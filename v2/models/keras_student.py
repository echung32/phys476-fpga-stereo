"""Low-resolution full-frame correlation student for stereo depth estimation.

Architecture:
  1. Shared tiny feature extractor (left + right images, weight-shared)
  2. Correlation cost volume across disparity range [0, max_disp)
  3. 2-D disparity regression head (small conv stack → soft-argmin)

Input contract:
  - left_image:  (H, W, 1)  float32, pixel values in [0, 1]
  - right_image: (H, W, 1)  float32, pixel values in [0, 1]

Output:
  - disparity:   (H, W, 1)  float32, predicted disparity in pixels
                 (scaled by max_disp; multiply by (original_W / W) to get
                  full-resolution pixel disparities)
"""

from __future__ import annotations

import os

os.environ.setdefault("KERAS_BACKEND", "torch")

import keras
import numpy as np


# ---------------------------------------------------------------------------
# Cost-volume layer
# ---------------------------------------------------------------------------

class CorrelationCostVolume(keras.layers.Layer):
    """Compute a 1-D correlation cost volume along the horizontal (disparity) axis.

    For each disparity offset d in [0, max_disp):
      cost[:, :, w, d] = mean(left_feat[:, :, w, :] * right_feat[:, :, w-d, :])

    Zero-padding is used where the right index would go out of bounds.

    Shapes:
      inputs[0]  left_feat:   (B, H, W, C)
      inputs[1]  right_feat:  (B, H, W, C)
      output:                 (B, H, W, max_disp)
    """

    def __init__(self, max_disp: int = 48, **kwargs):
        super().__init__(**kwargs)
        self.max_disp = max_disp

    def call(self, inputs):
        left_feat, right_feat = inputs
        # Build the cost volume slice-by-slice.
        # We stack along a new axis to avoid Python-level loops in the graph
        # where possible. For small max_disp this is fine.
        slices = []
        for d in range(self.max_disp):
            if d == 0:
                # No shift: direct dot product
                cost_d = keras.ops.mean(left_feat * right_feat, axis=-1, keepdims=True)
            else:
                # Shift right_feat d pixels to the right (i.e. compare left[w] with right[w-d])
                # Pad left with d zeros, then crop to original width
                pad_width = [[0, 0], [0, 0], [d, 0], [0, 0]]
                right_shifted = keras.ops.pad(right_feat, pad_width)
                # Crop back to original width W
                right_shifted = right_shifted[:, :, : keras.ops.shape(left_feat)[2], :]
                cost_d = keras.ops.mean(left_feat * right_shifted, axis=-1, keepdims=True)
            slices.append(cost_d)
        # Concatenate along last axis → (B, H, W, max_disp)
        return keras.ops.concatenate(slices, axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({"max_disp": self.max_disp})
        return config


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_correlation_student(
    *,
    height: int = 96,
    width: int = 320,
    max_disp: int = 48,
    feature_channels: int = 16,
    learning_rate: float = 1e-3,
) -> keras.Model:
    """Build the full-frame correlation student model.

    Args:
        height: Input image height in pixels.
        width:  Input image width in pixels.
        max_disp: Maximum disparity range (exclusive). The model outputs
            disparity values in [0, max_disp).
        feature_channels: Number of channels in the shared feature extractor.
        learning_rate: Adam learning rate.

    Returns:
        Compiled Keras model with inputs ["left", "right"] and output "disparity".
    """
    left_input  = keras.Input(shape=(height, width, 1), name="left")
    right_input = keras.Input(shape=(height, width, 1), name="right")

    # ---- shared feature extractor (weight-tied) ----
    conv1 = keras.layers.Conv2D(
        feature_channels, (3, 3), padding="same", activation="relu", name="feat_conv1"
    )
    conv2 = keras.layers.Conv2D(
        feature_channels, (3, 3), padding="same", activation="relu", name="feat_conv2"
    )
    conv3 = keras.layers.Conv2D(
        feature_channels // 2, (3, 3), padding="same", activation="relu", name="feat_conv3"
    )

    def extract(x):
        return conv3(conv2(conv1(x)))

    left_feat  = extract(left_input)   # (B, H, W, C/2)
    right_feat = extract(right_input)  # (B, H, W, C/2)

    # ---- cost volume ----
    cost_vol = CorrelationCostVolume(max_disp=max_disp, name="cost_volume")(
        [left_feat, right_feat]
    )  # (B, H, W, max_disp)

    # ---- disparity regression head ----
    x = keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu", name="disp_conv1")(cost_vol)
    x = keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu", name="disp_conv2")(x)
    logits = keras.layers.Conv2D(max_disp, (1, 1), padding="same", name="disp_logits")(x)
    # (B, H, W, max_disp)

    # Soft-argmin: differentiable disparity regression
    weights = keras.layers.Softmax(axis=-1, name="disp_softmax")(logits)
    # Build disparity index tensor [0, 1, ..., max_disp-1] and broadcast
    disp_indices = keras.ops.cast(
        keras.ops.arange(max_disp), dtype="float32"
    )  # (max_disp,)
    disparity = keras.ops.sum(
        weights * disp_indices, axis=-1, keepdims=True
    )  # (B, H, W, 1)

    model = keras.Model(
        inputs={"left": left_input, "right": right_input},
        outputs=disparity,
        name="correlation_student",
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=_masked_smooth_l1,
        metrics=[_mean_abs_error_valid],
    )
    return model


# ---------------------------------------------------------------------------
# Loss and metrics
# ---------------------------------------------------------------------------

def _masked_smooth_l1(y_true, y_pred):
    """Smooth-L1 loss computed only over valid (non-negative) GT disparity pixels.

    y_true shape: (B, H, W, 2) — channel 0: disparity, channel 1: valid mask (1/0)
    y_pred shape: (B, H, W, 1) — predicted disparity
    """
    gt_disp  = y_true[..., :1]      # (B, H, W, 1)
    valid    = y_true[..., 1:2]     # (B, H, W, 1)  float mask
    diff     = keras.ops.abs(gt_disp - y_pred)
    huber    = keras.ops.where(diff < 1.0, 0.5 * diff ** 2, diff - 0.5)
    masked   = huber * valid
    n_valid  = keras.ops.maximum(keras.ops.sum(valid), 1.0)
    return keras.ops.sum(masked) / n_valid


def _mean_abs_error_valid(y_true, y_pred):
    """Mean absolute error over valid pixels."""
    gt_disp = y_true[..., :1]
    valid   = y_true[..., 1:2]
    diff    = keras.ops.abs(gt_disp - y_pred) * valid
    n_valid = keras.ops.maximum(keras.ops.sum(valid), 1.0)
    return keras.ops.sum(diff) / n_valid
