"""Stereo-safe augmentation utilities.

ALL transforms operate on (left_image, right_image, disparity, valid_mask)
simultaneously to preserve epipolar geometry.

Forbidden operations (enforced by design — none of the functions below perform):
  - Horizontal flips
  - Independent left/right geometric transforms
  - Rotations that violate rectified epipolar geometry
  - Vertical stereo offsets
  - Any geometry change without disparity-map remap

Allowed operations implemented here:
  - Shared crop (same random crop for all four arrays)
  - Shared resize with disparity scaling by horizontal scale factor
  - Shared pad / center-crop to a fixed output size
  - Shared photometric jitter (brightness / contrast / gamma) applied identically
    to both left and right, with a separate optional per-image version for colour
    consistency simulation
  - Shared mild Gaussian noise
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class AugmentConfig:
    """Tunable knobs for the augmentation pipeline."""

    # Shared random crop: output size; None means no crop
    crop_h: int | None = None
    crop_w: int | None = None

    # Shared resize: target size (H, W); None means no resize
    resize_h: int | None = None
    resize_w: int | None = None

    # Photometric jitter ranges (multiplicative, centred on 1.0)
    brightness_range: tuple[float, float] = (0.8, 1.2)
    contrast_range:   tuple[float, float] = (0.8, 1.2)
    gamma_range:      tuple[float, float] = (0.8, 1.2)

    # Mild Gaussian noise (std as fraction of [0,1] range)
    noise_std: float = 0.02

    # Probability of applying each photometric op
    prob_jitter:  float = 0.5
    prob_noise:   float = 0.3


def augment(
    left: np.ndarray,
    right: np.ndarray,
    disparity: np.ndarray,
    valid_mask: np.ndarray,
    *,
    config: AugmentConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply stereo-safe augmentations in place.

    Args:
        left:       (H, W) float32 grayscale image in [0, 1].
        right:      (H, W) float32 grayscale image in [0, 1].
        disparity:  (H, W) float32 disparity map.
        valid_mask: (H, W) bool mask — True where disparity is defined.
        config:     AugmentConfig instance.
        rng:        NumPy RNG for reproducibility.

    Returns:
        Augmented (left, right, disparity, valid_mask).
    """
    # 1. Shared random crop
    if config.crop_h is not None and config.crop_w is not None:
        left, right, disparity, valid_mask = _shared_crop(
            left, right, disparity, valid_mask,
            crop_h=config.crop_h, crop_w=config.crop_w, rng=rng,
        )

    # 2. Shared resize (with disparity rescaling)
    if config.resize_h is not None and config.resize_w is not None:
        left, right, disparity, valid_mask = _shared_resize(
            left, right, disparity, valid_mask,
            out_h=config.resize_h, out_w=config.resize_w,
        )

    # 3. Photometric jitter (same transform applied to both left and right)
    if rng.random() < config.prob_jitter:
        left, right = _shared_photometric_jitter(
            left, right,
            brightness_range=config.brightness_range,
            contrast_range=config.contrast_range,
            gamma_range=config.gamma_range,
            rng=rng,
        )

    # 4. Shared mild Gaussian noise
    if rng.random() < config.prob_noise:
        noise = rng.normal(0.0, config.noise_std, size=left.shape).astype(np.float32)
        left  = np.clip(left  + noise, 0.0, 1.0)
        right = np.clip(right + noise, 0.0, 1.0)

    return left, right, disparity, valid_mask


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _shared_crop(
    left: np.ndarray,
    right: np.ndarray,
    disparity: np.ndarray,
    valid_mask: np.ndarray,
    *,
    crop_h: int,
    crop_w: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    h, w = left.shape[:2]
    if crop_h > h or crop_w > w:
        # Fall back to centre-crop if target is larger than source
        return _shared_centre_crop(left, right, disparity, valid_mask, crop_h=min(crop_h, h), crop_w=min(crop_w, w))
    row0 = int(rng.integers(0, h - crop_h + 1))
    col0 = int(rng.integers(0, w - crop_w + 1))
    sl_r = slice(row0, row0 + crop_h)
    sl_c = slice(col0, col0 + crop_w)
    return (
        left[sl_r, sl_c],
        right[sl_r, sl_c],
        disparity[sl_r, sl_c],
        valid_mask[sl_r, sl_c],
    )


def _shared_centre_crop(
    left: np.ndarray,
    right: np.ndarray,
    disparity: np.ndarray,
    valid_mask: np.ndarray,
    *,
    crop_h: int,
    crop_w: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    h, w = left.shape[:2]
    row0 = (h - crop_h) // 2
    col0 = (w - crop_w) // 2
    sl_r = slice(row0, row0 + crop_h)
    sl_c = slice(col0, col0 + crop_w)
    return (
        left[sl_r, sl_c],
        right[sl_r, sl_c],
        disparity[sl_r, sl_c],
        valid_mask[sl_r, sl_c],
    )


def _shared_resize(
    left: np.ndarray,
    right: np.ndarray,
    disparity: np.ndarray,
    valid_mask: np.ndarray,
    *,
    out_h: int,
    out_w: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Nearest-or-bilinear resize; scale disparity by horizontal scale factor."""
    from PIL import Image

    orig_h, orig_w = left.shape[:2]
    h_scale = out_h / orig_h
    w_scale = out_w / orig_w

    def _resize_img(arr: np.ndarray) -> np.ndarray:
        pil = Image.fromarray((arr * 255).clip(0, 255).astype(np.uint8))
        pil = pil.resize((out_w, out_h), Image.BILINEAR)
        return np.array(pil, dtype=np.float32) / 255.0

    def _resize_disp(arr: np.ndarray) -> np.ndarray:
        # Use nearest-neighbour for disparity to avoid interpolation artefacts
        pil = Image.fromarray(arr.astype(np.float32), mode="F")
        pil = pil.resize((out_w, out_h), Image.NEAREST)
        return np.array(pil, dtype=np.float32) * w_scale

    def _resize_mask(arr: np.ndarray) -> np.ndarray:
        pil = Image.fromarray(arr.astype(np.uint8) * 255)
        pil = pil.resize((out_w, out_h), Image.NEAREST)
        return np.array(pil) > 127

    return (
        _resize_img(left),
        _resize_img(right),
        _resize_disp(disparity),
        _resize_mask(valid_mask),
    )


def _shared_photometric_jitter(
    left: np.ndarray,
    right: np.ndarray,
    *,
    brightness_range: tuple[float, float],
    contrast_range: tuple[float, float],
    gamma_range: tuple[float, float],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the same brightness / contrast / gamma to both images."""
    brt = float(rng.uniform(*brightness_range))
    con = float(rng.uniform(*contrast_range))
    gam = float(rng.uniform(*gamma_range))

    def _jitter(img: np.ndarray) -> np.ndarray:
        img = img * brt                            # brightness
        mean = float(np.mean(img))
        img = mean + (img - mean) * con            # contrast
        img = np.clip(img, 0.0, 1.0)
        img = img ** (1.0 / gam)                   # gamma (valid because img ≥ 0)
        return np.clip(img, 0.0, 1.0)

    return _jitter(left), _jitter(right)


def centre_crop_pad(
    left: np.ndarray,
    right: np.ndarray,
    disparity: np.ndarray,
    valid_mask: np.ndarray,
    *,
    out_h: int,
    out_w: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Centre-crop then zero-pad to (out_h, out_w) — stereo-safe, geometry-neutral."""
    h, w = left.shape[:2]
    # Crop if too large
    if h > out_h or w > out_w:
        ch = min(h, out_h)
        cw = min(w, out_w)
        left, right, disparity, valid_mask = _shared_centre_crop(
            left, right, disparity, valid_mask, crop_h=ch, crop_w=cw
        )
        h, w = left.shape[:2]
    # Pad if too small
    if h < out_h or w < out_w:
        ph = max(0, out_h - h)
        pw = max(0, out_w - w)
        pt, pb = ph // 2, ph - ph // 2
        pl, pr = pw // 2, pw - pw // 2
        left      = np.pad(left,      ((pt, pb), (pl, pr)))
        right     = np.pad(right,     ((pt, pb), (pl, pr)))
        disparity = np.pad(disparity, ((pt, pb), (pl, pr)), constant_values=0.0)
        valid_mask = np.pad(valid_mask, ((pt, pb), (pl, pr)), constant_values=False)
    return left, right, disparity, valid_mask
