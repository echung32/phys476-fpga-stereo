"""Convert the trained correlation student to hls4ml firmware.

The full-frame correlation model cannot be synthesised directly for an
Artix-class FPGA.  Instead this script exports the shared feature extractor
sub-model (feat_conv1 → feat_conv3) as a standalone Keras model and converts
that to hls4ml for hardware deployment.

The feature extractor operates on a single (H, W, 1) grayscale image and
produces a (H, W, C/2) feature map.  On-chip disparity matching is then
done using a WTA correlator that can be implemented with fixed BRAM bandwidth.

Usage:
    KERAS_BACKEND=torch uv run python -m v2.hls4ml.convert_student \
        --model logs/<run>/checkpoints/best.keras \
        --output-dir v2/hls4ml/student_hls

    # Full parity check (requires parity_batch.npz from training):
    KERAS_BACKEND=torch uv run python -m v2.hls4ml.convert_student \
        --model logs/<run>/checkpoints/best.keras \
        --parity-batch logs/<run>/parity_batch.npz \
        --output-dir v2/hls4ml/student_hls
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")

import hls4ml
import keras
import numpy as np
import yaml

from v2.models.keras_student import CorrelationCostVolume
from v2.training.hf_utils import resolve_repo_path


# ---------------------------------------------------------------------------
# Feature-extractor sub-model extraction
# ---------------------------------------------------------------------------

def extract_feature_extractor(full_model: keras.Model) -> keras.Model:
    """Return a standalone Keras model for the shared feature extractor.

    The extractor maps (H, W, 1) → (H, W, C/2) and uses the same weights as
    the trained full-frame model.  It is the only part exported to hls4ml.
    """
    feat_layer_names = ["feat_conv1", "feat_conv2", "feat_conv3"]
    # Build a new functional model using the same layer objects (shared weights)
    try:
        conv1 = full_model.get_layer("feat_conv1")
        conv2 = full_model.get_layer("feat_conv2")
        conv3 = full_model.get_layer("feat_conv3")
    except ValueError as exc:
        raise ValueError(
            "Could not find feature extractor layers in the loaded model.  "
            "Make sure you are loading a correlation_student model trained "
            "with v2/training/train_stereo.py."
        ) from exc

    # Infer input shape from conv1
    h = full_model.input["left"].shape[1]
    w = full_model.input["left"].shape[2]
    inp = keras.Input(shape=(h, w, 1), name="image")
    out = conv3(conv2(conv1(inp)))
    return keras.Model(inputs=inp, outputs=out, name="feature_extractor")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Export the stereo feature extractor to hls4ml")
    parser.add_argument("--model",        default="v2/exports/student_model.keras",
                        help="Path to the full trained correlation_student .keras file")
    parser.add_argument("--parity-batch", default="",
                        help="Path to .npz with 'left'/'right' keys for parity check (optional)")
    parser.add_argument("--output-dir",   default="v2/hls4ml/student_hls")
    parser.add_argument("--backend",      default="Vitis")
    parser.add_argument("--precision",    default="fixed<16,6>")
    parser.add_argument("--export-full",  action="store_true",
                        help="Attempt to export the full model instead of just the extractor "
                             "(may fail for large input sizes)")
    args = parser.parse_args()

    model_path = resolve_repo_path(args.model)
    output_dir = resolve_repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {model_path} …", flush=True)
    full_model = keras.models.load_model(
        model_path,
        custom_objects={"CorrelationCostVolume": CorrelationCostVolume},
    )

    if args.export_full:
        export_model = full_model
        print("Exporting full model (may be too large for synthesis).", flush=True)
    else:
        export_model = extract_feature_extractor(full_model)
        export_model.summary(print_fn=lambda s: print(s, flush=True))
        # Save the extractor separately so it can be inspected / loaded later
        extractor_path = output_dir / "feature_extractor.keras"
        export_model.save(str(extractor_path))
        print(f"Saved feature extractor to {extractor_path}", flush=True)

    config = hls4ml.utils.config_from_keras_model(
        export_model,
        granularity="name",
        backend=args.backend,
        default_precision=args.precision,
    )
    config_path = output_dir / "hls_config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(f"Saved hls4ml config to {config_path}", flush=True)

    hls_model = hls4ml.converters.convert_from_keras_model(
        model=export_model,
        output_dir=str(output_dir),
        backend=args.backend,
        io_type="io_stream",
        hls_config=config,
    )
    hls_model.compile()
    print("hls4ml compilation successful.", flush=True)

    # ---- parity check ----
    parity_metrics: dict[str, object] = {"backend": args.backend, "precision": args.precision}

    if args.parity_batch:
        parity_path = resolve_repo_path(args.parity_batch)
        parity = np.load(parity_path)

        if args.export_full:
            # Expect left/right keys
            left  = parity["left"].astype(np.float32)
            right = parity["right"].astype(np.float32)
            keras_pred = full_model.predict({"left": left, "right": right}, verbose=0)
            hls_pred   = hls_model.predict({"left": left, "right": right})
        else:
            # Feature-extractor parity: use left images only
            images = parity.get("left", parity.get("images")).astype(np.float32)
            keras_pred = export_model.predict(images, verbose=0).reshape(len(images), -1)
            hls_pred   = hls_model.predict(images).reshape(len(images), -1)

        parity_metrics.update({
            "samples": int(len(keras_pred)),
            "mean_abs_error": float(np.mean(np.abs(keras_pred - hls_pred))),
            "max_abs_error":  float(np.max(np.abs(keras_pred - hls_pred))),
        })

    metrics_path = output_dir / "parity_metrics.json"
    metrics_path.write_text(json.dumps(parity_metrics, indent=2), encoding="utf-8")
    print(f"Saved parity metrics to {metrics_path}", flush=True)
    print(json.dumps(parity_metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
