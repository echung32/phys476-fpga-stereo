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

from v2.training.hf_utils import resolve_repo_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert the Keras stereo student to hls4ml")
    parser.add_argument("--model", default="v2/exports/student_patch_model.keras")
    parser.add_argument("--parity-batch", default="v2/exports/parity_batch.npz")
    parser.add_argument("--output-dir", default="v2/hls4ml/student_hls")
    parser.add_argument("--backend", default="Vitis")
    parser.add_argument("--precision", default="fixed<16,6>")
    args = parser.parse_args()

    model_path = resolve_repo_path(args.model)
    parity_path = resolve_repo_path(args.parity_batch)
    output_dir = resolve_repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = keras.models.load_model(model_path)
    config = hls4ml.utils.config_from_keras_model(
        model,
        granularity="name",
        backend=args.backend,
        default_precision=args.precision,
    )
    config_path = output_dir / "hls_config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    hls_model = hls4ml.converters.convert_from_keras_model(
        model=model,
        output_dir=str(output_dir),
        backend=args.backend,
        io_type="io_stream",
        hls_config=config,
    )
    hls_model.compile()

    parity = np.load(parity_path)
    features = parity["features"].astype(np.float32)
    keras_predictions = model.predict(features, verbose=0).reshape(-1)
    hls_predictions = hls_model.predict(features).reshape(-1)

    metrics = {
        "samples": int(len(features)),
        "mean_abs_error": float(np.mean(np.abs(keras_predictions - hls_predictions))),
        "max_abs_error": float(np.max(np.abs(keras_predictions - hls_predictions))),
        "backend": args.backend,
        "precision": args.precision,
    }
    metrics_path = output_dir / "parity_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Saved hls4ml config to {config_path}")
    print(f"Saved parity metrics to {metrics_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()