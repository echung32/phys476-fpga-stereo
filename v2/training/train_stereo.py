"""GT-supervised training for the correlation student stereo model.

Usage (SceneFlow only, quick smoke test):
    KERAS_BACKEND=torch uv run python -m v2.training.train_stereo \
        --run-name smoke_correlation \
        --chunk-size 32 --epochs 1 --batch-size 4 \
        --val-limit 4 --sceneflow-limit 64

Usage (full mixed run):
    KERAS_BACKEND=torch uv run python -m v2.training.train_stereo \
        --run-name full_mixed \
        --driving-stereo-dir v2/data/raw/driving_stereo \
        --kitti2015-dir v2/data/raw/kitti2015 \
        --kitti2012-dir v2/data/raw/kitti2012
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import os
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")

import keras
import numpy as np

from v2.models.keras_student import build_correlation_student, _masked_smooth_l1, _mean_abs_error_valid
from v2.training.augmentation import AugmentConfig, augment, centre_crop_pad
from v2.training.hf_utils import resolve_repo_path
from v2.training.stereo_data import (
    StereoExample,
    build_mixed_manifest,
    get_driving_stereo_manifest,
    get_kitti2012_manifest,
    get_kitti2015_manifest,
    get_sceneflow_manifest,
    iter_manifest_chunks,
    load_kitti_hf_examples,
    write_manifest,
    load_manifest,
)


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrainConfig:
    run_name: str
    output_dir: str
    sceneflow_dataset: str
    kitti_hf_dataset: str
    driving_stereo_dir: str
    kitti2015_dir: str
    kitti2012_dir: str
    sceneflow_limit: int        # 0 = use all
    kitti_hf_limit: int
    val_limit: int
    target_height: int
    target_width: int
    max_disp: int
    feature_channels: int
    learning_rate: float
    epochs: int
    batch_size: int
    chunk_size: int             # examples per training chunk (SceneFlow streaming)
    augment: bool
    seed: int
    # Sampling fractions for mixed manifest
    sceneflow_frac: float
    driving_stereo_frac: float
    kitti_frac: float


# ---------------------------------------------------------------------------
# Tensor-building helpers
# ---------------------------------------------------------------------------

def _prepare_inputs(
    examples: list[StereoExample],
    *,
    target_h: int,
    target_w: int,
    max_disp: int,
    aug_config: AugmentConfig | None,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert examples to (left_batch, right_batch, y_true_batch) tensors.

    y_true shape: (N, H, W, 2) — channel 0 = disparity, channel 1 = valid mask.
    Disparity is clipped to [0, max_disp).
    """
    lefts, rights, y_trues = [], [], []
    for ex in examples:
        left  = ex.left_image
        right = ex.right_image
        disp  = ex.disparity
        valid = ex.valid_mask.astype(np.float32)

        if aug_config is not None:
            left, right, disp, valid_bool = augment(
                left, right, disp, valid.astype(bool),
                config=aug_config, rng=rng,
            )
            valid = valid_bool.astype(np.float32)

        # Fit to fixed target size
        left, right, disp, valid_bool = centre_crop_pad(
            left, right, disp, valid.astype(bool),
            out_h=target_h, out_w=target_w,
        )
        valid = valid_bool.astype(np.float32)

        # Clip disparity to model range
        disp  = np.clip(disp, 0.0, max_disp - 1.0)
        valid = valid * (disp < max_disp).astype(np.float32)

        lefts.append(left[..., np.newaxis])   # (H, W, 1)
        rights.append(right[..., np.newaxis])
        y_true = np.stack([disp, valid], axis=-1)  # (H, W, 2)
        y_trues.append(y_true)

    return (
        np.stack(lefts,   axis=0).astype(np.float32),
        np.stack(rights,  axis=0).astype(np.float32),
        np.stack(y_trues, axis=0).astype(np.float32),
    )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _train_on_chunk(
    model: keras.Model,
    examples: list[StereoExample],
    *,
    config: TrainConfig,
    aug_config: AugmentConfig | None,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Train the model on a single chunk of examples.  Returns loss/metric dict."""
    left_batch, right_batch, y_true_batch = _prepare_inputs(
        examples,
        target_h=config.target_height,
        target_w=config.target_width,
        max_disp=config.max_disp,
        aug_config=aug_config,
        rng=rng,
    )
    history = model.fit(
        {"left": left_batch, "right": right_batch},
        y_true_batch,
        epochs=1,
        batch_size=config.batch_size,
        verbose=0,
    )
    return {k: float(v[-1]) for k, v in history.history.items()}


def _evaluate(
    model: keras.Model,
    examples: list[StereoExample],
    *,
    config: TrainConfig,
    rng: np.random.Generator,
) -> dict[str, float]:
    left_batch, right_batch, y_true_batch = _prepare_inputs(
        examples,
        target_h=config.target_height,
        target_w=config.target_width,
        max_disp=config.max_disp,
        aug_config=None,   # no augmentation during evaluation
        rng=rng,
    )
    result = model.evaluate(
        {"left": left_batch, "right": right_batch},
        y_true_batch,
        batch_size=config.batch_size,
        verbose=0,
        return_dict=True,
    )
    return {k: float(v) for k, v in result.items()}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train the correlation student stereo model")
    # Dataset sources
    parser.add_argument("--sceneflow-dataset",  default="olivermao/sceneflow")
    parser.add_argument("--kitti-hf-dataset",   default="UniflexAI/mini_kitti")
    parser.add_argument("--driving-stereo-dir", default="",  help="Local root of DrivingStereo (empty = skip)")
    parser.add_argument("--kitti2015-dir",      default="",  help="Local root of KITTI 2015 (empty = skip)")
    parser.add_argument("--kitti2012-dir",      default="",  help="Local root of KITTI 2012 (empty = skip)")
    # Dataset limits
    parser.add_argument("--sceneflow-limit",    type=int, default=0)
    parser.add_argument("--kitti-hf-limit",     type=int, default=88)
    parser.add_argument("--val-limit",          type=int, default=16)
    # Model
    parser.add_argument("--target-height",      type=int, default=96)
    parser.add_argument("--target-width",       type=int, default=320)
    parser.add_argument("--max-disp",           type=int, default=48)
    parser.add_argument("--feature-channels",   type=int, default=16)
    parser.add_argument("--learning-rate",      type=float, default=1e-3)
    # Training
    parser.add_argument("--epochs",             type=int, default=10)
    parser.add_argument("--batch-size",         type=int, default=4)
    parser.add_argument("--chunk-size",         type=int, default=256,
                        help="Examples per SceneFlow streaming chunk")
    parser.add_argument("--no-augment",         action="store_true")
    parser.add_argument("--seed",               type=int, default=0)
    # Sampling fractions
    parser.add_argument("--sceneflow-frac",     type=float, default=0.50)
    parser.add_argument("--driving-stereo-frac",type=float, default=0.35)
    parser.add_argument("--kitti-frac",         type=float, default=0.15)
    # Output
    parser.add_argument("--output-root",        default="logs")
    parser.add_argument("--run-name",           default="correlation_stereo")
    parser.add_argument("--output-dir",         default="")
    args = parser.parse_args()

    # --- output directory ---
    if args.output_dir:
        run_dir = resolve_repo_path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = resolve_repo_path(Path(args.output_root) / f"{timestamp}_{args.run_name}")
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir     = run_dir / "checkpoints";  ckpt_dir.mkdir(exist_ok=True)
    manifest_dir = run_dir / "manifests";    manifest_dir.mkdir(exist_ok=True)
    hls_dir      = run_dir / "hls4ml";       hls_dir.mkdir(exist_ok=True)

    config = TrainConfig(
        run_name=args.run_name,
        output_dir=str(run_dir),
        sceneflow_dataset=args.sceneflow_dataset,
        kitti_hf_dataset=args.kitti_hf_dataset,
        driving_stereo_dir=args.driving_stereo_dir,
        kitti2015_dir=args.kitti2015_dir,
        kitti2012_dir=args.kitti2012_dir,
        sceneflow_limit=args.sceneflow_limit,
        kitti_hf_limit=args.kitti_hf_limit,
        val_limit=args.val_limit,
        target_height=args.target_height,
        target_width=args.target_width,
        max_disp=args.max_disp,
        feature_channels=args.feature_channels,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        augment=not args.no_augment,
        seed=args.seed,
        sceneflow_frac=args.sceneflow_frac,
        driving_stereo_frac=args.driving_stereo_frac,
        kitti_frac=args.kitti_frac,
    )
    (run_dir / "run_config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    print(f"Run directory: {run_dir}", flush=True)

    rng = np.random.default_rng(config.seed)

    # --- build model ---
    model = build_correlation_student(
        height=config.target_height,
        width=config.target_width,
        max_disp=config.max_disp,
        feature_channels=config.feature_channels,
        learning_rate=config.learning_rate,
    )
    model.summary(print_fn=lambda s: print(s, flush=True))

    # --- augmentation config ---
    aug_config: AugmentConfig | None = None
    if config.augment:
        aug_config = AugmentConfig(
            crop_h=config.target_height,
            crop_w=config.target_width,
        )

    # --- load / build manifests ---
    print("Building manifests …", flush=True)

    # Scene Flow
    _, sceneflow_manifest = get_sceneflow_manifest(config.sceneflow_dataset)
    if config.sceneflow_limit > 0:
        sceneflow_manifest = sceneflow_manifest[: config.sceneflow_limit]
    write_manifest(manifest_dir / "sceneflow_manifest.json", sceneflow_manifest)
    print(f"  SceneFlow: {len(sceneflow_manifest)} entries", flush=True)

    # DrivingStereo (optional)
    ds_manifest: list[dict] | None = None
    if config.driving_stereo_dir:
        try:
            ds_manifest = get_driving_stereo_manifest(config.driving_stereo_dir)
            write_manifest(manifest_dir / "driving_stereo_manifest.json", ds_manifest)
            print(f"  DrivingStereo: {len(ds_manifest)} entries", flush=True)
        except FileNotFoundError as exc:
            print(f"  DrivingStereo skipped: {exc}", flush=True)

    # KITTI 2015 (optional)
    kitti2015_manifest: list[dict] | None = None
    if config.kitti2015_dir:
        try:
            kitti2015_manifest = get_kitti2015_manifest(config.kitti2015_dir)
            write_manifest(manifest_dir / "kitti2015_manifest.json", kitti2015_manifest)
            print(f"  KITTI 2015: {len(kitti2015_manifest)} entries", flush=True)
        except FileNotFoundError as exc:
            print(f"  KITTI 2015 skipped: {exc}", flush=True)

    # KITTI 2012 (optional)
    kitti2012_manifest: list[dict] | None = None
    if config.kitti2012_dir:
        try:
            kitti2012_manifest = get_kitti2012_manifest(config.kitti2012_dir)
            write_manifest(manifest_dir / "kitti2012_manifest.json", kitti2012_manifest)
            print(f"  KITTI 2012: {len(kitti2012_manifest)} entries", flush=True)
        except FileNotFoundError as exc:
            print(f"  KITTI 2012 skipped: {exc}", flush=True)

    # Mixed manifest
    mixed_manifest = build_mixed_manifest(
        sceneflow_manifest=sceneflow_manifest,
        driving_stereo_manifest=ds_manifest,
        kitti2015_manifest=kitti2015_manifest,
        kitti2012_manifest=kitti2012_manifest,
        sceneflow_frac=config.sceneflow_frac,
        driving_stereo_frac=config.driving_stereo_frac,
        kitti_frac=config.kitti_frac,
        rng=rng,
    )
    write_manifest(manifest_dir / "mixed_train_manifest.json", mixed_manifest)
    print(f"  Mixed train: {len(mixed_manifest)} entries", flush=True)

    # Validation: mini_kitti HF examples (cheap + always available)
    print("Loading validation examples …", flush=True)
    val_examples = load_kitti_hf_examples(
        config.kitti_hf_dataset, split="validation", max_examples=config.val_limit
    )
    write_manifest(manifest_dir / "validation_manifest.json", [e.metadata for e in val_examples])
    print(f"  Validation: {len(val_examples)} examples", flush=True)

    # --- training loop ---
    training_log: list[dict[str, float | int]] = []
    best_val_mae = float("inf")
    log_csv_path = ckpt_dir / "training_log.csv"
    csv_file = open(log_csv_path, "w", newline="")
    csv_writer: csv.DictWriter | None = None

    for epoch in range(1, config.epochs + 1):
        print(f"\n=== Epoch {epoch}/{config.epochs} ===", flush=True)
        epoch_losses: list[float] = []
        epoch_maes:   list[float] = []
        chunk_count   = 0
        example_count = 0

        for chunk_examples in iter_manifest_chunks(mixed_manifest, config.chunk_size):
            metrics = _train_on_chunk(
                model, chunk_examples,
                config=config, aug_config=aug_config, rng=rng,
            )
            chunk_count   += 1
            example_count += len(chunk_examples)
            epoch_losses.append(metrics.get("loss", float("nan")))
            epoch_maes.append(metrics.get("_mean_abs_error_valid", float("nan")))
            if chunk_count % 10 == 0:
                print(
                    f"  chunk {chunk_count:4d}  examples {example_count:6d}  "
                    f"loss {epoch_losses[-1]:.4f}  mae {epoch_maes[-1]:.4f}",
                    flush=True,
                )

        train_loss = float(np.nanmean(epoch_losses))
        train_mae  = float(np.nanmean(epoch_maes))

        # Validation
        val_metrics = _evaluate(model, val_examples, config=config, rng=rng)
        val_mae = val_metrics.get("_mean_abs_error_valid", float("nan"))

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_mae":  train_mae,
            "val_loss":   val_metrics.get("loss", float("nan")),
            "val_mae":    val_mae,
        }
        training_log.append(row)

        # CSV header on first row
        if csv_writer is None:
            csv_writer = csv.DictWriter(csv_file, fieldnames=list(row.keys()))
            csv_writer.writeheader()
        csv_writer.writerow(row)
        csv_file.flush()

        print(
            f"  Epoch {epoch} summary  train_loss={train_loss:.4f}  "
            f"train_mae={train_mae:.4f}  val_mae={val_mae:.4f}",
            flush=True,
        )

        # Save checkpoints
        model.save(str(ckpt_dir / "latest.keras"))
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            model.save(str(ckpt_dir / "best.keras"))
            print(f"  ** New best val MAE: {best_val_mae:.4f} — saved best.keras **", flush=True)

    csv_file.close()

    # Final artefacts
    (run_dir / "training_history.json").write_text(
        json.dumps(training_log, indent=2), encoding="utf-8"
    )
    model.save(str(run_dir / "student_model.keras"))
    print(f"\nTraining complete. Best val MAE: {best_val_mae:.4f}", flush=True)
    print(f"Artefacts saved to: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
