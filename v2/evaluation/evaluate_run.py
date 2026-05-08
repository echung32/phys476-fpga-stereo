"""Evaluate a saved stereo training run on held-out data.

This script scores the best checkpoint from a saved run against:
  - an unseen DrivingStereo holdout derived from examples not present in the
    run's mixed training manifest
  - the full mini-KITTI validation split used as a lightweight external check

It also saves qualitative sample panels showing the input image, ground-truth
    disparity, prediction, and absolute error.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")

import keras
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from v2.models.keras_student import CorrelationCostVolume, _masked_smooth_l1, _mean_abs_error_valid
from v2.training.hf_utils import resolve_repo_path
from v2.training.stereo_data import StereoExample, iter_manifest_chunks, load_kitti_hf_examples
from v2.training.train_stereo import TrainConfig, _prepare_inputs


def _load_run_config(run_dir: Path) -> TrainConfig:
    config_data = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    config_data.setdefault("driving_holdout_limit", 0)
    return TrainConfig(**config_data)


def _load_model(checkpoint_path: Path) -> keras.Model:
    return keras.models.load_model(
        checkpoint_path,
        custom_objects={
            "CorrelationCostVolume": CorrelationCostVolume,
            "_masked_smooth_l1": _masked_smooth_l1,
            "_mean_abs_error_valid": _mean_abs_error_valid,
        },
        compile=False,
    )


def _make_manifest_key(entry: dict) -> str:
    return json.dumps(entry, sort_keys=True)


def _select_driving_holdout(
    manifests_dir: Path,
    *,
    limit: int,
    seed: int,
) -> list[dict]:
    mixed_manifest = json.loads((manifests_dir / "mixed_train_manifest.json").read_text(encoding="utf-8"))
    driving_manifest = json.loads((manifests_dir / "driving_stereo_manifest.json").read_text(encoding="utf-8"))
    seen_keys = {_make_manifest_key(entry) for entry in mixed_manifest}
    holdout_entries = [entry for entry in driving_manifest if _make_manifest_key(entry) not in seen_keys]

    if not holdout_entries:
        raise ValueError("No unseen DrivingStereo holdout entries were found for this run")

    rng = np.random.default_rng(seed)
    indices = np.arange(len(holdout_entries))
    rng.shuffle(indices)
    if limit > 0:
        indices = indices[:limit]
    return [holdout_entries[int(index)] for index in indices]


def _update_metric_sums(metric_sums: dict[str, float], y_true_batch: np.ndarray, y_pred_batch: np.ndarray) -> None:
    gt_disp = y_true_batch[..., 0]
    valid = y_true_batch[..., 1] > 0.5
    pred_disp = y_pred_batch[..., 0]
    abs_error = np.abs(gt_disp - pred_disp)
    valid_error = abs_error[valid]

    metric_sums["examples"] += float(y_true_batch.shape[0])
    metric_sums["valid_pixels"] += float(np.count_nonzero(valid))
    metric_sums["mae_sum"] += float(valid_error.sum())
    metric_sums["rmse_sum"] += float(np.square(valid_error).sum())
    metric_sums["bad1_sum"] += float(np.count_nonzero(valid_error > 1.0))
    metric_sums["bad3_sum"] += float(np.count_nonzero(valid_error > 3.0))
    metric_sums["bad5_sum"] += float(np.count_nonzero(valid_error > 5.0))


def _finalize_metric_sums(metric_sums: dict[str, float]) -> dict[str, float]:
    valid_pixels = max(metric_sums["valid_pixels"], 1.0)
    return {
        "examples": int(metric_sums["examples"]),
        "valid_pixels": int(metric_sums["valid_pixels"]),
        "mae": metric_sums["mae_sum"] / valid_pixels,
        "rmse": math.sqrt(metric_sums["rmse_sum"] / valid_pixels),
        "bad_1px": metric_sums["bad1_sum"] / valid_pixels,
        "bad_3px": metric_sums["bad3_sum"] / valid_pixels,
        "bad_5px": metric_sums["bad5_sum"] / valid_pixels,
    }


def _sample_summary(example: StereoExample, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, object]:
    gt_disp = y_true[..., 0]
    valid = y_true[..., 1] > 0.5
    pred_disp = y_pred[..., 0]
    abs_error = np.abs(gt_disp - pred_disp)
    valid_error = abs_error[valid]
    sample_mae = float(valid_error.mean()) if valid_error.size else float("nan")
    sample_p95 = float(np.quantile(valid_error, 0.95)) if valid_error.size else float("nan")

    metadata = {key: value for key, value in example.metadata.items()}
    return {
        "metadata": metadata,
        "valid_pixels": int(np.count_nonzero(valid)),
        "mae": sample_mae,
        "p95_abs_error": sample_p95,
    }


def _save_sample_panel(
    output_path: Path,
    example: StereoExample,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    title: str,
) -> dict[str, object]:
    gt_disp = y_true[..., 0]
    valid = y_true[..., 1] > 0.5
    pred_disp = y_pred[..., 0]
    abs_error = np.abs(gt_disp - pred_disp)
    masked_gt = np.where(valid, gt_disp, np.nan)
    masked_pred = np.where(valid, pred_disp, np.nan)
    masked_err = np.where(valid, abs_error, np.nan)

    max_disp = float(np.nanpercentile(masked_gt, 99)) if np.count_nonzero(valid) else 1.0
    max_err = float(np.nanpercentile(masked_err, 99)) if np.count_nonzero(valid) else 1.0

    figure, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
    axes[0].imshow(example.left_image, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0].set_title("Left")
    axes[1].imshow(masked_gt, cmap="magma", vmin=0.0, vmax=max_disp)
    axes[1].set_title("Ground Truth")
    axes[2].imshow(masked_pred, cmap="magma", vmin=0.0, vmax=max_disp)
    axes[2].set_title("Prediction")
    axes[3].imshow(masked_err, cmap="inferno", vmin=0.0, vmax=max_err)
    axes[3].set_title("Abs Error")

    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])

    summary = _sample_summary(example, y_true, y_pred)
    figure.suptitle(
        f"{title} | mae={summary['mae']:.3f} | p95={summary['p95_abs_error']:.3f}",
        fontsize=12,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    summary["panel_path"] = str(output_path)
    return summary


def _evaluate_examples(
    model: keras.Model,
    examples: list[StereoExample],
    *,
    config: TrainConfig,
    sample_dir: Path,
    sample_prefix: str,
    sample_count: int,
    batch_size: int,
    rng_seed: int,
) -> dict[str, object]:
    metric_sums = {
        "examples": 0.0,
        "valid_pixels": 0.0,
        "mae_sum": 0.0,
        "rmse_sum": 0.0,
        "bad1_sum": 0.0,
        "bad3_sum": 0.0,
        "bad5_sum": 0.0,
    }
    sample_summaries: list[dict[str, object]] = []
    rng = np.random.default_rng(rng_seed)

    for offset in range(0, len(examples), batch_size):
        chunk_examples = examples[offset : offset + batch_size]
        left_batch, right_batch, y_true_batch = _prepare_inputs(
            chunk_examples,
            target_h=config.target_height,
            target_w=config.target_width,
            max_disp=config.max_disp,
            aug_config=None,
            rng=rng,
        )
        y_pred_batch = model.predict(
            {"left": left_batch, "right": right_batch},
            batch_size=config.batch_size,
            verbose=0,
        )

        _update_metric_sums(metric_sums, y_true_batch, y_pred_batch)

        while len(sample_summaries) < sample_count and len(sample_summaries) < len(examples):
            local_index = len(sample_summaries) - offset
            if local_index < 0 or local_index >= len(chunk_examples):
                break
            panel_path = sample_dir / f"{sample_prefix}_{len(sample_summaries):02d}.png"
            sample_summaries.append(
                _save_sample_panel(
                    panel_path,
                    chunk_examples[local_index],
                    y_true_batch[local_index],
                    y_pred_batch[local_index],
                    title=f"{sample_prefix} sample {len(sample_summaries)}",
                )
            )

    return {
        "metrics": _finalize_metric_sums(metric_sums),
        "samples": sample_summaries,
    }


def _evaluate_manifest_entries(
    model: keras.Model,
    entries: list[dict],
    *,
    config: TrainConfig,
    sample_dir: Path,
    sample_prefix: str,
    sample_count: int,
    chunk_size: int,
) -> dict[str, object]:
    examples: list[StereoExample] = []
    for chunk_examples in iter_manifest_chunks(
        entries,
        chunk_size,
        loader_workers=config.loader_workers,
    ):
        examples.extend(chunk_examples)
    return _evaluate_examples(
        model,
        examples,
        config=config,
        sample_dir=sample_dir,
        sample_prefix=sample_prefix,
        sample_count=sample_count,
        batch_size=chunk_size,
        rng_seed=config.seed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved stereo training run")
    parser.add_argument("--run-dir", required=True, help="Path to the saved training run directory")
    parser.add_argument("--checkpoint", default="", help="Optional checkpoint path (defaults to run_dir/checkpoints/best.keras)")
    parser.add_argument("--output-dir", default="", help="Output directory for evaluation artefacts")
    parser.add_argument("--driving-holdout-limit", type=int, default=2048, help="Number of unseen DrivingStereo holdout examples to evaluate (0 = all)")
    parser.add_argument("--sample-count", type=int, default=6, help="Number of qualitative sample panels per dataset")
    parser.add_argument("--eval-chunk-size", type=int, default=256, help="Examples per evaluation chunk")
    args = parser.parse_args()

    run_dir = resolve_repo_path(args.run_dir)
    checkpoint_path = resolve_repo_path(args.checkpoint) if args.checkpoint else run_dir / "checkpoints" / "best.keras"
    output_dir = resolve_repo_path(args.output_dir) if args.output_dir else run_dir / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    config = _load_run_config(run_dir)
    model = _load_model(checkpoint_path)

    driving_holdout_entries = _select_driving_holdout(
        run_dir / "manifests",
        limit=args.driving_holdout_limit,
        seed=config.seed + 17,
    )
    driving_result = _evaluate_manifest_entries(
        model,
        driving_holdout_entries,
        config=config,
        sample_dir=output_dir / "samples" / "driving_holdout",
        sample_prefix="driving_holdout",
        sample_count=args.sample_count,
        chunk_size=args.eval_chunk_size,
    )

    mini_kitti_examples = load_kitti_hf_examples(
        config.kitti_hf_dataset,
        split="validation",
        max_examples=0,
    )
    mini_kitti_result = _evaluate_examples(
        model,
        mini_kitti_examples,
        config=config,
        sample_dir=output_dir / "samples" / "mini_kitti_validation",
        sample_prefix="mini_kitti_validation",
        sample_count=args.sample_count,
        batch_size=args.eval_chunk_size,
        rng_seed=config.seed + 29,
    )

    report = {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "config": asdict(config),
        "driving_holdout": {
            "selected_examples": len(driving_holdout_entries),
            **driving_result,
        },
        "mini_kitti_validation": {
            "selected_examples": len(mini_kitti_examples),
            **mini_kitti_result,
        },
    }
    report_path = output_dir / "evaluation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved evaluation report to: {report_path}")


if __name__ == "__main__":
    main()