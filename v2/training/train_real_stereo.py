from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")

import keras
import matplotlib.pyplot as plt
import numpy as np

from v2.models.keras_student import build_patch_student
from v2.training.hf_utils import resolve_repo_path
from v2.training.patch_dataset import build_training_examples, compute_disparity_metrics, reconstruct_disparity_map
from v2.training.stereo_data import StereoExample, load_kitti_examples, load_sceneflow_examples, write_manifest


@dataclass(frozen=True)
class RunConfig:
    pretrain_dataset: str
    finetune_dataset: str
    pretrain_max_examples: int
    finetune_max_examples: int
    validation_max_examples: int
    max_positions_per_image: int
    window_size: int
    max_disp: int
    negatives_per_positive: int
    pretrain_epochs: int
    finetune_epochs: int
    batch_size: int
    learning_rate: float
    conv_filters: int
    dense_units: int
    seed: int


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the real-data stereo student model")
    parser.add_argument("--pretrain-dataset", default="olivermao/sceneflow")
    parser.add_argument("--finetune-dataset", default="UniflexAI/mini_kitti")
    parser.add_argument("--pretrain-max-examples", type=int, default=0)
    parser.add_argument("--finetune-max-examples", type=int, default=8)
    parser.add_argument("--validation-max-examples", type=int, default=2)
    parser.add_argument("--max-positions-per-image", type=int, default=512)
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--max-disp", type=int, default=16)
    parser.add_argument("--negatives-per-positive", type=int, default=1)
    parser.add_argument("--pretrain-epochs", type=int, default=2)
    parser.add_argument("--finetune-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--conv-filters", type=int, default=8)
    parser.add_argument("--dense-units", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="v2/exports")
    parser.add_argument("--manifest-dir", default="v2/data/manifests")
    parser.add_argument("--parity-samples", type=int, default=512)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    output_dir = resolve_repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir = resolve_repo_path(args.manifest_dir)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    run_config = RunConfig(
        pretrain_dataset=args.pretrain_dataset,
        finetune_dataset=args.finetune_dataset,
        pretrain_max_examples=args.pretrain_max_examples,
        finetune_max_examples=args.finetune_max_examples,
        validation_max_examples=args.validation_max_examples,
        max_positions_per_image=args.max_positions_per_image,
        window_size=args.window_size,
        max_disp=args.max_disp,
        negatives_per_positive=args.negatives_per_positive,
        pretrain_epochs=args.pretrain_epochs,
        finetune_epochs=args.finetune_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        conv_filters=args.conv_filters,
        dense_units=args.dense_units,
        seed=args.seed,
    )
    (output_dir / "run_config.json").write_text(json.dumps(asdict(run_config), indent=2), encoding="utf-8")

    model = build_patch_student(
        window_size=args.window_size,
        conv_filters=args.conv_filters,
        dense_units=args.dense_units,
        learning_rate=args.learning_rate,
    )

    history_payload: dict[str, list[dict[str, float]]] = {"pretrain": [], "finetune": []}

    if args.pretrain_max_examples > 0:
        pretrain_examples = load_sceneflow_examples(
            args.pretrain_dataset,
            split="train",
            max_examples=args.pretrain_max_examples,
        )
        write_manifest(
            manifest_dir / "sceneflow_train_manifest.json",
            [example.metadata for example in pretrain_examples],
        )
        pretrain_arrays = build_training_examples(
            pretrain_examples,
            window_size=args.window_size,
            max_disp=args.max_disp,
            max_positions_per_image=args.max_positions_per_image,
            negatives_per_positive=args.negatives_per_positive,
            seed=int(rng.integers(0, 1_000_000_000)),
        )
        pretrain_history = model.fit(
            pretrain_arrays.features,
            pretrain_arrays.labels,
            epochs=args.pretrain_epochs,
            batch_size=args.batch_size,
            callbacks=_build_callbacks(checkpoint_dir, stage="pretrain"),
            verbose=2,
        )
        history_payload["pretrain"] = _history_to_records(pretrain_history.history)

    finetune_examples = load_kitti_examples(
        args.finetune_dataset,
        split="train",
        max_examples=args.finetune_max_examples,
    )
    validation_examples = load_kitti_examples(
        args.finetune_dataset,
        split="validation",
        max_examples=args.validation_max_examples,
    )
    write_manifest(
        manifest_dir / "kitti_train_manifest.json",
        [example.metadata for example in finetune_examples],
    )
    write_manifest(
        manifest_dir / "kitti_validation_manifest.json",
        [example.metadata for example in validation_examples],
    )

    validation_arrays = build_training_examples(
        validation_examples,
        window_size=args.window_size,
        max_disp=args.max_disp,
        max_positions_per_image=args.max_positions_per_image,
        negatives_per_positive=args.negatives_per_positive,
        seed=int(rng.integers(0, 1_000_000_000)),
    )

    finetune_arrays = build_training_examples(
        finetune_examples,
        window_size=args.window_size,
        max_disp=args.max_disp,
        max_positions_per_image=args.max_positions_per_image,
        negatives_per_positive=args.negatives_per_positive,
        seed=int(rng.integers(0, 1_000_000_000)),
    )
    finetune_history = model.fit(
        finetune_arrays.features,
        finetune_arrays.labels,
        epochs=args.finetune_epochs,
        batch_size=args.batch_size,
        validation_data=(validation_arrays.features, validation_arrays.labels),
        callbacks=_build_callbacks(checkpoint_dir, stage="finetune"),
        verbose=2,
    )
    history_payload["finetune"] = _history_to_records(finetune_history.history)

    parity_batch = finetune_arrays.features[: min(args.parity_samples, len(finetune_arrays.features))]
    parity_labels = finetune_arrays.labels[: min(args.parity_samples, len(finetune_arrays.labels))]
    parity_path = output_dir / "parity_batch.npz"
    np.savez(parity_path, features=parity_batch.astype(np.float32), labels=parity_labels.astype(np.float32))

    metrics_payload = _evaluate_model(
        model,
        validation_examples,
        output_dir=output_dir,
        window_size=args.window_size,
        max_disp=args.max_disp,
    )

    model_path = output_dir / "student_patch_model.keras"
    model.save(model_path)

    history_path = output_dir / "training_history.json"
    history_path.write_text(json.dumps(history_payload, indent=2), encoding="utf-8")
    metrics_path = output_dir / "validation_metrics.json"
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    print(f"Saved model to {model_path}")
    print(f"Saved parity batch to {parity_path}")
    print(f"Saved history to {history_path}")
    print(f"Saved validation metrics to {metrics_path}")
    print(json.dumps(metrics_payload, indent=2))


def _evaluate_model(
    model: keras.Model,
    validation_examples: list[StereoExample],
    *,
    output_dir: Path,
    window_size: int,
    max_disp: int,
) -> dict[str, object]:
    per_example_metrics: list[dict[str, object]] = []
    aggregate: dict[str, float] = {
        "accuracy_within_1px": 0.0,
        "accuracy_within_3px": 0.0,
        "rmse": 0.0,
        "mae": 0.0,
    }

    for index, example in enumerate(validation_examples):
        predicted_disparity, valid_mask = reconstruct_disparity_map(
            model,
            example.left_image,
            example.right_image,
            window_size=window_size,
            max_disp=max_disp,
        )
        combined_mask = valid_mask & example.valid_mask
        metrics = compute_disparity_metrics(predicted_disparity, example.disparity, combined_mask)
        for key in aggregate:
            aggregate[key] += float(metrics[key])
        entry = {**example.metadata, **metrics}
        per_example_metrics.append(entry)
        if index == 0:
            _save_visualization(output_dir / "validation_example.png", example, predicted_disparity)
            np.save(output_dir / "validation_prediction.npy", predicted_disparity.astype(np.float32))

    count = max(len(per_example_metrics), 1)
    aggregate = {key: value / count for key, value in aggregate.items()}
    aggregate["examples"] = len(per_example_metrics)
    aggregate["per_example"] = per_example_metrics
    return aggregate


def _save_visualization(path: Path, example: StereoExample, predicted_disparity: np.ndarray) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].imshow(example.left_image, cmap="gray", interpolation="nearest")
    axes[0].set_title("Left")
    axes[1].imshow(example.disparity, cmap="plasma", interpolation="nearest")
    axes[1].set_title("Ground Truth")
    axes[2].imshow(predicted_disparity, cmap="plasma", interpolation="nearest")
    axes[2].set_title("Predicted")
    for axis in axes:
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def _history_to_records(history: dict[str, list[float]]) -> list[dict[str, float]]:
    keys = list(history.keys())
    if not keys:
        return []
    length = len(history[keys[0]])
    return [{key: float(history[key][index]) for key in keys} for index in range(length)]


def _build_callbacks(checkpoint_dir: Path, *, stage: str) -> list[keras.callbacks.Callback]:
    return [
        keras.callbacks.TerminateOnNaN(),
        keras.callbacks.CSVLogger(checkpoint_dir / f"{stage}_training_log.csv", append=False),
        keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_dir / f"{stage}_best.keras",
            monitor="val_loss" if stage == "finetune" else "loss",
            save_best_only=True,
            mode="min",
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_dir / f"{stage}_latest.keras",
            monitor="val_loss" if stage == "finetune" else "loss",
            save_best_only=False,
            mode="min",
        ),
    ]


if __name__ == "__main__":
    main()