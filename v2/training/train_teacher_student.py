from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import math
import os
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")

import keras
import numpy as np

from v2.hls4ml.convert_student import main as convert_student_main
from v2.models.keras_student import build_patch_student
from v2.models.keras_teacher import build_patch_teacher
from v2.training.hf_utils import resolve_repo_path
from v2.training.patch_dataset import build_training_examples, compute_disparity_metrics, reconstruct_disparity_map
from v2.training.stereo_data import (
    StereoExample,
    get_sceneflow_manifest,
    load_kitti_examples,
    load_sceneflow_examples_from_manifest,
    write_manifest,
)
from v2.training.visualization import save_prediction_grid, save_training_input_grid


@dataclass(frozen=True)
class TeacherStudentRunConfig:
    pretrain_dataset: str
    finetune_dataset: str
    sceneflow_epochs: int
    sceneflow_chunk_examples: int
    sceneflow_positions_per_image: int
    sceneflow_negatives_per_positive: int
    sceneflow_manifest_limit: int
    kitti_positions_per_image: int
    kitti_negatives_per_positive: int
    teacher_finetune_epochs: int
    student_finetune_epochs: int
    batch_size: int
    teacher_conv_filters: int
    teacher_dense_units: int
    student_conv_filters: int
    student_dense_units: int
    teacher_learning_rate: float
    student_learning_rate: float
    distill_alpha: float
    distill_temperature: float
    seed: int
    output_dir: str


def main() -> None:
    parser = argparse.ArgumentParser(description="Train teacher and student stereo models with Scene Flow pretraining and KITTI fine-tuning")
    parser.add_argument("--pretrain-dataset", default="olivermao/sceneflow")
    parser.add_argument("--finetune-dataset", default="UniflexAI/mini_kitti")
    parser.add_argument("--sceneflow-epochs", type=int, default=1)
    parser.add_argument("--sceneflow-chunk-examples", type=int, default=128)
    parser.add_argument("--sceneflow-positions-per-image", type=int, default=64)
    parser.add_argument("--sceneflow-negatives-per-positive", type=int, default=3)
    parser.add_argument("--sceneflow-manifest-limit", type=int, default=0)
    parser.add_argument("--kitti-positions-per-image", type=int, default=1024)
    parser.add_argument("--kitti-negatives-per-positive", type=int, default=3)
    parser.add_argument("--teacher-finetune-epochs", type=int, default=20)
    parser.add_argument("--student-finetune-epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--teacher-conv-filters", type=int, default=32)
    parser.add_argument("--teacher-dense-units", type=int, default=128)
    parser.add_argument("--student-conv-filters", type=int, default=16)
    parser.add_argument("--student-dense-units", type=int, default=64)
    parser.add_argument("--teacher-learning-rate", type=float, default=5e-4)
    parser.add_argument("--student-learning-rate", type=float, default=1e-3)
    parser.add_argument("--distill-alpha", type=float, default=0.5)
    parser.add_argument("--distill-temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--output-root", default="logs")
    parser.add_argument("--run-name", default="scene_teacher_student")
    args = parser.parse_args()

    if args.output_dir:
        run_dir = resolve_repo_path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = resolve_repo_path(Path(args.output_root) / f"{timestamp}_{args.run_name}")
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir = run_dir / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    hls_dir = run_dir / "hls4ml"
    hls_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    config = TeacherStudentRunConfig(
        pretrain_dataset=args.pretrain_dataset,
        finetune_dataset=args.finetune_dataset,
        sceneflow_epochs=args.sceneflow_epochs,
        sceneflow_chunk_examples=args.sceneflow_chunk_examples,
        sceneflow_positions_per_image=args.sceneflow_positions_per_image,
        sceneflow_negatives_per_positive=args.sceneflow_negatives_per_positive,
        sceneflow_manifest_limit=args.sceneflow_manifest_limit,
        kitti_positions_per_image=args.kitti_positions_per_image,
        kitti_negatives_per_positive=args.kitti_negatives_per_positive,
        teacher_finetune_epochs=args.teacher_finetune_epochs,
        student_finetune_epochs=args.student_finetune_epochs,
        batch_size=args.batch_size,
        teacher_conv_filters=args.teacher_conv_filters,
        teacher_dense_units=args.teacher_dense_units,
        student_conv_filters=args.student_conv_filters,
        student_dense_units=args.student_dense_units,
        teacher_learning_rate=args.teacher_learning_rate,
        student_learning_rate=args.student_learning_rate,
        distill_alpha=args.distill_alpha,
        distill_temperature=args.distill_temperature,
        seed=args.seed,
        output_dir=str(run_dir),
    )
    (run_dir / "run_config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")

    teacher = build_patch_teacher(
        window_size=5,
        conv_filters=args.teacher_conv_filters,
        dense_units=args.teacher_dense_units,
        learning_rate=args.teacher_learning_rate,
    )
    student = build_patch_student(
        window_size=5,
        conv_filters=args.student_conv_filters,
        dense_units=args.student_dense_units,
        learning_rate=args.student_learning_rate,
    )

    print(f"Run directory: {run_dir}", flush=True)
    print("Using Scene Flow for large-scale synthetic pretraining and mini_kitti for real-domain fine-tuning/evaluation.", flush=True)

    finetune_examples = load_kitti_examples(args.finetune_dataset, split="train", max_examples=88)
    validation_examples = load_kitti_examples(args.finetune_dataset, split="validation", max_examples=88)
    write_manifest(manifests_dir / "kitti_train_manifest.json", [example.metadata for example in finetune_examples])
    write_manifest(manifests_dir / "kitti_validation_manifest.json", [example.metadata for example in validation_examples])

    _, sceneflow_manifest = get_sceneflow_manifest(args.pretrain_dataset, split=None)
    if args.sceneflow_manifest_limit > 0:
        sceneflow_manifest = sceneflow_manifest[: args.sceneflow_manifest_limit]
    write_manifest(manifests_dir / "sceneflow_manifest.json", sceneflow_manifest)
    print(f"Scene Flow manifest entries: {len(sceneflow_manifest)}", flush=True)

    preview_scene = load_sceneflow_examples_from_manifest(sceneflow_manifest[:2], dataset_id=args.pretrain_dataset)
    save_training_input_grid(
        run_dir / "training_input_grid.png",
        [
            ("Scene Flow A", preview_scene[0]),
            ("Scene Flow B", preview_scene[1]),
            ("KITTI Train A", finetune_examples[0]),
            ("KITTI Train B", finetune_examples[1]),
        ],
    )

    stage_history: dict[str, list[dict[str, float | int]]] = {"teacher_sceneflow": [], "teacher_kitti": [], "student_sceneflow": [], "student_kitti": []}
    teacher_kitti_val_arrays = build_training_examples(
        validation_examples[: min(16, len(validation_examples))],
        max_positions_per_image=256,
        negatives_per_positive=args.kitti_negatives_per_positive,
        seed=int(rng.integers(0, 1_000_000_000)),
    )

    _train_sceneflow_stage(
        model=teacher,
        stage_name="teacher_sceneflow",
        manifest=sceneflow_manifest,
        chunk_examples=args.sceneflow_chunk_examples,
        positions_per_image=args.sceneflow_positions_per_image,
        negatives_per_positive=args.sceneflow_negatives_per_positive,
        batch_size=args.batch_size,
        rng=rng,
        epochs=args.sceneflow_epochs,
        history=stage_history,
        checkpoints_dir=checkpoints_dir,
        validation_arrays=teacher_kitti_val_arrays,
    )

    teacher_train_arrays = build_training_examples(
        finetune_examples,
        max_positions_per_image=args.kitti_positions_per_image,
        negatives_per_positive=args.kitti_negatives_per_positive,
        seed=int(rng.integers(0, 1_000_000_000)),
    )
    teacher_finetune_history = teacher.fit(
        teacher_train_arrays.features,
        teacher_train_arrays.labels,
        validation_data=(teacher_kitti_val_arrays.features, teacher_kitti_val_arrays.labels),
        epochs=args.teacher_finetune_epochs,
        batch_size=args.batch_size,
        callbacks=_build_callbacks(checkpoints_dir, "teacher_kitti"),
        verbose=2,
    )
    stage_history["teacher_kitti"] = _history_to_records(teacher_finetune_history.history)
    teacher.save(run_dir / "teacher_patch_model.keras")

    _train_sceneflow_stage(
        model=student,
        stage_name="student_sceneflow",
        manifest=sceneflow_manifest,
        chunk_examples=args.sceneflow_chunk_examples,
        positions_per_image=args.sceneflow_positions_per_image,
        negatives_per_positive=args.sceneflow_negatives_per_positive,
        batch_size=args.batch_size,
        rng=rng,
        epochs=args.sceneflow_epochs,
        history=stage_history,
        checkpoints_dir=checkpoints_dir,
        validation_arrays=teacher_kitti_val_arrays,
        teacher=teacher,
        distill_alpha=args.distill_alpha,
        distill_temperature=args.distill_temperature,
    )

    student_train_arrays = build_training_examples(
        finetune_examples,
        max_positions_per_image=args.kitti_positions_per_image,
        negatives_per_positive=args.kitti_negatives_per_positive,
        seed=int(rng.integers(0, 1_000_000_000)),
    )
    distill_targets = _distill_targets(
        teacher,
        student_train_arrays.features,
        student_train_arrays.labels,
        alpha=args.distill_alpha,
        temperature=args.distill_temperature,
        batch_size=args.batch_size,
    )
    student_finetune_history = student.fit(
        student_train_arrays.features,
        distill_targets,
        validation_data=(teacher_kitti_val_arrays.features, teacher_kitti_val_arrays.labels),
        epochs=args.student_finetune_epochs,
        batch_size=args.batch_size,
        callbacks=_build_callbacks(checkpoints_dir, "student_kitti"),
        verbose=2,
    )
    stage_history["student_kitti"] = _history_to_records(student_finetune_history.history)

    student_path = run_dir / "student_patch_model.keras"
    teacher_path = run_dir / "teacher_patch_model.keras"
    student.save(student_path)
    teacher.save(teacher_path)

    parity_path = run_dir / "parity_batch.npz"
    np.savez(
        parity_path,
        features=student_train_arrays.features[: min(4096, len(student_train_arrays.features))].astype(np.float32),
        labels=student_train_arrays.labels[: min(4096, len(student_train_arrays.labels))].astype(np.float32),
    )

    metrics_payload = _evaluate_models(teacher, student, validation_examples, run_dir)
    (run_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    (run_dir / "training_history.json").write_text(json.dumps(stage_history, indent=2), encoding="utf-8")

    original_argv = list(os.sys.argv)
    try:
        os.sys.argv = [
            "convert_student",
            "--model",
            str(student_path),
            "--parity-batch",
            str(parity_path),
            "--output-dir",
            str(hls_dir),
        ]
        convert_student_main()
    finally:
        os.sys.argv = original_argv

    print(json.dumps(metrics_payload, indent=2), flush=True)


def _train_sceneflow_stage(
    *,
    model: keras.Model,
    stage_name: str,
    manifest: list[dict[str, str]],
    chunk_examples: int,
    positions_per_image: int,
    negatives_per_positive: int,
    batch_size: int,
    rng: np.random.Generator,
    epochs: int,
    history: dict[str, list[dict[str, float | int]]],
    checkpoints_dir: Path,
    validation_arrays,
    teacher: keras.Model | None = None,
    distill_alpha: float = 0.5,
    distill_temperature: float = 1.0,
) -> None:
    chunk_count = math.ceil(len(manifest) / chunk_examples)
    for epoch in range(epochs):
        print(f"{stage_name} epoch {epoch + 1}/{epochs}", flush=True)
        for chunk_index in range(chunk_count):
            chunk_manifest = manifest[chunk_index * chunk_examples : (chunk_index + 1) * chunk_examples]
            examples = load_sceneflow_examples_from_manifest(chunk_manifest)
            arrays = build_training_examples(
                examples,
                max_positions_per_image=positions_per_image,
                negatives_per_positive=negatives_per_positive,
                seed=int(rng.integers(0, 1_000_000_000)),
            )
            targets = arrays.labels
            if teacher is not None:
                targets = _distill_targets(
                    teacher,
                    arrays.features,
                    arrays.labels,
                    alpha=distill_alpha,
                    temperature=distill_temperature,
                    batch_size=batch_size,
                )
            fit_history = model.fit(
                arrays.features,
                targets,
                validation_data=(validation_arrays.features, validation_arrays.labels),
                epochs=1,
                batch_size=batch_size,
                verbose=0,
            )
            record = {key: float(values[-1]) for key, values in fit_history.history.items()}
            record.update({"epoch": epoch + 1, "chunk": chunk_index + 1, "chunks": chunk_count})
            history[stage_name].append(record)
            print(
                f"{stage_name} chunk {chunk_index + 1}/{chunk_count}: loss={record.get('loss', 0.0):.4f} val_loss={record.get('val_loss', 0.0):.4f}",
                flush=True,
            )
            if (chunk_index + 1) % 10 == 0 or chunk_index + 1 == chunk_count:
                model.save(checkpoints_dir / f"{stage_name}_epoch{epoch + 1:02d}_chunk{chunk_index + 1:04d}.keras")


def _distill_targets(
    teacher: keras.Model,
    features: np.ndarray,
    hard_labels: np.ndarray,
    *,
    alpha: float,
    temperature: float,
    batch_size: int,
) -> np.ndarray:
    teacher_logits = teacher.predict(features, batch_size=batch_size, verbose=0).reshape(-1)
    clipped = np.clip(teacher_logits, 1e-6, 1.0 - 1e-6)
    if temperature != 1.0:
        logits = np.log(clipped / (1.0 - clipped)) / temperature
        softened = 1.0 / (1.0 + np.exp(-logits))
    else:
        softened = clipped
    return (alpha * hard_labels + (1.0 - alpha) * softened).astype(np.float32)


def _build_callbacks(checkpoints_dir: Path, prefix: str) -> list[keras.callbacks.Callback]:
    return [
        keras.callbacks.TerminateOnNaN(),
        keras.callbacks.CSVLogger(checkpoints_dir / f"{prefix}_training_log.csv", append=False),
        keras.callbacks.ModelCheckpoint(
            filepath=checkpoints_dir / f"{prefix}_best.keras",
            monitor="val_loss",
            save_best_only=True,
            mode="min",
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=checkpoints_dir / f"{prefix}_latest.keras",
            monitor="val_loss",
            save_best_only=False,
            mode="min",
        ),
    ]


def _history_to_records(history: dict[str, list[float]]) -> list[dict[str, float]]:
    keys = list(history.keys())
    if not keys:
        return []
    length = len(history[keys[0]])
    return [{key: float(history[key][index]) for key in keys} for index in range(length)]


def _evaluate_models(
    teacher: keras.Model,
    student: keras.Model,
    validation_examples: list[StereoExample],
    run_dir: Path,
) -> dict[str, object]:
    teacher_rows: list[dict[str, object]] = []
    student_rows: list[dict[str, object]] = []
    prediction_rows: list[tuple[StereoExample, np.ndarray, np.ndarray]] = []

    for example in validation_examples[:8]:
        teacher_prediction, teacher_mask = reconstruct_disparity_map(teacher, example.left_image, example.right_image)
        student_prediction, student_mask = reconstruct_disparity_map(student, example.left_image, example.right_image)
        teacher_metrics = compute_disparity_metrics(teacher_prediction, example.disparity, teacher_mask & example.valid_mask)
        student_metrics = compute_disparity_metrics(student_prediction, example.disparity, student_mask & example.valid_mask)
        teacher_rows.append({**example.metadata, **teacher_metrics})
        student_rows.append({**example.metadata, **student_metrics})
        prediction_rows.append((example, teacher_prediction.astype(np.float32), student_prediction.astype(np.float32)))

    save_prediction_grid(run_dir / "student_teacher_results.png", prediction_rows[:4])
    return {
        "teacher_average": _average_metrics(teacher_rows),
        "student_average": _average_metrics(student_rows),
        "teacher_per_example": teacher_rows,
        "student_per_example": student_rows,
    }


def _average_metrics(rows: list[dict[str, object]]) -> dict[str, float | int]:
    if not rows:
        return {"examples": 0}
    keys = ["accuracy_within_1px", "accuracy_within_3px", "rmse", "mae"]
    averaged = {key: float(np.mean([float(row[key]) for row in rows])) for key in keys}
    averaged["examples"] = len(rows)
    return averaged


if __name__ == "__main__":
    main()