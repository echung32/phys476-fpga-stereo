from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from v2.training.stereo_data import StereoExample


def save_training_input_grid(
    path: Path,
    rows: list[tuple[str, StereoExample]],
) -> None:
    figure, axes = plt.subplots(len(rows), 3, figsize=(12, 4 * len(rows)))
    if len(rows) == 1:
        axes = np.array([axes])

    for row_index, (label, example) in enumerate(rows):
        axes[row_index, 0].imshow(example.left_image, cmap="gray", interpolation="nearest")
        axes[row_index, 0].set_title(f"{label} Left")
        axes[row_index, 1].imshow(example.right_image, cmap="gray", interpolation="nearest")
        axes[row_index, 1].set_title(f"{label} Right")
        axes[row_index, 2].imshow(example.disparity, cmap="plasma", interpolation="nearest")
        axes[row_index, 2].set_title(f"{label} Disparity")
        for axis in axes[row_index]:
            axis.axis("off")

    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def save_prediction_grid(
    path: Path,
    rows: list[tuple[StereoExample, np.ndarray, np.ndarray]],
) -> None:
    figure, axes = plt.subplots(len(rows), 4, figsize=(16, 4 * len(rows)))
    if len(rows) == 1:
        axes = np.array([axes])

    for row_index, (example, teacher_prediction, student_prediction) in enumerate(rows):
        finite_values = np.concatenate(
            [
                example.disparity[np.isfinite(example.disparity)],
                teacher_prediction[np.isfinite(teacher_prediction)],
                student_prediction[np.isfinite(student_prediction)],
            ]
        )
        vmin = float(np.min(finite_values)) if finite_values.size else 0.0
        vmax = float(np.max(finite_values)) if finite_values.size else 1.0

        axes[row_index, 0].imshow(example.left_image, cmap="gray", interpolation="nearest")
        axes[row_index, 0].set_title(f"{example.metadata.get('image_id', row_index)} Left")
        axes[row_index, 1].imshow(example.disparity, cmap="plasma", interpolation="nearest", vmin=vmin, vmax=vmax)
        axes[row_index, 1].set_title("Ground Truth")
        axes[row_index, 2].imshow(teacher_prediction, cmap="plasma", interpolation="nearest", vmin=vmin, vmax=vmax)
        axes[row_index, 2].set_title("Teacher Prediction")
        axes[row_index, 3].imshow(student_prediction, cmap="plasma", interpolation="nearest", vmin=vmin, vmax=vmax)
        axes[row_index, 3].set_title("Student Prediction")
        for axis in axes[row_index]:
            axis.axis("off")

    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)