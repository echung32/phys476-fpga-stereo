from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from v2.training.stereo_data import StereoExample


@dataclass(frozen=True)
class PatchTrainingExamples:
    features: np.ndarray
    labels: np.ndarray
    metadata: list[dict[str, str | float | int]]


def build_training_examples(
    stereo_examples: list[StereoExample],
    *,
    window_size: int = 5,
    max_disp: int = 16,
    max_positions_per_image: int = 512,
    negatives_per_positive: int = 1,
    seed: int = 0,
) -> PatchTrainingExamples:
    feature_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []
    metadata: list[dict[str, str | float | int]] = []
    rng = np.random.default_rng(seed)

    for example_index, example in enumerate(stereo_examples):
        sampled = sample_training_examples(
            example.left_image,
            example.right_image,
            example.disparity,
            valid_mask=example.valid_mask,
            window_size=window_size,
            max_disp=max_disp,
            max_positions=max_positions_per_image,
            negatives_per_positive=negatives_per_positive,
            seed=int(rng.integers(0, 1_000_000_000)),
        )
        feature_chunks.append(sampled.features)
        label_chunks.append(sampled.labels)
        metadata.extend(
            {
                **example.metadata,
                "example_index": example_index,
                "row": int(position["row"]),
                "col": int(position["col"]),
            }
            for position in sampled.metadata
        )

    if not feature_chunks:
        raise ValueError("No stereo examples were provided for training")

    return PatchTrainingExamples(
        features=np.concatenate(feature_chunks, axis=0),
        labels=np.concatenate(label_chunks, axis=0),
        metadata=metadata,
    )


def sample_training_examples(
    left_image: np.ndarray,
    right_image: np.ndarray,
    disparity_map: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
    window_size: int = 5,
    max_disp: int = 16,
    max_positions: int = 4096,
    negatives_per_positive: int = 1,
    seed: int = 0,
) -> PatchTrainingExamples:
    radius = window_size // 2
    left_windows = sliding_window_view(left_image, (window_size, window_size))
    right_windows = sliding_window_view(right_image, (window_size, window_size))

    num_rows, num_cols = left_windows.shape[:2]
    center_disparity = disparity_map[radius : radius + num_rows, radius : radius + num_cols].astype(np.int16)
    if valid_mask is None:
        center_valid = np.ones_like(center_disparity, dtype=bool)
    else:
        center_valid = valid_mask[radius : radius + num_rows, radius : radius + num_cols]

    col_indices = np.broadcast_to(np.arange(num_cols), (num_rows, num_cols))
    candidate_mask = center_valid & (center_disparity >= 0) & (center_disparity < max_disp) & (col_indices >= center_disparity)
    valid_positions = np.argwhere(candidate_mask)
    if valid_positions.size == 0:
        raise ValueError("No valid training positions were found for the selected disparity range")

    rng = np.random.default_rng(seed)
    if max_positions < len(valid_positions):
        sampled_indices = rng.choice(len(valid_positions), size=max_positions, replace=False)
        chosen_positions = valid_positions[sampled_indices]
    else:
        chosen_positions = valid_positions

    candidate_disparities = np.arange(max_disp, dtype=np.int16)
    feature_list: list[np.ndarray] = []
    label_list: list[float] = []
    metadata: list[tuple[int, int]] = []

    for row_idx, col_idx in chosen_positions:
        true_disp = int(center_disparity[row_idx, col_idx])
        left_patch = left_windows[row_idx, col_idx].astype(np.float32)

        right_patch = right_windows[row_idx, col_idx - true_disp]
        feature_list.append(_combine_patches(left_patch, right_patch))
        label_list.append(1.0)
        metadata.append((int(row_idx + radius), int(col_idx + radius)))

        negative_choices = candidate_disparities[candidate_disparities != true_disp]
        sample_count = min(negatives_per_positive, negative_choices.size)
        sampled_negatives = rng.choice(negative_choices, size=sample_count, replace=False)
        for candidate_disp in np.atleast_1d(sampled_negatives):
            right_patch = right_windows[row_idx, col_idx - int(candidate_disp)]
            feature_list.append(_combine_patches(left_patch, right_patch))
            label_list.append(0.0)
            metadata.append((int(row_idx + radius), int(col_idx + radius)))

    features = np.asarray(feature_list, dtype=np.float32)
    labels = np.asarray(label_list, dtype=np.float32)
    return PatchTrainingExamples(
        features=features,
        labels=labels,
        metadata=[{"row": row, "col": col} for row, col in metadata],
    )


def reconstruct_disparity_map(
    model,
    left_image: np.ndarray,
    right_image: np.ndarray,
    *,
    window_size: int = 5,
    max_disp: int = 16,
    batch_size: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    radius = window_size // 2
    left_windows = sliding_window_view(left_image, (window_size, window_size))
    right_windows = sliding_window_view(right_image, (window_size, window_size))

    num_rows, num_cols = left_windows.shape[:2]
    start_col = max_disp - 1
    if start_col >= num_cols:
        raise ValueError("Disparity range exceeds the available image width")

    left_valid = left_windows[:, start_col:]
    flat_left = left_valid.reshape(-1, window_size * window_size).astype(np.float32) / 255.0
    score_planes = np.empty((max_disp, left_valid.shape[0], left_valid.shape[1]), dtype=np.float32)

    for disparity in range(max_disp):
        right_slice = right_windows[:, start_col - disparity : num_cols - disparity]
        flat_right = right_slice.reshape(-1, window_size * window_size).astype(np.float32) / 255.0
        features = np.concatenate((flat_left, flat_right), axis=1).reshape(-1, window_size, window_size, 2)
        scores = model.predict(features, batch_size=batch_size, verbose=0).reshape(-1)
        score_planes[disparity] = scores.reshape(left_valid.shape[0], left_valid.shape[1])

    predicted = score_planes.argmax(axis=0).astype(np.uint8)
    disparity_out = np.zeros_like(left_image, dtype=np.uint8)
    valid_mask = np.zeros_like(left_image, dtype=bool)

    row_slice = slice(radius, radius + predicted.shape[0])
    col_slice = slice(radius + start_col, radius + start_col + predicted.shape[1])
    disparity_out[row_slice, col_slice] = predicted
    valid_mask[row_slice, col_slice] = True
    return disparity_out, valid_mask


def compute_disparity_metrics(
    predicted_disparity: np.ndarray,
    ground_truth: np.ndarray,
    valid_mask: np.ndarray,
) -> dict[str, float | int]:
    comparison_mask = valid_mask & np.isfinite(ground_truth) & (ground_truth >= 0.0)
    if not np.any(comparison_mask):
        raise ValueError("No valid pixels were available for metric computation")

    errors = np.abs(predicted_disparity.astype(np.float32) - ground_truth.astype(np.float32))
    return {
        "valid_pixels": int(comparison_mask.sum()),
        "accuracy_within_1px": float(np.mean(errors[comparison_mask] <= 1.0)),
        "accuracy_within_3px": float(np.mean(errors[comparison_mask] <= 3.0)),
        "rmse": float(np.sqrt(np.mean(np.square(errors[comparison_mask], dtype=np.float32)))),
        "mae": float(np.mean(errors[comparison_mask])),
    }


def _combine_patches(left_patch: np.ndarray, right_patch: np.ndarray) -> np.ndarray:
    left_norm = left_patch.astype(np.float32) / 255.0
    right_norm = right_patch.astype(np.float32) / 255.0
    return np.stack((left_norm, right_norm), axis=-1)