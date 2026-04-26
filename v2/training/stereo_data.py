from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import json
import tarfile
from typing import Iterable
from zipfile import ZipFile

from datasets import load_dataset
from huggingface_hub import snapshot_download
import numpy as np
from PIL import Image

from v2.training.hf_utils import get_hf_kwargs, resolve_repo_path


@dataclass(frozen=True)
class StereoExample:
    left_image: np.ndarray
    right_image: np.ndarray
    disparity: np.ndarray
    valid_mask: np.ndarray
    metadata: dict[str, str | float | int]


def load_kitti_examples(
    dataset_id: str,
    *,
    split: str,
    max_examples: int,
) -> list[StereoExample]:
    dataset = load_dataset(dataset_id, split=split, streaming=True, **get_hf_kwargs())
    examples: list[StereoExample] = []

    for sample in dataset:
        left_image = _to_grayscale(_decode_image_field(sample["left_image"]))
        right_image = _to_grayscale(_decode_image_field(sample["right_image"]))
        disparity = _decode_disparity_field(sample["disparity"])
        valid_mask = np.isfinite(disparity) & (disparity >= 0.0)
        examples.append(
            StereoExample(
                left_image=left_image,
                right_image=right_image,
                disparity=disparity,
                valid_mask=valid_mask,
                metadata={
                    "dataset": dataset_id,
                    "split": split,
                    "sequence_id": str(sample.get("sequence_id", "")),
                    "image_id": str(sample.get("image_id", len(examples))),
                    "fx": float(sample.get("fx", 0.0)),
                    "baseline": float(sample.get("baseline", 0.0)),
                },
            )
        )
        if len(examples) >= max_examples:
            break

    return examples


def load_sceneflow_examples(
    dataset_id: str,
    *,
    split: str,
    max_examples: int,
    cache_dir: str | Path = "v2/data/raw/sceneflow",
) -> list[StereoExample]:
    cache_path = resolve_repo_path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    local_root = Path(
        snapshot_download(
            repo_id=dataset_id,
            repo_type="dataset",
            allow_patterns=["frames_clean.tar", "disparity.tar"],
            local_dir=cache_path,
            **get_hf_kwargs(),
        )
    )
    frames_tar_path = local_root / "frames_clean.tar"
    disparity_tar_path = local_root / "disparity.tar"

    split_tag = split.upper()
    manifest_entries = _build_sceneflow_manifest(frames_tar_path, disparity_tar_path, split_tag)
    examples: list[StereoExample] = []
    with tarfile.open(frames_tar_path, "r") as frames_tar, tarfile.open(disparity_tar_path, "r") as disparity_tar:
        for entry in manifest_entries[:max_examples]:
            left_image = _to_grayscale(_read_tar_image(frames_tar, entry["left"]))
            right_image = _to_grayscale(_read_tar_image(frames_tar, entry["right"]))
            disparity = _read_pfm(_read_tar_bytes(disparity_tar, entry["disparity"]))
            valid_mask = np.isfinite(disparity) & (disparity >= 0.0)
            examples.append(
                StereoExample(
                    left_image=left_image,
                    right_image=right_image,
                    disparity=disparity,
                    valid_mask=valid_mask,
                    metadata={
                        "dataset": dataset_id,
                        "split": split,
                        "left_member": entry["left"],
                    },
                )
            )
    return examples


def get_sceneflow_manifest(
    dataset_id: str,
    *,
    split: str | None = None,
    cache_dir: str | Path = "v2/data/raw/sceneflow",
) -> tuple[Path, list[dict[str, str]]]:
    cache_path = resolve_repo_path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    local_root = Path(
        snapshot_download(
            repo_id=dataset_id,
            repo_type="dataset",
            allow_patterns=["frames_clean.tar", "disparity.tar"],
            local_dir=cache_path,
            **get_hf_kwargs(),
        )
    )
    frames_tar_path = local_root / "frames_clean.tar"
    disparity_tar_path = local_root / "disparity.tar"
    split_tag = split.upper() if split else ""
    manifest_entries = _build_sceneflow_manifest(frames_tar_path, disparity_tar_path, split_tag)
    return local_root, manifest_entries


def load_sceneflow_examples_from_manifest(
    manifest_entries: Iterable[dict[str, str]],
    *,
    cache_dir: str | Path = "v2/data/raw/sceneflow",
    dataset_id: str = "olivermao/sceneflow",
) -> list[StereoExample]:
    cache_path = resolve_repo_path(cache_dir)
    frames_tar_path = cache_path / "frames_clean.tar"
    disparity_tar_path = cache_path / "disparity.tar"
    entries = list(manifest_entries)

    if not frames_tar_path.exists() or not disparity_tar_path.exists():
        local_root, _ = get_sceneflow_manifest(dataset_id, split=None, cache_dir=cache_dir)
        frames_tar_path = local_root / "frames_clean.tar"
        disparity_tar_path = local_root / "disparity.tar"

    examples: list[StereoExample] = []
    with tarfile.open(frames_tar_path, "r") as frames_tar, tarfile.open(disparity_tar_path, "r") as disparity_tar:
        for entry in entries:
            left_image = _to_grayscale(_read_tar_image(frames_tar, entry["left"]))
            right_image = _to_grayscale(_read_tar_image(frames_tar, entry["right"]))
            disparity = _read_pfm(_read_tar_bytes(disparity_tar, entry["disparity"]))
            valid_mask = np.isfinite(disparity) & (disparity >= 0.0)
            examples.append(
                StereoExample(
                    left_image=left_image,
                    right_image=right_image,
                    disparity=disparity,
                    valid_mask=valid_mask,
                    metadata={
                        "dataset": dataset_id,
                        "left_member": entry["left"],
                        "right_member": entry["right"],
                        "disparity_member": entry["disparity"],
                    },
                )
            )
    return examples


def write_manifest(path_like: str | Path, entries: list[dict[str, str | float | int]]) -> Path:
    path = resolve_repo_path(path_like)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    return path


def _decode_image_field(value) -> np.ndarray:
    if isinstance(value, bytes):
        with Image.open(BytesIO(value)) as image:
            return np.array(image)
    return np.asarray(value)


def _decode_disparity_field(value) -> np.ndarray:
    if isinstance(value, bytes) and value.startswith(b"PK"):
        with ZipFile(BytesIO(value)) as archive:
            member = archive.namelist()[0]
            with archive.open(member) as handle:
                return np.load(handle).astype(np.float32)

    if isinstance(value, bytes):
        with Image.open(BytesIO(value)) as image:
            disparity = np.array(image)
        if disparity.dtype == np.uint16:
            return disparity.astype(np.float32) / 256.0
        return disparity.astype(np.float32)

    return np.asarray(value, dtype=np.float32)


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image.astype(np.float32)
    rgb = image[..., :3].astype(np.float32)
    return 0.2989 * rgb[..., 0] + 0.5870 * rgb[..., 1] + 0.1140 * rgb[..., 2]


def _build_sceneflow_manifest(
    frames_tar_path: Path,
    disparity_tar_path: Path,
    split_tag: str,
) -> list[dict[str, str]]:
    with tarfile.open(frames_tar_path, "r") as frames_tar, tarfile.open(disparity_tar_path, "r") as disparity_tar:
        frame_names = {member.name for member in frames_tar.getmembers() if member.isfile()}
        disparity_names = {member.name for member in disparity_tar.getmembers() if member.isfile()}

    manifest_entries: list[dict[str, str]] = []
    for left_name in sorted(frame_names):
        upper_name = left_name.upper()
        if split_tag and f"/{split_tag}/" not in upper_name:
            continue
        if not left_name.endswith(".png"):
            continue
        if not (left_name.startswith("left/") or "/left/" in left_name):
            continue

        if left_name.startswith("left/"):
            right_name = left_name.replace("left/", "right/", 1)
        else:
            right_name = left_name.replace("/left/", "/right/")
        disparity_name = _guess_sceneflow_disparity_name(left_name)
        if right_name in frame_names and disparity_name in disparity_names:
            manifest_entries.append({"left": left_name, "right": right_name, "disparity": disparity_name})

    if not manifest_entries:
        raise ValueError(f"No Scene Flow examples found for split {split_tag!r}")

    return manifest_entries


def _guess_sceneflow_disparity_name(left_name: str) -> str:
    name = left_name.replace("frames_cleanpass", "disparity")
    name = name.replace("frames_clean", "disparity")
    return name[:-4] + ".pfm"


def _read_tar_bytes(archive: tarfile.TarFile, member_name: str) -> bytes:
    member = archive.getmember(member_name)
    handle = archive.extractfile(member)
    if handle is None:
        raise FileNotFoundError(member_name)
    return handle.read()


def _read_tar_image(archive: tarfile.TarFile, member_name: str) -> np.ndarray:
    with Image.open(BytesIO(_read_tar_bytes(archive, member_name))) as image:
        return np.array(image)


def _read_pfm(payload: bytes) -> np.ndarray:
    stream = BytesIO(payload)
    header = stream.readline().decode("ascii").strip()
    if header not in {"Pf", "PF"}:
        raise ValueError(f"Unsupported PFM header: {header}")
    width, height = map(int, stream.readline().decode("ascii").strip().split())
    scale = float(stream.readline().decode("ascii").strip())
    endian = "<" if scale < 0 else ">"
    channels = 3 if header == "PF" else 1
    data = np.frombuffer(stream.read(), dtype=endian + "f")
    array = np.reshape(data, (height, width, channels) if channels == 3 else (height, width))
    array = np.flipud(array)
    if channels == 3:
        array = array[..., 0]
    return array.astype(np.float32)