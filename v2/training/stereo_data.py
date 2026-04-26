"""Stereo dataset loading utilities.

Supports:
  - Scene Flow (olivermao/sceneflow HF tar archives)
  - DrivingStereo (local disk, official download layout)
  - KITTI 2015 / KITTI 2012 (local disk, official download layout)
  - mini_kitti / any KITTI-style HuggingFace dataset (streaming)

All loaders return StereoExample with grayscale float32 images in [0, 1].

Mixed-dataset manifests:
  Each manifest entry is a dict with a "source" key that identifies the
  dataset loader, plus source-specific path fields.  Use
  ``build_mixed_manifest`` to construct a shuffled mix, and
  ``load_example_from_manifest_entry`` to materialise any entry.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import json
import tarfile
from typing import Iterable, Iterator
from zipfile import ZipFile

from datasets import load_dataset
from huggingface_hub import snapshot_download
import numpy as np
from PIL import Image

from v2.training.hf_utils import get_hf_kwargs, resolve_repo_path


# ---------------------------------------------------------------------------
# Core data type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StereoExample:
    left_image: np.ndarray    # (H, W) float32 in [0, 1]
    right_image: np.ndarray   # (H, W) float32 in [0, 1]
    disparity: np.ndarray     # (H, W) float32 pixels
    valid_mask: np.ndarray    # (H, W) bool
    metadata: dict[str, str | float | int]


# ---------------------------------------------------------------------------
# Scene Flow (olivermao/sceneflow HF tar archives)
# ---------------------------------------------------------------------------

def get_sceneflow_manifest(
    dataset_id: str = "olivermao/sceneflow",
    *,
    split: str | None = None,
    cache_dir: str | Path = "v2/data/raw/sceneflow",
) -> tuple[Path, list[dict[str, str]]]:
    """Download Scene Flow tars (if needed) and return (local_root, manifest).

    Each manifest entry: {"source": "sceneflow", "left": ..., "right": ..., "disparity": ...}
    """
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
    entries = _build_sceneflow_manifest(frames_tar_path, disparity_tar_path, split_tag)
    tagged = [
        {"source": "sceneflow", "dataset_id": dataset_id, "tar_root": str(local_root), **e}
        for e in entries
    ]
    return local_root, tagged


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
            if left_image.max() > 1.0:
                left_image = left_image / 255.0
                right_image = right_image / 255.0
            valid_mask = np.isfinite(disparity) & (disparity >= 0.0)
            examples.append(
                StereoExample(
                    left_image=left_image,
                    right_image=right_image,
                    disparity=disparity,
                    valid_mask=valid_mask,
                    metadata={
                        "source": "sceneflow",
                        "dataset_id": dataset_id,
                        "left_member": entry["left"],
                        "right_member": entry.get("right", ""),
                        "disparity_member": entry.get("disparity", ""),
                    },
                )
            )
    return examples


# ---------------------------------------------------------------------------
# KITTI-style HuggingFace dataset (mini_kitti and similar)
# ---------------------------------------------------------------------------

def load_kitti_hf_examples(
    dataset_id: str,
    *,
    split: str,
    max_examples: int = 0,
) -> list[StereoExample]:
    """Load from a KITTI-style HuggingFace streaming dataset."""
    dataset = load_dataset(dataset_id, split=split, streaming=True, **get_hf_kwargs())
    examples: list[StereoExample] = []

    for sample in dataset:
        left_image = _to_grayscale(_decode_image_field(sample["left_image"]))
        right_image = _to_grayscale(_decode_image_field(sample["right_image"]))
        disparity = _decode_disparity_field(sample["disparity"])
        if left_image.max() > 1.0:
            left_image = left_image / 255.0
            right_image = right_image / 255.0
        valid_mask = np.isfinite(disparity) & (disparity >= 0.0)
        examples.append(
            StereoExample(
                left_image=left_image,
                right_image=right_image,
                disparity=disparity,
                valid_mask=valid_mask,
                metadata={
                    "source": "kitti_hf",
                    "dataset_id": dataset_id,
                    "split": split,
                    "sequence_id": str(sample.get("sequence_id", "")),
                    "image_id": str(sample.get("image_id", len(examples))),
                    "fx": float(sample.get("fx", 0.0)),
                    "baseline": float(sample.get("baseline", 0.0)),
                },
            )
        )
        if max_examples > 0 and len(examples) >= max_examples:
            break

    return examples


# Backward-compat alias used by hls4ml converter and evaluation scripts
def load_kitti_examples(
    dataset_id: str,
    *,
    split: str,
    max_examples: int = 0,
) -> list[StereoExample]:
    return load_kitti_hf_examples(dataset_id, split=split, max_examples=max_examples)


# ---------------------------------------------------------------------------
# DrivingStereo (local disk)
# ---------------------------------------------------------------------------
#
# Official download layout:
#   <root>/
#     train-left-image/<seq>/<frame>.jpg
#     train-right-image/<seq>/<frame>.jpg
#     train-disparity/<seq>/<frame>.png    (uint16, /256 → float pixels)
#
# Download: https://drivingstereo-dataset.github.io/

def get_driving_stereo_manifest(
    root_dir: str | Path,
    *,
    split: str = "train",
) -> list[dict[str, str]]:
    """Build a manifest for DrivingStereo from a local disk root."""
    root = resolve_repo_path(root_dir)
    left_root  = root / f"{split}-left-image"
    right_root = root / f"{split}-right-image"
    disp_root  = root / f"{split}-disparity"

    if not left_root.exists():
        raise FileNotFoundError(
            f"DrivingStereo left-image directory not found: {left_root}\n"
            "Download the dataset from https://drivingstereo-dataset.github.io/ "
            f"and extract into {root}"
        )

    entries: list[dict[str, str]] = []
    for left_path in sorted(left_root.rglob("*.jpg")):
        rel = left_path.relative_to(left_root)
        right_path = right_root / rel
        disp_path  = disp_root  / rel.with_suffix(".png")
        if right_path.exists() and disp_path.exists():
            entries.append({
                "source": "driving_stereo",
                "left":      str(left_path),
                "right":     str(right_path),
                "disparity": str(disp_path),
                "split": split,
            })
    if not entries:
        raise ValueError(f"No DrivingStereo examples found under {root}")
    return entries


def load_driving_stereo_example(entry: dict[str, str]) -> StereoExample:
    with Image.open(entry["left"]) as img:
        left_image = _to_grayscale(np.array(img, dtype=np.float32)) / 255.0
    with Image.open(entry["right"]) as img:
        right_image = _to_grayscale(np.array(img, dtype=np.float32)) / 255.0
    with Image.open(entry["disparity"]) as img:
        disp_arr = np.array(img)
    if disp_arr.dtype == np.uint16:
        disparity = disp_arr.astype(np.float32) / 256.0
    else:
        disparity = disp_arr.astype(np.float32)
    valid_mask = disparity > 0.0
    return StereoExample(
        left_image=left_image,
        right_image=right_image,
        disparity=disparity,
        valid_mask=valid_mask,
        metadata={"source": "driving_stereo", "left": entry["left"], "split": entry.get("split", "")},
    )


# ---------------------------------------------------------------------------
# KITTI 2015 / KITTI 2012 (local disk)
# ---------------------------------------------------------------------------
#
# KITTI 2015 layout (data_scene_flow.zip):
#   <root>/training/image_2/   (left)
#   <root>/training/image_3/   (right)
#   <root>/training/disp_occ_0/ (uint16/256 → float)
#
# KITTI 2012 layout (data_stereo_flow.zip):
#   <root>/training/colored_0/  (left)
#   <root>/training/colored_1/  (right)
#   <root>/training/disp_occ/   (uint16/256 → float)

def get_kitti2015_manifest(
    root_dir: str | Path,
    *,
    split: str = "training",
) -> list[dict[str, str]]:
    """Build manifest for KITTI 2015 stereo from local disk."""
    root = resolve_repo_path(root_dir) / split
    left_root  = root / "image_2"
    right_root = root / "image_3"
    disp_root  = root / "disp_occ_0"

    if not left_root.exists():
        raise FileNotFoundError(
            f"KITTI 2015 directory not found: {left_root}\n"
            "Download data_scene_flow.zip from http://www.cvlibs.net/datasets/kitti/"
        )

    entries: list[dict[str, str]] = []
    for left_path in sorted(left_root.glob("*.png")):
        right_path = right_root / left_path.name
        disp_path  = disp_root  / left_path.name
        if right_path.exists() and disp_path.exists():
            entries.append({
                "source": "kitti2015",
                "left":      str(left_path),
                "right":     str(right_path),
                "disparity": str(disp_path),
                "split": split,
            })
    if not entries:
        raise ValueError(f"No KITTI 2015 examples found under {root}")
    return entries


def get_kitti2012_manifest(
    root_dir: str | Path,
    *,
    split: str = "training",
) -> list[dict[str, str]]:
    """Build manifest for KITTI 2012 stereo from local disk."""
    root = resolve_repo_path(root_dir) / split
    left_root  = root / "colored_0"
    right_root = root / "colored_1"
    disp_root  = root / "disp_occ"

    if not left_root.exists():
        raise FileNotFoundError(
            f"KITTI 2012 directory not found: {left_root}\n"
            "Download data_stereo_flow.zip from http://www.cvlibs.net/datasets/kitti/"
        )

    entries: list[dict[str, str]] = []
    for left_path in sorted(left_root.glob("*.png")):
        right_path = right_root / left_path.name
        disp_path  = disp_root  / left_path.name
        if right_path.exists() and disp_path.exists():
            entries.append({
                "source": "kitti2012",
                "left":      str(left_path),
                "right":     str(right_path),
                "disparity": str(disp_path),
                "split": split,
            })
    if not entries:
        raise ValueError(f"No KITTI 2012 examples found under {root}")
    return entries


def load_kitti_disk_example(entry: dict[str, str]) -> StereoExample:
    """Load a single KITTI 2012 or 2015 disk example."""
    with Image.open(entry["left"]) as img:
        left_image = _to_grayscale(np.array(img, dtype=np.float32)) / 255.0
    with Image.open(entry["right"]) as img:
        right_image = _to_grayscale(np.array(img, dtype=np.float32)) / 255.0
    with Image.open(entry["disparity"]) as img:
        disp_arr = np.array(img)
    if disp_arr.dtype == np.uint16:
        disparity = disp_arr.astype(np.float32) / 256.0
    else:
        disparity = disp_arr.astype(np.float32)
    valid_mask = disparity > 0.0
    return StereoExample(
        left_image=left_image,
        right_image=right_image,
        disparity=disparity,
        valid_mask=valid_mask,
        metadata={"source": entry["source"], "left": entry["left"], "split": entry.get("split", "")},
    )


# ---------------------------------------------------------------------------
# Generic manifest entry loader
# ---------------------------------------------------------------------------

def load_example_from_manifest_entry(
    entry: dict[str, str],
    *,
    sceneflow_cache_dir: str | Path = "v2/data/raw/sceneflow",
    sceneflow_dataset_id: str = "olivermao/sceneflow",
    _open_tars: dict[str, tuple[tarfile.TarFile, tarfile.TarFile]] | None = None,
) -> StereoExample:
    """Materialise a StereoExample from any manifest entry dict."""
    source = entry["source"]

    if source == "sceneflow":
        cache_path = resolve_repo_path(sceneflow_cache_dir)
        tar_root = entry.get("tar_root", str(cache_path))
        if _open_tars is not None and tar_root in _open_tars:
            frames_tar, disparity_tar = _open_tars[tar_root]
            left_image  = _to_grayscale(_read_tar_image(frames_tar, entry["left"]))
            right_image = _to_grayscale(_read_tar_image(frames_tar, entry["right"]))
            disparity   = _read_pfm(_read_tar_bytes(disparity_tar, entry["disparity"]))
        else:
            ft_path = Path(tar_root) / "frames_clean.tar"
            dt_path = Path(tar_root) / "disparity.tar"
            with tarfile.open(ft_path, "r") as ft, tarfile.open(dt_path, "r") as dt:
                left_image  = _to_grayscale(_read_tar_image(ft, entry["left"]))
                right_image = _to_grayscale(_read_tar_image(ft, entry["right"]))
                disparity   = _read_pfm(_read_tar_bytes(dt, entry["disparity"]))
        if left_image.max() > 1.0:
            left_image  = left_image  / 255.0
            right_image = right_image / 255.0
        valid_mask = np.isfinite(disparity) & (disparity >= 0.0)
        return StereoExample(
            left_image=left_image, right_image=right_image,
            disparity=disparity, valid_mask=valid_mask,
            metadata={"source": "sceneflow", "dataset_id": sceneflow_dataset_id, "left_member": entry["left"]},
        )

    if source in ("kitti2015", "kitti2012"):
        return load_kitti_disk_example(entry)

    if source == "driving_stereo":
        return load_driving_stereo_example(entry)

    raise ValueError(f"Unknown manifest entry source: {source!r}")


# ---------------------------------------------------------------------------
# Mixed-dataset manifest builder
# ---------------------------------------------------------------------------

def build_mixed_manifest(
    *,
    sceneflow_manifest: list[dict[str, str]] | None = None,
    driving_stereo_manifest: list[dict[str, str]] | None = None,
    kitti2015_manifest: list[dict[str, str]] | None = None,
    kitti2012_manifest: list[dict[str, str]] | None = None,
    sceneflow_frac: float = 0.50,
    driving_stereo_frac: float = 0.35,
    kitti_frac: float = 0.15,
    target_size: int = 0,
    rng: np.random.Generator | None = None,
) -> list[dict[str, str]]:
    """Compose a mixed manifest at the requested sampling fractions.

    If target_size is 0 the count is the maximum drawable without replacement.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    pools: list[tuple[list[dict[str, str]], float]] = []
    if sceneflow_manifest:
        pools.append((sceneflow_manifest, sceneflow_frac))
    if driving_stereo_manifest:
        pools.append((driving_stereo_manifest, driving_stereo_frac))
    kitti_combined: list[dict[str, str]] = []
    if kitti2015_manifest:
        kitti_combined.extend(kitti2015_manifest)
    if kitti2012_manifest:
        kitti_combined.extend(kitti2012_manifest)
    if kitti_combined:
        pools.append((kitti_combined, kitti_frac))

    if not pools:
        raise ValueError("No datasets provided to build_mixed_manifest")

    total_frac = sum(f for _, f in pools)
    pools = [(p, f / total_frac) for p, f in pools]

    if target_size == 0:
        target_size = min(int(len(p) / f) for p, f in pools)

    mixed: list[dict[str, str]] = []
    for pool, frac in pools:
        n = int(target_size * frac)
        if n == 0:
            continue
        indices = rng.choice(len(pool), size=n, replace=(n > len(pool)))
        mixed.extend(pool[i] for i in indices)

    rng.shuffle(mixed)
    return mixed


# ---------------------------------------------------------------------------
# Streaming chunk iterator
# ---------------------------------------------------------------------------

def iter_manifest_chunks(
    manifest: list[dict[str, str]],
    chunk_size: int,
    *,
    sceneflow_cache_dir: str | Path = "v2/data/raw/sceneflow",
    sceneflow_dataset_id: str = "olivermao/sceneflow",
) -> Iterator[list[StereoExample]]:
    """Yield StereoExample lists in chunks, opening SceneFlow tars once per chunk."""
    cache_path = resolve_repo_path(sceneflow_cache_dir)

    for i in range(0, len(manifest), chunk_size):
        chunk = manifest[i : i + chunk_size]
        examples: list[StereoExample] = []

        sf_entries    = [e for e in chunk if e["source"] == "sceneflow"]
        other_entries = [e for e in chunk if e["source"] != "sceneflow"]

        if sf_entries:
            tar_root = sf_entries[0].get("tar_root", str(cache_path))
            ft_path  = Path(tar_root) / "frames_clean.tar"
            dt_path  = Path(tar_root) / "disparity.tar"
            with tarfile.open(ft_path, "r") as ft, tarfile.open(dt_path, "r") as dt:
                open_tars = {tar_root: (ft, dt)}
                for entry in sf_entries:
                    examples.append(
                        load_example_from_manifest_entry(
                            entry,
                            sceneflow_cache_dir=sceneflow_cache_dir,
                            sceneflow_dataset_id=sceneflow_dataset_id,
                            _open_tars=open_tars,
                        )
                    )

        for entry in other_entries:
            examples.append(
                load_example_from_manifest_entry(
                    entry,
                    sceneflow_cache_dir=sceneflow_cache_dir,
                    sceneflow_dataset_id=sceneflow_dataset_id,
                )
            )

        yield examples


# ---------------------------------------------------------------------------
# Manifest I/O
# ---------------------------------------------------------------------------

def write_manifest(path_like: str | Path, entries: list[dict]) -> Path:
    path = resolve_repo_path(path_like)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    return path


def load_manifest(path_like: str | Path) -> list[dict]:
    path = resolve_repo_path(path_like)
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

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
        frame_names    = {m.name for m in frames_tar.getmembers()    if m.isfile()}
        disparity_names = {m.name for m in disparity_tar.getmembers() if m.isfile()}

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
    scale  = float(stream.readline().decode("ascii").strip())
    endian = "<" if scale < 0 else ">"
    channels = 3 if header == "PF" else 1
    data  = np.frombuffer(stream.read(), dtype=endian + "f")
    array = np.reshape(data, (height, width, channels) if channels == 3 else (height, width))
    array = np.flipud(array)
    if channels == 3:
        array = array[..., 0]
    return array.astype(np.float32)
