# v2 Neural Stereo Pipeline

`v2` is now the real-data neural migration track.

The primary path is:

- direct ground-truth supervision with no teacher or distillation stage
- a low-resolution shared-feature correlation student model
- mixed-dataset training led by DrivingStereo with KITTI support and optional Scene Flow
- hls4ml conversion and parity checks on the extracted feature encoder

## Implemented pieces

- `training/hf_utils.py`: `.env`-backed Hugging Face token resolution and repo-relative paths
- `training/stereo_data.py`: Scene Flow, DrivingStereo, KITTI 2012/2015, and KITTI-style HF loaders plus mixed manifests
- `training/augmentation.py`: stereo-safe crop, resize, jitter, and padding utilities
- `models/keras_student.py`: low-resolution correlation student model
- `training/train_stereo.py`: end-to-end GT-only mixed-data training CLI
- `hls4ml/convert_student.py`: feature-extractor export plus hls4ml parity CLI
- `MODEL.md`: model, training flow, and architecture notes

## Environment

```bash
uv sync
```

The training and conversion commands assume Keras 3 with the Torch backend:

```bash
KERAS_BACKEND=torch
```

## Real-data training

Minimal validated run:

```bash
KERAS_BACKEND=torch uv run python -m v2.training.train_stereo \
	--run-name smoke_correlation \
	--driving-stereo-dir v2/data/raw/driving_stereo \
	--kitti2015-dir v2/data/raw/kitti2015 \
	--kitti2012-dir v2/data/raw/kitti2012 \
	--driving-stereo-limit 16 \
	--kitti2015-limit 8 \
	--kitti2012-limit 8 \
	--train-epoch-size 24 \
	--epochs 1 \
	--batch-size 2 \
	--chunk-size 8 \
	--val-limit 4 \
	--no-augment
```

Full mixed-data training adds local-disk datasets:

```bash
KERAS_BACKEND=torch CUDA_VISIBLE_DEVICES=0 uv run python -m v2.training.train_stereo \
	--run-name full_mixed \
	--driving-stereo-dir v2/data/raw/driving_stereo \
	--kitti2015-dir v2/data/raw/kitti2015 \
	--kitti2012-dir v2/data/raw/kitti2012 \
	--train-epoch-size 16384
```

## hls4ml conversion

```bash
KERAS_BACKEND=torch uv run python -m v2.hls4ml.convert_student \
	--model logs/<run>/checkpoints/best.keras \
	--parity-batch logs/<run>/parity_batch.npz \
	--output-dir v2/hls4ml/test_run
```

## Outputs

- `logs/<run>/`: checkpoints, manifests, training history, parity batch, metrics, and final model
- `v2/hls4ml/<run>/`: conversion config, feature-extractor model, and parity metrics

## Current dataset path

- KITTI validation: `UniflexAI/mini_kitti`
- DrivingStereo local root: `v2/data/raw/driving_stereo`
- KITTI 2015 local root: `v2/data/raw/kitti2015`
- KITTI 2012 local root: `v2/data/raw/kitti2012`
- Optional Scene Flow synthetic training: `olivermao/sceneflow`

The smoke training path is validated in this workspace. The active default path no longer depends on Scene Flow; larger runs can proceed directly from DrivingStereo plus KITTI, with Scene Flow available only as an optional synthetic add-on.