# v2 Neural Stereo Pipeline

`v2` is now the real-data neural migration track.

The primary path is:

- real stereo images and disparity labels from Hugging Face datasets
- a compact native Keras student model with only hls4ml-friendly layers
- full disparity reconstruction in software over candidate disparities `0..15`
- hls4ml conversion and software parity checks on the trained student

## Implemented pieces

- `training/hf_utils.py`: `.env`-backed Hugging Face token resolution and repo-relative paths
- `training/stereo_data.py`: real stereo dataset loaders for KITTI-style Hugging Face datasets and Scene Flow tar archives
- `training/patch_dataset.py`: patch sampling, reconstruction, and disparity metrics for real images
- `models/keras_student.py`: native Keras student model
- `training/train_real_stereo.py`: end-to-end pretrain, fine-tune, evaluate, and export CLI
- `hls4ml/convert_student.py`: hls4ml conversion and parity CLI
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
KERAS_BACKEND=torch uv run python -m v2.training.train_real_stereo \
	--pretrain-max-examples 0 \
	--finetune-max-examples 2 \
	--validation-max-examples 1 \
	--max-positions-per-image 64 \
	--finetune-epochs 1 \
	--batch-size 64 \
	--output-dir v2/exports/test_run \
	--manifest-dir v2/data/manifests/test_run
```

Scene Flow pretraining can be enabled by setting `--pretrain-max-examples` above zero.

## hls4ml conversion

```bash
KERAS_BACKEND=torch uv run python -m v2.hls4ml.convert_student \
	--model v2/exports/test_run/student_patch_model.keras \
	--parity-batch v2/exports/test_run/parity_batch.npz \
	--output-dir v2/hls4ml/test_run
```

## Outputs

- `v2/exports/`: trained model, parity batch, training history, validation metrics, and predicted disparity artifacts
- `v2/data/manifests/`: sampled dataset manifests used for a run
- `v2/hls4ml/`: conversion config and parity metrics

## Current dataset path

- KITTI fine-tune/eval: `UniflexAI/mini_kitti`
- Scene Flow pretrain: `olivermao/sceneflow`

The KITTI path is fully validated in this workspace. The Scene Flow loader is implemented against the Hugging Face tar archive layout and is ready for larger pretraining runs.