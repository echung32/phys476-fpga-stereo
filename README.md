# phys476

This repository now supports two tracks in parallel:

- `v1`: the existing Verilog stereo baseline, preserved as the hardware-reference path
- `v2`: a student-only low-resolution correlation stereo pipeline with hls4ml export checks

## Current layout

```text
phys476/
├── v1/
│   ├── data/                             baseline `.hex` inputs and ground truth
│   ├── output/                           baseline simulation outputs
│   ├── rtl/                              Verilog RTL, constraints, and testbench
│   ├── README.md                         baseline usage and references
│   └── plot_depth.py                     baseline depth visualizer
├── v2/
│   ├── MODEL.md
│   ├── docs/README.md
│   ├── hls4ml/convert_student.py
│   ├── models/keras_student.py
│   └── training/{augmentation.py,hf_utils.py,stereo_data.py,train_stereo.py}
├── REPORT.md
└── pyproject.toml                        uv-managed Python project
```

## Python environment

This repo now uses `uv` for Python environment and dependency management.

```bash
uv sync
```

That creates `.venv/` and installs the Python dependencies declared in `pyproject.toml`.

## Baseline visualization

Use the existing simulation output with the `uv` environment:

```bash
uv run python v1/plot_depth.py
```

Optional overrides still work:

```bash
uv run python v1/plot_depth.py --depth output/depth_out.hex --gt data/gt_disparity.hex
```

## v2 real-data neural pipeline

The primary `v2` path is now teacher-free. It trains a low-resolution full-frame correlation student directly from ground-truth disparity using DrivingStereo-first mixed stereo datasets.

Default data sources:

- KITTI validation source: `UniflexAI/mini_kitti`
- DrivingStereo main training source on local disk
- KITTI 2012/2015 local-disk adapters for real-domain supervision
- Optional Scene Flow synthetic source: `olivermao/sceneflow`

Run a minimal end-to-end smoke pass:

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

Run a mixed-data training pass once local datasets are ready:

```bash
bash v2/data/setup_datasets.sh
KERAS_BACKEND=torch CUDA_VISIBLE_DEVICES=0 uv run python -m v2.training.train_stereo \
    --run-name full_mixed \
    --driving-stereo-dir v2/data/raw/driving_stereo \
    --kitti2015-dir v2/data/raw/kitti2015 \
    --kitti2012-dir v2/data/raw/kitti2012 \
    --train-epoch-size 16384
```

Launch a longer GPU run inside tmux, capture stdout/stderr to a descriptive log,
and record GPU/progress snapshots in a second tmux window:

```bash
scripts/run_train_tmux.sh --attach
```

Useful overrides:

```bash
scripts/run_train_tmux.sh \
    --epochs 20 \
    --batch-size 80 \
    --learning-rate 3e-4 \
    --train-epoch-size 32768
```

Convert the trained student feature extractor with hls4ml and run parity checks:

```bash
KERAS_BACKEND=torch uv run python -m v2.hls4ml.convert_student \
    --model logs/<run>/checkpoints/best.keras \
    --parity-batch logs/<run>/parity_batch.npz \
    --output-dir v2/hls4ml/test_run
```

Outputs:

- `logs/*`: run config, manifests, checkpoints, training history, metrics, and parity batch
- `v2/hls4ml/*`: generated hls4ml config, extracted feature-extractor model, and parity metrics

## Verilog baseline references

The current hardware reference implementation remains in place:

- `v1/rtl/sources_1/new/`: RTL modules
- `v1/rtl/sim_1/new/`: simulation testbench
- `v1/data/` and `v1/output/`: `.hex` inputs and outputs used by the baseline reference path

`v1` now owns the preserved `.hex`-driven baseline; `v2` no longer depends on `.hex` handlers for its primary training and inference path.
