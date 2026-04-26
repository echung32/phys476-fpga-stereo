# phys476

This repository now supports two tracks in parallel:

- `v1`: the existing Verilog stereo baseline, preserved as the hardware-reference path
- `v2`: a real-data neural stereo pipeline built around a compact Keras student model and hls4ml conversion checks

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
│   └── training/{hf_utils.py,patch_dataset.py,stereo_data.py,train_real_stereo.py}
├── NN_MIGRATION_PLAN.md
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

The primary `v2` path now trains on real stereo images and disparity labels from Hugging Face datasets. The student model is a native Keras patch matcher that keeps the hardware-facing 5x5 patch plus 16-candidate disparity search assumption, while using a software inference loop to reconstruct a full disparity map.

Default datasets:

- Scene Flow pretraining source: `olivermao/sceneflow`
- KITTI fine-tune and evaluation source: `UniflexAI/mini_kitti`

Run a minimal end-to-end real-data training pass:

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

Convert the trained Keras student with hls4ml and run software parity checks:

```bash
KERAS_BACKEND=torch uv run python -m v2.hls4ml.convert_student \
    --model v2/exports/test_run/student_patch_model.keras \
    --parity-batch v2/exports/test_run/parity_batch.npz \
    --output-dir v2/hls4ml/test_run
```

Outputs:

- `v2/exports/*`: trained Keras model, manifests, validation metrics, parity batch, and validation visualizations
- `v2/hls4ml/*`: generated hls4ml config and software parity metrics

## Verilog baseline references

The current hardware reference implementation remains in place:

- `v1/rtl/sources_1/new/`: RTL modules
- `v1/rtl/sim_1/new/`: simulation testbench
- `v1/data/` and `v1/output/`: `.hex` inputs and outputs used by the baseline reference path

`v1` now owns the preserved `.hex`-driven baseline; `v2` no longer depends on `.hex` handlers for its primary training and inference path.
