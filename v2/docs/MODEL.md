# v2 Model — Correlation Student

## Overview

`v2` implements a **student-only** stereo depth pipeline trained directly from
ground-truth disparity.  There is no teacher, no distillation, and no
pseudo-labelling.  The primary architecture is a low-resolution full-frame
correlation network designed to run efficiently on real data from multiple
large datasets.

---

## Architecture

### Correlation Student (`v2/models/keras_student.py`)

**Input contract**

| Tensor | Shape        | dtype   | Range  |
|--------|-------------|---------|--------|
| left   | (H, W, 1)   | float32 | [0, 1] |
| right  | (H, W, 1)   | float32 | [0, 1] |

Default input size: **96 × 320** (height × width).

**Pipeline**

1. **Shared feature extractor** — same Conv2D weights applied to left and right:
   - `Conv2D(16, 3×3, same, relu)` → `feat_conv1`
   - `Conv2D(16, 3×3, same, relu)` → `feat_conv2`
   - `Conv2D(8,  3×3, same, relu)` → `feat_conv3`
   Output: (H, W, 8) feature maps for each view.

2. **Correlation cost volume** (`CorrelationCostVolume`):
   For each disparity offset d ∈ [0, max_disp):
   - shift right feature map d pixels to the right (zero-pad left)
   - compute per-pixel dot product with left features → scalar cost
   Output: (H, W, max_disp) cost volume.  Default `max_disp = 48`.

3. **Disparity regression head**:
   - `Conv2D(64, 3×3, same, relu)` → `disp_conv1`
   - `Conv2D(32, 3×3, same, relu)` → `disp_conv2`
   - `Conv2D(max_disp, 1×1)`        → `disp_logits`
   - `Softmax` + weighted sum (soft-argmin) → disparity scalar per pixel

**Output**: (H, W, 1) float32 disparity in pixels ∈ [0, max_disp).

**Loss**: masked smooth-L1 (Huber) over valid GT pixels only.

---

## Training

### Script

```
KERAS_BACKEND=torch uv run python -m v2.training.train_stereo [flags]
```

### Datasets (priority order)

| Dataset        | Source                        | Role      | Fraction |
|----------------|-------------------------------|-----------|----------|
| DrivingStereo  | local disk (official download)| train     | 85 %     |
| KITTI 2015/12  | local disk (official download)| train     | 15 %     |
| Scene Flow     | `olivermao/sceneflow` (HF tar)| optional  | 0 % default |
| mini_kitti     | `UniflexAI/mini_kitti` (HF)   | val only  | —        |

### Key flags

| Flag                  | Default | Description                         |
|-----------------------|---------|-------------------------------------|
| `--epochs`            | 10      | Full passes over the mixed manifest |
| `--batch-size`        | 4       | Images per gradient step            |
| `--chunk-size`        | 256     | Examples loaded per training chunk  |
| `--train-epoch-size`  | 16384   | Examples drawn into each mixed epoch |
| `--max-disp`          | 48      | Disparity range (pixels)            |
| `--target-height`     | 96      | Input image height                  |
| `--target-width`      | 320     | Input image width                   |
| `--driving-stereo-dir`| ""      | Path to DrivingStereo root (skip if absent) |
| `--kitti2015-dir`     | ""      | Path to KITTI 2015 root (skip if absent)    |
| `--kitti2012-dir`     | ""      | Path to KITTI 2012 root (skip if absent)    |
| `--sceneflow-dataset` | ""      | Optional Scene Flow dataset id              |

### Smoke test (CPU, no GPU required)

```bash
KERAS_BACKEND=torch uv run python -m v2.training.train_stereo \
    --run-name smoke \
   --driving-stereo-dir v2/data/raw/driving_stereo \
   --kitti2015-dir v2/data/raw/kitti2015 \
   --kitti2012-dir v2/data/raw/kitti2012 \
   --driving-stereo-limit 16 --kitti2015-limit 8 --kitti2012-limit 8 \
   --train-epoch-size 24 --epochs 1 --batch-size 2 \
   --chunk-size 8 \
    --val-limit 4 --no-augment
```

---

## Augmentation Policy

All transforms are **stereo-safe** (applied identically to left, right,
disparity, and valid mask):

- shared random crop
- shared resize (disparity scaled by horizontal scale factor)
- shared photometric jitter (brightness / contrast / gamma)
- shared mild Gaussian noise

**Forbidden**: horizontal flips, independent geometric transforms, rotations.

---

## Hardware Export (hls4ml)

The **feature extractor sub-model** is exported for FPGA synthesis:

```bash
KERAS_BACKEND=torch uv run python -m v2.hls4ml.convert_student \
    --model logs/<run>/checkpoints/best.keras \
    --output-dir v2/hls4ml/student_hls
```

The extractor maps a single (H, W, 1) image → (H, W, 8) feature map.
On-chip disparity matching is implemented as a WTA correlator outside the
hls4ml-synthesised block.

---

## Dataset Download

### DrivingStereo

```bash
# Download from https://drivingstereo-dataset.github.io/
# Extract into v2/data/raw/driving_stereo/ so that:
#   v2/data/raw/driving_stereo/train-left-image/<seq>/<frame>.jpg exists
```

### KITTI 2015

```bash
# Download data_scene_flow.zip from http://www.cvlibs.net/datasets/kitti/
# Extract into v2/data/raw/kitti2015/ so that:
#   v2/data/raw/kitti2015/training/image_2/<xxxxxx_10.png> exists
```

### KITTI 2012

```bash
# Download data_stereo_flow.zip from http://www.cvlibs.net/datasets/kitti/
# Extract into v2/data/raw/kitti2012/ so that:
#   v2/data/raw/kitti2012/training/colored_0/<xxxxxx_10.png> exists
```
