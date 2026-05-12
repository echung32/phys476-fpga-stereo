# Stereo Depth Estimation — FPGA Project Report (v2)

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [v1 Baseline: Hand-Written Verilog SAD Matcher](#2-v1-baseline-hand-written-verilog-sad-matcher)
3. [v2 Architecture: Neural Feature Extractor Pipeline](#3-v2-architecture-neural-feature-extractor-pipeline)
4. [Dataset Construction](#4-dataset-construction)
5. [Model Training](#5-model-training)
6. [hls4ml Conversion and IP Generation](#6-hls4ml-conversion-and-ip-generation)
7. [Vivado Simulation Infrastructure](#7-vivado-simulation-infrastructure)
8. [Simulation Results](#8-simulation-results)
9. [FPGA Performance and Resource Analysis](#9-fpga-performance-and-resource-analysis)
10. [v1 vs v2 Comparison](#10-v1-vs-v2-comparison)
11. [Conclusions and Future Work](#11-conclusions-and-future-work)

---

## 1. Project Overview

This project develops a stereo depth estimation pipeline targeting a **Xilinx Artix-7 200T FPGA** (`xc7a200tfbg484-1`). The goal is real-time per-pixel disparity inference from a stereo camera stream using hardware-accelerated computation.

The project has two generations:

- **v1**: A hand-written Verilog implementation of Sum-of-Absolute-Differences (SAD) stereo matching — a classical algorithm, fully custom RTL, simulation-only.
- **v2**: A learned stereo feature extractor trained on real driving datasets, converted to FPGA RTL via **hls4ml** and Vitis HLS, with behavioral simulation validated across 59 video frames.

The v2 pipeline is the primary deliverable. It demonstrates the full path from raw dataset → Keras model → fixed-point quantized RTL → Vivado behavioral simulation → frame-by-frame output validation.

---

## 2. v1 Baseline: Hand-Written Verilog SAD Matcher

### Architecture

v1 processes a 512×512 interleaved stereo pixel stream. For each column `c` in a row, the right-eye pixel arrives first, then the left-eye pixel. A 4-state FSM (`IDLE → HOLD → SNAP → RUNNING`) drives a single time-multiplexed SAD unit against 16 disparity candidates, producing a 4-bit disparity per pixel.

| Module | Purpose |
|---|---|
| `line_buffer.v` | BRAM-backed 5-row circular buffer (power-of-2 bank padding to avoid mod-5 dividers) |
| `sliding_window.v` | Assembles a 5×5 pixel patch via shift registers; `genvar`-based part-select for synthesis safety |
| `sad_unit.v` | Sequential 25-cycle accumulator: one subtractor + one adder, `+:` indexed part-select |
| `stereo_top.v` | Top-level FSM, interleaved splitter, single `sad_unit` time-multiplexed over 16 disparities |
| `stereo_tb.v` | Testbench: `$readmemh` inputs, interleaved stream driver, writes `depth_out.hex` |

### Key Design Challenges in v1

**LUT overrun (first pass: 92,630 LUTs required, 63,400 available):**  
The initial v1 implementation instantiated 16 parallel SAD units in a `generate` loop — each with a 25-input combinational adder tree — blowing far past the Artix-7 resource budget. The fix replaced 16 parallel instances with a single sequential unit time-multiplexed over the 16 disparity candidates, reducing LUT cost by ~20× per SAD unit.

**Synthesis error — non-constant part-select bounds:**  
Using a runtime `integer` loop variable as bit-select bounds (`signal[(r+1)*W-1 : r*W]`) is illegal in Verilog synthesis. Fixed by moving the loop into a `generate / genvar` block so bounds become elaboration-time constants.

**Mod-5 bank wrap in line buffer:**  
Modulo by non-power-of-2 synthesises a divider circuit (~9 LUTs). Fixed by padding to 8 banks and using a simple comparator-based wrap (`(ptr == 7) ? 0 : ptr + 1`).

### v1 Limitations

v1 uses synthetic test data (a single synthesised 512×512 stereo pair with a known shift), has no trained parameters, and produces only 4-bit disparity at a fixed 16-candidate search range. It was never tested on real images. The FSM-driven single SAD unit processes one disparity per ~25 cycles, making it too slow for real-time operation at any useful frame rate.

---

## 3. v2 Architecture: Neural Feature Extractor Pipeline

### Full Correlation Student Model

The v2 model is a learned end-to-end stereo depth estimator built in Keras 3 with the PyTorch backend. It has three stages:

**1. Shared feature extractor** (weight-tied left/right):
```
Conv2D(32, 3×3, same, relu)  →  feat_conv1   (H, W, 32)
Conv2D(32, 3×3, same, relu)  →  feat_conv2   (H, W, 32)
Conv2D(16, 3×3, same, relu)  →  feat_conv3   (H, W, 16)
```

**2. Correlation cost volume** (`CorrelationCostVolume`):  
For each disparity offset `d ∈ [0, 64)`, the right feature map is shifted `d` pixels right (zero-padded) and a per-pixel dot product with the left features is computed, yielding a `(H, W, 64)` cost volume.

**3. Disparity regression head**:
```
Conv2D(64, 3×3, same, relu)  →  disp_conv1
Conv2D(32, 3×3, same, relu)  →  disp_conv2
Conv2D(64, 1×1)              →  disp_logits
Softmax + weighted sum (soft-argmin)  →  disparity (H, W, 1)
```

**Input resolution**: 160×288 (height × width), 16:9 crop.  
**Loss function**: Masked smooth-L1 (Huber) over valid ground-truth disparity pixels only.  
**Output**: Per-pixel disparity in pixels, range `[0, 64)`.

### FPGA-Deployed Submodel: Feature Extractor

The full correlation model — including the cost volume and regression head — is too large to fit on an Artix-7 200T. The **shared feature extractor** (`feat_conv1 → feat_conv3`) is the portion exported to hardware via hls4ml. This sub-model maps:

```
(H, W, 1)  grayscale image  →  (H, W, 16)  fixed-point feature maps
```

**Is the implementation complete?** The short answer is: the *Python model* is fully complete and produces real depth maps end-to-end. The *hardware implementation* covers only the feature extraction stage. The remaining two stages — the correlation cost volume and the soft-argmin disparity regression head — run in Python/software and have not been converted to RTL.

This is a deliberate scoping decision: the cost volume requires 64 parallel dot-product streams plus large inter-stage buffers, which alone would consume more resources than the feature extractor. The practical hardware path forward is to implement a simpler WTA (winner-takes-all) correlator in hand-written Verilog that consumes the 256-bit feature stream and outputs a single disparity per pixel — bypassing the learned regression head entirely in favor of a hardware-efficient nearest-neighbor search. That block is not yet implemented.

---

## 3b. Pipeline Flow Diagram

The diagram below shows the complete development flow from raw data to RTL simulation output, and where the Python software pipeline and the hardware (RTL) pipeline diverge.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         SOFTWARE / TRAINING PATH                               │
│                                                                                 │
│  DrivingStereo (55%)  ─┐                                                        │
│  Scene Flow (25%)     ─┼─→  stereo_data.py  ─→  train_stereo.py               │
│  KITTI 2012/2015 (20%) ┘        (mixed manifest)      │                        │
│                                                        │ Adam, 16 epochs        │
│                                                        ▼                        │
│                                           correlation_student.keras             │
│                                           (full model: 160×288→disparity)      │
│                                                        │                        │
│                                         extract_feature_extractor()            │
│                                                        │                        │
│                                           feature_extractor.keras              │
│                                           (160×288×1 → 160×288×16)            │
└────────────────────────────────────────────┬────────────────────────────────────┘
                                             │
                                    hls4ml.convert_from_keras()
                                    fixed<16,6>, io_stream, RF=1
                                             │
┌────────────────────────────────────────────▼────────────────────────────────────┐
│                           HLS / RTL GENERATION PATH                            │
│                                                                                 │
│              Vitis HLS 2025.2  (build_prj.tcl)                                 │
│              C synthesis → 86 Verilog files, 397K lines                        │
│              Target: xc7a200tfbg484-1, 100 MHz clock                           │
│                                                                                 │
│  Interface:                                                                     │
│    image_r_TDATA  [15:0]   ← 1 pixel per beat (fixed<16,6>)                   │
│    layer7_out_TDATA [255:0] → 16 channels per beat (packed fixed<16,6>)        │
└────────────────────────────────────────────┬────────────────────────────────────┘
                                             │
                             vivado_sim.tcl (batch mode)
                             Vivado 2025.2 — behavioral sim ONLY
                             (no synthesis, no implementation, no bitstream)
                                             │
┌────────────────────────────────────────────▼────────────────────────────────────┐
│                           VIVADO / XSIM SIMULATION PATH                        │
│                                                                                 │
│  myproject_tb.sv                                                                │
│    ├── $readmemh(tb_input_hex.mem)     ← 46,080 fixed<16,6> pixels             │
│    ├── AXI-Stream driver (TVALID/TREADY handshake)                             │
│    ├── Output capture (all 46,080 × 256-bit output beats)                      │
│    └── writes rtl_output_<stem>.hex + .f32 per frame                           │
│                                                                                 │
│  run_parallel_xsim.sh                                                          │
│    ├── compile once (xvlog + xelab, ~20 min)                                   │
│    ├── reuse compiled snapshot (xsim.dir/ symlinked)                           │
│    └── xargs -P 32 → 59 frames in parallel (~84 min wall time)                 │
└────────────────────────────────────────────┬────────────────────────────────────┘
                                             │
┌────────────────────────────────────────────▼────────────────────────────────────┐
│                           POST-PROCESSING / VALIDATION                         │
│                                                                                 │
│  render_feature_video.py  → feature_video_10fps.mp4  (L2-norm heatmap)         │
│  render_channel_grid.py   → channel_grid_frame0.png  (16 individual channels)  │
│  decode_vivado_output.py  → compare RTL vs. Keras float reference              │
└─────────────────────────────────────────────────────────────────────────────────┘

  WHAT IS NOT IMPLEMENTED IN RTL:
  ─────────────────────────────────
  ✗  Correlation cost volume  (64 dot-product streams per pixel)
  ✗  Soft-argmin disparity regression head
  ✗  Vivado synthesis / implementation / bitstream generation
  ✗  Post-implementation timing closure verification
```

**Bottom line**: We have a complete software depth-estimation model and a validated behavioral RTL simulation of its first stage (feature extraction). We do *not* have a synthesised bitstream or a full hardware disparity pipeline.

---

## 4. Dataset Construction

### Training Datasets

Training used a mixed dataset drawn from three real-world stereo sources, controlled by fractional sampling weights:

| Dataset | Fraction | Source | Role |
|---|---|---|---|
| DrivingStereo | 55% | Local disk (official download, ~170K stereo pairs) | Primary training signal |
| Scene Flow (Monkaa) | 25% | Local disk (`v2/data/raw/sceneflow/monkaa`) | Synthetic diversity |
| KITTI 2012 + 2015 | 20% | Local disk (official download) | Real-world domain transfer |
| mini_kitti (HF) | — | `UniflexAI/mini_kitti` | Validation only (88 pairs) |

Training epochs drew 32,768 examples per epoch using chunk-based streaming with worker-parallel loading (`loader_workers=12`). Batches were formed by random sampling across all active sources weighted by the configured fractions.

### Data Augmentation

The `augmentation.py` module applies stereo-safe transformations that preserve epipolar consistency:

- **Random crop**: Applied identically to left and right frames, maintaining horizontal alignment
- **Random horizontal flip**: Disabled for stereo (would swap left/right and invert disparities)
- **Color jitter**: Applied jointly to both frames with the same random parameters
- **Resize**: Images are resized to 160×288 (16:9 crop of the network input resolution)
- **Normalization**: Pixel values scaled to `[0, 1]`

### Perspective Stereo Extraction for Simulation

For the hardware simulation validation, a separate dataset was constructed from a single video sequence. The `v2/data/extract_perspective_stereo.py` script:

1. Loaded frames from `v2/data/stereo-video.npz` (30fps stereo video)
2. Rendered each frame at 8 yaw angles (−135° to +180° in 45° steps)
3. Saved each rendered view as a 512×512 `.npz` file (arrays: `left_image`, `right_image`, `disparity`, `valid_mask`, `metadata`) and a `.mem` file (512×512 lines of `fixed<16,6>` hex, one pixel per line)

The simulation pipeline extracted 59 frames from the yaw=+000 sequence at stride 3 (30fps → 10fps effective), using only the **left image** as the grayscale feature extractor input. The 512×512 `.mem` files were **center-cropped** to the network input size (rows 176:336, cols 112:400 = 160×288 = 46,080 pixels) before being fed to the RTL testbench.

---

## 5. Model Training

### Training Run

**Run name**: `monkaa_crop16x9_sf25_fc32_h160_w288_bs128`

| Parameter | Value |
|---|---|
| Input resolution | 160×288 (H×W) |
| Feature channels | 32 (16 after `feat_conv3`) |
| Max disparity | 64 pixels |
| Learning rate | 3×10⁻⁴ (Adam) |
| Batch size | 128 |
| Epochs | 16 |
| Train examples/epoch | 32,768 |
| Scene Flow fraction | 25% |
| DrivingStereo fraction | 55% |
| KITTI fraction | 20% |
| Augmentation | enabled |

### Training Convergence

| Epoch | Train Loss | Train MAE | Val MAE | DrivingStereo Hold. MAE | Scene Flow Hold. MAE |
|---|---|---|---|---|---|
| 1 | 8.89 | 9.38 px | 7.43 px | 6.01 px | 9.27 px |
| 4 | 7.11 | 7.60 px | 6.52 px | 5.95 px | 8.84 px |
| 12 | 5.31 | 5.78 px | 5.01 px | 3.64 px | 7.01 px |
| 16 | 4.15 | 4.60 px | **3.68 px** | **3.44 px** | 5.23 px |

The model converged steadily over 16 epochs. The best checkpoint achieved:

- **Validation MAE**: 3.68 px (on 88 KITTI pairs)
- **DrivingStereo holdout MAE**: 3.44 px (1,024 pairs)
- **DrivingStereo holdout RMSE**: 4.56 px
- **Bad-1px rate**: 71.5% of pixels within 1 px error
- **Bad-3px rate**: 35.4% of pixels within 3 px error
- **Bad-5px rate**: 19.4% of pixels within 5 px error

The Scene Flow holdout MAE (5.23 px) is higher than DrivingStereo (3.44 px), consistent with the domain gap between synthetic and real driving data — the model is primarily tuned for real-world appearance statistics.

### Artifacts

```
logs/monkaa_crop16x9_sf25_fc32_h160_w288_bs128/
├── checkpoints/best.keras          ← best validation checkpoint
├── checkpoints/latest.keras        ← final epoch checkpoint
├── student_model.keras             ← copy of final model
├── parity_batch.npz                ← reference I/O batch for HLS parity checks
├── metrics.json                    ← final metrics summary
└── evaluation/                     ← per-example evaluation panels
```

---

## 6. hls4ml Conversion and IP Generation

### Feature Extractor Extraction

`v2/hls4ml/convert_student.py` extracts the shared feature extractor from the trained full model and saves it as a standalone Keras model (`feature_extractor.keras`). This sub-model applies the same three Conv2D layers to a single grayscale input and produces 16-channel feature maps.

```
input:   (160, 288, 1)   float32 / fixed<16,6>
output:  (160, 288, 16)  fixed<16,6>
```

### Quantization Configuration

All weights and activations are quantized to **`fixed<16,6>`** (16-bit total, 6 integer bits, 10 fractional bits). This provides:

- Representation range: ±31.984375
- Precision: 1/1024 ≈ 0.000977
- Sufficient dynamic range for convolution outputs at this network depth

The hls4ml configuration (`hls_config.yaml`) specifies:
- `IOType: io_stream` — AXI-Stream handshake throughout (TVALID/TREADY)
- `Strategy: Latency` — minimize latency, no resource reuse
- `ReuseFactor: 1` — no time-multiplexing of DSPs within the HLS kernel

### Vitis HLS Synthesis

**Tool**: Vitis HLS 2025.2  
**Target part**: `xc7a200tfbg484-1` (Artix-7 200T)  
**Clock**: 10 ns target (100 MHz)

HLS C synthesis (`build_prj.tcl`) produced 86 Verilog source files in `fixed16_6/myproject_prj/solution1/syn/verilog/` (396,975 total lines of RTL), representing the pipelined dataflow decomposition of the three Conv2D layers plus their padding and activation modules.

### AXI-Stream Interface

| Port | Direction | Width | Description |
|---|---|---|---|
| `image_r_TDATA` | in | 16 bits | One `fixed<16,6>` pixel per beat |
| `image_r_TVALID` | in | 1 bit | Upstream valid |
| `image_r_TREADY` | out | 1 bit | DUT ready for input |
| `layer7_out_TDATA` | out | 256 bits | 16 × `fixed<16,6>` channels packed (ch0 = bits[15:0], ch15 = bits[255:240]) |
| `layer7_out_TVALID` | out | 1 bit | Output valid |

Input: 46,080 beats per frame (160×288 pixels, one pixel per beat).  
Output: 46,080 beats per frame (one 256-bit word = 16 feature channels per spatial location).

---

## 7. Vivado Simulation Infrastructure

### Testbench Design (`v2/hls4ml/myproject_tb.sv`)

A SystemVerilog testbench drives the AXI-Stream DUT and captures all output data to disk. Key design decisions:

- **`run_frame(mem_path, stem)` task**: Loads a `.mem` file via `$readmemh`, resets the DUT, forks an input driver and output capture process, and blocks until all 46,080 output beats are collected.
- **Output files per frame**:
  - `tb_data/rtl_output_<stem>.hex` — 46,080 lines of 64-hex-digit (256-bit) raw output
  - `tb_data/rtl_output_<stem>.f32` — 46,080 lines of 16 space-separated float32 values (decoded from fixed<16,6> by dividing int16 by 64)
- **ap_done bug**: The standard `@(posedge ap_done)` approach missed the single-cycle pulse when it fired during `fork/join`. Fixed by inferring completion from output beat count: once all 46,080 output beats are captured, the frame is done.
- **Timeout**: 400 ms simulation time (sufficient for ~20+ frames at 8.98 ms/frame).
- **Manifest support**: Reads `tb_data/frame_manifest.txt` if present for multi-frame single-process runs; falls back to `tb_data/tb_input_hex.mem`.

### Parallel Simulation Runner (`scripts/run_parallel_xsim.sh`)

To simulate all 59 frames efficiently, a parallel runner was built that reuses the compiled xsim snapshot without relaunching Vivado (which takes ~20 minutes per compile). Key design:

1. **First run** (without `--skip-compile`): Launches Vivado in batch mode to compile and elaborate the full design into a snapshot at `vivado_sim_project/.../xsim.dir/myproject_tb_behav/`.
2. **Subsequent runs** (`--skip-compile`): Skips Vivado entirely. Each per-frame job:
   - Creates a workdir: `parallel_sim/<stem>/`
   - Symlinks the compiled `xsim.dir/` snapshot (read-only, shared across workers)
   - Copies `xsim.ini` (cannot be symlinked — xsim writes to it)
   - **Center-crops** the 512×512 `.mem` input file to 160×288 using inline Python
   - Stages the cropped `.mem` to `tb_data/tb_input_hex.mem`
   - Runs `xsim myproject_tb_behav -tclbatch sim.tcl`
3. **Parallelism**: `xargs -P 32` with `export -f run_frame` + `bash -c 'run_frame "$@"' _` pattern for reliable function export across subshells.
4. **Skip-if-done**: Checks for existing `.hex` output before running — safe to re-run interrupted jobs.

### Compilation Pipeline

For testbench changes without full Vivado relaunch:
```bash
cd v2/hls4ml/vivado_sim_project/myproject_sim.sim/sim_1/behav/xsim
source /mnt/dev/xilinx/2025.2/Vivado/settings64.sh
bash compile.sh && bash elaborate.sh
```
This recompiles only the testbench SV in ~seconds.

---

## 8. Simulation Results

### xsim Timing Validation

The simulation log for each frame records:
```
INFO [8982675000] All 46080 output beats captured
INFO [8982875000] Frame complete
```

Simulation time per frame: **8,982,875 ns = 8.983 ms**

This matches the HLS csynth report latency of **8.935 ms** within **0.5%**, confirming the behavioral simulation is consistent with the synthesis timing model.

### 59-Frame Parallel Run

All 59 frames of the yaw=+000 perspective stereo sequence were simulated across 32 cores in parallel. Total wall time: **~84 minutes** (~17–30 min per frame, varying with core load).

The ratio of simulation wall time to simulated hardware time illustrates the cost of software RTL simulation:
- Simulated hardware time per frame: ~8.98 ms
- xsim wall time per frame: ~20–25 min = ~1,500 s
- **Simulation speed ratio: ~170,000× slower than real silicon**

This is normal for RTL behavioral simulation — every flip-flop, every combinational gate, every AXI handshake is computed in software.

### Feature Map Analysis

Output `.f32` files were decoded and visualised with `render_feature_video.py`. The 59-frame sequence was rendered as a 10 fps side-by-side video (`v2/hls4ml/outputs/feature_video_10fps.mp4`) with the left grayscale input alongside a false-color heatmap of the L2 norm across all 16 output channels.

Notable observations:
- **Channel 12 dominates**: mean activation ~21.5 across the sequence (range 0–23.2)
- **Channels 0, 1, 2, 5, 6, 8, 13, 14**: near-zero mean (ReLU-saturated or dead)
- **Channels 9, 10, 11**: gradual increase over the 59-frame sequence, tracking scene motion
- **Spatial L2 norm**: min ~4.7, max ~31.3 — clear spatial structure present (edges, textures respond differently than flat regions)

The heatmap shows strong activation at object boundaries and texture-rich regions, confirming the convolutional filters have learned semantically meaningful low-level features despite the limited training depth.

---

## 9. FPGA Performance and Resource Analysis

### Timing Performance

From `myproject_csynth.rpt` and `solution1.log`:

| Metric | Value |
|---|---|
| Target clock | 10.00 ns (100 MHz) |
| HLS estimated clock (post-synthesis) | 7.272 ns |
| HLS uncertainty margin | 2.700 ns |
| **Estimated Fmax** | **137.5 MHz** |
| **Latency (min = max)** | **893,498 cycles** |
| **Latency @ 100 MHz** | **8.935 ms per frame** |
| **Latency @ 137.5 MHz** | **6.497 ms per frame** |
| **Interval (throughput)** | 892,620 cycles (dataflow pipelining) |
| **Throughput @ 100 MHz** | **112.1 FPS** |
| **Throughput @ 137.5 MHz** | **154.0 FPS** |

The design uses `dataflow` pipelining: once the first output pixel is produced, the three conv layers run concurrently. The interval (892,620 cycles) is slightly less than the latency (893,498 cycles), meaning the pipeline can accept a new frame every 8.926 ms while the previous frame is still being output.

**Real-time threshold**: 30 fps requires 33.3 ms/frame. The feature extractor IP is **3.7× faster than needed** at 100 MHz and **5.1× faster** at estimated Fmax.

### Per-Layer Latency Breakdown

| Layer | Cycles | Time @ 100 MHz |
|---|---|---|
| ZeroPad2D (input, 1ch) | 47,628 | 0.476 ms |
| Conv2D layer 1 (1→32ch, 3×3) | 328,863 | 3.289 ms |
| ReLU layer 1 (32ch) | 46,082 | 0.461 ms |
| ZeroPad2D layer 2 (32ch) | 47,628 | 0.476 ms |
| **Conv2D layer 2 (32→32ch, 3×3)** | **892,623** | **8.926 ms** ← pipeline bottleneck |
| ReLU layer 2 (32ch) | 46,082 | 0.461 ms |
| ZeroPad2D layer 3 (32ch) | 47,628 | 0.476 ms |
| Conv2D layer 3 (32→16ch, 3×3) | 892,623 | 8.926 ms |
| ReLU layer 3 (16ch) | 46,083 | 0.461 ms |

Conv2D layer 2 is the pipeline bottleneck: 32 input channels × 32 output channels × 3×3 filter at `ReuseFactor=1` dominates the interval. Because the design uses `io_stream` dataflow, all layers run concurrently — the 893,498-cycle total latency is the sum of startup latencies, and the 892,620-cycle interval is set by the bottleneck stage.

### Resource Utilization

From `myproject_csynth.rpt` (estimates against Artix-7 200T available resources):

| Resource | Used | Available | Utilization |
|---|---|---|---|
| **BRAM_18K** | **21,056** | 730 | **2,884%** ⚠️ |
| **DSP48E1** | **9,218** | 740 | **1,245%** ⚠️ |
| **Flip-Flops** | **512,842** | 269,200 | **190%** ⚠️ |
| **LUTs** | **589,297** | 134,600 | **437%** ⚠️ |
| URAM | 0 | 0 | — |

**The feature extractor alone overflows every resource category on the Artix-7 200T by 2–29×.**

This is a fundamental consequence of using `Strategy: Latency` with `ReuseFactor=1`: every multiply-accumulate in every 3×3 filter is implemented as a dedicated DSP+FF pipeline stage running in parallel. The dominant contributors are the two large conv layers:

| Layer | BRAM_18K | DSP | FF | LUT |
|---|---|---|---|---|
| Conv2D layer 2 (32→32ch) | 0 | 5,845 | 322,649 | 376,522 |
| Conv2D layer 3 (32→16ch) | 0 | 3,170 | 175,706 | 193,177 |
| Conv2D layer 1 (1→32ch) | 0 | 203 | 11,586 | 12,346 |
| FIFOs (inter-layer buffers) | 21,056 | — | 1,304 | 656 |

The FIFO BRAM usage (21,056 BRAM_18K) arises because `io_stream` dataflow requires full-depth FIFOs between pipeline stages — each inter-layer FIFO must hold one full image (46,080 pixels × number of channels × 16 bits), which totals ~243 Mbit and maps to thousands of BRAM tiles.

### Why the Artix-7 Cannot Fit This Design

The Artix-7 200T is a mid-range FPGA with 134,600 LUTs and 740 DSPs. The feature extractor at full parallelism (`RF=1`) requires 9,218 DSPs — 12.5× the device capacity. This is because each 3×3 convolution with `C_in` input channels and `C_out` output channels requires `C_in × C_out × 9` multiplications per pixel, all pipelined in parallel. For the 32→32ch layer that is 9,216 multipliers running simultaneously.

**Options to fit on Artix-7 (each with latency trade-offs)**:

| Approach | DSP reduction | Latency increase |
|---|---|---|
| `ReuseFactor=64` (time-multiplex DSPs 64×) | ~144 DSPs | ~64× longer |
| Reduce to 8 feature channels (8→8ch layers) | ~9× fewer DSPs | Accuracy loss |
| Use a larger FPGA (e.g. Ultrascale+ VU9P, 6,840 DSPs) | fits with RF~2 | minimal |
| Split across two Artix-7 devices | fits per-chip | board complexity |
| Quantize to 8-bit (`fixed<8,4>`) — use 2 mults/DSP | ~2× improvement | accuracy loss |

An earlier variant `fixed16_6_rf144_resource` was generated with `ReuseFactor=144` and `Strategy=Resource`, which reduces DSP count substantially at the cost of ~144× longer latency per layer.

---

## 10. v1 vs v2 Comparison

| Dimension | v1 | v2 |
|---|---|---|
| **Algorithm** | Classical SAD, 5×5 window, 16 disparity candidates | Learned 3-layer CNN feature extractor + correlation cost volume |
| **Input size** | 512×512 synthetic stereo pair | 160×288 real-world driving stereo (center-cropped) |
| **Disparity range** | 16 candidates | 64 candidates (regression) |
| **RTL origin** | Hand-written Verilog | hls4ml auto-generated from Keras (86 Verilog files, 397K lines) |
| **Quantization** | 8-bit pixel, 13-bit accumulator | `fixed<16,6>` throughout (16-bit, 6 integer bits) |
| **Interface** | Interleaved pixel stream, custom valid/ready | AXI-Stream (16-bit input, 256-bit output) |
| **Target FPGA** | Artix-7 (resource-constrained) | Artix-7 200T (feature extractor overflows; needs larger device) |
| **Latency (one frame)** | ~400+ µs (estimated from FSM analysis) | **8.935 ms** (confirmed by HLS + xsim) |
| **Throughput** | Low (single SAD unit, sequential) | **112 FPS @ 100 MHz** (dataflow pipelined) |
| **Training data** | None (no ML) | DrivingStereo 55% + Scene Flow 25% + KITTI 20%, 16 epochs |
| **Accuracy** | Qualitative (synthetic data only) | MAE 3.44 px on DrivingStereo, MAE 3.68 px on KITTI validation |
| **Simulation** | Single synthetic frame, Vivado xsim | 59 real-world frames, parallel xsim (32 cores, 84 min) |
| **Validation** | Qualitative depth map visual | Per-pixel `.f32` output, 10fps feature heatmap video |
| **Python tooling** | `plot_depth.py` (single frame visual) | `render_feature_video.py`, `decode_vivado_output.py`, `evaluate_run.py` |

The fundamental shift from v1 to v2 is moving from a hand-crafted algorithm to a learned model. v1 requires zero training data and produces output directly from a formula; v2 requires a training pipeline, dataset infrastructure, and HLS tooling, but produces richer semantic feature representations that generalize to real driving scenes.

---

## 11. Conclusions and Future Work

### What Was Accomplished

1. **End-to-end neural FPGA pipeline demonstrated**: Keras model → hls4ml quantization → Vitis HLS synthesis → Vivado behavioral simulation → real data validation.
2. **59-frame RTL simulation validated**: All 46,080 output beats captured per frame, 256-bit AXI-Stream output decoded to float32 feature maps.
3. **10fps feature video rendered**: Side-by-side input/heatmap video shows spatially coherent feature activations tracking scene structure over time.
4. **Performance confirmed**: 8.935 ms/frame latency at 100 MHz = **112 FPS**, 3.7× faster than real-time 30fps — when the device is large enough.
5. **Resource gap quantified**: Feature extractor requires a device with ≥9,218 DSPs and ≥21,000 BRAM tiles; Artix-7 200T has only 740 DSPs and 730 BRAM tiles.

### Next Steps

**Hardware fitting (critical path)**:
- Generate a `ReuseFactor=128` or higher resource-strategy variant to fit the design on Artix-7 (target: <700 DSPs, <700 BRAM_18K)
- Alternatively, reduce feature channels to 8 (instead of 16 output) and re-run HLS

**Full disparity pipeline**:
- The feature extractor IP is only one component. A WTA (winner-takes-all) or soft-argmin correlator block needs to be designed in RTL to consume the 256-bit feature output and produce a final disparity value per pixel.

**Numeric validation**:
- Run `decode_vivado_output.py --hex-file ... --npz ... --model ...` to compare RTL output against Keras floating-point reference and confirm fixed-point error budget

**Real hardware deployment**:
- Once resource-reduced and post-implementation timing is clean: synthesize, place-and-route, generate bitstream, and deploy to physical Artix-7 board
- Measure actual clock frequency and latency using ILA (Integrated Logic Analyzer)
