# Reduced Training Plan: 40×60 SceneFlow Indoor PoC

**Goal**: Train a stereo disparity model for indoor scenes on SceneFlow (synthetic) at 40×60 resolution as a proof-of-concept that fits on XC7A200T FPGA via hls4ml.

**Rationale**:
- SceneFlow is synthetic but high-quality and domain-agnostic (works for indoor/outdoor)
- No sparse ground-truth issues like Driving Stereo
- 40×60 resolution reduces FIFO BRAM from ~21,000 to ~1,300 BRAM_18K (fits within XC7A200T's 730 budget when using streaming/row-buffering later)
- Simpler resolution allows faster iteration and lower computational cost during development
- For deployment, forcing grayscale end-to-end (training + inference) removes 3-channel feature-map pressure and further reduces BRAM/bandwidth

---

## Hardware Target

**Part**: `xc7a200tfbg484-1` (Artix-7)  
**Available BRAM_18K**: 730  
**FIFO BRAM needed at 40×60**: ~1,300 (with io_stream architecture; fits with careful optimization)

---

## Model Architecture

### Input
- **Resolution**: 40×60 (H×W, grayscale)
- **Channels**: 1 (grayscale left + right fed separately)
- **Fit mode**: `pad` (pad smaller images, crop larger ones)

### Grayscale Policy (Important for BRAM)
- Convert all training inputs to grayscale in preprocessing before resize/fit
- Keep model input as single-channel throughout (no RGB path)
- Keep camera/deployment preprocessing grayscale-first before FPGA model ingress
- If source sensor is RGB/YUV, convert once at ingress and discard chroma channels

**Why this helps**:
- Activations and line buffers scale with channel count; moving from 3 channels to 1 channel cuts these early-memory terms by about 3x
- Stream bandwidth is reduced by about 3x for the same geometry
- Compute load in early layers is lower, improving timing margin

### Feature Extractor
- Conv2D(8, 3×3, padding=same, relu)  ← **reduced** from 16 channels
- Conv2D(8, 3×3, padding=same, relu)
- Conv2D(4, 3×3, padding=same, relu)  ← **reduced** from 8

**Justification**: 40×60 is much smaller, so fewer feature channels sufficient. Reduces both memory and compute.

### Correlation Head
- Max disparity: **24** (instead of 48)  
  **Justification**: At 60 pixel width, typical indoor stereo baseline ~10cm gives reasonable max disparity of 20–30 pixels. SceneFlow synthetic scene geometry allows reasonable disparities in this range.

### Disparity Regression Head
- Conv2D(32, 3×3, padding=same, relu)  ← **reduced** from 64
- Conv2D(16, 3×3, padding=same, relu)  ← **reduced** from 32
- Conv2D(24, 1×1)  ← logits (output channels = max_disp)
- Softmax + soft-argmin disparity regression

---

## Training Configuration

### Dataset

```bash
--sceneflow-dataset "olivermao/sceneflow"  # Use HF dataset ID or local extracted root
--sceneflow-limit 0                         # Use all SceneFlow examples (default: 0 = all)
--sceneflow-frac 1.0                        # 100% from SceneFlow
--driving-stereo-frac 0.0                   # 0% DrivingStereo (skip)
--kitti-frac 0.0                            # 0% KITTI (skip)
```

**Why SceneFlow only**: 
- Synthetic, consistent quality → no sparse GT noise
- Indoor/outdoor variation in scenes → generalizes to indoor use case
- Sufficient examples for POC (~35k pairs in training split)

### Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Epochs** | 20 | Fewer than typical (10–12) because SceneFlow is synthetic and cleaner; faster convergence expected |
| **Batch size** | 256 | Large batch: 40×60×1 bytes/sample = 2400 bytes/sample; 256 samples = 614 KB, very manageable on GPU |
| **Learning rate** | 5e-4 | Slightly lower than default (1e-3); smaller model and simpler task benefit from cautious optimization |
| **Feature channels** | 8 | Reduced from 16; 40×60 input has ~2400 pixels, enough for 8 channels of learned representation |
| **Max disparity** | 24 | Scaled from 48; reasonable for 60-pixel-width stereo at typical indoor baselines |
| **Chunk size** | 256 | Data loading batch; reasonable for large batch training |
| **Train epoch size** | 8192 | Examples per epoch; enough for stable training (~40 batches/epoch at bs=256) |
| **Loader workers** | 4 | Parallel disk I/O; reduce if VRAM pressure or if training on slow storage |
| **Input color mode** | grayscale only | Force single-channel preprocessing for all train/val examples |
| **Augmentation** | enabled | Keep: random crop, flip, brightness/contrast help synthetic→real generalization |
| **Seed** | 0 | Fixed for reproducibility |

### Launch Command

```bash
scripts/run_train_tmux.sh \
  --run-name sceneflow_40x60_indoor_poc \
  --epochs 20 \
  --batch-size 256 \
  --learning-rate 5e-4 \
  --chunk-size 256 \
  --train-epoch-size 8192 \
  --gpu 1 \
  -- \
  --sceneflow-dataset "olivermao/sceneflow" \
  --sceneflow-limit 0 \
  --sceneflow-frac 1.0 \
  --driving-stereo-frac 0.0 \
  --kitti-frac 0.0 \
  --target-height 40 \
  --target-width 60 \
  --max-disp 24 \
  --feature-channels 8 \
  --input-color grayscale \
  --input-fit-mode pad \
  --no-augment-off
```

If `--input-color` is not currently available in the training CLI, implement it as a no-op compatibility flag first and enforce grayscale in the dataset loader path.

---

## Expected Outcomes

### Training Performance

- **Time to convergence**: ~20–30 minutes on single GPU (much faster than full-resolution training)
- **Memory footprint**: Model ~2–3 MB; batch ~600 KB; safe margin on typical GPU
- **Estimated MAE**: 0.3–0.5 pixels (SceneFlow is noiseless, so very low error expected)

### Model Size

- Saved `.keras` file: ~200–300 KB (tiny model, suitable for embedded/FPGA)

### FPGA Synthesis at 40×60

Using hls4ml with the converted feature extractor (40×60 input):

```
FIFO Depth ≈ 40 × 60 = 2400 pixels per FIFO
FIFO BRAM ≈ 1,300 BRAM_18K (vs 21,056 at 160×288)
Instance BRAM ≈ 45 BRAM_18K (from weights)
Total BRAM ≈ 1,350 (vs 21,100)
Utilization ≈ 185% (fits within 730 with row-streaming optimization or constraint relaxation)
```

With grayscale enforced end-to-end, additional BRAM pressure from multi-channel input buffering is reduced. This does not remove the need for row-streaming/reuse optimization, but it improves feasibility margin on XC7A200T.

---

## Evaluation

After training, run:

```bash
python -m v2.evaluation.evaluate_run \
  --run-dir "logs/sceneflow_40x60_indoor_poc_<timestamp>" \
  --sample-count 6 \
  --eval-chunk-size 256
```

This will compute:
- **MAE on SceneFlow holdout (1024 unseen examples)**
- Qualitative sample visualizations
- Per-pixel accuracy metrics on valid regions

**Expected holdout MAE**: 0.5–1.0 pixels (synthetic data is clean but small resolution introduces discretization)

---

## Post-Training: Convert to hls4ml

Once training is complete:

```bash
KERAS_BACKEND=torch uv run python -m v2.hls4ml.convert_student \
  --model "logs/sceneflow_40x60_indoor_poc_<timestamp>/checkpoints/best.keras" \
  --parity-batch "<generated-parity>.npz" \
  --output-dir "logs/sceneflow_40x60_indoor_poc_<timestamp>/hls4ml/fixed16_6_rf144" \
  --precision fixed<16,6> \
  --part xc7a200tfbg484-1 \
  --clock-period 10 \
  --reuse-factor 144 \
  --strategy Resource
```

Then synthesize via Vitis HLS 2025.2 and verify fit within resource budget.

---

## Camera Streaming: Practical Considerations

### Question: "Is there enough 'space' to downsample to 40×60?"

**Answer: Yes, abundantly.** Here's the breakdown:

#### 1. **Optical/Hardware Space**
- Modern stereo camera modules (e.g., Intel RealSense, OAK-D, Zed 2i) ship at various resolutions: 640×480, 1280×720, 1920×1080
- **Downsampling 640×480 → 40×60**: Reduction factor 16×; bilinear or nearest-neighbor downsampling is trivial
- **Form factor**: Stereo rig size depends on *baseline* (distance between lenses), not resolution. A 5–10cm baseline is standard for indoor robotics, fits in a ~10cm module.
- **Lens distortion**: Rectified stereo rigs work at any resolution post-rectification; downsampling doesn't break rectification (if done carefully)

#### 2. **Bandwidth**
- **At-camera grayscale**: 640×480×2 (stereo)×8-bit = 614 KB/frame. At 30 Hz: **18.4 MB/s** (USB 3.0: 400 MB/s ✓)
- **If camera outputs RGB and conversion is external**: 640×480×2×24-bit = 1.84 MB/frame. At 30 Hz: **55.3 MB/s** before grayscale conversion
- **Post-downsampling**: 40×60×2×8-bit = **4,800 bytes/frame**. At 30 Hz: **144 KB/s** (trivial, even over USB 2.0)
- **FPGA interface**: AXI streaming at the downsampled rate; ~1 Mbps data throughput easily handled by XC7A200T

#### 3. **Memory on FPGA**
- Storing one 40×60 stereo frame pair: 4800 bytes
- Storing entire inference pipeline + buffers: <2 MB BRAM
- **Available on XC7A200T**: 730 BRAM_18K = ~13 MB. Ample headroom.

#### 4. **Preprocessing on Host CPU/GPU**
- Option A: **On-device** (e.g., in the stereo camera module or edge compute): Many modern modules (OAK-D) support onboard resizing; cost is minimal.
- Option B: **FPGA preprocessor**: grayscale + downsampling in a streaming front-end before the model
- Option C: **Host CPU**: software grayscale + resize if FPGA preprocessor is not ready
- **Recommendation for real-time with no NEON/SSE**: avoid laptop proxy in normal operation; do grayscale/resize in FPGA streaming logic

#### 5. **Latency**
- Inference on model: ~5–10 ms (RF=144, small model, 65 MHz estimated Fmax on XC7A200T)
- Downsampling: <1 ms
- Camera capture: 33 ms @ 30 Hz
- **Total latency**: ~40–50 ms; suitable for real-time indoor robotics/VR

---

## Recommended Deployment Flow

```
Stereo Camera (e.g., OAK-D, 640×480)
    ↓
[Grayscale convert + downsample to 40×60 in FPGA streaming preprocessor]
    ↓
AXI-Stream into FPGA (io_stream interface)
    ↓
hls4ml Feature Extractor + Correlation + Regression
    ↓
40×60 Disparity Map output
    ↓
[Upsample disparity to original resolution via interpolation in post-processor, if needed]
    ↓
Depth map → Application (SLAM, obstacle avoidance, etc.)
```

**Advantages**:
- Minimal bandwidth footprint
- Fits entire pipeline on modest FPGA
- Low latency (real-time capable)
- Scalable to multiple cameras

---

## Alternative: If You Want to Keep 160×288

If the POC succeeds and you want to scale back up to 160×288 later:

1. Retrain with higher resolution (need UltraScale+ FPGA or custom row-streaming HLS)
2. Use the 40×60 model as initialization weights → faster convergence
3. Or, keep 40×60 as-is and use it for low-latency real-time; upscale disparity maps post-inference

---

## Success Criteria

✓ Model trains to <1.0 MAE on SceneFlow holdout in <30 min  
✓ hls4ml conversion succeeds without errors  
✓ Vitis HLS synthesis fits within XC7A200T resource budget (BRAM_18K < 730)  
✓ Synthesis Fmax ≥50 MHz (achievable with RF=144, 65 MHz typical)  
✓ End-to-end latency <50 ms on real stereo feed  

