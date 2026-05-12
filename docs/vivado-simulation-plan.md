# Vivado Simulation Pipeline Plan

**Goal**: Drive the hls4ml feature extractor RTL through a full Vivado simulation —
feeding real image pixels in, collecting feature-map outputs, and validating numerically
against Python reference outputs.

**Working directory**: `v2/hls4ml/fixed16_6/`  
Copied from `logs/monkaa_crop16x9_sf25_fc32_h160_w288_bs128/hls4ml/fixed16_6/` (baseline synthesis,
full precision fixed<16,6>, no reuse factor tuning — all resources available).

---

## Starting Point

C/RTL synthesis has already completed. Everything needed for simulation is present:

| Artifact | Path | Status |
|---|---|---|
| 86 Verilog RTL files | `v2/hls4ml/fixed16_6/myproject_prj/solution1/syn/verilog/` | ✅ done |
| Top-level module | `myproject.v` | ✅ done |
| HLS C++ testbench | `myproject_test.cpp` | ✅ done |
| HLS build TCL | `build_prj.tcl` | ✅ done (all stages present) |
| Test data reference | `parity_batch.npz` (in logs run root) | ✅ needs conversion to .dat |
| Parity metrics | `parity_metrics.json` | ✅ MAE 0.021, max err 0.43 |
| Real-world stereo video | `v2/data/stereo-video.mp4` | ✅ 177 frames, NYC street scene |
| Video camera metadata | `v2/data/stereo-video.npz` | ✅ metric 3D tracks, 14k pts/frame |
| Extraction script | `v2/data/extract_perspective_stereo.py` | ✅ done, tested |
| Extracted 160×288 crops | `v2/data/perspective_stereo/` | ❌ not yet run (command below) |
| hls4ml output in v2/ | `v2/hls4ml/fixed16_6/` | ❌ not yet copied (command below) |

The build TCL already contains commented-out stages for `csim`, `cosim`, `export`, and `vsynth`.
All three stages below (co-sim, IP export, Vivado project) are wired up — they just need to be invoked.

**First-time setup — copy hls4ml output into v2/**:
```bash
cd /mnt/dev/xilinx/phys476
cp -r logs/monkaa_crop16x9_sf25_fc32_h160_w288_bs128/hls4ml/fixed16_6 v2/hls4ml/fixed16_6
```
All subsequent work runs from `v2/hls4ml/fixed16_6/`. The original in `logs/` is untouched.

---

## Available Tools

| Tool | Binary | Purpose |
|---|---|---|
| **Vitis HLS** | `/mnt/dev/xilinx/2025.2/Vitis/bin/vitis-run` | C sim, RTL co-sim, IP export |
| **Vivado** | `/mnt/dev/xilinx/2025.2/Vivado/bin/vivado` | Vivado project, behavioral sim, xsim |
| **xsim** | bundled in Vivado | RTL simulation backend (used by both cosim and Vivado sim) |
| **Python / uv** | system `uv` | Test vector generation, numeric comparison |

Settings scripts:
```bash
source /mnt/dev/xilinx/2025.2/Vitis/settings64.sh  # adds vitis-run to PATH
source /mnt/dev/xilinx/2025.2/Vivado/settings64.sh  # adds vivado to PATH
```

---

## Top-Level Module Interface

The synthesised DUT has an `ap_ctrl_hs` handshake and AXI-Stream data ports:

```
module myproject (
  // AXI-Stream input: one fixed<16,6> pixel per beat
  input  [15:0]  image_r_TDATA,
  input          image_r_TVALID,
  output         image_r_TREADY,

  // AXI-Stream output: 16 feature channels × 16 bits = 256-bit bus, one pixel position per beat
  output [255:0] layer7_out_TDATA,
  output         layer7_out_TVALID,
  input          layer7_out_TREADY,

  // Control (ap_ctrl_hs)
  input          ap_clk,
  input          ap_rst_n,   // active-low synchronous reset
  input          ap_start,
  output         ap_done,
  output         ap_ready,
  output         ap_idle
);
```

**Throughput**: 160 × 288 = 46,080 input beats per inference.  
**Output beats**: 46,080 (one 256-bit word of 16 feature values per pixel position).  
**Reported latency**: 893,498 clock cycles ≈ 8.9 ms at 100 MHz.

---

## Part / Board for Simulation

For **behavioral simulation**, the target part has zero effect on simulation correctness.
The RTL files already contain the design; xsim simulates the netlist as-is.

For the **Vivado project**, use:

```
xcvu9p-flga2104-2L-e   (VCU118 board, UltraScale+)
```

This avoids any resource overflow warnings during elaboration/synthesis-for-sim.
The `project.tcl` `$part` variable will need updating from `xc7a200tfbg484-1`.

---

## Test Data Sources

Two sources of test frames are available. Both produce 160×288 grayscale images in `fixed<16,6>` format.

### Source A — `parity_batch.npz` (8 SceneFlow frames, quick)

Available at `logs/monkaa_crop16x9_sf25_fc32_h160_w288_bs128/parity_batch.npz` (stays in logs/).  
Used by `gen_tb_vectors.py` (Stage 1). Good for fast co-sim turnaround.

### Source B — Real-world stereo video (NYC street scene)

`v2/data/stereo-video.mp4` — 177-frame VR180 stereo video.  
`v2/data/stereo-video.npz` — camera poses + 25,073 metric 3D tracks (1.89–22 m depth).

**Extraction command** (run once, ~2 min):
```bash
cd /mnt/dev/xilinx/phys476
uv run python -m v2.data.extract_perspective_stereo \
    --video v2/data/stereo-video.mp4 \
    --metadata v2/data/stereo-video.npz \
    --output-dir v2/data/perspective_stereo \
    --fov 90 \
    --out-height 160 --out-width 288 \
    --yaw-angles 0 45 90 -45 -90 \
    --frame-stride 3 \
    --export-hex
```

**Why `--fov 90`**: at 90° horizontal FoV the 2048-px-wide source maps to 1.78 source pixels per
output pixel — no upsampling needed.  Smaller FoV (e.g. 60°) would upsample 1.5×.

**Why `--out-height 160 --out-width 288`**: exact model input size. The model was trained on
160×288 and its convolutional layers are fixed — any other size is rejected.  
Do **not** use `--out-size 512` (the old default) for simulation test vectors.

**What the extraction produces**:
- `.npz` files with keys `left_image (160,288,1) float32`, `right_image`, `disparity (160,288) float32`,
  `valid_mask (160,288) uint8`. GT disparity comes from metric 3D tracks projected into the
  perspective crop — sparse (~7–16% pixel coverage) but metrically accurate.
- `.mem` files (46,080 lines of 4-digit hex) for direct `$readmemh` use in the SV testbench.
  These are `fixed<16,6>` encoded: `pixel_int16 = round(pixel_float * 64)`.

**Disparity scale note**: at 90° FoV, `f_x = 144 px`. With VR180 baseline B ≈ 63.5 mm,
disparity for this NYC scene ranges ~0.4–5.6 px (depths 2–20 m). Well within `max_disp=64`
but on the small end compared to SceneFlow training data. Fine for RTL simulation validation.

---

## Pipeline Overview

```
parity_batch.npz
     │
     ▼
[Stage 1] Python: generate .dat test vectors
     │  tb_data/tb_input_features.dat
     │  tb_data/tb_output_predictions.dat
     │
     ▼
[Stage 2] Vitis HLS co-simulation  (vitis-run --mode hls --tcl)
     │  Drives C++ testbench against xsim RTL backend
     │  Produces tb_data/rtl_cosim_results.log
     │  Quick pass/fail validation
     │
     ▼
[Stage 3] Vitis HLS IP export  (vitis-run --mode hls --tcl)
     │  export_design -format ip_catalog
     │  Produces myproject_prj/solution1/impl/ip/  (.zip)
     │
     ▼
[Stage 4] Vivado project + standalone simulation
          create_project  →  add RTL sources  →  write SV testbench
          →  launch_simulation  →  waveform analysis / numeric check
```

---

## Stage 1: Generate Test Vectors (Python)

The C++ testbench (`myproject_test.cpp`) reads two plain-text files:

- `tb_data/tb_input_features.dat` — one line per inference, space-separated float values  
  representing the flattened input image (160 × 288 = 46,080 floats).
- `tb_data/tb_output_predictions.dat` — one line per inference, space-separated float values  
  representing the expected output (160 × 288 × 16 = 737,280 floats).

The SV testbench (Stage 4b) also needs:
- `tb_data/tb_input_hex.mem` — 46,080 lines of 4-digit hex, one `fixed<16,6>` pixel per line.

**Option A — from `parity_batch.npz`** (existing 8 SceneFlow frames, fast):

**Script to write**: `v2/hls4ml/gen_tb_vectors.py`

```python
"""Generate tb_data .dat files from parity_batch.npz for hls4ml co-simulation."""
import argparse, pathlib
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parity-batch", required=True)
    ap.add_argument("--hls-dir",      required=True,
                    help="Path to fixed16_6/ hls4ml output dir")
    ap.add_argument("--model",        required=True,
                    help="Path to feature_extractor.keras for reference output")
    args = ap.parse_args()

    import os; os.environ.setdefault("KERAS_BACKEND", "torch")
    import keras
    from v2.models.keras_student import CorrelationCostVolume, _masked_smooth_l1, _mean_abs_error_valid

    data   = np.load(args.parity_batch)
    images = data["images"]          # (N, 160, 288, 1) float32 in [0,1]
    N      = images.shape[0]

    model  = keras.models.load_model(
        args.model,
        custom_objects={"CorrelationCostVolume": CorrelationCostVolume,
                        "_masked_smooth_l1": _masked_smooth_l1,
                        "_mean_abs_error_valid": _mean_abs_error_valid},
    )
    ref_out = model.predict(images, batch_size=1)  # (N, 160, 288, 16)

    tb = pathlib.Path(args.hls_dir) / "tb_data"
    tb.mkdir(exist_ok=True)

    with open(tb / "tb_input_features.dat", "w") as f_in, \
         open(tb / "tb_output_predictions.dat", "w") as f_out:
        for i in range(N):
            inp = images[i].flatten()           # (46080,)
            out = ref_out[i].flatten()          # (737280,)
            f_in.write(" ".join(f"{v:.6f}" for v in inp) + "\n")
            f_out.write(" ".join(f"{v:.6f}" for v in out) + "\n")

    print(f"Wrote {N} inference pairs → {tb}")

if __name__ == "__main__":
    main()
```

**Run (from parity_batch.npz)**:
```bash
cd /mnt/dev/xilinx/phys476
KERAS_BACKEND=torch uv run python -m v2.hls4ml.gen_tb_vectors \
  --parity-batch logs/monkaa_crop16x9_sf25_fc32_h160_w288_bs128/parity_batch.npz \
  --hls-dir     v2/hls4ml/fixed16_6 \
  --model       v2/hls4ml/fixed16_6/feature_extractor.keras
```

This produces the two `.dat` files required by both C-sim and RTL co-sim.

**Option B — from extracted real-world crops** (Source B above):

The extraction script already writes `.mem` files alongside each `.npz` when `--export-hex` is
passed. Copy one to `tb_data/` for the SV testbench, and generate the `.dat` counterpart:

```bash
# Copy a specific frame's hex file to tb_data
cp v2/data/perspective_stereo/frame00000_yaw+000_pitch+00.mem \
   v2/hls4ml/fixed16_6/tb_data/tb_input_hex.mem

# Generate matching .dat (extend gen_tb_vectors.py to accept --npz-input)
KERAS_BACKEND=torch uv run python -m v2.hls4ml.gen_tb_vectors \
  --npz-input   v2/data/perspective_stereo/frame00000_yaw+000_pitch+00.npz \
  --hls-dir     v2/hls4ml/fixed16_6 \
  --model       v2/hls4ml/fixed16_6/feature_extractor.keras
```

**Stage 4c is already done**: `export_fixed16_hex()` is implemented in `extract_perspective_stereo.py`
and called when `--export-hex` is passed. The `.mem` files it produces are the exact format needed
by `$readmemh` in the SV testbench.

---

## Stage 2: HLS C Simulation + RTL Co-simulation

The `build_prj.tcl` accepts `-csim`, `-cosim`, `-synth`, `-export`, `-vsynth` flags via `build_opt.tcl`.
C synthesis is already done, so only csim and cosim need to run.

### 2a. C Simulation (software model, fast sanity check)

```bash
cd v2/hls4ml/fixed16_6
source /mnt/dev/xilinx/2025.2/Vitis/settings64.sh
vitis-run --mode hls --tcl build_prj.tcl -csim
```

Produces `tb_data/csim_results.log` — the C model's output values.  
Already run (file exists); re-run only if test vectors change.

### 2b. RTL Co-simulation (RTL under xsim, driven by C++ testbench)

```bash
cd v2/hls4ml/fixed16_6
source /mnt/dev/xilinx/2025.2/Vitis/settings64.sh
vitis-run --mode hls --tcl build_prj.tcl -cosim
```

What this does internally:
1. Recompiles `myproject_test.cpp` with `-DRTL_SIM`
2. Runs `cosim_design -trace_level all -setup` (generates `run_sim.tcl` + xsim project)
3. Patches `run_sim.tcl` to remove recursive waveform logging
4. `cd myproject_prj/solution1/sim/verilog/ && source run_sim.tcl` → launches xsim
5. Writes `tb_data/rtl_cosim_results.log`
6. Prints a pass/fail report at `myproject_prj/solution1/sim/report/myproject_cosim.rpt`

**Expected runtime**: Long — ~893,498 cycles × 8 frames ≈ 7M simulation cycles.  
Use tmux (`scripts/run_train_tmux.sh` pattern) or `nohup`. Budget 30–90 minutes.

### 2c. Validation (compare csim vs cosim outputs)

```bash
vitis-run --mode hls --tcl build_prj.tcl -cosim -validation
```

Or manually (from `v2/hls4ml/fixed16_6/`):
```bash
diff tb_data/csim_results.log tb_data/rtl_cosim_results.log
```

The existing `build_prj.tcl` validation proc does a file diff and prints `INFO: Test PASSED` or `ERROR: Test failed`.

---

## Stage 3: IP Export

Package the synthesised design as a Vivado IP catalog entry:

```bash
cd v2/hls4ml/fixed16_6
source /mnt/dev/xilinx/2025.2/Vitis/settings64.sh
vitis-run --mode hls --tcl build_prj.tcl -export
```

Output: `v2/hls4ml/fixed16_6/myproject_prj/solution1/impl/ip/xilinx_com_hls_myproject_1_0.zip`

This ZIP is the packaged IP core. It contains:
- All Verilog RTL sources
- IP-XACT metadata (`component.xml`)
- Port definitions matching the AXI-Stream interface above

Alternatively, skip the zip and directly use the `syn/verilog/` directory as a file-set in Vivado.

---

## Stage 4: Vivado Project and Standalone Simulation

This stage creates a clean Vivado project, adds sources, writes a SystemVerilog testbench,
and runs behavioral simulation — fully decoupled from the HLS flow.

### 4a. TCL script: `v2/hls4ml/vivado_sim.tcl`

Create this script (invoked with `vivado -mode batch -source vivado_sim.tcl`):

```tcl
# vivado_sim.tcl — create Vivado project, add hls4ml RTL, run behavioral simulation

set hls_dir   [file normalize "v2/hls4ml/fixed16_6"]
set rtl_dir   "$hls_dir/myproject_prj/solution1/syn/verilog"
set tb_dir    "$hls_dir/tb_data"
set proj_name "myproject_sim"
set proj_dir  "v2/hls4ml/vivado_sim_project"
set part      "xcvu9p-flga2104-2L-e"   ;# simulation: part doesn't affect behavior

# ---- Create project ----
create_project $proj_name $proj_dir -part $part -force
set_property simulator_language Mixed [current_project]

# ---- Add all HLS-generated Verilog RTL ----
add_files -norecurse [glob $rtl_dir/*.v]
set_property file_type {Verilog} [get_files -filter {FILE_TYPE == "Verilog"}]

# ---- Add simulation testbench ----
add_files -fileset sim_1 -norecurse "v2/hls4ml/myproject_tb.sv"

# Set top for sim
set_property top myproject_tb [get_filesets sim_1]
set_property top_lib xil_defaultlib [get_filesets sim_1]

# ---- Launch behavioral simulation (xsim) ----
launch_simulation -simset sim_1 -mode behavioral

# Run for enough time to cover one full inference
#   893498 cycles × 10ns + margin = ~10ms
run 10 ms

# Save waveform database
save_wave_config "$proj_dir/${proj_name}.wcfg"

close_sim
close_project
exit
```

Run:
```bash
cd /mnt/dev/xilinx/phys476
source /mnt/dev/xilinx/2025.2/Vivado/settings64.sh
vivado -mode batch -source v2/hls4ml/vivado_sim.tcl \
       -log v2/hls4ml/vivado_sim.log \
       -journal v2/hls4ml/vivado_sim.jou
```

### 4b. SystemVerilog testbench: `v2/hls4ml/myproject_tb.sv`

The testbench must:
1. Drive `ap_clk` at 100 MHz (10 ns period)
2. Assert synchronous active-low reset for several cycles
3. Assert `ap_start`
4. Feed 46,080 pixel values in AXI-Stream format (TVALID/TDATA, wait for TREADY)
5. Accept 46,080 × 16-channel output beats (TREADY=1, capture TDATA)
6. Print output values or compare to expected

```systemverilog
`timescale 1ns/1ps
module myproject_tb;

  // ---- Clock and reset ----
  logic        clk   = 0;
  logic        rst_n = 0;
  always #5 clk = ~clk;   // 100 MHz

  // ---- DUT signals ----
  logic [15:0]  in_tdata;
  logic         in_tvalid = 0;
  logic         in_tready;
  logic [255:0] out_tdata;
  logic         out_tvalid;
  logic         out_tready = 1;   // always ready to accept output
  logic         ap_start   = 0;
  logic         ap_done;
  logic         ap_ready;
  logic         ap_idle;

  myproject dut (
    .image_r_TDATA    (in_tdata),
    .image_r_TVALID   (in_tvalid),
    .image_r_TREADY   (in_tready),
    .layer7_out_TDATA  (out_tdata),
    .layer7_out_TVALID (out_tvalid),
    .layer7_out_TREADY (out_tready),
    .ap_clk           (clk),
    .ap_rst_n         (rst_n),
    .ap_start         (ap_start),
    .ap_done          (ap_done),
    .ap_ready         (ap_ready),
    .ap_idle          (ap_idle)
  );

  // ---- Pixel data: loaded from $readmemh or task ----
  localparam H = 160, W = 288;
  localparam N_PIXELS = H * W;           // 46080
  localparam N_FEATURES = N_PIXELS * 16; // 737280

  logic [15:0] input_pixels  [0:N_PIXELS-1];
  logic [15:0] output_feats  [0:N_FEATURES-1];  // captured

  // Load pre-converted hex vectors (see Stage 1b below)
  initial $readmemh("tb_data/tb_input_hex.mem",  input_pixels);

  integer beat, feat_beat;

  initial begin
    // 1. Reset
    rst_n = 0;
    repeat(10) @(posedge clk);
    rst_n = 1;
    repeat(5)  @(posedge clk);

    // 2. Start
    ap_start = 1;
    @(posedge clk); #1;
    ap_start = 0;

    // 3. Stream in pixels
    for (beat = 0; beat < N_PIXELS; beat++) begin
      in_tdata  = input_pixels[beat];
      in_tvalid = 1;
      do @(posedge clk); while (!in_tready);
      #1;
    end
    in_tvalid = 0;

    // 4. Wait for ap_done
    @(posedge ap_done);
    $display("INFO: ap_done asserted — inference complete");

    // 5. Print first output word for sanity check
    $display("INFO: first output beat = %h", output_feats[0]);

    $finish;
  end

  // ---- Capture output beats ----
  initial begin
    feat_beat = 0;
    forever begin
      @(posedge clk);
      if (out_tvalid && out_tready) begin
        output_feats[feat_beat] = out_tdata[15:0];  // channel 0 of beat
        feat_beat++;
        if (feat_beat >= N_FEATURES) $display("INFO: all output beats captured");
      end
    end
  end

endmodule
```

**Note on pixel format**: `tb_input_hex.mem` needs 46,080 lines of 4-digit hex, one 16-bit
fixed<16,6> value per line. The Stage 1 Python script must also write this file.

### 4c. ~~Add hex vector output to Stage 1 Python script~~ (already done)

`export_fixed16_hex()` in `v2/data/extract_perspective_stereo.py` handles this. When running
the extraction with `--export-hex`, a `.mem` file is written alongside each `.npz` with exactly
46,080 lines of `fixed<16,6>` hex values — ready for `$readmemh`.

Format reminder:
```
# fixed<16,6>: pixel_int16 = clip(round(pixel_float * 2^6), -32768, 32767)
# Written as uint16 hex, 4 digits, one per line:
0031   # = 0x31 = 49 decimal → 49/64 = 0.766 (mid-gray pixel)
0026   # = 0x26 = 38 decimal → 38/64 = 0.594
...
```

---

## Stage 5: Waveform Analysis in Vivado GUI (optional)

After the batch simulation run, open the saved waveform database interactively:

```bash
source /mnt/dev/xilinx/2025.2/Vivado/settings64.sh
vivado v2/hls4ml/vivado_sim_project/myproject_sim.wcfg
```

Useful signals to probe:
- `image_r_TDATA` / `image_r_TVALID` / `image_r_TREADY` — input handshake
- `layer7_out_TDATA[15:0]` — feature channel 0 output (lowest 16 bits of 256-bit bus)
- `ap_start`, `ap_done`, `ap_idle` — control state transitions
- Any internal FIFO `num_data_valid` signals — to verify pipeline fill/drain

---

## Numeric Validation (Python post-processing)

After co-simulation (Stage 2) or Vivado simulation (Stage 4), validate RTL output numerically:

```bash
KERAS_BACKEND=torch uv run python - <<'EOF'
import numpy as np

# Load RTL cosim results (space-separated float lines)
with open("v2/hls4ml/fixed16_6/tb_data/rtl_cosim_results.log") as f:
    rtl_vals = np.array([[float(x) for x in line.split()] for line in f if line.strip()])

# Load Python reference
with open("v2/hls4ml/fixed16_6/tb_data/tb_output_predictions.dat") as f:
    ref_vals = np.array([[float(x) for x in line.split()] for line in f if line.strip()])

mae = np.mean(np.abs(rtl_vals - ref_vals))
maxe = np.max(np.abs(rtl_vals - ref_vals))
print(f"RTL vs Python MAE: {mae:.5f}  max: {maxe:.5f}")
# Expected: similar to parity_metrics.json (MAE ~0.02, max ~0.43)
EOF
```

Expected result: RTL output should match C-sim within quantisation error (~0.02 MAE).
Any larger discrepancy indicates a simulation or testbench bug.

---

## Summary: Implementation Checklist for New Session

```
[ ] Stage 0a: Copy hls4ml output to v2/: cp -r logs/.../hls4ml/fixed16_6 v2/hls4ml/fixed16_6
[ ] Stage 0b: Run extraction script to generate 160×288 perspective crops + .mem files
              (command in "Test Data Sources" section above)
[ ] Stage 1a: Write v2/hls4ml/gen_tb_vectors.py
[ ] Stage 1b: Run gen_tb_vectors.py → tb_data/*.dat  (from parity_batch.npz or extracted npz)
[ ] Stage 1c: Copy .mem from perspective_stereo/ to tb_data/ (or gen_tb_vectors.py writes it)
[ ] Stage 2a: Run vitis-run -csim  (verify csim_results.log refreshes)
[ ] Stage 2b: Run vitis-run -cosim  (long, use tmux)
[ ] Stage 2c: Diff csim vs cosim logs (expect PASSED)
[ ] Stage 3:  Run vitis-run -export → IP zip
[ ] Stage 4a: Write v2/hls4ml/vivado_sim.tcl
[ ] Stage 4b: Write v2/hls4ml/myproject_tb.sv
[ ] Stage 4c: ✅ DONE — export_fixed16_hex() in extract_perspective_stereo.py
[ ] Stage 4d: Run vivado -mode batch -source vivado_sim.tcl
[ ] Stage 5:  (optional) Open Vivado GUI, load .wcfg, probe waveforms
[ ] Stage 6:  Run Python numeric validation, confirm MAE ~0.02
```

---

## File Layout After Completion

```
v2/
  hls4ml/
    convert_student.py        (existing)
    gen_tb_vectors.py         ← NEW (Stage 1)
    vivado_sim.tcl            ← NEW (Stage 4a)
    myproject_tb.sv           ← NEW (Stage 4b)
    vivado_sim.log            ← NEW (Stage 4d)
    vivado_sim.jou            ← NEW (Stage 4d)
    fixed16_6/                ← COPIED (Stage 0a) from logs/.../hls4ml/fixed16_6
      build_prj.tcl
      myproject_test.cpp
      feature_extractor.keras
      tb_data/
        tb_input_features.dat     ← NEW (Stage 1)
        tb_output_predictions.dat ← NEW (Stage 1)
        tb_input_hex.mem          ← copied from perspective_stereo/ or gen_tb_vectors.py
        csim_results.log          (copied from logs/, re-generated by Stage 2a)
        rtl_cosim_results.log     ← NEW (Stage 2b)
      myproject_prj/
        solution1/
          syn/verilog/*.v   (86 RTL files, copied)
          sim/              ← NEW (Stage 2b)
          impl/ip/          ← NEW (Stage 3)
    vivado_sim_project/     ← NEW (Stage 4d)
      myproject_sim.xpr
      myproject_sim.wcfg
  data/
    extract_perspective_stereo.py  (existing, updated)
    stereo-video.mp4               (existing — 177 frames, NYC street)
    stereo-video.npz               (existing — camera poses + 3D tracks)
    perspective_stereo/            ← NEW (Stage 0b extraction)
      frame00000_yaw+000_pitch+00.npz   # left/right 160×288 float32, sparse GT disparity
      frame00000_yaw+000_pitch+00.mem   # 46080-line hex file for $readmemh
      ...  (one .npz + .mem per frame × yaw angle)
      manifest.json

logs/monkaa_crop16x9_sf25_fc32_h160_w288_bs128/hls4ml/fixed16_6/
  (original synthesis output — read-only reference, do not modify)
```
