	# Stereo Depth Estimation — FPGA Project Report

## Overview

This project implements a real-time stereo depth estimation pipeline in
Verilog, targeting a Xilinx FPGA in Vivado. The input is a simulated stereo
camera stream: two 512×512 grayscale frames delivered as an **interleaved**
pixel stream — for each column `c` in a row, the right-eye pixel is sent
first, immediately followed by the left-eye pixel. The output is a per-pixel
disparity map produced by Local SAD (Sum of Absolute Differences) matching
over a 5×5 window and a search range of 16 disparities.

The project is simulation-only. A testbench reads synthetic image data from
`.hex` files, streams it through the Verilog pipeline, and writes the output
disparity map to a `.hex` file that can be visualised with a Python script.

---

## Project Structure

```
phys476/
├── v1/
│   ├── rtl/sources_1/new/
│   │   ├── line_buffer.v      — BRAM-backed row store (flat 1-D, wr_en_out + wr_en_d1_out)
│   │   ├── sliding_window.v   — 5×5 patch assembler
│   │   ├── sad_unit.v         — sequential SAD accumulator (25-cycle)
│   │   └── stereo_top.v       — top-level: interleaved splitter + 4-state FSM + single SAD unit
│   ├── rtl/sim_1/new/
│   │   └── stereo_tb.v        — testbench: interleaved stream, v1-local paths, valid/ready
│   ├── data/
│   │   ├── left_img.hex       — 512×512 synthetic left-eye image
│   │   ├── right_img.hex      — 512×512 synthetic right-eye image (shifted)
│   │   └── gt_disparity.hex   — ground-truth disparity map
│   ├── output/
│   │   ├── depth_out.hex      — simulation output (written by testbench)
│   │   └── depth_map.png      — visualiser output (2×2 layout)
│   └── plot_depth.py          — Python visualiser
└── REPORT.md                  — this file
```

---

## Architecture

### Data flow

```
pixel_in (8-bit interleaved stream: right[c] then left[c] for each column c)
    │
    ▼
┌─────────────┐
│   Splitter  │  col_cnt[0]=0 → right eye (even slots)
│             │  col_cnt[0]=1 → left eye  (odd slots)
└──────┬──────┘
       │                         │
  left_wr_en               right_wr_en
       │                         │
  ┌────▼────┐               ┌────▼────┐
  │line_buf │               │line_buf │   5 rows × 512 cols each, BRAM-backed
  │wr_en_out│               │wr_en_out│
  │         │               │wr_en_d1_out (1-cycle early, for right sw)
  └────┬────┘               └────┬────┘
       │ (wr_en_out)              │ (wr_en_d1_out)
  ┌────▼──────┐             ┌────▼──────┐
  │sliding_win│             │sliding_win│   5×5 pixel patch, shift-register
  └────┬──────┘             └────┬──────┘
       │                         │
  patch_left               patch_right_base
                                 │
                          ┌──────▼──────────┐
                          │  Delay chain    │   16 × 200-bit FF registers
                          │  d=0..15        │   patch_right_delayed[d]
                          └──────┬──────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   stereo_top FSM        │
                    │   IDLE→HOLD(3)→SNAP     │
                    │        →RUNNING         │
                    │   single sad_unit,      │
                    │   time-multiplexed ×16  │
                    └────────────┬────────────┘
                                 │
                            disp_out (4-bit)
```

### Module descriptions

| Module | Purpose | Key internals |
|---|---|---|
| `line_buffer` | Stores the last NUM_ROWS rows of pixels | Flat 1-D BRAM (4096×8), rotating bank pointer, synchronous read; exposes `wr_en_out` (2-cycle delayed) and `wr_en_d1_out` (1-cycle delayed) |
| `sliding_window` | Converts vertical column to a 5×5 2-D patch | Per-row shift register (5 deep × 5 rows = 25 FFs per image), generate loop for synthesis-safe indexing |
| `sad_unit` | Computes SAD between two 5×5 patches | Sequential 25-cycle accumulator: one subtractor + one adder, `+:` indexed part-select |
| `stereo_top` | Top-level manager | 4-state FSM (IDLE/HOLD/SNAP/RUNNING), interleaved splitter (`col_cnt[0]`), single `sad_unit` instance, running-minimum register |
| `stereo_tb` | Testbench | `$readmemh` with v1-local relative paths, interleaved right-then-left column pairs, valid/ready handshake, writes `depth_out.hex` |

---

## Problem Log and Design Decisions

### Problem 1 — `sliding_window.v`: synthesis error "r is not a constant"

**Error:**
```
[Synth 8-35] 'r' is not a constant [sliding_window.v:63]
[Synth 8-9960] range must be bounded by constant expressions
```

**Root cause:**  
The original `always` block used `r` (an `integer` loop variable) as the
high/low bounds of a bit-select on `col_in`:

```verilog
// BROKEN — 'r' is a runtime integer, not a constant
col_in[(r+1)*DATA_WIDTH-1 : r*DATA_WIDTH]
```

Verilog's `signal[hi:lo]` part-select syntax requires **both bounds to be
constants** known at elaboration time. An `integer` is a runtime quantity and
is rejected by Vivado's synthesiser.

**Fix:**  
The entire shift-and-insert loop was moved into a `generate / for (genvar
gr_sh ...)` block. A `genvar` is resolved at elaboration time, so the slice
bounds become constants (e.g. `col_in[7:0]`, `col_in[15:8]`, …). The inner
column loop (`c_i`) remains a runtime `integer` because array *index* accesses
(unlike part-select bounds) are permitted to be runtime variables.

```verilog
// FIXED — gr_sh is a genvar, resolved at elaboration time
wire [DATA_WIDTH-1:0] col_pixel;
assign col_pixel = col_in[(gr_sh+1)*DATA_WIDTH-1 : gr_sh*DATA_WIDTH];
```

---

### Problem 2 — First DRC: 92,630 LUTs required, 63,400 available

**Error:**
```
[DRC UTLZ-1] This design requires 92630 LUT as Logic cells
but only 63400 compatible sites are available.
```

**Root cause analysis:**  
The synthesis log showed that the dominant contributors were in `sad_unit` and
`stereo_top`:

| Contributor | LUT estimate | Reason |
|---|---|---|
| 25 parallel ABS-diff datapaths × 16 instances | ~8,000 | Each 9-bit subtractor + conditional negate ~20 LUTs; replicated 16× |
| 25-operand unrolled adder tree × 16 instances | ~4,224 | Binary adder tree on 25 × 9-bit values; replicated 16× |
| 4-stage parallel comparator tree | ~105 | 15 × 16-bit comparators |
| `line_buffer` mod-5 bank address logic | ~30 | Modulo by non-power-of-2 requires a divider circuit |

Total: ~12,000+ LUTs from SAD alone, with additional routing pressure from
16 × 200-bit fanout buses.

**Fixes applied:**

#### Fix A — `sad_unit.v`: replace parallel datapath with sequential accumulator

Instead of 25 parallel subtractors and a 25-input adder tree (synthesised
once per SAD unit, replicated 16 times), the new implementation uses a single
subtractor and single adder that iterate over the 25 pixel pairs over 25
consecutive clock cycles.

- LUT cost: ~750 → ~30 per instance
- Latency: 2 cycles → 26 cycles per SAD result
- The `+:` indexed part-select operator is used for the variable pixel index,
  which is synthesis-safe (variable base, constant width)

#### Fix B — `stereo_top.v`: replace 16 parallel SAD units with one time-multiplexed unit

Instead of instantiating 16 copies of `sad_unit` in a `generate` loop, a
single instance is driven by a 3-state FSM:

- **`ST_IDLE`**: waits for `both_valid`
- **`ST_SNAP`**: latches `patch_left` and all 16 `patch_right_delayed[d]`
  into snapshot registers; kicks off SAD for `d=0`
- **`ST_RUNNING`**: on each `sad_valid_out`, updates a running `min_cost` /
  `min_disp` register; starts next disparity; emits `disp_out` after `d=15`

The 4-stage parallel comparator tree was also removed; it is replaced by the
running-minimum register (one 13-bit comparator, one 4-bit mux, ~5 LUTs).

The pixel splitter was simplified: since `IMG_WIDTH = 512 = 2^9`, the MSB of
`col_cnt` is 0 for the left half and 1 for the right half — a zero-LUT decode
replacing a 10-bit comparator.

#### Fix C — `line_buffer.v` v2: power-of-2 bank count

`NUM_ROWS = 5` banks were padded to `NUM_BANKS = 8` (next power of two).
Bank wrap-around changed from `% 5` (a mod-by-5 divider, ~9 LUTs) to
`(wr_bank == NUM_BANKS-1) ? 0 : wr_bank + 1` (a simple comparator).

---

### Problem 3 — Second DRC: still 92,630 LUTs (unchanged)

**Error:** Identical LUT count as Problem 2 despite rewriting all three files.

**Diagnosis:**  
The synthesis log revealed the actual cause:

```
WARNING: [Synth 8-7186] Applying attribute ram_style = "block" is ignored,
object 'mem[0][0]' is not inferred as ram due to incorrect usage
[line_buffer.v:113]
```

This warning repeated for every single element of the array. The synthesis
report showed:

```
|  u_left_lb  | line_buffer  |  98777 LUTs |
|  u_right_lb | line_buffer  |  98778 LUTs |
```

**Root cause:**  
Vivado's BRAM inference engine requires a **1-D array** accessed by a single
address expression. The previous `line_buffer` declared storage as a 2-D
array:

```verilog
// BROKEN — 2-D array, Vivado cannot infer BRAM
reg [DATA_WIDTH-1:0] mem [0:NUM_BANKS-1][0:IMG_WIDTH-1];
...
mem[wr_bank][wr_addr] <= pixel_in;  // variable first index = rejected
```

When the first index is a runtime variable, Vivado treats every element as an
individual flip-flop. With `8 × 512 = 4096` elements × 8 bits = 32,768 FFs
per instance, and each FF needing a mux to select it, this produces ~98,000
LUTs per `line_buffer` instance — which is where almost all of the 92,630
reported LUTs were coming from.

The `(* ram_style = "block" *)` attribute was silently ignored because the
array shape did not match the inference pattern.

**Fix — `line_buffer.v` v3: flat 1-D array**

The 2-D array is replaced by a single 1-D array of depth
`NUM_BANKS × IMG_WIDTH`:

```verilog
// FIXED — 1-D array, Vivado infers RAMB36E2
(* ram_style = "block" *)
reg [DATA_WIDTH-1:0] mem [0:MEM_DEPTH-1];

wire [MEM_ADDR_W-1:0] wr_addr = {wr_bank, col_addr};
always @(posedge clk)
    if (wr_en) mem[wr_addr] <= pixel_in;
```

The bank and column are concatenated into a single flat address
(`{bank[2:0], col[8:0]}` = 12-bit address for depth 4096). Both the write and
read ports use this 1-D address, which matches Vivado's canonical
synchronous-RAM inference template and causes it to map the storage to
`RAMB36E2` block RAM primitives instead of flip-flops.

Read addresses are also registered one cycle before the read data is needed
(synchronous read), satisfying Vivado's requirement for BRAM inference.

**Expected resource change:**  
- Before: ~98,000 LUTs × 2 instances from failed BRAM inference
- After: 2 × `RAMB36E2` block RAM primitives, ~0 LUTs

**After re-running synthesis**, the total LUT count should drop well below the
63,400 available on the target device.

> **Note:** After editing source files, always reset the synthesis run in
> Vivado before re-synthesising (Flow Navigator → right-click Synthesis →
> Reset Synthesis Run). Vivado caches the previous netlist checkpoint and will
> report stale results if the run is not reset.

---

## Synthesis Resource Summary (Expected After All Fixes)

| Resource | Before fixes | After fixes (estimate) |
|---|---|---|
| LUT as Logic | 92,630 | < 5,000 |
| LUT6 | 84,624 | < 4,500 |
| Flip-Flops | ~7,686 | ~7,500 (shift chain dominates) |
| BRAM (RAMB36E2) | 0 (inference failed) | 4 (2 per line_buffer × 2 eyes) |
| DSP48 | 0 | 0 |

The dominant remaining resource after the fixes is the 16 × 200-bit
right-eye disparity shift chain (3,200 FFs), which is correct and expected.

---

## Latency and Throughput

| Stage | Latency |
|---|---|
| `line_buffer` fill (5 rows) | 5 × 512 = 2,560 pixel clocks |
| `sliding_window` fill (5 cols) | 5 pixel clocks |
| SAD per disparity (sequential) | 26 cycles |
| 16 disparities per output pixel | 16 × 26 = 416 cycles |
| **Total pipeline fill** | ~2,981 cycles |

At 100 MHz, one complete 512×512 depth frame takes approximately:

```
512 × 512 × 416 / 100,000,000 ≈ 1.09 seconds
```

This is acceptable for a lab simulation. A real-time implementation at
30 fps would require parallelism restored (or a faster clock / smaller
disparity range).

---

## Key Verilog Synthesis Lessons

1. **Part-select bounds must be constants.** `signal[hi:lo]` requires `hi` and
   `lo` to be elaboration-time constants. Use `genvar` in a `generate` block
   (not `integer` in an `always` block) when the index comes from a loop
   counter. The `+:` indexed part-select (`signal[base +: width]`) is the
   correct way to use a variable base with a constant width.

2. **BRAM inference requires a 1-D array.** Vivado only infers `RAMB36E2`
   from a 1-D `reg` array with a single address, a synchronous write, and a
   synchronous (registered) read. A 2-D array with any variable index is
   treated as flip-flops regardless of the `ram_style` attribute.

3. **Parallel instantiation is the primary LUT budget risk.** Replicating a
   datapath N times (via `generate`) multiplies its LUT cost by N. The
   standard mitigation is time-multiplexing: one instance driven by an FSM
   that cycles through all N inputs over N clock cycles.

4. **Always reset the synthesis run after editing sources.** Vivado caches
   synthesised netlists as `.dcp` checkpoints. Stale results will be reported
   if you re-run synthesis without first resetting the run.

---

## Simulation Debugging Log

The following problems were encountered and resolved after the RTL was
complete and simulation was first attempted.

---

### Sim Bug 1 — xelab linker: `crt1.o` / `crti.o` not found

**Symptom:**
```
ERROR: [XSIM 43-3409] Failed to link the design
ld: cannot find crt1.o: No such file or directory
ld: cannot find crti.o: No such file or directory
```

**Root cause:**  
xelab uses the host GCC toolchain to link the compiled simulation objects.
The linker searches `LIBRARY_PATH` for startup libraries (`crt1.o`, `crti.o`).
On this system the libraries live in `/usr/lib64` but `LIBRARY_PATH` was unset.

**Fix:**
```bash
export LIBRARY_PATH=/usr/lib64:${LIBRARY_PATH:-}
```

---

### Sim Bug 2 — xsim: `libncurses.so.5` not found

**Symptom:**
```
xsim: error while loading shared libraries: libncurses.so.5:
cannot open shared object file: No such file or directory
```

**Root cause:**  
xsim is bundled with its own older shared libraries under the Vivado Flatpak
installation tree. The system only provides `libncurses.so.6`; xsim requires
version 5 explicitly.

**Fix:**
```bash
export LD_LIBRARY_PATH=\
  /home/aericio/.var/app/com.github.corna.Vivado/data/.xinstall/xic/lib/lnx64.o/Rhel/10:\
  ${LD_LIBRARY_PATH:-}
```
This prepends the Vivado-bundled `Rhel/10` compatibility library directory,
which contains `libncurses.so.5`.

---

### Sim Bug 3 — `$readmemh` cannot find hex files

**Symptom:**  
Simulation ran but all pixel values were `x` throughout; `$readmemh` silently
failed.

**Root cause:**  
xsim resolves `$readmemh` paths relative to its working directory
(`phys476project.sim/sim_1/behav/xsim/`), not the source tree. The hex files
are now kept under `v1/data/`, so the testbench needs explicit paths that stay
valid after the repo reorganization.

**Fix:**  
Use explicit v1-local relative paths in all `$readmemh` calls so the preserved
baseline remains self-contained under `v1/`:
```verilog
$readmemh("../../data/left_img.hex",  left_img);
$readmemh("../../data/right_img.hex", right_img);
```
Output is written to the matching `v1/output/` path:
```verilog
fd = $fopen("../../output/depth_out.hex", "w");
```

---

### Sim Bug 4 — Only ~1,195 output pixels produced (expected ~249,936)

**Symptom:**  
`depth_out.hex` contained only ~1,195 lines instead of the expected ~249,936.

**Root cause:**  
`stereo_top` had no backpressure signal. The pixel stream flowed at 1 pixel
per clock regardless of whether the SAD engine was busy. The FSM is stalled for
~416 cycles per output pixel, so almost every trigger was missed and the input
stream raced ahead.

**Fix:**  
Added a `ready` output to `stereo_top`, driven combinatorially:
```verilog
assign ready = (state == ST_IDLE);
```
The testbench `send_pixel` task was updated to implement a valid/ready
handshake — it holds `pix_valid` high and waits for a rising edge where
`ready` is also high before advancing to the next pixel.

---

### Sim Bug 5 — `line_buffer` bank wrap: simulation mismatch with synthesis

**Symptom:**  
The read bank index occasionally computed the wrong bank, causing garbage pixel
columns in the output.

**Root cause:**  
The read-bank combinational expression used a conditional subtraction:
```verilog
// BROKEN
(wr_bank + k + 1 >= NUM_BANKS) ? (wr_bank + k + 1) - NUM_BANKS
                                : (wr_bank + k + 1)
```
With `k` as a `genvar`, the comparison was ambiguous in some Vivado versions
and produced incorrect results in simulation.

**Fix:**  
Replaced with a bitmask. Because `NUM_BANKS` is a power of two this is
equivalent and synthesises to zero LUTs:
```verilog
wire [BANK_W-1:0] rd_bank_comb;
assign rd_bank_comb = (wr_bank + k[BANK_W-1:0] + 1'b1) & (NUM_BANKS - 1);
```

---

### Sim Bug 6 — Output contains `x` values (`disp_out` never valid)

**Symptom:**  
All output lines in `depth_out.hex` were `0x` (unknown / high-impedance).

**Root cause:**  
The BRAM memory array (`mem`) is uninitialised in simulation. Before any
address has been written, a read returns `x`. The `x` value propagates through
the sliding window and SAD unit, causing `sad_out` to be `x`, which leaves
`min_cost` as `x` and `disp_out` permanently unknown.

**Fix:**  
Added an `initial` block to zero-initialise the BRAM array:
```verilog
integer init_i;
initial begin
    for (init_i = 0; init_i < MEM_DEPTH; init_i = init_i + 1)
        mem[init_i] = 8'h00;
end
```

---

### Sim Bug 7 — BRAM read latency misalignment: sliding window filled with stale data

**Symptom:**  
Output pixels were produced but contained visibly wrong values — the pipeline
primed after many more rows than expected and pixel data was offset.

**Root cause:**  
The synchronous BRAM has 2-cycle read latency: the address is registered on
one clock edge and the data appears one clock later. The sliding window's
`wr_en` was being driven by the raw (undelayed) `wr_en` signal, so it shifted
in the BRAM output one cycle too early (before the data had arrived).

**Fix:**  
Added a `wr_en_out` output to `line_buffer`: a 2-cycle pipeline-delayed version
of `wr_en`. This signal goes high exactly when the BRAM read data for the
corresponding pixel is valid on `pixel_out`. Both sliding windows are now
driven by their respective `wr_en_out`:
```verilog
reg wr_en_d1;
always @(posedge clk) begin
    wr_en_d1  <= wr_en;
    wr_en_out <= wr_en_d1;
end
```

---

### Sim Bug 8 — Testbench timeout: `#2500000000` overflows 32-bit signed integer

**Symptom:**  
The simulation appeared to finish instantly. The timeout fired at time 0 ns.

**Root cause:**  
Verilog time literals are 32-bit signed integers before the `timescale` unit
is applied. `2,500,000,000` exceeds `2^31 − 1 = 2,147,483,647` and wraps to
a small negative value, which the simulator treats as 0.

**Fix:**  
Replace the single large delay with a loop of smaller delays:
```verilog
integer r;
for (r = 0; r < 1600; r = r + 1)
    #1000000;  // 1 ms each × 1600 = 1.6 s total budget
```

---

### Sim Bug 9 — Only ~86,000 output pixels (expected ~249,936): wrong stream order

**Symptom:**  
About one-third of the expected pixels were produced; the first valid output
appeared very late in the simulation.

**Root cause:**  
The testbench was streaming all left-eye pixels first, then all right-eye
pixels. When the FSM fired for the left pixel at column `c`, the right-eye
line buffer had not yet received any data for that row. The right sliding
window's `valid_out` was therefore low, suppressing the FSM trigger for most
of the frame.

**Intermediate fix (later superseded):**  
Swapped the testbench loop order to send the right-eye row before the left-eye
row, and updated the pixel splitter to treat `col_cnt[9] = 0` as right eye.
This produced the correct pixel count but left the delay chain misaligned by
~493 columns (see Bug 11).

**Final fix:**  
The stream was changed to fully interleaved column pairs (Bug 11, Change A),
which simultaneously fixed the pixel count and the delay-chain alignment.
The half-row ordering described above no longer exists in the codebase.

---

### Sim Bug 10 — FSM triggered 2 cycles before BRAM data settled: `ST_HOLD` added

**Symptom:**  
~259,000 pixels produced but accuracy was only ~18% (each of the 16 disparity
values appeared with almost equal frequency — ~16,224 times each — regardless
of the scene content).

**Cause (partial):**  
After the stream-order fix (Bug 9), the FSM triggered on the raw `left_wr_en`
signal. At that point `ready` deasserted immediately, but the BRAM still needed
2 more clock cycles to produce valid read data. The FSM moved straight to
`ST_SNAP` while the sliding windows contained stale data from the previous
column.

**Fix:**  
Added `ST_HOLD` state between `ST_IDLE` and `ST_SNAP`. The FSM waits 3 cycles
in `ST_HOLD` before snapping (the count was later extended from 2 to 3 as part
of the Bug 11 timing re-alignment):
```verilog
ST_IDLE: begin
    if (left_wr_en & left_sw_valid & right_sw_valid) begin
        state    <= ST_HOLD;
        hold_cnt <= 0;
    end
end
ST_HOLD: begin
    hold_cnt <= hold_cnt + 1;
    if (hold_cnt == 2'd2) state <= ST_SNAP;
end
```

---

### Sim Bug 11 — Accuracy 18.1%: right-eye delay chain fundamentally misaligned

**Symptom:**  
After Bug 10 was fixed the simulation produced ~259,000 pixels with a
perfectly uniform distribution across all 16 disparity values. Accuracy
remained at 18.1% (RMSE 5.58 px).

**Root cause (confirmed by analysis):**  
The stream layout at the time was *half-row*: all 512 right-eye pixels for
a row were sent before any left-eye pixel. The delay chain shifts on every
accepted right-eye pixel. By the time the FSM fired for left column `c = 19`
(the first valid column), the delay chain had already been shifted 512 times
and held patches from right columns 511, 510, …, 496. But for disparity `d`
to be correct, `delay[d]` must hold the right-eye patch centred at column
`c − d` — i.e. columns 19, 18, …, 3. The mismatch of ~493 columns meant
all 16 SAD costs were nearly equal (comparing unrelated image regions),
producing the observed uniform output.

**Fix — three coordinated changes:**

#### Change A — Testbench: interleaved pixel stream

Instead of right-row then left-row, the testbench now sends column pairs:
for each column `c`, right pixel `c` is sent immediately before left pixel
`c`. This means that when the FSM fires at left column `c`, the delay chain
has most recently advanced for right column `c` — giving exact column-level
alignment.

```verilog
for (col = 0; col < IMG_WIDTH; col = col + 1) begin
    send_pixel(right_img[row * IMG_WIDTH + col]);  // right first
    send_pixel(left_img [row * IMG_WIDTH + col]);  // then left
end
```

#### Change B — `stereo_top`: pixel splitter updated for interleaved stream

`col_cnt[9]` (MSB, half-row boundary) is replaced by `col_cnt[0]` (LSB,
alternating):
```verilog
wire is_right = ~col_cnt[0];  // even slot → right eye
wire is_left  =  col_cnt[0];  // odd slot  → left eye
```

#### Change C — `line_buffer` + `stereo_top`: timing re-alignment

With interleaved streaming the right pixel at column `c` arrives 1 cycle
before the left pixel at column `c`. For `delay[0]` to hold the patch
centred at right column `c` (not `c − 1`) when `ST_SNAP` fires, the right
sliding window must shift in right column `c` before the delay chain samples
it.

The solution:
- Added `wr_en_d1_out` (1-cycle delayed `wr_en`) as a new output on
  `line_buffer`.
- The right sliding window is now driven by `right_lb.wr_en_d1_out` instead
  of `right_lb.wr_en_out`. This shifts the right patch in 1 cycle earlier
  (at BRAM-output cycle T+1) so the delay chain, which fires at T+2, captures
  the post-shift state containing right column `c`.
- `ST_HOLD` was extended from 2 to 3 cycles so that the left sliding window
  (still driven by `wr_en_out`) also has time to shift left column `c` in
  before `ST_SNAP` reads `patch_left`.

**Timing summary after the fix:**

| Cycle offset | Event |
|---|---|
| T | right[c] accepted; right BRAM write |
| T+1 | left[c] accepted; FSM IDLE→HOLD; right sw shifts right[c] in (wr_en_d1_out) |
| T+2 | HOLD cycle 1; delay chain latches post-shift right patch: `delay[0]` = {right[c−4..c]} |
| T+3 | HOLD cycle 2; left sw shifts left[c] in (wr_en_out) |
| T+4 | HOLD cycle 3 → ST_SNAP: `patch_left` = {left[c−4..c]}, `delay[d]` = {right[c−d−4..c−d]} |

**Result after fix:**  
Accuracy: **87.3%** (|err| ≤ 1, foreground), RMSE **2.62 px**.  
The remaining ~12.7% errors are expected boundary effects where the 5×5 SAD
window straddles two depth regions.

---

## Final Simulation Results

| Metric | Value |
|---|---|
| Output pixels | 259,579 |
| Accuracy (|err| ≤ 1, foreground) | **87.3%** |
| RMSE (foreground) | **2.62 px** |
| Disparity 4 accuracy | 87.8% |
| Disparity 8 accuracy | 88.8% |
| Disparity 12 accuracy | 85.3% |
