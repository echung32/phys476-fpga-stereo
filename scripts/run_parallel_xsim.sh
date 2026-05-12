#!/usr/bin/env bash
# run_parallel_xsim.sh — Reuse compiled xsim snapshot and run multiple frames in parallel.
#
# First run: compiles the new testbench via Vivado (needed once after SV changes)
#            and simulates frame 0 as a baseline check.
# Subsequent runs: --skip-compile reuses the existing xsim.dir/ directly.
#
# Usage:
#   # Full run: recompile + 2fps frames (every 5th, yaw +000), 16 parallel workers
#   bash scripts/run_parallel_xsim.sh
#
#   # Skip recompile (testbench already compiled):
#   bash scripts/run_parallel_xsim.sh --skip-compile
#
#   # Custom: all yaw angles, 4 frames, 8 workers
#   bash scripts/run_parallel_xsim.sh --skip-compile --all-yaw --n-frames 4 --jobs 8
#
# Outputs land in: v2/hls4ml/parallel_sim/<frame_stem>/tb_data/rtl_output_<stem>.{hex,f32}
#
set -euo pipefail

# ---- Defaults ---------------------------------------------------------------
WORKSPACE=/mnt/dev/xilinx/phys476
HLS_DIR=$WORKSPACE/v2/hls4ml/fixed16_6
VIVADO_SIM_TCL=$WORKSPACE/v2/hls4ml/vivado_sim.tcl
XSIM_DIR=$WORKSPACE/v2/hls4ml/vivado_sim_project/myproject_sim.sim/sim_1/behav/xsim
MEM_DIR=$WORKSPACE/v2/data/perspective_stereo
OUT_BASE=$WORKSPACE/v2/hls4ml/parallel_sim
LOG_DIR=$WORKSPACE/v2/hls4ml/logs
JOBS=$(nproc)          # default: use all cores
SKIP_COMPILE=0
YAW="+000"
ALL_YAW=0
N_FRAMES=13            # ~2fps from 59 available (every 5th)
FRAME_STEP=5           # take every Nth frame from sorted list

# ---- Parse args -------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-compile)  SKIP_COMPILE=1 ;;
    --all-yaw)       ALL_YAW=1 ;;
    --yaw)           YAW="$2"; shift ;;
    --n-frames)      N_FRAMES="$2"; shift ;;
    --frame-step)    FRAME_STEP="$2"; shift ;;
    --jobs)          JOBS="$2"; shift ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
  shift
done

mkdir -p "$OUT_BASE" "$LOG_DIR"
source /mnt/dev/xilinx/2025.2/Vivado/settings64.sh

# ---- Step 1: Recompile testbench via Vivado ---------------------------------
if [[ $SKIP_COMPILE -eq 0 ]]; then
  echo "==== Step 1: Recompiling testbench + running frame 0 via Vivado ===="
  COMPILE_LOG=$LOG_DIR/vivado_sim_compile_$(date +%Y%m%d_%H%M%S).log

  # Generate center-crop 160×288 hex mem from 512×512 perspective_stereo npz
  # (the raw .mem files are full 512×512; we need the center 160×288 = 46080 pixels)
  python3 - <<'PYEOF' "$MEM_DIR/frame00000_yaw+000_pitch+00.mem" "$HLS_DIR/tb_data/tb_input_hex.mem"
import sys, pathlib
src_lines = pathlib.Path(sys.argv[1]).read_text().splitlines()
# 512x512 source; center crop to 160x288: rows 176:336, cols 112:400
crop = []
for r in range(176, 336):
    crop.extend(src_lines[r*512+112 : r*512+400])
pathlib.Path(sys.argv[2]).write_text('\n'.join(crop) + '\n')
print(f'Cropped {len(crop)} lines to {sys.argv[2]}')
PYEOF
  # Remove any manifest so testbench uses single-frame fallback
  rm -f "$HLS_DIR/tb_data/frame_manifest.txt"

  vivado -mode batch -source "$VIVADO_SIM_TCL" 2>&1 | tee "$COMPILE_LOG"

  # Copy frame 0 output to parallel_sim/
  FRAME0_OUT=$OUT_BASE/frame00000_yaw+000_pitch+00
  mkdir -p "$FRAME0_OUT/tb_data"
  if [[ -f $XSIM_DIR/tb_data/rtl_output_frame0.hex ]]; then
    cp "$XSIM_DIR/tb_data/rtl_output_frame0.hex" "$FRAME0_OUT/tb_data/"
    cp "$XSIM_DIR/tb_data/rtl_output_frame0.f32" "$FRAME0_OUT/tb_data/" 2>/dev/null || true
    echo "Frame 0 output saved to $FRAME0_OUT/tb_data/"
  fi
  SKIP_FIRST=1
else
  echo "==== Skipping recompile — using existing xsim.dir/ ===="
  SKIP_FIRST=0
fi

# ---- Step 2: Select frames for parallel simulation --------------------------
if [[ $ALL_YAW -eq 1 ]]; then
  PATTERN="*.mem"
else
  PATTERN="*yaw${YAW}_pitch+00.mem"
fi

# Build sorted list of .mem files, subsampled at FRAME_STEP
mapfile -t ALL_MEMS < <(find "$MEM_DIR" -maxdepth 1 -name "$PATTERN" | sort)
SELECTED_MEMS=()
IDX=0
COUNT=0
for m in "${ALL_MEMS[@]}"; do
  if (( IDX % FRAME_STEP == 0 )) && (( COUNT < N_FRAMES )); then
    SELECTED_MEMS+=("$m")
    (( COUNT++ )) || true
  fi
  (( IDX++ )) || true
done

# If we compiled frame 0 above, skip it to avoid duplicating
if [[ $SKIP_FIRST -eq 1 ]]; then
  REMAINING_MEMS=()
  for m in "${SELECTED_MEMS[@]}"; do
    STEM=$(basename "$m" .mem)
    if [[ $STEM == "frame00000_yaw+000_pitch+00" ]]; then
      echo "  (skipping $STEM — already simulated in compile step)"
    else
      REMAINING_MEMS+=("$m")
    fi
  done
  SELECTED_MEMS=("${REMAINING_MEMS[@]+"${REMAINING_MEMS[@]}"}")
fi

echo "==== Step 2: Parallel simulation of ${#SELECTED_MEMS[@]} frame(s) with $JOBS workers ===="
for m in "${SELECTED_MEMS[@]}"; do
  echo "  $m"
done

# ---- Per-frame xsim runner --------------------------------------------------
run_frame() {
  local MEM_PATH="$1"
  local STEM
  STEM=$(basename "$MEM_PATH" .mem)
  local WORKDIR="$OUT_BASE/$STEM"
  local FRAME_LOG="$LOG_DIR/xsim_${STEM}.log"

  mkdir -p "$WORKDIR/tb_data"

  # Skip if already successfully completed
  if [[ -f $WORKDIR/tb_data/rtl_output_${STEM}.hex ]]; then
    echo "  [$(date +%H:%M:%S)] SKIP   $STEM (already done)"
    return 0
  fi

  # Symlink compiled xsim.dir so xsim can find the snapshot
  if [[ ! -L $WORKDIR/xsim.dir ]]; then
    ln -s "$XSIM_DIR/xsim.dir" "$WORKDIR/xsim.dir"
  fi
  # Copy xsim.ini (contains library paths, must not be symlinked — xsim resolves
  # relative paths in it from cwd, so a copy is safest)
  cp "$XSIM_DIR/xsim.ini" "$WORKDIR/xsim.ini"

  # Generate center-crop 160×288 hex mem from 512×512 source
  python3 - <<'PYEOF' "$MEM_PATH" "$WORKDIR/tb_data/tb_input_hex.mem"
import sys, pathlib
src_lines = pathlib.Path(sys.argv[1]).read_text().splitlines()
# 512x512 source; center crop rows 176:336, cols 112:400 = 160x288 = 46080 pixels
crop = []
for r in range(176, 336):
    crop.extend(src_lines[r*512+112 : r*512+400])
pathlib.Path(sys.argv[2]).write_text('\n'.join(crop) + '\n')
PYEOF

  # Write a minimal xsim TCL: run until $finish
  cat > "$WORKDIR/sim.tcl" <<'TCL'
run -all
TCL

  # Run xsim
  echo "  [$(date +%H:%M:%S)] START  $STEM"
  (
    cd "$WORKDIR"
    xsim myproject_tb_behav -tclbatch sim.tcl -log simulate.log 2>&1
  ) > "$FRAME_LOG" 2>&1

  # Check success — testbench writes hex file when all beats captured
  if [[ -f $WORKDIR/tb_data/rtl_output_frame0.hex ]]; then
    mv "$WORKDIR/tb_data/rtl_output_frame0.hex" "$WORKDIR/tb_data/rtl_output_${STEM}.hex"
    mv "$WORKDIR/tb_data/rtl_output_frame0.f32" "$WORKDIR/tb_data/rtl_output_${STEM}.f32" 2>/dev/null || true
    echo "  [$(date +%H:%M:%S)] DONE   $STEM → $WORKDIR/tb_data/"
  elif grep -q "Frame complete" "$FRAME_LOG"; then
    # Output file already named correctly (stem was passed correctly)
    echo "  [$(date +%H:%M:%S)] DONE   $STEM"
  else
    echo "  [$(date +%H:%M:%S)] ERROR  $STEM — check $FRAME_LOG"
  fi
}
export -f run_frame
export XSIM_DIR OUT_BASE LOG_DIR

# ---- Run in parallel using GNU parallel or xargs ----------------------------
if command -v parallel &>/dev/null; then
  printf '%s\n' "${SELECTED_MEMS[@]}" | parallel -j "$JOBS" run_frame {}
else
  # Fallback: xargs -P
  printf '%s\n' "${SELECTED_MEMS[@]}" | xargs -P "$JOBS" -I{} bash -c 'run_frame "$@"' _ {}
fi

echo ""
echo "==== All frames complete ===="
echo "Outputs: $OUT_BASE/**/tb_data/rtl_output_*.f32"
echo ""
echo "Generate video:"
echo "  KERAS_BACKEND=torch uv run python -m v2.hls4ml.render_feature_video \\"
echo "    --f32-dir $OUT_BASE \\"
echo "    --output  $WORKSPACE/v2/hls4ml/feature_video.mp4"
