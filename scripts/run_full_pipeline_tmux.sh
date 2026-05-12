#!/usr/bin/env bash
# run_full_pipeline_tmux.sh — Run csim, then wait for it to finish,
# then chain cosim → export → Vivado sim in a single tmux session.
# This avoids needing to manually trigger each stage.

set -euo pipefail

WORKSPACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HLS_DIR="$WORKSPACE_ROOT/v2/hls4ml/fixed16_6"
VITIS_SETTINGS="/mnt/dev/xilinx/2025.2/Vitis/settings64.sh"
VIVADO_SETTINGS="/mnt/dev/xilinx/2025.2/Vivado/settings64.sh"
SESSION_NAME="hls_pipeline"
LOG_DIR="$WORKSPACE_ROOT/v2/hls4ml/logs"
ATTACH="${1:-0}"

mkdir -p "$LOG_DIR"

LOG_FILE="$LOG_DIR/pipeline_$(date +%Y%m%d_%H%M%S).log"

# Helper that writes build_opt.tcl with the given opt values
OPT_WRITER='
write_opt() {
    local csim=$1 synth=$2 cosim=$3 valid=$4 export_ip=$5
    printf "array set opt { reset 0 csim %s synth %s cosim %s validation %s export %s vsynth 0 fifo_opt 0 }\n" \
        "$csim" "$synth" "$cosim" "$valid" "$export_ip" \
        > '"$HLS_DIR"'/build_opt.tcl
}
'

# Build the full command sequence
read -r -d '' PIPELINE_CMD <<'EOF_CMD' || true
source /mnt/dev/xilinx/2025.2/Vitis/settings64.sh

cd /mnt/dev/xilinx/phys476/v2/hls4ml/fixed16_6

echo "=== Stage 2a: C Simulation ===" && \
printf 'array set opt { reset 0 csim 1 synth 0 cosim 0 validation 0 export 0 vsynth 0 fifo_opt 0 }\n' > build_opt.tcl && \
vitis-run --mode hls --tcl build_prj.tcl && \
echo "=== csim DONE ===" && \

echo "=== Stage 2b: RTL Co-Simulation + Validation ===" && \
printf 'array set opt { reset 0 csim 0 synth 0 cosim 1 validation 1 export 0 vsynth 0 fifo_opt 0 }\n' > build_opt.tcl && \
vitis-run --mode hls --tcl build_prj.tcl && \
echo "=== cosim + validation DONE ===" && \

echo "=== Stage 3: IP Export ===" && \
printf 'array set opt { reset 0 csim 0 synth 0 cosim 0 validation 0 export 1 vsynth 0 fifo_opt 0 }\n' > build_opt.tcl && \
vitis-run --mode hls --tcl build_prj.tcl && \
echo "=== export DONE ===" && \

source /mnt/dev/xilinx/2025.2/Vivado/settings64.sh && \
cd /mnt/dev/xilinx/phys476 && \
echo "=== Stage 4d: Vivado Behavioral Simulation ===" && \
vivado -mode batch \
    -source v2/hls4ml/vivado_sim.tcl \
    -log v2/hls4ml/vivado_sim.log \
    -journal v2/hls4ml/vivado_sim.jou && \
echo "=== Vivado sim DONE ===" && \

echo "=== Stage 6: Numeric Validation ===" && \
cd /mnt/dev/xilinx/phys476 && \
KERAS_BACKEND=torch uv run python -m v2.hls4ml.validate_rtl_output \
    --hls-dir v2/hls4ml/fixed16_6

EOF_CMD

WRAPPED="{ $PIPELINE_CMD ; } 2>&1 | tee $LOG_FILE; echo \"Pipeline exit \$? — log: $LOG_FILE\""

echo "=== Full Simulation Pipeline Launcher ==="
echo "Log: $LOG_FILE"
echo "Session: $SESSION_NAME"
echo ""

tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true
tmux new-session -d -s "$SESSION_NAME" -x 220 -y 50
tmux send-keys -t "$SESSION_NAME" "$WRAPPED" Enter

echo "Launched tmux session '$SESSION_NAME'."
echo "Monitor: tmux attach -t $SESSION_NAME"
echo "Log:     tail -f $LOG_FILE"

if [[ "${ATTACH}" == "1" ]]; then
    exec tmux attach -t "$SESSION_NAME"
fi
