#!/usr/bin/env bash
# run_sim_tmux.sh — Launch the full HLS + Vivado simulation pipeline in tmux.
#
# Stages executed in sequence:
#   1. Vitis HLS C simulation  (csim)
#   2. Vitis HLS RTL co-simulation  (cosim)  ← ~30-90 min
#   3. Vitis HLS IP export
#   4. Vivado batch behavioral simulation    ← ~30-90 min
#
# Usage (from workspace root):
#   bash scripts/run_sim_tmux.sh [--attach] [--dry-run]
#   bash scripts/run_sim_tmux.sh --stage csim     # run only csim
#   bash scripts/run_sim_tmux.sh --stage cosim    # run only cosim
#   bash scripts/run_sim_tmux.sh --stage export   # run only export
#   bash scripts/run_sim_tmux.sh --stage vivado   # run only Vivado sim
#
set -euo pipefail

WORKSPACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HLS_DIR="$WORKSPACE_ROOT/v2/hls4ml/fixed16_6"
VITIS_SETTINGS="/mnt/dev/xilinx/2025.2/Vitis/settings64.sh"
VIVADO_SETTINGS="/mnt/dev/xilinx/2025.2/Vivado/settings64.sh"
SESSION_NAME="hls_sim"
STAGE="all"
ATTACH=0
DRY_RUN=0
LOG_DIR="$WORKSPACE_ROOT/v2/hls4ml/logs"

usage() {
    cat <<'EOF'
Usage: scripts/run_sim_tmux.sh [options]

Options:
  --stage STAGE   One of: all, csim, cosim, export, vivado  (default: all)
  --attach        Attach to the tmux session after launch
  --dry-run       Print commands without executing
  --help          Show this message
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --stage)   STAGE="$2"; shift 2 ;;
        --attach)  ATTACH=1;   shift   ;;
        --dry-run) DRY_RUN=1;  shift   ;;
        --help)    usage; exit 0        ;;
        *) echo "Unknown argument: $1"; usage; exit 1 ;;
    esac
done

mkdir -p "$LOG_DIR"

# Build the command sequence for the requested stage(s)
# Emit an inline command that writes build_opt.tcl with desired opt values,
# then sources Vitis settings and runs vitis-run --mode hls --tcl.
# build_opt.tcl is always overwritten before the run to avoid stale settings.
build_hls_cmd() {
    local stage="$1"
    local csim=0 synth=0 cosim=0 validation=0 export_ip=0
    case "$stage" in
        csim)       csim=1 ;;
        cosim)      cosim=1 ;;
        validation) cosim=1; validation=1 ;;
        export)     export_ip=1 ;;
    esac
    local opt_block
    opt_block="array set opt { reset 0 csim $csim synth $synth cosim $cosim validation $validation export $export_ip vsynth 0 fifo_opt 0 }"
    echo "printf '%s\\n' '$opt_block' > $HLS_DIR/build_opt.tcl && source $VITIS_SETTINGS && cd $HLS_DIR && vitis-run --mode hls --tcl build_prj.tcl"
}

build_vivado_cmd() {
    echo "source $VIVADO_SETTINGS && cd $WORKSPACE_ROOT && vivado -mode batch -source v2/hls4ml/vivado_sim.tcl -log v2/hls4ml/vivado_sim.log -journal v2/hls4ml/vivado_sim.jou"
}

case "$STAGE" in
    all)
        CMDS=(
            "$(build_hls_cmd csim)"
            "$(build_hls_cmd validation)"
            "$(build_hls_cmd export)"
            "$(build_vivado_cmd)"
        )
        ;;
    csim)       CMDS=("$(build_hls_cmd csim)") ;;
    cosim)      CMDS=("$(build_hls_cmd cosim)") ;;
    validation) CMDS=("$(build_hls_cmd validation)") ;;
    export)     CMDS=("$(build_hls_cmd export)") ;;
    vivado)     CMDS=("$(build_vivado_cmd)") ;;
    *)
        echo "ERROR: Unknown stage '$STAGE'. Must be one of: all, csim, cosim, validation, export, vivado"
        exit 1
        ;;
esac

# Join commands with && so each stage only runs if the previous passed
FULL_CMD=""
for cmd in "${CMDS[@]}"; do
    if [[ -z "$FULL_CMD" ]]; then
        FULL_CMD="$cmd"
    else
        FULL_CMD="$FULL_CMD && $cmd"
    fi
done

LOG_FILE="$LOG_DIR/sim_${STAGE}_$(date +%Y%m%d_%H%M%S).log"
WRAPPED_CMD="{ $FULL_CMD ; } 2>&1 | tee $LOG_FILE; echo \"EXIT \$? — log: $LOG_FILE\""

echo "=== HLS/Vivado Simulation Launcher ==="
echo "Stage    : $STAGE"
echo "HLS dir  : $HLS_DIR"
echo "Log      : $LOG_FILE"
echo "Session  : $SESSION_NAME"
echo ""
echo "Full command:"
echo "  $FULL_CMD"
echo ""

if [[ "$DRY_RUN" == "1" ]]; then
    echo "(dry-run — not launching)"
    exit 0
fi

# Kill existing session if present, then create new one
tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true
tmux new-session -d -s "$SESSION_NAME" -x 220 -y 50

# Send command to tmux window
tmux send-keys -t "$SESSION_NAME" "$WRAPPED_CMD" Enter

echo "Launched tmux session '$SESSION_NAME'."
echo "Monitor with:   tmux attach -t $SESSION_NAME"
echo "Log tail:        tail -f $LOG_FILE"

if [[ "$ATTACH" == "1" ]]; then
    exec tmux attach -t "$SESSION_NAME"
fi
