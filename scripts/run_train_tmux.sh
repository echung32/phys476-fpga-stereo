#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/run_train_tmux.sh [launcher flags] [-- extra train_stereo flags]

Launch the DrivingStereo-first trainer inside a detached tmux session on GPU 1,
log live terminal output to logs/<run-name>.log, and open a monitor window that
records GPU memory/utilization snapshots plus the latest CSV progress row.

Launcher flags:
  --run-name NAME            Override the generated descriptive run name
  --session-name NAME        Override the tmux session name (defaults to run name)
  --gpu ID                   GPU index to use (default: 1)
  --epochs N                 Epoch count (default: 12)
  --batch-size N             Batch size (default: 64)
  --learning-rate VALUE      Learning rate (default: 5e-4)
  --chunk-size N             Examples per training chunk (default: 512)
  --train-epoch-size N       Training examples per epoch (default: 32768)
    --val-limit N              HF validation examples (default: 0 = full split)
    --driving-holdout-limit N  Unseen DrivingStereo holdout examples (default: 512)
  --target-vram-gb N         Target used for batch-size suggestions (default: 40)
  --monitor-interval SEC     GPU/progress polling interval (default: 30)
  --attach                   Attach to the tmux session after launch
  --dry-run                  Print the resolved command/session details without launching
  --help                     Show this message

Anything after -- is passed through directly to v2.training.train_stereo.

Examples:
  scripts/run_train_tmux.sh --attach
  scripts/run_train_tmux.sh --epochs 20 --batch-size 80 --learning-rate 3e-4
  scripts/run_train_tmux.sh -- --feature-channels 24 --max-disp 64
EOF
}

shell_join() {
    local out=""
    local arg=""
    for arg in "$@"; do
        if [[ -n "$out" ]]; then
            out+=" "
        fi
        out+="$(printf '%q' "$arg")"
    done
    printf '%s' "$out"
}

sanitize_token() {
    printf '%s' "$1" | tr -c '[:alnum:]' '_'
}

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/.." && pwd)

run_name=""
session_name=""
gpu="1"
epochs="12"
batch_size="64"
learning_rate="5e-4"
chunk_size="512"
train_epoch_size="32768"
val_limit="16"
driving_holdout_limit="512"
target_vram_gb="40"
monitor_interval="30"
attach_after_launch=0
dry_run=0
extra_args=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-name)
            run_name="$2"
            shift 2
            ;;
        --session-name)
            session_name="$2"
            shift 2
            ;;
        --gpu)
            gpu="$2"
            shift 2
            ;;
        --epochs)
            epochs="$2"
            shift 2
            ;;
        --batch-size)
            batch_size="$2"
            shift 2
            ;;
        --learning-rate)
            learning_rate="$2"
            shift 2
            ;;
        --chunk-size)
            chunk_size="$2"
            shift 2
            ;;
        --train-epoch-size)
            train_epoch_size="$2"
            shift 2
            ;;
        --val-limit)
            val_limit="$2"
            shift 2
            ;;
        --driving-holdout-limit)
            driving_holdout_limit="$2"
            shift 2
            ;;
        --target-vram-gb)
            target_vram_gb="$2"
            shift 2
            ;;
        --monitor-interval)
            monitor_interval="$2"
            shift 2
            ;;
        --attach)
            attach_after_launch=1
            shift
            ;;
        --dry-run)
            dry_run=1
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        --)
            shift
            extra_args=("$@")
            break
            ;;
        *)
            echo "Unknown launcher flag: $1" >&2
            echo "Use -- to pass flags through to train_stereo.py" >&2
            exit 2
            ;;
    esac
done

if ! command -v tmux >/dev/null 2>&1; then
    echo "tmux is required but was not found in PATH." >&2
    exit 1
fi
if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required but was not found in PATH." >&2
    exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi is required but was not found in PATH." >&2
    exit 1
fi

timestamp=$(date +%Y%m%d_%H%M%S)
lr_token=$(sanitize_token "$learning_rate")
if [[ -z "$run_name" ]]; then
    run_name="drivingstereo_gpu${gpu}_bs${batch_size}_lr${lr_token}_e${epochs}_${timestamp}"
fi
if [[ -z "$session_name" ]]; then
    session_name="$run_name"
fi

log_dir="$repo_root/logs"
output_dir="$log_dir/$run_name"
log_file="$log_dir/$run_name.log"
done_file="$output_dir/.train.done"
status_file="$output_dir/.train.status"
train_launcher="$output_dir/tmux_train_launcher.sh"
monitor_launcher="$output_dir/tmux_monitor_launcher.sh"
target_vram_mb=$((target_vram_gb * 1024))

train_cmd=(
    env
    KERAS_BACKEND=torch
    CUDA_VISIBLE_DEVICES="$gpu"
    uv run python -m v2.training.train_stereo
    --run-name "$run_name"
    --output-dir "$output_dir"
    --driving-stereo-dir v2/data/raw/driving_stereo
    --kitti2015-dir v2/data/raw/kitti2015
    --kitti2012-dir v2/data/raw/kitti2012
    --epochs "$epochs"
    --batch-size "$batch_size"
    --learning-rate "$learning_rate"
    --chunk-size "$chunk_size"
    --train-epoch-size "$train_epoch_size"
    --val-limit "$val_limit"
    --driving-holdout-limit "$driving_holdout_limit"
)
if [[ ${#extra_args[@]} -gt 0 ]]; then
    train_cmd+=("${extra_args[@]}")
fi
train_cmd_str=$(shell_join "${train_cmd[@]}")

if [[ "$dry_run" -eq 1 ]]; then
    cat <<EOF
Session:    $session_name
Run name:   $run_name
GPU:        $gpu
Output dir: $output_dir
Log file:   $log_file
Command:
$train_cmd_str
EOF
    exit 0
fi

mkdir -p "$log_dir"
if [[ -e "$log_file" ]]; then
    echo "Refusing to overwrite existing log file: $log_file" >&2
    exit 1
fi
if [[ -e "$output_dir" ]] && [[ -n "$(ls -A "$output_dir" 2>/dev/null)" ]]; then
    echo "Refusing to reuse non-empty output directory: $output_dir" >&2
    exit 1
fi
if tmux has-session -t "$session_name" 2>/dev/null; then
    echo "tmux session '$session_name' already exists." >&2
    exit 1
fi

mkdir -p "$output_dir"
rm -f "$done_file" "$status_file"
: > "$log_file"

cat > "$train_launcher" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd $(printf '%q' "$repo_root")
log_file=$(printf '%q' "$log_file")
done_file=$(printf '%q' "$done_file")
status_file=$(printf '%q' "$status_file")
printf '[launcher] %s run_name=%s gpu=%s batch_size=%s learning_rate=%s epochs=%s train_epoch_size=%s chunk_size=%s target_vram_gb=%s\n' \
  "\$(date -Iseconds)" \
  $(printf '%q' "$run_name") \
  $(printf '%q' "$gpu") \
  $(printf '%q' "$batch_size") \
  $(printf '%q' "$learning_rate") \
  $(printf '%q' "$epochs") \
  $(printf '%q' "$train_epoch_size") \
  $(printf '%q' "$chunk_size") \
  $(printf '%q' "$target_vram_gb") | tee -a "\$log_file"
printf '[launcher] command: %s\n' $(printf '%q' "$train_cmd_str") | tee -a "\$log_file"
set +e
$train_cmd_str 2>&1 | tee -a "\$log_file"
status=\${PIPESTATUS[0]}
set -e
printf '%s\n' "\$status" > "\$status_file"
touch "\$done_file"
printf '[launcher] %s exit_code=%s\n' "\$(date -Iseconds)" "\$status" | tee -a "\$log_file"
printf '[launcher] train pane is idle. Exit this shell when you are done inspecting the session.\n' | tee -a "\$log_file"
exec bash
EOF

cat > "$monitor_launcher" <<EOF
#!/usr/bin/env bash
set -euo pipefail
log_file=$(printf '%q' "$log_file")
done_file=$(printf '%q' "$done_file")
status_file=$(printf '%q' "$status_file")
output_dir=$(printf '%q' "$output_dir")
gpu=$(printf '%q' "$gpu")
batch_size=$(printf '%q' "$batch_size")
target_vram_mb=$(printf '%q' "$target_vram_mb")
monitor_interval=$(printf '%q' "$monitor_interval")
printf '[monitor] %s started gpu=%s interval=%ss target_vram_mb=%s\n' "\$(date -Iseconds)" "\$gpu" "\$monitor_interval" "\$target_vram_mb" | tee -a "\$log_file"
while [[ ! -e "\$done_file" ]]; do
    mem_used_mb=""
    mem_free_mb=""
    util_gpu=""
    util_mem=""
    read -r mem_used_mb mem_free_mb util_gpu util_mem < <(
        nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu,utilization.memory --format=csv,noheader,nounits \
            | awk -F', *' -v gpu="\$gpu" '\$1 == gpu {print \$2, \$3, \$4, \$5}'
    )
    if [[ -z "\$mem_used_mb" ]]; then
        mem_used_mb="na"
        mem_free_mb="na"
        util_gpu="na"
        util_mem="na"
        suggested_batch_size="na"
    else
        suggested_batch_size=\$(awk -v bs="\$batch_size" -v used="\$mem_used_mb" -v target="\$target_vram_mb" 'BEGIN { if (used <= 0) { print bs } else { printf "%d", int((bs * target / used) + 0.5) } }')
    fi
    if [[ -s "\$output_dir/checkpoints/training_log.csv" ]]; then
        last_csv=\$(tail -n 1 "\$output_dir/checkpoints/training_log.csv")
    else
        last_csv="pending"
    fi
    printf '[monitor] %s gpu=%s mem_used_mb=%s mem_free_mb=%s util_gpu=%s util_mem=%s suggested_batch_size=%s last_csv=%s\n' "\$(date -Iseconds)" "\$gpu" "\$mem_used_mb" "\$mem_free_mb" "\$util_gpu" "\$util_mem" "\$suggested_batch_size" "\$last_csv" | tee -a "\$log_file"
    sleep "\$monitor_interval"
done
status=\$(cat "\$status_file" 2>/dev/null || printf 'unknown')
if [[ -f "\$output_dir/metrics.json" ]]; then
    printf '[monitor] %s training complete; metrics.json is available at %s\n' "\$(date -Iseconds)" "\$output_dir/metrics.json" | tee -a "\$log_file"
else
    printf '[monitor] %s training ended without metrics.json; exit_code=%s\n' "\$(date -Iseconds)" "\$status" | tee -a "\$log_file"
fi
printf '[monitor] monitor window is idle. Exit this shell when you are done inspecting the session.\n' | tee -a "\$log_file"
exec bash
EOF

chmod +x "$train_launcher" "$monitor_launcher"

tmux new-session -d -s "$session_name" -n train -c "$repo_root" "$train_launcher"
tmux new-window -d -t "$session_name" -n monitor -c "$repo_root" "$monitor_launcher"

cat <<EOF
Launched tmux session: $session_name
Train window:          $session_name:train
Monitor window:        $session_name:monitor
Run directory:         $output_dir
Log file:              $log_file

Attach with:
  tmux attach -t $session_name
EOF

if [[ "$attach_after_launch" -eq 1 ]]; then
    exec tmux attach -t "$session_name"
fi