#!/usr/bin/env bash
set -euo pipefail
log_file=/home/ethan/github/phys476/logs/monkaa_crop16x9_sf25_fc32_h160_w288_bs128.log
done_file=/home/ethan/github/phys476/logs/monkaa_crop16x9_sf25_fc32_h160_w288_bs128/.train.done
status_file=/home/ethan/github/phys476/logs/monkaa_crop16x9_sf25_fc32_h160_w288_bs128/.train.status
output_dir=/home/ethan/github/phys476/logs/monkaa_crop16x9_sf25_fc32_h160_w288_bs128
gpu=0
batch_size=128
monitor_interval=30
printf '[monitor] %s started gpu=%s interval=%ss\n' "$(date -Iseconds)" "$gpu" "$monitor_interval" | tee -a "$log_file"
while [[ ! -e "$done_file" ]]; do
    mem_used_mb=""
    mem_free_mb=""
    util_gpu=""
    util_mem=""
    read -r mem_used_mb mem_free_mb util_gpu util_mem < <(
        nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu,utilization.memory --format=csv,noheader,nounits             | awk -F', *' -v gpu="$gpu" '$1 == gpu {print $2, $3, $4, $5}'
    )
    if [[ -z "$mem_used_mb" ]]; then
        mem_used_mb="na"
        mem_free_mb="na"
        util_gpu="na"
        util_mem="na"
    fi
    if [[ -s "$output_dir/checkpoints/training_log.csv" ]]; then
        last_csv=$(tail -n 1 "$output_dir/checkpoints/training_log.csv")
    else
        last_csv="pending"
    fi
    printf '[monitor] %s gpu=%s batch_size=%s mem_used_mb=%s mem_free_mb=%s util_gpu=%s util_mem=%s last_csv=%s\n' "$(date -Iseconds)" "$gpu" "$batch_size" "$mem_used_mb" "$mem_free_mb" "$util_gpu" "$util_mem" "$last_csv" | tee -a "$log_file"
    sleep "$monitor_interval"
done
status=$(cat "$status_file" 2>/dev/null || printf 'unknown')
if [[ -f "$output_dir/metrics.json" ]]; then
    printf '[monitor] %s training complete; metrics.json is available at %s\n' "$(date -Iseconds)" "$output_dir/metrics.json" | tee -a "$log_file"
else
    printf '[monitor] %s training ended without metrics.json; exit_code=%s\n' "$(date -Iseconds)" "$status" | tee -a "$log_file"
fi
printf '[monitor] monitor window is idle. Exit this shell when you are done inspecting the session.\n' | tee -a "$log_file"
exec bash
