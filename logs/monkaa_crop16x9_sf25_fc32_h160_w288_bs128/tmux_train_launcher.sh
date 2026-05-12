#!/usr/bin/env bash
set -euo pipefail
cd /home/ethan/github/phys476
log_file=/home/ethan/github/phys476/logs/monkaa_crop16x9_sf25_fc32_h160_w288_bs128.log
done_file=/home/ethan/github/phys476/logs/monkaa_crop16x9_sf25_fc32_h160_w288_bs128/.train.done
status_file=/home/ethan/github/phys476/logs/monkaa_crop16x9_sf25_fc32_h160_w288_bs128/.train.status
printf '[launcher] %s run_name=%s gpu=%s batch_size=%s learning_rate=%s epochs=%s train_epoch_size=%s chunk_size=%s\n'   "$(date -Iseconds)"   monkaa_crop16x9_sf25_fc32_h160_w288_bs128   0   128   3e-4   16   32768     1024 | tee -a "$log_file"
printf '[launcher] command: %s\n' env\ KERAS_BACKEND=torch\ CUDA_VISIBLE_DEVICES=0\ uv\ run\ python\ -m\ v2.training.train_stereo\ --run-name\ monkaa_crop16x9_sf25_fc32_h160_w288_bs128\ --output-dir\ /home/ethan/github/phys476/logs/monkaa_crop16x9_sf25_fc32_h160_w288_bs128\ --driving-stereo-dir\ v2/data/raw/driving_stereo\ --kitti2015-dir\ v2/data/raw/kitti2015\ --kitti2012-dir\ v2/data/raw/kitti2012\ --epochs\ 16\ --batch-size\ 128\ --learning-rate\ 3e-4\ --chunk-size\ 1024\ --train-epoch-size\ 32768\ --val-limit\ 0\ --driving-holdout-limit\ 1024\ --sceneflow-dataset\ v2/data/raw/sceneflow/monkaa\ --feature-channels\ 32\ --target-height\ 160\ --target-width\ 288\ --input-fit-mode\ crop\ --max-disp\ 64\ --sceneflow-frac\ 0.25\ --driving-stereo-frac\ 0.55\ --kitti-frac\ 0.20\ --loader-workers\ 12 | tee -a "$log_file"
set +e
env KERAS_BACKEND=torch CUDA_VISIBLE_DEVICES=0 uv run python -m v2.training.train_stereo --run-name monkaa_crop16x9_sf25_fc32_h160_w288_bs128 --output-dir /home/ethan/github/phys476/logs/monkaa_crop16x9_sf25_fc32_h160_w288_bs128 --driving-stereo-dir v2/data/raw/driving_stereo --kitti2015-dir v2/data/raw/kitti2015 --kitti2012-dir v2/data/raw/kitti2012 --epochs 16 --batch-size 128 --learning-rate 3e-4 --chunk-size 1024 --train-epoch-size 32768 --val-limit 0 --driving-holdout-limit 1024 --sceneflow-dataset v2/data/raw/sceneflow/monkaa --feature-channels 32 --target-height 160 --target-width 288 --input-fit-mode crop --max-disp 64 --sceneflow-frac 0.25 --driving-stereo-frac 0.55 --kitti-frac 0.20 --loader-workers 12 2>&1 | tee -a "$log_file"
status=${PIPESTATUS[0]}
set -e
printf '%s\n' "$status" > "$status_file"
touch "$done_file"
printf '[launcher] %s exit_code=%s\n' "$(date -Iseconds)" "$status" | tee -a "$log_file"
printf '[launcher] train pane is idle. Exit this shell when you are done inspecting the session.\n' | tee -a "$log_file"
exec bash
