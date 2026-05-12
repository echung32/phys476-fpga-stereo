# Two-GPU Training Handoff (Other System)

Purpose: run two reduced-resolution training configs in parallel on the 2-GPU box, then compare holdout metrics and synthesis viability.

## Preconditions
- Repo checked out and dependencies installed on the target system
- Access to both GPUs
- Enough local disk for logs and checkpoints

## Recommended runs (parallel)

Run A (baseline):
```bash
scripts/run_train_tmux.sh \
  --run-name sceneflow_40x60_gray_fc8_md24 \
  --epochs 20 \
  --batch-size 256 \
  --learning-rate 5e-4 \
  --chunk-size 256 \
  --train-epoch-size 8192 \
  --gpu 0 \
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

Run B (lower-memory variant):
```bash
scripts/run_train_tmux.sh \
  --run-name sceneflow_40x60_gray_fc6_md20 \
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
  --max-disp 20 \
  --feature-channels 6 \
  --input-color grayscale \
  --input-fit-mode pad \
  --no-augment-off
```

## Smoke-test first (quick)
Before full runs, launch each config with:
- `--epochs 1`
- `--train-epoch-size 512`
- `--sceneflow-limit 2048`

Goal: verify data pipeline, grayscale path, checkpoint write, and evaluation hooks.

## Evaluation after training
For each completed run:
```bash
python -m v2.evaluation.evaluate_run \
  --run-dir "logs/<run_name_timestamp>" \
  --sample-count 6 \
  --eval-chunk-size 256
```

## What to report back
- Best validation/holdout MAE
- Wall-clock train time per epoch
- Any OOM or loader stalls
- Chosen run directory for hls4ml conversion

## Notes
- If `--input-color grayscale` is not implemented yet, use current loader grayscale behavior and record that this flag is pending.
- Keep both runs on identical software/driver stack for fair comparison.
