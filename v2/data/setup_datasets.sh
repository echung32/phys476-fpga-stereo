#!/usr/bin/env bash
# setup_datasets.sh — extract KITTI 2015 and KITTI 2012 after download.
#
# Run from the repo root:
#   bash v2/data/setup_datasets.sh
#
# Prerequisites (already downloaded by training pipeline):
#   v2/data/raw/kitti2015/data_scene_flow.zip   (~1.6 GB)
#   v2/data/raw/kitti2012/data_stereo_flow.zip  (~1.9 GB)
#
# DrivingStereo requires manual registration + download from:
#   https://drivingstereo-dataset.github.io/
# Extract into v2/data/raw/driving_stereo/ with layout:
#   train-left-image/<seq>/<frame>.jpg
#   train-right-image/<seq>/<frame>.jpg
#   train-disparity/<seq>/<frame>.png

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RAW="$REPO_ROOT/v2/data/raw"

# ---- KITTI 2015 ----
KITTI15_ZIP="$RAW/kitti2015/data_scene_flow.zip"
KITTI15_DIR="$RAW/kitti2015"
if [[ -f "$KITTI15_ZIP" ]]; then
    echo "Extracting KITTI 2015 …"
    unzip -q -n "$KITTI15_ZIP" -d "$KITTI15_DIR"
    echo "  Done → $KITTI15_DIR/training/"
else
    echo "WARNING: $KITTI15_ZIP not found — still downloading?"
fi

# ---- KITTI 2012 ----
KITTI12_ZIP="$RAW/kitti2012/data_stereo_flow.zip"
KITTI12_DIR="$RAW/kitti2012"
if [[ -f "$KITTI12_ZIP" ]]; then
    echo "Extracting KITTI 2012 …"
    unzip -q -n "$KITTI12_ZIP" -d "$KITTI12_DIR"
    echo "  Done → $KITTI12_DIR/training/"
else
    echo "WARNING: $KITTI12_ZIP not found — still downloading?"
fi

echo ""
echo "Dataset status:"
echo "  SceneFlow   : $RAW/sceneflow/ ($(du -sh "$RAW/sceneflow" 2>/dev/null | cut -f1 || echo 'not found'))"
echo "  KITTI 2015  : $KITTI15_DIR/training/ ($(ls "$KITTI15_DIR/training/image_2/" 2>/dev/null | wc -l || echo 0) left frames)"
echo "  KITTI 2012  : $KITTI12_DIR/training/ ($(ls "$KITTI12_DIR/training/colored_0/" 2>/dev/null | wc -l || echo 0) left frames)"
echo "  DrivingStereo: $RAW/driving_stereo/ ($(ls "$RAW/driving_stereo/train-left-image/" 2>/dev/null | wc -l || echo 'not found') seqs)"
echo ""
echo "Once all datasets are ready, launch full training with:"
echo "  KERAS_BACKEND=torch CUDA_VISIBLE_DEVICES=0 uv run python -m v2.training.train_stereo \\"
echo "      --driving-stereo-dir v2/data/raw/driving_stereo \\"
echo "      --kitti2015-dir v2/data/raw/kitti2015 \\"
echo "      --kitti2012-dir v2/data/raw/kitti2012 \\"
echo "      --epochs 20 --batch-size 4 --chunk-size 256 \\"
echo "      --run-name full_mixed"
