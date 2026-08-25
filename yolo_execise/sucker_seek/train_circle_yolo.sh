#!/usr/bin/env bash
# Usage: ./train_circle_yolo.sh /absolute/path/to/dataset [epochs]
set -euo pipefail
DATASET_PATH="${1:?用法: ./train_circle_yolo.sh /path/to/dataset [epochs]}"
EPOCHS="${2:-150}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export YOLO_CONFIG_DIR="$SCRIPT_DIR/.ultralytics"
export MPLCONFIGDIR="$SCRIPT_DIR/.matplotlib"
mkdir -p "$MPLCONFIGDIR"
python3 "$SCRIPT_DIR/../yolo_body_wing_classification/run_local_yolo.py" detect train \
  data="$DATASET_PATH/data.yaml" model="$SCRIPT_DIR/../body.pt" epochs="$EPOCHS" imgsz=960 batch=4 \
  project="$SCRIPT_DIR/runs" name=circle
