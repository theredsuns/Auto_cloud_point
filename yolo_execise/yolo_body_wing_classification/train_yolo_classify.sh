#!/usr/bin/env bash
# Usage: ./train_yolo_classify.sh /absolute/path/to/dataset
set -euo pipefail
DATASET_PATH="${1:?Usage: ./train_yolo_classify.sh /path/to/dataset}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export YOLO_CONFIG_DIR="$SCRIPT_DIR/.ultralytics"
python3 "$SCRIPT_DIR/run_local_yolo.py" classify train \
  data="$DATASET_PATH" model=yolo11s-cls.pt epochs=100 imgsz=640 batch=16 \
  project="$SCRIPT_DIR/runs" name=body_wing_classifier
