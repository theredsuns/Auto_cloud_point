#!/usr/bin/env bash
# 本机 ZED 手动取帧：按 Enter 后才执行 SAM2、质量检查和 ICP。

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

export PYTHONUNBUFFERED=1
export YOLO_CONFIG_DIR="${SCRIPT_DIR}/.ultralytics"
export MPLCONFIGDIR="${SCRIPT_DIR}/.matplotlib"
mkdir -p "${YOLO_CONFIG_DIR}" "${MPLCONFIGDIR}" "${SCRIPT_DIR}/models"

python3 -m py_compile \
    zed_yolo_sam2_icp_reconstruct.py \
    zed_yolo_sam2_live.py

exec python3 -u zed_yolo_sam2_icp_reconstruct.py \
    --sam2-config sam2/sam2/configs/sam2.1/sam2.1_hiera_t.yaml \
    --sam2-checkpoint sam2/checkpoints/sam2.1_hiera_tiny.pt \
    --target-class "Wing and Body" \
    --confidence 0.25 \
    --max-depth-m 2.3 \
    --voxel-m 0.012 \
    --min-fitness 0.35 \
    --max-rmse-m 0.035 \
    --output models/local_complete_object \
    --manual-capture \
    --mesh-on-save \
    "$@"
