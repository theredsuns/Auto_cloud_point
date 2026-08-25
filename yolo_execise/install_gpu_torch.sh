#!/usr/bin/env bash
# Install a CUDA 12.1 PyTorch build compatible with NVIDIA driver 535/CUDA 12.2.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="$SCRIPT_DIR/libraries/yolo_gpu_runtime"
mkdir -p "$TARGET"
python3 -m pip install --upgrade --no-cache-dir --timeout 600 --retries 10 --target "$TARGET" \
  --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.5.1+cu121 torchvision==0.20.1+cu121
