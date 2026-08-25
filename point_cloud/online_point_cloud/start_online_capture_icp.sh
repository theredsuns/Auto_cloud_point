#!/usr/bin/env bash
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$HOME/object_seek/point_cloud/start_remote_zed_sam2_capture.sh" \
  --auto-icp --output-root "$SCRIPT_DIR"
