#!/usr/bin/env bash
set -eo pipefail
ssh -o BatchMode=yes -o ConnectTimeout=10 skki@192.168.50.55 \
  'bash ~/zed_code/scripts/start_remote_zed_live.sh'
ssh -o BatchMode=yes -o ConnectTimeout=10 skki@192.168.50.55 \
  'bash ~/zed_code/scripts/start_zed_tcp_bridge.sh'

# ROS 2 discovery across the Wi-Fi is unreliable here.  This one SSH tunnel
# carries the remote machine's locally subscribed ZED frames instead.
TUNNEL_PID=""
if ! ss -ltn '( sport = :47777 )' | grep -q ':47777'; then
  ssh -f -N -o BatchMode=yes -o ExitOnForwardFailure=yes \
    -L 127.0.0.1:47777:127.0.0.1:47777 skki@192.168.50.55
  TUNNEL_PID="$(pgrep -n -f '127.0.0.1:47777:127.0.0.1:47777')"
fi
cleanup() { [ -n "${TUNNEL_PID:-}" ] && kill "$TUNNEL_PID" 2>/dev/null || true; }
trap cleanup EXIT INT TERM
python3 "$HOME/object_seek/point_cloud/remote_zed_sam2_capture.py" "$@"
