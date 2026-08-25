#!/usr/bin/env bash
set -eo pipefail

if ss -ltn '( sport = :47777 )' 2>/dev/null | grep -q ':47777'; then
  exit 0
fi

nohup python3 "$HOME/zed_code/scripts/zed_direct_tcp_capture_bridge.py" \
  > "$HOME/zed_code/generated/runtime_logs/zed_direct_tcp_bridge.log" 2>&1 < /dev/null &
