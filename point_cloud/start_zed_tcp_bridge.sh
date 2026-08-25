#!/usr/bin/env bash
# Remote-side companion script, installed in ~/zed_code/scripts.
# The calibrated ZED/arm stack runs in ROS domain 36; the TCP bridge must
# use the same DDS configuration or it cannot see the already-open wrapper.
set -eo pipefail

# A dead/zombie Python process can remain in `pgrep` after an SSH client
# closes. Test the actual listening socket instead.
if ss -ltn '( sport = :47777 )' 2>/dev/null | grep -q ':47777'; then
  pgrep -f '^python3 .*/zed_tcp_capture_bridge.py$' | head -1 || true
  exit 0
fi

source "$HOME/agx_arm_ws/install/setup.bash"
export ROS_DOMAIN_ID=36
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI="file://$HOME/zed_code/arm_control/cyclonedds_agx_eth0.xml"
source "$HOME/zed_code/generated/install/setup.bash"
mkdir -p "$HOME/zed_code/generated/runtime_logs"
nohup python3 "$HOME/zed_code/scripts/zed_tcp_capture_bridge.py" \
  > "$HOME/zed_code/generated/runtime_logs/zed_tcp_capture_bridge.log" 2>&1 &
echo $!
