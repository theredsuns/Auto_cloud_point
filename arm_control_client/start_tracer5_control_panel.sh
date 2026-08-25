#!/usr/bin/env bash
# Start the local UI with Tracer5 DDS traffic over the wired Ethernet link.
set -eo pipefail

export ROS_DOMAIN_ID=36
export ROS_LOCALHOST_ONLY=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTRTPS_DEFAULT_PROFILES_FILE="$HOME/object_seek/arm_control_client/fastdds_tracer5.xml"
export RMW_FASTRTPS_USE_QOS_FROM_XML=1
# Use the wired SSH bridge so cmd_vel and odometry stay on the vehicle host.
export TRACER5_USE_SSH_BRIDGE=1
export TRACER5_REMOTE_HOST=192.168.50.55
source /opt/ros/humble/setup.bash

echo "[info] Starting remote vehicle IMU/system service (~/start_system.sh)…"
# The remote start_system.sh owns the Tracer5 IMU and publishes the filtered
# odometry used by the panel.  It is deliberately detached: that command is a
# long-running bring-up process.  No cmd_vel is sent here.
ssh -o BatchMode=yes -o ConnectTimeout=10 skki@192.168.50.55 '
  # Bracketed initial letters keep pgrep from matching this shell command
  # itself (whose arguments contain the process-name pattern).
  if pgrep -af "[e]kf_filter_node|[I]MU_publisher|[t]racer_base_node" >/dev/null; then
    echo "[tracer5] IMU/system process already running."
  else
    mkdir -p "$HOME/.local/state/tracer5_control"
    nohup bash -lc "cd \"$HOME\" && exec ./start_system.sh" \
      >"$HOME/.local/state/tracer5_control/start_system.log" 2>&1 < /dev/null &
    echo "[tracer5] started ~/start_system.sh in background."
  fi
' || echo "[warning] Remote IMU start failed or SSH is unavailable; UI will wait for /tracer5/odometry/filtered."

exec python3 "$HOME/object_seek/arm_control_client/tracer5_control_panel.py"
