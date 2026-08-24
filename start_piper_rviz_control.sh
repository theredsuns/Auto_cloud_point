#!/usr/bin/env bash
set -eo pipefail
LOCAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source /opt/ros/humble/setup.bash
source "$HOME/robot_arm/install/setup.bash"
set -u

python3 "$LOCAL_ROOT/ros_piper_ssh_bridge.py" --allow-motion &
bridge_pid=$!
trap 'kill "$bridge_pid" 2>/dev/null || true' EXIT INT TERM

ros2 launch agx_arm_description display.launch.py \
  arm_type:=piper_l follow:=true control:=true
