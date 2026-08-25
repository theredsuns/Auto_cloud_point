#!/usr/bin/env bash
set -eo pipefail

source /opt/ros/humble/setup.bash
source "$HOME/robot_arm/install/setup.bash"
set -u

python3 "$HOME/object_seek/arm_control_client/ros_piper_ssh_bridge.py" --allow-motion &
bridge_pid=$!
trap 'kill "$bridge_pid" 2>/dev/null || true' EXIT INT TERM

ros2 launch agx_arm_description display.launch.py \
  arm_type:=piper_l follow:=true control:=true
