#!/usr/bin/env bash
set -eo pipefail

source /opt/ros/humble/setup.bash
# The workspace-wide setup references an old deleted demo package. Source
# only the installed PiPER description package required by this preview.
source "$HOME/robot_arm/install/agx_arm_description/share/agx_arm_description/local_setup.bash"
set -u

ros2 launch agx_arm_description display.launch.py \
  arm_type:=piper_l follow:=false control:=false control_topic:=preview/joint_states &
rviz_pid=$!
trap 'kill "$rviz_pid" 2>/dev/null || true' EXIT INT TERM

python3 "$HOME/object_seek/arm_control_client/piper_control_panel.py" --preview
