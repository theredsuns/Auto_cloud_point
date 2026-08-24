#!/usr/bin/env bash
set -eo pipefail
LOCAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source /opt/ros/humble/setup.bash
source "$HOME/robot_arm/install/setup.bash"
set -u

python3 "$LOCAL_ROOT/ros_piper_ssh_bridge.py" --allow-motion --moveit &
bridge_pid=$!
ros2 launch agx_arm_moveit static_virtual_joint_tfs.launch.py arm_type:=piper_l &
tf_pid=$!
ros2 launch agx_arm_moveit rsp.launch.py arm_type:=piper_l follow:=true &
rsp_pid=$!
ros2 launch agx_arm_moveit move_group.launch.py arm_type:=piper_l follow:=true &
move_group_pid=$!
trap 'kill "$bridge_pid" "$tf_pid" "$rsp_pid" "$move_group_pid" 2>/dev/null || true' EXIT INT TERM
ros2 launch agx_arm_moveit moveit_rviz.launch.py arm_type:=piper_l
