#!/usr/bin/env bash
# Read the AGX TCP pose from this laptop over fixed Fast DDS peer discovery.
set -euo pipefail
export ROS_DOMAIN_ID=36
export ROS_LOCALHOST_ONLY=0
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI="file://$HOME/object_seek/arm_control_client/cyclonedds_agx_laptop.xml"
ros2 daemon stop >/dev/null 2>&1 || true
exec ros2 topic echo /feedback/tcp_pose --no-daemon
