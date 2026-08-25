#!/usr/bin/env bash
# Remote-side companion script, installed in ~/zed_code/scripts.
# If the calibrated ZED wrapper already owns the camera, the TCP bridge
# reuses its RGB/depth topics.  Otherwise start the lightweight scanner.
set -eo pipefail

source "$HOME/agx_arm_ws/install/setup.bash"
export ROS_DOMAIN_ID=36
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI="file://$HOME/zed_code/arm_control/cyclonedds_agx_eth0.xml"

if pgrep -f '[z]ed_wrapper.*zed_camera.launch.py' >/dev/null; then
  echo 'reusing existing zed_wrapper stream'
  exit 0
fi

if pgrep -f '/apriltag_zed_visp/zed_rigid_scanner' >/dev/null; then
  pgrep -f '/apriltag_zed_visp/zed_rigid_scanner' | head -1
  exit 0
fi

source "$HOME/zed_code/generated/install/setup.bash"
mkdir -p "$HOME/zed_code/generated/runtime_logs"
nohup ros2 run apriltag_zed_visp zed_rigid_scanner \
  --no-preview --no-3d --image-every 4 --jpeg-quality 30 \
  > "$HOME/zed_code/generated/runtime_logs/remote_live_scanner.log" 2>&1 &
echo $!
