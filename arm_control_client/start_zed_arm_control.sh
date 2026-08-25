#!/usr/bin/env bash
set -eo pipefail

# The Tracer5 vehicle IMU is published in ROS domain 36.  The panel uses
# this to express the camera attitude relative to its startup pose.
export ROS_DOMAIN_ID=36
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTRTPS_DEFAULT_PROFILES_FILE="$HOME/object_seek/arm_control_client/fastdds_tracer5.xml"
export RMW_FASTRTPS_USE_QOS_FROM_XML=1
source /opt/ros/humble/setup.bash
source "$HOME/robot_arm/install/setup.bash"
set -u

echo "[info] Configuring remote PiPER CAN: can1 @ 1 Mbit/s…"
# PiPER-L feedback on this arm is 1 Mbit/s.  A previous 500 kbit/s setting
# caused ERROR-PASSIVE/BUS-OFF and therefore 0 Hz feedback in the panel.
# This only resets SocketCAN; it never enables or moves the arm.
ssh -o BatchMode=yes -o ConnectTimeout=10 skki@192.168.50.55 '
  set -e
  can_info=$(ip -details link show can1 2>/dev/null || true)
  if ! printf "%s" "$can_info" | grep -q "bitrate 1000000"; then
    sudo -n ip link set can1 down
    sudo -n ip link set can1 type can bitrate 1000000 restart-ms 100
    sudo -n ip link set can1 up
  fi
  ip -details link show can1 | sed -n "1,8p"
' || echo "[warning] Remote CAN setup failed; panel will remain safely motion-blocked until feedback is restored."

echo "[info] Starting remote ZED image publisher…"
ssh -o BatchMode=yes -o ConnectTimeout=10 skki@192.168.50.55 \
  'bash ~/zed_code/scripts/start_remote_zed_live.sh' || \
  echo "[warning] Remote ZED start failed; the control panel will still open."

python3 "$HOME/object_seek/point_cloud/remote_zed_viewer.py" &
zed_viewer_pid=$!
ros2 launch agx_arm_description display.launch.py \
  arm_type:=piper_l namespace:=piper_preview follow:=false control:=false \
  control_topic:=/preview/joint_states \
  rvizconfig:="$HOME/object_seek/arm_control_client/piper_preview.rviz" &
rviz_pid=$!
trap 'kill "$zed_viewer_pid" "$rviz_pid" 2>/dev/null || true' EXIT INT TERM

python3 "$HOME/object_seek/arm_control_client/piper_control_panel.py" --preview
