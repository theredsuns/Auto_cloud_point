#!/usr/bin/env bash
# One local command: remote ZED RGB/depth/SAM2 capture + PiPER slider control.
set -eo pipefail
LOCAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source /opt/ros/humble/setup.bash
source "$HOME/robot_arm/install/agx_arm_description/share/agx_arm_description/local_setup.bash"
set -u

CAPTURE_SCRIPT="$HOME/object_seek/point_cloud/online_point_cloud/start_online_capture_icp.sh"
PANEL_SCRIPT="$LOCAL_ROOT/piper_control_panel.py"

echo "[info] Checking remote PiPER CAN1 (1 Mbps)…"
ssh -o BatchMode=yes -o ConnectTimeout=10 skki@192.168.50.55 '
  can_info=$(ip -details link show can1 2>/dev/null || true)
  if ! printf "%s" "$can_info" | grep -q "bitrate 1000000" || ! printf "%s" "$can_info" | grep -q "can state ERROR-ACTIVE"; then
    sudo -n ip link set can1 down
    sudo -n ip link set can1 type can bitrate 1000000 restart-ms 100
    sudo -n ip link set can1 up
  fi
' || echo "[warning] CAN1 precheck failed; inspect the remote arm connection before moving."

echo "[info] Starting remote ZED online capture (RGB/depth/masks save locally)…"
bash "$CAPTURE_SCRIPT" &
capture_pid=$!

echo "[info] Starting PiPER pose preview…"
ros2 launch agx_arm_description display.launch.py \
  arm_type:=piper_l follow:=false control:=false control_topic:=preview/joint_states &
rviz_pid=$!

cleanup() {
  kill "$capture_pid" "$rviz_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "[info] Opening PiPER control panel…"
python3 "$PANEL_SCRIPT" --preview
