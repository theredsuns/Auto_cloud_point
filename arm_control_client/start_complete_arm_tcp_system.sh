#!/usr/bin/env bash
# One-command launcher for the remote ZED + AGX/PiPER TCP stack and local UI.
# It starts no arm motion, enable, trajectory, or pose command.
set -eo pipefail

LOCAL_ROOT="$HOME/object_seek/arm_control_client"
REMOTE="skki@192.168.50.55"

# Local vehicle IMU (Tracer5) remains on ROS domain 36.
export ROS_DOMAIN_ID=36
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTRTPS_DEFAULT_PROFILES_FILE="$LOCAL_ROOT/fastdds_tracer5.xml"
export RMW_FASTRTPS_USE_QOS_FROM_XML=1
source /opt/ros/humble/setup.bash
if [[ -f "$HOME/robot_arm/install/setup.bash" ]]; then
    source "$HOME/robot_arm/install/setup.bash"
fi

echo "[1/4] Starting/reusing remote Tracer5 IMU and filtered odometry…"
ssh -o BatchMode=yes -o ConnectTimeout=10 "$REMOTE" '
  # Avoid a false positive from the remote shell command itself.
  if pgrep -af "[e]kf_filter_node|[I]MU_publisher|[t]racer_base_node" >/dev/null; then
    echo "[tracer5] vehicle IMU/system already running."
  else
    mkdir -p "$HOME/.local/state/tracer5_control"
    nohup bash -lc "cd \"$HOME\" && exec ./start_system.sh" \
      >"$HOME/.local/state/tracer5_control/start_system.log" 2>&1 < /dev/null &
    echo "[tracer5] started ~/start_system.sh in background."
  fi
'

echo "[2/4] Starting/reusing remote CAN, ZED, AGX control and calibrated TCP stack…"
ssh -o BatchMode=yes -o ConnectTimeout=10 "$REMOTE" \
    '~/zed_code/arm_control/start_remote_agx_zed_stack.sh'

echo "[3/4] Starting local RViz arm preview…"
ros2 launch agx_arm_description display.launch.py \
    arm_type:=piper_l namespace:=piper_preview follow:=false control:=false \
    control_topic:=/preview/joint_states \
    rvizconfig:="$LOCAL_ROOT/piper_preview.rviz" &
rviz_pid=$!
trap 'kill "$rviz_pid" 2>/dev/null || true' EXIT INT TERM

echo "[4/4] Opening combined PiPER + Tracer5 control panel…"
# Use the wired SSH bridge so cmd_vel and odometry stay on the vehicle host.
export TRACER5_USE_SSH_BRIDGE=1
export TRACER5_REMOTE_HOST=192.168.50.55
python3 "$LOCAL_ROOT/piper_control_panel.py" --preview
