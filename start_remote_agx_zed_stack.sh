#!/usr/bin/env bash
# Start the remote ZED + AGX/PiPER ROS stack without commanding arm motion.
# Some vendor ROS setup files reference optional variables (for example
# COLCON_TRACE) before defining them, so do not enable `set -u` here.
set -eo pipefail

REMOTE_ROOT="$HOME/zed_code"
LOG_ROOT="$REMOTE_ROOT/runtime_logs"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$LOG_ROOT/agx_zed_$STAMP"
mkdir -p "$LOG_DIR"

echo "[1/4] Activating the PiPER USB-CAN at 1 Mbps (no motor command)..."
bash "$HOME/agx_arm_ws/src/agx_arm_ros/scripts/can_activate.sh" 1000000 1-2.4.2:1.0 \
    >"$LOG_DIR/can_activate.log" 2>&1

source "$HOME/agx_arm_ws/install/setup.bash"
# The workspace setup sets a localhost-only Cyclone profile. Override it
# afterwards so the AGX stack and this laptop discover each other on eth0.
export ROS_DOMAIN_ID=36
export ROS_LOCALHOST_ONLY=0
unset FASTRTPS_DEFAULT_PROFILES_FILE
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI="file://$REMOTE_ROOT/arm_control/cyclonedds_agx_eth0.xml"

if pgrep -f '[z]ed_wrapper.*zed_camera.launch.py' >/dev/null; then
    echo "[2/4] ZED wrapper already running; keeping the existing camera process."
else
    echo "[2/4] Starting ZED wrapper in background..."
    nohup ros2 launch zed_wrapper zed_camera.launch.py \
        camera_model:=zed publish_tf:=false publish_map_tf:=false \
        >"$LOG_DIR/zed.log" 2>&1 < /dev/null &
    echo $! > "$LOG_DIR/zed.pid"
fi

sleep 4
if pgrep -f '[a]gx_arm_ctrl_single' >/dev/null; then
    echo "[3/4] AGX arm ROS control already running; keeping the existing process."
else
    echo "[3/4] Starting PiPER CAN control only (MoveIt disabled to prevent joint-state feedback from overriding TCP commands)..."
    # Do not launch the MoveIt stack here.  Its joint_state_broadcaster was
    # publishing zero joint feedback on /control/joint_states, which the AGX
    # node also treats as a command input and therefore overrides /move_p.
    # auto_home is explicitly off: starting this stack never moves the arm.
    nohup ros2 launch agx_arm_ctrl start_single_agx_arm.launch.py \
        can_port:=can1 arm_type:=piper effector_type:=none auto_home:=false \
        speed_percent:=50 \
        'tcp_offset:=[0.009, 0.005, 0.062, -0.059, -1.524, 0.067]' \
        >"$LOG_DIR/agx_arm.log" 2>&1 < /dev/null &
    echo $! > "$LOG_DIR/agx_arm.pid"
fi

sleep 2
# The vendor launch may initialise its runtime parameter to 100 even when a
# launch argument is supplied.  Set it explicitly; this does not command any
# motion, it only limits later /control/move_p goals.
ros2 param set /agx_arm_ctrl_single_node speed_percent 50 >"$LOG_DIR/speed_percent.log" 2>&1 || true
if pgrep -f '[s]tatic_transform_publisher.*tcp_link.*zed_camera_link' >/dev/null; then
    echo "[4/4] Static tcp_link -> ZED transform already running."
else
    echo "[4/4] Publishing static tcp_link -> ZED transform in background..."
    nohup ros2 run tf2_ros static_transform_publisher \
        0.0 0.0 -0.015 0.0 -0.050 0.0 tcp_link zed_camera_link \
        >"$LOG_DIR/static_tf.log" 2>&1 < /dev/null &
    echo $! > "$LOG_DIR/static_tf.pid"
fi

echo "[ready] No arm motion was commanded. Logs: $LOG_DIR"
echo "[check] source ~/agx_arm_ws/install/setup.bash && ros2 topic list"
