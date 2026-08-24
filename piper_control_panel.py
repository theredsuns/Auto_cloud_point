#!/usr/bin/env python3
"""Local GUI for guarded PiPER-L control over an SSH connection."""
import math
import json
import re
import shlex
import socket
import subprocess
import threading
import time
import queue
import tkinter as tk
import argparse
import os
import sys
import threading
from datetime import datetime
from pathlib import Path
from threading import Lock
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

REMOTE_HOST = "skki@192.168.50.55"
REMOTE_DIR = "~/zed_code/arm_control"
JOINT_NAMES = ("J1", "J2", "J3", "J4", "J5", "J6")
STEP_DEG = 1.0
JOINT_LIMITS = ((-150.0, 150.0), (0.0, 180.0), (-170.0, 0.0),
                (-100.0, 100.0), (-70.0, 70.0), (-120.0, 120.0))
# PiPER's mechanical zero pose.  It is sent through the same guarded,
# low-speed path as every other movement command.
HOME_JOINTS_DEG = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
# Vehicle EKF output.  Its pose orientation is the filtered IMU attitude in
# the same frame used by the vehicle odometry.
IMU_TOPIC = "/tracer5/odometry/filtered"
TRACER5_FASTDDS_PROFILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "fastdds_tracer5.xml"
)
# Arm-base origin expressed in the vehicle IMU frame, in metres.  Positive Z
# means that the arm base is 75 cm above the IMU in the vehicle's +Z direction.
ARM_BASE_FROM_IMU_M = (0.0, 0.0, 0.75)
# The AGX controller's calibrated TCP is offset from the raw PiPER flange.
# Cartesian fields shown in this UI are TCP coordinates, whereas the vendor
# SDK EndPoseCtrl command expects the raw flange/end-pose frame.
TCP_FROM_FLANGE_TRANSLATION_MM = (9.0, 5.0, 62.0)
TCP_FROM_FLANGE_RPY_RAD = (-0.059, -1.524, 0.067)
CARTESIAN_STEPS = (50.0, 50.0, 50.0, 1.0, 1.0, 1.0)
# Absolute-coordinate ranges used by the continuous-control sliders.  They do
# not bypass the guarded remote IK checks; they only limit what the GUI can
# enter in one gesture.
CARTESIAN_Z_MIN_MM = 0.0
CARTESIAN_Z_MAX_MM = 610.0
CARTESIAN_SLIDER_LIMITS = ((-1000.0, 1000.0), (-1000.0, 1000.0), (CARTESIAN_Z_MIN_MM, CARTESIAN_Z_MAX_MM),
                           (-180.0, 180.0), (-90.0, 180.0), (-180.0, 180.0))
CLIENT_DIR = os.path.dirname(os.path.abspath(__file__))
REMOTE_CAPTURE_SCRIPT = os.path.join(
    os.path.dirname(CLIENT_DIR), "point_cloud", "online_point_cloud", "start_online_capture_icp.sh"
)
MOTION_RECORDS_PATH = os.path.join(CLIENT_DIR, "motion_records.json")
VEHICLE_MOTION_RECORDS_PATH = os.path.join(CLIENT_DIR, "vehicle_motion_records.json")
AUTO_CAPTURE_PLANS_PATH = os.path.join(CLIENT_DIR, "auto_capture_plans.json")
CONTROL_SETTINGS_PATH = os.path.join(CLIENT_DIR, "control_settings.json")
ONLINE_SCAN_ROOT = Path(CLIENT_DIR).parent / "point_cloud" / "online_point_cloud"


def _quat_multiply(left, right):
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return (
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    )


def _quat_inverse(quaternion):
    x, y, z, w = quaternion
    norm = x * x + y * y + z * z + w * w
    if norm < 1e-12:
        return (0.0, 0.0, 0.0, 1.0)
    return (-x / norm, -y / norm, -z / norm, w / norm)


def _quat_to_rpy_deg(quaternion):
    x, y, z, w = quaternion
    sinr = 2.0 * (w * x + y * z)
    cosr = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr, cosr)
    sinp = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch = math.asin(sinp)
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny, cosy)
    return tuple(math.degrees(value) for value in (roll, pitch, yaw))


def _rpy_deg_to_quat(roll_deg, pitch_deg, yaw_deg):
    roll, pitch, yaw = (math.radians(value) for value in (roll_deg, pitch_deg, yaw_deg))
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def _rotate_vector(quaternion, vector):
    """Rotate a 3D vector by an xyzw quaternion."""
    rotated = _quat_multiply(
        _quat_multiply(quaternion, (vector[0], vector[1], vector[2], 0.0)),
        _quat_inverse(quaternion),
    )
    return rotated[:3]


def _tcp_pose_to_raw_flange_pose(tcp_pose):
    """Convert UI calibrated-TCP mm/RPY target to PiPER raw flange EndPose.

    AGX's tcp_offset is a rigid transform from flange to TCP.  The direct
    CAN SDK has no tcp_offset parameter, so it must receive the inverse
    transform.  Without this conversion a current flange pitch of about 85°
    was incorrectly commanded as a TCP pitch near -2°, causing 0x4 errors.
    """
    tx, ty, tz, roll, pitch, yaw = (float(value) for value in tcp_pose)
    tcp_orientation = _rpy_deg_to_quat(roll, pitch, yaw)
    offset_orientation = _rpy_deg_to_quat(
        *(math.degrees(value) for value in TCP_FROM_FLANGE_RPY_RAD)
    )
    flange_orientation = _quat_multiply(tcp_orientation, _quat_inverse(offset_orientation))
    offset_in_base = _rotate_vector(flange_orientation, TCP_FROM_FLANGE_TRANSLATION_MM)
    flange_rpy = _quat_to_rpy_deg(flange_orientation)
    return (
        tx - offset_in_base[0], ty - offset_in_base[1], tz - offset_in_base[2],
        *flange_rpy,
    )


class ImuTracker:
    """Receive the filtered vehicle pose and express its attitude from startup."""
    def __init__(self, topic=IMU_TOPIC):
        self.lock = Lock()
        self.current = None
        self.current_position = None
        self.initial = None
        self.error = None
        self.node = None
        self.executor = None
        try:
            # The vehicle uses ROS domain 36.  Make direct panel startup work
            # too: users should not have to remember a separate `source` or
            # export command just to receive the filtered odometry.
            os.environ.setdefault("ROS_DOMAIN_ID", "36")
            os.environ.setdefault("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")
            if os.path.isfile(TRACER5_FASTDDS_PROFILE):
                os.environ.setdefault("FASTRTPS_DEFAULT_PROFILES_FILE", TRACER5_FASTDDS_PROFILE)
                os.environ.setdefault("RMW_FASTRTPS_USE_QOS_FROM_XML", "1")
            import rclpy
            from rclpy.executors import SingleThreadedExecutor
            from rclpy.node import Node
            from nav_msgs.msg import Odometry
            self.rclpy = rclpy
            if not rclpy.ok():
                rclpy.init()
            self.node = Node("piper_camera_imu_display")
            self.node.create_subscription(Odometry, topic, self._callback, 20)
            self.executor = SingleThreadedExecutor()
            self.executor.add_node(self.node)
            self.thread = threading.Thread(target=self.executor.spin, daemon=True)
            self.thread.start()
        except Exception as exc:
            self.error = str(exc)

    def _callback(self, message):
        quaternion = message.pose.pose.orientation
        value = (quaternion.x, quaternion.y, quaternion.z, quaternion.w)
        if sum(component * component for component in value) < 1e-10:
            return
        with self.lock:
            self.current = value
            position = message.pose.pose.position
            self.current_position = (position.x, position.y, position.z)
            if self.initial is None:
                self.initial = value

    def relative_quaternion(self):
        with self.lock:
            if self.current is None or self.initial is None:
                return None
            return _quat_multiply(_quat_inverse(self.initial), self.current)

    def current_pose(self):
        with self.lock:
            if self.current is None or self.current_position is None:
                return None
            return self.current, self.current_position

    def reset_origin(self):
        with self.lock:
            if self.current is not None:
                self.initial = self.current
                return True
        return False

    def close(self):
        if self.executor:
            self.executor.shutdown()
        if self.node:
            self.node.destroy_node()


class PreviewPublisher:
    """ROS publisher used only to drive the local RViz preview model."""
    def __init__(self):
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import JointState
        self.rclpy = rclpy
        self.JointState = JointState
        if not rclpy.ok():
            rclpy.init()
        self.node = Node("piper_target_preview")
        self.publisher = self.node.create_publisher(JointState, "preview/joint_states", 10)
        self.degrees = [0.0] * 6
        # Keep RViz's complete kinematic chain visible even while the real
        # CAN arm is offline. This is a local preview topic only; it never
        # traverses SSH and cannot command the physical arm.
        self.publish([0.0] * 6)

    def publish(self, degrees):
        self.degrees = list(degrees)
        msg = self.JointState()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.name = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        msg.position = [math.radians(value) for value in degrees]
        self.publisher.publish(msg)

    def republish(self):
        """Keep late-starting robot_state_publisher/RViz supplied with TF."""
        self.publish(self.degrees)

    def close(self):
        self.node.destroy_node()


class PiperPanel(tk.Tk):
    def __init__(self, preview=False):
        super().__init__()
        self.title("PiPER-L 远程控制（受限低速）")
        self.tk.call("tk", "scaling", 1.0)
        self.resizable(True, True)
        self.geometry("1240x900")
        self.minsize(1040, 760)
        self.current = [None] * 6
        # Wi-Fi client isolation prevents direct DDS to the laptop.  Reuse
        # the SSH vehicle bridge inside this same Tk window.
        os.environ.setdefault("TRACER5_USE_SSH_BRIDGE", "0")
        os.environ.setdefault("TRACER5_REMOTE_HOST", "192.168.50.55")
        from tracer5_control_panel import TracerRos
        self.tracer = TracerRos(lambda: None)
        self.vehicle_status = tk.StringVar(value="Tracer5：正在连接…")
        self.vehicle_pose = tk.StringVar(value="等待 IMU 融合定位…")
        self.vehicle_v = tk.DoubleVar(value=0.10)
        self.vehicle_w = tk.DoubleVar(value=0.55)
        saved_settings = self._load_control_settings()
        self.vehicle_position_tolerance = tk.DoubleVar(value=saved_settings.get("vehicle_position_tolerance_cm", 0.5))
        self.vehicle_yaw_tolerance = tk.DoubleVar(value=saved_settings.get("vehicle_yaw_tolerance_deg", 0.5))
        self.no_target_wait_s = tk.DoubleVar(value=saved_settings.get("no_target_wait_s", 0.2))
        self.min_target_confidence = tk.DoubleVar(value=saved_settings.get("min_target_confidence_pct", 50.0))
        self.capture_settle_s = tk.DoubleVar(value=saved_settings.get("capture_settle_s", 2.0))
        self.tracer.set_tolerances(self.vehicle_position_tolerance.get(), self.vehicle_yaw_tolerance.get())
        self.vehicle_rx, self.vehicle_ry, self.vehicle_ryaw = tk.DoubleVar(), tk.DoubleVar(), tk.DoubleVar()
        self.vehicle_ax, self.vehicle_ay, self.vehicle_ayaw = tk.DoubleVar(), tk.DoubleVar(), tk.DoubleVar()
        self.vehicle_keep_yaw = tk.BooleanVar(value=True)
        self.vehicle_recording = False
        # Keep one ordered queue.  A route can mix relative and absolute
        # waypoints, and the list order is the actual execution order.
        self.vehicle_motion_records = self._load_vehicle_motion_records()
        self.vehicle_record_kind = tk.StringVar(value="相对")
        self.vehicle_record_x = tk.DoubleVar(value=0.0)
        self.vehicle_record_y = tk.DoubleVar(value=0.0)
        self.vehicle_record_yaw = tk.DoubleVar(value=0.0)
        self.vehicle_route_running = False
        self.vehicle_route_cancelled = False
        self.vehicle_auto_capture_running = False
        self.auto_capture_plan = []
        self.saved_auto_capture_plans = self._load_saved_auto_capture_plans()
        self.auto_capture_plan_name = tk.StringVar()
        self.auto_capture_plan_running = False
        self.auto_capture_plan_cancelled = False
        self.can_ready = False
        self.targets = [tk.StringVar(value="--") for _ in range(6)]
        self.current_labels = []
        self.sliders = []
        self.relative_inputs = [tk.StringVar(value="0") for _ in range(6)]
        self.cartesian_inputs = [tk.StringVar(value="0") for _ in range(6)]
        self.continuous_sliders = []
        self.preview = PreviewPublisher() if preview else None
        self.imu = ImuTracker()
        self.safety_ok = tk.BooleanVar(value=False)
        self.speed_percent = tk.StringVar(value="50")
        self.status = tk.StringVar(value="尚未读取远程机械臂状态")
        self.flange_pose = tk.StringVar(value="末端法兰：尚未读取")
        self.camera_pose = tk.StringVar(value="末端相对小车初始坐标：等待滤波里程计与末端位姿…")
        self.origin_flange = None
        self.origin_vehicle_position = None
        self.origin_vehicle_orientation = None
        self.current_flange = None
        self.last_end_relative_pose = None
        self.cartesian_busy = False
        self.routine_running = False
        self.routine_auto_capture = False
        self.refreshing = False
        self.capture_process = None
        self.capture_module = None
        self.capture_detector = None
        self.capture_predictor = None
        # None means this panel session has not captured anything yet.  It is
        # intentionally never restored from disk on startup: closing and
        # reopening the program always starts a fresh scan unless the user
        # explicitly chooses a historical scan below.
        self.capture_scan = None
        self.capture_busy = False
        self.cloud_mtime = None
        self.last_cloud_points = None
        self.last_cloud_colors = None
        self.remote_camera = None
        self.camera_tunnel = None
        self.auto_cloud_process = None
        self.auto_camera_enabled = False
        self.auto_camera_display_pending = False
        self.auto_camera_display_frame = None
        self.auto_capture_popup = None
        self.auto_capture_popup_photo = None
        self.fusion_queue: queue.Queue[tuple[Path, str, int] | None] = queue.Queue()
        self.fusion_worker = None
        self.fusion_pending = 0
        self.fusion_lock = threading.Lock()
        self.auto_capture_failed = False
        self.capture_detection_info = tk.StringVar(
            value="抓拍设置：识别度≥50%，无目标等待 0.2 s，抓拍停顿 2.0 s"
        )
        self.auto_capture_phase_info = tk.StringVar(value="自动抓拍：待命")
        self.capture_countdown_info = tk.StringVar(value="抓拍计时：未开始")
        # In automatic workflows, an empty view is evidence too: retain it
        # without creating a mask/point cloud, then continue the route.
        self.skip_no_target_capture = tk.BooleanVar(value=True)
        # This plain value is deliberately read by capture workers. Tk
        # variables must only ever be accessed by Tk's main thread.
        self._no_target_skip_active = True
        self._no_target_wait_s_active = 0.2
        self._min_target_confidence_active = 0.50
        self._capture_settle_s_active = 2.0
        # The capture worker must not send an arm command until Tk has
        # actually created the RGB/depth window.  Without this handshake a
        # fast CAN command can begin before the queued OpenCV draw runs,
        # making the linked workflow look as though ZED never opened.
        self.auto_camera_window_drawn = threading.Event()
        self.motion_records = self._load_motion_records()
        self.routine_tree = None
        self._build()
        self.after(200, self._keep_preview_alive)
        self.after(500, self._auto_refresh)
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.refresh_status()

    def _keep_preview_alive(self):
        if self.preview and self.winfo_exists():
            self.preview.republish()
            self.after(200, self._keep_preview_alive)

    def _auto_refresh(self):
        if self.winfo_exists():
            self.refresh_status(automatic=True)
            self._refresh_vehicle_panel()
            self.after(500, self._auto_refresh)

    def _build_camera_panel(self, parent):
        panel = ttk.LabelFrame(parent, text="远程 ZED 抓拍与点云", padding=10)
        panel.grid(row=0, column=1, sticky="nsew", padx=(14, 0))
        ttk.Label(panel, textvariable=self.camera_status, foreground="#204a87", wraplength=540).grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 7)
        )
        self.camera_label = ttk.Label(
            panel, text="未开启 ZED\n点击“开启抓拍”后显示 RGB + Depth", anchor="center", width=66,
            relief="sunken", padding=4,
        )
        self.camera_label.grid(row=1, column=0, columnspan=2, sticky="nsew")
        ttk.Button(panel, text="开启抓拍（RGB + Depth）", command=self.enable_remote_capture).grid(
            row=2, column=0, sticky="ew", padx=(0, 4), pady=(8, 0)
        )
        ttk.Button(panel, text="抓拍并自动分割（YOLO + SAM2）", command=self.capture_integrated).grid(
            row=2, column=1, sticky="ew", padx=(4, 0), pady=(8, 0)
        )
        ttk.Label(
            panel,
            text="开启抓拍后显示远程 RGB + Depth。按“抓拍并自动分割”或主窗口 Enter 即保存、生成掩码并刷新融合点云。",
            foreground="#555555", wraplength=540,
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=(8, 0))
        ttk.Separator(panel).grid(row=4, column=0, columnspan=2, sticky="ew", pady=8)
        ttk.Label(panel, text="实时融合点云预览", font=("Sans", 10, "bold")).grid(row=5, column=0, sticky="w")
        ttk.Button(panel, text="弹出 / 放大 3D 点云", command=self.open_cloud_popup).grid(row=5, column=1, sticky="e")
        figure = Figure(figsize=(5.6, 2.45), dpi=100, facecolor="#181818")
        self.cloud_axes = figure.add_subplot(111, projection="3d")
        self.cloud_axes.set_facecolor("#181818")
        self.cloud_axes.tick_params(colors="#bbbbbb", labelsize=6)
        self.cloud_axes.set_title("尚未生成点云", color="white", fontsize=9)
        self.cloud_canvas = FigureCanvasTkAgg(figure, master=panel)
        self.cloud_canvas.get_tk_widget().grid(row=6, column=0, columnspan=2, sticky="nsew", pady=(4, 0))

    def _set_no_target_skip(self):
        """Copy a UI option into worker-safe state on the Tk thread."""
        self._no_target_skip_active = bool(self.skip_no_target_capture.get())

    def _set_capture_detection_options(self):
        """Validate Tk inputs then snapshot them for the capture worker."""
        wait_s = min(10.0, max(0.2, float(self.no_target_wait_s.get())))
        confidence_pct = min(100.0, max(1.0, float(self.min_target_confidence.get())))
        settle_s = min(10.0, max(0.0, float(self.capture_settle_s.get())))
        self.no_target_wait_s.set(wait_s)
        self.min_target_confidence.set(confidence_pct)
        self.capture_settle_s.set(settle_s)
        self._set_no_target_skip()
        self._no_target_wait_s_active = wait_s
        self._min_target_confidence_active = confidence_pct / 100.0
        self._capture_settle_s_active = settle_s
        self.capture_detection_info.set(
            f"设置：识别度≥{confidence_pct:.0f}%\n无目标等待 {wait_s:.1f} s；抓拍停顿 {settle_s:.1f} s"
        )

    def _save_capture_detection_settings(self):
        try:
            self._set_capture_detection_options()
            settings = self._load_control_settings()
            settings.update({"no_target_wait_s": self._no_target_wait_s_active,
                             "min_target_confidence_pct": self.min_target_confidence.get(),
                             "capture_settle_s": self._capture_settle_s_active})
            with open(CONTROL_SETTINGS_PATH, "w", encoding="utf-8") as handle:
                json.dump(settings, handle, ensure_ascii=False, indent=2)
            self.status.set(
                f"全局抓拍设置已保存：停顿 {self._capture_settle_s_active:.1f} 秒，"
                f"等待 {self._no_target_wait_s_active:.1f} 秒，最低识别度 {self.min_target_confidence.get():.0f}%。"
            )
        except (ValueError, TypeError, OSError) as error:
            messagebox.showerror("保存抓拍设置失败", str(error), parent=self)

    def _set_auto_capture_phase(self, text):
        """Update the sidebar from either a worker or the Tk thread."""
        self.after(0, lambda value=str(text): self.auto_capture_phase_info.set(f"自动抓拍：{value}"))

    def _set_capture_countdown(self, label, remaining, total):
        self.after(0, lambda name=str(label), left=max(0.0, remaining), duration=total:
                   self.capture_countdown_info.set(
                       f"{name}：剩余 {left:.1f} / {duration:.1f} s"
                   ))

    def restart_camera_bridge(self):
        self.remote_camera.close()
        self.camera_status.set("远程相机：正在重新连接…")
        self.camera_connecting = False

    def enable_remote_capture(self):
        self.camera_enabled = True
        self.restart_camera_bridge()

    def _load_capture_models(self):
        """Load exactly the existing remote-capture YOLO/SAM2 implementation."""
        if self.capture_module is not None:
            return True
        try:
            point_cloud_root = str(ONLINE_SCAN_ROOT.parent)
            if point_cloud_root not in sys.path:
                sys.path.insert(0, point_cloud_root)
            import remote_zed_sam2_capture as capture_module
            self.capture_module = capture_module
            self.capture_detector = capture_module.YOLO(str(capture_module.SEGMENT_WEIGHTS))
            self.capture_predictor = capture_module.load_sam2(
                capture_module.PROJECT / "yolo_execise" / "libraries" / "sam2" / "sam2" / "configs" / "sam2.1" / "sam2.1_hiera_t.yaml",
                capture_module.PROJECT / "yolo_execise" / "libraries" / "sam2" / "checkpoints" / "sam2.1_hiera_tiny.pt",
            )
            return True
        except Exception as error:
            self.after(0, lambda: self.status.set(f"YOLO/SAM2 加载失败：{error}"))
            return False

    def _ensure_capture_scan(self):
        if self.capture_scan is not None:
            return self.capture_scan
        scan = ONLINE_SCAN_ROOT / datetime.now().strftime("scan_%Y%m%d_%H%M%S")
        for name in ("images", "depth", "masks", "preview"):
            (scan / name).mkdir(parents=True, exist_ok=True)
        (scan / "camera_intrinsics.json").write_text(json.dumps({
            "width": 1280, "height": 720, "fx": 672.917602539, "fy": 672.917602539,
            "cx": 639.5, "cy": 359.5, "depth_unit": "meter", "min_depth_m": .2, "max_depth_m": 8.0,
        }, indent=2), encoding="utf-8")
        self.capture_scan = scan
        return scan

    @staticmethod
    def _fusion_message(scan, frame_id):
        """Explain whether a saved frame actually entered the fused model."""
        try:
            status = json.loads((Path(scan) / "fusion" / "fusion_status.json").read_text(encoding="utf-8"))
            accepted = int(status.get("frame_count", 0))
            total = int(status.get("input_frame_count", 0))
            state = status.get("last_frame_status") if status.get("last_frame_id") == frame_id else None
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return "点云已保存，但无法读取融合报告。"
        if state in {"anchor", "camera_pose_anchor", "accepted", "camera_pose_plus_icp"}:
            return f"该帧已进入 ICP 融合（当前 {accepted}/{total} 张）。"
        reasons = {
            "skipped_no_pose_consistent_icp_path": "该帧已生成单帧点云，但没有通过 ZED 位姿约束的 ICP 连接；请补拍与已融合面有重叠的中间视角。",
            "skipped_no_pose_guided_icp": "该帧已生成单帧点云，但以第一张相机位姿为初值的 ICP 未找到可靠重叠，未加入融合。",
            "rejected_ambiguous_icp": "该帧已生成单帧点云，但 ICP 特征匹配不可靠，未加入融合。",
            "rejected_pose_jump": "该帧已生成单帧点云，但 ICP 与 ZED 相机位姿差异过大，疑似前后面误匹配，未加入融合。",
            "unknown": "该帧已生成单帧点云，但融合状态未知。",
        }
        return reasons.get(state, f"该帧已生成单帧点云，但未进入融合（状态：{state or 'unknown'}）。")

    def start_new_capture_scan(self):
        """Forget any selected history; next capture creates a new scan."""
        if self.capture_busy or self.routine_running:
            messagebox.showwarning("抓拍进行中", "请等待当前抓拍或规则运动结束后再切换点云文件。", parent=self)
            return
        self.capture_scan = None
        self.cloud_mtime = None
        self.last_cloud_points = None
        self.last_cloud_colors = None
        self.status.set("已切换为新的点云会话：下一次抓拍将创建新的 scan_ 文件夹，不会使用历史点云。")

    def select_existing_capture_scan(self):
        """Explicitly opt in to adding new captures to one historical scan."""
        if self.capture_busy or self.routine_running:
            messagebox.showwarning("抓拍进行中", "请等待当前抓拍或规则运动结束后再选择历史点云。", parent=self)
            return
        selected = filedialog.askdirectory(
            parent=self,
            title="选择要继续融合的历史 scan_ 文件夹",
            initialdir=str(ONLINE_SCAN_ROOT),
        )
        if not selected:
            return
        scan = Path(selected).resolve()
        # A user may have clicked images/depth/masks inside a scan by habit.
        if scan.name in {"images", "depth", "masks", "preview", "clouds", "fusion"}:
            scan = scan.parent
        required = (scan / "images", scan / "depth", scan / "masks", scan / "camera_intrinsics.json")
        if not all(path.exists() for path in required):
            messagebox.showerror(
                "不是有效扫描文件夹",
                "请选择包含 images、depth、masks 和 camera_intrinsics.json 的 scan_ 文件夹。",
                parent=self,
            )
            return
        self.capture_scan = scan
        self.cloud_mtime = None
        self.status.set(f"已选择历史点云：{scan.name}。之后抓拍会追加到该文件夹并继续 ICP 融合。")

    def capture_integrated(self):
        """Capture/save in this UI; no standalone OpenCV live window is opened."""
        if not self.camera_enabled:
            self.enable_remote_capture()
            self.status.set("远程 ZED 正在连接；画面出现后再按 Enter 或点击抓拍。")
            return
        if self.capture_busy:
            self.status.set("上一张正在进行 YOLO/SAM2 分割与点云融合，请等待。")
            return
        self.capture_busy = True
        self.status.set("正在抓取远程 RGB + 深度…")
        threading.Thread(target=self._capture_integrated_worker, daemon=True).start()

    def _capture_integrated_worker(self):
        try:
            if not self._load_capture_models():
                return
            bgr, depth = self.remote_camera.capture_frame()
            if bgr is None or depth is None:
                self.after(0, lambda: self.status.set("抓拍失败：未收到远程深度；请等待画面稳定后重试。"))
                return
            self.last_depth = depth
            self.after(0, lambda: self.camera_status.set("远程相机：已抓拍，正在 YOLO + SAM2 分割…"))
            module = self.capture_module
            box, yolo_points, yolo_labels, _yolo_details = module.yolo_object_prompt(self.capture_detector, bgr)
            automatic = box is not None
            if box is None:
                height, width = bgr.shape[:2]
                box = np.array([width * .04, height * .04, width * .96, height * .96], np.float32)
            self.capture_predictor.set_image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            mask = module.sam_from_prompt(self.capture_predictor, box, yolo_points, yolo_labels)
            preview = bgr.copy()
            preview[mask] = (preview[mask] * .42 + np.array((0, 220, 0)) * .58).astype(np.uint8)
            scan = self._ensure_capture_scan()
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            cv2.imwrite(str(scan / "images" / f"{stamp}.jpg"), bgr)
            np.save(scan / "depth" / f"{stamp}_depth.npy", depth)
            cv2.imwrite(str(scan / "masks" / f"{stamp}.png"), mask.astype(np.uint8) * 255)
            cv2.imwrite(str(scan / "preview" / f"{stamp}.jpg"), preview)
            tcp = module.read_capture_tcp_pose()
            with (scan / "captures.jsonl").open("a", encoding="utf-8") as manifest:
                manifest.write(json.dumps({
                    "id": stamp,
                    "pose_source": "calibrated_tcp" if tcp else "unavailable",
                    "world_from_camera": tcp["world_from_camera"] if tcp else None,
                    "calibrated_tcp": tcp,
                }, ensure_ascii=False) + "\n")
            pipeline = ONLINE_SCAN_ROOT / "online_icp_pipeline.py"
            result = subprocess.run([sys.executable, str(pipeline), str(scan), stamp], text=True, capture_output=True)
            if result.returncode:
                message = result.stderr[-400:] or result.stdout[-400:]
                self.after(0, lambda: self.status.set(f"已保存 {stamp}，但点云融合失败：{message}"))
            else:
                message = self._fusion_message(scan, stamp)
                self.after(0, lambda message=message: self.status.set(f"已保存 {stamp}；{message}"))
        except Exception as error:
            self.after(0, lambda: self.status.set(f"抓拍处理失败：{error}"))
        finally:
            self.capture_busy = False

    def _refresh_camera_frame(self):
        if not self.winfo_exists():
            return
        if not self.camera_enabled or self.capture_busy:
            self.after(120, self._refresh_camera_frame)
            return
        if not self.camera_frame_pending:
            self.camera_frame_pending = True
            threading.Thread(target=self._camera_frame_worker, daemon=True).start()
        self.after(120, self._refresh_camera_frame)

    def _camera_frame_worker(self):
        frame, depth = self.remote_camera.live_frame()
        if frame is None and not self.camera_connecting:
            self.camera_connecting = True
            self._start_remote_camera_bridge()
            frame, depth = self.remote_camera.live_frame()
        self.camera_frame_pending = False
        if frame is None:
            self.after(0, lambda: self.camera_status.set("远程相机：等待 ZED/SSH 桥接…"))
            return
        shown = cv2.resize(frame, (280, 158), interpolation=cv2.INTER_AREA)
        depth_shown = self._colour_depth(depth if depth is not None else self.last_depth, shown.shape[1], shown.shape[0])
        rgb = cv2.cvtColor(np.hstack((shown, depth_shown)), cv2.COLOR_BGR2RGB)
        self.after(0, lambda: self._apply_camera_frame(rgb))

    @staticmethod
    def _colour_depth(depth, width, height):
        panel = np.zeros((height, width, 3), dtype=np.uint8)
        if depth is None:
            cv2.putText(panel, "Depth waiting...", (12, height // 2), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 220, 255), 1)
            return panel
        valid = np.isfinite(depth) & (depth > .2) & (depth < 8.0)
        scaled = np.zeros(depth.shape, dtype=np.uint8)
        scaled[valid] = np.clip((depth[valid] - .2) / 7.8 * 255, 0, 255).astype(np.uint8)
        panel = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
        panel[~valid] = 0
        return cv2.resize(panel, (width, height), interpolation=cv2.INTER_NEAREST)

    def _apply_camera_frame(self, rgb):
        if not self.winfo_exists() or self.camera_label is None:
            return
        photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.camera_photo = photo  # Tk must retain a reference to the image.
        self.camera_label.configure(image=photo, text="")
        self.camera_status.set("远程相机：实时画面已连接")

    def _refresh_cloud_preview(self):
        if not self.winfo_exists():
            return
        # Never pick the newest file from a previous program run.  Preview
        # only the scan created in this session, or the historical scan the
        # operator explicitly selected with “选择历史点云继续融合”.
        fused = self.capture_scan / "fusion" / "fused_icp.ply" if self.capture_scan else None
        if fused is not None and fused.is_file():
            mtime = fused.stat().st_mtime_ns
            if mtime != self.cloud_mtime:
                self.cloud_mtime = mtime
                threading.Thread(target=self._cloud_preview_worker, args=(fused,), daemon=True).start()
        self.after(1500, self._refresh_cloud_preview)

    def _cloud_preview_worker(self, cloud_path, show_popup=False):
        try:
            import open3d as o3d
            cloud = o3d.io.read_point_cloud(str(cloud_path))
            points = np.asarray(cloud.points)
            colors = np.asarray(cloud.colors)
            if len(points) == 0:
                return
            limit = min(len(points), 18000)
            indices = np.linspace(0, len(points) - 1, limit, dtype=np.int64)
            points = points[indices]
            if len(colors) == len(cloud.points):
                colors = colors[indices]
            else:
                colors = np.full((len(points), 3), .82)
            self.after(0, lambda: self._apply_cloud_preview(points, colors, cloud_path.name, show_popup))
        except Exception:
            return

    def _apply_cloud_preview(self, points, colors, name, show_popup=False):
        if not self.winfo_exists() or self.cloud_axes is None or self.cloud_canvas is None:
            return
        # Preserve the user's current interactive 3D camera angle on refresh.
        elev, azim = self.cloud_axes.elev, self.cloud_axes.azim
        self.cloud_axes.clear()
        self.cloud_axes.set_facecolor("#181818")
        self.cloud_axes.scatter(points[:, 0], points[:, 1], points[:, 2],
                                c=colors, s=.35, depthshade=False)
        self.cloud_axes.set_title("实时融合点云（鼠标拖动旋转，滚轮缩放）", color="white", fontsize=9)
        self.cloud_axes.set_xlabel("X", color="#bbbbbb", fontsize=7)
        self.cloud_axes.set_ylabel("Y", color="#bbbbbb", fontsize=7)
        self.cloud_axes.set_zlabel("Z", color="#bbbbbb", fontsize=7)
        self.cloud_axes.tick_params(colors="#bbbbbb", labelsize=6)
        self.cloud_axes.view_init(elev=elev, azim=azim)
        self.cloud_canvas.draw_idle()
        self.last_cloud_points, self.last_cloud_colors = points, colors
        if show_popup:
            self.open_cloud_popup()
        if self.cloud_popup and self.cloud_popup.winfo_exists():
            self._draw_cloud_popup()

    def _show_fused_cloud_window(self, scan):
        """Refresh and raise the fused-cloud popup after every completed frame."""
        fused = Path(scan) / "fusion" / "fused_icp.ply"
        if fused.is_file():
            threading.Thread(target=self._cloud_preview_worker, args=(fused, True), daemon=True).start()

    def open_cloud_popup(self):
        if self.last_cloud_points is None:
            messagebox.showinfo("尚无点云", "请先完成至少一次抓拍和点云融合。")
            return
        if self.cloud_popup and self.cloud_popup.winfo_exists():
            self.cloud_popup.lift()
            return
        popup = tk.Toplevel(self)
        popup.title("实时融合点云 3D 视图")
        popup.geometry("1000x760")
        popup.minsize(640, 480)
        popup.resizable(True, True)
        self.cloud_popup = popup
        figure = Figure(figsize=(9, 6.5), dpi=100, facecolor="#181818")
        popup.axes = figure.add_subplot(111, projection="3d")
        popup.canvas = FigureCanvasTkAgg(figure, master=popup)
        popup.canvas.get_tk_widget().pack(fill="both", expand=True)
        self._draw_cloud_popup()

    def _draw_cloud_popup(self):
        popup = self.cloud_popup
        if not popup or not popup.winfo_exists() or self.last_cloud_points is None:
            return
        axes = popup.axes
        elev, azim = axes.elev, axes.azim
        axes.clear(); axes.set_facecolor("#181818")
        axes.scatter(self.last_cloud_points[:, 0], self.last_cloud_points[:, 1], self.last_cloud_points[:, 2],
                     c=self.last_cloud_colors, s=.6, depthshade=False)
        axes.set_title("融合点云（左键拖动旋转，滚轮缩放）", color="white")
        axes.tick_params(colors="#bbbbbb")
        axes.view_init(elev=elev, azim=azim)
        popup.canvas.draw_idle()

    def _start_remote_camera_bridge(self):
        """Reuse the established remote ZED publisher and TCP bridge."""
        try:
            result = subprocess.run(
                ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", REMOTE_HOST,
                 "bash ~/zed_code/scripts/start_remote_zed_live.sh && bash ~/zed_code/scripts/start_zed_tcp_bridge.sh"],
                text=True, capture_output=True, timeout=25,
            )
            if result.returncode:
                detail = (result.stderr or result.stdout).strip()[-220:]
                self.after(0, lambda: self.camera_status.set(f"远程相机启动失败：{detail or 'SSH/远程脚本错误'}"))
                return
            if self.camera_tunnel is None or self.camera_tunnel.poll() is not None:
                self.camera_tunnel = subprocess.Popen([
                    "ssh", "-N", "-o", "BatchMode=yes", "-o", "ExitOnForwardFailure=yes",
                    "-L", "127.0.0.1:47778:127.0.0.1:47777", REMOTE_HOST,
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                time.sleep(0.4)
                if self.camera_tunnel.poll() is not None:
                    self.after(0, lambda: self.camera_status.set("远程相机隧道启动失败；请关闭旧的 Remote ZED capture 窗口后重试。"))
        finally:
            self.camera_connecting = False

    def _build(self):
        # Keep the six-joint strip compact: the notebook above it is the
        # primary working area (car routing / Cartesian controls).
        style = ttk.Style(self)
        style.configure("Joint.TLabel", font=("Sans", 9), padding=(0, 0))
        style.configure("Joint.TButton", font=("Sans", 9), padding=(5, 0))
        style.configure("Joint.TEntry", font=("Sans", 9), padding=(2, 0))
        outer = ttk.Frame(self, padding=14)
        outer.grid(row=0, column=0, sticky="nsew")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        for column in range(6):
            outer.columnconfigure(column, weight=1)
        outer.columnconfigure(6, weight=0, minsize=230)
        outer.rowconfigure(6, weight=1)
        ttk.Label(outer, text="PiPER-L 本机控制面板", font=("Sans", 14, "bold")).grid(row=0, column=0, columnspan=6, sticky="w")
        vehicle_summary = ttk.LabelFrame(outer, text="Tracer5 当前信息", padding=8)
        vehicle_summary.grid(row=0, column=6, rowspan=8, sticky="nsew", padx=(10, 0))
        ttk.Label(vehicle_summary, textvariable=self.vehicle_status, foreground="#204a87", wraplength=210).pack(anchor="w", fill="x")
        ttk.Separator(vehicle_summary).pack(fill="x", pady=8)
        ttk.Label(vehicle_summary, textvariable=self.vehicle_pose, foreground="#4e6f30", wraplength=210).pack(anchor="w", fill="x")
        ttk.Separator(vehicle_summary).pack(fill="x", pady=8)
        ttk.Label(vehicle_summary, text="自动抓拍", font=("Sans", 9, "bold")).pack(anchor="w")
        ttk.Label(vehicle_summary, textvariable=self.capture_detection_info, foreground="#7a4b00", wraplength=210).pack(anchor="w", fill="x", pady=(3, 0))
        ttk.Label(vehicle_summary, textvariable=self.auto_capture_phase_info, foreground="#204a87", wraplength=210).pack(anchor="w", fill="x", pady=(7, 0))
        ttk.Label(vehicle_summary, textvariable=self.capture_countdown_info, foreground="#7a4b00", wraplength=210).pack(anchor="w", fill="x", pady=(3, 0))
        ttk.Label(vehicle_summary, text="当前位置每 0.5 秒刷新；详细控制在 Tracer5 小车控制页。", foreground="#666666", wraplength=210).pack(anchor="w", pady=(10, 0))
        ttk.Label(outer, textvariable=self.status, foreground="#204a87", wraplength=680).grid(row=1, column=0, columnspan=6, pady=(5, 12), sticky="w")
        settings = ttk.LabelFrame(outer, text="全局设置（速度 / 自动抓拍）", padding=(8, 5))
        settings.grid(row=2, column=0, columnspan=6, sticky="ew", pady=(0, 10))
        for column in range(9):
            settings.columnconfigure(column, weight=1 if column in {0, 5, 8} else 0)
        ttk.Button(settings, text="刷新姿态 / 状态", command=self.refresh_status).grid(row=0, column=0, columnspan=3, sticky="ew", padx=(0, 8))
        ttk.Label(settings, text="运动速度 (%)").grid(row=0, column=3, sticky="e", padx=(3, 2))
        ttk.Spinbox(settings, from_=1, to=100, increment=1, textvariable=self.speed_percent, width=5).grid(row=0, column=4, sticky="w")
        ttk.Button(settings, text="应用速度", command=self.apply_speed_percent).grid(row=0, column=5, sticky="ew", padx=(5, 8))
        ttk.Checkbutton(settings, text="无合格目标：保存非目标帧并继续", variable=self.skip_no_target_capture,
                        command=self._set_no_target_skip).grid(row=0, column=6, columnspan=3, sticky="w")
        ttk.Label(settings, text="抓拍停顿(s)：").grid(row=1, column=0, sticky="e", padx=(0, 2), pady=(5, 0))
        ttk.Spinbox(settings, from_=0.0, to=10.0, increment=0.2, textvariable=self.capture_settle_s, width=5).grid(row=1, column=1, sticky="w", pady=(5, 0))
        ttk.Label(settings, text="无目标等待(s)：").grid(row=1, column=2, sticky="e", padx=(6, 2), pady=(5, 0))
        ttk.Spinbox(settings, from_=0.2, to=10.0, increment=0.2, textvariable=self.no_target_wait_s, width=5).grid(row=1, column=3, sticky="w", pady=(5, 0))
        ttk.Label(settings, text="最低识别度(%)：").grid(row=1, column=4, sticky="e", padx=(6, 2), pady=(5, 0))
        ttk.Spinbox(settings, from_=1, to=100, increment=1, textvariable=self.min_target_confidence, width=5).grid(row=1, column=5, sticky="w", pady=(5, 0))
        ttk.Button(settings, text="保存全局设置", command=self._save_capture_detection_settings).grid(row=1, column=6, columnspan=3, sticky="ew", padx=(8, 0), pady=(5, 0))
        ttk.Label(outer, textvariable=self.flange_pose, foreground="#204a87", wraplength=680).grid(row=3, column=0, columnspan=6, pady=(0, 10), sticky="w")
        ttk.Label(outer, textvariable=self.camera_pose, foreground="#4e6f30", wraplength=680).grid(row=4, column=0, columnspan=5, pady=(0, 8), sticky="w")
        ttk.Button(outer, text="重设小车初始坐标", command=self.reset_camera_origin).grid(row=4, column=5, padx=4, pady=(0, 8))
        ttk.Label(outer, text="末端控制（相对机械臂底座的标定 TCP；XYZ:mm，RPY:°）", font=("Sans", 10, "bold")).grid(row=5, column=0, columnspan=6, sticky="w", pady=(0, 3))
        motion_pages = ttk.Notebook(outer, width=980, height=360)
        motion_pages.grid(row=6, column=0, columnspan=6, sticky="nsew", pady=(0, 6))
        discrete_page = ttk.Frame(motion_pages, padding=20)
        continuous_page = ttk.Frame(motion_pages, padding=20)
        routine_page = ttk.Frame(motion_pages, padding=20)
        vehicle_page = ttk.Frame(motion_pages, padding=6)
        auto_plan_page = ttk.Frame(motion_pages, padding=8)
        # Pages inherit the notebook's available area and grow with the main
        # window; no tab has a fixed pixel height.
        motion_pages.add(discrete_page, text="间断运动（输入后发送）")
        motion_pages.add(continuous_page, text="连续运动（滑块 / +/- 直接执行）")
        motion_pages.add(routine_page, text="规则运动（记录位置）")
        motion_pages.add(vehicle_page, text="Tracer5 小车控制")
        motion_pages.add(auto_plan_page, text="自动抓拍规划")
        motion_pages.bind("<<NotebookTabChanged>>", self._motion_page_changed)

        self._build_discrete_cartesian_page(discrete_page)
        self._build_continuous_cartesian_page(continuous_page)
        self._build_routine_page(routine_page)
        self._build_vehicle_page(vehicle_page)
        self._build_auto_capture_plan_page(auto_plan_page)
        ttk.Button(outer, text="结束上一段运动（停止 / 清除占用）", command=self.stop_cartesian_motion).grid(row=7, column=0, columnspan=3, sticky="ew", padx=(0, 4), pady=(0, 7))
        capture_actions = ttk.Frame(outer)
        capture_actions.grid(row=7, column=3, columnspan=3, sticky="ew", pady=(0, 7))
        for column in range(3):
            capture_actions.columnconfigure(column, weight=1)
        ttk.Button(capture_actions, text="启动远程 ZED 抓拍", command=self.launch_remote_capture).grid(row=0, column=0, sticky="ew", padx=(0, 3))
        ttk.Button(capture_actions, text="选择历史点云继续融合", command=self.select_existing_capture_scan).grid(row=0, column=1, sticky="ew", padx=3)
        ttk.Button(capture_actions, text="新建点云会话", command=self.start_new_capture_scan).grid(row=0, column=2, sticky="ew", padx=(3, 0))
        joint_panel = ttk.LabelFrame(outer, text="关节控制", padding=(5, 3))
        joint_panel.grid(row=8, column=0, columnspan=6, sticky="w")
        ttk.Label(joint_panel, text="关节", style="Joint.TLabel").grid(row=0, column=0, padx=3)
        ttk.Label(joint_panel, text="当前角度 (°)", style="Joint.TLabel").grid(row=0, column=1, padx=3)
        ttk.Label(joint_panel, text="目标角度 (°)", style="Joint.TLabel").grid(row=0, column=2, columnspan=2, padx=3)
        ttk.Label(joint_panel, text="相对变化 (°)", style="Joint.TLabel").grid(row=0, column=4, padx=3)
        ttk.Label(joint_panel, text="应用", style="Joint.TLabel").grid(row=0, column=5, padx=3)
        for index, name in enumerate(JOINT_NAMES):
            row = index + 1
            ttk.Label(joint_panel, text=name, style="Joint.TLabel").grid(row=row, column=0, padx=3, pady=0)
            label = ttk.Label(joint_panel, text="--", width=9, style="Joint.TLabel")
            label.grid(row=row, column=1, padx=3, pady=0)
            self.current_labels.append(label)
            low, high = JOINT_LIMITS[index]
            slider = ttk.Scale(joint_panel, from_=low, to=high, length=180,
                               command=lambda value, i=index: self.slider_changed(i, value))
            slider.grid(row=row, column=2, padx=(3, 5), pady=0)
            self.sliders.append(slider)
            ttk.Label(joint_panel, textvariable=self.targets[index], width=8, style="Joint.TLabel").grid(row=row, column=3, padx=(0, 3), pady=0)
            ttk.Entry(joint_panel, textvariable=self.relative_inputs[index], width=6, style="Joint.TEntry").grid(row=row, column=4, padx=3, pady=0)
            ttk.Button(joint_panel, text="应用", style="Joint.TButton", command=lambda i=index: self.apply_relative(i)).grid(row=row, column=5, padx=3, pady=0)
        ttk.Separator(outer).grid(row=9, column=0, columnspan=6, sticky="ew", pady=5)
        ttk.Checkbutton(outer, text="我已确认机械臂周围无人、无障碍物，且可随时按下实体急停", variable=self.safety_ok).grid(row=10, column=0, columnspan=6, sticky="w")
        ttk.Button(outer, text="使能电机（不会运动）", command=self.enable_motors).grid(row=11, column=0, columnspan=6, sticky="ew", pady=(4, 2))
        ttk.Button(outer, text="恢复初始姿态（零位，当前速度）", command=self.move_home).grid(row=12, column=0, columnspan=6, sticky="ew", pady=2)
        ttk.Button(outer, text="发送低速目标（当前速度）", command=self.send_target).grid(row=13, column=0, columnspan=6, sticky="ew")
        ttk.Label(outer, text="连续运动每次 +/- 为 XYZ 10 mm、姿态 1°；滑块松开时执行。速度由上方“应用速度”设置。", foreground="#555555").grid(row=14, column=0, columnspan=6, pady=(4, 0), sticky="w")

    def _build_vehicle_page(self, page):
        """Tracer5 controls embedded in the PiPER window (no second UI)."""
        # Keep all vehicle functions reachable inside the fixed-height arm
        # notebook by using a second, compact set of pages.
        tabs = ttk.Notebook(page)
        tabs.pack(fill="both", expand=True)
        manual = ttk.Frame(tabs, padding=12)
        relative = ttk.Frame(tabs, padding=12)
        absolute = ttk.Frame(tabs, padding=12)
        planned = ttk.Frame(tabs, padding=12)
        records = ttk.Frame(tabs, padding=12)
        tabs.add(manual, text="手动 / 即停")
        tabs.add(relative, text="相对运动")
        tabs.add(absolute, text="绝对运动")
        tabs.add(planned, text="规划回放")
        tabs.add(records, text="小车记录")
        vehicle_split = tk.PanedWindow(manual, orient="horizontal", sashwidth=8, sashrelief="raised", bd=0)
        vehicle_split.pack(fill="x", pady=(0, 4))
        vehicle_left = ttk.Frame(vehicle_split)
        vehicle_right = ttk.Frame(vehicle_split)
        vehicle_split.add(vehicle_left, minsize=260)
        vehicle_split.add(vehicle_right, minsize=260)
        ttk.Label(vehicle_left, textvariable=self.vehicle_status, foreground="#204a87").pack(anchor="w")
        ttk.Label(vehicle_left, textvariable=self.vehicle_pose, foreground="#4e6f30").pack(anchor="w", pady=(2, 0))
        ttk.Label(vehicle_right, text="可拖动中间分隔条，调整状态区与参数区宽度", foreground="#666666").pack(anchor="w")
        speed = ttk.Frame(manual); speed.pack(fill="x")
        ttk.Label(speed, text="线速度").pack(side="left")
        ttk.Spinbox(speed, from_=0.02, to=0.40, increment=0.01, textvariable=self.vehicle_v, width=7).pack(side="left", padx=5)
        ttk.Label(speed, text="角速度").pack(side="left", padx=(18, 0))
        ttk.Spinbox(speed, from_=0.10, to=1.50, increment=0.05, textvariable=self.vehicle_w, width=7).pack(side="left", padx=5)
        ttk.Button(speed, text="小车即停", command=self._vehicle_stop).pack(side="right")
        tolerance = ttk.Frame(manual); tolerance.pack(fill="x", pady=(5, 0))
        ttk.Label(tolerance, text="到达位置误差 cm").pack(side="left")
        ttk.Spinbox(tolerance, from_=0.1, to=20.0, increment=0.1, textvariable=self.vehicle_position_tolerance, width=7).pack(side="left", padx=5)
        ttk.Label(tolerance, text="角度误差 °").pack(side="left", padx=(14, 0))
        ttk.Spinbox(tolerance, from_=0.1, to=20.0, increment=0.1, textvariable=self.vehicle_yaw_tolerance, width=7).pack(side="left", padx=5)
        ttk.Button(tolerance, text="应用误差", command=self._apply_vehicle_tolerances).pack(side="left", padx=6)
        ttk.Button(tolerance, text="保存该误差", command=self._save_vehicle_tolerances).pack(side="left", padx=(0, 4))
        keys = ttk.Frame(manual); keys.pack(pady=6)
        ttk.Button(keys, text="↑前进", width=8, command=lambda: self._vehicle_manual(1, 0)).pack(side="left", padx=3)
        ttk.Button(keys, text="↓后退", width=8, command=lambda: self._vehicle_manual(-1, 0)).pack(side="left", padx=3)
        ttk.Button(keys, text="←左转", width=8, command=lambda: self._vehicle_manual(0, 1)).pack(side="left", padx=3)
        ttk.Button(keys, text="→右转", width=8, command=lambda: self._vehicle_manual(0, -1)).pack(side="left", padx=3)
        ttk.Button(keys, text="停止", width=8, command=self._vehicle_stop).pack(side="left", padx=3)
        ttk.Label(relative, text="以当前车体为原点：X 前方为正，Y 左侧为正；X/Y 步进 10 cm。").pack(anchor="w")
        rel_row = ttk.Frame(relative); rel_row.pack(pady=16)
        for label, var in (("X cm", self.vehicle_rx), ("Y cm", self.vehicle_ry), ("Yaw°", self.vehicle_ryaw)):
            ttk.Label(rel_row, text=label).pack(side="left", padx=(8, 2))
            ttk.Spinbox(rel_row, from_=-1000, to=1000, increment=10 if label != "Yaw°" else 5, textvariable=var, width=8).pack(side="left")
        ttk.Button(relative, text="运行相对目标", command=self._vehicle_relative).pack(fill="x", ipady=5)
        ttk.Label(absolute, text="以本次启动的车体位置/朝向为原点；X/Y 步进 10 cm。").pack(anchor="w")
        abs_row = ttk.Frame(absolute); abs_row.pack(pady=12)
        for label, var in (("X cm", self.vehicle_ax), ("Y cm", self.vehicle_ay), ("Yaw°", self.vehicle_ayaw)):
            ttk.Label(abs_row, text=label).pack(side="left", padx=(8, 2))
            ttk.Spinbox(abs_row, from_=-1000, to=1000, increment=10 if label != "Yaw°" else 5, textvariable=var, width=8).pack(side="left")
        ttk.Checkbutton(absolute, text="保持当前 Yaw", variable=self.vehicle_keep_yaw).pack(anchor="w")
        ttk.Button(absolute, text="运行绝对目标", command=self._vehicle_absolute).pack(fill="x", ipady=5, pady=(8, 3))
        ttk.Button(absolute, text="将当前位置设为本次绝对原点", command=self.tracer.reset_origin).pack(fill="x", ipady=3)
        ttk.Label(planned, text="开始记录后，用“手动 / 即停”页驾驶；结束记录后可回放。", foreground="#555555").pack(anchor="w")
        ttk.Button(planned, text="开始 / 结束记录手动轨迹", command=self._vehicle_record).pack(fill="x", ipady=6, pady=(16, 5))
        ttk.Button(planned, text="回放记录的轨迹", command=self._vehicle_replay).pack(fill="x", ipady=6)
        ttk.Label(records, text="手动录入小车路径点", font=("Sans", 11, "bold")).pack(anchor="w")
        ttk.Label(records, text="输入 X、Y、Yaw 后选择“相对”或“绝对”加入列表；列表顺序就是点 1、点 2… 的运行顺序。", foreground="#555555").pack(anchor="w", pady=(2, 4))
        self.vehicle_record_tree = ttk.Treeview(records, columns=("name", "type", "x", "y", "yaw"), show="headings", height=3)
        for key, text, width in (("name", "路径点", 90), ("type", "模式", 80), ("x", "X cm", 120), ("y", "Y cm", 120), ("yaw", "Yaw°", 120)):
            self.vehicle_record_tree.heading(key, text=text)
            self.vehicle_record_tree.column(key, width=width, anchor="center")
        self.vehicle_record_tree.pack(fill="both", expand=True, pady=(3, 4))
        entry = ttk.Frame(records); entry.pack(fill="x")
        ttk.Combobox(entry, textvariable=self.vehicle_record_kind, values=("相对", "绝对"), state="readonly", width=6).pack(side="left", padx=3)
        for label, variable in (("X cm", self.vehicle_record_x), ("Y cm", self.vehicle_record_y), ("Yaw°", self.vehicle_record_yaw)):
            ttk.Label(entry, text=label).pack(side="left", padx=(8, 2))
            ttk.Spinbox(entry, textvariable=variable, from_=-1000, to=1000, increment=10 if label != "Yaw°" else 5, width=7).pack(side="left")
        ttk.Button(entry, text="添加", command=self._add_vehicle_record).pack(side="left", padx=5)
        row = ttk.Frame(records); row.pack(fill="x", pady=(3, 0))
        ttk.Button(row, text="上移选中点", command=lambda: self._move_vehicle_record(-1)).pack(side="left", fill="x", expand=True, padx=(0, 3))
        ttk.Button(row, text="下移选中点", command=lambda: self._move_vehicle_record(1)).pack(side="left", fill="x", expand=True, padx=3)
        ttk.Button(row, text="删除选中", command=self._delete_vehicle_record).pack(side="left", fill="x", expand=True, padx=(3, 0))
        run_row = ttk.Frame(records); run_row.pack(fill="x", pady=(3, 0))
        ttk.Button(run_row, text="依次运行", command=self._run_vehicle_records).pack(side="left", fill="x", expand=True, padx=(0, 2))
        ttk.Button(run_row, text="自动抓拍并运行", command=self._run_vehicle_auto_capture).pack(side="left", fill="x", expand=True, padx=2)
        ttk.Button(run_row, text="保存记录", command=self._save_vehicle_motion_records).pack(side="left", fill="x", expand=True, padx=2)
        ttk.Button(run_row, text="删除全部", command=self._clear_vehicle_motion_records).pack(side="left", fill="x", expand=True, padx=(2, 0))
        self._refresh_vehicle_records()
        return
        for col in range(6):
            page.columnconfigure(col, weight=1)
        ttk.Label(page, textvariable=self.vehicle_status, foreground="#204a87").grid(row=0, column=0, columnspan=6, sticky="w")
        ttk.Label(page, textvariable=self.vehicle_pose, foreground="#4e6f30").grid(row=1, column=0, columnspan=6, sticky="w", pady=(3, 8))
        ttk.Label(page, text="线速度").grid(row=2, column=0, sticky="e")
        ttk.Spinbox(page, from_=0.02, to=0.40, increment=0.01, textvariable=self.vehicle_v, width=7).grid(row=2, column=1, sticky="w")
        ttk.Label(page, text="角速度").grid(row=2, column=2, sticky="e")
        ttk.Spinbox(page, from_=0.10, to=1.50, increment=0.05, textvariable=self.vehicle_w, width=7).grid(row=2, column=3, sticky="w")
        ttk.Button(page, text="小车即停", command=self._vehicle_stop).grid(row=2, column=4, columnspan=2, sticky="ew")
        ttk.Button(page, text="↑ 前进", command=lambda: self._vehicle_manual(1, 0)).grid(row=3, column=2, sticky="ew", pady=(8, 2))
        ttk.Button(page, text="← 左转", command=lambda: self._vehicle_manual(0, 1)).grid(row=4, column=1, sticky="ew", padx=3)
        ttk.Button(page, text="停止", command=self._vehicle_stop).grid(row=4, column=2, sticky="ew", padx=3)
        ttk.Button(page, text="→ 右转", command=lambda: self._vehicle_manual(0, -1)).grid(row=4, column=3, sticky="ew", padx=3)
        ttk.Button(page, text="↓ 后退", command=lambda: self._vehicle_manual(-1, 0)).grid(row=5, column=2, sticky="ew", pady=2)
        ttk.Separator(page).grid(row=6, column=0, columnspan=6, sticky="ew", pady=5)
        ttk.Label(page, text="相对：X前/Y左（cm）").grid(row=7, column=0, sticky="e")
        for col, variable, label in ((1, self.vehicle_rx, "X"), (2, self.vehicle_ry, "Y"), (3, self.vehicle_ryaw, "Yaw°")):
            ttk.Label(page, text=label).grid(row=7, column=col, sticky="e")
            ttk.Spinbox(page, from_=-1000, to=1000, increment=10 if label != "Yaw°" else 5, textvariable=variable, width=7).grid(row=8, column=col)
        ttk.Button(page, text="运行相对目标", command=self._vehicle_relative).grid(row=8, column=4, columnspan=2, sticky="ew")
        ttk.Label(page, text="绝对：启动时车体坐标（cm）").grid(row=9, column=0, sticky="e", pady=(8, 0))
        for col, variable, label in ((1, self.vehicle_ax, "X"), (2, self.vehicle_ay, "Y"), (3, self.vehicle_ayaw, "Yaw°")):
            ttk.Label(page, text=label).grid(row=9, column=col, sticky="e", pady=(8, 0))
            ttk.Spinbox(page, from_=-1000, to=1000, increment=10 if label != "Yaw°" else 5, textvariable=variable, width=7).grid(row=10, column=col)
        ttk.Checkbutton(page, text="保持当前 Yaw", variable=self.vehicle_keep_yaw).grid(row=10, column=0, sticky="w")
        ttk.Button(page, text="运行绝对目标", command=self._vehicle_absolute).grid(row=10, column=4, columnspan=2, sticky="ew")
        ttk.Button(page, text="开始/结束记录手动小车轨迹", command=self._vehicle_record).grid(row=11, column=0, columnspan=3, sticky="ew", pady=(8, 0))
        ttk.Button(page, text="回放小车轨迹", command=self._vehicle_replay).grid(row=11, column=3, columnspan=3, sticky="ew", pady=(8, 0))

    def _build_auto_capture_plan_page(self, page):
        """One sortable queue composed from existing arm and vehicle records."""
        ttk.Label(page, text="自动抓拍规划", font=("Sans", 11, "bold")).pack(anchor="w")
        ttk.Label(
            page,
            text="不在这里重复录入坐标。点击“生成规划”后，已有的机械臂相机位置和小车路径点会加入同一队列；可上移/下移调整执行顺序。相机步骤到位后自动抓拍并融合点云。",
            foreground="#555555", wraplength=760,
        ).pack(anchor="w", pady=(2, 5))
        self.auto_capture_plan_tree = ttk.Treeview(
            page, columns=("step", "action", "detail"), show="headings", height=10
        )
        for key, text, width in (("step", "步骤", 75), ("action", "动作", 150), ("detail", "记录来源 / 目标", 520)):
            self.auto_capture_plan_tree.heading(key, text=text)
            self.auto_capture_plan_tree.column(key, width=width, anchor="center" if key != "detail" else "w")
        self.auto_capture_plan_tree.pack(fill="both", expand=True, pady=(0, 6))
        row = ttk.Frame(page); row.pack(fill="x")
        ttk.Button(row, text="从已有记录生成规划", command=self._generate_auto_capture_plan).pack(side="left", fill="x", expand=True, padx=(0, 3))
        ttk.Button(row, text="上移选中步骤", command=lambda: self._move_auto_capture_plan(-1)).pack(side="left", fill="x", expand=True, padx=3)
        ttk.Button(row, text="下移选中步骤", command=lambda: self._move_auto_capture_plan(1)).pack(side="left", fill="x", expand=True, padx=3)
        ttk.Button(row, text="复制选中步骤", command=self._duplicate_auto_capture_plan_step).pack(side="left", fill="x", expand=True, padx=3)
        ttk.Button(row, text="移除选中步骤", command=self._remove_auto_capture_plan_step).pack(side="left", fill="x", expand=True, padx=(3, 0))
        saved = ttk.Frame(page); saved.pack(fill="x", pady=(5, 0))
        ttk.Label(saved, text="规划名称：").pack(side="left")
        ttk.Entry(saved, textvariable=self.auto_capture_plan_name, width=18).pack(side="left", padx=(0, 4))
        ttk.Button(saved, text="保存规划", command=self._save_named_auto_capture_plan).pack(side="left", padx=(0, 8))
        ttk.Label(saved, text="已保存：").pack(side="left")
        self.saved_auto_capture_plan_picker = ttk.Combobox(
            saved, textvariable=self.auto_capture_plan_name, state="readonly", width=18
        )
        self.saved_auto_capture_plan_picker.pack(side="left", padx=(0, 4))
        self.saved_auto_capture_plan_picker.bind("<<ComboboxSelected>>", lambda _event: None)
        ttk.Button(saved, text="加载规划", command=self._load_named_auto_capture_plan).pack(side="left", padx=2)
        ttk.Button(saved, text="删除保存", command=self._delete_named_auto_capture_plan).pack(side="left", padx=(2, 0))
        self._refresh_saved_auto_capture_plan_picker()
        ttk.Button(page, text="运行规划", command=self._run_auto_capture_plan).pack(fill="x", ipady=4, pady=(6, 0))

    def _generate_auto_capture_plan(self):
        """Build the plan from the existing record pages without new input."""
        plan = []
        for index, record in enumerate(self.motion_records, 1):
            plan.append({"kind": "camera", "name": f"相机位置 {index}", "pose": list(record["pose"])})
        for index, record in enumerate(self.vehicle_motion_records, 1):
            plan.append({"kind": "vehicle", "name": f"小车点 {index}", "record": dict(record)})
        self.auto_capture_plan = plan
        self._refresh_auto_capture_plan_tree()
        self.status.set(f"自动抓拍规划已生成：相机 {len(self.motion_records)} 个，小车 {len(self.vehicle_motion_records)} 个；请调整顺序后运行。")

    @staticmethod
    def _copy_auto_capture_plan(plan):
        copied = []
        for step in plan:
            if not isinstance(step, dict):
                continue
            if step.get("kind") == "camera":
                pose = step.get("pose")
                if isinstance(pose, list) and len(pose) == 6:
                    copied.append({"kind": "camera", "name": str(step.get("name", "相机位置")),
                                   "pose": [float(value) for value in pose]})
            elif step.get("kind") == "vehicle":
                record = step.get("record")
                if isinstance(record, dict) and all(key in record for key in ("type", "x", "y", "yaw")):
                    copied.append({"kind": "vehicle", "name": str(step.get("name", "小车点")),
                                   "record": {"type": str(record["type"]), "x": float(record["x"]),
                                              "y": float(record["y"]), "yaw": float(record["yaw"])}})
        return copied

    @classmethod
    def _load_saved_auto_capture_plans(cls):
        try:
            with open(AUTO_CAPTURE_PLANS_PATH, "r", encoding="utf-8") as handle:
                plans = json.load(handle)
            if not isinstance(plans, dict):
                raise ValueError("plans must be an object")
            return {str(name): cls._copy_auto_capture_plan(plan)
                    for name, plan in plans.items() if isinstance(plan, list)}
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return {}

    def _save_saved_auto_capture_plans(self):
        with open(AUTO_CAPTURE_PLANS_PATH, "w", encoding="utf-8") as handle:
            json.dump(self.saved_auto_capture_plans, handle, ensure_ascii=False, indent=2)

    def _refresh_saved_auto_capture_plan_picker(self):
        if hasattr(self, "saved_auto_capture_plan_picker"):
            self.saved_auto_capture_plan_picker["values"] = sorted(self.saved_auto_capture_plans)

    def _save_named_auto_capture_plan(self):
        name = self.auto_capture_plan_name.get().strip()
        if not name:
            return messagebox.showwarning("请输入名称", "请先为规划输入一个名称。", parent=self)
        if not self.auto_capture_plan:
            return messagebox.showwarning("尚无规划", "请先生成或加载一个规划后再保存。", parent=self)
        if name in self.saved_auto_capture_plans and not messagebox.askyesno(
            "覆盖规划", f"“{name}”已存在，确认用当前规划覆盖吗？", parent=self
        ):
            return
        self.saved_auto_capture_plans[name] = self._copy_auto_capture_plan(self.auto_capture_plan)
        self._save_saved_auto_capture_plans()
        self._refresh_saved_auto_capture_plan_picker()
        self.auto_capture_plan_name.set(name)
        self.status.set(f"已保存规划“{name}”（{len(self.auto_capture_plan)} 个步骤）；重启后可直接加载。")

    def _load_named_auto_capture_plan(self):
        name = self.auto_capture_plan_name.get().strip()
        plan = self.saved_auto_capture_plans.get(name)
        if plan is None:
            return messagebox.showwarning("请选择规划", "请选择一个已保存的规划。", parent=self)
        self.auto_capture_plan = self._copy_auto_capture_plan(plan)
        self._refresh_auto_capture_plan_tree()
        self.status.set(f"已加载规划“{name}”（{len(self.auto_capture_plan)} 个步骤），可直接运行或调整。")

    def _delete_named_auto_capture_plan(self):
        name = self.auto_capture_plan_name.get().strip()
        if name not in self.saved_auto_capture_plans:
            return messagebox.showwarning("请选择规划", "请选择一个已保存的规划。", parent=self)
        if not messagebox.askyesno("删除保存规划", f"确认删除规划“{name}”？", parent=self):
            return
        del self.saved_auto_capture_plans[name]
        self._save_saved_auto_capture_plans()
        self._refresh_saved_auto_capture_plan_picker()
        self.auto_capture_plan_name.set("")
        self.status.set(f"已删除保存规划“{name}”。")

    def _refresh_auto_capture_plan_tree(self):
        if not hasattr(self, "auto_capture_plan_tree"):
            return
        self.auto_capture_plan_tree.delete(*self.auto_capture_plan_tree.get_children())
        for index, step in enumerate(self.auto_capture_plan):
            if step["kind"] == "camera":
                pose = step["pose"]
                detail = "TCP: " + ", ".join(f"{value:.1f}" for value in pose)
                action = "机械臂到相机位 + 抓拍"
            else:
                record = step["record"]
                detail = f'{record["type"]}：X={record["x"]:.1f} cm，Y={record["y"]:.1f} cm，Yaw={record["yaw"]:.1f}°'
                action = "小车移动"
            self.auto_capture_plan_tree.insert("", "end", iid=str(index), values=(f"步骤 {index + 1}", action, detail))

    def _selected_auto_capture_plan_index(self):
        selection = self.auto_capture_plan_tree.selection() if hasattr(self, "auto_capture_plan_tree") else ()
        return int(selection[0]) if selection else None

    def _move_auto_capture_plan(self, direction):
        index = self._selected_auto_capture_plan_index()
        if index is None:
            return messagebox.showinfo("请选择步骤", "请先选择需要调整顺序的自动抓拍步骤。", parent=self)
        new_index = max(0, min(len(self.auto_capture_plan) - 1, index + direction))
        if index == new_index:
            return
        step = self.auto_capture_plan.pop(index)
        self.auto_capture_plan.insert(new_index, step)
        self._refresh_auto_capture_plan_tree()
        self.auto_capture_plan_tree.selection_set(str(new_index))
        self.auto_capture_plan_tree.focus(str(new_index))

    def _duplicate_auto_capture_plan_step(self):
        """Insert an independent copy of the selected step immediately after it."""
        index = self._selected_auto_capture_plan_index()
        if index is None:
            return messagebox.showinfo("请选择步骤", "请先选择需要复制的自动抓拍步骤。", parent=self)
        step = self.auto_capture_plan[index]
        if step["kind"] == "camera":
            duplicate = {"kind": "camera", "name": step["name"], "pose": list(step["pose"])}
        else:
            duplicate = {"kind": "vehicle", "name": step["name"], "record": dict(step["record"])}
        new_index = index + 1
        self.auto_capture_plan.insert(new_index, duplicate)
        self._refresh_auto_capture_plan_tree()
        self.auto_capture_plan_tree.selection_set(str(new_index))
        self.auto_capture_plan_tree.focus(str(new_index))
        self.status.set(f"已复制步骤 {index + 1}；副本已插入为步骤 {new_index + 1}。")

    def _remove_auto_capture_plan_step(self):
        index = self._selected_auto_capture_plan_index()
        if index is None:
            return
        self.auto_capture_plan.pop(index)
        self._refresh_auto_capture_plan_tree()

    def _run_auto_capture_plan(self):
        """Run the combined, user-sorted arm/camera and vehicle queue."""
        if not self.auto_capture_plan:
            return messagebox.showwarning("尚无规划", "请先点击“从已有记录生成规划”，并确认步骤顺序。", parent=self)
        if self.auto_capture_plan_running or self.routine_running or self.cartesian_busy or self.vehicle_route_running:
            return messagebox.showwarning("运动进行中", "请等待当前机械臂或小车运动完成，或使用停止按钮中断。", parent=self)
        camera_steps = sum(step["kind"] == "camera" for step in self.auto_capture_plan)
        if camera_steps and (not self.can_ready or not self.current_flange):
            return messagebox.showwarning("CAN 未就绪", "未收到有效机械臂反馈，不能执行相机抓拍步骤。", parent=self)
        pose, origin, _source, stamp, _state, _name = self.tracer.snapshot()
        if any(step["kind"] == "vehicle" for step in self.auto_capture_plan):
            if pose is None or origin is None or time.monotonic() - stamp > 1.0:
                return messagebox.showwarning("小车未定位", "未收到有效 Tracer5 融合定位，不能执行小车步骤。", parent=self)
        if not self.safety_ok.get():
            return messagebox.showwarning("需要安全确认", "请先勾选底部的机械臂安全确认。", parent=self)
        if not messagebox.askyesno(
            "确认自动抓拍规划",
            f"将按当前排序执行 {len(self.auto_capture_plan)} 个步骤，其中相机抓拍 {camera_steps} 次。\n"
            "相机步骤：机械臂到位 → 等待 2 秒 → YOLO/SAM2 抓拍并融合点云。\n"
            "小车步骤：到达对应目标后才会继续。可用停止按钮或小车即停中断。\n\n确认现场安全？",
            parent=self,
        ):
            return
        self.auto_capture_plan_running = True
        self.auto_capture_plan_cancelled = False
        self._set_capture_detection_options()
        self.vehicle_route_cancelled = False
        self.routine_running = True
        self.routine_auto_capture = True
        self.auto_capture_failed = False
        plan = [dict(step) for step in self.auto_capture_plan]
        speed = (float(self.vehicle_v.get()), float(self.vehicle_w.get()))
        self.status.set("自动抓拍规划已启动：正在准备远程 ZED…")
        threading.Thread(target=self._run_auto_capture_plan_worker, args=(plan, speed), daemon=True).start()

    def _run_auto_capture_plan_worker(self, plan, vehicle_speed):
        try:
            if not self._prepare_auto_capture():
                return
            captures = 0
            for index, step in enumerate(plan, 1):
                if not self.routine_running or self.auto_capture_plan_cancelled:
                    self.after(0, lambda: self.status.set("自动抓拍规划已停止。"))
                    return
                if step["kind"] == "camera":
                    target = list(step["pose"])
                    self._set_auto_capture_phase(f"机械臂运动：规划步骤 {index}/{len(plan)}")
                    self.after(0, lambda i=index, n=len(plan): self.status.set(f"自动抓拍规划 {i}/{n}：机械臂正在前往相机位…"))
                    result = self.run_remote(self._cartesian_command(target))
                    # piper_safe_end_pose blocks until its CAN end-pose
                    # feedback confirms arrival, so no ROS TCP topic wait is
                    # required here.
                    if result.returncode:
                        self.after(0, lambda i=index: self.status.set(f"自动抓拍规划：第 {i} 步机械臂未确认到位，已停止。"))
                        return
                    deadline = time.monotonic() + self._capture_settle_s_active
                    while self.routine_running and not self.auto_capture_plan_cancelled and time.monotonic() < deadline:
                        self._set_auto_capture_phase("机械臂到位：抓拍停顿计时")
                        self._set_capture_countdown("抓拍停顿", deadline - time.monotonic(), self._capture_settle_s_active)
                        self._refresh_auto_camera(f"step {index}: stable 2 seconds")
                        time.sleep(0.1)
                    if not self.routine_running or self.auto_capture_plan_cancelled:
                        return
                    captures += 1
                    if not self._capture_auto_frame(captures):
                        return
                else:
                    if not self._run_auto_capture_vehicle_step(step["record"], index, len(plan), *vehicle_speed):
                        return
            self.after(0, lambda: self.status.set("自动抓拍规划完成：正在恢复机械臂初始零位…"))
            args = " ".join(f"{value:.3f}" for value in HOME_JOINTS_DEG)
            home = (
                f"cd {REMOTE_DIR} && python3 piper_safe_move.py --can can1 "
                f"--joint-deg {args} --speed {self._joint_speed_level()} --execute --confirm ARM_CLEAR"
            )
            result = self.run_remote(home)
            if result.returncode or not self._wait_for_home_joint_pose(timeout_s=70.0):
                self.after(0, lambda: self.status.set("规划拍摄已完成、点云正在后台融合，但机械臂复位未确认；ZED 保持开启。"))
                return
            self.after(0, lambda: self.status.set(f"自动抓拍规划完成：已抓拍 {captures} 帧，点云继续后台融合；机械臂已回到初始零位。"))
        except Exception as error:
            self.auto_capture_failed = True
            self.after(0, lambda: self.status.set(f"自动抓拍规划失败：{error}"))
        finally:
            self.vehicle_route_running = False
            self.auto_capture_plan_running = False
            self.routine_auto_capture = False
            self.routine_running = False
            if not self.auto_capture_failed:
                self._close_auto_capture_views()
            self.after(0, self.refresh_status)

    def _run_auto_capture_vehicle_step(self, record, index, total, linear, angular):
        """One vehicle waypoint for the combined plan; returns only after arrival."""
        from tracer5_control_panel import Pose2D, wrap_angle
        pose, origin, _source, stamp, _state, _name = self.tracer.snapshot()
        if pose is None or origin is None or time.monotonic() - stamp > 1.0:
            self.after(0, lambda: self.status.set("自动抓拍规划：小车定位不可用，已停止。"))
            return False
        x, y = float(record["x"]) / 100.0, float(record["y"]) / 100.0
        if record["type"] == "相对":
            target = Pose2D(pose.x + math.cos(pose.yaw) * x - math.sin(pose.yaw) * y,
                            pose.y + math.sin(pose.yaw) * x + math.cos(pose.yaw) * y,
                            wrap_angle(pose.yaw + math.radians(float(record["yaw"]))))
            straight = abs(y) < 1e-6 and abs(float(record["yaw"])) < 1e-6
        else:
            target = Pose2D(origin.x + math.cos(origin.yaw) * x - math.sin(origin.yaw) * y,
                            origin.y + math.sin(origin.yaw) * x + math.cos(origin.yaw) * y,
                            wrap_angle(origin.yaw + math.radians(float(record["yaw"]))))
            straight = False
        self._set_auto_capture_phase(f"小车运动：规划点 {index}/{total}")
        ok, reason = self.tracer.start_target(target, f"规划小车点 {index}/{total}", linear, angular, straight=straight)
        if not ok:
            self.after(0, lambda r=reason: self.status.set(f"自动抓拍规划：小车目标未启动：{r}"))
            return False
        self.vehicle_route_running = True
        self.after(0, lambda: self.status.set(f"自动抓拍规划 {index}/{total}：小车正在移动…"))
        deadline = time.monotonic() + max(30.0, 8.0 + math.hypot(x, y) / max(linear, 0.02) * 4.0)
        while time.monotonic() < deadline:
            if not self.routine_running or self.auto_capture_plan_cancelled or self.vehicle_route_cancelled:
                self.tracer.stop_all()
                return False
            _pose, _origin, _source, newest, state, _name = self.tracer.snapshot()
            if time.monotonic() - newest > 1.0:
                self.tracer.stop_all()
                self.after(0, lambda: self.status.set("自动抓拍规划：小车定位超时，已停止。"))
                return False
            if state == "idle":
                self.vehicle_route_running = False
                self._set_auto_capture_phase("小车已到位：继续下一步骤")
                return True
            time.sleep(0.1)
        self.tracer.stop_all()
        self.after(0, lambda: self.status.set("自动抓拍规划：小车到位超时，已停止。"))
        return False

    def _refresh_vehicle_records(self):
        if not hasattr(self, "vehicle_record_tree"):
            return
        self.vehicle_record_tree.delete(*self.vehicle_record_tree.get_children())
        for index, item in enumerate(self.vehicle_motion_records):
            self.vehicle_record_tree.insert(
                "", "end", iid=str(index),
                values=(f"点 {index + 1}", item["type"], f'{item["x"]:.2f}', f'{item["y"]:.2f}', f'{item["yaw"]:.2f}'),
            )

    @staticmethod
    def _load_vehicle_motion_records():
        """Load saved vehicle waypoints; ignore malformed legacy entries."""
        try:
            with open(VEHICLE_MOTION_RECORDS_PATH, "r", encoding="utf-8") as handle:
                raw = json.load(handle)
            records = []
            for item in raw if isinstance(raw, list) else []:
                kind = item.get("type")
                if kind not in ("相对", "绝对"):
                    continue
                records.append({
                    "type": kind,
                    "x": float(item["x"]),
                    "y": float(item["y"]),
                    "yaw": float(item["yaw"]),
                })
            return records
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            return []

    def _save_vehicle_motion_records(self):
        try:
            with open(VEHICLE_MOTION_RECORDS_PATH, "w", encoding="utf-8") as handle:
                json.dump(self.vehicle_motion_records, handle, ensure_ascii=False, indent=2)
            self.vehicle_status.set(f"Tracer5：已保存 {len(self.vehicle_motion_records)} 个路径点，下次启动会自动载入。")
            self.status.set(f"小车路径记录已保存：{VEHICLE_MOTION_RECORDS_PATH}")
        except OSError as error:
            messagebox.showerror("保存小车记录失败", str(error), parent=self)

    def _clear_vehicle_motion_records(self):
        if not self.vehicle_motion_records:
            return
        if not messagebox.askyesno("确认删除", "确认删除全部小车路径点？保存文件也会同步清空。\n此操作不会让小车或机械臂运动。", parent=self):
            return
        self.vehicle_motion_records.clear()
        self._save_vehicle_motion_records()
        self._refresh_vehicle_records()

    def _add_vehicle_record(self):
        """Append a manually entered relative/absolute waypoint."""
        self.vehicle_motion_records.append({
            "type": self.vehicle_record_kind.get(),
            "x": float(self.vehicle_record_x.get()),
            "y": float(self.vehicle_record_y.get()),
            "yaw": float(self.vehicle_record_yaw.get()),
        })
        self._save_vehicle_motion_records()
        self._refresh_vehicle_records()
        last = len(self.vehicle_motion_records) - 1
        self.vehicle_record_tree.selection_set(str(last))
        self.vehicle_record_tree.focus(str(last))

    def _move_vehicle_record(self, direction):
        selection = self.vehicle_record_tree.selection()
        if not selection:
            return messagebox.showinfo("请选择路径点", "请先在列表中选择要调整顺序的路径点。", parent=self)
        old_index = int(selection[0])
        new_index = max(0, min(len(self.vehicle_motion_records) - 1, old_index + direction))
        if old_index == new_index:
            return
        item = self.vehicle_motion_records.pop(old_index)
        self.vehicle_motion_records.insert(new_index, item)
        self._save_vehicle_motion_records()
        self._refresh_vehicle_records()
        self.vehicle_record_tree.selection_set(str(new_index))
        self.vehicle_record_tree.focus(str(new_index))

    def _delete_vehicle_record(self):
        selection = self.vehicle_record_tree.selection()
        if not selection:
            return
        index = int(selection[0])
        self.vehicle_motion_records.pop(index)
        self._save_vehicle_motion_records()
        self._refresh_vehicle_records()

    def _run_vehicle_records(self):
        """Run manually entered waypoints in their displayed order."""
        if not self.vehicle_motion_records:
            return messagebox.showwarning("没有路径点", "请先输入 X、Y、Yaw 并添加至少一个路径点。", parent=self)
        if self.vehicle_route_running:
            return messagebox.showinfo("路径运行中", "正在依次运行已记录的路径点；可用“小车即停”中断。", parent=self)
        pose, origin, _source, stamp, _state, _name = self.tracer.snapshot()
        if pose is None or origin is None or time.monotonic() - stamp > 1.0:
            return messagebox.showwarning("小车未定位", "尚未收到有效的 Tracer5 融合定位，不能运行路径。", parent=self)
        self.vehicle_route_running = True
        self.vehicle_route_cancelled = False
        records = [dict(item) for item in self.vehicle_motion_records]
        linear, angular = float(self.vehicle_v.get()), float(self.vehicle_w.get())
        threading.Thread(
            target=self._run_vehicle_records_worker,
            args=(records, linear, angular), daemon=True,
        ).start()

    def _run_vehicle_auto_capture(self):
        """At every vehicle location, run the saved arm/ZED views and capture."""
        if not self.vehicle_motion_records:
            return messagebox.showwarning("没有路径点", "请先添加至少一个小车路径点。", parent=self)
        if not self.motion_records:
            return messagebox.showwarning("没有相机位姿", "请先在“规则运动（记录位置）”中记录至少一个机械臂相机位姿。", parent=self)
        if self.vehicle_auto_capture_running or self.vehicle_route_running or self.routine_running or self.cartesian_busy:
            active = []
            if self.vehicle_auto_capture_running:
                active.append("小车自动抓拍")
            if self.vehicle_route_running:
                active.append("小车路径")
            if self.routine_running:
                active.append("机械臂/自动规划")
            if self.cartesian_busy:
                active.append("机械臂末端目标")
            return messagebox.showwarning(
                "任务进行中",
                "当前仍被占用：" + "、".join(active) + "。\n请等待完成；如确认任务已异常结束，可使用全局“结束上一段运动（停止）”清除状态。",
                parent=self,
            )
        pose, _origin, _source, stamp, _state, _name = self.tracer.snapshot()
        if pose is None or time.monotonic() - stamp > 1.0:
            return messagebox.showwarning("小车未定位", "尚未收到有效 Tracer5 融合定位，不能开始自动抓拍。", parent=self)
        if not self.can_ready or not self.current_flange:
            return messagebox.showwarning("CAN 未就绪", "未收到有效机械臂反馈，不能执行机械臂/ZED 自动抓拍。", parent=self)
        if not self.safety_ok.get():
            return messagebox.showwarning("需要安全确认", "请先勾选底部的机械臂安全确认。", parent=self)
        if not messagebox.askyesno(
            "确认小车自动抓拍",
            f"将在当前位置及点 1 到点 {len(self.vehicle_motion_records)} 的每个小车位置，\n"
            f"依次运行 {len(self.motion_records)} 个机械臂相机位姿；每个相机位姿到位后由 ZED 自动抓拍。\n"
            "YOLO/SAM2、保存和点云融合均自动执行；可用“小车即停”或全局停止中断。\n\n确认现场安全？",
            parent=self,
        ):
            return
        self.vehicle_auto_capture_running = True
        self.vehicle_route_cancelled = False
        self.auto_capture_plan_cancelled = False
        # Reuse the existing tested ZED/Yolo/SAM2 capture path.  That path
        # uses routine_running only as a cooperative cancel flag.
        self.routine_running = True
        self.routine_auto_capture = True
        self.auto_capture_failed = False
        self._set_capture_detection_options()
        records = [dict(item) for item in self.vehicle_motion_records]
        arm_records = [{"name": item["name"], "pose": list(item["pose"])} for item in self.motion_records]
        speed = (float(self.vehicle_v.get()), float(self.vehicle_w.get()))
        self.status.set("小车-机械臂联动抓拍已启动：正在准备远程 ZED…")
        threading.Thread(
            target=self._run_vehicle_auto_capture_worker, args=(records, arm_records, speed), daemon=True
        ).start()

    def _capture_arm_views_at_vehicle_location(self, arm_records, location_name, capture_index):
        """Move the arm/ZED through every saved view at one stationary car pose."""
        for arm_index, record in enumerate(arm_records, start=1):
            if not self.routine_running or self.vehicle_route_cancelled or self.auto_capture_plan_cancelled:
                return False, capture_index
            target = list(record["pose"])
            self._set_auto_capture_phase(f"机械臂运动：{location_name} 相机位 {arm_index}/{len(arm_records)}")
            self.after(0, lambda a=arm_index, n=len(arm_records), place=location_name: self.status.set(
                f"{place}：机械臂正在前往相机位姿 {a}/{n}…"
            ))
            result = self.run_remote(self._cartesian_command(target))
            if result.returncode:
                self.after(0, lambda place=location_name, a=arm_index: self.status.set(
                    f"{place}：机械臂相机位姿 {a} 未确认到位，已停止。"
                ))
                return False, capture_index
            deadline = time.monotonic() + self._capture_settle_s_active
            while self.routine_running and not self.vehicle_route_cancelled and time.monotonic() < deadline:
                self._set_auto_capture_phase("机械臂到位：抓拍停顿计时")
                self._set_capture_countdown("抓拍停顿", deadline - time.monotonic(), self._capture_settle_s_active)
                self._refresh_auto_camera(f"{location_name}: arm view {arm_index} stable")
                time.sleep(0.1)
            if not self.routine_running or self.vehicle_route_cancelled:
                return False, capture_index
            capture_index += 1
            self.after(0, lambda c=capture_index, place=location_name: self.status.set(
                f"{place}：ZED 正在抓拍第 {c} 帧…"
            ))
            if not self._capture_auto_frame(capture_index):
                return False, capture_index
        return True, capture_index

    def _run_vehicle_auto_capture_worker(self, records, arm_records, vehicle_speed):
        try:
            if not self._prepare_auto_capture():
                return
            capture_index = 0
            ok, capture_index = self._capture_arm_views_at_vehicle_location(
                arm_records, "当前位置", capture_index
            )
            if not ok:
                return
            for index, record in enumerate(records, start=1):
                if not self.routine_running or self.vehicle_route_cancelled:
                    self.after(0, lambda: self.status.set("小车-机械臂联动抓拍已停止。"))
                    return
                if not self._run_auto_capture_vehicle_step(record, index, len(records), *vehicle_speed):
                    return
                deadline = time.monotonic() + 0.5
                while self.routine_running and not self.vehicle_route_cancelled and time.monotonic() < deadline:
                    self._refresh_auto_camera(f"vehicle point {index}: settling")
                    time.sleep(0.1)
                if not self.routine_running or self.vehicle_route_cancelled:
                    return
                ok, capture_index = self._capture_arm_views_at_vehicle_location(
                    arm_records, f"小车点 {index}", capture_index
                )
                if not ok:
                    return
            self.after(0, lambda: self.status.set("小车-机械臂联动抓拍完成：正在恢复机械臂初始零位…"))
            args = " ".join(f"{value:.3f}" for value in HOME_JOINTS_DEG)
            home = (
                f"cd {REMOTE_DIR} && python3 piper_safe_move.py --can can1 "
                f"--joint-deg {args} --speed {self._joint_speed_level()} --execute --confirm ARM_CLEAR"
            )
            result = self.run_remote(home)
            if result.returncode or not self._wait_for_home_joint_pose(timeout_s=70.0):
                self.after(0, lambda: self.status.set("抓拍已完成，但机械臂复位未确认；ZED 将保持开启。"))
                return
            self.after(0, lambda n=capture_index: self.status.set(
                f"小车-机械臂联动抓拍完成：已保存 {n} 帧，点云继续后台融合；机械臂已回到初始零位。"
            ))
        except Exception as error:
            self.auto_capture_failed = True
            self.after(0, lambda: self.status.set(f"小车自动抓拍失败：{error}"))
        finally:
            self.vehicle_route_running = False
            self.vehicle_auto_capture_running = False
            self.routine_auto_capture = False
            self.routine_running = False
            if not self.auto_capture_failed:
                self._close_auto_capture_views()
            self.after(0, self.refresh_status)

    def _run_vehicle_records_worker(self, records, linear, angular):
        """Wait for TracerRos's closed-loop target controller per waypoint."""
        from tracer5_control_panel import Pose2D, wrap_angle
        try:
            total = len(records)
            for index, item in enumerate(records, start=1):
                if self.vehicle_route_cancelled:
                    self.after(0, lambda: self.vehicle_status.set("Tracer5：已停止依次运行。"))
                    return
                pose, origin, _source, stamp, _state, _name = self.tracer.snapshot()
                if pose is None or origin is None or time.monotonic() - stamp > 1.0:
                    self.after(0, lambda i=index: self.vehicle_status.set(f"Tracer5：点 {i} 未启动，定位数据不可用。"))
                    return
                x, y = float(item["x"]) / 100.0, float(item["y"]) / 100.0
                if item["type"] == "相对":
                    target = Pose2D(
                        pose.x + math.cos(pose.yaw) * x - math.sin(pose.yaw) * y,
                        pose.y + math.sin(pose.yaw) * x + math.cos(pose.yaw) * y,
                        wrap_angle(pose.yaw + math.radians(float(item["yaw"]))),
                    )
                    straight = abs(y) < 1e-6 and abs(float(item["yaw"])) < 1e-6
                else:
                    target = Pose2D(
                        origin.x + math.cos(origin.yaw) * x - math.sin(origin.yaw) * y,
                        origin.y + math.sin(origin.yaw) * x + math.cos(origin.yaw) * y,
                        wrap_angle(origin.yaw + math.radians(float(item["yaw"]))),
                    )
                    straight = False
                ok, reason = self.tracer.start_target(target, f"路径点 {index}/{total}", linear, angular, straight=straight)
                if not ok:
                    self.after(0, lambda i=index, r=reason: self.vehicle_status.set(f"Tracer5：点 {i} 未运行：{r}"))
                    return
                self.after(0, lambda i=index, n=total: self.vehicle_status.set(f"Tracer5：正在运行路径点 {i}/{n}…"))
                # Generous timeout: this is only a safety stop if odometry or
                # drive feedback ceases; normal completion is decided by TracerRos.
                deadline = time.monotonic() + max(30.0, 8.0 + math.hypot(x, y) / max(linear, 0.02) * 4.0)
                while time.monotonic() < deadline:
                    if self.vehicle_route_cancelled:
                        self.tracer.stop_all()
                        self.after(0, lambda: self.vehicle_status.set("Tracer5：已停止依次运行。"))
                        return
                    _pose, _origin, _source, newest, state, _name = self.tracer.snapshot()
                    if time.monotonic() - newest > 1.0:
                        self.tracer.stop_all()
                        self.after(0, lambda i=index: self.vehicle_status.set(f"Tracer5：点 {i} 定位超时，已停止。"))
                        return
                    if state == "idle":
                        break
                    time.sleep(0.10)
                else:
                    self.tracer.stop_all()
                    self.after(0, lambda i=index: self.vehicle_status.set(f"Tracer5：点 {i} 超时，已停止。"))
                    return
            self.after(0, lambda: self.vehicle_status.set(f"Tracer5：全部 {len(records)} 个路径点已完成。"))
        finally:
            self.vehicle_route_running = False

    def _vehicle_manual(self, linear_sign, angular_sign):
        self.tracer.set_manual(float(linear_sign) * self.vehicle_v.get(), float(angular_sign) * self.vehicle_w.get())

    def _vehicle_stop(self):
        self.vehicle_route_cancelled = True
        self.auto_capture_plan_cancelled = True
        self.tracer.stop_all()

    def _apply_vehicle_tolerances(self):
        self.tracer.set_tolerances(self.vehicle_position_tolerance.get(), self.vehicle_yaw_tolerance.get())
        self.vehicle_status.set(f"Tracer5：到达误差已设为 {self.vehicle_position_tolerance.get():.1f} cm / {self.vehicle_yaw_tolerance.get():.1f}°")

    @staticmethod
    def _load_control_settings():
        """Load benign UI defaults; a missing/bad file simply uses defaults."""
        try:
            with open(CONTROL_SETTINGS_PATH, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            position = float(data.get("vehicle_position_tolerance_cm", 0.5))
            yaw = float(data.get("vehicle_yaw_tolerance_deg", 0.5))
            wait_s = float(data.get("no_target_wait_s", 0.2))
            confidence_pct = float(data.get("min_target_confidence_pct", 50.0))
            settle_s = float(data.get("capture_settle_s", 2.0))
            return {
                "vehicle_position_tolerance_cm": min(20.0, max(0.1, position)),
                "vehicle_yaw_tolerance_deg": min(20.0, max(0.1, yaw)),
                "no_target_wait_s": min(10.0, max(0.2, wait_s)),
                "min_target_confidence_pct": min(100.0, max(1.0, confidence_pct)),
                "capture_settle_s": min(10.0, max(0.0, settle_s)),
            }
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return {}

    def _save_vehicle_tolerances(self):
        """Persist the displayed tolerances and immediately apply them."""
        try:
            position = min(20.0, max(0.1, float(self.vehicle_position_tolerance.get())))
            yaw = min(20.0, max(0.1, float(self.vehicle_yaw_tolerance.get())))
            self.vehicle_position_tolerance.set(position)
            self.vehicle_yaw_tolerance.set(yaw)
            with open(CONTROL_SETTINGS_PATH, "w", encoding="utf-8") as handle:
                settings = self._load_control_settings()
                settings.update({
                    "vehicle_position_tolerance_cm": position,
                    "vehicle_yaw_tolerance_deg": yaw,
                })
                json.dump(settings, handle, ensure_ascii=False, indent=2)
            self._apply_vehicle_tolerances()
            self.vehicle_status.set(
                f"Tracer5：已保存误差 {position:.1f} cm / {yaw:.1f}°；以后启动会自动使用。"
            )
        except (OSError, ValueError, TypeError) as error:
            messagebox.showerror("保存误差失败", str(error), parent=self)

    def _vehicle_relative(self):
        pose, *_ = self.tracer.snapshot()
        if pose is None:
            return messagebox.showwarning("小车未定位", "尚未收到 tracer5 的 IMU 融合定位。", parent=self)
        x, y = self.vehicle_rx.get() / 100.0, self.vehicle_ry.get() / 100.0
        from tracer5_control_panel import Pose2D, wrap_angle
        target = Pose2D(pose.x + math.cos(pose.yaw) * x - math.sin(pose.yaw) * y,
                        pose.y + math.sin(pose.yaw) * x + math.cos(pose.yaw) * y,
                        wrap_angle(pose.yaw + math.radians(self.vehicle_ryaw.get())))
        self.tracer.start_target(target, "相对目标", self.vehicle_v.get(), self.vehicle_w.get(), straight=abs(y) < 1e-6 and abs(self.vehicle_ryaw.get()) < 1e-6)

    def _vehicle_absolute(self):
        pose, origin, *_ = self.tracer.snapshot()
        if pose is None or origin is None:
            return messagebox.showwarning("小车未定位", "尚未收到 tracer5 的 IMU 融合定位。", parent=self)
        from tracer5_control_panel import Pose2D, wrap_angle
        x, y = self.vehicle_ax.get() / 100.0, self.vehicle_ay.get() / 100.0
        yaw = pose.yaw if self.vehicle_keep_yaw.get() else wrap_angle(origin.yaw + math.radians(self.vehicle_ayaw.get()))
        target = Pose2D(origin.x + math.cos(origin.yaw) * x - math.sin(origin.yaw) * y,
                        origin.y + math.sin(origin.yaw) * x + math.cos(origin.yaw) * y, yaw)
        self.tracer.start_target(target, "绝对目标", self.vehicle_v.get(), self.vehicle_w.get())

    def _vehicle_record(self):
        if self.vehicle_recording:
            self.vehicle_recording = False
            count = self.tracer.end_recording()
            self.vehicle_status.set(f"Tracer5：已记录 {count} 段手动运动")
        else:
            self.vehicle_recording = self.tracer.begin_recording()
            self.vehicle_status.set("Tracer5：正在记录手动运动")

    def _vehicle_replay(self):
        ok, reason = self.tracer.start_plan(self.vehicle_v.get() / 0.10, self.vehicle_w.get() / 0.55)
        if not ok:
            messagebox.showwarning("无法回放", reason, parent=self)

    def _refresh_vehicle_panel(self):
        pose, origin, source, stamp, state, name = self.tracer.snapshot()
        if pose is None or time.monotonic() - stamp > 1.0:
            error = getattr(self.tracer, "error", "")
            self.vehicle_status.set(
                f"Tracer5：远程定位不可用：{error}" if error
                else "Tracer5：等待远程 IMU 融合定位…"
            )
            self.vehicle_pose.set("Tracer5：当前坐标不可用（等待融合定位）")
            return
        if origin:
            wx, wy = pose.x - origin.x, pose.y - origin.y
            x = math.cos(origin.yaw) * wx + math.sin(origin.yaw) * wy
            y = -math.sin(origin.yaw) * wx + math.cos(origin.yaw) * wy
            self.vehicle_pose.set(f"Tracer5（{source}）：X={x*100:.1f} cm，Y={y*100:.1f} cm，Yaw={math.degrees((pose.yaw-origin.yaw+math.pi)%(2*math.pi)-math.pi):.1f}°")
        activity = "手动待命" if state == "idle" else f"正在执行{name}"
        self.vehicle_status.set(f"Tracer5：{activity}，命令 /tracer5/cmd_vel")

    def _build_discrete_cartesian_page(self, page):
        """One-shot absolute target editor; motion starts only after Send."""
        for column in range(6):
            page.columnconfigure(column, weight=1)
        ttk.Label(
            page, text="标定 TCP 的绝对目标（相对机械臂底座）", font=("Sans", 12, "bold")
        ).grid(row=0, column=0, columnspan=6, sticky="w", pady=(0, 12))
        for index, name in enumerate(("X", "Y", "Z", "Roll", "Pitch", "Yaw")):
            column = index % 3 * 2
            row = index // 3 + 1
            ttk.Label(page, text=name, font=("Sans", 12)).grid(row=row, column=column, padx=(3, 8), pady=9, sticky="e")
            field = ttk.Frame(page)
            field.grid(row=row, column=column + 1, padx=(1, 18), pady=9, sticky="ew")
            field.columnconfigure(0, weight=1)
            ttk.Entry(field, textvariable=self.cartesian_inputs[index], width=14, font=("Sans", 13)).grid(row=0, column=0, sticky="ew")
        ttk.Button(page, text="取当前末端值", command=self.sync_cartesian_target).grid(row=3, column=0, columnspan=3, sticky="ew", ipady=8, padx=(0, 7), pady=(16, 0))
        ttk.Button(page, text="发送末端低速目标（当前速度）", command=self.send_cartesian_target).grid(row=3, column=3, columnspan=3, sticky="ew", ipady=8, padx=(7, 0), pady=(16, 0))
        ttk.Button(page, text="记录当前目标到规则运动", command=self.record_current_target).grid(row=4, column=0, columnspan=6, sticky="ew", ipady=8, pady=(10, 0))

    def _build_continuous_cartesian_page(self, page):
        """Direct nudge UI. Scale only moves after the mouse is released."""
        ttk.Label(page, text="按 +/- 立即移动：XYZ 每次 10 mm；Roll/Pitch/Yaw 每次 1°。滑块松开后按当前位置直接移动。", foreground="#555555").grid(row=0, column=0, columnspan=6, sticky="w", pady=(0, 4))
        for index, name in enumerate(("X", "Y", "Z", "Roll", "Pitch", "Yaw")):
            row = index + 1
            low, high = CARTESIAN_SLIDER_LIMITS[index]
            ttk.Label(page, text=name, width=6).grid(row=row, column=0, sticky="e", padx=(2, 4), pady=2)
            ttk.Button(page, text="−", width=2, command=lambda i=index: self.adjust_cartesian(i, -1, direct=True)).grid(row=row, column=1)
            slider = ttk.Scale(page, from_=low, to=high, length=340, command=lambda value, i=index: self.continuous_slider_changed(i, value))
            slider.grid(row=row, column=2, padx=5)
            slider.bind("<ButtonRelease-1>", lambda _event, i=index: self.continuous_slider_released(i))
            self.continuous_sliders.append(slider)
            ttk.Entry(page, textvariable=self.cartesian_inputs[index], width=9).grid(row=row, column=3, padx=3)
            ttk.Button(page, text="+", width=2, command=lambda i=index: self.adjust_cartesian(i, 1, direct=True)).grid(row=row, column=4)
            ttk.Label(page, text=f"{low:.0f} ～ {high:.0f}", width=12, foreground="#666666").grid(row=row, column=5, sticky="w")
        ttk.Button(page, text="以真实末端值重置连续控制", command=self.sync_cartesian_target).grid(row=7, column=0, columnspan=3, sticky="ew", padx=(0, 3), pady=(5, 0))
        ttk.Button(page, text="记录当前目标到规则运动", command=self.record_current_target).grid(row=7, column=3, columnspan=3, sticky="ew", padx=(3, 0), pady=(5, 0))

    def _build_routine_page(self, page):
        """Saved Cartesian TCP targets. Nothing moves from this page by itself."""
        ttk.Label(
            page,
            text="在“间断运动”页设置绝对 TCP 目标后点“记录当前目标”。选中一条后才可运行，且仍会要求安全确认。",
            foreground="#555555",
            wraplength=700,
        ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 7))
        columns = ("name", "x", "y", "z", "roll", "pitch", "yaw")
        self.routine_tree = ttk.Treeview(page, columns=columns, show="headings", height=5, selectmode="browse")
        labels = ("记录", "X mm", "Y mm", "Z mm", "Roll °", "Pitch °", "Yaw °")
        widths = (90, 105, 105, 105, 105, 105, 105)
        for column, label, width in zip(columns, labels, widths):
            self.routine_tree.heading(column, text=label)
            self.routine_tree.column(column, width=width, anchor="center", stretch=False)
        self.routine_tree.grid(row=1, column=0, columnspan=3, sticky="nsew")
        self.routine_tree.bind("<<TreeviewSelect>>", self.load_selected_record)
        self._refresh_routine_tree()
        ttk.Button(page, text="运行选中记录（当前速度）", command=self.run_selected_record).grid(row=2, column=0, sticky="ew", padx=(0, 4), pady=(8, 0))
        ttk.Button(page, text="自动依次运行全部（当前速度）", command=self.run_all_records).grid(row=2, column=1, sticky="ew", padx=4, pady=(8, 0))
        ttk.Button(page, text="删除选中记录", command=self.delete_selected_record).grid(row=2, column=2, sticky="ew", padx=(4, 0), pady=(8, 0))
        ttk.Button(page, text="上移选中位置", command=lambda: self.move_selected_record(-1)).grid(row=3, column=0, sticky="ew", padx=(0, 4), pady=(5, 0))
        ttk.Button(page, text="下移选中位置", command=lambda: self.move_selected_record(1)).grid(row=3, column=1, sticky="ew", padx=4, pady=(5, 0))
        ttk.Button(page, text="清空全部记录", command=self.clear_motion_records).grid(row=3, column=2, sticky="ew", padx=(4, 0), pady=(5, 0))
        ttk.Button(page, text="自动抓拍 + 依次运行（每点到达后 2 秒）", command=self.auto_capture_button_clicked).grid(
            row=4, column=0, columnspan=3, sticky="ew", pady=(8, 0)
        )

    def _load_motion_records(self):
        try:
            with open(MOTION_RECORDS_PATH, "r", encoding="utf-8") as handle:
                records = json.load(handle)
            if not isinstance(records, list):
                raise ValueError("records is not a list")
            cleaned = []
            for record in records:
                pose = record.get("pose") if isinstance(record, dict) else None
                if isinstance(pose, list) and len(pose) == 6:
                    cleaned.append({"name": str(record.get("name", "记录")), "pose": [float(value) for value in pose]})
            # The list order is the execution order.  Old files could retain
            # a deleted record's name (for example "位置 2" as the first
            # row), so normalise names whenever they are loaded.
            self._renumber_motion_records(cleaned)
            return cleaned
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return []

    def _save_motion_records(self):
        with open(MOTION_RECORDS_PATH, "w", encoding="utf-8") as handle:
            json.dump(self.motion_records, handle, ensure_ascii=False, indent=2)

    @staticmethod
    def _renumber_motion_records(records):
        """Keep visible names identical to the actual execution order."""
        for number, record in enumerate(records, 1):
            record["name"] = f"位置 {number}"

    def _refresh_routine_tree(self):
        if not self.routine_tree:
            return
        for item in self.routine_tree.get_children():
            self.routine_tree.delete(item)
        for index, record in enumerate(self.motion_records):
            values = [record["name"]] + [f"{value:.2f}" for value in record["pose"]]
            self.routine_tree.insert("", "end", iid=str(index), values=values)

    def record_current_target(self):
        try:
            pose = [float(value.get()) for value in self.cartesian_inputs]
        except ValueError:
            messagebox.showerror("输入错误", "当前绝对末端目标不是有效数字，无法记录。")
            return
        if not CARTESIAN_Z_MIN_MM <= pose[2] <= CARTESIAN_Z_MAX_MM:
            messagebox.showwarning("Z 轴超出范围", f"记录的 Z 必须在 {CARTESIAN_Z_MIN_MM:.0f}～{CARTESIAN_Z_MAX_MM:.0f} mm 内。")
            return
        # Append is always the next execution position; existing names are
        # normalised first in case this is an older saved file.
        self._renumber_motion_records(self.motion_records)
        record = {"name": f"位置 {len(self.motion_records) + 1}", "pose": pose}
        self.motion_records.append(record)
        self._save_motion_records()
        self._refresh_routine_tree()
        self.status.set(f"已记录 {record['name']}；不会自动运动。")

    def _selected_record_index(self):
        if not self.routine_tree:
            return None
        selected = self.routine_tree.selection()
        if not selected:
            messagebox.showwarning("未选择记录", "请先在规则运动页选择一条记录。")
            return None
        return int(selected[0])

    def load_selected_record(self, _event=None):
        """Selecting a row always loads that exact row's stored TCP target."""
        index = self._selected_record_index()
        if index is None:
            return
        record = self.motion_records[index]
        for target, value in zip(self.cartesian_inputs, record["pose"]):
            target.set(f"{value:.2f}")
        for slider, value, limits in zip(self.continuous_sliders, record["pose"], CARTESIAN_SLIDER_LIMITS):
            slider.set(min(limits[1], max(limits[0], value)))
        self.status.set(f"已载入 {record['name']}：点击“运行选中记录”才会运动。")

    def move_selected_record(self, direction):
        index = self._selected_record_index()
        if index is None:
            return
        target_index = index + direction
        if not 0 <= target_index < len(self.motion_records):
            return
        self.motion_records[index], self.motion_records[target_index] = (
            self.motion_records[target_index], self.motion_records[index]
        )
        # Names represent execution order, so make the visible numbering
        # match the order used by the automatic sequence.
        self._renumber_motion_records(self.motion_records)
        self._save_motion_records()
        self._refresh_routine_tree()
        self.routine_tree.selection_set(str(target_index))
        self.routine_tree.focus(str(target_index))
        self.status.set(f"已调整顺序；自动运行将按位置 1 → {len(self.motion_records)} 执行。")

    def run_selected_record(self):
        index = self._selected_record_index()
        if index is None:
            return
        record = self.motion_records[index]
        self.load_selected_record()
        self.status.set(f"已载入 {record['name']}；等待安全确认后发送。")
        self.send_cartesian_target(direct=False)

    def run_all_records(self):
        """Run saved TCP targets in list order, waiting for feedback per step."""
        if not self.motion_records:
            messagebox.showwarning("尚无记录", "请先记录至少一个末端目标。", parent=self)
            return
        if self.routine_running or self.cartesian_busy:
            messagebox.showwarning("运动进行中", "请等待当前末端运动结束，或使用全局停止按钮。")
            return
        if not self.can_ready or not self.current_flange:
            messagebox.showwarning("CAN 未就绪", "未收到有效机械臂反馈，已禁止自动运行。")
            return
        if not self.safety_ok.get():
            messagebox.showwarning("需要安全确认", "请先勾选安全确认。")
            return
        if not messagebox.askyesno(
            "确认自动依次运行",
            f"将按列表顺序以当前 {self._configured_speed()}% 速度运行 {len(self.motion_records)} 个记录位置。\n"
            "每个位置确认到位后才会执行下一条；可用全局停止按钮中断。\n\n确认现场安全？",
        ):
            return
        self.routine_running = True
        threading.Thread(target=self._run_all_records_worker, daemon=True).start()

    def auto_capture_button_clicked(self):
        """GUI boundary: never leave a button click looking like a no-op."""
        self.status.set("已收到自动抓拍按钮：正在检查安全条件…")
        self.update_idletasks()
        try:
            self.run_all_with_auto_capture()
        except Exception as error:
            self.status.set(f"自动抓拍按钮异常：{error}")
            messagebox.showerror("自动抓拍启动失败", str(error), parent=self)

    def run_all_with_auto_capture(self):
        """Safely run every recorded view and capture one automatic frame there."""
        if not self.motion_records:
            self.status.set("自动抓拍未启动：没有规则运动记录。")
            messagebox.showwarning("尚无记录", "请先记录至少一个末端目标。")
            return
        if self.routine_running or self.cartesian_busy:
            self.status.set("自动抓拍未启动：已有机械臂运动正在进行。")
            messagebox.showwarning("运动进行中", "请等待当前运动结束，或使用全局停止按钮。", parent=self)
            return
        if not self.can_ready or not self.current_flange:
            self.status.set("自动抓拍未启动：CAN 或标定 TCP 反馈尚未就绪。")
            messagebox.showwarning("CAN 未就绪", "未收到有效机械臂反馈，已禁止自动运行。", parent=self)
            return
        if not self.safety_ok.get():
            self.status.set("自动抓拍未启动：请先勾选底部的安全确认。")
            messagebox.showwarning("需要安全确认", "请先勾选安全确认。", parent=self)
            return
        if not messagebox.askyesno(
            "确认自动抓拍规则运动",
            f"将启动远程 ZED，按位置 1 到 {len(self.motion_records)} 逐点低速运动。\n"
            "每点确认到位后按全局“抓拍停顿”设置等待；随后立即 YOLO + SAM2、保存并融合点云，再前往下一位置。\n"
            "全部成功后会自动回到机械臂零位，并关闭自动相机/点云显示窗口；文件会保留。\n"
            "识别、深度或点云失败会立即停止后续运动；全程可用全局停止按钮中断。\n\n"
            "确认现场安全？",
            parent=self,
        ):
            return
        self.routine_running = True
        self.routine_auto_capture = True
        self.auto_capture_failed = False
        self._set_capture_detection_options()
        self.status.set("自动抓拍已启动：正在加载 YOLO/SAM2，并准备远程 ZED 画面…")
        self.update_idletasks()
        threading.Thread(target=self._run_all_with_auto_capture_worker, daemon=True).start()

    def _prepare_auto_capture(self):
        """Start the established remote bridge and wait until it delivers RGB+depth."""
        if not self._load_capture_models():
            return False
        self.after(0, lambda: self.status.set("正在启动远程 ZED，等待 RGB + 深度画面…"))
        result = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", REMOTE_HOST,
             "bash ~/zed_code/scripts/start_remote_zed_live.sh && bash ~/zed_code/scripts/start_zed_tcp_bridge.sh"],
            text=True, capture_output=True, timeout=30,
        )
        if result.returncode:
            detail = (result.stderr or result.stdout).strip()[-500:]
            self.after(0, lambda: self.status.set(f"远程 ZED 启动失败：{detail}"))
            return False
        # The standalone capture program may already own this local forward.
        # Reuse a live port instead of trying to bind 47778 a second time.
        try:
            with socket.create_connection(("127.0.0.1", 47778), timeout=1.0):
                tunnel_ready = True
        except OSError:
            tunnel_ready = False
        if not tunnel_ready and (self.camera_tunnel is None or self.camera_tunnel.poll() is not None):
            self.camera_tunnel = subprocess.Popen(
                ["ssh", "-N", "-o", "BatchMode=yes", "-o", "ExitOnForwardFailure=yes",
                 "-L", "127.0.0.1:47778:127.0.0.1:47777", REMOTE_HOST],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            time.sleep(0.8)
        if not tunnel_ready:
            try:
                with socket.create_connection(("127.0.0.1", 47778), timeout=1.0):
                    tunnel_ready = True
            except OSError:
                tunnel_ready = False
        if not tunnel_ready:
            self.after(0, lambda: self.status.set("远程 ZED SSH 隧道启动失败。"))
            return False
        self.remote_camera = self.capture_module.RemoteCapture(port=47778)
        self.auto_camera_enabled = True
        scan = self._ensure_capture_scan()
        deadline = time.monotonic() + 25.0
        while self.routine_running and time.monotonic() < deadline:
            rgb, depth = self.remote_camera.live_frame()
            if rgb is not None and depth is not None:
                self.auto_camera_window_drawn.clear()
                self._show_auto_camera(rgb, depth, "等待机械臂运行第一个记录位置")
                # Give the Tk main loop a short chance to paint the embedded
                # preview before the first arm movement starts.
                if not self.auto_camera_window_drawn.wait(3.0):
                    self.after(0, lambda: self.status.set(
                        "已收到远程 RGB+深度，但主窗口预览尚未绘制；请检查本机图形桌面。"
                    ))
                return True
            time.sleep(0.5)
        self.after(0, lambda: self.status.set("远程 ZED 画面或深度未在 25 秒内就绪，已停止。"))
        return False

    def _start_auto_cloud_viewer(self, scan):
        """Open the rotatable Open3D window only once fused data exists."""
        viewer = ONLINE_SCAN_ROOT / "live_fusion_viewer.py"
        fused = Path(scan) / "fusion" / "fused_icp.ply"
        if (not viewer.is_file() or not fused.is_file() or
                (self.auto_cloud_process is not None and self.auto_cloud_process.poll() is None)):
            return
        log_path = Path(scan) / "fusion" / "live_viewer.log"
        try:
            log = open(log_path, "a", encoding="utf-8")
            self.auto_cloud_process = subprocess.Popen(
                [sys.executable, str(viewer), str(scan)], stdout=log, stderr=subprocess.STDOUT
            )
            self.status.set("首帧点云已生成：已打开可旋转/缩放的实时 3D 点云窗口。")
        except OSError as error:
            self.status.set(f"实时 3D 点云窗口启动失败：{error}")

    def _show_auto_camera(self, bgr, depth, message):
        """Queue camera drawing on Tk's main thread, never on a worker."""
        if not self.auto_camera_enabled:
            return
        self.auto_camera_display_frame = (
            bgr.copy(), None if depth is None else depth.copy(), str(message)
        )
        if not self.auto_camera_display_pending:
            self.auto_camera_display_pending = True
            self.after(0, self._draw_auto_camera)

    def _draw_auto_camera(self):
        """Draw the automatic preview inside Tk; never start an OpenCV GUI."""
        self.auto_camera_display_pending = False
        if (not self.auto_camera_enabled or not self.winfo_exists() or
                self.auto_camera_display_frame is None):
            return
        bgr, depth, message = self.auto_camera_display_frame
        shown = cv2.resize(bgr, (640, 360), interpolation=cv2.INTER_AREA)
        depth_view = self.capture_module.depth_preview(depth, (640, 360))
        panel = np.hstack((shown, depth_view))
        # OpenCV's built-in Hershey font cannot render Chinese.  Keep this
        # display line ASCII so it never becomes unreadable question marks;
        # detailed Chinese state remains in the Tk control panel.
        ascii_message = str(message).encode("ascii", "ignore").decode().strip() or "processing"
        cv2.putText(panel, "AUTO: " + ascii_message, (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    .65, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(panel, "RGB                                      Depth", (12, 348),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv2.LINE_AA)
        rgb = cv2.cvtColor(panel, cv2.COLOR_BGR2RGB)
        if self.camera_label is None:
            return
        photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.camera_photo = photo  # Tk must retain a reference to the image.
        self.camera_label.configure(image=photo, text="")
        self.camera_status.set("自动规划：RGB + Depth 预览中")
        self.auto_camera_window_drawn.set()

    def _show_auto_capture_popup(self, bgr, depth, message):
        """Show one non-blocking Tk preview for the current captured frame."""
        self.after(0, lambda image=bgr.copy(), depth_image=depth.copy(), text=str(message):
                   self._draw_auto_capture_popup(image, depth_image, text))

    def _draw_auto_capture_popup(self, bgr, depth, message):
        if not self.winfo_exists():
            return
        popup = self.auto_capture_popup
        if popup is None or not popup.winfo_exists():
            popup = tk.Toplevel(self)
            popup.title("自动抓拍：当前帧（处理完成后自动关闭）")
            popup.transient(self)
            popup.geometry("980x560")
            popup.minsize(640, 380)
            popup.resizable(True, True)
            popup.label = ttk.Label(popup)
            popup.label.pack(fill="both", expand=True, padx=6, pady=6)
            self.auto_capture_popup = popup
        shown = cv2.resize(bgr, (480, 270), interpolation=cv2.INTER_AREA)
        depth_view = self.capture_module.depth_preview(depth, (480, 270))
        panel = np.hstack((shown, depth_view))
        cv2.putText(panel, str(message).encode("ascii", "ignore").decode() or "capturing",
                    (12, 28), cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 255, 255), 2, cv2.LINE_AA)
        photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB)))
        self.auto_capture_popup_photo = photo
        popup.label.configure(image=photo)
        popup.lift()

    def _close_auto_capture_popup(self):
        popup = self.auto_capture_popup
        self.auto_capture_popup = None
        self.auto_capture_popup_photo = None
        if popup is not None and popup.winfo_exists():
            popup.destroy()

    def _refresh_auto_camera(self, message):
        if not self.remote_camera:
            return
        bgr, depth = self.remote_camera.live_frame()
        if bgr is not None:
            self._show_auto_camera(bgr, depth, message)

    def _save_non_target_capture(self, bgr, _depth, sequence_index, reason):
        """Persist an empty view outside the ICP input directories."""
        scan = self._ensure_capture_scan()
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        image_dir = scan / "non_targets" / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(image_dir / f"{stamp}.jpg"), bgr)
        with (scan / "non_targets.jsonl").open("a", encoding="utf-8") as manifest:
            manifest.write(json.dumps({
                "id": stamp, "sequence_index": sequence_index,
                "reason": reason, "included_in_fusion": False,
            }, ensure_ascii=False) + "\n")
        return stamp

    def _enqueue_fusion(self, scan, stamp, sequence_index):
        """Fuse frames serially in the background so motion never waits on ICP."""
        with self.fusion_lock:
            self.fusion_pending += 1
            pending = self.fusion_pending
            if self.fusion_worker is None or not self.fusion_worker.is_alive():
                self.fusion_worker = threading.Thread(target=self._fusion_worker_loop, daemon=True)
                self.fusion_worker.start()
        self.fusion_queue.put((Path(scan), stamp, sequence_index))
        self._set_auto_capture_phase(f"点云融合后台队列：等待 {pending} 帧")
        self.after(0, lambda i=sequence_index, count=pending: self.status.set(
            f"位置 {i}：图片/掩码已保存，点云与 ICP 已转入后台队列（{count} 帧待处理）；继续下一步。"
        ))
        self.after(0, self._close_auto_capture_popup)

    def _fusion_worker_loop(self):
        while True:
            task = self.fusion_queue.get()
            if task is None:
                return
            scan, stamp, sequence_index = task
            self._set_auto_capture_phase(f"后台点云生成与 ICP 融合：第 {sequence_index} 帧")
            result = subprocess.run(
                [sys.executable, str(ONLINE_SCAN_ROOT / "online_icp_pipeline.py"), str(scan), stamp],
                text=True, capture_output=True,
            )
            if result.returncode:
                detail = (result.stderr or result.stdout).strip()[-500:]
                self.after(0, lambda i=sequence_index, message=detail: self.status.set(
                    f"位置 {i}：后台点云融合失败：{message}"
                ))
            else:
                message = self._fusion_message(scan, stamp)
                self.after(0, lambda i=sequence_index, text=message: self.status.set(
                    f"位置 {i}：后台融合完成；{text}"
                ))
                self.after(0, lambda value=scan: self._show_fused_cloud_window(value))
            with self.fusion_lock:
                self.fusion_pending = max(0, self.fusion_pending - 1)
                pending = self.fusion_pending
            self._set_auto_capture_phase(
                "后台融合空闲" if not pending else f"后台点云融合：剩余 {pending} 帧"
            )
            self.fusion_queue.task_done()

    def _capture_auto_frame(self, sequence_index):
        """Capture a target frame, or record a non-target view and continue."""
        module = self.capture_module
        self._set_auto_capture_phase("拍摄并进行 YOLO 识别")
        self._set_capture_countdown("抓拍计时", 0.0, self._capture_settle_s_active)
        no_target_deadline = time.monotonic() + self._no_target_wait_s_active
        while True:
            if not self.routine_running:
                return False
            bgr, depth = self.remote_camera.capture_frame()
            if bgr is None or depth is None:
                self.auto_capture_failed = True
                self.after(0, lambda: self.status.set("自动抓拍失败：未收到完整 RGB + 深度，已停止后续运动。"))
                self.after(0, self._close_auto_capture_popup)
                return False
            self._show_auto_camera(bgr, depth, f"位置 {sequence_index}：正在自动识别")
            self._show_auto_capture_popup(bgr, depth, f"capture {sequence_index}: YOLO")
            box, yolo_points, yolo_labels, yolo_details = module.yolo_object_prompt(self.capture_detector, bgr)
            best_confidence = max((float(detail[1]) for detail in yolo_details), default=0.0)
            self.after(0, lambda confidence=best_confidence: self.capture_detection_info.set(
                f"本帧识别度：{confidence:.0%}（要求≥{self._min_target_confidence_active:.0%}）\n"
                f"无目标等待 {self._no_target_wait_s_active:.1f} s；抓拍停顿 {self._capture_settle_s_active:.1f} s"
            ))
            if box is not None and best_confidence >= self._min_target_confidence_active:
                break
            if box is not None:
                if not self._no_target_skip_active:
                    self.auto_capture_failed = True
                    self.after(0, lambda: self.status.set("自动抓拍识别度低于要求，已停止后续运动。"))
                    return False
                stamp = self._save_non_target_capture(
                    bgr, depth, sequence_index, "confidence_below_threshold"
                )
                self.after(0, lambda i=sequence_index, s=stamp, confidence=best_confidence: self.status.set(
                    f"位置 {i}：识别度 {confidence:.0%} 低于要求，已仅保存图片 {s}（未融合），立即继续下一步。"
                ))
                self._set_auto_capture_phase("低识别度：保存图片后跳过融合")
                self.after(0, self._close_auto_capture_popup)
                return True
            if not self._no_target_skip_active:
                self.auto_capture_failed = True
                self.after(0, lambda: self.status.set("自动抓拍未识别到 body/wing，未保存该帧并已停止后续运动。"))
                self.after(0, self._close_auto_capture_popup)
                return False
            remaining = no_target_deadline - time.monotonic()
            if remaining <= 0:
                reason = ("no_target_detected" if box is None else "confidence_below_threshold")
                stamp = self._save_non_target_capture(bgr, depth, sequence_index, reason)
                self.after(0, lambda i=sequence_index, s=stamp: self.status.set(
                    f"位置 {i}：{self._no_target_wait_s_active:.1f} 秒内未发现合格目标"
                    f"（最高 {best_confidence:.0%}），已保存非目标帧 {s}（未融合），继续下一步。"
                ))
                self._set_auto_capture_phase("未发现目标：保存图片后跳过融合")
                self.after(0, self._close_auto_capture_popup)
                return True
            self.after(0, lambda i=sequence_index, seconds=max(0.0, remaining): self.status.set(
                f"位置 {i}：目标未达到 {self._min_target_confidence_active:.0%}，继续检测（剩余 {seconds:.1f} 秒）…"
            ))
            self._set_capture_countdown("无目标检测", remaining, self._no_target_wait_s_active)
            time.sleep(min(0.35, remaining))
        yolo_preview = bgr.copy()
        x1, y1, x2, y2 = np.round(box).astype(int)
        cv2.rectangle(yolo_preview, (x1, y1), (x2, y2), (0, 255, 255), 3)
        labels_text = []
        for detail_index, (label, confidence, detail_box) in enumerate(yolo_details):
            dx1, dy1, dx2, dy2 = np.round(detail_box).astype(int)
            cv2.rectangle(yolo_preview, (dx1, dy1), (dx2, dy2), (255, 180, 0), 2)
            text = f"{label} {confidence:.0%}"
            labels_text.append(text)
            cv2.putText(yolo_preview, text, (dx1, max(28, dy1 - 7 - detail_index * 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 180, 0), 2, cv2.LINE_AA)
        self._show_auto_camera(yolo_preview, depth, "YOLO: " + (", ".join(labels_text) or "object"))
        self.after(0, lambda i=sequence_index, labels_text=labels_text: self.status.set(
            f"位置 {i}：YOLO 检测到 {', '.join(labels_text) or '目标'}，正在 SAM2 分割并生成点云…"
        ))
        self.capture_predictor.set_image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        self._set_auto_capture_phase("SAM2 分割")
        mask = module.sam_from_prompt(self.capture_predictor, box, yolo_points, yolo_labels)
        if int(mask.sum()) < 100:
            self.auto_capture_failed = True
            self.after(0, lambda: self.status.set("SAM2 掩码过小，未保存该帧并已停止后续运动。"))
            self.after(0, self._close_auto_capture_popup)
            return False
        scan = self._ensure_capture_scan()
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        preview = bgr.copy()
        preview[mask] = (preview[mask] * .42 + np.array((0, 220, 0)) * .58).astype(np.uint8)
        self._show_auto_camera(preview, depth, f"位置 {sequence_index}：YOLO + SAM2 掩码已生成")
        cv2.imwrite(str(scan / "images" / f"{stamp}.jpg"), bgr)
        np.save(scan / "depth" / f"{stamp}_depth.npy", depth)
        cv2.imwrite(str(scan / "masks" / f"{stamp}.png"), mask.astype(np.uint8) * 255)
        cv2.imwrite(str(scan / "preview" / f"{stamp}.jpg"), preview)
        tcp = module.read_capture_tcp_pose()
        with (scan / "captures.jsonl").open("a", encoding="utf-8") as manifest:
            manifest.write(json.dumps({"id": stamp,
                "pose_source": "calibrated_tcp" if tcp else "unavailable",
                "world_from_camera": tcp["world_from_camera"] if tcp else None,
                "calibrated_tcp": tcp}, ensure_ascii=False) + "\n")
        self._enqueue_fusion(scan, stamp, sequence_index)
        return True

    def _run_all_with_auto_capture_worker(self):
        try:
            if not self._prepare_auto_capture():
                return
            for index, record in enumerate(self.motion_records, 1):
                if not self.routine_running:
                    return
                target = list(record["pose"])
                self._set_auto_capture_phase(f"机械臂运动：相机位 {index}/{len(self.motion_records)}")
                self.after(0, lambda i=index: self.status.set(f"自动抓拍 {i}/{len(self.motion_records)}：正在前往位置 {i}…"))
                result = self.run_remote(self._cartesian_command(target))
                if result.returncode:
                    self.after(0, lambda i=index: self.status.set(f"位置 {i} 未确认到位，已停止后续运动。"))
                    return
                self.after(0, lambda i=index, settle=self._capture_settle_s_active: self.status.set(f"位置 {i} 已到达，稳定等待 {settle:.1f} 秒…"))
                deadline = time.monotonic() + self._capture_settle_s_active
                while self.routine_running and time.monotonic() < deadline:
                    self._set_auto_capture_phase("机械臂到位：抓拍停顿计时")
                    self._set_capture_countdown("抓拍停顿", deadline - time.monotonic(), self._capture_settle_s_active)
                    self._refresh_auto_camera(f"位置 {index} 已到达，稳定等待")
                    time.sleep(.1)
                if not self.routine_running or not self._capture_auto_frame(index):
                    return
            self.after(0, lambda: self.status.set("全部位置抓拍完成；正在自动恢复机械臂初始零位…"))
            args = " ".join(f"{value:.3f}" for value in HOME_JOINTS_DEG)
            home = (
                f"cd {REMOTE_DIR} && python3 piper_safe_move.py --can can1 "
                f"--joint-deg {args} --speed {self._joint_speed_level()} "
                "--execute --confirm ARM_CLEAR"
            )
            result = self.run_remote(home)
            if result.returncode:
                detail = (result.stdout + result.stderr).strip()[-600:]
                self.after(0, lambda: self.status.set(f"抓拍点云已保存，但自动复位失败：{detail}"))
                return
            if not self._wait_for_home_joint_pose(timeout_s=70.0):
                self.after(0, lambda: self.status.set(
                    "抓拍点云已保存，但尚未确认机械臂已回到零位；ZED 将保持运行。"
                ))
                return
            self._close_auto_capture_views()
            self.after(0, lambda: self.status.set(
                "自动抓拍规则运动完成：图片已保存，点云继续后台融合；机械臂已恢复初始零位。"
            ))
        except Exception as error:
            self.auto_capture_failed = True
            self.after(0, lambda: self.status.set(f"自动抓拍规则运动失败：{error}"))
        finally:
            # Preserve RGB/depth after a failure.  Previously this cleanup
            # immediately destroyed the only evidence of a failed first
            # capture, so the arm stopped at position 1 and the user saw no
            # ZED window at all.  Successful completion still closes it.
            if self.routine_auto_capture and not self.auto_capture_failed:
                self._close_auto_capture_views()
            elif self.auto_capture_failed:
                self.after(0, lambda: self.status.set(
                    "自动抓拍已在失败点停止；RGB+Depth 窗口保持打开，请查看最后一帧和上方原因。"
                ))
            self.routine_auto_capture = False
            self.routine_running = False
            self.after(0, self.refresh_status)

    def _close_auto_capture_views(self):
        """Stop capture only after home is confirmed; files remain on disk."""
        # Block stale queued display callbacks before releasing the camera.
        self.auto_camera_enabled = False
        self.auto_camera_display_frame = None
        self.after(0, self._close_auto_capture_popup)
        if self.auto_cloud_process and self.auto_cloud_process.poll() is None:
            self.auto_cloud_process.terminate()
        if self.remote_camera:
            self.remote_camera.close()
            self.remote_camera = None
        if self.camera_tunnel and self.camera_tunnel.poll() is None:
            self.camera_tunnel.terminate()
            self.camera_tunnel = None
        # These exact patterns only match the image-only scanner and its TCP
        # bridge launched by this capture workflow, never the arm controller.
        self.run_remote(
            "pkill -f '[z]ed_rigid_scanner.*--image-only' || true; "
            "pkill -f '[z]ed_tcp_capture_bridge.py' || true"
        )

    def _wait_for_home_joint_pose(self, timeout_s):
        """Require real CAN feedback near the zero pose before stopping ZED."""
        deadline = time.monotonic() + timeout_s
        while self.routine_running and time.monotonic() < deadline:
            result = self.run_remote(f"cd {REMOTE_DIR} && python3 piper_status.py --can can1 --seconds 0.1")
            match = re.search(r"joint_angles_raw_mdeg:\s*\[([^]]+)\]", result.stdout + result.stderr)
            if result.returncode == 0 and match:
                try:
                    angles = [float(value.strip()) / 1000.0 for value in match.group(1).split(",")]
                    if len(angles) == 6 and max(abs(value) for value in angles) <= 1.5:
                        return True
                except ValueError:
                    pass
            time.sleep(.5)
        return False

    def _run_all_records_worker(self):
        try:
            for index, record in enumerate(self.motion_records, 1):
                if not self.routine_running:
                    self.after(0, lambda: self.status.set("规则运动已由停止按钮中断。"))
                    return
                target = list(record["pose"])
                command = self._cartesian_command(target)
                self.after(0, lambda i=index, name=record["name"]: self.status.set(
                    f"规则运动 {i}/{len(self.motion_records)}：正在运行 {name}…"
                ))
                result = self.run_remote(command)
                if result.returncode:
                    output = (result.stdout + result.stderr).strip()
                    self.after(0, lambda: messagebox.showerror("规则运动失败", output[-1000:]))
                    return
                self.after(0, lambda i=index, total=len(self.motion_records): self.status.set(
                    f"第 {i}/{total} 个位置已到达，停留 5 秒…"
                ))
                # Do not block the stop button: check the run flag throughout
                # the dwell rather than using one uninterruptible sleep(5).
                dwell_deadline = time.monotonic() + 5.0
                while self.routine_running and time.monotonic() < dwell_deadline:
                    time.sleep(0.1)
                if not self.routine_running:
                    self.after(0, lambda: self.status.set("规则运动已由停止按钮中断。"))
                    return
            self.after(0, lambda: self.status.set("规则运动全部完成。"))
        finally:
            self.routine_running = False
            self.after(0, self.refresh_status)

    def _wait_for_tcp_target(self, target, timeout_s, show_camera=False, require_routine=True):
        """Wait for measured calibrated TCP feedback, never merely ROS publish success."""
        deadline = time.monotonic() + timeout_s
        while (self.routine_running or not require_routine) and time.monotonic() < deadline:
            if show_camera:
                self._refresh_auto_camera("机械臂正在运动到目标位置")
            result = self.run_remote(
                "source ~/agx_arm_ws/install/setup.bash && "
                "export ROS_DOMAIN_ID=36 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp "
                "CYCLONEDDS_URI=file://$HOME/zed_code/arm_control/cyclonedds_agx_eth0.xml && "
                "python3 ~/zed_code/arm_control/read_calibrated_tcp_pose.py --timeout 0.6"
            )
            match = re.search(r"TCP_POSE_JSON=(.*)", result.stdout + result.stderr)
            if result.returncode == 0 and match and match.group(1).strip() != "null":
                try:
                    pose = json.loads(match.group(1))
                    current = [float(value) * 1000.0 for value in pose["position_m"]]
                    current += list(_quat_to_rpy_deg(tuple(pose["quaternion_xyzw"])))
                    distance_error = max(abs(a - b) for a, b in zip(current[:3], target[:3]))
                    angle_error = max(abs((a - b + 180.0) % 360.0 - 180.0)
                                      for a, b in zip(current[3:], target[3:]))
                    if distance_error <= 5.0 and angle_error <= 3.0:
                        return True
                except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                    pass
            # Do not add a long blind delay between a confirmed arm pose and
            # the next capture.  SSH/topic sampling itself takes some time;
            # a short retry interval keeps automatic capture responsive.
            time.sleep(.25)
        return False

    def delete_selected_record(self):
        index = self._selected_record_index()
        if index is None:
            return
        name = self.motion_records[index]["name"]
        del self.motion_records[index]
        # Fill the deleted slot immediately: deleting 位置 1 makes the old
        # 位置 2 become the new 位置 1, both on screen and in the JSON file.
        self._renumber_motion_records(self.motion_records)
        self._save_motion_records()
        self._refresh_routine_tree()
        self.status.set(f"已删除 {name}。")

    def clear_motion_records(self):
        if not self.motion_records:
            return
        if not messagebox.askyesno("确认清空", "确认删除全部规则运动记录？此操作不会让机械臂运动。"):
            return
        self.motion_records.clear()
        self._save_motion_records()
        self._refresh_routine_tree()
        self.status.set("已清空全部规则运动记录。")

    def launch_remote_capture(self):
        """Launch the established SSH ZED + YOLO/SAM2 capture program unchanged."""
        if self.capture_process and self.capture_process.poll() is None:
            messagebox.showinfo("远程抓拍已运行", "远程 ZED 抓拍程序已在独立窗口中运行。")
            return
        if not os.path.isfile(REMOTE_CAPTURE_SCRIPT):
            messagebox.showerror("找不到抓拍程序", f"未找到原远程抓拍脚本：\n{REMOTE_CAPTURE_SCRIPT}")
            return
        try:
            self.capture_process = subprocess.Popen(
                ["bash", REMOTE_CAPTURE_SCRIPT], cwd=os.path.dirname(REMOTE_CAPTURE_SCRIPT)
            )
        except OSError as error:
            messagebox.showerror("启动抓拍失败", str(error))
            return
        self.status.set("已启动原有远程 ZED 抓拍窗口；按 Enter/Space 抓拍并进行 YOLO/SAM2 审核。")

    def _motion_page_changed(self, event):
        """Entering continuous mode must never reuse an old typed target."""
        if event.widget.index("current") == 1 and not self.cartesian_busy:
            self.sync_cartesian_target()

    def slider_changed(self, index, value):
        self.targets[index].set(f"{float(value):.2f}")
        self.publish_preview()

    def publish_preview(self):
        if not self.preview:
            return
        try:
            values = [float(target.get()) for target in self.targets]
        except ValueError:
            return
        self.preview.publish(values)

    def close(self):
        self.routine_running = False
        self._close_auto_capture_popup()
        if self.tracer:
            self.tracer.close()
        if self.remote_camera:
            self.remote_camera.close()
        if self.camera_tunnel and self.camera_tunnel.poll() is None:
            self.camera_tunnel.terminate()
        if self.preview:
            self.preview.close()
        if self.imu:
            self.imu.close()
            if getattr(self.imu, "rclpy", None) and self.imu.rclpy.ok():
                self.imu.rclpy.shutdown()
        self.destroy()

    def run_remote(self, remote_command):
        # Direct CAN Cartesian moves wait for real end-pose feedback; allow
        # their guarded 12 s settle window plus SSH overhead.
        return subprocess.run(["ssh", "-o", "BatchMode=yes", REMOTE_HOST, remote_command], text=True, capture_output=True, timeout=40)

    def _configured_speed(self):
        try:
            return min(100, max(1, int(self.speed_percent.get())))
        except ValueError:
            return 50

    def _joint_speed_level(self):
        """PiPER's guarded joint script accepts 1..10, unlike ROS 1..100%."""
        return min(10, max(1, round(self._configured_speed() / 10.0)))

    def apply_speed_percent(self):
        """Change the AGX controller limit only; this never sends a pose."""
        try:
            speed = int(self.speed_percent.get())
        except ValueError:
            messagebox.showwarning("速度输入错误", "速度必须是 1 到 100 的整数百分比。")
            return
        if not 1 <= speed <= 100:
            messagebox.showwarning("速度超出安全范围", "界面允许的速度范围是 1% 到 100%。")
            return
        self.status.set(f"正在把远程机械臂速度限制设置为 {speed}%（不会运动）…")
        threading.Thread(target=self._apply_speed_percent_worker, args=(speed,), daemon=True).start()

    def _apply_speed_percent_worker(self, speed):
        command = (
            "source ~/agx_arm_ws/install/setup.bash && "
            "export ROS_DOMAIN_ID=36 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp "
            "CYCLONEDDS_URI=file://$HOME/zed_code/arm_control/cyclonedds_agx_eth0.xml && "
            f"ros2 param set /agx_arm_ctrl_single_node speed_percent {speed} && "
            "ros2 param get /agx_arm_ctrl_single_node speed_percent"
        )
        result = self.run_remote(command)
        output = (result.stdout + result.stderr).strip()
        if result.returncode or f"Integer value is: {speed}" not in output:
            self.after(0, lambda: messagebox.showerror("速度设置失败", output[-1000:] or "远程控制器未确认速度参数。"))
            return
        self.after(0, lambda: self.status.set(f"远程机械臂速度已设置为 {speed}%（未发送运动目标）。"))

    def refresh_status(self, automatic=False):
        if self.refreshing:
            return
        self.refreshing = True
        if not automatic:
            self.status.set("正在读取远程机械臂状态…")
        threading.Thread(target=self._refresh_worker, args=(automatic,), daemon=True).start()

    def _refresh_worker(self, automatic=False):
        result = self.run_remote(
            f"source ~/agx_arm_ws/install/setup.bash && "
            f"export ROS_DOMAIN_ID=36 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp "
            f"CYCLONEDDS_URI=file://$HOME/zed_code/arm_control/cyclonedds_agx_eth0.xml && "
            f"python3 {REMOTE_DIR}/read_calibrated_tcp_pose.py --timeout 1.0; "
            f"cd {REMOTE_DIR} && python3 piper_status.py --can can1 --seconds 0.05"
        )
        text = result.stdout + result.stderr
        match = re.search(r"joint_angles_raw_mdeg:\s*\[([^]]+)\]", text)
        feedback = re.search(r"joint_feedback_hz:\s*([0-9.]+)", text)
        flange = re.search(r"flange_pose_raw:\s*\[([^]]+)\]", text)
        mode = re.search(r"Control Mode:\s*([^\n]+)", text)
        arm = re.search(r"Arm Status:\s*([^\n]+)", text)
        if result.returncode or not match:
            self.after(0, lambda: self._finish_refresh_error("读取失败：" + text[-300:].replace("\n", " ")))
            return
        try:
            values = [float(value.strip()) / 1000.0 for value in match.group(1).split(",")]
            if len(values) != 6:
                raise ValueError
        except ValueError:
            self.after(0, lambda: self._finish_refresh_error("读取失败：关节反馈格式异常"))
            return
        feedback_hz = float(feedback.group(1)) if feedback else 0.0
        if feedback_hz < 5.0:
            self.after(0, self._mark_can_unavailable)
            return
        try:
            flange_values = [float(value.strip()) / 1000.0 for value in flange.group(1).split(",")] if flange else None
            if flange_values is not None and len(flange_values) != 6:
                flange_values = None
        except ValueError:
            flange_values = None
        tcp_values = None
        tcp_match = re.search(r"TCP_POSE_JSON=(.*)", text)
        if tcp_match and tcp_match.group(1).strip() != "null":
            try:
                tcp = json.loads(tcp_match.group(1))
                position = tcp["position_m"]
                orientation = tcp["quaternion_xyzw"]
                if len(position) == 3 and len(orientation) == 4:
                    tcp_values = [value * 1000.0 for value in position] + list(_quat_to_rpy_deg(tuple(orientation)))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                tcp_values = None
        self.after(0, lambda: self._apply_status(
            values,
            mode.group(1) if mode else "未知",
            arm.group(1) if arm else "未知",
            flange_values,
            tcp_values,
            update_controls=not automatic,
        ))

    def _mark_can_unavailable(self):
        self.refreshing = False
        self.can_ready = False
        self.current = [None] * 6
        self.status.set("CAN 无有效关节反馈（0 Hz）：请检查机械臂上电、急停、CAN 线和终端电阻；已禁止使能/运动。")

    def _apply_status(self, values, mode, arm, flange_values=None, tcp_values=None, update_controls=True):
        self.refreshing = False
        self.can_ready = True
        self.current = values
        self.current_flange = flange_values
        if update_controls:
            for index, value in enumerate(values):
                self.current_labels[index].configure(text=f"{value:.3f}")
                self.targets[index].set(f"{value:.3f}")
                self.sliders[index].set(value)
        self.status.set(
            f"已连接：控制模式 {mode}；机械臂状态 {arm}。"
            + ("请先小步调整一个关节。" if update_controls else "末端与相机姿态持续更新中。")
        )
        calibrated_end = tcp_values or flange_values
        if calibrated_end:
            x, y, z, rx, ry, rz = calibrated_end
            title = "标定 TCP" if tcp_values else "末端法兰（TCP 话题暂不可用）"
            self.flange_pose.set(
                f"{title}：X={x:.2f} mm, Y={y:.2f} mm, Z={z:.2f} mm；"
                f"RX={rx:.2f}°, RY={ry:.2f}°, RZ={rz:.2f}°"
            )
        else:
            self.flange_pose.set("末端法兰：未收到有效位姿反馈")
        # Keep the user's slider targets untouched during automatic refresh,
        # while driving RViz with the actual arm feedback so the model animates
        # as the physical arm moves.
        if self.preview:
            if update_controls:
                self.publish_preview()
            else:
                self.preview.publish(values)
        self._update_camera_pose(calibrated_end)

    def _finish_refresh_error(self, message):
        self.refreshing = False
        self.status.set(message)

    def reset_camera_origin(self):
        if self.imu.current_pose() is None or self.origin_flange is None:
            messagebox.showwarning("尚未就绪", "请等待 IMU 与标定 TCP 位姿数据出现后再重设起始姿态。")
            return
        self.origin_flange = None
        self.origin_vehicle_position = None
        self.origin_vehicle_orientation = None
        self.camera_pose.set("末端相对小车初始坐标：已重设，等待下一帧数据…")

    def _update_camera_pose(self, flange_values):
        if not flange_values:
            return
        # Control coordinates are always TCP relative to the rigid arm base.
        # They must not change when the vehicle/IMU moves.
        x, y, z, rx, ry, rz = flange_values
        self.last_end_relative_pose = (x, y, z, rx, ry, rz)
        vehicle_pose = self.imu.current_pose()
        if vehicle_pose is None:
            self.camera_pose.set("末端相对小车初始坐标：等待 /tracer5/odometry/filtered（ROS_DOMAIN_ID=36）…")
            return
        vehicle_orientation, vehicle_position = vehicle_pose
        flange_orientation = _rpy_deg_to_quat(rx, ry, rz)
        # Assumption: arm base is rigidly mounted to the vehicle/IMU frame and
        # the camera axes equal the flange axes.  The zero frame is captured
        # when both sensors first provide valid data.
        end_orientation = _quat_multiply(vehicle_orientation, flange_orientation)
        flange_from_base_m = (x / 1000.0, y / 1000.0, z / 1000.0)
        end_from_imu_m = tuple(
            base + flange for base, flange in zip(ARM_BASE_FROM_IMU_M, flange_from_base_m)
        )
        rotated_end_offset = _rotate_vector(vehicle_orientation, end_from_imu_m)
        end_position = tuple(
            base + offset for base, offset in zip(vehicle_position, rotated_end_offset)
        )
        if self.origin_vehicle_position is None:
            self.origin_flange = (x, y, z)
            self.origin_vehicle_position = vehicle_position
            self.origin_vehicle_orientation = vehicle_orientation
        relative_orientation = _quat_multiply(
            _quat_inverse(self.origin_vehicle_orientation), end_orientation
        )
        roll, pitch, yaw = _quat_to_rpy_deg(relative_orientation)
        position_from_vehicle_start = _rotate_vector(
            _quat_inverse(self.origin_vehicle_orientation),
            tuple(value - origin for value, origin in zip(end_position, self.origin_vehicle_position)),
        )
        dx, dy, dz = (value * 1000.0 for value in position_from_vehicle_start)
        self.camera_pose.set(
            f"末端相对小车初始：X={dx:.2f} mm, Y={dy:.2f} mm, Z={dz:.2f} mm；"
            f"Roll={roll:.2f}°, Pitch={pitch:.2f}°, Yaw={yaw:.2f}°"
        )
        # last_end_relative_pose intentionally remains the arm-base TCP pose.

    def sync_cartesian_target(self):
        if not self.last_end_relative_pose:
            messagebox.showwarning("尚未就绪", "请等待滤波里程计和末端位姿数据。")
            return
        for target, value in zip(self.cartesian_inputs, self.last_end_relative_pose):
            target.set(f"{value:.2f}")
        for slider, value in zip(self.continuous_sliders, self.last_end_relative_pose):
            slider.set(value)

    def continuous_slider_changed(self, index, value):
        """Update the visible target while dragging; do not command yet."""
        self.cartesian_inputs[index].set(f"{float(value):.2f}")

    def continuous_slider_released(self, _index):
        if self.cartesian_busy:
            self.status.set("上一段连续运动尚未完成，请等待真实末端反馈。")
            return
        self.send_cartesian_target(direct=True)

    def adjust_cartesian(self, index, direction, direct=False):
        """Edit an interval target, or execute one guarded direct nudge."""
        if direct:
            if self.cartesian_busy:
                self.status.set("上一段连续运动尚未完成，请等待真实末端反馈。")
                return
            if not self.last_end_relative_pose:
                messagebox.showwarning("尚未就绪", "请等待滤波里程计和末端位姿数据。")
                return
            # Start every direct nudge from real feedback rather than a stale
            # target left by a previous command.
            self.sync_cartesian_target()
            step = 10.0 if index < 3 else 1.0
            value = self.last_end_relative_pose[index] + direction * step
            self.cartesian_inputs[index].set(f"{value:.2f}")
            self.continuous_sliders[index].set(value)
            self.send_cartesian_target(direct=True)
            return
        try:
            value = float(self.cartesian_inputs[index].get())
        except ValueError:
            messagebox.showerror("输入错误", "请先填入有效的末端数值。")
            return
        self.cartesian_inputs[index].set(f"{value + direction * CARTESIAN_STEPS[index]:.2f}")

    def _cartesian_command(self, target):
        # The AGX ROS participant is currently not discoverable on this
        # machine, despite the CAN driver being healthy.  Use the existing
        # vendor-SDK guard instead: it sends EndPoseCtrl directly over can1
        # and exits successfully only after live end-pose feedback reaches
        # the requested target.  --continuous removes only the old 55 mm
        # software nudge cap; firmware IK/joint limits remain enforced.
        raw_flange_target = _tcp_pose_to_raw_flange_pose(target)
        args = " ".join(f"{value:.3f}" for value in raw_flange_target)
        return (
            f"cd {REMOTE_DIR} && python3 piper_safe_end_pose.py --can can1 "
            f"--pose {args} --continuous --execute --confirm ARM_CLEAR"
        )

    def send_cartesian_target(self, direct=False):
        if not self.can_ready or not self.current_flange:
            messagebox.showwarning("CAN 未就绪", "未收到有效关节和末端反馈，已禁止发送末端目标。")
            return
        if not self.safety_ok.get():
            messagebox.showwarning("需要安全确认", "请确认现场安全后勾选安全确认。")
            return
        try:
            target = [float(value.get()) for value in self.cartesian_inputs]
        except ValueError:
            messagebox.showerror("输入错误", "末端 XYZ/Roll/Pitch/Yaw 必须都是数字。")
            return
        if not CARTESIAN_Z_MIN_MM <= target[2] <= CARTESIAN_Z_MAX_MM:
            messagebox.showwarning(
                "Z 轴超出范围",
                f"末端 Z 必须在 {CARTESIAN_Z_MIN_MM:.0f}～{CARTESIAN_Z_MAX_MM:.0f} mm 内；本次目标不会发送。",
            )
            # Return the continuous slider to its safe range too.  The typed
            # value is deliberately retained so the operator can correct it.
            if len(self.continuous_sliders) > 2:
                self.continuous_sliders[2].set(
                    min(CARTESIAN_Z_MAX_MM, max(CARTESIAN_Z_MIN_MM, target[2]))
                )
            return
        if not direct:
            answer = messagebox.askyesno("确认发送末端目标", f"将按机械臂底座坐标中的标定 TCP 目标，以当前 {self._configured_speed()}% 速度执行。\n确认周围无人、无障碍物且可触及实体急停？")
            if not answer:
                return
        # The UI displays calibrated TCP coordinates. The AGX TCP interface
        # applies the configured tcp_offset before commanding PiPER.
        command = self._cartesian_command(target)
        self.cartesian_busy = True
        self.status.set("正在执行连续微动…" if direct else "正在发送受限末端目标…")
        threading.Thread(target=self._cartesian_move_worker, args=(command, target), daemon=True).start()

    def _cartesian_move_worker(self, command, target):
        try:
            result = self.run_remote(command)
            output = (result.stdout + result.stderr).strip()
            if result.returncode:
                self.after(0, lambda: messagebox.showerror("远程拒绝或失败", output[-1200:]))
                return
            self.after(0, lambda: self.status.set("真实 CAN 末端反馈已确认到位；可继续下一段微动。"))
        except Exception as error:
            self.after(0, lambda: messagebox.showerror("末端控制通信失败", str(error), parent=self))
        finally:
            # SSH timeouts used to terminate this worker before clearing the
            # guard, leaving every later task permanently labelled "进行中".
            self.cartesian_busy = False
            self.after(0, self.refresh_status)

    def stop_cartesian_motion(self):
        """Request the AGX controller's real stop service, never a new pose."""
        answer = messagebox.askyesno(
            "确认停止", "将立即结束当前末端运动。停止后需要重新使能电机才能继续运动。\n"
            "若现场有碰撞风险，请优先按实体急停。\n\n确认停止？"
        )
        if not answer:
            return
        # Release every local workflow guard immediately.  The remote stop
        # request below still protects the physical arm, but a hung SSH/ZED
        # worker must never leave the GUI permanently unusable.
        self.routine_running = False
        self.routine_auto_capture = False
        self.cartesian_busy = False
        self.vehicle_route_cancelled = True
        self.vehicle_route_running = False
        self.vehicle_auto_capture_running = False
        self.auto_capture_plan_cancelled = True
        self.auto_capture_plan_running = False
        self.tracer.stop_all()
        self.status.set("已清除本地任务占用，正在请求远程停止当前末端运动…")
        threading.Thread(target=self._stop_cartesian_worker, daemon=True).start()

    def _stop_cartesian_worker(self):
        command = (
            "source ~/agx_arm_ws/install/setup.bash && "
            "export ROS_DOMAIN_ID=36 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp "
            "CYCLONEDDS_URI=file://$HOME/zed_code/arm_control/cyclonedds_agx_eth0.xml && "
            "ros2 service call /emergency_stop std_srvs/srv/Empty '{}'"
        )
        result = self.run_remote(command)
        output = (result.stdout + result.stderr).strip()
        self.cartesian_busy = False
        if result.returncode:
            self.after(0, lambda: messagebox.showerror("停止失败", output[-1200:]))
        else:
            self.after(0, lambda: self.status.set("已请求停止当前末端运动；请重新使能电机后再发送新目标。"))
        self.after(0, self.refresh_status)

    def adjust(self, index, delta):
        if self.current[index] is None:
            messagebox.showwarning("尚无姿态", "请先点击“刷新姿态 / 状态”。")
            return
        try:
            value = float(self.targets[index].get()) + delta
        except ValueError:
            value = self.current[index] + delta
        low, high = JOINT_LIMITS[index]
        self.sliders[index].set(min(high, max(low, value)))

    def apply_relative(self, index):
        """Add the entered delta to the target; this never moves the arm."""
        try:
            delta = float(self.relative_inputs[index].get())
            base = float(self.targets[index].get())
        except ValueError:
            messagebox.showerror("相对角度错误", "相对变化和当前目标必须是数字。")
            return
        low, high = JOINT_LIMITS[index]
        target = base + delta
        if not low <= target <= high:
            messagebox.showwarning("超过关节限位", f"J{index + 1} 目标 {target:.2f}° 超出 {low}° 到 {high}°。")
            return
        self.sliders[index].set(target)
        self.relative_inputs[index].set("0")
        self.publish_preview()

    def enable_motors(self):
        if not self.can_ready:
            messagebox.showwarning("CAN 未就绪", "未收到有效关节反馈，已禁止使能。请先刷新姿态并恢复 CAN 通信。")
            return
        if not self.safety_ok.get():
            messagebox.showwarning("需要安全确认", "请确认现场安全后勾选安全确认。")
            return
        answer = messagebox.askyesno(
            "确认使能", "这会使电机上电保持当前位置，但不会发送关节运动目标。\n"
            "确认机械臂周围无人且可触及实体急停？"
        )
        if not answer:
            return
        command = f"cd {REMOTE_DIR} && python3 piper_enable.py --can can1 --execute --confirm ARM_CLEAR"
        self.status.set("正在使能远程 PiPER 电机…")
        threading.Thread(target=self._enable_worker, args=(command,), daemon=True).start()

    def _enable_worker(self, command):
        result = self.run_remote(command)
        output = (result.stdout + result.stderr).strip()
        if result.returncode:
            self.after(0, lambda: messagebox.showerror("使能失败", output[-1000:]))
        else:
            self.after(0, lambda: self.status.set("电机已使能；请先小幅调整一个关节，再发送低速目标。"))
        self.after(0, self.refresh_status)

    def send_target(self):
        if not self.can_ready:
            messagebox.showwarning("CAN 未就绪", "未收到有效关节反馈，已禁止发送运动目标。")
            return
        if not self.safety_ok.get():
            messagebox.showwarning("需要安全确认", "请确认现场安全后勾选安全确认。")
            return
        if any(value is None for value in self.current):
            messagebox.showwarning("尚无姿态", "请先读取实时姿态。")
            return
        try:
            target = [float(item.get()) for item in self.targets]
        except ValueError:
            messagebox.showerror("输入错误", "六个目标角度必须是数字。")
            return
        for index, (value, limits) in enumerate(zip(target, JOINT_LIMITS), 1):
            if not limits[0] <= value <= limits[1]:
                messagebox.showerror("超过关节限位", f"J{index} 必须在 {limits[0]}° 到 {limits[1]}° 之间。")
                return
        speed = self._configured_speed()
        joint_speed = self._joint_speed_level()
        answer = messagebox.askyesno("确认发送", f"将以 {speed}% 速度发送此目标。确认机械臂周围无人且可触及实体急停？")
        if not answer:
            return
        args = " ".join(f"{value:.3f}" for value in target)
        command = f"cd {REMOTE_DIR} && python3 piper_safe_move.py --can can1 --joint-deg {args} --speed {joint_speed} --execute --confirm ARM_CLEAR"
        self.status.set("正在发送低速目标…")
        threading.Thread(target=self._move_worker, args=(command,), daemon=True).start()

    def move_home(self):
        """Return to the configured mechanical zero using the guarded move path."""
        if not self.can_ready:
            messagebox.showwarning("CAN 未就绪", "未收到有效关节反馈，已禁止发送运动目标。")
            return
        if not self.safety_ok.get():
            messagebox.showwarning("需要安全确认", "请确认现场安全后勾选安全确认。")
            return
        speed = self._configured_speed()
        joint_speed = self._joint_speed_level()
        answer = messagebox.askyesno(
            "确认恢复初始姿态",
            f"将以 {speed}% 速度移动至初始零位 [0, 0, 0, 0, 0, 0]°。\n"
            "确认路径及周围空间无碰撞风险，并可触及实体急停？",
        )
        if not answer:
            return
        for index, value in enumerate(HOME_JOINTS_DEG):
            self.targets[index].set(f"{value:.3f}")
            self.sliders[index].set(value)
        self.publish_preview()
        args = " ".join(f"{value:.3f}" for value in HOME_JOINTS_DEG)
        command = f"cd {REMOTE_DIR} && python3 piper_safe_move.py --can can1 --joint-deg {args} --speed {joint_speed} --execute --confirm ARM_CLEAR"
        self.status.set("正在以低速恢复初始姿态…")
        threading.Thread(target=self._move_worker, args=(command,), daemon=True).start()

    def _move_worker(self, command):
        result = self.run_remote(command)
        output = (result.stdout + result.stderr).strip()
        if result.returncode:
            self.after(0, lambda: messagebox.showerror("远程端拒绝或失败", output[-1000:]))
            self.after(0, self.refresh_status)
            return
        self.after(0, lambda: self.status.set("目标已发送。请观察机械臂，再刷新确认实际姿态。"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview", action="store_true", help="publish target pose to a local RViz preview")
    args = parser.parse_args()
    PiperPanel(preview=args.preview).mainloop()
