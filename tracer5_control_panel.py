#!/usr/bin/env python3
"""Tracer5 local chassis control panel.

The panel only publishes geometry_msgs/Twist to /tracer5/cmd_vel.  Position
control is a conservative closed loop over /tracer5/odometry/filtered, whose
orientation is EKF-fused with the vehicle IMU.  It never configures CAN or
starts the remote hardware stack.
"""

from __future__ import annotations

import math
import os
import json
import shlex
import subprocess
import threading
import time
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import messagebox, ttk


ROOT = Path(__file__).resolve().parent
CMD_TOPIC = "/tracer5/cmd_vel"
FILTERED_ODOM_TOPIC = "/tracer5/odometry/filtered"
RAW_ODOM_TOPIC = "/tracer5/odom"
CONTROL_HZ = 20.0
POSITION_TOLERANCE_M = 0.005  # 0.5 cm: target should closely match displayed pose.
YAW_TOLERANCE_RAD = math.radians(0.5)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def yaw_from_quaternion(q) -> float:
    sin_yaw = 2.0 * (q.w * q.z + q.x * q.y)
    cos_yaw = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(sin_yaw, cos_yaw)


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw: float


@dataclass(frozen=True)
class PlanSegment:
    linear: float
    angular: float
    duration: float


class TracerRos:
    """Thread-safe ROS driver. All physical commands originate here."""

    def __init__(self, on_pose_update) -> None:
        self.error = ""
        self._lock = threading.RLock()
        self._running = True
        self._on_pose_update = on_pose_update
        self.pose: Pose2D | None = None
        self.pose_source = ""
        self.last_pose_time = 0.0
        self.initial_pose: Pose2D | None = None
        self.initial_source = ""
        self.manual = (0.0, 0.0)
        self.target: Pose2D | None = None
        self.target_name = ""
        self.target_drive_straight = False
        self.target_phase = "idle"
        self.target_speed = (0.10, 0.55)
        self.position_tolerance_m = POSITION_TOLERANCE_M
        self.yaw_tolerance_rad = YAW_TOLERANCE_RAD
        self.auto_state = "idle"
        self.plan: list[PlanSegment] = []
        self.active_plan: list[PlanSegment] = []
        self.play_index = 0
        self.play_segment_start = 0.0
        self._last_command = (0.0, 0.0)
        self._recording = False
        self._record_last_command = (0.0, 0.0)
        self._record_last_time = 0.0
        self._record_segments: list[PlanSegment] = []
        self.node = None
        self.rclpy = None
        self.bridge = None
        self.use_ssh_bridge = os.environ.get("TRACER5_USE_SSH_BRIDGE", "0") == "1"

        try:
            if self.use_ssh_bridge:
                self._start_ssh_bridge()
                return
            os.environ.setdefault("ROS_DOMAIN_ID", "36")
            os.environ.setdefault("ROS_LOCALHOST_ONLY", "0")
            os.environ.setdefault("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")
            profile = ROOT / "fastdds_tracer5.xml"
            if profile.is_file():
                os.environ.setdefault("FASTRTPS_DEFAULT_PROFILES_FILE", str(profile))
                os.environ.setdefault("RMW_FASTRTPS_USE_QOS_FROM_XML", "1")
            import rclpy
            from geometry_msgs.msg import Twist
            from nav_msgs.msg import Odometry
            from rclpy.executors import SingleThreadedExecutor
            from rclpy.node import Node

            self.rclpy, self.Twist = rclpy, Twist
            if not rclpy.ok():
                rclpy.init()
            self.node = Node("tracer5_local_control_panel")
            self.publisher = self.node.create_publisher(Twist, CMD_TOPIC, 10)
            # Filtered odometry is preferred: it contains the IMU fusion.
            self.node.create_subscription(Odometry, FILTERED_ODOM_TOPIC,
                                          lambda msg: self._odom(msg, "filtered"), 20)
            self.node.create_subscription(Odometry, RAW_ODOM_TOPIC,
                                          lambda msg: self._odom(msg, "raw"), 10)
            self.executor = SingleThreadedExecutor()
            self.executor.add_node(self.node)
            self.thread = threading.Thread(target=self._spin, daemon=True)
            self.thread.start()
        except Exception as exc:
            self.error = str(exc)

    def _start_ssh_bridge(self) -> None:
        """Use SSH when Wi-Fi permits laptop→AGX but not DDS peer traffic."""
        host = os.environ.get("TRACER5_REMOTE_HOST", "192.168.50.55")
        remote = (
            "source /opt/ros/humble/setup.bash; "
            "source ~/5G_slam_edge/install/setup.bash; "
            "exec python3 ~/tracer5_control/bin/tracer5_ssh_control_bridge.py"
        )
        self.bridge = subprocess.Popen(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", f"skki@{host}", f"bash -lc {shlex.quote(remote)}"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1,
        )
        self.thread = threading.Thread(target=self._spin_ssh, daemon=True)
        self.thread.start()
        self.stderr_thread = threading.Thread(target=self._read_bridge_errors, daemon=True)
        self.stderr_thread.start()

    def _read_bridge_errors(self) -> None:
        if not self.bridge or not self.bridge.stderr:
            return
        for line in self.bridge.stderr:
            text = line.strip()
            if text:
                self.error = f"远程控制通道：{text[-180:]}"

    def _spin_ssh(self) -> None:
        if not self.bridge or not self.bridge.stdout:
            return
        # Reader and controller are separated so an idle remote pose stream
        # never delays a stop command from the UI.
        reader = threading.Thread(target=self._read_bridge_poses, daemon=True)
        reader.start()
        period = 1.0 / CONTROL_HZ
        while self._running and self.bridge.poll() is None:
            started = time.monotonic()
            self._control_tick(started)
            elapsed = time.monotonic() - started
            if elapsed < period:
                time.sleep(period - elapsed)
        if self._running and not self.error:
            self.error = "远程 SSH 控制通道已关闭"

    def _read_bridge_poses(self) -> None:
        if not self.bridge or not self.bridge.stdout:
            return
        for line in self.bridge.stdout:
            try:
                event = json.loads(line)
                if event.get("type") != "pose":
                    continue
                pose = Pose2D(float(event["x"]), float(event["y"]), float(event["yaw"]))
            except (ValueError, TypeError, KeyError, json.JSONDecodeError):
                continue
            with self._lock:
                self.pose = pose
                self.pose_source = str(event.get("source", "filtered"))
                self.last_pose_time = time.monotonic()
                if self.initial_pose is None or (self.pose_source == "filtered" and self.initial_source != "filtered"):
                    self.initial_pose = pose
                    self.initial_source = self.pose_source
            self._on_pose_update()

    def _odom(self, msg, source: str) -> None:
        pose = msg.pose.pose
        received = Pose2D(float(pose.position.x), float(pose.position.y), yaw_from_quaternion(pose.orientation))
        with self._lock:
            # Keep raw odometry only as fallback; do not overwrite a live EKF pose.
            if source == "raw" and self.pose_source == "filtered" and time.monotonic() - self.last_pose_time < 1.0:
                return
            self.pose, self.pose_source, self.last_pose_time = received, source, time.monotonic()
            if self.initial_pose is None or (source == "filtered" and self.initial_source != "filtered"):
                self.initial_pose = received
                self.initial_source = source
        self._on_pose_update()

    def _spin(self) -> None:
        period = 1.0 / CONTROL_HZ
        while self._running and self.rclpy and self.rclpy.ok():
            started = time.monotonic()
            self.executor.spin_once(timeout_sec=0.02)
            self._control_tick(started)
            elapsed = time.monotonic() - started
            if elapsed < period:
                time.sleep(period - elapsed)

    def _control_tick(self, now: float) -> None:
        with self._lock:
            if self.auto_state == "target":
                command = self._target_command_locked()
            elif self.auto_state == "plan":
                command = self._plan_command_locked(now)
            else:
                command = self.manual
            self._publish_locked(*command)

    def _target_command_locked(self) -> tuple[float, float]:
        if self.pose is None or self.target is None:
            self.auto_state = "idle"
            return 0.0, 0.0
        dx, dy = self.target.x - self.pose.x, self.target.y - self.pose.y
        distance = math.hypot(dx, dy)
        yaw_error = wrap_angle(self.target.yaw - self.pose.yaw)
        v_limit, w_limit = self.target_speed
        if self.target_drive_straight:
            # Pure relative X commands mean "drive along the current vehicle
            # heading".  Do not steer to compensate IMU/odometry side drift,
            # otherwise a nominally straight 10 cm move visibly oscillates.
            remaining = math.cos(self.target.yaw) * dx + math.sin(self.target.yaw) * dy
            if abs(remaining) < self.position_tolerance_m:
                self.auto_state, self.target_name, self.target_drive_straight, self.target_phase = "idle", "", False, "idle"
                return 0.0, 0.0
            linear = math.copysign(clamp(0.75 * abs(remaining), 0.035, v_limit), remaining)
            return linear, 0.0
        # Non-holonomic chassis: use a stable turn → drive → final-turn
        # sequence.  The earlier "drive and steer continuously" controller
        # could oscillate left/right on a short lateral target.
        if distance <= self.position_tolerance_m:
            self.target_phase = "final_turn"
        if self.target_phase == "turn_to_path":
            bearing = math.atan2(dy, dx)
            heading_error = wrap_angle(bearing - self.pose.yaw)
            if abs(heading_error) <= math.radians(2.0):
                self.target_phase = "drive"
            else:
                return 0.0, clamp(1.2 * heading_error, -w_limit, w_limit)
        if self.target_phase == "drive":
            bearing = math.atan2(dy, dx)
            heading_error = wrap_angle(bearing - self.pose.yaw)
            if abs(heading_error) > math.radians(12.0):
                self.target_phase = "turn_to_path"
                return 0.0, clamp(1.2 * heading_error, -w_limit, w_limit)
            linear = clamp(0.70 * distance, 0.030, v_limit)
            angular = clamp(0.65 * heading_error, -0.45 * w_limit, 0.45 * w_limit)
            return linear, angular
        # Hold position and turn only once the translation is completed.  A
        # differential chassis can drift slightly while making that last
        # turn, so return to path correction if the position leaves tolerance.
        if self.target_phase == "final_turn" and distance > self.position_tolerance_m:
            self.target_phase = "turn_to_path"
            return self._target_command_locked()
        if abs(yaw_error) <= self.yaw_tolerance_rad:
            self.auto_state, self.target_name, self.target_drive_straight, self.target_phase = "idle", "", False, "idle"
            return 0.0, 0.0
        return 0.0, clamp(1.0 * yaw_error, -0.55 * w_limit, 0.55 * w_limit)

    def _plan_command_locked(self, now: float) -> tuple[float, float]:
        if self.play_index >= len(self.active_plan):
            self.auto_state = "idle"
            return 0.0, 0.0
        segment = self.active_plan[self.play_index]
        if now - self.play_segment_start >= segment.duration:
            self.play_index += 1
            self.play_segment_start = now
            return self._plan_command_locked(now)
        return segment.linear, segment.angular

    def _publish_locked(self, linear: float, angular: float) -> None:
        if self.use_ssh_bridge:
            if not self.bridge or self.bridge.poll() is not None or not self.bridge.stdin:
                return
            try:
                self.bridge.stdin.write(json.dumps({"type": "twist", "linear": float(linear), "angular": float(angular)}) + "\n")
                self.bridge.stdin.flush()
            except (BrokenPipeError, OSError):
                self.error = "远程 SSH 控制通道已断开"
                return
        else:
            if self.node is None:
                return
            msg = self.Twist()
            msg.linear.x, msg.angular.z = float(linear), float(angular)
            self.publisher.publish(msg)
        self._last_command = (float(linear), float(angular))

    def _record_transition_locked(self, command: tuple[float, float], now: float) -> None:
        if not self._recording:
            return
        old = self._record_last_command
        duration = now - self._record_last_time
        if duration > 0.05 and (abs(old[0]) > 1e-6 or abs(old[1]) > 1e-6):
            self._record_segments.append(PlanSegment(old[0], old[1], duration))
        self._record_last_command, self._record_last_time = command, now

    def set_manual(self, linear: float, angular: float) -> None:
        now = time.monotonic()
        with self._lock:
            if self.auto_state != "idle":
                return
            command = (float(linear), float(angular))
            if command != self.manual:
                self._record_transition_locked(command, now)
            self.manual = command
            self._publish_locked(*command)

    def reset_origin(self) -> bool:
        with self._lock:
            if self.pose is None:
                return False
            self.initial_pose = self.pose
            self.initial_source = self.pose_source
            return True

    def set_tolerances(self, position_cm: float, yaw_deg: float) -> None:
        with self._lock:
            self.position_tolerance_m = clamp(float(position_cm) / 100.0, 0.001, 0.20)
            self.yaw_tolerance_rad = math.radians(clamp(float(yaw_deg), 0.1, 20.0))

    def snapshot(self):
        with self._lock:
            return self.pose, self.initial_pose, self.pose_source, self.last_pose_time, self.auto_state, self.target_name

    def start_target(self, target: Pose2D, name: str, linear: float, angular: float, straight: bool = False) -> tuple[bool, str]:
        with self._lock:
            if self.pose is None:
                return False, "尚未收到小车定位数据"
            if time.monotonic() - self.last_pose_time > 1.0:
                return False, "定位数据已超时；小车可能离线"
            self.manual = (0.0, 0.0)
            self.target, self.target_name, self.target_drive_straight = target, name, bool(straight)
            self.target_phase = "drive" if straight else "turn_to_path"
            self.target_speed = (clamp(float(linear), 0.02, 0.40), clamp(float(angular), 0.10, 1.50))
            self.auto_state = "target"
            return True, ""

    def begin_recording(self) -> bool:
        with self._lock:
            if self.auto_state != "idle":
                return False
            self._recording = True
            self._record_segments = []
            self._record_last_command = self.manual
            self._record_last_time = time.monotonic()
            return True

    def end_recording(self) -> int:
        with self._lock:
            if not self._recording:
                return len(self.plan)
            self._record_transition_locked((0.0, 0.0), time.monotonic())
            self._recording = False
            self.plan = list(self._record_segments)
            self._record_segments = []
            return len(self.plan)

    def is_recording(self) -> bool:
        with self._lock:
            return self._recording

    def start_plan(self, linear_scale: float, angular_scale: float) -> tuple[bool, str]:
        with self._lock:
            if not self.plan:
                return False, "没有已记录的运动"
            # Scale only velocities; segment duration stays unchanged. Keep
            # the saved recording untouched so repeated playback does not
            # become slower and slower.
            ls, ws = clamp(linear_scale, 0.1, 1.0), clamp(angular_scale, 0.1, 1.0)
            self.active_plan = [PlanSegment(clamp(s.linear * ls, -0.40, 0.40), clamp(s.angular * ws, -1.50, 1.50), s.duration) for s in self.plan]
            self.play_index, self.play_segment_start, self.auto_state = 0, time.monotonic(), "plan"
            self.manual = (0.0, 0.0)
            return True, ""

    def stop_all(self) -> None:
        with self._lock:
            self.auto_state, self.target, self.target_name, self.target_drive_straight, self.target_phase = "idle", None, "", False, "idle"
            self.manual = (0.0, 0.0)
            self._record_transition_locked((0.0, 0.0), time.monotonic())
            for _ in range(5):
                self._publish_locked(0.0, 0.0)

    def close(self) -> None:
        self.stop_all()
        self._running = False
        if self.bridge and self.bridge.poll() is None:
            self.bridge.terminate()
        if self.node:
            self.node.destroy_node()
        if self.rclpy and self.rclpy.ok():
            self.rclpy.shutdown()


class TracerControlPanel(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Tracer5 小车本机控制")
        self.minsize(790, 650)
        self.geometry("900x760")
        self.option_add("*Font", "Sans 11")
        self.linear_speed = tk.DoubleVar(value=0.10)
        self.angular_speed = tk.DoubleVar(value=0.55)
        self.status = tk.StringVar(value="初始化 ROS…")
        self.pose_text = tk.StringVar(value="位置：等待 /tracer5/odometry/filtered（IMU 融合）…")
        self.absolute_current_text = tk.StringVar(value="当前绝对坐标：等待定位数据…")
        self.relative_x, self.relative_y, self.relative_yaw = tk.DoubleVar(), tk.DoubleVar(), tk.DoubleVar()
        self.absolute_x, self.absolute_y, self.absolute_yaw = tk.DoubleVar(), tk.DoubleVar(), tk.DoubleVar()
        self.absolute_keep_yaw = tk.BooleanVar(value=True)
        self._pressed: set[str] = set()
        self.ros = TracerRos(self._pose_changed)
        self._build()
        self.bind_all("<KeyPress>", self._key_press)
        self.bind_all("<KeyRelease>", self._key_release)
        self.protocol("WM_DELETE_WINDOW", self._close)
        self.after(150, self._refresh)

    def _build(self) -> None:
        root = ttk.Frame(self, padding=12)
        root.pack(fill="both", expand=True)
        ttk.Label(root, text="Tracer5 小车控制", font=("Sans", 17, "bold")).pack(anchor="w")
        ttk.Label(root, text=f"仅发布 {CMD_TOPIC}；不会控制机械臂。定位使用 IMU 融合的 filtered odometry。", foreground="#555").pack(anchor="w")
        ttk.Label(root, textvariable=self.status, foreground="#1769aa").pack(anchor="w", pady=(5, 2))
        ttk.Label(root, textvariable=self.pose_text, foreground="#487a1e", font=("Sans", 11, "bold")).pack(anchor="w", pady=(0, 8))

        speed = ttk.LabelFrame(root, text="全局速度限制（手动、相对、绝对、规划回放均使用）", padding=8)
        speed.pack(fill="x")
        self._speed_row(speed, "线速度 m/s", self.linear_speed, 0.02, 0.40, 0.01, 0)
        self._speed_row(speed, "角速度 rad/s", self.angular_speed, 0.10, 1.50, 0.05, 1)

        notebook = ttk.Notebook(root)
        notebook.pack(fill="both", expand=True, pady=10)
        self._manual_tab(notebook)
        self._relative_tab(notebook)
        self._absolute_tab(notebook)
        self._plan_tab(notebook)

        ttk.Button(root, text="即停：取消所有运动并连续发送零速度", command=self._stop_all).pack(fill="x", ipady=7)

    def _speed_row(self, parent, label, variable, minimum, maximum, increment, row):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8))
        ttk.Scale(parent, variable=variable, from_=minimum, to=maximum, orient="horizontal").grid(row=row, column=1, sticky="ew")
        ttk.Spinbox(parent, textvariable=variable, from_=minimum, to=maximum, increment=increment, width=7).grid(row=row, column=2, padx=8)
        parent.columnconfigure(1, weight=1)

    def _manual_tab(self, notebook) -> None:
        tab = ttk.Frame(notebook, padding=16)
        notebook.add(tab, text="手动方向")
        ttk.Label(tab, text="按住方向键或按钮移动；松开即自动停车。", foreground="#555").pack()
        grid = ttk.Frame(tab)
        grid.pack(expand=True, pady=20)
        self._move_button(grid, "↑\n前进", "up", 0, 1)
        self._move_button(grid, "←\n左转", "left", 1, 0)
        ttk.Button(grid, text="停止\nSpace", command=self._stop_all, width=12).grid(row=1, column=1, padx=8, pady=8, ipady=8)
        self._move_button(grid, "→\n右转", "right", 1, 2)
        self._move_button(grid, "↓\n后退", "down", 2, 1)

    def _motion_fields(self, parent, variables) -> None:
        labels = (("X cm", variables[0]), ("Y cm", variables[1]), ("Yaw °", variables[2]))
        for column, (label, variable) in enumerate(labels):
            ttk.Label(parent, text=label).grid(row=0, column=column * 2, sticky="e", padx=(5, 4), pady=8)
            step = 10.0 if "cm" in label else 5.0
            ttk.Spinbox(parent, textvariable=variable, from_=-1000.0, to=1000.0, increment=step, width=11).grid(row=0, column=column * 2 + 1, sticky="w", padx=(0, 10), pady=8)

    def _relative_tab(self, notebook) -> None:
        tab = ttk.Frame(notebook, padding=16)
        notebook.add(tab, text="相对运动")
        ttk.Label(tab, text="以当前车体位置和朝向为原点：X 前方为正，Y 左方为正。X/Y 每格 10 cm。", foreground="#555").pack(anchor="w")
        fields = ttk.Frame(tab)
        fields.pack(anchor="w", pady=12)
        self._motion_fields(fields, (self.relative_x, self.relative_y, self.relative_yaw))
        ttk.Button(tab, text="移动至相对目标", command=self._go_relative).pack(anchor="w", fill="x", ipady=5)

    def _absolute_tab(self, notebook) -> None:
        tab = ttk.Frame(notebook, padding=16)
        notebook.add(tab, text="绝对运动")
        ttk.Label(tab, text="以首次获得的 IMU 融合位置为原点。X/Y 每格 10 cm；可随时重设原点。下方输入框是目标，不是当前值。", foreground="#555").pack(anchor="w")
        ttk.Label(tab, textvariable=self.absolute_current_text, foreground="#487a1e", font=("Sans", 11, "bold")).pack(anchor="w", pady=(8, 0))
        fields = ttk.Frame(tab)
        fields.pack(anchor="w", pady=12)
        self._motion_fields(fields, (self.absolute_x, self.absolute_y, self.absolute_yaw))
        ttk.Checkbutton(
            tab,
            text="保持当前航向（默认；忽略 Yaw 目标，避免仅改位置时被强制回到初始朝向）",
            variable=self.absolute_keep_yaw,
        ).pack(anchor="w", pady=(0, 10))
        buttons = ttk.Frame(tab)
        buttons.pack(fill="x")
        ttk.Button(buttons, text="移动至绝对目标", command=self._go_absolute).pack(side="left", fill="x", expand=True, ipady=5, padx=(0, 5))
        ttk.Button(buttons, text="将当前位置设为绝对原点", command=self._reset_origin).pack(side="left", fill="x", expand=True, ipady=5, padx=(5, 0))

    def _plan_tab(self, notebook) -> None:
        tab = ttk.Frame(notebook, padding=16)
        notebook.add(tab, text="规划运动")
        ttk.Label(tab, text="点击“开始记录”后用手动方向控制；“结束记录”会保存每段速度和持续时间。回放会从当前位置重复这些指令。", wraplength=720, foreground="#555").pack(anchor="w")
        self.plan_text = tk.StringVar(value="未记录运动")
        ttk.Label(tab, textvariable=self.plan_text, foreground="#1769aa").pack(anchor="w", pady=18)
        row = ttk.Frame(tab)
        row.pack(fill="x")
        self.record_button = ttk.Button(row, text="开始记录", command=self._toggle_record)
        self.record_button.pack(side="left", fill="x", expand=True, padx=(0, 5), ipady=6)
        ttk.Button(row, text="启动回放", command=self._start_plan).pack(side="left", fill="x", expand=True, padx=(5, 0), ipady=6)

    def _move_button(self, parent, label, direction, row, column) -> None:
        button = ttk.Button(parent, text=label, width=12)
        button.grid(row=row, column=column, padx=8, pady=8, ipady=8)
        button.bind("<ButtonPress-1>", lambda _e: self._press(direction))
        button.bind("<ButtonRelease-1>", lambda _e: self._release(direction))

    def _manual_motion(self, direction: str) -> tuple[float, float]:
        v, w = float(self.linear_speed.get()), float(self.angular_speed.get())
        return {"up": (v, 0.0), "down": (-v, 0.0), "left": (0.0, w), "right": (0.0, -w)}[direction]

    def _press(self, direction: str) -> None:
        self._pressed.add(direction)
        self.ros.set_manual(*self._manual_motion(direction))

    def _release(self, direction: str) -> None:
        self._pressed.discard(direction)
        if not self._pressed:
            self.ros.set_manual(0.0, 0.0)

    def _key_press(self, event) -> None:
        mapping = {"Up": "up", "Down": "down", "Left": "left", "Right": "right"}
        if event.keysym in mapping:
            self._press(mapping[event.keysym])
        elif event.keysym in ("space", "Escape"):
            self._stop_all()

    def _key_release(self, event) -> None:
        mapping = {"Up": "up", "Down": "down", "Left": "left", "Right": "right"}
        if event.keysym in mapping:
            self._release(mapping[event.keysym])

    def _go_relative(self) -> None:
        pose, _, _, _, _, _ = self.ros.snapshot()
        if pose is None:
            messagebox.showwarning("无法运动", "尚未收到 /tracer5/odometry/filtered")
            return
        forward, left = self.relative_x.get() / 100.0, self.relative_y.get() / 100.0
        target = Pose2D(pose.x + math.cos(pose.yaw) * forward - math.sin(pose.yaw) * left,
                        pose.y + math.sin(pose.yaw) * forward + math.cos(pose.yaw) * left,
                        wrap_angle(pose.yaw + math.radians(self.relative_yaw.get())))
        pure_x = abs(left) < 1e-6 and abs(self.relative_yaw.get()) < 1e-6
        self._start_target(target, "相对直行 X" if pure_x else "相对目标", straight=pure_x)

    def _go_absolute(self) -> None:
        pose, origin, _, _, _, _ = self.ros.snapshot()
        if origin is None or pose is None:
            messagebox.showwarning("无法运动", "尚未收到 IMU 融合定位")
            return
        x, y = self.absolute_x.get() / 100.0, self.absolute_y.get() / 100.0
        # Absolute UI coordinates are expressed in the vehicle frame at
        # startup: X is the original forward direction and Y its left side.
        # Convert that target into the odometry/world frame for the controller.
        target_x = origin.x + math.cos(origin.yaw) * x - math.sin(origin.yaw) * y
        target_y = origin.y + math.sin(origin.yaw) * x + math.cos(origin.yaw) * y
        target_yaw = pose.yaw if self.absolute_keep_yaw.get() else wrap_angle(origin.yaw + math.radians(self.absolute_yaw.get()))
        target = Pose2D(target_x, target_y, target_yaw)
        self._start_target(target, "绝对目标")

    def _start_target(self, target, name, straight=False) -> None:
        ok, reason = self.ros.start_target(target, name, self.linear_speed.get(), self.angular_speed.get(), straight=straight)
        if not ok:
            messagebox.showwarning("无法开始", reason)

    def _reset_origin(self) -> None:
        if self.ros.reset_origin():
            self.absolute_x.set(0.0); self.absolute_y.set(0.0); self.absolute_yaw.set(0.0)
        else:
            messagebox.showwarning("无法重设", "尚未收到定位数据")

    def _toggle_record(self) -> None:
        if self.ros.is_recording():
            count = self.ros.end_recording()
            self.record_button.configure(text="开始记录")
            self.plan_text.set(f"已记录 {count} 段运动；回放将从当前车位重复该轨迹")
        elif self.ros.begin_recording():
            self.record_button.configure(text="结束记录")
            self.plan_text.set("记录中：现在可用方向键/按钮驾驶小车")

    def _start_plan(self) -> None:
        # Plan already stores the speed used during recording; scale down if
        # current limits are below that historical speed for a safer replay.
        # Default UI speeds (0.10 m/s and 0.55 rad/s) replay a recording at
        # its original speed. Changing global speed scales every recorded
        # segment, while the hard command limits remain enforced.
        ok, reason = self.ros.start_plan(self.linear_speed.get() / 0.10, self.angular_speed.get() / 0.55)
        if not ok:
            messagebox.showwarning("无法回放", reason)

    def _stop_all(self) -> None:
        self._pressed.clear()
        self.ros.stop_all()

    def _pose_changed(self) -> None:
        pass  # Tk is updated by _refresh on its own thread.

    def _refresh(self) -> None:
        pose, origin, source, stamp, state, name = self.ros.snapshot()
        now = time.monotonic()
        if self.ros.error:
            self.status.set(f"ROS 初始化失败：{self.ros.error}")
        elif pose is None or now - stamp > 1.0:
            self.status.set("等待小车在线：未收到 /tracer5/odometry/filtered（请先在远程机启动底盘驱动）")
        else:
            activity = {"idle": "手动待命", "target": f"正在执行{name}", "plan": "正在回放规划"}[state]
            self.status.set(f"小车在线：定位来源 {source}，{activity}；命令话题 {CMD_TOPIC}")
        if pose and origin:
            world_dx, world_dy = pose.x - origin.x, pose.y - origin.y
            # Display in the initial vehicle frame, not the odometry/world
            # axes, so a pure forward motion changes only X.
            dx = math.cos(origin.yaw) * world_dx + math.sin(origin.yaw) * world_dy
            dy = -math.sin(origin.yaw) * world_dx + math.cos(origin.yaw) * world_dy
            self.pose_text.set(f"相对初始车体原点：X={dx * 100:.1f} cm，Y={dy * 100:.1f} cm，Yaw={math.degrees(wrap_angle(pose.yaw-origin.yaw)):.1f}°")
            self.absolute_current_text.set(
                f"当前绝对坐标：X={dx * 100:.1f} cm，Y={dy * 100:.1f} cm，Yaw={math.degrees(wrap_angle(pose.yaw-origin.yaw)):.1f}°"
            )
        self.after(150, self._refresh)

    def _close(self) -> None:
        self._stop_all()
        self.ros.close()
        self.destroy()


if __name__ == "__main__":
    TracerControlPanel().mainloop()
