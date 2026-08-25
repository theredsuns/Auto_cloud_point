#!/usr/bin/env python3
"""Expose a remote PiPER-L as ROS JointState topics through SSH.

This is intentionally a small, guarded bridge for RViz joint sliders. It does
not enable the arm. Motion is disabled unless --allow-motion is supplied.
"""
import argparse
import json
import math
import re
import shlex
import subprocess
import threading

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.node import Node
from control_msgs.action import FollowJointTrajectory
from sensor_msgs.msg import JointState

REMOTE = "skki@192.168.50.55"
REMOTE_DIR = "~/zed_code/arm_control"
JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]


class RemotePiperBridge(Node):
    def __init__(self, allow_motion, moveit):
        super().__init__("remote_piper_ssh_bridge")
        self.allow_motion = allow_motion
        self.current = None
        self.control_base = None
        self.slider_zero = None
        self.last_sent = None
        self.busy = False
        self.poll_busy = False
        self.stream_proc = None
        self.feedback_pub = self.create_publisher(JointState, "feedback/joint_states", 10)
        self.create_subscription(JointState, "control/joint_states", self.control_callback, 10)
        self.stream_thread = threading.Thread(target=self._stream_worker, daemon=True)
        self.stream_thread.start()
        self.trajectory_server = None
        if moveit:
            self.trajectory_server = ActionServer(
                self, FollowJointTrajectory, "arm_controller/follow_joint_trajectory",
                execute_callback=self.execute_trajectory,
                goal_callback=self.accept_trajectory,
                cancel_callback=lambda _: CancelResponse.ACCEPT)
        self.get_logger().info("Remote PiPER bridge ready; motion=" + ("enabled" if allow_motion else "disabled"))

    @staticmethod
    def ssh(command, timeout=12):
        return subprocess.run(
            ["ssh", "-o", "BatchMode=yes", REMOTE, command],
            text=True, capture_output=True, timeout=timeout)

    def _stream_worker(self):
        """Maintain one SSH connection so MoveIt receives fresh joint states."""
        while rclpy.ok():
            try:
                self.stream_proc = subprocess.Popen(
                    ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", REMOTE,
                     f"cd {REMOTE_DIR} && exec python3 piper_status_stream.py"],
                    text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
                for line in self.stream_proc.stdout:
                    try:
                        payload = json.loads(line)
                        values = [math.radians(value / 1000.0) for value in payload["joint_mdeg"]]
                        if len(values) != 6:
                            continue
                        self.current = values
                        msg = JointState()
                        msg.header.stamp = self.get_clock().now().to_msg()
                        msg.name = JOINTS
                        msg.position = values
                        self.feedback_pub.publish(msg)
                    except (ValueError, KeyError, TypeError):
                        continue
            except Exception as exc:
                self.get_logger().warning(f"State stream error: {exc}")
            time.sleep(1.0)

    def poll_status(self):
        if self.busy or self.poll_busy:
            return
        self.poll_busy = True
        threading.Thread(target=self._poll_worker, daemon=True).start()

    def close(self):
        if self.stream_proc and self.stream_proc.poll() is None:
            self.stream_proc.terminate()

    def _poll_worker(self):
        try:
            result = self.ssh(f"cd {REMOTE_DIR} && python3 piper_status.py --can can1 --seconds 0.1")
            match = re.search(r"joint_angles_raw_mdeg:\s*\[([^]]+)\]", result.stdout)
            if result.returncode or not match:
                detail = (result.stderr or result.stdout).strip().replace("\n", " ")[-240:]
                self.get_logger().warning("Remote feedback unavailable: " + detail)
                return
            values = [math.radians(float(x.strip()) / 1000.0) for x in match.group(1).split(",")]
            if len(values) != 6:
                return
            self.current = values
            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name = JOINTS
            msg.position = values
            self.feedback_pub.publish(msg)
        except Exception as exc:
            self.get_logger().warning(f"Remote feedback error: {exc}")
        finally:
            self.poll_busy = False

    def control_callback(self, message):
        if not self.allow_motion or self.current is None or self.busy:
            return
        requested = dict(zip(message.name, message.position))
        if not all(name in requested for name in JOINTS):
            return
        slider_values = [requested[name] for name in JOINTS]
        # joint_state_publisher_gui may initialize from the URDF with nonzero
        # angles. Capture that first message as its neutral position.
        if self.slider_zero is None:
            self.slider_zero = list(slider_values)
            self.control_base = list(self.current)
            self.get_logger().info("Captured current slider positions as neutral; move one slider slightly to command motion")
            return
        offsets = [value - neutral for value, neutral in zip(slider_values, self.slider_zero)]
        # Interpret slider changes as relative displacement, never an absolute
        # all-zero/home target.
        if max(abs(math.degrees(value)) for value in offsets) > 5.0:
            self.get_logger().warning("Ignored slider value: each relative offset must stay within +/-5 degrees")
            return
        if max(abs(value) for value in offsets) < math.radians(0.02):
            self.control_base = list(self.current)
        target = [base + offset for base, offset in zip(self.control_base, offsets)]
        if self.last_sent and max(abs(a - b) for a, b in zip(target, self.last_sent)) < math.radians(0.1):
            return
        self.last_sent = target
        threading.Thread(target=self._send_worker, args=(target,), daemon=True).start()

    def _send_worker(self, target):
        self.busy = True
        try:
            degrees = " ".join(f"{math.degrees(value):.3f}" for value in target)
            command = (
                f"cd {REMOTE_DIR} && python3 piper_safe_move.py --can can1 "
                f"--joint-deg {degrees} --speed 5 --execute --confirm ARM_CLEAR"
            )
            result = self.ssh(command)
            if result.returncode:
                self.get_logger().error("Remote command rejected: " + (result.stderr or result.stdout).strip()[-300:])
            else:
                self.get_logger().info("Sent guarded 5% joint target")
        except Exception as exc:
            self.get_logger().error(f"Remote command error: {exc}")
        finally:
            self.busy = False

    def accept_trajectory(self, goal):
        if not self.allow_motion or self.current is None or self.busy:
            return GoalResponse.REJECT
        if not goal.trajectory.points or not all(name in goal.trajectory.joint_names for name in JOINTS):
            return GoalResponse.REJECT
        final = dict(zip(goal.trajectory.joint_names, goal.trajectory.points[-1].positions))
        target = [final[name] for name in JOINTS]
        if max(abs(math.degrees(a - b)) for a, b in zip(target, self.current)) > 5.0:
            self.get_logger().warning("Rejected MoveIt plan: final target exceeds 5 degrees per joint")
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def execute_trajectory(self, goal_handle):
        request = goal_handle.request.trajectory
        final = dict(zip(request.joint_names, request.points[-1].positions))
        target = [final[name] for name in JOINTS]
        self.busy = True
        result = FollowJointTrajectory.Result()
        try:
            degrees = " ".join(f"{math.degrees(value):.3f}" for value in target)
            command = (f"cd {REMOTE_DIR} && python3 piper_safe_move.py --can can1 "
                       f"--joint-deg {degrees} --speed 5 --execute --confirm ARM_CLEAR")
            reply = self.ssh(command, timeout=20)
            if reply.returncode:
                result.error_code = FollowJointTrajectory.Result.INVALID_GOAL
                result.error_string = (reply.stderr or reply.stdout).strip()[-500:]
                goal_handle.abort()
            else:
                result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
                goal_handle.succeed()
        except Exception as exc:
            result.error_code = FollowJointTrajectory.Result.INVALID_GOAL
            result.error_string = str(exc)
            goal_handle.abort()
        finally:
            self.busy = False
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-motion", action="store_true", help="allow guarded commands from RViz sliders")
    parser.add_argument("--moveit", action="store_true", help="accept guarded MoveIt end-effector plans")
    args = parser.parse_args()
    rclpy.init()
    node = RemotePiperBridge(args.allow_motion, args.moveit)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
