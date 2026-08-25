#!/usr/bin/env python3
"""Live Tracer5 state monitor: filtered XYZ/RPY plus raw IMU acceleration."""
from __future__ import annotations

import math
import os
import time

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Imu


def quaternion_to_rpy_degrees(x: float, y: float, z: float, w: float) -> tuple[float, float, float]:
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch_term = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch = math.asin(pitch_term)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return tuple(math.degrees(value) for value in (roll, pitch, yaw))


class TracerStateMonitor(Node):
    def __init__(self) -> None:
        super().__init__("tracer5_state_monitor")
        self.odom: Odometry | None = None
        self.imu: Imu | None = None
        self.create_subscription(Odometry, "/tracer5/odometry/filtered", self._on_odom, 10)
        self.create_subscription(Imu, "/tracer5/IMU_data", self._on_imu, 10)
        self.create_timer(0.1, self._print_state)

    def _on_odom(self, message: Odometry) -> None:
        self.odom = message

    def _on_imu(self, message: Imu) -> None:
        self.imu = message

    def _print_state(self) -> None:
        if self.odom is None:
            return
        position = self.odom.pose.pose.position
        orientation = self.odom.pose.pose.orientation
        roll, pitch, yaw = quaternion_to_rpy_degrees(
            orientation.x, orientation.y, orientation.z, orientation.w
        )
        if self.imu is None:
            acceleration = "waiting for IMU"
        else:
            linear = self.imu.linear_acceleration
            acceleration = f"ax={linear.x:+.3f}  ay={linear.y:+.3f}  az={linear.z:+.3f} m/s²"
        line = (
            f"XYZ (m): x={position.x:+.3f}  y={position.y:+.3f}  z={position.z:+.3f}"
            f" | RPY (deg): roll={roll:+.1f}  pitch={pitch:+.1f}  yaw={yaw:+.1f}"
            f" | IMU accel: {acceleration}"
        )
        print("\r" + line.ljust(180), end="", flush=True)


def main() -> None:
    print(f"ROS_DOMAIN_ID={os.environ.get('ROS_DOMAIN_ID', '0')} | Ctrl+C to quit")
    rclpy.init()
    node = TracerStateMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        print()


if __name__ == "__main__":
    main()
