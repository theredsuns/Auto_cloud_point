#!/usr/bin/env python3
"""Remote half of the Wi-Fi SSH bridge for the Tracer5 control panel.

It is intentionally small: JSON twist commands arrive on stdin, while fused
odometry is returned as newline-delimited JSON on stdout.  It must run on the
vehicle computer because Wi-Fi client isolation prevents DDS from reaching the
laptop directly.
"""

from __future__ import annotations

import json
import os
import queue
import select
import sys
import time


def yaw_from_quaternion(q) -> float:
    import math
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def main() -> None:
    os.environ.setdefault("ROS_DOMAIN_ID", "36")
    os.environ.setdefault("ROS_LOCALHOST_ONLY", "0")
    os.environ.setdefault("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")
    os.environ.setdefault("FASTRTPS_DEFAULT_PROFILES_FILE", os.path.expanduser("~/.fastdds/fastdds_select5G.xml"))
    os.environ.setdefault("RMW_FASTRTPS_USE_QOS_FROM_XML", "1")

    import rclpy
    from geometry_msgs.msg import Twist
    from nav_msgs.msg import Odometry
    from rclpy.node import Node

    commands: queue.Queue[tuple[float, float]] = queue.Queue()

    def stdin_reader() -> None:
        for line in sys.stdin:
            try:
                payload = json.loads(line)
                if payload.get("type") == "twist":
                    commands.put((float(payload["linear"]), float(payload["angular"])))
            except (ValueError, TypeError, KeyError, json.JSONDecodeError):
                continue

    rclpy.init()
    node = Node("tracer5_ssh_control_bridge")
    publisher = node.create_publisher(Twist, "/tracer5/cmd_vel", 10)
    latest = {"pose": None, "source": "", "filtered_at": 0.0, "printed_at": 0.0}

    def receive(msg: Odometry, source: str) -> None:
        now = time.monotonic()
        if source == "raw" and now - latest["filtered_at"] < 1.0:
            return
        pose = msg.pose.pose
        latest["pose"] = (float(pose.position.x), float(pose.position.y), yaw_from_quaternion(pose.orientation))
        latest["source"] = source
        if source == "filtered":
            latest["filtered_at"] = now

    node.create_subscription(Odometry, "/tracer5/odometry/filtered", lambda msg: receive(msg, "filtered"), 20)
    node.create_subscription(Odometry, "/tracer5/odom", lambda msg: receive(msg, "raw"), 10)
    import threading
    threading.Thread(target=stdin_reader, daemon=True).start()

    linear = angular = 0.0
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.01)
            try:
                while True:
                    linear, angular = commands.get_nowait()
            except queue.Empty:
                pass
            message = Twist()
            message.linear.x, message.angular.z = linear, angular
            publisher.publish(message)
            now = time.monotonic()
            if latest["pose"] is not None and now - latest["printed_at"] >= 0.08:
                x, y, yaw = latest["pose"]
                print(json.dumps({"type": "pose", "x": x, "y": y, "yaw": yaw, "source": latest["source"]}), flush=True)
                latest["printed_at"] = now
            time.sleep(0.04)
    finally:
        for _ in range(5):
            publisher.publish(Twist())
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
