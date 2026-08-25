#!/usr/bin/env python3
"""Display the compressed left image published by the remote ZED ROS2 node."""

import os
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CompressedImage


LOG_DIR = Path(__file__).resolve().parent / ".ros_logs"
LOG_DIR.mkdir(exist_ok=True)
os.environ.setdefault("ROS_LOG_DIR", str(LOG_DIR))


class RemoteZedViewer(Node):
    def __init__(self):
        super().__init__("remote_zed_viewer")
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            # Keep only the newest camera frame.  For a live preview, a
            # dropped old frame is preferable to seconds of accumulated lag.
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        self.subscription = self.create_subscription(
            CompressedImage, "/zed_scanner/left_image/compressed", self.show, qos
        )
        self.frames = 0

    def show(self, message):
        image = cv2.imdecode(np.frombuffer(message.data, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return
        self.frames += 1
        cv2.putText(image, f"Remote ZED | frames {self.frames} | Q quit", (14, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("Remote ZED live", image)
        if (cv2.waitKey(1) & 0xFF) in (27, ord("q"), ord("Q")):
            rclpy.shutdown()


def main():
    rclpy.init()
    node = RemoteZedViewer()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
