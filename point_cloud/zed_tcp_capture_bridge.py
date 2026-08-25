#!/usr/bin/env python3
"""Run on the ZED computer: carry its local ROS images over one SSH tunnel.

Protocol is deliberately tiny: L returns the current JPEG, C returns a JPEG
and a requested float32 depth image.  This avoids ROS 2 multicast discovery
between the two computers.
"""
from __future__ import annotations

import socket
import struct
import threading
import time
import zlib

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Empty

HOST, PORT = "127.0.0.1", 47777


class Bridge(Node):
    def __init__(self):
        super().__init__("zed_tcp_capture_bridge")
        qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1,
                         reliability=ReliabilityPolicy.BEST_EFFORT)
        self.jpeg: bytes | None = None
        self.capture_jpeg: bytes | None = None
        self.depth: np.ndarray | None = None
        # A ZED wrapper may already own the physical camera (for example
        # when the calibrated TCP stack is running).  Reuse that stream
        # instead of opening the camera a second time, which causes
        # "CAMERA STREAM FAILED TO START".
        self.wrapper_jpeg: bytes | None = None
        self.wrapper_depth: np.ndarray | None = None
        self.preview_depth: np.ndarray | None = None
        self.waiting_for_depth = threading.Event()
        self.depth_ready = threading.Event()
        self.create_subscription(CompressedImage, "/zed_scanner/left_image/compressed", self.image_cb, qos)
        self.create_subscription(CompressedImage, "/zed_scanner/capture_image/compressed", self.capture_image_cb, qos)
        self.create_subscription(Image, "/zed_scanner/depth_image", self.depth_cb, qos)
        self.create_subscription(CompressedImage, "/zed/zed_node/left/color/rect/image/compressed", self.wrapper_image_cb, qos)
        self.create_subscription(Image, "/zed/zed_node/depth/depth_registered", self.wrapper_depth_cb, qos)
        self.capture_pub = self.create_publisher(Empty, "/zed_scanner/capture_request", 1)

    def image_cb(self, msg):
        self.jpeg = bytes(msg.data)

    def wrapper_image_cb(self, msg):
        self.wrapper_jpeg = bytes(msg.data)
        # Live display works through exactly the same tunnel.
        self.jpeg = self.wrapper_jpeg

    def wrapper_depth_cb(self, msg):
        if msg.encoding != "32FC1":
            return
        values = np.frombuffer(msg.data, dtype=np.float32)
        if values.size == msg.width * msg.height:
            self.wrapper_depth = values.reshape(msg.height, msg.width).copy()
            self.preview_depth = self.wrapper_depth

    def capture_image_cb(self, msg):
        self.capture_jpeg = bytes(msg.data)
        if self.depth is not None:
            self.depth_ready.set()

    def depth_cb(self, msg):
        if not self.waiting_for_depth.is_set() or msg.encoding != "32FC1":
            return
        values = np.frombuffer(msg.data, dtype=np.float32)
        if values.size == msg.width * msg.height:
            self.depth = values.reshape(msg.height, msg.width).copy()
            self.waiting_for_depth.clear()
            if self.capture_jpeg is not None:
                self.depth_ready.set()

    def capture(self):
        # Do not start a second ZED instance when the wrapper already owns
        # it. The most recent synchronized-enough RGB/depth pair is the
        # available native wrapper frame and is sufficient for a still scan.
        if self.wrapper_jpeg is not None and self.wrapper_depth is not None:
            return self.wrapper_jpeg, self.wrapper_depth
        self.depth = None
        self.capture_jpeg = None
        self.depth_ready.clear()
        self.waiting_for_depth.set()
        self.capture_pub.publish(Empty())
        self.depth_ready.wait(5.0)
        self.waiting_for_depth.clear()
        return self.capture_jpeg, self.depth


def send_packet(conn: socket.socket, jpeg: bytes | None, depth: np.ndarray | None = None):
    if jpeg is None:
        conn.sendall(b"N")
        return
    raw_depth = b"" if depth is None else zlib.compress(depth.astype(np.float32).tobytes(), 1)
    conn.sendall(b"O" + struct.pack("!II", len(jpeg), len(raw_depth)) + jpeg + raw_depth)


def serve_client(conn: socket.socket, node: Bridge):
    """Serve one tunnel client without stalling preview/capture reconnects."""
    with conn:
        conn.settimeout(2.0)
        while rclpy.ok():
            try:
                command = conn.recv(1)
            except socket.timeout:
                break
            if not command:
                break
            if command == b"L":
                try:
                    send_packet(conn, node.jpeg, node.preview_depth)
                except OSError:
                    break
            elif command == b"C":
                try:
                    jpeg, depth = node.capture()
                    send_packet(conn, jpeg, depth)
                except OSError:
                    break
            else:
                break


def serve(node: Bridge):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((HOST, PORT)); server.listen(3)
        server.settimeout(1.0)
        print(f"[ready] SSH TCP bridge at {HOST}:{PORT}", flush=True)
        while rclpy.ok():
            try:
                conn, _ = server.accept()
            except socket.timeout:
                continue
            threading.Thread(target=serve_client, args=(conn, node), daemon=True).start()


def main():
    rclpy.init(); node = Bridge()
    executor = rclpy.executors.MultiThreadedExecutor(); executor.add_node(node)
    thread = threading.Thread(target=executor.spin, daemon=True); thread.start()
    try:
        serve(node)
    finally:
        executor.shutdown()
        node.destroy_node()
        # A launch/signal shutdown can already have closed the default
        # context; do not turn that normal condition into a bridge crash.
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
