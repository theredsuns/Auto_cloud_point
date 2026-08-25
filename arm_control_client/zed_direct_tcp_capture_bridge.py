#!/usr/bin/env python3
"""Minimal ZED RGB/depth TCP server for the local capture panel.

No ROS nodes, point-cloud generation, mapping, mesh, or GUI are started on
the AGX.  The protocol matches RemoteCapture: L/C -> JPEG + float32 depth.
"""
from __future__ import annotations

import socket
import struct
import threading
import zlib

import cv2
import numpy as np
import pyzed.sl as sl

HOST, PORT = "127.0.0.1", 47777


class DirectZed:
    def __init__(self):
        self.lock = threading.Lock()
        self.camera = sl.Camera()
        self.image, self.depth = sl.Mat(), sl.Mat()
        init = sl.InitParameters()
        init.camera_resolution = sl.RESOLUTION.HD720
        init.depth_mode = sl.DEPTH_MODE.NEURAL
        init.coordinate_units = sl.UNIT.METER
        if self.camera.open(init) != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError("ZED camera open failed")
        self.runtime = sl.RuntimeParameters()

    def frame(self, include_depth):
        with self.lock:
            if self.camera.grab(self.runtime) != sl.ERROR_CODE.SUCCESS:
                return None, None
            self.camera.retrieve_image(self.image, sl.VIEW.LEFT)
            bgra = self.image.get_data()
            if bgra is None:
                return None, None
            ok, encoded = cv2.imencode(".jpg", bgra[:, :, :3], [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ok:
                return None, None
            if not include_depth:
                return bytes(encoded), None
            self.camera.retrieve_measure(self.depth, sl.MEASURE.DEPTH)
            depth = self.depth.get_data()
            return bytes(encoded), None if depth is None else np.asarray(depth, dtype=np.float32).copy()

    def close(self):
        self.camera.close()


def send_frame(connection, zed, include_depth):
    jpeg, depth = zed.frame(include_depth)
    if jpeg is None or (include_depth and depth is None):
        connection.sendall(b"N")
        return
    raw_depth = zlib.compress(depth.tobytes(), 1) if depth is not None else b""
    connection.sendall(b"O" + struct.pack("!II", len(jpeg), len(raw_depth)) + jpeg + raw_depth)


def client_loop(connection, zed):
    with connection:
        connection.settimeout(15)
        while True:
            try:
                command = connection.recv(1)
            except OSError:
                return
            if command not in {b"L", b"C"}:
                return
            try:
                send_frame(connection, zed, command == b"C")
            except OSError:
                return


def main():
    zed = DirectZed()
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT)); server.listen(3)
    try:
        while True:
            connection, _address = server.accept()
            threading.Thread(target=client_loop, args=(connection, zed), daemon=True).start()
    finally:
        server.close(); zed.close()


if __name__ == "__main__":
    main()
