#!/usr/bin/env python3
"""Preview a remote ZED stream and manually save object-reconstruction images."""

import argparse
import csv
import os
from pathlib import Path
import select
import shutil
import sys
import time
from typing import Any, Optional

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage

LOCAL_YOLO_RUNTIME = Path(__file__).resolve().parent / "yolo_runtime"
if LOCAL_YOLO_RUNTIME.is_dir():
    # Insert after importing the ROS/OpenCV system packages. Their compatible
    # NumPy build is already loaded, while YOLO can use its newer local helpers.
    sys.path.insert(0, str(LOCAL_YOLO_RUNTIME))
os.environ.setdefault(
    "YOLO_CONFIG_DIR", str(Path(__file__).resolve().parent / ".ultralytics")
)
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib")
)


class ObjectCapture(Node):
    def __init__(self, arguments: argparse.Namespace) -> None:
        super().__init__("zed_object_capture")
        self.arguments = arguments
        self.root = arguments.output.expanduser().resolve()
        self.image_dir = self.root / "images"
        self.mask_dir = self.root / "masks"
        self.overlay_dir = self.root / "overlays"
        for directory in (self.image_dir, self.mask_dir, self.overlay_dir):
            directory.mkdir(parents=True, exist_ok=True)

        self.model: Optional[Any] = None
        if arguments.model is not None:
            try:
                from ultralytics import YOLO
            except ImportError as error:
                raise RuntimeError(
                    "A YOLO model was requested, but ultralytics is not installed. "
                    "Install it in a local virtual environment first."
                ) from error
            self.model = YOLO(str(arguments.model.expanduser().resolve()))

        self.latest_message: Optional[CompressedImage] = None
        self.latest_image: Optional[np.ndarray] = None
        self.latest_mask: Optional[np.ndarray] = None
        self.latest_overlay: Optional[np.ndarray] = None
        self.pending_capture = False
        self.received_count = 0
        self.saved_count = 0
        self.finished = False
        self.last_received_at = time.monotonic()
        self.min_free_bytes = int(arguments.min_free_gb * 1024**3)

        log_path = self.root / "capture_log.csv"
        new_log = not log_path.exists() or log_path.stat().st_size == 0
        self.log_file = log_path.open("a", newline="", encoding="utf-8", buffering=1)
        self.log_writer = csv.writer(self.log_file)
        if new_log:
            self.log_writer.writerow(
                [
                    "filename",
                    "ros_stamp_sec",
                    "ros_stamp_nanosec",
                    "has_yolo_mask",
                    "bytes",
                ]
            )

        self.subscription = self.create_subscription(
            CompressedImage,
            arguments.topic,
            self.image_callback,
            qos_profile_sensor_data,
        )
        self.status_timer = self.create_timer(5.0, self.report_status)
        mode = (
            f"YOLO recognition ({arguments.model})"
            if self.model is not None
            else "capture-only (no YOLO model)"
        )
        self.get_logger().info(f"Mode: {mode}")
        self.get_logger().info(f"Topic: {arguments.topic}")
        self.get_logger().info(f"Local dataset: {self.root}")

    def infer_objects(
        self, image: np.ndarray
    ) -> tuple[Optional[np.ndarray], list[tuple[int, int, int, int, int, float, str]], str]:
        if self.model is None:
            return None, [], "NO MODEL - capture images for labeling"

        results = self.model.predict(
            source=image,
            conf=self.arguments.confidence,
            imgsz=self.arguments.imgsz,
            device=self.arguments.device,
            verbose=False,
        )
        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return None, [], "TARGET NOT FOUND"

        coordinates = result.boxes.xyxy.detach().cpu().numpy()
        classes = result.boxes.cls.detach().cpu().numpy().astype(int)
        confidences = result.boxes.conf.detach().cpu().numpy()
        candidates = [
            index
            for index, class_id in enumerate(classes)
            if self.arguments.class_id < 0 or class_id == self.arguments.class_id
        ]
        if not candidates:
            return None, [], "TARGET CLASS NOT FOUND"

        detections = []
        for index in candidates:
            x1, y1, x2, y2 = coordinates[index].round().astype(int)
            class_id = int(classes[index])
            class_name = result.names.get(class_id, str(class_id))
            detections.append(
                (x1, y1, x2, y2, class_id, float(confidences[index]), class_name)
            )

        binary_mask = None
        if result.masks is not None:
            masks = result.masks.data.detach().cpu().numpy()
            selected = max(
                candidates, key=lambda index: int(np.count_nonzero(masks[index]))
            )
            mask = cv2.resize(
                masks[selected],
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            binary_mask = np.where(mask > 0.5, 255, 0).astype(np.uint8)

        return binary_mask, detections, f"TARGETS: {len(detections)}"

    def image_callback(self, message: CompressedImage) -> None:
        self.received_count += 1
        self.last_received_at = time.monotonic()
        encoded = np.frombuffer(message.data, dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image is None:
            self.get_logger().warning("Could not decode a compressed ZED frame")
            return

        mask, detections, label = self.infer_objects(image)
        overlay = image.copy()
        if mask is not None:
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(overlay, contours, -1, (0, 255, 0), 3, cv2.LINE_AA)
        for x1, y1, x2, y2, _class_id, confidence, class_name in detections:
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.drawMarker(
                overlay,
                (center_x, center_y),
                (0, 0, 255),
                cv2.MARKER_CROSS,
                24,
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                overlay,
                f"{class_name} {confidence:.2f} center=({center_x},{center_y})",
                (x1, max(25, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
        cv2.putText(
            overlay,
            label,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0) if detections else (0, 180, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            "Enter: save locally | Q: quit",
            (20, overlay.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        self.latest_message = message
        self.latest_image = image
        self.latest_mask = mask
        self.latest_overlay = overlay

        cv2.imshow("ZED Object Capture", overlay)
        key = cv2.waitKey(1) & 0xFF
        if key in (10, 13):
            self.save_latest()
        elif key in (ord("q"), ord("Q"), 27):
            self.finished = True

        if self.pending_capture:
            self.pending_capture = False
            self.save_latest()

    def request_capture(self) -> None:
        self.pending_capture = True
        self.get_logger().info("Waiting for the next new ZED frame...")

    def save_latest(self) -> None:
        if self.latest_message is None or self.latest_image is None:
            self.get_logger().warning("No ZED frame is available yet")
            return
        free_bytes = shutil.disk_usage(self.root).free
        if free_bytes < self.min_free_bytes:
            self.get_logger().error(
                f"Local saving stopped: only {free_bytes / 1024**3:.2f} GiB is free"
            )
            self.finished = True
            return

        stamp = self.latest_message.header.stamp
        filename = (
            f"zed_{stamp.sec:010d}_{stamp.nanosec:09d}_"
            f"{self.saved_count + 1:06d}.jpg"
        )
        image_path = self.image_dir / filename
        overlay_path = self.overlay_dir / filename
        image_ok = cv2.imwrite(
            str(image_path),
            self.latest_image,
            [cv2.IMWRITE_JPEG_QUALITY, self.arguments.jpeg_quality],
        )
        overlay_ok = cv2.imwrite(
            str(overlay_path),
            self.latest_overlay,
            [cv2.IMWRITE_JPEG_QUALITY, self.arguments.jpeg_quality],
        )
        if not image_ok or not overlay_ok:
            self.get_logger().error(f"Could not save {filename}")
            return

        has_mask = self.latest_mask is not None
        if has_mask:
            # COLMAP expects image.jpg.png for a corresponding image.jpg mask.
            cv2.imwrite(str(self.mask_dir / f"{filename}.png"), self.latest_mask)

        self.saved_count += 1
        self.log_writer.writerow(
            [
                filename,
                stamp.sec,
                stamp.nanosec,
                int(has_mask),
                len(self.latest_message.data),
            ]
        )
        self.get_logger().info(
            f"Saved #{self.saved_count}: {filename}; "
            f"mask={'yes' if has_mask else 'no'}"
        )
        if (
            self.arguments.max_images
            and self.saved_count >= self.arguments.max_images
        ):
            self.finished = True

    def report_status(self) -> None:
        if time.monotonic() - self.last_received_at > 5.0:
            self.get_logger().warning(
                "No ZED image received; check the remote scanner and ROS_DOMAIN_ID"
            )

    def close(self) -> None:
        self.log_file.close()
        cv2.destroyAllWindows()


def parse_arguments() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Recognize YOLO objects and save images only on this computer."
    )
    parser.add_argument(
        "--topic", default="/zed_scanner/left_image/compressed"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/home/nkk/object_seek/datasets/object_01"),
    )
    parser.add_argument(
        "--model",
        type=Path,
        help="custom Ultralytics YOLO detection or segmentation weights, e.g. best.pt",
    )
    parser.add_argument(
        "--class-id",
        type=int,
        default=-1,
        help="target class ID; -1 selects the largest detected instance",
    )
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--min-free-gb", type=float, default=5.0)
    arguments, ros_arguments = parser.parse_known_args()
    if not 0.0 < arguments.confidence <= 1.0:
        parser.error("--confidence must be in (0, 1]")
    if not 1 <= arguments.jpeg_quality <= 100:
        parser.error("--jpeg-quality must be between 1 and 100")
    if arguments.max_images < 0 or arguments.min_free_gb < 0:
        parser.error("--max-images and --min-free-gb must be non-negative")
    return arguments, ros_arguments


def main() -> int:
    arguments, ros_arguments = parse_arguments()
    root = arguments.output.expanduser().resolve()
    ros_log_dir = root / ".ros_logs"
    ros_log_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("ROS_LOG_DIR", str(ros_log_dir))
    rclpy.init(args=ros_arguments)
    try:
        node = ObjectCapture(arguments)
    except Exception as error:
        print(f"Cannot start object capture: {error}", file=sys.stderr)
        rclpy.shutdown()
        return 1

    print(
        "\nFocus the image window and press Enter to save. "
        "Terminal Enter also saves the next frame. Type q to quit.\n",
        flush=True,
    )
    try:
        while rclpy.ok() and not node.finished:
            readable, _, _ = select.select([sys.stdin], [], [], 0.05)
            if readable:
                command = sys.stdin.readline()
                if command == "" or command.strip().lower() == "q":
                    break
                if command.strip() == "":
                    node.request_capture()
            rclpy.spin_once(node, timeout_sec=0.05)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info(
            f"Stopped. Received {node.received_count}, saved {node.saved_count}."
        )
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
