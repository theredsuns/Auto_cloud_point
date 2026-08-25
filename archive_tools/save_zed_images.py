#!/usr/bin/env python3
"""Save a remote ZED compressed-image ROS 2 topic on this computer only."""

import argparse
import csv
import os
from pathlib import Path
import select
import shutil
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage


class ZedImageSaver(Node):
    def __init__(
        self,
        topic: str,
        output_dir: Path,
        max_images: int,
        min_free_gb: float,
    ) -> None:
        super().__init__("zed_local_image_saver")
        self.output_dir = output_dir
        self.max_images = max_images
        self.min_free_bytes = int(min_free_gb * 1024**3)
        self.saved_count = 0
        self.received_count = 0
        self.last_received_at = time.monotonic()
        self.pending_captures = 0
        self.finished = False

        self.output_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.output_dir / "capture_log.csv"
        log_is_empty = not log_path.exists() or log_path.stat().st_size == 0
        self.log_file = log_path.open("a", newline="", encoding="utf-8", buffering=1)
        self.log_writer = csv.writer(self.log_file)
        if log_is_empty:
            self.log_writer.writerow(
                [
                    "filename",
                    "ros_stamp_sec",
                    "ros_stamp_nanosec",
                    "received_unix_time",
                    "format",
                    "bytes",
                ]
            )

        self.subscription = self.create_subscription(
            CompressedImage, topic, self.image_callback, qos_profile_sensor_data
        )
        self.status_timer = self.create_timer(5.0, self.report_status)
        self.get_logger().info(f"Listening: {topic}")
        self.get_logger().info(f"Local output: {self.output_dir}")
        self.get_logger().info(
            f"Manual capture; maximum: {self.max_images or 'unlimited'} images"
        )

    def image_callback(self, message: CompressedImage) -> None:
        self.received_count += 1
        self.last_received_at = time.monotonic()
        if self.finished or self.pending_captures == 0:
            return

        image_format = message.format.lower()
        if "jpeg" in image_format or "jpg" in image_format:
            extension = ".jpg"
        elif "png" in image_format:
            extension = ".png"
        else:
            self.get_logger().warning(
                f"Unsupported compressed image format '{message.format}'; frame skipped"
            )
            return

        free_bytes = shutil.disk_usage(self.output_dir).free
        if free_bytes < self.min_free_bytes:
            self.get_logger().error(
                "Saving stopped: local free space is below the configured minimum "
                f"({free_bytes / 1024**3:.2f} GiB remaining)"
            )
            self.finished = True
            return

        stamp = message.header.stamp
        filename = (
            f"zed_{stamp.sec:010d}_{stamp.nanosec:09d}_"
            f"{self.saved_count + 1:06d}{extension}"
        )
        destination = self.output_dir / filename
        temporary = self.output_dir / f".{filename}.part"

        try:
            with temporary.open("wb") as image_file:
                image_file.write(bytes(message.data))
            os.replace(temporary, destination)
        except OSError as error:
            self.get_logger().error(f"Could not save {destination}: {error}")
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            return

        self.saved_count += 1
        self.pending_captures -= 1
        self.log_writer.writerow(
            [
                filename,
                stamp.sec,
                stamp.nanosec,
                f"{time.time():.6f}",
                message.format,
                len(message.data),
            ]
        )

        self.get_logger().info(
            f"Saved #{self.saved_count}: {filename} "
            f"({free_bytes / 1024**3:.2f} GiB free locally)"
        )

        if self.max_images and self.saved_count >= self.max_images:
            self.get_logger().info(
                f"Reached the requested limit of {self.max_images} images"
            )
            self.finished = True

    def request_capture(self) -> None:
        if self.finished:
            return
        self.pending_captures += 1
        self.get_logger().info(
            f"Capture requested; waiting for a new ZED frame "
            f"(pending: {self.pending_captures})"
        )

    def report_status(self) -> None:
        seconds_without_data = time.monotonic() - self.last_received_at
        if self.received_count == 0 or seconds_without_data > 5.0:
            self.get_logger().warning(
                "No image received recently; check ROS_DOMAIN_ID, the remote scanner, "
                "and /zed_scanner/left_image/compressed"
            )

    def close(self) -> None:
        self.log_file.close()


def parse_arguments() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Save /zed_scanner/left_image/compressed on the local computer."
    )
    parser.add_argument(
        "--topic",
        default="/zed_scanner/left_image/compressed",
        help="sensor_msgs/msg/CompressedImage topic",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/home/nkk/object_seek/captured_images"),
        help="local output directory",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="stop after this many images; 0 means unlimited",
    )
    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=2.0,
        help="stop before local free space falls below this value",
    )
    arguments, ros_arguments = parser.parse_known_args()
    if arguments.max_images < 0:
        parser.error("--max-images must be non-negative")
    if arguments.min_free_gb < 0:
        parser.error("--min-free-gb must be non-negative")
    return arguments, ros_arguments


def main() -> int:
    arguments, ros_arguments = parse_arguments()
    output_dir = arguments.output.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    ros_log_dir = output_dir / ".ros_logs"
    ros_log_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("ROS_LOG_DIR", str(ros_log_dir))
    rclpy.init(args=ros_arguments)
    node = ZedImageSaver(
        topic=arguments.topic,
        output_dir=output_dir,
        max_images=arguments.max_images,
        min_free_gb=arguments.min_free_gb,
    )
    print("\n[Enter] capture next ZED frame    [q + Enter] quit\n", flush=True)
    try:
        while rclpy.ok() and not node.finished:
            readable, _, _ = select.select([sys.stdin], [], [], 0.1)
            if readable:
                command = sys.stdin.readline()
                if command == "" or command.strip().lower() == "q":
                    break
                if command.strip() == "":
                    node.request_capture()
                else:
                    print("Press Enter to capture, or type q then Enter to quit.", flush=True)
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info(
            f"Stopped. Received {node.received_count}, saved {node.saved_count} images."
        )
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
