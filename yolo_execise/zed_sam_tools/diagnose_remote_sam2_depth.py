#!/usr/bin/env python3
"""One-shot, headless diagnostic for a synchronized remote ZED RGB/cloud pair."""

import argparse
from pathlib import Path

import cv2
import message_filters
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, PointCloud2

from remote_zed_yolo_sam2_icp import cloud_for_mask
from zed_yolo_sam2_icp_reconstruct import select_target
from zed_yolo_sam2_live import ROOT, load_sam2, load_yolo


class Diagnostic(Node):
    def __init__(self, args):
        super().__init__("remote_sam2_depth_diagnostic")
        self.args = args
        self.yolo = load_yolo(args.weights)
        self.sam2 = load_sam2(args.sam2_config, args.sam2_checkpoint)
        image = message_filters.Subscriber(
            self, CompressedImage, args.image_topic, qos_profile=qos_profile_sensor_data
        )
        cloud = message_filters.Subscriber(
            self, PointCloud2, args.cloud_topic, qos_profile=qos_profile_sensor_data
        )
        sync = message_filters.ApproximateTimeSynchronizer([image, cloud], 2, 0.20)
        sync.registerCallback(self.callback)
        self.sync = sync
        self.done = False

    def callback(self, image_message, cloud_message):
        if self.done:
            return
        self.done = True
        bgr = cv2.imdecode(np.frombuffer(image_message.data, np.uint8), cv2.IMREAD_COLOR)
        result = self.yolo.predict(bgr, conf=self.args.confidence, verbose=False)[0]
        selected = select_target(
            result, self.sam2, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), self.args.target_class
        )
        if selected is None:
            print("RESULT: no matching YOLO + SAM2 target", flush=True)
        else:
            box, score, label, mask = selected
            points, _colors = cloud_for_mask(cloud_message, mask, bgr)
            print(f"RESULT: label={label}; confidence={score:.3f}; mask_pixels={int(mask.sum())}; depth_points={len(points)}", flush=True)
            x1, y1, x2, y2 = np.round(box).astype(int)
            overlay = bgr.copy()
            overlay[mask] = (overlay[mask] * 0.5 + np.array((0, 255, 0)) * 0.5).astype(np.uint8)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imwrite(str(self.args.output), overlay)
        rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, default=ROOT / "best.pt")
    parser.add_argument("--sam2-config", type=Path, required=True)
    parser.add_argument("--sam2-checkpoint", type=Path, required=True)
    parser.add_argument("--target-class", default="Wing and Body")
    parser.add_argument("--image-topic", default="/zed_scanner/left_image/compressed")
    parser.add_argument("--cloud-topic", default="/zed_scanner/live_cloud")
    parser.add_argument("--confidence", type=float, default=0.70)
    parser.add_argument("--output", type=Path, default=ROOT / "logs" / "remote_sam2_mask.png")
    args, ros_args = parser.parse_known_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rclpy.init(args=ros_args)
    node = Diagnostic(args)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
