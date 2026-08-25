#!/usr/bin/env python3
"""Display remote ZED RGB locally; Enter stores RGB + SAM2 mask locally only."""

import argparse
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, HistoryPolicy, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Empty

from zed_yolo_sam2_icp_reconstruct import select_target
from zed_yolo_sam2_live import ROOT, load_sam2, load_yolo


class RemotePhotoCapture(Node):
    def __init__(self, args):
        super().__init__("remote_yolo_sam2_photo_capture")
        self.args = args
        self.yolo = load_yolo(args.weights)
        self.sam2 = load_sam2(args.sam2_config, args.sam2_checkpoint)
        self.lock = threading.Lock()
        self.latest_bgr = self.capture_bgr = self.capture_overlay = self.capture_mask = None
        self.latest_depth = None
        self.capture_confidence = 0.0
        self.capture_ok = False
        self.capture_reason = "Waiting for a SAM2 result"
        self.last_saved_mask = None
        self.pending_capture = None
        self.status = "Waiting for remote ZED image"
        self.worker = None
        self.last_start = 0.0
        self.analysis_requested = False
        self.analysis_enabled = False
        # Keep one frame only: processing an old ROS queue makes the visual
        # stream look delayed even if the camera itself is live.
        live_qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1,
                              reliability=ReliabilityPolicy.BEST_EFFORT,
                              durability=DurabilityPolicy.VOLATILE)
        self.subscription = self.create_subscription(CompressedImage, args.image_topic,
                                                     self.on_image, live_qos)
        self.depth_subscription = self.create_subscription(Image, args.depth_topic,
                                                           self.on_depth, live_qos)
        self.capture_request = self.create_publisher(Empty, "/zed_scanner/capture_request", 1)
        self.timer = self.create_timer(0.025, self.update_ui)
        cv2.namedWindow("Remote ZED — local photo capture", cv2.WINDOW_NORMAL)
        self.get_logger().info(f"Remote image: {args.image_topic}; local save: {args.output}")

    def on_image(self, message):
        bgr = cv2.imdecode(np.frombuffer(message.data, np.uint8), cv2.IMREAD_COLOR)
        if bgr is not None:
            with self.lock:
                self.latest_bgr = bgr

    def on_depth(self, message):
        if message.encoding != "32FC1": return
        depth = np.frombuffer(message.data, dtype=np.float32).reshape(message.height, message.width)
        payload = None
        with self.lock:
            self.latest_depth = depth.copy()
            if self.pending_capture is not None:
                payload = (*self.pending_capture, self.latest_depth.copy())
                self.pending_capture = None
        if payload is not None:
            self.write_capture(*payload)

    def start_inference_if_ready(self):
        with self.lock:
            if self.latest_bgr is None or (self.worker and self.worker.is_alive()):
                return
            if not self.analysis_enabled:
                return
            if (not self.analysis_requested and
                    time.monotonic() - self.last_start < self.args.process_interval):
                return
            image = self.latest_bgr.copy()
            self.last_start = time.monotonic()
            self.analysis_requested = False
            self.status = "SAM2 analysing frame; live image remains visible"
            self.worker = threading.Thread(target=self.infer, args=(image,), daemon=True)
            self.worker.start()

    def infer(self, bgr):
        try:
            result = self.yolo.predict(bgr, conf=self.args.confidence, verbose=False)[0]
            selected = select_target(result, self.sam2,
                                     cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), self.args.target_class)
            overlay = bgr.copy()
            mask = np.zeros(bgr.shape[:2], dtype=np.uint8)
            status = "No matching YOLO + SAM2 target"
            capture_ok = False
            reason = "未检测到目标"
            if selected is not None:
                box, confidence, label, found_mask = selected
                mask = found_mask.astype(np.uint8) * 255
                overlay[found_mask] = (0.5 * overlay[found_mask] +
                                       0.5 * np.array((0, 255, 0))).astype(np.uint8)
                x1, y1, x2, y2 = np.round(box).astype(int)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(overlay, f"{label} {confidence:.0%}", (x1, max(28, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
                sharpness = cv2.Laplacian(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
                area = float(found_mask.mean())
                reasons = []
                if confidence < self.args.min_confidence: reasons.append("置信度偏低")
                if sharpness < self.args.min_sharpness: reasons.append("画面模糊，请停稳相机")
                if area < self.args.min_mask_area: reasons.append("目标太小，请靠近")
                if area > self.args.max_mask_area: reasons.append("目标不完整，请后退")
                with self.lock:
                    prior = self.last_saved_mask
                if prior is not None and prior.shape == found_mask.shape:
                    inter = np.logical_and(prior, found_mask).sum()
                    union = np.logical_or(prior, found_mask).sum()
                    if union and inter / union > self.args.max_duplicate_iou:
                        reasons.append("与上次角度太接近，请绕物体移动")
                capture_ok = not reasons
                reason = "可拍：按 Enter 保存本地照片" if capture_ok else "不建议拍：" + "；".join(reasons)
                status = (f"SAM2 {label} {confidence:.0%} | sharp={sharpness:.0f} | "
                          f"mask={area:.1%} | {reason}")
            with self.lock:
                self.capture_bgr, self.capture_overlay = bgr, overlay
                self.capture_mask, self.status = mask, status
                self.capture_confidence, self.capture_ok, self.capture_reason = confidence if selected else 0.0, capture_ok, reason
        except Exception as exc:
            with self.lock:
                self.status = f"Inference error: {type(exc).__name__}: {exc}"

    def save_capture(self):
        with self.lock:
            if self.capture_bgr is None:
                self.status = "No SAM2 result yet; wait briefly before saving"
                return
            if not self.capture_ok:
                self.status = self.capture_reason
                return
            if self.pending_capture is not None:
                self.status = "Already waiting for remote depth frame"
                return
            self.pending_capture = (datetime.now().strftime("%Y%m%d_%H%M%S_%f"), self.capture_bgr.copy(), self.capture_mask.copy(), self.capture_overlay.copy())
            self.status = "Capture requested: waiting for remote depth frame"
        self.capture_request.publish(Empty())

    def write_capture(self, stamp, bgr, mask, overlay, depth):
        dirs = {name: self.args.output / name for name in ("images", "masks", "preview", "depth")}
        for directory in dirs.values(): directory.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dirs["images"] / f"zed_{stamp}.jpg"), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(dirs["masks"] / f"zed_{stamp}.png"), mask)
        cv2.imwrite(str(dirs["preview"] / f"zed_{stamp}.jpg"), overlay, [cv2.IMWRITE_JPEG_QUALITY, 95])
        np.save(dirs["depth"] / f"zed_{stamp}_depth.npy", depth)
        with self.lock:
            self.last_saved_mask = mask > 0
            self.status = f"Saved RGB + mask + preview + depth: zed_{stamp}"

    def update_ui(self):
        self.start_inference_if_ready()
        with self.lock:
            frame = self.latest_bgr.copy() if self.latest_bgr is not None else None
            overlay = (None if not self.analysis_enabled or self.capture_overlay is None
                       else self.capture_overlay.copy())
            status = self.status
        if frame is None:
            waiting = np.zeros((480, 854, 3), dtype=np.uint8)
            cv2.putText(waiting, "Waiting for remote ZED RGB + depth...", (55, 230),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 2, cv2.LINE_AA)
            cv2.imshow("Remote ZED — local photo capture", waiting)
            cv2.waitKey(1)
            return
        if overlay is not None:
            small = cv2.resize(overlay, (frame.shape[1] // 4, frame.shape[0] // 4))
            h, w = small.shape[:2]
            frame[104:104+h, frame.shape[1]-w-12:frame.shape[1]-12] = small
            cv2.rectangle(frame, (frame.shape[1]-w-12, 104), (frame.shape[1]-12, 104+h), (0,255,255), 2)
            cv2.putText(frame, "last SAM2 result", (frame.shape[1]-w-12, 96),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1, cv2.LINE_AA)
        cv2.putText(frame, "Remote ZED live | YOLO + SAM2 runs locally", (16, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2, cv2.LINE_AA)
        color = (0, 220, 0) if "可拍" in status else (0, 255, 255)
        cv2.putText(frame, status, (16, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                    color, 2, cv2.LINE_AA)
        cv2.putText(frame, "Enter: save RGB + SAM2 mask locally | Q/Esc: quit", (16, frame.shape[0]-18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255,255,255), 2, cv2.LINE_AA)
        cv2.imshow("Remote ZED — local photo capture", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("a"), ord("A")):
            with self.lock:
                self.analysis_enabled = not self.analysis_enabled
                self.analysis_requested = self.analysis_enabled
                self.status = ("YOLO + SAM2 enabled: analysing current frame" if self.analysis_enabled
                               else "Raw remote image only (YOLO + SAM2 disabled)")
        elif key in (10, 13): self.save_capture()
        elif key in (27, ord("q")): rclpy.shutdown()

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, default=ROOT / "best.pt")
    parser.add_argument("--sam2-config", type=Path, required=True)
    parser.add_argument("--sam2-checkpoint", type=Path, required=True)
    parser.add_argument("--target-class", default="Wing and Body")
    parser.add_argument("--image-topic", default="/zed_scanner/left_image/compressed")
    parser.add_argument("--depth-topic", default="/zed_scanner/depth_image")
    parser.add_argument("--output", type=Path, default=ROOT / "datasets" / "remote_capture")
    parser.add_argument("--confidence", type=float, default=0.70)
    parser.add_argument("--process-interval", type=float, default=6.0,
                        help="Seconds between SAM2 updates while A mode is enabled.")
    parser.add_argument("--min-confidence", type=float, default=0.75)
    parser.add_argument("--min-sharpness", type=float, default=70.0)
    parser.add_argument("--min-mask-area", type=float, default=0.02)
    parser.add_argument("--max-mask-area", type=float, default=0.55)
    parser.add_argument("--max-duplicate-iou", type=float, default=0.88)
    args, ros_args = parser.parse_known_args()
    rclpy.init(args=ros_args)
    node = RemotePhotoCapture(args)
    executor = SingleThreadedExecutor(); executor.add_node(node)
    try:
        while rclpy.ok(): executor.spin_once(timeout_sec=0.02)
    except KeyboardInterrupt:
        pass
    finally:
        executor.remove_node(node); node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()


if __name__ == "__main__":
    main()
