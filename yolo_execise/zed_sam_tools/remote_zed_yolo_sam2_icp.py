#!/usr/bin/env python3
"""Local YOLO + SAM2 + Open3D reconstruction from remote ZED ROS 2 topics.

SAM2 can be slow on a CPU.  ROS callbacks therefore only retain the newest
synchronised image/cloud pair; a background worker performs inference while
the OpenCV and Open3D windows remain responsive on the main thread.
"""

import argparse
import threading
import time
import traceback
from pathlib import Path

import cv2
import message_filters
import numpy as np
import open3d as o3d
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, PointCloud2
from sensor_msgs_py import point_cloud2

from zed_yolo_sam2_icp_reconstruct import make_cloud, register, save_model, select_target
from zed_yolo_sam2_live import ROOT, load_sam2, load_yolo


def cloud_for_mask(message: PointCloud2, mask: np.ndarray, bgr: np.ndarray):
    rows = np.asarray(list(point_cloud2.read_points(
        message, field_names=("x", "y", "z", "u", "v"), skip_nans=True
    )))
    if rows.size == 0:
        return np.empty((0, 3)), np.empty((0, 3))
    xyz = np.column_stack((rows["x"], rows["y"], rows["z"])).astype(np.float64)
    uv = np.rint(np.column_stack((rows["u"], rows["v"]))).astype(np.int32)
    height, width = mask.shape
    valid = ((uv[:, 0] >= 0) & (uv[:, 0] < width) &
             (uv[:, 1] >= 0) & (uv[:, 1] < height))
    xyz, uv = xyz[valid], uv[valid]
    chosen = mask[uv[:, 1], uv[:, 0]]
    xyz, uv = xyz[chosen], uv[chosen]
    colors = bgr[uv[:, 1], uv[:, 0], ::-1].astype(np.float64) / 255.0
    return xyz, colors


class RemoteObjectReconstructor(Node):
    def __init__(self, args):
        super().__init__("remote_zed_yolo_sam2_icp")
        self.args = args
        self.yolo = load_yolo(args.weights)
        self.sam2 = load_sam2(args.sam2_config, args.sam2_checkpoint)
        assert self.sam2 is not None
        self.lock = threading.Lock()
        self.latest_pair = None
        self.latest_preview = None
        self.latest_overlay = None
        self.last_result_time = None
        self.worker = None
        self.last_started = 0.0
        self.last_status = "Waiting for synchronized remote image + cloud"
        self.last_point_count = 0
        self.accepted = self.rejected = 0
        self.previous_local = None
        self.world_from_previous = np.eye(4)
        self.built = o3d.geometry.PointCloud()
        self.pending_current = None
        self.pending_built = None

        self.viewer = o3d.visualization.Visualizer()
        self.viewer.create_window("Local SAM2 masked ICP reconstruction", 1200, 850)
        self.built_visual = o3d.geometry.PointCloud()
        self.current_visual = o3d.geometry.PointCloud()
        self.viewer.add_geometry(self.built_visual)
        self.viewer.add_geometry(self.current_visual)
        # Independent image subscription drives the live preview.  The
        # synchronised pair below is only for 3D reconstruction.
        self.preview_sub = self.create_subscription(
            CompressedImage, args.image_topic, self.update_preview,
            qos_profile_sensor_data)
        image_sub = message_filters.Subscriber(
            self, CompressedImage, args.image_topic, qos_profile=qos_profile_sensor_data)
        cloud_sub = message_filters.Subscriber(
            self, PointCloud2, args.cloud_topic, qos_profile=qos_profile_sensor_data)
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [image_sub, cloud_sub], queue_size=8, slop=args.sync_slop)
        self.sync.registerCallback(self.cache_pair)
        self.timer = self.create_timer(0.025, self.update_ui)
        self.get_logger().info(f"Image: {args.image_topic}; cloud: {args.cloud_topic}")

    def update_preview(self, image_message):
        bgr = cv2.imdecode(np.frombuffer(image_message.data, np.uint8), cv2.IMREAD_COLOR)
        if bgr is not None:
            with self.lock:
                self.latest_preview = bgr

    def cache_pair(self, image_message, cloud_message):
        bgr = cv2.imdecode(np.frombuffer(image_message.data, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            return
        with self.lock:
            self.latest_pair = (bgr, cloud_message)
            self.latest_preview = bgr

    def maybe_start_worker(self):
        with self.lock:
            busy = self.worker is not None and self.worker.is_alive()
            if busy or self.latest_pair is None:
                return
            if time.monotonic() - self.last_started < self.args.min_process_interval:
                return
            bgr, cloud_message = self.latest_pair
            self.last_started = time.monotonic()
            self.last_status = "SAM2 is analysing the latest frame (window remains live)"
            self.worker = threading.Thread(
                target=self.process_pair, args=(bgr.copy(), cloud_message), daemon=True)
            self.worker.start()

    def process_pair(self, bgr, cloud_message):
        try:
            result = self.yolo.predict(bgr, conf=self.args.confidence, verbose=False)[0]
            selected = select_target(result, self.sam2,
                                     cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), self.args.target_class)
            overlay = bgr.copy()
            if selected is None:
                with self.lock:
                    self.latest_overlay = overlay
                    self.last_point_count = 0
                    self.last_status = "No matching YOLO + SAM2 target"
                return
            box, confidence, label, mask = selected
            points, colors = cloud_for_mask(cloud_message, mask, bgr)
            x1, y1, x2, y2 = np.round(box).astype(int)
            overlay[mask] = (overlay[mask] * 0.5 + np.array((0, 255, 0)) * 0.5).astype(np.uint8)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(overlay, f"{label} {confidence:.0%}", (x1, max(28, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
            with self.lock:
                self.latest_overlay = overlay
                self.last_result_time = time.monotonic()
                self.last_point_count = len(points)
                if len(points) < self.args.min_points:
                    self.last_status = f"SAM2 mask has only {len(points)} valid depth points"
                    return
                local = make_cloud(points, colors, self.args.voxel_m)
                if self.previous_local is None:
                    world_from_current = np.eye(4)
                    self.accepted += 1
                    status = "Anchor frame accepted"
                else:
                    fit = register(local, self.previous_local, self.args.voxel_m)
                    if fit.fitness < self.args.min_fitness or fit.inlier_rmse > self.args.max_rmse_m:
                        self.rejected += 1
                        world_from_current = None
                        status = f"ICP rejected: fitness={fit.fitness:.2f}, rmse={fit.inlier_rmse:.3f} m"
                    else:
                        world_from_current = self.world_from_previous @ fit.transformation
                        self.accepted += 1
                        status = f"ICP accepted: fitness={fit.fitness:.2f}, rmse={fit.inlier_rmse:.3f} m"
                if world_from_current is not None:
                    in_world = o3d.geometry.PointCloud(local)
                    in_world.transform(world_from_current)
                    self.built += in_world
                    self.built = self.built.voxel_down_sample(self.args.voxel_m)
                    self.previous_local = local
                    self.world_from_previous = world_from_current
                    self.pending_current = in_world
                    self.pending_built = o3d.geometry.PointCloud(self.built)
                self.last_status = status
                self.get_logger().info(f"{status}; SAM2 mask depth points={len(points)}")
        except Exception as exc:
            self.get_logger().error(traceback.format_exc())
            with self.lock:
                self.last_status = f"Processing error: {type(exc).__name__}: {exc}"

    def update_ui(self):
        self.maybe_start_worker()
        with self.lock:
            if self.pending_built is not None:
                self.built_visual.points = self.pending_built.points
                self.built_visual.colors = self.pending_built.colors
                self.current_visual.points = self.pending_current.points
                self.current_visual.colors = self.pending_current.colors
                self.pending_built = self.pending_current = None
                self.viewer.update_geometry(self.built_visual)
                self.viewer.update_geometry(self.current_visual)
            # Never use the old SAM2 frame as the main view.  SAM2 inference
            # is deliberately slow on CPU, while this preview must stay live.
            frame = self.latest_preview
            result_preview = self.latest_overlay
            status, count, accepted, rejected = (self.last_status, self.last_point_count,
                                                  self.accepted, self.rejected)
        if frame is not None:
            frame = frame.copy()
            cv2.putText(frame, f"SAM2 masked | accepted {accepted} | rejected {rejected}", (16, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, status, (16, 57), cv2.FONT_HERSHEY_SIMPLEX, 0.53,
                        (255, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f"SAM2 mask depth points: {count}", (16, 84),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, "S: save   R: reset   Ctrl-C: quit", (16, frame.shape[0] - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
            # The small inset makes it clear that this is the most recent
            # completed SAM2 output, not a frozen camera stream.
            if result_preview is not None:
                inset_w = max(220, frame.shape[1] // 4)
                inset_h = int(inset_w * result_preview.shape[0] / result_preview.shape[1])
                inset = cv2.resize(result_preview, (inset_w, inset_h))
                y0, x0 = 104, frame.shape[1] - inset_w - 12
                frame[y0:y0 + inset_h, x0:x0 + inset_w] = inset
                cv2.rectangle(frame, (x0, y0), (x0 + inset_w, y0 + inset_h), (0, 255, 255), 2)
                cv2.putText(frame, "last completed SAM2 result", (x0, y0 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow("Remote ZED: local YOLO + SAM2", frame)
        self.viewer.poll_events()
        self.viewer.update_renderer()
        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            with self.lock:
                if len(self.built.points):
                    save_model(self.built, self.args.output, self.args.mesh_on_save)
                    self.last_status = f"Saved {self.args.output.with_suffix('.ply')}"
        elif key == ord("r"):
            with self.lock:
                self.built.clear(); self.previous_local = None
                self.world_from_previous = np.eye(4); self.accepted = self.rejected = 0
                self.pending_built = o3d.geometry.PointCloud()
                self.pending_current = o3d.geometry.PointCloud()
                self.last_status = "Reconstruction reset"

    def destroy_node(self):
        self.viewer.destroy_window()
        cv2.destroyAllWindows()
        super().destroy_node()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run local YOLO+SAM2+ICP on remote ZED topics.")
    parser.add_argument("--weights", type=Path, default=ROOT / "best.pt")
    parser.add_argument("--sam2-config", type=Path, required=True)
    parser.add_argument("--sam2-checkpoint", type=Path, required=True)
    parser.add_argument("--target-class", default="Wing and Body")
    parser.add_argument("--image-topic", default="/zed_scanner/left_image/compressed")
    parser.add_argument("--cloud-topic", default="/zed_scanner/live_cloud")
    parser.add_argument("--sync-slop", type=float, default=0.18)
    parser.add_argument("--min-process-interval", type=float, default=12.0)
    parser.add_argument("--confidence", type=float, default=0.70)
    parser.add_argument("--voxel-m", type=float, default=0.012)
    parser.add_argument("--min-points", type=int, default=150)
    parser.add_argument("--min-fitness", type=float, default=0.55)
    parser.add_argument("--max-rmse-m", type=float, default=0.025)
    parser.add_argument("--output", type=Path, default=ROOT / "models" / "remote_recognized_object")
    parser.add_argument("--mesh-on-save", action="store_true")
    args, ros_args = parser.parse_known_args()
    if not args.weights.is_file(): parser.error(f"YOLO weights not found: {args.weights}")
    rclpy.init(args=ros_args)
    node = RemoteObjectReconstructor(args)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    try:
        while rclpy.ok():
            executor.spin_once(timeout_sec=0.02)
    except KeyboardInterrupt:
        pass
    finally:
        executor.remove_node(node)
        node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
