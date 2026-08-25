#!/usr/bin/env python3
"""Manually capture validated local ZED RGB/depth/mask/preview groups."""

import json
import argparse
import math
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pyzed.sl as sl
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Imu

from zed_yolo_sam2_live import ROOT, load_sam2, load_yolo


MIN_DEPTH_M = 0.20
MAX_DEPTH_M = 2.30
MIN_MASK_RATIO = 0.005
MAX_MASK_RATIO = 0.75
MIN_VALID_DEPTH_RATIO = 0.35
MIN_CONNECTED_DEPTH_RATIO = 0.55
PROJECT_ROOT = ROOT.parent.parent
ONLINE_ICP_PIPELINE = PROJECT_ROOT / "point_cloud" / "online_point_cloud" / "online_icp_pipeline.py"
LIVE_FUSION_VIEWER = PROJECT_ROOT / "point_cloud" / "online_point_cloud" / "live_fusion_viewer.py"


def depth_live_preview(depth: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Show metric ZED depth beside RGB; invalid depth stays black."""
    valid = np.isfinite(depth) & (depth > MIN_DEPTH_M) & (depth < MAX_DEPTH_M)
    normalized = np.zeros(depth.shape, dtype=np.uint8)
    normalized[valid] = np.clip(
        (depth[valid] - MIN_DEPTH_M) / (MAX_DEPTH_M - MIN_DEPTH_M) * 255,
        0,
        255,
    ).astype(np.uint8)
    shown = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    shown[~valid] = 0
    shown = cv2.resize(shown, size, interpolation=cv2.INTER_NEAREST)
    cv2.putText(shown, "Depth: blue=near, red=far, black=invalid", (12, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    return shown


def quaternion_to_rpy_degrees(x: float, y: float, z: float, w: float) -> tuple[float, float, float]:
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return tuple(math.degrees(value) for value in (roll, pitch, yaw))


class TracerImuRecorder(Node):
    """Keep the newest vehicle IMU sample for each locally captured frame."""

    def __init__(self, topic: str) -> None:
        super().__init__("local_zed_imu_recorder")
        self._lock = threading.Lock()
        self._sample: dict | None = None
        self.create_subscription(Imu, topic, self._on_imu, 20)

    def _on_imu(self, message: Imu) -> None:
        orientation = message.orientation
        roll, pitch, yaw = quaternion_to_rpy_degrees(
            orientation.x, orientation.y, orientation.z, orientation.w
        )
        acceleration = message.linear_acceleration
        with self._lock:
            self._sample = {
                "stamp_sec": int(message.header.stamp.sec),
                "stamp_nanosec": int(message.header.stamp.nanosec),
                "orientation_xyzw": [orientation.x, orientation.y, orientation.z, orientation.w],
                "rpy_deg": [roll, pitch, yaw],
                "linear_acceleration_mps2": [acceleration.x, acceleration.y, acceleration.z],
                "received_monotonic": time.monotonic(),
            }

    def snapshot(self) -> dict | None:
        with self._lock:
            if self._sample is None:
                return None
            sample = dict(self._sample)
        sample["age_s"] = time.monotonic() - sample.pop("received_monotonic")
        return sample


def select_part_box(image: np.ndarray, part_name: str) -> np.ndarray | None:
    """Let the user draw one frozen-image box for a physical object part."""
    title = f"Draw {part_name} box - Enter confirm, C cancel"
    shown = image.copy()
    cv2.putText(
        shown,
        f"Drag a tight box around {part_name}",
        (16, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    x, y, width, height = cv2.selectROI(
        title, shown, showCrosshair=True, fromCenter=False
    )
    cv2.destroyWindow(title)
    if width < 8 or height < 8:
        return None
    return np.array((x, y, x + width, y + height), dtype=np.float32)


def clean_part_mask(mask: np.ndarray) -> np.ndarray:
    """Remove tiny speckles without deleting a valid disconnected part."""
    binary = mask.astype(bool).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    minimum_area = max(80, int(binary.size * 0.0002))
    keep = np.zeros(binary.shape, dtype=bool)
    for label in range(1, count):
        if int(stats[label, cv2.CC_STAT_AREA]) >= minimum_area:
            keep |= labels == label
    keep = cv2.morphologyEx(
        keep.astype(np.uint8),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    return keep.astype(bool)


def segment_box(
    predictor,
    box: np.ndarray,
    points: list[tuple[float, float]] | None = None,
    labels: list[int] | None = None,
) -> np.ndarray:
    point_coords = (
        np.asarray(points, dtype=np.float32)
        if points
        else None
    )
    point_labels = (
        np.asarray(labels, dtype=np.int32)
        if labels
        else None
    )
    masks, _scores, _logits = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        box=box[None, :],
        multimask_output=False,
    )
    return clean_part_mask(np.squeeze(masks[0]))


def manual_part_masks(
    predictor, bgr: np.ndarray, rgb: np.ndarray, depth: np.ndarray,
    capture_mode: str, auto_boxes: dict[str, np.ndarray] | None = None,
):
    """Select, correct and confirm the part(s) chosen for this session."""
    part_specs = {
        "body": ("BODY", (0, 255, 0)),
        "wing": ("WING", (0, 165, 255)),
    }
    selected_parts = (
        ("wing",) if capture_mode == "wing" else
        ("body",) if capture_mode == "body" else
        ("body", "wing")
    )
    window = "Confirm " + " + ".join(part_specs[key][0] for key in selected_parts) + " masks"
    force_manual_boxes = False
    while True:
        boxes = {}
        for key in selected_parts:
            # R explicitly discards the YOLO box and makes the next pass
            # request a fresh user-drawn box.
            box = None if force_manual_boxes else (auto_boxes or {}).get(key)
            if box is None:
                box = select_part_box(bgr, part_specs[key][0])
                if box is None:
                    return None
            boxes[key] = box

        predictor.set_image(rgb)
        parts = {}
        for key in selected_parts:
            part_name, color = part_specs[key]
            parts[part_name] = {
                "box": boxes[key],
                "points": [],
                "labels": [],
                "color": color,
            }
        for part in parts.values():
            part["mask"] = segment_box(predictor, part["box"])

        active = [part_specs[selected_parts[0]][0]]
        click_events = []
        correction_enabled = [False]

        def mouse_callback(event, x, y, _flags, _parameter):
            if not correction_enabled[0]:
                return
            if event == cv2.EVENT_LBUTTONDOWN:
                click_events.append((active[0], float(x), float(y), 1))
            elif event == cv2.EVENT_RBUTTONDOWN:
                click_events.append((active[0], float(x), float(y), 0))

        cv2.namedWindow(window)
        cv2.setMouseCallback(window, mouse_callback)
        redraw_boxes = False
        while True:
            while click_events:
                part_name, x, y, point_label = click_events.pop(0)
                part = parts[part_name]
                part["points"].append((x, y))
                part["labels"].append(point_label)
                part["mask"] = segment_box(
                    predictor, part["box"], part["points"], part["labels"]
                )

            body_mask = parts.get("BODY", {}).get(
                "mask", np.zeros(depth.shape, dtype=bool)
            )
            wing_mask = parts.get("WING", {}).get(
                "mask", np.zeros(depth.shape, dtype=bool)
            )
            combined_mask = body_mask | wing_mask
            quality = {
                name: quality_metrics(part["mask"].astype(np.uint8) * 255, depth)
                for name, part in parts.items()
            }
            combined_ok, _combined_reason, combined_metrics = quality_metrics(
                combined_mask.astype(np.uint8) * 255, depth
            )
            all_ok = combined_ok and all(item[0] for item in quality.values())

            preview = bgr.copy()
            preview[body_mask] = (
                preview[body_mask] * 0.45
                + np.array((0, 255, 0), dtype=np.float32) * 0.55
            ).astype(np.uint8)
            preview[wing_mask] = (
                preview[wing_mask] * 0.45
                + np.array((0, 165, 255), dtype=np.float32) * 0.55
            ).astype(np.uint8)
            for part_name, part in parts.items():
                x1, y1, x2, y2 = np.round(part["box"]).astype(int)
                thickness = 4 if part_name == active[0] else 2
                cv2.rectangle(
                    preview, (x1, y1), (x2, y2), part["color"], thickness
                )
                cv2.putText(
                    preview,
                    part_name,
                    (x1, max(24, y1 - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    part["color"],
                    2,
                    cv2.LINE_AA,
                )
                for (x, y), point_label in zip(
                    part["points"], part["labels"]
                ):
                    center = (int(round(x)), int(round(y)))
                    if point_label == 1:
                        cv2.circle(preview, center, 7, part["color"], -1)
                        cv2.circle(preview, center, 8, (0, 0, 0), 2)
                    else:
                        cv2.line(
                            preview,
                            (center[0] - 7, center[1] - 7),
                            (center[0] + 7, center[1] + 7),
                            (0, 0, 255),
                            3,
                        )
                        cv2.line(
                            preview,
                            (center[0] + 7, center[1] - 7),
                            (center[0] - 7, center[1] + 7),
                            (0, 0, 255),
                            3,
                        )
            status = "PASS" if all_ok else "WARN (manual save allowed)"
            status += " | POINT EDIT ON" if correction_enabled[0] else " | P for point edit"
            cv2.putText(
                preview,
                f"{status} | Active={active[0]} | mode={capture_mode}",
                (16, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                preview,
                "P=enable points | Left=add | Right=remove | R=redraw box | Enter=save",
                (16, 58),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.47,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            quality_text = " | ".join(
                f"{name}: {item[1]}" for name, item in quality.items()
            )
            cv2.putText(preview, f"{quality_text} | Enter/S=save", (16, 84),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2,
                        cv2.LINE_AA)
            cv2.imshow(window, preview)
            key = cv2.waitKey(20) & 0xFF
            if key in (ord("p"), ord("P")):
                correction_enabled[0] = not correction_enabled[0]
            if "BODY" in parts and key in (ord("b"), ord("B")):
                active[0] = "BODY"
            elif "WING" in parts and key in (ord("w"), ord("W")):
                active[0] = "WING"
            elif key in (ord("u"), ord("U")):
                part = parts[active[0]]
                if part["points"]:
                    part["points"].pop()
                    part["labels"].pop()
                    part["mask"] = segment_box(
                        predictor,
                        part["box"],
                        part["points"],
                        part["labels"],
                    )
            elif key in (ord("c"), ord("C")):
                part = parts[active[0]]
                part["points"].clear()
                part["labels"].clear()
                part["mask"] = segment_box(predictor, part["box"])
            elif key in (ord("x"), ord("X")):
                part = parts[active[0]]
                kept = [
                    (point, label)
                    for point, label in zip(part["points"], part["labels"])
                    if label != 0
                ]
                part["points"] = [point for point, _label in kept]
                part["labels"] = [label for _point, label in kept]
                part["mask"] = segment_box(
                    predictor,
                    part["box"],
                    part["points"],
                    part["labels"],
                )
            elif key in (ord("a"), ord("A")):
                for part in parts.values():
                    part["points"].clear()
                    part["labels"].clear()
                    part["mask"] = segment_box(predictor, part["box"])
            if key in (ord("r"), ord("R")):
                cv2.destroyWindow(window)
                redraw_boxes = True
                break
            if key in (27, ord("q"), ord("Q")):
                cv2.destroyWindow(window)
                return None
            masks_nonempty = all(
                np.count_nonzero(part["mask"]) >= 80
                for part in parts.values()
            )
            if masks_nonempty and key in (
                10,
                13,
                32,
                ord("s"),
                ord("S"),
            ):
                cv2.destroyWindow(window)
                return {
                    "body_box": boxes.get("body"),
                    "wing_box": boxes.get("wing"),
                    "body_mask": body_mask,
                    "wing_mask": wing_mask,
                    "combined_mask": combined_mask,
                    "body_quality": quality.get("BODY", (None, None, {}))[2],
                    "wing_quality": quality.get("WING", (None, None, {}))[2],
                    "combined_quality": combined_metrics,
                    "automatic_quality_passed": all_ok,
                    "preview": preview,
                }
        if redraw_boxes:
            force_manual_boxes = True
            continue


def quality_metrics(mask: np.ndarray, depth: np.ndarray) -> tuple[bool, str, dict]:
    mask_pixels = int(np.count_nonzero(mask))
    image_pixels = int(mask.size)
    mask_ratio = mask_pixels / max(1, image_pixels)
    if mask_ratio < MIN_MASK_RATIO:
        return False, "mask too small", {"mask_ratio": mask_ratio}
    if mask_ratio > MAX_MASK_RATIO:
        return False, "mask too large", {"mask_ratio": mask_ratio}

    valid = (
        (mask > 0)
        & np.isfinite(depth)
        & (depth > MIN_DEPTH_M)
        & (depth < MAX_DEPTH_M)
    )
    valid_pixels = int(np.count_nonzero(valid))
    valid_ratio = valid_pixels / max(1, mask_pixels)
    if valid_ratio < MIN_VALID_DEPTH_RATIO:
        return (
            False,
            f"valid depth too low ({valid_ratio:.0%})",
            {"mask_ratio": mask_ratio, "valid_depth_ratio": valid_ratio},
        )

    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        valid.astype(np.uint8), 8
    )
    component_areas = (
        np.sort(stats[1:, cv2.CC_STAT_AREA])[::-1]
        if count > 1
        else np.empty(0, dtype=np.int32)
    )
    largest = int(component_areas[0]) if len(component_areas) else 0
    largest_two = int(component_areas[:2].sum())
    connected_ratio = largest_two / max(1, valid_pixels)
    metrics = {
        "mask_pixels": mask_pixels,
        "mask_ratio": mask_ratio,
        "valid_depth_pixels": valid_pixels,
        "valid_depth_ratio": valid_ratio,
        # Kept under the old key so previously written reconstruction tools
        # remain compatible. It now means the combined body + wing coverage.
        "largest_depth_component_ratio": connected_ratio,
        "largest_one_depth_component_ratio": largest / max(1, valid_pixels),
        "kept_depth_components": min(2, len(component_areas)),
        "median_depth_m": float(np.median(depth[valid])),
    }
    if connected_ratio < MIN_CONNECTED_DEPTH_RATIO:
        return (
            False,
            f"body + wing depth fragmented ({connected_ratio:.0%} connected)",
            metrics,
        )
    return True, "qualified", metrics


def choose_capture_mode() -> str:
    """Ask once which physical part(s) this capture session contains."""
    options = {
        "1": ("wing", "仅拍摄机翼"),
        "2": ("body", "仅拍摄机体"),
        "3": ("both", "拍摄机翼和机体"),
    }
    while True:
        print("\n采集模式：1) 仅机翼  2) 仅机体  3) 机翼 + 机体", flush=True)
        choice = input("请选择 [1/2/3]: ").strip()
        if choice in options:
            mode, label = options[choice]
            print(f"[设置] {label}", flush=True)
            return mode
        print("请输入 1、2 或 3。", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Local ZED manual SAM2 capture")
    parser.add_argument(
        "--auto-body", action="store_true",
        help="use body.pt to propose a BODY box (off by default; manual boxes are safer)",
    )
    parser.add_argument("--imu-topic", default="/tracer5/IMU_data")
    parser.add_argument("--imu-domain-id", default="36",
                        help="ROS domain containing the vehicle IMU (default: 36)")
    parser.add_argument("--no-imu", action="store_true",
                        help="capture without vehicle IMU metadata/ICP yaw constraint")
    args = parser.parse_args()
    # The vehicle publishes in domain 36.  Make the requested domain explicit
    # instead of inheriting an unrelated terminal's default domain 0.
    os.environ["ROS_DOMAIN_ID"] = str(args.imu_domain_id)
    capture_mode = choose_capture_mode()
    session_id = datetime.now().strftime("scan_%Y%m%d_%H%M%S")
    output = ROOT / "datasets/local_offline_capture" / session_id
    for directory in (
        "images",
        "depth",
        "confidence",
        "masks",
        "body_masks",
        "wing_masks",
        "preview",
    ):
        (output / directory).mkdir(parents=True, exist_ok=True)

    print("[初始化] 正在加载 SAM2 手动双部件分割器（CPU 模式请稍等）...", flush=True)
    sam2 = load_sam2(
        ROOT / "sam2/sam2/configs/sam2.1/sam2.1_hiera_t.yaml",
        ROOT / "sam2/checkpoints/sam2.1_hiera_tiny.pt",
    )
    auto_model_path = ROOT.parent / "body.pt"
    body_yolo = load_yolo(auto_model_path) if args.auto_body and auto_model_path.is_file() else None
    if args.auto_body and body_yolo is None:
        print(f"[warning] Auto YOLO model missing: {auto_model_path}; manual box mode only.", flush=True)
    elif body_yolo is not None:
        print(f"[初始化] 自动 Body 检测模型：{auto_model_path}", flush=True)
    else:
        print("[初始化] 默认手动框选：不会自动把 BODY 写入当前掩码。", flush=True)

    imu_recorder = None
    imu_executor = None
    imu_thread = None
    if not args.no_imu:
        try:
            if not rclpy.ok():
                rclpy.init()
            imu_recorder = TracerImuRecorder(args.imu_topic)
            imu_executor = SingleThreadedExecutor()
            imu_executor.add_node(imu_recorder)
            imu_thread = threading.Thread(target=imu_executor.spin, daemon=True)
            imu_thread.start()
            print(f"[初始化] 记录车辆 IMU：{args.imu_topic}（ROS_DOMAIN_ID={os.environ['ROS_DOMAIN_ID']}）", flush=True)
        except Exception as error:
            print(f"[warning] Vehicle IMU unavailable; continuing without IMU constraint: {error}", flush=True)
            if imu_recorder is not None:
                imu_recorder.destroy_node()
            imu_recorder = imu_executor = imu_thread = None

    camera = sl.Camera()
    parameters = sl.InitParameters()
    parameters.camera_resolution = sl.RESOLUTION.HD720
    parameters.camera_fps = 30
    parameters.depth_mode = sl.DEPTH_MODE.ULTRA
    parameters.coordinate_units = sl.UNIT.METER
    print("[初始化] 正在打开本机 ZED...", flush=True)
    status = camera.open(parameters)
    if status != sl.ERROR_CODE.SUCCESS:
        raise SystemExit(f"无法打开本机 ZED：{status}")

    tracking_error = camera.enable_positional_tracking(
        sl.PositionalTrackingParameters()
    )
    tracking_enabled = tracking_error == sl.ERROR_CODE.SUCCESS
    camera_pose = sl.Pose()

    calibration = (
        camera.get_camera_information()
        .camera_configuration.calibration_parameters.left_cam
    )
    (output / "camera_intrinsics.json").write_text(
        json.dumps(
            {
                "width": 1280,
                "height": 720,
                "fx": float(calibration.fx),
                "fy": float(calibration.fy),
                "cx": float(calibration.cx),
                "cy": float(calibration.cy),
                "depth_unit": "meter",
                "min_depth_m": MIN_DEPTH_M,
                "max_depth_m": MAX_DEPTH_M,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    manifest = (output / "captures.jsonl").open(
        "a", encoding="utf-8", buffering=1
    )

    image_zed, depth_zed, confidence_zed = sl.Mat(), sl.Mat(), sl.Mat()
    saved_count = rejected_count = 0
    fusion_job = None
    fusion_viewer = None
    last_status = "Ready: hold still and press Enter"
    print(f"[就绪] 数据目录：{output}", flush=True)
    print("按 Enter 检查并保存；Q 退出。", flush=True)

    try:
        while True:
            if camera.grab() != sl.ERROR_CODE.SUCCESS:
                continue
            camera.retrieve_image(image_zed, sl.VIEW.LEFT)
            camera.retrieve_measure(depth_zed, sl.MEASURE.DEPTH)
            camera.retrieve_measure(confidence_zed, sl.MEASURE.CONFIDENCE)
            bgr = cv2.cvtColor(image_zed.get_data(), cv2.COLOR_BGRA2BGR)
            depth = depth_zed.get_data().copy()
            confidence = confidence_zed.get_data().copy()

            shown = bgr.copy()
            cv2.putText(
                shown,
                f"Saved {saved_count} | Rejected {rejected_count}",
                (16, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                shown,
                last_status,
                (16, 57),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                shown,
                "Enter: freeze -> YOLO + SAM2 auto mask | Q: quit",
                (16, shown.shape[0] - 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("Local ZED offline capture", shown)
            cv2.imshow(
                "Local ZED depth",
                depth_live_preview(depth, (shown.shape[1], shown.shape[0])),
            )
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
            if key not in (10, 13):
                continue

            print("[capture] 正在检查当前帧，请保持相机不动...", flush=True)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            auto_boxes = {}
            if body_yolo is not None and capture_mode in ("body", "both"):
                detected = body_yolo(bgr, verbose=False)[0]
                if detected.boxes is not None and len(detected.boxes):
                    best = int(np.argmax(detected.boxes.conf.cpu().numpy()))
                    auto_boxes["body"] = detected.boxes.xyxy[best].cpu().numpy()
                    print("[capture] YOLO found BODY; SAM2 mask generated automatically.", flush=True)
                else:
                    print("[capture] YOLO found no BODY; please draw a box manually.", flush=True)
            selected = manual_part_masks(sam2, bgr, rgb, depth, capture_mode, auto_boxes)
            detection_mode = f"YOLO Body + SAM2 ({capture_mode})"
            if selected is None:
                rejected_count += 1
                last_status = "CANCEL: body/wing selection not saved"
                print(f"[不合格] {last_status}", flush=True)
                continue

            body_mask = selected["body_mask"].astype(np.uint8) * 255
            wing_mask = selected["wing_mask"].astype(np.uint8) * 255
            mask = selected["combined_mask"].astype(np.uint8) * 255
            metrics = selected["combined_quality"]
            metrics.setdefault("valid_depth_ratio", 0.0)
            metrics.setdefault("largest_depth_component_ratio", 0.0)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            preview = selected["preview"]

            cv2.imwrite(str(output / "images" / f"{timestamp}.jpg"), bgr)
            np.save(output / "depth" / f"{timestamp}_depth.npy", depth)
            np.save(
                output / "confidence" / f"{timestamp}_confidence.npy",
                confidence,
            )
            cv2.imwrite(str(output / "masks" / f"{timestamp}.png"), mask)
            cv2.imwrite(
                str(output / "body_masks" / f"{timestamp}.png"), body_mask
            )
            cv2.imwrite(
                str(output / "wing_masks" / f"{timestamp}.png"), wing_mask
            )
            cv2.imwrite(str(output / "preview" / f"{timestamp}.jpg"), preview)

            pose_matrix = None
            pose_state = "disabled"
            if tracking_enabled:
                state = camera.get_position(
                    camera_pose, sl.REFERENCE_FRAME.WORLD
                )
                pose_state = str(state)
                if state == sl.POSITIONAL_TRACKING_STATE.OK:
                    pose_matrix = np.asarray(
                        camera_pose.pose_data().m, dtype=np.float64
                    ).tolist()
            manifest.write(
                json.dumps(
                    {
                        "id": timestamp,
                        "detection_mode": detection_mode,
                        "confidence": None,
                        "quality": metrics,
                        "body_quality": selected["body_quality"],
                        "wing_quality": selected["wing_quality"],
                        "automatic_quality_passed": selected[
                            "automatic_quality_passed"
                        ],
                        "body_box": (
                            selected["body_box"].tolist()
                            if selected["body_box"] is not None else None
                        ),
                        "wing_box": (
                            selected["wing_box"].tolist()
                            if selected["wing_box"] is not None else None
                        ),
                        "pose_state": pose_state,
                        "world_from_camera": pose_matrix,
                        "tracer5_imu": (
                            imu_recorder.snapshot() if imu_recorder is not None else None
                        ),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            saved_count += 1
            save_kind = (
                "ACCEPT"
                if selected["automatic_quality_passed"]
                else "MANUAL SAVE (quality warning)"
            )
            last_status = (
                f"{save_kind}: depth {metrics['valid_depth_ratio']:.0%}, "
                f"connected {metrics['largest_depth_component_ratio']:.0%}"
            )
            print(
                f"[已保存] {timestamp}; {last_status}; "
                f"mode={detection_mode}",
                flush=True,
            )
            # Build in the background so the live RGB/depth windows remain
            # responsive.  The persistent Open3D viewer reloads the fused PLY
            # whenever this job completes.
            if ONLINE_ICP_PIPELINE.is_file():
                if fusion_job is None or fusion_job.poll() is not None:
                    fusion_job = subprocess.Popen(
                        [sys.executable, str(ONLINE_ICP_PIPELINE), str(output), timestamp]
                    )
                    print("[点云] 正在后台更新 ICP；融合窗口会自动刷新。", flush=True)
                else:
                    print("[点云] 上一帧 ICP 尚未完成，本帧已保存，下一次保存后会更新。", flush=True)
                if LIVE_FUSION_VIEWER.is_file() and (
                    fusion_viewer is None or fusion_viewer.poll() is not None
                ):
                    fusion_viewer = subprocess.Popen(
                        [sys.executable, str(LIVE_FUSION_VIEWER), str(output)]
                    )
            else:
                print(f"[warning] Online ICP pipeline missing: {ONLINE_ICP_PIPELINE}", flush=True)
    finally:
        manifest.close()
        if imu_executor is not None:
            imu_executor.shutdown()
        if imu_thread is not None:
            imu_thread.join(timeout=1.0)
        if imu_recorder is not None:
            imu_recorder.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        if tracking_enabled:
            camera.disable_positional_tracking()
        camera.close()
        cv2.destroyAllWindows()
        print(
            f"[结束] 合格保存 {saved_count} 组，不合格 {rejected_count} 次；"
            f"目录：{output}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n已停止采集。", flush=True)
