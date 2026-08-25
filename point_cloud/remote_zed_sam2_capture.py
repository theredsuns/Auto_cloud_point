#!/usr/bin/env python3
"""Capture remote ZED RGB/depth on this PC, then review a YOLO+SAM2 mask.

Live view: Enter requests one remote depth frame.  Review: Enter saves and
returns to live view; P enables SAM2 point edits (left add/right remove); R
draws a replacement box and reruns SAM2; Q quits without saving the frame.
"""
from __future__ import annotations

import json
import argparse
import re
import socket
import struct
import subprocess
import sys
import time
import zlib
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
YOLO_DIR = PROJECT / "yolo_execise" / "yolo_body_wing_classification"
SAM_DIR = PROJECT / "yolo_execise" / "zed_sam_tools"
sys.path.insert(0, str(YOLO_DIR))
import run_local_yolo  # noqa: E402,F401
from ultralytics import YOLO  # noqa: E402
sys.path.insert(0, str(SAM_DIR))
from zed_yolo_sam2_live import load_sam2  # noqa: E402

# Preview is 320x180; a saved capture is the native ZED HD720 image.
REMOTE_WIDTH = 1280
REMOTE_HEIGHT = 720
DEFAULT_INTRINSICS = {"fx": 672.917602539, "fy": 672.917602539,
                      "cx": 639.5, "cy": 359.5}
SEGMENT_WEIGHTS = PROJECT / "yolo_execise" / "yolo_body_wing_dataset" / "runs" / "body_wing" / "weights" / "best.pt"
ONLINE_ROOT = ROOT / "online_point_cloud"
REMOTE_HOST = "skki@192.168.50.55"
# This is the fixed transform published by start_remote_agx_zed_stack.sh:
# tcp_link -> zed_camera_link.  It is part of the camera mounting calibration,
# not a per-frame motion estimate.  Depth is registered to this ZED camera
# mount in the current bridge configuration.
TCP_FROM_ZED_CAMERA_TRANSLATION_M = (0.0, 0.0, -0.015)
TCP_FROM_ZED_CAMERA_RPY_RAD = (0.0, -0.050, 0.0)


def tcp_pose_to_matrix(position_m, quaternion_xyzw):
    """Return arm-base/world_from_camera from the calibrated TCP pose."""
    x, y, z, w = (float(value) for value in quaternion_xyzw)
    norm = float(np.sqrt(x * x + y * y + z * z + w * w))
    if norm < 1e-9:
        raise ValueError("zero TCP quaternion")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    matrix[:3, 3] = [float(value) for value in position_m]
    return matrix


def rpy_matrix(roll, pitch, yaw):
    """Homogeneous transform rotation from ROS roll/pitch/yaw values."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])
    return matrix


def tcp_from_zed_camera_matrix():
    """Return the fixed calibrated mount transform tcp_link<-zed_camera_link."""
    matrix = rpy_matrix(*TCP_FROM_ZED_CAMERA_RPY_RAD)
    matrix[:3, 3] = TCP_FROM_ZED_CAMERA_TRANSLATION_M
    return matrix


def read_capture_tcp_pose():
    """Read the calibrated camera/TCP pose from the arm computer at capture time.

    The TCP was calibrated after mounting the camera, so the arm base is a
    stable world frame for a stationary vehicle.  A missing pose never blocks
    a capture; that frame simply falls back to normal ICP.
    """
    command = (
        "source ~/agx_arm_ws/install/setup.bash && "
        "export ROS_DOMAIN_ID=36 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp "
        "CYCLONEDDS_URI=file://$HOME/zed_code/arm_control/cyclonedds_agx_eth0.xml && "
        "python3 ~/zed_code/arm_control/read_calibrated_tcp_pose.py --timeout 2.0"
    )
    try:
        result = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=3", REMOTE_HOST, command],
            text=True, capture_output=True, timeout=5,
        )
        match = re.search(r"TCP_POSE_JSON=(.*)", result.stdout + result.stderr)
        if result.returncode or not match or match.group(1).strip() == "null":
            return None
        pose = json.loads(match.group(1))
        position = pose["position_m"]
        quaternion = pose["quaternion_xyzw"]
        if len(position) != 3 or len(quaternion) != 4:
            return None
        world_from_tcp = tcp_pose_to_matrix(position, quaternion)
        # Previous versions incorrectly stored world_from_tcp under the name
        # world_from_camera.  Compose the fixed camera mount extrinsic so a
        # ZED depth point is transformed from the actual camera frame.
        world_from_camera = world_from_tcp @ tcp_from_zed_camera_matrix()
        return {
            "position_m": [float(value) for value in position],
            "quaternion_xyzw": [float(value) for value in quaternion],
            "world_from_tcp": world_from_tcp.tolist(),
            "tcp_from_zed_camera": tcp_from_zed_camera_matrix().tolist(),
            "world_from_camera": world_from_camera.tolist(),
            "camera_pose_source": "feedback/tcp_pose + tcp_link_to_zed_camera_link",
        }
    except (OSError, subprocess.TimeoutExpired, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None


def clean_mask(mask: np.ndarray) -> np.ndarray:
    binary = mask.astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if count > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        binary = (labels == largest).astype(np.uint8)
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))).astype(bool)


def sam_from_prompt(predictor, box, points, labels):
    masks, _scores, _logits = predictor.predict(
        box=np.asarray(box, np.float32)[None, :],
        point_coords=np.asarray(points, np.float32) if points else None,
        point_labels=np.asarray(labels, np.int32) if labels else None,
        multimask_output=False,
    )
    return clean_mask(np.asarray(masks[0], dtype=bool))


class RemoteCapture:
    """Client for the bridge carried inside an SSH port-forward."""
    def __init__(self, host="127.0.0.1", port=47777):
        self.host, self.port, self.socket = host, port, None

    def _connect(self):
        if self.socket is None:
            self.socket = socket.create_connection((self.host, self.port), timeout=3)
            self.socket.settimeout(8)

    def _read_exact(self, count):
        data = b""
        while len(data) < count:
            chunk = self.socket.recv(count - len(data))
            if not chunk:
                raise ConnectionError("remote ZED bridge closed the tunnel")
            data += chunk
        return data

    def _request(self, command):
        try:
            self._connect(); self.socket.sendall(command)
            if self._read_exact(1) != b"O":
                return None, None
            jpeg_size, depth_size = struct.unpack("!II", self._read_exact(8))
            jpeg = self._read_exact(jpeg_size)
            depth_bytes = self._read_exact(depth_size) if depth_size else b""
            bgr = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
            if bgr is None:
                return None, None
            depth = None
            if depth_bytes:
                values = np.frombuffer(zlib.decompress(depth_bytes), dtype=np.float32)
                if values.size == bgr.shape[0] * bgr.shape[1]:
                    depth = values.reshape(bgr.shape[:2]).copy()
            return bgr, depth
        except (OSError, ConnectionError, zlib.error):
            if self.socket is not None:
                self.socket.close()
            self.socket = None
            return None, None

    def live_frame(self):
        return self._request(b"L")

    def capture_frame(self):
        return self._request(b"C")

    def close(self):
        if self.socket is not None:
            self.socket.close()


def depth_preview(depth: np.ndarray | None, size: tuple[int, int]) -> np.ndarray:
    """Colourise the low-rate metric depth map without hiding invalid pixels."""
    width, height = size
    shown = np.zeros((height, width, 3), np.uint8)
    if depth is None:
        cv2.putText(shown, "Depth waiting...", (18, height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 220, 255), 2)
        return shown
    valid = np.isfinite(depth) & (depth > .2) & (depth < 8.0)
    normalized = np.zeros(depth.shape, np.uint8)
    normalized[valid] = np.clip((depth[valid] - .2) / 7.8 * 255, 0, 255).astype(np.uint8)
    coloured = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    coloured[~valid] = 0
    shown = cv2.resize(coloured, (width, height), interpolation=cv2.INTER_NEAREST)
    cv2.putText(shown, "Live depth (about 1 Hz)", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
    return shown


def yolo_object_prompt(model, bgr):
    """Return one full-object SAM2 prompt from all relevant YOLO detections.

    The trained model has separate ``body`` and ``wing`` classes.  Choosing
    only the highest-confidence box silently loses the other class, so use a
    union box and a few positive points from the YOLO segmentation masks.
    """
    result = model(bgr, verbose=False, imgsz=640, conf=.20)[0]
    if result.boxes is None or len(result.boxes) == 0:
        return None, [], [], []
    boxes = result.boxes.xyxy.detach().cpu().numpy()
    scores = result.boxes.conf.detach().cpu().numpy()
    classes = result.boxes.cls.detach().cpu().numpy().astype(int)
    height, width = bgr.shape[:2]
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    # Tiny boxes are normally reflections or incomplete predictions.  The
    # remaining threshold stays intentionally modest: SAM2 performs the fine
    # silhouette separation after YOLO provides the object location.
    valid = np.flatnonzero((scores >= .35) & (areas >= width * height * .0025))
    if not len(valid):
        return None, [], [], []
    names = getattr(result, "names", {})
    class_names = [str(names.get(int(class_id), class_id)) for class_id in classes]
    # The body is the physical main shell.  A common failure case in this
    # workshop is floor/stand being classified as a high-confidence wing, so
    # choose a body anchor whenever one is available rather than blindly
    # choosing the maximum confidence prediction.
    body_candidates = [index for index in valid if class_names[index].lower() == "body"]
    anchor_pool = body_candidates if body_candidates else list(valid)
    anchor = max(anchor_pool, key=lambda index: float(scores[index]) * np.sqrt(max(1.0, areas[index])))

    mask_data = None
    masks = getattr(result, "masks", None)
    if masks is not None and masks.data is not None:
        mask_data = masks.data.detach().cpu().numpy()

    def mask_for(index):
        if mask_data is None or index >= len(mask_data):
            return None
        return cv2.resize(mask_data[index], (width, height), interpolation=cv2.INTER_NEAREST) > .5

    anchor_mask = mask_for(anchor)
    selected = [anchor]
    for index in valid:
        if index == anchor or scores[index] < max(.35, scores[anchor] * .45):
            continue
        # Ultralytics sometimes emits a second, larger box of the same body
        # instance.  Its segmentation frequently leaks into the workshop
        # background; it is a duplicate prediction, not another object part.
        if class_names[index].lower() == class_names[anchor].lower():
            continue
        candidate_mask = mask_for(index)
        # Include duplicate body detections and an attached wing, but never a
        # separate floor/stand wing prediction.  The 21x21 dilation permits a
        # thin real joint between body and wing without joining distant items.
        connected = False
        if anchor_mask is not None and candidate_mask is not None:
            overlap = np.count_nonzero(anchor_mask & candidate_mask)
            connected = overlap > 0 or bool((cv2.dilate(anchor_mask.astype(np.uint8), np.ones((21, 21), np.uint8)) & candidate_mask).any())
        else:
            ax1, ay1, ax2, ay2 = boxes[anchor]
            bx1, by1, bx2, by2 = boxes[index]
            connected = max(ax1, bx1) <= min(ax2, bx2) and max(ay1, by1) <= min(ay2, by2)
        if connected:
            selected.append(index)
    selected = np.asarray(selected, dtype=int)
    selected_boxes = boxes[selected]
    box = np.array((selected_boxes[:, 0].min(), selected_boxes[:, 1].min(),
                    selected_boxes[:, 2].max(), selected_boxes[:, 3].max()), dtype=np.float32)
    details = []
    for index in selected:
        label = class_names[index]
        details.append((label, float(scores[index]), boxes[index].astype(np.float32)))

    points, labels = [], []
    if mask_data is not None:
        foreground = np.zeros((height, width), dtype=bool)
        for index in selected:
            if index < len(mask_data):
                item = cv2.resize(mask_data[index], (width, height), interpolation=cv2.INTER_NEAREST) > .5
                foreground |= item
        ys, xs = np.where(foreground)
        if len(xs):
            # Three spatially separated foreground points make SAM2 less
            # likely to stop at only the body or only the wing.
            for fraction in (.20, .50, .80):
                point_index = min(len(xs) - 1, int(len(xs) * fraction))
                points.append((float(xs[point_index]), float(ys[point_index])))
                labels.append(1)
    return box, points, labels, details


def best_yolo_box(model, bgr):
    """Backward-compatible full-object box helper for older review callers."""
    box, _points, _labels, _details = yolo_object_prompt(model, bgr)
    return box


def select_box(bgr):
    x, y, w, h = cv2.selectROI("R: redraw SAM2 box, Enter confirms", bgr, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow("R: redraw SAM2 box, Enter confirms")
    return None if w < 8 or h < 8 else np.array([x, y, x + w, y + h], np.float32)


def review_mask(predictor, detector, bgr, depth):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    box, points, labels, yolo_details = yolo_object_prompt(detector, bgr)
    automatic = box is not None
    if box is None:
        # Keep the workflow automatic even when a rare view is missed by YOLO.
        h, w = bgr.shape[:2]
        box = np.array([w * .04, h * .04, w * .96, h * .96], np.float32)
        yolo_details = []
    # Encoding the image is the expensive SAM2 step. Do it once per capture;
    # subsequent P/C/X corrections reuse that image embedding.
    predictor.set_image(rgb)
    mask = sam_from_prompt(predictor, box, points, labels)
    point_edit = False
    clicks = []
    cursor = [bgr.shape[1] // 2, bgr.shape[0] // 2]
    title = "Remote ZED YOLO + SAM2 review"

    def mouse(event, x, y, _flags, _userdata):
        cursor[:] = [x, y]
        if not point_edit:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            clicks.append((float(x), float(y), 1))
        elif event == cv2.EVENT_RBUTTONDOWN:
            clicks.append((float(x), float(y), 0))

    cv2.namedWindow(title)
    cv2.setMouseCallback(title, mouse)
    while True:
        while clicks:
            x, y, label = clicks.pop(0)
            points.append((x, y)); labels.append(label)
            mask = sam_from_prompt(predictor, box, points, labels)
        preview = bgr.copy()
        preview[mask] = (preview[mask] * .42 + np.array((0, 220, 0)) * .58).astype(np.uint8)
        x1, y1, x2, y2 = np.round(box).astype(int)
        cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 255), 2)
        for detail_index, (label, confidence, detail_box) in enumerate(yolo_details):
            dx1, dy1, dx2, dy2 = np.round(detail_box).astype(int)
            cv2.rectangle(preview, (dx1, dy1), (dx2, dy2), (255, 180, 0), 1)
            cv2.putText(preview, f"{label} {confidence:.0%}", (dx1, max(92, dy1 - 5 - detail_index * 18)),
                        cv2.FONT_HERSHEY_SIMPLEX, .48, (255, 180, 0), 1, cv2.LINE_AA)
        for (x, y), label in zip(points, labels):
            color = (0, 255, 0) if label else (0, 0, 255)
            if label:
                cv2.circle(preview, (round(x), round(y)), 6, color, -1)
            else:
                cv2.drawMarker(preview, (round(x), round(y)), color,
                               markerType=cv2.MARKER_TILTED_CROSS, markerSize=14, thickness=2)
        valid = mask & np.isfinite(depth) & (depth > .2) & (depth < 8.0)
        text = "YOLO + SAM2" if automatic else "SAM2 automatic fallback (R=manual box)"
        cv2.putText(preview, text + (" | P: point edit ON" if point_edit else " | P: enable point edit"),
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 255, 255), 2)
        cv2.putText(preview, "Enter=save | P=edit | left/C=green add | right/X=red X remove | Z=undo | R=redraw | Q=cancel",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, .42, (0, 255, 255), 1)
        cv2.putText(preview, f"mask={mask.mean():.1%} depth-valid={valid.sum()/max(1, mask.sum()):.1%}",
                    (10, 72), cv2.FONT_HERSHEY_SIMPLEX, .42, (255, 255, 255), 1)
        cv2.imshow(title, preview)
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("p"), ord("P")):
            point_edit = True
        elif point_edit and key in (ord("c"), ord("C")):
            clicks.append((float(cursor[0]), float(cursor[1]), 1))
        elif point_edit and key in (ord("x"), ord("X")):
            clicks.append((float(cursor[0]), float(cursor[1]), 0))
        elif key in (ord("z"), ord("Z")) and points:
            points.pop(); labels.pop()
            mask = sam_from_prompt(predictor, box, points, labels)
        elif key in (ord("r"), ord("R")):
            new_box = select_box(bgr)
            if new_box is not None:
                box, points, labels, yolo_details = new_box, [], [], []
                predictor.set_image(rgb)
                mask = sam_from_prompt(predictor, box, points, labels)
                automatic = False
        elif key in (10, 13):
            cv2.destroyWindow(title)
            return mask, preview, box, automatic
        elif key in (27, ord("q"), ord("Q")):
            cv2.destroyWindow(title)
            return None


def main():
    parser = argparse.ArgumentParser(description="Remote ZED capture with YOLO + SAM2 review")
    parser.add_argument("--auto-icp", action="store_true", help="build each saved online frame and refresh fused ICP")
    parser.add_argument("--output-root", type=Path, default=ONLINE_ROOT,
                        help="directory holding online scan folders")
    args = parser.parse_args()
    output = args.output_root.expanduser().resolve() / datetime.now().strftime("scan_%Y%m%d_%H%M%S")
    for name in ("images", "depth", "masks", "preview"):
        (output / name).mkdir(parents=True, exist_ok=True)
    (output / "camera_intrinsics.json").write_text(json.dumps({
        "width": REMOTE_WIDTH, "height": REMOTE_HEIGHT, **DEFAULT_INTRINSICS,
        "depth_unit": "meter", "min_depth_m": .2, "max_depth_m": 8.0}, indent=2), encoding="utf-8")
    if not SEGMENT_WEIGHTS.is_file():
        raise SystemExit(f"YOLO segmentation weights not found: {SEGMENT_WEIGHTS}")
    manifest = (output / "captures.jsonl").open("a", encoding="utf-8", buffering=1)
    detector = YOLO(str(SEGMENT_WEIGHTS))
    predictor = load_sam2(PROJECT / "yolo_execise" / "libraries" / "sam2" / "sam2" / "configs" / "sam2.1" / "sam2.1_hiera_t.yaml",
                          PROJECT / "yolo_execise" / "libraries" / "sam2" / "checkpoints" / "sam2.1_hiera_tiny.pt")
    node = RemoteCapture()
    fusion_viewer = None
    saved = 0
    print(f"[ready] {output}\nWaiting for the SSH ZED bridge. Enter=capture -> YOLO + SAM2; P/R are used in review; Q=quit")
    try:
        while True:
            live, live_depth = node.live_frame()
            if live is None:
                blank = np.zeros((360, 640, 3), np.uint8)
                cv2.putText(blank, "Waiting for remote ZED / SSH bridge...", (25, 180),
                            cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 220, 255), 2)
                cv2.imshow("Remote ZED capture", blank)
                if cv2.waitKey(250) & 0xFF in (27, ord("q"), ord("Q")):
                    break
                continue
            shown = live.copy()
            cv2.putText(shown, f"Remote ZED | saved {saved} | Enter/Space=capture + SAM2 | Q=quit", (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), 2)
            # RGB remains fast at 320x180.  Depth is a separate low-rate panel
            # so seeing it does not force full neural depth on every preview frame.
            panel = depth_preview(live_depth, (shown.shape[1], shown.shape[0]))
            cv2.imshow("Remote ZED capture", np.hstack((shown, panel)))
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
            if key not in (10, 13, ord(" ")):
                continue
            waiting = np.zeros_like(np.hstack((shown, panel)))
            cv2.putText(waiting, "Capturing full RGB + depth, please wait...", (20, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 220, 255), 2)
            cv2.imshow("Remote ZED capture", waiting)
            cv2.waitKey(1)
            bgr, depth = node.capture_frame()
            if bgr is None or depth is None:
                print("[warning] depth frame timeout; retry Enter/Space")
                continue
            # Sample the calibrated camera/TCP pose immediately after the
            # remote bridge returns this captured RGB/depth pair. Operators
            # should let the arm settle before pressing Enter/Space.
            capture_tcp = read_capture_tcp_pose()
            selected = review_mask(predictor, detector, bgr, depth)
            if selected is None:
                continue
            mask, preview, box, automatic = selected
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            cv2.imwrite(str(output / "images" / f"{stamp}.jpg"), bgr)
            np.save(output / "depth" / f"{stamp}_depth.npy", depth)
            cv2.imwrite(str(output / "masks" / f"{stamp}.png"), mask.astype(np.uint8) * 255)
            cv2.imwrite(str(output / "preview" / f"{stamp}.jpg"), preview)
            manifest.write(json.dumps({
                "id": stamp,
                "pose_source": "calibrated_tcp" if capture_tcp else "unavailable",
                "world_from_camera": capture_tcp["world_from_camera"] if capture_tcp else None,
                "calibrated_tcp": capture_tcp,
            }, ensure_ascii=False) + "\n")
            pose_text = "TCP pose saved" if capture_tcp else "TCP pose unavailable"
            print(f"[saved] {stamp} | {'YOLO+SAM2' if automatic else 'manual SAM2'} | {pose_text}")
            saved += 1
            if args.auto_icp:
                pipeline = args.output_root.expanduser().resolve() / "online_icp_pipeline.py"
                viewer = args.output_root.expanduser().resolve() / "live_fusion_viewer.py"
                if not pipeline.is_file():
                    print(f"[warning] Online ICP script not found: {pipeline}")
                    continue
                print("[cloud] Building this frame and refreshing ICP fusion…")
                result = subprocess.run(
                    [sys.executable, str(pipeline), str(output), stamp],
                    text=True, capture_output=True,
                )
                if result.stdout:
                    print(result.stdout.rstrip())
                if result.returncode:
                    print("[warning] Online cloud/ICP failed:\n" + result.stderr[-1200:])
                elif viewer.is_file() and (fusion_viewer is None or fusion_viewer.poll() is not None):
                    fusion_viewer = subprocess.Popen([sys.executable, str(viewer), str(output)])
    finally:
        manifest.close()
        node.close(); cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
