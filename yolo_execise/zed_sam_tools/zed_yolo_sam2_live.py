#!/usr/bin/env python3
"""Live ZED object recognition with a segmented colored point cloud.

The detector uses the local ``best.pt`` model.  Supplying both SAM2 paths turns
the YOLO boxes into instance masks; without them the program deliberately uses
a rectangular fallback and labels the overlay accordingly.
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pyzed.sl as sl


ROOT = Path(__file__).resolve().parent


def load_yolo(weights: Path):
    # Load system OpenCV/NumPy first: local YOLO runtime contains its own NumPy.
    runtime = ROOT / "yolo_runtime"
    if runtime.is_dir():
        sys.path.insert(0, str(runtime))
    os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")
    from ultralytics import YOLO
    return YOLO(str(weights))


def load_sam2(config: Path | None, checkpoint: Path | None):
    if config is None or checkpoint is None:
        return None
    # SAM2 and its small Python-only dependencies are kept project-local so
    # they do not overwrite the existing ROS, YOLO or Open3D environments.
    for directory in (ROOT / "sam2_runtime", ROOT / "sam2", ROOT / "yolo_runtime"):
        if directory.is_dir() and str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        import torch
    except ImportError as error:
        raise RuntimeError(
            "SAM2 was requested but is not installed. Install SAM2 in the "
            "active Python environment, then pass --sam2-config and "
            "--sam2-checkpoint."
        ) from error
    if not config.is_file() or not checkpoint.is_file():
        raise RuntimeError("SAM2 config or checkpoint path does not exist.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"SAM2 device: {device}")
    # Resolve the compatibility symlink before comparing paths.  Otherwise an
    # absolute config from libraries/sam2 incorrectly falls back to Hydra's
    # unsupported absolute-name mode.
    bundled_configs = (ROOT / "sam2" / "sam2" / "configs").resolve()
    try:
        # SAM2's Hydra package resolves its own config names relative to this
        # directory rather than accepting an absolute filename.
        config_name = "configs/" + str(config.resolve().relative_to(bundled_configs))
    except ValueError:
        config_name = str(config)
    model = build_sam2(config_name, str(checkpoint), device=device)
    return SAM2ImagePredictor(model)


def clean_mask(mask: np.ndarray) -> np.ndarray:
    binary = (mask.astype(np.uint8) * 255)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if count > 1:
        keep = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        binary = np.where(labels == keep, 255, 0).astype(np.uint8)
    binary = cv2.morphologyEx(
        binary, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
    )
    return binary.astype(bool)


def box_mask(shape: tuple[int, int], box: np.ndarray) -> np.ndarray:
    height, width = shape
    x1, y1, x2, y2 = np.round(box).astype(int)
    mask = np.zeros((height, width), dtype=bool)
    mask[max(0, y1):min(height, y2), max(0, x1):min(width, x2)] = True
    return mask


def instance_mask(predictor, rgb: np.ndarray, box: np.ndarray) -> np.ndarray:
    if predictor is None:
        return box_mask(rgb.shape[:2], box)
    predictor.set_image(rgb)
    masks, _scores, _logits = predictor.predict(
        point_coords=None, point_labels=None, box=box[None, :],
        multimask_output=False,
    )
    return clean_mask(np.squeeze(masks[0]))


def masked_cloud(xyz: np.ndarray, bgr: np.ndarray, mask: np.ndarray,
                 stride: int, max_depth_m: float):
    valid = mask & np.isfinite(xyz[:, :, 0]) & np.isfinite(xyz[:, :, 1]) \
        & np.isfinite(xyz[:, :, 2]) & (xyz[:, :, 2] > 0) \
        & (xyz[:, :, 2] <= max_depth_m * 1000.0)
    rows, cols = np.where(valid)
    rows, cols = rows[::stride], cols[::stride]
    # ZED XYZ coordinates are millimetres because we set UNIT.MILLIMETER.
    points = xyz[rows, cols, :3] / 1000.0
    colors = bgr[rows, cols, ::-1] / 255.0
    return points.astype(np.float64), colors.astype(np.float64)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Display recognized ZED objects and their masked point clouds."
    )
    parser.add_argument("--weights", type=Path, default=ROOT / "best.pt")
    parser.add_argument("--confidence", type=float, default=0.65)
    parser.add_argument("--sam2-config", type=Path)
    parser.add_argument("--sam2-checkpoint", type=Path)
    parser.add_argument("--max-depth-m", type=float, default=8.0)
    parser.add_argument("--point-stride", type=int, default=3)
    parser.add_argument("--no-3d", action="store_true")
    args = parser.parse_args()
    if not args.weights.is_file():
        raise SystemExit(f"YOLO weights not found: {args.weights}")
    if args.point_stride < 1:
        raise SystemExit("--point-stride must be at least 1")

    yolo = load_yolo(args.weights)
    sam2 = load_sam2(args.sam2_config, args.sam2_checkpoint)
    segmentation_mode = "SAM2 mask" if sam2 else "YOLO box fallback (not segmentation)"
    print(f"Mode: {segmentation_mode}")

    camera = sl.Camera()
    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.HD720
    init.camera_fps = 30
    init.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    init.coordinate_units = sl.UNIT.MILLIMETER
    status = camera.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        raise SystemExit(f"Could not open ZED: {status}")

    image_zed, xyz_zed = sl.Mat(), sl.Mat()
    visualizer = None
    cloud = None
    if not args.no_3d:
        try:
            import open3d as o3d
            visualizer = o3d.visualization.Visualizer()
            visualizer.create_window("ZED recognized-object point cloud", 1100, 800)
            cloud = o3d.geometry.PointCloud()
            visualizer.add_geometry(cloud)
        except Exception as error:
            print(f"3D viewer unavailable ({error}); continuing with 2D overlay.")

    colors = ((0, 255, 0), (255, 160, 0), (255, 0, 255), (0, 255, 255))
    frame = 0
    try:
        while True:
            if camera.grab() != sl.ERROR_CODE.SUCCESS:
                continue
            camera.retrieve_image(image_zed, sl.VIEW.LEFT)
            bgr = cv2.cvtColor(image_zed.get_data(), cv2.COLOR_BGRA2BGR)
            camera.retrieve_measure(xyz_zed, sl.MEASURE.XYZRGBA)
            xyz = xyz_zed.get_data()
            output = bgr.copy()
            result = yolo.predict(bgr, conf=args.confidence, verbose=False)[0]
            all_points, all_colors = [], []
            if result.boxes is not None:
                boxes = result.boxes.xyxy.detach().cpu().numpy()
                classes = result.boxes.cls.detach().cpu().numpy().astype(int)
                scores = result.boxes.conf.detach().cpu().numpy()
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                for index, (box, class_id, score) in enumerate(zip(boxes, classes, scores)):
                    mask = instance_mask(sam2, rgb, box)
                    points, point_colors = masked_cloud(
                        xyz, bgr, mask, args.point_stride, args.max_depth_m
                    )
                    if len(points):
                        all_points.append(points)
                        all_colors.append(point_colors)
                    color = colors[index % len(colors)]
                    x1, y1, x2, y2 = np.round(box).astype(int)
                    label = f"{result.names[class_id]} {score:.0%}"
                    cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(output, label, (x1, max(25, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
                    output[mask] = (0.55 * output[mask] + 0.45 * np.array(color)).astype(np.uint8)
            cv2.putText(output, segmentation_mode, (16, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow("ZED YOLO + SAM2 recognized objects", output)
            if visualizer is not None and frame % 2 == 0:
                import open3d as o3d
                if all_points:
                    cloud.points = o3d.utility.Vector3dVector(np.vstack(all_points))
                    cloud.colors = o3d.utility.Vector3dVector(np.vstack(all_colors))
                else:
                    cloud.clear()
                visualizer.update_geometry(cloud)
                visualizer.poll_events()
                visualizer.update_renderer()
            frame += 1
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        camera.close()
        cv2.destroyAllWindows()
        if visualizer is not None:
            visualizer.destroy_window()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
