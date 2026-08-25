#!/usr/bin/env python3
"""Generate one colored object point cloud from RGB, depth, and mask files.

Expected input directory (default: object_pointcloud_input):
  rgb.jpg                  RGB image
  depth.npy                float depth image in metres
  mask.png                 white=object, black=background mask
  camera_intrinsics.json   optional camera calibration
"""

import argparse
import json
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import open3d as o3d

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_INTRINSICS = {
    # ZED S/N 21417, HD720 calibration. Replace with the camera's own values
    # for accurate geometry when another camera is used.
    "fx": 672.917602539,
    "fy": 672.917602539,
    "cx": 620.836303711,
    "cy": 351.999664307,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a single-frame colored object point cloud."
    )
    parser.add_argument("--input", type=Path, default=Path("object_pointcloud_input"))
    parser.add_argument("--output", type=Path, default=Path("object_pointcloud_output"))
    parser.add_argument("--min-depth-m", type=float, default=0.20)
    parser.add_argument("--max-depth-m", type=float, default=8.0)
    parser.add_argument("--voxel-size-m", type=float, default=0.004)
    return parser.parse_args()


def load_intrinsics(input_dir: Path) -> dict:
    calibration = input_dir / "camera_intrinsics.json"
    if not calibration.exists():
        print("[info] camera_intrinsics.json not found; using built-in ZED HD720 calibration.")
        return DEFAULT_INTRINSICS.copy()
    try:
        values = json.loads(calibration.read_text(encoding="utf-8"))
        missing = set(DEFAULT_INTRINSICS) - set(values)
        if missing:
            raise ValueError(f"missing fields: {', '.join(sorted(missing))}")
        return {key: float(values[key]) for key in DEFAULT_INTRINSICS}
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise SystemExit(f"Invalid {calibration}: {error}")


def save_preview(points: np.ndarray, colors: np.ndarray, output: Path) -> None:
    # A sampled preview keeps rendering fast even for dense depth images.
    sample = max(1, len(points) // 100_000)
    shown_points, shown_colors = points[::sample], colors[::sample]
    figure = plt.figure(figsize=(8, 7), facecolor="white")
    axes = figure.add_subplot(111, projection="3d")
    axes.scatter(
        shown_points[:, 0], shown_points[:, 1], shown_points[:, 2],
        c=shown_colors, s=0.4, depthshade=False,
    )
    axes.set_xlabel("X (m)")
    axes.set_ylabel("Y (m)")
    axes.set_zlabel("Z (m)")
    axes.set_title(f"Single-frame object point cloud ({len(points):,} points)")
    axes.view_init(elev=18, azim=-65)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main():
    args = parse_args()
    if args.min_depth_m >= args.max_depth_m or args.voxel_size_m <= 0:
        raise SystemExit("Depth limits and voxel size must be positive and ordered.")

    input_dir, output_dir = args.input.resolve(), args.output.resolve()
    rgb_path, depth_path, mask_path = (
        input_dir / "rgb.jpg", input_dir / "depth.npy", input_dir / "mask.png"
    )
    missing = [str(path) for path in (rgb_path, depth_path, mask_path) if not path.is_file()]
    if missing:
        raise SystemExit("Missing required input file(s):\n  " + "\n  ".join(missing))

    bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    depth = np.load(depth_path)
    if bgr is None or mask is None or depth.ndim != 2:
        raise SystemExit("Could not read rgb.jpg, mask.png, or 2-D depth.npy.")
    if depth.shape != mask.shape or depth.shape != bgr.shape[:2]:
        raise SystemExit(
            f"Image sizes must match: RGB={bgr.shape[:2]}, mask={mask.shape}, depth={depth.shape}."
        )

    intrinsics = load_intrinsics(input_dir)
    valid = (
        (mask > 0) & np.isfinite(depth) & (depth > args.min_depth_m)
        & (depth < args.max_depth_m)
    )
    rows, columns = np.where(valid)
    if not len(rows):
        raise SystemExit("No valid object depth pixels remain after masking.")

    z = depth[rows, columns].astype(np.float64)
    points = np.column_stack((
        (columns - intrinsics["cx"]) * z / intrinsics["fx"],
        (rows - intrinsics["cy"]) * z / intrinsics["fy"], z,
    ))
    colors = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)[rows, columns] / 255.0
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(colors)
    cloud = cloud.voxel_down_sample(args.voxel_size_m)

    output_dir.mkdir(parents=True, exist_ok=True)
    ply_path = output_dir / "single_frame_object.ply"
    preview_path = output_dir / "single_frame_object_preview.png"
    o3d.io.write_point_cloud(str(ply_path), cloud, compressed=True)
    saved_points, saved_colors = np.asarray(cloud.points), np.asarray(cloud.colors)
    save_preview(saved_points, saved_colors, preview_path)
    print(f"Saved {len(saved_points):,} points: {ply_path}")
    print(f"Saved preview: {preview_path}")


if __name__ == "__main__":
    main()
