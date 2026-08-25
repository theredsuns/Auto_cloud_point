#!/usr/bin/env python3
"""Convert saved local ZED RGB/depth/mask groups into colored point clouds."""

import argparse
from pathlib import Path
import warnings

import cv2
import numpy as np
import open3d as o3d
from scipy.ndimage import distance_transform_edt


ROOT = Path(__file__).resolve().parent / "datasets/local_zed_capture"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--latest",
        type=int,
        default=0,
        help="only process the latest N captures; 0 processes all captures",
    )
    parser.add_argument("--output-name", default="fused_object.ply")
    parser.add_argument("--min-depth-m", type=float, default=0.2)
    parser.add_argument("--max-depth-m", type=float, default=8.0)
    parser.add_argument("--voxel-size-m", type=float, default=0.004)
    parser.add_argument(
        "--repair-mask-depth",
        action="store_true",
        help=(
            "combine selected frames, reject implausible depths, and fill every "
            "mask pixel from the nearest reliable object depth"
        ),
    )
    parser.add_argument(
        "--flat-mask",
        action="store_true",
        help=(
            "place the complete mask on one robust median-depth plane; this "
            "removes all floating layers but intentionally discards depth shape"
        ),
    )
    # ZED S/N 21417, HD720 calibration read on 2026-07-25.
    parser.add_argument("--fx", type=float, default=672.917602539)
    parser.add_argument("--fy", type=float, default=672.917602539)
    parser.add_argument("--cx", type=float, default=620.836303711)
    parser.add_argument("--cy", type=float, default=351.999664307)
    parser.add_argument("--view", action="store_true")
    args = parser.parse_args()
    if args.latest < 0:
        parser.error("--latest must be non-negative")
    if args.min_depth_m >= args.max_depth_m:
        parser.error("--min-depth-m must be smaller than --max-depth-m")
    if args.voxel_size_m <= 0:
        parser.error("--voxel-size-m must be positive")
    if Path(args.output_name).name != args.output_name:
        parser.error("--output-name must be a filename, not a path")
    return args


def load_cloud(depth_path, args):
    capture_id = depth_path.name.removesuffix("_depth.npy")
    image_path = ROOT / "images" / f"{capture_id}.jpg"
    mask_path = ROOT / "masks" / f"{capture_id}.png"
    if not image_path.is_file() or not mask_path.is_file():
        raise RuntimeError(f"incomplete capture group: {capture_id}")

    depth = np.load(depth_path)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if depth is None or mask is None or bgr is None:
        raise RuntimeError(f"could not read capture group: {capture_id}")
    if depth.shape != mask.shape or depth.shape != bgr.shape[:2]:
        raise RuntimeError(f"shape mismatch in capture group: {capture_id}")

    valid = (
        (mask > 0)
        & np.isfinite(depth)
        & (depth > args.min_depth_m)
        & (depth < args.max_depth_m)
    )
    rows, columns = np.where(valid)
    z = depth[rows, columns].astype(np.float64)
    points = np.column_stack(
        (
            (columns - args.cx) * z / args.fx,
            (rows - args.cy) * z / args.fy,
            z,
        )
    )
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    colors = rgb[rows, columns].astype(np.float64) / 255.0

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(colors)
    return capture_id, cloud


def build_mask_repaired_cloud(depth_paths, args):
    depths = []
    masks = []
    latest_bgr = None
    for depth_path in depth_paths:
        capture_id = depth_path.name.removesuffix("_depth.npy")
        image_path = ROOT / "images" / f"{capture_id}.jpg"
        mask_path = ROOT / "masks" / f"{capture_id}.png"
        depth = np.load(depth_path)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        latest_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if mask is None or latest_bgr is None:
            raise RuntimeError(f"incomplete capture group: {capture_id}")
        depths.append(depth.astype(np.float32))
        masks.append(mask > 0)

    depth_stack = np.stack(depths)
    mask_stack = np.stack(masks)
    target_mask = mask_stack.sum(axis=0) >= (len(masks) + 1) // 2
    reliable = (
        mask_stack
        & np.isfinite(depth_stack)
        & (depth_stack > args.min_depth_m)
        & (depth_stack < args.max_depth_m)
    )
    samples = np.where(reliable, depth_stack, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        median_depth = np.nanmedian(samples, axis=0)

    known = target_mask & np.isfinite(median_depth)
    if not np.any(known):
        raise RuntimeError("no reliable depth remains inside the masks")

    if args.flat_mask:
        stable_depth = float(np.nanmedian(median_depth[known]))
        repaired_depth = np.full(median_depth.shape, stable_depth, dtype=np.float32)
        print(f"Stable mask plane depth: {stable_depth:.3f} m")
    else:
        # Find the nearest reliable target pixel for every bad/missing target pixel.
        nearest = distance_transform_edt(
            ~known, return_distances=False, return_indices=True
        )
        filled_depth = median_depth[tuple(nearest)].astype(np.float32)

        # Smooth only the repaired field; retain measured median values where reliable.
        smoothed = cv2.GaussianBlur(filled_depth, (9, 9), 0)
        repaired_depth = np.where(known, median_depth, smoothed)
    rows, columns = np.where(target_mask)
    z = repaired_depth[rows, columns].astype(np.float64)
    points = np.column_stack(
        (
            (columns - args.cx) * z / args.fx,
            (rows - args.cy) * z / args.fy,
            z,
        )
    )
    rgb = cv2.cvtColor(latest_bgr, cv2.COLOR_BGR2RGB)
    colors = rgb[rows, columns].astype(np.float64) / 255.0
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(colors)
    repaired_count = int(np.count_nonzero(target_mask & ~known))
    return cloud, int(np.count_nonzero(known)), repaired_count


def main():
    args = parse_args()
    output_dir = ROOT / "pointcloud"
    output_dir.mkdir(parents=True, exist_ok=True)

    depth_paths = sorted((ROOT / "depth").glob("*_depth.npy"))
    if args.latest:
        depth_paths = depth_paths[-args.latest :]
    if not depth_paths:
        raise SystemExit("No saved depth files were found.")

    if args.repair_mask_depth or args.flat_mask:
        fused, measured_count, repaired_count = build_mask_repaired_cloud(
            depth_paths, args
        )
        fused = fused.voxel_down_sample(args.voxel_size_m)
        output_path = output_dir / args.output_name
        if not o3d.io.write_point_cloud(str(output_path), fused, compressed=True):
            raise RuntimeError(f"could not write {output_path}")
        print(
            f"Mask-guided cloud: {measured_count} measured pixels, "
            f"{repaired_count} repaired pixels, {len(fused.points)} output points."
        )
        print(f"Saved: {output_path.resolve()}")
        if args.view:
            o3d.visualization.draw_geometries(
                [fused],
                window_name="Mask-guided repaired ZED point cloud",
                width=1200,
                height=850,
            )
        return

    clouds = []
    for index, depth_path in enumerate(depth_paths, 1):
        capture_id, cloud = load_cloud(depth_path, args)
        destination = output_dir / f"{capture_id}.ply"
        if not o3d.io.write_point_cloud(str(destination), cloud, compressed=True):
            raise RuntimeError(f"could not write {destination}")
        clouds.append(cloud)
        print(
            f"[{index}/{len(depth_paths)}] {capture_id}: "
            f"{len(cloud.points)} points",
            flush=True,
        )

    fused = o3d.geometry.PointCloud()
    for cloud in clouds:
        fused += cloud
    fused = fused.voxel_down_sample(args.voxel_size_m)
    if len(fused.points) >= 20:
        fused, _ = fused.remove_statistical_outlier(
            nb_neighbors=20, std_ratio=1.5
        )

    output_path = output_dir / args.output_name
    if not o3d.io.write_point_cloud(str(output_path), fused, compressed=True):
        raise RuntimeError(f"could not write {output_path}")
    print(f"Fused {len(clouds)} captures into {len(fused.points)} points.")
    print(f"Saved: {output_path.resolve()}")

    if args.view:
        o3d.visualization.draw_geometries(
            [fused],
            window_name="Local ZED depth + SAM2 fused point cloud",
            width=1200,
            height=850,
        )


if __name__ == "__main__":
    main()
