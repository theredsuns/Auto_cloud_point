#!/usr/bin/env python3
"""Fuse saved masked ZED depth with globally optimized COLMAP camera poses."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d


def quaternion_rotation(q):
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def read_colmap_images(path):
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    poses = {}
    for line in lines[::2]:
        fields = line.split()
        rotation_world_to_camera = quaternion_rotation(
            np.asarray(fields[1:5], dtype=np.float64)
        )
        translation_world_to_camera = np.asarray(fields[5:8], dtype=np.float64)
        capture_id = Path(fields[9]).stem
        rotation_camera_to_world = rotation_world_to_camera.T
        camera_center = -rotation_camera_to_world @ translation_world_to_camera
        poses[capture_id] = (rotation_camera_to_world, camera_center)
    return poses


def robust_metric_scale(items, colmap_poses):
    zed_centers = {
        item["id"]: np.asarray(item["world_from_camera"], dtype=np.float64)[:3, 3]
        for item in items
        if item.get("world_from_camera") is not None
    }
    ratios = []
    pair_details = []
    ids = [item["id"] for item in items]
    for first, second in zip(ids, ids[1:]):
        if first not in colmap_poses or second not in colmap_poses:
            continue
        if first not in zed_centers or second not in zed_centers:
            continue
        colmap_step = np.linalg.norm(
            colmap_poses[second][1] - colmap_poses[first][1]
        )
        zed_step = np.linalg.norm(zed_centers[second] - zed_centers[first])
        if colmap_step < 0.03 or zed_step < 0.02:
            continue
        ratio = zed_step / colmap_step
        ratios.append(ratio)
        pair_details.append((first, second, ratio))
    if len(ratios) < 8:
        raise RuntimeError("Too few adjacent registered views to recover metric scale")
    ratios = np.asarray(ratios)
    median = float(np.median(ratios))
    mad = float(np.median(np.abs(ratios - median)))
    tolerance = max(0.035 * median, 3.5 * 1.4826 * mad)
    inliers = np.abs(ratios - median) <= tolerance
    scale = float(np.median(ratios[inliers]))
    return scale, int(inliers.sum()), len(ratios), float(mad)


def rotation_outlier_ids(items, colmap_poses, threshold_degrees=30.0):
    zed_rotations = {
        item["id"]: np.asarray(item["world_from_camera"], dtype=np.float64)[:3, :3]
        for item in items
        if item.get("world_from_camera") is not None
    }
    ids = [item["id"] for item in items]
    bad_incidents = {}
    for first, second in zip(ids, ids[1:]):
        if first not in colmap_poses or second not in colmap_poses:
            continue
        if first not in zed_rotations or second not in zed_rotations:
            continue
        colmap_delta = colmap_poses[first][0].T @ colmap_poses[second][0]
        zed_delta = zed_rotations[first].T @ zed_rotations[second]
        difference = colmap_delta.T @ zed_delta
        cosine = np.clip((np.trace(difference) - 1.0) / 2.0, -1.0, 1.0)
        error_degrees = float(np.degrees(np.arccos(cosine)))
        if error_degrees > threshold_degrees:
            bad_incidents[first] = bad_incidents.get(first, 0) + 1
            bad_incidents[second] = bad_incidents.get(second, 0) + 1
    return {capture_id for capture_id, count in bad_incidents.items() if count >= 2}


def load_frame(root, item, intrinsics, args):
    capture_id = item["id"]
    depth = np.load(root / "depth" / f"{capture_id}_depth.npy")
    mask = cv2.imread(
        str(root / args.mask_dir / f"{capture_id}.png"), cv2.IMREAD_GRAYSCALE
    )
    bgr = cv2.imread(str(root / "images" / f"{capture_id}.jpg"), cv2.IMREAD_COLOR)
    if mask is None or bgr is None:
        return None

    object_mask = (mask > 0).astype(np.uint8)
    if args.mask_erode_px:
        kernel = np.ones((3, 3), np.uint8)
        object_mask = cv2.erode(object_mask, kernel, iterations=args.mask_erode_px)
    valid = (
        (object_mask > 0)
        & np.isfinite(depth)
        & (depth > args.min_depth_m)
        & (depth < args.max_depth_m)
    )
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        valid.astype(np.uint8), 8
    )
    if component_count <= 1:
        return None
    component_order = 1 + np.argsort(stats[1:, cv2.CC_STAT_AREA])[::-1]
    largest_area = int(stats[component_order[0], cv2.CC_STAT_AREA])
    kept_components = [
        int(label)
        for label in component_order[:2]
        if int(stats[label, cv2.CC_STAT_AREA]) >= max(80, int(largest_area * 0.025))
    ]
    valid = np.isin(labels, kept_components)

    rows, columns = np.where(valid)
    z = depth[rows, columns].astype(np.float64)
    points = np.column_stack(
        (
            (columns - intrinsics["cx"]) * z / intrinsics["fx"],
            (rows - intrinsics["cy"]) * z / intrinsics["fy"],
            z,
        )
    )
    colors = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)[rows, columns] / 255.0
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(colors)
    cloud = cloud.voxel_down_sample(args.frame_voxel_m)
    if len(cloud.points) < args.min_frame_points:
        return None
    cloud, _ = cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.6)
    return np.asarray(cloud.points), np.asarray(cloud.colors)


def repeated_view_filter(points, frame_ids, voxel_size, minimum_views):
    origin = points.min(axis=0) - voxel_size
    cells = np.floor((points - origin) / voxel_size).astype(np.int64)
    minimum = cells.min(axis=0)
    cells -= minimum
    span = cells.max(axis=0) + 1
    cell_keys = (cells[:, 0] * span[1] + cells[:, 1]) * span[2] + cells[:, 2]

    frame_cell = np.rec.fromarrays((cell_keys, frame_ids), names=("cell", "frame"))
    unique_frame_cell = np.unique(frame_cell)
    supported_cells, view_counts = np.unique(
        unique_frame_cell["cell"], return_counts=True
    )
    supported_cells = supported_cells[view_counts >= minimum_views]
    return np.isin(cell_keys, supported_cells), len(supported_cells)


def write_outline_preview(cloud, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    points = np.asarray(cloud.points)
    center = np.median(points, axis=0)
    centered = points - center
    _, _, axes = np.linalg.svd(centered, full_matrices=False)
    aligned = centered @ axes.T
    radius = np.linalg.norm(aligned, axis=1)
    aligned = aligned[radius <= np.quantile(radius, 0.997)]
    stride = max(1, len(aligned) // 180000)

    figure, plots = plt.subplots(1, 3, figsize=(15, 5), facecolor="white")
    for plot, (axis_x, axis_y, title) in zip(
        plots,
        ((0, 1, "front"), (0, 2, "top"), (1, 2, "side")),
    ):
        plot.scatter(
            aligned[::stride, axis_x],
            aligned[::stride, axis_y],
            s=0.45,
            c="#174a7e",
            linewidths=0,
        )
        plot.set_aspect("equal", "box")
        plot.set_title(title)
        plot.grid(alpha=0.15)
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def render_and_validate_views(
    cloud, root, mask_dir, used_ids, colmap_poses, scale, intrinsics, output_path
):
    points = np.asarray(cloud.points)
    colors = (np.asarray(cloud.colors)[:, ::-1] * 255).astype(np.uint8)
    sample_indices = sorted(set((0, len(used_ids) // 2, len(used_ids) - 1)))
    renders = []
    metrics = []
    for index in sample_indices:
        capture_id = used_ids[index]
        rotation, camera_center = colmap_poses[capture_id]
        camera = (points - scale * camera_center) @ rotation
        z = camera[:, 2]
        visible = z > 0.1
        columns = np.rint(
            intrinsics["fx"] * camera[:, 0] / np.maximum(z, 1e-6)
            + intrinsics["cx"]
        ).astype(np.int32)
        rows = np.rint(
            intrinsics["fy"] * camera[:, 1] / np.maximum(z, 1e-6)
            + intrinsics["cy"]
        ).astype(np.int32)
        height, width = cv2.imread(
            str(root / mask_dir / f"{capture_id}.png"), cv2.IMREAD_GRAYSCALE
        ).shape
        visible &= (
            (columns >= 0)
            & (columns < width)
            & (rows >= 0)
            & (rows < height)
        )
        selected = np.flatnonzero(visible)
        pixels = rows[selected].astype(np.int64) * width + columns[selected]
        order = np.argsort(z[selected])
        _, first = np.unique(pixels[order], return_index=True)
        nearest = selected[order[first]]

        occupied = np.zeros((height, width), dtype=np.uint8)
        occupied[rows[nearest], columns[nearest]] = 255
        kernel = np.ones((5, 5), np.uint8)
        occupied = cv2.dilate(occupied, kernel, iterations=1)
        image = np.full((height, width, 3), 245, dtype=np.uint8)
        image[occupied > 0] = (126, 74, 23)
        mask = cv2.imread(
            str(root / mask_dir / f"{capture_id}.png"), cv2.IMREAD_GRAYSCALE
        )
        predicted = occupied > 0
        expected = mask > 0
        intersection = int(np.count_nonzero(predicted & expected))
        union = int(np.count_nonzero(predicted | expected))
        precision = intersection / max(int(np.count_nonzero(predicted)), 1)
        coverage = intersection / max(int(np.count_nonzero(expected)), 1)
        iou = intersection / max(union, 1)
        metrics.append(
            {
                "id": capture_id,
                "projection_precision": precision,
                "mask_coverage": coverage,
                "projection_iou": iou,
            }
        )
        cv2.putText(
            image,
            f"{capture_id}  IoU={iou:.2f}",
            (18, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )
        renders.append(image)
    cv2.imwrite(str(output_path), np.hstack(renders))
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--mask-dir", default="masks")
    parser.add_argument(
        "--allow-fallback-masks",
        action="store_true",
        help="include previous-box SAM masks that may contain only one object part",
    )
    parser.add_argument("--min-depth-m", type=float, default=0.20)
    parser.add_argument("--max-depth-m", type=float, default=2.30)
    parser.add_argument("--min-valid-ratio", type=float, default=0.55)
    parser.add_argument("--min-connected-ratio", type=float, default=0.75)
    parser.add_argument("--mask-erode-px", type=int, default=2)
    parser.add_argument("--frame-voxel-m", type=float, default=0.008)
    parser.add_argument("--support-voxel-m", type=float, default=0.018)
    parser.add_argument("--minimum-views", type=int, default=2)
    parser.add_argument("--fused-voxel-m", type=float, default=0.007)
    parser.add_argument("--min-frame-points", type=int, default=400)
    args = parser.parse_args()

    root = args.input.resolve()
    output = args.output.resolve() if args.output else root / "colmap_reconstruction"
    output.mkdir(parents=True, exist_ok=True)
    intrinsics = json.loads((root / "camera_intrinsics.json").read_text())
    items = [
        json.loads(line)
        for line in (root / "captures.jsonl").read_text().splitlines()
        if line.strip()
    ]
    colmap_poses = read_colmap_images(root / "colmap_global/text/images.txt")
    scale, scale_inliers, scale_pairs, scale_mad = robust_metric_scale(
        items, colmap_poses
    )
    pose_outliers = rotation_outlier_ids(items, colmap_poses)
    print(
        f"COLMAP registered: {len(colmap_poses)}/{len(items)}; "
        f"metric scale={scale:.6f} ({scale_inliers}/{scale_pairs} step inliers); "
        f"rotation outliers={len(pose_outliers)}",
        flush=True,
    )

    all_points = []
    all_colors = []
    all_frame_ids = []
    used_ids = []
    for item in items:
        quality = item["quality"]
        if item["id"] not in colmap_poses:
            continue
        if (
            not args.allow_fallback_masks
            and "previous box" in item.get("detection_mode", "").lower()
        ):
            print(f"[skip] incomplete-mask risk: {item['id']}", flush=True)
            continue
        if item["id"] in pose_outliers:
            print(f"[skip] 180-degree pose jump: {item['id']}", flush=True)
            continue
        if quality["valid_depth_ratio"] < args.min_valid_ratio:
            continue
        if quality["largest_depth_component_ratio"] < args.min_connected_ratio:
            continue
        frame = load_frame(root, item, intrinsics, args)
        if frame is None:
            continue
        camera_points, colors = frame
        rotation, colmap_center = colmap_poses[item["id"]]
        world_points = camera_points @ rotation.T + scale * colmap_center
        frame_index = len(used_ids)
        all_points.append(world_points)
        all_colors.append(colors)
        all_frame_ids.append(np.full(len(world_points), frame_index, dtype=np.int16))
        used_ids.append(item["id"])
        print(
            f"[{len(used_ids)}] prepared {item['id']}: {len(world_points)} points",
            flush=True,
        )

    if len(used_ids) < 8:
        raise RuntimeError("Too few quality frames have global camera poses")
    points = np.concatenate(all_points)
    colors = np.concatenate(all_colors)
    frame_ids = np.concatenate(all_frame_ids)
    keep, supported_cell_count = repeated_view_filter(
        points, frame_ids, args.support_voxel_m, args.minimum_views
    )
    points, colors = points[keep], colors[keep]
    if len(points) < 1000:
        raise RuntimeError("Repeated-view filtering removed nearly all points")

    fused = o3d.geometry.PointCloud()
    fused.points = o3d.utility.Vector3dVector(points)
    fused.colors = o3d.utility.Vector3dVector(colors)
    fused = fused.voxel_down_sample(args.fused_voxel_m)
    fused, _ = fused.remove_radius_outlier(
        nb_points=4, radius=args.support_voxel_m * 1.5
    )
    fused, _ = fused.remove_statistical_outlier(nb_neighbors=24, std_ratio=1.4)

    ply_path = output / "fused_colmap_depth.ply"
    preview_path = output / "fused_colmap_depth_preview.png"
    perspective_path = output / "fused_colmap_perspective_preview.jpg"
    o3d.io.write_point_cloud(str(ply_path), fused, compressed=True)
    write_outline_preview(fused, preview_path)
    validation = render_and_validate_views(
        fused,
        root,
        args.mask_dir,
        used_ids,
        colmap_poses,
        scale,
        intrinsics,
        perspective_path,
    )
    summary = {
        "registered_images": len(colmap_poses),
        "used_quality_frames": len(used_ids),
        "mask_directory": args.mask_dir,
        "fallback_masks_allowed": args.allow_fallback_masks,
        "used_capture_ids": used_ids,
        "metric_scale": scale,
        "rejected_rotation_outliers": sorted(pose_outliers),
        "scale_step_inliers": scale_inliers,
        "scale_step_pairs": scale_pairs,
        "scale_mad": scale_mad,
        "support_voxel_m": args.support_voxel_m,
        "minimum_views": args.minimum_views,
        "supported_cells": supported_cell_count,
        "output_points": len(fused.points),
        "projection_validation": validation,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Saved {len(fused.points)} points: {ply_path}", flush=True)
    print(f"Saved outline preview: {preview_path}", flush=True)
    print(f"Saved perspective preview: {perspective_path}", flush=True)


if __name__ == "__main__":
    main()
