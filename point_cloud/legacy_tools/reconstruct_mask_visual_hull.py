#!/usr/bin/env python3
"""Build a sharp mask-constrained visual-hull point cloud from an offline scan."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from scipy.ndimage import binary_closing, binary_erosion

from reconstruct_local_offline import write_preview


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--voxel-m", type=float, default=0.02)
    parser.add_argument("--support-ratio", type=float, default=0.58)
    parser.add_argument("--keyframe-step", type=int, default=2)
    args = parser.parse_args()

    root = args.input.resolve()
    reconstruction = root / "offline_reconstruction"
    metrics = json.loads(
        (reconstruction / "registration_metrics.json").read_text()
    )
    accepted_ids = [
        frame["id"] for frame in metrics["frames"] if frame["accepted"]
    ]
    poses = np.load(reconstruction / "accepted_camera_poses.npy")
    if len(accepted_ids) != len(poses):
        raise RuntimeError("accepted frame IDs and poses do not match")

    keyframe_indices = list(range(0, len(accepted_ids), args.keyframe_step))
    if keyframe_indices[-1] != len(accepted_ids) - 1:
        keyframe_indices.append(len(accepted_ids) - 1)
    intrinsics = json.loads((root / "camera_intrinsics.json").read_text())

    depth_fusion = o3d.io.read_point_cloud(
        str(reconstruction / "fused_offline.ply")
    )
    xyz = np.asarray(depth_fusion.points)
    lower = np.quantile(xyz, 0.005, axis=0) - 0.08
    upper = np.quantile(xyz, 0.995, axis=0) + 0.08
    axes = [
        np.arange(lower[axis], upper[axis] + args.voxel_m, args.voxel_m)
        for axis in range(3)
    ]
    shape = tuple(len(axis) for axis in axes)
    grid = np.stack(
        np.meshgrid(axes[0], axes[1], axes[2], indexing="ij"), axis=-1
    ).reshape(-1, 3)
    seen = np.zeros(len(grid), dtype=np.uint16)
    votes = np.zeros(len(grid), dtype=np.uint16)

    for progress, index in enumerate(keyframe_indices, 1):
        capture_id = accepted_ids[index]
        mask = cv2.imread(
            str(root / "masks" / f"{capture_id}.png"),
            cv2.IMREAD_GRAYSCALE,
        )
        camera_from_world = np.linalg.inv(poses[index])
        camera = (
            grid @ camera_from_world[:3, :3].T
            + camera_from_world[:3, 3]
        )
        z = camera[:, 2]
        valid_z = z > 0.20
        columns = np.zeros(len(grid), dtype=np.int32)
        rows = np.zeros(len(grid), dtype=np.int32)
        columns[valid_z] = np.rint(
            intrinsics["fx"] * camera[valid_z, 0] / z[valid_z]
            + intrinsics["cx"]
        ).astype(np.int32)
        rows[valid_z] = np.rint(
            intrinsics["fy"] * camera[valid_z, 1] / z[valid_z]
            + intrinsics["cy"]
        ).astype(np.int32)
        visible = (
            valid_z
            & (columns >= 0)
            & (columns < mask.shape[1])
            & (rows >= 0)
            & (rows < mask.shape[0])
        )
        selected = np.flatnonzero(visible)
        seen[selected] += 1
        inside = mask[rows[selected], columns[selected]] > 0
        votes[selected[inside]] += 1
        print(
            f"[{progress}/{len(keyframe_indices)}] carved {capture_id}",
            flush=True,
        )

    minimum_views = max(4, len(keyframe_indices) // 4)
    support = votes.astype(np.float32) / np.maximum(seen, 1)
    occupied = (
        (seen >= minimum_views) & (support >= args.support_ratio)
    ).reshape(shape)
    occupied = binary_closing(occupied, iterations=1)
    surface = occupied & ~binary_erosion(occupied, iterations=1)
    surface_points = grid[surface.reshape(-1)]
    if len(surface_points) < 100:
        raise RuntimeError("visual hull is empty; lower --support-ratio")

    # Average image colors from keyframes whose masks contain each surface point.
    color_sum = np.zeros((len(surface_points), 3), dtype=np.float64)
    color_count = np.zeros(len(surface_points), dtype=np.uint16)
    for index in keyframe_indices:
        capture_id = accepted_ids[index]
        bgr = cv2.imread(str(root / "images" / f"{capture_id}.jpg"))
        mask = cv2.imread(
            str(root / "masks" / f"{capture_id}.png"),
            cv2.IMREAD_GRAYSCALE,
        )
        camera_from_world = np.linalg.inv(poses[index])
        camera = (
            surface_points @ camera_from_world[:3, :3].T
            + camera_from_world[:3, 3]
        )
        z = camera[:, 2]
        valid_z = z > 0.20
        columns = np.zeros(len(surface_points), dtype=np.int32)
        rows = np.zeros(len(surface_points), dtype=np.int32)
        columns[valid_z] = np.rint(
            intrinsics["fx"] * camera[valid_z, 0] / z[valid_z]
            + intrinsics["cx"]
        ).astype(np.int32)
        rows[valid_z] = np.rint(
            intrinsics["fy"] * camera[valid_z, 1] / z[valid_z]
            + intrinsics["cy"]
        ).astype(np.int32)
        valid = (
            valid_z
            & (columns >= 0)
            & (columns < mask.shape[1])
            & (rows >= 0)
            & (rows < mask.shape[0])
        )
        ids = np.flatnonzero(valid)
        ids = ids[mask[rows[ids], columns[ids]] > 0]
        color_sum[ids] += bgr[rows[ids], columns[ids], ::-1]
        color_count[ids] += 1
    colors = color_sum / np.maximum(color_count[:, None], 1) / 255.0
    colors[color_count == 0] = np.array([0.75, 0.75, 0.75])

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(surface_points)
    cloud.colors = o3d.utility.Vector3dVector(colors)
    output = reconstruction / "mask_visual_hull.ply"
    o3d.io.write_point_cloud(str(output), cloud, compressed=True)
    write_preview(cloud, reconstruction / "mask_visual_hull_preview.png")
    summary = {
        "keyframes": len(keyframe_indices),
        "grid_shape": shape,
        "voxel_m": args.voxel_m,
        "minimum_views": minimum_views,
        "support_ratio": args.support_ratio,
        "surface_points": len(surface_points),
    }
    (reconstruction / "mask_visual_hull_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Saved {len(surface_points)} sharp hull points: {output}", flush=True)


if __name__ == "__main__":
    main()
