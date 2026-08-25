#!/usr/bin/env python3
"""Offline quality-gated reconstruction of saved local ZED capture groups."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d


def prepared_cloud(root, item, intrinsics, args):
    capture_id = item["id"]
    depth = np.load(root / "depth" / f"{capture_id}_depth.npy")
    mask = cv2.imread(
        str(root / "masks" / f"{capture_id}.png"), cv2.IMREAD_GRAYSCALE
    )
    bgr = cv2.imread(
        str(root / "images" / f"{capture_id}.jpg"), cv2.IMREAD_COLOR
    )
    if mask is None or bgr is None:
        raise RuntimeError(f"incomplete capture group: {capture_id}")

    valid = (
        (mask > 0)
        & np.isfinite(depth)
        & (depth > args.min_depth_m)
        & (depth < args.max_depth_m)
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        valid.astype(np.uint8), 8
    )
    if count <= 1:
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
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    colors = rgb[rows, columns].astype(np.float64) / 255.0
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(colors)
    cloud = cloud.voxel_down_sample(args.frame_voxel_m)
    if len(cloud.points) < args.min_points:
        return None
    cloud, _ = cloud.remove_statistical_outlier(
        nb_neighbors=20, std_ratio=1.8
    )
    cloud.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=args.frame_voxel_m * 4.0, max_nn=50
        )
    )
    return cloud


def register(source, target, voxel, initial):
    transform = initial.copy()
    result = None
    for multiplier, iterations in ((8.0, 60), (4.0, 45), (2.0, 35)):
        result = o3d.pipelines.registration.registration_icp(
            source,
            target,
            voxel * multiplier,
            transform,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=iterations
            ),
        )
        transform = result.transformation
    return result


def better(candidate, current):
    if current is None:
        return True
    if candidate.fitness != current.fitness:
        return candidate.fitness > current.fitness
    return candidate.inlier_rmse < current.inlier_rmse


def write_preview(cloud, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xyz = np.asarray(cloud.points)
    rgb = np.asarray(cloud.colors)
    center = np.median(xyz, axis=0)
    centered = xyz - center
    _, _, axes = np.linalg.svd(centered, full_matrices=False)
    aligned = centered @ axes.T
    radius = np.linalg.norm(aligned, axis=1)
    keep = radius <= np.quantile(radius, 0.995)
    aligned, rgb = aligned[keep], rgb[keep]
    stride = max(1, len(aligned) // 120000)

    figure, plots = plt.subplots(1, 3, figsize=(15, 5), facecolor="#202124")
    for plot, (axis_x, axis_y, title) in zip(
        plots,
        ((0, 1, "front"), (0, 2, "top"), (1, 2, "side")),
    ):
        plot.set_facecolor("#202124")
        plot.scatter(
            aligned[::stride, axis_x],
            aligned[::stride, axis_y],
            s=0.3,
            c=rgb[::stride],
            linewidths=0,
        )
        plot.set_aspect("equal", "box")
        plot.set_title(title, color="white")
        plot.tick_params(colors="white")
    figure.tight_layout()
    figure.savefig(path, dpi=180, facecolor=figure.get_facecolor())
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--min-depth-m", type=float, default=0.20)
    parser.add_argument("--max-depth-m", type=float, default=2.30)
    parser.add_argument("--min-valid-ratio", type=float, default=0.55)
    parser.add_argument("--min-connected-ratio", type=float, default=0.75)
    parser.add_argument("--min-points", type=int, default=300)
    parser.add_argument("--frame-voxel-m", type=float, default=0.015)
    parser.add_argument("--fused-voxel-m", type=float, default=0.012)
    parser.add_argument("--min-fitness", type=float, default=0.25)
    parser.add_argument("--max-rmse-m", type=float, default=0.045)
    args = parser.parse_args()

    root = args.input.resolve()
    output = (
        args.output.resolve()
        if args.output
        else root / "offline_reconstruction"
    )
    output.mkdir(parents=True, exist_ok=True)
    intrinsics = json.loads((root / "camera_intrinsics.json").read_text())
    items = [
        json.loads(line)
        for line in (root / "captures.jsonl").read_text().splitlines()
        if line.strip()
    ]
    candidates = [
        item
        for item in items
        if item["quality"]["valid_depth_ratio"] >= args.min_valid_ratio
        and item["quality"]["largest_depth_component_ratio"]
        >= args.min_connected_ratio
        and item["world_from_camera"] is not None
    ]
    print(
        f"Quality prefilter: {len(candidates)}/{len(items)} frames",
        flush=True,
    )

    accepted = []
    metrics = []
    previous_cloud = None
    previous_zed_pose = None
    world_from_previous = np.eye(4)
    for candidate_index, item in enumerate(candidates, 1):
        cloud = prepared_cloud(root, item, intrinsics, args)
        if cloud is None:
            metrics.append({"id": item["id"], "accepted": False, "reason": "few points"})
            continue
        zed_pose = np.asarray(item["world_from_camera"], dtype=np.float64)
        if previous_cloud is None:
            accepted.append((item, cloud, np.eye(4)))
            previous_cloud = cloud
            previous_zed_pose = zed_pose
            metrics.append(
                {"id": item["id"], "accepted": True, "reason": "anchor"}
            )
            print(f"[{candidate_index}/{len(candidates)}] anchor {item['id']}", flush=True)
            continue

        zed_initial = np.linalg.inv(previous_zed_pose) @ zed_pose
        attempts = [
            ("zed", zed_initial),
            ("identity", np.eye(4)),
            ("zed_inverse", np.linalg.inv(zed_initial)),
        ]
        best_result = None
        best_mode = ""
        for mode, initial in attempts:
            result = register(
                cloud, previous_cloud, args.frame_voxel_m, initial
            )
            if better(result, best_result):
                best_result, best_mode = result, mode
            if (
                result.fitness >= args.min_fitness
                and result.inlier_rmse <= args.max_rmse_m
                and mode == "zed"
            ):
                break

        accepted_frame = (
            best_result.fitness >= args.min_fitness
            and best_result.inlier_rmse <= args.max_rmse_m
        )
        record = {
            "id": item["id"],
            "accepted": bool(accepted_frame),
            "fitness": float(best_result.fitness),
            "rmse_m": float(best_result.inlier_rmse),
            "initialization": best_mode,
        }
        metrics.append(record)
        status = "accept" if accepted_frame else "reject"
        print(
            f"[{candidate_index}/{len(candidates)}] {status} {item['id']}: "
            f"fitness={best_result.fitness:.3f}, "
            f"rmse={best_result.inlier_rmse:.4f}, init={best_mode}",
            flush=True,
        )
        if not accepted_frame:
            continue

        world_from_current = (
            world_from_previous @ best_result.transformation
        )
        accepted.append((item, cloud, world_from_current))
        previous_cloud = cloud
        previous_zed_pose = zed_pose
        world_from_previous = world_from_current

    if len(accepted) < 2:
        raise SystemExit("Fewer than two frames passed offline registration.")

    fused = o3d.geometry.PointCloud()
    for _item, cloud, pose in accepted:
        transformed = o3d.geometry.PointCloud(cloud)
        transformed.transform(pose)
        fused += transformed
    fused = fused.voxel_down_sample(args.fused_voxel_m)
    fused, _ = fused.remove_statistical_outlier(
        nb_neighbors=30, std_ratio=1.2
    )
    fused, _ = fused.remove_radius_outlier(
        nb_points=5, radius=args.fused_voxel_m * 3.0
    )

    ply_path = output / "fused_offline.ply"
    o3d.io.write_point_cloud(str(ply_path), fused, compressed=True)
    np.save(
        output / "accepted_camera_poses.npy",
        np.stack([pose for _item, _cloud, pose in accepted]),
    )
    (output / "registration_metrics.json").write_text(
        json.dumps(
            {
                "input_frames": len(items),
                "quality_candidates": len(candidates),
                "accepted_frames": len(accepted),
                "frames": metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_preview(fused, output / "fused_preview.png")
    print(
        f"Saved {len(fused.points)} points from {len(accepted)} frames: "
        f"{ply_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
