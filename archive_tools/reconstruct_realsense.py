#!/usr/bin/env python3
"""Fuse masked RealSense RGB-D frames with sequential multi-scale ICP."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d


def load_intrinsics(path: Path) -> tuple[float, float, float, float]:
    matrix = np.loadtxt(path)
    return float(matrix[0, 0]), float(matrix[1, 1]), float(
        matrix[0, 2]
    ), float(matrix[1, 2])


def load_cloud(
    root: Path,
    index: int,
    intrinsics: tuple[float, float, float, float],
    depth_scale: float,
) -> o3d.geometry.PointCloud:
    name = f"{index:06d}.png"
    depth = cv2.imread(str(root / "depth" / name), cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(str(root / "masks" / name), cv2.IMREAD_GRAYSCALE)
    bgr = cv2.imread(str(root / "rgb" / name), cv2.IMREAD_COLOR)
    if depth is None or mask is None or bgr is None:
        raise RuntimeError(f"Could not load RGB-D frame {name}")

    valid = (mask > 0) & (depth > 300) & (depth < 10000)
    row, column = np.where(valid)
    z = depth[row, column].astype(np.float64) * depth_scale
    fx, fy, cx, cy = intrinsics
    points = np.column_stack(
        ((column - cx) * z / fx, (row - cy) * z / fy, z)
    )
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(
        rgb[row, column].astype(np.float64) / 255.0
    )
    return cloud


def prepared(
    cloud: o3d.geometry.PointCloud, voxel: float
) -> o3d.geometry.PointCloud:
    down = cloud.voxel_down_sample(voxel)
    down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 4.0, max_nn=50)
    )
    return down


def register_pair(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    initial: np.ndarray | None = None,
) -> o3d.pipelines.registration.RegistrationResult:
    transform = np.eye(4) if initial is None else initial.copy()
    result = None
    for voxel, distance, iterations in (
        (0.10, 0.40, 80),
        (0.05, 0.20, 60),
        (0.025, 0.08, 50),
    ):
        source_down = prepared(source, voxel)
        target_down = prepared(target, voxel)
        result = o3d.pipelines.registration.registration_icp(
            source_down,
            target_down,
            distance,
            transform,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=iterations
            ),
        )
        transform = result.transformation
    assert result is not None
    return result


def optimize_poses(
    clouds: list[o3d.geometry.PointCloud],
    initial_poses: list[np.ndarray],
    adjacent_transforms: list[np.ndarray],
) -> tuple[list[np.ndarray], list[dict[str, float | int]]]:
    down = [prepared(cloud, 0.04) for cloud in clouds]
    graph = o3d.pipelines.registration.PoseGraph()
    for pose in initial_poses:
        graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(pose))

    for index, transform in enumerate(adjacent_transforms, start=1):
        information = (
            o3d.pipelines.registration.get_information_matrix_from_point_clouds(
                down[index], down[index - 1], 0.10, transform
            )
        )
        graph.edges.append(
            o3d.pipelines.registration.PoseGraphEdge(
                index, index - 1, transform, information, uncertain=False
            )
        )

    closures = []
    for source_index in range(2, len(clouds)):
        for target_index in range(source_index - 2, -1, -1):
            predicted = (
                np.linalg.inv(initial_poses[target_index])
                @ initial_poses[source_index]
            )
            result = o3d.pipelines.registration.registration_icp(
                down[source_index],
                down[target_index],
                0.16,
                predicted,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(
                    max_iteration=60
                ),
            )
            if result.fitness < 0.30 or result.inlier_rmse > 0.065:
                continue
            information = (
                o3d.pipelines.registration.get_information_matrix_from_point_clouds(
                    down[source_index],
                    down[target_index],
                    0.10,
                    result.transformation,
                )
            )
            graph.edges.append(
                o3d.pipelines.registration.PoseGraphEdge(
                    source_index,
                    target_index,
                    result.transformation,
                    information,
                    uncertain=True,
                )
            )
            closures.append(
                {
                    "source": source_index,
                    "target": target_index,
                    "fitness": float(result.fitness),
                    "rmse_m": float(result.inlier_rmse),
                }
            )

    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=0.10,
        edge_prune_threshold=0.25,
        reference_node=0,
    )
    o3d.pipelines.registration.global_optimization(
        graph,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        option,
    )
    return [node.pose for node in graph.nodes], closures


def write_preview(cloud: o3d.geometry.PointCloud, output: Path) -> None:
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
    keep = radius < np.quantile(radius, 0.995)
    aligned = aligned[keep]
    rgb = rgb[keep]

    fig, plots = plt.subplots(1, 3, figsize=(15, 5), facecolor="#202124")
    for plot, (x_axis, y_axis, title) in zip(
        plots,
        ((0, 1, "front"), (0, 2, "top"), (1, 2, "side")),
    ):
        plot.set_facecolor("#202124")
        stride = max(1, len(aligned) // 150000)
        plot.scatter(
            aligned[::stride, x_axis],
            aligned[::stride, y_axis],
            s=0.25,
            c=rgb[::stride],
            linewidths=0,
        )
        plot.set_aspect("equal", "box")
        plot.set_title(title, color="white")
        plot.tick_params(colors="white")
    fig.suptitle(
        f"Masked RGB-D fusion: {len(xyz):,} points", color="white"
    )
    fig.tight_layout()
    fig.savefig(output, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--depth-scale", type=float, default=0.001)
    arguments = parser.parse_args()

    root = arguments.input.resolve()
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    intrinsics = load_intrinsics(root / "cam_K.txt")
    frame_count = len(list((root / "rgb").glob("*.png")))
    clouds = [
        load_cloud(root, index, intrinsics, arguments.depth_scale)
        for index in range(frame_count)
    ]

    global_poses = [np.eye(4)]
    adjacent_transforms = []
    metrics = []
    for index in range(1, frame_count):
        result = register_pair(clouds[index], clouds[index - 1])
        adjacent_transforms.append(result.transformation)
        global_poses.append(global_poses[-1] @ result.transformation)
        item = {
            "source": index,
            "target": index - 1,
            "fitness": float(result.fitness),
            "rmse_m": float(result.inlier_rmse),
        }
        metrics.append(item)
        print(
            f"{index:02d}->{index - 1:02d}: "
            f"fitness={result.fitness:.3f}, rmse={result.inlier_rmse:.4f} m",
            flush=True,
        )

    global_poses, closures = optimize_poses(
        clouds, global_poses, adjacent_transforms
    )
    print(f"Accepted loop closures: {len(closures)}", flush=True)

    fused = o3d.geometry.PointCloud()
    for cloud, pose in zip(clouds, global_poses):
        transformed = o3d.geometry.PointCloud(cloud)
        transformed.transform(pose)
        fused += transformed
    fused = fused.voxel_down_sample(0.015)
    fused, _ = fused.remove_statistical_outlier(
        nb_neighbors=25, std_ratio=2.0
    )

    o3d.io.write_point_cloud(str(output / "fused_masked.ply"), fused)
    np.save(output / "camera_poses.npy", np.stack(global_poses))
    (output / "registration_metrics.json").write_text(
        json.dumps(
            {"adjacent": metrics, "loop_closures": closures}, indent=2
        ),
        encoding="utf-8",
    )
    write_preview(fused, output / "fused_preview.png")
    print(f"Fused points: {len(fused.points):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
