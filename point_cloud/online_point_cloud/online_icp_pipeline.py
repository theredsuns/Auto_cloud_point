#!/usr/bin/env python3
"""Build per-frame PLY clouds and refresh the online ICP fusion."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import open3d as o3d
import numpy as np

POINT_CLOUD_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(POINT_CLOUD_ROOT))
import object_pointcloud_menu as pcm  # noqa: E402

# A cylindrical front and rear opening can look geometrically similar to ICP.
# The calibrated ZED TCP pose is therefore not merely an initial guess: ICP
# may refine it only inside these small capture-pose error bounds.
# Robot/TCP calibration can carry a modest systematic error, especially when
# the arm is extended.  These bounds allow that correction while still
# rejecting the 0.6--0.7 m jump produced when two cylinder openings match.
MAX_POSE_CONSISTENT_ICP_TRANSLATION_M = 0.220
MAX_POSE_CONSISTENT_ICP_ROTATION_DEG = 15.0


def remove_reflection_speckles(cloud: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    """Discard isolated 3-D reflection speckles without fitting/smoothing surfaces.

    The masks decide *what* belongs to the object.  This is deliberately only
    a spatial-neighbour test on that masked cloud: it cannot invent depth or
    bend a real surface, but a point (or tiny group) floating away from the
    shell has too few nearby 3-D neighbours and is discarded.
    """
    if cloud.is_empty():
        return cloud
    cleaned = cloud.voxel_down_sample(0.004)
    before = len(cleaned.points)
    # First remove small detached reflection groups.  The object is one
    # connected physical shell, while bright depth failures form tiny clouds
    # around it.  Do not retain every DBSCAN component merely because it is
    # dense enough locally.
    if len(cleaned.points) >= 100:
        labels = np.asarray(cleaned.cluster_dbscan(eps=0.032, min_points=10, print_progress=False))
        valid = labels[labels >= 0]
        if len(valid):
            sizes = np.bincount(valid)
            keep = np.flatnonzero(sizes >= max(500, int(sizes.max() * 0.012)))
            selected = np.flatnonzero(np.isin(labels, keep))
            if len(selected):
                cleaned = cleaned.select_by_index(selected)
    if len(cleaned.points) >= 10:
        cleaned, _ = cleaned.remove_radius_outlier(nb_points=6, radius=0.025)
    if len(cleaned.points) >= 40:
        cleaned, _ = cleaned.remove_statistical_outlier(nb_neighbors=32, std_ratio=0.68)
    removed = before - len(cleaned.points)
    if removed:
        print(f"[filter] removed {removed:,} isolated/reflection points from fused cloud.")
    return cleaned


def transform_rotation_degrees(transform: np.ndarray) -> float:
    """Return the unsigned rotation angle of a homogeneous transform."""
    rotation = transform[:3, :3]
    cosine = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def fuse_from_capture_camera_poses(clouds, poses):
    """Pose-initialised global GICP in the first capture's camera frame.

    ``poses`` are calibrated world-from-camera matrices captured together with
    RGB/depth.  The first camera is the modelling origin; every later camera
    gets an explicit root-from-camera initial transform before ICP is run.
    Thus ZED/TCP position selects the intended side of a symmetric cylinder,
    while ICP supplies the millimetre-level final alignment.
    """
    usable = [(frame_id, cloud, np.asarray(poses[frame_id], dtype=np.float64))
              for frame_id, cloud in clouds if frame_id in poses]
    if not usable:
        raise RuntimeError("No clouds have a saved calibrated camera pose")
    # Build pose-guided pairwise ICP edges.  The relative camera pose is the
    # initial transform, and ICP gives the final source-to-target transform.
    # A large difference from the recorded camera movement is rejected as a
    # likely front/rear cylinder ambiguity.
    links = []
    for source_index in range(len(usable)):
        _source_id, source_cloud, source_pose = usable[source_index]
        for target_index in range(source_index):
            _target_id, target_cloud, target_pose = usable[target_index]
            initial = np.linalg.inv(target_pose) @ source_pose
            transform, result = pcm.refine_icp(source_cloud, target_cloud, initial)
            pose_correction = transform @ np.linalg.inv(initial)
            correction_translation = float(np.linalg.norm(pose_correction[:3, 3]))
            correction_rotation = transform_rotation_degrees(pose_correction)
            if (result.fitness >= 0.50 and result.inlier_rmse <= 0.030 and
                    correction_translation <= MAX_POSE_CONSISTENT_ICP_TRANSLATION_M and
                    correction_rotation <= MAX_POSE_CONSISTENT_ICP_ROTATION_DEG):
                links.append((float(result.fitness), source_index, target_index, transform,
                              float(result.inlier_rmse), correction_translation, correction_rotation,
                              initial))

    parent = list(range(len(usable)))
    def find(item):
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item
    def join(a, b):
        a, b = find(a), find(b)
        if a == b:
            return False
        parent[b] = a
        return True

    tree = []
    for link in sorted(links, key=lambda item: item[0], reverse=True):
        if join(link[1], link[2]):
            tree.append(link)
    groups = {}
    for index in range(len(usable)):
        groups.setdefault(find(index), []).append(index)
    # The first capture is always the modelling origin.  Do not silently move
    # the origin to a later, easier-to-match cylinder view.
    root = 0
    component = groups[find(root)]
    adjacency = {index: [] for index in component}
    for fitness, source, target, transform, rmse, correction_translation, correction_rotation, initial in tree:
        if source in adjacency and target in adjacency:
            # transform is target-from-source.  Store both directions so all
            # transforms are ultimately accumulated into the first/root view.
            adjacency[source].append((target, np.linalg.inv(transform), initial, fitness, rmse,
                                      correction_translation, correction_rotation))
            adjacency[target].append((source, transform, initial, fitness, rmse,
                                      correction_translation, correction_rotation))
    root_from = {root: np.eye(4)}
    edge_info = {}
    pending = [root]
    while pending:
        current = pending.pop()
        for neighbour, current_from_neighbour, initial, fitness, rmse, correction_translation, correction_rotation in adjacency[current]:
            if neighbour not in root_from:
                root_from[neighbour] = root_from[current] @ current_from_neighbour
                edge_info[neighbour] = (fitness, rmse, correction_translation, correction_rotation, initial)
                pending.append(neighbour)

    merged = o3d.geometry.PointCloud()
    report = []
    fusion_index = 0
    for index, (frame_id, cloud, pose) in enumerate(usable):
        if index not in root_from:
            report.append({"frame_id": frame_id, "status": "skipped_no_pose_guided_icp"})
            continue
        transformed = o3d.geometry.PointCloud(cloud)
        transformed.transform(root_from[index])
        merged += transformed.voxel_down_sample(0.003)
        fusion_index += 1
        entry = {
            "frame_id": frame_id,
            "status": "camera_pose_anchor" if index == root else "camera_pose_plus_icp",
            "fusion_index": fusion_index,
            "root_from_frame": root_from[index].tolist(),
            "camera_initial_from_first": (np.linalg.inv(usable[root][2]) @ pose).tolist(),
        }
        if index in edge_info:
            fitness, rmse, correction_translation, correction_rotation, _initial = edge_info[index]
            entry.update({"fitness": fitness, "rmse_m": rmse,
                          "pose_correction_m": correction_translation,
                          "pose_correction_deg": correction_rotation})
        report.append(entry)
    return remove_reflection_speckles(merged), report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("scan", type=Path)
    parser.add_argument("frame_id")
    args = parser.parse_args()
    scan = args.scan.resolve()
    intrinsics = pcm.load_intrinsics(scan)
    cloud_dir = scan / "clouds"; cloud_dir.mkdir(exist_ok=True)
    cloud_file = cloud_dir / f"frame_{args.frame_id}.ply"
    if not cloud_file.exists():
        image = scan / "images" / f"{args.frame_id}.jpg"
        depth = scan / "depth" / f"{args.frame_id}_depth.npy"
        mask = scan / "masks" / f"{args.frame_id}.png"
        _, cloud = pcm.make_cloud((args.frame_id, image, depth, mask, None), intrinsics)
        if not o3d.io.write_point_cloud(str(cloud_file), cloud, compressed=True):
            raise RuntimeError(f"Could not write {cloud_file}")
        print(f"[cloud] {args.frame_id}: {len(cloud.points):,} points")
    clouds = []
    for path in sorted(cloud_dir.glob("frame_*.ply")):
        cloud = o3d.io.read_point_cloud(str(path))
        if not cloud.is_empty():
            clouds.append((path.stem.removeprefix("frame_"), cloud))
    fusion_dir = scan / "fusion"; fusion_dir.mkdir(exist_ok=True)
    frame_ids = [frame_id for frame_id, _cloud in clouds]
    # Remote captures store the calibrated camera TCP in captures.jsonl.  It
    # is a much stronger initial pose than cylindrical geometry: front and
    # rear openings no longer look interchangeable to ICP.
    saved_poses, pose_path = pcm.find_saved_poses(scan, frame_ids)
    required_poses = 1 if len(frame_ids) <= 1 else max(2, int(np.ceil(len(frame_ids) * 0.8)))
    if len(saved_poses) >= required_poses:
        print(f"[camera-pose] Using {len(saved_poses)}/{len(frame_ids)} capture-time camera poses as GICP initialisation; only verified ICP frames enter the final model.")
        fused, report = fuse_from_capture_camera_poses(clouds, saved_poses)
    else:
        if saved_poses:
            print(f"[warning] Only {len(saved_poses)}/{len(frame_ids)} capture-time TCP poses; falling back to sequential ICP.")
        imu_yaws = pcm.find_saved_imu_yaws(scan, frame_ids)
        if imu_yaws:
            print(f"[imu] Using capture-time yaw constraints for {len(imu_yaws)}/{len(clouds)} frame(s).")
        fused, report = pcm.fuse_icp(
            clouds, total=len(clouds), debug_dir=fusion_dir / "icp_debug", imu_yaws=imu_yaws
        )
    # Do not run the aggressive cluster/outlier cleanup here: it can delete a
    # valid surface visible in only one online capture. ICP acceptance plus
    # voxel merging above is the online cleanup policy.
    fused_path = fusion_dir / "fused_icp.ply"
    o3d.io.write_point_cloud(str(fused_path), fused, compressed=True)
    (fusion_dir / "icp_registration_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    fused_frame_ids = [entry["frame_id"] for entry in report
                       if entry.get("status") in (
                           "camera_pose_anchor", "camera_pose_plus_icp", "camera_pose_only",
                           "anchor", "accepted",
                       )]
    current_entry = next((entry for entry in report if entry.get("frame_id") == args.frame_id), {})
    (fusion_dir / "fusion_status.json").write_text(json.dumps({
        "frame_count": len(fused_frame_ids), "input_frame_count": len(clouds),
        "fused_frame_ids": fused_frame_ids, "camera_pose_count": len(saved_poses),
        "mode": "camera_pose_plus_icp" if len(saved_poses) >= required_poses else "sequential_icp",
        "last_frame_id": args.frame_id,
        "last_frame_status": current_entry.get("status", "unknown"),
        "last_frame_detail": current_entry,
    }, indent=2), encoding="utf-8")
    print(f"[fusion] {len(clouds)} frame(s), {len(fused.points):,} points -> {fused_path}")


if __name__ == "__main__":
    main()
