#!/usr/bin/env python3
"""Interactive point-cloud generator for ordered image/depth/mask folders."""

import json
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import open3d as o3d

matplotlib.use("Agg")
import matplotlib.pyplot as plt


INPUT_ROOT = Path("object_pointcloud_input")
DEFAULT_INTRINSICS = {"fx": 672.917602539, "fy": 672.917602539,
                      "cx": 620.836303711, "cy": 351.999664307}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
MAX_ICP_STEP_TRANSLATION_M = 0.60
MAX_ICP_STEP_ROTATION_DEG = 35.0
# The ZED tracker can drift by several decimetres while the operator walks
# around a large object. Permit a substantial correction only when ICP itself
# has strong support; never start a disconnected second registration branch.
MAX_POSE_REFINEMENT_TRANSLATION_M = 0.60
MAX_POSE_REFINEMENT_ROTATION_DEG = 30.0
# With an AprilTag anchor, favour one clean continuous surface over attempting
# to recover a symmetric far side after a bad local registration.
TAG_MAX_ICP_CORRECTION_M = 0.15
TAG_MAX_ICP_CORRECTION_DEG = 10.0
MIN_ADJACENT_ICP_FITNESS = 0.45
# A point is allowed into the final sequential model only after a *mutual*
# nearest-neighbour match with the immediately preceding accepted view.  This
# is the multi-frame test that rejects one-view reflection speckles.
ADJACENT_CONFIRM_DISTANCE_M = 0.028
MIN_ADJACENT_CONFIRMED_FEATURES = 80
MAX_IMU_YAW_SEED_OFFSET_DEG = 6.0
# ZED confidence uses 0 for the most reliable depth and 100 for the least.
# This is applied only when the optional confidence/ capture folder exists.
MAX_ZED_CONFIDENCE = 50.0
# A full-object mask can contain isolated reflected/background regions.  Keep
# all large regions (so two real bodies remain possible), but remove small
# disconnected depth islands before they become floating 3-D clusters.
MIN_MASK_DEPTH_COMPONENT_RATIO = 0.05
# Reflection errors usually form tiny sparse 3-D islands.  These values are
# deliberately measured in metres, rather than pixels, so the cleanup remains
# appropriate when the camera is moved farther from the object.
SCATTER_RADIUS_M = 0.035
SCATTER_MIN_NEIGHBORS = 5
SCATTER_STATISTICAL_NEIGHBORS = 20
SCATTER_STATISTICAL_STD_RATIO = 0.85
FUSED_CLUSTER_EPS_M = 0.040
FUSED_CLUSTER_MIN_POINTS = 20
FUSED_CLUSTER_MIN_RATIO = 0.005
FUSED_CLEAN_RADIUS_M = 0.045
FUSED_CLEAN_MIN_NEIGHBORS = 12
FUSED_CLEAN_STD_RATIO = 0.40
# Adjacent-frame ICP residuals are typically around 1 cm. Fuse at a slightly
# larger scale so repeated observations become one centroid instead of layers.
FUSED_CONSENSUS_VOXEL_M = 0.015
FUSED_SUPPORT_VOXEL_M = 0.030
FUSED_MIN_VIEWS = 2
# GICP uses local surface covariances instead of treating every measured point
# as equally reliable.  These scales are deliberately tied to each ICP level:
# coarse registration keeps a broad Huber basin, while fine registration uses
# Tukey to reject reflective depth jumps completely.
GICP_HUBER_RATIO = 0.60
GICP_TUKEY_RATIO = 0.80
# The printed AprilTags on the front of the object provide the only
# non-symmetric geometric reference in this capture.  Their real edge length
# is deliberately kept here (rather than guessed from pixels).
APRILTAG_SIZE_M = 0.064
APRILTAG_MAX_REPROJECTION_ERROR_PX = 3.0
PROJECT_ROOT = Path(__file__).resolve().parent


def base_id(path: Path, is_depth=False) -> str:
    """Return a common frame ID; accepts `frame.npy` and `frame_depth.npy`."""
    name = path.stem
    return name.removesuffix("_depth") if is_depth else name


def load_intrinsics(root: Path) -> dict:
    path = root / "camera_intrinsics.json"
    if not path.exists():
        print("[info] No camera_intrinsics.json; using the built-in ZED HD720 calibration.")
        return DEFAULT_INTRINSICS.copy()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return {key: float(data[key]) for key in DEFAULT_INTRINSICS}
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(f"Invalid {path}: {error}")


def find_frames(root: Path):
    # Accept both the requested names and the common names already used here.
    image_dir = next(
        (
            root / name for name in ("image", "images", "rgb")
            if (root / name).is_dir()
        ),
        root / "rgb",
    )
    depth_dir = root / "depth"
    # Only the complete mask defines the point-cloud silhouette. A scene can
    # contain more than one fuselage, so body/wing masks must not be used to
    # associate or repair parts.
    full_mask_dir = root / "mask" if (root / "mask").is_dir() else root / "masks"
    absent = [str(d) for d in (image_dir, depth_dir, full_mask_dir) if not d.is_dir()]
    if absent:
        raise SystemExit("Missing required folder(s):\n  " + "\n  ".join(absent))
    images = {base_id(p): p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS}
    depths = {base_id(p, True): p for p in depth_dir.glob("*.npy")}
    confidence_dir = root / "confidence"
    confidence = (
        {
            p.stem.removesuffix("_confidence"): p
            for p in confidence_dir.glob("*.npy")
        }
        if confidence_dir.is_dir() else {}
    )
    full_masks = {
        base_id(p): p for p in full_mask_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    }
    common = sorted(images.keys() & depths.keys() & set(full_masks))
    if not common:
        raise SystemExit(
            "No matching frames. Files must share a name, e.g. "
            "image/0001.jpg, depth/0001.npy, mask/0001.png."
        )
    print(f"[info] Found {len(common)} complete frame(s).")
    if confidence:
        print(
            f"[info] Found ZED confidence for "
            f"{len(set(common) & set(confidence))}/{len(common)} frame(s)."
        )
    for group, entries in (("image", images), ("depth", depths), (full_mask_dir.name, full_masks)):
        unmatched = sorted(set(entries) - set(common))
        if unmatched:
            print(f"[warning] {group}: ignored {len(unmatched)} unmatched file(s).")
    return [
        (
            frame_id,
            images[frame_id],
            depths[frame_id],
            full_masks[frame_id],
            confidence.get(frame_id),
        )
        for frame_id in common
    ]


def make_cloud(frame, intrinsics, min_depth=0.20, max_depth=8.0):
    frame_id, image_path, depth_path, full_mask_path, confidence_path = frame
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    depth = np.load(depth_path)
    if bgr is None or depth.ndim != 2:
        raise ValueError("could not decode image or 2-D depth array")
    full_mask = cv2.imread(str(full_mask_path), cv2.IMREAD_GRAYSCALE)
    if full_mask is None or full_mask.shape != depth.shape:
        raise ValueError(f"bad mask size in {full_mask_path.name}")

    object_mask = full_mask > 0
    # Each valid total-mask pixel is projected using its *measured* metric
    # depth. Do not clamp against a frame-wide depth statistic or interpolate
    # missing pixels: a long, obliquely viewed object naturally has a large
    # real depth span, and fabricated depths distort its single-frame shape.
    valid = (
        object_mask
        & np.isfinite(depth)
        & (depth > min_depth)
        & (depth < max_depth)
    )
    if confidence_path is not None:
        confidence = np.load(confidence_path)
        if confidence.shape != depth.shape:
            raise ValueError(f"bad confidence size in {confidence_path.name}")
        valid &= np.isfinite(confidence) & (confidence <= MAX_ZED_CONFIDENCE)
    component_count, component_labels, component_stats, _ = cv2.connectedComponentsWithStats(
        valid.astype(np.uint8), connectivity=8
    )
    if component_count > 1:
        areas = component_stats[1:, cv2.CC_STAT_AREA]
        largest = int(areas.max())
        minimum = max(100, int(largest * MIN_MASK_DEPTH_COMPONENT_RATIO))
        keep_labels = np.flatnonzero(areas >= minimum) + 1
        valid &= np.isin(component_labels, keep_labels)
        removed = int(np.count_nonzero(component_labels) - np.count_nonzero(valid))
        if removed:
            print(f"[info] {frame_id}: removed {removed:,} pixels from small disconnected depth regions.")
    rows, cols = np.where(valid)
    if len(rows) < 100:
        raise ValueError(f"only {len(rows)} valid masked depth pixels")
    z = depth[rows, cols].astype(np.float64)
    points = np.column_stack(((cols-intrinsics["cx"])*z/intrinsics["fx"],
                              (rows-intrinsics["cy"])*z/intrinsics["fy"], z))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)[rows, cols] / 255.0
    cloud = o3d.geometry.PointCloud()
    cloud.points, cloud.colors = o3d.utility.Vector3dVector(points), o3d.utility.Vector3dVector(rgb)
    cloud = cloud.voxel_down_sample(0.006)
    # Remove only spatially isolated points. Unlike component filtering, this
    # keeps a dense but separate wing or a second physical body intact.
    if len(cloud.points) >= SCATTER_MIN_NEIGHBORS:
        before = len(cloud.points)
        cloud, _ = cloud.remove_radius_outlier(
            nb_points=SCATTER_MIN_NEIGHBORS,
            radius=SCATTER_RADIUS_M,
        )
        # Radius filtering removes isolated speckles.  The statistical pass
        # then removes the remaining sparse reflection clusters without
        # smoothing, filling, or changing any measured object depth.
        if len(cloud.points) >= SCATTER_STATISTICAL_NEIGHBORS:
            cloud, _ = cloud.remove_statistical_outlier(
                nb_neighbors=SCATTER_STATISTICAL_NEIGHBORS,
                std_ratio=SCATTER_STATISTICAL_STD_RATIO,
            )
        removed_3d = before - len(cloud.points)
        if removed_3d:
            print(f"[info] {frame_id}: removed {removed_3d:,} isolated/reflection 3-D points.")
    print(
        f"[info] {frame_id}: mask pixels={np.count_nonzero(object_mask):,}; "
        f"raw-depth points={len(rows):,}."
    )
    return frame_id, cloud


def save_preview(cloud, output: Path, title: str):
    points, colors = np.asarray(cloud.points), np.asarray(cloud.colors)
    if not len(points): return
    step = max(1, len(points)//100_000)
    fig = plt.figure(figsize=(8, 7), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(points[::step, 0], points[::step, 1], points[::step, 2], c=colors[::step], s=.4, depthshade=False)
    ax.set(xlabel="X (m)", ylabel="Y (m)", zlabel="Z (m)", title=title)
    ax.view_init(elev=18, azim=-65); fig.tight_layout(); fig.savefig(output, dpi=180); plt.close(fig)


def save_cloud(cloud, output: Path, name: str, clean=True):
    cloud = cloud.voxel_down_sample(0.004)
    if clean:
        cloud, _ = cloud.remove_statistical_outlier(nb_neighbors=24, std_ratio=1.3)
    o3d.io.write_point_cloud(str(output / f"{name}.ply"), cloud, compressed=True)
    save_preview(cloud, output / f"{name}_preview.png", f"{name}: {len(cloud.points):,} points")
    return cloud


def build_all(frames, intrinsics):
    clouds = []
    for index, frame in enumerate(frames, 1):
        try:
            frame_id, cloud = make_cloud(frame, intrinsics)
            clouds.append((frame_id, cloud))
            print(f"[{index}/{len(frames)}] {frame_id}: {len(cloud.points):,} points")
        except (ValueError, OSError) as error:
            print(f"[{index}/{len(frames)}] skip {frame[0]}: {error}")
    if not clouds: raise SystemExit("No valid point cloud could be generated.")
    return clouds


def repeated_view_cloud(clouds, voxel=0.025, minimum_views=2):
    """Keep cells observed by multiple frames; preserve separate wing pieces."""
    sampled = [cloud.voxel_down_sample(0.012) for cloud in clouds]
    cells_per_frame = []
    for cloud in sampled:
        points = np.asarray(cloud.points)
        cells_per_frame.append(np.unique(np.floor(points / voxel).astype(np.int32), axis=0))
    if not cells_per_frame:
        return o3d.geometry.PointCloud()
    cells, counts = np.unique(np.vstack(cells_per_frame), axis=0, return_counts=True)
    supported = cells[counts >= minimum_views]

    def packed(array):
        contiguous = np.ascontiguousarray(array)
        return contiguous.view(np.dtype((np.void, contiguous.dtype.itemsize * 3))).ravel()

    supported_packed = packed(supported)
    kept_points, kept_colors = [], []
    for cloud in sampled:
        points, colors = np.asarray(cloud.points), np.asarray(cloud.colors)
        keys = np.floor(points / voxel).astype(np.int32)
        keep = np.isin(packed(keys), supported_packed)
        kept_points.append(points[keep]); kept_colors.append(colors[keep])
    output = o3d.geometry.PointCloud()
    output.points = o3d.utility.Vector3dVector(np.vstack(kept_points))
    output.colors = o3d.utility.Vector3dVector(np.vstack(kept_colors))
    return output.voxel_down_sample(0.006)


def clean_fused_cloud(cloud):
    """Remove detached noise clusters while retaining meaningful sub-parts."""
    # This is a true fusion step: Open3D stores one centroid/color per voxel,
    # averaging repeated observations that remain a few millimetres apart
    # after ICP. A 6 mm output voxel preserved those layers visibly.
    cloud = cloud.voxel_down_sample(FUSED_CONSENSUS_VOXEL_M)
    if len(cloud.points) < FUSED_CLUSTER_MIN_POINTS:
        return cloud
    labels = np.asarray(cloud.cluster_dbscan(
        eps=FUSED_CLUSTER_EPS_M,
        min_points=FUSED_CLUSTER_MIN_POINTS,
        print_progress=False,
    ))
    valid_labels = labels[labels >= 0]
    if len(valid_labels):
        sizes = np.bincount(valid_labels)
        minimum_size = max(
            400, int(sizes.max() * FUSED_CLUSTER_MIN_RATIO)
        )
        keep_labels = np.flatnonzero(sizes >= minimum_size)
        keep_indices = np.flatnonzero(np.isin(labels, keep_labels))
        if len(keep_indices):
            cloud = cloud.select_by_index(keep_indices)
    cloud, _ = cloud.remove_radius_outlier(
        nb_points=FUSED_CLEAN_MIN_NEIGHBORS,
        radius=FUSED_CLEAN_RADIUS_M,
    )
    if len(cloud.points) >= 40:
        cloud, _ = cloud.remove_statistical_outlier(
            nb_neighbors=50, std_ratio=FUSED_CLEAN_STD_RATIO
        )
    return cloud


def consensus_fuse(clouds, voxel=FUSED_SUPPORT_VOXEL_M,
                   minimum_views=FUSED_MIN_VIEWS):
    """Keep only spatial cells independently observed by several frames."""
    if len(clouds) < minimum_views:
        return o3d.geometry.PointCloud()

    def packed(cells):
        contiguous = np.ascontiguousarray(cells)
        return contiguous.view(np.dtype((np.void, contiguous.dtype.itemsize * 3))).ravel()

    per_cloud = []
    all_cells = []
    for cloud in clouds:
        points = np.asarray(cloud.points)
        colors = np.asarray(cloud.colors)
        if not len(points):
            continue
        cells = np.floor(points / voxel).astype(np.int32)
        unique = np.unique(cells, axis=0)
        per_cloud.append((points, colors, cells))
        all_cells.append(unique)
    if len(all_cells) < minimum_views:
        return o3d.geometry.PointCloud()
    cells, counts = np.unique(np.vstack(all_cells), axis=0, return_counts=True)
    supported = cells[counts >= minimum_views]
    supported_keys = packed(supported)
    kept_points, kept_colors = [], []
    for points, colors, cells in per_cloud:
        keep = np.isin(packed(cells), supported_keys)
        if np.any(keep):
            kept_points.append(points[keep])
            kept_colors.append(colors[keep])
    if not kept_points:
        return o3d.geometry.PointCloud()
    fused = o3d.geometry.PointCloud()
    fused.points = o3d.utility.Vector3dVector(np.vstack(kept_points))
    fused.colors = o3d.utility.Vector3dVector(np.vstack(kept_colors))
    return fused.voxel_down_sample(FUSED_CONSENSUS_VOXEL_M)


def find_saved_poses(root: Path, frame_ids):
    """Find the capture log matching this input set and load metric poses."""
    candidates = [root / "captures.jsonl"]
    candidates.extend((PROJECT_ROOT / "datasets" / "local_offline_capture").glob("*/captures.jsonl"))
    wanted, best_poses, best_path = set(frame_ids), {}, None
    for path in candidates:
        if not path.is_file():
            continue
        poses = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            pose = item.get("world_from_camera")
            if item.get("id") in wanted and pose is not None:
                poses[item["id"]] = np.asarray(pose, dtype=np.float64)
        if len(poses) > len(best_poses):
            best_poses, best_path = poses, path
    return best_poses, best_path


def find_saved_imu_yaws(root: Path, frame_ids):
    """Load capture-time vehicle yaw values recorded by local_zed_capture.

    IMU acceleration is intentionally not integrated for translation.  Only
    the relative yaw of a rigidly mounted camera is stable enough to constrain
    a cylindrical object's otherwise ambiguous ICP rotation.
    """
    wanted, yaws = set(frame_ids), {}
    path = root / "captures.jsonl"
    if not path.is_file():
        return yaws
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        imu = item.get("tracer5_imu") or {}
        rpy = imu.get("rpy_deg")
        if item.get("id") in wanted and isinstance(rpy, list) and len(rpy) == 3:
            try:
                # Ignore a stale sample rather than constraining ICP with it.
                if float(imu.get("age_s", float("inf"))) <= 1.0:
                    yaws[item["id"]] = float(rpy[2])
            except (TypeError, ValueError):
                continue
    return yaws


def detect_front_apriltag_poses(frames, intrinsics):
    """Return tag_from_camera poses for the most repeatedly seen AprilTag.

    A cylindrical body has no unique yaw for ICP.  The tag is therefore used
    only as an absolute front reference; all poses below are expressed in that
    tag's coordinate system.  Selecting one repeatedly observed ID avoids
    assuming that separately printed tags have known relative positions.
    """
    if not hasattr(cv2, "aruco"):
        print("[warning] OpenCV was built without aruco; AprilTag anchoring is unavailable.")
        return {}, None, []
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    params = cv2.aruco.DetectorParameters_create()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.cornerRefinementMaxIterations = 50
    detections = {}
    half = APRILTAG_SIZE_M / 2.0
    object_corners = np.array(
        [[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]],
        dtype=np.float64,
    )
    camera_matrix = np.array(
        [[intrinsics["fx"], 0, intrinsics["cx"]],
         [0, intrinsics["fy"], intrinsics["cy"]], [0, 0, 1]], dtype=np.float64,
    )
    for frame_id, image_path, *_ in frames:
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        corners, ids, _ = cv2.aruco.detectMarkers(image, dictionary, parameters=params)
        if ids is None:
            continue
        for corner, tag_id in zip(corners, ids.ravel()):
            ok, rvec, tvec = cv2.solvePnP(
                object_corners, corner.reshape(4, 2), camera_matrix, None,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if not ok:
                continue
            projected, _ = cv2.projectPoints(
                object_corners, rvec, tvec, camera_matrix, None
            )
            error = float(np.mean(np.linalg.norm(
                projected.reshape(4, 2) - corner.reshape(4, 2), axis=1
            )))
            if error > APRILTAG_MAX_REPROJECTION_ERROR_PX:
                continue
            rotation, _ = cv2.Rodrigues(rvec)
            camera_from_tag = np.eye(4)
            camera_from_tag[:3, :3] = rotation
            camera_from_tag[:3, 3] = tvec.ravel()
            detections.setdefault(int(tag_id), []).append(
                (frame_id, np.linalg.inv(camera_from_tag), error)
            )
    if not detections:
        print("[warning] No valid AprilTag 36h11 detections in the RGB frames.")
        return {}, None, []
    tag_id, observations = max(detections.items(), key=lambda item: len(item[1]))
    poses = {frame_id: pose for frame_id, pose, _ in observations}
    print(
        f"[info] AprilTag 36h11 ID {tag_id}: {len(poses)} reliable front anchor frame(s), "
        f"size={APRILTAG_SIZE_M * 100:.1f} cm."
    )
    return poses, tag_id, [
        {"frame_id": frame_id, "tag_id": tag_id, "reprojection_error_px": error}
        for frame_id, _, error in observations
    ]


def refine_global_poses(clouds, initial_poses):
    """Globally distribute ICP drift using reliable adjacent/loop edges."""
    if len(clouds) < 3:
        return initial_poses, 0
    down = [cloud.voxel_down_sample(0.025) for cloud in clouds]
    for cloud in down:
        cloud.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=0.075, max_nn=40)
        )
    graph = o3d.pipelines.registration.PoseGraph()
    for pose in initial_poses:
        graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(pose))

    edges = 0

    def try_add(source_index, target_index, uncertain, min_fitness,
                max_rmse, max_translation, max_angle):
        nonlocal edges
        initial = (
            np.linalg.inv(initial_poses[target_index])
            @ initial_poses[source_index]
        )
        result = o3d.pipelines.registration.registration_icp(
            down[source_index], down[target_index], 0.08, initial,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60),
        )
        correction = np.linalg.inv(initial) @ result.transformation
        translation = float(np.linalg.norm(correction[:3, 3]))
        angle = float(np.degrees(np.arccos(np.clip(
            (np.trace(correction[:3, :3]) - 1) / 2, -1, 1
        ))))
        if (
            result.fitness < min_fitness or result.inlier_rmse > max_rmse
            or translation > max_translation or angle > max_angle
        ):
            return
        information = (
            o3d.pipelines.registration.get_information_matrix_from_point_clouds(
                down[source_index], down[target_index], 0.08,
                result.transformation,
            )
        )
        graph.edges.append(o3d.pipelines.registration.PoseGraphEdge(
            source_index, target_index, result.transformation, information,
            uncertain=uncertain,
        ))
        edges += 1

    # Preserve reliable local relations as fixed edges.
    for source in range(1, len(clouds)):
        try_add(source, source - 1, False, 0.40, 0.025, 0.60, 30.0)

    # Only add conservative loop closures. These stop a full walk around the
    # body from accumulating enough error to form a second shell.
    for source in range(3, len(clouds)):
        for target in range(0, source - 2):
            try_add(source, target, True, 0.55, 0.020, 0.15, 10.0)

    if edges < 2:
        return initial_poses, 0
    o3d.pipelines.registration.global_optimization(
        graph,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        o3d.pipelines.registration.GlobalOptimizationOption(
            max_correspondence_distance=0.08,
            edge_prune_threshold=0.35,
            reference_node=0,
        ),
    )
    return [node.pose for node in graph.nodes], edges


def fuse_with_saved_poses(clouds, poses, total=None, tag_poses=None, tag_id=None):
    """Use ZED poses as adjacent ICP initialisation, not final registration."""
    accepted_clouds, accepted_raw, accepted_poses, report = [], [], [], []
    total = total or "?"
    tag_poses = tag_poses or {}
    use_tag_anchor = bool(tag_poses)
    tag_started = not use_tag_anchor
    tag_chain_broken = False
    previous_id = None
    previous_cloud = None
    world_from_previous = None
    for index, (frame_id, cloud) in enumerate(clouds, 1):
        zed_pose = poses.get(frame_id)
        tag_pose = tag_poses.get(frame_id)
        if tag_pose is not None:
            # A repeated view of the same physical tag is an absolute pose,
            # not a soft ICP suggestion.  This prevents the circular shell
            # from snapping 180 degrees onto itself.
            world_from_current = tag_pose.copy()
            transformed = o3d.geometry.PointCloud(cloud)
            transformed.transform(world_from_current)
            accepted_clouds.append(transformed)
            accepted_raw.append(cloud)
            accepted_poses.append(world_from_current)
            previous_id = frame_id
            previous_cloud = cloud
            world_from_previous = world_from_current
            tag_started = True
            tag_chain_broken = False
            report.append({"frame_id": frame_id, "status": "anchor_apriltag", "tag_id": tag_id})
            print(f"[{index}/{total}] AprilTag anchor {frame_id} (ID {tag_id})")
            continue
        if not tag_started:
            report.append({"frame_id": frame_id, "status": "skipped_before_first_apriltag"})
            print(f"[{index}/{total}] {frame_id}: waiting for first AprilTag anchor; skipped.")
            continue
        if use_tag_anchor and tag_chain_broken:
            report.append({"frame_id": frame_id, "status": "skipped_after_tag_icp_break"})
            print(f"[{index}/{total}] {frame_id}: skipped after unreliable ICP; keeping clean anchored surface.")
            continue
        if zed_pose is None:
            report.append({"frame_id": frame_id, "status": "rejected_missing_pose"})
            continue
        if previous_cloud is None:
            world_from_current = zed_pose.copy()
            transformed = o3d.geometry.PointCloud(cloud)
            transformed.transform(world_from_current)
            accepted_clouds.append(transformed)
            accepted_raw.append(cloud)
            accepted_poses.append(world_from_current)
            previous_id = frame_id
            previous_cloud = cloud
            world_from_previous = world_from_current
            report.append({"frame_id": frame_id, "status": "anchor_saved_pose"})
            print(f"[{index}/{total}] ICP anchor {frame_id}")
            continue

        # Predicted transform maps the current camera cloud into the previous
        # camera cloud. ICP refines this local relation before it is composed
        # into the world transform, so global tracking drift does not turn into
        # the layered overlap visible in the old fused result.
        predicted = np.linalg.inv(poses[previous_id]) @ zed_pose
        candidate, result = refine_icp(cloud, previous_cloud, predicted)
        correction = np.linalg.inv(predicted) @ candidate
        translation = float(np.linalg.norm(correction[:3, 3]))
        angle = float(np.degrees(np.arccos(np.clip(
            (np.trace(correction[:3, :3]) - 1) / 2, -1, 1
        ))))
        record = {
            "frame_id": frame_id,
            "previous_frame_id": previous_id,
            "icp_fitness": float(result.fitness),
            "icp_rmse_m": float(result.inlier_rmse),
            "pose_correction_m": translation,
            "pose_correction_deg": angle,
        }
        if (
            result.fitness >= MIN_ADJACENT_ICP_FITNESS
            and result.inlier_rmse <= 0.025
            and translation <= (TAG_MAX_ICP_CORRECTION_M if use_tag_anchor else MAX_POSE_REFINEMENT_TRANSLATION_M)
            and angle <= (TAG_MAX_ICP_CORRECTION_DEG if use_tag_anchor else MAX_POSE_REFINEMENT_ROTATION_DEG)
        ):
            relative = candidate
            record["status"] = "accepted_adjacent_icp"
        else:
            # Do not put a frame back using its unrefined tracking pose. It
            # is precisely those frames that created the visible layered
            # overlap in the fused cloud. Individual frame PLY files are
            # still written by cloud_stream for later inspection.
            record["status"] = "rejected_icp_quality"
            if use_tag_anchor:
                tag_chain_broken = True
            report.append(record)
            print(
                f"[{index}/{total}] ICP {frame_id}: rejected; "
                f"fitness={result.fitness:.3f}, rmse={result.inlier_rmse:.4f} m, "
                f"correction={translation:.3f} m/{angle:.1f} deg"
            )
            # Advance the *registration reference* to the immediately
            # previous capture, but do not add this unreliable cloud to the
            # fused result. The predicted ZED step keeps one continuous global
            # chain; the next frame can then ICP-match its true neighbour
            # instead of trying to match a distant view of the first anchor.
            world_from_previous = world_from_previous @ predicted
            previous_id = frame_id
            previous_cloud = cloud
            continue

        world_from_current = world_from_previous @ relative
        transformed = o3d.geometry.PointCloud(cloud)
        transformed.transform(world_from_current)
        accepted_clouds.append(transformed)
        accepted_raw.append(cloud)
        accepted_poses.append(world_from_current)
        previous_id = frame_id
        previous_cloud = cloud
        world_from_previous = world_from_current
        report.append(record)
        print(
            f"[{index}/{total}] ICP {frame_id}: {record['status']}; "
            f"fitness={result.fitness:.3f}, rmse={result.inlier_rmse:.4f} m, "
            f"correction={translation:.3f} m/{angle:.1f} deg"
        )
    if not accepted_clouds:
        raise SystemExit("No frames with saved camera poses could be fused.")
    # Pose-graph ICP has no knowledge of the AprilTag and can pull an anchored
    # circular body back into its symmetric, wrong orientation.  With tags we
    # retain the hard anchors and only use the reliable ordered local links.
    refined_poses, graph_edges = (
        (accepted_poses, 0) if use_tag_anchor
        else refine_global_poses(accepted_raw, accepted_poses)
    )
    if graph_edges:
        print(f"[info] Global pose graph: {graph_edges} reliable ICP edges.")
        accepted_clouds = []
        for cloud, pose in zip(accepted_raw, refined_poses):
            transformed = o3d.geometry.PointCloud(cloud)
            transformed.transform(pose)
            accepted_clouds.append(transformed)
    # Do not require two observations per voxel. Exterior surfaces and wing
    # edges are often visible from only one view; requiring a second view was
    # the main reason the previous fused cloud looked too sparse.
    final = consensus_fuse(accepted_clouds)
    if final.is_empty():
        print("[warning] Too little multi-view overlap; using unfiltered fusion.")
        final = o3d.geometry.PointCloud()
        for cloud in accepted_clouds:
            final += cloud
    return clean_fused_cloud(final), report


def refine_icp(source_full, target_full, initial):
    """Robust multi-scale GICP refinement from a supplied pose.

    GICP models each local patch using its covariance, which is much less
    sensitive to curved shells than ordinary point-to-plane ICP.  The robust
    loss prevents the isolated depth values caused by specular reflections
    from steering the camera pose.
    """
    transform, result = initial.copy(), None
    for voxel, threshold in ((0.050, 0.16), (0.020, 0.070), (0.008, 0.028)):
        source = source_full.voxel_down_sample(voxel)
        target = target_full.voxel_down_sample(voxel)
        params = o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 3, max_nn=30)
        source.estimate_normals(params); target.estimate_normals(params)
        # Do not use Tukey at the coarse stage: a valid next frame can still
        # have centimetres of initial pose error.  At the two finer levels it
        # suppresses points whose residual is characteristic of a reflection.
        if voxel >= 0.050:
            kernel = o3d.pipelines.registration.HuberLoss(threshold * GICP_HUBER_RATIO)
        else:
            kernel = o3d.pipelines.registration.TukeyLoss(threshold * GICP_TUKEY_RATIO)
        estimator = o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(
            1e-6, kernel
        )
        result = o3d.pipelines.registration.registration_generalized_icp(
            source, target, threshold, transform,
            estimator,
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50),
        )
        transform = result.transformation
    return transform, result


def centroid_yaw_initial(source, target, yaw_deg):
    """Place source's centroid at target's, trying an object turn about Y."""
    angle = np.radians(yaw_deg)
    rotation = np.array([
        [np.cos(angle), 0.0, np.sin(angle)],
        [0.0, 1.0, 0.0],
        [-np.sin(angle), 0.0, np.cos(angle)],
    ])
    source_center = np.asarray(source.points).mean(axis=0)
    target_center = np.asarray(target.points).mean(axis=0)
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = target_center - rotation @ source_center
    return transform


def refine_icp_with_yaw_search(source_full, target_full, previous_transform,
                               imu_yaw_delta_deg: float | None = None):
    """Try nearby viewing-angle changes before the normal fine ICP.

    A scanner circling an object sees each frame in a different camera frame.
    Identity initialisation gives zero correspondences even for a valid next
    view.  The low-resolution search is inexpensive and is only used when
    saved camera poses are unavailable.
    """
    source_coarse = source_full.voxel_down_sample(0.050)
    target_coarse = target_full.voxel_down_sample(0.050)
    if len(source_coarse.points) < 20 or len(target_coarse.points) < 20:
        return refine_icp(source_full, target_full, previous_transform)
    candidates = [previous_transform]
    # With a vehicle IMU, seed only near the measured relative yaw (both
    # signs are tried because camera and vehicle axes can be mounted opposite
    # to one another). This rules out the repeated circular opening on the
    # far side before GICP sees it. Without an IMU retain the conservative
    # adjacent-view search used for offline image sets.
    if imu_yaw_delta_deg is not None:
        delta = abs(float(imu_yaw_delta_deg))
        yaw_values = []
        for sign in (-1.0, 1.0):
            center = sign * delta
            yaw_values.extend((center - MAX_IMU_YAW_SEED_OFFSET_DEG,
                               center,
                               center + MAX_IMU_YAW_SEED_OFFSET_DEG))
    else:
        yaw_values = (-45, -30, -15, 0, 15, 30, 45)
    candidates.extend(centroid_yaw_initial(source_coarse, target_coarse, yaw)
                      for yaw in sorted(set(yaw_values)))
    best_initial, best_score = candidates[0], (-1.0, np.inf)
    for initial in candidates:
        coarse = o3d.pipelines.registration.registration_icp(
            source_coarse, target_coarse, 0.18, initial,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=35),
        )
        score = (float(coarse.fitness), -float(coarse.inlier_rmse))
        if score > best_score:
            best_initial, best_score = coarse.transformation, score
    return refine_icp(source_full, target_full, best_initial)


def save_icp_correspondence_debug(source_full, target_full, transform, output: Path, frame_id: str):
    """Save the actual final-ICP inlier pairs in one coloured PLY.

    Red = sampled points from the newly captured cloud after its proposed
    transform; green = nearest points in the existing accumulated cloud;
    blue = midpoint of each accepted pair; yellow = sparse ISS geometric
    keypoints that also have a valid counterpart.  The red/green/blue points
    are the actual ICP support/correspondence points, not invented image
    features.
    """
    output.mkdir(parents=True, exist_ok=True)
    source = o3d.geometry.PointCloud(source_full).voxel_down_sample(0.020)
    target = o3d.geometry.PointCloud(target_full).voxel_down_sample(0.020)
    source.transform(transform)
    source_points, target_points = np.asarray(source.points), np.asarray(target.points)
    if not len(source_points) or not len(target_points):
        return None, 0, 0
    tree = o3d.geometry.KDTreeFlann(target)
    source_indices, target_indices = [], []
    # This threshold is intentionally the same scale as the final ICP stage.
    for index, point in enumerate(source_points):
        found, indices, squared = tree.search_knn_vector_3d(point, 1)
        if found and squared[0] <= 0.028 ** 2:
            source_indices.append(index)
            target_indices.append(indices[0])
    if not source_indices:
        return None, 0, 0
    matched_source = source_points[source_indices]
    matched_target = target_points[target_indices]
    midpoints = (matched_source + matched_target) * .5
    # ISS provides a sparse, human-readable subset: corners, rim breaks,
    # holes and asymmetric details.  Showing only these yellow points makes
    # it clear during capture whether the next view still shares geometry.
    keypoint_count = 0
    keypoints = o3d.geometry.keypoint.compute_iss_keypoints(
        source.voxel_down_sample(0.020), salient_radius=.080,
        non_max_radius=.040, min_neighbors=5,
    )
    keypoints_points = np.asarray(keypoints.points)
    matched_keypoints = []
    for point in keypoints_points:
        found, _indices, squared = tree.search_knn_vector_3d(point, 1)
        if found and squared[0] <= 0.028 ** 2:
            matched_keypoints.append(point)
    if matched_keypoints:
        matched_keypoints = np.asarray(matched_keypoints)
        keypoint_count = len(matched_keypoints)
        points = np.vstack((matched_source, matched_target, midpoints, matched_keypoints))
    else:
        points = np.vstack((matched_source, matched_target, midpoints))
    colors = np.vstack((
        np.tile((1.0, 0.10, 0.10), (len(matched_source), 1)),
        np.tile((0.10, 1.0, 0.10), (len(matched_target), 1)),
        np.tile((0.10, 0.55, 1.0), (len(midpoints), 1)),
    ))
    if keypoint_count:
        colors = np.vstack((colors, np.tile((1.0, 0.90, 0.05), (keypoint_count, 1))))
    debug = o3d.geometry.PointCloud()
    debug.points = o3d.utility.Vector3dVector(points)
    debug.colors = o3d.utility.Vector3dVector(colors)
    path = output / f"{frame_id}_icp_correspondences.ply"
    o3d.io.write_point_cloud(str(path), debug, compressed=True)
    return path, len(matched_source), keypoint_count


def confirmed_adjacent_cloud(previous_full, current_full,
                             distance=ADJACENT_CONFIRM_DISTANCE_M):
    """Return only mutually matched surface features from adjacent views.

    A source point must choose a target point as its nearest neighbour *and*
    that target must choose the same source point.  The stored point is the
    pair midpoint, not two overlaid copies.  Thus every output feature has
    been observed in two adjacent frames; a one-frame reflection can never
    enter the final model.  Previously confirmed midpoints are deliberately
    never removed by later views.
    """
    previous = o3d.geometry.PointCloud(previous_full).voxel_down_sample(0.012)
    current = o3d.geometry.PointCloud(current_full).voxel_down_sample(0.012)
    previous_points = np.asarray(previous.points)
    current_points = np.asarray(current.points)
    if not len(previous_points) or not len(current_points):
        return o3d.geometry.PointCloud(), 0
    previous_tree = o3d.geometry.KDTreeFlann(previous)
    current_tree = o3d.geometry.KDTreeFlann(current)
    current_indices, previous_indices = [], []
    for index, point in enumerate(current_points):
        found, indices, squared = previous_tree.search_knn_vector_3d(point, 1)
        if not found or squared[0] > distance * distance:
            continue
        previous_index = indices[0]
        reciprocal, reverse_indices, reverse_squared = current_tree.search_knn_vector_3d(
            previous_points[previous_index], 1
        )
        if reciprocal and reverse_indices[0] == index and reverse_squared[0] <= distance * distance:
            current_indices.append(index)
            previous_indices.append(previous_index)
    if not current_indices:
        return o3d.geometry.PointCloud(), 0
    # One midpoint represents one physical feature.  Keeping both raw points
    # was the source of visible double layers when ICP had a few mm residual.
    midpoints = (current_points[current_indices] + previous_points[previous_indices]) * 0.5
    confirmed = o3d.geometry.PointCloud()
    confirmed.points = o3d.utility.Vector3dVector(midpoints)
    if current.has_colors() and previous.has_colors():
        colors = (np.asarray(current.colors)[current_indices] +
                  np.asarray(previous.colors)[previous_indices]) * 0.5
        confirmed.colors = o3d.utility.Vector3dVector(colors)
    return confirmed.voxel_down_sample(0.012), len(current_indices)


def fuse_icp(clouds, total=None, debug_dir: Path | None = None, imu_yaws=None):
    """Sequential, coarse-to-fine ICP for files captured in viewing order."""
    iterator = iter(clouds)
    try:
        anchor_id, anchor_cloud = next(iterator)
    except StopIteration:
        raise SystemExit("No valid point cloud could be generated.")
    total = total or "?"
    accumulated = anchor_cloud
    # Registration target is intentionally *only* the previous accepted
    # view.  Matching against the full accumulated model lets a new right-end
    # view match an old, geometrically identical left-end opening.
    previous_accepted = anchor_cloud
    accepted_clouds = [anchor_cloud.voxel_down_sample(0.012)]
    # Keep the clean, accepted full views in ``accumulated``.  An exterior
    # surface can be genuinely occluded in the next view, so simple stitching
    # must never delete it merely because it lacks overlap.
    confirmed = o3d.geometry.PointCloud()
    last_transform = np.eye(4)
    report = [{"frame_id": anchor_id, "status": "anchor"}]
    imu_yaws = imu_yaws or {}
    previous_frame_id = anchor_id
    for index, (frame_id, source_full) in enumerate(iterator, 2):
        # Crucially, use the last accepted camera pose as the next initial
        # pose. Starting every frame at identity caused wrong alignments.
        imu_delta = None
        if frame_id in imu_yaws and previous_frame_id in imu_yaws:
            imu_delta = (imu_yaws[frame_id] - imu_yaws[previous_frame_id] + 180.0) % 360.0 - 180.0
        transform, result = refine_icp_with_yaw_search(
            source_full, previous_accepted, last_transform, imu_delta
        )
        debug_path, correspondence_count, keypoint_count = (None, 0, 0)
        if debug_dir is not None:
            debug_path, correspondence_count, keypoint_count = save_icp_correspondence_debug(
                source_full, previous_accepted, transform, debug_dir, frame_id
            )
        # Cylindrical/near-symmetric views can look deceptively close to ICP.
        # Do not accept a weak fit: it produces a convincing but displaced
        # second shell that is worse than leaving that view unmerged.
        if result.fitness < 0.45 or result.inlier_rmse > 0.025:
            print(f"[warning] {frame_id}: ambiguous/bad ICP (fitness={result.fitness:.3f}, RMSE={result.inlier_rmse:.4f} m); skipped.")
            report.append({"frame_id": frame_id, "status": "rejected_ambiguous_icp", "fitness": result.fitness, "rmse_m": result.inlier_rmse,
                           "correspondence_points": correspondence_count, "keypoints": keypoint_count, "debug_cloud": str(debug_path) if debug_path else None})
            continue
        relative = np.linalg.inv(last_transform) @ transform
        step_translation = float(np.linalg.norm(relative[:3, 3]))
        step_angle = float(np.degrees(np.arccos(np.clip((np.trace(relative[:3, :3]) - 1) / 2, -1, 1))))
        # A symmetric fuselage can give ICP a deceptively good score after it
        # snaps to another orientation. Adjacent capture files should not make
        # a large pose jump, so do not merge such a frame into the model.
        # Turning around an object 3 m away legitimately creates metres of
        # translation in the camera-coordinate transform.  Judge translation
        # relative to that viewing radius instead of a fixed 0.60 m limit.
        viewing_radius = max(
            float(np.linalg.norm(np.asarray(source_full.points).mean(axis=0))),
            float(np.linalg.norm(np.asarray(previous_accepted.points).mean(axis=0))),
        )
        allowed_translation = max(
            MAX_ICP_STEP_TRANSLATION_M,
            2.0 * viewing_radius * np.sin(np.radians(step_angle) / 2.0) + 0.40,
        )
        if step_translation > allowed_translation or step_angle > 35.0:
            print(f"[warning] {frame_id}: pose jump ({step_translation:.3f} m, {step_angle:.1f} deg); skipped.")
            report.append({"frame_id": frame_id, "status": "rejected_pose_jump", "fitness": result.fitness, "rmse_m": result.inlier_rmse, "step_translation_m": step_translation, "step_rotation_deg": step_angle,
                           "correspondence_points": correspondence_count, "keypoints": keypoint_count, "debug_cloud": str(debug_path) if debug_path else None})
            continue
        accepted = source_full.transform(transform)
        pair_cloud, pair_count = confirmed_adjacent_cloud(previous_accepted, accepted)
        # Record the adjacent mutual matches for diagnosis, but do not delete
        # an otherwise valid full frame merely because a portion of it is not
        # visible in the next view.  The user selected simple stitching:
        # geometric checks guard the pose; every point from an accepted frame
        # remains in the fused result.
        if pair_count < MIN_ADJACENT_CONFIRMED_FEATURES:
            print(
                f"[info] {frame_id}: {pair_count} mutually matched adjacent features; "
                "keeping full frame after valid ICP."
            )
        accepted_clouds.append(accepted.voxel_down_sample(0.012))
        confirmed += pair_cloud
        accumulated += accepted
        accumulated = accumulated.voxel_down_sample(0.006)
        previous_accepted = accepted
        previous_frame_id = frame_id
        last_transform = transform
        print(f"[{index}/{total}] ICP {frame_id}: fitness={result.fitness:.3f}, RMSE={result.inlier_rmse:.4f} m, step={step_translation:.3f} m/{step_angle:.1f} deg, confirmed={pair_count:,}")
        report.append({"frame_id": frame_id, "status": "accepted", "fitness": result.fitness, "rmse_m": result.inlier_rmse, "step_translation_m": step_translation, "step_rotation_deg": step_angle,
                       "correspondence_points": correspondence_count, "pair_confirmed_points": pair_count,
                       "keypoints": keypoint_count, "imu_yaw_delta_deg": imu_delta,
                       "debug_cloud": str(debug_path) if debug_path else None})
    # Simple stitching mode: retain all depth-filtered points from every ICP
    # accepted frame.  Do not run final cluster/statistical cleanup or
    # two-frame-only filtering, because both remove true non-overlapping
    # surfaces and make the fused shape poorer than a single frame.
    return accumulated, report


def main():
    root = INPUT_ROOT.resolve()
    frames, intrinsics = find_frames(root), load_intrinsics(root)
    print("[info] Point selection uses only the complete masks/ silhouette.")
    print("\n1) Generate one selected frame\n2) Generate a point cloud for every frame\n3) Generate every frame and fuse sequentially with ICP")
    mode = input("Enter mode [1/2/3]: ").strip()
    # Every menu run receives its own timestamped directory, so no output is
    # overwritten when the same input is reconstructed with new parameters.
    output = (root.parent / "object_pointcloud_output" / datetime.now().strftime("%Y%m%d_%H%M%S")).resolve()
    output.mkdir(parents=True, exist_ok=False)
    print(f"[info] This run will be saved in: {output}")
    if mode == "1":
        for index, frame in enumerate(frames, 1): print(f"  {index}: {frame[0]}")
        choice = input(f"Frame number [1-{len(frames)}]: ").strip()
        try: frame = frames[int(choice)-1]
        except (ValueError, IndexError): raise SystemExit("Invalid frame number.")
        frame_id, cloud = make_cloud(frame, intrinsics)
        cloud = save_cloud(cloud, output, f"single_{frame_id}", clean=False)
        print(f"Saved {len(cloud.points):,} points to {output}")
    elif mode == "2":
        for frame_id, cloud in build_all(frames, intrinsics):
            save_cloud(cloud, output, f"frame_{frame_id}", clean=False)
        print(f"Saved individual point clouds to {output}")
    elif mode == "3":
        def cloud_stream():
            for index, frame in enumerate(frames, 1):
                try:
                    frame_id, cloud = make_cloud(frame, intrinsics)
                    print(f"[{index}/{len(frames)}] {frame_id}: {len(cloud.points):,} points")
                    # Defer expensive statistical filtering to the final fused
                    # cloud. Repeating it for 48 frames exhausts memory.
                    save_cloud(cloud, output, f"frame_{frame_id}", clean=False)
                    yield frame_id, cloud
                except (ValueError, OSError) as error:
                    print(f"[{index}/{len(frames)}] skip {frame[0]}: {error}")

        poses, pose_path = find_saved_poses(root, [frame[0] for frame in frames])
        if len(poses) >= max(8, int(len(frames) * 0.8)):
            print(f"[info] Using {len(poses)} saved camera poses from {pose_path}.")
            tag_poses, tag_id, tag_report = detect_front_apriltag_poses(frames, intrinsics)
            fused, registration_report = fuse_with_saved_poses(
                cloud_stream(), poses, total=len(frames),
                tag_poses=tag_poses, tag_id=tag_id,
            )
            registration_report = tag_report + registration_report
        else:
            print("[warning] No matching saved camera poses; falling back to sequential ICP.")
            imu_yaws = find_saved_imu_yaws(root, [frame[0] for frame in frames])
            if imu_yaws:
                print(f"[info] Using vehicle IMU yaw constraints for {len(imu_yaws)}/{len(frames)} frame(s).")
            fused, registration_report = fuse_icp(
                cloud_stream(), total=len(frames), imu_yaws=imu_yaws
            )
        fused = save_cloud(fused, output, "fused_icp")
        (output / "icp_registration_report.json").write_text(
            json.dumps(registration_report, indent=2), encoding="utf-8"
        )
        print(f"Saved fused cloud ({len(fused.points):,} points) to {output}")
    else:
        raise SystemExit("Mode must be 1, 2, or 3.")


if __name__ == "__main__": main()
