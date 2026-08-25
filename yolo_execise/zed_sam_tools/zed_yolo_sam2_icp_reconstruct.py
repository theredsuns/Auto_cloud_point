#!/usr/bin/env python3
"""Strict YOLO + SAM2 ZED object reconstruction using multi-scale ICP.

Only SAM2-masked depth points enter the reconstruction.  Frames whose ICP fit
fails the configurable quality gate are displayed but never fused.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import pyzed.sl as sl

from zed_yolo_sam2_live import ROOT, clean_mask, load_sam2, load_yolo, masked_cloud


def make_cloud(points: np.ndarray, colors: np.ndarray, voxel: float) -> o3d.geometry.PointCloud:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(colors)
    cloud = cloud.voxel_down_sample(voxel)
    if len(cloud.points) > 30:
        cloud.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 4, max_nn=40)
        )
    return cloud


def register(source: o3d.geometry.PointCloud, target: o3d.geometry.PointCloud,
             voxel: float, initial: np.ndarray | None = None
             ) -> o3d.pipelines.registration.RegistrationResult:
    """Coarse-to-fine point-to-plane ICP; result maps source into target."""
    transform = np.eye(4) if initial is None else initial.copy()
    result = None
    for multiplier, iterations in ((6.0, 50), (3.0, 35), (1.5, 25)):
        result = o3d.pipelines.registration.registration_icp(
            source, target, voxel * multiplier, transform,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=iterations),
        )
        transform = result.transformation
    assert result is not None
    return result


def select_target(result, predictor, rgb, class_name: str | None):
    """Return one stable semantic instance: highest-confidence matching box."""
    if result.boxes is None or len(result.boxes) == 0:
        return None
    boxes = result.boxes.xyxy.detach().cpu().numpy()
    scores = result.boxes.conf.detach().cpu().numpy()
    classes = result.boxes.cls.detach().cpu().numpy().astype(int)
    candidates = [
        i for i, class_id in enumerate(classes)
        if class_name is None or result.names[class_id] == class_name
    ]
    if not candidates:
        return None
    index = max(candidates, key=lambda i: float(scores[i]))
    predictor.set_image(rgb)
    masks, _scores, _logits = predictor.predict(
        point_coords=None, point_labels=None, box=boxes[index][None, :],
        multimask_output=False,
    )
    return boxes[index], float(scores[index]), result.names[classes[index]], clean_mask(np.squeeze(masks[0]))


def select_target_parts(result, predictor, rgb, class_name: str | None):
    """Segment the body and wing with several overlapping SAM box prompts."""
    if result.boxes is None or len(result.boxes) == 0:
        return None
    boxes = result.boxes.xyxy.detach().cpu().numpy()
    scores = result.boxes.conf.detach().cpu().numpy()
    classes = result.boxes.cls.detach().cpu().numpy().astype(int)
    candidates = [
        index
        for index, class_id in enumerate(classes)
        if class_name is None or result.names[class_id] == class_name
    ]
    if not candidates:
        return None
    primary_index = max(candidates, key=lambda index: float(scores[index]))
    primary_box = boxes[primary_index].astype(np.float32)

    def overlap_fraction(first, second):
        x1, y1 = np.maximum(first[:2], second[:2])
        x2, y2 = np.minimum(first[2:], second[2:])
        intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        first_area = max(1.0, (first[2] - first[0]) * (first[3] - first[1]))
        second_area = max(1.0, (second[2] - second[0]) * (second[3] - second[1]))
        return intersection / min(first_area, second_area)

    trusted_boxes = [primary_box]
    for index in candidates:
        if index == primary_index:
            continue
        if overlap_fraction(primary_box, boxes[index]) >= 0.12:
            trusted_boxes.append(boxes[index].astype(np.float32))

    predictor.set_image(rgb)

    def segment(box):
        masks, _mask_scores, _logits = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box[None, :],
            multimask_output=False,
        )
        return clean_mask(np.squeeze(masks[0]))

    union = np.zeros(rgb.shape[:2], dtype=bool)
    used_masks = 0
    for box in trusted_boxes:
        part = segment(box)
        if np.count_nonzero(part) >= max(120, int(part.size * 0.002)):
            union |= part
            used_masks += 1
    if not np.any(union):
        return None
    rows, columns = np.where(union)
    union_box = np.array(
        (columns.min(), rows.min(), columns.max(), rows.max()), dtype=np.float32
    )
    label = f"{result.names[classes[primary_index]]} ({used_masks} SAM prompts)"
    return union_box, float(scores[primary_index]), label, union


def segment_from_previous_box(
    predictor, rgb: np.ndarray, previous_box: np.ndarray
) -> tuple[np.ndarray, float, str, np.ndarray]:
    """Use the last image-space target box when YOLO misses a nearby view."""
    height, width = rgb.shape[:2]
    x1, y1, x2, y2 = previous_box.astype(np.float64)
    pad_x = max(12.0, (x2 - x1) * 0.12)
    pad_y = max(12.0, (y2 - y1) * 0.12)
    box = np.array(
        [
            max(0.0, x1 - pad_x),
            max(0.0, y1 - pad_y),
            min(float(width - 1), x2 + pad_x),
            min(float(height - 1), y2 + pad_y),
        ],
        dtype=np.float32,
    )
    predictor.set_image(rgb)
    masks, _scores, _logits = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=box[None, :],
        multimask_output=False,
    )
    mask = clean_mask(np.squeeze(masks[0]))
    rows, columns = np.where(mask)
    if len(rows):
        tracked_box = np.array(
            [
                float(columns.min()),
                float(rows.min()),
                float(columns.max()),
                float(rows.max()),
            ],
            dtype=np.float32,
        )
    else:
        tracked_box = previous_box.copy()
    return tracked_box, 0.0, "SAM2 previous-box fallback", mask


def save_model(cloud: o3d.geometry.PointCloud, output: Path, mesh: bool) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(output.with_suffix(".ply")), cloud)
    if not mesh or len(cloud.points) < 300:
        return
    work = o3d.geometry.PointCloud(cloud)
    work.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.06, max_nn=60))
    work.orient_normals_consistent_tangent_plane(20)
    surface, density = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(work, depth=9)
    density = np.asarray(density)
    surface.remove_vertices_by_mask(density < np.quantile(density, 0.03))
    surface.remove_degenerate_triangles()
    surface.remove_duplicated_triangles()
    surface.remove_unreferenced_vertices()
    o3d.io.write_triangle_mesh(str(output.with_suffix(".obj")), surface)


def main() -> int:
    parser = argparse.ArgumentParser(description="Fuse SAM2 object point clouds with quality-gated ICP.")
    parser.add_argument("--weights", type=Path, default=ROOT / "best.pt")
    parser.add_argument("--sam2-config", type=Path, required=True)
    parser.add_argument("--sam2-checkpoint", type=Path, required=True)
    parser.add_argument("--target-class", help="Exact YOLO class, e.g. 'Wing and Body'.")
    parser.add_argument("--confidence", type=float, default=0.70)
    parser.add_argument("--voxel-m", type=float, default=0.012)
    parser.add_argument("--max-depth-m", type=float, default=8.0)
    parser.add_argument("--point-stride", type=int, default=2)
    parser.add_argument("--infer-every", type=int, default=2)
    parser.add_argument(
        "--manual-capture",
        action="store_true",
        help="only run segmentation and ICP after Enter is pressed",
    )
    parser.add_argument("--min-fitness", type=float, default=0.55)
    parser.add_argument("--max-rmse-m", type=float, default=0.025)
    parser.add_argument("--output", type=Path, default=ROOT / "models" / "recognized_object")
    parser.add_argument("--mesh-on-save", action="store_true")
    args = parser.parse_args()
    if not args.weights.is_file():
        raise SystemExit(f"YOLO weights not found: {args.weights}")
    if args.voxel_m <= 0 or args.point_stride < 1 or args.infer_every < 1:
        raise SystemExit("--voxel-m must be positive; stride and interval must be at least 1")

    # SAM2 is deliberately mandatory: no box-mask fallback can enter the model.
    yolo = load_yolo(args.weights)
    sam2 = load_sam2(args.sam2_config, args.sam2_checkpoint)
    assert sam2 is not None
    print("Strict mode: YOLO type + SAM2 mask + ICP fusion")

    camera = sl.Camera()
    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.HD720
    init.camera_fps = 30
    init.depth_mode = sl.DEPTH_MODE.ULTRA
    init.coordinate_units = sl.UNIT.MILLIMETER
    status = camera.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        raise SystemExit(f"Could not open ZED: {status}")
    tracking_status = camera.enable_positional_tracking(sl.PositionalTrackingParameters())
    tracking_enabled = tracking_status == sl.ERROR_CODE.SUCCESS
    print(
        "ZED positional guidance: "
        + ("enabled" if tracking_enabled else f"unavailable ({tracking_status})"),
        flush=True,
    )

    image_zed, xyz_zed = sl.Mat(), sl.Mat()
    camera_pose = sl.Pose()
    viewer = o3d.visualization.Visualizer()
    viewer.create_window("SAM2 masked ICP reconstruction", 1200, 850)
    built_visual = o3d.geometry.PointCloud()
    current_visual = o3d.geometry.PointCloud()
    viewer.add_geometry(built_visual)
    viewer.add_geometry(current_visual)
    built = o3d.geometry.PointCloud()
    previous_local = None
    previous_camera_pose = None
    last_target_box = None
    world_from_previous = np.eye(4)
    accepted = rejected = frame = 0
    capture_requested = False
    last_guidance = ""
    last_status = (
        "Ready: press Enter to evaluate this view"
        if args.manual_capture
        else "Waiting for target"
    )
    try:
        while True:
            if camera.grab() != sl.ERROR_CODE.SUCCESS:
                continue
            camera.retrieve_image(image_zed, sl.VIEW.LEFT)
            bgr = cv2.cvtColor(image_zed.get_data(), cv2.COLOR_BGRA2BGR)
            overlay = bgr.copy()
            current_camera_pose = None
            movement_text = ""
            movement_guidance = ""
            if tracking_enabled:
                pose_state = camera.get_position(
                    camera_pose, sl.REFERENCE_FRAME.WORLD
                )
                if pose_state == sl.POSITIONAL_TRACKING_STATE.OK:
                    current_camera_pose = np.asarray(
                        camera_pose.pose_data().m, dtype=np.float64
                    ).copy()
                    # ZED pose translation follows the configured millimetre unit,
                    # while all Open3D clouds use metres.
                    current_camera_pose[:3, 3] /= 1000.0
                    if previous_camera_pose is not None:
                        camera_delta = (
                            np.linalg.inv(previous_camera_pose)
                            @ current_camera_pose
                        )
                        move_cm = np.linalg.norm(camera_delta[:3, 3]) * 100.0
                        cosine = np.clip(
                            (np.trace(camera_delta[:3, :3]) - 1.0) / 2.0,
                            -1.0,
                            1.0,
                        )
                        turn_deg = float(np.degrees(np.arccos(cosine)))
                        movement_text = (
                            f"From last accepted: {move_cm:.1f} cm, "
                            f"{turn_deg:.1f} deg"
                        )
                        if move_cm > 20.0 or turn_deg > 12.0:
                            movement_guidance = (
                                "Too far: move back toward last accepted view"
                            )
                        elif move_cm < 3.0 and turn_deg < 2.0:
                            movement_guidance = (
                                "Move sideways 5-10 cm before next capture"
                            )
                        else:
                            movement_guidance = (
                                "Step looks good: hold still and press Enter"
                            )
            should_process = (
                capture_requested
                if args.manual_capture
                else frame % args.infer_every == 0
            )
            if should_process:
                capture_requested = False
                captured_camera_pose = (
                    None
                    if current_camera_pose is None
                    else current_camera_pose.copy()
                )
                last_status = "Evaluating YOLO + SAM2 mask and depth..."
                print("[capture] evaluating current view...", flush=True)
                camera.retrieve_measure(xyz_zed, sl.MEASURE.XYZRGBA)
                selected = select_target(
                    yolo.predict(bgr, conf=args.confidence, verbose=False)[0], sam2,
                    cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), args.target_class,
                )
                used_box_fallback = False
                if selected is None and last_target_box is not None:
                    selected = segment_from_previous_box(
                        sam2,
                        cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
                        last_target_box,
                    )
                    used_box_fallback = True
                    print(
                        "[fallback] YOLO missed; SAM2 is using the previous target box",
                        flush=True,
                    )
                if selected is None:
                    last_status = "No matching YOLO + SAM2 target"
                    last_guidance = "Keep the whole target visible and centered; retry"
                    rejected += 1
                    print(f"[rejected] {last_status}", flush=True)
                else:
                    box, confidence, label, mask = selected
                    last_target_box = box.copy()
                    points, colors = masked_cloud(xyz_zed.get_data(), bgr, mask,
                                                  args.point_stride, args.max_depth_m)
                    if len(points) < 100:
                        last_status = "Target mask has insufficient valid depth"
                        last_guidance = "Move closer or improve surface texture/light; retry"
                        rejected += 1
                        print(f"[rejected] {last_status}: {len(points)} points", flush=True)
                    else:
                        local = make_cloud(points, colors, args.voxel_m)
                        if previous_local is None:
                            world_from_current = np.eye(4)
                            accepted += 1
                            last_status = "Anchor frame accepted"
                            last_guidance = ""
                            print(f"[accepted] {last_status}: {len(local.points)} points", flush=True)
                        else:
                            tracking_initial = None
                            if (
                                previous_camera_pose is not None
                                and captured_camera_pose is not None
                            ):
                                tracking_initial = (
                                    np.linalg.inv(previous_camera_pose)
                                    @ captured_camera_pose
                                )
                            result = register(
                                local,
                                previous_local,
                                args.voxel_m,
                                tracking_initial,
                            )
                            start_mode = (
                                "ZED tracking"
                                if tracking_initial is not None
                                else "identity"
                            )
                            if (
                                tracking_initial is not None
                                and result.fitness < args.min_fitness
                            ):
                                alternatives = [
                                    ("identity retry", np.eye(4)),
                                    (
                                        "inverse tracking retry",
                                        np.linalg.inv(tracking_initial),
                                    ),
                                ]
                                for mode, initial in alternatives:
                                    candidate = register(
                                        local,
                                        previous_local,
                                        args.voxel_m,
                                        initial,
                                    )
                                    if (
                                        candidate.fitness > result.fitness
                                        or (
                                            candidate.fitness == result.fitness
                                            and candidate.inlier_rmse
                                            < result.inlier_rmse
                                        )
                                    ):
                                        result = candidate
                                        start_mode = mode
                                print(
                                    f"[ICP] best initialization: {start_mode}",
                                    flush=True,
                                )
                            if result.fitness < args.min_fitness or result.inlier_rmse > args.max_rmse_m:
                                rejected += 1
                                last_status = (f"ICP rejected: fitness={result.fitness:.2f}, "
                                               f"rmse={result.inlier_rmse:.3f} m ({start_mode})")
                                if result.fitness < args.min_fitness:
                                    last_guidance = (
                                        "Low overlap: return toward last accepted view; "
                                        "halve the step"
                                    )
                                else:
                                    last_guidance = (
                                        "Depth/alignment error: hold still, improve "
                                        "texture/light, retry"
                                    )
                                print(f"[rejected] {last_status}", flush=True)
                                world_from_current = None
                            else:
                                world_from_current = world_from_previous @ result.transformation
                                accepted += 1
                                last_guidance = ""
                                last_status = (f"ICP accepted: fitness={result.fitness:.2f}, "
                                               f"rmse={result.inlier_rmse:.3f} m ({start_mode})")
                                print(f"[accepted] {last_status}", flush=True)
                        if world_from_current is not None:
                            in_world = o3d.geometry.PointCloud(local)
                            in_world.transform(world_from_current)
                            built += in_world
                            built = built.voxel_down_sample(args.voxel_m)
                            previous_local = local
                            if captured_camera_pose is not None:
                                previous_camera_pose = captured_camera_pose
                            world_from_previous = world_from_current
                            built_visual.points = built.points
                            built_visual.colors = built.colors
                            current_visual.points = in_world.points
                            current_visual.colors = in_world.colors
                            viewer.update_geometry(built_visual)
                            viewer.update_geometry(current_visual)
                    x1, y1, x2, y2 = np.round(box).astype(int)
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    overlay[mask] = (overlay[mask] * 0.5 + np.array((0, 255, 0)) * 0.5).astype(np.uint8)
                    detection_text = (
                        "SAM2 previous-box fallback"
                        if used_box_fallback
                        else f"{label} {confidence:.0%}"
                    )
                    cv2.putText(overlay, detection_text, (x1, max(28, y1-8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(overlay, f"SAM2 masked | accepted {accepted} | rejected {rejected}", (16, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(overlay, last_status, (16, 57), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (255, 255, 0), 2, cv2.LINE_AA)
            if movement_text:
                cv2.putText(
                    overlay, movement_text, (16, 86),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA,
                )
            display_guidance = last_guidance or movement_guidance
            if display_guidance:
                cv2.putText(
                    overlay, display_guidance, (16, 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 165, 255), 2, cv2.LINE_AA,
                )
            controls = (
                "Enter: evaluate view  S: save  R: reset  Q/Esc: quit"
                if args.manual_capture
                else "S: save  R: reset  Q/Esc: quit"
            )
            cv2.putText(overlay, controls, (16, overlay.shape[0]-18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow("ZED YOLO + SAM2 + ICP", overlay)
            viewer.poll_events()
            viewer.update_renderer()
            key = cv2.waitKey(1) & 0xFF
            if args.manual_capture and key in (10, 13):
                capture_requested = True
                last_guidance = ""
                last_status = "Capture requested; hold camera still..."
            elif key == ord("s") and len(built.points):
                save_model(built, args.output, args.mesh_on_save)
                last_status = f"Saved {args.output.with_suffix('.ply')}"
            elif key == ord("r"):
                built.clear()
                previous_local = None
                previous_camera_pose = None
                last_target_box = None
                world_from_previous = np.eye(4)
                accepted = rejected = 0
                capture_requested = False
                last_guidance = ""
                last_status = (
                    "Reconstruction reset; press Enter for anchor view"
                    if args.manual_capture
                    else "Reconstruction reset"
                )
            elif key in (27, ord("q")):
                break
            frame += 1
    finally:
        if tracking_enabled:
            camera.disable_positional_tracking()
        camera.close()
        cv2.destroyAllWindows()
        viewer.destroy_window()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
