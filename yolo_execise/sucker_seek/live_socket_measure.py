#!/usr/bin/env python3
"""实时测量固定球窝的口沿尺寸与中心偏差。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
REAL_SENSE_RUNTIME = Path(__file__).resolve().parents[1] / "libraries" / "realsense_runtime"
if REAL_SENSE_RUNTIME.is_dir():
    sys.path.insert(0, str(REAL_SENSE_RUNTIME))
from measure_socket_geometry import measure, parse_roi  # noqa: E402


def robust_xyz(xyz: np.ndarray, x: int, y: int) -> np.ndarray | None:
    """读取圆心附近 5×5 像素的中位三维点，避开单点深度空洞。"""
    patch = xyz[max(0, y - 2):y + 3, max(0, x - 2):x + 3, :3].reshape(-1, 3)
    patch = patch[np.isfinite(patch).all(axis=1)]
    patch = patch[(patch[:, 2] > 0) & (patch[:, 2] < 10_000)]
    return np.median(patch, axis=0) if len(patch) else None


def robust_depth_mm(depth: np.ndarray, x: int, y: int, scale: float) -> float | None:
    patch = depth[max(0, y - 2):y + 3, max(0, x - 2):x + 3].reshape(-1)
    patch = patch[patch > 0]
    return float(np.median(patch) * scale * 1000) if len(patch) else None


def robust_ring_depth_mm(depth: np.ndarray, x: int, y: int, radius: int, scale: float) -> float | None:
    """黑色中心容易无深度，改从白色圆环内侧取深度中值。"""
    left, top = max(0, x - radius), max(0, y - radius)
    right, bottom = min(depth.shape[1], x + radius + 1), min(depth.shape[0], y + radius + 1)
    yy, xx = np.ogrid[top:bottom, left:right]
    distance = np.hypot(xx - x, yy - y)
    samples = depth[top:bottom, left:right][(distance > .68 * radius) & (distance < .92 * radius)]
    samples = samples[samples > 0]
    return float(np.median(samples) * scale * 1000) if len(samples) else None


def detect_socket(frame: np.ndarray, roi: tuple[int, int, int, int], detect_width: int):
    """在缩小画面上检测，返回原图像素坐标，避免实时画面卡顿。"""
    height, width = frame.shape[:2]
    scale = max(1.0, width / detect_width)
    if scale == 1.0:
        return measure(frame, roi)
    small = cv2.resize(frame, (round(width / scale), round(height / scale)), interpolation=cv2.INTER_AREA)
    x, y, roi_width, roi_height = roi
    small_roi = (round(x / scale), round(y / scale), round(roi_width / scale), round(roi_height / scale))
    outer, inner = measure(small, small_roi)
    return tuple(round(value * scale) for value in outer), tuple(round(value * scale) for value in inner)


def matches_color_calibration(frame: np.ndarray, center_x: int, center_y: int, radius: int, calibration: dict) -> bool:
    """按手动框选时的比例和像素数量，验证当前候选区域。"""
    reference_width, reference_height = calibration["roi_width"], calibration["roi_height"]
    scale = (2 * radius) / min(reference_width, reference_height)
    width, height = round(reference_width * scale), round(reference_height * scale)
    left, top = round(center_x - width / 2), round(center_y - height / 2)
    if left < 0 or top < 0 or left + width > frame.shape[1] or top + height > frame.shape[0]:
        return False
    hsv = cv2.cvtColor(frame[top:top + height, left:left + width], cv2.COLOR_BGR2HSV)
    white = (hsv[:, :, 1] < calibration["white_saturation_max"]) & (hsv[:, :, 2] > calibration["white_value_min"])
    black = hsv[:, :, 2] < calibration["black_value_max"]
    tolerance = calibration["tolerance"]
    expected_white = calibration["white_pixels"] * scale * scale
    expected_black = calibration["black_pixels"] * scale * scale
    return (
        abs(float(white.sum()) - expected_white) <= expected_white * tolerance
        and abs(float(black.sum()) - expected_black) <= expected_black * tolerance
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="打开摄像头，实时测量球窝。")
    parser.add_argument("--camera", type=int, default=0, help="摄像头编号，默认 0")
    parser.add_argument("--zed", action="store_true", help="使用 ZED 左目摄像头")
    parser.add_argument("--realsense", action="store_true", help="使用 Intel RealSense 彩色与深度摄像头")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--roi", nargs=4, type=int, metavar=("X", "Y", "W", "H"), help="固定球窝区域")
    parser.add_argument("--select-roi", action="store_true", help="在首帧用鼠标框选球窝区域")
    parser.add_argument("--mm-per-pixel", type=float, help="毫米/像素标定值")
    parser.add_argument("--outer-diameter-mm", type=float, default=60.0, help="球窝外圈实际直径，默认 60 mm")
    parser.add_argument("--inner-diameter-mm", type=float, default=40.0, help="球窝内圈实际直径，默认 40 mm")
    parser.add_argument("--max-z-mm", type=float, default=500.0, help="有效测量的最大 Z 距离，默认 500 mm")
    parser.add_argument("--max-model-z-mm", type=float, help="圆环尺寸模型允许的最远距离；默认不以尺寸距离过滤")
    parser.add_argument("--min-model-z-mm", type=float, default=0.0, help="圆环尺寸模型允许的最近距离；用于排除过大的圆")
    parser.add_argument("--use-depth-gate", action="store_true", help="启用深度 Z 阈值过滤（默认只做完整图案识别）")
    parser.add_argument("--detect-width", type=int, default=1280, help="圆检测使用的最大画面宽度，默认 1280（保留小圆细节）")
    parser.add_argument("--detect-every", type=int, default=5, help="每隔多少帧重新检测一次，默认 5")
    parser.add_argument("--show-gray", action="store_true", help="显示圆检测使用的灰度调试画面")
    parser.add_argument("--color-calibration", type=Path, default=Path("sucker_seek/socket_color_calibration.json"), help="手动框选生成的白黑像素标定 JSON")
    args = parser.parse_args()
    if args.zed and args.realsense:
        parser.error("--zed 与 --realsense 只能选择一个")
    if args.detect_width < 320 or args.detect_every < 1:
        parser.error("--detect-width 至少为 320，--detect-every 至少为 1")
    if args.min_model_z_mm < 0:
        parser.error("--min-model-z-mm 必须不小于 0")
    if args.max_model_z_mm is not None and args.min_model_z_mm >= args.max_model_z_mm:
        parser.error("--min-model-z-mm 必须小于 --max-model-z-mm")
    if args.max_model_z_mm is None and args.min_model_z_mm > 0:
        parser.error("使用 --min-model-z-mm 时必须同时提供 --max-model-z-mm")
    calibration = None
    if args.color_calibration.is_file():
        try:
            calibration = json.loads(args.color_calibration.read_text(encoding="utf-8"))
            required = {"roi_width", "roi_height", "white_pixels", "black_pixels", "tolerance", "white_saturation_max", "white_value_min", "black_value_max"}
            if not required <= calibration.keys():
                raise ValueError("缺少必要字段")
            total = calibration["roi_width"] * calibration["roi_height"]
            if calibration["white_pixels"] + calibration["black_pixels"] > total * 1.02:
                print("[颜色标定] 白黑像素存在重叠，已忽略该旧标定；请重新运行 manual_socket_calibration.py。")
                calibration = None
            else:
                print(f"[颜色标定] 已加载 {args.color_calibration}")
        except (OSError, ValueError, json.JSONDecodeError) as error:
            raise SystemExit(f"无法读取颜色标定文件 {args.color_calibration}: {error}")
    camera = zed = image = xyz_zed = calibration = None
    rs = rs_pipeline = rs_align = rs_depth_scale = rs_intrinsics = None
    if args.zed:
        try:
            import pyzed.sl as sl
        except ImportError as error:
            raise SystemExit("未安装 ZED SDK Python 模块 pyzed.sl。") from error
        zed = sl.Camera()
        init = sl.InitParameters()
        init.camera_resolution = sl.RESOLUTION.HD720
        init.camera_fps = 30
        init.coordinate_units = sl.UNIT.MILLIMETER
        if zed.open(init) != sl.ERROR_CODE.SUCCESS:
            raise SystemExit("无法打开 ZED；请检查 ZED 连接、驱动和 CUDA 环境。")
        image = sl.Mat()
        xyz_zed = sl.Mat()
        calibration = zed.get_camera_information().camera_configuration.calibration_parameters.left_cam
    elif args.realsense:
        try:
            import pyrealsense2 as rs
        except ImportError as error:
            raise SystemExit("未安装 pyrealsense2。请安装 Intel RealSense SDK 的 Python 模块后重试。") from error
        rs_pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, 30)
        if args.use_depth_gate:
            config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, 30)
        profile = rs_pipeline.start(config)
        if args.use_depth_gate:
            rs_depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
            rs_align = rs.align(rs.stream.color)
        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        rs_intrinsics = color_profile.get_intrinsics()
    else:
        camera = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
        if not camera.isOpened():
            camera = cv2.VideoCapture(args.camera)
        if not camera.isOpened():
            raise SystemExit(f"无法打开摄像头 {args.camera}；请检查连接，或改用 --camera 1。")
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    window = "Socket measurement: Q / Esc quit"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    frame_number = 0
    last_measurement = None
    last_error = "正在寻找白色圆环与黑色中心"
    try:
        while True:
            rs_depth = None
            if zed is not None:
                ok = zed.grab() == sl.ERROR_CODE.SUCCESS
                if ok:
                    zed.retrieve_image(image, sl.VIEW.LEFT)
                    zed.retrieve_measure(xyz_zed, sl.MEASURE.XYZRGBA)
                    frame = cv2.cvtColor(image.get_data(), cv2.COLOR_BGRA2BGR)
            elif rs_pipeline is not None:
                frames = rs_pipeline.wait_for_frames()
                aligned = rs_align.process(frames) if args.use_depth_gate else frames
                color_frame = aligned.get_color_frame()
                depth_frame = aligned.get_depth_frame() if args.use_depth_gate else None
                ok = bool(color_frame) and (not args.use_depth_gate or bool(depth_frame))
                if ok:
                    frame = np.asanyarray(color_frame.get_data())
                    if args.use_depth_gate:
                        rs_depth = np.asanyarray(depth_frame.get_data())
            else:
                ok, frame = camera.read()
            if not ok:
                print("无法读取摄像头画面")
                break
            if args.select_roi and args.roi is None:
                selected = cv2.selectROI("Drag to select socket ROI, Enter confirm", frame, fromCenter=False, showCrosshair=True)
                cv2.destroyWindow("Drag to select socket ROI, Enter confirm")
                if selected[2] <= 0 or selected[3] <= 0:
                    raise SystemExit("未选择有效球窝区域")
                args.roi = [int(value) for value in selected]
                print(f"[ROI] 已选择：{args.roi}")
            shown = frame.copy()
            if args.show_gray:
                cv2.imshow("Socket detection grayscale", cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            try:
                roi = parse_roi(args.roi, frame.shape[1], frame.shape[0])
                frame_number += 1
                if frame_number % args.detect_every == 1 or last_measurement is None:
                    try:
                        last_measurement = detect_socket(frame, roi, args.detect_width)
                        last_error = ""
                    except ValueError as error:
                        last_measurement = None
                        last_error = str(error)
                if last_measurement is None:
                    raise ValueError(last_error)
                (cx, cy, radius), (ix, iy, inner_radius) = last_measurement
                offset_px = ((cx - ix) ** 2 + (cy - iy) ** 2) ** .5
                diameter = 2 * radius
                text = f"diameter: {diameter}px | offset: {offset_px:.2f}px"
                # 已知外/内圈尺寸的针孔模型深度，是本程序输出的测量结果。
                # 深度相机只用于 0.5 m 距离门槛，不参与该数值的计算。
                if calibration is not None:
                    focal = (float(calibration.fx) + float(calibration.fy)) / 2
                elif rs_intrinsics is not None:
                    focal = (float(rs_intrinsics.fx) + float(rs_intrinsics.fy)) / 2
                else:
                    focal = None
                model_z = None
                depth_estimates: list[float] = []
                if focal is not None:
                    outer_model_z = focal * args.outer_diameter_mm / diameter
                    depth_estimates = [outer_model_z]
                    if inner_radius > 0:
                        inner_model_z = focal * args.inner_diameter_mm / (2 * inner_radius)
                        depth_estimates.append(inner_model_z)
                    model_z = float(np.median(depth_estimates))
                    ratio = (2 * inner_radius) / diameter
                    text += f" | inner/outer:{ratio:.2f} | Zouter:{outer_model_z:.0f}mm"
                    if len(depth_estimates) == 2:
                        text += f" Zinner:{inner_model_z:.0f}mm"
                    text += f" model Z:{model_z:.0f}mm"
                    size_valid = True
                    if args.max_model_z_mm is not None:
                        # 投影面积门槛：距离不超过 max_model_z 时，圆的像素面积不得更小。
                        outer_area = np.pi * (diameter / 2) ** 2
                        inner_area = np.pi * inner_radius ** 2
                        outer_min_area = np.pi * (focal * args.outer_diameter_mm / (2 * args.max_model_z_mm)) ** 2
                        inner_min_area = np.pi * (focal * args.inner_diameter_mm / (2 * args.max_model_z_mm)) ** 2
                        size_valid = outer_area >= outer_min_area and inner_area >= inner_min_area
                        if args.min_model_z_mm > 0:
                            outer_max_area = np.pi * (focal * args.outer_diameter_mm / (2 * args.min_model_z_mm)) ** 2
                            inner_max_area = np.pi * (focal * args.inner_diameter_mm / (2 * args.min_model_z_mm)) ** 2
                            size_valid = size_valid and outer_area <= outer_max_area and inner_area <= inner_max_area
                else:
                    size_valid = False
                sensor_z = None
                range_rejected = False
                if args.use_depth_gate and xyz_zed is not None:
                    point = robust_xyz(xyz_zed.get_data(), cx, cy)
                    if point is not None:
                        sensor_z = float(point[2])
                        range_rejected = sensor_z >= args.max_z_mm
                elif args.use_depth_gate and rs_depth is not None:
                    # 小圆中心为黑色，优先取白色环带的深度；中心点仅作回退。
                    depth_mm = robust_ring_depth_mm(rs_depth, cx, cy, radius, rs_depth_scale)
                    if depth_mm is None:
                        depth_mm = robust_depth_mm(rs_depth, cx, cy, rs_depth_scale)
                    if depth_mm is not None:
                        sensor_z = depth_mm
                        range_rejected = sensor_z >= args.max_z_mm
                if args.mm_per_pixel is not None:
                    text += f" | {diameter * args.mm_per_pixel:.2f}mm / {offset_px * args.mm_per_pixel:.2f}mm"
                # 合格图案：白色外环、黑色内圆（由 measure() 的颜色评分保证），
                # 且直径比接近已知 40/60=0.667。倾斜会改变投影，故留出容差。
                # 颜色/尺寸模式已在 measure() 中完成筛选；深度相机模式只以真实 Z 做门槛。
                # 不再以像素尺寸推算深度作为拒绝条件，避免倾斜或轻微拟合误差漏检。
                # 深度仅作单向远距离过滤。近距离常无有效深度，不能因此丢弃颜色识别结果。
                color_valid = calibration is None or matches_color_calibration(frame, cx, cy, radius, calibration)
                valid = not range_rejected and size_valid and color_valid
                # 只有实际读到 Z >= 500 mm 的目标不画框、不输出识别结果。
                if valid:
                    cv2.circle(shown, (cx, cy), radius, (0, 220, 0), 2)
                    cv2.circle(shown, (ix, iy), inner_radius, (255, 0, 255), 2)
                    cv2.drawMarker(shown, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 18, 2)
                    cv2.drawMarker(shown, (ix, iy), (255, 0, 255), cv2.MARKER_CROSS, 18, 2)
                    cv2.line(shown, (cx, cy), (ix, iy), (0, 255, 255), 2)
                    cv2.putText(shown, text, (18, 34), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 0), 3)
                    cv2.putText(shown, text, (18, 34), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 0), 1)
            except ValueError as error:
                cv2.putText(shown, str(error), (18, 34), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 255), 2)
            if args.roi is not None:
                x, y, w, h = parse_roi(args.roi, frame.shape[1], frame.shape[0])
                cv2.rectangle(shown, (x, y), (x + w, y + h), (255, 255, 0), 1)
            cv2.imshow(window, shown)
            if cv2.waitKey(1) & 0xFF in (27, ord("q"), ord("Q")):
                break
    finally:
        if camera is not None:
            camera.release()
        if zed is not None:
            zed.close()
        if rs_pipeline is not None:
            rs_pipeline.stop()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
