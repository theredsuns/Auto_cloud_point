#!/usr/bin/env python3
"""无训练的球窝几何测量：拟合口沿圆与中心基准小圆。"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


def parse_roi(values: list[int] | None, width: int, height: int) -> tuple[int, int, int, int]:
    if values is None:
        return 0, 0, width, height
    x, y, roi_width, roi_height = values
    if roi_width <= 0 or roi_height <= 0:
        raise ValueError("ROI 宽高必须大于 0")
    return max(0, x), max(0, y), min(roi_width, width - max(0, x)), min(roi_height, height - max(0, y))


def measure(image: np.ndarray, roi: tuple[int, int, int, int]):
    x0, y0, width, height = roi
    crop = image[y0:y0 + height, x0:x0 + width]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    blurred = cv2.GaussianBlur(gray, (9, 9), 1.5)
    outer = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=max(80, min(width, height) // 6),
        param1=110, param2=30, minRadius=max(25, min(width, height) // 40), maxRadius=min(300, min(width, height) // 2),
    )
    if outer is None:
        raise ValueError("未找到球窝口沿圆；请通过 --roi 缩小到球窝区域")
    # 用“白色圆环 + 中间黑色”给圆候选评分，抑制背景中的普通圆形边缘。
    # 工业现场有曝光变化：白环允许低饱和、中高亮；后续黑心评分负责排除白色背景。
    # 目标白环在实际画面中会偏灰，且有紫色散斑；用低饱和、中等亮度描述它。
    # 黑心以较低亮度描述。两者分别在环形区与中心区判断。
    white = (hsv[:, :, 1] < 180) & (hsv[:, :, 2] >= 60)
    black = hsv[:, :, 2] < 95
    def color_metrics(circle: np.ndarray) -> tuple[float, float]:
        cx, cy, radius = np.round(circle).astype(int)
        # 只计算该候选圆的局部区域；避免每个候选都扫描整张 1280×720 图像。
        left, top = max(0, cx - radius - 2), max(0, cy - radius - 2)
        right, bottom = min(width, cx + radius + 3), min(height, cy + radius + 3)
        yy, xx = np.ogrid[top:bottom, left:right]
        distance = np.hypot(xx - cx, yy - cy)
        ring = (distance > .58 * radius) & (distance < .98 * radius)
        core = distance < .50 * radius
        return float(white[top:bottom, left:right][ring].mean()), float(black[top:bottom, left:right][core].mean())
    # 只接受完整落在画面（或 ROI）内的圆。被边缘截断的圆不用于识别/测量。
    complete = [
        circle for circle in outer[0]
        if circle[0] - circle[2] >= 2 and circle[1] - circle[2] >= 2
        and circle[0] + circle[2] <= width - 3 and circle[1] + circle[2] <= height - 3
    ]
    if not complete:
        raise ValueError("未找到完整位于画面内的球窝圆环")
    # 二维码可能含黑白格，但无法同时满足“环形白 + 中心连续黑”的比例。
    color_candidates = []
    for candidate in complete[:80]:
        white_ring, black_core = color_metrics(candidate)
        if white_ring >= .18 and black_core >= .45:
            color_candidates.append((2 * white_ring + black_core, candidate))
    if not color_candidates:
        raise ValueError("未找到白色环带包围黑色中心的完整圆")
    cx, cy, radius = np.round(max(color_candidates, key=lambda item: item[0])[1]).astype(int)
    core_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(core_mask, (cx, cy), round(radius * .58), 255, -1)
    black_core = cv2.bitwise_and(black.astype(np.uint8) * 255, black.astype(np.uint8) * 255, mask=core_mask)
    contours, _ = cv2.findContours(black_core, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("找到白色圆环，但未找到中间黑色圆")
    contour = max(contours, key=cv2.contourArea)
    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        raise ValueError("中间黑色圆轮廓无效")
    ix, iy = round(moments["m10"] / moments["m00"]), round(moments["m01"] / moments["m00"])
    inner_radius = round((cv2.contourArea(contour) / np.pi) ** .5)
    center_offset = np.hypot(ix - cx, iy - cy)
    radius_ratio = inner_radius / radius
    if (
        cv2.contourArea(contour) < .10 * np.pi * radius * radius
        or center_offset > .25 * radius
        or not .40 <= radius_ratio <= .80
    ):
        raise ValueError("候选圆的黑色中心不连续或未位于白环中心")
    return (cx + x0, cy + y0, radius), (ix + x0, iy + y0, inner_radius)


def main() -> int:
    parser = argparse.ArgumentParser(description="不经 YOLO 训练，直接测球窝口沿与中心基准圆的偏差。")
    parser.add_argument("source", type=Path, help="图片文件或目录")
    parser.add_argument("--roi", nargs=4, type=int, metavar=("X", "Y", "W", "H"), help="球窝区域，建议固定相机时设置")
    parser.add_argument("--mm-per-pixel", type=float, help="同一测量平面的毫米/像素标定值")
    parser.add_argument("--output", type=Path, default=Path("sucker_seek/geometry_results"))
    args = parser.parse_args()
    images = [args.source] if args.source.is_file() else sorted(p for p in args.source.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    if not images:
        raise SystemExit("没有可处理的图片")
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in images:
        image = cv2.imread(str(path))
        if image is None:
            continue
        try:
            roi = parse_roi(args.roi, image.shape[1], image.shape[0])
            (cx, cy, radius), (ix, iy, inner_radius) = measure(image, roi)
        except ValueError as error:
            print(f"[跳过] {path.name}: {error}")
            continue
        offset_px = float(np.hypot(cx - ix, cy - iy))
        row = {"image": path.name, "socket_cx_px": cx, "socket_cy_px": cy, "socket_diameter_px": 2 * radius, "reference_cx_px": ix, "reference_cy_px": iy, "center_offset_px": round(offset_px, 3)}
        if args.mm_per_pixel is not None:
            row["socket_diameter_mm"] = round(2 * radius * args.mm_per_pixel, 3)
            row["center_offset_mm"] = round(offset_px * args.mm_per_pixel, 3)
        rows.append(row)
        shown = image.copy()
        cv2.circle(shown, (cx, cy), radius, (0, 220, 0), 2)
        cv2.circle(shown, (ix, iy), inner_radius, (255, 0, 255), 2)
        cv2.drawMarker(shown, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
        cv2.drawMarker(shown, (ix, iy), (255, 0, 255), cv2.MARKER_CROSS, 20, 2)
        cv2.line(shown, (cx, cy), (ix, iy), (0, 255, 255), 2)
        cv2.putText(shown, f"offset {offset_px:.1f}px", (cx + 12, cy - 12), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 255), 2)
        cv2.imwrite(str(args.output / path.name), shown)
        print(f"[测量] {path.name}: 口径={2*radius}px，圆心偏差={offset_px:.2f}px")
    fields = ["image", "socket_cx_px", "socket_cy_px", "socket_diameter_px", "reference_cx_px", "reference_cy_px", "center_offset_px"]
    if args.mm_per_pixel is not None:
        fields += ["socket_diameter_mm", "center_offset_mm"]
    with (args.output / "socket_measurements.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
