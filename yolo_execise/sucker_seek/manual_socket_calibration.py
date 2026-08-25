#!/usr/bin/env python3
"""手动框选球窝，记录白色/黑色像素数量，供实时颜色筛选使用。"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "libraries" / "realsense_runtime"
if RUNTIME.is_dir():
    sys.path.insert(0, str(RUNTIME))

# 白/黑必须互斥，灰色过渡区域不计入任一类，才能让像素数量可稳定比较。
WHITE_SAT_MAX, WHITE_VALUE_MIN, BLACK_VALUE_MAX = 150, 90, 70


def statistics(image: np.ndarray, roi: tuple[int, int, int, int]) -> dict[str, int | float]:
    x, y, width, height = roi
    hsv = cv2.cvtColor(image[y:y + height, x:x + width], cv2.COLOR_BGR2HSV)
    white = (hsv[:, :, 1] < WHITE_SAT_MAX) & (hsv[:, :, 2] > WHITE_VALUE_MIN)
    black = hsv[:, :, 2] < BLACK_VALUE_MAX
    total = int(width * height)
    white_pixels, black_pixels = int(white.sum()), int(black.sum())
    return {
        "roi_x": x, "roi_y": y, "roi_width": width, "roi_height": height,
        "total_pixels": total, "white_pixels": white_pixels, "black_pixels": black_pixels,
        "white_ratio": round(white_pixels / total, 6), "black_ratio": round(black_pixels / total, 6),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="框选完整球窝，标定白环/黑心像素数量。")
    parser.add_argument("--output", type=Path, default=Path("sucker_seek/socket_color_calibration.json"))
    parser.add_argument("--tolerance", type=float, default=.20, help="后续允许的白/黑像素比例误差，默认 ±20%%")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()
    if not 0 < args.tolerance < .8:
        parser.error("--tolerance 必须在 0 到 0.8 之间")
    try:
        import pyrealsense2 as rs
    except ImportError as error:
        raise SystemExit("未找到 pyrealsense2。") from error
    pipeline, config = rs.pipeline(), rs.config()
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, 30)
    pipeline.start(config)
    window = "Manual socket color calibration"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    try:
        while True:
            color = pipeline.wait_for_frames().get_color_frame()
            if not color:
                continue
            frame = np.asanyarray(color.get_data())
            shown = frame.copy()
            message = "Space冻结后拖动框选球窝；Enter确认；Q退出"
            cv2.putText(shown, message, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 0, 0), 3)
            cv2.putText(shown, message, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 1)
            cv2.imshow(window, shown)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord(" "), ord("f"), ord("F")):
                frozen = frame.copy()
                roi = cv2.selectROI("Drag to select complete socket, Enter confirm", frozen, fromCenter=False, showCrosshair=True)
                cv2.destroyWindow("Drag to select complete socket, Enter confirm")
                if roi[2] <= 0 or roi[3] <= 0:
                    continue
                data = statistics(frozen, tuple(int(value) for value in roi))
                data.update({
                    "calibrated_at": datetime.now(timezone.utc).isoformat(),
                    "tolerance": args.tolerance,
                    "white_saturation_max": WHITE_SAT_MAX,
                    "white_value_min": WHITE_VALUE_MIN,
                    "black_value_max": BLACK_VALUE_MAX,
                })
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                print(json.dumps(data, ensure_ascii=False, indent=2))
                print(f"[已保存] {args.output}")
                return 0
            if key in (27, ord("q"), ord("Q")):
                return 0
    finally:
        pipeline.stop(); cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
