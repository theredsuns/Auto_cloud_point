#!/usr/bin/env python3
"""YOLO 圆检测，并输出每张图片中各圆心的两两距离。"""
from __future__ import annotations

import argparse
import csv
import itertools
import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib"))
sys.path.insert(0, str(ROOT / "yolo_body_wing_classification"))
import run_local_yolo  # noqa: F401,E402
from ultralytics import YOLO  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="检测 circle 并计算检测框中心的距离。")
    parser.add_argument("source", type=Path, help="图片文件或图片目录")
    parser.add_argument("--weights", type=Path, default=Path(__file__).resolve().parent / "runs/circle/weights/best.pt")
    parser.add_argument("--confidence", type=float, default=.5)
    parser.add_argument("--mm-per-pixel", type=float, help="标定后的毫米/像素；提供后额外输出毫米距离")
    parser.add_argument("--output", type=Path, default=Path("sucker_seek/results"))
    args = parser.parse_args()
    if not args.weights.is_file():
        raise SystemExit(f"找不到权重：{args.weights}；请先训练或通过 --weights 指定。")
    images = [args.source] if args.source.is_file() else sorted(p for p in args.source.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    if not images:
        raise SystemExit("没有可处理的图片")
    args.output.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.weights))
    rows: list[dict[str, object]] = []
    for image_path in images:
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        result = model(image, conf=args.confidence, verbose=False)[0]
        centers = []
        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
                centers.append(((x1+x2)/2, (y1+y2)/2, float(box.conf[0])))
        centers.sort(key=lambda item: (item[0], item[1]))
        shown = image.copy()
        for number, (x, y, confidence) in enumerate(centers, 1):
            point = (round(x), round(y))
            cv2.drawMarker(shown, point, (0, 0, 255), cv2.MARKER_CROSS, 18, 2)
            cv2.putText(shown, f"C{number} {confidence:.0%}", (point[0]+8, point[1]-8), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 220, 0), 2)
        for (i, first), (j, second) in itertools.combinations(enumerate(centers, 1), 2):
            distance_px = float(np.hypot(first[0]-second[0], first[1]-second[1]))
            row: dict[str, object] = {"image": image_path.name, "circle_a": i, "circle_b": j, "distance_px": round(distance_px, 3)}
            if args.mm_per_pixel is not None:
                row["distance_mm"] = round(distance_px * args.mm_per_pixel, 3)
            rows.append(row)
            midpoint = (round((first[0]+second[0])/2), round((first[1]+second[1])/2))
            cv2.line(shown, (round(first[0]), round(first[1])), (round(second[0]), round(second[1])), (0, 255, 255), 2)
            cv2.putText(shown, f"{distance_px:.1f}px", midpoint, cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 255, 255), 2)
        cv2.imwrite(str(args.output / image_path.name), shown)
    fields = ["image", "circle_a", "circle_b", "distance_px"] + (["distance_mm"] if args.mm_per_pixel is not None else [])
    with (args.output / "distances.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    print(f"[完成] 标注图：{args.output}；距离表：{args.output / 'distances.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
