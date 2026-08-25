#!/usr/bin/env python3
"""交互式标注圆心与半径，结果保存为 circle_annotations.json。

操作：每个圆依次点击“圆心”和“圆周上一点”；U 撤销、S 保存、N 下一张、Q 退出。
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def load_annotations(path: Path) -> dict[str, list[dict[str, float]]]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def main() -> int:
    parser = argparse.ArgumentParser(description="用两次鼠标点击标注圆：圆心、圆周。")
    parser.add_argument("images", type=Path, help="待标注图片文件夹")
    parser.add_argument("--output", type=Path, default=Path("sucker_seek/circle_annotations.json"))
    args = parser.parse_args()
    image_paths = [p for p in sorted(args.images.iterdir()) if p.suffix.lower() in IMAGE_EXTENSIONS]
    if not image_paths:
        raise SystemExit(f"未在 {args.images} 找到图片")
    annotations = load_annotations(args.output)
    pending_center: tuple[int, int] | None = None

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"[跳过] 无法读取：{image_path}")
            continue
        circles = annotations.setdefault(image_path.name, [])

        def redraw() -> None:
            canvas = image.copy()
            for index, circle in enumerate(circles, start=1):
                center = (round(circle["cx"]), round(circle["cy"]))
                radius = round(circle["r"])
                cv2.circle(canvas, center, radius, (0, 220, 0), 2)
                cv2.drawMarker(canvas, center, (0, 0, 255), cv2.MARKER_CROSS, 14, 2)
                cv2.putText(canvas, str(index), (center[0] + 8, center[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 220, 0), 2)
            message = "中心后点圆周 | U撤销 S保存 N下一张 Q退出"
            if pending_center:
                cv2.drawMarker(canvas, pending_center, (0, 255, 255), cv2.MARKER_CROSS, 16, 2)
                message = "请点击同一圆的圆周"
            cv2.putText(canvas, message, (15, 32), cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 0, 0), 3)
            cv2.putText(canvas, message, (15, 32), cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 1)
            cv2.imshow("Circle annotation", canvas)

        def mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
            nonlocal pending_center
            if event != cv2.EVENT_LBUTTONDOWN:
                return
            if pending_center is None:
                pending_center = (x, y)
            else:
                radius = math.dist(pending_center, (x, y))
                if radius >= 2:
                    circles.append({"cx": pending_center[0], "cy": pending_center[1], "r": round(radius, 2)})
                pending_center = None
            redraw()

        cv2.namedWindow("Circle annotation", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Circle annotation", mouse)
        redraw()
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key in (ord("u"), ord("U")):
                if pending_center is not None:
                    pending_center = None
                elif circles:
                    circles.pop()
                redraw()
            elif key in (ord("s"), ord("S")):
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(annotations, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"[已保存] {args.output}")
                redraw()
            elif key in (ord("n"), ord("N"), 13, 32):
                break
            elif key in (ord("q"), ord("Q"), 27):
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(annotations, ensure_ascii=False, indent=2), encoding="utf-8")
                cv2.destroyAllWindows()
                return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(annotations, ensure_ascii=False, indent=2), encoding="utf-8")
    cv2.destroyAllWindows()
    print(f"[完成] 已保存 {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
