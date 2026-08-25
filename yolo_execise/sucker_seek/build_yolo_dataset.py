#!/usr/bin/env python3
"""把圆心/半径标注导出为 YOLO 检测数据集（单类别 circle）。"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import cv2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("images", type=Path)
    parser.add_argument("annotations", type=Path)
    parser.add_argument("--output", type=Path, default=Path("sucker_seek/dataset"))
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not 0.05 <= args.val_ratio < 0.5:
        parser.error("--val-ratio 必须在 0.05 到 0.5 之间")
    annotations = json.loads(args.annotations.read_text(encoding="utf-8"))
    samples = [(args.images / name, circles) for name, circles in annotations.items() if circles and (args.images / name).is_file()]
    if len(samples) < 10:
        raise SystemExit(f"仅有 {len(samples)} 张含圆的已标注图。请至少标注 10 张，建议 100+ 张再训练。")
    output = args.output.resolve()
    if output.exists():
        if not args.overwrite:
            raise SystemExit(f"输出目录已存在：{output}；请加 --overwrite 或换目录")
        shutil.rmtree(output)
    random.Random(args.seed).shuffle(samples)
    validation = set(range(max(1, round(len(samples) * args.val_ratio))))
    for split in ("train", "val"):
        (output / "images" / split).mkdir(parents=True)
        (output / "labels" / split).mkdir(parents=True)
    for index, (image_path, circles) in enumerate(samples):
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"[跳过] 无法读取：{image_path}")
            continue
        height, width = image.shape[:2]
        labels = []
        for circle in circles:
            cx, cy, radius = float(circle["cx"]), float(circle["cy"]), float(circle["r"])
            x1, y1, x2, y2 = max(0., cx-radius), max(0., cy-radius), min(float(width), cx+radius), min(float(height), cy+radius)
            if x2 <= x1 or y2 <= y1:
                continue
            labels.append(f"0 {(x1+x2)/2/width:.6f} {(y1+y2)/2/height:.6f} {(x2-x1)/width:.6f} {(y2-y1)/height:.6f}")
        if not labels:
            continue
        split = "val" if index in validation else "train"
        shutil.copy2(image_path, output / "images" / split / image_path.name)
        (output / "labels" / split / f"{image_path.stem}.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")
    (output / "data.yaml").write_text(f"path: {output}\ntrain: images/train\nval: images/val\nnames:\n  0: circle\n", encoding="utf-8")
    print(f"[完成] 数据集：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
