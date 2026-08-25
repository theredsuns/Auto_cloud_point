#!/usr/bin/env python3
"""Prepare a two-class YOLO classification dataset from input/body and input/wing."""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# Without negative examples a classifier must label every empty scene as body
# or wing.  Background is therefore a required third class for live use.
CLASSES = ("background", "body", "wing")


def image_files(folder: Path) -> list[Path]:
    return sorted(path for path in folder.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Split separately supplied body/wing photos for YOLO classification."
    )
    parser.add_argument("--input", type=Path, default=Path("yolo_body_wing_classification/input"))
    parser.add_argument("--output", type=Path, default=Path("yolo_body_wing_classification/dataset"))
    parser.add_argument("--val-ratio", type=float, default=.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not .05 <= args.val_ratio < .5:
        parser.error("--val-ratio must be between 0.05 and 0.5")
    input_root, output = args.input.resolve(), args.output.resolve()
    groups = {name: image_files(input_root / name) for name in CLASSES}
    missing = [name for name, files in groups.items() if not files]
    if missing:
        raise SystemExit(
            "No images for: " + ", ".join(missing) + "\n"
            f"Put images in {input_root}/body and {input_root}/wing."
        )
    if output.exists():
        if not args.overwrite:
            raise SystemExit(f"Output already exists: {output}\nUse --overwrite to replace it.")
        shutil.rmtree(output)
    rng = random.Random(args.seed)
    totals = {}
    for class_name, files in groups.items():
        if len(files) < 10:
            print(f"[warning] {class_name} has only {len(files)} images; collect at least 100 varied photos.")
        rng.shuffle(files)
        val_count = max(1, round(len(files) * args.val_ratio))
        for split, subset in (("val", files[:val_count]), ("train", files[val_count:])):
            destination = output / split / class_name
            destination.mkdir(parents=True, exist_ok=True)
            for index, image_path in enumerate(subset):
                # A prefix prevents equal camera filenames from overwriting.
                name = f"{index:05d}_{image_path.name}"
                shutil.copy2(image_path, destination / name)
        totals[class_name] = len(files)
    print(f"[done] Dataset created: {output}")
    print("[done] " + ", ".join(f"{name}={count}" for name, count in totals.items()))
    print("[next] Run train_yolo_classify.sh. Keep photos of the same physical view together when possible.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
