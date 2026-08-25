#!/usr/bin/env python3
"""Browse generated point-cloud preview images with A/B keys."""

import argparse
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parent


def latest_run() -> Path:
    runs = [
        path for path in ROOT.iterdir()
        if path.is_dir() and any(path.glob("*_preview.png"))
    ]
    if not runs:
        raise SystemExit("No point-cloud preview images found in object_pointcloud_output.")
    return max(runs, key=lambda path: path.stat().st_mtime)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Browse point-cloud preview PNGs: A=previous, B=next."
    )
    parser.add_argument(
        "--input", type=Path,
        help="output run folder; defaults to the latest run with preview images",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    folder = (args.input or latest_run()).resolve()
    previews = sorted(folder.glob("*_preview.png"))
    if not previews:
        raise SystemExit(f"No *_preview.png files in {folder}")

    index = 0
    window = "Point cloud previews | A=previous, B=next, Q/Esc=quit"
    while True:
        path = previews[index]
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise SystemExit(f"Could not read {path}")
        label = f"[{index + 1}/{len(previews)}] {path.name}"
        cv2.putText(
            image, label, (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.72,
            (0, 0, 0), 4, cv2.LINE_AA,
        )
        cv2.putText(
            image, label, (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.72,
            (255, 255, 255), 2, cv2.LINE_AA,
        )
        cv2.imshow(window, image)
        key = cv2.waitKey(0) & 0xFF
        if key in (27, ord("q"), ord("Q")):
            break
        if key in (ord("a"), ord("A")):
            index = max(0, index - 1)
        elif key in (ord("b"), ord("B")):
            index = min(len(previews) - 1, index + 1)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
