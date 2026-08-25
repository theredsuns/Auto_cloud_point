#!/usr/bin/env python3
"""Interactively test the trained body/wing classifier on photos.

Place test images in test_images/ or pass an image/folder with --source.
Keys: A previous, B next, Q/Esc quit, S save annotated preview.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

# Importing this module configures the bundled Torch/SymPy/Ultralytics runtime.
import run_local_yolo  # noqa: F401
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = ROOT / "runs" / "body_wing_classifier-3" / "weights" / "best.pt"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_images(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    if source.is_dir():
        return sorted(path for path in source.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)
    raise FileNotFoundError(f"No such image or folder: {source}")


def predict(model: YOLO, image_path: Path):
    result = model(str(image_path), verbose=False)[0]
    top = int(result.probs.top1)
    confidence = float(result.probs.top1conf)
    names = result.names
    label = names[top] if isinstance(names, dict) else names[top]
    return str(label), confidence


def main():
    parser = argparse.ArgumentParser(description="Test the trained body/wing classification model.")
    parser.add_argument("--source", type=Path, default=ROOT / "test_images")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--no-gui", action="store_true", help="Print results without opening an image window.")
    args = parser.parse_args()
    weights = args.weights.resolve()
    if not weights.is_file():
        raise SystemExit(f"Model not found: {weights}")
    images = find_images(args.source.resolve())
    if not images:
        raise SystemExit(f"No image files in: {args.source.resolve()}")
    model = YOLO(str(weights))
    print(f"[info] model: {weights}")
    if args.no_gui:
        for path in images:
            label, confidence = predict(model, path)
            print(f"{path.name}: {label}  {confidence:.1%}")
        return

    index = 0
    output = ROOT / "test_results"
    output.mkdir(exist_ok=True)
    window = "Body/Wing test: A previous | B next | S save | Q quit"
    while True:
        image_path = images[index]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"[skip] unreadable: {image_path}")
            index = (index + 1) % len(images)
            continue
        label, confidence = predict(model, image_path)
        color = (0, 220, 0) if label == "body" else (0, 165, 255)
        text = f"{label.upper()}  {confidence:.1%}"
        cv2.rectangle(image, (10, 10), (min(image.shape[1] - 10, 380), 64), (0, 0, 0), -1)
        cv2.putText(image, text, (20, 48), cv2.FONT_HERSHEY_SIMPLEX, .9, color, 2, cv2.LINE_AA)
        cv2.putText(image, f"{index + 1}/{len(images)}  {image_path.name}", (15, image.shape[0] - 18), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(window, image)
        key = cv2.waitKey(0) & 0xFF
        if key in (27, ord("q"), ord("Q")):
            break
        if key in (ord("a"), ord("A")):
            index = (index - 1) % len(images)
        elif key in (ord("b"), ord("B")):
            index = (index + 1) % len(images)
        elif key in (ord("s"), ord("S")):
            saved = output / f"{image_path.stem}_{label}_{confidence:.3f}.jpg"
            cv2.imwrite(str(saved), image)
            print(f"[saved] {saved}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
