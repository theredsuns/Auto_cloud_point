#!/usr/bin/env python3
"""Create/review full masks from body.pt detections and SAM2.

Enter: save current mask and next image.  P: redraw a box, then rerun SAM2.
Q/Esc: stop.  The masks are written to object_pointcloud_input/masks/ and are
ready for run_pointcloud.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "object_pointcloud_input"
IMAGE_DIR = INPUT / "images"
MASK_DIR = INPUT / "masks"
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT / "yolo_execise" / "yolo_body_wing_classification"))
import run_local_yolo  # noqa: E402,F401
from ultralytics import YOLO  # noqa: E402
sys.path.insert(0, str(PROJECT / "yolo_execise" / "zed_sam_tools"))
from zed_yolo_sam2_live import load_sam2  # noqa: E402

MODEL_PATH = PROJECT / "yolo_execise" / "body.pt"
SAM_CONFIG = PROJECT / "yolo_execise" / "libraries" / "sam2" / "sam2" / "configs" / "sam2.1" / "sam2.1_hiera_t.yaml"
SAM_CHECKPOINT = PROJECT / "yolo_execise" / "libraries" / "sam2" / "checkpoints" / "sam2.1_hiera_tiny.pt"
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def sam_mask(predictor, rgb, box):
    predictor.set_image(rgb)
    masks, _, _ = predictor.predict(box=np.asarray(box, np.float32)[None, :], multimask_output=False)
    return np.asarray(masks[0], dtype=bool)


def overlay(image, mask, title):
    shown = image.copy()
    if mask is not None and np.any(mask):
        color = np.zeros_like(shown); color[:, :, 1] = 220
        shown[mask] = cv2.addWeighted(shown[mask], .45, color[mask], .55, 0)
    cv2.rectangle(shown, (0, 0), (shown.shape[1], 76), (0, 0, 0), -1)
    cv2.putText(shown, title, (15, 29), cv2.FONT_HERSHEY_SIMPLEX, .62, (255, 255, 255), 2)
    cv2.putText(shown, "Enter: save + next | P: redraw SAM box | Q: quit", (15, 58), cv2.FONT_HERSHEY_SIMPLEX, .52, (0, 255, 255), 2)
    return shown


def main():
    if not IMAGE_DIR.is_dir() or not MODEL_PATH.is_file():
        raise SystemExit(f"Need {IMAGE_DIR} and {MODEL_PATH}")
    images = sorted(path for path in IMAGE_DIR.iterdir() if path.suffix.lower() in EXTENSIONS)
    if not images: raise SystemExit("No images found.")
    MASK_DIR.mkdir(exist_ok=True)
    detector, predictor = YOLO(str(MODEL_PATH)), load_sam2(SAM_CONFIG, SAM_CHECKPOINT)
    for index, image_path in enumerate(images, 1):
        image = cv2.imread(str(image_path)); rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = detector(image, verbose=False)[0]
        mask = None
        if result.boxes is not None and len(result.boxes):
            best = int(np.argmax(result.boxes.conf.cpu().numpy()))
            box = result.boxes.xyxy[best].cpu().numpy()
            mask = sam_mask(predictor, rgb, box)
            status = "YOLO + SAM2"
        else:
            status = "NO YOLO BODY: press P to draw SAM box"
        while True:
            shown = overlay(image, mask, f"{index}/{len(images)} {image_path.name} | {status}")
            cv2.imshow("Auto mask review", shown)
            key = cv2.waitKey(0) & 0xFF
            if key in (27, ord('q'), ord('Q')):
                cv2.destroyAllWindows(); return
            if key in (13, 10):
                if mask is not None and np.any(mask):
                    cv2.imwrite(str(MASK_DIR / f"{image_path.stem}.png"), mask.astype(np.uint8) * 255)
                    print(f"[saved] {image_path.name}")
                else: print(f"[skip] {image_path.name}: no mask")
                break
            if key in (ord('p'), ord('P')):
                x, y, w, h = cv2.selectROI("Draw BODY box for SAM2", image, showCrosshair=True, fromCenter=False)
                cv2.destroyWindow("Draw BODY box for SAM2")
                if w > 8 and h > 8:
                    mask = sam_mask(predictor, rgb, (x, y, x + w, y + h)); status = "SAM2 corrected manually"
    cv2.destroyAllWindows()


if __name__ == "__main__": main()
