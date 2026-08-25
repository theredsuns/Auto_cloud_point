#!/usr/bin/env python3
"""Recover body + wing masks from an existing saved ZED capture session."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from zed_yolo_sam2_icp_reconstruct import select_target_parts
from zed_yolo_sam2_live import ROOT, load_sam2, load_yolo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.10)
    args = parser.parse_args()
    root = args.input.resolve()
    output_masks = root / "masks_two_parts"
    output_previews = root / "preview_two_parts"
    output_masks.mkdir(exist_ok=True)
    output_previews.mkdir(exist_ok=True)

    yolo = load_yolo(ROOT / "best.pt")
    sam2 = load_sam2(
        ROOT / "sam2/sam2/configs/sam2.1/sam2.1_hiera_t.yaml",
        ROOT / "sam2/checkpoints/sam2.1_hiera_tiny.pt",
    )
    images = sorted((root / "images").glob("*.jpg"))
    recovered = retained = 0
    for progress, image_path in enumerate(images, 1):
        capture_id = image_path.stem
        bgr = cv2.imread(str(image_path))
        old_mask = cv2.imread(
            str(root / "masks" / f"{capture_id}.png"), cv2.IMREAD_GRAYSCALE
        )
        result = yolo.predict(bgr, conf=args.confidence, verbose=False)[0]
        selected = select_target_parts(
            result,
            sam2,
            cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
            "Wing and Body",
        )
        if selected is None:
            combined = old_mask > 0
            retained += 1
            mode = "old"
        else:
            _box, _confidence, _label, new_mask = selected
            # Never remove a part that was present in the originally saved mask.
            combined = (old_mask > 0) | new_mask
            added = int(np.count_nonzero(combined)) - int(np.count_nonzero(old_mask))
            if added >= 300:
                recovered += 1
                mode = f"+{added} px"
            else:
                retained += 1
                mode = "unchanged"
        mask = combined.astype(np.uint8) * 255
        preview = bgr.copy()
        preview[combined] = (
            preview[combined] * 0.5
            + np.array((0, 255, 0), dtype=np.float32) * 0.5
        ).astype(np.uint8)
        cv2.imwrite(str(output_masks / f"{capture_id}.png"), mask)
        cv2.imwrite(str(output_previews / f"{capture_id}.jpg"), preview)
        print(f"[{progress}/{len(images)}] {capture_id}: {mode}", flush=True)
    print(
        f"Done: expanded={recovered}, retained={retained}; masks={output_masks}",
        flush=True,
    )


if __name__ == "__main__":
    main()
