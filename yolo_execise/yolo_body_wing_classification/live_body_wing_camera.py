#!/usr/bin/env python3
"""Run real-time body/wing classification from a USB/web camera."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

import run_local_yolo  # Configures the bundled local YOLO runtime.
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = ROOT / "runs" / "body_wing_classifier-3" / "weights" / "best.pt"


def main():
    parser = argparse.ArgumentParser(description="Real-time body/wing camera classifier.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index; try 1 if 0 is not the desired camera.")
    parser.add_argument("--zed", action="store_true", help="Read the ZED left camera through the ZED SDK.")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--min-confidence", type=float, default=.55)
    args = parser.parse_args()
    weights = args.weights.resolve()
    if not weights.is_file():
        raise SystemExit(f"Model not found: {weights}")
    camera = None
    zed = None
    zed_image = None
    if args.zed:
        try:
            import pyzed.sl as sl
        except ImportError as error:
            raise SystemExit(f"ZED SDK Python module is unavailable: {error}")
        zed = sl.Camera()
        parameters = sl.InitParameters()
        parameters.camera_resolution = sl.RESOLUTION.HD720
        parameters.camera_fps = 30
        if zed.open(parameters) != sl.ERROR_CODE.SUCCESS:
            raise SystemExit("Cannot open ZED. Check its USB connection and close other ZED programs.")
        zed_image = sl.Mat()
    else:
        camera = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
        if not camera.isOpened():
            raise SystemExit(
                f"Cannot open USB camera {args.camera}. No /dev/video device is available; "
                "use --zed for the ZED camera."
            )
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    model = YOLO(str(weights))
    window = "Body/Wing live: Q quit"
    print(f"[info] {'ZED left camera' if args.zed else f'USB camera {args.camera}'} open. Press Q to quit.")
    try:
        while True:
            if zed is not None:
                ok = zed.grab() == sl.ERROR_CODE.SUCCESS
                if ok:
                    zed.retrieve_image(zed_image, sl.VIEW.LEFT)
                    frame = cv2.cvtColor(zed_image.get_data(), cv2.COLOR_BGRA2BGR)
                else:
                    frame = None
            else:
                ok, frame = camera.read()
            if not ok:
                print("[warning] camera frame read failed")
                break
            result = model(frame, verbose=False)[0]
            top = int(result.probs.top1)
            confidence = float(result.probs.top1conf)
            label = result.names[top] if isinstance(result.names, dict) else result.names[top]
            # A model trained with the required background class can reject an
            # empty/unrelated scene.  Its displayed detection confidence is
            # deliberately 0% in that case.
            reliable = confidence >= args.min_confidence and label != "background"
            if reliable:
                color = (0, 220, 0) if label == "body" else (0, 165, 255)
                text = f"{str(label).upper()} {confidence:.1%}"
                # Classification has no learned coordinates, so no fake
                # full-screen object box is drawn here.
            else:
                color = (0, 0, 255)
                text = "NO DETECTION 0%"
            cv2.rectangle(frame, (12, 12), (min(frame.shape[1] - 10, 430), 72), (0, 0, 0), -1)
            cv2.putText(frame, text, (24, 53), cv2.FONT_HERSHEY_SIMPLEX, .95, color, 2, cv2.LINE_AA)
            cv2.putText(frame, "Q: quit", (18, frame.shape[0] - 18), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow(window, frame)
            if (cv2.waitKey(1) & 0xFF) in (27, ord("q"), ord("Q")):
                break
    finally:
        if camera is not None:
            camera.release()
        if zed is not None:
            zed.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
