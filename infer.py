#!/usr/bin/env python3
"""Run inspection on an image, a directory, or a video.

``--demo`` / ``--mock`` draws an obvious MOCK overlay and does not load weights.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from db.database import get_engine, get_session, init_db, log_inspection
from inspection.decision import load_thresholds
from inspection.images import read_bgr
from inspection.inference import Inspector, save_prediction

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _inspect_image(inspector: Inspector, path: Path, default, per_class, output_dir: Path, log: bool, device: str, threshold_note: str):
    image = read_bgr(path)
    if image is None:
        print(f"unreadable {path}")
        return
    payload = inspector.predict(image, default, per_class)
    image_out, json_out = save_prediction(output_dir, path.stem, payload)
    print(
        f"{path.name} {payload['result']} mock={payload['mock']} "
        f"{payload['inference_ms']:.1f}ms detections={len(payload['detections'])} -> {image_out.name}"
    )
    if log:
        engine = get_engine()
        init_db(engine)
        session = get_session(engine)
        log_inspection(
            session,
            image_name=path.name,
            model_version=payload["model_version"],
            overall_result=payload["result"],
            max_confidence=payload["max_confidence"],
            inference_ms=payload["inference_ms"],
            device=device if not payload["mock"] else "mock",
            threshold_config=threshold_note,
            notes=payload["notes"],
            detections=payload["detections"],
        )
    return json_out


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect an image, directory, or video.")
    parser.add_argument("--image", default=None)
    parser.add_argument("--dir", default=None)
    parser.add_argument("--video", default=None)
    parser.add_argument("--webcam", action="store_true")
    parser.add_argument("--weights", default="outputs/training/yolov8_shop/weights/best.pt")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-log", action="store_true")
    parser.add_argument("--thresholds", default="configs/thresholds.yaml")
    args = parser.parse_args()
    mock = args.demo or args.mock
    weights = ROOT / args.weights
    if not mock and not weights.exists():
        print("No weights found. Re-run with --demo to exercise the pipeline. Mock output is labeled MOCK.")
        return 1
    default, per_class, status = load_thresholds(ROOT / args.thresholds)
    threshold_note = f"{args.thresholds} status={status} default={default}"
    inspector = Inspector(weights if weights.exists() else None, mock=mock, device=args.device)
    output_dir = ROOT / "outputs" / "predictions"
    targets: list[Path] = []
    if args.image:
        path = Path(args.image)
        targets.append(path if path.is_absolute() else ROOT / path)
    if args.dir:
        directory = Path(args.dir)
        if not directory.is_absolute():
            directory = ROOT / directory
        targets.extend(sorted(path for path in directory.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES))
    for path in targets:
        _inspect_image(inspector, path, default, per_class, output_dir, not args.no_log, args.device, threshold_note)
    if args.video or args.webcam:
        import cv2

        source = 0 if args.webcam else str(Path(args.video) if Path(args.video).is_absolute() else ROOT / args.video)
        capture = cv2.VideoCapture(source)
        if not capture.isOpened():
            print(f"could not open video source {source}")
            return 1 if args.video else 0
        ok, frame = capture.read()
        capture.release()
        if not ok:
            print("video source opened but no frame was read")
            return 1
        payload = inspector.predict(frame, default, per_class)
        stem = "webcam" if args.webcam else Path(str(args.video)).stem
        save_prediction(output_dir, stem, payload)
        print(f"{stem} {payload['result']} mock={payload['mock']}")
    if not targets and not args.video and not args.webcam:
        parser.error("pass --image, --dir, --video, or --webcam")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
