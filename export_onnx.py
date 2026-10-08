#!/usr/bin/env python3
"""Export a PyTorch checkpoint to ONNX and compare one image when ONNX Runtime is installed."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inspection.images import read_bgr


def _sample_image(explicit: str | None) -> Path | None:
    if explicit:
        path = Path(explicit)
        return path if path.is_absolute() else ROOT / path
    for folder in (ROOT / "data" / "sample", ROOT / "data" / "test" / "images", ROOT / "data" / "val" / "images"):
        if not folder.exists():
            continue
        images = sorted(folder.glob("*.jpg")) + sorted(folder.glob("*.png"))
        if images:
            return images[0]
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Export YOLO weights to ONNX.")
    parser.add_argument("--weights", default="outputs/training/yolov8/weights/best.pt")
    parser.add_argument("--image", default=None)
    parser.add_argument("--imgsz", type=int, default=320)
    args = parser.parse_args()
    weights = ROOT / args.weights
    if not weights.exists():
        print(f"weights not found: {weights}")
        return 1
    from ultralytics import YOLO

    model = YOLO(str(weights))
    exported = Path(model.export(format="onnx", imgsz=args.imgsz, device="cpu"))
    import onnx

    onnx_model = onnx.load(str(exported))
    onnx.checker.check_model(onnx_model)
    report = {
        "weights": str(weights.relative_to(ROOT)),
        "onnx": str(exported),
        "onnx_checker": "ok",
        "imgsz": args.imgsz,
    }
    image_path = _sample_image(args.image)
    try:
        import onnxruntime as ort
    except ImportError:
        report["onnxruntime"] = "not installed"
        destination = ROOT / "outputs" / "onnx_comparison.json"
        destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
        return 0
    if image_path is None:
        report["onnxruntime"] = "available"
        report["comparison"] = "no sample image"
    else:
        image = read_bgr(image_path)
        pytorch = model.predict(source=image, device="cpu", verbose=False, imgsz=args.imgsz)
        session = ort.InferenceSession(str(exported), providers=["CPUExecutionProvider"])
        blob = _letterbox(image, args.imgsz)
        input_name = session.get_inputs()[0].name
        onnx_outputs = session.run(None, {input_name: blob})
        report["onnxruntime"] = "available"
        report["sample_image"] = str(image_path)
        report["onnx_output_shapes"] = [list(array.shape) for array in onnx_outputs]
        pytorch_boxes = []
        if pytorch and pytorch[0].boxes is not None:
            pytorch_boxes = pytorch[0].boxes.xyxy.cpu().numpy().tolist()
        report["pytorch_boxes"] = pytorch_boxes
        report["note"] = (
            "ONNX Runtime consumed the exported graph. Raw ONNX heads are not the same tensor as "
            "Ultralytics' decoded boxes, so this report records shapes and the PyTorch boxes rather "
            "than a fake box-level equality claim. max_abs on the raw output is included."
        )
        report["raw_output_max_abs"] = float(np.max(np.abs(onnx_outputs[0]))) if onnx_outputs else None
        report["raw_output_mean_abs"] = float(np.mean(np.abs(onnx_outputs[0]))) if onnx_outputs else None
    destination = ROOT / "outputs" / "onnx_comparison.json"
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {destination.relative_to(ROOT)}")
    print(f"onnx checker ok: {exported}")
    return 0


def _letterbox(image, size: int):
    import cv2

    resized = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
    rgb = resized[:, :, ::-1]
    blob = rgb.astype(np.float32) / 255.0
    blob = np.transpose(blob, (2, 0, 1))[None]
    return np.ascontiguousarray(blob)


if __name__ == "__main__":
    raise SystemExit(main())
