"""Real YOLOv8 inference and an explicitly labeled mock mode."""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from inspection.classes import CLASS_NAMES
from inspection.decision import Detection, decide
from inspection.images import write_bgr
from inspection.labels import yolo_to_xyxy

MOCK_MODEL_VERSION = "mock-demo"
MOCK_NOTE = "MOCK detection. This is not model inference."

def draw_detections(image: np.ndarray, detections: list[Detection], mock: bool) -> np.ndarray:
    canvas = image.copy()
    for detection in detections:
        x1, y1, x2, y2 = [int(round(value)) for value in detection.bbox]
        color = (0, 0, 255) if mock else (0, 180, 0)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        prefix = "MOCK " if mock else ""
        text = f"{prefix}{detection.class_name} {detection.confidence:.2f}"
        cv2.putText(
            canvas,
            text,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )
    if mock:
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 36), (0, 0, 180), -1)
        cv2.putText(
            canvas,
            "MOCK - NOT MODEL INFERENCE",
            (8, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return canvas


class Inspector:
    def __init__(self, weights: Path | None, mock: bool, device: str = "cpu"):
        self.weights = weights
        self.mock = mock
        self.device = device
        self.model = None
        self.model_version = MOCK_MODEL_VERSION if mock else "unloaded"
        if not mock:
            self._load()

    def _load(self) -> None:
        if self.weights is None or not Path(self.weights).exists():
            raise FileNotFoundError(f"weights not found: {self.weights}")

        from ultralytics import YOLO

        weights = Path(self.weights)
        self.model = YOLO(str(weights))
        run_name = weights.parent.parent.name if weights.parent.name == "weights" else ""
        self.model_version = f"{run_name}:{weights.stem}" if run_name else weights.stem

    def predict(self, image: np.ndarray, default: float, per_class: dict[str, float]) -> dict:
        started = time.perf_counter()
        if self.mock or self.model is None:
            detections = _mock_detections(image)
            mock = True
            model_version = MOCK_MODEL_VERSION
        else:
            operating_floor = min([float(default), *[float(v) for v in per_class.values()]])
            detections = self._yolo(image, conf=max(0.001, operating_floor))
            mock = False
            model_version = self.model_version

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        decision = decide(detections, default=default, per_class=per_class)
        annotated = draw_detections(image, detections, mock=mock)

        return {
            "result": decision["result"],
            "model_version": model_version,
            "inference_ms": round(elapsed_ms, 3),
            "detections": [item.as_dict() for item in detections],
            "mock": mock,
            "notes": MOCK_NOTE if mock else "",
            "max_confidence": decision["max_confidence"],
            "annotated": annotated,
            "decision": decision,
        }

    def _yolo(self, image: np.ndarray, conf: float) -> list[Detection]:
        # Ask YOLO for every candidate that could possibly cross an operating
        # threshold. This avoids Ultralytics' default conf=0.25 becoming a hidden
        # policy while also avoiding thousands of near-zero boxes in output.
        results = self.model.predict(
            source=image,
            device=self.device,
            conf=float(conf),
            verbose=False,
        )
        detections: list[Detection] = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                class_id = int(box.cls.item())
                name = CLASS_NAMES[class_id] if 0 <= class_id < len(CLASS_NAMES) else str(class_id)
                xyxy = box.xyxy[0].tolist()
                detections.append(
                    Detection(
                        class_name=name,
                        confidence=float(box.conf.item()),
                        bbox=(
                            float(xyxy[0]),
                            float(xyxy[1]),
                            float(xyxy[2]),
                            float(xyxy[3]),
                        ),
                    )
                )
        return detections


def _mock_detections(image: np.ndarray) -> list[Detection]:
    """A fixed, visibly fake box so the demo runs before weights exist."""
    height, width = image.shape[:2]
    cx, cy, bw, bh = 0.50, 0.50, 0.30, 0.20
    x1, y1, x2, y2 = yolo_to_xyxy(cx, cy, bw, bh, width, height)
    return [
        Detection(
            class_name="corrosion",
            confidence=0.90,
            bbox=(x1, y1, x2, y2),
        )
    ]


def save_prediction(output_dir: Path, stem: str, payload: dict) -> tuple[Path, Path]:
    import json

    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / f"{stem}.jpg"
    json_path = output_dir / f"{stem}.json"
    write_bgr(image_path, payload["annotated"])
    public = {
        key: value
        for key, value in payload.items()
        if key not in {"annotated", "decision"}
    }
    json_path.write_text(json.dumps(public, indent=2) + "\n", encoding="utf-8")
    return image_path, json_path
