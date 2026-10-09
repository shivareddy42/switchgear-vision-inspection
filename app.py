#!/usr/bin/env python3
"""FastAPI inspection service.

The detector is loaded once per process. Runtime configuration is explicit via
environment variables so the API and CLI can use the same checkpoint/device.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inspection.decision import load_thresholds
from inspection.inference import Inspector

DEFAULT_WEIGHTS = "outputs/training/yolov8_shop/weights/best.pt"
DEFAULT_THRESHOLDS = "configs/thresholds.yaml"

app = FastAPI(title="Switchgear vision inspection")
_inspector: Inspector | None = None


class DetectionModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    class_name: str = Field(alias="class")
    confidence: float
    bbox: list[float]


class PredictResponse(BaseModel):
    result: str
    model_version: str
    inference_ms: float
    detections: list[DetectionModel]
    mock: bool


def _runtime_weights() -> Path:
    configured = os.getenv("MODEL_WEIGHTS", DEFAULT_WEIGHTS)
    path = Path(configured)
    return path if path.is_absolute() else ROOT / path


def _runtime_device() -> str:
    return os.getenv("MODEL_DEVICE", "cpu")


def _thresholds_path() -> Path:
    configured = os.getenv("THRESHOLDS_PATH", DEFAULT_THRESHOLDS)
    path = Path(configured)
    return path if path.is_absolute() else ROOT / path


def get_inspector() -> Inspector:
    global _inspector
    if _inspector is None:
        weights = _runtime_weights()
        mock = not weights.exists()
        _inspector = Inspector(
            weights if weights.exists() else None,
            mock=mock,
            device=_runtime_device(),
        )
    return _inspector


def set_inspector(inspector: Inspector | None) -> None:
    """Testing hook; pass None to force runtime reinitialization."""
    global _inspector
    _inspector = inspector


@app.get("/health")
def health() -> dict:
    weights = _runtime_weights()
    return {
        "status": "ok",
        "weights": str(weights.relative_to(ROOT) if weights.is_relative_to(ROOT) else weights),
        "weights_present": weights.exists(),
        "device": _runtime_device(),
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)) -> dict:
    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="empty upload")

    image = _decode(payload)
    if image is None:
        raise HTTPException(status_code=400, detail="could not decode image")

    default, per_class, _status = load_thresholds(_thresholds_path())
    result = get_inspector().predict(image, default, per_class)
    return {
        "result": result["result"],
        "model_version": result["model_version"],
        "inference_ms": result["inference_ms"],
        "detections": result["detections"],
        "mock": result["mock"],
    }


def _decode(payload: bytes):
    import cv2

    array = np.frombuffer(payload, dtype=np.uint8)
    return cv2.imdecode(array, cv2.IMREAD_COLOR)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
