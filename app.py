#!/usr/bin/env python3
"""FastAPI inspection service. The model is loaded once, not per request."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inspection.decision import load_thresholds
from inspection.inference import Inspector

app = FastAPI(title="Switchgear vision inspection")
_inspector: Inspector | None = None


class DetectionModel(BaseModel):
    class_name: str
    confidence: float
    bbox: list[float]


class PredictResponse(BaseModel):
    result: str
    model_version: str
    inference_ms: float
    detections: list[dict]
    mock: bool


def get_inspector() -> Inspector:
    global _inspector
    if _inspector is None:
        weights = ROOT / "outputs" / "training" / "yolov8" / "weights" / "best.pt"
        mock = not weights.exists()
        _inspector = Inspector(weights if weights.exists() else None, mock=mock, device="cpu")
    return _inspector


def set_inspector(inspector: Inspector) -> None:
    global _inspector
    _inspector = inspector


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)) -> dict:
    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="empty upload")
    image = _decode(payload)
    if image is None:
        raise HTTPException(status_code=400, detail="could not decode image")
    default, per_class, _status = load_thresholds(ROOT / "configs" / "thresholds.yaml")
    result = get_inspector().predict(image, default, per_class)
    # The documented response shape. mock is extra so a demo cannot be mistaken for inference.
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
