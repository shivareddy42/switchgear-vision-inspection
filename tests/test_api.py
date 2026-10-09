"""API schema with a mocked inspector. No GPU and no weight file."""

import io

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

import app as app_module
from inspection.decision import Detection
from inspection.inference import MOCK_MODEL_VERSION


class _FakeInspector:
    model_version = MOCK_MODEL_VERSION
    mock = True

    def predict(self, image, default, per_class):
        detection = Detection("weld_porosity", 0.87, (10.0, 12.0, 40.0, 50.0))
        return {
            "result": "FAIL",
            "model_version": MOCK_MODEL_VERSION,
            "inference_ms": 1.5,
            "detections": [detection.as_dict()],
            "mock": True,
            "notes": "MOCK",
            "max_confidence": 0.87,
            "annotated": image,
            "decision": {"result": "FAIL"},
        }


@pytest.fixture(autouse=True)
def reset_inspector():
    app_module.set_inspector(None)
    yield
    app_module.set_inspector(None)


def _jpeg_bytes() -> bytes:
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[:] = (30, 60, 90)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    return encoded.tobytes()


def test_health():
    client = TestClient(app_module.app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_schema_uses_mocked_model():
    app_module.set_inspector(_FakeInspector())
    client = TestClient(app_module.app)
    response = client.post("/predict", files={"file": ("panel.jpg", io.BytesIO(_jpeg_bytes()), "image/jpeg")})
    assert response.status_code == 200
    body = response.json()
    assert body["result"] == "FAIL"
    assert body["model_version"] == MOCK_MODEL_VERSION
    assert isinstance(body["inference_ms"], float)
    assert body["mock"] is True
    assert body["detections"][0]["class"] == "weld_porosity"
    assert body["detections"][0]["confidence"] == 0.87
    assert body["detections"][0]["bbox"] == [10.0, 12.0, 40.0, 50.0]


def test_missing_weights_do_not_silently_enable_mock(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_WEIGHTS", str(tmp_path / "missing.pt"))
    monkeypatch.delenv("MODEL_MOCK", raising=False)
    response = TestClient(app_module.app).post("/predict", files={"file": ("panel.jpg", _jpeg_bytes(), "image/jpeg")})
    assert response.status_code == 503


def test_mock_requires_explicit_configuration(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_WEIGHTS", str(tmp_path / "missing.pt"))
    monkeypatch.setenv("MODEL_MOCK", "1")
    response = TestClient(app_module.app).post("/predict", files={"file": ("panel.jpg", _jpeg_bytes(), "image/jpeg")})
    assert response.status_code == 200
    assert response.json()["mock"] is True


def test_invalid_upload_is_rejected():
    response = TestClient(app_module.app).post("/predict", files={"file": ("bad.jpg", b"invalid", "image/jpeg")})
    assert response.status_code == 400
