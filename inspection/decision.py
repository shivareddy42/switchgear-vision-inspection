"""PASS/FAIL from defect detections and confidence thresholds.

A detection triggers FAIL only when its confidence is strictly greater than
the threshold for its class. Equality does not exceed the threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import yaml

from inspection.classes import CLASS_NAMES


@dataclass(frozen=True)
class Detection:
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]

    def as_dict(self) -> dict:
        return {
            "class": self.class_name,
            "confidence": round(float(self.confidence), 6),
            "bbox": [round(float(value), 2) for value in self.bbox],
        }


def load_thresholds(path) -> tuple[float, dict[str, float], str]:
    payload = yaml.safe_load(Path_read(path))
    default = float(payload.get("default_confidence_threshold", 0.50))
    per_class = {str(key): float(value) for key, value in (payload.get("per_class") or {}).items()}
    status = str(payload.get("status", "unspecified"))
    unknown = sorted(set(per_class) - set(CLASS_NAMES))
    if unknown:
        raise ValueError(f"unknown classes in thresholds: {unknown}")
    if not all(math.isfinite(value) and 0 <= value <= 1 for value in [default, *per_class.values()]):
        raise ValueError("confidence thresholds must be finite values in 0..1")
    return default, per_class, status


def Path_read(path) -> str:
    from pathlib import Path

    return Path(path).read_text(encoding="utf-8")


def threshold_for(class_name: str, default: float, per_class: dict[str, float]) -> float:
    return float(per_class.get(class_name, default))


def decide(
    detections: list[Detection],
    default: float = 0.50,
    per_class: dict[str, float] | None = None,
) -> dict:
    per_class = per_class or {}
    failing: list[Detection] = []
    kept: list[dict] = []
    max_confidence = 0.0
    for detection in detections:
        max_confidence = max(max_confidence, float(detection.confidence))
        limit = threshold_for(detection.class_name, default, per_class)
        triggered = float(detection.confidence) > limit
        row = detection.as_dict()
        row["threshold"] = limit
        row["triggered_fail"] = triggered
        kept.append(row)
        if triggered:
            failing.append(detection)
    result = "FAIL" if failing else "PASS"
    return {
        "result": result,
        "max_confidence": round(max_confidence, 6),
        "failing_classes": sorted({item.class_name for item in failing}),
        "detections": kept,
    }
