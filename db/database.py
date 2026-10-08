"""Engine and session helpers."""

from __future__ import annotations

import os
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from db.models import Base, DetectionRow, Inspection
from inspection.paths import repo_root


def default_url() -> str:
    override = os.environ.get("INSPECTION_DATABASE_URL")
    if override:
        return override
    path = repo_root() / "outputs" / "inspections.sqlite"
    path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{path}"


def get_engine(url: str | None = None):
    chosen = url or default_url()
    connect_args = {"check_same_thread": False} if chosen.startswith("sqlite") else {}
    return create_engine(chosen, connect_args=connect_args, future=True)


def init_db(engine) -> None:
    Base.metadata.create_all(engine)


def get_session(engine) -> Session:
    factory = sessionmaker(bind=engine, future=True)
    return factory()


def log_inspection(
    session: Session,
    *,
    image_name: str,
    model_version: str,
    overall_result: str,
    max_confidence: float,
    inference_ms: float,
    device: str,
    threshold_config: str,
    notes: str,
    detections: list[dict],
) -> Inspection:
    row = Inspection(
        image_name=image_name,
        model_version=model_version,
        overall_result=overall_result,
        max_confidence=max_confidence,
        inference_ms=inference_ms,
        device=device,
        threshold_config=threshold_config,
        notes=notes,
    )
    session.add(row)
    session.flush()
    for detection in detections:
        x1, y1, x2, y2 = detection["bbox"]
        session.add(
            DetectionRow(
                inspection_id=row.inspection_id,
                class_name=detection["class"],
                confidence=float(detection["confidence"]),
                x1=float(x1),
                y1=float(y1),
                x2=float(x2),
                y2=float(y2),
            )
        )
    session.commit()
    session.refresh(row)
    return row


def recent_inspections(session: Session, limit: int = 20) -> list[Inspection]:
    return list(session.query(Inspection).order_by(Inspection.inspection_id.desc()).limit(limit))
