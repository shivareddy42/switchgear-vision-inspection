"""Inspection and detection tables.

Column types stay in the SQLAlchemy portable subset so a SQL Server URL can
replace the local SQLite URL without a schema rewrite.
"""

from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Inspection(Base):
    __tablename__ = "inspections"

    inspection_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    image_name: Mapped[str] = mapped_column(String(512))
    model_version: Mapped[str] = mapped_column(String(128))
    overall_result: Mapped[str] = mapped_column(String(8))
    max_confidence: Mapped[float] = mapped_column(Float)
    inference_ms: Mapped[float] = mapped_column(Float)
    device: Mapped[str] = mapped_column(String(64))
    threshold_config: Mapped[str] = mapped_column(Text)
    notes: Mapped[str] = mapped_column(Text, default="")
    detections: Mapped[list["DetectionRow"]] = relationship(back_populates="inspection")


class DetectionRow(Base):
    __tablename__ = "detections"

    detection_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    inspection_id: Mapped[int] = mapped_column(ForeignKey("inspections.inspection_id"))
    class_name: Mapped[str] = mapped_column(String(64))
    confidence: Mapped[float] = mapped_column(Float)
    x1: Mapped[float] = mapped_column(Float)
    y1: Mapped[float] = mapped_column(Float)
    x2: Mapped[float] = mapped_column(Float)
    y2: Mapped[float] = mapped_column(Float)
    inspection: Mapped[Inspection] = relationship(back_populates="detections")
