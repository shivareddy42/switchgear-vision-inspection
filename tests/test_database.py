"""SQLite inspection log."""

from db.database import get_engine, get_session, init_db, log_inspection, recent_inspections


def test_insert_inspection_and_detection(tmp_path):
    url = f"sqlite:///{tmp_path / 'inspections.sqlite'}"
    engine = get_engine(url)
    init_db(engine)
    session = get_session(engine)
    row = log_inspection(
        session,
        image_name="plate.jpg",
        model_version="test-weights",
        overall_result="FAIL",
        max_confidence=0.87,
        inference_ms=12.5,
        device="cpu",
        threshold_config="example",
        notes="",
        detections=[{"class": "corrosion", "confidence": 0.87, "bbox": [1, 2, 30, 40]}],
    )
    assert row.inspection_id == 1
    assert len(row.detections) == 1
    assert row.detections[0].class_name == "corrosion"
    assert row.detections[0].x2 == 30
    again = recent_inspections(session, limit=5)
    assert again[0].image_name == "plate.jpg"
    assert again[0].overall_result == "FAIL"
