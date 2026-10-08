#!/usr/bin/env python3
"""Export a flat CSV that Power BI can import. No PBIX file is produced.

Each row is one date, hour, and defect class. Inspection totals are repeated
on every class row for that hour so a visual can filter by defect_class
without losing the hour's pass/fail counts. Hours with no detections get one
row with an empty defect_class.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from db.database import get_engine, get_session, init_db
from db.models import DetectionRow, Inspection


def main() -> int:
    parser = argparse.ArgumentParser(description="Export dashboard CSV.")
    parser.add_argument("--output", default="outputs/inspection_summary.csv")
    parser.add_argument("--database-url", default=None)
    args = parser.parse_args()
    engine = get_engine(args.database_url)
    init_db(engine)
    session = get_session(engine)
    inspections = list(session.query(Inspection).all())
    buckets: dict[tuple, dict] = {}
    for inspection in inspections:
        stamp = inspection.timestamp
        key_time = (stamp.date().isoformat(), stamp.hour)
        bucket = buckets.setdefault(
            key_time,
            {"inspections": [], "by_class": defaultdict(list)},
        )
        bucket["inspections"].append(inspection)
        for detection in inspection.detections:
            bucket["by_class"][detection.class_name].append(detection)
    destination = ROOT / args.output
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "date",
        "hour",
        "total_inspections",
        "passes",
        "fails",
        "failure_rate",
        "average_inference_ms",
        "defect_class",
        "defect_count",
        "average_confidence",
    ]
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for (day, hour), bucket in sorted(buckets.items()):
            rows = bucket["inspections"]
            total = len(rows)
            fails = sum(1 for row in rows if row.overall_result == "FAIL")
            passes = total - fails
            avg_ms = sum(row.inference_ms for row in rows) / total if total else 0.0
            base = {
                "date": day,
                "hour": hour,
                "total_inspections": total,
                "passes": passes,
                "fails": fails,
                "failure_rate": f"{(fails / total):.4f}" if total else "",
                "average_inference_ms": f"{avg_ms:.3f}",
            }
            if not bucket["by_class"]:
                writer.writerow({**base, "defect_class": "", "defect_count": 0, "average_confidence": ""})
                continue
            for class_name, detections in sorted(bucket["by_class"].items()):
                avg_conf = sum(item.confidence for item in detections) / len(detections)
                writer.writerow(
                    {
                        **base,
                        "defect_class": class_name,
                        "defect_count": len(detections),
                        "average_confidence": f"{avg_conf:.4f}",
                    }
                )
    print(f"wrote {destination.relative_to(ROOT)} from {len(inspections)} inspections")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
