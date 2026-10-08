#!/usr/bin/env python3
"""Print the most recent inspection rows."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from db.database import get_engine, get_session, init_db, recent_inspections


def main() -> int:
    parser = argparse.ArgumentParser(description="Show recent inspection log rows.")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--database-url", default=None)
    args = parser.parse_args()
    engine = get_engine(args.database_url)
    init_db(engine)
    session = get_session(engine)
    rows = recent_inspections(session, limit=args.limit)
    if not rows:
        print("no inspections logged")
        return 0
    for row in rows:
        print(
            f"{row.inspection_id} {row.timestamp} {row.image_name} {row.overall_result} "
            f"max_conf={row.max_confidence:.3f} {row.inference_ms:.1f}ms model={row.model_version} "
            f"detections={len(row.detections)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
