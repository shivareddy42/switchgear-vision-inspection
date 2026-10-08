#!/usr/bin/env python3
"""Check raw images and copy the keepers into data/cleaned."""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inspection.quality import assess_file, flag_duplicates

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Quality-check raw images.")
    parser.add_argument("--raw", default="data/raw")
    parser.add_argument("--cleaned", default="data/cleaned")
    parser.add_argument("--report", default="outputs/data_quality_report.csv")
    args = parser.parse_args()
    raw = ROOT / args.raw
    cleaned = ROOT / args.cleaned
    report_path = ROOT / args.report
    images = sorted(path for path in raw.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
    records = []
    for path in images:
        record = assess_file(path)
        record["class_name"] = path.parent.name
        record["relative_path"] = str(path.relative_to(raw))
        records.append(record)
    records = flag_duplicates(records)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["relative_path", "class_name", "filename", "sha256", "dhash", "width", "height", "status", "reason"]
    with report_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    if cleaned.exists():
        shutil.rmtree(cleaned)
    kept = 0
    for record in records:
        if record["status"] != "ok":
            continue
        source = raw / record["relative_path"]
        destination = cleaned / record["relative_path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        kept += 1
    print(f"checked {len(records)} kept {kept} report {report_path.relative_to(ROOT)}")
    counts: dict[str, int] = {}
    for record in records:
        counts[record["status"]] = counts.get(record["status"], 0) + 1
    for status, count in sorted(counts.items()):
        print(f"  {status}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
