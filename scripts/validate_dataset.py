#!/usr/bin/env python3
"""Validate image/label pairing and split leakage."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inspection.validation import validate_tree, write_report


def main() -> int:
    report = validate_tree(ROOT / "data")
    destination = ROOT / "outputs" / "dataset_validation_report.json"
    write_report(report, destination)
    print(
        f"train {report['train_count']} val {report['val_count']} test {report['test_count']} "
        f"errors {len(report['errors'])} warnings {len(report['warnings'])}"
    )
    for warning in report["warnings"]:
        print(f"  warning: {warning}")
    for error in report["errors"]:
        print(f"  error: {error}")
    print(f"wrote {destination.relative_to(ROOT)}")
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
