# Plan match report

This file is not the shop-inspect generator. It was written on 2026-10-09 after the 700-row check, which until then had only been run in memory. No verification script or report was saved beside `generate.py`.

## Command re-run

Interpreter: `/workspace/switchgear-vision-inspection/.venv/bin/python`

```bash
PYTHONDONTWRITEBYTECODE=1 /workspace/switchgear-vision-inspection/.venv/bin/python - <<'PY'
import csv, sys
from importlib.machinery import SourceFileLoader
sys.dont_write_bytecode = True
mod = SourceFileLoader("shop_generate", "/cursor/stores/bc-ae1e9d8d-7c93-47e6-8801-d8044527a7eb/synthetic-enclosures/generate.py").load_module()
import numpy as np
plans, counts = mod.build_plan()
splits = mod.assign_splits(len(plans))
manifest = "/workspace/switchgear-vision-inspection/data/shop/manifest.csv"
with open(manifest, newline="") as f:
    rows = list(csv.DictReader(f))
mism = 0
for i, row in enumerate(rows):
    seed = int(row["seed"])
    planned = ",".join(str(c) for c in sorted(plans[seed]))
    if not (seed == i and splits[seed] == row["split"] and planned == row["class_ids"] and int(row["width"]) == mod.IMAGE_W and int(row["height"]) == mod.IMAGE_H and row["generator_version"] == mod.GENERATOR_VERSION and row["synthetic"] == "true"):
        mism += 1
print("numpy", np.__version__)
print("plan", len(plans), counts.tolist())
print("rows", len(rows), "mismatches", mism)
print("match", f"{len(rows)-mism}/{len(rows)}")
PY
```

## Result

```
numpy 2.5.2
plan 700 [196, 197, 193, 194, 195, 195, 197]
rows 700 mismatches 0
match 700/700
```

`build_plan()` and `assign_splits()` were called. No images were written. Class ids were compared after sorting, which is how `generate.py` writes `class_ids` (`labels.sort` by class id before the CSV field is joined).

Python for this check: 3.12.3 (main, Aug 31 2026, 10:18:26) [GCC 13.3.0]

numpy: 2.5.2
