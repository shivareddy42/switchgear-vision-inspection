# Recovered shop-inspect-1.0.0 generator

`generate.py` is the actual source exported from Cursor's
`synthetic-enclosures/generate.py`, not `scripts/synthesize_enclosures.py`.
It is preserved byte for byte, including historical paths and generated README
text. SHA-256:
`602137072a30e85ad93640fa4bc21b349a897ce5538fe794c353629b1d2ae854`.
The exported adjacent manifest is byte-identical to `data/shop/manifest.csv`.
The export also contained stats, class names, a dataset YAML, and unpinned
requirements; the committed manifest and dataset README remain authoritative.
The later supplied check report is preserved in `plan-match-report.md`.

## Environment and evidence

The original rendering Python version, dependency lockfile, and exact command
were not saved. The later plan check used Python 3.12.3, NumPy 2.5.2,
OpenCV 5.0.0.93, and Pillow 12.3.0. `requirements.txt` here pins that later
environment; it is not a recovered render lockfile. JPEG encoding, floating
point calculations, and library/platform changes can affect byte identity.

The source's `build_plan()` and `assign_splits()` can be checked independently
against all 700 manifest rows. A successful plan check does not prove that all
700 images have been reproduced. Pixel verification is separate and reports
each selected seed. See `verification.json` for the integration check and its
exact scope/environment. Do not infer plant accuracy from this synthetic set.

Integration check on macOS ARM64, Python 3.12.14: **700/700 plan rows matched**;
seeds 0–9 gave **10/10 identical label files and 0/10 identical JPEG files**.
For seed 0, decoding both JPEGs gave mean absolute channel difference 0.001361
on the 0–255 scale, maximum difference 3, and 0.1414% of pixels changed.
This is a small measured difference for that seed, not proof of its cause or
of byte-identical reproduction on Linux. The original Linux-render report's
sample identity has not been independently confirmed here.

The portable launcher also completed a full 700-image rebuild with four workers
in about 97 seconds. Its manifest and all **700 label files matched exactly**;
**0/700 JPEG files matched byte for byte**. Across the decoded images, mean
absolute channel difference was 0.001819 on the 0–255 scale, 0.1757% of pixels
changed, and the maximum channel difference was 135. The small aggregate error
does not rule out larger local differences. These results demonstrate functional
regeneration while documenting that exact pixel/byte reproducibility remains
unconfirmed for the unknown original environment.

## Run from the repository root

```bash
python3 -m venv .venv-shop
source .venv-shop/bin/activate
python -m pip install -r scripts/shop_inspect/requirements.txt

# Source checksum, filenames, seeds, classes, dimensions, flags, and splits
python -m scripts.shop_inspect.verify

# Independently render these seeds using the original 700-image plan/splits
python -m scripts.shop_inspect.verify --seeds 0 1 2 3 4 5 6 7 8 9

# Generate all 700 into a fresh folder (can take substantial time)
python -m scripts.shop_inspect.cli --output /tmp/shop-inspect-rebuild --workers 4

# Check all regenerated images and labels against the committed benchmark
python - <<'PY'
import csv
from pathlib import Path
expected = Path('data/shop')
actual = Path('/tmp/shop-inspect-rebuild')
with (expected / 'manifest.csv').open(newline='') as f:
    rows = list(csv.DictReader(f))
with (actual / 'manifest.csv').open(newline='') as f:
    rebuilt = list(csv.DictReader(f))
assert rows == rebuilt, 'Manifest mismatch'
mismatches = []
for row in rows:
    image = Path(row['filename'])
    label = Path('labels') / row['split'] / (image.stem + '.txt')
    for relative in (image, label):
        if (actual / relative).read_bytes() != (expected / relative).read_bytes():
            mismatches.append(str(relative))
print(f'{len(rows)} images checked; {len(mismatches)} file mismatches')
assert not mismatches, mismatches
PY
```

The verifier exits nonzero on a source/plan mismatch or any selected image or
label byte mismatch. Report mismatches as such; do not change the original
generator or golden files to make verification pass.

Use the portable launcher instead of running archived `generate.py` directly:
the original defaults target `/cursor/stores`, use Linux system font paths,
and delete the output's `images/` and `labels/` directories. The launcher sets
the preview font paths to the bundled fonts, rejects nonempty destinations and
destinations inside `data/shop`, and writes `preview.jpg` beside the new data.
It does not change rendering or JPEG settings. Its multiprocessing workers
import the archived module normally.

Do not use the original `--limit` for benchmark verification: it truncates the
plan and reassigns splits using the new length; its validation also requires
at least 700 images. The verifier renders selected seeds using the full plan.
The archived generated README contains pre-training statements; it is historical
output, not a replacement for the repository's current model card.

## Tests and assets

```bash
python -m pip install pytest
python -m pytest tests/test_shop_inspect.py -q
```

Tests protect source identity, all 700 planned rows, valid mask bounds,
repeat-render determinism, bundled font loading, and output-directory protection.
Byte comparison to historical samples is explicit via the verifier above,
rather than assumed portable across architectures.

The exported, unmodified Liberation Sans Regular/Bold fonts identify themselves
as version 2.1.5 and SIL Open Font License 1.1 in their metadata. Their copyright
notices and license are in `assets/LICENSE`, copied from
https://github.com/liberationfonts/liberation-fonts/blob/main/LICENSE.
The repository's MIT license does not replace this font license.
