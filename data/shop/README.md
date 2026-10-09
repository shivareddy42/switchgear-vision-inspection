# Synthetic shop-inspection enclosure defects

This folder is the in-repo copy of the shop-inspect renders. It is not
`data/synthetic/` (the earlier diagram set) and it is not the Wikimedia split
in `data/train`, `data/val`, and `data/test`. `configs/data.yaml` points here
for the next training run. Do not score this folder in the same metric as
those other domains.

These images are **procedural renders**, not photographs of a plant, shop, or
factory. They were drawn with a pinhole camera, painted or brushed metal
shading, and defect masks. They are not frames taken from the licensed
reference photos, and they are not copies of any catalog or website.

Do **not** treat this folder as a factory dataset or interpret its benchmark as
plant performance. The held-out synthetic-domain test is documented in
`docs/MODEL_CARD.md` and `outputs/training/yolov8_shop/`. These renders are
procedural, and their score does not transfer to plant photos.

## What the pictures show

Clean manufacturing-inspection views of floor-standing switchgear cabinets:

- gray painted metal, or bare brushed metal on some seeds
- shop lighting (overhead fluorescent bars, soft floor shadow)
- mild perspective, with a side panel and door rebate visible
- seams, louvers, a handle, hinges, and shaded fastener heads
- some open doors looking onto straight copper busbars on dark insulator blocks

Defects are small and local: a scratch, a dent, pits in a weld bead, a crack
along a weld, a small oxidation patch, one busbar shifted off its insulator
line, or one missing or loose fastener. Cabinets are not wrecked, rusted out,
or overgrown.

Geometry and materials were checked against two licensed photos used only as
visual reference (not as pixels in this set):

- `licensed-enclosures/unlabeled/001-electrical-switchgear.jpg` (gray cabinet line, aisle, shop lighting)
- `licensed-enclosures/unlabeled/013-2500a-copper-busbars-in-motor-control-panel.jpg` (straight copper busbars, dark supports, galvanized cheek)

## Classes

YOLO class ids:

| id | name | mask |
|----|------|------|
| 0 | scratch | groove and bright shoulder painted on the door |
| 1 | dent | elliptical sheet deformation shaded into the door |
| 2 | weld_porosity | pits and rims drawn on a weld bead |
| 3 | weld_crack | crack pixels drawn along a weld bead |
| 4 | corrosion | small rust patch shaded into the paint |
| 5 | misaligned_busbar | visible pixels of the one shifted copper bar |
| 6 | missing_or_loose_component | empty countersunk hole, or the loose fastener head |

Each image has 1–3 of these. The label box is the axis-aligned bound of that
mask after the same perspective that placed the pixels. Boxes are normalized
`class x_center y_center width height` and are never zero-area.

## Splits

Train / val / test = 70 / 15 / 15 of the original renders.
Assignment uses `numpy.random.Generator(42)` once, after the render
list exists and before any augmentation. This generator does not write
augmented copies. Each seed is one file and one split.

This build: **700** unique images (seeds `0` … `699`).

Per-class image counts (an image can count toward more than one class):

| class | images |
|-------|--------|
| 0 scratch | 196 |
| 1 dent | 197 |
| 2 weld_porosity | 193 |
| 3 weld_crack | 194 |
| 4 corrosion | 195 |
| 5 misaligned_busbar | 195 |
| 6 missing_or_loose_component | 197 |

Split counts: train 490, val 105, test 105.

## Reproducibility note

The committed images, labels, split assignment, and manifest are sufficient to
re-run training and evaluation on this exact dataset. The original
`shop-inspect-1.0.0` render generator is **not currently included in this
public repository**, so the pixels themselves cannot yet be regenerated from
seed alone.

A separate, simpler procedural generator is included at
`scripts/synthesize_enclosures.py`; it produces `data/synthetic/` and is not
the source of this shop-inspect benchmark.

## Parameters (shop-inspect-1.0.0)

| name | value |
|------|--------|
| `PLAN_SEED` | 20261008 |
| `SPLIT_SEED` | 42 |
| `NOISE_SEED_OFFSET` | 99991 |
| `TEXTURE_NOISE_SEED` | 12345 |
| image size | 960 × 640 |
| JPEG quality | 92, 4:2:2, optimize off |
| images | at least 700, and a multiple of 20 |
| per-class minimum | 100 |
| defects per image | 1–3, distinct classes |
| camera FOV | 40–48° horizontal |
| camera yaw | about 12–30° (aisle a bit more), never straight-on orthographic |
| pitch | about −4° to +5° |
| metal | painted gray or brushed, orange-peel noise, edge bevel, specular |
| busbar shift | 34–48 mm off the insulator centerline, bar kept straight |
| blur | Gaussian σ 0.30–0.65 px after the mask is stored |
| extra augmentation | none |

`manifest.csv` columns: `filename,seed,split,class_ids,width,height,generator_version,synthetic`.
`synthetic` is `true` on every row. `class_ids` is a comma-separated list of YOLO ids.
