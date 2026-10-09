#!/usr/bin/env python3
"""Procedural switchgear-style metal enclosures.

These images are drawn in code. They are not factory photographs, not the
Wikimedia set under data/raw, and Each file is one class. The YOLO box is the
bounding rectangle of the defect mask drawn in that image.

Deterministic for a given --seed, --per-class, and image size. The split
uses the same seed and is applied to these originals before any training
augmentation. A generation seed belongs to one image and therefore one split.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inspection.classes import CLASS_NAMES  # noqa: E402
from inspection.hashing import sha256_file  # noqa: E402

WIDTH = 640
HEIGHT = 480
NOTE = (
    "Procedural render. Not a factory photograph. Not a Wikimedia file. "
    "Not part of a plant-owned validation dataset."
)


def image_seed(master_seed: int, class_id: int, index: int) -> int:
    """One positive integer per image. Unique for class_id < 100 and index < 100000."""
    return master_seed * 1_000_003 + class_id * 100_000 + index


def _clip_rect(x: int, y: int, w: int, h: int, width: int, height: int) -> tuple[int, int, int, int]:
    x = int(np.clip(x, 0, width - 2))
    y = int(np.clip(y, 0, height - 2))
    w = int(np.clip(w, 2, width - x))
    h = int(np.clip(h, 2, height - y))
    return x, y, w, h


def _bolt_sites(rect: tuple[int, int, int, int], spacing: int) -> list[tuple[int, int]]:
    x, y, w, h = rect
    sites: list[tuple[int, int]] = []
    inset = 16
    step = max(28, spacing)
    for xx in range(x + inset + 8, x + w - inset - 8, step):
        sites.append((xx, y + inset))
        sites.append((xx, y + h - inset))
    for yy in range(y + step, y + h - step, step):
        sites.append((x + inset, yy))
        sites.append((x + w - inset, yy))
    return sites


def _sample_geometry(rng: np.random.Generator, force_open: bool) -> dict:
    ow = int(rng.integers(int(WIDTH * 0.56), int(WIDTH * 0.80)))
    oh = int(rng.integers(int(HEIGHT * 0.64), int(HEIGHT * 0.86)))
    ox = int(rng.integers(24, max(25, WIDTH - ow - 24)))
    oy = int(rng.integers(18, max(19, HEIGHT - oh - 18)))
    open_door = bool(force_open or rng.random() < 0.30)
    frame = 18
    if open_door:
        dw = int(np.clip(ow * float(rng.uniform(0.30, 0.40)), 78, ow * 0.46))
        dh = oh - 20
        dx = max(6, ox - dw + 16)
        dy = oy + 10
    else:
        dx, dy = ox + frame, oy + frame
        dw, dh = ow - 2 * frame, oh - 2 * frame
    dx, dy, dw, dh = _clip_rect(dx, dy, dw, dh, WIDTH, HEIGHT)
    spacing = int(rng.integers(36, 54))
    bolts = _bolt_sites((dx, dy, dw, dh), spacing)
    welds = [
        ((ox + 8, oy + 8), (ox + 58, oy + 8)),
        ((ox + ow - 58, oy + 8), (ox + ow - 8, oy + 8)),
        ((ox + 8, oy + oh - 8), (ox + 58, oy + oh - 8)),
        ((ox + ow - 58, oy + oh - 8), (ox + ow - 8, oy + oh - 8)),
        ((ox + 8, oy + 8), (ox + 8, oy + 56)),
        ((ox + ow - 8, oy + oh - 56), (ox + ow - 8, oy + oh - 8)),
    ]
    busbars: list[tuple[int, int, int, int]] = []
    if open_door:
        ix, iy = ox + 16, oy + 22
        iw, ih = ow - 32, oh - 44
        bar_w = int(np.clip(iw / 11, 16, 28))
        gap = iw // 4
        for i in range(3):
            bx = ix + gap * (i + 1) - bar_w // 2
            by = iy + 18
            bh = ih - 36
            busbars.append(_clip_rect(bx, by, bar_w, bh, WIDTH, HEIGHT))
    return {
        "outer": (ox, oy, ow, oh),
        "door": (dx, dy, dw, dh),
        "open": open_door,
        "bolts": bolts,
        "welds": welds,
        "busbars": busbars,
        "metal": (
            int(rng.integers(128, 158)),
            int(rng.integers(136, 168)),
            int(rng.integers(142, 176)),
        ),
    }


def _background(rng: np.random.Generator) -> np.ndarray:
    wall = int(rng.integers(78, 96))
    image = np.full((HEIGHT, WIDTH, 3), wall, np.uint8)
    image[:, :, 0] = np.clip(image[:, :, 0].astype(np.int16) + int(rng.integers(-6, 4)), 0, 255)
    noise = rng.integers(0, 10, size=(HEIGHT, WIDTH, 1), dtype=np.uint8)
    image = cv2.add(image, np.repeat(noise, 3, axis=2))
    horizon = int(rng.integers(int(HEIGHT * 0.72), int(HEIGHT * 0.82)))
    floor = int(rng.integers(58, 74))
    image[horizon:, :] = floor
    image[horizon:, :, 1] = np.clip(image[horizon:, :, 1].astype(np.int16) + 4, 0, 255)
    return image


def _shade_rect(image: np.ndarray, rect: tuple[int, int, int, int], color: tuple[int, int, int]) -> None:
    x, y, w, h = rect
    patch = np.empty((h, w, 3), np.float32)
    patch[:] = color
    vertical = np.linspace(12, -16, h, dtype=np.float32).reshape(h, 1, 1)
    brush = np.tile(np.linspace(-6, 6, w, dtype=np.float32), (h, 1))
    patch += vertical
    patch[:, :, 0] += brush * 0.15
    patch[:, :, 1] += brush * 0.25
    patch[:, :, 2] += brush * 0.35
    image[y : y + h, x : x + w] = np.clip(patch, 0, 255).astype(np.uint8)


def _draw_bolt(image: np.ndarray, center: tuple[int, int], radius: int = 7) -> None:
    cx, cy = center
    cv2.circle(image, (cx, cy), radius, (70, 74, 78), -1, lineType=cv2.LINE_AA)
    cv2.circle(image, (cx, cy), max(2, radius - 3), (168, 174, 180), -1, lineType=cv2.LINE_AA)
    cv2.circle(image, (cx - 1, cy - 1), max(1, radius // 4), (210, 214, 218), -1, lineType=cv2.LINE_AA)


def _draw_hole(image: np.ndarray, mask: np.ndarray, center: tuple[int, int], radius: int = 7) -> None:
    cx, cy = center
    cv2.circle(image, (cx, cy), radius, (28, 30, 32), -1, lineType=cv2.LINE_AA)
    cv2.circle(image, (cx, cy), max(2, radius - 3), (12, 12, 14), -1, lineType=cv2.LINE_AA)
    cv2.circle(mask, (cx, cy), radius, 255, -1, lineType=cv2.LINE_8)


def _draw_cabinet(image: np.ndarray, geom: dict) -> None:
    ox, oy, ow, oh = geom["outer"]
    shadow = image.copy()
    cv2.ellipse(
        shadow,
        (ox + ow // 2, oy + oh - 6),
        (ow // 2, 10),
        0,
        0,
        360,
        (40, 42, 44),
        -1,
        lineType=cv2.LINE_AA,
    )
    image[:] = cv2.addWeighted(shadow, 0.35, image, 0.65, 0)
    _shade_rect(image, geom["outer"], tuple(max(0, c - 28) for c in geom["metal"]))
    if geom["open"]:
        ix, iy = ox + 14, oy + 14
        iw, ih = ow - 28, oh - 28
        _shade_rect(image, (ix, iy, iw, ih), (46, 52, 58))
        for bar in geom["busbars"]:
            _shade_rect(image, bar, (42, 92, 168))
            bx, by, bw, bh = bar
            cv2.rectangle(image, (bx, by - 8), (bx + bw, by), (90, 94, 98), -1)
            cv2.rectangle(image, (bx, by + bh), (bx + bw, by + bh + 8), (90, 94, 98), -1)
    _shade_rect(image, geom["door"], geom["metal"])
    dx, dy, dw, dh = geom["door"]
    cv2.rectangle(image, (dx, dy), (dx + dw - 1, dy + dh - 1), (86, 90, 94), 2, lineType=cv2.LINE_AA)
    # Frame-to-door seam on the closed cabinet, or the opening edge when the door is swung.
    ox, oy, ow, oh = geom["outer"]
    cv2.rectangle(image, (ox + 8, oy + 8), (ox + ow - 9, oy + oh - 9), (60, 64, 68), 1, lineType=cv2.LINE_8)
    for p1, p2 in geom["welds"]:
        cv2.line(image, p1, p2, (186, 190, 194), 6, lineType=cv2.LINE_AA)
    hx = dx + dw - 22
    hy = dy + dh // 2 - 28
    cv2.rectangle(image, (hx, hy), (hx + 8, hy + 56), (96, 100, 106), -1)
    for hinge_y in (dy + 24, dy + dh // 2, dy + dh - 36):
        cv2.rectangle(image, (dx + 2, hinge_y), (dx + 12, hinge_y + 16), (80, 84, 88), -1)


def _paint_scratch(image: np.ndarray, mask: np.ndarray, rng: np.random.Generator, rect: tuple[int, int, int, int]) -> None:
    x, y, w, h = rect
    strokes = int(rng.integers(1, 4))
    for _ in range(strokes):
        px = int(rng.integers(x + 18, x + w - 18))
        py = int(rng.integers(y + 18, y + h - 18))
        points = [(px, py)]
        steps = int(rng.integers(12, 28))
        for _step in range(steps):
            px = int(np.clip(px + int(rng.integers(2, 9)), x + 8, x + w - 8))
            py = int(np.clip(py + int(rng.integers(-3, 4)), y + 8, y + h - 8))
            points.append((px, py))
        chain = np.array(points, np.int32).reshape(-1, 1, 2)
        cv2.polylines(image, [chain], False, (40, 42, 46), 2, lineType=cv2.LINE_8)
        offset = chain.copy()
        offset[:, 0, 1] += 1
        cv2.polylines(image, [offset], False, (214, 218, 222), 1, lineType=cv2.LINE_8)
        cv2.polylines(mask, [chain], False, 255, 2, lineType=cv2.LINE_8)


def _paint_dent(image: np.ndarray, mask: np.ndarray, rng: np.random.Generator, rect: tuple[int, int, int, int]) -> None:
    x, y, w, h = rect
    cx = int(rng.integers(x + w // 5, x + 4 * w // 5))
    cy = int(rng.integers(y + h // 5, y + 4 * h // 5))
    ax = int(rng.integers(22, min(48, w // 3)))
    ay = int(rng.integers(14, min(32, h // 4)))
    angle = float(rng.integers(0, 180))
    dent = np.zeros(mask.shape, np.uint8)
    cv2.ellipse(dent, (cx, cy), (ax, ay), angle, 0, 360, 255, -1, lineType=cv2.LINE_8)
    ys, xs = np.where(dent > 0)
    if len(xs) == 0:
        return
    span = max(1, int(xs.max() - xs.min()))
    shade = 0.52 + 0.85 * ((xs - xs.min()) / span)
    pix = image[ys, xs].astype(np.float32)
    image[ys, xs] = np.clip(pix * shade[:, None], 0, 255).astype(np.uint8)
    cv2.ellipse(image, (cx, cy), (ax, ay), angle, 0, 360, (70, 74, 78), 2, lineType=cv2.LINE_AA)
    mask[dent > 0] = 255


def _paint_porosity(image: np.ndarray, mask: np.ndarray, rng: np.random.Generator, welds: list) -> None:
    p1, p2 = welds[int(rng.integers(0, len(welds)))]
    pits = int(rng.integers(8, 16))
    for _ in range(pits):
        t = float(rng.random())
        x = int(p1[0] + t * (p2[0] - p1[0]) + int(rng.integers(-3, 4)))
        y = int(p1[1] + t * (p2[1] - p1[1]) + int(rng.integers(-3, 4)))
        radius = int(rng.integers(2, 4))
        cv2.circle(image, (x, y), radius, (18, 18, 20), -1, lineType=cv2.LINE_8)
        cv2.circle(mask, (x, y), radius, 255, -1, lineType=cv2.LINE_8)


def _paint_crack(image: np.ndarray, mask: np.ndarray, rng: np.random.Generator, welds: list) -> None:
    p1, p2 = welds[int(rng.integers(0, len(welds)))]
    steps = int(rng.integers(8, 14))
    points = []
    for i in range(steps + 1):
        t = i / steps
        x = int(p1[0] + t * (p2[0] - p1[0]) + int(rng.integers(-2, 3)))
        y = int(p1[1] + t * (p2[1] - p1[1]) + int(rng.integers(-2, 3)))
        points.append((x, y))
    chain = np.array(points, np.int32).reshape(-1, 1, 2)
    cv2.polylines(image, [chain], False, (8, 8, 10), 2, lineType=cv2.LINE_8)
    cv2.polylines(mask, [chain], False, 255, 2, lineType=cv2.LINE_8)


def _paint_corrosion(image: np.ndarray, mask: np.ndarray, rng: np.random.Generator, rect: tuple[int, int, int, int]) -> None:
    x, y, w, h = rect
    blob = np.zeros(mask.shape, np.uint8)
    cx = int(rng.integers(x + 24, x + w - 24))
    cy = int(rng.integers(y + h // 3, y + h - 20))
    for _ in range(int(rng.integers(4, 7))):
        ox = int(np.clip(cx + int(rng.integers(-28, 29)), x + 8, x + w - 8))
        oy = int(np.clip(cy + int(rng.integers(-18, 19)), y + 8, y + h - 8))
        ax = int(rng.integers(10, 26))
        ay = int(rng.integers(8, 18))
        cv2.ellipse(blob, (ox, oy), (ax, ay), float(rng.integers(0, 180)), 0, 360, 255, -1, lineType=cv2.LINE_8)
    ys, xs = np.where(blob > 0)
    if len(xs) == 0:
        return
    color = np.array([int(rng.integers(20, 45)), int(rng.integers(70, 110)), int(rng.integers(140, 185))], np.float32)
    jitter = rng.integers(-22, 23, size=(len(xs), 3))
    image[ys, xs] = np.clip(color + jitter, 0, 255).astype(np.uint8)
    mask[blob > 0] = 255


def _paint_misaligned(image: np.ndarray, mask: np.ndarray, rng: np.random.Generator, geom: dict) -> None:
    bars = geom["busbars"]
    choice = int(rng.integers(0, len(bars)))
    bx, by, bw, bh = bars[choice]
    shift = int(rng.integers(22, 38)) * (1 if rng.random() < 0.5 else -1)
    moved = _clip_rect(bx + shift, by + int(rng.integers(-4, 5)), bw, bh, WIDTH, HEIGHT)
    # Cover the original bar and its supports, then draw the shifted bar.
    cover = _clip_rect(bx, by - 8, bw, bh + 16, WIDTH, HEIGHT)
    _shade_rect(image, cover, (46, 52, 58))
    _shade_rect(image, moved, (42, 92, 168))
    mx, my, mw, mh = moved
    cv2.rectangle(image, (mx, my - 8), (mx + mw, my), (90, 94, 98), -1)
    cv2.rectangle(image, (mx, my + mh), (mx + mw, my + mh + 8), (90, 94, 98), -1)
    cv2.rectangle(mask, (mx, my), (mx + mw - 1, my + mh - 1), 255, -1)


def _paint_missing(image: np.ndarray, mask: np.ndarray, rng: np.random.Generator, geom: dict) -> None:
    sites = geom["bolts"]
    if not sites:
        raise RuntimeError("cabinet has no bolt sites")
    index = int(rng.integers(0, len(sites)))
    center = sites[index]
    loose = bool(rng.random() < 0.5)
    _draw_hole(image, mask, center, radius=8)
    if loose:
        shift = int(rng.integers(16, 26))
        loose_center = (center[0] + shift, center[1] + int(rng.integers(-6, 7)))
        loose_center = (int(np.clip(loose_center[0], 8, WIDTH - 8)), int(np.clip(loose_center[1], 8, HEIGHT - 8)))
        _draw_bolt(image, loose_center, radius=7)
        cv2.circle(mask, loose_center, 8, 255, -1, lineType=cv2.LINE_8)


def _apply_defect(image: np.ndarray, mask: np.ndarray, rng: np.random.Generator, class_id: int, geom: dict) -> None:
    if class_id == 0:
        _paint_scratch(image, mask, rng, geom["door"])
    elif class_id == 1:
        _paint_dent(image, mask, rng, geom["door"])
    elif class_id == 2:
        _paint_porosity(image, mask, rng, geom["welds"])
    elif class_id == 3:
        _paint_crack(image, mask, rng, geom["welds"])
    elif class_id == 4:
        _paint_corrosion(image, mask, rng, geom["door"])
    elif class_id == 5:
        _paint_misaligned(image, mask, rng, geom)
    elif class_id == 6:
        _paint_missing(image, mask, rng, geom)
    else:
        raise ValueError(f"unknown class id {class_id}")


def _perspective(image: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    src = np.float32([[0, 0], [WIDTH - 1, 0], [WIDTH - 1, HEIGHT - 1], [0, HEIGHT - 1]])
    jitter = rng.uniform(-10, 10, size=(4, 2)).astype(np.float32)
    dst = src + jitter
    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image, matrix, (WIDTH, HEIGHT), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    warped_mask = cv2.warpPerspective(mask, matrix, (WIDTH, HEIGHT), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
    return warped, warped_mask


def mask_to_yolo(mask: np.ndarray, class_id: int) -> list[tuple[int, float, float, float, float]]:
    """One box around every mask pixel. Empty masks produce no box."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return []
    height, width = mask.shape
    x1 = float(xs.min())
    y1 = float(ys.min())
    x2 = float(xs.max()) + 1.0
    y2 = float(ys.max()) + 1.0
    cx = ((x1 + x2) / 2.0) / width
    cy = ((y1 + y2) / 2.0) / height
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height
    cx = float(np.clip(cx, 0.0, 1.0))
    cy = float(np.clip(cy, 0.0, 1.0))
    bw = float(np.clip(bw, 1.0 / width, 1.0))
    bh = float(np.clip(bh, 1.0 / height, 1.0))
    return [(class_id, cx, cy, bw, bh)]


def render_enclosure(class_id: int, index: int, master_seed: int = 42) -> tuple[np.ndarray, np.ndarray, list[tuple[int, float, float, float, float]], int]:
    """Return BGR image, defect mask, YOLO rows, and the generation seed."""
    if class_id < 0 or class_id >= len(CLASS_NAMES):
        raise ValueError(f"class_id must be 0..{len(CLASS_NAMES) - 1}")
    seed = image_seed(master_seed, class_id, index)
    last_error = "no attempt"
    for attempt in range(4):
        rng = np.random.default_rng(np.random.SeedSequence([master_seed, class_id, index, attempt]))
        image = _background(rng)
        geom = _sample_geometry(rng, force_open=(class_id == 5))
        _draw_cabinet(image, geom)
        for site in geom["bolts"]:
            _draw_bolt(image, site)
        mask = np.zeros((HEIGHT, WIDTH), np.uint8)
        _apply_defect(image, mask, rng, class_id, geom)
        image, mask = _perspective(image, mask, rng)
        mask = np.where(mask > 0, np.uint8(255), np.uint8(0))
        rows = mask_to_yolo(mask, class_id)
        if rows and int(mask.sum()) >= 20 * 255:
            return image, mask, rows, seed
        last_error = f"empty or tiny mask on attempt {attempt}"
    raise RuntimeError(f"class {class_id} index {index}: {last_error}")


def assign_splits(records: list[dict], seed: int = 42) -> None:
    """Stratified 70/15/15. Mutates each record's split. Seeds stay in one split."""
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    grouped: dict[int, list[dict]] = {}
    for record in records:
        grouped.setdefault(int(record["class_id"]), []).append(record)
    for class_id in sorted(grouped):
        group = grouped[class_id]
        order = rng.permutation(len(group))
        n = len(group)
        n_train = (n * 70) // 100
        n_val = (n * 15) // 100
        shuffled = [group[int(i)] for i in order]
        for record in shuffled[:n_train]:
            record["split"] = "train"
        for record in shuffled[n_train : n_train + n_val]:
            record["split"] = "val"
        for record in shuffled[n_train + n_val :]:
            record["split"] = "test"
    seeds_by_split: dict[str, set[int]] = {"train": set(), "val": set(), "test": set()}
    for record in records:
        seeds_by_split[record["split"]].add(int(record["seed"]))
    overlap = (seeds_by_split["train"] & seeds_by_split["val"]) | (seeds_by_split["train"] & seeds_by_split["test"]) | (
        seeds_by_split["val"] & seeds_by_split["test"]
    )
    if overlap:
        raise RuntimeError(f"generation seed crossed splits: {sorted(overlap)[:5]}")


def _write_jpeg(path: Path, image_bgr: np.ndarray) -> None:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    Image.fromarray(rgb).save(path, format="JPEG", quality=90, subsampling=0)


def _write_label(path: Path, rows: list[tuple[int, float, float, float, float]]) -> None:
    lines = [f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}" for class_id, cx, cy, bw, bh in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _clear_files(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for item in directory.iterdir():
        if item.is_file():
            item.unlink()


def write_preview(examples: list[tuple[np.ndarray, list[tuple[int, float, float, float, float]], str]], destination: Path) -> None:
    """Grid of one labeled example per class, marked synthetic."""
    cell_w, cell_h = 320, 240
    columns = 4
    rows = 2
    title_h = 36
    canvas = Image.new("RGB", (columns * cell_w, title_h + rows * cell_h), (24, 26, 28))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((8, 10), "SYNTHETIC enclosures — procedural, not factory photos", fill=(240, 220, 80), font=font)
    colors = {
        0: (80, 180, 255),
        1: (255, 180, 60),
        2: (180, 120, 255),
        3: (255, 80, 80),
        4: (255, 140, 40),
        5: (80, 220, 140),
        6: (220, 220, 80),
    }
    for slot, (image_bgr, yolo_rows, caption) in enumerate(examples):
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        tile = Image.fromarray(rgb).resize((cell_w, cell_h), Image.Resampling.BILINEAR)
        tile_draw = ImageDraw.Draw(tile)
        for class_id, cx, cy, bw, bh in yolo_rows:
            x1 = int((cx - bw / 2) * cell_w)
            y1 = int((cy - bh / 2) * cell_h)
            x2 = int((cx + bw / 2) * cell_w)
            y2 = int((cy + bh / 2) * cell_h)
            tile_draw.rectangle((x1, y1, x2, y2), outline=colors[class_id], width=2)
        tile_draw.rectangle((0, 0, cell_w, 16), fill=(0, 0, 0))
        tile_draw.text((4, 2), caption, fill=colors[yolo_rows[0][0]], font=font)
        col = slot % columns
        row = slot // columns
        canvas.paste(tile, (col * cell_w, title_h + row * cell_h))
    # The eighth cell states what the grid is. The tile size is independent of the source frame.
    note = Image.new("RGB", (cell_w, cell_h), (18, 20, 22))
    note_draw = ImageDraw.Draw(note)
    note_draw.text((8, 16), "Not the 6,800-image", fill=(230, 230, 230), font=font)
    note_draw.text((8, 36), "factory target.", fill=(230, 230, 230), font=font)
    note_draw.text((8, 64), "Not a 0.93 mAP claim.", fill=(230, 230, 230), font=font)
    note_draw.text((8, 92), "Wikimedia photos stay", fill=(230, 230, 230), font=font)
    note_draw.text((8, 112), "in data/train|val|test.", fill=(230, 230, 230), font=font)
    canvas.paste(note, (3 * cell_w, title_h + cell_h))
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, format="JPEG", quality=90, subsampling=0)


def generate(output: Path, per_class: int, master_seed: int, preview: Path, preview_copy: Path | None) -> dict[str, int]:
    if per_class < 1:
        raise ValueError("per_class must be positive")
    images_dir = output / "images"
    labels_dir = output / "labels"
    splits_dir = output / "splits"
    _clear_files(images_dir)
    _clear_files(labels_dir)
    _clear_files(splits_dir)
    records: list[dict] = []
    preview_examples = []
    hashes: dict[str, str] = {}
    for class_id, class_name in enumerate(CLASS_NAMES):
        for index in range(per_class):
            image, _mask, rows, seed = render_enclosure(class_id, index, master_seed)
            filename = f"{class_name}_{index:04d}.jpg"
            image_path = images_dir / filename
            label_path = labels_dir / f"{class_name}_{index:04d}.txt"
            _write_jpeg(image_path, image)
            _write_label(label_path, rows)
            digest = sha256_file(image_path)
            if digest in hashes:
                raise RuntimeError(f"duplicate pixels: {filename} matches {hashes[digest]}")
            hashes[digest] = filename
            label_text = label_path.read_text(encoding="utf-8").strip()
            records.append(
                {
                    "filename": filename,
                    "relative_path": (images_dir / filename).resolve().relative_to(ROOT).as_posix(),
                    "class_id": class_id,
                    "class_name": class_name,
                    "seed": seed,
                    "split": "",
                    "sha256": digest,
                    "width": WIDTH,
                    "height": HEIGHT,
                    "n_boxes": len(rows),
                    "yolo": label_text,
                    "domain": "synthetic",
                    "generator": "scripts/synthesize_enclosures.py",
                    "note": NOTE,
                }
            )
            if index == 0:
                preview_examples.append((image, rows, f"{class_id} {class_name}"))
            if (index + 1) % 25 == 0:
                print(f"{class_name} {index + 1}/{per_class}", flush=True)
    if len({int(record["seed"]) for record in records}) != len(records):
        raise RuntimeError("generation seeds are not unique")
    assign_splits(records, seed=master_seed)
    fieldnames = [
        "filename",
        "relative_path",
        "class_id",
        "class_name",
        "seed",
        "split",
        "sha256",
        "width",
        "height",
        "n_boxes",
        "yolo",
        "domain",
        "generator",
        "note",
    ]
    with (output / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    with (output / "splits.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["filename", "split", "seed", "class_id", "class_name", "sha256", "n_boxes", "domain"],
        )
        writer.writeheader()
        for record in records:
            writer.writerow({key: record[key] for key in writer.fieldnames})
    for split_name in ("train", "val", "test"):
        names = sorted(record["relative_path"] for record in records if record["split"] == split_name)
        (splits_dir / f"{split_name}.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    write_preview(preview_examples, preview)
    if preview_copy is not None:
        preview_copy.parent.mkdir(parents=True, exist_ok=True)
        preview_copy.write_bytes(preview.read_bytes())
    counts = {name: 0 for name in CLASS_NAMES}
    split_counts = {"train": 0, "val": 0, "test": 0}
    for record in records:
        counts[record["class_name"]] += 1
        split_counts[record["split"]] += 1
    print(f"unique_images {len(records)}", flush=True)
    for name in CLASS_NAMES:
        print(f"class {name} {counts[name]}", flush=True)
    for split_name, count in split_counts.items():
        print(f"split {split_name} {count}", flush=True)
    return {"images": len(records), **counts, **split_counts}


def main() -> int:
    parser = argparse.ArgumentParser(description="Draw synthetic switchgear enclosures and split them.")
    parser.add_argument("--per-class", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="data/synthetic")
    parser.add_argument("--preview", default="outputs/synthetic_enclosure_preview.jpg")
    parser.add_argument("--preview-copy", default="/cursor/stores/self/media/synthetic_enclosure_preview.jpg")
    args = parser.parse_args()
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    preview = Path(args.preview)
    if not preview.is_absolute():
        preview = ROOT / preview
    preview_copy = Path(args.preview_copy) if args.preview_copy else None
    # This script must not write the Wikimedia split.
    resolved = output.resolve()
    for name in ("raw", "cleaned", "annotated", "train", "val", "test"):
        banned = (ROOT / "data" / name).resolve()
        if resolved == banned or banned in resolved.parents:
            print(f"Refusing to write into the Wikimedia tree: {output}")
            return 1
    generate(output, args.per_class, args.seed, preview, preview_copy)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
