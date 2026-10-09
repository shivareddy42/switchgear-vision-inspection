#!/usr/bin/env python3
"""Clean shop-inspection renders of switchgear cabinets with local defects.

Procedural camera model (not a flat diagram, not a photograph). Every YOLO
box is the bounding box of the defect mask drawn for that image.

Determinism
-----------
- Image seed ``s`` drives only that image (``numpy.random.Generator(s)``).
- Class bundles come from ``PLAN_SEED`` (see ``build_plan``).
- Train/val/test is a shuffle of those renders with ``SPLIT_SEED`` 42,
  applied before any augmentation. This script does not augment further.
- Final sensor noise uses ``Generator(s + NOISE_SEED_OFFSET)``.

The look follows the geometry and materials of two licensed reference
photos (gray aisle switchgear; straight copper busbars on dark supports).
It does not copy those frames, and it is not a rusted, outdoor, or
abandoned installation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

GENERATOR_VERSION = "shop-inspect-1.0.0"
PLAN_SEED = 20261008
SPLIT_SEED = 42
NOISE_SEED_OFFSET = 99991
TEXTURE_NOISE_SEED = 12345
IMAGE_W = 960
IMAGE_H = 640
MIN_IMAGES = 700
MIN_PER_CLASS = 100
JPEG_QUALITY = 92

CLASS_NAMES = [
    "scratch",
    "dent",
    "weld_porosity",
    "weld_crack",
    "corrosion",
    "misaligned_busbar",
    "missing_or_loose_component",
]

# Surface ids written into the id buffer. Defect paint tests these.
ID_FLOOR = 1
ID_WALL = 2
ID_CEIL = 3
ID_DOOR = 10
ID_FRAME = 11
ID_SIDE = 12
ID_WELD0 = 30
ID_INTERIOR = 40
ID_BUS = 50

FONT_REG = "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"

BOX_COLORS = [
    (40, 210, 230),
    (240, 150, 40),
    (210, 80, 230),
    (230, 60, 60),
    (70, 180, 70),
    (60, 130, 240),
    (230, 200, 40),
]


def srgb_to_lin(c):
    a = np.asarray(c, np.float32)
    return np.power(np.clip(a, 0.0, 1.0), 2.2).astype(np.float32)


def normalize(v):
    v = np.asarray(v, np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


def unit_rows(v):
    n = np.sqrt(np.sum(v * v, axis=-1, keepdims=True)) + 1e-8
    return v / n


# ---------------------------------------------------------------------------
# Shared tileable noise (fixed seed, identical in every process)
# ---------------------------------------------------------------------------

_LOW = None
_HIGH = None


def _ensure_noise():
    global _LOW, _HIGH
    if _LOW is not None:
        return
    rng = np.random.default_rng(TEXTURE_NOISE_SEED)
    n = rng.random((512, 512)).astype(np.float32)
    low = cv2.GaussianBlur(n, (0, 0), 14)
    high = n - cv2.GaussianBlur(n, (0, 0), 1.4)
    _LOW = ((low - low.mean()) / (low.std() + 1e-6)).astype(np.float32)
    _HIGH = ((high - high.mean()) / (high.std() + 1e-6)).astype(np.float32)


def sample_tex(tex, u, v, scale, phase):
    h, w = tex.shape
    x = np.mod((u * scale + phase[0]) * (w - 1), w - 1.001)
    y = np.mod((v * scale + phase[1]) * (h - 1), h - 1.001)
    return tex[y.astype(np.int32), x.astype(np.int32)]


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------

class Camera:
    def __init__(self, eye, target, fov_deg, width, height, roll=0.0):
        self.eye = np.asarray(eye, np.float32)
        self.W = int(width)
        self.H = int(height)
        forward = normalize(np.asarray(target, np.float32) - self.eye)
        world_up = np.array([0.0, 1.0, 0.0], np.float32)
        right = np.cross(forward, world_up)
        if np.linalg.norm(right) < 1e-5:
            world_up = np.array([0.0, 0.0, 1.0], np.float32)
            right = np.cross(forward, world_up)
        right = normalize(right)
        up = normalize(np.cross(right, forward))
        if abs(roll) > 1e-6:
            c, s = np.cos(roll), np.sin(roll)
            right, up = c * right + s * up, -s * right + c * up
            right = normalize(right)
            up = normalize(up)
        self.forward = forward.astype(np.float32)
        self.right = right.astype(np.float32)
        self.up = up.astype(np.float32)
        self.f = float((self.W / 2.0) / np.tan(np.deg2rad(fov_deg) / 2.0))

    def project(self, pts):
        rel = np.asarray(pts, np.float32) - self.eye
        z = rel @ self.forward
        x = rel @ self.right
        y = rel @ self.up
        z_safe = np.where(np.abs(z) < 1e-6, 1e-6, z)
        sx = self.W / 2.0 + self.f * x / z_safe
        sy = self.H / 2.0 - self.f * y / z_safe
        return np.stack([sx, sy], axis=-1).astype(np.float32), z.astype(np.float32)


class Buffers:
    def __init__(self, w, h):
        self.w = w
        self.h = h
        self.color = np.zeros((h, w, 3), np.float32)
        self.zbuf = np.full((h, w), np.inf, np.float32)
        self.idbuf = np.zeros((h, w), np.int16)
        self.masks = np.zeros((7, h, w), np.uint8)
        self._dirs = None

    def dirs(self, cam: Camera):
        if self._dirs is None:
            xs = (np.arange(cam.W, dtype=np.float32) + 0.5 - cam.W / 2.0) / cam.f
            ys = -(np.arange(cam.H, dtype=np.float32) + 0.5 - cam.H / 2.0) / cam.f
            dirs = (
                cam.right.reshape(1, 1, 3) * xs.reshape(1, -1, 1)
                + cam.up.reshape(1, 1, 3) * ys.reshape(-1, 1, 1)
                + cam.forward.reshape(1, 1, 3)
            )
            self._dirs = dirs.astype(np.float32)
        return self._dirs


def plane_t(cam, origin, normal, dirs):
    num = float(np.dot(np.asarray(origin, np.float32) - cam.eye, normal))
    den = dirs @ np.asarray(normal, np.float32)
    t = np.full(den.shape, np.inf, np.float32)
    ok = np.abs(den) > 1e-6
    t[ok] = (num / den[ok]).astype(np.float32)
    t[t < 0.08] = np.inf
    return t


def shade_points(P, N, albedo, spec_strength, shininess, spec_tint, lights, eye, spec_mul=None):
    """Lambert + Blinn-Phong. ``albedo`` is linear RGB."""
    V = eye.reshape(1, 1, 3) - P
    V = unit_rows(V)
    if N.ndim == 1:
        N = np.broadcast_to(N, P.shape).astype(np.float32)
    if albedo.ndim == 1:
        rgb = np.broadcast_to(albedo, P.shape).astype(np.float32) * 0.20
    else:
        rgb = albedo * 0.20
    spec_tint = np.asarray(spec_tint, np.float32)
    for L, color, inten in lights:
        L = np.asarray(L, np.float32)
        color = np.asarray(color, np.float32)
        ndl = np.clip(np.sum(N * L, axis=-1, keepdims=True), 0.0, 1.0)
        H = unit_rows(L.reshape(1, 1, 3) + V)
        ndh = np.clip(np.sum(N * H, axis=-1, keepdims=True), 0.0, 1.0)
        spec = np.power(ndh, shininess) * spec_strength
        if spec_mul is not None:
            spec = spec * spec_mul
        rgb = rgb + albedo * (ndl * color * inten) + spec * spec_tint * color * inten
    return rgb.astype(np.float32)


def blit_quad(buf, cam, origin, axis_u, axis_v, normal, rgb, surface_id, mask_layers=None):
    """Warp a linear RGB texture onto a planar quad and z-composite it.

    Texture row 0 / col 0 is the origin corner. ``axis_u`` runs along +col,
    ``axis_v`` along +row. ``mask_layers`` is a list of (class_id, uint8 HxW).
    """
    normal = np.asarray(normal, np.float32)
    origin = np.asarray(origin, np.float32)
    axis_u = np.asarray(axis_u, np.float32)
    axis_v = np.asarray(axis_v, np.float32)
    if float(np.dot(normal, cam.eye - origin)) <= 1e-4:
        return
    th, tw = rgb.shape[:2]
    corners = np.stack([
        origin,
        origin + axis_u,
        origin + axis_u + axis_v,
        origin + axis_v,
    ]).astype(np.float32)
    proj, z = cam.project(corners)
    if np.any(z < 0.20) or not np.all(np.isfinite(proj)):
        return
    if np.any(np.abs(proj) > 8000):
        return
    area = cv2.contourArea(proj.reshape(-1, 1, 2))
    if area < 12:
        return
    src = np.array(
        [[0, 0], [tw - 1, 0], [tw - 1, th - 1], [0, th - 1]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(src, proj.astype(np.float32))
    alpha = np.ones((th, tw, 1), np.float32)
    rgba = np.concatenate([np.ascontiguousarray(rgb), alpha], axis=-1)
    warped = cv2.warpPerspective(
        rgba,
        M,
        (cam.W, cam.H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )
    coverage = warped[:, :, 3] > 0.55
    t = plane_t(cam, origin, normal, buf.dirs(cam))
    valid = coverage & np.isfinite(t) & (t < buf.zbuf)
    if not np.any(valid):
        return
    buf.color[valid] = warped[:, :, :3][valid]
    buf.zbuf[valid] = t[valid]
    buf.idbuf[valid] = np.int16(surface_id)
    if mask_layers:
        for cls, mtex in mask_layers:
            if mtex is None:
                continue
            wm = cv2.warpPerspective(
                mtex,
                M,
                (cam.W, cam.H),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            hit = valid & (wm > 0)
            buf.masks[cls][hit] = 255


def render_sphere(buf, cam, center, radius, albedo_lin, spec, shin, spec_tint, lights, mask_class=None):
    center = np.asarray(center, np.float32)
    rel = center - cam.eye
    z = float(np.dot(rel, cam.forward))
    if z < 0.18:
        return
    x = float(np.dot(rel, cam.right))
    y = float(np.dot(rel, cam.up))
    sx = cam.W / 2.0 + cam.f * x / z
    sy = cam.H / 2.0 - cam.f * y / z
    r_pix = cam.f * radius / z
    if r_pix < 1.15:
        return
    x0 = max(0, int(np.floor(sx - r_pix - 1)))
    x1 = min(cam.W, int(np.ceil(sx + r_pix + 2)))
    y0 = max(0, int(np.floor(sy - r_pix - 1)))
    y1 = min(cam.H, int(np.ceil(sy + r_pix + 2)))
    if x0 >= x1 or y0 >= y1:
        return
    xs = np.arange(x0, x1, dtype=np.float32) + 0.5
    ys = np.arange(y0, y1, dtype=np.float32) + 0.5
    X, Y = np.meshgrid(xs, ys)
    dx = (X - sx) / r_pix
    dy = (sy - Y) / r_pix
    r2 = dx * dx + dy * dy
    inside = r2 <= 1.0
    nz = np.sqrt(np.clip(1.0 - r2, 0.0, 1.0))
    N = (
        dx[..., None] * cam.right
        + dy[..., None] * cam.up
        + nz[..., None] * cam.forward
    )
    zpix = z - nz * radius
    closer = inside & (zpix < buf.zbuf[y0:y1, x0:x1])
    if not np.any(closer):
        return
    alb = np.broadcast_to(albedo_lin, (y1 - y0, x1 - x0, 3)).astype(np.float32).copy()
    # Dark hex-ish socket and a slot, so the head reads as a fastener.
    socket = inside & (r2 < 0.20) & (nz > 0.35)
    slot = inside & (np.abs(dy) < 0.11) & (np.abs(dx) < 0.42) & (nz > 0.25)
    alb[socket] *= 0.28
    alb[slot] *= 0.45
    P = cam.eye.reshape(1, 1, 3) + np.array([0, 0, 0], np.float32)
    # View vector approximates the camera direction for these small heads.
    P = np.broadcast_to(center, alb.shape).astype(np.float32).copy()
    rgb = shade_points(P, N, alb, spec, shin, spec_tint, lights, cam.eye)
    sub_c = buf.color[y0:y1, x0:x1]
    sub_z = buf.zbuf[y0:y1, x0:x1]
    sub_c[closer] = rgb[closer]
    sub_z[closer] = zpix[closer]
    buf.color[y0:y1, x0:x1] = sub_c
    buf.zbuf[y0:y1, x0:x1] = sub_z
    if mask_class is not None:
        sub_m = buf.masks[mask_class][y0:y1, x0:x1]
        sub_m[closer] = 255
        buf.masks[mask_class][y0:y1, x0:x1] = sub_m


def _surface_sel(idbuf, surface_ids):
    sel = np.zeros(idbuf.shape, np.bool_)
    for sid in surface_ids:
        sel |= idbuf == sid
    return sel


def paint_polyline(buf, pts, surface_ids, thickness, cls, kind):
    pts = np.asarray(pts, np.float32)
    if len(pts) < 2:
        return 0
    if np.any(~np.isfinite(pts)) or np.any(np.abs(pts) > 8000):
        return 0
    layer = np.zeros((buf.h, buf.w), np.uint8)
    cv2.polylines(layer, [np.round(pts).astype(np.int32)], False, 255, int(thickness), cv2.LINE_8)
    sel = (layer > 0) & _surface_sel(buf.idbuf, surface_ids)
    n = int(sel.sum())
    if n == 0:
        return 0
    if kind == "scratch":
        buf.color[sel] = buf.color[sel] * 0.20 + np.array([0.012, 0.012, 0.014], np.float32)
        dil = cv2.dilate(layer, np.ones((3, 3), np.uint8))
        sh = (dil > 0) & (layer == 0) & _surface_sel(buf.idbuf, surface_ids)
        buf.color[sh] = np.clip(buf.color[sh] * 1.55 + 0.035, 0.0, 4.0)
        buf.masks[cls][sel | sh] = 255
    elif kind == "crack":
        buf.color[sel] = buf.color[sel] * 0.12 + np.array([0.01, 0.008, 0.006], np.float32)
        dil = cv2.dilate(layer, np.ones((3, 3), np.uint8))
        sh = (dil > 0) & (layer == 0) & _surface_sel(buf.idbuf, surface_ids)
        buf.color[sh] = np.clip(buf.color[sh] * 1.35 + 0.02, 0.0, 4.0)
        buf.masks[cls][sel | sh] = 255
    return n


def paint_pits(buf, centers, radii, surface_ids, cls):
    layer = np.zeros((buf.h, buf.w), np.uint8)
    rim = np.zeros_like(layer)
    for c, r in zip(centers, radii):
        if not np.isfinite(c).all():
            continue
        rr = int(max(2, round(float(r))))
        cc = (int(round(float(c[0]))), int(round(float(c[1]))))
        cv2.circle(layer, cc, rr, 255, -1, lineType=cv2.LINE_8)
        cv2.circle(rim, cc, rr + 1, 255, 1, lineType=cv2.LINE_8)
    sel = (layer > 0) & _surface_sel(buf.idbuf, surface_ids)
    if not np.any(sel):
        return 0
    buf.color[sel] = buf.color[sel] * 0.10 + np.array([0.008, 0.007, 0.006], np.float32)
    rsel = (rim > 0) & (layer == 0) & _surface_sel(buf.idbuf, surface_ids)
    buf.color[rsel] = np.clip(buf.color[rsel] * 1.5 + 0.04, 0.0, 4.0)
    buf.masks[cls][sel | rsel] = 255
    return int((sel | rsel).sum())


def paint_hole(buf, center, radius_px, surface_ids, cls):
    if not np.isfinite(center).all():
        return 0
    layer = np.zeros((buf.h, buf.w), np.uint8)
    rim = np.zeros_like(layer)
    cc = (int(round(float(center[0]))), int(round(float(center[1]))))
    r = int(max(3, round(float(radius_px))))
    cv2.circle(layer, cc, r, 255, -1, lineType=cv2.LINE_8)
    cv2.circle(rim, cc, r + 2, 255, 2, lineType=cv2.LINE_8)
    sel = (layer > 0) & _surface_sel(buf.idbuf, surface_ids)
    if not np.any(sel):
        return 0
    # Countersunk hole: darker toward the middle, metal rim around it.
    yy, xx = np.mgrid[0:buf.h, 0:buf.w]
    dist = np.sqrt((xx - cc[0]) ** 2 + (yy - cc[1]) ** 2)
    fall = np.clip(dist / (r + 1e-6), 0.0, 1.0)
    shade = (0.05 + 0.20 * fall)[..., None]
    buf.color[sel] = buf.color[sel] * shade[sel] + np.array([0.01, 0.01, 0.012], np.float32)
    rsel = (rim > 0) & (layer == 0) & _surface_sel(buf.idbuf, surface_ids)
    buf.color[rsel] = np.clip(buf.color[rsel] * 1.25 + 0.03, 0.0, 4.0)
    buf.masks[cls][sel | rsel] = 255
    return int((sel | rsel).sum())


def add_floor_shadow(buf, cam, footprint, light_dir):
    cast = -np.asarray(light_dir, np.float32)
    cast[1] = 0.0
    cast = cast / (np.linalg.norm(cast) + 1e-8) * 0.55
    pts = np.concatenate([footprint, footprint + cast], axis=0).astype(np.float32)
    proj, z = cam.project(pts)
    if np.any(z < 0.15) or np.any(~np.isfinite(proj)):
        return
    hull = cv2.convexHull(proj.reshape(-1, 1, 2).astype(np.float32))
    sh = np.zeros((buf.h, buf.w), np.float32)
    cv2.fillConvexPoly(sh, np.round(hull.reshape(-1, 2)).astype(np.int32), 1.0)
    sh = cv2.GaussianBlur(sh, (0, 0), 13)
    floor = buf.idbuf == ID_FLOOR
    buf.color[floor] *= (1.0 - 0.42 * sh[floor, None])


# ---------------------------------------------------------------------------
# Metal texture
# ---------------------------------------------------------------------------

def make_metal_texture(
    origin, axis_u, axis_v, normal, tw, th, lights, eye,
    base_srgb, brushed, phase, spec_strength, shininess, spec_tint,
    features,
):
    """Shade one rectangular metal face. ``features`` may include bevel, ao,
    louvers, joint, nameplate, bolt shadows, a dent, and a corrosion patch.
    Returns linear RGB and a list of (class_id, mask).
    """
    _ensure_noise()
    us = np.linspace(0.0, 1.0, tw, dtype=np.float32)
    vs = np.linspace(0.0, 1.0, th, dtype=np.float32)
    U, V = np.meshgrid(us, vs)
    origin = np.asarray(origin, np.float32)
    axis_u = np.asarray(axis_u, np.float32)
    axis_v = np.asarray(axis_v, np.float32)
    normal = normalize(normal)
    P = origin + U[..., None] * axis_u + V[..., None] * axis_v
    tu = axis_u / (np.linalg.norm(axis_u) + 1e-8)
    tv = axis_v / (np.linalg.norm(axis_v) + 1e-8)
    N = np.broadcast_to(normal, P.shape).astype(np.float32).copy()

    if features.get("bevel", True):
        bw = features.get("bevel_w", 0.045)
        tilt_u = np.zeros_like(U)
        tilt_v = np.zeros_like(V)
        left = U < bw
        right = U > 1.0 - bw
        top = V < bw
        bot = V > 1.0 - bw
        tilt_u[left] = -((bw - U[left]) / bw)
        tilt_u[right] = (U[right] - (1.0 - bw)) / bw
        tilt_v[top] = -((bw - V[top]) / bw)
        tilt_v[bot] = (V[bot] - (1.0 - bw)) / bw
        N = N + (tilt_u[..., None] * 1.35) * tu + (tilt_v[..., None] * 1.35) * tv

    if features.get("weld_crown"):
        across = tu
        ang = (U - 0.5) * np.pi
        N = normal * np.cos(ang)[..., None] + across * np.sin(ang)[..., None]

    low = sample_tex(_LOW, U, V, features.get("noise_scale", 3.5), phase)
    high = sample_tex(_HIGH, U, V, features.get("noise_scale", 3.5) * 2.2, phase)
    base = srgb_to_lin(base_srgb)
    if brushed:
        lines = np.sin((U if features.get("brush_along", "u") == "u" else V) * features.get("brush_freq", 420.0) + phase[0] * 6.0)
        alb = base * (0.94 + 0.025 * lines + 0.02 * high)[..., None]
    else:
        alb = base * (0.95 + 0.035 * low + 0.012 * high)[..., None]
    alb = np.clip(alb, 0.0, 1.5).astype(np.float32)

    if features.get("ao", True):
        edge = np.minimum(np.minimum(U, 1.0 - U), np.minimum(V, 1.0 - V))
        ao = np.clip(edge / 0.035, 0.62, 1.0)
        alb *= ao[..., None]

    spec_mul = np.ones((th, tw, 1), np.float32)
    masks = {}

    louvers = features.get("louvers")
    if louvers:
        u0, u1, v0, count, slot_h, gap = louvers
        for i in range(count):
            y0 = v0 + i * (slot_h + gap)
            slot = (V >= y0) & (V <= y0 + slot_h) & (U >= u0) & (U <= u1)
            alb[slot] *= 0.18
            lip = (V > y0 + slot_h) & (V < y0 + slot_h + slot_h * 0.35) & (U >= u0) & (U <= u1)
            alb[lip] *= 1.22
            N[slot] = normal * 0.2 - tv * 0.8

    joint = features.get("joint_v")
    if joint is not None:
        band = np.abs(V - joint) < features.get("joint_half", 0.008)
        alb[band] *= 0.55
        N[band] = unit_rows(N[band] - tv * 0.4)

    plate = features.get("nameplate")
    if plate:
        pu, pv, pw, ph = plate
        on = (np.abs(U - pu) < pw / 2) & (np.abs(V - pv) < ph / 2)
        alb[on] *= 1.08
        alb[on] = alb[on] * 0.92 + srgb_to_lin(np.array([0.78, 0.78, 0.76], np.float32)) * 0.08
        # Rivets, not text.
        for du, dv in ((-0.42, -0.38), (0.42, -0.38), (-0.42, 0.38), (0.42, 0.38)):
            rr = ((U - (pu + du * pw)) / (pw * 0.06)) ** 2 + ((V - (pv + dv * ph)) / (ph * 0.10)) ** 2
            riv = rr < 1.0
            alb[riv] *= 0.55

    for bu, bv, bru in features.get("bolt_shadows", []):
        du = (U - bu) / bru
        dv = (V - bv) / (bru * features.get("bolt_shadow_aspect", 1.15))
        sh = np.clip(1.0 - (du * du + dv * dv), 0.0, 1.0) ** 1.4
        alb *= (1.0 - 0.38 * sh)[..., None]

    dent = features.get("dent")
    if dent is not None:
        cu, cv, au, av, ang, strength = dent
        du = U - cu
        dv = V - cv
        ca, sa = np.cos(ang), np.sin(ang)
        lu = (du * ca + dv * sa) / max(au, 1e-4)
        lv = (-du * sa + dv * ca) / max(av, 1e-4)
        r2 = lu * lu + lv * lv
        inside = r2 <= 1.0
        fall = np.clip(1.0 - r2, 0.0, 1.0) * inside
        radial = lu[..., None] * tu + lv[..., None] * tv
        radial = unit_rows(radial)
        tilt_amt = (strength * np.sqrt(np.clip(r2, 0.0, 1.0)) * inside)[..., None]
        N = N + tilt_amt * (-radial)
        alb = alb * (1.0 - 0.42 * fall[..., None])
        # Rim catch on the light side of the dent.
        rim = inside & (r2 > 0.55) & (r2 < 1.0)
        alb[rim] *= 1.08
        m = np.zeros((th, tw), np.uint8)
        m[inside] = 255
        masks[1] = m

    corr = features.get("corrosion")
    if corr is not None:
        cu, cv, rad, blob_seed = corr
        rr = np.random.default_rng(int(blob_seed))
        m = np.zeros((th, tw), np.uint8)
        for _k in range(6):
            ang = float(rr.random() * np.pi * 2)
            dist = float(rr.random() * rad * 0.55)
            rx = rad * float(rr.uniform(0.40, 0.95))
            ry = rad * float(rr.uniform(0.32, 0.80))
            cu2 = cu + np.cos(ang) * dist
            cv2 = cv + np.sin(ang) * dist
            du = (U - cu2) / max(rx, 1e-4)
            dv = (V - cv2) / max(ry, 1e-4)
            m[(du * du + dv * dv) <= 1.0] = 255
        on = m > 0
        rust = srgb_to_lin(np.array([0.62, 0.34, 0.16], np.float32))
        rust2 = srgb_to_lin(np.array([0.42, 0.24, 0.12], np.float32))
        mix = np.clip(0.5 + 0.5 * high, 0.0, 1.0)
        alb[on] = (rust * mix[on, None] + rust2 * (1.0 - mix[on, None])) * (0.85 + 0.25 * low[on, None])
        spec_mul[on] = 0.04
        # A few darker pits inside the patch so it reads as oxidation, not a sticker.
        pit = on & (high > 1.15)
        alb[pit] *= 0.72
        masks[4] = m

    N = unit_rows(N)
    # Roughness breakup so the highlight is not a plastic blob.
    spec_mul = spec_mul * np.clip(0.82 + 0.18 * high, 0.45, 1.25)[..., None]
    rgb = shade_points(
        P, N, alb, spec_strength, shininess, spec_tint, lights, np.asarray(eye, np.float32), spec_mul
    )
    return np.clip(rgb, 0.0, 8.0).astype(np.float32), masks


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

def _bolt_uvs():
    """Inset from the door edge so heads sit on visible skin, clear of the louvers."""
    pts = []
    for v in (0.50, 0.66, 0.82):
        pts.append((0.18, v))
        pts.append((0.82, v))
    pts.append((0.50, 0.78))
    return pts


def _interior_bolt_uvs():
    """On the open back panel, inset from the frame and clear of the busbars."""
    return [(0.30, 0.74), (0.62, 0.74), (0.30, 0.20), (0.70, 0.20)]


def sample_params(seed, classes):
    """Every visual choice for one image, drawn in a fixed order from ``seed``."""
    rng = np.random.default_rng(int(seed))
    classes = sorted({int(c) for c in classes})
    p = {"seed": int(seed), "classes": classes}
    if 5 in classes:
        p["mode"] = "open"
    elif any(c in classes for c in (2, 3, 6)):
        p["mode"] = "close"
    else:
        p["mode"] = "aisle" if rng.random() < 0.36 else "single"

    p["brushed"] = bool(rng.random() < 0.22)
    p["gray"] = float(rng.uniform(0.46, 0.60))
    p["floor"] = "red" if rng.random() < 0.40 else "concrete"
    p["w"] = float(rng.uniform(0.72, 0.92))
    p["h"] = float(rng.uniform(1.85, 2.15))
    p["d"] = float(rng.uniform(0.48, 0.68))
    sign = -1.0 if rng.random() < 0.5 else 1.0
    if p["mode"] == "open":
        # Camera on the right so the left-hung door does not block the bars.
        sign = 1.0
    p["yaw_sign"] = sign
    if p["mode"] == "aisle":
        p["yaw"] = sign * float(rng.uniform(0.40, 0.58))
    elif p["mode"] == "open":
        p["yaw"] = float(rng.uniform(0.22, 0.40))
    else:
        p["yaw"] = sign * float(rng.uniform(0.26, 0.46))
    p["pitch"] = float(rng.uniform(-0.07, 0.08))
    p["roll"] = float(rng.uniform(-0.015, 0.015))
    p["fov"] = float(rng.uniform(40.0, 48.0))
    if p["mode"] == "close":
        p["fill"] = float(rng.uniform(0.78, 0.90))
    elif p["mode"] == "aisle":
        p["fill"] = float(rng.uniform(0.86, 0.96))
    else:
        p["fill"] = float(rng.uniform(0.70, 0.84))
    p["open_angle"] = float(rng.uniform(np.deg2rad(100), np.deg2rad(118)))
    p["exposure"] = float(rng.uniform(0.95, 1.22))
    p["blur"] = float(rng.uniform(0.30, 0.65))
    p["noise"] = float(rng.uniform(0.007, 0.016))
    p["vignette"] = float(rng.uniform(0.10, 0.20))
    p["n_sections"] = int(rng.integers(3, 5))
    p["louver_count"] = int(rng.integers(5, 8))
    p["phase"] = (float(rng.random()), float(rng.random()))
    lx = float(rng.uniform(-0.45, 0.55))
    p["key"] = normalize([lx, float(rng.uniform(0.75, 1.15)), float(rng.uniform(0.35, 0.75))])
    p["fill_light"] = normalize([-lx * 0.6, 0.45, 0.55])
    p["handle_side"] = 1.0  # latch side, opposite the left hinge

    bolts = _bolt_uvs()
    p["bolt_uvs"] = bolts
    p["fastener_index"] = int(rng.integers(0, len(bolts)))
    p["fastener_loose"] = bool(rng.random() < 0.5)
    p["loose_du"] = float(rng.choice([-1.0, 1.0])) * float(rng.uniform(0.09, 0.13))
    p["loose_dv"] = float(rng.choice([-1.0, 1.0])) * float(rng.uniform(0.035, 0.06))

    # Skin defects stay in separate zones so a scratch is not buried in a dent.
    p["scratch"] = {
        "u0": float(rng.uniform(0.22, 0.38)),
        "v0": float(rng.uniform(0.46, 0.58)),
        "u1": float(rng.uniform(0.55, 0.78)),
        "v1": float(rng.uniform(0.62, 0.74)),
        "jag": float(rng.uniform(0.008, 0.02)),
        "n": int(rng.integers(4, 7)),
    }
    p["dent"] = {
        "u": float(rng.uniform(0.48, 0.72)),
        "v": float(rng.uniform(0.50, 0.70)),
        "au": float(rng.uniform(0.07, 0.11)),
        "av": float(rng.uniform(0.05, 0.08)),
        "ang": float(rng.uniform(0, np.pi)),
        "strength": float(rng.uniform(1.4, 2.1)),
    }
    if p["mode"] == "aisle":
        p["dent"]["au"] *= 1.25
        p["dent"]["av"] *= 1.25
        p["scratch"]["u1"] = min(0.84, p["scratch"]["u1"] + 0.06)
    p["corrosion"] = {
        "u": float(rng.uniform(0.22, 0.70)),
        "v": float(rng.uniform(0.80, 0.90)),
        "rad": float(rng.uniform(0.035, 0.055) * (1.35 if p["mode"] == "aisle" else 1.0)),
        "blob_seed": int(rng.integers(0, 2**31 - 1)),
    }
    # The near vertical corner faces the camera. The far corner is edge-on.
    _ = int(rng.integers(0, 3))
    p["weld_which"] = 1 if p["yaw_sign"] > 0 else 0
    _ = int(rng.integers(0, 2))
    p["weld_which_b"] = p["weld_which"]
    p["weld_t0"] = float(rng.uniform(0.25, 0.55))
    p["weld_len"] = float(rng.uniform(0.12, 0.22))
    p["pit_n"] = int(rng.integers(5, 9))
    p["crack_n"] = int(rng.integers(8, 14))
    p["crack_amp"] = float(rng.uniform(0.18, 0.42))
    p["bus_phase"] = int(rng.integers(0, 3))
    p["bus_dy"] = float(rng.choice([-1.0, 1.0])) * float(rng.uniform(0.040, 0.055))
    if p["mode"] == "open":
        # Keep skin defects off the copper, on the open back panel.
        p["scratch"]["u0"], p["scratch"]["v0"] = 0.15, 0.16
        p["scratch"]["u1"], p["scratch"]["v1"] = 0.48, 0.30
        p["dent"]["u"], p["dent"]["v"] = 0.74, 0.16
        p["dent"]["au"], p["dent"]["av"] = 0.09, 0.055
        p["corrosion"]["u"], p["corrosion"]["v"] = 0.62, 0.90
        p["corrosion"]["rad"] = 0.040
    p["sections_jitter"] = float(rng.uniform(0.0, 0.01))
    if p["mode"] == "close" and (2 in classes or 3 in classes) and 6 in classes:
        right = p["weld_which"] == 1
        cands = [
            i for i, (u, v) in enumerate(p["bolt_uvs"])
            if 0.42 <= v <= 0.70 and ((u > 0.6) if right else (u < 0.4))
        ]
        if cands:
            p["fastener_index"] = cands[p["fastener_index"] % len(cands)]
    if p["mode"] == "close":
        # The close camera frames the fastener or the near weld, not the whole
        # door. Keep any skin defects inside that same patch.
        bu, bv = p["bolt_uvs"][p["fastener_index"]]
        p["dent"]["u"] = float(np.clip(bu + (0.10 if bu < 0.5 else -0.10), 0.22, 0.78))
        p["dent"]["v"] = float(np.clip(bv - 0.06, 0.40, 0.70))
        p["dent"]["au"] = 0.055
        p["dent"]["av"] = 0.040
        p["corrosion"]["u"] = float(np.clip(bu + ( -0.08 if bu > 0.5 else 0.08), 0.20, 0.80))
        p["corrosion"]["v"] = float(np.clip(bv + 0.05, 0.46, 0.74))
        p["corrosion"]["rad"] = 0.032
        p["scratch"]["u0"] = float(np.clip(min(bu, p["dent"]["u"]) - 0.02, 0.16, 0.50))
        p["scratch"]["v0"] = float(np.clip(bv - 0.02, 0.40, 0.62))
        p["scratch"]["u1"] = p["scratch"]["u0"] + 0.20
        p["scratch"]["v1"] = p["scratch"]["v0"] + 0.07
    return p


def lights_from(p):
    key = (p["key"], np.array([1.0, 0.98, 0.93], np.float32), 1.15)
    fill = (p["fill_light"], np.array([0.82, 0.88, 1.0], np.float32), 0.28)
    rim = (normalize([0.15, 0.55, -0.35]), np.array([0.9, 0.95, 1.0], np.float32), 0.22)
    return [key, fill, rim]


def paint_color(p, gain=1.0):
    g = np.clip(p["gray"] * gain, 0.35, 0.85)
    if p["brushed"]:
        g = np.clip(0.70 + 0.08 * (p["gray"] - 0.5), 0.64, 0.80)
        return np.array([g * 0.99, g, g * 0.98], np.float32), True
    return np.array([g * 0.985, g, g * 0.975], np.float32), False


# ---------------------------------------------------------------------------
# Scene geometry
# ---------------------------------------------------------------------------

def _face(buf, cam, origin, axis_u, axis_v, normal, tw, th, p, lights, base_srgb, brushed, surface_id, features, phase=None, material=None):
    features = dict(features)
    if material is None:
        spec_strength = 0.62 if brushed else 0.16
        shininess = 120.0 if brushed else 55.0
        spec_tint = np.array([1.0, 1.0, 1.0], np.float32) if brushed else np.array([0.9, 0.92, 0.95], np.float32)
    else:
        spec_strength, shininess, spec_tint = material
    rgb, masks = make_metal_texture(
        origin, axis_u, axis_v, normal, tw, th, lights, cam.eye,
        base_srgb, brushed, phase if phase is not None else p["phase"],
        spec_strength, shininess, np.asarray(spec_tint, np.float32),
        features,
    )
    layers = [(cls, m) for cls, m in masks.items()]
    blit_quad(buf, cam, origin, axis_u, axis_v, normal, rgb, surface_id, layers)


def _aa_box_faces(center, sx, sy, sz):
    """Axis-aligned box. Yields origin, axis_u, axis_v, normal for 6 faces.
    Sizes are half-extents.
    """
    c = np.asarray(center, np.float32)
    ex = np.array([1.0, 0.0, 0.0], np.float32)
    ey = np.array([0.0, 1.0, 0.0], np.float32)
    ez = np.array([0.0, 0.0, 1.0], np.float32)
    specs = [
        (ez, ex, ey, sz, sx, sy),
        (-ez, -ex, ey, sz, sx, sy),
        (ex, -ez, ey, sx, sz, sy),
        (-ex, ez, ey, sx, sz, sy),
        (ey, ex, -ez, sy, sx, sz),
        (-ey, ex, ez, sy, sx, sz),
    ]
    for normal, au, av, hn, hu, hv in specs:
        origin = c + normal * hn - au * hu - av * hv
        yield origin, au * (2 * hu), av * (2 * hv), normal


def _ray_t(dirs_comp, eye_comp, plane_value, looking_sign):
    """Intersect eye + t*dir with a axis-aligned plane. ``looking_sign`` is -1
    when the ray component must be negative to hit the plane from the front."""
    t = np.full(dirs_comp.shape, np.inf, np.float32)
    if looking_sign < 0:
        m = dirs_comp < -1e-4
    else:
        m = dirs_comp > 1e-4
    t[m] = (plane_value - eye_comp) / dirs_comp[m]
    t[(t < 0.08) | ~np.isfinite(t)] = np.inf
    return t


def add_room(buf, cam, p, lights):
    """Infinite shop box: floor, walls, ceiling. No uncovered black pixels."""
    _ensure_noise()
    dirs = buf.dirs(cam)
    eye = cam.eye
    t_floor = _ray_t(dirs[:, :, 1], eye[1], np.float32(0.0), -1)
    t_back = _ray_t(dirs[:, :, 2], eye[2], np.float32(-1.7), -1)
    t_ceil = _ray_t(dirs[:, :, 1], eye[1], np.float32(p["h"] + 1.05), +1)
    t_left = _ray_t(dirs[:, :, 0], eye[0], np.float32(-2.4), -1)
    t_right = _ray_t(dirs[:, :, 0], eye[0], np.float32(7.2), +1)
    stack = np.stack([t_floor, t_back, t_ceil, t_left, t_right], axis=0)
    nearest = np.min(stack, axis=0)
    which = np.argmin(stack, axis=0).astype(np.int16)
    valid = np.isfinite(nearest) & (nearest < 80.0)
    P = eye.reshape(1, 1, 3) + nearest[..., None] * dirs
    key = np.asarray(lights[0][0], np.float32)
    normals = np.zeros(P.shape, np.float32)
    normals[which == 0] = (0.0, 1.0, 0.0)
    normals[which == 1] = (0.0, 0.0, 1.0)
    normals[which == 2] = (0.0, -1.0, 0.0)
    normals[which == 3] = (1.0, 0.0, 0.0)
    normals[which == 4] = (-1.0, 0.0, 0.0)
    ndl = np.clip(np.sum(normals * key.reshape(1, 1, 3), axis=-1), 0.0, 1.0)

    if p["floor"] == "red":
        floor_alb = srgb_to_lin(np.array([0.50, 0.29, 0.24], np.float32))
    else:
        floor_alb = srgb_to_lin(np.array([0.52, 0.51, 0.49], np.float32))
    wall_alb = srgb_to_lin(np.array([0.60, 0.61, 0.59], np.float32))
    ceil_alb = srgb_to_lin(np.array([0.52, 0.53, 0.52], np.float32))

    fn = sample_tex(_LOW, np.mod(P[:, :, 0], 6.0) / 6.0, np.mod(P[:, :, 2] + 3.0, 6.0) / 6.0, 2.4, (0.15, 0.4))
    wn = sample_tex(_LOW, np.mod(P[:, :, 0] + 2.0, 6.0) / 6.0, np.mod(P[:, :, 1], 4.0) / 4.0, 1.8, (0.55, 0.2))
    alb = np.broadcast_to(wall_alb, P.shape).astype(np.float32).copy()
    floor_m = valid & (which == 0)
    wall_m = valid & ((which == 1) | (which == 3) | (which == 4))
    ceil_m = valid & (which == 2)
    alb[floor_m] = floor_alb * (0.93 + 0.07 * fn[floor_m, None])
    jx = np.abs(np.mod(P[:, :, 0] + 0.35, 1.25) - 0.625) < 0.010
    jz = np.abs(np.mod(P[:, :, 2] + 0.15, 1.25) - 0.625) < 0.010
    alb[floor_m & (jx | jz)] *= 0.80
    alb[wall_m] = wall_alb * (0.96 + 0.04 * wn[wall_m, None])
    top = np.clip((P[:, :, 1] - 0.4) / 2.2, 0.0, 1.0)
    alb[wall_m] *= (0.78 + 0.32 * top[wall_m, None])
    alb[ceil_m] = ceil_alb
    rgb = alb * (0.30 + 0.90 * ndl[..., None])
    for fx in (0.15, 1.7, 3.5):
        tube = ceil_m & (np.abs(P[:, :, 0] - fx) < 0.48) & (np.abs(P[:, :, 2] - 0.15) < 0.07)
        rgb[tube] = np.array([2.4, 2.35, 2.2], np.float32)
    buf.color[valid] = rgb[valid].astype(np.float32)
    buf.zbuf[valid] = nearest[valid]
    id_of = np.array([ID_FLOOR, ID_WALL, ID_CEIL, ID_WALL, ID_WALL], np.int16)
    buf.idbuf[valid] = id_of[which[valid]]


def _door_map(u, y, t, angle, hinge_x, hinge_z):
    c, s = np.cos(angle), np.sin(angle)
    x = hinge_x + u * c - t * s
    z = hinge_z + u * s + t * c
    return np.array([x, y, z], np.float32)


def _door_axes(angle):
    c, s = np.cos(angle), np.sin(angle)
    eu = np.array([c, 0.0, s], np.float32)
    et = np.array([-s, 0.0, c], np.float32)
    ey = np.array([0.0, 1.0, 0.0], np.float32)
    return eu, ey, et


def add_oriented_box(buf, cam, p, lights, center, su, sy, st, angle, hinge, base_srgb, brushed, surface_id, features):
    """Box in door-local (u, y, t) space. ``su, sy, st`` are half extents."""
    eu, ey, et = _door_axes(angle)
    hx, hz = hinge
    # center is (u, y, t)
    cu, cy, ct = center
    c = _door_map(cu, cy, ct, angle, hx, hz)
    faces = [
        (et, eu, ey, st, su, sy),
        (-et, -eu, ey, st, su, sy),
        (eu, -et, ey, su, st, sy),
        (-eu, et, ey, su, st, sy),
        (ey, eu, -et, sy, su, st),
        (-ey, eu, et, sy, su, st),
    ]
    for normal, au, av, hn, hu, hv in faces:
        origin = c + normal * hn - au * hu - av * hv
        _face(
            buf, cam, origin, au * (2 * hu), av * (2 * hv), normal,
            48, 96, p, lights, base_srgb, brushed, surface_id, features,
        )


def cabinet_layout(p, x0):
    """World-space measurements for one section."""
    w, h, d = p["w"], p["h"], p["d"]
    front = d / 2.0
    fb = 0.050
    plinth = 0.055
    x_l, x_r = x0, x0 + w
    y_b, y_t = plinth + fb * 0.35, h - fb * 0.55
    open_x0, open_x1 = x_l + fb, x_r - fb
    open_y0, open_y1 = y_b, y_t
    door_gap = 0.008
    door = {
        "u0": 0.0,
        "v0": 0.0,
        "w": (open_x1 - open_x0) - 2 * door_gap,
        "h": (open_y1 - open_y0) - 2 * door_gap,
        "hinge_x": open_x0 + door_gap,
        "hinge_z": front - 0.018,
        "y0": open_y0 + door_gap,
    }
    return {
        "x0": x0, "w": w, "h": h, "d": d, "front": front, "fb": fb,
        "plinth": plinth, "open": (open_x0, open_x1, open_y0, open_y1),
        "door": door,
    }


def add_cabinet(buf, cam, p, lights, x0, detailed, with_defects, angle_override=None):
    lay = cabinet_layout(p, x0)
    w, h, d = lay["w"], lay["h"], lay["d"]
    front = lay["front"]
    fb = lay["fb"]
    plinth = lay["plinth"]
    x_l = lay["x0"]
    x_r = x_l + w
    base_srgb, brushed = paint_color(p, gain=0.96 + 0.03 * ((int(x0 * 10) % 5) / 5.0))
    phase = (p["phase"][0] + x0 * 0.17, p["phase"][1])
    classes = set(p["classes"]) if with_defects else set()
    angle = 0.0 if angle_override is None else angle_override
    door_open = angle > 0.2

    # Plinth front and visible side.
    _face(
        buf, cam,
        np.array([x_l + 0.01, plinth, front + 0.004], np.float32),
        np.array([w - 0.02, 0, 0], np.float32),
        np.array([0, -plinth, 0], np.float32),
        np.array([0, 0, 1], np.float32),
        80, 24, p, lights, base_srgb * 0.72, brushed, ID_FRAME,
        {"bevel": True, "bevel_w": 0.08, "ao": True}, phase,
    )

    # Side panels.
    side_n = np.array([1.0, 0, 0], np.float32) if p["yaw_sign"] > 0 else np.array([-1.0, 0, 0], np.float32)
    if p["yaw_sign"] > 0:
        origin = np.array([x_r, h, -d / 2], np.float32)
        axis_u = np.array([0, 0, d], np.float32)
        axis_v = np.array([0, -h, 0], np.float32)
        normal = np.array([1.0, 0, 0], np.float32)
    else:
        origin = np.array([x_l, h, d / 2], np.float32)
        axis_u = np.array([0, 0, -d], np.float32)
        axis_v = np.array([0, -h, 0], np.float32)
        normal = np.array([-1.0, 0, 0], np.float32)
    _face(
        buf, cam, origin, axis_u, axis_v, normal, 180, 420, p, lights,
        base_srgb, brushed, ID_SIDE, {"bevel": True, "bevel_w": 0.03, "ao": True}, phase,
    )

    # Front frame: four bars around the opening.
    ox0, ox1, oy0, oy1 = lay["open"]
    # top rail
    _face(
        buf, cam,
        np.array([x_l, h, front], np.float32),
        np.array([w, 0, 0], np.float32),
        np.array([0, -(h - oy1), 0], np.float32),
        np.array([0, 0, 1], np.float32),
        160, 48, p, lights, base_srgb, brushed, ID_FRAME,
        {"bevel": True, "bevel_w": 0.12, "ao": True}, phase,
    )
    # bottom rail (above plinth)
    _face(
        buf, cam,
        np.array([x_l, oy0, front], np.float32),
        np.array([w, 0, 0], np.float32),
        np.array([0, -(oy0 - plinth), 0], np.float32),
        np.array([0, 0, 1], np.float32),
        160, 36, p, lights, base_srgb, brushed, ID_FRAME,
        {"bevel": True, "bevel_w": 0.15, "ao": True}, phase,
    )
    # left stile
    _face(
        buf, cam,
        np.array([x_l, oy1, front], np.float32),
        np.array([ox0 - x_l, 0, 0], np.float32),
        np.array([0, -(oy1 - oy0), 0], np.float32),
        np.array([0, 0, 1], np.float32),
        48, 400, p, lights, base_srgb, brushed, ID_FRAME,
        {"bevel": True, "bevel_w": 0.18, "ao": True}, phase,
    )
    # right stile
    _face(
        buf, cam,
        np.array([ox1, oy1, front], np.float32),
        np.array([x_r - ox1, 0, 0], np.float32),
        np.array([0, -(oy1 - oy0), 0], np.float32),
        np.array([0, 0, 1], np.float32),
        48, 400, p, lights, base_srgb, brushed, ID_FRAME,
        {"bevel": True, "bevel_w": 0.18, "ao": True}, phase,
    )
    # Jamb reveal (depth of the door rebate), darker.
    jamb_z0 = front - 0.020
    reveal = srgb_to_lin(base_srgb) * 0.45
    # left jamb facing +X
    _face(
        buf, cam,
        np.array([ox0, oy1, front], np.float32),
        np.array([0, 0, jamb_z0 - front], np.float32),
        np.array([0, -(oy1 - oy0), 0], np.float32),
        np.array([1, 0, 0], np.float32),
        16, 200, p, lights, base_srgb * 0.55, False, ID_FRAME,
        {"bevel": False, "ao": False}, phase,
    )
    _face(
        buf, cam,
        np.array([ox1, oy1, jamb_z0], np.float32),
        np.array([0, 0, front - jamb_z0], np.float32),
        np.array([0, -(oy1 - oy0), 0], np.float32),
        np.array([-1, 0, 0], np.float32),
        16, 200, p, lights, base_srgb * 0.55, False, ID_FRAME,
        {"bevel": False, "ao": False}, phase,
    )

    # Welds on the outer front vertical corners and the top front edge.
    welds = _weld_strips(lay)
    weld_ids = []
    for i, (origin, axis_u, axis_v, normal) in enumerate(welds):
        sid = ID_WELD0 + i
        weld_ids.append(sid)
        _face(
            buf, cam, origin, axis_u, axis_v, normal,
            36, 280 if i < 2 else 200, p, lights,
            np.clip(base_srgb * np.array([0.78, 0.72, 0.62], np.float32) + np.array([0.08, 0.05, 0.02], np.float32), 0, 1),
            False, sid,
            {"bevel": False, "ao": False, "weld_crown": True, "noise_scale": 6.0},
            phase,
        )

    # Door.
    door = lay["door"]
    hinge = (door["hinge_x"], door["hinge_z"])
    eu, ey, et = _door_axes(angle)
    # Outer face at t = 0.004 (just proud of the hinge plane) when closed;
    # the hinge plane is already recessed from the frame.
    t_outer = 0.006
    y_top = door["y0"] + door["h"]
    origin = _door_map(0.0, y_top, t_outer, angle, *hinge)
    axis_u = eu * door["w"]
    axis_v = -ey * door["h"]
    normal = et  # outward
    feats = {
        "bevel": True,
        "bevel_w": 0.028,
        "ao": True,
        "louvers": (0.12, 0.88, 0.045, p["louver_count"], 0.012, 0.010),
        "joint_v": 0.40,
        "nameplate": (0.22, 0.28, 0.20, 0.07) if detailed else None,
        "bolt_shadows": [],
    }
    if (not door_open) and detailed:
        skip_idx = p["fastener_index"] if (6 in classes and not p["fastener_loose"]) else -1
        for i, (bu, bv) in enumerate(p["bolt_uvs"]):
            if i == skip_idx:
                continue
            feats["bolt_shadows"].append((bu, bv, 0.020))
    if with_defects and (not door_open) and 1 in classes:
        d = p["dent"]
        feats["dent"] = (d["u"], d["v"], d["au"], d["av"], d["ang"], d["strength"])
    if with_defects and (not door_open) and 4 in classes:
        c = p["corrosion"]
        feats["corrosion"] = (c["u"], c["v"], c["rad"], c["blob_seed"])
    _face(
        buf, cam, origin, axis_u, axis_v, normal,
        420, 640, p, lights, base_srgb, brushed, ID_DOOR, feats, phase,
    )

    # Handle and hinges, only when the door outer face is the thing we look at
    # (closed). When open, the door is swung aside; still draw them on the door.
    if detailed:
        _add_hardware(buf, cam, p, lights, door, angle, hinge, base_srgb, brushed, classes if with_defects else set())

    if door_open and detailed:
        _add_interior(buf, cam, p, lights, lay, classes if with_defects else set(), base_srgb)

    return {
        "layout": lay,
        "door_origin": origin,
        "door_axis_u": axis_u,
        "door_axis_v": axis_v,
        "door_normal": normal,
        "angle": angle,
        "hinge": hinge,
        "welds": welds,
        "weld_ids": weld_ids,
        "brushed": brushed,
    }


def _weld_strips(lay):
    """Three raised beads: left corner, right corner, top front edge."""
    x_l, w, h, d = lay["x0"], lay["w"], lay["h"], lay["d"]
    front = lay["front"]
    x_r = x_l + w
    half = 0.016
    strips = []
    # Left vertical corner, bead facing the bisector of -X and +Z.
    bis = normalize([-0.65, 0.0, 0.75])
    across = normalize(np.cross(np.array([0, 1, 0], np.float32), bis))
    mid = np.array([x_l, 0.0, front], np.float32)
    origin = mid + np.array([0, h * 0.98, 0], np.float32) - across * half
    strips.append((origin, across * (2 * half), np.array([0, -h * 0.92, 0], np.float32), bis))
    # Right vertical corner.
    bis = normalize([0.65, 0.0, 0.75])
    across = normalize(np.cross(np.array([0, 1, 0], np.float32), bis))
    mid = np.array([x_r, 0.0, front], np.float32)
    origin = mid + np.array([0, h * 0.98, 0], np.float32) - across * half
    strips.append((origin, across * (2 * half), np.array([0, -h * 0.92, 0], np.float32), bis))
    # Top front edge, bead facing up-and-out.
    bis = normalize([0.0, 0.55, 0.84])
    across = normalize(np.cross(np.array([1, 0, 0], np.float32), bis))
    # across should run vertically-ish across the bead; length runs along X.
    mid = np.array([x_l, h, front], np.float32)
    origin = mid - across * half
    strips.append((origin, np.array([w, 0, 0], np.float32), across * (2 * half), bis))
    return strips


def _add_hardware(buf, cam, p, lights, door, angle, hinge, base_srgb, brushed, classes):
    # Hinges: three short steel barrels on the hinge stile.
    steel = np.array([0.55, 0.56, 0.57], np.float32)
    for fy in (0.12, 0.48, 0.82):
        cy = door["y0"] + door["h"] * fy
        add_oriented_box(
            buf, cam, p, lights,
            center=(0.012, cy, 0.012),
            su=0.012, sy=0.045, st=0.010,
            angle=angle, hinge=hinge,
            base_srgb=steel, brushed=True, surface_id=ID_FRAME,
            features={"bevel": True, "bevel_w": 0.2, "ao": False},
        )
    # Vertical handle.
    side = p["handle_side"]
    cu = door["w"] * (0.90 if side > 0 else 0.10)
    cy = door["y0"] + door["h"] * 0.48
    handle_srgb = np.array([0.62, 0.63, 0.64], np.float32)
    add_oriented_box(
        buf, cam, p, lights,
        center=(cu, cy, 0.020),
        su=0.010, sy=0.11, st=0.009,
        angle=angle, hinge=hinge,
        base_srgb=handle_srgb, brushed=True, surface_id=ID_DOOR,
        features={"bevel": True, "bevel_w": 0.25, "ao": False},
    )
    # Small round lock below the handle.
    lock_c = _door_map(cu, cy - 0.16, 0.016, angle, hinge[0], hinge[1])
    render_sphere(
        buf, cam, lock_c, 0.014,
        srgb_to_lin(np.array([0.45, 0.46, 0.47], np.float32)),
        0.45, 60.0, np.array([1, 1, 1], np.float32), lights_from(p),
    )
    # Fastener heads on the closed door.
    if angle < 0.2:
        loose = 6 in classes and p["fastener_loose"]
        missing = 6 in classes and not p["fastener_loose"]
        for i, (bu, bv) in enumerate(p["bolt_uvs"]):
            if missing and i == p["fastener_index"]:
                continue
            u = bu * door["w"]
            # bv is down the texture (v=0 at top). World y decreases as bv increases.
            y = door["y0"] + door["h"] * (1.0 - bv)
            t = 0.012
            if loose and i == p["fastener_index"]:
                u = u + p["loose_du"] * door["w"]
                y = y - p["loose_dv"] * door["h"]
                t = 0.020
            c = _door_map(u, y, t, angle, hinge[0], hinge[1])
            render_sphere(
                buf, cam, c, 0.014,
                srgb_to_lin(np.array([0.74, 0.75, 0.76], np.float32)),
                0.75, 90.0, np.array([1, 1, 1], np.float32), lights_from(p),
                mask_class=6 if (loose and i == p["fastener_index"]) else None,
            )


def _add_interior(buf, cam, p, lights, lay, classes, base_srgb):
    ox0, ox1, oy0, oy1 = lay["open"]
    d = lay["d"]
    back_z = -d / 2.0 + 0.06
    # Light gray back panel, similar to an open MCC interior.
    origin = np.array([ox0 + 0.02, oy1 - 0.04, back_z], np.float32)
    axis_u = np.array([(ox1 - ox0) - 0.04, 0, 0], np.float32)
    axis_v = np.array([0, -((oy1 - oy0) - 0.08), 0], np.float32)
    feats = {"bevel": True, "bevel_w": 0.02, "ao": True, "noise_scale": 2.0, "bolt_shadows": []}
    if 1 in classes:
        dnt = p["dent"]
        feats["dent"] = (dnt["u"], dnt["v"], dnt["au"], dnt["av"], dnt["ang"], dnt["strength"])
    if 4 in classes:
        cor = p["corrosion"]
        feats["corrosion"] = (cor["u"], min(0.86, cor["v"]), cor["rad"], cor["blob_seed"])
    missing_fastener = 6 in classes and not p["fastener_loose"]
    interior_bolts = _interior_bolt_uvs()
    defect_i = p["fastener_index"] % len(interior_bolts)
    for i, (bu, bv) in enumerate(interior_bolts):
        if missing_fastener and i == defect_i:
            continue
        feats["bolt_shadows"].append((bu, bv, 0.022))
    _face(
        buf, cam, origin, axis_u, axis_v, np.array([0, 0, 1], np.float32),
        360, 480, p, lights, np.array([0.78, 0.79, 0.78], np.float32), False, ID_INTERIOR,
        feats,
        phase=(0.55, 0.25),
    )
    # Remember the panel so screen-space marks (scratch, empty hole) land on it.
    p["_back_panel"] = (origin, axis_u, axis_v)
    # Galvanized return on the camera-side interior wall.
    origin = np.array([ox1 - 0.015, oy1 - 0.04, back_z], np.float32)
    axis_u = np.array([0, 0, lay["front"] - 0.08 - back_z], np.float32)
    axis_v = np.array([0, -((oy1 - oy0) - 0.08), 0], np.float32)
    _face(
        buf, cam, origin, axis_u, axis_v, np.array([-1, 0, 0], np.float32),
        160, 420, p, lights, np.array([0.74, 0.75, 0.73], np.float32), True, ID_INTERIOR,
        {"bevel": False, "ao": True, "brush_freq": 280, "brush_along": "v"},
        phase=(0.9, 0.4),
    )

    # Three straight copper bars. One may be shifted off the insulator line.
    bar_x0 = ox0 + 0.10
    bar_x1 = ox1 - 0.10
    length = bar_x1 - bar_x0
    y_mid = (oy0 + oy1) / 2.0
    ys = [y_mid - 0.16, y_mid, y_mid + 0.16]
    z_bar = back_z + 0.16
    copper = np.array([0.78, 0.46, 0.32], np.float32)
    mis = p["bus_phase"] if 5 in classes else -1
    for i, y in enumerate(ys):
        dy = p["bus_dy"] if i == mis else 0.0
        cy = y + dy
        # Insulator blocks stay on the nominal line, behind the bar.
        for fx in (0.22, 0.78):
            cx = bar_x0 + length * fx
            for origin, au, av, normal in _aa_box_faces(
                np.array([cx, y - 0.01, z_bar - 0.045], np.float32),
                0.028, 0.034, 0.030,
            ):
                _face(
                    buf, cam, origin, au, av, normal, 40, 36, p, lights,
                    np.array([0.07, 0.065, 0.065], np.float32), False, ID_INTERIOR,
                    {"bevel": True, "bevel_w": 0.15, "ao": False},
                    phase=(0.1 * i, fx),
                    material=(0.04, 12.0, np.array([1.0, 1.0, 1.0], np.float32)),
                )
        half_h, half_t = 0.022, 0.008
        center = np.array([(bar_x0 + bar_x1) / 2, cy, z_bar], np.float32)
        for origin, au, av, normal in _aa_box_faces(center, length / 2, half_h, half_t):
            # Only the faces that can face the camera need copper shading detail.
            _face(
                buf, cam, origin, au, av, normal, 300, 36, p, lights,
                copper, False, ID_BUS + i,
                {"bevel": True, "bevel_w": 0.18, "ao": False, "noise_scale": 5.0},
                phase=(0.2, i * 0.3),
                material=(0.95, 160.0, np.array([1.0, 0.72, 0.48], np.float32)),
            )
            if i == mis:
                # Mark the whole visible bar. Re-blit is wasteful; instead stamp
                # the mask by a second warp of a full-white texture through the
                # same quad. Done below via id buffer after all bars exist.
                pass
        # Clamp bolts. They ride on the bar, so a shifted bar takes them along.
        for fx in (0.22, 0.78):
            cx = bar_x0 + length * fx
            render_sphere(
                buf, cam,
                np.array([cx, cy + half_h * 0.15, z_bar + half_t + 0.007], np.float32),
                0.009,
                srgb_to_lin(np.array([0.78, 0.79, 0.80], np.float32)),
                0.7, 80.0, np.array([1, 1, 1], np.float32), lights,
                mask_class=5 if i == mis else None,
            )
    # Stamp every visible pixel of the misaligned bar (surface id) into the mask.
    if mis >= 0:
        sid = ID_BUS + mis
        buf.masks[5][buf.idbuf == sid] = 255

    # Fastener heads on the back panel. A missing one leaves the countersunk
    # hole painted later; a loose one is shifted and is the labeled part.
    panel = p.get("_back_panel")
    if panel is not None:
        bo, bau, bav = panel
        loose = 6 in classes and p["fastener_loose"]
        interior_bolts = _interior_bolt_uvs()
        for i, (bu, bv) in enumerate(interior_bolts):
            if missing_fastener and i == (p["fastener_index"] % len(interior_bolts)):
                continue
            u, v = bu, bv
            if loose and i == (p["fastener_index"] % len(interior_bolts)):
                u = float(np.clip(u + p["loose_du"] * 0.45, 0.18, 0.82))
                v = float(np.clip(v + p["loose_dv"] * 0.35, 0.14, 0.80))
            center = bo + bau * u + bav * v + np.array([0.0, 0.0, 0.014], np.float32)
            render_sphere(
                buf, cam, center, 0.013,
                srgb_to_lin(np.array([0.74, 0.75, 0.76], np.float32)),
                0.7, 80.0, np.array([1, 1, 1], np.float32), lights,
                mask_class=6 if (loose and i == (p["fastener_index"] % len(interior_bolts))) else None,
            )


def apply_screen_defects(buf, cam, p, built):
    classes = set(p["classes"])
    door = built["layout"]["door"]
    origin = built["door_origin"]
    axis_u = built["door_axis_u"]
    axis_v = built["door_axis_v"]
    angle = built["angle"]

    if 0 in classes:
        sc = p["scratch"]
        n = sc["n"]
        if angle < 0.2:
            host_o, host_u, host_v, host_ids = origin, axis_u, axis_v, [ID_DOOR]
        elif p.get("_back_panel") is not None:
            host_o, host_u, host_v = p["_back_panel"]
            host_ids = [ID_INTERIOR]
        else:
            host_o = None
        if host_o is not None:
            pts = []
            for i in range(n):
                t = i / (n - 1)
                u = sc["u0"] + (sc["u1"] - sc["u0"]) * t
                v = sc["v0"] + (sc["v1"] - sc["v0"]) * t
                v = v + np.sin(t * np.pi * 2.0) * sc["jag"]
                pts.append(host_o + host_u * u + host_v * v)
            proj, z = cam.project(np.stack(pts))
            if np.all(z > 0.2):
                paint_polyline(buf, proj, host_ids, 2, 0, "scratch")

    if 4 in classes and angle >= 0.2:
        # Open cabinets keep a small rust patch on the lower front rail, in UV
        # of that rail. Already skipped on the door; paint it in screen space
        # on the frame as a compact cluster of spots is less accurate than a
        # texture. Corrosion on open images is placed on the back panel by
        # reusing the door-closed texture path only. Here, if the door is open
        # the corrosion mask was not drawn. Handle that in add_cabinet by
        # putting corrosion on the back panel. Nothing to do if the mask exists.
        pass

    if 2 in classes or 3 in classes:
        welds = built["welds"]
        ids = built["weld_ids"]
        which = p["weld_which"]
        which_b = p["weld_which_b"]
        if 2 in classes:
            _paint_weld_defect(buf, cam, p, welds[which], ids[which], "pits")
        if 3 in classes:
            use = which if (2 in classes and which_b == which) else which_b
            crack_p = p
            if 2 in classes and use == which:
                crack_p = dict(p)
                crack_p["weld_t0"] = min(0.72, p["weld_t0"] + p["weld_len"] + 0.05)
            _paint_weld_defect(buf, cam, crack_p, welds[use], ids[use], "crack")

    if 6 in classes and not p["fastener_loose"]:
        if angle < 0.2:
            bu, bv = p["bolt_uvs"][p["fastener_index"]]
            world = origin + axis_u * bu + axis_v * bv
            host_ids = [ID_DOOR]
        elif p.get("_back_panel") is not None:
            bo, buv, bvct = p["_back_panel"]
            bolts = _interior_bolt_uvs()
            bu, bv = bolts[p["fastener_index"] % len(bolts)]
            world = bo + buv * bu + bvct * bv
            host_ids = [ID_INTERIOR]
        else:
            world = None
        if world is not None:
            proj, z = cam.project(np.asarray(world, np.float32).reshape(1, 3))
            if z[0] > 0.2:
                r_pix = cam.f * 0.013 / float(z[0])
                paint_hole(buf, proj[0], r_pix, host_ids, 6)


def _paint_weld_defect(buf, cam, p, weld, surface_id, kind):
    origin, axis_u, axis_v, _normal = weld
    # Centerline is mid-way across the bead (axis_v is the across-bead axis
    # for vertical welds where axis_v runs down the length... see _weld_strips.
    # Left/right welds: axis_u is across (short), axis_v is along (long).
    # Top weld: axis_u is along (long), axis_v is across (short).
    along_is_v = np.linalg.norm(axis_v) > np.linalg.norm(axis_u)
    t0 = p["weld_t0"]
    t1 = min(0.92, t0 + p["weld_len"])
    if kind == "pits":
        n = p["pit_n"]
        centers = []
        radii = []
        for i in range(n):
            t = t0 + (t1 - t0) * (i + 0.5) / n
            if along_is_v:
                world = origin + axis_u * 0.5 + axis_v * t
            else:
                world = origin + axis_u * t + axis_v * 0.5
            proj, z = cam.project(world.reshape(1, 3))
            if z[0] <= 0.2:
                continue
            centers.append(proj[0])
            radii.append(max(2.2, cam.f * 0.0065 / float(z[0])))
        if centers:
            paint_pits(buf, centers, radii, [surface_id], 2)
    else:
        n = p["crack_n"]
        pts = []
        amp = p["crack_amp"]
        for i in range(n):
            t = t0 + (t1 - t0) * i / (n - 1)
            jitter = np.sin(i * 2.3) * amp
            if along_is_v:
                world = origin + axis_u * (0.5 + jitter * 0.15) + axis_v * t
            else:
                world = origin + axis_u * t + axis_v * (0.5 + jitter * 0.15)
            pts.append(world)
        proj, z = cam.project(np.stack(pts))
        if np.all(z > 0.2):
            paint_polyline(buf, proj, [surface_id], 2, 3, "crack")


def frame_camera(p, corners):
    corners = np.asarray(corners, np.float32)
    target = corners.mean(axis=0)
    target = target + np.array([0.0, -0.05, 0.0], np.float32)
    dist = 4.0
    yaw, pitch = p["yaw"], p["pitch"]
    cam = None
    for _ in range(12):
        eye = target + dist * np.array([
            np.sin(yaw) * np.cos(pitch),
            np.sin(pitch),
            np.cos(yaw) * np.cos(pitch),
        ], np.float32)
        cam = Camera(eye, target, p["fov"], IMAGE_W, IMAGE_H, p["roll"])
        proj, z = cam.project(corners)
        if np.any(z < 0.35):
            dist *= 1.12
            continue
        span = np.maximum(proj.max(0) - proj.min(0), 1.0)
        fill = max(span[0] / IMAGE_W, span[1] / IMAGE_H)
        if fill < 1e-3:
            dist *= 0.8
            continue
        dist *= fill / p["fill"]
    return cam


def subject_corners(p):
    """Corners the camera fitter keeps in frame. Close and open modes use a
    small window so a weld, fastener, or busbar is large enough to inspect."""
    w, h, d = p["w"], p["h"], p["d"]
    front = d / 2.0
    if p["mode"] == "aisle":
        total_w = p["n_sections"] * (w + 0.012)
        xs, ys, zs = [0.0, total_w], [0.0, h], [-d / 2.0, front]
    elif p["mode"] == "open":
        # Interior around the three bars. Include the frame and a bit of the
        # swung door so the shot still reads as an open cabinet.
        xs = [-0.22, w + 0.04]
        ys = [h * 0.30, h * 0.78]
        zs = [-d / 2.0, front + 0.20]
    elif p["mode"] == "close":
        # Door spans almost the full cabinet height. v=0 is the top of the door.
        door_top = h - 0.045
        door_bot = 0.10
        door_h = door_top - door_bot
        if 6 in p["classes"] and 2 not in p["classes"] and 3 not in p["classes"]:
            bu, bv = p["bolt_uvs"][p["fastener_index"]]
            by = door_bot + door_h * (1.0 - bv)
            bx = 0.06 + bu * (w - 0.12)
            xs = [bx - 0.28, bx + 0.28]
            ys = [by - 0.26, by + 0.26]
        elif 2 in p["classes"] or 3 in p["classes"]:
            which = p["weld_which"] if 2 in p["classes"] else p["weld_which_b"]
            y0, y1 = h * 0.28, h * 0.80
            if which == 1:
                xs = [w * 0.42, w + 0.08]
            else:
                xs = [-0.08, w * 0.58]
            ys = [y0, y1]
        else:
            xs = [0.02, w * 0.98]
            ys = [h * 0.36, h * 0.84]
        zs = [front - 0.08, front + 0.08]
    else:
        xs, ys, zs = [-0.02, w + 0.02], [0.0, h], [-d / 2.0, front]
    corners = [(x, y, z) for x in xs for y in ys for z in zs]
    return np.array(corners, np.float32)


def tonemap(linear, exposure):
    x = np.clip(linear * exposure, 0.0, None)
    x = x / (1.0 + 0.12 * x)
    return np.power(np.clip(x, 0.0, 1.0), 1.0 / 2.2).astype(np.float32)


def mask_to_yolo(mask, w, h):
    ys, xs = np.where(mask > 0)
    if xs.size < 6:
        return None
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    bw = (x2 - x1 + 1) / float(w)
    bh = (y2 - y1 + 1) / float(h)
    cx = (x1 + x2 + 1) / 2.0 / float(w)
    cy = (y1 + y2 + 1) / 2.0 / float(h)
    if bw <= 0 or bh <= 0:
        return None
    if bw >= 1.0:
        bw = 0.999
        cx = 0.5
    if bh >= 1.0:
        bh = 0.999
        cy = 0.5
    cx = float(np.clip(cx, bw / 2.0, 1.0 - bw / 2.0))
    cy = float(np.clip(cy, bh / 2.0, 1.0 - bh / 2.0))
    return cx, cy, bw, bh


def render_image(seed, classes):
    """Return uint8 RGB, list of (cls, cx, cy, bw, bh), and the mode name."""
    _ensure_noise()
    p = sample_params(seed, classes)
    lights = lights_from(p)
    cam = frame_camera(p, subject_corners(p))
    buf = Buffers(IMAGE_W, IMAGE_H)
    add_room(buf, cam, p, lights)

    if p["mode"] == "aisle":
        gap = 0.012
        for i in range(p["n_sections"]):
            x0 = i * (p["w"] + gap)
            built = add_cabinet(
                buf, cam, p, lights, x0,
                detailed=(i == 0),
                with_defects=(i == 0),
                angle_override=0.0,
            )
            if i == 0:
                primary = built
    else:
        angle = p["open_angle"] if p["mode"] == "open" else 0.0
        primary = add_cabinet(
            buf, cam, p, lights, 0.0,
            detailed=True, with_defects=True, angle_override=angle,
        )

    apply_screen_defects(buf, cam, p, primary)

    # Contact shadow from the nearest cabinet footprint.
    lay = primary["layout"]
    fp = np.array([
        [lay["x0"], 0.01, -lay["d"] / 2],
        [lay["x0"] + lay["w"], 0.01, -lay["d"] / 2],
        [lay["x0"] + lay["w"], 0.01, lay["d"] / 2],
        [lay["x0"], 0.01, lay["d"] / 2],
    ], np.float32)
    add_floor_shadow(buf, cam, fp, p["key"])

    rgb = tonemap(buf.color, p["exposure"])
    if p["blur"] > 0.05:
        rgb = cv2.GaussianBlur(rgb, (0, 0), p["blur"])
    nrng = np.random.default_rng(int(seed) + NOISE_SEED_OFFSET)
    rgb = rgb + nrng.normal(0.0, p["noise"], rgb.shape).astype(np.float32)
    yy = np.linspace(-1.0, 1.0, IMAGE_H, dtype=np.float32)[:, None]
    xx = np.linspace(-1.0, 1.0, IMAGE_W, dtype=np.float32)[None, :]
    vig = 1.0 - p["vignette"] * (xx * xx + yy * yy)
    rgb = np.clip(rgb * vig[..., None], 0.0, 1.0)
    image = (rgb * 255.0 + 0.5).astype(np.uint8)

    labels = []
    for cls in p["classes"]:
        box = mask_to_yolo(buf.masks[cls], IMAGE_W, IMAGE_H)
        if box is None:
            raise RuntimeError(f"empty mask for class {cls} ({CLASS_NAMES[cls]}) seed {seed} mode {p['mode']}")
        labels.append((cls, *box))
    labels.sort(key=lambda r: r[0])
    return image, labels, p["mode"]


# ---------------------------------------------------------------------------
# Dataset plan, splits, IO
# ---------------------------------------------------------------------------

def build_plan(min_images=MIN_IMAGES, min_per_class=MIN_PER_CLASS):
    rng = np.random.default_rng(PLAN_SEED)
    counts = np.zeros(7, dtype=np.int32)
    plans = []
    while True:
        if len(plans) >= min_images and int(counts.min()) >= min_per_class and len(plans) % 20 == 0:
            break
        if len(plans) > 4000:
            raise RuntimeError(f"could not balance plan, counts={counts.tolist()}")
        k = int(rng.integers(1, 4))
        weight = (float(counts.max() if len(plans) else 0) + 4.0) - counts.astype(np.float64)
        weight = np.maximum(weight, 0.35)
        chosen = []
        for _ in range(k):
            w = weight.copy()
            for c in chosen:
                w[c] = 0.0
            if float(w.sum()) <= 0:
                break
            w /= w.sum()
            c = int(rng.choice(7, p=w))
            chosen.append(c)
            weight[c] = 0.0
        for c in chosen:
            counts[c] += 1
        plans.append(chosen)
    return plans, counts


def assign_splits(n, seed=SPLIT_SEED):
    """70/15/15 by integer percent, so 700 images is 490/105/105.

    ``n * 0.70`` is not exact in float64 (it can be just under 490), so the
    cut uses integer arithmetic. The shuffle itself is ``Generator(seed)``.
    """
    rng = np.random.default_rng(seed)
    order = np.arange(n)
    rng.shuffle(order)
    n_train = (n * 70) // 100
    n_val = (n * 15) // 100
    splits = np.empty(n, dtype=object)
    splits[order[:n_train]] = "train"
    splits[order[n_train:n_train + n_val]] = "val"
    splits[order[n_train + n_val:]] = "test"
    return splits


def _save_image(path, rgb):
    path.parent.mkdir(parents=True, exist_ok=True)
    im = Image.fromarray(rgb, mode="RGB")
    im.save(path, format="JPEG", quality=JPEG_QUALITY, subsampling=1, optimize=False)


def _save_labels(path, labels):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}" for cls, cx, cy, bw, bh in labels]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _render_job(job):
    seed, classes, split, out_dir = job
    image, labels, mode = render_image(seed, classes)
    stem = f"enc_{seed:05d}"
    rel_img = f"images/{split}/{stem}.jpg"
    rel_lbl = f"labels/{split}/{stem}.txt"
    _save_image(Path(out_dir) / rel_img, image)
    _save_labels(Path(out_dir) / rel_lbl, labels)
    return {
        "filename": rel_img,
        "seed": seed,
        "split": split,
        "class_ids": ",".join(str(c) for c, *_ in labels),
        "width": IMAGE_W,
        "height": IMAGE_H,
        "generator_version": GENERATOR_VERSION,
        "synthetic": "true",
        "mode": mode,
        "labels": labels,
    }


def write_readme(path: Path, n, counts, split_counts):
    text = f"""# Synthetic shop-inspection enclosure defects

These images are **procedural renders**, not photographs of a plant, shop, or
factory. They were drawn with a pinhole camera, painted or brushed metal
shading, and defect masks. They are not frames taken from the licensed
reference photos, and they are not copies of any catalog or website.

Do **not** report this folder as a 6,800-image factory dataset.
Do **not** report a 0.93 mAP (or any other measured accuracy) from these
files. Nothing here was trained or scored.

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
Assignment uses `numpy.random.Generator({SPLIT_SEED})` once, after the render
list exists and before any augmentation. This generator does not write
augmented copies. Each seed is one file and one split.

This build: **{n}** unique images (seeds `0` … `{n - 1}`).

Per-class image counts (an image can count toward more than one class):

| class | images |
|-------|--------|
"""
    for i, name in enumerate(CLASS_NAMES):
        text += f"| {i} {name} | {int(counts[i])} |\n"
    text += f"""
Split counts: train {split_counts['train']}, val {split_counts['val']}, test {split_counts['test']}.

## Reproduce

```bash
python3 generate.py --output /path/to/synthetic-enclosures
```

Requires Python 3, numpy, opencv-python-headless, and Pillow. No network
access and no downloaded photographs.

## Parameters ({GENERATOR_VERSION})

| name | value |
|------|--------|
| `PLAN_SEED` | {PLAN_SEED} |
| `SPLIT_SEED` | {SPLIT_SEED} |
| `NOISE_SEED_OFFSET` | {NOISE_SEED_OFFSET} |
| `TEXTURE_NOISE_SEED` | {TEXTURE_NOISE_SEED} |
| image size | {IMAGE_W} × {IMAGE_H} |
| JPEG quality | {JPEG_QUALITY}, 4:2:2, optimize off |
| images | at least {MIN_IMAGES}, and a multiple of 20 |
| per-class minimum | {MIN_PER_CLASS} |
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
"""
    path.write_text(text, encoding="utf-8")


def write_yolo_sidecar(path: Path):
    (path / "classes.txt").write_text("\n".join(CLASS_NAMES) + "\n", encoding="utf-8")
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASS_NAMES))
    (path / "data.yaml").write_text(
        "path: .\ntrain: images/train\nval: images/val\ntest: images/test\n"
        f"names:\n{names}\n",
        encoding="utf-8",
    )


def build_preview(rows, dataset_dir: Path, preview_path: Path):
    """One labeled example per class, boxes taken from the saved label files."""
    chosen = {}
    for row in rows:
        label_path = dataset_dir / "labels" / row["split"] / (Path(row["filename"]).stem + ".txt")
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            cls = int(parts[0])
            bw, bh = float(parts[3]), float(parts[4])
            area = bw * bh
            prev = chosen.get(cls)
            if prev is None or area > prev[0] or (area == prev[0] and row["seed"] < prev[1]):
                chosen[cls] = (area, row["seed"], row, line)
    if len(chosen) < 7:
        missing = [CLASS_NAMES[i] for i in range(7) if i not in chosen]
        raise RuntimeError(f"preview missing classes: {missing}")

    thumb_w, thumb_h = 420, 280
    pad = 14
    header = 64
    caption_h = 36
    cols, rows_n = 4, 2
    sheet_w = cols * thumb_w + (cols + 1) * pad
    sheet_h = header + rows_n * (thumb_h + caption_h) + (rows_n + 1) * pad
    sheet = Image.new("RGB", (sheet_w, sheet_h), (28, 30, 32))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype(FONT_BOLD, 22)
    font_s = ImageFont.truetype(FONT_REG, 16)
    draw.text((pad, 12), "Synthetic shop-inspection enclosures", fill=(240, 240, 238), font=font)
    draw.text(
        (pad, 38),
        "Generated renders, not plant photos. Not a 6,800-image factory set and not a 0.93 mAP result.",
        fill=(190, 194, 196),
        font=font_s,
    )
    order = list(range(7))
    for i, cls in enumerate(order):
        _area, _seed, row, _line = chosen[cls]
        r, c = divmod(i, cols)
        if i >= 4:
            # Center the last row of three.
            c = i - 4
            x = pad + c * (thumb_w + pad) + (thumb_w + pad) // 2
        else:
            x = pad + c * (thumb_w + pad)
        y = header + pad + r * (thumb_h + caption_h + pad)
        img = Image.open(dataset_dir / row["filename"]).convert("RGB")
        # Draw every box; emphasize this class with a thicker stroke.
        lab = (dataset_dir / "labels" / row["split"] / (Path(row["filename"]).stem + ".txt")).read_text(encoding="utf-8")
        arr = np.array(img)
        for line in lab.splitlines():
            pc = line.split()
            k = int(pc[0])
            cx, cy, bw, bh = map(float, pc[1:])
            W, H = arr.shape[1], arr.shape[0]
            x1 = int(round((cx - bw / 2) * W))
            y1 = int(round((cy - bh / 2) * H))
            x2 = int(round((cx + bw / 2) * W))
            y2 = int(round((cy + bh / 2) * H))
            color = BOX_COLORS[k]
            thick = 3 if k == cls else 1
            cv2.rectangle(arr, (x1, y1), (x2, y2), color, thick, lineType=cv2.LINE_8)
        tile = Image.fromarray(arr).resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        sheet.paste(tile, (x, y))
        cap = f"{cls}  {CLASS_NAMES[cls].replace('_', ' ')}"
        draw.text((x, y + thumb_h + 6), cap, fill=BOX_COLORS[cls], font=font_s)
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(preview_path, format="JPEG", quality=90, subsampling=1)
    return preview_path


def validate(dataset_dir: Path, rows):
    seeds = [r["seed"] for r in rows]
    if len(seeds) != len(set(seeds)):
        raise RuntimeError("duplicate seeds")
    if len(rows) < MIN_IMAGES:
        raise RuntimeError(f"only {len(rows)} images")
    splits = {"train": 0, "val": 0, "test": 0}
    counts = np.zeros(7, dtype=np.int32)
    hashes = set()
    seed_split = {}
    for r in rows:
        if r["synthetic"] != "true":
            raise RuntimeError("synthetic flag not true")
        if r["generator_version"] != GENERATOR_VERSION:
            raise RuntimeError("version mismatch")
        if int(r["width"]) != IMAGE_W or int(r["height"]) != IMAGE_H:
            raise RuntimeError("size mismatch")
        splits[r["split"]] += 1
        if r["seed"] in seed_split and seed_split[r["seed"]] != r["split"]:
            raise RuntimeError("seed in two splits")
        seed_split[r["seed"]] = r["split"]
        img_path = dataset_dir / r["filename"]
        lbl_path = dataset_dir / "labels" / r["split"] / (img_path.stem + ".txt")
        if not img_path.is_file() or not lbl_path.is_file():
            raise RuntimeError(f"missing file for seed {r['seed']}")
        data = img_path.read_bytes()
        digest = hashlib.md5(data).hexdigest()
        if digest in hashes:
            raise RuntimeError(f"duplicate image bytes at {img_path}")
        hashes.add(digest)
        im = Image.open(img_path)
        if im.size != (IMAGE_W, IMAGE_H):
            raise RuntimeError("jpeg size mismatch")
        ids = []
        for line in lbl_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) != 5:
                raise RuntimeError(f"bad label line in {lbl_path}")
            cls = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:])
            if not (0 <= cls <= 6):
                raise RuntimeError("class out of range")
            if bw <= 0 or bh <= 0:
                raise RuntimeError("zero-area box")
            if not (0 <= cx <= 1 and 0 <= cy <= 1 and bw <= 1 and bh <= 1):
                raise RuntimeError("box outside 0..1")
            if cx - bw / 2 < -1e-6 or cy - bh / 2 < -1e-6 or cx + bw / 2 > 1 + 1e-6 or cy + bh / 2 > 1 + 1e-6:
                raise RuntimeError("box extends outside the image")
            ids.append(cls)
        if not (1 <= len(ids) <= 3):
            raise RuntimeError(f"expected 1-3 defects, got {len(ids)} for seed {r['seed']}")
        if len(set(ids)) != len(ids):
            raise RuntimeError("duplicate class in one image")
        manifest_ids = [int(x) for x in r["class_ids"].split(",") if x != ""]
        if manifest_ids != sorted(ids):
            raise RuntimeError("manifest class_ids do not match label file")
        for c in ids:
            counts[c] += 1
    n = len(rows)
    # 70/15/15 using the same floor rule as assign_splits.
    expect = {
        "train": (n * 70) // 100,
        "val": (n * 15) // 100,
    }
    expect["test"] = n - expect["train"] - expect["val"]
    if splits != expect:
        raise RuntimeError(f"split counts {splits} != {expect}")
    if int(counts.min()) < MIN_PER_CLASS:
        raise RuntimeError(f"class counts below minimum: {counts.tolist()}")
    return counts, splits, len(hashes)


def generate_dataset(out_dir: Path, workers: int, limit: int = 0):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for sub in ("images", "labels"):
        shutil.rmtree(out_dir / sub, ignore_errors=True)
    plans, plan_counts = build_plan()
    if limit:
        plans = plans[:limit]
    n = len(plans)
    splits = assign_splits(n)
    print(f"plan images={n} per-class={plan_counts.tolist()}", flush=True)
    jobs = [(i, plans[i], splits[i], str(out_dir)) for i in range(n)]
    rows = [None] * n
    t0 = time.time()
    if workers <= 1:
        for i, job in enumerate(jobs):
            rows[i] = _render_job(job)
            if i < 3 or (i + 1) % 25 == 0:
                dt = time.time() - t0
                print(f"  {i+1}/{n}  {dt:.1f}s  mode={rows[i]['mode']}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(_render_job, job): job[0] for job in jobs}
            done = 0
            for fut in as_completed(futs):
                seed = futs[fut]
                rows[seed] = fut.result()
                done += 1
                if done <= 3 or done % 50 == 0 or done == n:
                    dt = time.time() - t0
                    print(f"  {done}/{n}  {dt:.1f}s", flush=True)
    rows = [r for r in rows if r is not None]
    rows.sort(key=lambda r: r["seed"])
    manifest = out_dir / "manifest.csv"
    fields = ["filename", "seed", "split", "class_ids", "width", "height", "generator_version", "synthetic"]
    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in fields})
    counts, split_counts, n_hash = validate(out_dir, rows)
    write_readme(out_dir / "README.md", len(rows), counts, split_counts)
    write_yolo_sidecar(out_dir)
    stats = {
        "generator_version": GENERATOR_VERSION,
        "unique_images": len(rows),
        "unique_seeds": len({r["seed"] for r in rows}),
        "unique_md5": n_hash,
        "seeds": [rows[0]["seed"], rows[-1]["seed"]],
        "per_class": {CLASS_NAMES[i]: int(counts[i]) for i in range(7)},
        "splits": split_counts,
        "plan_seed": PLAN_SEED,
        "split_seed": SPLIT_SEED,
        "image_size": [IMAGE_W, IMAGE_H],
        "elapsed_sec": round(time.time() - t0, 2),
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2), flush=True)
    return rows, stats


def render_samples(dest: Path):
    dest.mkdir(parents=True, exist_ok=True)
    jobs = []
    for cls in range(7):
        jobs.append((80000 + cls, [cls]))
    jobs.append((80010, [0, 1, 4]))
    jobs.append((80011, [5, 3]))
    jobs.append((80012, [2, 6]))
    for seed, classes in jobs:
        image, labels, mode = render_image(seed, classes)
        stem = f"sample_{seed}_{mode}_{'-'.join(CLASS_NAMES[c] for c in classes)}"
        _save_image(dest / f"{stem}.jpg", image)
        boxed = image.copy()
        for cls, cx, cy, bw, bh in labels:
            x1 = int(round((cx - bw / 2) * IMAGE_W))
            y1 = int(round((cy - bh / 2) * IMAGE_H))
            x2 = int(round((cx + bw / 2) * IMAGE_W))
            y2 = int(round((cy + bh / 2) * IMAGE_H))
            cv2.rectangle(boxed, (x1, y1), (x2, y2), BOX_COLORS[cls], 2)
        _save_image(dest / f"{stem}_box.jpg", boxed)
        print(stem, [(CLASS_NAMES[c], round(bw, 3), round(bh, 3)) for c, cx, cy, bw, bh in labels], flush=True)


def main():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    parser = argparse.ArgumentParser(description="Generate synthetic shop-inspection enclosure defects")
    parser.add_argument("--output", type=Path, default=Path("/cursor/stores/self/synthetic-enclosures"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0, help="render only the first N planned images")
    parser.add_argument("--samples", type=Path, default=None, help="write a handful of QA images and exit")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--preview", type=Path, default=Path("/cursor/stores/self/media/synthetic_enclosure_preview.jpg"))
    args = parser.parse_args()
    if args.samples:
        render_samples(args.samples)
        return
    if args.plan_only:
        plans, counts = build_plan()
        print(f"images={len(plans)} per-class={counts.tolist()}")
        return
    rows, stats = generate_dataset(args.output, args.workers, args.limit)
    preview = build_preview(rows, args.output, args.preview)
    # Keep a copy next to the dataset as well.
    shutil.copy2(preview, args.output / "preview.jpg")
    print(f"preview {preview} bytes={preview.stat().st_size}", flush=True)


if __name__ == "__main__":
    main()
