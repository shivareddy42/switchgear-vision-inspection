#!/usr/bin/env python3
"""Download images listed in a manifest. This is not a crawler.

It fetches only the URLs in the manifest. It does not follow links in HTML,
search the web, or discover new pages.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import unquote
from urllib.request import Request, urlopen

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inspection.hashing import dhash_image, hamming, sha256_bytes, sha256_file
from inspection.quality import NEAR_DUPLICATE_BITS

USER_AGENT = "switchgear-vision-inspection/0.1 (educational prototype; image provenance; contact: local)"
MIME_EXT = {
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/bmp": ".bmp",
    "image/tiff": ".tif",
}


def _slug(value: str) -> str:
    stem = Path(unquote(value.split("?")[0])).name
    stem = re.sub(r"\.[A-Za-z0-9]{1,5}$", "", stem)
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._")
    return (stem or "image")[:80]


def _read_manifest(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _append_row(path: Path, fieldnames: list[str], row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _existing_hashes(log_path: Path, output_root: Path) -> tuple[set[str], list[tuple[str, int]]]:
    """Load exact hashes from the log and perceptual hashes from stored images.

    The previous implementation only compared perceptual hashes within the
    current download batch, which allowed a near-duplicate to slip through on a
    later collection pass. Scanning the small stored public set makes
    deduplication stable across repeated runs.
    """
    shas: set[str] = set()
    hashes: list[tuple[str, int]] = []

    if log_path.exists():
        with log_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("status") != "ok":
                    continue
                if row.get("sha256"):
                    shas.add(row["sha256"])

    if output_root.exists():
        for path in sorted(item for item in output_root.rglob("*") if item.is_file()):
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}:
                continue
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            shas.add(sha256_file(path))
            hashes.append((str(path.relative_to(ROOT)), dhash_image(image)))

    return shas, hashes


def download_one(url: str, timeout: float) -> tuple[bytes, str]:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=timeout) as response:
        content_type = (response.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        payload = response.read()
    return payload, content_type


def main() -> int:
    parser = argparse.ArgumentParser(description="Download the images listed in a manifest.")
    parser.add_argument("--manifest", default="data/source_manifest.csv")
    parser.add_argument("--output-root", default="data/raw")
    parser.add_argument("--log", default="data/download_log.csv")
    parser.add_argument("--failed", default="data/failed_urls.csv")
    parser.add_argument("--min-side", type=int, default=96)
    parser.add_argument("--delay", type=float, default=1.0, help="Seconds to wait between requests.")
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--near-bits", type=int, default=NEAR_DUPLICATE_BITS)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    log_path = Path(args.log)
    if not log_path.is_absolute():
        log_path = ROOT / log_path
    failed_path = Path(args.failed)
    if not failed_path.is_absolute():
        failed_path = ROOT / failed_path

    rows = _read_manifest(manifest_path)
    known_sha, known_hashes = _existing_hashes(log_path, output_root)
    perceptual_hashes: list[tuple[str, int]] = list(known_hashes)
    log_fields = [
        "filename",
        "source_url",
        "source_page",
        "download_timestamp",
        "sha256",
        "width",
        "height",
        "status",
    ]
    fail_fields = ["url", "error"]
    ok = 0
    for index, row in enumerate(rows):
        if index:
            time.sleep(args.delay)
        url = (row.get("url") or "").strip()
        target_class = (row.get("target_class") or "unassigned").strip()
        source_page = (row.get("source_page") or "").strip()
        timestamp = datetime.now(timezone.utc).isoformat()
        record = {
            "filename": "",
            "source_url": url,
            "source_page": source_page,
            "download_timestamp": timestamp,
            "sha256": "",
            "width": "",
            "height": "",
            "status": "failed",
        }
        try:
            payload, content_type = download_one(url, args.timeout)
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            record["status"] = "failed"
            _append_row(log_path, log_fields, record)
            _append_row(failed_path, fail_fields, {"url": url, "error": str(exc)})
            print(f"FAIL {url} {exc}")
            continue
        if content_type not in MIME_EXT:
            message = f"rejected mime {content_type or 'missing'}"
            record["status"] = "rejected_mime"
            _append_row(log_path, log_fields, record)
            _append_row(failed_path, fail_fields, {"url": url, "error": message})
            print(f"REJECT {url} {message}")
            continue
        image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            record["status"] = "corrupt"
            _append_row(log_path, log_fields, record)
            _append_row(failed_path, fail_fields, {"url": url, "error": "decode failed"})
            print(f"REJECT {url} decode failed")
            continue
        height, width = image.shape[:2]
        record["width"] = str(width)
        record["height"] = str(height)
        digest = sha256_bytes(payload)
        record["sha256"] = digest
        if min(width, height) < args.min_side:
            record["status"] = "too_small"
            _append_row(log_path, log_fields, record)
            print(f"REJECT {url} too small {width}x{height}")
            continue
        perceptual = dhash_image(image)
        if digest in known_sha:
            record["status"] = "duplicate_sha256"
            record["filename"] = _slug(url) + MIME_EXT[content_type]
            _append_row(log_path, log_fields, record)
            print(f"DUP sha {url}")
            continue
        near = next((name for name, other in perceptual_hashes if hamming(perceptual, other) <= args.near_bits), None)
        if near:
            record["status"] = "duplicate_perceptual"
            record["filename"] = near
            _append_row(log_path, log_fields, record)
            print(f"DUP phash {url} ~ {near}")
            continue
        filename = _slug(url) + MIME_EXT[content_type]
        destination = output_root / target_class / filename
        if destination.exists():
            filename = f"{_slug(url)}_{digest[:8]}{MIME_EXT[content_type]}"
            destination = output_root / target_class / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
        record["filename"] = str(destination.relative_to(ROOT))
        record["status"] = "ok"
        known_sha.add(digest)
        perceptual_hashes.append((record["filename"], perceptual))
        _append_row(log_path, log_fields, record)
        ok += 1
        print(f"OK {record['filename']} {width}x{height}")
    print(f"downloaded_ok {ok} of {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
