#!/usr/bin/env python3
"""
Wayback Machine image scraper using the CDX API.

Downloads archived web page snapshots for a given URL across multiple years
(default 2020-2025), extracts images from each snapshot, and stores them in
year-based subdirectories.

Usage:
    python wayback_machine.py \
        --url https://www.bbc.co.uk/news \
        --output-dir wayback_images_by_year/output/sites/bbc-co-uk \
        --start-year 2020 --end-year 2025 \
        --max-images-per-year 50
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import logging
import os
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

CDX_API = "https://web.archive.org/cdx/search/cdx"
WAYBACK_BASE = "https://web.archive.org/web"
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MIN_WIDTH = 100
MIN_HEIGHT = 100
MIN_FILE_SIZE = 5 * 1024
USER_AGENT = (
    "Mozilla/5.0 (compatible; TFG-UAB-Scraper/1.0; "
    "+https://github.com/aminasifar1/tfg-how-common-are-ia-images-on-web)"
)


class WaybackScraper:

    def __init__(
        self,
        output_dir: str | Path,
        start_year: int = 2020,
        end_year: int = 2025,
        max_snapshots_per_year: int = 5,
        max_images_per_year: int = 50,
        delay: float = 1.5,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.start_year = start_year
        self.end_year = end_year
        self.max_snapshots_per_year = max_snapshots_per_year
        self.max_images_per_year = max_images_per_year
        self.delay = delay

        self.cdx_cache_dir = self.output_dir.parent / "cdx_cache" if self.output_dir.parent.exists() else self.output_dir / "cdx_cache"
        self.html_cache_dir = self.output_dir.parent / "html_cache" if self.output_dir.parent.exists() else self.output_dir / "html_cache"
        self.logs_dir = self.output_dir.parent / "logs" if self.output_dir.parent.exists() else self.output_dir / "logs"

        for d in [self.cdx_cache_dir, self.html_cache_dir, self.logs_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        self._seen_hashes: set[str] = set()

    def scrape(self, url: str) -> dict[int, int]:
        results: dict[int, int] = {}

        for year in range(self.start_year, self.end_year + 1):
            logger.info("=== Year %d ===", year)
            year_dir = self.output_dir / str(year)
            year_dir.mkdir(parents=True, exist_ok=True)

            snapshots = self._get_snapshots(url, year)
            if not snapshots:
                logger.info("  No snapshots found for %d", year)
                results[year] = 0
                continue

            logger.info("  Found %d snapshots, using up to %d",
                        len(snapshots), self.max_snapshots_per_year)

            images_saved = 0
            for snap_ts in snapshots[: self.max_snapshots_per_year]:
                if images_saved >= self.max_images_per_year:
                    break

                wb_url = f"{WAYBACK_BASE}/{snap_ts}/{url}"
                logger.info("  Snapshot %s ...", snap_ts)

                soup = self._fetch_snapshot(wb_url)
                if soup is None:
                    continue

                for img_url in self._extract_image_urls(soup, wb_url):
                    if images_saved >= self.max_images_per_year:
                        break
                    if self._download_image(img_url, year_dir):
                        images_saved += 1

                time.sleep(self.delay)

            results[year] = images_saved
            logger.info("  Year %d: %d images saved", year, images_saved)

        self._save_manifest(url, results)
        return results

    # ------------------------------------------------------------------
    # CDX API
    # ------------------------------------------------------------------

    def _get_snapshots(self, url: str, year: int) -> list[str]:
        params = {
            "url": url,
            "output": "json",
            "fl": "timestamp,statuscode,mimetype",
            "from": f"{year}0101",
            "to": f"{year}1231",
            "filter": "statuscode:200",
            "filter": "mimetype:text/html",
            "collapse": "timestamp:6",
            "limit": str(self.max_snapshots_per_year * 3),
        }
        try:
            resp = self.session.get(CDX_API, params=params, timeout=30)
            resp.raise_for_status()
            rows = resp.json()
            if len(rows) < 2:
                return []
            return [row[0] for row in rows[1:]]
        except Exception as exc:
            logger.warning("CDX query failed for %s/%d: %s", url, year, exc)
            return []

    # ------------------------------------------------------------------
    # Fetch & extract
    # ------------------------------------------------------------------

    def _fetch_snapshot(self, wb_url: str) -> BeautifulSoup | None:
        try:
            resp = self.session.get(wb_url, timeout=20)
            resp.raise_for_status()
            return BeautifulSoup(resp.content, "html.parser")
        except Exception as exc:
            logger.warning("Failed to fetch snapshot %s: %s", wb_url, exc)
            return None

    def _extract_image_urls(self, soup: BeautifulSoup, page_url: str) -> list[str]:
        urls: list[str] = []
        for tag in soup.find_all(["img", "source"]):
            src = tag.get("src") or tag.get("data-src") or ""
            src = src.strip()
            if not src or src.startswith("data:"):
                continue

            if src.startswith("//web.archive.org") or src.startswith("/web/"):
                src = "https:" + src if src.startswith("//") else "https://web.archive.org" + src
            elif not src.startswith("http"):
                src = urljoin(page_url, src)

            if "/web/" in src:
                match = re.search(r"/web/\d+(?:im_|if_|)/(https?://.*)", src)
                if match:
                    src = match.group(1)

            ext = Path(urlparse(src).path).suffix.lower()
            if ext in VALID_EXTENSIONS or not ext:
                urls.append(src)
        return urls

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    def _download_image(self, img_url: str, dest_dir: Path) -> bool:
        try:
            resp = self.session.get(img_url, timeout=10)
            resp.raise_for_status()
            data = resp.content
        except Exception:
            return False

        if len(data) < MIN_FILE_SIZE:
            return False

        content_hash = hashlib.md5(data).hexdigest()
        if content_hash in self._seen_hashes:
            return False

        try:
            img = Image.open(io.BytesIO(data))
            img.verify()
            img = Image.open(io.BytesIO(data))
        except Exception:
            return False

        w, h = img.size
        if w < MIN_WIDTH or h < MIN_HEIGHT:
            return False

        fmt = (img.format or "").lower()
        ext = {"jpeg": ".jpeg", "png": ".png", "webp": ".webp"}.get(fmt)
        if ext is None:
            return False

        self._seen_hashes.add(content_hash)
        filename = content_hash + ext
        (dest_dir / filename).write_bytes(data)
        return True

    # ------------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------------

    def _save_manifest(self, url: str, results: dict[int, int]) -> None:
        manifest_path = self.output_dir / "manifest.csv"
        with open(manifest_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["url", "year", "images_downloaded"])
            for year, count in sorted(results.items()):
                writer.writerow([url, year, count])
        logger.info("Manifest saved to %s", manifest_path)


# ======================================================================
# CLI
# ======================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Wayback Machine image scraper via CDX API")
    parser.add_argument("--url", required=True, help="Website URL to query")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Output directory (year subfolders created inside)")
    parser.add_argument("--start-year", type=int, default=2020, help="First year (default: 2020)")
    parser.add_argument("--end-year", type=int, default=2025, help="Last year (default: 2025)")
    parser.add_argument("--max-snapshots-per-year", type=int, default=5,
                        help="Max CDX snapshots to fetch per year (default: 5)")
    parser.add_argument("--max-images-per-year", type=int, default=50,
                        help="Max images to download per year (default: 50)")
    parser.add_argument("--delay", type=float, default=1.5,
                        help="Delay between snapshot requests (default: 1.5)")
    args = parser.parse_args()

    scraper = WaybackScraper(
        output_dir=args.output_dir,
        start_year=args.start_year,
        end_year=args.end_year,
        max_snapshots_per_year=args.max_snapshots_per_year,
        max_images_per_year=args.max_images_per_year,
        delay=args.delay,
    )
    scraper.scrape(args.url)


if __name__ == "__main__":
    main()
