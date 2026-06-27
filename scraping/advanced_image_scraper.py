#!/usr/bin/env python3
"""
Advanced web image scraper with BeautifulSoup + optional Playwright fallback.

Extracts images from HTML <img>, <picture>, <source> tags and CSS background
properties.  Filters by size, format and deduplicates via content + perceptual
hashing.  Stores per-image metadata (HTML tag, parent, CSS classes, zone) for
downstream zone-level analysis.

Usage:
    python advanced_image_scraper.py \
        --url https://www.bbc.co.uk/news \
        --output-dir output/bbc \
        --max-images 150 --max-pages 3

    python advanced_image_scraper.py \
        --url https://www.lavanguardia.com \
        --output-dir output/lavanguardia \
        --use-playwright-fallback --accept-cookies
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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import imagehash
import requests
from bs4 import BeautifulSoup, Tag
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MIN_WIDTH = 100
MIN_HEIGHT = 100
MIN_FILE_SIZE = 5 * 1024  # 5 KB
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
)

ZONE_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("header_nav", ("header", "navbar", "nav", "topbar", "masthead", "menu")),
    ("hero_banner", ("hero", "banner", "jumbotron", "carousel", "slider", "cover", "splash")),
    ("ad_sponsored", ("advert", "sponsor", "promo", "dfp", "adsbygoogle", "banner-ad", "ad-slot", "ad-unit")),
    ("thumbnail_gallery", ("gallery", "thumbnail", "thumb", "grid", "masonry", "tile", "mosaic")),
    ("product_catalog", ("product", "plp", "catalog", "shop", "listing", "item-card", "price")),
    ("sidebar_related", ("sidebar", "aside", "widget", "related", "recommend", "recirculation")),
    ("article_content", ("article", "content", "post", "story", "teaser", "body", "main")),
    ("profile_icon", ("avatar", "profile", "icon", "logo", "badge")),
    ("footer", ("footer", "bottom")),
]


@dataclass
class ImageRecord:
    filename: str
    image_url: str
    page_url: str
    html_tag: str
    parent_tag: str
    classes: str
    element_id: str
    width: int
    height: int
    file_size: int
    format: str
    content_hash: str
    perceptual_hash: str
    zone: str


def classify_zone(classes: str, element_id: str, parent_tag: str) -> str:
    context = f"{classes} {element_id} {parent_tag}".lower()
    tokens = context.split()
    for zone, hints in ZONE_KEYWORDS:
        for hint in hints:
            for token in tokens:
                if token == hint or token.startswith(hint + "-") or token.startswith(hint + "_") \
                        or token.endswith("-" + hint) or token.endswith("_" + hint):
                    return zone
    return "unclassified"


class AdvancedImageScraper:

    def __init__(
        self,
        output_dir: str | Path,
        max_images: int = 200,
        max_pages: int = 5,
        min_images_per_page: int = 0,
        delay: float = 1.0,
        use_playwright: bool = False,
        accept_cookies: bool = False,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / "images"
        self.metadata_dir = self.output_dir / "metadata"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

        self.max_images = max_images
        self.max_pages = max_pages
        self.min_images_per_page = min_images_per_page
        self.delay = delay
        self.use_playwright = use_playwright
        self.accept_cookies = accept_cookies

        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})

        self._seen_content_hashes: set[str] = set()
        self._seen_perceptual_hashes: set[str] = set()
        self._visited_urls: set[str] = set()
        self._records: list[ImageRecord] = []
        self._playwright_browser: Any = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scrape(self, start_url: str) -> list[ImageRecord]:
        to_visit = [(start_url, 0)]
        pages_crawled = 0

        while to_visit and len(self._records) < self.max_images and pages_crawled < self.max_pages:
            url, depth = to_visit.pop(0)
            if url in self._visited_urls:
                continue
            self._visited_urls.add(url)
            pages_crawled += 1

            logger.info(
                "Page %d/%d  images=%d/%d  url=%s",
                pages_crawled, self.max_pages, len(self._records), self.max_images, url,
            )

            soup = self._fetch_page(url)
            if soup is None:
                continue

            images_before = len(self._records)
            self._extract_images(soup, url)
            images_found = len(self._records) - images_before
            logger.info("  Found %d new images on this page", images_found)

            if self.use_playwright and images_found < self.min_images_per_page:
                logger.info("  Few images via BS4 (%d < %d), trying Playwright fallback...",
                            images_found, self.min_images_per_page)
                pw_soup = self._fetch_page_playwright(url)
                if pw_soup is not None:
                    self._extract_images(pw_soup, url)
                    pw_new = len(self._records) - images_before - images_found
                    logger.info("  Playwright added %d extra images", pw_new)

            if depth < self.max_pages - 1:
                links = self._extract_links(soup, url)
                for link in links:
                    if link not in self._visited_urls:
                        to_visit.append((link, depth + 1))

            time.sleep(self.delay)

        self._save_metadata_csv()
        self._close_playwright()
        logger.info("Scraping complete: %d images saved to %s", len(self._records), self.images_dir)
        return self._records

    # ------------------------------------------------------------------
    # Page fetching
    # ------------------------------------------------------------------

    def _fetch_page(self, url: str) -> BeautifulSoup | None:
        try:
            resp = self.session.get(url, timeout=15)
            resp.raise_for_status()
            return BeautifulSoup(resp.content, "html.parser")
        except Exception as exc:
            logger.warning("Failed to fetch %s: %s", url, exc)
            return None

    def _fetch_page_playwright(self, url: str) -> BeautifulSoup | None:
        try:
            from playwright.sync_api import sync_playwright

            if self._playwright_browser is None:
                pw = sync_playwright().start()
                self._playwright_browser = pw.chromium.launch(headless=True)

            page = self._playwright_browser.new_page()
            page.set_extra_http_headers({"User-Agent": USER_AGENT})
            page.goto(url, wait_until="networkidle", timeout=30_000)

            if self.accept_cookies:
                self._dismiss_cookie_banner(page)

            page.wait_for_timeout(2000)
            html = page.content()
            page.close()
            return BeautifulSoup(html, "html.parser")
        except Exception as exc:
            logger.warning("Playwright fallback failed for %s: %s", url, exc)
            return None

    @staticmethod
    def _dismiss_cookie_banner(page: Any) -> None:
        selectors = [
            "button:has-text('Accept')",
            "button:has-text('Aceptar')",
            "button:has-text('Acepto')",
            "button:has-text('Agree')",
            "button:has-text('OK')",
            "button:has-text('Got it')",
            "[id*='cookie'] button",
            "[class*='cookie'] button",
            "[id*='consent'] button",
            "[class*='consent'] button",
        ]
        for sel in selectors:
            try:
                btn = page.query_selector(sel)
                if btn and btn.is_visible():
                    btn.click()
                    page.wait_for_timeout(500)
                    return
            except Exception:
                continue

    def _close_playwright(self) -> None:
        if self._playwright_browser is not None:
            try:
                self._playwright_browser.close()
            except Exception:
                pass
            self._playwright_browser = None

    # ------------------------------------------------------------------
    # Image extraction
    # ------------------------------------------------------------------

    def _extract_images(self, soup: BeautifulSoup, page_url: str) -> None:
        for tag in soup.find_all(["img", "picture", "source"]):
            if len(self._records) >= self.max_images:
                return
            self._process_tag(tag, page_url)

        for tag in soup.find_all(style=True):
            if len(self._records) >= self.max_images:
                return
            self._process_css_background(tag, page_url)

    def _process_tag(self, tag: Tag, page_url: str) -> None:
        img_url = (
            tag.get("src")
            or tag.get("data-src")
            or tag.get("data-lazy-src")
            or tag.get("srcset", "").split(",")[0].strip().split(" ")[0]
        )
        if not img_url:
            return
        img_url = self._resolve_url(img_url, page_url)
        if not img_url:
            return

        parent = tag.parent
        parent_tag = parent.name if parent else ""
        classes = " ".join(tag.get("class", []))
        if parent:
            classes += " " + " ".join(parent.get("class", []))
        element_id = tag.get("id", "")
        if parent and parent.get("id"):
            element_id += " " + parent.get("id", "")

        self._download_and_store(
            img_url=img_url,
            page_url=page_url,
            html_tag=tag.name,
            parent_tag=parent_tag,
            classes=classes.strip(),
            element_id=element_id.strip(),
        )

    def _process_css_background(self, tag: Tag, page_url: str) -> None:
        style = tag.get("style", "")
        match = re.search(r"url\(['\"]?(.*?)['\"]?\)", style)
        if not match:
            return
        img_url = self._resolve_url(match.group(1), page_url)
        if not img_url:
            return

        parent = tag.parent
        parent_tag = parent.name if parent else ""
        classes = " ".join(tag.get("class", []))
        element_id = tag.get("id", "")

        self._download_and_store(
            img_url=img_url,
            page_url=page_url,
            html_tag=f"{tag.name}[css-bg]",
            parent_tag=parent_tag,
            classes=classes.strip(),
            element_id=element_id.strip(),
        )

    # ------------------------------------------------------------------
    # Download, filter, store
    # ------------------------------------------------------------------

    def _download_and_store(
        self,
        img_url: str,
        page_url: str,
        html_tag: str,
        parent_tag: str,
        classes: str,
        element_id: str,
    ) -> None:
        try:
            resp = self.session.get(img_url, timeout=10)
            resp.raise_for_status()
            data = resp.content
        except Exception:
            return

        if len(data) < MIN_FILE_SIZE:
            return

        content_hash = hashlib.md5(data).hexdigest()
        if content_hash in self._seen_content_hashes:
            return

        try:
            img = Image.open(io.BytesIO(data))
            img.verify()
            img = Image.open(io.BytesIO(data))
        except Exception:
            return

        w, h = img.size
        if w < MIN_WIDTH or h < MIN_HEIGHT:
            return

        fmt = (img.format or "").lower()
        ext = {"jpeg": ".jpeg", "png": ".png", "webp": ".webp"}.get(fmt)
        if ext is None:
            return

        try:
            phash = str(imagehash.phash(img))
        except Exception:
            phash = ""
        if phash and phash in self._seen_perceptual_hashes:
            return

        self._seen_content_hashes.add(content_hash)
        if phash:
            self._seen_perceptual_hashes.add(phash)

        filename = content_hash + ext
        dest = self.images_dir / filename
        dest.write_bytes(data)

        zone = classify_zone(classes, element_id, parent_tag)

        record = ImageRecord(
            filename=filename,
            image_url=img_url,
            page_url=page_url,
            html_tag=html_tag,
            parent_tag=parent_tag,
            classes=classes,
            element_id=element_id,
            width=w,
            height=h,
            file_size=len(data),
            format=fmt,
            content_hash=content_hash,
            perceptual_hash=phash,
            zone=zone,
        )
        self._records.append(record)

    # ------------------------------------------------------------------
    # Links
    # ------------------------------------------------------------------

    def _extract_links(self, soup: BeautifulSoup, page_url: str) -> list[str]:
        base_domain = urlparse(page_url).netloc
        links: list[str] = []
        for a in soup.find_all("a", href=True):
            href = self._resolve_url(a["href"], page_url)
            if href and urlparse(href).netloc == base_domain:
                links.append(href)
        return links

    @staticmethod
    def _resolve_url(url: str, base_url: str) -> str | None:
        if not url or not isinstance(url, str):
            return None
        url = url.strip()
        if url.startswith("data:"):
            return None
        if url.startswith("//"):
            return urlparse(base_url).scheme + ":" + url
        if not url.startswith("http"):
            return urljoin(base_url, url)
        return url

    # ------------------------------------------------------------------
    # Metadata CSV
    # ------------------------------------------------------------------

    def _save_metadata_csv(self) -> None:
        csv_path = self.metadata_dir / "images_metadata.csv"
        fieldnames = [
            "filename", "image_url", "page_url", "html_tag", "parent_tag",
            "classes", "element_id", "width", "height", "file_size",
            "format", "content_hash", "perceptual_hash", "zone",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in self._records:
                writer.writerow(r.__dict__)
        logger.info("Metadata saved to %s (%d rows)", csv_path, len(self._records))


# ======================================================================
# CLI
# ======================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Advanced web image scraper with zone classification",
    )
    parser.add_argument("--url", required=True, help="Website URL to scrape")
    parser.add_argument("--output-dir", required=True, help="Output directory for images + metadata")
    parser.add_argument("--max-images", type=int, default=200, help="Max images to download (default: 200)")
    parser.add_argument("--max-pages", type=int, default=5, help="Max pages to crawl (default: 5)")
    parser.add_argument("--min-images-per-page", type=int, default=0,
                        help="Min images before triggering Playwright fallback (default: 0)")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between requests in seconds (default: 1.0)")
    parser.add_argument("--use-playwright-fallback", action="store_true",
                        help="Use Playwright when BS4 finds too few images")
    parser.add_argument("--accept-cookies", action="store_true",
                        help="Auto-dismiss cookie banners (requires Playwright)")
    args = parser.parse_args()

    scraper = AdvancedImageScraper(
        output_dir=args.output_dir,
        max_images=args.max_images,
        max_pages=args.max_pages,
        min_images_per_page=args.min_images_per_page,
        delay=args.delay,
        use_playwright=args.use_playwright_fallback,
        accept_cookies=args.accept_cookies,
    )
    scraper.scrape(args.url)


if __name__ == "__main__":
    main()
