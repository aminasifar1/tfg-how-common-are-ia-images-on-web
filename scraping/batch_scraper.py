#!/usr/bin/env python3
"""
Batch orchestrator: reads a websites CSV and runs advanced_image_scraper
for each site (or a single site by index).

Usage:
    # All sites
    python batch_scraper.py \
        --csv data/websites-list.csv \
        --output-dir batch_scrape_results \
        --max-images 150 --max-pages 2

    # Single site (1-based index)
    python batch_scraper.py \
        --csv data/websites-list.csv \
        --output-dir batch_scrape_results \
        --site-index 3
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from advanced_image_scraper import AdvancedImageScraper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def _slug(url: str, org_name: str) -> str:
    host = re.sub(r"^https?://", "", url).split("/")[0]
    host = re.sub(r"^www\.", "", host)
    host = re.sub(r"[^a-zA-Z0-9]", "-", host).strip("-")
    org = re.sub(r"[^a-zA-Z0-9]", "-", org_name).strip("-").lower()
    return f"{host}__{org}"


def run_batch(
    csv_path: Path,
    output_dir: Path,
    max_images: int = 200,
    max_pages: int = 5,
    min_images_per_page: int = 0,
    delay: float = 1.0,
    use_playwright: bool = False,
    accept_cookies: bool = False,
    site_index: int | None = None,
    extra_args: list[str] | None = None,
) -> dict:
    df = pd.read_csv(csv_path)
    required = {"url", "organization_name"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"CSV must have columns: {required}. Found: {set(df.columns)}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"run_{timestamp}"
    sites_dir = run_dir / "sites"
    logs_dir = run_dir / "logs"
    sites_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    if site_index is not None:
        idx = site_index - 1
        if idx < 0 or idx >= len(df):
            raise IndexError(f"site-index {site_index} out of range (1..{len(df)})")
        df = df.iloc[[idx]]
        logger.info("Running single site: index=%d", site_index)

    summary: list[dict] = []

    for i, row in df.iterrows():
        site_num = int(i) + 1
        url = str(row["url"]).strip()
        org = str(row["organization_name"]).strip()
        sector = str(row.get("sector", "")).strip()
        slug = f"{site_num:03d}_{_slug(url, org)}"
        site_dir = sites_dir / slug

        logger.info("=" * 60)
        logger.info("Site %d/%d: %s  (%s)", site_num, len(df), org, url)
        logger.info("=" * 60)

        scraper = AdvancedImageScraper(
            output_dir=str(site_dir),
            max_images=max_images,
            max_pages=max_pages,
            min_images_per_page=min_images_per_page,
            delay=delay,
            use_playwright=use_playwright,
            accept_cookies=accept_cookies,
        )

        try:
            records = scraper.scrape(url)
            n = len(records)
            logger.info("Site %s: %d images scraped", slug, n)
        except Exception as exc:
            logger.error("Site %s FAILED: %s", slug, exc)
            n = 0

        summary.append({
            "site": slug,
            "url": url,
            "organization_name": org,
            "sector": sector,
            "images_scraped": n,
        })

    summary_path = run_dir / "batch_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"run": timestamp, "sites": summary}, f, indent=2, ensure_ascii=False)
    logger.info("Batch complete. Summary: %s", summary_path)

    return {"run_dir": str(run_dir), "sites": summary}


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch image scraper for website list CSV")
    parser.add_argument("--csv", type=Path, required=True, help="CSV with url, organization_name columns")
    parser.add_argument("--output-dir", type=Path, required=True, help="Base output directory")
    parser.add_argument("--max-images", type=int, default=200, help="Max images per site (default: 200)")
    parser.add_argument("--max-pages", type=int, default=5, help="Max pages to crawl per site (default: 5)")
    parser.add_argument("--min-images-per-page", type=int, default=0,
                        help="Min images before Playwright fallback (default: 0)")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between requests (default: 1.0)")
    parser.add_argument("--use-playwright-fallback", action="store_true",
                        help="Use Playwright when BS4 finds too few images")
    parser.add_argument("--accept-cookies", action="store_true",
                        help="Auto-dismiss cookie banners")
    parser.add_argument("--site-index", type=int, default=None,
                        help="1-based index to scrape only one site from the CSV")
    args, extra = parser.parse_known_args()

    run_batch(
        csv_path=args.csv,
        output_dir=args.output_dir,
        max_images=args.max_images,
        max_pages=args.max_pages,
        min_images_per_page=args.min_images_per_page,
        delay=args.delay,
        use_playwright=args.use_playwright_fallback,
        accept_cookies=args.accept_cookies,
        site_index=args.site_index,
        extra_args=extra,
    )


if __name__ == "__main__":
    main()
