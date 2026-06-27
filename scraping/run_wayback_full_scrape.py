#!/usr/bin/env python3
"""
Batch Wayback Machine scraper: reads a websites CSV and scrapes archived
images for each site across years 2020-2025.

Usage:
    python run_wayback_full_scrape.py \
        --csv data/websites-news-arts.csv \
        --output wayback_images_by_year/output
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

from wayback_machine import WaybackScraper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def _site_slug(url: str) -> str:
    host = re.sub(r"^https?://", "", url).split("/")[0]
    host = re.sub(r"^www\.", "", host)
    return re.sub(r"[^a-zA-Z0-9]", "-", host).strip("-")


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch Wayback Machine scraper")
    parser.add_argument("--csv", type=Path, required=True, help="CSV with url column")
    parser.add_argument("--output", type=Path, required=True, help="Output root directory")
    parser.add_argument("--start-year", type=int, default=2020)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--max-snapshots-per-year", type=int, default=5)
    parser.add_argument("--max-images-per-year", type=int, default=50)
    parser.add_argument("--delay", type=float, default=1.5)
    parser.add_argument("--site-index", type=int, default=None,
                        help="1-based index to process a single site")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if "url" not in df.columns:
        raise ValueError("CSV must have a 'url' column")

    if args.site_index is not None:
        idx = args.site_index - 1
        if idx < 0 or idx >= len(df):
            raise IndexError(f"site-index {args.site_index} out of range (1..{len(df)})")
        df = df.iloc[[idx]]

    sites_dir = args.output / "sites"
    sites_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for _, row in df.iterrows():
        url = str(row["url"]).strip()
        slug = _site_slug(url)
        site_dir = sites_dir / slug

        logger.info("=" * 60)
        logger.info("Wayback scrape: %s -> %s", url, site_dir)
        logger.info("=" * 60)

        scraper = WaybackScraper(
            output_dir=site_dir,
            start_year=args.start_year,
            end_year=args.end_year,
            max_snapshots_per_year=args.max_snapshots_per_year,
            max_images_per_year=args.max_images_per_year,
            delay=args.delay,
        )
        results = scraper.scrape(url)
        all_results[slug] = results

    summary_path = args.output / "wayback_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Wayback batch complete. Summary: %s", summary_path)


if __name__ == "__main__":
    main()
