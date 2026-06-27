#!/usr/bin/env python3
"""
Extract images that belong to article body content, filtering out
navigation, ads, sidebars, headers and footers.

Reads the images_metadata.csv produced by the scraper and copies only
images whose zone == 'article_content' into a separate directory.

Usage:
    python extract_article_content_images.py \
        --scrape-dir results/batch_scrape_results/run_20260605_151433 \
        --output-dir results/article_content_images
"""

from __future__ import annotations

import argparse
import csv
import glob
import logging
import shutil
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ARTICLE_ZONES = {"article_content", "articulo_contenido"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract article-content images from scrape results")
    parser.add_argument("--scrape-dir", type=Path, required=True,
                        help="Root of a scrape run (contains sites/)")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Output directory for article content images")
    args = parser.parse_args()

    sites_dir = args.scrape_dir / "sites"
    if not sites_dir.exists():
        sites_dir = args.scrape_dir
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []

    for meta_path in sorted(sites_dir.glob("*/metadata/images_metadata.csv")):
        site_dir = meta_path.parent.parent
        site_name = site_dir.name
        images_dir = site_dir / "images"

        df = pd.read_csv(meta_path)
        if "zone" not in df.columns:
            logger.warning("No 'zone' column in %s, skipping", meta_path)
            continue

        article_df = df[df["zone"].isin(ARTICLE_ZONES)]
        if article_df.empty:
            continue

        dest = args.output_dir / site_name
        dest.mkdir(parents=True, exist_ok=True)

        for _, row in article_df.iterrows():
            filename = row["filename"]
            src = images_dir / filename
            if src.exists():
                shutil.copy2(src, dest / filename)
                all_rows.append({
                    "site": site_name,
                    "filename": filename,
                    "image_url": row.get("image_url", ""),
                    "page_url": row.get("page_url", ""),
                    "zone": row["zone"],
                })

        logger.info("%s: %d article images extracted", site_name, len(article_df))

    if all_rows:
        out_csv = args.output_dir / "article_content_images.csv"
        pd.DataFrame(all_rows).to_csv(out_csv, index=False)
        logger.info("Total: %d article images -> %s", len(all_rows), out_csv)
    else:
        logger.info("No article content images found")


if __name__ == "__main__":
    main()
