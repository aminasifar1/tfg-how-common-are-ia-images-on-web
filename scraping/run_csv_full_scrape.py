#!/usr/bin/env python3
"""Convenience launcher for batch_scraper.  Equivalent to calling batch_scraper
directly but with shorter flag names matching the SLURM env vars."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from batch_scraper import run_batch


def main() -> None:
    parser = argparse.ArgumentParser(description="Full batch scrape from CSV")
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-images", type=int, default=200)
    parser.add_argument("--max-pages", type=int, default=5)
    parser.add_argument("--min-images-per-page", type=int, default=0)
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--use-playwright-fallback", action="store_true")
    parser.add_argument("--accept-cookies", action="store_true")
    parser.add_argument("--site-index", type=int, default=None)
    args, extra = parser.parse_known_args()

    run_batch(
        csv_path=args.csv,
        output_dir=args.output,
        max_images=args.max_images,
        max_pages=args.max_pages,
        min_images_per_page=args.min_images_per_page,
        delay=args.delay,
        use_playwright=args.use_playwright_fallback,
        accept_cookies=args.accept_cookies,
        site_index=args.site_index,
    )


if __name__ == "__main__":
    main()
