#!/usr/bin/env python3
"""
Collect all AI-classified images into a single folder, organized by website.

Reads classification result CSVs, filters images with predicted_label == 1,
and copies the original images into output/<site>/.

Usage:
    python create_ai_images_folder.py \
        --results-dir results/classification_results/run_20260607_200images \
        --images-dir results/batch_scrape_results/run_20260605_151433/sites \
        --output-dir results/ai_images_all
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect AI-classified images into one folder")
    parser.add_argument("--results-dir", type=Path, required=True,
                        help="Directory with classification CSVs (predictions_long.csv or per-site CSVs)")
    parser.add_argument("--images-dir", type=Path, required=True,
                        help="Root directory with scraped images (sites/*/images/)")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Output directory for AI images")
    parser.add_argument("--threshold", type=float, default=0.35,
                        help="Score threshold for AI classification (default: 0.35)")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    predictions_file = args.results_dir / "predictions_long.csv"
    if predictions_file.exists():
        df = pd.read_csv(predictions_file)
    else:
        csvs = list(args.results_dir.rglob("*predictions*.csv"))
        if not csvs:
            logger.error("No prediction CSVs found in %s", args.results_dir)
            return
        df = pd.concat([pd.read_csv(c) for c in csvs], ignore_index=True)

    score_col = "score" if "score" in df.columns else None
    if score_col:
        ai_df = df[df[score_col] >= args.threshold]
    elif "predicted_label" in df.columns:
        ai_df = df[df["predicted_label"] == 1]
    else:
        logger.error("No score or predicted_label column found")
        return

    logger.info("Found %d AI images (threshold=%.2f)", len(ai_df), args.threshold)

    copied = 0
    for _, row in ai_df.iterrows():
        image_path = None
        for col in ["image_path", "stored_path", "local_path"]:
            if col in row.index and pd.notna(row[col]):
                p = Path(str(row[col]))
                if p.exists():
                    image_path = p
                    break

        if image_path is None:
            for col in ["image_path", "stored_path", "local_path"]:
                if col in row.index and pd.notna(row[col]):
                    parts = Path(str(row[col])).parts
                    for i, part in enumerate(parts):
                        if part == "sites" and i + 2 < len(parts):
                            candidate = args.images_dir / parts[i + 1] / "images" / parts[-1]
                            if candidate.exists():
                                image_path = candidate
                                break
                if image_path:
                    break

        if image_path is None:
            continue

        site_name = image_path.parent.parent.name if image_path.parent.name == "images" else "unknown"
        dest_dir = args.output_dir / site_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, dest_dir / image_path.name)
        copied += 1

    logger.info("Copied %d AI images to %s", copied, args.output_dir)


if __name__ == "__main__":
    main()
