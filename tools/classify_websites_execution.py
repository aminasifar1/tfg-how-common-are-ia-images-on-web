#!/usr/bin/env python3
"""Classify website images and export analysis-ready CSV files.

Expected input: either a CSV with one row per image or a directory of images.

CSV mode typical columns:
- organization_name
- sector
- subsector
- page_url
- image_url or stored_path or image_path
- image_type (optional but recommended)
- site_size_proxy (optional numeric proxy for site size/relevance)

Directory mode:
- scans image files recursively under --images-dir
- classifies every image file found
- writes a CSV with one row per image

The script writes:
- predictions_long.csv
- summary_by_sector.csv
- summary_by_site.csv
- summary_by_image_type.csv
- summary_by_sector_and_image_type.csv
- summary_by_sector_and_site_size.csv, only if site_size_proxy is present
- sector_ai_ranking.csv
- analysis_summary.json

Extra split outputs:
- csv_by_page_type/<sector>_predictions.csv
- csv_by_page_type/<sector>_summary_by_website.csv
- csv_by_page_type/<sector>_summary_by_image_type.csv
- csv_by_website/<sector>/<website>_predictions.csv
- csv_by_website/<sector>/<website>_summary.csv
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_MODEL_DIR = Path("/home/aaasifar/spai-hf")
VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    return str(value).strip()


def _safe_filename(value: Any) -> str:
    """Convert sector/site names into safe filenames."""
    text = _clean_text(value)
    if not text:
        text = "unknown"

    text = text.lower()
    text = re.sub(r"[^\w.-]+", "_", text, flags=re.UNICODE)
    text = text.strip("_.")

    return text[:120] or "unknown"


def _pick_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    normalized = {str(col).strip().lower(): col for col in df.columns}
    for candidate in candidates:
        match = normalized.get(candidate.lower())
        if match is not None:
            return match
    return None


def _resolve_image_input(row: pd.Series) -> str | None:
    for column in ["stored_path", "image_path", "local_path"]:
        if column in row.index:
            value = _clean_text(row.get(column))
            if value:
                path = Path(value)
                if path.exists():
                    return str(path)

    for column in ["image_url", "url", "image"]:
        if column in row.index:
            value = _clean_text(row.get(column))
            if value:
                return value

    return None


def _collect_image_files(images_dir: Path) -> list[Path]:
    if not images_dir.exists() or not images_dir.is_dir():
        raise ValueError(f"Images directory does not exist or is not a directory: {images_dir}")

    return sorted(
        p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTS
    )


def _to_numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _safe_mean(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.mean())


def _safe_median(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.median())


def _safe_std(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if len(values) < 2:
        return None
    return float(values.std(ddof=1))


def _rate(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.mean())


def _aggregate_frame(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouped = df.groupby(group_cols, dropna=False)

    for keys, group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)

        record: dict[str, Any] = {}
        for col, key in zip(group_cols, keys):
            record[col] = key

        record["total_images"] = int(len(group))
        record["ai_images"] = int(group["is_ai"].sum())
        record["real_images"] = int((group["is_ai"] == 0).sum())
        record["ai_rate"] = float(group["is_ai"].mean())
        record["real_rate"] = float((group["is_ai"] == 0).mean())
        record["score_mean"] = float(group["score"].mean())
        record["score_median"] = float(group["score"].median())
        record["score_std"] = float(group["score"].std(ddof=1)) if len(group) > 1 else None
        record["score_min"] = float(group["score"].min())
        record["score_max"] = float(group["score"].max())
        rows.append(record)

    result = pd.DataFrame(rows)
    if not result.empty:
        sort_cols = group_cols + ["total_images"]
        result = result.sort_values(
            sort_cols,
            ascending=[True] * len(group_cols) + [False],
        ).reset_index(drop=True)
    return result


def _export_split_csvs(pred_df: pd.DataFrame, output_dir: Path) -> dict[str, Any]:
    """Export one CSV per page type/sector and one CSV per website."""
    artifacts: dict[str, Any] = {
        "csv_by_page_type_dir": str(output_dir / "csv_by_page_type"),
        "csv_by_website_dir": str(output_dir / "csv_by_website"),
        "page_type_prediction_csvs": [],
        "page_type_summary_csvs": [],
        "website_prediction_csvs": [],
        "website_summary_csvs": [],
    }

    by_page_type_dir = output_dir / "csv_by_page_type"
    by_website_dir = output_dir / "csv_by_website"

    by_page_type_dir.mkdir(parents=True, exist_ok=True)
    by_website_dir.mkdir(parents=True, exist_ok=True)

    # 1) One predictions CSV per sector/page type
    for sector, sector_df in pred_df.groupby("sector", dropna=False):
        sector_name = _safe_filename(sector)

        sector_predictions_csv = by_page_type_dir / f"{sector_name}_predictions.csv"
        sector_summary_by_website_csv = by_page_type_dir / f"{sector_name}_summary_by_website.csv"
        sector_summary_by_image_type_csv = by_page_type_dir / f"{sector_name}_summary_by_image_type.csv"

        sector_df.to_csv(sector_predictions_csv, index=False)

        _aggregate_frame(sector_df, ["organization_name"]).to_csv(
            sector_summary_by_website_csv,
            index=False,
        )

        _aggregate_frame(sector_df, ["image_type"]).to_csv(
            sector_summary_by_image_type_csv,
            index=False,
        )

        artifacts["page_type_prediction_csvs"].append(str(sector_predictions_csv))
        artifacts["page_type_summary_csvs"].extend(
            [
                str(sector_summary_by_website_csv),
                str(sector_summary_by_image_type_csv),
            ]
        )

    # 2) One predictions CSV per website, organized inside its sector folder
    for (sector, website), site_df in pred_df.groupby(
        ["sector", "organization_name"],
        dropna=False,
    ):
        sector_name = _safe_filename(sector)
        website_name = _safe_filename(website)

        website_dir = by_website_dir / sector_name
        website_dir.mkdir(parents=True, exist_ok=True)

        website_predictions_csv = website_dir / f"{website_name}_predictions.csv"
        website_summary_csv = website_dir / f"{website_name}_summary.csv"

        site_df.to_csv(website_predictions_csv, index=False)

        _aggregate_frame(site_df, ["organization_name"]).to_csv(
            website_summary_csv,
            index=False,
        )

        artifacts["website_prediction_csvs"].append(str(website_predictions_csv))
        artifacts["website_summary_csvs"].append(str(website_summary_csv))

    return artifacts


def classify_manifest(
    metadata_csv: Path,
    output_dir: Path,
    model_dir: Path,
    threshold: float,
    max_images: int,
) -> dict[str, Any]:
    try:
        from inference import EndpointHandler
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Could not import the local inference stack. Run this inside the project environment "
            "with the repo dependencies installed."
        ) from exc

    df = pd.read_csv(metadata_csv)
    if df.empty:
        raise ValueError(f"Empty metadata CSV: {metadata_csv}")

    image_col = _pick_column(
        df,
        ["image_url", "url", "image", "stored_path", "image_path", "local_path"],
    )
    if image_col is None:
        raise ValueError(
            "Could not find an image column. Expected one of: image_url, url, image, "
            "stored_path, image_path, local_path"
        )

    sector_col = _pick_column(df, ["sector"])
    org_col = _pick_column(df, ["organization_name", "organization", "site_name", "web_name"])
    subsector_col = _pick_column(df, ["subsector"])
    page_col = _pick_column(df, ["page_url", "source_page", "page"])
    image_type_col = _pick_column(df, ["image_type", "img_type", "type"])
    site_size_col = _pick_column(
        df,
        [
            "site_size_proxy",
            "site_size",
            "rank_proxy",
            "relevance_proxy",
            "alexa_rank",
            "tranco_rank",
        ],
    )

    if max_images > 0:
        df = df.head(max_images).copy()

    output_dir.mkdir(parents=True, exist_ok=True)

    os.environ["SPAI_THRESHOLD"] = str(threshold)
    handler = EndpointHandler(path=str(model_dir))

    rows: list[dict[str, Any]] = []
    failures = 0
    failure_samples: list[str] = []

    for idx, (_, row) in enumerate(df.iterrows(), start=1):
        image_input = _resolve_image_input(row)
        if not image_input:
            failures += 1
            if len(failure_samples) < 10:
                failure_samples.append(f"row {idx}: no image input found")
            continue

        try:
            pred = handler({"inputs": image_input})
            score = float(pred["score"])
            predicted_label = int(pred["predicted_label"])

            record = {
                "row_index": idx,
                "image_input": image_input,
                "score": score,
                "predicted_label": predicted_label,
                "predicted_label_name": pred["predicted_label_name"],
                "threshold": float(pred["threshold"]),
                "is_ai": int(predicted_label == 1),
                "organization_name": _clean_text(row.get(org_col)) if org_col else "",
                "sector": _clean_text(row.get(sector_col)) if sector_col else "",
                "subsector": _clean_text(row.get(subsector_col)) if subsector_col else "",
                "page_url": _clean_text(row.get(page_col)) if page_col else "",
                "image_url": _clean_text(row.get(image_col)) if image_col else "",
                "image_type": _clean_text(row.get(image_type_col)) if image_type_col else "",
                "site_size_proxy": _to_numeric_series(pd.Series([row.get(site_size_col)])).iloc[0]
                if site_size_col
                else np.nan,
            }
            rows.append(record)
        except Exception as exc:
            failures += 1
            if len(failure_samples) < 10:
                failure_samples.append(f"row {idx}: {image_input}: {exc}")

    if not rows:
        detail = ""
        if failure_samples:
            detail = " First failures: " + " | ".join(failure_samples)
        raise RuntimeError("No images could be classified from the provided CSV." + detail)

    pred_df = pd.DataFrame(rows)
    pred_df["site_size_proxy"] = pd.to_numeric(pred_df["site_size_proxy"], errors="coerce")

    predictions_csv = output_dir / "predictions_long.csv"
    pred_df.to_csv(predictions_csv, index=False)

    split_artifacts = _export_split_csvs(pred_df, output_dir)

    summary_by_sector = _aggregate_frame(pred_df, ["sector"])
    summary_by_site = _aggregate_frame(pred_df, ["sector", "organization_name"])
    summary_by_image_type = _aggregate_frame(pred_df, ["sector", "image_type"])
    summary_by_sector_and_image_type = _aggregate_frame(
        pred_df,
        ["sector", "organization_name", "image_type"],
    )

    summary_by_sector.to_csv(output_dir / "summary_by_sector.csv", index=False)
    summary_by_site.to_csv(output_dir / "summary_by_site.csv", index=False)
    summary_by_image_type.to_csv(output_dir / "summary_by_image_type.csv", index=False)
    summary_by_sector_and_image_type.to_csv(
        output_dir / "summary_by_sector_and_image_type.csv",
        index=False,
    )

    sector_size_summary = None
    if pred_df["site_size_proxy"].notna().any():
        sector_size_summary = (
            pred_df.groupby(["sector", "organization_name"], dropna=False)
            .agg(
                total_images=("score", "size"),
                ai_rate=("is_ai", "mean"),
                score_mean=("score", "mean"),
                score_median=("score", "median"),
                site_size_proxy=("site_size_proxy", "mean"),
            )
            .reset_index()
        )
        sector_size_summary.to_csv(output_dir / "summary_by_sector_and_site_size.csv", index=False)

    corr_ai_rate = None
    corr_score_mean = None
    if pred_df["site_size_proxy"].notna().sum() >= 3:
        site_level = (
            pred_df.groupby(["sector", "organization_name"], dropna=False)
            .agg(
                ai_rate=("is_ai", "mean"),
                score_mean=("score", "mean"),
                site_size_proxy=("site_size_proxy", "mean"),
            )
            .reset_index()
        )
        valid = site_level.dropna(subset=["site_size_proxy", "ai_rate"])
        if len(valid) >= 3:
            corr_ai_rate = float(valid["site_size_proxy"].corr(valid["ai_rate"]))

        valid_score = site_level.dropna(subset=["site_size_proxy", "score_mean"])
        if len(valid_score) >= 3:
            corr_score_mean = float(valid_score["site_size_proxy"].corr(valid_score["score_mean"]))

    sector_ai_order = summary_by_sector.sort_values("ai_rate", ascending=False)[
        ["sector", "ai_rate", "score_mean"]
    ]
    if not sector_ai_order.empty:
        sector_ai_order.to_csv(output_dir / "sector_ai_ranking.csv", index=False)

    summary = {
        "metadata_csv": str(metadata_csv),
        "output_dir": str(output_dir),
        "threshold": float(threshold),
        "total_rows_input": int(len(df)),
        "classified": int(len(pred_df)),
        "failed": int(failures),
        "failure_samples": failure_samples,
        "ai_count": int(pred_df["is_ai"].sum()),
        "real_count": int((pred_df["is_ai"] == 0).sum()),
        "ai_rate": float(pred_df["is_ai"].mean()),
        "score_mean": float(pred_df["score"].mean()),
        "score_median": float(pred_df["score"].median()),
        "score_std": float(pred_df["score"].std(ddof=1)) if len(pred_df) > 1 else None,
        "site_size_proxy_correlation_ai_rate": corr_ai_rate,
        "site_size_proxy_correlation_score_mean": corr_score_mean,
        "artifacts": {
            "predictions_long_csv": str(predictions_csv),
            "summary_by_sector_csv": str(output_dir / "summary_by_sector.csv"),
            "summary_by_site_csv": str(output_dir / "summary_by_site.csv"),
            "summary_by_image_type_csv": str(output_dir / "summary_by_image_type.csv"),
            "summary_by_sector_and_image_type_csv": str(
                output_dir / "summary_by_sector_and_image_type.csv"
            ),
            "summary_by_sector_and_site_size_csv": str(
                output_dir / "summary_by_sector_and_site_size.csv"
            )
            if sector_size_summary is not None
            else None,
            "sector_ai_ranking_csv": str(output_dir / "sector_ai_ranking.csv"),
            "csv_by_page_type_dir": split_artifacts["csv_by_page_type_dir"],
            "csv_by_website_dir": split_artifacts["csv_by_website_dir"],
            "page_type_prediction_csvs": split_artifacts["page_type_prediction_csvs"],
            "page_type_summary_csvs": split_artifacts["page_type_summary_csvs"],
            "website_prediction_csvs": split_artifacts["website_prediction_csvs"],
            "website_summary_csvs": split_artifacts["website_summary_csvs"],
        },
        "columns_used": {
            "image": image_col,
            "sector": sector_col,
            "organization_name": org_col,
            "subsector": subsector_col,
            "page_url": page_col,
            "image_type": image_type_col,
            "site_size_proxy": site_size_col,
        },
    }

    summary_json = output_dir / "analysis_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    return summary


def classify_images_dir(
    images_dir: Path,
    output_dir: Path,
    model_dir: Path,
    threshold: float,
    max_images: int,
) -> dict[str, Any]:
    try:
        from inference import EndpointHandler
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Could not import the local inference stack. Run this inside the project environment "
            "with the repo dependencies installed."
        ) from exc

    image_paths = _collect_image_files(images_dir)
    if not image_paths:
        raise ValueError(f"No images found in: {images_dir}")

    if max_images > 0:
        image_paths = image_paths[:max_images]

    output_dir.mkdir(parents=True, exist_ok=True)

    os.environ["SPAI_THRESHOLD"] = str(threshold)
    handler = EndpointHandler(path=str(model_dir))

    rows: list[dict[str, Any]] = []
    failures = 0
    failure_samples: list[str] = []

    for idx, img_path in enumerate(image_paths, start=1):
        try:
            pred = handler({"inputs": str(img_path)})
            score = float(pred["score"])
            predicted_label = int(pred["predicted_label"])

            rows.append(
                {
                    "row_index": idx,
                    "image_input": str(img_path),
                    "image_path": str(img_path),
                    "stored_path": str(img_path),
                    "score": score,
                    "predicted_label": predicted_label,
                    "predicted_label_name": pred["predicted_label_name"],
                    "threshold": float(pred["threshold"]),
                    "is_ai": int(predicted_label == 1),
                    "organization_name": img_path.parent.name,
                    "sector": img_path.parent.parent.name
                    if img_path.parent.parent != images_dir
                    else "",
                    "subsector": "",
                    "page_url": "",
                    "image_url": "",
                    "image_type": "",
                    "site_size_proxy": np.nan,
                }
            )
        except Exception as exc:
            failures += 1
            if len(failure_samples) < 10:
                failure_samples.append(f"{img_path}: {exc}")

    if not rows:
        detail = ""
        if failure_samples:
            detail = " First failures: " + " | ".join(failure_samples)
        raise RuntimeError("No images could be classified from the provided directory." + detail)

    pred_df = pd.DataFrame(rows)

    predictions_csv = output_dir / "predictions_long.csv"
    pred_df.to_csv(predictions_csv, index=False)

    split_artifacts = _export_split_csvs(pred_df, output_dir)

    summary_by_sector = _aggregate_frame(pred_df, ["sector"])
    summary_by_site = _aggregate_frame(pred_df, ["sector", "organization_name"])
    summary_by_image_type = _aggregate_frame(pred_df, ["sector", "image_type"])
    summary_by_sector_and_image_type = _aggregate_frame(
        pred_df,
        ["sector", "organization_name", "image_type"],
    )

    summary_by_sector.to_csv(output_dir / "summary_by_sector.csv", index=False)
    summary_by_site.to_csv(output_dir / "summary_by_site.csv", index=False)
    summary_by_image_type.to_csv(output_dir / "summary_by_image_type.csv", index=False)
    summary_by_sector_and_image_type.to_csv(
        output_dir / "summary_by_sector_and_image_type.csv",
        index=False,
    )

    sector_ai_order = summary_by_sector.sort_values("ai_rate", ascending=False)[
        ["sector", "ai_rate", "score_mean"]
    ]
    if not sector_ai_order.empty:
        sector_ai_order.to_csv(output_dir / "sector_ai_ranking.csv", index=False)

    summary = {
        "images_dir": str(images_dir),
        "output_dir": str(output_dir),
        "threshold": float(threshold),
        "total_images_input": int(len(image_paths)),
        "classified": int(len(pred_df)),
        "failed": int(failures),
        "failure_samples": failure_samples,
        "ai_count": int(pred_df["is_ai"].sum()),
        "real_count": int((pred_df["is_ai"] == 0).sum()),
        "ai_rate": float(pred_df["is_ai"].mean()),
        "score_mean": float(pred_df["score"].mean()),
        "score_median": float(pred_df["score"].median()),
        "score_std": float(pred_df["score"].std(ddof=1)) if len(pred_df) > 1 else None,
        "artifacts": {
            "predictions_long_csv": str(predictions_csv),
            "summary_by_sector_csv": str(output_dir / "summary_by_sector.csv"),
            "summary_by_site_csv": str(output_dir / "summary_by_site.csv"),
            "summary_by_image_type_csv": str(output_dir / "summary_by_image_type.csv"),
            "summary_by_sector_and_image_type_csv": str(
                output_dir / "summary_by_sector_and_image_type.csv"
            ),
            "sector_ai_ranking_csv": str(output_dir / "sector_ai_ranking.csv"),
            "csv_by_page_type_dir": split_artifacts["csv_by_page_type_dir"],
            "csv_by_website_dir": split_artifacts["csv_by_website_dir"],
            "page_type_prediction_csvs": split_artifacts["page_type_prediction_csvs"],
            "page_type_summary_csvs": split_artifacts["page_type_summary_csvs"],
            "website_prediction_csvs": split_artifacts["website_prediction_csvs"],
            "website_summary_csvs": split_artifacts["website_summary_csvs"],
        },
    }

    summary_json = output_dir / "analysis_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify website images and export analysis-ready CSV files."
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--metadata-csv",
        type=Path,
        help="CSV with one row per image and metadata columns.",
    )
    input_group.add_argument(
        "--images-dir",
        type=Path,
        help="Directory with images to classify recursively.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where CSV summaries will be written.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Path to the SPAI model repository root.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.35,
        help="Decision threshold used to convert score into predicted_label.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Maximum number of images to classify. 0 = all.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.images_dir is not None:
        summary = classify_images_dir(
            images_dir=args.images_dir,
            output_dir=args.output_dir,
            model_dir=args.model_dir,
            threshold=args.threshold,
            max_images=args.max_images,
        )
    else:
        summary = classify_manifest(
            metadata_csv=args.metadata_csv,
            output_dir=args.output_dir,
            model_dir=args.model_dir,
            threshold=args.threshold,
            max_images=args.max_images,
        )

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()