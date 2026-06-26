#!/usr/bin/env python3
"""Cruza TODAS las imágenes "article content" (`extract_article_content_images.py`,
heurística por clases CSS) con las predicciones del clasificador SPAI sobre el
mismo scraping (`classification_results/run_20260607_200images/predictions_long.csv`).

A diferencia de una primera versión de este cruce, aquí NO se filtra por
is_ai==1: se conservan TODAS las imágenes (reales + IA), porque para calcular
tasas de IA por tipo de publicación o por sitio se necesita el denominador
completo (total de imágenes), no solo las marcadas como IA.

Genera, en --out:
  - article_content_images_with_predictions.csv : una fila por imagen
    "article content" con predicción cruzada (is_ai, score,
    predicted_label_name, threshold), tanto reales como IA.
  - article_content_ai_rate_by_site.csv : total / ai_count / real_count /
    ai_share por sitio.

Uso:
    python3 cross_reference_article_content_predictions.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--article-content",
        default="article_content_images/article_content_images.csv",
        help="CSV generado por extract_article_content_images.py",
    )
    parser.add_argument(
        "--predictions",
        default="classification_results/run_20260607_200images/predictions_long.csv",
        help="predictions_long.csv del clasificador SPAI sobre el mismo scraping",
    )
    parser.add_argument("--out", default="article_content_images", help="Carpeta de salida")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    articles = pd.read_csv(args.article_content, dtype=str, keep_default_na=False)
    preds = pd.read_csv(args.predictions, dtype={"image_path": str})
    preds_small = preds[["image_path", "score", "predicted_label", "predicted_label_name", "threshold", "is_ai"]]

    merged = articles.merge(preds_small, left_on="image_abs_path", right_on="image_path", how="left")
    unmatched = merged["is_ai"].isna().sum()
    if unmatched:
        print(f"[AVISO] {unmatched}/{len(merged)} imágenes 'article content' sin predicción cruzada (no estaban en {args.predictions}), se descartan")

    with_preds = merged.dropna(subset=["is_ai"]).copy()
    with_preds = with_preds.drop(columns=["image_path"])
    with_preds["is_ai"] = with_preds["is_ai"].astype(int)

    out_csv = out_dir / "article_content_images_with_predictions.csv"
    with_preds.to_csv(out_csv, index=False)
    print(f"Escrito {out_csv} ({len(with_preds)} filas, {int(with_preds['is_ai'].sum())} IA / {len(with_preds)} total)")

    counts = with_preds.groupby("sitio")["is_ai"].agg(total="size", ai_count="sum").reset_index()
    counts["real_count"] = counts["total"] - counts["ai_count"]
    counts["ai_share"] = counts["ai_count"] / counts["total"]
    counts_csv = out_dir / "article_content_ai_rate_by_site.csv"
    counts.to_csv(counts_csv, index=False)
    print(f"Escrito {counts_csv}")
    print("\nTasa de IA en 'article content', por sitio (con denominador completo):")
    print(counts.to_string(index=False))


if __name__ == "__main__":
    main()
