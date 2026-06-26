#!/usr/bin/env python3
"""Combina los resultados por sitio de `classify_wayback_years.py`
(carpetas `wayback_classification_results/<sitio>/`) en CSVs únicos para
analizar todo en conjunto sin entrar en cada carpeta.

Genera, en --out (por defecto el propio --root):
  - combined_yearly_summary.csv : una fila por (sitio, año), igual que
    `wayback_yearly_summary.csv` de cada sitio pero con columna `sitio`.
  - combined_all_predictions.csv : una fila por imagen (todos los sitios y
    años), igual que `wayback_all_years_predictions.csv` de cada sitio pero
    con columna `sitio`.
  - global_yearly_summary.csv : una fila por año, totales/medias agregando
    TODOS los sitios juntos.

Uso:
    python3 combine_wayback_results.py --root wayback_classification_results
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="wayback_classification_results", help="Carpeta con un subdirectorio por sitio")
    parser.add_argument("--out", default=None, help="Carpeta de salida (por defecto, la misma que --root)")
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out) if args.out else root
    out_dir.mkdir(parents=True, exist_ok=True)

    site_dirs = sorted(p for p in root.iterdir() if p.is_dir() and (p / "wayback_yearly_summary.csv").exists())
    if not site_dirs:
        raise SystemExit(f"No se encontró ningún wayback_yearly_summary.csv en {root}")

    summary_frames = []
    pred_frames = []
    for site_dir in site_dirs:
        sitio = site_dir.name

        summary = pd.read_csv(site_dir / "wayback_yearly_summary.csv")
        summary.insert(0, "sitio", sitio)
        summary = summary.drop(columns=["predictions_csv", "histogram_png", "bars_png"], errors="ignore")
        summary_frames.append(summary)

        preds_path = site_dir / "wayback_all_years_predictions.csv"
        if preds_path.exists():
            preds = pd.read_csv(preds_path)
            preds.insert(0, "sitio", sitio)
            pred_frames.append(preds)

    combined_summary = pd.concat(summary_frames, ignore_index=True)
    combined_summary_path = out_dir / "combined_yearly_summary.csv"
    combined_summary.to_csv(combined_summary_path, index=False)
    print(f"Escrito {combined_summary_path} ({len(combined_summary)} filas)")

    combined_preds = pd.concat(pred_frames, ignore_index=True)
    combined_preds_path = out_dir / "combined_all_predictions.csv"
    combined_preds.to_csv(combined_preds_path, index=False)
    print(f"Escrito {combined_preds_path} ({len(combined_preds)} filas)")

    global_summary = combined_summary.groupby("year", as_index=False).agg(
        total_images=("total_images", "sum"),
        real_count=("real_count", "sum"),
        ai_count=("ai_count", "sum"),
    )
    global_summary["ai_share"] = global_summary["ai_count"] / global_summary["total_images"]
    global_summary["real_share"] = global_summary["real_count"] / global_summary["total_images"]

    score_means = combined_preds.groupby("year")["score"].mean().rename("score_mean")
    global_summary = global_summary.merge(score_means, on="year")

    global_summary_path = out_dir / "global_yearly_summary.csv"
    global_summary.to_csv(global_summary_path, index=False)
    print(f"Escrito {global_summary_path} ({len(global_summary)} filas)")

    print("\nResumen global por año (todos los sitios):")
    print(global_summary.to_string(index=False))

    per_site = combined_summary.groupby("sitio", as_index=False).agg(
        total_images=("total_images", "sum"),
        real_count=("real_count", "sum"),
        ai_count=("ai_count", "sum"),
    )
    per_site["ai_share"] = per_site["ai_count"] / per_site["total_images"]
    print("\nResumen por sitio (todos los años):")
    print(per_site.to_string(index=False))


if __name__ == "__main__":
    main()
