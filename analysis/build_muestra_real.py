#!/usr/bin/env python3
"""Recoge, por cada sitio de la categoría "news", las N imágenes clasificadas
como "real" con menor score de IA (las más confiadamente "no-IA"), como
conjunto de comparación frente a `muestra_ia_noticias/`.

Misma estructura que `muestra_ia_noticias/`: una subcarpeta por sitio con
archivos `<score:.4f>_<hash>.<ext>` y un `index.csv` con columnas
`sitio,score_ia,archivo,origen`.

Uso:
    python3 build_muestra_real.py --out muestra_real_noticias --top 5
"""
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

NEWS_SITES = [
    "001_euronews-com__euronews",
    "002_www-lavanguardia-com__la-vanguardia",
    "003_www-rtve-es__rtve-noticias",
    "004_www-elmundo-es__el-mundo",
    "005_www-bbc-co-uk__bbc-news",
]

CSV_ROOT = Path("classification_results/run_20260607_200images/csv_by_website")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, help="Carpeta de salida")
    parser.add_argument("--top", type=int, default=5, help="Nº de imágenes 'real' (menor score) por sitio")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    index_rows = []
    for slug in NEWS_SITES:
        csv_path = CSV_ROOT / slug / "images_predictions.csv"
        rows = list(csv.DictReader(open(csv_path)))
        real_rows = [r for r in rows if r["predicted_label_name"] == "real"]
        real_rows.sort(key=lambda r: float(r["score"]))

        site_dir = out_dir / slug
        site_dir.mkdir(parents=True, exist_ok=True)

        for r in real_rows[: args.top]:
            src = Path(r["stored_path"])
            score = float(r["score"])
            dst_name = f"{score:.4f}_{src.name}"
            dst = site_dir / dst_name
            shutil.copy2(src, dst)
            index_rows.append({
                "sitio": slug,
                "score_ia": score,
                "archivo": (site_dir / dst_name).relative_to(out_dir).as_posix(),
                "origen": str(src),
            })
            print(f"  {slug}: {src.name} (score={score:.6g}) -> {dst}")

    with open(out_dir / "index.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sitio", "score_ia", "archivo", "origen"])
        writer.writeheader()
        writer.writerows(index_rows)

    print(f"Listo: {len(index_rows)} imágenes en {out_dir}/")


if __name__ == "__main__":
    main()
