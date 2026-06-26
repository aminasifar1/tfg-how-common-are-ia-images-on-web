#!/usr/bin/env python3
"""Combina los `manifest.csv` de cada sitio
(`wayback_images_by_year/output/sites/<sitio>/manifest.csv`) en un único CSV,
añadiendo una columna `sitio`, para ver el estado de la descarga
(downloaded/cached/not_found/too_small/too_large/failed) de todas las
imágenes de todos los sitios sin entrar en cada carpeta.

Genera, en --out (por defecto el propio --root):
  - combined_manifest.csv : una fila por intento de descarga (todos los
    sitios), igual que cada `manifest.csv` pero con columna `sitio`.
  - combined_manifest_status_by_year.csv : recuento de cada `status` por
    (sitio, año).

Uso:
    python3 combine_wayback_manifests.py --root wayback_images_by_year/output
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="wayback_images_by_year/output", help="Carpeta con sites/<sitio>/manifest.csv")
    parser.add_argument("--out", default=None, help="Carpeta de salida (por defecto, --root)")
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out) if args.out else root
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_paths = sorted((root / "sites").glob("*/manifest.csv"))
    if not manifest_paths:
        raise SystemExit(f"No se encontró ningún manifest.csv en {root / 'sites'}")

    frames = []
    for path in manifest_paths:
        sitio = path.parent.name
        df = pd.read_csv(path)
        if df.empty:
            continue
        df.insert(0, "sitio", sitio)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined_path = out_dir / "combined_manifest.csv"
    combined.to_csv(combined_path, index=False)
    print(f"Escrito {combined_path} ({len(combined)} filas)")

    status_long = combined.groupby(["sitio", "year", "status"]).size().reset_index(name="count")
    status_by_year = status_long.pivot_table(index=["sitio", "year"], columns="status", values="count", fill_value=0).reset_index()
    status_by_year.columns.name = None
    status_by_year_path = out_dir / "combined_manifest_status_by_year.csv"
    status_by_year.to_csv(status_by_year_path, index=False)
    print(f"Escrito {status_by_year_path} ({len(status_by_year)} filas)")

    print("\nEstados por sitio (todos los años):")
    per_site_long = combined.groupby(["sitio", "status"]).size().reset_index(name="count")
    per_site = per_site_long.pivot_table(index="sitio", columns="status", values="count", fill_value=0)
    print(per_site.to_string())


if __name__ == "__main__":
    main()
