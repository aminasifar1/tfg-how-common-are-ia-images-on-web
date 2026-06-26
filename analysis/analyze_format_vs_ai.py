#!/usr/bin/env python3
"""Cruza el formato de archivo de cada imagen con el resultado de clasificación IA.

Parte de predictions_long.csv (salida de un run de clasificación, que ya trae
is_ai/predicted_label_name por imagen y la ruta del archivo guardado) y deriva
el formato a partir de la extensión de image_path -no hace falta reabrir los
images_metadata.csv de cada sitio: la extensión coincide siempre con
output_format, ya verificado-. Genera tres tablas:

  1. format_by_website.csv        -> total/ai/real/ai_rate por (web, formato)
  2. predominant_format_by_website.csv -> formato dominante de cada web + su ai_rate
  3. format_ai_overall.csv        -> total/ai/real/ai_rate por formato (todas las webs)
"""
import argparse
import os

import pandas as pd


def derive_format(image_path: str) -> str:
    ext = os.path.splitext(image_path)[1].lstrip('.').lower()
    return ext or 'desconocido'


def derive_website(sector: str) -> str:
    # 'sector' trae en realidad la carpeta del sitio, p.ej. '001_euronews-com__euronews';
    # 'organization_name' viene vacío ("images") en estos CSVs de clasificación.
    name = sector.split('__', 1)[-1] if '__' in sector else sector
    return name.replace('-', ' ')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', default='classification_results/run_20260607_200images/predictions_long.csv')
    parser.add_argument('--output-dir', default=None, help='Por defecto, la misma carpeta que --predictions')
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.dirname(args.predictions)

    df = pd.read_csv(args.predictions)
    df['format'] = df['image_path'].map(derive_format)
    df['website'] = df['sector'].map(derive_website)

    # 1) Por web y formato
    by_site_format = (
        df.groupby(['website', 'format'])
        .agg(total_images=('is_ai', 'size'), ai_images=('is_ai', 'sum'))
        .reset_index()
    )
    by_site_format['real_images'] = by_site_format['total_images'] - by_site_format['ai_images']
    by_site_format['ai_rate'] = by_site_format['ai_images'] / by_site_format['total_images']
    site_totals = by_site_format.groupby('website')['total_images'].transform('sum')
    by_site_format['share_in_site'] = by_site_format['total_images'] / site_totals
    by_site_format = by_site_format.sort_values(['website', 'total_images'], ascending=[True, False])
    path_by_site_format = os.path.join(output_dir, 'format_by_website.csv')
    by_site_format.to_csv(path_by_site_format, index=False)

    # 2) Formato dominante de cada web
    dominant = (
        by_site_format.sort_values(['website', 'total_images'], ascending=[True, False])
        .groupby('website')
        .first()
        .reset_index()
        .rename(columns={
            'format': 'predominant_format',
            'total_images': 'images_in_predominant_format',
            'ai_rate': 'ai_rate_predominant_format',
            'share_in_site': 'share_predominant_format',
        })
    )
    site_overall = (
        df.groupby('website')
        .agg(site_total_images=('is_ai', 'size'), site_ai_images=('is_ai', 'sum'))
        .reset_index()
    )
    site_overall['site_ai_rate'] = site_overall['site_ai_images'] / site_overall['site_total_images']
    dominant = dominant.merge(site_overall, on='website')
    dominant = dominant[[
        'website', 'predominant_format', 'images_in_predominant_format',
        'share_predominant_format', 'ai_rate_predominant_format',
        'site_total_images', 'site_ai_rate',
    ]].sort_values('website')
    path_dominant = os.path.join(output_dir, 'predominant_format_by_website.csv')
    dominant.to_csv(path_dominant, index=False)

    # 3) Relación global formato <-> IA (todas las webs juntas)
    overall = (
        df.groupby('format')
        .agg(total_images=('is_ai', 'size'), ai_images=('is_ai', 'sum'))
        .reset_index()
    )
    overall['real_images'] = overall['total_images'] - overall['ai_images']
    overall['ai_rate'] = overall['ai_images'] / overall['total_images']
    overall = overall.sort_values('total_images', ascending=False)
    path_overall = os.path.join(output_dir, 'format_ai_overall.csv')
    overall.to_csv(path_overall, index=False)

    print(f"Imágenes clasificadas analizadas: {len(df)}")
    print(f"\n[1] Por web y formato -> {path_by_site_format}")
    print(by_site_format.to_string(index=False))
    print(f"\n[2] Formato predominante por web -> {path_dominant}")
    print(dominant.to_string(index=False))
    print(f"\n[3] Relación global formato <-> IA -> {path_overall}")
    print(overall.to_string(index=False))


if __name__ == '__main__':
    main()
