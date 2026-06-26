#!/usr/bin/env python3
"""Cruza la "zona de página" donde aparece cada imagen con el resultado de
clasificación IA (is_ai), para ver en qué zonas se concentran las imágenes IA.

No existe un campo "zona de página" como tal en los metadatos: hay que
derivarlo de pistas estructurales/CSS que sí se capturaron al rastrear cada
imagen (parent_tag, html_tag, classes, element_id, en images_metadata.csv de
cada sitio). Se construye una heurística por palabras clave -en la línea de
HIGH_PRIORITY_PATH_HINTS / AD_IMAGE_KEYWORDS del scraper- que clasifica cada
imagen en una zona según el primer patrón que coincide, comprobando tokens de
clase completos (no subcadenas sueltas) para evitar falsos positivos del tipo
'ad'~'shadow' que ya tuvimos que corregir antes.

Limitación honesta: muchos sitios modernos usan nombres de clase
ofuscados/hasheados (p.ej. 'DUvEbv RyH8GD qtdoVZ...'), que no llevan ninguna
pista semántica -> esas imágenes caen en 'sin_clasificar'. Eso no es un fallo
del cruce: es una limitación real de los datos disponibles.
"""
import argparse
import glob
import os

import pandas as pd

IMAGES_DIR_DEFAULT = 'images scraped/run_20260604_084542/sites'

# Orden de prioridad: la primera zona cuyo patrón aparezca como token (o
# prefijo de token con guion/guion bajo, al estilo BEM) gana. Palabras como
# 'ad' se evitan deliberadamente sueltas -> se usan formas más específicas
# (advert, sponsor, promo, dfp, adsbygoogle) que no aparecen dentro de
# palabras comunes ('shadow', 'gradient', 'load'...).
ZONE_KEYWORDS = [
    ('cabecera_navegacion', ('header', 'navbar', 'nav', 'topbar', 'masthead', 'menu')),
    ('hero_portada_banner', ('hero', 'banner', 'jumbotron', 'carousel', 'slider', 'cover', 'splash')),
    ('publicidad_patrocinado', ('advert', 'sponsor', 'promo', 'dfp', 'adsbygoogle', 'banner-ad', 'ad-slot', 'ad-unit')),
    ('galeria_miniaturas_grid', ('gallery', 'thumbnail', 'thumb', 'grid', 'masonry', 'tile', 'mosaic')),
    ('producto_catalogo', ('product', 'plp', 'catalog', 'shop', 'listing', 'item-card', 'price')),
    ('barra_lateral_relacionados', ('sidebar', 'aside', 'widget', 'related', 'recommend', 'recirculation')),
    ('articulo_contenido', ('article', 'content', 'post', 'story', 'teaser', 'body', 'main')),
    ('perfil_avatar_icono', ('avatar', 'profile', 'icon', 'logo', 'badge')),
    ('pie_pagina', ('footer', 'bottom')),
]


def classify_zone(context: str) -> str:
    tokens = context.split()
    for zone, hints in ZONE_KEYWORDS:
        for hint in hints:
            for token in tokens:
                if token == hint or token.startswith(hint + '-') or token.startswith(hint + '_') \
                        or token.endswith('-' + hint) or token.endswith('_' + hint):
                    return zone
    return 'sin_clasificar'


def load_metadata(images_dir: str) -> pd.DataFrame:
    rows = []
    pattern = os.path.join(images_dir, '*', 'metadata', 'images_metadata.csv')
    for path in glob.glob(pattern):
        site_dir = os.path.basename(os.path.dirname(os.path.dirname(path)))
        cols = ['filename', 'html_tag', 'parent_tag', 'classes', 'element_id']
        try:
            d = pd.read_csv(path, usecols=cols)
        except Exception:
            continue
        d['site_dir'] = site_dir
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', default='classification_results/run_20260607_200images/predictions_long.csv')
    parser.add_argument('--images-dir', default=IMAGES_DIR_DEFAULT)
    parser.add_argument('--output-dir', default=None)
    args = parser.parse_args()
    output_dir = args.output_dir or os.path.dirname(args.predictions)

    preds = pd.read_csv(args.predictions)
    # site_dir + filename a partir de la ruta guardada: .../sites/<site_dir>/images/<filename>
    parts = preds['image_path'].str.split('/')
    preds['filename'] = parts.str[-1]
    preds['site_dir'] = parts.str[-3]

    meta = load_metadata(args.images_dir)
    meta['context'] = (
        meta[['classes', 'parent_tag', 'html_tag', 'element_id']]
        .fillna('')
        .agg(' '.join, axis=1)
        .str.lower()
    )
    meta['zone'] = meta['context'].map(classify_zone)

    df = preds.merge(meta[['site_dir', 'filename', 'zone']], on=['site_dir', 'filename'], how='left')
    df['zone'] = df['zone'].fillna('sin_metadata')

    matched = (df['zone'] != 'sin_metadata').sum()
    print(f"Imágenes clasificadas: {len(df)} | con metadata de contexto cruzada: {matched} ({matched/len(df):.1%})")

    by_zone = (
        df.groupby('zone')
        .agg(total_images=('is_ai', 'size'), ai_images=('is_ai', 'sum'))
        .reset_index()
    )
    by_zone['real_images'] = by_zone['total_images'] - by_zone['ai_images']
    by_zone['ai_rate'] = by_zone['ai_images'] / by_zone['total_images']
    by_zone['share_of_ai_total'] = by_zone['ai_images'] / by_zone['ai_images'].sum()
    by_zone = by_zone.sort_values('ai_images', ascending=False)
    path_zone = os.path.join(output_dir, 'zone_ai_overall.csv')
    by_zone.to_csv(path_zone, index=False)

    by_site_zone = (
        df.groupby(['site_dir', 'zone'])
        .agg(total_images=('is_ai', 'size'), ai_images=('is_ai', 'sum'))
        .reset_index()
    )
    by_site_zone['ai_rate'] = by_site_zone['ai_images'] / by_site_zone['total_images']
    path_site_zone = os.path.join(output_dir, 'zone_ai_by_website.csv')
    by_site_zone.sort_values(['site_dir', 'ai_images'], ascending=[True, False]).to_csv(path_site_zone, index=False)

    print(f"\n[1] Zona de página <-> IA, global -> {path_zone}")
    print(by_zone.to_string(index=False))
    print(f"\n[2] Detalle por web y zona -> {path_site_zone}")


if __name__ == '__main__':
    main()
