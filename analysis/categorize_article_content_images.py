#!/usr/bin/env python3
"""Tipifica TODAS las imágenes "article content" con predicción
(`article_content_images/article_content_images_with_predictions.csv`,
generado por `cross_reference_article_content_predictions.py`) según el tipo
de publicación de la página de origen (`source_url`/`title`): noticia,
reportaje/feature, magazine, contenido institucional, comercial/promocional,
servicio/consejos, turismo descriptivo...

Importante: se tipifican y agregan TODAS las imágenes (reales + IA), no solo
las IA, para poder calcular `ai_share` con el denominador correcto por tipo
de publicación y por sitio.

Genera, en --out (por defecto la misma carpeta de entrada):
  - article_content_images_tipificado.csv : todas las imágenes con predicción
    + columna `tipo_publicacion`.
  - article_content_tipo_publicacion_summary.csv : total/ai_count/real_count/
    ai_share por tipo_publicacion.
  - article_content_news_images.csv : solo filas con tipo_publicacion ==
    "noticia" (reales + IA), con su ai_share por sitio.
  - <out>/noticias_ia/<sitio>/<archivo> : symlinks a las imágenes de noticias
    detectadas como IA, para revisión visual (subconjunto, no usar para tasas).

Uso:
    python3 categorize_article_content_images.py
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd


def categorize(row: pd.Series) -> str:
    url = row["source_url"]
    title = (row["title"] or "").strip().lower()

    if "bbc.co.uk/future" in url or "bbc.co.uk/earth/story" in url:
        return "reportaje_feature"

    if "magazine.artstation.com" in url:
        return "reportaje_magazine_arte"

    if "mibebeyyo.elmundo.es" in url:
        return "servicio_consejos"

    if "spain.info" in url:
        return "turismo_descriptivo"

    if "loyola.edu" in url:
        if "/blog/" in url:
            return "blog_institucional"
        if "/news/" in url:
            return "noticia"
        return "otro_institucional"

    if "uab.cat" in url and ("detall-noticia" in url or "detall-de-noticia" in url or "news-detail" in url):
        return "noticia"

    if "lavanguardia.com" in url and ("club.lavanguardia.com" in url or "suscripciones.lavanguardia.com" in url):
        return "comercial_promocional"

    if url.rstrip("/") == "https://www.lavanguardia.com":
        if "oferta" in title or "por tiempo limitado" in title:
            return "comercial_promocional"
        return "noticia"

    if "euronews.com" in url:
        return "noticia"

    if "lonelyplanet.com" in url or "italia.it" in url:
        return "turismo_descriptivo"

    return "sin_clasificar"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--input",
        default="article_content_images/article_content_images_with_predictions.csv",
        help="CSV generado por cross_reference_article_content_predictions.py",
    )
    parser.add_argument("--out", default=None, help="Carpeta de salida (por defecto, la del --input)")
    parser.add_argument(
        "--no-links",
        action="store_true",
        help="No crear symlinks a las imágenes IA de noticias, solo generar los CSVs",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out) if args.out else input_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path, dtype=str, keep_default_na=False)
    df["is_ai"] = df["is_ai"].astype(int)
    df["tipo_publicacion"] = df.apply(categorize, axis=1)

    tipificado_csv = out_dir / "article_content_images_tipificado.csv"
    df.to_csv(tipificado_csv, index=False)
    print(f"Escrito {tipificado_csv} ({len(df)} filas)")

    by_type = df.groupby("tipo_publicacion").agg(
        total_images=("is_ai", "size"),
        ai_count=("is_ai", "sum"),
        n_articulos=("source_url", "nunique"),
        sitios=("sitio", lambda s: ", ".join(sorted(s.unique()))),
    )
    by_type["real_count"] = by_type["total_images"] - by_type["ai_count"]
    by_type["ai_share"] = by_type["ai_count"] / by_type["total_images"]
    by_type = by_type.sort_values("total_images", ascending=False)
    summary_csv = out_dir / "article_content_tipo_publicacion_summary.csv"
    by_type.to_csv(summary_csv)
    print(f"Escrito {summary_csv}")
    print("\nTasa de IA por tipo de publicación (todas las imágenes 'article content'):")
    print(by_type.to_string())

    news = df[df["tipo_publicacion"] == "noticia"].copy()
    news_csv = out_dir / "article_content_news_images.csv"
    news.to_csv(news_csv, index=False)
    print(f"\nEscrito {news_csv} ({len(news)} filas, {int(news['is_ai'].sum())} IA / {len(news)} total)")

    by_site = news.groupby("sitio").agg(
        total_images=("is_ai", "size"),
        ai_count=("is_ai", "sum"),
        n_articulos=("source_url", "nunique"),
    )
    by_site["real_count"] = by_site["total_images"] - by_site["ai_count"]
    by_site["ai_share"] = by_site["ai_count"] / by_site["total_images"]
    print("\nTasa de IA dentro de 'noticia', por sitio:")
    print(by_site.to_string())

    if not args.no_links:
        ai_news = news[news["is_ai"] == 1]
        for _, r in ai_news.iterrows():
            src = Path(r["image_abs_path"])
            if not src.exists():
                continue
            site_out = out_dir / "noticias_ia" / r["sitio"]
            site_out.mkdir(parents=True, exist_ok=True)
            dst = site_out / src.name
            if dst.exists() or dst.is_symlink():
                continue
            try:
                dst.symlink_to(src)
            except OSError:
                shutil.copy2(src, dst)


if __name__ == "__main__":
    main()
