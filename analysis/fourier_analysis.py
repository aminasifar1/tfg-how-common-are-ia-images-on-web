#!/usr/bin/env python3
"""Genera el espectro de Fourier (magnitud, log-escalada) de cada imagen de
una carpeta, para inspección visual de patrones de frecuencia típicos de
imágenes generadas por IA (p.ej. rejillas/periodicidades de upsampling).

Para cada imagen de --src se guarda en --out un panel con la imagen original
(escala de grises) y su espectro de magnitud |FFT2| (log-escalado, con
frecuencia cero centrada).

Uso:
    python3 fourier_analysis.py --src muestra_ia_noticias --out muestra_ia_noticias_fourier
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def fourier_magnitude(gray: np.ndarray) -> np.ndarray:
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    return np.log1p(np.abs(fshift))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", required=True, help="Carpeta de entrada con imágenes")
    parser.add_argument("--out", required=True, help="Carpeta de salida para los espectros")
    args = parser.parse_args()

    src_dir = Path(args.src)
    out_dir = Path(args.out)

    image_paths = sorted(p for p in src_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    print(f"Procesando {len(image_paths)} imágenes de {src_dir}/")

    for path in image_paths:
        rel = path.relative_to(src_dir)
        out_path = (out_dir / rel).with_suffix(".png")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        img = Image.open(path).convert("L")
        gray = np.array(img, dtype=np.float64)
        magnitude = fourier_magnitude(gray)

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(img, cmap="gray")
        axes[0].set_title("Original")
        axes[0].axis("off")

        axes[1].imshow(magnitude, cmap="viridis")
        axes[1].set_title("Espectro |FFT2| (log)")
        axes[1].axis("off")

        fig.suptitle(rel.as_posix(), fontsize=8)
        fig.tight_layout()
        fig.savefig(out_path, dpi=120)
        plt.close(fig)

        print(f"  {rel} -> {out_path.relative_to(out_dir.parent)}")

    print(f"Listo. Espectros guardados en {out_dir}/")


if __name__ == "__main__":
    main()
