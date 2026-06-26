#!/usr/bin/env python3
"""Extrae el "ruido residual" espacial de cada imagen y su espectro de
Fourier — técnica forense clásica (similar a PRNU/SRM): se filtra la imagen
para quitarle el contenido "real" (formas, bordes grandes) y se calcula

    residual = original - suavizada (filtro de mediana)

quedando solo la textura fina / ruido de sensor / posibles artefactos de
generación. Una cámara real deja un ruido residual con un espectro
relativamente plano (ruido ~blanco); un generador de IA puede dejar
periodicidades (rejilla de upsampling) que se ven como picos en el espectro
de ESTE residual, normalmente más limpios que en el espectro de la imagen
original (donde el contenido de la escena domina).

Para cada imagen se guarda un panel: original | ruido residual | espectro del
residual (con los picos detectados, reutilizando `analyze()` de
`fourier_peak_detector.py`). Y un `picos.csv` con las mismas métricas
(num_peaks, peak_energy, max_residual...) pero calculadas sobre el espectro
del RUIDO RESIDUAL en vez de sobre la imagen original.

Uso:
    python3 noise_residual.py --src muestra_ia_noticias --out muestra_ia_noticias_ruido --median-size 3
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter

from fourier_peak_detector import analyze

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def make_panel(img: Image.Image, residual: np.ndarray, result: dict, title: str, out_path: Path) -> None:
    logmag = result["logmag"]
    valid = result["valid"]
    peak_mask = result["peak_mask"]
    ys, xs = np.nonzero(peak_mask)

    vmax = np.abs(residual).max() or 1.0

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3))

    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(residual, cmap="seismic", vmin=-vmax, vmax=vmax)
    axes[1].set_title("Ruido residual (original - mediana)")
    axes[1].axis("off")

    axes[2].imshow(logmag, cmap="viridis")
    axes[2].scatter(xs, ys, s=18, facecolors="none", edgecolors="red", linewidths=0.8)
    axes[2].set_title(f"Espectro del residual + picos (picos={result['num_peaks']})")
    axes[2].axis("off")

    fig.suptitle(title, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", required=True, help="Carpeta de entrada con imágenes")
    parser.add_argument("--out", required=True, help="Carpeta de salida")
    parser.add_argument("--median-size", type=int, default=3, help="Tamaño del filtro de mediana usado como 'denoised' (impar, por defecto 3)")
    parser.add_argument("--k", type=float, default=4.0, help="Umbral = media + k*desviación del residuo del espectro (por defecto 4.0)")
    parser.add_argument("--window", type=int, default=5, help="Tamaño de ventana para máximos locales (impar, por defecto 5)")
    parser.add_argument("--margin", type=int, default=3, help="Píxeles extra a excluir alrededor de la cruz/DC al buscar picos (por defecto 3)")
    args = parser.parse_args()

    src_dir = Path(args.src)
    out_dir = Path(args.out)
    panels_dir = out_dir / "paneles"
    panels_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(p for p in src_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    print(f"Procesando {len(image_paths)} imágenes de {src_dir}/ (median-size={args.median_size}, k={args.k})")

    rows = []
    for path in image_paths:
        rel = path.relative_to(src_dir)

        img = Image.open(path).convert("L")
        gray = np.array(img, dtype=np.float64)

        denoised = np.array(img.filter(ImageFilter.MedianFilter(size=args.median_size)), dtype=np.float64)
        residual = gray - denoised

        result = analyze(residual, k=args.k, window=args.window, margin=args.margin)

        panel_path = (panels_dir / rel).with_suffix(".png")
        panel_path.parent.mkdir(parents=True, exist_ok=True)
        make_panel(img, residual, result, rel.as_posix(), panel_path)

        rows.append({
            "sitio": rel.parts[0] if len(rel.parts) > 1 else "",
            "archivo": rel.as_posix(),
            "residual_std": round(float(residual.std()), 4),
            "num_peaks": result["num_peaks"],
            "peak_energy": round(result["peak_energy"], 4),
            "max_residual": round(result["max_residual"], 4),
            "mean_residual": round(result["mean_residual"], 4),
            "std_residual": round(result["std_residual"], 4),
            "threshold": round(result["threshold"], 4),
        })
        print(f"  {rel}: ruido_std={residual.std():.3f}, picos={result['num_peaks']}, "
              f"energia={result['peak_energy']:.2f}, max_res={result['max_residual']:.2f}")

    csv_path = out_dir / "picos.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sitio", "archivo", "residual_std", "num_peaks", "peak_energy", "max_residual", "mean_residual", "std_residual", "threshold"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Listo. Paneles en {panels_dir}/, scores en {csv_path}")


if __name__ == "__main__":
    main()
