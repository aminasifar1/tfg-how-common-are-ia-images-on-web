#!/usr/bin/env python3
"""Detector de picos periódicos en el espectro de Fourier, basado en la idea
mostrada en https://youtu.be/q5_PrTvNypY: una imagen "natural" tiene un
espectro de magnitud que cae suavemente con la frecuencia (perfil radial 1/f)
más una cruz central (artefacto de ventaneo por los bordes de la imagen). Una
imagen generada/retocada por IA frecuentemente añade, ENCIMA de ese fondo
suave, una constelación de picos periódicos fuera de la cruz central,
producidos por las capas de upsampling del generador.

Para cada imagen:
  1. Espectro de magnitud log-escalado |FFT2| (centrado).
  2. Perfil radial promedio del espectro -> "fondo" esperado para una imagen
     sin estructura periódica.
  3. Residuo = espectro - fondo radial.
  4. Se enmascara el disco central (DC) y una banda horizontal/vertical
     (la cruz de ventaneo), que NO cuentan como "picos".
  5. Se buscan máximos locales del residuo, fuera de esa máscara, que superen
     `media + k * desviación` -> "picos periódicos".
  6. Score por imagen: nº de picos y energía total del residuo en esos picos.

Para cada imagen se guarda un panel: original | espectro log con los picos
marcados | residuo (fondo radial restado, cruz/DC enmascarados) con los picos
marcados. Además se escribe `picos.csv` con el score de cada imagen.

Uso:
    python3 fourier_peak_detector.py --src muestra_ia_noticias --out muestra_ia_noticias_picos --k 4.0
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def radial_profile(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Perfil radial promedio de `data` respecto a su centro, y la matriz de
    distancias enteras (en píxeles) de cada punto al centro."""
    h, w = data.shape
    cy, cx = h / 2.0, w / 2.0
    yy, xx = np.indices((h, w))
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(int)
    r_max = r.max()
    tbin = np.bincount(r.ravel(), data.ravel(), minlength=r_max + 1)
    nr = np.bincount(r.ravel(), minlength=r_max + 1).astype(np.float64)
    nr[nr == 0] = 1
    return tbin / nr, r


def local_maxima_mask(arr: np.ndarray, size: int) -> np.ndarray:
    """True donde `arr` es el máximo dentro de una ventana `size x size`
    centrada en ese punto."""
    pad = size // 2
    padded = np.pad(arr, pad, mode="edge")
    windows = sliding_window_view(padded, (size, size))
    local_max = windows.max(axis=(-2, -1))
    return arr >= local_max


def erode_mask(mask: np.ndarray, size: int) -> np.ndarray:
    """True solo donde `mask` es True en toda una ventana `size x size`
    centrada en ese punto (erosión morfológica). Se usa para no contar como
    "pico" nada pegado al borde de la cruz/DC enmascarados, donde el residuo
    tiene una discontinuidad fuerte que no es un pico periódico real."""
    pad = size // 2
    padded = np.pad(mask, pad, mode="constant", constant_values=False)
    windows = sliding_window_view(padded, (size, size))
    return windows.all(axis=(-2, -1))


def analyze(gray: np.ndarray, k: float, window: int, margin: int) -> dict:
    h, w = gray.shape
    F = np.fft.fftshift(np.fft.fft2(gray))
    logmag = np.log1p(np.abs(F))

    profile, r = radial_profile(logmag)
    background = profile[r]
    residual = logmag - background

    cy, cx = h / 2.0, w / 2.0
    yy, xx = np.indices((h, w))
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    dc_radius = max(4, int(0.02 * min(h, w)))
    cross_width = max(2, int(0.004 * min(h, w)))
    mask_invalid = (dist <= dc_radius) | (np.abs(yy - cy) <= cross_width) | (np.abs(xx - cx) <= cross_width)
    valid = ~mask_invalid

    mu = residual[valid].mean()
    sigma = residual[valid].std()
    threshold = mu + k * sigma

    # Excluir además un margen alrededor de la cruz/DC: ahí el residuo tiene
    # una discontinuidad fuerte (no un pico periódico) por el cambio brusco
    # de fondo radial entre los radios "dentro" y "fuera" de la cruz.
    valid_search = erode_mask(valid, size=2 * margin + 1)

    is_local_max = local_maxima_mask(residual, size=window)
    peak_mask = is_local_max & valid_search & (residual > threshold)

    num_peaks = int(peak_mask.sum())
    peak_energy = float(residual[peak_mask].sum()) if num_peaks else 0.0
    max_residual = float(residual[valid].max())

    return {
        "logmag": logmag,
        "residual": residual,
        "valid": valid,
        "peak_mask": peak_mask,
        "num_peaks": num_peaks,
        "peak_energy": peak_energy,
        "max_residual": max_residual,
        "mean_residual": float(mu),
        "std_residual": float(sigma),
        "threshold": float(threshold),
    }


def make_panel(img: Image.Image, result: dict, title: str, out_path: Path) -> None:
    logmag = result["logmag"]
    residual = result["residual"]
    valid = result["valid"]
    peak_mask = result["peak_mask"]
    ys, xs = np.nonzero(peak_mask)

    residual_display = np.where(valid, residual, np.nan)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3))

    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(logmag, cmap="viridis")
    axes[1].scatter(xs, ys, s=18, facecolors="none", edgecolors="red", linewidths=0.8)
    axes[1].set_title("Espectro |FFT2| (log) + picos")
    axes[1].axis("off")

    cmap_res = matplotlib.cm.get_cmap("magma").copy()
    cmap_res.set_bad(color="black")
    axes[2].imshow(residual_display, cmap=cmap_res)
    axes[2].scatter(xs, ys, s=18, facecolors="none", edgecolors="cyan", linewidths=0.8)
    axes[2].set_title(f"Residuo radial (picos={result['num_peaks']})")
    axes[2].axis("off")

    fig.suptitle(title, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", required=True, help="Carpeta de entrada con imágenes")
    parser.add_argument("--out", required=True, help="Carpeta de salida")
    parser.add_argument("--k", type=float, default=4.0, help="Umbral = media + k*desviación del residuo (por defecto 4.0)")
    parser.add_argument("--window", type=int, default=5, help="Tamaño de ventana para máximos locales (impar, por defecto 5)")
    parser.add_argument("--margin", type=int, default=3, help="Píxeles extra a excluir alrededor de la cruz/DC al buscar picos (por defecto 3)")
    args = parser.parse_args()

    src_dir = Path(args.src)
    out_dir = Path(args.out)
    panels_dir = out_dir / "paneles"
    panels_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(p for p in src_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    print(f"Procesando {len(image_paths)} imágenes de {src_dir}/ (k={args.k}, window={args.window})")

    rows = []
    for path in image_paths:
        rel = path.relative_to(src_dir)

        img = Image.open(path).convert("L")
        gray = np.array(img, dtype=np.float64)

        result = analyze(gray, k=args.k, window=args.window, margin=args.margin)

        panel_path = (panels_dir / rel).with_suffix(".png")
        panel_path.parent.mkdir(parents=True, exist_ok=True)
        make_panel(img, result, rel.as_posix(), panel_path)

        rows.append({
            "sitio": rel.parts[0] if len(rel.parts) > 1 else "",
            "archivo": rel.as_posix(),
            "num_peaks": result["num_peaks"],
            "peak_energy": round(result["peak_energy"], 4),
            "max_residual": round(result["max_residual"], 4),
            "mean_residual": round(result["mean_residual"], 4),
            "std_residual": round(result["std_residual"], 4),
            "threshold": round(result["threshold"], 4),
        })
        print(f"  {rel}: picos={result['num_peaks']}, energia={result['peak_energy']:.2f}, "
              f"max_res={result['max_residual']:.2f}, umbral={result['threshold']:.2f}")

    csv_path = out_dir / "picos.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sitio", "archivo", "num_peaks", "peak_energy", "max_residual", "mean_residual", "std_residual", "threshold"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Listo. Paneles en {panels_dir}/, scores en {csv_path}")


if __name__ == "__main__":
    main()
