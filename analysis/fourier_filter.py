#!/usr/bin/env python3
"""Aplica un filtro paso-alto en el dominio de Fourier a cada imagen de una
carpeta y guarda tanto la imagen filtrada (espacial) como un panel
comparativo (original | filtrada | espectro de la filtrada).

El filtro paso-alto elimina el "blob" de baja frecuencia que domina el
espectro y deja solo bordes/texturas finas. Al recalcular el espectro de la
imagen ya filtrada, cualquier patrón periódico (rejilla/"estrella" de picos,
típico de artefactos de upsampling de algunos generadores) queda mucho más
visible que en el espectro original, donde queda eclipsado por la energía de
baja frecuencia.

Uso:
    python3 fourier_filter.py --src muestra_ia_noticias --out muestra_ia_noticias_filtrado --cutoff 0.10
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


def highpass_filter(gray: np.ndarray, cutoff_frac: float) -> np.ndarray:
    """Devuelve la imagen filtrada en paso alto (dominio espacial), tras
    anular un disco central de radio `cutoff_frac * min(H, W) / 2` en el
    espectro de Fourier."""
    h, w = gray.shape
    f = np.fft.fftshift(np.fft.fft2(gray))

    cy, cx = h / 2, w / 2
    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    radius = cutoff_frac * min(h, w) / 2
    mask = dist > radius

    f_hp = f * mask
    img_back = np.fft.ifft2(np.fft.ifftshift(f_hp))
    return np.abs(img_back)


def to_uint8(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-12:
        return np.zeros_like(arr, dtype=np.uint8)
    return ((arr - lo) / (hi - lo) * 255.0).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", required=True, help="Carpeta de entrada con imágenes")
    parser.add_argument("--out", required=True, help="Carpeta de salida")
    parser.add_argument(
        "--cutoff", type=float, default=0.10,
        help="Radio del filtro paso-alto, como fracción de min(alto,ancho)/2 (por defecto 0.10)",
    )
    args = parser.parse_args()

    src_dir = Path(args.src)
    out_dir = Path(args.out)
    images_dir = out_dir / "imagenes_filtradas"
    panels_dir = out_dir / "paneles"

    image_paths = sorted(p for p in src_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    print(f"Procesando {len(image_paths)} imágenes de {src_dir}/ (cutoff={args.cutoff})")

    for path in image_paths:
        rel = path.relative_to(src_dir)

        img = Image.open(path).convert("L")
        gray = np.array(img, dtype=np.float64)

        filtered = highpass_filter(gray, args.cutoff)
        filtered_u8 = to_uint8(filtered)

        # Imagen filtrada como archivo independiente
        img_out_path = (images_dir / rel).with_suffix(".png")
        img_out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(filtered_u8, mode="L").save(img_out_path)

        # Espectro de la imagen ya filtrada (para ver si aparece la "estrella")
        f_filtered = np.fft.fftshift(np.fft.fft2(filtered))
        spectrum = np.log1p(np.abs(f_filtered))

        # Panel comparativo
        panel_out_path = (panels_dir / rel).with_suffix(".png")
        panel_out_path.parent.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(img, cmap="gray")
        axes[0].set_title("Original")
        axes[0].axis("off")

        axes[1].imshow(filtered_u8, cmap="gray")
        axes[1].set_title(f"Paso alto (cutoff={args.cutoff})")
        axes[1].axis("off")

        axes[2].imshow(spectrum, cmap="viridis")
        axes[2].set_title("Espectro de la filtrada")
        axes[2].axis("off")

        fig.suptitle(rel.as_posix(), fontsize=8)
        fig.tight_layout()
        fig.savefig(panel_out_path, dpi=120)
        plt.close(fig)

        print(f"  {rel} -> {img_out_path.relative_to(out_dir.parent)}")

    print(f"Listo. Imágenes filtradas en {images_dir}/, paneles en {panels_dir}/")


if __name__ == "__main__":
    main()
