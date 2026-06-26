#!/usr/bin/env python3
"""Aplica una regla de decisión simple, basada en las métricas calculadas por
`noise_residual.py` (ruido residual + picos en su espectro), para clasificar
cada imagen como "sospechosa de patrón de IA" o no, y compara el resultado
con el grupo real (carpeta de origen: IA = `muestra_ia_noticias`, Real =
`muestra_real_noticias`) y con el `score_ia` del clasificador SPAI.

Regla (umbral de `residual_std` ajustado al barrido de `compare_picos.py`
sobre ESTE MISMO conjunto de 50 imágenes -> sirve para ver qué imágenes caen
a un lado u otro, NO es un umbral validado en datos nuevos):

    sospechosa_IA  si  num_peaks > 0           (picos en espectro del residual)
                   o   residual_std < --umbral (ruido de sensor anormalmente bajo)

Uso:
    python3 clasificar_picos.py \
        --ia-picos muestra_ia_noticias_ruido/picos.csv --ia-index muestra_ia_noticias/index.csv \
        --real-picos muestra_real_noticias_ruido/picos.csv --real-index muestra_real_noticias/index.csv \
        --umbral 6.79 --out clasificacion_picos.csv
"""
from __future__ import annotations

import argparse
import csv


def load_index(path: str) -> dict[str, float]:
    rows = list(csv.DictReader(open(path)))
    return {r["archivo"]: float(r["score_ia"]) for r in rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ia-picos", required=True)
    parser.add_argument("--ia-index", required=True)
    parser.add_argument("--real-picos", required=True)
    parser.add_argument("--real-index", required=True)
    parser.add_argument("--umbral", type=float, default=6.79, help="Umbral de residual_std (por debajo = sospechoso)")
    parser.add_argument("--out", default="clasificacion_picos.csv")
    args = parser.parse_args()

    ia_index = load_index(args.ia_index)
    real_index = load_index(args.real_index)

    out_rows = []
    for picos_path, index_map, grupo_real in [
        (args.ia_picos, ia_index, "IA"),
        (args.real_picos, real_index, "Real"),
    ]:
        for r in csv.DictReader(open(picos_path)):
            archivo = r["archivo"]
            num_peaks = int(r["num_peaks"])
            residual_std = float(r["residual_std"])
            score_ia = index_map.get(archivo)

            sospechosa = num_peaks > 0 or residual_std < args.umbral
            veredicto = "IA" if sospechosa else "Real"
            acierto = veredicto == grupo_real

            out_rows.append({
                "archivo": archivo,
                "grupo_real": grupo_real,
                "score_ia_spai": f"{score_ia:.4f}" if score_ia is not None else "",
                "residual_std": residual_std,
                "num_peaks": num_peaks,
                "peak_energy": r["peak_energy"],
                "veredicto_fourier": veredicto,
                "acierto": "si" if acierto else "no",
            })

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["archivo", "grupo_real", "score_ia_spai", "residual_std", "num_peaks", "peak_energy", "veredicto_fourier", "acierto"])
        writer.writeheader()
        writer.writerows(out_rows)

    total = len(out_rows)
    aciertos = sum(1 for r in out_rows if r["acierto"] == "si")
    print(f"Regla: sospechosa_IA si num_peaks>0 o residual_std < {args.umbral}")
    print(f"Acierto global: {aciertos}/{total} ({100 * aciertos / total:.1f}%)\n")

    for grupo in ["IA", "Real"]:
        sub = [r for r in out_rows if r["grupo_real"] == grupo]
        a = sum(1 for r in sub if r["acierto"] == "si")
        print(f"  {grupo}: {a}/{len(sub)} correctos ({100 * a / len(sub):.1f}%)")

    flagged = [r for r in out_rows if r["veredicto_fourier"] == "IA"]
    print(f"\nImagenes marcadas como 'IA' por este filtro ({len(flagged)}/{total}):")
    for r in flagged:
        marca = "OK " if r["acierto"] == "si" else "FALLO"
        print(f"  [{marca}] ({r['grupo_real']}) {r['archivo']}  residual_std={r['residual_std']:.3f}  num_peaks={r['num_peaks']}")

    not_flagged_ia = [r for r in out_rows if r["grupo_real"] == "IA" and r["veredicto_fourier"] != "IA"]
    print(f"\nImagenes IA NO detectadas por este filtro ({len(not_flagged_ia)}/{sum(1 for r in out_rows if r['grupo_real']=='IA')}):")
    for r in not_flagged_ia:
        print(f"  {r['archivo']}  residual_std={r['residual_std']:.3f}  num_peaks={r['num_peaks']}")

    print(f"\nDetalle completo en {args.out}")


if __name__ == "__main__":
    main()
