#!/usr/bin/env python3
"""Combina los `picos.csv` de dos grupos (p.ej. imágenes IA vs "real") y
compara sus distribuciones de `num_peaks`, `peak_energy` y `max_residual`,
para ver si alguna de estas métricas del detector de picos de Fourier separa
realmente IA de real, y con qué umbral.

Uso:
    python3 compare_picos.py \
        --ia muestra_ia_noticias_picos/picos.csv --label-ia "IA" \
        --real muestra_real_noticias_picos/picos.csv --label-real "Real" \
        --out comparacion_picos.png
"""
from __future__ import annotations

import argparse
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load(path: str) -> list[dict]:
    return list(csv.DictReader(open(path)))


def stats(rows: list[dict], col: str) -> dict:
    vals = np.array([float(r[col]) for r in rows])
    return {
        "n": len(vals),
        "media": vals.mean(),
        "mediana": np.median(vals),
        "min": vals.min(),
        "max": vals.max(),
        "vals": vals,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ia", required=True, help="picos.csv del grupo IA")
    parser.add_argument("--real", required=True, help="picos.csv del grupo real")
    parser.add_argument("--label-ia", default="IA", help="Etiqueta para el grupo IA")
    parser.add_argument("--label-real", default="Real", help="Etiqueta para el grupo real")
    parser.add_argument("--out", default="comparacion_picos.png", help="Figura de salida")
    args = parser.parse_args()

    ia_rows = load(args.ia)
    real_rows = load(args.real)

    all_metrics = ["residual_std", "num_peaks", "peak_energy", "max_residual"]
    metrics = [m for m in all_metrics if m in ia_rows[0] and m in real_rows[0]]

    print(f"{'metrica':<14} {'grupo':<8} {'n':>3} {'media':>8} {'mediana':>8} {'min':>8} {'max':>8}")
    for m in metrics:
        for label, rows in [(args.label_ia, ia_rows), (args.label_real, real_rows)]:
            s = stats(rows, m)
            print(f"{m:<14} {label:<8} {s['n']:>3} {s['media']:>8.3f} {s['mediana']:>8.3f} {s['min']:>8.3f} {s['max']:>8.3f}")
        print()

    # Barrido de umbrales para la métrica más informativa: si existe
    # `residual_std`, valores BAJOS son sospechosos (imagen "demasiado
    # limpia"/sin ruido de sensor) -> % por DEBAJO del umbral. Si no, usamos
    # `max_residual` (picos del espectro), donde valores ALTOS son
    # sospechosos -> % por ENCIMA del umbral.
    if "residual_std" in metrics:
        thr_metric, direction = "residual_std", "below"
    else:
        thr_metric, direction = "max_residual", "above"

    ia_vals = np.array([float(r[thr_metric]) for r in ia_rows])
    real_vals = np.array([float(r[thr_metric]) for r in real_rows])
    all_thr = sorted(set(round(v, 2) for v in np.concatenate([ia_vals, real_vals])))

    cmp_op = "<" if direction == "below" else ">"
    print(f"Umbral {thr_metric} -> % IA {cmp_op} umbral vs % Real {cmp_op} umbral")
    for t in all_thr:
        if direction == "below":
            pct_ia = 100 * (ia_vals < t).mean()
            pct_real = 100 * (real_vals < t).mean()
        else:
            pct_ia = 100 * (ia_vals > t).mean()
            pct_real = 100 * (real_vals > t).mean()
        print(f"  {cmp_op} {t:6.2f}  ->  IA: {pct_ia:5.1f}%   Real: {pct_real:5.1f}%   (diferencia: {pct_ia - pct_real:+.1f} pts)")

    # Gráfico: dispersión + caja para cada métrica
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4.5))
    for ax, m in zip(axes, metrics):
        data = [np.array([float(r[m]) for r in real_rows]), np.array([float(r[m]) for r in ia_rows])]
        labels = [args.label_real, args.label_ia]
        ax.boxplot(data, tick_labels=labels, showmeans=True)
        for i, d in enumerate(data, start=1):
            jitter = (np.random.rand(len(d)) - 0.5) * 0.2
            ax.scatter(np.full(len(d), i) + jitter, d, alpha=0.6, s=18, color="tab:red")
        ax.set_title(m)

    fig.suptitle("Comparación de métricas del detector de picos (Fourier)")
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    plt.close(fig)
    print(f"\nFigura guardada en {args.out}")


if __name__ == "__main__":
    main()
