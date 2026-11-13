"""Regenera la comparación multisemilla en formato de una columna IEEE."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "articulo" / "resultados_derivados" / "multisemilla" / "resultados_por_semilla.csv"
FIGURES = ROOT / "articulo" / "figuras"
PNG_PATH = FIGURES / "accuracy_multisemilla_e1_e4.png"
SVG_PATH = FIGURES / "accuracy_multisemilla_e1_e4.svg"


def main():
    frame = pd.read_csv(SOURCE)
    frame = frame[frame["experimento"].isin(["E1", "E4"])]
    pivot = frame.pivot(index="semilla", columns="experimento", values="accuracy_int8_percent")
    pivot = pivot.loc[sorted(pivot.index)]
    if list(pivot.columns) != ["E1", "E4"] or len(pivot) != 10:
        raise ValueError("El consolidado no contiene los diez pares E1/E4 esperados")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7.5,
        }
    )

    fig, axis = plt.subplots(figsize=(3.45, 4.15), constrained_layout=True)
    y = np.arange(len(pivot))
    e1 = pivot["E1"].to_numpy(dtype=float)
    e4 = pivot["E4"].to_numpy(dtype=float)

    for row, value_e1, value_e4 in zip(y, e1, e4):
        axis.plot([value_e1, value_e4], [row, row], color="#9EADBA", linewidth=1.2, zorder=1)
    axis.scatter(e1, y, s=28, color="#1F6D8A", marker="o", label="E1", zorder=2)
    axis.scatter(e4, y, s=28, color="#E87533", marker="s", label="E4", zorder=3)

    axis.set_yticks(y, [str(seed) for seed in pivot.index])
    axis.invert_yaxis()
    axis.set_xlim(94.5, 100.25)
    axis.xaxis.set_major_formatter(lambda value, position: f"{value:.0f}%")
    axis.set_xlabel("Exactitud INT8")
    axis.set_ylabel("Semilla")
    axis.grid(True, axis="x", color="#D9E2F3", linewidth=0.55)
    axis.set_axisbelow(True)
    axis.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG_PATH, dpi=600, bbox_inches="tight")
    fig.savefig(SVG_PATH, bbox_inches="tight")
    plt.close(fig)

    print(pivot.to_string())
    print(f"\nPNG: {PNG_PATH}")
    print(f"SVG: {SVG_PATH}")


if __name__ == "__main__":
    main()
