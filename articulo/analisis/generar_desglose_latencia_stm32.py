"""Genera el desglose y la distribución empírica de latencia para E1 y E4."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
E1_PATH = ROOT / "resultados" / "arciniegas" / "e1" / "benchmark_end_to_end" / "benchall_1332.csv"
E4_PATH = ROOT / "resultados" / "arciniegas" / "e4" / "benchmark_end_to_end" / "benchall_1332.csv"

FIGURES = ROOT / "articulo" / "figuras"
DERIVED = ROOT / "articulo" / "resultados_derivados"
PNG_PATH = FIGURES / "desglose_latencia_presupuesto_stm32.png"
SVG_PATH = FIGURES / "desglose_latencia_presupuesto_stm32.svg"
SUMMARY_PATH = DERIVED / "resumen_latencia_presupuesto_stm32.csv"

HOP_MS = 80.0
STAGES = [
    ("Características", "us_features", "#4C78A8"),
    ("DNN INT8", "us_dnn", "#F58518"),
    ("K-means", "us_kmeans", "#54A24B"),
]


def summarize(name, path):
    frame = pd.read_csv(path)
    required = {"us_features", "us_dnn", "us_kmeans", "us_total"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Faltan columnas en {path}: {required - set(frame.columns)}")
    if len(frame) != 1332:
        raise ValueError(f"Se esperaban 1332 ventanas en {path} y se encontraron {len(frame)}")

    row = {"experiment": name, "windows": len(frame)}
    for _, column, _ in STAGES:
        row[f"{column}_mean"] = float(frame[column].mean())
    row["us_total_mean"] = float(frame["us_total"].mean())
    row["us_total_p95"] = float(frame["us_total"].quantile(0.95))
    row["us_total_max"] = float(frame["us_total"].max())
    row["occupancy_mean_pct"] = row["us_total_mean"] / (HOP_MS * 1000.0) * 100.0
    row["occupancy_p95_pct"] = row["us_total_p95"] / (HOP_MS * 1000.0) * 100.0
    row["occupancy_max_pct"] = row["us_total_max"] / (HOP_MS * 1000.0) * 100.0
    row["mean_margin_ms"] = HOP_MS - row["us_total_mean"] / 1000.0
    row["max_case_margin_ms"] = HOP_MS - row["us_total_max"] / 1000.0
    return row


def create_figure(summary):
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

    fig, (ax_stages, ax_distribution) = plt.subplots(
        2,
        1,
        figsize=(3.45, 5.0),
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1.15, 1.0]},
    )

    y_positions = np.arange(len(summary))
    left = np.zeros(len(summary))
    for stage_name, column, color in STAGES:
        values_ms = summary[f"{column}_mean"].to_numpy(dtype=float) / 1000.0
        ax_stages.barh(
            y_positions,
            values_ms,
            left=left,
            height=0.52,
            color=color,
            label=stage_name,
        )
        for y, x0, value in zip(y_positions, left, values_ms):
            ax_stages.text(
                x0 + value / 2.0,
                y,
                f"{value:.3f}",
                ha="center",
                va="center",
                fontsize=6.8,
                color="white" if stage_name != "DNN INT8" else "#222222",
                fontweight="bold",
            )
        left += values_ms

    for y, total in zip(y_positions, summary["us_total_mean"] / 1000.0):
        ax_stages.text(total + 0.07, y, f"{total:.3f} ms", va="center", fontsize=7)
    ax_stages.set_yticks(y_positions, summary["experiment"])
    ax_stages.invert_yaxis()
    ax_stages.set_xlim(0, 7.5)
    ax_stages.set_xlabel("Latencia media (ms)")
    ax_stages.set_title("(a) Desglose del pipeline")
    ax_stages.legend(loc="lower center", bbox_to_anchor=(0.5, -0.40), ncol=3, frameon=False)

    colors = {"E1": "#4C78A8", "E4": "#E45756"}
    latency_frames = {
        "E1": pd.read_csv(E1_PATH)["us_total"].to_numpy(dtype=float) / 1000.0,
        "E4": pd.read_csv(E4_PATH)["us_total"].to_numpy(dtype=float) / 1000.0,
    }
    all_values = np.concatenate(list(latency_frames.values()))
    for experiment, values in latency_frames.items():
        ordered = np.sort(values)
        cumulative = np.arange(1, ordered.size + 1) / ordered.size
        ax_distribution.step(
            ordered,
            cumulative,
            where="post",
            linewidth=1.15,
            color=colors[experiment],
            label=experiment,
        )
        p95 = float(np.quantile(values, 0.95))
        ax_distribution.plot(
            p95,
            0.95,
            marker="o",
            markersize=3.6,
            color=colors[experiment],
        )

    span = float(all_values.max() - all_values.min())
    pad = max(span * 0.07, 0.001)
    ax_distribution.set_xlim(all_values.min() - pad, all_values.max() + pad)
    ax_distribution.set_ylim(0, 1.01)
    ax_distribution.set_xlabel("Latencia total por ventana (ms)")
    ax_distribution.set_ylabel("Proporción acumulada")
    ax_distribution.set_title("(b) Distribución empírica en 1332 ventanas")
    ax_distribution.axhline(0.95, color="#777777", linewidth=0.6, linestyle="--")
    ax_distribution.text(
        0.02,
        0.89,
        "Los círculos indican P95",
        transform=ax_distribution.transAxes,
        fontsize=6.8,
        color="#555555",
    )
    ax_distribution.legend(loc="lower right", frameon=False)

    for axis in (ax_stages, ax_distribution):
        axis.grid(True, axis="x", color="#E0E0E0", linewidth=0.45)
        axis.set_axisbelow(True)

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG_PATH, dpi=600, bbox_inches="tight")
    fig.savefig(SVG_PATH, bbox_inches="tight")
    plt.close(fig)


def main():
    summary = pd.DataFrame(
        [
            summarize("E1", E1_PATH),
            summarize("E4", E4_PATH),
        ]
    )
    DERIVED.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_PATH, index=False)
    create_figure(summary)
    print(summary.to_string(index=False))
    print(f"\nPNG: {PNG_PATH}")
    print(f"SVG: {SVG_PATH}")
    print(f"CSV: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
