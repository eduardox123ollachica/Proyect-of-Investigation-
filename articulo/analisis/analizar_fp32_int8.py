"""Resume y grafica la comparación FP32--INT8 sin repetir entrenamientos."""

from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean, median
from xml.sax.saxutils import escape

import matplotlib.pyplot as plt


ARTICLE_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = ARTICLE_DIR.parent
FIGURES_DIR = ARTICLE_DIR / "figuras"
DERIVED_DIR = ARTICLE_DIR / "resultados_derivados"
COLORS = {
    "FP32": "#3B78A8",
    "INT8": "#E07A3F",
    "E1": "#2A6F8F",
    "E2": "#6AA84F",
    "E3": "#8E63A9",
    "E4": "#D95F4F",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def svg_text(x: float, y: float, value: str, size: int = 15, **attrs: str) -> str:
    extra = " ".join(
        f'{key.replace("_", "-")}="{escape(str(val))}"' for key, val in attrs.items()
    )
    return f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" {extra}>{escape(value)}</text>'


def canonical_data() -> list[dict[str, float | str]]:
    output = []
    for experiment in ("E1", "E2", "E3", "E4"):
        path = (
            PROJECT_DIR
            / "resultados"
            / "arciniegas"
            / experiment.lower()
            / "entrenamiento"
            / "metricas_tinyml_fp32_int8.csv"
        )
        rows = read_rows(path)
        fp32 = next(row for row in rows if row["modelo"] == "TFLite_FP32")
        int8 = next(row for row in rows if row["modelo"] == "TFLite_INT8")
        output.append(
            {
                "experimento": experiment,
                "accuracy_fp32_percent": float(fp32["accuracy_percent"]),
                "accuracy_int8_percent": float(int8["accuracy_percent"]),
                "delta_int8_fp32_pp": float(int8["accuracy_percent"])
                - float(fp32["accuracy_percent"]),
                "archivo_fp32_bytes": float(fp32["archivo_bytes"]),
                "archivo_int8_bytes": float(int8["archivo_bytes"]),
                "tensores_fp32_bytes": float(fp32["memoria_tensores_estimada_bytes"]),
                "tensores_int8_bytes": float(int8["memoria_tensores_estimada_bytes"]),
            }
        )
    return output


def multiseed_data() -> list[dict[str, float | int | str]]:
    rows = read_rows(
        ARTICLE_DIR
        / "resultados_derivados"
        / "multisemilla"
        / "resultados_por_semilla.csv"
    )
    return [
        {
            "experimento": row["experimento"],
            "semilla": int(row["semilla"]),
            "delta": float(row["delta_int8_vs_keras_pp"]),
        }
        for row in rows
    ]


def canonical_panel(x: int, y: int, width: int, height: int, rows: list[dict]) -> list[str]:
    output = [svg_text(x + width / 2, y, "(a) Exactitud canónica", 21, text_anchor="middle", font_weight="bold")]
    left, top = x + 75, y + 45
    plot_w, plot_h = width - 105, height - 110
    y_min, y_max = 93.5, 100.3

    def sy(value: float) -> float:
        return top + plot_h - (value - y_min) / (y_max - y_min) * plot_h

    output.append(f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="white" stroke="#444"/>')
    for tick in (94, 96, 98, 100):
        yy = sy(tick)
        output.append(f'<line x1="{left}" y1="{yy}" x2="{left + plot_w}" y2="{yy}" stroke="#dddddd"/>')
        output.append(svg_text(left - 10, yy + 5, str(tick), 12, text_anchor="end"))
    centers = [left + plot_w * fraction for fraction in (0.14, 0.38, 0.62, 0.86)]
    bar_w = 42
    for center, row in zip(centers, rows):
        for offset, key, color in (
            (-bar_w, "accuracy_fp32_percent", COLORS["FP32"]),
            (0, "accuracy_int8_percent", COLORS["INT8"]),
        ):
            value = float(row[key])
            bar_y = sy(value)
            output.append(f'<rect x="{center + offset}" y="{bar_y}" width="{bar_w}" height="{top + plot_h - bar_y}" fill="{color}"/>')
            output.append(svg_text(center + offset + bar_w / 2, bar_y - 6, f"{value:.2f}", 10, text_anchor="middle"))
        output.append(svg_text(center - bar_w / 2, top + plot_h + 27, str(row["experimento"]), 14, text_anchor="middle", font_weight="bold"))
    output.append(f'<rect x="{left + 15}" y="{top + 12}" width="18" height="12" fill="{COLORS["FP32"]}"/>')
    output.append(svg_text(left + 40, top + 23, "TFLite FP32", 12))
    output.append(f'<rect x="{left + 145}" y="{top + 12}" width="18" height="12" fill="{COLORS["INT8"]}"/>')
    output.append(svg_text(left + 170, top + 23, "TFLite INT8", 12))
    output.append(f'<text x="{left - 48}" y="{top + plot_h / 2}" font-size="14" text-anchor="middle" transform="rotate(-90 {left - 48} {top + plot_h / 2})">Exactitud (%)</text>')
    return output


def multiseed_panel(x: int, y: int, width: int, height: int, rows: list[dict]) -> list[str]:
    output = [svg_text(x + width / 2, y, "(b) Cambio de exactitud en diez semillas", 20, text_anchor="middle", font_weight="bold")]
    left, top = x + 75, y + 45
    plot_w, plot_h = width - 105, height - 115
    seeds = sorted({int(row["semilla"]) for row in rows})
    y_min, y_max = -0.35, 0.425

    def sx(seed: int) -> float:
        return left + seeds.index(seed) / (len(seeds) - 1) * plot_w

    def sy(value: float) -> float:
        return top + plot_h - (value - y_min) / (y_max - y_min) * plot_h

    output.append(f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="white" stroke="#444"/>')
    for tick in (-0.3, -0.15, 0, 0.15, 0.3):
        yy = sy(tick)
        output.append(f'<line x1="{left}" y1="{yy}" x2="{left + plot_w}" y2="{yy}" stroke="{("#555" if tick == 0 else "#dddddd")}" stroke-width="{(2 if tick == 0 else 1)}"/>')
        output.append(svg_text(left - 10, yy + 5, f"{tick:+.2f}", 11, text_anchor="end"))
    for seed in seeds:
        output.append(svg_text(sx(seed), top + plot_h + 25, str(seed), 11, text_anchor="middle"))
    for experiment, marker_offset in (("E1", -7.5), ("E2", -2.5), ("E3", 2.5), ("E4", 7.5)):
        selected = sorted((row for row in rows if row["experimento"] == experiment), key=lambda row: row["semilla"])
        points = " ".join(f'{sx(int(row["semilla"])) + marker_offset:.1f},{sy(float(row["delta"])):.1f}' for row in selected)
        output.append(f'<polyline points="{points}" fill="none" stroke="{COLORS[experiment]}" stroke-width="2" opacity="0.70"/>')
        for row in selected:
            output.append(f'<circle cx="{sx(int(row["semilla"])) + marker_offset}" cy="{sy(float(row["delta"]))}" r="5" fill="{COLORS[experiment]}"/>')
    output.append(svg_text(left + plot_w / 2, top + plot_h + 50, "Semilla", 14, text_anchor="middle"))
    output.append(f'<text x="{left - 52}" y="{top + plot_h / 2}" font-size="14" text-anchor="middle" transform="rotate(-90 {left - 52} {top + plot_h / 2})">INT8 − FP32 (pp)</text>')
    for index, experiment in enumerate(("E1", "E2", "E3", "E4")):
        legend_x = left + 20 + 55 * index
        output.append(f'<circle cx="{legend_x}" cy="{top + 20}" r="5" fill="{COLORS[experiment]}"/>')
        output.append(svg_text(legend_x + 10, top + 25, experiment, 12))
    return output


def footprint_panel(x: int, y: int, width: int, height: int, canonical: list[dict]) -> list[str]:
    output = [svg_text(x + width / 2, y, "(c) Huella del modelo", 20, text_anchor="middle", font_weight="bold")]
    row = canonical[0]
    categories = [
        ("Archivo TFLite", float(row["archivo_fp32_bytes"]) / 1024, float(row["archivo_int8_bytes"]) / 1024),
        ("Tensores estimados", float(row["tensores_fp32_bytes"]) / 1024, float(row["tensores_int8_bytes"]) / 1024),
    ]
    left, top = x + 90, y + 55
    plot_w, plot_h = width - 125, height - 130
    max_value = 5.6

    def sy(value: float) -> float:
        return top + plot_h - value / max_value * plot_h

    output.append(f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="white" stroke="#444"/>')
    for tick in range(0, 6):
        yy = sy(float(tick))
        output.append(f'<line x1="{left}" y1="{yy}" x2="{left + plot_w}" y2="{yy}" stroke="#dddddd"/>')
        output.append(svg_text(left - 10, yy + 5, str(tick), 11, text_anchor="end"))
    centers = [left + plot_w * 0.30, left + plot_w * 0.72]
    bar_w = 55
    for center, (label, fp32, int8) in zip(centers, categories):
        for offset, value, color in ((-bar_w, fp32, COLORS["FP32"]), (0, int8, COLORS["INT8"])):
            bar_y = sy(value)
            output.append(f'<rect x="{center + offset}" y="{bar_y}" width="{bar_w}" height="{top + plot_h - bar_y}" fill="{color}"/>')
            output.append(svg_text(center + offset + bar_w / 2, bar_y - 7, f"{value:.2f}", 11, text_anchor="middle"))
        reduction = 100 * (1 - int8 / fp32)
        output.append(svg_text(center - bar_w / 2, top + plot_h + 26, label, 12, text_anchor="middle"))
        output.append(svg_text(center - bar_w / 2, top + plot_h + 46, f"reducción {reduction:.1f} %", 11, text_anchor="middle", fill="#555555"))
    output.append(f'<text x="{left - 55}" y="{top + plot_h / 2}" font-size="14" text-anchor="middle" transform="rotate(-90 {left - 55} {top + plot_h / 2})">KiB</text>')
    output.append(f'<rect x="{left + 15}" y="{top - 18}" width="18" height="12" fill="{COLORS["FP32"]}"/>')
    output.append(svg_text(left + 40, top - 7, "FP32", 12))
    output.append(f'<rect x="{left + 95}" y="{top - 18}" width="18" height="12" fill="{COLORS["INT8"]}"/>')
    output.append(svg_text(left + 120, top - 7, "INT8", 12))
    return output


def write_figure(canonical: list[dict], multiseed: list[dict], path: Path) -> None:
    width, height = 1500, 1120
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<g font-family="Arial, Helvetica, sans-serif" fill="#172B3A">',
        svg_text(width / 2, 38, "Comparación entre TFLite FP32 e INT8", 27, text_anchor="middle", font_weight="bold"),
    ]
    parts.extend(canonical_panel(40, 82, 1420, 440, canonical))
    parts.extend(multiseed_panel(35, 585, 760, 500, multiseed))
    parts.extend(footprint_panel(820, 585, 640, 500, canonical))
    parts.extend(["</g>", "</svg>"])
    path.write_text("\n".join(parts), encoding="utf-8")


def write_png(canonical: list[dict], multiseed: list[dict], path: Path) -> None:
    """Genera la versión raster usada por LaTeX a partir de los mismos datos."""
    fig = plt.figure(figsize=(14, 9), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=(1.0, 1.05))
    ax_a = fig.add_subplot(grid[0, :])
    ax_b = fig.add_subplot(grid[1, 0])
    ax_c = fig.add_subplot(grid[1, 1])

    experiments = [str(row["experimento"]) for row in canonical]
    positions = list(range(len(experiments)))
    width = 0.36
    fp32 = [float(row["accuracy_fp32_percent"]) for row in canonical]
    int8 = [float(row["accuracy_int8_percent"]) for row in canonical]
    bars_fp32 = ax_a.bar([x - width / 2 for x in positions], fp32, width, label="TFLite FP32", color=COLORS["FP32"])
    bars_int8 = ax_a.bar([x + width / 2 for x in positions], int8, width, label="TFLite INT8", color=COLORS["INT8"])
    ax_a.set_ylim(93.0, 100.5)
    ax_a.set_ylabel("Exactitud (%)")
    ax_a.set_xticks(positions, experiments)
    ax_a.set_title("(a) Exactitud canónica", fontweight="bold")
    ax_a.grid(axis="y", alpha=0.25)
    ax_a.legend(loc="lower right")
    ax_a.bar_label(bars_fp32, fmt="%.2f", padding=2, fontsize=9)
    ax_a.bar_label(bars_int8, fmt="%.2f", padding=2, fontsize=9)

    seeds = sorted({int(row["semilla"]) for row in multiseed})
    offsets = {"E1": -0.24, "E2": -0.08, "E3": 0.08, "E4": 0.24}
    for experiment in ("E1", "E2", "E3", "E4"):
        selected = sorted(
            (row for row in multiseed if row["experimento"] == experiment),
            key=lambda row: int(row["semilla"]),
        )
        ax_b.plot(
            [index + offsets[experiment] for index in range(len(seeds))],
            [float(row["delta"]) for row in selected],
            marker="o",
            markersize=4,
            linewidth=1.2,
            label=experiment,
            color=COLORS[experiment],
        )
    ax_b.axhline(0.0, color="#444444", linewidth=1)
    ax_b.set_xticks(range(len(seeds)), [str(seed) for seed in seeds], rotation=45)
    ax_b.set_ylim(-0.35, 0.425)
    ax_b.set_xlabel("Semilla")
    ax_b.set_ylabel("INT8 − FP32 (pp)")
    ax_b.set_title("(b) Cambio en diez semillas", fontweight="bold")
    ax_b.grid(axis="y", alpha=0.25)
    ax_b.legend(ncol=4, fontsize=8, loc="upper left")

    row = canonical[0]
    categories = ["Archivo TFLite", "Tensores estimados"]
    fp32_kib = [float(row["archivo_fp32_bytes"]) / 1024, float(row["tensores_fp32_bytes"]) / 1024]
    int8_kib = [float(row["archivo_int8_bytes"]) / 1024, float(row["tensores_int8_bytes"]) / 1024]
    category_positions = [0, 1]
    bars_fp32 = ax_c.bar([x - width / 2 for x in category_positions], fp32_kib, width, label="FP32", color=COLORS["FP32"])
    bars_int8 = ax_c.bar([x + width / 2 for x in category_positions], int8_kib, width, label="INT8", color=COLORS["INT8"])
    ax_c.set_xticks(category_positions, categories)
    ax_c.set_ylabel("KiB")
    ax_c.set_title("(c) Huella del modelo", fontweight="bold")
    ax_c.grid(axis="y", alpha=0.25)
    ax_c.legend()
    ax_c.bar_label(bars_fp32, fmt="%.2f", padding=2, fontsize=9)
    ax_c.bar_label(bars_int8, fmt="%.2f", padding=2, fontsize=9)

    fig.suptitle("Comparación entre TFLite FP32 e INT8", fontsize=16, fontweight="bold")
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_outputs(canonical: list[dict], multiseed: list[dict]) -> None:
    canonical_path = DERIVED_DIR / "comparacion_fp32_int8_canonica.csv"
    with canonical_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(canonical[0].keys()))
        writer.writeheader()
        writer.writerows(canonical)

    summaries = []
    for experiment in ("E1", "E2", "E3", "E4"):
        deltas = [float(row["delta"]) for row in multiseed if row["experimento"] == experiment]
        summaries.append(
            {
                "experimento": experiment,
                "n_semillas": len(deltas),
                "media_delta_pp": mean(deltas),
                "mediana_delta_pp": median(deltas),
                "min_delta_pp": min(deltas),
                "max_delta_pp": max(deltas),
                "mejora": sum(value > 1e-12 for value in deltas),
                "sin_cambio": sum(abs(value) <= 1e-12 for value in deltas),
                "degrada": sum(value < -1e-12 for value in deltas),
            }
        )
    summary_path = DERIVED_DIR / "resumen_fp32_int8_multisemilla.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)
    for row in summaries:
        print(row)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    canonical = canonical_data()
    multiseed = multiseed_data()
    write_figure(canonical, multiseed, FIGURES_DIR / "comparacion_fp32_int8.svg")
    write_png(canonical, multiseed, FIGURES_DIR / "comparacion_fp32_int8.png")
    write_outputs(canonical, multiseed)


if __name__ == "__main__":
    main()
