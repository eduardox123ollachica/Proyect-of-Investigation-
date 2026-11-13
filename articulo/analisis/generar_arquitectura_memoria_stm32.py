"""Genera la figura de arquitectura DNN y ocupación de memoria del STM32."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parents[2]
FIGURES = ROOT / "articulo" / "figuras"
DERIVED = ROOT / "articulo" / "resultados_derivados"
PNG_PATH = FIGURES / "arquitectura_red_memoria_stm32.png"
SVG_PATH = FIGURES / "arquitectura_red_memoria_stm32.svg"
SUMMARY_PATH = DERIVED / "resumen_arquitectura_memoria_stm32.csv"

FLASH_CAPACITY = 512 * 1024
RAM_CAPACITY = 128 * 1024
MODEL_BYTES = 4480
FIRMWARE_FLASH_BYTES = 94200
STATIC_RAM_BYTES = 19312

LAYERS = [
    ("Entrada", "21 caract.", 0, "#D9EAF7"),
    ("Densa 1", "20 neur.\nReLU", 21 * 20 + 20, "#9ECAE1"),
    ("Densa 2", "10 neur.\nReLU", 20 * 10 + 10, "#6BAED6"),
    ("Salida", "3 clases\nSoftmax", 10 * 3 + 3, "#3182BD"),
]


def add_box(axis, x, y, width, height, color, title, subtitle, parameters):
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.025",
        facecolor=color,
        edgecolor="#24445C",
        linewidth=0.9,
    )
    axis.add_patch(box)
    axis.text(x + width / 2, y + height * 0.70, title, ha="center", va="center", fontsize=7.6, fontweight="bold")
    axis.text(x + width / 2, y + height * 0.43, subtitle, ha="center", va="center", fontsize=6.7)
    if parameters:
        axis.text(
            x + width / 2,
            y + height * 0.14,
            f"{parameters} par.",
            ha="center",
            va="center",
            fontsize=6.6,
            color="#333333",
        )


def create_figure():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
        }
    )

    fig, (ax_network, ax_memory) = plt.subplots(
        2,
        1,
        figsize=(3.45, 4.55),
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1.12, 1.0]},
    )

    ax_network.set_xlim(0, 1)
    ax_network.set_ylim(0, 1)
    ax_network.axis("off")
    ax_network.set_title("(a) Arquitectura del clasificador", pad=3)

    width = 0.17
    height = 0.48
    y = 0.28
    positions = [0.03, 0.285, 0.54, 0.795]
    for index, ((title, subtitle, parameters, color), x) in enumerate(zip(LAYERS, positions)):
        add_box(ax_network, x, y, width, height, color, title, subtitle, parameters)
        if index < len(LAYERS) - 1:
            ax_network.annotate(
                "",
                xy=(positions[index + 1] - 0.012, y + height / 2),
                xytext=(x + width + 0.012, y + height / 2),
                arrowprops={"arrowstyle": "->", "linewidth": 0.9, "color": "#555555"},
            )
    ax_network.text(0.5, 0.10, "Total: 683 parámetros | TFLite INT8: 4480 B", ha="center", fontsize=7.5, fontweight="bold")

    categories = ["Flash firmware", "RAM estática"]
    used = [FIRMWARE_FLASH_BYTES, STATIC_RAM_BYTES]
    capacities = [FLASH_CAPACITY, RAM_CAPACITY]
    percentages = [value / capacity * 100 for value, capacity in zip(used, capacities)]
    colors = ["#4C78A8", "#54A24B"]

    y_positions = [1, 0]
    for y_pos, category, value, capacity, percentage, color in zip(
        y_positions, categories, used, capacities, percentages, colors
    ):
        ax_memory.barh(y_pos, 100, color="#E6E6E6", height=0.48)
        ax_memory.barh(y_pos, percentage, color=color, height=0.48)
        ax_memory.text(
            percentage + 1.4,
            y_pos,
            f"{percentage:.2f} %\n{value:,} / {capacity:,} B",
            va="center",
            fontsize=7.1,
        )

    model_percentage = MODEL_BYTES / FLASH_CAPACITY * 100
    ax_memory.plot(model_percentage, 1, marker="|", markersize=15, markeredgewidth=1.5, color="#F58518")
    ax_memory.annotate(
        f"Modelo INT8: {MODEL_BYTES:,} B ({model_percentage:.2f} %)",
        xy=(model_percentage, 1.18),
        xytext=(29, 1.55),
        arrowprops={"arrowstyle": "->", "linewidth": 0.7, "color": "#A65400"},
        fontsize=6.8,
        color="#7A3E00",
        ha="center",
    )

    ax_memory.set_xlim(0, 100)
    ax_memory.set_ylim(-0.55, 1.75)
    ax_memory.set_yticks(y_positions, categories)
    ax_memory.set_xlabel("Porcentaje de la capacidad del STM32F446RE")
    ax_memory.set_title("(b) Ocupación medida del firmware completo", pad=3)
    ax_memory.grid(True, axis="x", color="#DDDDDD", linewidth=0.5)
    ax_memory.set_axisbelow(True)

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG_PATH, dpi=600, bbox_inches="tight")
    fig.savefig(SVG_PATH, bbox_inches="tight")
    plt.close(fig)

    summary = pd.DataFrame(
        [
            {
                "resource": "int8_model_file",
                "used_bytes": MODEL_BYTES,
                "capacity_bytes": FLASH_CAPACITY,
                "capacity_percent": model_percentage,
                "scope": "archivo TFLite INT8",
            },
            {
                "resource": "full_firmware_flash",
                "used_bytes": FIRMWARE_FLASH_BYTES,
                "capacity_bytes": FLASH_CAPACITY,
                "capacity_percent": percentages[0],
                "scope": "firmware completo E1/E4",
            },
            {
                "resource": "full_firmware_static_ram",
                "used_bytes": STATIC_RAM_BYTES,
                "capacity_bytes": RAM_CAPACITY,
                "capacity_percent": percentages[1],
                "scope": "RAM estatica del firmware completo E1/E4",
            },
        ]
    )
    DERIVED.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_PATH, index=False)
    return summary


def main():
    summary = create_figure()
    print(summary.to_string(index=False))
    print(f"\nPNG: {PNG_PATH}")
    print(f"SVG: {SVG_PATH}")
    print(f"CSV: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
