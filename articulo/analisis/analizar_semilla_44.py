"""Genera evidencia reproducible para el análisis detallado de la semilla 44.

Reutiliza resultados existentes; no entrena ni modifica modelos. Solo requiere
la biblioteca estándar de Python y genera SVG más un resumen CSV.
"""

from __future__ import annotations

import csv
from pathlib import Path
from xml.sax.saxutils import escape


ARTICLE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ARTICLE_DIR.parent
RESULTS_DIR = PROJECT_ROOT / "resultados" / "arciniegas" / "multisemilla" / "ejecuciones"
FIGURES_DIR = ARTICLE_DIR / "figuras"
DERIVED_DIR = ARTICLE_DIR / "resultados_derivados"
CLASS_NAMES = ["DETENIDO", "MEDIA", "ALTA"]
COLORS = {"E1": "#2A6F8F", "E4": "#E76F51"}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_experiment(experiment: str) -> dict:
    folder = RESULTS_DIR / experiment.lower() / "seed_44"
    probabilities = read_rows(folder / "probabilidades_test.csv")
    media_margins = [
        float(row["P_MEDIA"]) - float(row["P_DETENIDO"])
        for row in probabilities
        if int(row["label"]) == 1
    ]
    errors = [row for row in probabilities if row["correcto"].lower() != "true"]
    media_errors = [row for row in errors if int(row["label"]) == 1]

    confusion_rows = read_rows(folder / "matriz_confusion.csv")
    confusion = [
        [int(row["PRED_DETENIDO"]), int(row["PRED_MEDIA"]), int(row["PRED_ALTA"])]
        for row in confusion_rows
    ]
    history = read_rows(folder / "historial_entrenamiento.csv")
    metrics = read_rows(folder / "metricas_tinyml_fp32_int8.csv")
    comparison = read_rows(folder / "comparacion_predicciones_fp32_int8.csv")
    keras = next(row for row in metrics if row["modelo"] == "Keras_FP32")
    int8 = next(row for row in metrics if row["modelo"] == "TFLite_INT8")

    return {
        "media_margins": media_margins,
        "errors": errors,
        "media_errors": media_errors,
        "confusion": confusion,
        "train_accuracy": 100.0 * float(history[-1]["accuracy"]),
        "keras_accuracy": float(keras["accuracy_percent"]),
        "int8_accuracy": float(int8["accuracy_percent"]),
        "int8_delta": float(int8["delta_accuracy_pp_vs_keras"]),
        "changed_predictions": sum(
            row["cambio_prediccion"].lower() == "true" for row in comparison
        ),
    }


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def svg_text(x: float, y: float, value: str, size: int = 15, **attributes: str) -> str:
    extra = " ".join(
        f'{key.replace("_", "-")}="{escape(str(val))}"'
        for key, val in attributes.items()
    )
    return f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" {extra}>{escape(value)}</text>'


def confusion_panel(x: int, y: int, matrix: list[list[int]], title: str) -> list[str]:
    output = [svg_text(x + 225, y, title, 20, text_anchor="middle", font_weight="bold")]
    cell = 92
    grid_x, grid_y = x + 120, y + 55
    output.append(svg_text(grid_x + 1.5 * cell, grid_y - 18, "Predicción", 14, text_anchor="middle"))
    for index, label in enumerate(CLASS_NAMES):
        output.append(svg_text(grid_x + (index + 0.5) * cell, grid_y + 18, label, 12, text_anchor="middle"))
        output.append(svg_text(grid_x - 12, grid_y + (index + 0.62) * cell, label, 12, text_anchor="end"))
    for row in range(3):
        for column in range(3):
            value = matrix[row][column]
            intensity = value / 540
            shade = int(246 - 150 * intensity)
            fill = f"rgb({shade},{min(250, shade + 18)},{min(255, shade + 25)})"
            output.append(
                f'<rect x="{grid_x + column * cell}" y="{grid_y + 30 + row * cell}" '
                f'width="{cell}" height="{cell}" fill="{fill}" stroke="#ffffff" stroke-width="2"/>'
            )
            output.append(
                svg_text(
                    grid_x + (column + 0.5) * cell,
                    grid_y + 30 + (row + 0.57) * cell,
                    str(value),
                    18,
                    text_anchor="middle",
                    font_weight="bold",
                    fill="#ffffff" if intensity > 0.55 else "#17324D",
                )
            )
    output.append(
        f'<text x="{grid_x - 82}" y="{grid_y + 180}" font-size="14" text-anchor="middle" '
        f'transform="rotate(-90 {grid_x - 82} {grid_y + 180})">Clase real</text>'
    )
    return output


def ecdf_panel(x: int, y: int, width: int, height: int, data: dict[str, dict]) -> list[str]:
    output = [svg_text(x + width / 2, y, "(c) Ventanas reales de clase MEDIA", 20, text_anchor="middle", font_weight="bold")]
    left, top = x + 75, y + 45
    plot_w, plot_h = width - 100, height - 115
    all_values = data["E1"]["media_margins"] + data["E4"]["media_margins"]
    x_min, x_max = min(all_values), max(all_values)
    padding = 0.04 * (x_max - x_min)
    x_min, x_max = x_min - padding, x_max + padding

    def sx(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * plot_w

    def sy(value: float) -> float:
        return top + plot_h - value * plot_h

    output.append(f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#ffffff" stroke="#333333"/>')
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        yy = sy(fraction)
        output.append(f'<line x1="{left}" y1="{yy}" x2="{left + plot_w}" y2="{yy}" stroke="#dddddd"/>')
        output.append(svg_text(left - 10, yy + 5, f"{fraction:.2f}", 12, text_anchor="end"))
    zero_x = sx(0.0)
    output.append(f'<line x1="{zero_x}" y1="{top}" x2="{zero_x}" y2="{top + plot_h}" stroke="#333333" stroke-dasharray="7,5" stroke-width="2"/>')
    output.append(svg_text(zero_x + 7, top + 18, "frontera 0", 12, fill="#333333"))
    for experiment in ("E1", "E4"):
        ordered = sorted(data[experiment]["media_margins"])
        points = " ".join(
            f"{sx(value):.1f},{sy((index + 1) / len(ordered)):.1f}"
            for index, value in enumerate(ordered)
        )
        output.append(f'<polyline points="{points}" fill="none" stroke="{COLORS[experiment]}" stroke-width="3"/>')
    output.append(svg_text(left + plot_w / 2, top + plot_h + 45, "Margen P(MEDIA) − P(DETENIDO)", 14, text_anchor="middle"))
    output.append(
        f'<text x="{left - 55}" y="{top + plot_h / 2}" font-size="14" text-anchor="middle" '
        f'transform="rotate(-90 {left - 55} {top + plot_h / 2})">Proporción acumulada</text>'
    )
    output.append(f'<line x1="{left + 20}" y1="{top + 25}" x2="{left + 55}" y2="{top + 25}" stroke="{COLORS["E1"]}" stroke-width="3"/>')
    output.append(svg_text(left + 63, top + 30, "E1", 13))
    output.append(f'<line x1="{left + 110}" y1="{top + 25}" x2="{left + 145}" y2="{top + 25}" stroke="{COLORS["E4"]}" stroke-width="3"/>')
    output.append(svg_text(left + 153, top + 30, "E4", 13))
    return output


def accuracy_panel(x: int, y: int, width: int, height: int, data: dict[str, dict]) -> list[str]:
    output = [svg_text(x + width / 2, y, "(d) Entrenamiento frente a prueba", 20, text_anchor="middle", font_weight="bold")]
    left, top = x + 70, y + 45
    plot_w, plot_h = width - 100, height - 115
    y_min, y_max = 93.5, 100.5

    def sy(value: float) -> float:
        return top + plot_h - (value - y_min) / (y_max - y_min) * plot_h

    output.append(f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#ffffff" stroke="#333333"/>')
    for tick in (94, 96, 98, 100):
        yy = sy(tick)
        output.append(f'<line x1="{left}" y1="{yy}" x2="{left + plot_w}" y2="{yy}" stroke="#dddddd"/>')
        output.append(svg_text(left - 10, yy + 5, str(tick), 12, text_anchor="end"))
    centers = [left + plot_w * 0.30, left + plot_w * 0.72]
    bar_w = 55
    for center, experiment in zip(centers, ("E1", "E4")):
        train = data[experiment]["train_accuracy"]
        test = data[experiment]["keras_accuracy"]
        for offset, value, color in ((-bar_w, train, "#7A9E7E"), (0, test, "#D98C5F")):
            bar_y = sy(value)
            output.append(f'<rect x="{center + offset}" y="{bar_y}" width="{bar_w}" height="{top + plot_h - bar_y}" fill="{color}"/>')
            output.append(svg_text(center + offset + bar_w / 2, bar_y - 7, f"{value:.2f}", 11, text_anchor="middle"))
        int8_y = sy(data[experiment]["int8_accuracy"])
        output.append(f'<polygon points="{center + 15},{int8_y - 6} {center + 21},{int8_y} {center + 15},{int8_y + 6} {center + 9},{int8_y}" fill="#222222"/>')
        output.append(svg_text(center - bar_w / 2, top + plot_h + 28, experiment, 14, text_anchor="middle", font_weight="bold"))
    legend_y = top - 12
    output.append(f'<rect x="{left + 15}" y="{legend_y - 11}" width="18" height="12" fill="#7A9E7E"/>')
    output.append(svg_text(left + 40, legend_y, "Entrenamiento", 12))
    output.append(f'<rect x="{left + 145}" y="{legend_y - 11}" width="18" height="12" fill="#D98C5F"/>')
    output.append(svg_text(left + 170, legend_y, "Prueba FP32", 12))
    output.append(
        f'<text x="{left - 47}" y="{top + plot_h / 2}" font-size="14" text-anchor="middle" '
        f'transform="rotate(-90 {left - 47} {top + plot_h / 2})">Exactitud (%)</text>'
    )
    return output


def write_figure(data: dict[str, dict], path: Path) -> None:
    width, height = 1500, 1120
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<g font-family="Arial, Helvetica, sans-serif" fill="#172B3A">',
        svg_text(width / 2, 38, "Diagnóstico de la inversión observada con la semilla 44", 26, text_anchor="middle", font_weight="bold"),
    ]
    parts.extend(confusion_panel(40, 82, data["E1"]["confusion"], "(a) E1, semilla 44"))
    parts.extend(confusion_panel(770, 82, data["E4"]["confusion"], "(b) E4, semilla 44"))
    parts.extend(ecdf_panel(35, 585, 720, 500, data))
    parts.extend(accuracy_panel(770, 585, 690, 500, data))
    parts.extend(["</g>", "</svg>"])
    path.write_text("\n".join(parts), encoding="utf-8")


def write_summary(data: dict[str, dict], path: Path) -> None:
    fields = [
        "experimento", "semilla", "accuracy_train_final_percent",
        "accuracy_keras_test_percent", "accuracy_int8_test_percent",
        "errores_keras_total", "errores_keras_media_a_detenido",
        "mediana_margen_media_detenido", "p10_margen_media_detenido",
        "minimo_margen_media_detenido", "predicciones_cambiadas_fp32_int8",
        "delta_int8_vs_keras_pp",
    ]
    rows = []
    for experiment, values in data.items():
        margins = values["media_margins"]
        rows.append({
            "experimento": experiment,
            "semilla": 44,
            "accuracy_train_final_percent": values["train_accuracy"],
            "accuracy_keras_test_percent": values["keras_accuracy"],
            "accuracy_int8_test_percent": values["int8_accuracy"],
            "errores_keras_total": len(values["errors"]),
            "errores_keras_media_a_detenido": len(values["media_errors"]),
            "mediana_margen_media_detenido": percentile(margins, 0.5),
            "p10_margen_media_detenido": percentile(margins, 0.1),
            "minimo_margen_media_detenido": min(margins),
            "predicciones_cambiadas_fp32_int8": values["changed_predictions"],
            "delta_int8_vs_keras_pp": values["int8_delta"],
        })
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    data = {experiment: load_experiment(experiment) for experiment in ("E1", "E4")}
    write_figure(data, FIGURES_DIR / "analisis_semilla_44.svg")
    write_summary(data, DERIVED_DIR / "resumen_semilla_44.csv")


if __name__ == "__main__":
    main()
