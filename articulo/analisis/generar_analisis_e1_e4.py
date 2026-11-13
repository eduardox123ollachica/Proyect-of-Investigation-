"""Genera resultados derivados para el articulo a partir de RAWALL E1 y E4.

No entrena modelos ni se comunica con el STM32. Solo lee resultados existentes.
"""

from __future__ import annotations

import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from xml.sax.saxutils import escape


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUTS = {
    "E1": PROJECT_ROOT / "resultados/arciniegas/e1/validacion_rawall/rawall_todas.csv",
    "E4": PROJECT_ROOT / "resultados/arciniegas/e4/validacion_rawall/rawall_todas.csv",
}
OUTPUT_DIR = PROJECT_ROOT / "articulo/resultados_derivados"
FIGURE_DIR = PROJECT_ROOT / "articulo/figuras"
CLASS_NAMES = {0: "DETENIDO", 1: "VELOCIDAD MEDIA", 2: "VELOCIDAD ALTA"}
BOOTSTRAP_SEED = 20261011
BOOTSTRAP_ITERATIONS = 10_000


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def confusion_matrix(rows: list[dict[str, str]]) -> list[list[int]]:
    matrix = [[0 for _ in CLASS_NAMES] for _ in CLASS_NAMES]
    for row in rows:
        matrix[int(row["label_real"])][int(row["clase_stm32"])] += 1
    return matrix


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def metrics(matrix: list[list[int]]) -> list[dict[str, object]]:
    total = sum(sum(row) for row in matrix)
    correct = sum(matrix[i][i] for i in CLASS_NAMES)
    rows: list[dict[str, object]] = []
    per_class: list[tuple[int, float, float, float]] = []
    for label, name in CLASS_NAMES.items():
        tp = matrix[label][label]
        support = sum(matrix[label])
        predicted = sum(matrix[true][label] for true in CLASS_NAMES)
        precision = safe_div(tp, predicted)
        recall = safe_div(tp, support)
        f1 = safe_div(2 * precision * recall, precision + recall)
        per_class.append((support, precision, recall, f1))
        rows.append(
            {
                "nivel": "clase",
                "clase": name,
                "support": support,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "accuracy": "",
            }
        )
    rows.append(
        {
            "nivel": "promedio_macro",
            "clase": "MACRO",
            "support": total,
            "precision": sum(x[1] for x in per_class) / len(per_class),
            "recall": sum(x[2] for x in per_class) / len(per_class),
            "f1": sum(x[3] for x in per_class) / len(per_class),
            "accuracy": safe_div(correct, total),
        }
    )
    rows.append(
        {
            "nivel": "promedio_ponderado",
            "clase": "WEIGHTED",
            "support": total,
            "precision": sum(x[0] * x[1] for x in per_class) / total,
            "recall": sum(x[0] * x[2] for x in per_class) / total,
            "f1": sum(x[0] * x[3] for x in per_class) / total,
            "accuracy": safe_div(correct, total),
        }
    )
    return rows


def exact_mcnemar_pvalue(b: int, c: int) -> float:
    discordant = b + c
    if discordant == 0:
        return 1.0
    lower = min(b, c)
    lower_tail = sum(math.comb(discordant, k) for k in range(lower + 1)) / (2**discordant)
    return min(1.0, 2.0 * lower_tail)


def percentile(sorted_values: list[float], probability: float) -> float:
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def stratified_file_bootstrap(
    e1_rows: list[dict[str, str]], e4_rows: list[dict[str, str]]
) -> tuple[float, float, float]:
    paired_by_class: dict[int, dict[str, list[tuple[bool, bool]]]] = defaultdict(lambda: defaultdict(list))
    for row1, row4 in zip(e1_rows, e4_rows, strict=True):
        label = int(row1["label_real"])
        filename = row1["file"]
        paired_by_class[label][filename].append(
            (int(row1["clase_stm32"]) == label, int(row4["clase_stm32"]) == label)
        )

    rng = random.Random(BOOTSTRAP_SEED)
    differences: list[float] = []
    for _ in range(BOOTSTRAP_ITERATIONS):
        e1_correct = 0
        e4_correct = 0
        total = 0
        for label in sorted(paired_by_class):
            filenames = sorted(paired_by_class[label])
            for sampled in rng.choices(filenames, k=len(filenames)):
                for correct1, correct4 in paired_by_class[label][sampled]:
                    e1_correct += int(correct1)
                    e4_correct += int(correct4)
                    total += 1
        differences.append(100.0 * (e4_correct - e1_correct) / total)

    differences.sort()
    observed = 100.0 * (
        sum(int(row["clase_stm32"]) == int(row["label_real"]) for row in e4_rows)
        - sum(int(row["clase_stm32"]) == int(row["label_real"]) for row in e1_rows)
    ) / len(e1_rows)
    return observed, percentile(differences, 0.025), percentile(differences, 0.975)


def render_confusion_svg(experiment: str, matrix: list[list[int]], accuracy: float, path: Path) -> None:
    width, height = 760, 610
    left, top, cell = 245, 125, 110
    maximum = max(max(row) for row in matrix)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif}.title{font-size:24px;font-weight:700}.axis{font-size:16px;font-weight:700}.label{font-size:14px}.count{font-size:24px;font-weight:700}</style>',
        f'<text x="{width/2}" y="38" text-anchor="middle" class="title">Matriz de confusión INT8/STM32 — {escape(experiment)}</text>',
        f'<text x="{width/2}" y="68" text-anchor="middle" class="label">Accuracy: {accuracy * 100:.4f} % · n = {sum(sum(row) for row in matrix)}</text>',
        f'<text x="{left + 1.5*cell}" y="98" text-anchor="middle" class="axis">Predicción</text>',
        f'<text x="42" y="{top + 1.5*cell}" text-anchor="middle" class="axis" transform="rotate(-90 42 {top + 1.5*cell})">Clase real</text>',
    ]
    short_names = ["DETENIDO", "MEDIA", "ALTA"]
    for i, label in enumerate(short_names):
        parts.append(f'<text x="{left + (i+0.5)*cell}" y="{top-14}" text-anchor="middle" class="label">{label}</text>')
        parts.append(f'<text x="{left-14}" y="{top + (i+0.55)*cell}" text-anchor="end" class="label">{label}</text>')
    for true_label in CLASS_NAMES:
        for predicted_label in CLASS_NAMES:
            value = matrix[true_label][predicted_label]
            ratio = value / maximum if maximum else 0
            red = round(238 - 184 * ratio)
            green = round(245 - 137 * ratio)
            blue = round(250 - 157 * ratio)
            fill = f"rgb({red},{green},{blue})"
            text_color = "white" if ratio > 0.55 else "#17365D"
            x = left + predicted_label * cell
            y = top + true_label * cell
            parts.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{fill}" stroke="white" stroke-width="3"/>')
            parts.append(f'<text x="{x+cell/2}" y="{y+cell/2+8}" text-anchor="middle" class="count" fill="{text_color}">{value}</text>')
    parts.append('<text x="380" y="525" text-anchor="middle" class="label" fill="#555">Fuente: validación RAWALL sobre las 1332 ventanas del conjunto de prueba.</text>')
    parts.append('</svg>')
    path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    data = {experiment: read_rows(path) for experiment, path in INPUTS.items()}
    key_fields = ("indice_test", "file", "window", "label_real")
    keys_e1 = [tuple(row[field] for field in key_fields) for row in data["E1"]]
    keys_e4 = [tuple(row[field] for field in key_fields) for row in data["E4"]]
    if keys_e1 != keys_e4:
        raise ValueError("E1 y E4 no contienen las mismas ventanas en el mismo orden")

    matrices = {experiment: confusion_matrix(rows) for experiment, rows in data.items()}
    metric_rows: list[dict[str, object]] = []
    matrix_rows: list[dict[str, object]] = []
    for experiment, matrix in matrices.items():
        for row in metrics(matrix):
            metric_rows.append({"experimento": experiment, **row})
        for true_label, true_name in CLASS_NAMES.items():
            for predicted_label, predicted_name in CLASS_NAMES.items():
                matrix_rows.append(
                    {
                        "experimento": experiment,
                        "clase_real": true_name,
                        "clase_predicha": predicted_name,
                        "conteo": matrix[true_label][predicted_label],
                    }
                )
        accuracy = sum(matrix[i][i] for i in CLASS_NAMES) / sum(sum(row) for row in matrix)
        render_confusion_svg(
            experiment,
            matrix,
            accuracy,
            FIGURE_DIR / f"matriz_confusion_{experiment.lower()}_int8_stm32.svg",
        )

    with (OUTPUT_DIR / "metricas_int8_stm32_e1_e4.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)
    with (OUTPUT_DIR / "matrices_confusion_int8_stm32_e1_e4.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(matrix_rows[0]))
        writer.writeheader()
        writer.writerows(matrix_rows)

    correct_e1 = [int(row["clase_stm32"]) == int(row["label_real"]) for row in data["E1"]]
    correct_e4 = [int(row["clase_stm32"]) == int(row["label_real"]) for row in data["E4"]]
    both_correct = sum(a and b for a, b in zip(correct_e1, correct_e4, strict=True))
    e1_correct_e4_wrong = sum(a and not b for a, b in zip(correct_e1, correct_e4, strict=True))
    e1_wrong_e4_correct = sum(not a and b for a, b in zip(correct_e1, correct_e4, strict=True))
    both_wrong = sum(not a and not b for a, b in zip(correct_e1, correct_e4, strict=True))
    p_value = exact_mcnemar_pvalue(e1_correct_e4_wrong, e1_wrong_e4_correct)
    observed, ci_low, ci_high = stratified_file_bootstrap(data["E1"], data["E4"])
    comparison = {
        "ventanas": len(data["E1"]),
        "archivos_test": len({row["file"] for row in data["E1"]}),
        "ambos_correctos": both_correct,
        "e1_correcto_e4_incorrecto": e1_correct_e4_wrong,
        "e1_incorrecto_e4_correcto": e1_wrong_e4_correct,
        "ambos_incorrectos": both_wrong,
        "mcnemar_exacto_p": p_value,
        "diferencia_accuracy_e4_menos_e1_pp": observed,
        "bootstrap_archivos_estratificado_iteraciones": BOOTSTRAP_ITERATIONS,
        "bootstrap_semilla": BOOTSTRAP_SEED,
        "bootstrap_ic95_inferior_pp": ci_low,
        "bootstrap_ic95_superior_pp": ci_high,
    }
    (OUTPUT_DIR / "comparacion_pareada_e1_e4.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(comparison, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
