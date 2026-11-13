"""Consolida el factorial multisemilla E1--E4 y genera sus derivados."""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULTS = PROJECT_ROOT / "resultados/arciniegas/multisemilla/ejecuciones"
OUTPUT = PROJECT_ROOT / "articulo/resultados_derivados/multisemilla"
FIGURES = PROJECT_ROOT / "articulo/figuras"
T_CRITICAL_95 = {4: 2.776, 9: 2.262}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def macro_f1(y_true: list[int], y_pred: list[int]) -> float:
    scores = []
    for label in (0, 1, 2):
        tp = sum(t == label and p == label for t, p in zip(y_true, y_pred, strict=True))
        fp = sum(t != label and p == label for t, p in zip(y_true, y_pred, strict=True))
        fn = sum(t == label and p != label for t, p in zip(y_true, y_pred, strict=True))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(scores) / len(scores)


EXPERIMENTS = ("E1", "E2", "E3", "E4")


def collect() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for experiment_name in EXPERIMENTS:
        experiment = experiment_name.lower()
        for seed_dir in sorted((RESULTS / experiment).glob("seed_[0-9]*"), key=lambda p: int(p.name.split("_")[1])):
            seed = int(seed_dir.name.split("_")[1])
            tiny_path = seed_dir / "metricas_tinyml_fp32_int8.csv"
            pred_path = seed_dir / "comparacion_predicciones_fp32_int8.csv"
            history_path = seed_dir / "historial_entrenamiento.csv"
            if not all(path.is_file() for path in (tiny_path, pred_path, history_path)):
                continue
            tiny = {row["modelo"]: row for row in read_csv(tiny_path)}
            predictions = read_csv(pred_path)
            pred_key = "pred_tflite_int8" if "pred_tflite_int8" in predictions[0] else "pred_int8"
            y_true = [int(row["y_real"]) for row in predictions]
            y_pred = [int(row[pred_key]) for row in predictions]
            history = read_csv(history_path)
            rows.append(
                {
                    "experimento": experiment.upper(),
                    "semilla": seed,
                    "accuracy_keras_percent": float(tiny["Keras_FP32"]["accuracy_percent"]),
                    "accuracy_int8_percent": float(tiny["TFLite_INT8"]["accuracy_percent"]),
                    "macro_f1_int8_percent": 100.0 * macro_f1(y_true, y_pred),
                    "delta_int8_vs_keras_pp": float(tiny["TFLite_INT8"]["delta_accuracy_pp_vs_keras"]),
                    "accuracy_train_final_percent": 100.0 * float(history[-1]["accuracy"]),
                    "modelo_int8_bytes": int(float(tiny["TFLite_INT8"]["archivo_bytes"])),
                    "fuente": str(tiny_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                }
            )
    return rows


def describe(values: list[float]) -> dict[str, float]:
    n = len(values)
    mean = statistics.mean(values)
    sd = statistics.stdev(values) if n > 1 else 0.0
    df = n - 1
    t = T_CRITICAL_95.get(df)
    margin = t * sd / math.sqrt(n) if t is not None else float("nan")
    return {
        "n": n,
        "media": mean,
        "desviacion_estandar": sd,
        "mediana": statistics.median(values),
        "minimo": min(values),
        "maximo": max(values),
        "ic95_media_inferior": mean - margin,
        "ic95_media_superior": mean + margin,
    }


def exact_sign_test(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    lower = min(wins, losses)
    tail = sum(math.comb(n, k) for k in range(lower + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def render_svg(pairs: list[dict[str, float]], path: Path) -> None:
    width, height = 920, 560
    left, right, top, bottom = 80, 30, 70, 80
    plot_w, plot_h = width - left - right, height - top - bottom
    y_min, y_max = 90.0, 101.0
    seeds = [int(row["semilla"]) for row in pairs]
    def x(i: int) -> float:
        return left + (i + 0.5) * plot_w / len(pairs)
    def y(value: float) -> float:
        return top + (y_max - value) * plot_h / (y_max - y_min)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif}.title{font-size:23px;font-weight:700}.axis{font-size:14px}.legend{font-size:14px;font-weight:700}</style>',
        '<text x="460" y="34" text-anchor="middle" class="title">Robustez multisemilla: accuracy INT8</text>',
    ]
    for tick in range(90, 101, 2):
        yy = y(float(tick))
        parts.append(f'<line x1="{left}" y1="{yy}" x2="{width-right}" y2="{yy}" stroke="#D9E2F3"/>')
        parts.append(f'<text x="{left-12}" y="{yy+5}" text-anchor="end" class="axis">{tick}%</text>')
    colors = {"E1": "#1F6D8A", "E2": "#6AA84F", "E3": "#8E63A9", "E4": "#E87533"}
    offsets = {"E1": -9, "E2": -3, "E3": 3, "E4": 9}
    for i, row in enumerate(pairs):
        xx = x(i)
        for experiment in EXPERIMENTS:
            yy = y(float(row[experiment]))
            parts.append(f'<circle cx="{xx+offsets[experiment]}" cy="{yy}" r="5" fill="{colors[experiment]}"/>')
        parts.append(f'<text x="{xx}" y="{height-bottom+25}" text-anchor="middle" class="axis">{seeds[i]}</text>')
    parts.append(f'<text x="{width/2}" y="{height-18}" text-anchor="middle" class="axis">Semilla</text>')
    parts.append(f'<text x="22" y="{top+plot_h/2}" text-anchor="middle" class="axis" transform="rotate(-90 22 {top+plot_h/2})">Accuracy INT8 (%)</text>')
    for index, experiment in enumerate(EXPERIMENTS):
        xx = 590 + 70 * index
        parts.append(f'<circle cx="{xx}" cy="52" r="5" fill="{colors[experiment]}"/><text x="{xx+10}" y="57" class="legend">{experiment}</text>')
    parts.append('</svg>')
    path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    rows = collect()
    if not rows:
        raise RuntimeError("No se encontraron ejecuciones completas")
    fields = list(rows[0])
    with (OUTPUT / "resultados_por_semilla.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    by_experiment = {exp: [row for row in rows if row["experimento"] == exp] for exp in EXPERIMENTS}
    summaries = {
        exp: {
            "accuracy_int8_percent": describe([float(row["accuracy_int8_percent"]) for row in exp_rows]),
            "macro_f1_int8_percent": describe([float(row["macro_f1_int8_percent"]) for row in exp_rows]),
            "delta_int8_vs_keras_pp": describe([float(row["delta_int8_vs_keras_pp"]) for row in exp_rows]),
        }
        for exp, exp_rows in by_experiment.items()
    }
    by_seed = {
        exp: {int(row["semilla"]): row for row in exp_rows}
        for exp, exp_rows in by_experiment.items()
    }
    common_seeds = sorted(set.intersection(*(set(rows_by_seed) for rows_by_seed in by_seed.values())))
    paired: list[dict[str, float]] = []
    for seed in common_seeds:
        values = {
            exp: float(by_seed[exp][seed]["accuracy_int8_percent"])
            for exp in EXPERIMENTS
        }
        paired.append(
            {
                "semilla": seed,
                **values,
                "scaler_sin_pesos_E2_menos_E1_pp": values["E2"] - values["E1"],
                "scaler_con_pesos_E3_menos_E4_pp": values["E3"] - values["E4"],
                "pesos_sin_scaler_E4_menos_E1_pp": values["E4"] - values["E1"],
                "pesos_con_scaler_E3_menos_E2_pp": values["E3"] - values["E2"],
                "efecto_marginal_scaler_pp": ((values["E2"] + values["E3"]) - (values["E1"] + values["E4"])) / 2.0,
                "efecto_marginal_pesos_pp": ((values["E3"] + values["E4"]) - (values["E1"] + values["E2"])) / 2.0,
                "interaccion_pp": (values["E3"] - values["E2"]) - (values["E4"] - values["E1"]),
            }
        )

    contrast_names = (
        "scaler_sin_pesos_E2_menos_E1_pp",
        "scaler_con_pesos_E3_menos_E4_pp",
        "pesos_sin_scaler_E4_menos_E1_pp",
        "pesos_con_scaler_E3_menos_E2_pp",
        "efecto_marginal_scaler_pp",
        "efecto_marginal_pesos_pp",
        "interaccion_pp",
    )
    factorial_summary: dict[str, object] = {}
    for name in contrast_names:
        values = [float(row[name]) for row in paired]
        wins = sum(value > 0 for value in values)
        losses = sum(value < 0 for value in values)
        factorial_summary[name] = {
            "descripcion": describe(values),
            "positivos": wins,
            "negativos": losses,
            "empates": sum(value == 0 for value in values),
            "prueba_signos_exacta_p_bilateral": exact_sign_test(wins, losses),
        }
    summaries["contrastes_factoriales_pareados"] = factorial_summary
    # Alias conservado para no romper derivados históricos que leen este campo.
    summaries["comparacion_pareada"] = {
        "diferencia_E4_menos_E1_pp": factorial_summary["pesos_sin_scaler_E4_menos_E1_pp"]["descripcion"],
        "semillas_E4_supera_E1": factorial_summary["pesos_sin_scaler_E4_menos_E1_pp"]["positivos"],
        "semillas_E1_supera_E4": factorial_summary["pesos_sin_scaler_E4_menos_E1_pp"]["negativos"],
        "empates": factorial_summary["pesos_sin_scaler_E4_menos_E1_pp"]["empates"],
        "prueba_signos_exacta_p_bilateral": factorial_summary["pesos_sin_scaler_E4_menos_E1_pp"]["prueba_signos_exacta_p_bilateral"],
    }
    with (OUTPUT / "contrastes_factoriales_por_semilla.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired[0]))
        writer.writeheader()
        writer.writerows(paired)
    (OUTPUT / "resumen_multisemilla.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    render_svg(paired, FIGURES / "accuracy_multisemilla_factorial.svg")
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
