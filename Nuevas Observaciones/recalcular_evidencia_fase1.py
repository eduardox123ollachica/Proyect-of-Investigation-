"""Recalcula las tablas de la revisión CWRU desde los artefactos congelados.

Ejecutar con Python 3.12 y NumPy. El argumento --check comprueba el JSON
entregado sin escribirlo. El cálculo no ejecuta inferencia ni entrenamiento.
"""

import argparse
import hashlib
import json
from math import isclose
from pathlib import Path, PureWindowsPath
from statistics import mean, stdev

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path(__file__).with_name("EVIDENCIA_FASE1_CWRU.json")
OUTER = (
    "resultados/cwru/ampliacion_v2/evaluacion_exterior_v2_2026-10-27/"
    "20261027T222625_337707Z"
)
METHODOLOGY = "articulo/ampliacion_cwru/metodologia"
FROZEN = f"{METHODOLOGY}/seleccion_congelada_v2_2026-10-27/DECISIONES_CONGELADAS.json"
PHYSICAL = (
    f"{METHODOLOGY}/validacion_fisica_canonicos_completa_2026-10-28/"
    "RESUMEN_VALIDACION_FISICA.json"
)
COSTS = (
    "despliegue_stm32/costo_total_single_rounding_v2_180mhz_2026-10-26/"
    "COSTOS_TOTALES_SR_VALIDADOS.json"
)
LOADS = (0, 1, 2, 3)
SEEDS = (11, 22, 33, 42, 44, 55, 66, 77, 88, 99)
SUPPORT = ((28, 62, 85), (58, 161, 174), (58, 174, 174), (58, 174, 174))
BASELINE = "baseline_F14"
PERFORMANCE = "performance_only"
COST = "validated_total_cost"
PROCEDURES = (BASELINE, PERFORMANCE, COST)
ERRORS = ("normal_como_falla", "falla_como_normal", "confusion_IR_OR")
WEIGHTS = {"prioridad_deteccion": (1, 5, 0), "prioridad_subtipo": (1, 5, 1)}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def registered_path(relative, hashes):
    key = Path(relative).as_posix()
    path = ROOT / key
    if key not in hashes:
        hashes[key] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path


def read_json(relative, hashes):
    return json.loads(registered_path(relative, hashes).read_text(encoding="utf-8-sig"))


def prediction_metrics(exterior, hashes):
    partitions = {}
    metrics = {}
    for cell in exterior["cells"]:
        procedure = cell["procedure"]
        if cell["precision"] != "INT8" or procedure not in PROCEDURES:
            continue
        load, seed = cell["outer_load_hp"], cell["seed"]
        key = load, procedure, seed
        require(key not in metrics, f"Celda INT8 repetida: {key}")
        prediction_file = PureWindowsPath(cell["prediction_file"]).as_posix()
        relative = f"{OUTER}/{prediction_file}"
        with np.load(registered_path(relative, hashes), allow_pickle=False) as predictions:
            y = predictions["y_true"]
            pred = predictions["int8_predictions"]
            ids = predictions["global_window_ids"]
        require(y.shape == pred.shape == ids.shape == (sum(SUPPORT[load]),), key)
        require(np.unique(ids).size == ids.size, f"IDs repetidos: {key}")
        require(np.isin(y, (0, 1, 2)).all() and np.isin(pred, (0, 1, 2)).all(), key)
        if load not in partitions:
            partitions[load] = (ids, y)
        reference_ids, reference_y = partitions[load]
        require(np.array_equal(ids, reference_ids) and np.array_equal(y, reference_y), key)

        matrix = np.zeros((3, 3), dtype=np.int64)
        np.add.at(matrix, (y, pred), 1)
        require(np.array_equal(matrix, cell["confusion_matrix"]), key)
        require(tuple(matrix.sum(axis=1)) == SUPPORT[load], key)
        true_positives = matrix.diagonal()
        accuracy = float(true_positives.sum() / matrix.sum())
        macro_f1 = float((2 * true_positives / (matrix.sum(axis=0) + matrix.sum(axis=1))).mean())
        require(isclose(accuracy, cell["accuracy"], rel_tol=0.0, abs_tol=1e-12), key)
        require(isclose(macro_f1, cell["macro_f1"], rel_tol=0.0, abs_tol=1e-12), key)
        metrics[key] = {"matrix": matrix, "accuracy": accuracy, "macro_f1": macro_f1,
                        "model": cell["model_job_id"]}

    expected = {(load, procedure, seed) for load in LOADS
                for procedure in PROCEDURES for seed in SEEDS}
    require(metrics.keys() == expected, "Cobertura inesperada de las celdas exteriores INT8")
    all_ids = {int(window) for ids, _ in partitions.values() for window in ids}
    require(len(all_ids) == 1380, "Las cargas exteriores deben reunir 1380 ventanas distintas")
    return metrics, partitions


def summarize_errors(cells, support):
    # Las diez semillas se evalúan sobre las mismas ventanas de la carga.
    # Promediar primero las matrices equivale aquí a promediar las tasas.
    matrix = np.mean([cell["matrix"] for cell in cells], axis=0)
    n_normal, n_ir, n_or = support
    n_fault = n_ir + n_or
    counts = [float(matrix[0, 1] + matrix[0, 2]),
              float(matrix[1, 0] + matrix[2, 0]),
              float(matrix[1, 2] + matrix[2, 1])]
    denominators = [n_normal, n_fault, n_fault]
    loss = {
        scenario: 100 * sum(weight * count for weight, count in zip(weights, counts))
        / sum(support)
        for scenario, weights in WEIGHTS.items()
    }
    return {
        "matriz_media": matrix.tolist(),
        "conteos_medios_de_error": dict(zip(ERRORS, counts)),
        "denominadores_de_error": dict(zip(ERRORS, denominators)),
        "tasas_de_error_porcentaje": {
            error: 100 * count / denominator
            for error, count, denominator in zip(ERRORS, counts, denominators)
        },
        "perdida_por_100_ventanas": loss,
        "macro_f1_porcentaje": 100 * mean(cell["macro_f1"] for cell in cells),
        "exactitud_porcentaje": 100 * mean(cell["accuracy"] for cell in cells),
    }


def summarize_loads(metrics, physical):
    by_model = {row["model"]: row for row in physical["latencies"]}
    require(len(by_model) == len(physical["latencies"]) == 8, "Banco canónico incompleto")
    require(all((row["seed"], row["windows"], row["RUN"]) == (42, 42, 126)
                for row in by_model.values()), "Cobertura de tiempo canónico inesperada")
    coverage = physical["coverage"]
    require((coverage["RAW_pairs"], coverage["POST"], coverage["POST_unique"]) ==
            (336, 2760, 1380), "Cobertura RAW/POST inesperada")
    rows = []
    for load in LOADS:
        compared = {
            procedure: summarize_errors(
                [metrics[load, procedure, seed] for seed in SEEDS], SUPPORT[load]
            )
            for procedure in PROCEDURES
        }
        baseline_model = metrics[load, BASELINE, 42]["model"]
        cost_model = metrics[load, COST, 42]["model"]
        baseline_time, cost_time = (by_model[model]["mean_ms"]
                                    for model in (baseline_model, cost_model))
        require(all(by_model[model]["outer_load_hp"] == load
                    for model in (baseline_model, cost_model)), load)
        rows.append({
            "carga_hp": load,
            "ventanas_por_clase": dict(zip(("NORMAL", "IR", "OR"), SUPPORT[load])),
            "resultados": compared,
            "delta_macro_f1_costo_menos_F14_pp":
                compared[COST]["macro_f1_porcentaje"] - compared[BASELINE]["macro_f1_porcentaje"],
            "delta_exactitud_costo_menos_F14_pp":
                compared[COST]["exactitud_porcentaje"] - compared[BASELINE]["exactitud_porcentaje"],
            "delta_macro_f1_costo_menos_rendimiento_pp":
                compared[COST]["macro_f1_porcentaje"] - compared[PERFORMANCE]["macro_f1_porcentaje"],
            "tiempo_canonico_semilla_42": {
                "modelo_F14": baseline_model,
                "modelo_costo": cost_model,
                "mascara_costo": by_model[cost_model]["candidate"],
                "F14_ms": baseline_time,
                "costo_ms": cost_time,
                "reduccion_porcentaje": 100 * (baseline_time - cost_time) / baseline_time,
            },
        })
    return rows


def summarize_frozen_costs(frozen, costs):
    decisions = {row["outer_load_hp"]: row for row in frozen["decisions"]}
    require(sorted(decisions) == list(LOADS), "Decisiones congeladas incompletas")
    rows = costs["rows"]
    require(costs["status"] == "VALIDATED_TOTAL_COST" and len(rows) == 32,
            "Tabla de costos sin validación o incompleta")
    all_windows = set()
    load_banks = []
    for load in LOADS:
        fold = [row for row in rows if row["outer_load"] == load]
        candidate_names = set(decisions[load]["cost"]["candidate_summaries"])
        require(len(fold) == len(candidate_names) == 8, load)
        require({row["candidate"] for row in fold} == candidate_names, load)
        ids = fold[0]["benchmark_global_window_ids"]
        bank_loads = fold[0]["benchmark_loads"]
        validation_load = fold[0]["inner_validation_load"]
        require(len(ids) == len(set(ids)) == 28, load)
        require(validation_load == min(set(LOADS) - {load}), load)
        require(set(bank_loads) == set(LOADS) - {load, validation_load}, load)
        for row in fold:
            require(row["benchmark_global_window_ids"] == ids, load)
            require(row["benchmark_loads"] == bank_loads and
                    row["inner_validation_load"] == validation_load, load)
            require((row["seed"], row["unique_windows"], row["repetitions"], row["observations"])
                    == (11, 28, 3, 84) and row["validated_as_cost_input"], load)
        all_windows.update(ids)
        load_banks.append({"carga_exterior_hp": load, "carga_validacion_interna_hp": validation_load,
                           "cargas_banco_hp": bank_loads})
    require(len(all_windows) == 42 and sum(row["observations"] for row in rows) == 2688,
            "Cobertura temporal interna inesperada")

    decision = decisions[3]["cost"]
    timings = {row["candidate"]: row for row in rows if row["outer_load"] == 3}
    floor = decision["macro_f1_threshold"]
    ceiling = 0.8 * timings["F14"]["lower_ms"]
    candidates = []
    for mask, prediction in decision["candidate_summaries"].items():
        measured = timings[mask]
        for field in ("mean_ms", "lower_ms", "upper_ms"):
            require(measured[field] == decision["validated_total_costs"][mask][field], mask)
        score = prediction["mean_macro_f1_int8"]
        candidates.append({
            "mascara": mask,
            "macro_f1_interno_porcentaje": 100 * score,
            "media_ms": measured["mean_ms"],
            "L_ms": measured["lower_ms"],
            "U_ms": measured["upper_ms"],
            "cumple_guardia_detector": prediction["detector_admissible"],
            "cumple_piso_predictivo": score >= floor - 1e-12,
            "cumple_techo_temporal": measured["upper_ms"] <= ceiling,
        })
    predictive = [row["mascara"] for row in candidates
                  if row["cumple_guardia_detector"] and row["cumple_piso_predictivo"]]
    eligible = [row["mascara"] for row in candidates if row["mascara"] in predictive
                and row["cumple_techo_temporal"]]
    require(predictive == decision["predictive_eligible_candidates"] and
            eligible == decision["target_eligible_candidates"], "La decisión no se reproduce")
    require(not eligible and not decision["goal_met"] and
            decision["selected_candidate"] == "F14", "Respaldo 3 HP inesperado")
    return {
        "banco_interno": {
            "representantes": len(rows), "ventanas_unicas": len(all_windows),
            "ejecuciones": sum(row["observations"] for row in rows),
            "bancos_por_carga_exterior": load_banks,
        },
        "escenario_3_HP": {
            "piso_macro_f1_interno_porcentaje": 100 * floor, "techo_temporal_ms": ceiling,
            "mascara_respaldo": decision["selected_candidate"], "candidatas": candidates,
        },
    }


def summarize_detector(exterior, partitions, hashes):
    rows = []
    for cell in exterior["detector_cells"]:
        load, model = cell["outer_load_hp"], cell["model_job_id"]
        relative = f"{OUTER}/outer_eval_{load}/{model}_predicciones.npz"
        with np.load(registered_path(relative, hashes), allow_pickle=False) as stored:
            y, ids = stored["y_true"], stored["global_window_ids"]
            scores, rejected = stored["scores"], stored["rejected"]
            threshold = float(stored["threshold32"].item())
        expected_ids, expected_y = partitions[load]
        require(np.array_equal(ids, expected_ids) and np.array_equal(y, expected_y), model)
        require(np.isfinite(scores).all() and np.isfinite(threshold), model)
        require(threshold == cell["threshold_float32"] and
                np.array_equal(scores > threshold, rejected), model)
        normal_scores = scores[y == 0]
        normal_spec = float((~rejected[y == 0]).mean())
        require(normal_spec == cell["normal_spec"] and
                float(rejected[y == 1].mean()) == cell["ir_recall"] and
                float(rejected[y == 2].mean()) == cell["or_recall"], model)
        rows.append({
            "carga_hp": load, "procedimiento": cell["procedure"], "modelo": model,
            "umbral": threshold, "ventanas_normales": int(normal_scores.size),
            "normales_rechazadas": int(rejected[y == 0].sum()),
            "minimo_normal": float(normal_scores.min()),
            "maximo_normal": float(normal_scores.max()),
            "minimo_falla": float(scores[y != 0].min()),
            "especificidad_porcentaje": 100 * normal_spec,
        })
    expected = {(load, procedure) for load in LOADS for procedure in PROCEDURES}
    require({(row["carga_hp"], row["procedimiento"]) for row in rows} == expected and
            len(rows) == len(expected), "Cobertura del detector incompleta")
    return rows


def build_evidence():
    hashes = {}
    exterior = read_json(f"{OUTER}/RESULTADOS_EXTERIORES.json", hashes)
    frozen = read_json(FROZEN, hashes)
    physical = read_json(PHYSICAL, hashes)
    costs = read_json(COSTS, hashes)
    metrics, partitions = prediction_metrics(exterior, hashes)
    load_rows = summarize_loads(metrics, physical)
    improvements = [row["delta_macro_f1_costo_menos_rendimiento_pp"] for row in load_rows]
    paired = [
        100 * mean(metrics[load, COST, seed]["macro_f1"] - metrics[load, BASELINE, seed]["macro_f1"]
                   for load in LOADS)
        for seed in SEEDS
    ]
    return {
        "alcance": "Análisis descriptivo retrospectivo desde resultados y costos ya congelados.",
        "semillas_predictivas_INT8": SEEDS,
        "celdas_INT8_contrastadas_con_predicciones_NPZ": len(metrics),
        "comprobacion_fisica_canonica": physical["coverage"],
        "perdida_hipotetica": {
            "unidad": "unidades de penalización por 100 ventanas con la composición observada por carga",
            "formula": "100 × suma de (peso × conteo medio del error) / ventanas de la carga",
            "orden_de_errores": ERRORS,
            "pesos_por_escenario": WEIGHTS,
            "alcance": "Pesos ilustrativos; la latencia se informa aparte y no interviene en la pérdida.",
        },
        "comparacion_exterior_por_carga": load_rows,
        "diferencias_exteriores": {
            "delta_macro_f1_costo_menos_F14_pp": mean(paired),
            "DE_muestral_entre_semillas_pp": stdev(paired),
            "rango_entre_semillas_pp": [min(paired), max(paired)],
            "delta_macro_f1_costo_menos_rendimiento_pp": mean(improvements),
            "aporte_3_HP_a_delta_global_costo_menos_rendimiento_pp": improvements[3] / len(LOADS),
        },
        "seleccion_y_costos_internos": summarize_frozen_costs(frozen, costs),
        "detector_auxiliar_por_carga": summarize_detector(exterior, partitions, hashes),
        "insumos_sha256": hashes,
    }


def verify_against_saved(evidence, output_text, sanitized):
    if not sanitized:
        require(OUTPUT.read_text(encoding="utf-8") == output_text,
                "La evidencia guardada no coincide")
        return
    saved = json.loads(OUTPUT.read_text(encoding="utf-8"))
    calc = json.loads(output_text)
    saved_metrics = {k: v for k, v in saved.items() if k != "insumos_sha256"}
    calc_metrics = {k: v for k, v in calc.items() if k != "insumos_sha256"}
    require(saved_metrics == calc_metrics,
            "Las métricas recalculadas no coinciden con la evidencia guardada")
    saved_hashes = saved["insumos_sha256"]
    calc_hashes = calc["insumos_sha256"]
    require(set(saved_hashes) == set(calc_hashes),
            "El conjunto de insumos no coincide con la evidencia guardada")
    for key, saved_hash in saved_hashes.items():
        is_sanitized_outer_npz = (
            "/outer_eval_" in key and key.endswith("_predicciones.npz")
        )
        if not is_sanitized_outer_npz:
            require(calc_hashes[key] == saved_hash,
                    f"El SHA-256 del insumo no saneado cambió: {key}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Comparar sin reescribir el JSON")
    parser.add_argument(
        "--saneado",
        action="store_true",
        help="Modo de paquete público saneado (permite distinto SHA-256 solo en outer_eval_*_predicciones.npz)",
    )
    parser.add_argument(
        "--salida",
        type=Path,
        default=None,
        help="Ruta opcional para escribir el JSON recalculado sin sobrescribir EVIDENCIA_FASE1_CWRU.json",
    )
    args = parser.parse_args()
    sanitized = args.saneado or (ROOT / "MANIFEST_PUBLIC_SHA256.txt").is_file()
    evidence = build_evidence()
    output = json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.check:
        verify_against_saved(evidence, output, sanitized=sanitized)
    if args.salida is not None:
        args.salida.parent.mkdir(parents=True, exist_ok=True)
        args.salida.write_text(output, encoding="utf-8", newline="\n")
    elif not args.check:
        OUTPUT.write_text(output, encoding="utf-8", newline="\n")
    print(f"{OUTPUT.name}: {evidence['celdas_INT8_contrastadas_con_predicciones_NPZ']} celdas INT8 "
          f"contrastadas, {len(evidence['insumos_sha256'])} insumos identificados por SHA-256")


if __name__ == "__main__":
    main()
