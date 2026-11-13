"""Reglas deterministas V2: roles, P95 y seleccion; no ajusta estimadores.

Esquemas de filas:
  DNN: candidate, inner_validation_load, seed, status='PASS', macro_f1_int8.
  Detector: candidate, inner_validation_load, status='PASS', normal_spec,
            ir_recall y or_recall (tasas por registro agregadas por carga).
``candidates`` es una secuencia de ocho nombres en orden de catalogo.

Los margenes son decisiones de ingenieria, no pruebas de no inferioridad.
Los intervalos de costo son envolventes empiricas, no intervalos de confianza
ni cotas WCET. Este modulo nunca entrena, abre datos o habilita ejecuciones.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from fractions import Fraction
from math import fsum, isfinite
from numbers import Integral, Real
from statistics import median

import numpy as np

LOADS = (0, 1, 2, 3)
F1_LOSS_MARGIN = .01
DETECTOR_BA_LOSS_MARGIN = .01
DETECTOR_COMPONENT_LOSS_MARGIN = .02
TARGET_TOTAL_REDUCTION = .20
MAX_EPOCHS = 50
_EPS = 1e-12  # Solo frontera numerica de comparaciones, no margen predictivo.


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _integer(value, label):
    _require(isinstance(value, Integral) and not isinstance(value, (bool, np.bool_)),
             label + ' debe ser entero')
    return int(value)


def _finite_number(value, label):
    _require(isinstance(value, Real) and not isinstance(value, (bool, np.bool_)),
             label + ' debe ser numerico')
    value = float(value)
    _require(isfinite(value), label + ' debe ser finito')
    return value


def _rate(value, label):
    value = _finite_number(value, label)
    _require(0. <= value <= 1., label + ' debe estar entre cero y uno')
    return value


def component_roles(outer_load, inner_val=None):
    """Roles por carga, antes de tocar muestras o ajustar parametros.

    El registro N de la mayor carga disponible para entrenamiento se reserva
    para calibrar el detector. No se reajusta el detector despues de calibrar.
    La DNN/INT8 si utiliza todas las cargas de su entrenamiento autorizado.
    ``outer_excluded_loads`` significa excluidas de AJUSTE; en outer_refit
    esa misma carga constituye la evaluacion externa.
    """
    outer = _integer(outer_load, 'outer_load')
    _require(outer in LOADS, 'Carga externa fuera de 0..3')
    development = [load for load in LOADS if load != outer]
    if inner_val is not None:
        inner = _integer(inner_val, 'inner_val')
        _require(inner in development, 'Validacion interna debe pertenecer a desarrollo')
        training = [load for load in development if load != inner]
        evaluation = [inner]
    else:
        inner = None
        training, evaluation = development.copy(), [outer]
    calibration = [max(training)]
    fitting = [load for load in training if load not in calibration]
    _require(set(fitting).isdisjoint(calibration) and
             set(training).isdisjoint(evaluation) and
             set(fitting) | set(calibration) == set(training), 'Roles solapados')
    return dict(outer_load_hp=outer, inner_validation_load_hp=inner,
                development_loads=development, training_loads=training,
                dnn_fit_loads=training.copy(), int8_calibration_loads=training.copy(),
                detector_fit_normal_loads=fitting,
                detector_calibration_normal_loads=calibration,
                evaluation_loads=evaluation, outer_excluded_loads=[outer],
                detector_refit_after_calibration=False)


def weighted_p95(scores, record_ids):
    """P95 observado float32 con igual masa por registro y CDF racional.

    F(t) = (1/R) sum_r count(s_r <= t)/n_r. Se retorna el primer score
    observado cuya CDF alcanza 19/20, sin interpolacion. La decision es
    ``score > threshold``: los empates se aceptan. No garantiza 95 % futuro.
    """
    try:
        raw_scores, raw_ids = list(scores), list(record_ids)
    except TypeError as error:
        raise ValueError('Scores y registros deben ser secuencias') from error
    _require(raw_scores and len(raw_scores) == len(raw_ids), 'Calibracion vacia o longitudes distintas')
    numbers = [_finite_number(value, 'score') for value in raw_scores]
    _require(all(value >= 0. for value in numbers), 'Distancia negativa')
    with np.errstate(over='ignore', invalid='ignore'):
        values = np.asarray(numbers, dtype=np.float32)
    _require(values.ndim == 1 and np.isfinite(values).all(), 'Score no representable como float32 finito')
    ids = []
    for value in raw_ids:
        _require(isinstance(value, (str, Integral)) and not isinstance(value, (bool, np.bool_)),
                 'ID de registro debe ser texto o entero')
        value = str(value)
        _require(value and value.strip() == value, 'ID de registro vacio o malformado')
        ids.append(value)
    counts = Counter(ids)
    mass = defaultdict(Fraction)
    for value, record in zip(values, ids):
        mass[float(value)] += Fraction(1, len(counts) * counts[record])
    cumulative = Fraction(0, 1)
    for value in sorted(mass):
        cumulative += mass[value]
        if cumulative >= Fraction(19, 20):
            return np.float32(value)
    raise AssertionError('La CDF ponderada no alcanzo uno')


def final_epoch_count(best_epochs):
    """Mediana de nueve checkpoints internos 1..50, sin consulta externa."""
    values = list(best_epochs)
    _require(len(values) == 9, 'Se requieren nueve mejores epocas internas')
    values = [_integer(value, 'best_epoch') for value in values]
    _require(all(1 <= value <= MAX_EPOCHS for value in values), 'Epoca fuera de 1..50')
    return int(median(values))


def _evaluate(dnn_rows, detector_rows, candidates, inner_loads, seeds):
    candidates, inner_loads, seeds = list(candidates), list(inner_loads), list(seeds)
    _require(all(isinstance(name, str) and name for name in candidates), 'Nombres de candidatas malformados')
    _require(len(candidates) == 8 and len(set(candidates)) == 8 and 'F14' in candidates,
             'Ocho candidatas unicas con F14 requeridas')
    inner_loads = [_integer(load, 'inner_load') for load in inner_loads]
    seeds = [_integer(seed, 'seed') for seed in seeds]
    _require(len(inner_loads) == 3 and len(set(inner_loads)) == 3 and set(inner_loads) <= set(LOADS),
             'Se requieren tres cargas internas distintas')
    _require(len(seeds) == 3 and len(set(seeds)) == 3 and all(seed >= 0 for seed in seeds),
             'Se requieren tres semillas internas distintas')
    expected = {(name, load, seed) for name in candidates for load in inner_loads for seed in seeds}
    scores = {}
    for row in dnn_rows:
        _require(isinstance(row, dict), 'Fila DNN malformada')
        try:
            key = (row['candidate'], _integer(row['inner_validation_load'], 'inner_validation_load'),
                   _integer(row['seed'], 'seed'))
            _require(key in expected and key not in scores, 'Fila DNN ajena o duplicada')
            _require(row['status'] == 'PASS', 'Ajuste DNN no aprobado: no omitirlo del promedio')
            scores[key] = _rate(row['macro_f1_int8'], 'macro_f1_int8')
        except KeyError as error:
            raise ValueError('Campo DNN ausente: ' + str(error)) from error
    _require(set(scores) == expected, 'Matriz DNN incompleta: ocho por tres por tres')
    expected_det = {(name, load) for name in candidates for load in inner_loads}
    detector = {}
    for row in detector_rows:
        _require(isinstance(row, dict), 'Fila detector malformada')
        try:
            key = (row['candidate'], _integer(row['inner_validation_load'], 'inner_validation_load'))
            _require(key in expected_det and key not in detector, 'Fila detector ajena o duplicada')
            _require(row['status'] == 'PASS', 'Detector no aprobado: no omitirlo del promedio')
            metrics = {name: _rate(row[name], name) for name in ('normal_spec', 'ir_recall', 'or_recall')}
            metrics['fault_sensitivity'] = (metrics['ir_recall'] + metrics['or_recall']) / 2
            metrics['balanced_accuracy'] = (metrics['normal_spec'] + metrics['fault_sensitivity']) / 2
            detector[key] = metrics
        except KeyError as error:
            raise ValueError('Campo detector ausente: ' + str(error)) from error
    _require(set(detector) == expected_det, 'Matriz detector incompleta: ocho por tres')
    summary = {}
    for name in candidates:
        per_load = {load: fsum(scores[name, load, seed] for seed in sorted(seeds)) / len(seeds)
                    for load in sorted(inner_loads)}
        metrics = {metric: fsum(detector[name, load][metric] for load in sorted(inner_loads)) / len(inner_loads)
                   for metric in ('normal_spec', 'ir_recall', 'or_recall', 'fault_sensitivity', 'balanced_accuracy')}
        summary[name] = dict(mean_macro_f1_int8=fsum(per_load.values()) / len(per_load),
                             worst_load_mean_macro_f1_int8=min(per_load.values()),
                             per_load_mean_macro_f1_int8=per_load, detector_mean=metrics)
    reference = summary['F14']['detector_mean']
    admissible = []
    for name in candidates:
        metrics = summary[name]['detector_mean']
        passed = metrics['balanced_accuracy'] + _EPS >= reference['balanced_accuracy'] - DETECTOR_BA_LOSS_MARGIN
        passed = passed and all(metrics[key] + _EPS >= reference[key] - DETECTOR_COMPONENT_LOSS_MARGIN
                                for key in ('normal_spec', 'ir_recall', 'or_recall'))
        summary[name]['detector_admissible'] = bool(passed)
        if passed:
            admissible.append(name)
    _require('F14' in admissible, 'La referencia debe ser admisible frente a si misma')
    return candidates, inner_loads, seeds, summary, admissible


def _best_performance(names, candidates, summary):
    # No se utilizan dimension, latencia o probabilidades promediadas.
    # Una envolvente respecto al maximo evita diferencias de redondeo y una
    # comparacion por pares no transitiva. EPS no es un margen predictivo.
    best_mean = max(summary[name]['mean_macro_f1_int8'] for name in names)
    tied_mean = [name for name in names
                 if summary[name]['mean_macro_f1_int8'] >= best_mean - _EPS]
    best_worst = max(summary[name]['worst_load_mean_macro_f1_int8'] for name in tied_mean)
    tied_worst = [name for name in tied_mean
                  if summary[name]['worst_load_mean_macro_f1_int8'] >= best_worst - _EPS]
    return min(tied_worst, key=candidates.index)


def select_performance(dnn_rows, detector_rows, candidates, inner_loads, seeds):
    """Mejor media de nueve macro-F1 INT8 entre detectores admisibles."""
    candidates, _, _, summary, admissible = _evaluate(dnn_rows, detector_rows, candidates, inner_loads, seeds)
    selected = _best_performance(admissible, candidates, summary)
    return dict(selected_candidate=selected, procedure='performance_only',
                detector_admissible_candidates=admissible, candidate_summaries=summary,
                mean_macro_f1_int8=summary[selected]['mean_macro_f1_int8'],
                worst_load_mean_macro_f1_int8=summary[selected]['worst_load_mean_macro_f1_int8'],
                latency_or_dimension_used_for_performance_tie=False, probability_ensemble_used=False)


def select_cost(dnn_rows, detector_rows, candidates, inner_loads, seeds, costs, *, outer_load):
    """Seleccion por costo TOTAL validado, con guardias predictivas y de costo.

    ``costs`` mapea cada candidata a status='VALIDATED_TOTAL_COST', mean_ms,
    lower_ms, upper_ms, outer_load y benchmark_loads. Debe contener las ocho.
    Si se declara cost_scope, solo se admite 'full_pipeline'; los proxies se
    rechazan. El representante fija inner_validation=min(desarrollo), seed=11;
    las cargas del benchmark deben ser exactamente sus dos cargas de ajuste,
    excluyendo tanto la externa como esa validacion interna.

    Se exige U(S)<=0.8*L(F14). Entre intervalos que contienen el menor U se
    desempata por rendimiento/carga peor/catalogo. Si nadie cumple, se conserva
    F14 y se registra que no se alcanzo el objetivo; nunca se rellenan costos
    desconocidos con cero ni se omite un candidato sin medir.
    """
    candidates, inner_loads, _, summary, admissible = _evaluate(
        dnn_rows, detector_rows, candidates, inner_loads, seeds)
    outer = _integer(outer_load, 'outer_load')
    _require(outer in LOADS and set(inner_loads) == set(LOADS) - {outer}, 'Pliegue externo/internos incompatible')
    benchmark_training_loads = set(inner_loads) - {min(inner_loads)}
    _require(isinstance(costs, dict) and set(costs) == set(candidates), 'Se requieren costos totales de las ocho candidatas')
    checked = {}
    for name in candidates:
        row = costs[name]
        _require(isinstance(row, dict), 'Costo desconocido o malformado: ' + name)
        try:
            _require(row['status'] == 'VALIDATED_TOTAL_COST', 'Costo total no validado: ' + name)
            _require(row.get('cost_scope', 'full_pipeline') == 'full_pipeline'
                     and row.get('is_proxy', False) is False, 'Un proxy no es costo total validado')
            _require(_integer(row['outer_load'], 'cost.outer_load') == outer, 'Costo de otro pliegue externo')
            benchmark = [_integer(load, 'benchmark_load') for load in row['benchmark_loads']]
            _require(benchmark and len(set(benchmark)) == len(benchmark)
                     and set(benchmark) == benchmark_training_loads and outer not in benchmark,
                     'Benchmark debe usar solo las dos cargas de ajuste del representante')
            mean = _finite_number(row['mean_ms'], 'mean_ms')
            lower = _finite_number(row['lower_ms'], 'lower_ms')
            upper = _finite_number(row['upper_ms'], 'upper_ms')
            _require(0. < lower <= mean <= upper, 'Envolvente temporal invalida')
            checked[name] = dict(mean_ms=mean, lower_ms=lower, upper_ms=upper,
                                 benchmark_loads=benchmark)
        except KeyError as error:
            raise ValueError('Campo de costo ausente: ' + str(error)) from error
    performance = _best_performance(admissible, candidates, summary)
    threshold = max(summary['F14']['mean_macro_f1_int8'], summary[performance]['mean_macro_f1_int8']) - F1_LOSS_MARGIN
    predictive = [name for name in admissible if summary[name]['mean_macro_f1_int8'] + _EPS >= threshold]
    target = [name for name in predictive if checked[name]['upper_ms'] <=
              (1. - TARGET_TOTAL_REDUCTION) * checked['F14']['lower_ms'] + _EPS]
    common = dict(procedure='validated_total_cost', performance_selected_candidate=performance,
                  detector_admissible_candidates=admissible, predictive_eligible_candidates=predictive,
                  target_eligible_candidates=target, macro_f1_threshold=threshold,
                  candidate_summaries=summary, validated_total_costs=checked,
                  interval_interpretation='empirical_envelope_not_confidence_interval_or_WCET',
                  probability_ensemble_used=False)
    if not target:
        return dict(**common, selected_candidate='F14', goal_met=False,
                    reason='no_candidate_meets_total_cost_target', overlapping_cost_shortlist=[],
                    selected_satisfies_predictive_margin='F14' in predictive)
    minimum_upper = min(checked[name]['upper_ms'] for name in target)
    shortlist = [name for name in target if checked[name]['lower_ms'] <= minimum_upper + _EPS]
    selected = _best_performance(shortlist, candidates, summary)
    return dict(**common, selected_candidate=selected, goal_met=True,
                reason='total_cost_target_met', overlapping_cost_shortlist=shortlist,
                selected_satisfies_predictive_margin=True)
