"""Proyeccion pura RF/SVM/detector: escenarios PC, sin datos o entrenamiento.

Los factores por filas son supuestos de ingenieria, no complejidades o cotas
temporales garantizadas. El piloto no selecciona modelos por sus predicciones.
"""
from __future__ import annotations

from math import fsum, isfinite, log
from numbers import Integral, Real

CLASSICAL_STAGES = ('imports', 'data_and_scaler', 'fit', 'inference', 'artifact_IO')
DETECTOR_OVERHEAD_STAGES = ('imports', 'data_and_scaler', 'artifact_IO')
DETECTOR_CANDIDATES = ('F14', 'T6', 'S8', 'NM10', 'NE12', 'DE7', 'FE7')


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _integer(value, expected, label):
    _require(isinstance(value, Integral) and not isinstance(value, bool)
             and int(value) == expected, 'Alcance inesperado: ' + label)


def _time(value, label):
    _require(isinstance(value, Real) and not isinstance(value, bool), 'Tiempo no numerico: ' + label)
    value = float(value)
    _require(isfinite(value) and value >= 0., 'Tiempo negativo/no finito: ' + label)
    return value


def _timings(value, stages):
    _require(isinstance(value, dict) and set(value) == set(stages), 'Esquema de etapas incorrecto')
    checked = {name: _time(value[name], name) for name in stages}
    _require(fsum(checked.values()) > 0., 'Perfil temporal completamente nulo')
    return checked


def _process_gap(row, stage_sum, *, required=False):
    observed = row.get('process_elapsed_seconds')
    if observed is None:
        _require(not required, 'Falta tiempo de proceso del detector')
        return None
    observed = _time(observed, 'process_elapsed_seconds')
    _require(observed + 1e-9 >= stage_sum, 'Etapas mayores que el tiempo del proceso')
    return max(0., observed - stage_sum)


def _classical_rows(results):
    _require(isinstance(results, (list, tuple)) and len(results) == 16,
             'Se requieren ocho configuraciones RF y ocho SVM')
    checked, identifiers = {}, set()
    for row in results:
        _require(isinstance(row, dict) and row.get('status') == 'PASS', 'Resultado clasico no aprobado')
        family = row.get('family')
        config = row.get('config_index')
        _require(family in ('RF', 'SVM') and isinstance(config, Integral)
                 and not isinstance(config, bool) and 0 <= config <= 7, 'Familia/configuracion incorrecta')
        key = (family, int(config))
        identifier = row.get('job_id')
        _require(key not in checked and isinstance(identifier, str) and identifier
                 and identifier not in identifiers, 'Configuracion o identificador duplicados')
        _integer(row.get('training_windows'), 799, 'training_windows')
        _integer(row.get('validation_windows'), 175, 'validation_windows')
        times = _timings(row.get('timing_nonoverlapping_seconds'), CLASSICAL_STAGES)
        checked[key] = dict(job_id=identifier, stages=times, process_gap=_process_gap(row, fsum(times.values())))
        identifiers.add(identifier)
    _require(set(checked) == {(family, i) for family in ('RF', 'SVM') for i in range(8)},
             'Grilla clasica incompleta')
    return checked


def _classical_profile(stages, family, *, outer):
    rows = 1205 if outer else 812
    row_ratio = rows / 799.
    fit_ratio = (rows * log(rows)) / (799. * log(799)) if family == 'RF' else row_ratio ** 2
    inference_ratio=(406 / 175.) * (row_ratio if family=='SVM' else 1.)
    factors = dict(imports=1., data_and_scaler=row_ratio, fit=fit_ratio,
                   inference=inference_ratio, artifact_IO=1.)
    values = {name: stages[name] * factors[name] for name in CLASSICAL_STAGES}
    return dict(training_windows_assumed=rows, final_inference_windows_assumed=406,
                scale_factors=factors, stage_seconds_per_fit=values,
                seconds_per_fit=fsum(values.values()))


def _project_family(checked, family):
    inner_per_config, outer_fits = (36, 40) if family == 'RF' else (12, 4)
    profiles = {}
    for config in range(8):
        source = checked[family, config]
        profiles[str(config)] = dict(source_job_id=source['job_id'],
            inner=_classical_profile(source['stages'], family, outer=False),
            outer=_classical_profile(source['stages'], family, outer=True),
            projected_inner_fits=inner_per_config,
            observed_process_gap_seconds=source['process_gap'])
    inner = fsum(p['inner']['seconds_per_fit'] * inner_per_config for p in profiles.values())
    outer_index = max(profiles, key=lambda i: profiles[i]['outer']['seconds_per_fit'])
    outer = profiles[outer_index]['outer']['seconds_per_fit'] * outer_fits
    gaps = [p['observed_process_gap_seconds'] for p in profiles.values()]
    if all(value is not None for value in gaps):
        gap_scenario = fsum(value * inner_per_config for value in gaps) + max(gaps) * outer_fits
    else:
        gap_scenario = None
    stage_total = fsum((inner, outer))
    return dict(family=family, configuration_profiles=profiles,
        projected_inner_fits=8 * inner_per_config, projected_outer_fits=outer_fits,
        projected_total_fits=8 * inner_per_config + outer_fits,
        projected_inner_stage_seconds=inner, projected_outer_stage_seconds=outer,
        outer_sensitivity_source_config_index=int(outer_index),
        outer_config_rule='largest projected complete configuration profile, not stagewise mixed minima/maxima',
        projected_stage_seconds=stage_total, projected_stage_hours=stage_total / 3600.,
        observed_process_gap_scenario_seconds=gap_scenario,
        projected_with_observed_process_gap_seconds=None if gap_scenario is None else stage_total + gap_scenario,
        fit_scaling_assumption='n_log_n' if family == 'RF' else 'quadratic_rows',
        scaling_is_engineering_sensitivity_not_guaranteed_bound=True)


def _project_detector(result):
    _require(isinstance(result, dict) and result.get('status') == 'PASS', 'Detector no aprobado')
    for key, expected in (('fit_windows', 58), ('calibration_windows', 58),
                          ('validation_windows', 175), ('unique_representations', 7)):
        _integer(result.get(key), expected, key)
    overhead = _timings(result.get('timing_nonoverlapping_seconds'), DETECTOR_OVERHEAD_STAGES)
    rows = result.get('rows')
    _require(isinstance(rows, list) and len(rows) == 21, 'Detector requiere siete representaciones por tres repeticiones')
    seen, checked = set(), []
    for row in rows:
        _require(isinstance(row, dict) and row.get('status') == 'PASS', 'Repeticion detector no aprobada')
        candidate, rep = row.get('candidate'), row.get('rep')
        _require(candidate in DETECTOR_CANDIDATES and isinstance(rep, Integral)
                 and not isinstance(rep, bool) and 0 <= rep <= 2, 'Representacion/repeticion detector incorrecta')
        key = candidate, int(rep)
        _require(key not in seen, 'Repeticion detector duplicada')
        seen.add(key)
        checked.append(dict(candidate=candidate, rep=int(rep),
            fit_calibration_seconds=_time(row.get('fit_calibration_seconds'), 'fit_calibration_seconds'),
            inference_seconds=_time(row.get('inference_seconds'), 'inference_seconds')))
    _require(seen == {(name, rep) for name in DETECTOR_CANDIDATES for rep in range(3)},
             'Repeticiones detector incompletas')
    grouped_total = fsum(overhead.values()) + fsum(r['fit_calibration_seconds'] + r['inference_seconds'] for r in checked)
    gap = _process_gap(result, grouped_total, required=True)
    max_fit = max(r['fit_calibration_seconds'] for r in checked)
    max_inference = max(r['inference_seconds'] for r in checked)
    fit_inner = max_fit
    fit_outer = 2. * max_fit
    inference = max_inference * (406 / 175.)
    # Copiar TODO el overhead del worker agrupado a cada fit es un escenario
    # conservador; no dividir por siete o 21 ni multiplicar ajustes por semillas.
    overhead_per_fit = fsum(overhead.values())
    inner_per_fit = overhead_per_fit + fit_inner + inference
    outer_per_fit = overhead_per_fit + fit_outer + inference
    stage_total = 96 * inner_per_fit + 12 * outer_per_fit
    return dict(projected_inner_fits_before_cache=96, projected_outer_fits_before_cache=12,
        projected_total_fits_before_cache=108, repeated_per_DNN_seed=False,
        pilot_unique_representations=7, pilot_fit_calibration_repetitions=21,
        candidate_aliases={'NF12': 'F14'}, cache_savings_claimed=False,
        grouped_overhead_stage_seconds=overhead, grouped_overhead_copied_per_fit_seconds=overhead_per_fit,
        overhead_divided_by_representations_or_repetitions=False,
        maximum_observed_fit_calibration_seconds=max_fit,
        maximum_observed_inference_seconds=max_inference,
        inner_fit_calibration_scale_factor=1., outer_fit_calibration_scale_factor=2.,
        outer_fit_windows_assumed=116, calibration_windows_assumed=58,
        inference_windows_assumed=406, inference_scale_factor=406 / 175.,
        inner_seconds_per_fit=inner_per_fit, outer_seconds_per_fit=outer_per_fit,
        projected_stage_seconds=stage_total, projected_stage_hours=stage_total / 3600.,
        observed_grouped_process_gap_seconds=gap,
        observed_process_gap_scenario_seconds=108 * gap,
        projected_with_observed_process_gap_seconds=stage_total + 108 * gap,
        max_fit_and_inference_are_empirical_component_envelopes_not_one_observation=True,
        scaling_is_engineering_sensitivity_not_guaranteed_bound=True)


def project_classical_budget(results, detector_result):
    """Retorna escenarios PC separados y agregados; no consulta metricas.

    ``results``: 16 PASS (RF/SVM por config_index 0..7), ventanas 799/175,
    cinco etapas no solapadas y process_elapsed_seconds opcional.
    ``detector_result``: PASS, ventanas 58/58/175, siete representaciones y
    21 filas (candidate, rep 0..2, PASS, fit_calibration_seconds,
    inference_seconds), tres etapas de overhead y tiempo del proceso agrupado.
    """
    checked = _classical_rows(results)
    rf, svm = _project_family(checked, 'RF'), _project_family(checked, 'SVM')
    detector = _project_detector(detector_result)
    stage_total = fsum(item['projected_stage_seconds'] for item in (rf, svm, detector))
    gaps = [item['observed_process_gap_scenario_seconds'] for item in (rf, svm, detector)]
    gap_total = fsum(gaps) if all(value is not None for value in gaps) else None
    _require(isfinite(stage_total) and (gap_total is None or isfinite(gap_total)), 'Proyeccion no finita')
    with_gap = None if gap_total is None else stage_total + gap_total
    return dict(status='CLASSICAL_PLANNING_PROJECTION_ONLY', units='seconds', platform='PC',
        RF=rf, SVM=svm, detector=detector,
        aggregate=dict(projected_stage_seconds=stage_total, projected_stage_hours=stage_total / 3600.,
            observed_process_gap_scenario_seconds=gap_total,
            projected_with_observed_process_gap_seconds=with_gap,
            projected_with_observed_process_gap_hours=None if with_gap is None else with_gap / 3600.),
        pilot_scope=dict(additional_RF_fits=8, additional_SVM_fits=8,
            additional_detector_fit_calibration_calls=21, grouped_detector_worker_processes=1,
            repeated_measurements_are_not_independent_acquisitions=True,
            pilot_models_reused_as_confirmatory_results=False),
        assumptions=[
            'RF fit scales by n log(n); SVM fit scales quadratically with rows, solely as engineering scenarios.',
            'Classical data/scaler scales by training rows, inference to 406 rows; SVM inference also assumes support-vector work scales with training rows. These are sensitivities, not bounds.',
            'Imports and artifact I/O stay fixed per fit; RF depth changes and SVM support-vector changes remain data dependent.',
            'Every inner grid configuration retains its own complete timing profile; the outer unknown selection uses the largest projected profile.',
            'Seed and load differences can change runtime; one timing per configuration cannot bound those changes.',
            'The grouped detector overhead and its process gap are conservatively copied to each of 108 naive fits.',
            'Detector fit/calibration doubles as a combined block for outer fitting; calibration itself remains at most 58 rows.',
            'Detector fit/calibration and inference maxima can come from different repetitions; this is a component-envelope scenario.',
            'Actual caching may reduce work; no unmeasured saving is credited.',
        ],
        DNN_included=False, DNN_or_classical_training_launched=False, future_predictive_metrics_used=False,
        STM32_latency_measured=False, confidence_interval=False, guaranteed_ETA=False, WCET=False,
        cold_import_strategy_pending=True, cold_import_sensitivity_available=False,
        full_budget_frozen=False)
