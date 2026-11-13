"""Proyeccion pura del presupuesto DNN desde dos perfiles de tiempo PC.

No abre datos, entrena, selecciona modelos o mide latencia STM32. Dos perfiles
son un rango de escenarios de ingenieria, no una muestra para intervalos de
confianza ni una garantia de tiempo maximo. Los 32 representantes de costo
estan incluidos en los 288 ajustes internos, nunca se suman de nuevo.
"""
from __future__ import annotations

from math import fsum, isfinite
from numbers import Integral, Real


STAGES = (
    'imports', 'data_and_scaler', 'graph_compile', 'training_updates',
    'epoch_training_diagnostics', 'epoch_validation', 'FP32_conversion',
    'INT8_conversion', 'inference', 'artifact_IO',
)
INNER_FITS = 288
OUTER_FITS = 120
PILOT_TRAIN = 799
PILOT_VALIDATION = 175
PILOT_EPOCHS = 50
PILOT_BATCHES = 25


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _validate_result(row):
    _require(isinstance(row, dict), 'Cada resultado debe ser un diccionario')
    _require(row.get('status') == 'PASS', 'No proyectar desde un piloto fallido o incompleto')
    _require(row.get('mask') in ('F14', 'T6'), 'Solo se admiten los pilotos F14 y T6')
    expected = dict(training_windows=PILOT_TRAIN, validation_windows=PILOT_VALIDATION,
                    epochs=PILOT_EPOCHS, batches_per_epoch=PILOT_BATCHES,
                    optimizer_updates=PILOT_EPOCHS * PILOT_BATCHES)
    for key, value in expected.items():
        observed = row.get(key)
        _require(isinstance(observed, Integral) and not isinstance(observed, bool)
                 and int(observed) == value, 'Alcance inesperado en ' + key)
    timings = row.get('timing_nonoverlapping_seconds')
    _require(isinstance(timings, dict) and set(timings) == set(STAGES),
             'Se requieren exactamente diez etapas no solapadas en segundos')
    checked = {}
    for name in STAGES:
        value = timings[name]
        _require(isinstance(value, Real) and not isinstance(value, bool),
                 'Tiempo no numerico en ' + name)
        value = float(value)
        _require(isfinite(value) and value >= 0., 'Tiempo negativo o no finito en ' + name)
        checked[name] = value
    _require(fsum(checked.values()) > 0., 'Perfil temporal completamente nulo')
    return checked


def _profile(stages, epoch_count, *, outer):
    train_rows = 1205 if outer else 812
    batches = 38 if outer else 26
    epoch_ratio = epoch_count / PILOT_EPOCHS
    train_ratio = train_rows / PILOT_TRAIN
    validation_ratio = 0. if outer else 406 / PILOT_VALIDATION
    factors = dict(imports=1., data_and_scaler=train_ratio, graph_compile=1.,
        training_updates=(batches / PILOT_BATCHES) * epoch_ratio,
        epoch_training_diagnostics=train_ratio * epoch_ratio,
        epoch_validation=validation_ratio * epoch_ratio,
        FP32_conversion=1., INT8_conversion=train_ratio,
        inference=406 / PILOT_VALIDATION, artifact_IO=1.)
    projected = {name: stages[name] * factors[name] for name in STAGES}
    seconds = fsum(projected.values())
    fits = OUTER_FITS if outer else INNER_FITS
    return dict(training_windows_assumed=train_rows, final_evaluation_windows_assumed=406,
        epochs_assumed=epoch_count, batches_per_epoch=batches,
        optimizer_updates_assumed=batches * epoch_count,
        epoch_validation_windows_assumed=0 if outer else 406,
        timing_scale_factors=factors, stage_seconds_per_fit=projected,
        seconds_per_fit=seconds, maximum_fits_before_identical_reuse=fits,
        projected_stage_seconds_all_fits={name: value * fits for name, value in projected.items()},
        projected_seconds_all_fits=seconds * fits)


def project_dnn_budget(results):
    """Retorna escenarios de 25 y 50 epocas; los valores temporales son segundos.

    results contiene exactamente dos dict worker PASS, mask F14/T6, diez
    tiempos no solapados y el alcance 799/175, 50 epocas, 25 lotes y 1250
    actualizaciones. Los tiempos son duraciones PC ya observadas. No se usan
    accuracy/F1 ni resultados futuros para elegir factores o perfiles.

    Cada extremo minimo/maximo conserva un perfil completo coherente; no se
    mezclan minimos de distintas etapas como si fueran otra observacion.
    """
    _require(isinstance(results, (list, tuple)) and len(results) == 2,
             'Se requieren exactamente dos resultados piloto')
    checked = {}
    for row in results:
        timings = _validate_result(row)
        name = row['mask']
        _require(name not in checked, 'Mascara piloto duplicada')
        checked[name] = timings
    _require(set(checked) == {'F14', 'T6'}, 'Falta un perfil piloto')
    scenarios = {}
    for epochs in (25, 50):
        profiles = {}
        for name in ('F14', 'T6'):
            inner = _profile(checked[name], epochs, outer=False)
            outer = _profile(checked[name], epochs, outer=True)
            total = fsum((inner['projected_seconds_all_fits'], outer['projected_seconds_all_fits']))
            _require(isfinite(total), 'Proyeccion temporal no representable')
            profiles[name] = dict(inner=inner, outer=outer, projected_DNN_seconds=total,
                                  projected_DNN_hours=total / 3600.)
        totals = [item['projected_DNN_seconds'] for item in profiles.values()]
        scenarios['epochs_' + str(epochs)] = dict(epochs_assumed=epochs, profiles=profiles,
            range_between_two_pilot_profiles_seconds=dict(min=min(totals), max=max(totals)),
            conservative_max_seconds=max(totals), conservative_max_hours=max(totals) / 3600.,
            confidence_interval=False, guaranteed_ETA=False, WCET=False)
    observed = {name: fsum(stages.values()) for name, stages in checked.items()}
    return dict(status='DNN_PLANNING_PROJECTION_ONLY', units='seconds', platform='PC',
        scenarios=scenarios, maximum_DNN_fits_before_identical_reuse=INNER_FITS + OUTER_FITS,
        maximum_inner_DNN_fits=INNER_FITS, maximum_outer_DNN_fits=OUTER_FITS,
        maximum_FP32_plus_INT8_conversions=2 * (INNER_FITS + OUTER_FITS),
        cost_representatives_included_in_inner_fits=32, additional_cost_representative_fits=0,
        pilot_scope=dict(additional_DNN_fits=2, additional_conversions=4,
            observed_nonoverlapping_stage_seconds_by_mask=observed,
            observed_nonoverlapping_stage_seconds_total=fsum(observed.values()),
            included_in_projected_408=False, worker_limit_seconds_each=300,
            combined_worker_limit_seconds=600, limit_is_not_a_runtime_estimate=True),
        assumptions=[
            'Every projected fit includes fresh import, graph tracing, FP32 conversion and artifact I/O.',
            'Inner and outer maxima use 812/1205 training rows and 26/38 batches per epoch.',
            'Entire measured INT8 conversion scales by training rows; its fixed part was not separated.',
            'Data/scaler stage scales by training rows; this is an engineering approximation.',
            'Epoch diagnostics of training scale by training rows; inner validation scales to 406 rows.',
            'Outer fitting has no per-epoch external validation; final inference projects 406 rows.',
            'All 288 inner and 120 outer fits use each complete pilot timing profile in turn.',
            'Two pilot profiles do not provide confidence intervals or a guaranteed upper bound.',
        ],
        exclusions=['RF and SVM runtime', 'detector fitting/calibration runtime',
            'STM32 compilation/flashing/measurement', 'UART transfer or acquisition',
            'operator review time', 'unmeasured process startup/teardown outside worker stage timers'],
        other_stage_exclusions=['worker unattributed time', 'final result JSON write',
                                'process gap is added separately by the orchestrator as a scenario'],
        future_predictive_metrics_used=False, cost_selection_or_training_launched=False,
        STM32_latency_measured=False, classical_runtime_budget_pending=True,
        full_budget_frozen=False)
