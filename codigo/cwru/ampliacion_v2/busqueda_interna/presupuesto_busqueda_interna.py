"""Presupuesto DNN interno desde los32 perfiles recientes; sin TF ni ajustes.

build_budget(n_new_fits) es importable por el preparador. El modo --freeze
crea JSON/Markdown nuevos una sola vez, sin ejecutar la campaña.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[4]
PREP = ROOT/'articulo/ampliacion_cwru/metodologia/busqueda_interna_v2_2026-10-26'
REP = ROOT/'resultados/cwru/ampliacion_v2/representantes_v2_2026-10-25/20261025T171847_957295Z'
REP_CLOSURE = ROOT/'articulo/ampliacion_cwru/metodologia/cierre_representantes_v2_2026-10-25'
OLD_BUDGET = ROOT/'articulo/ampliacion_cwru/metodologia/cierre_piloto_clasicos_2026-10-25/PRESUPUESTO_CPU_PRELIMINAR.json'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def describe(values):
    return dict(count=len(values), min=min(values), mean=statistics.mean(values),
                median=statistics.median(values), max=max(values), sum=sum(values))


def measured_profiles():
    seal = REP_CLOSURE/'MANIFEST_CIERRE_SHA256.json'
    digest, declared = seal.with_suffix('.sha256').read_text(encoding='ascii').split()
    require(declared == seal.name and sha(seal) == digest, 'Sello de perfiles distinto')
    inherited = read(seal)
    classical_seal = OLD_BUDGET.parent/'MANIFEST_CIERRE_CLASICOS_SHA256.json'
    classical_digest, classical_declared = classical_seal.with_suffix('.sha256').read_text(encoding='ascii').split()
    require(classical_declared == classical_seal.name and sha(classical_seal) == classical_digest,
            'Sello del presupuesto clasico distinto')
    classical_entries = read(classical_seal)
    results = sorted(REP.glob('cost_*/resultado.json'))
    statuses = sorted((REP/'lotes').glob('*/ESTADO_*.json'))
    require(len(results) == len(statuses) == 32, 'Se esperan32perfiles completos')
    paths = results+statuses+[REP_CLOSURE/'CIERRE_REPRESENTANTES.json',OLD_BUDGET]
    # El presupuesto usa estas fuentes concretas del cierre; la reutilizacion
    # de ajustes completos es una comprobacion distinta del preparador.
    source_sha = {}
    for path in paths:
        relative = path.relative_to(ROOT).as_posix()
        current = sha(path)
        declared_sha = classical_entries.get(relative) if path == OLD_BUDGET else inherited.get(relative)
        require(declared_sha == current,
                'Fuente de presupuesto no corresponde al cierre: '+relative)
        source_sha[relative] = current
    source_sha[seal.relative_to(ROOT).as_posix()] = digest
    source_sha[seal.with_suffix('.sha256').relative_to(ROOT).as_posix()] = sha(seal.with_suffix('.sha256'))
    source_sha[classical_seal.relative_to(ROOT).as_posix()] = classical_digest
    source_sha[classical_seal.with_suffix('.sha256').relative_to(ROOT).as_posix()] = sha(classical_seal.with_suffix('.sha256'))
    rows = [read(p) for p in results]
    states = {row['job_id']:row for row in (read(p) for p in statuses)}
    require(len(states) == 32 and all(row['status'] == 'PASS' and row['seed'] == 11
        and row['job_id'] in states and states[row['job_id']]['status'] == 'PASS'
        and states[row['job_id']]['return_code'] == 0 for row in rows), 'Perfiles fallidos o duplicados')
    versions = rows[0]['versions']
    require(all(row['versions'] == versions and 1 <= row['epochs_executed'] <= 50
        and row['optimizer_updates'] == row['epochs_executed']*math.ceil(row['training_windows']/32)
        and row['weights_restored'] for row in rows), 'Versiones/epocas/pesos incompatibles')
    controller = [states[row['job_id']]['worker_elapsed_seconds'] for row in rows]
    worker = [row['elapsed_worker_seconds'] for row in rows]
    require(all(a >= b > 0 for a,b in zip(controller,worker)), 'Tiempo de proceso inconsistente')
    stages = {name:describe([row['timings_seconds'][name] for row in rows])
              for name in rows[0]['timings_seconds']}
    per_update = max(row['timings_seconds']['fit_including_checkpoint_IO']/row['optimizer_updates'] for row in rows)
    max_training_windows = max(row['training_windows'] for row in rows)
    require(max_training_windows == 812, 'Tamano maximo representativo distinto')
    max_updates = math.ceil(max_training_windows/32)*50
    fit50 = per_update*max_updates
    process_gap = max(a-b for a,b in zip(controller,worker))
    stress_seconds = sum(row['max'] for name,row in stages.items()
        if name != 'fit_including_checkpoint_IO')+fit50+process_gap
    return dict(representative_count=32, controller_process_seconds=describe(controller),
        internal_worker_seconds=describe(worker), stage_seconds=stages,
        epochs=dict(min=min(r['epochs_executed'] for r in rows),
            mean=statistics.mean(r['epochs_executed'] for r in rows),max=max(r['epochs_executed'] for r in rows),
            stopped_before50=sum(r['epochs_executed'] < 50 for r in rows),cap50=sum(r['epochs_executed'] == 50 for r in rows)),
        all50_epoch_stress=dict(max_training_windows=812,max_optimizer_updates=max_updates,
            worst_observed_fit_seconds_per_update=per_update, projected_fit_seconds=fit50,
            maximum_observed_controller_worker_gap=process_gap,
            component_envelope_seconds_per_fit=stress_seconds,
            interpretation='Engineering sensitivity; stagewise maxima from different jobs and linear extrapolation of short fits, not one measured profile or a bound'),
        source_sha256=source_sha,tested_versions=versions)


def build_budget(n_new_fits=240):
    require(isinstance(n_new_fits,int) and not isinstance(n_new_fits,bool) and 0 <= n_new_fits <= 288,
            'n_new_fits debe ser entero entre0y288')
    observed = measured_profiles()
    ctl = observed['controller_process_seconds']
    stress = observed['all50_epoch_stress']['component_envelope_seconds_per_fit']
    old = read(OLD_BUDGET)
    old_50 = old['scenarios']['epochs_50']
    old_warm = old_50['profiles']['F14']['with_observed_process_gap_seconds']
    old_cold = old_50['all_DNN_imports_like_initial_cold_sensitivity_seconds']
    cold_extra_per_DNN = (old_cold-old_warm)/old['total_maximum_fits_before_reuse']['DNN']
    return dict(status='BUDGET_FROZEN_FOR_INNER_DNN_ONLY', date_local='2026-10-26',timezone='America/Lima',
        scope='Remaining inner DNN fits only; RF/SVM, outer evaluation, new detector fits and hardware excluded',
        planned_inner_DNN_total=288, reused_DNN_fits=288-n_new_fits, n_new_fits=n_new_fits,
        expected_reuse_sources=dict(cost_representatives=32, supplementary_P0=16,
            complete_signatures_and_manifests_must_be_verified_by_preparer=True),
        epochs_policy=dict(maximum=50,patience=10,min_delta=1e-12,restore_best_weights=True, fixed50=False),
        conversions_per_new_fit=2, maximum_new_conversions=2*n_new_fits,
        limits=dict(worker_seconds=300,campaign_worker_seconds=7200,max_batch_seconds=1800,
            max_new_jobs=16, maximum_new_fits_total=n_new_fits,default_first_batch_seconds=600),
        first_user_gate=dict(max_new_jobs=1,max_seconds=600,audit_before_next_batch=True),
        recommended_following_batch=dict(max_new_jobs=16,max_seconds=1800,
            forecast_mean_process_minutes=16*ctl['mean']/60,
            forecast_component_stress_minutes=16*stress/60),
        frontier_rule='Do not start another worker unless both batch and cumulative campaign budgets retain300s; stop normally at a job boundary, do not truncate a fit to use leftover budget.',
        tested_versions=observed['tested_versions'], measured_reference=observed,
        forecast=dict(new_fits=n_new_fits, from_observed_mean_seconds=n_new_fits*ctl['mean'],
            from_observed_mean_minutes=n_new_fits*ctl['mean']/60,
            from_observed_max_process_seconds=n_new_fits*ctl['max'],
            from_observed_max_process_minutes=n_new_fits*ctl['max']/60,
            component_max_all50_epoch_stress_seconds=n_new_fits*stress,
            component_max_all50_epoch_stress_minutes=n_new_fits*stress/60,
            user_total_expectation_minutes=[60,120] if n_new_fits else [0,0],
            historical_cold_import_sensitivity_seconds=n_new_fits*(ctl['mean']+cold_extra_per_DNN),
            cold_import_extra_seconds_per_fit_from_prior_projection=cold_extra_per_DNN,
            cold_sensitivity_is_not_expected_runtime=True,
            observational_range_or_stress_is_not_a_confidence_interval_or_guarantee=True),
        prior_3h26_scope=dict(DNN=408,RF=328,SVM=100,detector_maximum=108,
            hours=old_50['profiles']['F14']['with_observed_process_gap_hours'],
            not_the_remaining_inner_DNN_scope=True),
        budget_interpretation=dict(campaign_seconds_are_cumulative_worker_process_time=True,
            batch_seconds_scope='Wall-clock batch budget enforced by runner; shared preparation/audit/import checks may add time outside workers.',
            timeout_termination_grace_is_additional=True,
            restarting_a_batch_does_not_reset_consumed_campaign_worker_budget=True,
            hitting_budget_does_not_guarantee_all_fits_completed=True,
            limits_may_not_be_raised_automatically=True),
        environment=dict(CPU_only=True,CPU_threads=1,sequential_workers=1,first_seed_order_fixed=True),
        training_started=False,new_training_fits=0,new_TFLite_invocations=0,hardware_connected=False,
        old_artifacts_modified=False, source_sha256=sha(__file__),
        qualifications=['The32 timing profiles are recent measured processes; only six reached50epochs.',
            'Future seeds and folds can change epochs, export time and process startup.',
            'Observed component maxima are mixed engineering projections, not measured full future runs.',
            'Cold import sensitivity from the older projection can exceed the2h worker budget; the runner must stop rather than silently extend it.',
            'RF/SVM pilot has already completed and is not repeated or included in this DNN budget.',
            'PC training duration is not STM32 latency; manuscript and frozen scientific decisions remain unchanged.'])


def write_budget(n_new_fits):
    target, document = PREP/'PRESUPUESTO_Y_LIMITES.json',PREP/'PRESUPUESTO_Y_LIMITES.md'
    require(not target.exists() and not document.exists(), 'No sobrescribir presupuesto congelado')
    budget = build_budget(n_new_fits)
    PREP.mkdir(parents=True,exist_ok=True)
    with target.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(budget,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
    f = budget['forecast']; profiles = budget['measured_reference']; ctl = profiles['controller_process_seconds']
    text = f'''# Presupuesto congelado: ajustes DNN internos restantes

Este presupuesto prepara {n_new_fits} ajustes nuevos de los 288 DNN internos. Se prevé
reutilizar {288-n_new_fits} resultados completos por firma, incluidos 32 representantes
y 16 P0 del suplemento. La reutilización se valida por separado antes de ejecutar.
No se incluyen RF/SVM, reajustes externos, nuevos ajustes del detector ni hardware.
El piloto clásico ya está cerrado y no se repite.

## Evidencia temporal y previsión

Los 32 procesos recientes aprobados sumaron {ctl['sum']:.3f} s: media
{ctl['mean']:.3f} s, mediana {ctl['median']:.3f} s y máximo {ctl['max']:.3f} s por ajuste,
incluyendo importación, entrenamiento, conversiones y cierre del proceso.
Las épocas fueron 11–50 (media {profiles['epochs']['mean']:.5f}); solo seis llegaron
a 50. Se conservan máximo 50, paciencia 10, min_delta=1e-12 y restauración de
mejores pesos. No son 50 épocas fijas.

| Proyección para los {n_new_fits} ajustes | Minutos |
|---|---:|
| Media de los procesos medidos × cantidad | {f['from_observed_mean_minutes']:.2f} |
| Máximo de proceso observado × cantidad | {f['from_observed_max_process_minutes']:.2f} |
| Sensibilidad conservadora de componentes con 50 épocas | {f['component_max_all50_epoch_stress_minutes']:.2f} |

Reservar aproximadamente **1–2 horas de trabajo restante**, repartido en lotes.
La sensibilidad a 50 épocas combina máximos de etapas de distintos procesos y
extrapola la peor razón observada de tiempo de ajuste por actualización a
1300 actualizaciones. Es una proyección de ingeniería, no una duración medida,
intervalo de confianza o garantía. Las comprobaciones previas y auditorías
pueden añadir tiempo. El arranque frío puede alargar los procesos y agotar
el presupuesto sin completar todos los ajustes.

La cifra anterior de unas 3 h 26 min tenía otro alcance: hasta 408 DNN,
328 RF, 100 SVM y 108 ajustes de detector antes de reutilización. No equivale
a esta etapa de DNN internas restantes.

## Límites y ejecución personal

- Primer lote: **un ajuste**, presupuesto de lote 600 s; revisar antes de ampliar.
- Lotes siguientes: como máximo **16 ajustes**, presupuesto de lote 1800 s.
- Cada proceso: máximo **300 s**, CPU con un hilo y ejecución secuencial.
- Presupuesto acumulado de procesos de la campaña: **7200 s**, persistente.

Con el perfil medido, 16 procesos sumarían unos
{budget['recommended_following_batch']['forecast_mean_process_minutes']:.2f} min; la
sensibilidad de componentes suma unos
{budget['recommended_following_batch']['forecast_component_stress_minutes']:.2f} min.
Conviene reservar **4–10 minutos por lote**, con techo solicitado de 30 minutos.
No iniciar el siguiente proceso si quedan menos de 300 s de lote o presupuesto
acumulado; detenerse en esa frontera conserva resultados completos. Los cinco
minutos por proceso no implican que cada ajuste vaya a tardar cinco minutos.

El presupuesto acumulado contabiliza procesos, no todo el tiempo de reloj:
verificación, preparación y auditoría pueden añadir tiempo fuera de ellos.
La terminación por timeout puede requerir una breve gracia adicional.
Reanudar no reinicia el presupuesto. No hay reintentos ni ampliación automática;
un fallo conserva el intento y detiene la cola para revisión.

El usuario ejecutará personalmente los comandos. Este documento y su generador
solo leen evidencias existentes: no entrenan, importan TensorFlow ni abren COM.
Los hashes de resultados, estados y cierres utilizados figuran en el JSON.
'''
    with document.open('x',encoding='utf-8',newline='\n') as stream:
        stream.write(text)
    print(json.dumps(dict(status=budget['status'],n_new_fits=n_new_fits,
        forecast_mean_minutes=f['from_observed_mean_minutes'],
        forecast_max_process_minutes=f['from_observed_max_process_minutes'],
        forecast_all50_component_stress_minutes=f['component_max_all50_epoch_stress_minutes'],
        limits=budget['limits'],source_sha256=budget['source_sha256']),ensure_ascii=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze',action='store_true',required=True)
    parser.add_argument('--new-fits',type=int,required=True)
    args = parser.parse_args()
    write_budget(args.new_fits)


if __name__ == '__main__':
    main()
