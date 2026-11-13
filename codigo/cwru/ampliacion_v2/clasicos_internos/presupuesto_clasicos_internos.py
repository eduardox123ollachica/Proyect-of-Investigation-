"""Presupuesto RF/SVM interno desde el piloto sellado, sin entrenar.

build_budget() solo lee JSON, hashes y el listado de artefactos existentes.
No importa NumPy, sklearn o TensorFlow ni carga matrices/modelos.
La proyección conserva384 ajustes por rol; no incorpora ajustes exteriores.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[4]
CLOSURE=ROOT/'articulo/ampliacion_cwru/metodologia/cierre_piloto_clasicos_2026-10-25'
SEAL=CLOSURE/'MANIFEST_CIERRE_CLASICOS_SHA256.json'
PILOT_CLOSURE=CLOSURE/'CIERRE_PILOTO_CLASICOS.json'
OLD_BUDGET=CLOSURE/'PRESUPUESTO_CPU_PRELIMINAR.json'
LIMITS=dict(RF_worker_seconds=60,SVM_worker_seconds=30,
    campaign_execution_verification_seconds=3600,maximum_new_fits=384,
    concurrent_workers=1,automatic_retry=False)


def require(ok,message):
    if not ok:
        raise ValueError(message)


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def describe(values):
    require(values and all(math.isfinite(v) and v>0 for v in values),'Tiempos inválidos')
    return dict(count=len(values),minimum=min(values),mean=statistics.mean(values),
        median=statistics.median(values),maximum=max(values),sum=math.fsum(values))


def audited_inputs():
    """Autenticar solo las fuentes temporales utilizadas, sin repetir el piloto."""
    digest,name=SEAL.with_suffix('.sha256').read_text(encoding='ascii').split()
    require(name == SEAL.name and digest == sha(SEAL),'Sello del piloto clásico distinto')
    entries=read(SEAL)
    sources={SEAL.relative_to(ROOT).as_posix():digest,
        SEAL.with_suffix('.sha256').relative_to(ROOT).as_posix():sha(SEAL.with_suffix('.sha256'))}
    def checked(path):
        path=Path(path).resolve()
        require(path.is_relative_to(ROOT.resolve()),'Fuente fuera del proyecto')
        relative=path.relative_to(ROOT).as_posix()
        current=sha(path)
        require(entries.get(relative) == current,'Fuente temporal no corresponde al cierre: '+relative)
        sources[relative]=current
        return read(path)
    closure,budget=checked(PILOT_CLOSURE),checked(OLD_BUDGET)
    require(closure['status']=='CLASSICAL_PILOT_AUDITED_NO_RETRAINING'
        and closure['jobs_passed']==17 and closure['failed_or_timeout_jobs']==0
        and closure['pilot_predictive_metrics_computed'] is False
        and closure['predictive_selection_or_external_evaluation_executed'] is False,
        'Piloto temporal no cerrado o alcance distinto')
    require(budget['classical_projection']['pilot_scope']['pilot_models_reused_as_confirmatory_results'] is False,
        'El piloto no es un ajuste confirmatorio reutilizable')
    run=(ROOT/closure['run']).resolve()
    require(run.is_relative_to((ROOT/'resultados/cwru/ampliacion_v2/piloto_clasicos_2026-10-25').resolve()),
        'Ejecución piloto fuera del destino esperado')
    authorization=checked(run/'EJECUCION_ACOTADA.json')
    require(authorization['scope']=='classical_runtime_pilot_only'
        and authorization['mass_training_enabled'] is False
        and authorization['outer_predictive_evaluation_enabled'] is False,'Autorización piloto distinta')
    pilot_jobs={job['job_id']:job for job in authorization['jobs'] if job['family'] in ('RF','SVM')}
    rows=closure['classifier_rows']
    require(len(rows)==len(pilot_jobs)==16
        and {r['job_id'] for r in rows}=={f'{family}_{i:02d}' for family in ('RF','SVM') for i in range(8)},
        'Perfiles de configuraciones incompletos o repetidos')
    versions=None
    for row in rows:
        identifier=row['job_id'];job=pilot_jobs[identifier]
        result=checked(run/identifier/'resultado.json')
        status=checked(run/f'ESTADO_{identifier}.json')
        require(result['status']=='PASS' and status['status']=='PASS'
            and status['return_code']==0 and not status['automatic_retry']
            and status['process_launch_error'] is None and status['result_error'] is None,
            'Proceso piloto fallido')
        require(result['confirmatory_fit'] is False and result['metrics_used_for_selection'] is False
            and result['validation_labels_used_for_metrics'] is False
            and result['requested_params']==row['params']==job['params']
            and result['config_index']==job['config_index']
            and result['timing_nonoverlapping_seconds']==row['stages_seconds']
            and status['process_elapsed_seconds']==row['process_seconds']
            and result['training_windows']==799 and result['validation_windows']==175,
            'Perfil de presupuesto incompatible con su resultado original')
        require(status['timeout_seconds']==(60 if row['family']=='RF' else 30), 'Límite piloto distinto')
        stages=list(row['stages_seconds'].values())
        require(all(math.isfinite(v) and v>=0 for v in stages)
            and 0<math.fsum(stages)<=row['process_seconds']+1e-9,'Tiempos no finitos o solapados')
        if versions is None:versions=result['versions']
        require(result['versions']==versions,'Entornos piloto distintos')
        expected={'procedencia.json','progreso.jsonl','resultado.json','salidas_diagnostico.npz'}
        if row['family']=='SVM':expected.add('scaler_train_only.npz')
        require({p.name for p in (run/identifier).iterdir() if p.is_file()}==expected,
            'Inventario piloto cambió; revisar antes de afirmar que no hay modelo persistido')
    # El orden y los valores del grid provienen de esta implementación sellada.
    checked_sources=[ROOT/'codigo/cwru/ampliacion_v2/piloto_clasicos/contrato_piloto.py',
        ROOT/'codigo/cwru/ampliacion_v2/piloto_clasicos/worker_tiempos.py',
        ROOT/'codigo/cwru/ampliacion_v2/piloto_clasicos/presupuesto_clasicos.py']
    for path in checked_sources:
        relative=path.relative_to(ROOT).as_posix()
        current=sha(path)
        require(entries.get(relative)==current,'Núcleo del piloto cambiado: '+relative)
        sources[relative]=current
    return closure,budget,rows,versions,sources


def build_budget():
    closure,old,rows,versions,sources=audited_inputs()
    counts={'RF':288,'SVM':96}
    per_config={'RF':36,'SVM':12}
    profiles=[];families={}
    projected_with_gaps=[];observed_repetitions=[];maximum_family=[]
    for family in ('RF','SVM'):
        current=sorted([row for row in rows if row['family']==family],key=lambda row:row['job_id'])
        require(len(current)==8,'Falta una configuración')
        observed=describe([row['process_seconds'] for row in current])
        family_projected=[]
        for index,row in enumerate(current):
            previous=old['classical_projection'][family]['configuration_profiles'][str(index)]
            require(previous['source_job_id']==row['job_id']
                and previous['projected_inner_fits']==per_config[family], 'Perfil escalado de otra configuración')
            gap=previous['observed_process_gap_seconds']
            scaled=previous['inner']['seconds_per_fit']
            require(math.isfinite(gap) and gap>=0 and math.isfinite(scaled) and scaled>0,
                'Proyección temporal no válida')
            config_total=(scaled+gap)*per_config[family]
            family_projected.append(config_total)
            profiles.append(dict(family=family,config_index=index,source_job_id=row['job_id'],
                original_params=row['params'],original_training_windows=799,original_validation_windows=175,
                observed_process_seconds=row['process_seconds'],
                inner_training_windows_assumed=previous['inner']['training_windows_assumed'],
                inner_validation_windows_assumed=previous['inner']['final_inference_windows_assumed'],
                projected_stage_seconds_per_fit=scaled,observed_process_gap_seconds=gap,
                projected_execution_seconds_per_fit=scaled+gap,
                logical_inner_fits=per_config[family],projected_total_seconds=config_total))
        family_seconds=math.fsum(family_projected)
        projected_with_gaps.append(family_seconds)
        repeated=observed['sum']*per_config[family]
        observed_repetitions.append(repeated)
        maximum=observed['maximum']*counts[family]
        maximum_family.append(maximum)
        families[family]=dict(logical_inner_fits=counts[family],grid_configurations=8,
            pilot_profiles=observed,observed_processes_repeated_seconds=repeated,
            projected_inner_scaled_with_gap_seconds=family_seconds,
            all_family_jobs_at_observed_max_seconds=maximum,
            worker_limit_seconds=LIMITS[family+'_worker_seconds'])
    projection=math.fsum(projected_with_gaps)
    observed=math.fsum(observed_repetitions)
    stress=math.fsum(maximum_family)
    require(stress<LIMITS['campaign_execution_verification_seconds'], 'Reevaluar límite: el máximo observado ya lo supera')
    source_self=sha(__file__)
    return dict(status='INNER_CLASSICAL_BUDGET_PREPARED_FROM_SEALED_TIMING_PILOT',
        date_local='2026-10-27',timezone='America/Lima',
        scope='Internal RF/SVM only; no DNN, detector, selection, outer fitting/evaluation or STM32',
        planned_inner_cells=384,RF_inner_fits=288,SVM_inner_fits=96,
        reused_confirmatory_fits=0,new_inner_fits=384,
        RF_inner_seeds=[11,22,33],SVM_repeat_per_seed=False,
        pilot_not_repeated_for_timing=True,pilot_models_reused=False,
        pilot_nonreuse_reasons=['All16 pilot classifiers declare confirmatory_fit=false.',
            'Pilot did not compute predictive metrics or persist fitted estimators.',
            'Future roles and signatures are evaluated by the confirmatory executor.'],
        model_fit_deduplication_across_swapped_validation_outer_roles_enabled=False,
        logical_roles_kept_distinct=True,configuration_profiles=profiles,families=families,
        forecast=dict(observed_processes_repeated_seconds=observed,
            observed_processes_repeated_minutes=observed/60,
            projected_inner_scaled_plus_observed_gap_seconds=projection,
            projected_inner_scaled_plus_observed_gap_minutes=projection/60,
            all_family_jobs_at_observed_family_max_seconds=stress,
            all_family_jobs_at_observed_family_max_minutes=stress/60,
            recommended_user_reservation_minutes=[30,60],
            added_model_serialization_loading_verification_time_not_measured=True,
            calendar_time_including_shared_checks_can_exceed_counted_budget=True,
            guarantee=False,confidence_interval=False),
        limits=dict(LIMITS),
        budget_accounting='Accumulated launch/worker execution and per-fit verification; shared preparation and checks can add wall time. Never start without a full family timeout remaining.',
        budget_remaining_persists_on_resume=True,no_automatic_budget_increase=True,
        failures_preserved_without_automatic_retry=True,one_worker_per_fit=True,
        CPU_threads=1,execution_owner='user_CLI',tested_versions=versions,
        source_sha256=sources,self_source_sha256=source_self,
        new_fits_during_budget_preparation=0,new_inferences_during_budget_preparation=0,
        sklearn_imported=False,TensorFlow_imported=False,hardware_connected=False,
        limitations=['One observed process per configuration does not bound new load or seed runtime.',
            'RF n log(n) and SVM quadratic row scaling are inherited engineering assumptions, not measured complexity guarantees.',
            'New estimator serialization and numerical artifact verification were not part of the timing pilot.',
            'Cold imports, background workload or storage delays can exhaust the fixed budget before all384 cells.',
            'The previous global3h26 projection included DNN and outer/detector work; it is not this classical-inner forecast.',
            'RF/SVM are PC comparators; these timings do not measure STM32 latency.'])
