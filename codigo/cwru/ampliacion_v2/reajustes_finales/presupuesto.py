"""Presupuesto trazable de reajustes finales; solo lee metadatos y perfiles.

Las proyecciones usan procesos internos ya cerrados. Entrenar con tres cargas,
épocas fijas y comprobar solo train cambia el alcance: ninguna extrapolación
constituye una duración garantizada, un IC o un límite de WCET.
"""
from __future__ import annotations

import collections
import copy
import math
from pathlib import Path
import statistics

import contrato as c

ROOT=c.ROOT
DNN_CLOSURE=ROOT/'articulo/ampliacion_cwru/metodologia/cierre_matriz_dnn_interna_v2_2026-10-26'
CLASSICAL_CLOSURE=ROOT/'articulo/ampliacion_cwru/metodologia/cierre_clasicos_internos_v2_2026-10-27'
DETECTOR_CLOSURE=ROOT/'articulo/ampliacion_cwru/metodologia/cierre_detector_interno_v2_2026-10-26'
DNN_RUN=ROOT/'resultados/cwru/ampliacion_v2/busqueda_interna_v2_2026-10-26/20261027T005636_578225Z'
CLASSICAL_RUN=ROOT/'resultados/cwru/ampliacion_v2/clasicos_internos_v2_2026-10-27/20261027T060928_938422Z'
EXPECTED_SEALS={
    DNN_CLOSURE/'MANIFEST_CIERRE_MATRIZ_SHA256.json':
        '0856eb2a024650dd27ad7babb2ef381fdf85f1d71607bc3286d9899902c589b7',
    CLASSICAL_CLOSURE/'MANIFEST_CIERRE_CLASICOS_INTERNOS_SHA256.json':
        'ae51648e781afd26568d0ab9e78e89a36706f240a46fa57dfb3ca7384b14205e',
    DETECTOR_CLOSURE/'MANIFEST_CIERRE_DETECTOR_SHA256.json':
        '252ba802ce209a0bc5c4290e89565d954b5b731eb78dd00275ee8c439dd90043'}


def _require(ok,message):
    if not ok:
        raise ValueError(message)


def _number(value,positive=False):
    return (isinstance(value,(int,float)) and not isinstance(value,bool)
        and math.isfinite(value) and (value>0 if positive else value>=0))


def _describe(values):
    values=list(values)
    _require(values and all(_number(value) for value in values),'Perfil temporal no finito o vacío')
    return dict(count=len(values),minimum=min(values),mean=statistics.mean(values),
        median=statistics.median(values),maximum=max(values),sum=math.fsum(values))


class _Authenticated:
    """Autenticar exactamente los JSON consumidos, sin cargar modelos o arrays."""
    def __init__(self,manifest,sources):
        self.sources=sources
        digest,name=manifest.with_suffix('.sha256').read_text(encoding='ascii').split()
        _require(name==manifest.name and digest==c.io.sha(manifest)==EXPECTED_SEALS[manifest],
            'Sello de perfiles distinto del cierre verificado')
        self.entries=c.io.read(manifest)
        _require(isinstance(self.entries,dict) and self.entries,'Manifiesto temporal vacío')
        for path in (manifest,manifest.with_suffix('.sha256')):
            sources[path.relative_to(ROOT).as_posix()]=c.io.sha(path)

    def read(self,path):
        path=Path(path).resolve()
        _require(path.is_relative_to(ROOT.resolve()),'Fuente temporal fuera del proyecto')
        relative=path.relative_to(ROOT).as_posix();actual=c.io.sha(path)
        _require(self.entries.get(relative)==actual,'Fuente temporal no corresponde a su cierre: '+relative)
        self.sources[relative]=actual
        return c.io.read(path)


def _measured_profiles():
    sources={}
    dnn_auth=_Authenticated(DNN_CLOSURE/'MANIFEST_CIERRE_MATRIZ_SHA256.json',sources)
    classical_auth=_Authenticated(CLASSICAL_CLOSURE/'MANIFEST_CIERRE_CLASICOS_INTERNOS_SHA256.json',sources)
    detector_auth=_Authenticated(DETECTOR_CLOSURE/'MANIFEST_CIERRE_DETECTOR_SHA256.json',sources)
    dnn_closure=dnn_auth.read(DNN_CLOSURE/'CIERRE_MATRIZ_INTERNA.json')
    classical_closure=classical_auth.read(CLASSICAL_CLOSURE/'CIERRE_CLASICOS_INTERNOS.json')
    detector_closure=detector_auth.read(DETECTOR_CLOSURE/'CIERRE_DETECTOR_INTERNO.json')
    _require(dnn_closure['new_fits']==240 and dnn_closure['failures']==0
        and classical_closure['completed_fits']==384 and classical_closure['failed_fits']==0,
        'Las campañas temporales no están completas')
    dnn=[];ids=set();versions=None
    state_paths=sorted((DNN_RUN/'lotes').glob('*/ESTADO_*.json'))
    _require(len(state_paths)==240,'Se esperan 240 procesos DNN nuevos cerrados')
    stage_names={'imports','data_and_scaler','fit_including_checkpoint_IO',
        'FP32_conversion','INT8_conversion','validation_inference_and_IO'}
    for path in state_paths:
        state=dnn_auth.read(path);job_id=state['job_id']
        _require(isinstance(job_id,str) and job_id.startswith('inner_')
            and Path(job_id).name==job_id and job_id not in ids,'Proceso DNN repetido o ajeno')
        ids.add(job_id)
        result=dnn_auth.read(DNN_RUN/job_id/'resultado.json')
        _require(state['status']==result['status']=='PASS' and state['return_code']==0
            and not state['automatic_retry'] and result['job_id']==job_id,
            'Perfil DNN fallido o reintentado')
        stage=result['timings_seconds'];process=state['worker_elapsed_seconds']
        n=result['training_windows'];epochs=result['epochs_executed'];updates=result['optimizer_updates']
        _require(set(stage)==stage_names and all(_number(value) for value in stage.values())
            and _number(process,True) and _number(result['elapsed_worker_seconds'],True)
            and process>=result['elapsed_worker_seconds'] and process>=math.fsum(stage.values())-1e-8
            and type(n) is int and n>0 and type(epochs) is int and 1<=epochs<=50
            and updates==epochs*math.ceil(n/32) and _number(stage['fit_including_checkpoint_IO'],True),
            'Tiempos o actualizaciones DNN inconsistentes')
        versions=result['versions'] if versions is None else versions
        _require(result['versions']==versions,'Los perfiles DNN mezclan versiones')
        dnn.append(dict(job_id=job_id,candidate=result['candidate'],training_windows=n,
            epochs=epochs,updates=updates,process_seconds=process,
            stage_seconds=stage,unattributed_process_seconds=max(0.,process-math.fsum(stage.values())),
            fit_seconds_per_update=stage['fit_including_checkpoint_IO']/updates))
    _require(abs(math.fsum(row['process_seconds'] for row in dnn)
        -dnn_closure['worker_controller_budget_seconds_consumed'])<1e-6,'Acumulado DNN distinto del cierre')
    classical=[];ids=set()
    step_paths=sorted((CLASSICAL_RUN/'pasos').glob('*.json'))
    _require(len(step_paths)==384,'Se esperan 384 pasos clásicos completos')
    for path in step_paths:
        state=classical_auth.read(path);job_id=state['job_id']
        _require(isinstance(job_id,str) and job_id.startswith('clasico_')
            and Path(job_id).name==job_id and job_id not in ids,'Proceso clásico repetido o ajeno')
        ids.add(job_id)
        result=classical_auth.read(CLASSICAL_RUN/job_id/'resultado.json')
        _require(state['status']==result['status']=='PASS' and state['return_code']==0
            and not state['automatic_retry'] and result['job_id']==job_id,'Perfil clásico fallido o reintentado')
        process=state['execution_verification_seconds'];fit=result['fit_seconds'];infer=result['inference_seconds']
        _require(result['family'] in ('RF','SVM') and type(result['config_index']) is int
            and all(_number(value) for value in (process,fit,infer)) and process>=fit+infer-1e-8
            and type(result['training_windows']) is int and result['training_windows']>0
            and type(result['validation_windows']) is int and result['validation_windows']>0,
            'Tiempos o dimensiones clásicos inconsistentes')
        classical.append(dict(job_id=job_id,family=result['family'],config_index=result['config_index'],
            training_windows=result['training_windows'],validation_windows=result['validation_windows'],
            process_seconds=process,fit_seconds=fit,inference_seconds=infer,
            other_seconds=max(0.,process-fit-infer)))
    _require(abs(math.fsum(row['process_seconds'] for row in classical)
        -classical_closure['campaign_execution_verification_seconds'])<1e-6,
        'Acumulado clásico distinto del cierre')
    _require(detector_closure['new_parameter_fits']==19 and detector_closure['new_unique_evaluations']==54
        and detector_closure['failed_units']==0 and _number(detector_closure['worker_process_seconds'],True),
        'La referencia del detector mezcla otro alcance')
    return dnn,classical,detector_closure,versions,sources


def _dnn_project(profile,n,updates):
    stage=profile['stage_seconds'];ratio=max(1.,n/profile['training_windows'])
    # Imports y residuo de proceso se conservan fijos; las restantes etapas se
    # escalan por filas como sensibilidad de ingeniería, no ley de complejidad.
    variable=math.fsum(stage[key] for key in ('data_and_scaler','FP32_conversion',
        'INT8_conversion','validation_inference_and_IO'))
    fit=updates*profile['fit_seconds_per_update']
    return stage['imports']+profile['unattributed_process_seconds']+ratio*variable+fit


def _classical_project(profile,n,stress=False):
    ratio=max(1.,n/profile['training_windows'])
    if stress:
        # RF n*log(n) y SVM ratio^3 son sensibilidades explícitas, no cotas.
        ratio=(n*math.log2(max(n,2)))/(profile['training_windows']
            *math.log2(max(profile['training_windows'],2))) if profile['family']=='RF' else ratio**3
        ratio=max(1.,ratio)
    return profile['other_seconds']+profile['fit_seconds']*ratio \
        +profile['inference_seconds']*max(1.,n/profile['validation_windows'])


def build_budget(jobs,rows):
    """Derivar previsión y límites sin fit, inferencia o valores de características."""
    counts=collections.Counter(job['family'] for job in jobs)
    _require(dict(counts)==dict(DNN=80,RF=40,SVM=4,detector=7)
        and len({job['job_id'] for job in jobs})==131,'Inventario final distinto de 131 trabajos congelados')
    _require(len(rows)==1380 and all(int(row['global_window_id'])==i for i,row in enumerate(rows)),
        'Índice de ventanas distinto del sellado')
    training_windows={str(outer):sum(int(row['load_hp'])!=outer for row in rows) for outer in range(4)}
    _require(training_windows=={'0':1205,'1':987,'2':974,'3':974},'Número de ventanas por ámbito distinto')
    dnn_jobs=[job for job in jobs if job['family']=='DNN']
    epochs=sum(job['epochs'] for job in dnn_jobs)
    updates=sum(job['epochs']*math.ceil(training_windows[str(job['outer_load_hp'])]/32) for job in dnn_jobs)
    _require(epochs==2400 and updates==80840,'Épocas o actualizaciones diferentes de las decisiones congeladas')
    dnn,classical,detector,versions,sources=_measured_profiles()
    projections=[]
    for job in jobs:
        family=job['family'];outer=job['outer_load_hp'];n=training_windows[str(outer)]
        _require(len(job['training_global_window_ids'])==n,'Ámbito train del presupuesto no corresponde al trabajo')
        if family=='DNN':
            profiles=[profile for profile in dnn if profile['candidate']==job['candidate']]
            _require(profiles,'No hay perfiles de la máscara seleccionada')
            count_updates=job['epochs']*math.ceil(n/32)
            scaled=[_dnn_project(profile,n,count_updates) for profile in profiles]
            stage_max={key:max(profile['stage_seconds'][key] for profile in profiles)
                for key in profiles[0]['stage_seconds'] if key!='fit_including_checkpoint_IO'}
            largest_ratio=max(1.,max(n/profile['training_windows'] for profile in profiles))
            stress=(stage_max['imports']+max(profile['unattributed_process_seconds'] for profile in profiles)
                +largest_ratio*sum(value for key,value in stage_max.items() if key!='imports')
                +count_updates*max(profile['fit_seconds_per_update'] for profile in profiles))
            projections.append(dict(job_id=job['job_id'],family=family,outer_load_hp=outer,
                training_windows=n,epochs=job['epochs'],optimizer_updates=count_updates,
                matched_profile_count=len(profiles),
                naive_process_seconds=statistics.mean(profile['process_seconds'] for profile in profiles),
                scaled_process_seconds=statistics.mean(scaled),mixed_max_sensitivity_seconds=stress,
                timeout_seconds=c.TIMEOUTS[family]))
        elif family in ('RF','SVM'):
            profiles=[profile for profile in classical if profile['family']==family
                and profile['config_index']==job['config_index']]
            _require(profiles,'No hay perfiles de la configuración clásica seleccionada')
            projections.append(dict(job_id=job['job_id'],family=family,outer_load_hp=outer,
                training_windows=n,config_index=job['config_index'],matched_profile_count=len(profiles),
                naive_process_seconds=statistics.mean(profile['process_seconds'] for profile in profiles),
                scaled_process_seconds=statistics.mean(_classical_project(profile,n) for profile in profiles),
                mixed_max_sensitivity_seconds=max(_classical_project(profile,n,True) for profile in profiles),
                timeout_seconds=c.TIMEOUTS[family]))
        else:
            # No hay perfil por parámetro/proceso comparable: se reserva el
            # watchdog completo, sin presentarlo como una duración medida.
            projections.append(dict(job_id=job['job_id'],family=family,outer_load_hp=outer,
                training_windows=n,matched_profile_count=0,naive_process_seconds=None,
                scaled_process_seconds=None,mixed_max_sensitivity_seconds=None,
                planning_worker_allowance_seconds=c.TIMEOUTS[family],timeout_seconds=c.TIMEOUTS[family]))
    allowance=sum(p.get('planning_worker_allowance_seconds',0) for p in projections)
    totals={key:math.fsum(p[key] for p in projections if p[key] is not None)+allowance for key in
        ('naive_process_seconds','scaled_process_seconds','mixed_max_sensitivity_seconds')}
    by_family={family:dict(count=counts[family],**{
        key:(math.fsum(p[key] for p in projections if p['family']==family) if family!='detector' else None)
        for key in ('naive_process_seconds','scaled_process_seconds','mixed_max_sensitivity_seconds')})
        for family in counts}
    by_family['detector']['planning_worker_allowance_seconds']=allowance
    sources[Path(__file__).resolve().relative_to(ROOT).as_posix()]=c.io.sha(Path(__file__))
    return dict(status='BUDGET_FROZEN_FOR_FINAL_REFITS_ONLY',date_local='2026-10-27',timezone='America/Lima',
        scope='131 frozen final refits only; no heldout inference/evaluation, hardware or manuscript edits',
        counts=dict(**counts,total=len(jobs)),fixed_DNN_epochs=epochs,optimizer_updates=updates,
        training_windows_by_outer=training_windows,limits=copy.deepcopy(c.LIMITS),
        measured_reference=dict(DNN=dict(profile_count=len(dnn),new_internal_fits_only=True,
            process_seconds=_describe(p['process_seconds'] for p in dnn),
            fit_seconds_per_optimizer_update=_describe(p['fit_seconds_per_update'] for p in dnn),
            epochs=_describe(p['epochs'] for p in dnn),
            training_windows=_describe(p['training_windows'] for p in dnn),
            reused_48_profiles_not_mixed=True,fit_includes_internal_validation_and_checkpoint_IO=True),
            classical={family:dict(profile_count=sum(p['family']==family for p in classical),
                process_seconds=_describe(p['process_seconds'] for p in classical if p['family']==family))
                for family in ('RF','SVM')},
            detector_mixed_reference=dict(new_parameter_fits=19,new_unique_evaluations=54,
                single_worker_process_seconds=detector['worker_process_seconds'],
                not_an_isolated_fit_or_future_process_measurement=True)),
        forecast=dict(**totals,**{key.removesuffix('_seconds')+'_minutes':value/60 for key,value in totals.items()},
            by_family=by_family,detector_worker_allowance_seconds=allowance,
            detector_estimate_available=False,user_reserve_minutes=[30,75],
            reserve_is_orientation_not_guarantee=True,component_max_sensitivity_is_not_expected_runtime=True,
            aggregate_includes_detector_full_watchdog_allowance_not_measured_detector_projection=True),
        per_job_projections=projections,tested_DNN_profile_versions=versions,
        estimation_method=dict(DNN='Match candidate; keep imports/process residual fixed; scale nonfit stages by max(1,final_rows/inner_train_rows); fit_seconds/inner_updates times fixed final updates.',
            DNN_stress='Mixed stage maxima, largest row ratio, maximum observed fit/update; maxima can belong to different fits and are not one measured run.',
            classical='Match family/configuration; fixed remainder plus fit scaled by train rows and inference scaled from validation rows to final train rows. Stress RF n*log2(n), SVM train-row-ratio cubed are explicit sensitivities, not complexity guarantees.',
            detector=f'No independent final-process profile. Reserve 7 x {c.TIMEOUTS["detector"]} s worker allowance from the current contract; previous 49.094 s mixed 19 fits + 54 evaluations is reference only. A separate synthetic cold-import check justifies the operational watchdog adjustment, without changing the scientific protocol.'),
        budget_interpretation=dict(campaign_seconds_are_cumulative_job_execution_and_verification=True,
            max_batch_seconds_scope='Monotonic batch boundary including per-job preparation/execution/verification; shared preflight/inspect may occur outside batch clock.',
            worker_timeout_is_not_a_global_hard_cap=True,watchdog_timeout_excludes_parent_verification=True,
            interruption_or_termination_grace_may_add_wall_time=True,
            no_worker_starts_without_its_complete_timeout_remaining=True,
            reentry_does_not_reset_campaign_consumed_seconds=True,no_automatic_budget_increase=True,
            hitting_budget_does_not_imply_completion=True),
        qualifications=[
            'Future refits train on three development loads, rather than the two used by measured internal profiles.',
            'Internal DNN fit stage includes validation, early-stopping checks and checkpoint I/O; final fixed-epoch training removes those operations but retains train diagnostics.',
            'INT8 conversion calibrates all final training rows; graph-freeze and inference diagnostics now use training rows only and can cost more than prior heldout diagnostics.',
            'TensorFlow import, graph tracing, converter initialization, antivirus/storage and cold startup can dominate small-model training time.',
            'Updates and row ratios describe workload; optimization, conversion and tree/kernel training need not scale linearly.',
            'The 30–75 minute reservation leaves headroom above the mixed-maximum sensitivity for shared preparation, verification and cold imports; that reservation is orientation, not a bound or confidence interval.',
            'Detector allowance sums worker watchdogs; parent materialization/verification is additional and no separate detector-process fit duration was measured.',
            'Shared verification/preparation, process termination grace and interruptions add elapsed wall time beyond forecast profiles.',
            'Budgets and per-process timeouts are enforced stop policies, not assurances all 131 fits will complete.',
            'PC training time does not measure STM32 inference latency.'],
        source_sha256=sources,new_training_fits=0,new_model_predictions=0,
        raw_or_feature_arrays_materialized=False,outer_evaluation_executed=False,
        hardware_connected=False,training_started=False,manuscript_modified=False)
