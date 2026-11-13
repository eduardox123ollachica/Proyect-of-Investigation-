"""Audita artefactos guardados y consolida presupuesto; no entrena o reinfiere."""
from __future__ import annotations
import argparse
import csv
import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[4]
PREP=ROOT/'articulo/ampliacion_cwru/metodologia/piloto_clasicos_2026-10-25'
PROTOCOL=ROOT/'articulo/ampliacion_cwru/metodologia/protocolo_v2_2026-10-25'
DNN=ROOT/'articulo/ampliacion_cwru/metodologia/cierre_piloto_tiempo_2026-10-25'
OUT=ROOT/'articulo/ampliacion_cwru/metodologia/cierre_piloto_clasicos_2026-10-25'
sys.path.insert(0,str(ROOT/'codigo/cwru/ampliacion_v2/piloto_clasicos'))
from contrato_piloto import build_jobs,validate_result
from presupuesto_clasicos import project_classical_budget


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))


def verify(path):
    digest,name=path.with_suffix('.sha256').read_text(encoding='ascii').split()
    require(name==path.name and digest==sha(path),'Sello modificado: '+str(path))
    entries=read(path)
    for name,digest in entries.items():
        target=(ROOT/name).resolve()
        require(target.is_relative_to(ROOT) and sha(target)==digest,'Recurso modificado: '+name)
    return entries


def global_budget(dnn,classical):
    c=classical['aggregate']
    c_stage=c['projected_stage_seconds'];c_gap=c['projected_with_observed_process_gap_seconds']
    require(c_gap is not None,'Falta residual de procesos clasicos')
    d=dnn['projection'];scenarios={}
    for name,scenario in d['scenarios'].items():
        profiles={}
        for mask,profile in scenario['profiles'].items():
            stage=profile['projected_DNN_seconds']+c_stage
            with_gap=(profile['projected_DNN_seconds']+
                408*d['observed_process_minus_stages_seconds_by_mask'][mask]+c_gap)
            profiles[mask]=dict(profile_kind=d['profile_kinds'][mask],stage_seconds=stage,
                stage_hours=stage/3600,with_observed_process_gap_seconds=with_gap,
                with_observed_process_gap_hours=with_gap/3600,
                includes_technical_recovery_overhead=mask=='T6')
        cold=scenario['all_fits_like_initial_cold_import_sensitivity_seconds']+c_gap
        scenarios[name]=dict(epochs_assumed=scenario['epochs_assumed'],profiles=profiles,
            all_DNN_imports_like_initial_cold_sensitivity_seconds=cold,
            all_DNN_imports_like_initial_cold_sensitivity_hours=cold/3600,
            cold_SVM_RF_imports_not_extrapolated=True)
    return dict(status='CPU_BUDGET_PRELIMINARY_FROM_AUDITED_PILOTS',units='seconds',platform='PC',
        scenarios=scenarios,classical_projection=classical,
        canonical_planning_reference='epochs_50/F14_single_complete_process_profile_plus_classical_with_observed_gaps',
        total_maximum_fits_before_reuse=dict(DNN=408,RF=328,SVM=100,detector=108),
        maximum_DNN_conversions=816,cost_representatives_included_in_DNN_inner=32,
        cost_representatives_added_again=0,
        pilots_excluded_from_future_counts=dict(DNN_training_attempts=3,
            DNN_export_recovery_training_updates=0,RF_fits=8,SVM_fits=8,detector_fit_calibration_calls=21),
        mask_or_hyperparameter_selection_executed=False,outer_predictive_evaluation_executed=False,
        caching_savings_credited=False,full_protocol_frozen=False,full_pipeline_STM32_cost_available=False,
        long_execution_owner='user',long_execution_launched_by_this_audit=False,
        limitations=[
            'Pilot durations are measured; future campaign durations are engineering projections, not guarantees or confidence intervals.',
            'F14 is a clean one-process DNN profile; T6 includes export recovery and repeated import/setup. The two are not a speed comparison between masks.',
            'The cold-import sensitivity changes DNN imports only, not an unmeasured cold profile of classical libraries.',
            'The detector scenario copies grouped overhead per naive fit; actual caching may reduce work but no saving is credited.',
            'PC fitting/inference times are not latency on STM32. Hardware compilation/flashing/measurement, UART, acquisition and human review are excluded.',
            'Remaining gates include the definitive training executor, campaign limits and representative full-pipeline STM32 cost measurements.',
        ])


def audit(run):
    run=run.resolve()
    require(run.parent==ROOT/'resultados/cwru/ampliacion_v2/piloto_clasicos_2026-10-25','Ruta de piloto incorrecta')
    run_entries=verify(run/'MANIFEST_RESULTADOS_SHA256.json')
    prep_entries=verify(PREP/'MANIFEST_PREPARACION_SHA256.json')
    protocol_entries=verify(PROTOCOL/'MANIFEST_PROTOCOLO_SHA256.json')
    dnn_entries=verify(DNN/'MANIFEST_CIERRE_PILOTO_SHA256.json')
    protocol=read(PROTOCOL/'protocolo_reglas.json');plan=read(PREP/'PLAN_PILOTO.json')
    source_job=read(PROTOCOL/'piloto_plan.json')['jobs'][0]
    require(plan['jobs']==build_jobs(protocol),'Cuadricula distinta de reglas selladas')
    authorization=read(run/'EJECUCION_ACOTADA.json')
    require(authorization['execution_initiated_by']=='user_CLI' and authorization['execution_owner']=='user'
        and authorization['jobs']==plan['jobs'] and authorization['worker_environment']==plan['worker_environment'],
        'Identidad de ejecucion distinta')
    unique=read(run.parent/'EJECUCION_UNICA.json')
    require(unique['run_id']==run.name and unique['preparation_manifest_sha256']==sha(PREP/'MANIFEST_PREPARACION_SHA256.json'),
        'Identidad de ejecucion unica incorrecta')
    summary=read(run/'RESUMEN_PILOTO.json')
    require(summary['status']=='PILOT_COMPLETED' and summary['jobs_requested']==summary['jobs_started']==summary['jobs_passed']==17
        and summary['projection_error'] is None and summary['mass_training_started'] is False
        and summary['outer_predictive_evaluation_executed'] is False,'Piloto incompleto o fuera de alcance')
    require(not list(run.glob('FALLO_*.json')),'No omitir fallos registrados')
    with (ROOT/'datos/cwru/particiones/ampliacion_v2/por_carga_2026-10-22/indice_ventanas_fuente.csv').open(encoding='utf-8-sig',newline='') as stream:
        index={int(r['global_window_id']):r for r in csv.DictReader(stream)}
    expected_train=source_job['train']['global_window_ids'];expected_val=source_job['validation']['global_window_ids']
    require(len(expected_train)==799 and len(expected_val)==175 and set(expected_train).isdisjoint(expected_val)
        and all(int(index[g]['load_hp']) in (1,2) for g in expected_train)
        and all(int(index[g]['load_hp'])==0 for g in expected_val),'Papeles de carga incorrectos')
    train_files={index[g]['file_id'] for g in expected_train};val_files={index[g]['file_id'] for g in expected_val}
    require(len(train_files)==14 and len(val_files)==7 and train_files.isdisjoint(val_files),'MAT compartidos')
    budget_input=read(run/'RESULTADOS_PARA_PRESUPUESTO.json')
    results=[];family_rows=[];hashes={};previous_scaler=None
    for job,status in zip(plan['jobs'],summary['statuses']):
        directory=run/job['job_id'];result=read(directory/'resultado.json');validate_result(result,job)
        require(read(run/('ESTADO_'+job['job_id']+'.json'))==status and status['job_id']==job['job_id']
            and status['status']=='PASS' and status['return_code']==0 and status['timeout_seconds']==job['timeout_seconds']
            and status['process_elapsed_seconds']<=status['timeout_seconds']
            and status['process_launch_error'] is None and status['result_error'] is None,'Estado de trabajo incorrecto')
        require(result['versions']==plan['tested_environment'] and result['no_TensorFlow_imported']
            and result['no_STM32_latency_measured'],'Versiones o alcance incorrectos')
        provenance=read(directory/'procedencia.json')
        require(provenance['train_global_window_ids']==expected_train and provenance['validation_global_window_ids']==expected_val
            and provenance['arrays_opened']==[a['path'] for a in provenance['source_arrays']]
            and provenance['load3_arrays_opened'] is False and provenance['raw_signals_or_MAT_opened'] is False,
            'Procedencia incorrecta')
        require(provenance['source_index_sha256']==sha(ROOT/'datos/cwru/particiones/ampliacion_v2/por_carga_2026-10-22/indice_ventanas_fuente.csv')
            and provenance['protocol_manifest_sha256']==sha(PROTOCOL/'MANIFEST_PROTOCOLO_SHA256.json'),'Indices distintos')
        require(len(provenance['source_arrays'])==2,'Fuentes de features distintas')
        for a in provenance['source_arrays']:
            require(Path(a['path']).name in ('X_features_train.npy','X_features_validation.npy') and a['dtype']=='float64',
                'Fuente no autorizada')
            if a['path'] not in hashes:hashes[a['path']]=sha(ROOT/a['path'])
            require(hashes[a['path']]==a['sha256'],'Cache fuente modificado')
        times=result['timing_nonoverlapping_seconds']
        require(all(math.isfinite(v) and v>=0 for v in times.values()),'Tiempo no finito')
        own_stages=sum(times.values())
        if job['family']=='DETECTOR':
            own_stages+=sum(r['fit_calibration_seconds']+r['inference_seconds'] for r in result['rows'])
        require(own_stages<=result['worker_elapsed_to_result_seconds']<=status['process_elapsed_seconds'],
            'Etapas solapadas o wall time incorrecto')
        events=[json.loads(line) for line in (directory/'progreso.jsonl').read_text(encoding='utf-8').splitlines()]
        require(events[0]['stage']=='worker_started' and events[-1]['stage']=='PASS'
            and all(b['elapsed_seconds']>=a['elapsed_seconds'] for a,b in zip(events,events[1:])),
            'Progreso incompleto o tiempo regresivo')
        annotated=dict(result,process_elapsed_seconds=status['process_elapsed_seconds'])
        results.append(annotated)
        if job['family']!='DETECTOR':
            require(result['validation_labels_used_for_metrics'] is False and result['model_classes']==[0,1,2],
                'Clases o uso de etiquetas incorrecto')
            with np.load(directory/'salidas_diagnostico.npz',allow_pickle=False) as stored:
                predictions=stored['predictions'];values=stored['values']
                require(predictions.shape==(175,) and predictions.dtype.kind in 'iu' and set(predictions.tolist())<={0,1,2}
                    and values.shape==(175,3) and np.isfinite(values).all()
                    and np.array_equal(stored['validation_global_window_ids'],expected_val)
                    and hashlib.sha256(predictions.tobytes()).hexdigest()==result['prediction_sha256'],'Salidas guardadas inconsistentes')
                if job['family']=='RF':
                    require(np.all(values>=0) and np.all(values<=1) and np.allclose(values.sum(axis=1),1,rtol=1e-12,atol=1e-12)
                        and np.array_equal(predictions,np.argmax(values,axis=1)),'Probabilidades RF inconsistentes')
            if job['family']=='SVM':
                with np.load(directory/'scaler_train_only.npz',allow_pickle=False) as scaler:
                    require(scaler['mean64'].dtype==scaler['scale64'].dtype==np.float64
                        and scaler['mean32'].dtype==scaler['scale32'].dtype==np.float32
                        and np.array_equal(scaler['mean64'].astype(np.float32),scaler['mean32'])
                        and np.array_equal(scaler['scale64'].astype(np.float32),scaler['scale32'])
                        and np.isfinite(scaler['mean64']).all() and np.isfinite(scaler['scale64']).all()
                        and (scaler['scale32']>0).all(),'Scaler SVM inconsistente')
                    signature=tuple(scaler[k].tobytes() for k in sorted(scaler.files))
                    require(previous_scaler is None or previous_scaler==signature,'El scaler cambia con C/gamma')
                    previous_scaler=signature
            family_rows.append(dict(job_id=job['job_id'],family=job['family'],params=job['params'],
                process_seconds=status['process_elapsed_seconds'],stages_seconds=times))
    require(budget_input==dict(classifiers=results[:16],detector=results[16]),'Entrada de presupuesto distinta de los workers')
    require(math.isclose(sum(s['process_elapsed_seconds'] for s in summary['statuses']),summary['observed_worker_seconds'],abs_tol=1e-8)
        and summary['elapsed_orchestrator_seconds']>=summary['observed_worker_seconds'],'Tiempo total inconsistente')
    detector=results[16];dj=plan['jobs'][16];detdir=run/'DETECTOR'
    fit_ids=source_job['detector_fit_role_only']['global_window_ids'];cal_ids=source_job['detector_calibration_role_only']['global_window_ids']
    require(detector['fit_global_window_ids']==fit_ids and detector['calibration_global_window_ids']==cal_ids
        and detector['refit_after_calibration'] is False and set(fit_ids).isdisjoint(cal_ids)
        and all(index[g]['class']=='N' and int(index[g]['load_hp'])==1 for g in fit_ids)
        and all(index[g]['class']=='N' and int(index[g]['load_hp'])==2 for g in cal_ids),'Ajuste/calibracion detector mezclados')
    detector_rows=[]
    keys={'mean64','scale64','var64','mean32','scale32','centroid32','threshold32','indices'}
    for candidate in dj['unique_candidates']:
        reps=[r for r in detector['rows'] if r['candidate']==candidate]
        require(len(reps)==3 and {r['rep'] for r in reps}=={0,1,2},'Repeticiones incompletas')
        with np.load(detdir/f'parametros_{candidate}.npz',allow_pickle=False) as params:
            require(set(params.files)==keys and params['indices'].tolist()==dj['indices_by_candidate'][candidate],
                'Indices detector inconsistentes')
            n=len(dj['indices_by_candidate'][candidate]);digest=hashlib.sha256()
            require(params['threshold32'].dtype==np.float32 and params['threshold32'].shape==()
                and np.isfinite(params['threshold32']) and params['threshold32']>=0,'Umbral invalido')
            for key in keys-{'threshold32','indices'}:
                require(params[key].shape==(n,) and np.isfinite(params[key]).all()
                    and params[key].dtype==(np.float64 if key.endswith('64') else np.float32),'Parametro detector invalido')
            require((params['scale64']>0).all() and (params['scale32']>0).all() and (params['var64']>=0).all()
                and np.array_equal(params['mean64'].astype(np.float32),params['mean32'])
                and np.array_equal(params['scale64'].astype(np.float32),params['scale32']),'Exportacion scaler detector distinta')
            for key in sorted(keys):
                a=np.asarray(params[key]);digest.update(key.encode());digest.update(str(a.dtype).encode());digest.update(a.tobytes())
            threshold_bits=f"{int(np.asarray(params['threshold32'],dtype='<f4').view('<u4').item()):08x}"
        with np.load(detdir/f'scores_{candidate}.npz',allow_pickle=False) as saved:
            scores=saved['scores']
            require(scores.shape==(175,) and scores.dtype==np.float32 and np.isfinite(scores).all() and (scores>=0).all()
                and np.array_equal(saved['validation_global_window_ids'],expected_val),'Scores guardados invalidos')
            score_hash=hashlib.sha256(scores.tobytes()).hexdigest()
        require(all(r['status']=='PASS' and r['params_sha256']==digest.hexdigest() and r['scores_sha256']==score_hash
            and r['threshold_bits']==threshold_bits for r in reps),'Hashes de repeticiones inconsistentes')
        detector_rows.append(dict(candidate=candidate,features=n,threshold_bits=threshold_bits,
            repeated_parameter_and_score_hashes_equal=True,
            fit_calibration_mean_seconds=statistics.mean(r['fit_calibration_seconds'] for r in reps),
            fit_calibration_max_seconds=max(r['fit_calibration_seconds'] for r in reps),
            inference_mean_seconds=statistics.mean(r['inference_seconds'] for r in reps)))
    require(detector['aliases']=={'NF12':'F14'} and dj['indices_by_candidate']['NF12']==dj['indices_by_candidate']['F14']
        and not list(detdir.glob('*NF12*.npz')),'No inventar medicion de alias')
    projection=project_classical_budget(results[:16],detector)
    require(projection==read(run/'PROYECCION_PRESUPUESTO_CLASICOS.json'),'Proyeccion temporal no reproducible')
    dnn=read(DNN/'CIERRE_PILOTO.json');global_projection=global_budget(dnn,projection)
    groups={name:dict(processes=sum(s['family']==name for s in summary['statuses']),
        observed_worker_seconds=sum(s['process_elapsed_seconds'] for s in summary['statuses'] if s['family']==name))
        for name in ('RF','SVM','DETECTOR')}
    report=dict(status='CLASSICAL_PILOT_AUDITED_NO_RETRAINING',generated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        run=run.relative_to(ROOT).as_posix(),jobs_passed=17,failed_or_timeout_jobs=0,
        observed_orchestrator_seconds=summary['elapsed_orchestrator_seconds'],observed_worker_seconds=summary['observed_worker_seconds'],
        observed_families=groups,classifier_rows=family_rows,detector_rows=detector_rows,
        run_resources_verified=len(run_entries),preparation_resources_verified=len(prep_entries),
        protocol_resources_verified=len(protocol_entries),DNN_closure_resources_verified=len(dnn_entries),
        exact_data_scope=dict(train_windows=799,validation_windows=175,train_MAT=14,validation_MAT=7,
            train_loads=[1,2],validation_loads=[0],detector_fit_windows=58,detector_calibration_windows=58,
            detector_fit_MAT=sorted({index[g]['file_id'] for g in fit_ids}),
            detector_calibration_MAT=sorted({index[g]['file_id'] for g in cal_ids})),
        detector_aliases={'NF12':'F14'},detector_unique_representations=7,detector_repeated_fit_calibration_calls=21,
        repeated_calls_are_not_seeds_or_independent_acquisitions=True,
        source_feature_cache_hashes_verified=hashes,pilot_predictive_metrics_computed=False,
        actual_pilot_fits_executed_by_user=dict(RF=8,SVM=8,detector_fit_calibration=21,DNN=0),
        projection_flag_interpretation='DNN_or_classical_training_launched=false belongs to the pure projection function; the actual user pilot did fit RF/SVM/detector as recorded here.',
        model_fitting_or_inference_executed_by_auditor=False,TensorFlow_or_sklearn_imported_by_auditor=False,
        source_feature_arrays_loaded_by_auditor=False,stored_diagnostic_NPZ_loaded_by_auditor=True,
        predictive_selection_or_external_evaluation_executed=False,manuscript_or_firmware_modified=False,
        global_cpu_budget_preliminary_ready=True,full_protocol_frozen=False,full_STM32_cost_available=False,
        limitations=[
            'Hashes, stored artifacts and guarded worker sources were checked; no independent OS-level trace of every file access was collected.',
            'Only first-repetition numeric parameter/score arrays were saved; subsequent repetitions are checked through their recorded matching hashes and runtime guards.',
            'The audit did not re-fit scalers/centroids, recompute P95 from source arrays or re-run prediction; it does not claim a new independent predictive experiment.',
            'Classical inference timing includes predict and additional diagnostic predict_proba/decision_function calls.',
            'First detector fit/calibration was 25.1741 ms versus approximately 1--2 ms for subsequent calls; it is retained in the observed maximum.',
            'This is a timing pilot, not a comparison of predictive accuracy or an ARM measurement.',
        ])
    paths={**run_entries,**prep_entries,**protocol_entries,**dnn_entries}
    for base,name in ((run,'MANIFEST_RESULTADOS_SHA256'),(PREP,'MANIFEST_PREPARACION_SHA256'),
                      (PROTOCOL,'MANIFEST_PROTOCOLO_SHA256'),(DNN,'MANIFEST_CIERRE_PILOTO_SHA256')):
        for extension in ('.json','.sha256'):
            path=base/(name+extension);paths[path.relative_to(ROOT).as_posix()]=sha(path)
    paths[(run.parent/'EJECUCION_UNICA.json').relative_to(ROOT).as_posix()]=sha(run.parent/'EJECUCION_UNICA.json')
    paths[(DNN/'CIERRE_PILOTO.json').relative_to(ROOT).as_posix()]=sha(DNN/'CIERRE_PILOTO.json')
    return report,global_projection,paths


def write_outputs(report,budget,paths):
    require(not OUT.exists(),'No sobrescribir cierre existente');OUT.mkdir()
    def save(name,value):
        path=OUT/name
        with path.open('x',encoding='utf-8',newline='\n') as stream:
            json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False);stream.write('\n')
    save('CIERRE_PILOTO_CLASICOS.json',report);save('PRESUPUESTO_CPU_PRELIMINAR.json',budget)
    projection=budget['classical_projection']
    lines=['# Cierre del piloto RF, SVM y detector','',
        f"Se aprobaron 17/17 procesos en {report['observed_orchestrator_seconds']:.3f} s de orquestador.",
        'El usuario ejecutó el piloto; esta auditoría no entrenó o reinfirió. Se conservaron sus',
        'artefactos y fuentes, sin abrir matrices fuente como arrays ni modificar el manuscrito.','',
        '| Bloque | Procesos del piloto | Tiempo observado de procesos (s) |',
        '|---|---:|---:|']
    for name,r in report['observed_families'].items():lines.append(f"| {name} | {r['processes']} | {r['observed_worker_seconds']:.3f} |")
    lines += ['', 'Los 13 minutos eran la suma de límites por proceso, no la duración esperada.',
        'Entrenamiento 1/2 HP: 799 ventanas de 14 MAT; diagnóstico 0 HP: 175 de siete MAT.',
        'Detector: N de MAT 98 para ajuste y N de MAT 99 para P95, 58/58, sin refit.',
        'Las siete representaciones conservaron hashes idénticos en sus tres repeticiones.',
        'NF12 es alias de F14; no se creó una medición duplicada. Estas repeticiones no son semillas.', '',
        'Se verificaron sellos, roles, versiones, configuraciones, salidas, scalers, parámetros,',
        'scores y tiempos. Se reprodujo la proyección desde los tiempos registrados, sin',
        'volver a ejecutar ajuste, calibración, predicción o hardware.', '',
        '## Proyección de campaña futura PC', '',
        '| Bloque | Ajustes máximos antes de reutilizar | Etapas (min) | Con residual de procesos (min) |',
        '|---|---:|---:|---:|']
    for name,count in (('RF',328),('SVM',100),('detector',108)):
        p=projection[name];lines.append(f"| {name} | {count} | {p['projected_stage_seconds']/60:.3f} | {p['projected_with_observed_process_gap_seconds']/60:.3f} |")
    agg=projection['aggregate'];lines.append(f"| Total clásico + detector | — | {agg['projected_stage_seconds']/60:.3f} | {agg['projected_with_observed_process_gap_seconds']/60:.3f} |")
    lines += ['', 'Son escenarios por tamaños de datos y tiempos medidos, no cotas ni duración garantizada.',
        'El overhead del detector se copia conservadoramente a cada ajuste antes de caché.',
        'RF/SVM incluyen la llamada diagnóstica adicional de inferencia; no son costos mínimos.',
        'El máximo inicial de 25.1741 ms del ajuste/calibración del detector se conserva.', '',
        'Los artefactos permiten comprobar consistencia; no hay una traza independiente del sistema',
        'operativo para todos los accesos a archivos. Se guardaron arrays de la primera repetición',
        'del detector y hashes de las otras dos. No se recalculó P95 con fuentes durante el cierre.', '',
        'Siguiente paso: implementar/verificar el ejecutor definitivo y fijar límites de campaña.',
        'Después, preparar los 32 representantes de costo total STM32. Están incluidos en las',
        '288 DNN internas: no se suman otra vez. Toda ejecución prolongada corresponde al usuario.', '',
        'Las métricas predictivas, selección y evaluación externa no se ejecutaron en este piloto.',
        'El protocolo completo y el costo total físico STM32 siguen pendientes.']
    (OUT/'CIERRE_PILOTO_CLASICOS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    lines=['# Presupuesto CPU preliminar consolidado','',
        'La duración observada del piloto clásico fue 62.687 s. Las siguientes horas corresponden',
        'a la campaña futura completa antes de reutilizar ajustes: DNN 408, RF 328, SVM 100 y',
        'detector hasta 108, con hasta 816 conversiones DNN. Los pilotos se contabilizan aparte.', '',
        '| Épocas | Perfil DNN | Solo etapas (h) | Con residual propio de procesos (h) |',
        '|---:|---|---:|---:|']
    for scenario in budget['scenarios'].values():
        for name,p in scenario['profiles'].items():
            kind='F14, proceso completo' if name=='F14' else 'T6, incluye recuperación'
            lines.append(f"| {scenario['epochs_assumed']} | {kind} | {p['stage_hours']:.6f} | {p['with_observed_process_gap_hours']:.6f} |")
    main=budget['scenarios']['epochs_50']['profiles']['F14']['with_observed_process_gap_seconds']
    hours=int(main//3600);minutes=round((main%3600)/60)
    lines += ['',f'La referencia de planificación limpia a 50 épocas es aproximadamente **{hours} h {minutes} min**.',
        'El perfil T6 compuesto suma importación y preparación repetidas por recuperación; el',
        'escenario resultante es conservador y no demuestra que T6 sea más lento.', '',
        '## Sensibilidad al arranque frío DNN', '',
        '| Épocas | Si cada importación DNN repite el primer arranque frío (h) |',
        '|---:|---:|']
    for s in budget['scenarios'].values():lines.append(f"| {s['epochs_assumed']} | {s['all_DNN_imports_like_initial_cold_sensitivity_hours']:.6f} |")
    lines += ['', 'Esta sensibilidad cambia únicamente la importación DNN; conserva los clásicos medidos.',
        'No se extrapola un arranque frío desconocido de RF/SVM ni se promete una duración.',
        'Los modelos, semillas y cargas futuros pueden cambiar el tiempo y el trabajo reutilizable.', '',
        'Los 32 representantes ya están dentro de las DNN internas. Pilotos y recuperación',
        'técnica no se suman como nuevos resultados confirmatorios o adquisiciones independientes.',
        'No se acredita ahorro de caché no medido.', '',
        'Se excluyen compilación/flasheo y medición física STM32, UART/adquisición y revisión humana.',
        'Tiempo de entrenamiento PC y latencia STM32 son alcances distintos.',
        'Faltan el ejecutor definitivo, límites de campaña y costo total físico de representantes.',
        'El presupuesto es preliminar; el protocolo completo no se declara congelado.', '',
        '**Ejecución:** corresponde al usuario. Este cierre no inició la campaña prolongada.']
    (OUT/'PRESUPUESTO_CPU_PRELIMINAR.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    paths=dict(paths)
    for path in [Path(__file__),*OUT.iterdir()]:
        if path.is_file():paths[path.relative_to(ROOT).as_posix()]=sha(path)
    manifest=OUT/'MANIFEST_CIERRE_CLASICOS_SHA256.json'
    with manifest.open('x',encoding='utf-8',newline='\n') as stream:json.dump(paths,stream,indent=2);stream.write('\n')
    manifest.with_suffix('.sha256').write_text(sha(manifest)+'  '+manifest.name+'\n',encoding='ascii')
    verify(manifest)
    return len(paths)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True);parser.add_argument('--write',action='store_true')
    args=parser.parse_args();report,budget,paths=audit(args.run_dir)
    count=write_outputs(report,budget,paths) if args.write else None
    print(json.dumps(dict(status=report['status'],jobs_passed=17,
        observed_seconds=report['observed_orchestrator_seconds'],resources_verified=report['run_resources_verified'],
        new_sealed_resources=count,classical_projected_minutes=budget['classical_projection']['aggregate']['projected_with_observed_process_gap_seconds']/60,
        global_50_epochs_reference_hours=budget['scenarios']['epochs_50']['profiles']['F14']['with_observed_process_gap_hours']),indent=2))
