"""Preparación y ejecución personal de evaluación exterior, sin entrenamiento."""
from __future__ import annotations
import argparse
import datetime
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import contrato_evaluacion as c
io=c.io


def prepare():
    io.require(not c.PLAN.exists() and not c.MANIFEST.exists(),'Preparación existente; no sobrescribir')
    protocol,rows,hashes,mapping,inventory,refit_plan=c.closed_context()
    jobs=c.build_jobs(protocol,rows,mapping,inventory,refit_plan);sourcehashes=c.sources()
    tests=io.read(c.PREP/'PRUEBAS_PREPARACION.json');synthetic=io.read(c.PREP/'PRUEBA_SINTETICA_EVALUACION.json')
    io.require(tests['status']=='PASS' and tests['errors']==tests['failures']==0
        and tests['source_sha256']==sourcehashes and tests['CWRU_predictions']==0,'Faltan pruebas de las fuentes finales')
    io.require(synthetic['status']=='PASS' and synthetic['source_sha256']==sourcehashes
        and synthetic['CWRU_values_opened'] is False and synthetic['new_training_fits']==0,
        'Falta ensayo sintético final sin ajustes')
    versions=c.refit.versions()
    io.require(versions==refit_plan['expected_versions']==synthetic['versions'],'Entorno diferente del fijado')
    inputs={p:hashes[p] for p in sorted({p for j in jobs for p in j['authorized_feature_source_paths']})}
    for p,digest in inputs.items():io.require(io.sha(io.confined(c.ROOT,p))==digest,'Caché modificado')
    counts={key:sum(j['expected_inference_counts'][key] for j in jobs) for key in jobs[0]['expected_inference_counts']}
    budget=dict(status='FIXED_OPERATIONAL_BUDGET_FOR_OUTER_EVALUATION_ONLY',limits=c.LIMITS,
        expected_counts=counts,heldout_windows_by_outer=[j['heldout_windows'] for j in jobs],
        expected_wall_minutes_orientation=[5,15],orientation_is_not_a_guarantee=True,
        worker_timeout_is_not_expected_duration=True,shared_preflight_excluded_from_campaign_counter=True,
        parent_verification_and_termination_can_add_wall_time=True,
        basis='Four fresh processes; prior synthetic TensorFlow cold import 50.371 s. No full evaluation runtime measured. Fixed batch1 invocations and saved-output checks; no model fitting.',
        automatic_budget_increase=False,outer_predictions_during_preparation=0)
    io.write(c.PREP/'PRESUPUESTO_Y_LIMITES.json',budget)
    plan=dict(status='OUTER_EVALUATION_EXECUTOR_PREPARED_PENDING_USER_RUN',jobs=jobs,
        selection_signature=c.refit.SELECTION_SIGNATURE,final_refit_manifest_sha256=c.CLOSED_SHA256,
        expected_versions=versions,source_sha256=sourcehashes,input_sha256=inputs,worker_environment=io.ENVIRONMENT,
        limits=c.LIMITS,evaluation_signatures={j['job_id']:c.signature(j,sourcehashes,versions,hashes) for j in jobs},
        expected_counts=counts,outer_evaluation_enabled=True,training_enabled=False,calibration_enabled=False,
        partial_metric_release_enabled=False,manuscript_edits_enabled=False,user_runs_evaluation=True,
        historical_operational_flags_preserved=True,rule_changes_from_outer_results_allowed=False,
        retrospective_dataset=True,seed_ensembles=False,detector_class_override=False)
    io.write(c.PLAN,plan)
    extra=[c.PLAN,c.CLOSED_MANIFEST,c.CLOSED_MANIFEST.with_suffix('.sha256'),
        *[c.ROOT/p for p in sourcehashes],*[c.ROOT/p for p in inputs]]
    extra += [c.CLOSED/name for name in ('CIERRE_REAJUSTES_FINALES.json','MAPA_PARA_EVALUACION_EXTERIOR.json','INVENTARIO_MODELOS_FINALES_131.json')]
    extra += [c.ROOT/path for model in inventory for path in model['paths'].values()]
    extra += [p for p in c.PREP.rglob('*') if p.is_file() and p not in (c.MANIFEST,c.MANIFEST.with_suffix('.sha256'))]
    io.write_manifest(c.MANIFEST,extra=extra)
    print(json.dumps(dict(status=plan['status'],workers=4,expected_counts=counts,
        preparation_sha256=io.sha(c.MANIFEST),new_training_fits=0,real_outer_predictions=0)))


def same(a,b):
    import numpy as np
    return a.dtype==b.dtype and a.shape==b.shape and np.array_equal(a,b)


def verify_fold(campaign,job,plan):
    """Comprobar artefactos guardados sin volver a invocar un clasificador."""
    import numpy as np
    import worker_evaluacion as worker
    from metricas_evaluacion import supervised,detector,quantization_report
    from numerica_representantes import normalize_f32,integer_argmax
    folder=Path(campaign)/job['job_id']
    io.verify_manifest(folder/'MANIFEST_OUTER_SHA256.json',exact_directory=folder)
    result=io.read(folder/'RESULTADO.json')
    io.require(result['status']=='PASS' and result['job_id']==job['job_id']
        and result['outer_load_hp']==job['outer_load_hp']
        and result['selection_signature']==plan['selection_signature']
        and result['evaluation_signature']==plan['evaluation_signatures'][job['job_id']]
        and result['versions']==plan['expected_versions'] and result['source_sha256']==plan['source_sha256'],
        'Resultado ajeno al contrato de evaluación')
    for key in ('training_executed','parameter_fit_executed','threshold_calibrated','model_conversion_executed',
        'scalers_refitted','INT8_recalibrated','detector_threshold_refitted','mask_or_seed_selection_executed',
        'seed_ensembles','detector_class_override'):
        io.require(result[key] is False,'Operación no autorizada: '+key)
    snapshot=folder/'entrada_evaluacion.npz'
    io.require(io.sha(snapshot)==result['input_snapshot_sha256'],'Snapshot modificado')
    with np.load(snapshot,allow_pickle=False) as saved:
        io.require(set(saved.files)=={'raw_features14','y_true','global_window_ids','record_ids'},'Snapshot ajeno')
        raw=saved['raw_features14'];y=saved['y_true'];ids=saved['global_window_ids'];records=saved['record_ids']
    worker.validate_heldout(job,raw,y,records.tolist(),result['provenance'])
    io.require(ids.dtype==np.int64 and ids.tolist()==job['heldout_global_window_ids']
        and records.dtype.kind=='U','IDs o MAT distintos')
    expected_models=worker.model_snapshot(job['models'])
    io.require(result['model_hashes_before']==result['model_hashes_after']==expected_models
        and result['source_sha256_before']==result['source_sha256_after']==plan['source_sha256'],
        'La evaluación alteró modelos o fuentes')
    by_id={r['model_job_id']:r for r in result['model_results']}
    io.require(len(by_id)==len(result['model_results'])==len(job['models'])
        and set(by_id)=={m['job_id'] for m in job['models']},'Modelos evaluados incompletos o duplicados')
    for model in job['models']:
        row=by_id[model['job_id']]
        io.require(row['family']==model['family'] and row['seed']==model['seed']
            and row['fit_signature']==model['fit_signature'],'Modelo distinto del inventario')
        file=io.confined(folder,row['prediction_file'])
        io.require(file.parent==folder and io.sha(file)==row['prediction_sha256'],'Predicciones modificadas')
        with np.load(file,allow_pickle=False) as a:
            io.require(same(a['global_window_ids'],ids) and same(a['y_true'],y) and same(a['record_ids'],records),
                'Filas o etiquetas de predicciones diferentes')
            if model['family']=='DNN':
                scaler=worker.load_scaler(c.ROOT/model['paths']['scaler_train_only.npz'],model['indices'],True)
                z=normalize_f32(raw[:,model['indices']],scaler['mean32'],scaler['scale32'])
                io.require(same(a['normalized_features'],z) and same(a['fp32_inputs'],z),'Transformación DNN distinta')
                fp=integer_argmax(a['fp32_probabilities']);iq=integer_argmax(a['int8_probabilities'])
                io.require(same(a['fp32_outputs'],a['fp32_probabilities']) and same(a['fp32_predictions'],fp)
                    and same(a['int8_predictions'],iq),'Predicciones DNN inconsistentes')
                inputq=row['int8_details']['input_quantization'];outputq=row['int8_details']['output_quantization']
                io.require(same(a['input_quantization'],np.asarray(inputq,dtype=np.float64))
                    and same(a['output_quantization'],np.asarray(outputq,dtype=np.float64)),
                    'Parámetros de cuantización guardados distintos')
                quant=quantization_report(z,a['int8_inputs'],a['int8_outputs'],a['fp32_probabilities'],a['int8_probabilities'],
                    y,records.tolist(),inputq,outputq,ids.tolist())
                io.require(quant==row['quantization_report'] and row['metrics']==dict(FP32=supervised(y,fp,records.tolist()),
                    INT8=supervised(y,iq,records.tolist())),'Métricas o cuantización DNN distintas')
            elif model['family'] in ('RF','SVM'):
                z=raw
                if model['family']=='SVM':
                    scaler=worker.load_scaler(c.ROOT/model['paths']['scaler_train_only.npz'],list(range(14)),False)
                    z=normalize_f32(raw,scaler['mean32'],scaler['scale32'])
                io.require(same(a['model_features'],z) and a['values'].shape==(len(y),3)
                    and np.isfinite(a['values']).all() and row['native_predict_rule'] is True
                    and row['metrics']==supervised(y,a['predictions'],records.tolist()),'Métricas clásicas distintas')
            else:
                from datos_detector import load_params
                from detector_numerico import detector_score
                params=load_params(c.ROOT/model['paths']['parametros.npz'])
                scores=detector_score(raw,params)
                io.require(same(a['scores'],scores) and same(a['rejected'],scores>params['threshold32'])
                    and np.asarray(a['threshold32']).tobytes()==np.asarray(params['threshold32']).tobytes()
                    and row['metrics']==detector(y,a['rejected'],records.tolist()) and row['class_override'] is False,
                    'Scores, umbral o métricas detector diferentes')
    io.require(all(result['inference_counts'][k]==v for k,v in job['expected_inference_counts'].items()),
        'Conteos de inferencias diferentes')
    return result


def aggregate_results(campaign,plan):
    from metricas_evaluacion import aggregate_rows,paired_differences
    cells=[];detector_cells=[];quant=[]
    for job in plan['jobs']:
        result=io.read(Path(campaign)/job['job_id']/'RESULTADO.json')
        by_id={r['model_job_id']:r for r in result['model_results']}
        def add(procedure,precision,seed,row,metrics):
            cells.append(dict(procedure=procedure,precision=precision,seed=seed,outer_load_hp=job['outer_load_hp'],
                model_job_id=row['model_job_id'],prediction_file=str(Path(job['job_id'])/row['prediction_file']),**metrics))
        for name,procedure in job['procedure_mapping'].items():
            for seed,identifier in procedure['DNN_by_seed'].items():
                row=by_id[identifier]
                for mode in ('FP32','INT8'):add(name,mode,int(seed),row,row['metrics'][mode])
            row=by_id[procedure['detector_job_id']]
            detector_cells.append(dict(procedure=name,outer_load_hp=job['outer_load_hp'],seed=None,
                model_job_id=row['model_job_id'],threshold_float32=row['threshold_float32'],**row['metrics']))
        for seed,identifier in job['RF_by_seed'].items():add('RF','native',int(seed),by_id[identifier],by_id[identifier]['metrics'])
        row=by_id[job['SVM_job_id']];add('SVM','native',None,row,row['metrics'])
        quant.extend(dict(model_job_id=r['model_job_id'],outer_load_hp=job['outer_load_hp'],**r['quantization_report'])
            for r in result['model_results'] if r['family']=='DNN')
    io.require(len(cells)==284 and len(detector_cells)==12 and len(quant)==80,'Expansión de aliases incompleta')
    summaries=aggregate_rows(cells)
    paired=[paired_differences(cells,left,right,'INT8') for left,right in
        [('performance_only','baseline_F14'),('validated_total_cost','baseline_F14'),('validated_total_cost','performance_only')]]
    detector_summaries=[]
    for procedure in ('baseline_F14','performance_only','validated_total_cost'):
        selected=[r for r in detector_cells if r['procedure']==procedure]
        io.require(len(selected)==4,'Se requieren cuatro evaluaciones lógicas del detector')
        detector_summaries.append(dict(procedure=procedure,loads=4,seed_replications=0,
            **{key:math.fsum(r[key] for r in selected)/4 for key in ('normal_spec','ir_recall','or_recall','balanced_accuracy')},
            model_job_ids=[r['model_job_id'] for r in selected]))
    return dict(status='ALL_FOUR_HELDOUT_SCENARIOS_COMPLETE_RESULTS_RELEASED',cells=cells,summary=summaries,
        paired_INT8_differences=paired,detector_cells=detector_cells,detector_summary=detector_summaries,
        quantization_per_model=quant,quantization_totals=dict(unique_model_window_pairs=sum(r['windows'] for r in quant),
            class_changes=sum(r['class_changes'] for r in quant),clipped_feature_values=sum(r['input_clipped_feature_values'] for r in quant),
            clipped_window_evaluations=sum(r['input_clipped_window_count'] for r in quant)),
        selection_signature=plan['selection_signature'],retrospective_evaluation=True,
        seeds_are_not_independent_motors=True,no_IID_window_CI=True,training_executed=False,
        rules_changed_after_results=False,physical_STM32_latency_or_memory_measured=False)


def release_results(campaign,plan):
    done,_=inspect_campaign(campaign,plan)
    io.require(done==[j['job_id'] for j in plan['jobs']] and len(done)==4,'No liberar resultados parciales')
    output=aggregate_results(campaign,plan)
    io.write(Path(campaign)/'RESULTADOS_EXTERIORES.json',output)
    return output


def inspect_campaign(campaign,plan):
    campaign=Path(campaign).resolve()
    io.require(campaign.parent==c.OUTPUT.resolve() and io.read(campaign/'CONTRATO.json')==c.campaign_contract(plan),'Campaña ajena')
    done=[];consumed=0.;gap=False
    for index,job in enumerate(plan['jobs']):
        path=campaign/'pasos'/f'{index:02d}.json'
        if not path.exists():
            gap=True;io.require(not (campaign/job['job_id']).exists(),'Evaluación iniciada sin cierre; auditar');continue
        io.require(not gap,'Hueco en el prefijo de evaluaciones')
        io.verify_manifest(campaign/'pasos'/f'{index:02d}_MANIFEST.json')
        state=io.read(path);seconds=state['execution_verification_seconds']
        io.require(state['status']=='PASS' and state['job_id']==job['job_id'] and not state['automatic_retry']
            and state['timeout_seconds']==job['timeout_seconds'] and type(seconds) in (int,float)
            and math.isfinite(seconds) and seconds>=0,'Evaluación previa fallida; no reintentar')
        verify_fold(campaign,job,plan);done.append(job['job_id']);consumed+=seconds
    io.require(len(list((campaign/'pasos').glob('[0-9][0-9].json')))==len(done),'Pasos ajenos al plan')
    return done,consumed


def can_start(timeout,consumed,budget):return budget-consumed>=timeout


def launch_one(campaign,job,plan):
    started=time.monotonic();process=None;code=None;error=None;request=campaign/'SOLICITUD_ACTIVA.json'
    try:
        io.write(request,dict(job_id=job['job_id'],outer_load_hp=job['outer_load_hp'],
            evaluation_signature=plan['evaluation_signatures'][job['job_id']]))
        env=os.environ.copy();env.update(plan['worker_environment'])
        with (campaign/(job['job_id']+'_stdout.log')).open('x',encoding='utf-8') as out,\
             (campaign/(job['job_id']+'_stderr.log')).open('x',encoding='utf-8') as err:
            process=subprocess.Popen([sys.executable,'-B',str(c.SOURCE/'worker_evaluacion.py'),'--campaign',
                str(campaign),'--outer-load',str(job['outer_load_hp'])],cwd=c.ROOT,env=env,stdout=out,stderr=err)
            code=process.wait(timeout=job['timeout_seconds'])
        if code==0:verify_fold(campaign,job,plan)
        else:error='WORKER_EXIT_NONZERO'
    except subprocess.TimeoutExpired:error='TIMEOUT'
    except KeyboardInterrupt:error='INTERRUPTED'
    except Exception as exc:error=type(exc).__name__+': '+str(exc)
    finally:
        if process is not None and process.poll() is None:
            process.kill();process.wait(timeout=15);code=process.returncode
        if request.exists():request.unlink()
    return dict(status='PASS' if error is None and code==0 else 'FAIL',job_id=job['job_id'],
        return_code=code,error=error,timeout_seconds=job['timeout_seconds'],automatic_retry=False,
        execution_verification_seconds=time.monotonic()-started)


def execute():
    plan=c.preflight(environment=True);c.OUTPUT.mkdir(parents=True,exist_ok=True)
    marker=c.OUTPUT/'CAMPANA_UNICA.json'
    io.require(not marker.exists(),'Evaluación ya iniciada: no repetir automáticamente; revisar sus evidencias')
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    io.write(marker,dict(run_id=stamp,preparation_sha256=io.sha(c.MANIFEST)))
    campaign=c.OUTPUT/stamp;campaign.mkdir();(campaign/'pasos').mkdir()
    io.write(campaign/'CONTRATO.json',c.campaign_contract(plan))
    lock=campaign/'ACTIVO.lock'
    with lock.open('x',encoding='utf-8') as stream:stream.write(str(os.getpid()))
    consumed=0.;done=[];summary=None
    print('Campaña: '+str(campaign),flush=True)
    try:
        for index,job in enumerate(plan['jobs']):
            if not can_start(job['timeout_seconds'],consumed,plan['limits']['campaign_execution_verification_seconds']):
                raise ValueError('BUDGET_BOUNDARY: sin tiempo completo para otro proceso; conservar evidencias')
            print(f"[{index+1}/4] EVALUACION {job['job_id']}; límite {job['timeout_seconds']} s",flush=True)
            state=launch_one(campaign,job,plan);consumed+=state['execution_verification_seconds']
            path=campaign/'pasos'/f'{index:02d}.json';io.write(path,state)
            extra=[path]+[p for p in (campaign/job['job_id']).rglob('*') if p.is_file()]
            extra += [p for p in campaign.glob(job['job_id']+'_*log')]
            io.write_manifest(campaign/'pasos'/f'{index:02d}_MANIFEST.json',extra=extra)
            print(job['job_id']+': '+state['status']+'; métricas pendientes del cierre completo',flush=True)
            if state['status']!='PASS':raise ValueError('Evaluación detenida para auditoría: '+job['job_id'])
            done.append(job['job_id'])
        release_results(campaign,plan)
        summary=dict(status='OUTER_EVALUATION_COMPLETED',completed_loads=4,planned_loads=4,
            campaign_execution_verification_seconds=consumed,training_executed=False,
            all_models_and_decisions_fixed_before_evaluation=True,results_released=True,automatic_retry=False)
    except BaseException as error:
        summary=dict(status='STOPPED_FOR_AUDIT_NO_FINAL_RESULTS',completed_loads=len(done),planned_loads=4,
            campaign_execution_verification_seconds=consumed,error=type(error).__name__+': '+str(error),
            traceback=traceback.format_exc(),results_released=False,automatic_retry=False)
    finally:lock.unlink()
    io.write(campaign/'RESUMEN_CAMPANA.json',summary)
    io.write_manifest(campaign/'MANIFEST_CAMPANA_SHA256.json',directory=campaign,
        extra=[c.PLAN,c.MANIFEST,c.MANIFEST.with_suffix('.sha256'),c.CLOSED_MANIFEST,c.CLOSED_MANIFEST.with_suffix('.sha256')])
    print(summary['status']+f": {summary['completed_loads']}/4 cargas.",flush=True)
    print('Evidencias: '+str(campaign),flush=True)
    return 0 if summary['status']=='OUTER_EVALUATION_COMPLETED' else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_mutually_exclusive_group(required=True)
    for name in ('prepare','preflight','run'):modes.add_argument('--'+name,action='store_true')
    parser.add_argument('--check-environment',action='store_true');args=parser.parse_args()
    if args.prepare:prepare();return 0
    if args.preflight:
        plan=c.preflight(environment=args.check_environment)
        print(json.dumps(dict(status='OUTER_EVALUATION_READ_ONLY_PREFLIGHT_PASS',workers=len(plan['jobs']),
            real_outer_predictions=0,training_fits=0,TensorFlow_imported='tensorflow' in sys.modules)));return 0
    return execute()


if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception:print(traceback.format_exc(),file=sys.stderr);raise SystemExit(1)
