"""Preparar y ejecutar reajustes finales; evaluación exterior deshabilitada."""
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

import contrato as c
io=c.io
WORKERS={'DNN':'worker_dnn.py','RF':'worker_clasicos.py','SVM':'worker_clasicos.py','detector':'worker_detector.py'}


def prepare():
    io.require(not c.PLAN.exists() and not c.MANIFEST.exists(),'Preparación existente; no sobrescribir')
    scopes,protocol,rows,hashes=c.frozen_contract()
    jobs=c.build_jobs(scopes,protocol,rows);sourcehashes=c.sources()
    tests=io.read(c.PREP/'PRUEBAS_PREPARACION_FINAL.json')
    synthetic=io.read(c.PREP/'PRUEBA_SINTETICA_REAL_INTENTO2.json')
    budget=io.read(c.PREP/'PRESUPUESTO_Y_LIMITES_FINAL.json')
    io.require(tests['status']=='PASS' and tests['errors']==tests['failures']==0
        and tests['source_sha256']==sourcehashes and tests['CWRU_training_fits']==0,
        'Faltan pruebas de las fuentes finales')
    io.require(synthetic['status']=='PASS' and synthetic['CWRU_arrays_opened'] is False
        and synthetic['source_sha256']==sourcehashes,'Falta prueba sintética de las fuentes finales')
    expected=c.versions()
    io.require(synthetic['versions']==expected and budget['limits']==c.LIMITS
        and budget['status']=='BUDGET_FROZEN_FOR_FINAL_REFITS_ONLY','Entorno/presupuesto diferentes')
    inputhashes={p:hashes[p] for p in sorted({p for j in jobs for p in j['authorized_feature_source_paths']})}
    for relative,digest in inputhashes.items():
        io.require(io.sha(io.confined(c.ROOT,relative))==digest,'Caché modificado antes de preparar')
    plan=dict(status='FINAL_REFIT_EXECUTOR_PREPARED_PENDING_USER_RUN',jobs=jobs,counts=dict(DNN=80,RF=40,SVM=4,detector=7),
        selection_signature=c.SELECTION_SIGNATURE,source_sha256=sourcehashes,input_sha256=inputhashes,
        expected_versions=expected,limits=c.LIMITS,worker_environment=io.ENVIRONMENT,
        fit_signatures={j['job_id']:c.fit_signature(j,protocol,hashes,sourcehashes,expected) for j in jobs},
        user_starts_training=True,outer_evaluation_enabled=False,outer_prediction_release_enabled=False,
        manuscript_edits_enabled=False,early_stopping=False,automatic_retry=False,
        default_max_new_jobs=1,default_max_seconds=600,all_training_is_retrospective=True,
        train_diagnostics_are_not_generalization_metrics=True)
    io.write(c.PLAN,plan)
    extra=[c.PLAN,*[c.ROOT/p for p in sourcehashes],*[c.ROOT/p for p in inputhashes],
        c.FROZEN/'MANIFEST_SELECCION_CONGELADA_SHA256.json',c.FROZEN/'MANIFEST_SELECCION_CONGELADA_SHA256.sha256',
        c.FROZEN/'CUERPO_FIRMA_SELECCION.json',c.FROZEN/'DECISIONES_CONGELADAS.json',c.FROZEN/'PLAN_REAJUSTES_PENDIENTES.json']
    extra += [p for p in c.PREP.rglob('*') if p.is_file() and p not in (c.MANIFEST,c.MANIFEST.with_suffix('.sha256'))]
    extra += [c.ROOT/p for p in budget['source_sha256']]
    io.write_manifest(c.MANIFEST,extra=extra)
    print(json.dumps(dict(status=plan['status'],jobs=len(jobs),counts=plan['counts'],new_CWRU_fits=0,
        preparation_sha256=io.sha(c.MANIFEST))))


def payload_for(job,plan,folder,context):
    provenance=c.materialize(job,folder,context)
    protocol=context[0]
    return dict(status='AUTHORIZED_FINAL_REFIT_PAYLOAD',family=job['family'],job=job,
        arrays_path=str((folder/'datos.npz').resolve()),arrays_sha256=io.sha(folder/'datos.npz'),
        fit_signature=plan['fit_signatures'][job['job_id']],selection_signature=c.SELECTION_SIGNATURE,
        source_sha256=plan['source_sha256'],expected_versions=plan['expected_versions'],provenance=provenance,
        worker_environment=plan['worker_environment'],**{k:protocol[k] for k in ('dnn','numeric_contract','quantization','detector')})


def _same(a,b):
    import numpy as np
    return a.dtype==b.dtype and a.shape==b.shape and np.array_equal(a,b)


def verify_job(campaign,job,plan):
    """Autenticar artefactos y reconstruir diagnósticos exclusivamente de desarrollo."""
    import numpy as np
    folder=campaign/job['job_id'];payload_path=campaign/'solicitudes'/job['job_id']/'payload.json'
    payload=io.read(payload_path)
    io.require(payload['job']==job and payload['fit_signature']==plan['fit_signatures'][job['job_id']]
        and io.sha(payload['arrays_path'])==payload['arrays_sha256'],'Payload modificado')
    io.verify_manifest(folder/'MANIFEST_JOB_SHA256.json',exact_directory=folder)
    result=io.read(folder/'RESULTADO.json')
    io.require(result['status']=='PASS' and result['job_id']==job['job_id'] and result['family']==job['family']
        and result['roles']==job['roles'] and result['outer_load_hp']==job['outer_load_hp']
        and result['selection_signature']==c.SELECTION_SIGNATURE
        and result['fit_signature']==plan['fit_signatures'][job['job_id']]
        and result['source_sha256']==plan['source_sha256']
        and not result['outer_evaluation_executed'],'Resultado ajeno al reajuste o evaluación exterior presente')
    io.require(all(plan['expected_versions'].get(k)==v for k,v in result['versions'].items()),'Versiones del resultado distintas')
    io.require(io.read(folder/'procedencia.json')==payload['provenance'],'Procedencia alterada')
    with np.load(payload['arrays_path'],allow_pickle=False) as saved:
        arrays={key:saved[key] for key in saved.files}
    family=job['family']
    if family=='DNN':
        import worker_dnn
        from numerica_representantes import balanced_weights,integer_argmax,metrics3,normalize_f32,quantize_int8
        from sklearn.preprocessing import StandardScaler
        raw,y,ids,records=worker_dnn.validate_payload(payload,arrays)
        scaler=StandardScaler().fit(raw[:,job['feature_indices']])
        with np.load(folder/'scaler_train_only.npz',allow_pickle=False) as saved:
            for key,expected in dict(mean64=scaler.mean_,scale64=scaler.scale_,var64=scaler.var_,
                mean32=scaler.mean_.astype(np.float32),scale32=scaler.scale_.astype(np.float32),
                feature_indices=np.asarray(job['feature_indices'],dtype=np.int64)).items():
                io.require(_same(saved[key],expected),'Scaler DNN no reconstruible: '+key)
        x=normalize_f32(raw[:,job['feature_indices']],scaler.mean_.astype(np.float32),scaler.scale_.astype(np.float32))
        state=io.read(folder/'estado_antes_conversion.json');epochs=job['epochs'];updates=(len(y)+31)//32
        io.require(state['epochs_executed']==result['epochs_executed']==epochs
            and state['final_epoch']==result['final_epoch']==epochs and len(state['history'])==epochs
            and state['optimizer_updates']==result['optimizer_updates']==epochs*updates
            and state['tracing_updates']==0 and not any(state[k] for k in ('weights_restored','early_stopping_used','validation_used')),
            'Épocas fijas/actualizaciones distintas')
        for index,row in enumerate(state['history'],1):
            io.require(row['epoch']==index and row['optimizer_updates']==index*updates and math.isfinite(row['loss']),
                'Historia de entrenamiento inválida')
        digest=__import__('hashlib').sha256()
        with np.load(folder/'checkpoint_antes_conversion.npz',allow_pickle=False) as saved:
            shapes=[(job['input_dimension'],20),(20,),(20,10),(10,),(10,3),(3,)]
            io.require(saved.files==[f'weight_{i:02d}' for i in range(6)],'Checkpoint incompleto')
            for key,shape in zip(saved.files,shapes):
                a=saved[key];io.require(a.dtype==np.float32 and a.shape==shape and np.isfinite(a).all(),'Pesos inválidos')
                digest.update(str(a.shape).encode('ascii'));digest.update(a.tobytes())
        io.require(digest.hexdigest()==state['model_weights_sha256']==result['final_weights_sha256'],'Huella de pesos incorrecta')
        counts,weights,_=balanced_weights(y)
        io.require(result['class_counts']==counts.tolist() and result['class_weights']==weights.tolist()
            and result['representative_calibration_rows']==len(y) and result['training_windows']==len(y)
            and result['parameters']==20*job['input_dimension']+263,'Scope/cuentas DNN distintos')
        for key,name in [('Keras','modelo.keras'),('FP32','modelo_fp32.tflite'),('INT8','modelo_int8.tflite')]:
            io.require(io.sha(folder/name)==result['model_sha256'][key],'Modelo DNN modificado: '+key)
        with np.load(folder/'diagnosticos_train_only.npz',allow_pickle=False) as diag:
            q,clipped=quantize_int8(x,*result['int8_details']['input_quantization'])
            fp=integer_argmax(diag['fp32_probabilities']);iq=integer_argmax(diag['int8_probabilities'])
            io.require(_same(diag['global_window_ids'],ids) and _same(diag['y_true'],y)
                and _same(diag['int8_inputs'],q) and clipped==result['training_saturated_input_values']
                and _same(diag['fp32_predictions'],fp) and _same(diag['int8_predictions'],iq)
                and result['training_only_diagnostics']==dict(FP32=metrics3(y,fp),INT8=metrics3(y,iq)),
                'Diagnósticos train DNN no reconstruibles')
    elif family in ('RF','SVM'):
        import joblib
        import worker_clasicos as worker
        a=worker.validate_payload(payload,arrays);x,scaler=worker.prepare_training(family,a['raw_training'])
        io.require(result['params']==job['params'] and result['seed']==job['seed']
            and result['training_global_window_ids']==job['training_global_window_ids'],'Parámetros clásicos diferentes')
        if scaler is not None:
            with np.load(folder/'scaler_train_only.npz',allow_pickle=False) as saved:
                io.require(set(saved.files)==set(scaler) and all(_same(saved[k],v) for k,v in scaler.items()),'Scaler SVM distinto')
        io.require(io.sha(folder/'modelo.joblib')==result['model_sha256'],'Modelo clásico alterado')
        model=joblib.load(folder/'modelo.joblib')
        io.require(all(model.get_params(deep=False).get(k)==v for k,v in job['params'].items())
            and np.array_equal(model.classes_,[0,1,2]) and model.n_features_in_==14,'Estimador ajeno a parámetros')
        if family=='RF':io.require(len(model.estimators_)==job['params']['n_estimators'],'RF incompleto')
        else:io.require(model.fit_status_==0,'SVM no convergió')
        pred,values=worker.training_prediction(model,family,x)
        with np.load(folder/'diagnostico_entrenamiento.npz',allow_pickle=False) as saved:
            io.require(_same(saved['global_window_ids'],a['training_ids']) and _same(saved['y_true'],a['y_training'])
                and _same(saved['predictions'],pred) and _same(saved['values'],values),'Replay clásico distinto')
    else:
        import worker_detector as worker
        from datos_detector import load_params
        from detector_numerico import fit_normality,detector_score
        a=worker.validate_payload(payload,arrays)
        params=load_params(folder/'parametros.npz')
        expected=fit_normality(a['raw_fit'],a['raw_calibration'],a['calibration_records'].tolist(),job['indices'])
        io.require(params['indices']==job['indices'] and params['fit_ids'].tolist()==job['fit_ids']
            and params['calibration_ids'].tolist()==job['calibration_ids'],'Ámbito detector distinto')
        for key in ('mean64','scale64','var64','mean32','scale32','centroid32','threshold32'):
            io.require(_same(np.asarray(params[key]),np.asarray(expected[key])),'Reconstrucción detector distinta: '+key)
        scores=detector_score(a['raw_calibration'],params)
        with np.load(folder/'calibracion.npz',allow_pickle=False) as saved:
            io.require(_same(saved['global_window_ids'],a['calibration_ids']) and _same(saved['scores'],scores)
                and _same(saved['rejected'],scores>params['threshold32']),'Calibración no reconstruible')
        io.require(worker.parameter_payload_sha256(folder/'parametros.npz')==result['parameter_payload_sha256']
            and result['refit_after_calibration'] is False,'Payload detector alterado/reajustado')
    return result


def inspect_campaign(campaign,plan):
    campaign=Path(campaign).resolve()
    io.require(campaign.parent==c.OUTPUT.resolve() and io.read(campaign/'CONTRATO.json')==c.campaign_contract(plan),
        'Campaña fuera del contrato')
    done=[];consumed=0.;gap=False
    for index,job in enumerate(plan['jobs']):
        state_path=campaign/'pasos'/f'{index:03d}.json'
        if not state_path.exists():
            gap=True
            io.require(not (campaign/job['job_id']).exists()
                and not (campaign/'solicitudes'/job['job_id']).exists(),'Trabajo iniciado sin cierre; auditar antes de continuar')
            continue
        io.require(not gap,'Hueco en prefijo de trabajos completos')
        io.verify_manifest(campaign/'pasos'/f'{index:03d}_MANIFEST.json')
        state=io.read(state_path)
        seconds=state['execution_verification_seconds']
        io.require(state['status']=='PASS' and state['job_id']==job['job_id'] and state['automatic_retry'] is False
            and state['timeout_seconds']==job['timeout_seconds'] and type(seconds) in (int,float)
            and math.isfinite(seconds) and seconds>=0,'Fallo previo/contador inválido; sin reintentos automáticos')
        verify_job(campaign,job,plan);done.append(job['job_id']);consumed+=seconds
    io.require(len(list((campaign/'pasos').glob('[0-9][0-9][0-9].json')))==len(done),'Pasos ajenos')
    return done,consumed


def can_start(timeout,consumed,budget):
    return budget-consumed>=timeout


def launch_one(campaign,job,plan,context):
    started=time.monotonic();process=None;error=None;code=None
    request=campaign/'SOLICITUD_ACTIVA.json'
    try:
        folder=campaign/'solicitudes'/job['job_id'];folder.mkdir(exist_ok=False)
        payload=payload_for(job,plan,folder,context);payload_path=folder/'payload.json';io.write(payload_path,payload)
        io.write(request,dict(job_id=job['job_id'],fit_signature=plan['fit_signatures'][job['job_id']],
            payload_path=payload_path.relative_to(c.ROOT).as_posix()))
        env=os.environ.copy();env.update(plan['worker_environment'])
        with (campaign/(job['job_id']+'_stdout.log')).open('x',encoding='utf-8') as stdout,\
             (campaign/(job['job_id']+'_stderr.log')).open('x',encoding='utf-8') as stderr:
            process=subprocess.Popen([sys.executable,'-B',str(c.SOURCE/WORKERS[job['family']]),'--payload',
                str(payload_path),'--output',str(campaign/job['job_id'])],cwd=c.ROOT,env=env,stdout=stdout,stderr=stderr)
            code=process.wait(timeout=job['timeout_seconds'])
        if code==0:verify_job(campaign,job,plan)
        else:error='WORKER_EXIT_NONZERO'
    except subprocess.TimeoutExpired:error='TIMEOUT'
    except KeyboardInterrupt:error='INTERRUPTED'
    except Exception as exc:error=type(exc).__name__+': '+str(exc)
    finally:
        if process is not None and process.poll() is None:
            process.kill();process.wait(timeout=15);code=process.returncode
        if request.exists():request.unlink()
    return dict(status='PASS' if error is None and code==0 else 'FAIL',job_id=job['job_id'],family=job['family'],
        return_code=code,error=error,execution_verification_seconds=time.monotonic()-started,
        timeout_seconds=job['timeout_seconds'],automatic_retry=False)


def execute(max_new_jobs=1,max_seconds=600):
    plan=c.preflight(environment=True)
    io.require(type(max_new_jobs) is int and 1<=max_new_jobs<=c.LIMITS['max_new_jobs'],'Cantidad fuera de límites')
    io.require(type(max_seconds) is int and 300<=max_seconds<=c.LIMITS['max_batch_seconds'],'Tiempo fuera de límites')
    c.OUTPUT.mkdir(parents=True,exist_ok=True)
    marker=c.OUTPUT/'CAMPANA_UNICA.json'
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    if not marker.exists():
        io.write(marker,dict(run_id=stamp,preparation_sha256=io.sha(c.MANIFEST)))
        campaign=c.OUTPUT/stamp;campaign.mkdir()
        for name in ('pasos','lotes','solicitudes'):(campaign/name).mkdir()
        io.write(campaign/'CONTRATO.json',c.campaign_contract(plan))
    else:
        info=io.read(marker);campaign=io.confined(c.OUTPUT,info['run_id'])
        io.require(info['preparation_sha256']==io.sha(c.MANIFEST),'Marcador de otra preparación')
    lock=campaign/'ACTIVO.lock'
    with lock.open('x',encoding='utf-8') as stream:stream.write(str(os.getpid()))
    print('Campaña: '+str(campaign),flush=True)
    summary=None
    try:
        io.require(not (campaign/'SOLICITUD_ACTIVA.json').exists(),'Solicitud activa/incompleta anterior')
        for previous in (campaign/'lotes').iterdir():
            io.verify_manifest(previous/'MANIFEST_LOTE_SHA256.json')
            io.require(io.read(previous/'RESUMEN_LOTE.json')['status'] in ('BATCH_COMPLETED','BATCH_BUDGET_BOUNDARY'),
                'Campaña completa/fallida/interrumpida; no repetir')
        done,consumed=inspect_campaign(campaign,plan)
        io.require(len(done)<len(plan['jobs']),'Campaña ya completa; no repetir')
        batch=campaign/'lotes'/stamp;batch.mkdir()
        io.write(batch/'PLAN_LOTE.json',dict(completed_before=done,max_new_jobs=max_new_jobs,max_seconds=max_seconds,
            campaign_execution_verification_seconds_before=consumed))
        started=time.monotonic();statuses=[];boundary=False
        context=c.inner.context()
        for index,job in enumerate(plan['jobs'][len(done):len(done)+max_new_jobs],len(done)):
            if not can_start(job['timeout_seconds'],time.monotonic()-started,max_seconds) or not can_start(
                job['timeout_seconds'],consumed,c.LIMITS['campaign_execution_verification_seconds']):
                boundary=True;break
            print(f"[{index+1}/131] INICIO {job['job_id']}; {job['family']}; límite {job['timeout_seconds']} s",flush=True)
            state=launch_one(campaign,job,plan,context);statuses.append(state)
            io.write(campaign/'pasos'/f'{index:03d}.json',state)
            consumed+=state['execution_verification_seconds']
            extra=[campaign/'pasos'/f'{index:03d}.json']
            extra += [p for name in ('solicitudes/'+job['job_id'],job['job_id'])
                for p in (campaign/name).rglob('*') if p.is_file()]
            extra += [p for p in campaign.glob(job['job_id']+'_*log')]
            io.write_manifest(campaign/'pasos'/f'{index:03d}_MANIFEST.json',extra=extra)
            print(job['job_id']+': '+state['status'],flush=True)
            if state['status']!='PASS':break
        passed=sum(s['status']=='PASS' for s in statuses);complete=len(done)+passed==len(plan['jobs'])
        status='FINAL_REFITS_COMPLETED_NO_OUTER_EVALUATION' if complete else 'STOPPED_FOR_AUDIT' if passed!=len(statuses) else (
            'BATCH_BUDGET_BOUNDARY' if boundary else 'BATCH_COMPLETED')
        summary=dict(status=status,completed_jobs=len(done)+passed,planned_jobs=131,remaining_jobs=131-len(done)-passed,
            jobs_started=len(statuses),jobs_passed=passed,batch_execution_verification_seconds=sum(s['execution_verification_seconds'] for s in statuses),
            campaign_execution_verification_seconds=consumed,batch_wall_seconds=time.monotonic()-started,
            outer_evaluation_executed=False,outer_prediction_release_enabled=False,automatic_retry=False)
        io.write(batch/'RESUMEN_LOTE.json',summary)
        io.write_manifest(batch/'MANIFEST_LOTE_SHA256.json',directory=batch,extra=[campaign/'CONTRATO.json',c.MANIFEST,
            c.MANIFEST.with_suffix('.sha256'),*[campaign/'pasos'/f'{i:03d}_MANIFEST.json' for i in range(len(done),len(done)+len(statuses))]])
        print(f"{status}: {summary['completed_jobs']}/131 trabajos completos.",flush=True)
        print('Evidencias: '+str(campaign),flush=True)
        return 1 if status=='STOPPED_FOR_AUDIT' else 0
    finally:lock.unlink()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_mutually_exclusive_group(required=True)
    for mode in ('prepare','preflight','run'):modes.add_argument('--'+mode,action='store_true')
    parser.add_argument('--check-environment',action='store_true')
    parser.add_argument('--max-new-jobs',type=int,default=1);parser.add_argument('--max-seconds',type=int,default=600)
    args=parser.parse_args()
    io.require(args.run or (args.max_new_jobs==1 and args.max_seconds==600),'Límites solo con --run')
    if args.prepare:prepare();return 0
    if args.preflight:
        plan=c.preflight(environment=args.check_environment)
        print(json.dumps(dict(status='FINAL_REFIT_READ_ONLY_PREFLIGHT_PASS',counts=plan['counts'],CWRU_training_fits=0,
            TensorFlow_imported='tensorflow' in sys.modules)));return 0
    return execute(args.max_new_jobs,args.max_seconds)


if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception:print(traceback.format_exc(),file=sys.stderr);raise SystemExit(1)
