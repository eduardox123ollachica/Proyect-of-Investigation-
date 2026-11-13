"""Suplemento exploratorio acotado; entrenamientos solo con --execute del usuario."""
from __future__ import annotations
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[4]
SOURCE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'codigo/cwru/ampliacion_v2/representantes'))
from ejecutar_representantes import (read,write,sha,verify_manifest,write_manifest,versions,ENVIRONMENT,
    preflight as original_preflight,verify_job as original_verify_job,PREP as ORIGINAL_PREP)
from contrato_datos import jobs,context

PREP=ROOT/'articulo/ampliacion_cwru/metodologia/suplemento_preprocesamiento_normalidad_2026-10-25'
OUTPUT=ROOT/'resultados/cwru/ampliacion_v2/suplemento_preprocesamiento_normalidad_2026-10-25'
REFERENCE=ROOT/'resultados/cwru/ampliacion_v2/representantes_v2_2026-10-25/20261025T171847_957295Z'
DIAG=ROOT/'articulo/ampliacion_cwru/metodologia/diagnostico_representantes_v2_2026-10-25'
LIMITS=dict(DNN_worker_seconds=300,detector_block_seconds=60,campaign_process_seconds=7200,
            maximum_new_DNN_fits=34,maximum_new_conversions=68,maximum_new_detector_fits=5)


def require(ok,message):
    if not ok:raise ValueError(message)


def source_hashes():
    return {p.relative_to(ROOT).as_posix():sha(p) for p in sorted(SOURCE.glob('*.py'))}


def imported_references():
    """Importacion con firma ORIGINAL verificada, no un entrenamiento renombrado."""
    original,_=original_preflight()
    mapping={}
    for job in jobs():
        if job['mode']!='P0' or job['seed']!=11 or job['inner_validation_load_hp']!=0:continue
        old=next(j for j in original['jobs'] if j['outer_load_hp']==3 and j['candidate']==job['candidate'])
        for key in ('feature_indices','training_global_window_ids','validation_global_window_ids',
                    'detector_fit_global_window_ids','detector_calibration_global_window_ids','roles','seed'):
            require(old[key]==job[key],'Referencia con distinto scope: '+key)
        result=original_verify_job(REFERENCE/old['job_id'],old,original['fit_signatures'][old['job_id']])
        require(result['versions']==original['tested_versions'],'Referencia con versiones distintas')
        mapping[job['job_id']]=dict(cell_id=job['job_id'],source_job_id=old['job_id'],
            source_result=(REFERENCE/old['job_id']/'resultado.json').relative_to(ROOT).as_posix(),
            source_result_sha256=sha(REFERENCE/old['job_id']/'resultado.json'),
            source_fit_signature=result['fit_signature'],new_training_fits=0,
            method='verified_original_signature_and_identical_identity_preprocessing_roles',
            original_training_source_sha256=original['source_sha256']['codigo/cwru/ampliacion_v2/representantes/worker_representantes.py'])
    require(len(mapping)==2,'Se requieren dos referencias P0 conservadas')
    return mapping


def signature(job,source,version_info,protocol):
    value=dict(job=job,sources=source,versions=version_info,dnn=protocol['dnn'],
        preprocessing='P0 identity; P1 log1p(x/positive_training_median) at indices0,3,5,7,10,12; float32; allzero scale1',
        scaling=protocol['numeric_contract'],quantization=protocol['quantization'])
    return __import__('hashlib').sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode('utf-8')).hexdigest()


def prepare():
    protocol,_,_=context()
    tests=read(PREP/'PRUEBAS_PREPARACION.json')
    require(tests['status']=='PASS' and tests['failures']==tests['errors']==tests['skipped']==0
            and tests['source_sha256']==source_hashes(),'Pruebas incompletas o fuentes distintas')
    verify_manifest(DIAG/'MANIFEST_CIERRE_SHA256.json')
    imported=imported_references();items=jobs();source=source_hashes()
    original,_=original_preflight()
    require(tests['versions']==original['tested_versions'],'Entorno de pruebas distinto del entrenamiento original')
    plan=dict(status='PREPARED_FOR_USER_EXECUTION',scope='exploratory_development_supplement_outer3_only',
        jobs=items,imported_references=imported,source_sha256=source,tested_versions=tests['versions'],
        fit_signatures={j['job_id']:signature(j,source,tests['versions'],protocol) for j in items if j['job_id'] not in imported},
        dnn=protocol['dnn'],limits=LIMITS,environment=ENVIRONMENT,new_DNN_fits=34,imported_DNN_cells=2,
        total_DNN_cells=36,new_detector_fits=5,imported_detector_controls=1,
        default_new_fits_per_invocation=1,source_training_core='unchanged_sealed_representatives_worker_functions',
        main_protocol_changed=False,full_protocol_frozen=False,outer_evaluation_enabled=False,
        cost_selection_enabled=False,mask_or_preprocessing_selection_enabled=False,manuscript_edits_enabled=False,
        detector_block_after_all_DNN=True,automatic_retry=False,overwrite_allowed=False,
        detector_main_selector_replaced=False,observed_validation_cannot_change_this_plan=True)
    write(PREP/'PLAN_SUPLEMENTO.json',plan)
    extra=[ROOT/name for name in source]
    for path in (ORIGINAL_PREP/'MANIFEST_PREPARACION_SHA256.json',DIAG/'MANIFEST_CIERRE_SHA256.json'):
        extra.extend([path,path.with_suffix('.sha256')])
    write_manifest(PREP/'MANIFEST_PREPARACION_SHA256.json',PREP,extra)
    print('PREPARADO:36celdas,2importadas,34fitsnuevos;0fitsreales ejecutados por agente.')


def preflight():
    verify_manifest(PREP/'MANIFEST_PREPARACION_SHA256.json')
    plan=read(PREP/'PLAN_SUPLEMENTO.json');protocol,_,_=context()
    require(plan['jobs']==jobs() and plan['source_sha256']==source_hashes() and plan['limits']==LIMITS
            and plan['environment']==ENVIRONMENT and plan['dnn']==protocol['dnn']
            and plan['outer_evaluation_enabled'] is False and plan['cost_selection_enabled'] is False,
            'Contrato o fuentes distintos')
    require(plan['imported_references']==imported_references(),'Referencia importada modificada')
    signatures={j['job_id']:signature(j,plan['source_sha256'],plan['tested_versions'],protocol)
                for j in plan['jobs'] if j['job_id'] not in plan['imported_references']}
    require(signatures==plan['fit_signatures'],'Firmas de ajuste distintas')
    return plan


def validate_run(directory):
    directory=directory.resolve()
    require(directory.parent==OUTPUT.resolve() and directory.is_dir(),'Carpeta de ejecucion ajena')
    plan=preflight()
    require(read(directory/'CONTRATO.json')==dict(preparation_sha256=sha(PREP/'MANIFEST_PREPARACION_SHA256.json'),
        plan_sha256=sha(PREP/'PLAN_SUPLEMENTO.json'),scope=plan['scope'],execution_owner='user_CLI'),
        'Contrato de ejecucion distinto')
    return plan


def validate_result(result,job,expected_signature):
    require(result.get('status')=='PASS' and result.get('job_id')==job['job_id']
            and result.get('candidate')==job['candidate'] and result.get('mode')==job['mode']
            and result.get('seed')==job['seed'] and result.get('roles')==job['roles']
            and result.get('fit_signature')==expected_signature,'Resultado incompleto o ajeno')
    require(result.get('training_windows')==len(job['training_global_window_ids'])
            and result.get('validation_windows')==len(job['validation_global_window_ids'])
            and result.get('representative_calibration_rows')==len(job['training_global_window_ids'])
            and result.get('parameters')==20*job['input_dimension']+263,'Scope/arquitectura diferentes')
    require(type(result.get('best_epoch')) is int and 1<=result['best_epoch']<=result['epochs_executed']<=50
            and result['weights_restored'] is True and result['outer_predictive_evaluation'] is False
            and result['mask_or_preprocessing_selected'] is False and result['STM32_cost_measured'] is False,
            'Checkpoint/alcance incorrecto')
    require(0<=result['metrics']['INT8']['macro_f1']<=1,'F1 no valido')
    detail=result.get('int8_details')
    if detail is not None:
        require(detail['input_dtype']==detail['output_dtype']=='int8'
                and set(detail['tensor_dtypes'])<={'int8','int32','int64'}
                and detail['operators']==['FULLY_CONNECTED']*3+['SOFTMAX'], 'Resultado no full INT8')
    return result


def verify_new_job(directory,job,plan):
    import numpy as np
    import hashlib
    from numerica_representantes import metrics3,EarlyStopping,normalize_f32,quantize_int8
    from contrato_datos import load_data
    from preprocesamiento import fit_params,apply
    entries=verify_manifest(directory/'MANIFEST_JOB_SHA256.json')
    actual={p.relative_to(ROOT).as_posix() for p in directory.iterdir() if p.is_file()
            and p.name not in ('MANIFEST_JOB_SHA256.json','MANIFEST_JOB_SHA256.sha256')}
    require(set(entries)==actual,'Recursos faltantes o nuevos')
    result=validate_result(read(directory/'resultado.json'),job,plan['fit_signatures'][job['job_id']])
    require(result['versions']==plan['tested_versions'],'Versiones distintas')
    _,rows,_=context();index={int(r['global_window_id']):r for r in rows}
    y=np.asarray([int(index[k]['label']) for k in job['validation_global_window_ids']],dtype=np.int64)
    with np.load(directory/'predicciones_validacion.npz',allow_pickle=False) as saved:
        require(saved['global_window_ids'].tolist()==job['validation_global_window_ids']
                and np.array_equal(saved['y_true'],y),'IDs/etiquetas distintos')
        for family,key in (('FP32','fp32_predictions'),('INT8','int8_predictions')):
            require(metrics3(y,saved[key])==result['metrics'][family],'Metricas inconsistentes')
        saved_validation_inputs=saved['int8_inputs'].copy()
    state=read(directory/'estado_antes_conversion.json');stop=EarlyStopping()
    for row in state['history']:
        improved,_=stop.observe(row['epoch'],row['val_loss'])
        require(improved==row['improved'] and stop.best_epoch==row['best_epoch'],'Historia ES distinta')
    require(stop.best_epoch==state['best_epoch']==result['best_epoch']
            and len(state['history'])==state['epochs_executed']==result['epochs_executed']
            and state['optimizer_updates']==state['epochs_executed']*((len(job['training_global_window_ids'])+31)//32)
            and (state['epochs_executed']==50 or stop.stopped),'ES incompleto')
    require([json.loads(line) for line in (directory/'historial.jsonl').read_text(encoding='utf-8').splitlines()]
            ==state['history'],'Historia incremental distinta')
    digest=hashlib.sha256()
    with np.load(directory/'checkpoint_antes_conversion.npz',allow_pickle=False) as final, \
         np.load(directory/f"checkpoint_epoca_{state['best_epoch']:02d}.npz",allow_pickle=False) as best:
        require(final.files==best.files and len(final.files)==6
                and all(np.array_equal(final[k],best[k]) for k in final.files),'Pesos no restaurados')
        for key in final.files:
            require(np.isfinite(final[key]).all(),'Pesos no finitos')
            digest.update(str(final[key].shape).encode('ascii'));digest.update(final[key].tobytes())
    require(digest.hexdigest()==state['model_weights_sha256']==result['freeze']['model_weights_sha256'],
            'Hash de pesos restaurados distinto')
    for family,name in (('FP32','modelo_fp32.tflite'),('INT8','modelo_int8.tflite')):
        require(sha(directory/name)==result['model_sha256'][family],'Modelo modificado')
    raw,y_train,raw_val,y_val,provenance=load_data(job)
    require(read(directory/'procedencia.json')==provenance,'Procedencia no reproducible')
    params=fit_params(raw,job['mode'])
    with np.load(directory/'preprocesamiento.npz',allow_pickle=False) as saved:
        require(str(saved['mode'].item())==job['mode'] and saved['indices'].tolist()==params['indices']
                and np.array_equal(saved['scales64'],params['scales64'])
                and np.array_equal(saved['scales32'],params['scales32']), 'Preprocesamiento no train-only')
    train_processed,val_processed=apply(raw,params),apply(raw_val,params)
    ix=job['feature_indices']
    with np.load(directory/'scaler_train_only.npz',allow_pickle=False) as scaler:
        require(np.allclose(np.mean(train_processed[:,ix],axis=0,dtype=np.float64),scaler['mean64'],rtol=1e-12,atol=1e-12)
                and np.allclose(np.var(train_processed[:,ix].astype(np.float64),axis=0),scaler['var64'],rtol=1e-12,atol=1e-12)
                and np.array_equal(scaler['mean32'],scaler['mean64'].astype(np.float32))
                and np.array_equal(scaler['scale32'],scaler['scale64'].astype(np.float32)), 'Scaler no reproducible sobre train')
        x=normalize_f32(train_processed[:,ix],scaler['mean32'],scaler['scale32'])
        val=normalize_f32(val_processed[:,ix],scaler['mean32'],scaler['scale32'])
    q_train,clipped_train=quantize_int8(x,*result['int8_details']['input_quantization'])
    q_val,clipped_val=quantize_int8(val,*result['int8_details']['input_quantization'])
    with np.load(directory/'cuantizacion_train.npz',allow_pickle=False) as saved:
        require(saved['global_window_ids'].tolist()==job['training_global_window_ids']
                and np.array_equal(saved['int8_inputs'],q_train),'Cuantizacion de train distinta')
    require(np.array_equal(q_val,saved_validation_inputs) and clipped_train==result['training_clipped_values']
            and clipped_val==result['validation_clipped_values'],'Cuantizacion de validacion distinta')
    return result


def inspect_run(directory):
    plan=validate_run(directory);consumed=0.
    for batch in sorted((directory/'lotes').iterdir()):
        verify_manifest(batch/'MANIFEST_LOTE_SHA256.json');summary=read(batch/'RESUMEN_LOTE.json')
        require(summary['status'] in ('BATCH_COMPLETED','SUPPLEMENT_COMPLETED'),'Fallo previo: preservar y auditar')
        consumed+=summary['process_seconds']
    pending=[j for j in plan['jobs'] if j['job_id'] not in plan['imported_references']]
    done=[]
    for job in pending:
        folder=directory/job['job_id']
        if folder.exists():verify_new_job(folder,job,plan);done.append(job['job_id'])
    require(done==[j['job_id'] for j in pending[:len(done)]],'Huecos en los ajustes nuevos')
    return plan,pending,done,consumed


def process_one(directory,batch,job_id,timeout,detector=False):
    started=time.perf_counter();process=None;code=None;error=None
    with (batch/(job_id+'_stdout.log')).open('x',encoding='utf-8') as out, \
         (batch/(job_id+'_stderr.log')).open('x',encoding='utf-8') as err:
        try:
            if detector:
                command=[sys.executable,'-B',str(SOURCE/'detector_suplemento.py'),'--output-dir',str(directory/'detector')]
            else:
                command=[sys.executable,'-B',str(SOURCE/'worker_suplemento.py'),'--run-dir',str(directory),
                         '--batch',str(batch),'--job-id',job_id]
            env=os.environ.copy();env.update(ENVIRONMENT)
            process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=out,stderr=err)
            code=process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:error='TIMEOUT'
        except KeyboardInterrupt:error='INTERRUPTED'
        except OSError as exc:error='LAUNCH_ERROR: '+str(exc)
        finally:
            if process is not None and process.poll() is None:
                process.kill();process.wait(timeout=30);code=process.returncode
    if error is None and code==0:
        try:
            plan=read(PREP/'PLAN_SUPLEMENTO.json')
            if detector:
                summary=read(directory/'detector'/'RESUMEN_DETECTOR.json')
                require(summary['status']=='PASS' and len(summary['results'])==6
                        and summary['new_fit_calls']==5 and summary['imported_reference_fits']==1,
                        'Bloque detector incompleto')
            else:
                job=next(j for j in plan['jobs'] if j['job_id']==job_id)
                verify_new_job(directory/job_id,job,plan)
        except Exception as exc:error=type(exc).__name__+': '+str(exc)
    status=dict(job_id=job_id,status='PASS' if code==0 and error is None else 'FAIL',return_code=code,
                error=error,process_seconds=time.perf_counter()-started,timeout_seconds=timeout,automatic_retry=False)
    write(batch/('ESTADO_'+job_id+'.json'),status)
    return status


def execute(directory=None,limit=1):
    require(type(limit) is int and 1<=limit<=34,'Limite nuevo entre1 y34')
    plan=preflight();require(versions()==plan['tested_versions'],'Usar Python del proyecto validado')
    OUTPUT.mkdir(parents=True,exist_ok=True)
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    if directory is None:
        write(OUTPUT/'EJECUCION_UNICA.json',dict(run_id=stamp,preparation_sha256=sha(PREP/'MANIFEST_PREPARACION_SHA256.json')))
        directory=OUTPUT/stamp;directory.mkdir();(directory/'lotes').mkdir()
        write(directory/'CONTRATO.json',dict(preparation_sha256=sha(PREP/'MANIFEST_PREPARACION_SHA256.json'),
            plan_sha256=sha(PREP/'PLAN_SUPLEMENTO.json'),scope=plan['scope'],execution_owner='user_CLI'))
        write(directory/'REFERENCIAS_IMPORTADAS.json',plan['imported_references'])
    directory=directory.resolve();require(directory.parent==OUTPUT.resolve(),'Directorio ajeno')
    lock=directory/'ACTIVO.lock'
    with lock.open('x',encoding='utf-8') as stream:stream.write(str(os.getpid()))
    try:
        plan,pending,done,consumed=inspect_run(directory)
        require(len(done)<34,'Suplemento ya completo')
        selected=pending[len(done):len(done)+limit];batch=directory/'lotes'/stamp;batch.mkdir()
        write(batch/'PLAN_LOTE.json',dict(job_ids=[j['job_id'] for j in selected],completed_before=done,
            imported_references=list(plan['imported_references']),previous_process_seconds=consumed))
        print('Evidencias: '+str(directory),flush=True);statuses=[]
        for job in selected:
            remaining=LIMITS['campaign_process_seconds']-consumed
            if remaining<=0:break
            print('INICIO '+job['job_id'],flush=True)
            status=process_one(directory,batch,job['job_id'],min(300,remaining))
            statuses.append(status);consumed+=status['process_seconds']
            print(job['job_id']+': '+status['status'],flush=True)
            if status['status']!='PASS':break
        passed=sum(s['status']=='PASS' for s in statuses);new_complete=len(done)+passed
        success=passed==len(selected);detector_ok=False
        if success and new_complete==34:
            remaining=LIMITS['campaign_process_seconds']-consumed
            if remaining>0:
                status=process_one(directory,batch,'DETECTOR_BLOCK',min(60,remaining),detector=True)
                statuses.append(status);consumed+=status['process_seconds'];detector_ok=status['status']=='PASS'
            success=detector_ok
        summary=dict(status='SUPPLEMENT_COMPLETED' if new_complete==34 and detector_ok else
            'BATCH_COMPLETED' if success else 'STOPPED_FOR_AUDIT',new_fits_completed=new_complete,
            imported_cells=2,total_DNN_cells_available=new_complete+2,total_DNN_cells_planned=36,
            new_fits_this_batch=passed,process_seconds=sum(s['process_seconds'] for s in statuses),
            campaign_process_seconds=consumed,detector_complete=detector_ok,outer_evaluation_executed=False,
            selection_executed=False,STM32_cost_measured=False)
        write(batch/'RESUMEN_LOTE.json',summary)
        extra=[directory/'CONTRATO.json',directory/'REFERENCIAS_IMPORTADAS.json',PREP/'MANIFEST_PREPARACION_SHA256.json',
               PREP/'MANIFEST_PREPARACION_SHA256.sha256']
        for job in selected[:min(len(statuses),len(selected))]:
            if (directory/job['job_id']).is_dir():
                extra.extend(p for p in (directory/job['job_id']).iterdir() if p.is_file())
        if (directory/'detector').is_dir():extra.extend(p for p in (directory/'detector').rglob('*') if p.is_file())
        write_manifest(batch/'MANIFEST_LOTE_SHA256.json',batch,extra)
        print(f"{summary['status']}: {new_complete}/34 ajustes nuevos; {new_complete+2}/36 celdas disponibles (2 importadas).",flush=True)
        print('Evidencias: '+str(directory),flush=True)
        return 0 if success else 1
    finally:lock.unlink()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--prepare',action='store_true');mode.add_argument('--execute',action='store_true')
    parser.add_argument('--resume',type=Path);parser.add_argument('--limit',type=int,default=1);args=parser.parse_args()
    require(args.execute or(args.resume is None and args.limit==1),'Resume/limit requierenexecute')
    if args.prepare:prepare()
    elif args.execute:raise SystemExit(execute(args.resume,args.limit))
    else:
        plan=preflight();print(json.dumps(dict(status='READ_ONLY_PREFLIGHT_PASS',new_training_started=False,
            cells=36,imports=2,new_fits=34,limits=LIMITS)))
