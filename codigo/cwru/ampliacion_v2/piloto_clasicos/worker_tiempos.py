"""Trabajo de tiempo autorizado: RF/SVM o siete detectores; sin TensorFlow."""
from __future__ import annotations
import time
START=time.perf_counter()
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import traceback

ROOT=Path(__file__).resolve().parents[4]
OUTPUT=ROOT/'resultados/cwru/ampliacion_v2/piloto_clasicos_2026-10-25'


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def write(path,value):
    with path.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')


def run(run_dir,job_id):
    require(run_dir.resolve().parent==OUTPUT.resolve(),'Ejecucion fuera del proyecto')
    authorization=read(run_dir/'EJECUCION_ACOTADA.json')
    require(authorization['scope']=='classical_runtime_pilot_only'
        and authorization['mass_training_enabled'] is False
        and authorization['outer_predictive_evaluation_enabled'] is False,'Contrato incompatible')
    for name,digest in authorization['source_sha256'].items():
        path=(ROOT/name).resolve()
        require(path.is_relative_to(ROOT) and sha(path)==digest,'Fuente modificada: '+name)
    for key,value in authorization['worker_environment'].items():
        require(os.environ.get(key)==value,'Entorno no fijado: '+key)
    job=next(j for j in authorization['jobs'] if j['job_id']==job_id)
    directory=run_dir/job_id;directory.mkdir(exist_ok=False)
    times={k:0. for k in ('imports','data_and_scaler','artifact_IO')}
    if job['family']!='DETECTOR':times.update(fit=0.,inference=0.)
    def save(callback):
        start=time.perf_counter();callback();times['artifact_IO']+=time.perf_counter()-start
    def event(stage,**values):
        start=time.perf_counter()
        row=dict(stage=stage,elapsed_seconds=time.perf_counter()-START,**values)
        with (directory/'progreso.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True);times['artifact_IO']+=time.perf_counter()-start
    event('worker_started',family=job['family'])
    start=time.perf_counter()
    import numpy as np
    import sklearn
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    sys.path.insert(0,str(ROOT/'codigo/cwru/ampliacion_v2/piloto'))
    from datos_piloto import load_verified_plan,load_job_data,read_csv,PARTITIONS
    from detector_numerico import fit_normality,detector_score
    times['imports']=time.perf_counter()-start
    event('imports_complete',sklearn=sklearn.__version__)
    start=time.perf_counter()
    source_job=load_verified_plan()['jobs'][0]
    require(source_job['mask']=='F14','El manifiesto de filas debe ser F14')
    x,y,v,validation_labels,provenance=load_job_data(source_job)
    require(x.shape==(799,14) and v.shape==(175,14) and x.dtype==v.dtype==np.float32,'Matrices inesperadas')
    extra={}
    if job['family']=='SVM':
        scaler=StandardScaler().fit(x)
        mean,scale=scaler.mean_.astype(np.float32),scaler.scale_.astype(np.float32)
        x=np.divide(np.subtract(x,mean,dtype=np.float32),scale,dtype=np.float32)
        v=np.divide(np.subtract(v,mean,dtype=np.float32),scale,dtype=np.float32)
        require(np.isfinite(x).all() and np.isfinite(v).all(),'Normalizacion no finita')
        scaler_arrays=dict(mean64=scaler.mean_,scale64=scaler.scale_,var64=scaler.var_,mean32=mean,scale32=scale)
    if job['family']=='DETECTOR':
        metadata={int(r['global_window_id']):r for r in read_csv(PARTITIONS/'indice_ventanas_fuente.csv')}
        local={g:i for i,g in enumerate(source_job['train']['global_window_ids'])}
        fit_ids=source_job['detector_fit_role_only']['global_window_ids']
        cal_ids=source_job['detector_calibration_role_only']['global_window_ids']
        require(len(fit_ids)==len(cal_ids)==58 and set(fit_ids).isdisjoint(cal_ids)
            and all(metadata[g]['class']=='N' and int(metadata[g]['load_hp'])==1 for g in fit_ids)
            and all(metadata[g]['class']=='N' and int(metadata[g]['load_hp'])==2 for g in cal_ids),
            'Papeles del detector no coinciden')
        fit_positions=[local[g] for g in fit_ids];cal_positions=[local[g] for g in cal_ids]
        require(np.all(y[fit_positions]==0) and np.all(y[cal_positions]==0),'Detector incluye fallas al ajustar')
        fitting=x[fit_positions];calibration=x[cal_positions]
        cal_records=[metadata[g]['file_id'] for g in cal_ids]
        require({metadata[g]['file_id'] for g in fit_ids}.isdisjoint(cal_records),'MAT detector compartido')
        extra.update(fit_global_window_ids=fit_ids,calibration_global_window_ids=cal_ids,
                     refit_after_calibration=False)
    times['data_and_scaler']=time.perf_counter()-start
    save(lambda:write(directory/'procedencia.json',provenance))
    if job['family']=='SVM':save(lambda:np.savez(directory/'scaler_train_only.npz',**scaler_arrays))
    event('data_ready',train=799,validation=175,load3_arrays_opened=False)
    if job['family'] in ('RF','SVM'):
        model=(RandomForestClassifier if job['family']=='RF' else SVC)(**job['params'])
        start=time.perf_counter();model.fit(x,y);times['fit']=time.perf_counter()-start
        require(np.array_equal(model.classes_,[0,1,2]),'Clases de modelo incorrectas')
        if job['family']=='RF':require(len(model.estimators_)==job['params']['n_estimators'],'Bosque incompleto')
        else:require(model.fit_status_==0,'SVM no convergio')
        event('fit_complete')
        start=time.perf_counter()
        predictions=model.predict(v)
        diagnostics=model.predict_proba(v) if job['family']=='RF' else model.decision_function(v)
        require(predictions.shape==(175,) and set(predictions.tolist())<={0,1,2}
            and diagnostics.shape==(175,3) and np.isfinite(diagnostics).all(),'Inferencia no finita o forma incorrecta')
        times['inference']=time.perf_counter()-start
        save(lambda:np.savez(directory/'salidas_diagnostico.npz',predictions=predictions,values=diagnostics,
            validation_global_window_ids=np.asarray(source_job['validation']['global_window_ids'],dtype=np.int64)))
        extra.update(config_index=job['config_index'],requested_params=job['params'],
            model_classes=model.classes_.astype(int).tolist(),prediction_sha256=hashlib.sha256(predictions.tobytes()).hexdigest(),
            model_input_dtype='float32',normalization='raw_float32' if job['family']=='RF' else 'train_only_scaler_explicit_float32',
            validation_labels_used_for_metrics=False)
        if job['family']=='SVM':extra['support_vectors_dtype']=str(model.support_vectors_.dtype)
    else:
        rows=[];baseline={}
        for candidate in job['unique_candidates']:
            for rep in range(3):
                start=time.perf_counter()
                params=fit_normality(fitting,calibration,cal_records,job['indices_by_candidate'][candidate])
                fit_seconds=time.perf_counter()-start
                start=time.perf_counter();scores=detector_score(v,params);infer_seconds=time.perf_counter()-start
                numeric={k:params[k] for k in ('mean64','scale64','var64','mean32','scale32','centroid32','threshold32')}
                numeric['indices']=np.asarray(params['indices'],dtype=np.int32)
                digest=hashlib.sha256()
                for key in sorted(numeric):
                    value=np.asarray(numeric[key]);digest.update(key.encode());digest.update(str(value.dtype).encode());digest.update(value.tobytes())
                signature=(digest.hexdigest(),hashlib.sha256(scores.tobytes()).hexdigest())
                if rep==0:
                    baseline[candidate]=signature
                    save(lambda:np.savez(directory/f'parametros_{candidate}.npz',**numeric))
                    save(lambda:np.savez(directory/f'scores_{candidate}.npz',scores=scores,
                        validation_global_window_ids=np.asarray(source_job['validation']['global_window_ids'],dtype=np.int64)))
                require(signature==baseline[candidate],'Repeticiones detector no coinciden bit a bit')
                rows.append(dict(status='PASS',candidate=candidate,rep=rep,fit_calibration_seconds=fit_seconds,
                    inference_seconds=infer_seconds,params_sha256=signature[0],scores_sha256=signature[1],
                    threshold_bits=f"{int(np.asarray(params['threshold32'],dtype='<f4').view('<u4')):08x}"))
            event('detector_candidate_complete',candidate=candidate,repetitions=3)
        extra.update(rows=rows,aliases=job['aliases'],unique_representations=7,
            mask_requests=8,unique_fit_calls=21,fit_windows=58,calibration_windows=58,
            aliases_are_not_additional_measurements=True,repetitions_are_not_training_seeds=True)
    result=dict(status='PASS',job_id=job_id,family=job['family'],training_windows=799,validation_windows=175,
        timing_nonoverlapping_seconds=times,worker_elapsed_to_result_seconds=time.perf_counter()-START,
        versions=dict(python=platform.python_version(),numpy=np.__version__,sklearn=sklearn.__version__),
        metrics_used_for_selection=False,confirmatory_fit=False,no_load3_arrays_opened=True,
        no_TensorFlow_imported=True,no_STM32_latency_measured=True,source_data_job_id=source_job['job_id'],
        final_result_write_and_exit_not_in_stage_times=True,**extra)
    write(directory/'resultado.json',result)
    event('PASS')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--job-id',required=True,choices=[f'RF_{i:02d}' for i in range(8)]+[f'SVM_{i:02d}' for i in range(8)]+['DETECTOR'])
    args=parser.parse_args()
    try:run(args.run_dir,args.job_id)
    except Exception as error:
        if args.run_dir.resolve().parent==OUTPUT.resolve() and args.run_dir.is_dir():
            path=args.run_dir/('FALLO_'+args.job_id+'.json')
            if not path.exists():write(path,dict(status='FAIL',error_type=type(error).__name__,error=str(error),
                traceback=traceback.format_exc(),automatic_retry=False))
        raise
