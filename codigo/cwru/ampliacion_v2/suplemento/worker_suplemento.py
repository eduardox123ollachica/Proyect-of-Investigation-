"""Worker del suplemento: un ajuste nuevo, sin habilitar seleccion o outer."""
from __future__ import annotations
import argparse
from pathlib import Path
import json
import os
import sys
import time
import traceback

START=time.perf_counter()
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'codigo/cwru/ampliacion_v2/representantes'))


def require(ok,message):
    if not ok:raise ValueError(message)


def run(run_dir,batch,job_id):
    from ejecutar_suplemento import validate_run,ENVIRONMENT,write,versions,write_manifest,validate_result
    plan=validate_run(run_dir)
    require(batch.resolve().parent==run_dir/'lotes' and (run_dir/'ACTIVO.lock').is_file(),'Lote no activo')
    require(job_id in json.loads((batch/'PLAN_LOTE.json').read_text(encoding='utf-8'))['job_ids'], 'Celda no solicitada')
    require(job_id not in plan['imported_references'],'No reentrenar referencia importada')
    require(versions()==plan['tested_versions'] and all(os.environ.get(k)==v for k,v in ENVIRONMENT.items()),
            'Versiones/entorno no comprobados')
    job=next(j for j in plan['jobs'] if j['job_id']==job_id)
    directory=run_dir/job_id;directory.mkdir(exist_ok=False)

    def event(stage,**extra):
        row=dict(stage=stage,elapsed_seconds=time.perf_counter()-START,**extra)
        with (directory/'progreso.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(row,allow_nan=False)+'\n')
        print(json.dumps(row),flush=True)

    event('start',job_id=job_id)
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from contrato_datos import load_data
    from preprocesamiento import fit_params,apply
    from numerica_representantes import normalize_f32,quantize_int8,metrics3
    from worker_representantes import configure_tf,fit_network,export_models,evaluate_lite
    tf,keras=configure_tf(job['seed'])
    raw,y,raw_val,y_val,provenance=load_data(job)
    write(directory/'procedencia.json',provenance)
    params=fit_params(raw,job['mode'])
    processed,processed_val=apply(raw,params),apply(raw_val,params)
    ix=job['feature_indices']
    scaler=StandardScaler().fit(processed[:,ix])
    mean32,scale32=scaler.mean_.astype(np.float32),scaler.scale_.astype(np.float32)
    x,val=normalize_f32(processed[:,ix],mean32,scale32),normalize_f32(processed_val[:,ix],mean32,scale32)
    np.savez(directory/'preprocesamiento.npz',mode=np.asarray(job['mode']),indices=np.asarray(params['indices'],dtype=np.int64),
        scales64=params['scales64'],scales32=params['scales32'])
    write(directory/'decision_preprocesamiento.json',dict(mode=job['mode'],fit_scope='training_only',
        positive_training_median=(job['mode']=='P1'),all_zero_train_columns=[i for i in params['indices'] if bool((raw[:,i]==0).all())],
        all_zero_scale_policy=1.0,no_validation_statistics_used=True))
    np.savez(directory/'scaler_train_only.npz',mean64=scaler.mean_,scale64=scaler.scale_,var64=scaler.var_,
             mean32=mean32,scale32=scale32)
    event('data_ready',training=len(x),validation=len(val),mode=job['mode'])
    started=time.perf_counter()
    model,state=fit_network(tf,keras,x,y,val,y_val,plan['dnn'],job['seed'],directory,event)
    fit_seconds=time.perf_counter()-started
    fp32,integer,original,export=export_models(tf,model,x,val,directory,event)
    fp,fp_raw,_,fp_details=evaluate_lite(tf,fp32,val,False)
    iq,iq_raw,iq_inputs,iq_details=evaluate_lite(tf,integer,val,True)
    fp_pred,iq_pred=np.argmax(fp,axis=1),np.argmax(iq,axis=1)
    train_q,train_clip=quantize_int8(x,*iq_details['input_quantization'])
    np.savez(directory/'cuantizacion_train.npz',global_window_ids=np.asarray(job['training_global_window_ids']),int8_inputs=train_q)
    np.savez(directory/'predicciones_validacion.npz',global_window_ids=np.asarray(job['validation_global_window_ids']),
        y_true=y_val,keras_batch1=original,fp32_probabilities=fp,int8_probabilities=iq,fp32_outputs=fp_raw,
        int8_outputs=iq_raw,int8_inputs=iq_inputs,fp32_predictions=fp_pred,int8_predictions=iq_pred)
    result=dict(status='PASS',job_id=job_id,candidate=job['candidate'],mode=job['mode'],seed=job['seed'],
        inner=job['inner_validation_load_hp'],outer=3,roles=job['roles'],
        fit_signature=plan['fit_signatures'][job_id],imported=False,new_training_fits=1,
        best_epoch=state['best_epoch'],epochs_executed=state['epochs_executed'],weights_restored=True,
        training_windows=len(x),validation_windows=len(val),parameters=model.count_params(),
        metrics=dict(FP32=metrics3(y_val,fp_pred),INT8=metrics3(y_val,iq_pred)),
        class_changes=int((fp_pred!=iq_pred).sum()),training_clipped_values=train_clip,
        validation_clipped_values=iq_details['input_clipped_values'],
        validation_input_values=int(val.size),representative_calibration_rows=export['representative_rows'],
        freeze=export['freeze'],fp32_details=fp_details,int8_details=iq_details,
        model_sha256=dict(FP32=__import__('hashlib').sha256(fp32).hexdigest(),INT8=__import__('hashlib').sha256(integer).hexdigest()),
        model_bytes=dict(FP32=len(fp32),INT8=len(integer)),versions=versions(),
        training_fit_seconds=fit_seconds,elapsed_worker_seconds=time.perf_counter()-START,
        outer_predictive_evaluation=False,mask_or_preprocessing_selected=False,STM32_cost_measured=False,
        detector_preprocessing_changed=False)
    validate_result(result,job,plan['fit_signatures'][job_id])
    write(directory/'resultado.json',result);event('PASS',macro_f1_INT8=result['metrics']['INT8']['macro_f1'])
    write_manifest(directory/'MANIFEST_JOB_SHA256.json',directory)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True);parser.add_argument('--batch',type=Path,required=True)
    parser.add_argument('--job-id',required=True);args=parser.parse_args()
    try:run(args.run_dir.resolve(),args.batch.resolve(),args.job_id)
    except Exception:
        traceback.print_exc();raise SystemExit(1)
