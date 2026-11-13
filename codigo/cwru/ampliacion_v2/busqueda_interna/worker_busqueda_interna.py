"""Worker DNN interno: funciones de entrenamiento/exportacion originales.

Importar este modulo no importa TensorFlow. Solo run() desde un lote activo
puede entrenar. No ajusta detector ni evalua la carga externa.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

START = time.perf_counter()
ROOT = Path(__file__).resolve().parents[4]
LEGACY = ROOT/'codigo/cwru/ampliacion_v2/representantes'
if str(LEGACY) not in sys.path:
    sys.path.insert(0,str(LEGACY))
from worker_representantes import configure_tf,fit_network,export_models,evaluate_lite


def require(ok, message):
    if not ok:
        raise ValueError(message)


def run(campaign,batch,job_id):
    from ejecutar_busqueda_interna import (validate_campaign,write,write_manifest,validate_result,
                                           ENVIRONMENT,legacy,PREP_MANIFEST,sha)
    plan = validate_campaign(campaign,cheap=True)
    job = next((j for j in plan['jobs'] if j['job_id'] == job_id),None)
    require(job is not None and job_id not in plan['imported'],'Worker no puede repetir una celda importada')
    require(batch.resolve().parent == campaign/'lotes' and (campaign/'EJECUCION_EN_CURSO.lock').is_file(),
            'Worker requiere lote activo y confinado')
    batch_plan = json.loads((batch/'PLAN_LOTE.json').read_text(encoding='utf-8'))
    require(job_id in batch_plan['job_ids'] and not (batch/('ESTADO_'+job_id+'.json')).exists(),
            'Ajuste no autorizado o ya cerrado')
    require(legacy.versions() == plan['tested_versions'],'Entorno Python distinto del preparado')
    require(all(os.environ.get(k) == v for k,v in ENVIRONMENT.items()),'Entorno determinista no fijado')
    directory = campaign/job_id
    directory.mkdir(exist_ok=False)
    def event(stage,**extra):
        row = dict(stage=stage,elapsed_seconds=time.perf_counter()-START,**extra)
        with (directory/'progreso.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(row,allow_nan=False)+'\n')
        print(json.dumps(row),flush=True)
    event('start',job_id=job_id)
    started=time.perf_counter()
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from datos_busqueda import load_job_data
    from numerica_representantes import normalize_f32,metrics3,integer_argmax,quantize_int8
    tf,keras=configure_tf(job['seed'])
    import_seconds=time.perf_counter()-started
    event('imports_complete',versions=legacy.versions())
    started=time.perf_counter()
    raw,y,raw_val,y_val,provenance=load_job_data(job)
    write(directory/'procedencia.json',provenance)
    indices=job['feature_indices']
    scaler=StandardScaler().fit(raw[:,indices])
    mean,scale=scaler.mean_.astype(np.float32),scaler.scale_.astype(np.float32)
    x,val=normalize_f32(raw[:,indices],mean,scale),normalize_f32(raw_val[:,indices],mean,scale)
    np.savez(directory/'scaler_train_only.npz',mean64=scaler.mean_,scale64=scaler.scale_,var64=scaler.var_,
             mean32=mean,scale32=scale)
    data_seconds=time.perf_counter()-started
    event('data_ready',training=len(x),validation=len(val))
    started=time.perf_counter()
    model,state=fit_network(tf,keras,x,y,val,y_val,plan['dnn'],job['seed'],directory,event)
    fit_seconds=time.perf_counter()-started
    event('best_checkpoint_restored',best_epoch=state['best_epoch'])
    fp32,int8,original,export=export_models(tf,model,x,val,directory,event)
    started=time.perf_counter()
    fp,fp_raw,_,fp_details=evaluate_lite(tf,fp32,val,False)
    iq,iq_raw,iq_inputs,iq_details=evaluate_lite(tf,int8,val,True)
    train_quantized,train_clipped=quantize_int8(x,*iq_details['input_quantization'])
    np.savez(directory/'cuantizacion_train.npz',global_window_ids=np.asarray(job['training_global_window_ids'],dtype=np.int64),
             int8_inputs=train_quantized)
    fp_pred,iq_pred=integer_argmax(fp),integer_argmax(iq)
    metrics=dict(FP32=metrics3(y_val,fp_pred),INT8=metrics3(y_val,iq_pred))
    np.savez(directory/'predicciones_validacion.npz',y_true=y_val,
        global_window_ids=np.asarray(job['validation_global_window_ids'],dtype=np.int64),
        keras_batch1=original,fp32_probabilities=fp,int8_probabilities=iq,
        fp32_outputs=fp_raw,int8_outputs=iq_raw,int8_inputs=iq_inputs,
        fp32_predictions=fp_pred,int8_predictions=iq_pred)
    inference_seconds=time.perf_counter()-started
    result=dict(status='PASS',job_id=job_id,fit_signature=plan['fit_signatures'][job_id],
        semantic_signature=plan['semantic_signatures'][job_id],candidate=job['candidate'],
        outer_load_hp=job['outer_load_hp'],inner_validation_load_hp=job['inner_validation_load_hp'],seed=job['seed'],
        purpose='fixed_inner_DNN_search_only',roles=job['roles'],training_windows=len(x),validation_windows=len(val),
        best_epoch=state['best_epoch'],epochs_executed=state['epochs_executed'],optimizer_updates=state['optimizer_updates'],
        weights_restored=True,architecture=[len(indices),20,10,3],parameters=model.count_params(),
        representative_calibration_rows=export['representative_rows'],freeze=export['freeze'],
        model_sha256=dict(FP32=hashlib.sha256(fp32).hexdigest(),INT8=hashlib.sha256(int8).hexdigest()),
        model_bytes=dict(FP32=len(fp32),INT8=len(int8)),metrics=metrics,
        FP32_INT8_class_changes=int(np.count_nonzero(fp_pred != iq_pred)),
        keras_batch1_FP32_class_changes=int(np.count_nonzero(integer_argmax(original) != fp_pred)),
        fp32_details=fp_details,int8_details=iq_details,training_saturated_input_values=train_clipped,
        detector_fit_executed=False,outer_evaluation_executed=False,predictive_selection_executed=False,
        STM32_cost_measured=False,retrospective_evaluation=True,versions=legacy.versions(),
        original_training_core_source_sha256=sha(LEGACY/'worker_representantes.py'),
        preparation_manifest_sha256=sha(PREP_MANIFEST),
        timings_seconds=dict(imports=import_seconds,data_and_scaler=data_seconds,fit_including_checkpoint_IO=fit_seconds,
            FP32_conversion=export['FP32_seconds'],INT8_conversion=export['INT8_seconds'],validation_inference_and_IO=inference_seconds),
        elapsed_worker_seconds=time.perf_counter()-START)
    validate_result(result,job,plan['fit_signatures'][job_id],plan['semantic_signatures'][job_id])
    write(directory/'resultado.json',result)
    event('PASS',best_epoch=state['best_epoch'],macro_f1_int8=metrics['INT8']['macro_f1'])
    write_manifest(directory/'MANIFEST_JOB_SHA256.json',directory)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--batch',type=Path,required=True)
    parser.add_argument('--job-id',required=True)
    args=parser.parse_args()
    try:
        run(args.campaign.resolve(),args.batch.resolve(),args.job_id)
    except Exception:
        print(traceback.format_exc(),file=sys.stderr,flush=True)
        raise SystemExit(1)
