"""Probar evaluación con modelos sintéticos existentes; no ajustar modelos."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from unittest.mock import patch

import contrato_evaluacion as c
io=c.io
SYNTHETIC_MODELS=c.ROOT/'articulo/ampliacion_cwru/metodologia/reajustes_finales_v2_2026-10-27/prueba_sintetica_real_intento2'


@contextmanager
def no_cwru_arrays():
    import numpy as np
    original=np.load;calls={'blocked_requests':0}
    def guarded(path,*args,**kwargs):
        if isinstance(path,(str,Path)) and Path(path).resolve().is_relative_to((c.ROOT/'datos/cwru').resolve()):
            calls['blocked_requests']+=1;raise RuntimeError('Ensayo sintético no abre datos CWRU')
        return original(path,*args,**kwargs)
    with patch.object(np,'load',guarded):
        try:np.load(c.ROOT/'datos/cwru/PROBE_MUST_BE_BLOCKED.npy')
        except RuntimeError:pass
        io.require(calls['blocked_requests']==1,'Guarda no operativa')
        yield calls
    io.require(calls['blocked_requests']==1,'Intento adicional de acceso CWRU')


def fixture(folder):
    import numpy as np
    models=[]
    for family in ('DNN','RF','SVM','detector'):
        base=SYNTHETIC_MODELS/family/'salida';result=io.read(base/'RESULTADO.json')
        io.verify_manifest(base/'MANIFEST_JOB_SHA256.json',exact_directory=base)
        names={'DNN':['modelo.keras','modelo_fp32.tflite','modelo_int8.tflite','scaler_train_only.npz'],
            'RF':['modelo.joblib'],'SVM':['modelo.joblib','scaler_train_only.npz'],
            'detector':['parametros.npz','calibracion.npz']}[family]
        paths={name:(base/name).relative_to(c.ROOT).as_posix() for name in names}
        models.append(dict(job_id=result['job_id'],family=family,outer_load_hp=0,seed=result.get('seed'),
            candidate=result.get('candidate'),epochs=result.get('epochs_executed'),config_index=result.get('config_index'),
            indices=result.get('indices',list(range(14))),roles=result['roles'],masks=result.get('masks'),
            procedures=result.get('procedures'),fit_signature=result['fit_signature'],
            selection_signature=result['selection_signature'],result_path=(base/'RESULTADO.json').relative_to(c.ROOT).as_posix(),
            result_sha256=io.sha(base/'RESULTADO.json'),paths=paths,
            artifact_sha256={name:io.sha(c.ROOT/path) for name,path in paths.items()}))
    records=np.repeat(np.asarray([f'SYNTH_{i}' for i in range(7)]),6)
    labels=np.repeat(np.asarray([0,1,1,1,2,2,2],dtype=np.int64),6)
    raw=np.random.default_rng(20261027).normal(size=(42,14)).astype(np.float32)
    raw[0,0]=np.float32(100000.)
    ids=list(range(1000000,1000042))
    job=dict(job_id='outer_eval_0',outer_load_hp=0,heldout_global_window_ids=ids,
        heldout_MAT=sorted(set(records.tolist())),heldout_windows=42,models=models,
        expected_counts={family:1 for family in ('DNN','RF','SVM','detector')},
        expected_inference_counts=dict(DNN_FP32_invocations=42,DNN_INT8_invocations=42,
            RF_predicted_rows=42,SVM_predicted_rows=42,detector_scored_rows=42),timeout_seconds=300)
    plan=dict(expected_versions=c.refit.versions(),source_sha256=c.sources(),worker_environment=io.ENVIRONMENT,
        selection_signature='synthetic_only_not_a_production_authorization',
        evaluation_signatures={'outer_eval_0':io.canonical(job)})
    provenance=dict(global_window_ids=ids,selected_loads_hp=[0],selected_record_ids=job['heldout_MAT'],
        shape=[42,14],loaded_dtype='float32',only_requested_rows_selected_or_materialized=True,
        raw_signals_or_MAT_opened=False,synthetic_data=True)
    with (folder/'synthetic_input.npz').open('xb') as stream:
        np.savez(stream,raw=raw,y=labels,records=records)
    io.write(folder/'FIXTURE.json',dict(job=job,plan=plan,provenance=provenance))


def child(folder):
    import numpy as np
    import worker_evaluacion as worker
    from ejecutar_evaluacion import verify_fold
    folder=Path(folder).resolve();payload=io.read(folder/'FIXTURE.json')
    plan,job=payload['plan'],payload['job']
    io.require(plan['source_sha256']==c.sources() and plan['expected_versions']==c.refit.versions(),'Fuentes/entorno cambiaron')
    campaign=folder/'campaign';campaign.mkdir(exist_ok=False)
    with no_cwru_arrays() as access:
        with np.load(folder/'synthetic_input.npz',allow_pickle=False) as a:
            raw,y,records=a['raw'],a['y'],a['records'].tolist()
        def authorize(actual_campaign,outer):
            io.require(Path(actual_campaign)==campaign and outer==0,'Fixture ajena')
            return plan,job
        with patch.object(c,'authorize',authorize),patch.object(worker,'load_heldout',return_value=(raw,y,records,payload['provenance'])):
            worker.run(campaign,0)
        result=verify_fold(campaign,job,plan)
        from sklearn.preprocessing import StandardScaler
        try:StandardScaler().fit(np.zeros((2,2),dtype=np.float32))
        except RuntimeError:guard=True
        else:guard=False
        io.require(guard,'Entrenamiento accidental no está bloqueado')
        io.require(result['model_hashes_before']==result['model_hashes_after'],'Modelos alterados')
        dnn=next(r for r in result['model_results'] if r['family']=='DNN')
        io.require(dnn['quantization_report']['input_clipped_feature_values']>0,'La prueba no ejercitó clipping')
    io.write(folder/'COMPROBACION.json',dict(status='PASS',four_families_verified=True,
        saved_output_verification_without_classifier_replay=True,clipping_verified=True,training_guard_verified=True,
        inference_counts=result['inference_counts'],CWRU_values_opened=False,blocked_CWRU_requests=access['blocked_requests'],
        new_training_fits=0,production_authorization_modified=False,fixture_authorization_patch_only=True))


def run():
    os.environ.update(io.ENVIRONMENT)
    folder=c.PREP/'prueba_sintetica';report_path=c.PREP/'PRUEBA_SINTETICA_EVALUACION.json'
    io.require(not folder.exists() and not report_path.exists(),'Ensayo existente; no sobrescribir o repetir')
    c.PREP.mkdir(parents=True,exist_ok=True);folder.mkdir(exist_ok=False)
    sources=c.sources();started=time.monotonic();error=None;code=None;summary={}
    try:
        with no_cwru_arrays():fixture(folder)
        env=os.environ.copy();env.update(io.ENVIRONMENT)
        with (folder/'stdout.log').open('x',encoding='utf-8') as out,(folder/'stderr.log').open('x',encoding='utf-8') as err:
            result=subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'--child',str(folder)],
                cwd=c.ROOT,env=env,stdout=out,stderr=err,timeout=300)
            code=result.returncode
        io.require(code==0,'Ensayo fallido; conservar logs')
        summary=io.read(folder/'COMPROBACION.json');io.require(summary['status']=='PASS','Comprobación fallida')
        io.require(c.sources()==sources,'Fuentes cambiaron durante ensayo')
    except Exception:error=traceback.format_exc()
    report=dict(status='PASS' if error is None else 'FAILED_PRESERVED_DO_NOT_RETRY',
        date_local='2026-10-27',timezone='America/Lima',completed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        versions=c.refit.versions(),source_sha256=sources,source_sha256_before=sources,source_sha256_after=c.sources(),
        runtime_seconds=time.monotonic()-started,returncode=code,failure=error,summary=summary,
        CWRU_values_opened=False,new_training_fits=0,real_outer_predictions=0,
        existing_synthetic_models_only=True,automatic_retry=False,production_authorization_modified=False)
    io.write(report_path,report)
    print(json.dumps(dict(status=report['status'],report=str(report_path),report_sha256=io.sha(report_path))))
    return 0 if error is None else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',action='store_true');parser.add_argument('--child',type=Path)
    args=parser.parse_args()
    if args.child:
        io.require(not args.run,'Argumentos incompatibles');child(args.child.resolve())
    else:
        io.require(args.run,'Usar --run explícito');raise SystemExit(run())
