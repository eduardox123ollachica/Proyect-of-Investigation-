"""Búsqueda interna RF/SVM: preparar sin entrenar; usuario ejecuta --run.

384 ajustes por rol, un proceso por ajuste, guardado incremental y sin
selección automática, evaluación exterior, DNN o hardware.
"""
from __future__ import annotations
import argparse
import datetime
import importlib.abc
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[4]
SOURCE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'codigo/cwru/ampliacion_v2/busqueda_interna'))


class NoDNN(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in ('tensorflow','keras','ai_edge_litert'):
            raise RuntimeError('El comparador clásico no importa un runtime DNN')
        return None


sys.meta_path.insert(0,NoDNN())
import numpy as np
import ejecutar_busqueda_interna as io
import datos_clasicos as data
from numerica_representantes import normalize_f32,metrics3,balanced_weights

PREP=ROOT/'articulo/ampliacion_cwru/metodologia/clasicos_internos_v2_2026-10-27'
OUTPUT=ROOT/'resultados/cwru/ampliacion_v2/clasicos_internos_v2_2026-10-27'
PLAN=PREP/'PLAN_CLASICOS_INTERNOS.json'
MANIFEST=PREP/'MANIFEST_PREPARACION_CLASICOS_SHA256.json'


def versions():
    return dict(python=platform.python_version(),**{name:importlib.metadata.version(package)
        for name,package in [('numpy','numpy'),('sklearn','scikit-learn'),('scipy','scipy'),('joblib','joblib')]})


def sources():
    dependencies=['busqueda_interna/datos_busqueda.py','busqueda_interna/ejecutar_busqueda_interna.py',
        'detector_interno/datos_detector.py','representantes/datos_representantes.py',
        'representantes/ejecutar_representantes.py','representantes/numerica_representantes.py',
        'protocolo/reglas_v2.py','piloto_clasicos/contrato_piloto.py']
    files=[*SOURCE.glob('*.py'),*[ROOT/'codigo/cwru/ampliacion_v2'/p for p in dependencies]]
    return {p.relative_to(ROOT).as_posix():io.sha(p) for p in sorted(files)}


def save_npz(path,**arrays):
    with path.open('xb') as stream:np.savez(stream,**arrays)


def prepare_inputs(family,raw,validation):
    io.require(raw.dtype == validation.dtype == np.float32 and raw.ndim == validation.ndim == 2
        and raw.shape[1] == validation.shape[1] == 14
        and np.isfinite(raw).all() and np.isfinite(validation).all(),'Entradas clásicas no finitas o distintas de F14float32')
    if family == 'RF':return raw,validation,None
    io.require(family == 'SVM','Familia desconocida')
    from sklearn.preprocessing import StandardScaler
    scaler=StandardScaler().fit(raw)
    arrays=dict(mean64=scaler.mean_,scale64=scaler.scale_,var64=scaler.var_,
        mean32=scaler.mean_.astype(np.float32),scale32=scaler.scale_.astype(np.float32))
    return normalize_f32(raw,arrays['mean32'],arrays['scale32']),normalize_f32(validation,arrays['mean32'],arrays['scale32']),arrays


def make_estimator(job):
    if job['family'] == 'RF':
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(**job['params'])
    io.require(job['family'] == 'SVM','Familia desconocida')
    from sklearn.svm import SVC
    return SVC(**job['params'])


def predict_estimator(model,family,validation):
    predictions=model.predict(validation)
    values=model.predict_proba(validation) if family == 'RF' else model.decision_function(validation)
    io.require(predictions.shape == (len(validation),) and predictions.dtype == np.int64
        and set(predictions.tolist()) <= {0,1,2} and values.shape == (len(validation),3)
        and np.isfinite(values).all(),'Salidas clásicas inválidas')
    # SVC.predict conserva la votación nativa; argmax(decision_function) puede diferir.
    return predictions,values


def preflight(environment=False,deep=True):
    io.verify_manifest(MANIFEST)
    plan=io.read(PLAN)
    io.require(plan['source_sha256'] == sources(),'Fuentes cambiadas')
    io.require(plan['limits'] == io.read(PREP/'PRESUPUESTO_Y_LIMITES.json')['limits']
        and plan['budget_sha256'] == io.sha(PREP/'PRESUPUESTO_Y_LIMITES.json'),'Presupuesto cambiado')
    io.require(plan['logical_cells'] == 384 and plan['RF_fits'] == 288 and plan['SVM_fits'] == 96
        and len(plan['jobs']) == 384 and plan['imported_fits'] == 0,'Inventario clásico cambiado')
    io.require(plan['worker_environment'] == io.ENVIRONMENT,'Entorno de worker cambiado')
    io.require(not any(plan[k] for k in ('selection_enabled','outer_evaluation_enabled','DNN_enabled',
        'detector_enabled','hardware_enabled','automatic_retry')), 'Alcance extendido')
    if environment:io.require(versions() == plan['tested_versions'],'Usar las versiones exactas del Python preparado')
    if deep:
        protocol,rows,hashes=data.context()
        io.require(plan['jobs'] == data.build_jobs(protocol,rows),'Grid distinto del protocolo')
        io.require(plan['fit_signatures'] == {j['job_id']:data.training_signature(j,protocol,hashes,
            plan['tested_versions'],plan['source_sha256']) for j in plan['jobs']},'Firmas cambiadas')
    return plan


def prepare():
    from presupuesto_clasicos_internos import build_budget
    io.require(not PLAN.exists() and not MANIFEST.exists(),'No sobrescribir preparación')
    io.preflight(check_environment=True)
    protocol,rows,hashes=data.context();jobs=data.build_jobs(protocol,rows)
    tests=io.read(PREP/'PRUEBAS_PREPARACION.json')
    io.require(tests['status'] == 'PASS' and tests['errors'] == tests['failures'] == 0
        and tests['source_sha256'] == sources() and tests['real_training_fits'] == tests['real_model_predictions'] == 0
        and tests['TensorFlow_imported'] is False,'Faltan pruebas sobre las fuentes finales')
    budget=build_budget();io.write(PREP/'PRESUPUESTO_Y_LIMITES.json',budget)
    prerequisite_files=[]
    for directory,filename,key,count in [
        ('cierre_matriz_dnn_interna_v2_2026-10-26','CIERRE_MATRIZ_INTERNA.json','cells',288),
        ('cierre_detector_interno_v2_2026-10-26','CIERRE_DETECTOR_INTERNO.json','logical_rows',96)]:
        folder=ROOT/'articulo/ampliacion_cwru/metodologia'/directory
        result_path=folder/filename
        seal=next(folder.glob('MANIFEST_CIERRE_*SHA256.json'))
        digest,name=seal.with_suffix('.sha256').read_text(encoding='ascii').split()
        io.require(name == seal.name and io.sha(seal) == digest
            and io.read(seal)[result_path.relative_to(ROOT).as_posix()] == io.sha(result_path)
            and io.read(result_path)[key] == count,'Requisito previo no cerrado o alterado')
        prerequisite_files += [result_path,seal,seal.with_suffix('.sha256')]
    tested=versions();source_hashes=sources()
    plan=dict(status='CLASSICAL_INNER_SEARCH_PREPARED_PENDING_USER_EXECUTION',jobs=jobs,logical_cells=384,
        RF_fits=288,SVM_fits=96,imported_fits=0,pilot_retrained=False,pilot_reused_as_confirmatory=False,
        tested_versions=tested,source_sha256=source_hashes,limits=budget['limits'],
        budget_sha256=io.sha(PREP/'PRESUPUESTO_Y_LIMITES.json'),
        fit_signatures={j['job_id']:data.training_signature(j,protocol,hashes,tested,source_hashes) for j in jobs},
        worker_environment=io.ENVIRONMENT,one_worker_process_per_fit=True,
        selection_enabled=False,outer_evaluation_enabled=False,DNN_enabled=False,detector_enabled=False,
        hardware_enabled=False,manuscript_edits_enabled=False,automatic_retry=False,
        retrospective_evaluation=True,user_starts_training=True,new_training_fits_during_preparation=0)
    io.write(PLAN,plan)
    command=f'& "C:\\Users\\LENOVO\\AppData\\Local\\Microsoft\\WindowsApps\\python3.12.exe" -B "{Path(__file__).resolve()}" --run'
    text=f'''# Búsqueda interna RF/SVM CWRU V2

Plan:384 ajustes por rol (288RF y96SVM). Cuatro escenarios exteriores,
tres validaciones internas, ocho configuraciones por familia. RF usa11/22/33;
SVM se ajusta una sola vez por configuración/pliegue, sin triplicar semillas.
Ambos usan F14. El grid y su orden proceden del contrato original del piloto.
RF recibe float32 sin escalar; SVM usa scaler ajustado únicamente con train,
exportado a float32 y aplicado con resta/división explícitas. SVC mantiene
predict/votación nativa; su decision_function no sustituye esa regla.

```powershell
{command}
```

Un proceso por ajuste, CPU/un hilo. Límites RF60s ySVM30s; presupuesto acumulado
3600s para procesos y comprobación individual. No incluye todo el preflight
o pausas entre comandos; es más conservador que contar solo subprocesos.
La parada se decide antes de iniciar otro ajuste, reservando su timeout
completo. La comprobación o terminación de un paso ya iniciado puede añadir
tiempo al alcanzar esa frontera; no es un límite exacto de reloj global.
Reserva orientativa30–60min, sin garantía. El presupuesto de esta carpeta
deriva de las16configuraciones medidas, sin repetir el piloto. No se amplían
límites ni se reintenta automáticamente. Fallos/interrupciones conservan
artefactos y detienen para auditoría. Guardado por modelo y paso con firmas.

El piloto fue solo temporal, confirmatory_fit=false y sin estimadores
persistidos; no se presenta como un resultado de búsqueda reutilizable.
Se mantienen los384ajustes por rol del protocolo, aunque algunos pares de
roles comparten entrenamiento; no se introduce una deduplicación nueva entre
pliegues. Esto no implica384adquisiciones independientes ni un test virgen.

Se guardan modelos joblib, parámetros, IDs, scaler SVM, predicciones,
probabilidades/vectores de decisión y métricas por clase. La reanudación
verifica modelos completos y firmas; un marcador único impide reiniciar el
presupuesto creando otra campaña. Un fallo necesita recuperación técnica
separada; no borrar locks o archivos para repetir. No se cargan modelos ajenos.

Al terminar imprime `CLASSICAL_INNER_COMPLETED: 384/384; RF288; SVM96`.
No necesita placa o COM. No selecciona configuraciones o máscaras, modifica
el detector, revela evaluación exterior ni edita el manuscrito. Las tasas
deficientes se conservan. Los tiempos PC no se presentan como latencia STM32.
'''
    (PREP/'README_EJECUCION.md').write_text(text,encoding='utf-8')
    files=[p for p in PREP.iterdir() if p.is_file()]+[ROOT/p for p in source_hashes]+prerequisite_files
    files += [ROOT/p for p in budget['source_sha256']]
    io.write_manifest(MANIFEST,extra=files)
    print(json.dumps(dict(status=plan['status'],RF_fits=288,SVM_fits=96,real_training_fits=0,
        manifest_sha256=io.sha(MANIFEST)),ensure_ascii=True))


def validate_campaign(campaign,deep=False):
    campaign=Path(campaign).resolve()
    io.require(campaign.parent == OUTPUT.resolve(),'Campaña fuera del destino')
    plan=preflight(environment=True,deep=deep)
    io.require(io.read(OUTPUT/'CAMPANA_UNICA.json') == dict(run_id=campaign.name,
        preparation_sha256=io.sha(MANIFEST)), 'Campaña no coincide con el marcador único')
    io.require(io.read(campaign/'CONTRATO.json') == dict(preparation_sha256=io.sha(MANIFEST),
        plan_sha256=io.sha(PLAN),scope='classical_inner_only',execution_owner='user_CLI'), 'Contrato cambiado')
    return plan


def worker(campaign,job_id):
    import joblib
    started=time.monotonic();plan=validate_campaign(campaign)
    io.require((campaign/'ACTIVO.lock').is_file(),'Worker requiere padre activo')
    job=next((j for j in plan['jobs'] if j['job_id'] == job_id),None)
    io.require(job is not None,'Trabajo ajeno al plan')
    request=io.read(campaign/'SOLICITUD_ACTIVA.json')
    io.require(request == dict(job_id=job_id,fit_signature=plan['fit_signatures'][job_id]),'Trabajo no autorizado')
    io.require(all(os.environ.get(k) == v for k,v in plan['worker_environment'].items()),'Entorno no fijado')
    folder=campaign/job_id;folder.mkdir(exist_ok=False)
    raw,y,raw_val,y_val,provenance=data.load_job_data(job)
    x,val,scaler=prepare_inputs(job['family'],raw,raw_val)
    io.write(folder/'procedencia.json',provenance)
    if scaler is not None:save_npz(folder/'scaler_train_only.npz',**scaler)
    counts,weights,_=balanced_weights(y)
    model=make_estimator(job)
    fit_start=time.monotonic();model.fit(x,y);fit_seconds=time.monotonic()-fit_start
    io.require(np.array_equal(model.classes_,[0,1,2]),'Clases de modelo incorrectas')
    if job['family'] == 'RF':io.require(len(model.estimators_) == job['params']['n_estimators'],'RF incompleto')
    else:io.require(model.fit_status_ == 0,'SVM no convergió')
    inference_start=time.monotonic();pred,values=predict_estimator(model,job['family'],val)
    inference_seconds=time.monotonic()-inference_start
    save_npz(folder/'predicciones_validacion.npz',global_window_ids=np.asarray(job['validation_global_window_ids'],dtype=np.int64),
        y_true=y_val,predictions=pred,values=values)
    joblib.dump(model,folder/'modelo.joblib',compress=3)
    result=dict(status='PASS',job_id=job_id,family=job['family'],config_index=job['config_index'],seed=job['seed'],
        outer_load_hp=job['outer_load_hp'],inner_validation_load_hp=job['inner_validation_load_hp'],roles=job['roles'],
        fit_signature=plan['fit_signatures'][job_id],params=job['params'],training_windows=len(y),validation_windows=len(y_val),
        input_dimension=14,model_classes=model.classes_.tolist(),class_counts=counts.tolist(),
        balanced_class_weights=weights.tolist(),metrics=metrics3(y_val,pred),versions=versions(),
        model_sha256=io.sha(folder/'modelo.joblib'),model_bytes=(folder/'modelo.joblib').stat().st_size,
        support_vectors_dtype=str(model.support_vectors_.dtype) if job['family']=='SVM' else None,
        preparation_manifest_sha256=io.sha(MANIFEST),normalization=provenance['normalization'],
        native_predict_rule=True,outer_evaluation_executed=False,selection_executed=False,
        DNN_fit_or_inference=False,detector_fit_or_score=False,STM32_cost_measured=False,
        fit_seconds=fit_seconds,inference_seconds=inference_seconds,worker_elapsed_seconds=time.monotonic()-started)
    io.write(folder/'resultado.json',result)
    io.write_manifest(folder/'MANIFEST_JOB_SHA256.json',directory=folder)


def verify_job(folder,job,plan):
    import joblib
    io.verify_manifest(folder/'MANIFEST_JOB_SHA256.json',exact_directory=folder)
    result=io.read(folder/'resultado.json')
    io.require(result['status'] == 'PASS' and all(result[k] == job[k] for k in (
        'job_id','family','config_index','seed','outer_load_hp','inner_validation_load_hp','roles','params'))
        and result['fit_signature'] == plan['fit_signatures'][job['job_id']]
        and result['versions'] == plan['tested_versions']
        and result['preparation_manifest_sha256'] == io.sha(MANIFEST),'Resultado ajeno al contrato')
    io.require(result['model_sha256'] == io.sha(folder/'modelo.joblib')
        and result['model_bytes'] == (folder/'modelo.joblib').stat().st_size,'Modelo modificado')
    raw,y,raw_val,y_val,provenance=data.load_job_data(job)
    io.require(io.read(folder/'procedencia.json') == provenance
        and result['training_windows'] == len(y) and result['validation_windows'] == len(y_val)
        and result['input_dimension'] == 14 and result['normalization'] == provenance['normalization'], 'Datos/procedencia distintos')
    _,val,scaler=prepare_inputs(job['family'],raw,raw_val)
    if scaler is None:io.require(not (folder/'scaler_train_only.npz').exists(),'RF no debe tener scaler')
    else:
        with np.load(folder/'scaler_train_only.npz',allow_pickle=False) as saved:
            io.require(set(saved.files) == set(scaler) and all(saved[k].dtype == v.dtype
                and saved[k].shape == v.shape and saved[k].tobytes() == v.tobytes() for k,v in scaler.items()),'Scaler no coincide con train')
    # Solo se deserializan modelos locales autenticados por el manifiesto anterior.
    model=joblib.load(folder/'modelo.joblib')
    expected_type=type(make_estimator(job))
    io.require(type(model) is expected_type and all(model.get_params()[k] == v for k,v in job['params'].items())
        and np.array_equal(model.classes_,[0,1,2]) and model.n_features_in_ == 14,'Estimador/clases distintos')
    if job['family']=='RF':io.require(len(model.estimators_) == job['params']['n_estimators'],'RF incompleto')
    else:io.require(model.fit_status_ == 0 and result['support_vectors_dtype'] == str(model.support_vectors_.dtype),'SVM incompleto')
    pred,values=predict_estimator(model,job['family'],val)
    with np.load(folder/'predicciones_validacion.npz',allow_pickle=False) as saved:
        io.require(saved['global_window_ids'].dtype == saved['y_true'].dtype == saved['predictions'].dtype == np.int64
            and saved['global_window_ids'].tolist() == job['validation_global_window_ids']
            and np.array_equal(saved['y_true'],y_val) and np.array_equal(saved['predictions'],pred)
            and saved['values'].dtype == values.dtype and np.array_equal(saved['values'],values), 'Predicciones no reproducibles')
    counts,weights,_=balanced_weights(y)
    io.require(result['metrics'] == metrics3(y_val,pred) and result['class_counts'] == counts.tolist()
        and result['balanced_class_weights'] == weights.tolist() and result['native_predict_rule'] is True
        and not any(result[k] for k in ('outer_evaluation_executed','selection_executed','DNN_fit_or_inference',
            'detector_fit_or_score','STM32_cost_measured')), 'Métricas o alcance inconsistentes')
    return result


def inspect_campaign(campaign):
    plan=validate_campaign(campaign,deep=True);done=[];consumed=0.
    for index,job in enumerate(plan['jobs']):
        state_path=campaign/'pasos'/f'{index:03d}.json'
        if not state_path.exists():
            io.require(not (campaign/job['job_id']).exists(),'Ajuste sin cierre; requiere auditoría')
            continue
        state=io.read(state_path)
        io.require(state['status'] == 'PASS' and state['job_id'] == job['job_id'] and not state['automatic_retry'],
            'Fallo previo; no reintentar automáticamente')
        io.require(state['timeout_seconds'] == job['timeout_seconds']
            and isinstance(state['execution_verification_seconds'],(int,float))
            and not isinstance(state['execution_verification_seconds'],bool)
            and np.isfinite(state['execution_verification_seconds']) and state['execution_verification_seconds'] >= 0,
            'Contador o timeout de un paso inválido')
        verify_job(campaign/job['job_id'],job,plan)
        done.append(job['job_id']);consumed+=state['execution_verification_seconds']
    io.require(done == [j['job_id'] for j in plan['jobs'][:len(done)]], 'Huecos en la campaña')
    io.require(len(list((campaign/'pasos').glob('*.json'))) == len(done),'Pasos ajenos al plan')
    return plan,done,consumed


def launch_one(campaign,job,plan,timeout):
    started=time.monotonic();process=None;error=None;code=None
    io.write(campaign/'SOLICITUD_ACTIVA.json',dict(job_id=job['job_id'],fit_signature=plan['fit_signatures'][job['job_id']]))
    env=os.environ.copy();env.update(plan['worker_environment'])
    try:
        with (campaign/(job['job_id']+'_stdout.log')).open('x',encoding='utf-8') as out,\
             (campaign/(job['job_id']+'_stderr.log')).open('x',encoding='utf-8') as err:
            process=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),'--worker',
                '--campaign',str(campaign),'--job-id',job['job_id']],cwd=ROOT,env=env,stdout=out,stderr=err)
            code=process.wait(timeout=timeout)
        if code == 0:verify_job(campaign/job['job_id'],job,plan)
    except subprocess.TimeoutExpired:error='TIMEOUT'
    except KeyboardInterrupt:error='INTERRUPTED'
    except Exception as failure:error=type(failure).__name__+': '+str(failure)
    finally:
        if process is not None and process.poll() is None:
            process.kill();process.wait(timeout=10);code=process.returncode
        # Solo retirar nuestra solicitud cerrada; los resultados/logs permanecen.
        (campaign/'SOLICITUD_ACTIVA.json').unlink()
    return dict(status='PASS' if error is None and code == 0 else 'FAIL',job_id=job['job_id'],
        return_code=code,error=error,execution_verification_seconds=time.monotonic()-started,
        timeout_seconds=timeout,automatic_retry=False)


def can_start(timeout,consumed,budget):return budget-consumed >= timeout


def run():
    plan=preflight(environment=True);OUTPUT.mkdir(parents=True,exist_ok=True)
    marker=OUTPUT/'CAMPANA_UNICA.json'
    if marker.exists():
        info=io.read(marker);campaign=io.confined(OUTPUT,info['run_id'])
        io.require(info['preparation_sha256'] == io.sha(MANIFEST),'Marcador de otra preparación')
        previous=io.read(campaign/'RESUMEN_CAMPANA.json')
        io.require(previous['status'] == 'CAMPAIGN_BUDGET_BOUNDARY','Campaña completa, fallida o interrumpida: no repetir')
        io.verify_manifest(campaign/'MANIFEST_CAMPANA_SHA256.json')
    else:
        stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        io.write(marker,dict(run_id=stamp,preparation_sha256=io.sha(MANIFEST)))
        campaign=OUTPUT/stamp;campaign.mkdir();(campaign/'pasos').mkdir()
        io.write(campaign/'CONTRATO.json',dict(preparation_sha256=io.sha(MANIFEST),
            plan_sha256=io.sha(PLAN),scope='classical_inner_only',execution_owner='user_CLI'))
    lock=campaign/'ACTIVO.lock'
    with lock.open('x',encoding='utf-8') as stream:stream.write(str(os.getpid()))
    print('Evidencias: '+str(campaign),flush=True)
    try:
        plan,done,consumed=inspect_campaign(campaign)
        for index,job in enumerate(plan['jobs'][len(done):],start=len(done)):
            timeout=job['timeout_seconds']
            if not can_start(timeout,consumed,plan['limits']['campaign_execution_verification_seconds']):break
            print(f"[{index+1}/384] INICIO {job['job_id']}; limite{timeout}s",flush=True)
            state=launch_one(campaign,job,plan,timeout)
            io.write(campaign/'pasos'/f'{index:03d}.json',state)
            consumed+=state['execution_verification_seconds']
            print(job['job_id']+': '+state['status'],flush=True)
            if state['status'] != 'PASS':raise ValueError('Ajuste detenido; revisar '+job['job_id'])
            done.append(job['job_id'])
        complete=len(done) == 384
        summary=dict(status='CLASSICAL_INNER_COMPLETED' if complete else 'CAMPAIGN_BUDGET_BOUNDARY',
            completed_fits=len(done),planned_fits=384,remaining_fits=384-len(done),
            RF_completed=sum('_RF_' in j for j in done),SVM_completed=sum('_SVM_' in j for j in done),
            campaign_execution_verification_seconds=consumed,limits=plan['limits'],automatic_retry=False,
            outer_evaluation_executed=False,selection_executed=False,DNN_or_detector_fits=False)
    except BaseException as error:
        summary=dict(status='STOPPED_FOR_AUDIT',error=type(error).__name__+': '+str(error),
            traceback=traceback.format_exc(),automatic_retry=False)
    finally:lock.unlink()
    # Una frontera previa conserva su cierre; no sobrescribir evidencia.
    if (campaign/'RESUMEN_CAMPANA.json').exists():
        io.require(summary['status'] == 'CAMPAIGN_BUDGET_BOUNDARY','No sobrescribir cierre anterior')
        print('CAMPAIGN_BUDGET_BOUNDARY: presupuesto acumulado sin reiniciar',flush=True);return 2
    io.write(campaign/'RESUMEN_CAMPANA.json',summary)
    io.write_manifest(campaign/'MANIFEST_CAMPANA_SHA256.json',directory=campaign,extra=[MANIFEST,MANIFEST.with_suffix('.sha256'),PLAN])
    if summary['status'] == 'CLASSICAL_INNER_COMPLETED':print('CLASSICAL_INNER_COMPLETED: 384/384; RF288; SVM96',flush=True)
    else:print(summary['status']+': conservar evidencia para revisión',flush=True)
    print('Evidencias: '+str(campaign),flush=True)
    return 0 if summary['status'] == 'CLASSICAL_INNER_COMPLETED' else 2


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    for name in ('prepare','preflight','run','worker'):mode.add_argument('--'+name,action='store_true')
    parser.add_argument('--campaign',type=Path);parser.add_argument('--job-id')
    args=parser.parse_args()
    io.require((args.campaign is not None and args.job_id is not None) if args.worker
        else (args.campaign is None and args.job_id is None), 'Campaña/trabajo requieren modo worker y ambos argumentos')
    if args.prepare:prepare()
    elif args.preflight:preflight(environment=True);print('READ_ONLY_CLASSICAL_PREFLIGHT_PASS; 0 fits')
    elif args.worker:
        try:worker(args.campaign.resolve(),args.job_id)
        except Exception:print(traceback.format_exc(),file=sys.stderr,flush=True);sys.exit(1)
    else:sys.exit(run())
