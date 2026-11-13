"""Completar detector interno V2: preparar sin ajustes; ejecución por usuario.

Un único worker calcula19 parámetros nuevos y54 evaluaciones únicas.
Preserva23 parámetros y30 evaluaciones, alias96filas, sin DNN/selección/STM32.
"""
from __future__ import annotations
import argparse
import importlib.abc
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import datetime
import traceback

ROOT=Path(__file__).resolve().parents[4]
SOURCE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'codigo/cwru/ampliacion_v2/busqueda_interna'))
sys.path.insert(0,str(ROOT/'codigo/cwru/ampliacion_v2/piloto_clasicos'))


class NoDNN(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in ('tensorflow','keras','ai_edge_litert'):
            raise RuntimeError('Este bloque no importa runtime DNN')
        return None


sys.meta_path.insert(0,NoDNN())
import numpy as np
import ejecutar_busqueda_interna as base
from detector_numerico import fit_normality,detector_score
from numerica_representantes import detector_metrics,normalize_f32
from reglas_v2 import weighted_p95

PREP=ROOT/'articulo/ampliacion_cwru/metodologia/detector_interno_v2_2026-10-26'
OUTPUT=ROOT/'resultados/cwru/ampliacion_v2/detector_interno_v2_2026-10-26'
MANIFEST=PREP/'MANIFEST_PREPARACION_DETECTOR_SHA256.json'
PLAN=PREP/'PLAN_DETECTOR_INTERNO.json'
LIMITS=dict(worker_seconds=300,new_parameter_fits=19,new_unique_evaluations=54,
    logical_rows=96,parameter_scopes=42,automatic_retry=False,concurrent_workers=1)


def sources():
    files=[*SOURCE.glob('*.py'),ROOT/'codigo/cwru/ampliacion_v2/piloto_clasicos/detector_numerico.py',
        ROOT/'codigo/cwru/ampliacion_v2/representantes/numerica_representantes.py',
        ROOT/'codigo/cwru/ampliacion_v2/protocolo/reglas_v2.py']
    return {p.relative_to(ROOT).as_posix():base.sha(p) for p in sorted(files)}


def save_npz(path,**arrays):
    with path.open('xb') as stream:np.savez(stream,**arrays)


def same_bits(a,b):
    a,b=np.asarray(a),np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def preflight(environment=False):
    base.verify_manifest(MANIFEST)
    plan=base.read(PLAN)
    base.require(plan['source_sha256'] == sources() and plan['limits'] == LIMITS,'Fuentes o límites cambiados')
    base.preflight(check_environment=environment)
    for reference in list(plan['inventory']['reused_parameters'].values())+list(plan['inventory']['reused_evaluations'].values()):
        base.require(all(base.sha(ROOT/path) == digest for path,digest in reference['source_sha256'].items()),
            'Recurso de detector reutilizado cambiado')
    return plan


def prepare():
    import datos_detector as data
    base.require(not PLAN.exists() and not MANIFEST.exists(),'No sobrescribir preparación')
    base.preflight(check_environment=True)
    inventory=data.build_inventory()
    tests=base.read(PREP/'PRUEBAS_PREPARACION.json')
    base.require(tests['status'] == 'PASS' and tests['errors'] == tests['failures'] == 0
        and tests['source_sha256'] == sources() and tests['new_real_parameter_fits'] == 0
        and tests['new_real_evaluations'] == 0 and tests['TensorFlow_imported'] is False,
        'Faltan pruebas sobre fuentes finales sin ajustes reales')
    base.require(len(inventory['parameter_jobs']) == 42 and len(inventory['reused_parameters']) == 23
        and len(inventory['evaluation_jobs']) == 84 and len(inventory['reused_evaluations']) == 30
        and len(inventory['logical_jobs']) == 96 and len(inventory['reused_rows']) == 36,
        'Inventario incompatible')
    # Proyección de ingeniería desde el bloque previo5fit+6evaluaciones/8.275s.
    old_status=next((ROOT/'resultados/cwru/ampliacion_v2/suplemento_preprocesamiento_normalidad_2026-10-25/20261025T215652_507777Z/lotes').rglob('ESTADO_DETECTOR_BLOCK.json'))
    timing=base.read(old_status)
    base.require(timing['status'] == 'PASS','Bloque temporal anterior no aprobado')
    budget=dict(reference_seconds=timing['process_seconds'],reference_new_fits=5,reference_evaluations=6,
        reference_status_sha256=base.sha(old_status),reference_path=old_status.relative_to(ROOT).as_posix(),
        engineering_unit_scaling_seconds=timing['process_seconds']*(19+54)/(5+6),
        projection_is_not_a_guarantee=True,reserve_wall_minutes=[3,10],worker_limit_seconds=300,
        shared_preflight_and_validation_outside_worker_limit=True)
    closure=ROOT/'articulo/ampliacion_cwru/metodologia/cierre_matriz_dnn_interna_v2_2026-10-26/CIERRE_MATRIZ_INTERNA.json'
    closure_manifest=closure.parent/'MANIFEST_CIERRE_MATRIZ_SHA256.json'
    digest,name=closure_manifest.with_suffix('.sha256').read_text(encoding='ascii').split()
    base.require(name == closure_manifest.name and digest == base.sha(closure_manifest)
        and base.read(closure_manifest)[closure.relative_to(ROOT).as_posix()] == base.sha(closure),
        'Cierre DNN no coincide con su sello')
    base.require(base.read(closure)['cells'] == 288,'Matriz DNN no cerrada')
    plan=dict(status='DETECTOR_INTERNAL_EXECUTOR_PREPARED_PENDING_USER_EXECUTION',inventory=inventory,
        source_sha256=sources(),limits=LIMITS,budget=budget,
        DNN_closure_sha256=base.sha(closure),prior_DNN_preparation_sha256=base.sha(base.PREP_MANIFEST),
        DNN_fits_enabled=False,outer_evaluation_enabled=False,selection_enabled=False,
        hardware_enabled=False,manuscript_edits_enabled=False,new_real_parameter_fits_during_preparation=0,
        new_real_evaluations_during_preparation=0,tested_versions=base.legacy.versions())
    base.write(PLAN,plan)
    command=f'& "C:\\Users\\LENOVO\\AppData\\Local\\Microsoft\\WindowsApps\\python3.12.exe" -B "{Path(__file__).resolve()}" --run'
    text=f'''# Completar el detector interno CWRU V2

Preparación sin ajustes nuevos. La matriz tiene96filas lógicas;36 se conservan
y60 faltan. F14/NF12 comparten índices y parámetros: son84evaluaciones únicas,
30 conservadas y54 nuevas. Los parámetros son42ámbitos únicos,23 conservados
y19nuevos. No se repiten por semilla. Las firmas originales se conservan;
cuando un detector histórico no tenía firma individual se citan sus hashes y
su manifiesto de cobertura, sin inventar una firma de ejecución.

Se usa el núcleo original K=1: scaler/centroide con normales de la carga menor
de cada par, P95 con normales de la mayor, sin reajuste posterior. La validación
usa otra carga completa N/IR/OR. Umbral observado float32, CDF con igual masa
por MAT y rechazo estricto score>umbral. No utiliza el scaler ni errores DNN.

```powershell
{command}
```

Un worker secuencial, CPU e hilos limitados; máximo300s de proceso, sin
reintentos automáticos. Reserva orientativa3–10min incluyendo preflight y
comprobaciones. El bloque previo de5ajustes+6evaluaciones duró8.275s; escalar
por número de operaciones daría{budget['engineering_unit_scaling_seconds']:.1f}s,
pero mezcla tareas distintas y no garantiza duración. La preparación y la
verificación del padre pueden añadir tiempo fuera del límite del worker.

Guarda parámetros, IDs, calibración, scores, decisiones y métricas por MAT
incrementalmente. Un fallo o interrupción conserva el intento y bloquea repetir
con este comando; requiere auditoría de recuperación. No eliminar locks o marcador
único. Al terminar imprime `DETECTOR_INTERNAL_COMPLETED: 96/96 filas; 42/42 parametros`.
No necesita placa, COM, modelos nuevos DNN o cambios al manuscrito.

Este bloque no selecciona máscaras ni ejecuta las guardias del selector o
evaluación exterior. El mínimo absoluto0.95 del suplemento es descriptivo;
no reemplaza las guardias relativas a F14 del protocolo principal. Las tasas
deficientes se conservan como resultados, no se corrigen cambiando umbrales.
'''
    (PREP/'README_EJECUCION.md').write_text(text,encoding='utf-8')
    files=[p for p in PREP.iterdir() if p.is_file()]
    files += [ROOT/p for p in sources()]+[ROOT/p for p in inventory['source_sha256']]
    files += [closure,closure_manifest,closure_manifest.with_suffix('.sha256'),
        old_status,base.PREP_MANIFEST,base.PREP_MANIFEST.with_suffix('.sha256')]
    for ref in list(inventory['reused_parameters'].values())+list(inventory['reused_evaluations'].values()):
        files += [ROOT/p for p in ref['source_sha256']]
    base.write_manifest(MANIFEST,extra=files)
    print(json.dumps(dict(status=plan['status'],new_parameter_fits=19,new_unique_evaluations=54,
        missing_logical_rows=60,new_real_parameter_fits_during_preparation=0,
        forecast_seconds=budget['engineering_unit_scaling_seconds'],manifest_sha256=base.sha(MANIFEST)),ensure_ascii=True))


def validate_campaign(campaign):
    campaign=Path(campaign).resolve()
    base.require(campaign.parent == OUTPUT.resolve(),'Campaña ajena')
    plan=preflight(environment=True)
    base.require(base.read(OUTPUT/'EJECUCION_UNICA.json') == dict(run_id=campaign.name,
        preparation_sha256=base.sha(MANIFEST)),'Marcador de campaña diferente')
    base.require(base.read(campaign/'CONTRATO.json') == dict(preparation_sha256=base.sha(MANIFEST),
        plan_sha256=base.sha(PLAN),scope='normality_internal_only',execution_owner='user_CLI'), 'Contrato cambiado')
    return plan


def parameter_path(plan,campaign,param_id):
    reference=plan['inventory']['reused_parameters'].get(param_id)
    return ROOT/reference['params_path'] if reference else campaign/'parametros'/param_id/'parametros.npz'


def evaluation_source(plan,campaign,eval_id):
    reference=plan['inventory']['reused_evaluations'].get(eval_id)
    return (ROOT/reference['validation_path'],ROOT/reference['result_path'],reference['source_family']) if reference else (
        campaign/'evaluaciones'/eval_id/'validacion.npz',campaign/'evaluaciones'/eval_id/'resultado.json','new')


def worker(campaign):
    import datos_detector as data
    from datos_busqueda import context
    plan=validate_campaign(campaign)
    base.require((campaign/'ACTIVO.lock').is_file(),'Worker sin padre activo')
    base.require(all(os.environ.get(k) == v for k,v in base.ENVIRONMENT.items()),'Entorno de worker no fijado')
    ctx=context()
    def event(stage,**extra):
        row=dict(stage=stage,**extra)
        with (campaign/'progreso.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(row,allow_nan=False)+'\n')
        print(json.dumps(row,ensure_ascii=True),flush=True)
    fits=0;evaluations=0
    for job in plan['inventory']['parameter_jobs']:
        if job['param_id'] in plan['inventory']['reused_parameters']:continue
        folder=campaign/'parametros'/job['param_id'];folder.mkdir(exist_ok=False)
        fit,y_fit,fit_records,p_fit=data.load_rows(job['fit_ids'],context=ctx)
        cal,y_cal,cal_records,p_cal=data.load_rows(job['calibration_ids'],context=ctx)
        base.require(np.all(y_fit == 0) and np.all(y_cal == 0)
            and not set(fit_records)&set(cal_records),'Ajuste/calibración no normales o MAT compartidos')
        params=fit_normality(fit,cal,cal_records,job['indices']);fits+=1
        save_npz(folder/'parametros.npz',indices=np.asarray(params['indices'],dtype=np.int64),
            **{k:params[k] for k in ('mean64','scale64','var64','mean32','scale32','centroid32','threshold32')},
            fit_ids=np.asarray(job['fit_ids'],dtype=np.int64),calibration_ids=np.asarray(job['calibration_ids'],dtype=np.int64))
        save_npz(folder/'calibracion.npz',global_window_ids=np.asarray(job['calibration_ids'],dtype=np.int64),
            record_ids=np.asarray(cal_records),scores=detector_score(cal,params))
        base.write(folder/'procedencia.json',dict(fit=p_fit,calibration=p_cal,
            fit_normal_only=True,calibration_normal_only=True,refit_after_calibration=False))
        base.write(folder/'resultado.json',dict(status='PASS',param_id=job['param_id'],
            parameter_signature=job['parameter_signature'],new_fit_calls=1,
            parameter_payload_sha256=data.parameter_payload_sha256(folder/'parametros.npz'),
            threshold_bits=np.float32(params['threshold32']).tobytes().hex()))
        base.write_manifest(folder/'MANIFEST_UNIDAD_SHA256.json',directory=folder)
        event('PARAMETER_PASS',param_id=job['param_id'],new_fits_completed=fits,total_new_fits=19)
    for job in plan['inventory']['evaluation_jobs']:
        if job['eval_id'] in plan['inventory']['reused_evaluations']:continue
        folder=campaign/'evaluaciones'/job['eval_id'];folder.mkdir(exist_ok=False)
        params=data.load_params(parameter_path(plan,campaign,job['param_id']))
        raw,y,records,provenance=data.load_rows(job['validation_ids'],context=ctx)
        scores=detector_score(raw,params);rejected=scores > params['threshold32']
        metrics=detector_metrics(y,rejected,records);evaluations+=1
        save_npz(folder/'validacion.npz',global_window_ids=np.asarray(job['validation_ids'],dtype=np.int64),
            record_ids=np.asarray(records),y_true=y,scores=scores,rejected=rejected)
        base.write(folder/'procedencia.json',provenance)
        base.write(folder/'resultado.json',dict(status='PASS',eval_id=job['eval_id'],param_id=job['param_id'],
            evaluation_signature=job['evaluation_signature'],metrics=metrics,
            parameter_file_sha256=base.sha(parameter_path(plan,campaign,job['param_id'])),
            no_DNN_class_override=True,outer_evaluation_executed=False))
        base.write_manifest(folder/'MANIFEST_UNIDAD_SHA256.json',directory=folder)
        event('EVALUATION_PASS',eval_id=job['eval_id'],new_evaluations_completed=evaluations,total_new_evaluations=54)
    base.require(fits == 19 and evaluations == 54,'Cantidad real de operaciones distinta del plan')
    base.write(campaign/'CIERRE_WORKER.json',dict(status='PASS',new_parameter_fits=fits,
        new_unique_evaluations=evaluations,DNN_fits=0,TensorFlow_imported='tensorflow' in sys.modules))


def verify_outputs(plan,campaign):
    import datos_detector as data
    from datos_busqueda import context
    from sklearn.preprocessing import StandardScaler
    ctx=context()
    for job in plan['inventory']['parameter_jobs']:
        path=parameter_path(plan,campaign,job['param_id']);params=data.load_params(path)
        with np.load(path,allow_pickle=False) as saved:
            base.require(saved['fit_ids'].tolist() == job['fit_ids']
                and saved['calibration_ids'].tolist() == job['calibration_ids']
                and saved['indices'].tolist() == job['indices'],'Scope de parámetros distinto')
        if job['param_id'] in plan['inventory']['reused_parameters']:continue
        folder=path.parent;base.verify_manifest(folder/'MANIFEST_UNIDAD_SHA256.json',exact_directory=folder)
        result=base.read(folder/'resultado.json')
        base.require(result['status'] == 'PASS' and result['param_id'] == job['param_id']
            and result['parameter_signature'] == job['parameter_signature']
            and result['new_fit_calls'] == 1
            and result['parameter_payload_sha256'] == data.parameter_payload_sha256(path),'Resultado de parámetros inconsistente')
        fit,y_fit,fit_records,p_fit=data.load_rows(job['fit_ids'],context=ctx)
        cal,y_cal,cal_records,p_cal=data.load_rows(job['calibration_ids'],context=ctx)
        base.require(np.all(y_fit == 0) and np.all(y_cal == 0) and not set(fit_records)&set(cal_records),
            'Datos del detector no normales/disjuntos')
        scaler=StandardScaler().fit(fit[:,job['indices']])
        base.require(all(same_bits(params[k],v) for k,v in (
            ('mean64',scaler.mean_),('scale64',scaler.scale_),('var64',scaler.var_),
            ('mean32',scaler.mean_.astype(np.float32)),('scale32',scaler.scale_.astype(np.float32)))),
            'Scaler no reproduce únicamente normales de ajuste')
        centroid=np.mean(normalize_f32(fit[:,job['indices']],params['mean32'],params['scale32']),axis=0,dtype=np.float64).astype(np.float32)
        base.require(same_bits(centroid,params['centroid32']),'Centroide distinto')
        scores=detector_score(cal,params)
        base.require(same_bits(weighted_p95(scores,cal_records),params['threshold32'])
            and result['threshold_bits'] == params['threshold32'].tobytes().hex(),'Umbral distinto del P95 observado')
        base.require(base.read(folder/'procedencia.json') == dict(fit=p_fit,calibration=p_cal,
            fit_normal_only=True,calibration_normal_only=True,refit_after_calibration=False),'Procedencia de parámetros cambiada')
        with np.load(folder/'calibracion.npz',allow_pickle=False) as saved:
            base.require(saved['global_window_ids'].tolist() == job['calibration_ids']
                and saved['record_ids'].tolist() == cal_records and same_bits(saved['scores'],scores),'Calibración guardada distinta')
    evaluated={}
    for job in plan['inventory']['evaluation_jobs']:
        val_path,result_path,family=evaluation_source(plan,campaign,job['eval_id'])
        params=data.load_params(parameter_path(plan,campaign,job['param_id']))
        raw,y,records,provenance=data.load_rows(job['validation_ids'],context=ctx)
        scores=detector_score(raw,params);rejected=scores > params['threshold32']
        metrics=detector_metrics(y,rejected,records)
        with np.load(val_path,allow_pickle=False) as saved:
            base.require(saved['global_window_ids'].tolist() == job['validation_ids']
                and np.array_equal(saved['y_true'],y) and same_bits(saved['scores'],scores)
                and same_bits(saved['rejected'],rejected),'Validación no reproduce los scores/decisiones')
        result=base.read(result_path)
        recorded=result['detector_metrics'] if family == 'representantes' else result['metrics']
        base.require(result['status'] == 'PASS' and recorded == metrics,'Métricas no reproducibles por MAT')
        if family == 'new':
            base.verify_manifest(val_path.parent/'MANIFEST_UNIDAD_SHA256.json',exact_directory=val_path.parent)
            base.require(result['evaluation_signature'] == job['evaluation_signature']
                and result['param_id'] == job['param_id'] and result['eval_id'] == job['eval_id']
                and result['parameter_file_sha256'] == base.sha(parameter_path(plan,campaign,job['param_id']))
                and base.read(val_path.parent/'procedencia.json') == provenance,'Firma o procedencia de evaluación cambiada')
        evaluated[job['eval_id']]=metrics
    rows=[dict(row_id=j['row_id'],candidate=j['candidate'],outer_load_hp=j['outer_load_hp'],
        inner_validation_load=j['inner_validation_load_hp'],status='PASS',param_id=j['param_id'],eval_id=j['eval_id'],
        normal_spec=evaluated[j['eval_id']]['normal_spec'],ir_recall=evaluated[j['eval_id']]['ir_recall'],
        or_recall=evaluated[j['eval_id']]['or_recall'],balanced_accuracy=evaluated[j['eval_id']]['balanced_accuracy'],
        metrics=evaluated[j['eval_id']],reused_logical_row=j['row_id'] in plan['inventory']['reused_rows'])
        for j in plan['inventory']['logical_jobs']]
    base.require(len(rows) == 96 and sum(r['reused_logical_row'] for r in rows) == 36,'Matriz lógica no completa')
    return rows


def launch_worker(campaign,timeout=300):
    started=time.monotonic();process=None;error=None;code=None
    environment=os.environ.copy();environment.update(base.ENVIRONMENT)
    try:
        with (campaign/'stderr.log').open('x',encoding='utf-8') as err:
            process=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),'--worker','--campaign',str(campaign)],
                cwd=ROOT,env=environment,stderr=err)
            code=process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:error='TIMEOUT'
    except KeyboardInterrupt:error='INTERRUPTED'
    except OSError as failure:error='LAUNCH_ERROR: '+str(failure)
    finally:
        if process is not None and process.poll() is None:
            process.kill();process.wait(timeout=10);code=process.returncode
    return dict(status='PASS' if error is None and code == 0 else 'FAIL',return_code=code,error=error,
        worker_process_seconds=time.monotonic()-started,timeout_seconds=timeout,automatic_retry=False)


def run():
    plan=preflight(environment=True)
    OUTPUT.mkdir(parents=True,exist_ok=True)
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    base.write(OUTPUT/'EJECUCION_UNICA.json',dict(run_id=stamp,preparation_sha256=base.sha(MANIFEST)))
    campaign=OUTPUT/stamp;campaign.mkdir();(campaign/'parametros').mkdir();(campaign/'evaluaciones').mkdir()
    base.write(campaign/'CONTRATO.json',dict(preparation_sha256=base.sha(MANIFEST),plan_sha256=base.sha(PLAN),
        scope='normality_internal_only',execution_owner='user_CLI'))
    lock=campaign/'ACTIVO.lock'
    with lock.open('x',encoding='utf-8') as stream:stream.write(str(os.getpid()))
    print('Evidencias: '+str(campaign),flush=True)
    try:
        status=launch_worker(campaign,LIMITS['worker_seconds'])
        base.write(campaign/'ESTADO_WORKER.json',status)
        if status['status'] != 'PASS':raise ValueError('Worker no aprobado: '+str(status))
        worker_close=base.read(campaign/'CIERRE_WORKER.json')
        base.require(worker_close == dict(status='PASS',new_parameter_fits=19,new_unique_evaluations=54,
            DNN_fits=0,TensorFlow_imported=False),'Worker incompleto o alcance extendido')
        rows=verify_outputs(plan,campaign)
        base.write(campaign/'MATRIZ_DETECTOR_96.json',rows)
        summary=dict(status='DETECTOR_INTERNAL_COMPLETED',logical_rows=96,parameter_scopes=42,
            reused_parameter_scopes=23,new_parameter_fits=19,unique_evaluations=84,reused_unique_evaluations=30,
            new_unique_evaluations=54,reused_logical_rows=36,new_logical_rows=60,
            worker_process_seconds=status['worker_process_seconds'],all_scores_and_metrics_replayed=True,
            no_DNN_class_override=True,DNN_fits=0,outer_evaluation_executed=False,selection_executed=False,
            STM32_cost_measured=False,manuscript_modified=False,automatic_retry=False)
    except BaseException as error:
        summary=dict(status='STOPPED_FOR_AUDIT',error=type(error).__name__+': '+str(error),
            traceback=traceback.format_exc(),automatic_retry=False)
    finally:lock.unlink()
    base.write(campaign/'RESUMEN_DETECTOR_INTERNO.json',summary)
    base.write_manifest(campaign/'MANIFEST_CAMPANA_SHA256.json',directory=campaign,
        extra=[MANIFEST,MANIFEST.with_suffix('.sha256'),PLAN])
    print('DETECTOR_INTERNAL_COMPLETED: 96/96 filas; 42/42 parametros' if summary['status'] == 'DETECTOR_INTERNAL_COMPLETED'
        else 'STOPPED_FOR_AUDIT; conservar evidencia',flush=True)
    print('Evidencias: '+str(campaign),flush=True)
    return 0 if summary['status'] == 'DETECTOR_INTERNAL_COMPLETED' else 1


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare',action='store_true');mode.add_argument('--preflight',action='store_true')
    mode.add_argument('--run',action='store_true');mode.add_argument('--worker',action='store_true')
    parser.add_argument('--campaign',type=Path)
    args=parser.parse_args()
    base.require(args.worker == (args.campaign is not None),'--campaign requiere --worker')
    if args.prepare:prepare()
    elif args.preflight:
        preflight(environment=True);print('READ_ONLY_DETECTOR_PREFLIGHT_PASS; no ajustes nuevos')
    elif args.worker:
        try:worker(args.campaign.resolve())
        except Exception:
            print(traceback.format_exc(),file=sys.stderr,flush=True);sys.exit(1)
    else:sys.exit(run())
