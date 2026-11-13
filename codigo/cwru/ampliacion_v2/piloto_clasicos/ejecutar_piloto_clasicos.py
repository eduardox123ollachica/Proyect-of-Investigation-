"""Prepara o verifica el piloto; --execute queda para ejecucion del usuario."""
from __future__ import annotations
import argparse
import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

from contrato_piloto import build_jobs,validate_result

ROOT=Path(__file__).resolve().parents[4]
SOURCE=Path(__file__).resolve().parent
PREP=ROOT/'articulo/ampliacion_cwru/metodologia/piloto_clasicos_2026-10-25'
PROTOCOL=ROOT/'articulo/ampliacion_cwru/metodologia/protocolo_v2_2026-10-25'
DNN_CLOSURE=ROOT/'articulo/ampliacion_cwru/metodologia/cierre_piloto_tiempo_2026-10-25'
OUTPUT=ROOT/'resultados/cwru/ampliacion_v2/piloto_clasicos_2026-10-25'
ENVIRONMENT=dict(PYTHONHASHSEED='11',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',
    OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))


def write(path,value):
    with path.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')


def verify_manifest(path):
    digest,name=path.with_suffix('.sha256').read_text(encoding='ascii').split()
    require(name==path.name and digest==sha(path),'Sello distinto: '+str(path))
    entries=read(path)
    for name,digest in entries.items():
        target=(ROOT/name).resolve()
        require(target.is_relative_to(ROOT) and sha(target)==digest,'Recurso modificado: '+name)
    return entries


def source_plan():
    sys.path.insert(0,str(ROOT/'codigo/cwru/ampliacion_v2/piloto'))
    from datos_piloto import load_verified_plan
    prior=load_verified_plan()
    protocol=read(PROTOCOL/'protocolo_reglas.json')
    require(protocol['mass_training_enabled'] is False and protocol['outer_predictive_evaluation_enabled'] is False,
            'No convertir este piloto en busqueda')
    return prior['jobs'][0],build_jobs(protocol)


def prepare():
    data_job,jobs=source_plan()
    verify_manifest(DNN_CLOSURE/'MANIFEST_CIERRE_PILOTO_SHA256.json')
    tests=read(PREP/'PRUEBAS_PREPARACION.json')
    require(tests['status']=='PASS' and tests['failures']==tests['errors']==tests['skipped']==0
            and tests['source_AST_passed'] and tests['tests_run']>=30,'Faltan pruebas completas aprobadas')
    sources=sorted(SOURCE.glob('*.py'))
    require(all(tests['source_sha256'].get(p.relative_to(ROOT).as_posix())==sha(p) for p in sources),
            'Fuentes distintas de las pruebas')
    manifest=PREP/'MANIFEST_PREPARACION_SHA256.json'
    require(not manifest.exists() and not (PREP/'PLAN_PILOTO.json').exists(),'No sobrescribir preparacion')
    plan=dict(status='READY_FOR_USER_EXECUTION_NOT_RUN_BY_AGENT',scope='classical_runtime_pilot_only',
        execution_owner='user',agent_may_start_real_pilot=False,
        source_data_job_id=data_job['job_id'],source_plan_sha256=sha(PROTOCOL/'piloto_plan.json'),
        training_loads=[1,2],validation_loads=[0],training_windows=799,validation_windows=175,
        source_feature_arrays=['X_features_train.npy','X_features_validation.npy'],load3_arrays_authorized=False,
        normal_detector_fit_loads=[1],normal_detector_calibration_loads=[2],
        normal_detector_fit_windows=58,normal_detector_calibration_windows=58,
        jobs=jobs,maximum_processes=17,maximum_worker_seconds=780,
        timeout_sum_is_not_duration_estimate=True,workers_sequential=True,
        pilot_additional_RF_fits=8,pilot_additional_SVM_fits=8,pilot_unique_detector_fits=21,
        detector_aliases={'NF12':'F14'},detector_repetitions_are_not_seeds=True,
        metrics_used_for_selection=False,confirmatory_reuse=False,
        mass_training_enabled=False,outer_predictive_evaluation_enabled=False,
        manuscript_edits_enabled=False,automatic_retry=False,overwrite_allowed=False,
        tested_environment=tests['environment'],worker_environment=ENVIRONMENT,
        limits_are_project_engineering_choices=True)
    write(PREP/'PLAN_PILOTO.json',plan)
    paths=sources+[p for p in PREP.iterdir() if p.is_file()]
    paths += [ROOT/'codigo/cwru/ampliacion_v2/piloto/datos_piloto.py',
              ROOT/'codigo/cwru/ampliacion_v2/protocolo/reglas_v2.py']
    for base,name in ((PROTOCOL,'MANIFEST_PROTOCOLO_SHA256'),(DNN_CLOSURE,'MANIFEST_CIERRE_PILOTO_SHA256')):
        paths += [base/(name+'.json'),base/(name+'.sha256')]
    entries={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(set(paths))}
    write(manifest,entries)
    manifest.with_suffix('.sha256').write_text(sha(manifest)+'  '+manifest.name+'\n',encoding='ascii')
    verify_manifest(manifest)
    print(json.dumps(dict(status=plan['status'],processes=17,maximum_worker_seconds=780,
        sealed_resources=len(entries),real_models_trained_by_agent=0)))


def preflight():
    entries=verify_manifest(PREP/'MANIFEST_PREPARACION_SHA256.json')
    plan=read(PREP/'PLAN_PILOTO.json')
    data_job,jobs=source_plan()
    require(plan['jobs']==jobs and plan['source_data_job_id']==data_job['job_id']
        and plan['source_plan_sha256']==sha(PROTOCOL/'piloto_plan.json')
        and plan['execution_owner']=='user' and plan['mass_training_enabled'] is False,
        'Plan o responsable de ejecucion distintos')
    return plan,entries


def read_worker_result(path,job):
    try:
        result=read(path)
        validate_result(result,job)
        return result,None
    except (OSError,UnicodeError,json.JSONDecodeError,ValueError,KeyError,TypeError) as error:
        return None,type(error).__name__+': '+str(error)


def run_one(run_dir,job,environment):
    started=time.perf_counter();job_id=job['job_id'];process=None
    code=None;timedout=False;interrupted=False;launch_error=None
    with (run_dir/(job_id+'_stdout.log')).open('x',encoding='utf-8') as out, \
         (run_dir/(job_id+'_stderr.log')).open('x',encoding='utf-8') as err:
        try:
            process=subprocess.Popen([sys.executable,'-B',str(SOURCE/'worker_tiempos.py'),
                '--run-dir',str(run_dir),'--job-id',job_id],cwd=ROOT,env=environment,stdout=out,stderr=err)
            try:code=process.wait(timeout=job['timeout_seconds'])
            except subprocess.TimeoutExpired:
                timedout=True
                if process.poll() is None:process.kill()
                process.wait(timeout=30);code=process.returncode
            except KeyboardInterrupt:
                interrupted=True
                if process.poll() is None:process.kill()
                process.wait(timeout=30);code=process.returncode
        except OSError as error:launch_error=type(error).__name__+': '+str(error)
    result,result_error=read_worker_result(run_dir/job_id/'resultado.json',job)
    passed=not timedout and not interrupted and code==0 and result is not None
    status=dict(job_id=job_id,family=job['family'],status='PASS' if passed else
        'INTERRUPTED' if interrupted else 'TIMEOUT' if timedout else 'FAIL',return_code=code,
        process_elapsed_seconds=time.perf_counter()-started,timeout_seconds=job['timeout_seconds'],
        process_launch_error=launch_error,result_error=result_error,automatic_retry=False)
    write(run_dir/('ESTADO_'+job_id+'.json'),status)
    if passed:result=dict(result,process_elapsed_seconds=status['process_elapsed_seconds'])
    return status,result


def execute():
    plan,entries=preflight()
    expected=plan['tested_environment']
    current=dict(python=platform.python_version(),numpy=importlib.metadata.version('numpy'),
                 sklearn=importlib.metadata.version('scikit-learn'))
    require(current==expected,'Usar el entorno Python comprobado: '+str(expected)+'; actual: '+str(current))
    OUTPUT.mkdir(parents=True,exist_ok=True)
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    write(OUTPUT/'EJECUCION_UNICA.json',dict(run_id=stamp,preparation_manifest_sha256=sha(PREP/'MANIFEST_PREPARACION_SHA256.json')))
    run_dir=OUTPUT/stamp;run_dir.mkdir()
    authorization=dict(plan,source_sha256={n:h for n,h in entries.items() if n.endswith('.py')},
                       run_id=stamp,execution_initiated_by='user_CLI')
    write(run_dir/'EJECUCION_ACOTADA.json',authorization)
    environment=os.environ.copy();environment.update(ENVIRONMENT)
    statuses=[];results=[];started=time.perf_counter()
    for position,job in enumerate(plan['jobs'],1):
        print(f"[{position}/17] INICIO {job['job_id']}; limite {job['timeout_seconds']} s",flush=True)
        status,result=run_one(run_dir,job,environment)
        statuses.append(status)
        print(f"{job['job_id']}: {status['status']}; {status['process_elapsed_seconds']:.3f} s",flush=True)
        if status['status']!='PASS':break
        results.append(result)
    complete=len(results)==17;projection_error=None
    if complete:
        write(run_dir/'RESULTADOS_PARA_PRESUPUESTO.json',dict(classifiers=results[:16],detector=results[16]))
        try:
            from presupuesto_clasicos import project_classical_budget
            write(run_dir/'PROYECCION_PRESUPUESTO_CLASICOS.json',project_classical_budget(results[:16],results[16]))
        except (ValueError,KeyError,TypeError,OverflowError) as error:
            complete=False;projection_error=type(error).__name__+': '+str(error)
    summary=dict(status='PILOT_COMPLETED' if complete else 'PILOT_OUTPUT_VALIDATION_FAILED' if projection_error else
        'PILOT_STOPPED_ON_FIRST_FAILURE',jobs_requested=17,jobs_started=len(statuses),jobs_passed=len(results),
        statuses=statuses,projection_error=projection_error,elapsed_orchestrator_seconds=time.perf_counter()-started,
        observed_worker_seconds=sum(s['process_elapsed_seconds'] for s in statuses),maximum_worker_seconds=780,
        raw_or_load3_arrays_opened=False,TensorFlow_imported=False,mass_training_started=False,
        outer_predictive_evaluation_executed=False,models_not_used_for_selection=True,full_budget_frozen=False,
        STM32_cost_calibration_pending=True)
    write(run_dir/'RESUMEN_PILOTO.json',summary)
    resources=dict(entries)
    for p in sorted(run_dir.rglob('*')):
        if p.is_file():resources[p.relative_to(ROOT).as_posix()]=sha(p)
    for name in ('MANIFEST_PREPARACION_SHA256.json','MANIFEST_PREPARACION_SHA256.sha256'):
        path=PREP/name;resources[path.relative_to(ROOT).as_posix()]=sha(path)
    manifest=run_dir/'MANIFEST_RESULTADOS_SHA256.json';write(manifest,resources)
    manifest.with_suffix('.sha256').write_text(sha(manifest)+'  '+manifest.name+'\n',encoding='ascii')
    verify_manifest(manifest)
    print(('PILOTO COMPLETADO' if complete else 'PILOTO DETENIDO')+f": {len(results)}/17 procesos aprobados.",flush=True)
    print('Evidencias: '+str(run_dir),flush=True)
    if not complete:print(json.dumps(statuses[-1] if statuses else summary,ensure_ascii=True),flush=True)
    return 0 if complete else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--prepare',action='store_true');mode.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    if args.prepare:prepare()
    elif args.execute:raise SystemExit(execute())
    else:
        plan,_=preflight()
        print(json.dumps(dict(status='READ_ONLY_PREFLIGHT_PASS',real_models_trained=0,
            jobs=len(plan['jobs']),maximum_worker_seconds=780,execution_owner='user',execute_required=True)))
