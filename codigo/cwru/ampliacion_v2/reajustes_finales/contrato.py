"""Contrato de reajustes congelados. Sin fit, inferencia o lectura de valores al importar."""
from __future__ import annotations
import copy
import importlib.metadata
import os
from pathlib import Path
import platform
import sys

ROOT=Path(__file__).resolve().parents[4]
SOURCE=Path(__file__).resolve().parent
BASE=ROOT/'codigo/cwru/ampliacion_v2'
for directory in ('busqueda_interna','detector_interno','protocolo','representantes'):
    sys.path.insert(0,str(BASE/directory))
import ejecutar_busqueda_interna as io
import datos_busqueda as inner
from reglas_v2 import component_roles

FROZEN=ROOT/'articulo/ampliacion_cwru/metodologia/seleccion_congelada_v2_2026-10-27'
PREP=ROOT/'articulo/ampliacion_cwru/metodologia/reajustes_finales_v2_2026-10-27'
OUTPUT=ROOT/'resultados/cwru/ampliacion_v2/reajustes_finales_v2_2026-10-27'
PLAN=PREP/'PLAN_REAJUSTES.json'
MANIFEST=PREP/'MANIFEST_PREPARACION_REAJUSTES_SHA256.json'
SELECTION_SIGNATURE='c577339ac0c9450f99d40044850063e7090b19d5e81502eca0378c46e1b2c32f'
FROZEN_MANIFEST_SHA='0907315607b0ebf1e7479b46002b596e1f66ddf73202f4d80cdff1ec1aa88339'
TIMEOUTS={'DNN':300,'RF':90,'SVM':60,'detector':60}
LIMITS=dict(worker_seconds_by_family=TIMEOUTS,campaign_execution_verification_seconds=7200,
    max_batch_seconds=7200,max_new_jobs=131,automatic_retry=False,concurrent_workers=1)


def versions():
    return dict(python=platform.python_version(),**{name:importlib.metadata.version(package)
        for name,package in [('numpy','numpy'),('sklearn','scikit-learn'),('tensorflow','tensorflow'),
            ('keras','keras'),('scipy','scipy'),('joblib','joblib')]})


def sources():
    dependencies=['busqueda_interna/ejecutar_busqueda_interna.py','busqueda_interna/datos_busqueda.py',
        'representantes/ejecutar_representantes.py','representantes/datos_representantes.py',
        'representantes/worker_representantes.py','representantes/numerica_representantes.py',
        'protocolo/reglas_v2.py','detector_interno/datos_detector.py','piloto_clasicos/detector_numerico.py']
    files=[*SOURCE.glob('*.py'),*[BASE/p for p in dependencies]]
    return {p.relative_to(ROOT).as_posix():io.sha(p) for p in sorted(files)}


def frozen_contract():
    manifest=FROZEN/'MANIFEST_SELECCION_CONGELADA_SHA256.json'
    io.require(io.sha(manifest)==FROZEN_MANIFEST_SHA,'Sello de selección distinto del congelado')
    io.verify_manifest(manifest)
    body=io.read(FROZEN/'CUERPO_FIRMA_SELECCION.json')
    lock=io.read(FROZEN/'DECISIONES_CONGELADAS.json')
    scopes=io.read(FROZEN/'PLAN_REAJUSTES_PENDIENTES.json')
    io.require(io.canonical(body)==lock['selection_signature']==SELECTION_SIGNATURE
        and body['refit_scopes']==scopes and len(lock['decisions'])==4,'Selección/firma alterada')
    protocol,rows,hashes=inner.context()
    return scopes,protocol,rows,hashes


def _ids(rows,loads,normal=False):
    return [int(r['global_window_id']) for r in rows if int(r['load_hp']) in loads
        and (not normal or int(r['label'])==0)]


def build_jobs(scopes,protocol,rows):
    """Derivar exactamente 131 trabajos de las decisiones; no volver a seleccionar."""
    jobs=[]
    for original in scopes['DNN_unique_jobs']:
        job=copy.deepcopy(original);job['family']='DNN'
        job['feature_indices']=protocol['candidates'][job['candidate']].copy()
        job['input_dimension']=len(job['feature_indices'])
        jobs.append(job)
    for family,key in [('RF','RF_jobs'),('SVM','SVM_jobs')]:
        for original in scopes[key]:
            job=copy.deepcopy(original);job['family']=family
            seed=str(job['seed']) if job['seed'] is not None else 'none'
            job['job_id']=f"outer{job['outer_load_hp']}_{family}_c{job['config_index']:02d}_s{seed}"
            job['feature_indices']=list(range(14));job['input_dimension']=14
            jobs.append(job)
    for index,original in enumerate(scopes['detector_unique_scopes']):
        job=copy.deepcopy(original);job['family']='detector'
        job['job_id']=f"outer{job['outer_load_hp']}_detector_{index:02d}"
        job['fit_ids']=_ids(rows,job['roles']['detector_fit_normal_loads'],True)
        job['calibration_ids']=_ids(rows,job['roles']['detector_calibration_normal_loads'],True)
        jobs.append(job)
    for job in jobs:
        outer=job['outer_load_hp'];roles=component_roles(outer)
        io.require(type(outer) is int and job['roles']==roles,'Roles de reajuste alterados')
        job['training_global_window_ids']=_ids(rows,roles['training_loads'])
        selected=job['fit_ids']+job['calibration_ids'] if job['family']=='detector' else job['training_global_window_ids']
        job['authorized_feature_source_paths']=sorted({rows[i]['source_features'] for i in selected})
        job['timeout_seconds']=TIMEOUTS[job['family']]
        io.require(selected and all(int(rows[i]['load_hp'])!=outer for i in selected),'Reajuste contiene carga exterior')
        io.require(len({rows[i]['file_id'] for i in job['training_global_window_ids']})==21,'Se requieren 21 MAT de desarrollo')
        if job['family']=='DNN':
            io.require(type(job['epochs']) is int and 1<=job['epochs']<=50
                and job['seed'] in protocol['dnn']['outer_seeds'],'Épocas/semilla no fijadas')
        if job['family']=='detector':
            fit,cal=job['fit_ids'],job['calibration_ids']
            io.require(not set(fit)&set(cal) and set(fit+cal)<=set(job['training_global_window_ids'])
                and len({rows[i]['file_id'] for i in fit})==2
                and len({rows[i]['file_id'] for i in cal})==1,'Detector no separa 2 MAT/1 MAT normales')
    io.require(len(jobs)==len({j['job_id'] for j in jobs})==131
        and {f:sum(j['family']==f for j in jobs) for f in TIMEOUTS}==dict(DNN=80,RF=40,SVM=4,detector=7),
        'Inventario diferente del congelado')
    return jobs


def fit_signature(job,protocol,hashes,sourcehashes,expected_versions):
    return io.canonical(dict(job=job,selection_signature=SELECTION_SIGNATURE,
        source_sha256=sourcehashes,expected_versions=expected_versions,
        input_sha256={p:hashes[p] for p in job['authorized_feature_source_paths']},
        dnn=protocol['dnn'],numeric=protocol['numeric_contract'],quantization=protocol['quantization'],
        detector=protocol['detector'],classical=protocol['classical_comparators'],
        outer_evaluation_enabled=False,early_stopping=False))


def preflight(environment=False):
    io.verify_manifest(MANIFEST)
    plan=io.read(PLAN)
    scopes,protocol,rows,hashes=frozen_contract()
    io.require(plan['jobs']==build_jobs(scopes,protocol,rows) and plan['source_sha256']==sources()
        and plan['selection_signature']==SELECTION_SIGNATURE and plan['limits']==LIMITS
        and plan['worker_environment']==io.ENVIRONMENT,'Plan/fuentes/límites cambiados')
    io.require(not plan['outer_evaluation_enabled'] and not plan['manuscript_edits_enabled']
        and not plan['automatic_retry'] and plan['early_stopping'] is False,'Alcance ampliado')
    io.require(plan['fit_signatures']=={j['job_id']:fit_signature(j,protocol,hashes,plan['source_sha256'],
        plan['expected_versions']) for j in plan['jobs']},'Firmas diferentes')
    if environment:io.require(versions()==plan['expected_versions'],'Usar el Python exacto de la preparación')
    return plan


def campaign_contract(plan):
    return dict(preparation_manifest_sha256=io.sha(MANIFEST),plan_sha256=io.sha(PLAN),
        selection_signature=SELECTION_SIGNATURE,scope='final_refits_only_no_outer_predictions',
        execution_owner='user_CLI',automatic_retry=False)


def authorize(payload_path,output,family):
    """Workers solo reciben NPZ train-only de una solicitud activa firmada."""
    payload_path=Path(payload_path).resolve();output=Path(output).resolve()
    plan=preflight(environment=True)
    io.require(not output.exists() and output.parent.parent==OUTPUT.resolve(),'Salida existente o fuera de campaña')
    campaign=output.parent
    io.require((campaign/'ACTIVO.lock').is_file(),'No hay controlador activo para autorizar el worker')
    marker=io.read(OUTPUT/'CAMPANA_UNICA.json')
    io.require(marker==dict(run_id=campaign.name,preparation_sha256=io.sha(MANIFEST))
        and io.read(campaign/'CONTRATO.json')==campaign_contract(plan),'Campaña ajena al contrato')
    payload=io.read(payload_path);job=payload['job']
    io.require(output==campaign/job['job_id'],'La salida no corresponde al identificador del trabajo')
    expected=next((j for j in plan['jobs'] if j['job_id']==job['job_id']),None)
    io.require(job==expected and job['family']==payload['family']==family
        and payload['status']=='AUTHORIZED_FINAL_REFIT_PAYLOAD','Trabajo/payload no autorizado')
    request=dict(job_id=job['job_id'],fit_signature=plan['fit_signatures'][job['job_id']],
        payload_path=payload_path.relative_to(ROOT).as_posix())
    io.require(io.read(campaign/'SOLICITUD_ACTIVA.json')==request,'No hay solicitud activa correspondiente')
    io.require(payload_path==campaign/'solicitudes'/job['job_id']/'payload.json','Ruta de payload ajena')
    arrays=Path(payload['arrays_path']).resolve()
    io.require(arrays==payload_path.parent/'datos.npz' and io.sha(arrays)==payload['arrays_sha256'],'Arrays modificados o ajenos')
    io.require(payload['fit_signature']==plan['fit_signatures'][job['job_id']]
        and payload['selection_signature']==SELECTION_SIGNATURE and payload['source_sha256']==plan['source_sha256']
        and payload['expected_versions']==plan['expected_versions']
        and payload['worker_environment']==plan['worker_environment'],'Firma/entorno de payload distintos')
    protocol=io.read(io.PROTOCOL/'protocolo_reglas.json')
    for key in ('dnn','numeric_contract','quantization','detector'):
        io.require(payload[key]==protocol[key],'Configuración del payload alterada: '+key)
    io.require(all(os.environ.get(k)==v for k,v in plan['worker_environment'].items()),'Entorno de worker no fijado')
    return payload


def materialize(job,folder,context):
    """NPZ por trabajo; solo seleccionar filas autorizadas, nunca la carga reservada."""
    import numpy as np
    from datos_detector import load_rows
    protocol,rows,_=context
    scopes=io.read(FROZEN/'PLAN_REAJUSTES_PENDIENTES.json')
    expected=next(j for j in build_jobs(scopes,protocol,rows) if j['job_id']==job['job_id'])
    io.require(job==expected,'Trabajo modificado antes de leer valores')
    if job['family']!='detector':
        raw,y,records,training_provenance=load_rows(job['training_global_window_ids'],context=context)
        provenance=dict(training=training_provenance)
        arrays=dict(raw_training=raw,y_training=y,training_ids=np.asarray(job['training_global_window_ids'],dtype=np.int64),
            training_records=np.asarray(records,dtype=str))
    else:
        fit,y_fit,r_fit,p_fit=load_rows(job['fit_ids'],context=context)
        cal,y_cal,r_cal,p_cal=load_rows(job['calibration_ids'],context=context)
        io.require((y_fit==0).all() and (y_cal==0).all(),'Detector recibe muestras no normales')
        arrays=dict(raw_fit=fit,raw_calibration=cal,y_fit=y_fit,y_calibration=y_cal,
            fit_ids=np.asarray(job['fit_ids'],dtype=np.int64),calibration_ids=np.asarray(job['calibration_ids'],dtype=np.int64),
            fit_records=np.asarray(r_fit,dtype=str),calibration_records=np.asarray(r_cal,dtype=str))
        provenance=dict(fit=p_fit,calibration=p_cal)
    with (folder/'datos.npz').open('xb') as stream:np.savez(stream,**arrays)
    return provenance
