"""Contrato operacional de evaluación; preparar no materializa valores reservados."""
from __future__ import annotations
import copy
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[4]
SOURCE=Path(__file__).resolve().parent
BASE=ROOT/'codigo/cwru/ampliacion_v2'
for directory in ('reajustes_finales','busqueda_interna','detector_interno','representantes','protocolo'):
    sys.path.insert(0,str(BASE/directory))
import contrato as refit
io=refit.io

CLOSED=ROOT/'articulo/ampliacion_cwru/metodologia/cierre_reajustes_finales_v2_2026-10-27'
CLOSED_MANIFEST=CLOSED/'MANIFEST_CIERRE_REAJUSTES_FINALES_SHA256.json'
CLOSED_SHA256='5d90a9dc4e91b6d9295acd593878e31bf48c5b9e6117cb910ed52a618986bc8b'
PREP=ROOT/'articulo/ampliacion_cwru/metodologia/evaluacion_exterior_v2_2026-10-27'
OUTPUT=ROOT/'resultados/cwru/ampliacion_v2/evaluacion_exterior_v2_2026-10-27'
PLAN=PREP/'PLAN_EVALUACION.json'
MANIFEST=PREP/'MANIFEST_PREPARACION_EVALUACION_SHA256.json'
LIMITS=dict(worker_seconds=300,campaign_execution_verification_seconds=1800,
    workers=4,concurrent_workers=1,automatic_retry=False,training_enabled=False)


def sources():
    dependencies=['reajustes_finales/contrato.py','busqueda_interna/ejecutar_busqueda_interna.py',
        'busqueda_interna/datos_busqueda.py','detector_interno/datos_detector.py',
        'representantes/ejecutar_representantes.py','representantes/datos_representantes.py',
        'representantes/worker_representantes.py','representantes/numerica_representantes.py',
        'protocolo/reglas_v2.py','piloto_clasicos/detector_numerico.py']
    files=[*SOURCE.glob('*.py'),*[BASE/p for p in dependencies]]
    return {p.relative_to(ROOT).as_posix():io.sha(p) for p in sorted(files)}


def closed_context():
    io.require(io.sha(CLOSED_MANIFEST)==CLOSED_SHA256,'Cierre final distinto del auditado')
    io.verify_manifest(CLOSED_MANIFEST)
    closure=io.read(CLOSED/'CIERRE_REAJUSTES_FINALES.json')
    mapping=io.read(CLOSED/'MAPA_PARA_EVALUACION_EXTERIOR.json')
    inventory=io.read(CLOSED/'INVENTARIO_MODELOS_FINALES_131.json')
    refit_plan=io.read(refit.PLAN)
    _,protocol,rows,hashes=refit.frozen_contract()
    io.require(closure['status']=='ALL_FINAL_REFITS_COMPLETE_AUDITED_PENDING_OUTER_EVALUATION'
        and closure['completed_jobs']==131 and closure['failures']==closure['remaining_jobs']==0
        and closure['selection_signature']==mapping['selection_signature']==refit.SELECTION_SIGNATURE,
        'No se han cumplido los prerrequisitos de evaluación')
    return protocol,rows,hashes,mapping,inventory,refit_plan


def build_jobs(protocol,rows,mapping,inventory,refit_plan):
    io.require(len(inventory)==131 and len({r['job_id'] for r in inventory})==131,'Inventario final incompleto')
    by_id={j['job_id']:j for j in refit_plan['jobs']}
    io.require(len(refit_plan['jobs'])==len(by_id)==131,'Reajustes duplicados o incompletos')
    io.require(set(by_id)=={r['job_id'] for r in inventory},'Inventario y reajustes no coinciden')
    io.require([fold['outer_load_hp'] for fold in mapping['outer_scenarios']]==[0,1,2,3],'Cuatro cargas requeridas')
    jobs=[]
    for fold in mapping['outer_scenarios']:
        outer=fold['outer_load_hp'];io.require(type(outer) is int,'Carga no entera')
        selected=[r for r in rows if int(r['load_hp'])==outer]
        ids=[int(r['global_window_id']) for r in selected]
        records=sorted({r['file_id'] for r in selected},key=int)
        io.require(ids==fold['heldout_global_window_ids'] and records==fold['heldout_MAT']
            and len(records)==7 and ids==sorted(set(ids)),'Partición de evaluación alterada')
        models=[copy.deepcopy(r) for r in inventory if r['outer_load_hp']==outer]
        counts={f:sum(m['family']==f for m in models) for f in ('DNN','RF','SVM','detector')}
        io.require(counts==dict(DNN=20,RF=10,SVM=1,detector=1 if outer==3 else 2),'Modelos por carga incorrectos')
        for model in models:
            trained=by_id[model['job_id']]
            io.require(type(model['outer_load_hp']) is int and model['family']==trained['family']
                and io.canonical(model['roles'])==io.canonical(trained['roles'])==io.canonical(refit.component_roles(outer))
                and outer not in model['roles']['training_loads']
                and not set(ids)&set(trained['training_global_window_ids']), 'Evaluación y entrenamiento solapados')
            expected_indices=trained.get('feature_indices',trained.get('indices'))
            io.require(model['indices']==expected_indices and all(type(i) is int for i in model['indices'])
                and all(io.canonical(model.get(k))==io.canonical(trained.get(k))
                    for k in ('seed','candidate','epochs','config_index')),'Identidad del modelo distinta del reajuste')
            required={'DNN':{'modelo.keras','modelo_fp32.tflite','modelo_int8.tflite','scaler_train_only.npz'},
                'RF':{'modelo.joblib'},'SVM':{'modelo.joblib','scaler_train_only.npz'},
                'detector':{'parametros.npz','calibracion.npz'}}[model['family']]
            io.require(set(model['paths'])==set(model['artifact_sha256'])==required,'Artefactos incompletos')
            io.require(all(Path(path).name==name and Path(path).parent==Path(model['result_path']).parent
                for name,path in model['paths'].items()),'Artefactos fuera del directorio del modelo')
            if model['family']=='detector':
                io.require(not set(ids)&set(trained['fit_ids']+trained['calibration_ids']), 'Detector calibrado en evaluación')
            for name,path in model['paths'].items():
                io.require(io.sha(io.confined(ROOT,path))==model['artifact_sha256'][name],'Modelo/parámetro final modificado')
        local={m['job_id']:m for m in models};references=set()
        io.require(set(fold['DNN_procedures'])=={'baseline_F14','performance_only','validated_total_cost'},'Procedimientos alterados')
        for name,procedure in fold['DNN_procedures'].items():
            expected_seeds=[str(s) for s in protocol['dnn']['outer_seeds']]
            io.require(list(procedure['DNN_by_seed'])==expected_seeds,'Semillas incompletas o reordenadas')
            for seed,job_id in procedure['DNN_by_seed'].items():
                model=local[job_id]
                io.require(model['family']=='DNN' and model['seed']==int(seed)
                    and model['candidate']==procedure['candidate'] and model['epochs']==procedure['epochs'],
                    'Alias corresponde a otro clasificador')
                references.add(job_id)
            detector=local[procedure['detector_job_id']]
            io.require(detector['family']=='detector' and procedure['candidate'] in detector['masks'], 'Alias detector distinto')
            references.add(detector['job_id'])
            io.require(procedure['canonical_hardware_job_id']==procedure['DNN_by_seed']['42'],'Semilla canónica alterada')
        io.require(list(fold['RF_by_seed'])==[str(s) for s in protocol['dnn']['outer_seeds']], 'Semillas RF incompletas')
        for seed,job_id in fold['RF_by_seed'].items():
            io.require(local[job_id]['family']=='RF' and local[job_id]['seed']==int(seed),'Alias RF distinto')
            references.add(job_id)
        io.require(local[fold['SVM_job_id']]['family']=='SVM' and local[fold['SVM_job_id']]['seed'] is None,'SVM debe evaluarse una vez')
        references.add(fold['SVM_job_id'])
        io.require(references==set(local),'Modelos huérfanos o repetición no autorizada')
        n=len(ids)
        jobs.append(dict(job_id=f'outer_eval_{outer}',outer_load_hp=outer,heldout_global_window_ids=ids,
            heldout_MAT=records,heldout_windows=n,models=models,procedure_mapping=copy.deepcopy(fold['DNN_procedures']),
            RF_by_seed=copy.deepcopy(fold['RF_by_seed']),SVM_job_id=fold['SVM_job_id'],
            authorized_feature_source_paths=sorted({r['source_features'] for r in selected}),
            timeout_seconds=LIMITS['worker_seconds'],expected_counts=counts,
            expected_inference_counts=dict(DNN_FP32_invocations=20*n,DNN_INT8_invocations=20*n,
                RF_predicted_rows=10*n,SVM_predicted_rows=n,detector_scored_rows=counts['detector']*n)))
    io.require(sum(j['heldout_windows'] for j in jobs)==1380,'Corpus de evaluación distinto')
    return jobs


def signature(job,sourcehashes,versions,inputhashes):
    return io.canonical(dict(job=job,source_sha256=sourcehashes,versions=versions,
        input_sha256={p:inputhashes[p] for p in job['authorized_feature_source_paths']},
        selection_signature=refit.SELECTION_SIGNATURE,final_refit_manifest_sha256=CLOSED_SHA256,
        training_enabled=False,calibration_enabled=False,outer_evaluation_only=True))


def preflight(environment=False):
    io.verify_manifest(MANIFEST);plan=io.read(PLAN)
    protocol,rows,hashes,mapping,inventory,refit_plan=closed_context()
    io.require(plan['jobs']==build_jobs(protocol,rows,mapping,inventory,refit_plan)
        and plan['source_sha256']==sources() and plan['limits']==LIMITS
        and plan['worker_environment']==io.ENVIRONMENT,'Plan, fuentes o límites alterados')
    io.require(plan['selection_signature']==refit.SELECTION_SIGNATURE
        and plan['final_refit_manifest_sha256']==CLOSED_SHA256,'Selección o cierre final distinto')
    io.require(plan['evaluation_signatures']=={j['job_id']:signature(j,plan['source_sha256'],plan['expected_versions'],hashes)
        for j in plan['jobs']},'Firmas de evaluación distintas')
    io.require(plan['outer_evaluation_enabled'] is True and plan['training_enabled'] is False
        and plan['calibration_enabled'] is False and plan['partial_metric_release_enabled'] is False,
        'Alcance de ejecución alterado')
    if environment:io.require(refit.versions()==plan['expected_versions'],'Entorno Python diferente del fijado')
    return plan


def campaign_contract(plan):
    return dict(preparation_manifest_sha256=io.sha(MANIFEST),plan_sha256=io.sha(PLAN),
        final_refit_manifest_sha256=CLOSED_SHA256,selection_signature=refit.SELECTION_SIGNATURE,
        execution_owner='user_CLI',scope='controlled_outer_evaluation_without_training',partial_metric_release=False)


def authorize(campaign,outer):
    campaign=Path(campaign).resolve();plan=preflight(environment=True)
    io.require(type(outer) is int and outer in range(4),'Carga fuera de 0..3')
    io.require(campaign.parent==OUTPUT.resolve() and io.read(campaign/'CONTRATO.json')==campaign_contract(plan),
        'Campaña ajena al contrato')
    io.require(io.read(OUTPUT/'CAMPANA_UNICA.json')==dict(run_id=campaign.name,preparation_sha256=io.sha(MANIFEST))
        and (campaign/'ACTIVO.lock').is_file(),'Campaña no activa')
    job=plan['jobs'][outer]
    io.require(not (campaign/job['job_id']).exists(),'No sobrescribir o repetir evaluación iniciada')
    io.require(io.read(campaign/'SOLICITUD_ACTIVA.json')==dict(job_id=job['job_id'],outer_load_hp=outer,
        evaluation_signature=plan['evaluation_signatures'][job['job_id']]),'Solicitud activa distinta')
    io.require(all(os.environ.get(k)==v for k,v in plan['worker_environment'].items()),'Entorno worker no fijado')
    return plan,job
