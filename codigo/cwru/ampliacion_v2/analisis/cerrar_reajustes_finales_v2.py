"""Cerrar los 131 reajustes y sellar modelos antes de la evaluación exterior."""
from __future__ import annotations
import datetime
import json
import math
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'codigo/cwru/ampliacion_v2/reajustes_finales'))
import contrato as c
import ejecutar_reajustes as executor

CAMPAIGN=c.OUTPUT/'20261027T173452_969615Z'
OUT=ROOT/'articulo/ampliacion_cwru/metodologia/cierre_reajustes_finales_v2_2026-10-27'
FIRST=ROOT/'articulo/ampliacion_cwru/metodologia/cierre_primer_reajuste_final_v2_2026-10-27'


def inventory(plan):
    """Inventario de rutas y huellas, sin volver a seleccionar o inferir."""
    values=[]
    for job in plan['jobs']:
        folder=CAMPAIGN/job['job_id'];result=c.io.read(folder/'RESULTADO.json')
        names={'DNN':['modelo.keras','modelo_fp32.tflite','modelo_int8.tflite','scaler_train_only.npz'],
            'RF':['modelo.joblib'],'SVM':['modelo.joblib','scaler_train_only.npz'],
            'detector':['parametros.npz','calibracion.npz']}[job['family']]
        paths={name:(folder/name).relative_to(ROOT).as_posix() for name in names}
        values.append(dict(job_id=job['job_id'],family=job['family'],outer_load_hp=job['outer_load_hp'],
            seed=job.get('seed'),candidate=job.get('candidate'),indices=job.get('feature_indices',job.get('indices')),
            epochs=job.get('epochs'),config_index=job.get('config_index'),roles=job['roles'],
            procedures=job.get('procedures'),masks=job.get('masks'),
            fit_signature=result['fit_signature'],selection_signature=result['selection_signature'],
            result_path=(folder/'RESULTADO.json').relative_to(ROOT).as_posix(),
            result_sha256=c.io.sha(folder/'RESULTADO.json'),paths=paths,
            artifact_sha256={name:c.io.sha(ROOT/path) for name,path in paths.items()},
            manifest_path=(folder/'MANIFEST_JOB_SHA256.json').relative_to(ROOT).as_posix(),
            manifest_sha256=c.io.sha(folder/'MANIFEST_JOB_SHA256.json')))
    return values


def evaluation_map(plan,items,rows):
    lock=c.io.read(c.FROZEN/'DECISIONES_CONGELADAS.json')
    by_id={r['job_id']:r for r in items};result=[]
    for decision in lock['decisions']:
        outer=decision['outer_load_hp'];procedures={}
        choices={'baseline_F14':decision['baseline'],'performance_only':decision['performance']['choice'],
            'validated_total_cost':decision['cost']['choice']}
        for name,choice in choices.items():
            matches=[j for j in plan['jobs'] if j['family']=='DNN' and j['outer_load_hp']==outer
                and j['candidate']==choice['candidate'] and j['epochs']==choice['final_epochs']]
            c.io.require([j['seed'] for j in matches]==[11,22,33,42,44,55,66,77,88,99],
                'Semillas finales incompletas')
            detector=next(j for j in plan['jobs'] if j['family']=='detector'
                and j['outer_load_hp']==outer and choice['candidate'] in j['masks'])
            procedures[name]=dict(candidate=choice['candidate'],epochs=choice['final_epochs'],
                DNN_by_seed={str(j['seed']):j['job_id'] for j in matches},detector_job_id=detector['job_id'],
                canonical_hardware_job_id=next(j['job_id'] for j in matches if j['seed']==42))
        rf=[j for j in plan['jobs'] if j['family']=='RF' and j['outer_load_hp']==outer]
        svm=[j for j in plan['jobs'] if j['family']=='SVM' and j['outer_load_hp']==outer]
        c.io.require(len(rf)==10 and len(svm)==1,'Comparadores finales incompletos')
        selected=[r for r in rows if int(r['load_hp'])==outer]
        ids=[int(r['global_window_id']) for r in selected]
        records=sorted({r['file_id'] for r in selected},key=int)
        c.io.require(len(records)==7,'Cada evaluación por carga debe conservar siete MAT')
        result.append(dict(outer_load_hp=outer,heldout_global_window_ids=ids,heldout_MAT=records,
            heldout_windows=len(ids),DNN_procedures=procedures,
            RF_by_seed={str(j['seed']):j['job_id'] for j in rf},SVM_job_id=svm[0]['job_id']))
    referenced={job_id for fold in result for value in fold['DNN_procedures'].values()
        for job_id in list(value['DNN_by_seed'].values())+[value['detector_job_id']]}
    referenced.update(job_id for fold in result for job_id in list(fold['RF_by_seed'].values())+[fold['SVM_job_id']])
    c.io.require(referenced==set(by_id),'Modelos huérfanos o referencia fuera del inventario')
    return dict(status='MODEL_MAPPING_FIXED_PENDING_SEPARATE_OUTER_EVALUATION_EXECUTOR',
        selection_signature=c.SELECTION_SIGNATURE,outer_scenarios=result,inventory_job_ids=sorted(by_id),
        outer_evaluation_enabled=False,outer_predictions_released=False,automatic_execution=False,
        rule_changes_after_outer_results_allowed=False,selection_per_seed_allowed=False,
        seed_ensembles=False,canonical_hardware_seed=42,detector_class_override=False,
        previously_explored_dataset=True,retrospective_evaluation=True,new_motor_independence_claimed=False,
        evaluation_feature_values_materialized_here=False)


def main():
    io=c.io;started=time.monotonic()
    io.require(not OUT.exists(),'Cierre existente; no sobrescribir o repetir')
    plan=c.preflight(environment=True)
    io.require(io.read(c.OUTPUT/'CAMPANA_UNICA.json')==dict(run_id=CAMPAIGN.name,
        preparation_sha256=io.sha(c.MANIFEST)),'Campaña distinta de la preparada')
    io.require(not (CAMPAIGN/'ACTIVO.lock').exists() and not (CAMPAIGN/'SOLICITUD_ACTIVA.json').exists(),
        'Campaña activa o incompleta')
    io.verify_manifest(FIRST/'MANIFEST_CIERRE_PRIMER_REAJUSTE_SHA256.json')
    batch_rows=[]
    for batch in sorted((CAMPAIGN/'lotes').iterdir()):
        io.verify_manifest(batch/'MANIFEST_LOTE_SHA256.json')
        summary=io.read(batch/'RESUMEN_LOTE.json')
        io.require(summary['status'] in ('BATCH_COMPLETED','FINAL_REFITS_COMPLETED_NO_OUTER_EVALUATION')
            and summary['jobs_started']==summary['jobs_passed'] and summary['outer_evaluation_executed'] is False,
            'Lote fallido o ámbito distinto')
        batch_rows.append(summary)
    io.require(len(batch_rows)==2 and batch_rows[-1]['status']=='FINAL_REFITS_COMPLETED_NO_OUTER_EVALUATION'
        and batch_rows[-1]['completed_jobs']==131 and batch_rows[-1]['remaining_jobs']==0,'Campaña no completa')
    # Reutiliza el auditor preparado: scalers/pesos/historia DNN, replay nativo
    # RF/SVM sobre entrenamiento y reconstrucción determinista de parámetros K=1.
    done,consumed=executor.inspect_campaign(CAMPAIGN,plan)
    io.require(done==[j['job_id'] for j in plan['jobs']] and len(done)==131,'Inventario completo no verificado')
    io.require(math.isclose(consumed,batch_rows[-1]['campaign_execution_verification_seconds'],abs_tol=1e-6)
        and consumed<=c.LIMITS['campaign_execution_verification_seconds'],'Contador de campaña inconsistente')
    io.require(sum(j['epochs'] for j in plan['jobs'] if j['family']=='DNN')==2400,'Épocas distintas de las fijadas')
    from resumir_reajustes_finales_v2 import summarize
    diagnostics=summarize(plan,CAMPAIGN)
    items=inventory(plan);_,_,rows,_=c.frozen_contract()
    mapping=evaluation_map(plan,items,rows)
    OUT.mkdir(parents=True,exist_ok=False)
    io.write(OUT/'INVENTARIO_MODELOS_FINALES_131.json',items)
    io.write(OUT/'MAPA_PARA_EVALUACION_EXTERIOR.json',mapping)
    io.write(OUT/'DIAGNOSTICOS_ENTRENAMIENTO_Y_TIEMPOS.json',diagnostics)
    closure=dict(status='ALL_FINAL_REFITS_COMPLETE_AUDITED_PENDING_OUTER_EVALUATION',
        date_local='2026-10-27',timezone='America/Lima',verified_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        campaign=CAMPAIGN.relative_to(ROOT).as_posix(),selection_signature=c.SELECTION_SIGNATURE,
        preparation_manifest_sha256=io.sha(c.MANIFEST),completed_jobs=131,counts=plan['counts'],
        remaining_jobs=0,failures=0,retries=0,batches=len(batch_rows),DNN_epochs=2400,
        DNN_optimizer_updates=sum(io.read(CAMPAIGN/j['job_id']/'RESULTADO.json')['optimizer_updates']
            for j in plan['jobs'] if j['family']=='DNN'),
        campaign_execution_verification_seconds=consumed,sum_batch_wall_seconds=sum(r['batch_wall_seconds'] for r in batch_rows),
        timing_excludes_gap_between_user_commands=True,audit_seconds=time.monotonic()-started,
        new_independent_model_fits=0,new_DNN_TFLite_invocations=0,
        technical_verification=dict(DNN_scalers_reconstructed=80,SVM_scalers_reconstructed=4,
            RF_training_prediction_batches_replayed=40,SVM_training_prediction_batches_replayed=4,
            detector_parameter_scopes_reconstructed=7,these_are_not_independent_experiments=True),
        outer_evaluation_executed=False,outer_predictions_released=False,
        heldout_values_selected_or_materialized=False,TensorFlow_imported='tensorflow' in sys.modules,
        new_hardware_validation=False,manuscript_modified=False,
        engineering_fallback_outer3_preserved=True,absolute_detector_quality_not_guaranteed=True,
        immutable_model_inventory_before_outer_evaluation=True)
    io.require(closure['DNN_optimizer_updates']==80840 and closure['TensorFlow_imported'] is False,
        'Updates o runtime inesperados')
    io.write(OUT/'CIERRE_REAJUSTES_FINALES.json',closure)
    text=f'''# Reajustes finales completos y auditados

Campaña `{CAMPAIGN.name}`: **131/131 trabajos PASS**, sin fallos, reintentos
o pendientes. Se verificaron las cuatro decisiones, las fuentes, las versiones,
los payloads de datos, las firmas y los manifiestos de trabajo/paso/lote.

| Familia | Trabajos finalizados | Alcance |
|---|---:|---|
| DNN | 80 | Ocho ámbitos × diez semillas; 2400 épocas y 80 840 actualizaciones |
| RF | 40 | Cuatro configuraciones seleccionadas × diez semillas |
| SVM | 4 | Una configuración seleccionada por carga reservada |
| Detector | 7 | Normales de dos cargas para ajuste y de otra para calibración |

Tiempo acumulado de ejecución/verificación: **{consumed:.3f} s (18 min 52.315 s)**.
La suma de los relojes de ambos lotes es **{closure['sum_batch_wall_seconds']:.3f} s**;
ninguno incluye el intervalo entre comandos del usuario. Los tiempos son de PC,
no latencia STM32. La auditoría posterior se mide aparte.

El inventario guarda las rutas y huellas de todos los modelos, scalers y
parámetros. El mapa asocia de antemano los tres procedimientos DNN, RF, SVM y
detector con las cuatro cargas exteriores, sin volver a elegir máscaras,
semillas, épocas o umbrales. Conserva los ocho modelos DNN canónicos con
semilla 42 y los alias entre procedimientos.

## Qué se comprobó

Los reajustes DNN usan exactamente las épocas congeladas, los pesos finales y
el orden de calibración de sus ventanas de desarrollo. Se reconstruyeron
scalers, pesos e historias, y las métricas se recalcularon desde salidas de
entrenamiento guardadas, sin volver a invocar TFLite. RF/SVM se recargaron y
reprodujeron sus predicciones nativas de entrenamiento. Se reconstruyeron
scaler, centroide y P95 del detector y se contrastó la calibración. Estas
comprobaciones no constituyen nuevos experimentos independientes.

Los diagnósticos del archivo adjunto son exclusivamente de entrenamiento.
La cercanía FP32–INT8 y las métricas altas aquí no prueban generalización.
Los contenedores INT8 de T6 y FE7 son ligeramente mayores que los FP32;
no se afirmará ahorro universal de archivo ni se equipararán bytes TFLite
con Flash del firmware. Los avisos de deprecación se conservan en los logs;
no se cambian las bibliotecas fijadas al cerrar esta campaña.

## Siguiente etapa

Preparar y comprobar un ejecutor separado para una evaluación exterior
controlada de todos los modelos y procedimientos ya fijados. Debe consumir
solo el inventario sellado, generar predicciones por ID de ventana y métricas
por clase, carga, semilla y MAT, conservar diagnósticos FP32–INT8 y del detector,
y presentar resultados completos sin seleccionar semillas favorables o
cambiar decisiones con los resultados. No se crean nuevas particiones.

Esta auditoría **no ejecutó esa evaluación**. Las cargas reservadas lo están
respecto de cada reajuste; el corpus ya se exploró anteriormente y el estudio
continúa siendo retrospectivo, sin independencia de un nuevo motor. Las
semillas y ventanas solapadas no son adquisiciones físicas independientes.

El respaldo F14 del escenario 3 sigue registrado como objetivo no alcanzado:
no cumple margen predictivo interno ni ahorro temporal. La baja especificidad
interna del detector permanece como limitación. Se evaluarán sin modificar
umbrales ni convertir el detector en sustituto de la clase DNN.

No se modificaron modelos, datos, protocolo, firmware o manuscrito. Después
de la evaluación exterior quedará validar en STM32 los pipelines canónicos
finales y consolidar la redacción con sus límites.
'''
    (OUT/'RESULTADOS_Y_SIGUIENTE_ETAPA.md').write_text(text,encoding='utf-8')
    extra=[Path(__file__),Path(sys.modules['resumir_reajustes_finales_v2'].__file__),c.MANIFEST,
        c.MANIFEST.with_suffix('.sha256'),c.PLAN,c.OUTPUT/'CAMPANA_UNICA.json',
        FIRST/'MANIFEST_CIERRE_PRIMER_REAJUSTE_SHA256.json',FIRST/'MANIFEST_CIERRE_PRIMER_REAJUSTE_SHA256.sha256']
    extra += [p for p in CAMPAIGN.rglob('*') if p.is_file()]
    extra += [ROOT/p for p in diagnostics['source_sha256']]
    manifest=OUT/'MANIFEST_CIERRE_REAJUSTES_FINALES_SHA256.json'
    io.write_manifest(manifest,directory=OUT,extra=extra)
    print(json.dumps(dict(status=closure['status'],counts=plan['counts'],seconds=consumed,
        DNN_epochs=2400,DNN_updates=80840,manifest_sha256=io.sha(manifest),output=str(OUT)),ensure_ascii=True))


if __name__=='__main__':main()
