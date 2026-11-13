"""Auditar salidas exteriores y documentar resultados sin repetir inferencias."""
from __future__ import annotations
import datetime
import json
import math
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'codigo/cwru/ampliacion_v2/evaluacion_exterior'))
import contrato_evaluacion as c
import ejecutar_evaluacion as evaluator
io=c.io
CAMPAIGN=c.OUTPUT/'20261027T222625_337707Z'
OUT=ROOT/'articulo/ampliacion_cwru/metodologia/cierre_evaluacion_exterior_v2_2026-10-27'


def quantization_summary(result,jobs):
    """Conteos sobre pares modelo-ventana, no muestras físicas independientes."""
    models={m['job_id']:m for j in jobs for m in j['models']}
    total_features=0;changed_clipped=0;changed_unclipped=0;clipped_unchanged=0
    correct_lost=0;correct_gained=0;affected_models=0;by_load={};by_feature={}
    for q in result['quantization_per_model']:
        model=models[q['model_job_id']];outer=q['outer_load_hp']
        changed={r['row_index'] for r in q['class_change_rows']}
        clipped={i for i,value in enumerate(q['input_clipped_by_window']) if value>0}
        total_features+=q['windows']*q['features']
        changed_clipped+=len(changed&clipped);changed_unclipped+=len(changed-clipped)
        clipped_unchanged+=len(clipped-changed);affected_models+=bool(changed)
        correct_lost+=q['FP32_correct_INT8_incorrect'];correct_gained+=q['FP32_incorrect_INT8_correct']
        row=by_load.setdefault(str(outer),dict(model_window_pairs=0,class_changes=0,clipped_feature_values=0,clipped_window_evaluations=0))
        for key,value in [('model_window_pairs',q['windows']),('class_changes',q['class_changes']),
            ('clipped_feature_values',q['input_clipped_feature_values']),('clipped_window_evaluations',len(clipped))]:row[key]+=value
        for index,value in zip(model['indices'],q['input_clipped_by_feature']):
            if value:
                key=f"outer{outer}_{model['candidate']}_feature{index}"
                by_feature[key]=by_feature.get(key,0)+value
    totals=result['quantization_totals']
    io.require(changed_clipped+changed_unclipped==totals['class_changes']
        and changed_clipped+clipped_unchanged==totals['clipped_window_evaluations'],'Conteos cuantización inconsistentes')
    return dict(**totals,total_feature_values=total_features,affected_models=affected_models,
        clipped_feature_value_fraction=totals['clipped_feature_values']/total_features,
        clipped_window_evaluation_fraction=totals['clipped_window_evaluations']/totals['unique_model_window_pairs'],
        changed_on_clipped_window=changed_clipped,changed_on_unclipped_window=changed_unclipped,
        clipped_window_without_class_change=clipped_unchanged,FP32_correct_INT8_incorrect=correct_lost,
        FP32_incorrect_INT8_correct=correct_gained,by_load=by_load,clipped_feature_counts=by_feature,
        clipping_is_not_a_complete_causal_explanation=True,denominators_include_repeated_windows_across_models=True,
        quantization_benefit_not_universal=True)


def main():
    started=time.monotonic();io.require(not OUT.exists(),'Cierre existente; no repetir o sobrescribir')
    plan=c.preflight(environment=True)
    io.require(io.read(c.OUTPUT/'CAMPANA_UNICA.json')==dict(run_id=CAMPAIGN.name,preparation_sha256=io.sha(c.MANIFEST)),
        'Campaña distinta de la preparada')
    io.require(not (CAMPAIGN/'ACTIVO.lock').exists() and not (CAMPAIGN/'SOLICITUD_ACTIVA.json').exists(),
        'Campaña activa o incompleta')
    campaign_manifest=CAMPAIGN/'MANIFEST_CAMPANA_SHA256.json';io.verify_manifest(campaign_manifest)
    summary=io.read(CAMPAIGN/'RESUMEN_CAMPANA.json')
    io.require(summary['status']=='OUTER_EVALUATION_COMPLETED' and summary['completed_loads']==4
        and summary['results_released'] is True and summary['training_executed'] is False,'Campaña incompleta')
    done,seconds=evaluator.inspect_campaign(CAMPAIGN,plan)
    io.require(done==[j['job_id'] for j in plan['jobs']] and len(done)==4
        and math.isclose(seconds,summary['campaign_execution_verification_seconds'],abs_tol=1e-6),'Cierres/tiempos inconsistentes')
    original=io.read(CAMPAIGN/'RESULTADOS_EXTERIORES.json')
    recomputed=evaluator.aggregate_results(CAMPAIGN,plan)
    io.require(io.canonical(original)==io.canonical(recomputed),'Resumen publicado no reconstruible')
    _,rows,_,_,_,_=c.closed_context()
    import numpy as np
    counts={key:0 for key in plan['expected_counts']};canonical=[];models_evaluated=0
    for job in plan['jobs']:
        folder=CAMPAIGN/job['job_id'];result=io.read(folder/'RESULTADO.json')
        models_evaluated+=len(result['model_results'])
        for key in counts:counts[key]+=result['inference_counts'][key]
        ids=job['heldout_global_window_ids']
        with np.load(folder/'entrada_evaluacion.npz',allow_pickle=False) as a:
            io.require(a['global_window_ids'].tolist()==ids
                and a['y_true'].tolist()==[int(rows[i]['label']) for i in ids]
                and a['record_ids'].tolist()==[str(rows[i]['file_id']) for i in ids],
                'Etiquetas o MAT no corresponden al índice sellado')
        by_id={m['job_id']:m for m in job['models']}
        for row in result['model_results']:
            if row['family']=='DNN' and row['seed']==42:
                model=by_id[row['model_job_id']]
                canonical.append(dict(model_job_id=row['model_job_id'],outer_load_hp=job['outer_load_hp'],
                    candidate=row['candidate'],seed=42,epochs=row['epochs'],indices=row['indices'],
                    paths=model['paths'],artifact_sha256=model['artifact_sha256'],fit_signature=model['fit_signature'],
                    procedures=[name for name,p in job['procedure_mapping'].items()
                        if p['canonical_hardware_job_id']==row['model_job_id']],
                    metrics=row['metrics'],quantization_report=row['quantization_report'],
                    pending_hardware_validation=True,seed_selected_from_outer_performance=False))
    io.require(counts==plan['expected_counts'] and models_evaluated==131 and len(canonical)==8,'Inventario exterior incompleto')
    quant=quantization_summary(original,plan['jobs'])
    worst_quant=min(original['quantization_per_model'],key=lambda q:q['INT8_minus_FP32_macro_f1_pp'])
    best_quant=max(original['quantization_per_model'],key=lambda q:q['INT8_minus_FP32_macro_f1_pp'])
    groups=original['summary']['groups'];by_key={(g['procedure'],g['precision']):g for g in groups}
    selected=[('baseline_F14','INT8'),('performance_only','INT8'),('validated_total_cost','INT8'),('RF','native'),('SVM','native')]
    labels={'baseline_F14':'F14 base','performance_only':'Selector por rendimiento','validated_total_cost':'Selector por costo total','RF':'RF','SVM':'SVM'}
    table='\n'.join(f"| {labels[p]} | {by_key[p,m]['mean_macro_f1']*100:.4f} | {by_key[p,m]['mean_accuracy']*100:.4f} |" for p,m in selected)
    load_table='\n'.join('| '+labels[p]+' | '+' | '.join(f"{row['macro_f1_across_seeds']['mean']*100:.4f}" for row in by_key[p,m]['per_load'])+' |' for p,m in selected)
    paired_table='\n'.join(f"| {labels[r['left']]} − {labels[r['right']]} | {r['mean_macro_f1_difference_pp']:+.4f} | {r['mean_accuracy_difference_pp']:+.4f} |" for r in original['paired_INT8_differences'])
    detector_rows=[r for r in original['detector_cells'] if r['procedure']=='baseline_F14']
    rejected_N=sum(r['rejected_windows'] for cell in detector_rows for r in cell['per_record'] if r['label']==0)
    total_N=sum(r['windows'] for cell in detector_rows for r in cell['per_record'] if r['label']==0)
    detector_scope=dict(normal_specificity_equal_load_mean=.5,IR_recall_equal_load_mean=1.,OR_recall_equal_load_mean=1.,
        balanced_accuracy_equal_load_mean=.75,normal_rejected_windows=rejected_N,normal_windows=total_N,
        window_weighted_normal_rejection_fraction=rejected_N/total_N,
        equal_load_specificity_is_not_window_weighted_specificity=True,unique_normal_MAT_count=4,
        new_motor_independence_claimed=False)
    OUT.mkdir(parents=True,exist_ok=False)
    io.write(OUT/'METRICAS_Y_LIMITES.json',dict(groups=groups,paired=original['paired_INT8_differences'],
        detector_summary=original['detector_summary'],detector_cells=original['detector_cells'],
        detector_scope=detector_scope,quantization=quant,canonical_seed42_is_not_an_average=True,
        outer_results_cannot_change_rules=True,original_results_sha256=io.sha(CAMPAIGN/'RESULTADOS_EXTERIORES.json')))
    io.write(OUT/'MODELOS_CANONICOS_42_PENDIENTES_STM32.json',dict(status='PENDING_SEPARATE_HARDWARE_VALIDATION',
        models=canonical,unique_models=8,seed=42,seed_fixed_before_outer_evaluation=True,hardware_execution_enabled=False))
    closure=dict(status='OUTER_EVALUATION_COMPLETE_AUDITED_PENDING_FINAL_HARDWARE_AND_REPORTING',
        date_local='2026-10-27',timezone='America/Lima',verified_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        campaign=CAMPAIGN.relative_to(ROOT).as_posix(),campaign_manifest_sha256=io.sha(campaign_manifest),
        selection_signature=plan['selection_signature'],preparation_manifest_sha256=io.sha(c.MANIFEST),
        loads_completed=4,heldout_windows=1380,MAT_records=28,models_evaluated=131,
        logical_supervised_cells=284,inference_counts=counts,worker_and_verification_seconds=seconds,
        failures=0,retries=0,new_training_fits=0,new_classifier_invocations_during_audit=0,
        saved_output_metrics_reconstructed=True,detector_scores_recomputed_for_integrity=True,
        audit_seconds=time.monotonic()-started,TensorFlow_imported='tensorflow' in sys.modules,
        models_scalers_or_thresholds_modified=False,selection_modified=False,manuscript_modified=False,hardware_connected=False,
        retrospective_evaluation=True,window_IID_claimed=False,new_motor_generalization_demonstrated=False)
    io.require(closure['TensorFlow_imported'] is False,'Auditoría no requiere runtime TensorFlow')
    io.write(OUT/'CIERRE_EVALUACION_EXTERIOR.json',closure)
    text=f'''# Evaluación exterior CWRU V2: resultados auditados y límites

Campaña `{CAMPAIGN.name}`: **4/4 cargas PASS**, sin fallos o reintentos.
Firmas, versiones, modelos, scalers, umbrales, snapshots y predicciones fueron
verificados. Los resúmenes se reconstruyeron desde las salidas guardadas;
no se entrenó o volvió a invocar un clasificador. Los scores deterministas
del detector se recalcularon únicamente para comprobar integridad.

Tiempo registrado de procesos y verificación: **{seconds:.3f} s**. No es tiempo
de inferencia por ventana en STM32 ni incluye todos los pasos compartidos o
esta auditoría posterior. Se evaluaron 1380 ventanas de 28 MAT mediante
80 DNN, 40 RF, 4 SVM y 7 detectores únicos. Los aliases no son nuevos ajustes.

## Resultados supervisados

Medias con igual peso de las cuatro cargas y, cuando corresponde, diez
semillas. Las DNN de la tabla usan INT8; RF/SVM conservan su precisión nativa.

| Procedimiento | Macro-F1 medio (%) | Exactitud media (%) |
|---|---:|---:|
{table}

La comparación completa FP32/INT8, las métricas por clase, las dispersiones
descriptivas y los modelos canónicos están en los JSON adjuntos. La desviación
entre semillas condiciona el mismo corpus; no estima variación entre motores.
SVM tiene un único ajuste por carga, no diez réplicas artificiales.

| Macro-F1 por carga (%) | 0 HP | 1 HP | 2 HP | 3 HP |
|---|---:|---:|---:|---:|
{load_table}

| Diferencia pareada INT8 | Macro-F1 (pp) | Exactitud (pp) |
|---|---:|---:|
{paired_table}

## Interpretación que puede sostenerse

RF obtuvo el mayor macro-F1 predictivo en este corpus. No corresponde afirmar
superioridad general de la DNN o del selector por costo. Los tamaños joblib o
tiempos PC no representan su viabilidad, Flash o latencia en el microcontrolador.

Costo mejora macro-F1 medio frente a F14, pero pierde exactitud. En 0 HP,
T6 aumenta recall N de 0 a 95 %, mientras recall IR baja de 95 a 12.7419 %;
no mejora todas las clases. En 1 HP FE7 pierde 4.7625 puntos de macro-F1 frente
a F14; el margen de selección interna no garantizaba preservación exterior
por carga. En 2 HP los resultados son cercanos.

La ventaja de costo frente al selector por rendimiento depende fuertemente
del respaldo F14 en 3 HP: NF12 obtiene macro-F1 medio 65.2045 %, frente a
99.0594 % del respaldo. Este último NO alcanzó el objetivo temporal interno;
no se atribuirá ese beneficio al ahorro de cómputo. Se conservó la regla fijada,
sin elegir retrospectivamente máscara o semilla después de ver los resultados.

0 HP es la principal limitación de generalización por carga. Sus medias deben
mostrarse junto a los promedios globales. La semilla canónica 42 permanece
fijada para hardware; no representa necesariamente el centro de las diez
semillas y no se sustituirá por la mejor ejecución exterior.

## Detector y cuantización

Los tres procedimientos tienen la misma especificidad N media por carga:
**50 %**, recall IR/OR **100 %** y balanced accuracy **75 %**. La especificidad
por carga 0/1/2/3 HP es 0/100/100/0 %. Rechaza los 28 N de MAT 97 y los
58 N de MAT 100. El rechazo de {rejected_N}/{total_N} ventanas N es
**{100*rejected_N/total_N:.4f} %** si se pondera por ventana: no debe confundirse
con el promedio de cuatro cargas. Solo hay cuatro MAT normales. El detector
no es un confirmador fiable de fallas ante cambios de carga con estos datos;
el recall alto de falla debe presentarse junto al rechazo de señales normales.
No se recalibrará el umbral usando esta evaluación.

FP32–INT8 registra **{quant['class_changes']} cambios en {quant['affected_models']}/80 modelos**,
entre 27 600 pares modelo-ventana. Hay {quant['clipped_feature_values']} valores
recortados de {quant['total_feature_values']} y {quant['clipped_window_evaluations']}
evaluaciones de ventana con clipping. {quant['changed_on_clipped_window']} cambios
ocurren en ventanas con clipping y {quant['changed_on_unclipped_window']} sin él;
{quant['clipped_window_without_class_change']} ventanas recortadas no cambian de clase.
No se atribuirán todos los cambios causalmente al clipping ni se afirmará
beneficio universal de cuantización. Los denominadores repiten ventanas entre
modelos/semillas y no son adquisiciones independientes.

La mayor pérdida observada es **{worst_quant['INT8_minus_FP32_macro_f1_pp']:.4f} pp**
en `{worst_quant['model_job_id']}` y la mayor ganancia
**{best_quant['INT8_minus_FP32_macro_f1_pp']:+.4f} pp** en
`{best_quant['model_job_id']}`. Los cambios son muy variables entre semillas;
una diferencia pequeña en la media no permite declarar cuantización neutra
para todas las ejecuciones o cargas.

## Trabajo siguiente

Mantener este cierre como resultado del protocolo fijado. La evaluación es
retrospectiva sobre un corpus previamente explorado y no prueba independencia
de un motor nuevo; no se calculan intervalos IID o pruebas de no inferioridad
sobre ventanas solapadas. No ajustar decisiones confirmatorias para ocultar
resultados desfavorables. Cualquier modificación futura requeriría delimitarse
como un nuevo estudio exploratorio.

Quedan las validaciones PC–STM32 y mediciones físicas de los ocho modelos
canónicos con semilla 42, utilizando sus parámetros actuales, y la redacción
de estos resultados con las limitaciones indicadas. Los costos medidos durante
la selección pertenecen a representantes internos; todavía no sustituyen las
mediciones de esos pipelines finales. No se modificó el manuscrito.
'''
    (OUT/'RESULTADOS_Y_LIMITES.md').write_text(text,encoding='utf-8')
    extra=[Path(__file__),c.MANIFEST,c.MANIFEST.with_suffix('.sha256'),c.PLAN,
        c.OUTPUT/'CAMPANA_UNICA.json',c.CLOSED_MANIFEST,c.CLOSED_MANIFEST.with_suffix('.sha256')]
    extra += [p for p in CAMPAIGN.rglob('*') if p.is_file()]
    manifest=OUT/'MANIFEST_CIERRE_EVALUACION_EXTERIOR_SHA256.json'
    io.write_manifest(manifest,directory=OUT,extra=extra)
    print(json.dumps(dict(status=closure['status'],loads=4,seconds=seconds,
        summary=[dict(procedure=p,precision=m,F1=by_key[p,m]['mean_macro_f1'],accuracy=by_key[p,m]['mean_accuracy']) for p,m in selected],
        manifest_sha256=io.sha(manifest),output=str(OUT)),ensure_ascii=True))


if __name__=='__main__':main()
