"""Audita y consolida la matriz DNN interna completa; sin nuevos fits/inferencia.

Reconstruye comprobaciones numéricas con artefactos guardados y conserva
todos los sellos previos. No selecciona máscaras ni evalúa cargas externas.
"""
from __future__ import annotations

import datetime
import importlib.abc
import json
from pathlib import Path
import statistics
import sys
import time
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'codigo/cwru/ampliacion_v2/continuacion_interna'))


class ReadOnlyAudit(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in ('tensorflow','keras','ai_edge_litert'):
            raise RuntimeError('Cierre sin runtime ML: no entrenar ni inferir')
        return None


sys.meta_path.insert(0,ReadOnlyAudit())
import continuar_restantes as control
e=control.e
CAMPAIGN=control.CAMPAIGN
OUT=ROOT/'articulo/ampliacion_cwru/metodologia/cierre_matriz_dnn_interna_v2_2026-10-26'


def compatible_result(result,family):
    """Adaptar solo la lectura del P0 histórico, sin reescribir su firma."""
    if family != 'suplemento_P0':
        return result
    e.require(result['mode'] == 'P0','No incorporar otro preprocesamiento al cierre')
    mapped=dict(result)
    for target,source in (('outer_load_hp','outer'),('inner_validation_load_hp','inner'),
        ('FP32_INT8_class_changes','class_changes'),('training_saturated_input_values','training_clipped_values')):
        e.require(target not in result or result[target] == result[source],'Alias histórico contradictorio')
        mapped[target]=result[source]
    return mapped


def main():
    started=time.perf_counter()
    e.require(not OUT.exists(),'No sobrescribir un cierre previo')
    e.preflight(check_environment=True)
    e.verify_manifest(control.MANIFEST)
    e.require(not list(CAMPAIGN.glob('*.lock')),'Hay una ejecución o controlador activo')
    print('Verificando firmas, checkpoints, escaladores y métricas de los240 ajustes nuevos...',flush=True)
    # Una instantánea de metadatos inmutables evita releer el manifiesto de
    # todo el corpus y el plan de5.7MB por cada celda. Se verifica nuevamente
    # al acabar; las comprobaciones numéricas/artefactos por ajuste se ejecutan.
    import datos_busqueda as data
    original_context=data.context
    original_read=e.read
    context_snapshot=original_context()
    context_digest=e.canonical(context_snapshot)
    plan_snapshot=original_read(e.PLAN_FILE)
    plan_digest=e.sha(e.PLAN_FILE)
    def snapshot_read(path):
        return plan_snapshot if Path(path).resolve() == e.PLAN_FILE.resolve() else original_read(path)
    with patch.object(data,'context',return_value=context_snapshot),patch.object(e,'read',side_effect=snapshot_read):
        plan,jobs,done,consumed=e.inspect_campaign(CAMPAIGN)
    e.require(e.canonical(original_context()) == context_digest
        and e.canonical(context_snapshot) == context_digest and e.sha(e.PLAN_FILE) == plan_digest,
        'Cambió la instantánea de datos/plan durante la auditoría')
    e.require(len(done) == len(jobs) == 240,'Matriz nueva incompleta')
    closed=control.closed_snapshot(CAMPAIGN)
    e.require(closed[2:] == (done,consumed),'Discrepan las dos verificaciones del cierre')
    receipts=sorted(control.RECEIPTS.iterdir())
    e.require(len(receipts) == 1,'Revisar recibos adicionales antes de este cierre')
    receipt=receipts[0]
    e.verify_manifest(receipt/'MANIFEST_CONTROL_SHA256.json')
    result_control=e.read(receipt/'RESUMEN_CONTINUACION.json')
    e.require(result_control['status'] == 'CAMPAIGN_COMPLETED'
        and result_control['completed_new_fits'] == 240 and result_control['remaining_new_fits'] == 0
        and result_control['available_logical_cells'] == 288
        and abs(result_control['consumed_worker_seconds']-consumed) < 1e-7,'Recibo de controlador inconsistente')
    batches=sorted((CAMPAIGN/'lotes').iterdir())
    statuses=[]
    for batch in batches:
        statuses += [e.read(path) for path in sorted(batch.glob('ESTADO_*.json'))]
    e.require(len(statuses) == 240 and len({s['job_id'] for s in statuses}) == 240
        and all(s['status'] == 'PASS' and not s['automatic_retry'] for s in statuses),
        'Estados duplicados, incompletos o fallidos')
    e.require(all(s['timeout_seconds'] == 300 for s in statuses) and consumed < 7200,
        'Límites de ejecución incompatibles')
    matrix=[]
    source_folders=[]
    for job in plan['jobs']:
        imported=plan['imported'].get(job['job_id'])
        folder=ROOT/imported['source_directory'] if imported else CAMPAIGN/job['job_id']
        source_folders.append(folder)
        result=compatible_result(e.read(folder/'resultado.json'),imported['family'] if imported else 'new_inner_campaign')
        e.require(all(result[k] == job[k] for k in ('candidate','outer_load_hp','inner_validation_load_hp','seed','roles')),
            'Resultado con roles incompatibles')
        row=dict(job_id=job['job_id'],candidate=job['candidate'],outer_load_hp=job['outer_load_hp'],
            inner_validation_load_hp=job['inner_validation_load_hp'],seed=job['seed'],
            training_windows=result['training_windows'],validation_windows=result['validation_windows'],
            accuracy_FP32=result['metrics']['FP32']['accuracy'],macro_f1_FP32=result['metrics']['FP32']['macro_f1'],
            accuracy_INT8=result['metrics']['INT8']['accuracy'],macro_f1_INT8=result['metrics']['INT8']['macro_f1'],
            best_epoch=result['best_epoch'],epochs_executed=result['epochs_executed'],
            class_changes_FP32_INT8=result['FP32_INT8_class_changes'],
            training_clipped_feature_values=result['training_saturated_input_values'],
            validation_clipped_feature_values=result['int8_details']['input_clipped_values'],
            source_family=imported['family'] if imported else 'new_inner_campaign',
            source_directory=folder.relative_to(ROOT).as_posix(),source_fit_signature=result['fit_signature'],
            current_semantic_signature=plan['semantic_signatures'][job['job_id']],
            model_sha256=result['model_sha256'],source_result_sha256=e.sha(folder/'resultado.json'))
        matrix.append(row)
    e.require(len(matrix) == 288 and len({(r['outer_load_hp'],r['inner_validation_load_hp'],r['candidate'],r['seed'])
        for r in matrix}) == 288,'Cuadrícula no completa o duplicada')
    order=list(dict.fromkeys(j['candidate'] for j in plan['jobs']))
    aggregates=[]
    per_mask=[]
    for mask in order:
        current=[r for r in matrix if r['candidate'] == mask]
        e.require(len(current) == 36,'Una máscara carece de36 celdas')
        scores=[r['macro_f1_INT8'] for r in current]
        per_mask.append(dict(candidate=mask,cells=36,mean_macro_f1_INT8=statistics.mean(scores),
            minimum_cell_macro_f1_INT8=min(scores),maximum_cell_macro_f1_INT8=max(scores)))
        for outer in range(4):
            rows=[r for r in current if r['outer_load_hp'] == outer]
            e.require(len(rows) == 9 and {r['seed'] for r in rows} == {11,22,33},'Bloque interno incompleto')
            per_load={str(inner):statistics.mean(r['macro_f1_INT8'] for r in rows
                if r['inner_validation_load_hp'] == inner) for inner in range(4) if inner != outer}
            aggregates.append(dict(candidate=mask,outer_load_hp=outer,cells=9,
                mean_macro_f1_INT8=statistics.mean(r['macro_f1_INT8'] for r in rows),
                mean_macro_f1_FP32=statistics.mean(r['macro_f1_FP32'] for r in rows),
                mean_INT8_by_inner_validation_load=per_load,worst_load_mean_macro_f1_INT8=min(per_load.values()),
                median_best_epoch=int(statistics.median(r['best_epoch'] for r in rows)),
                no_selection_executed=True))
    epochs=[r['epochs_executed'] for r in matrix]
    budget_remaining=7200-consumed
    report=dict(status='INNER_DNN_MATRIX_COMPLETE_AUDITED_PENDING_DETECTOR_CLASSICAL_AND_SELECTION',
        date_local='2026-10-26',timezone='America/Lima',
        campaign=CAMPAIGN.relative_to(ROOT).as_posix(),controller_receipt=receipt.relative_to(ROOT).as_posix(),
        cells=288,new_fits=240,reused_fits=48,reused_representatives=32,reused_supplement_P0=16,
        new_fits_remaining=0,inner_seeds=[11,22,33],outer_scenarios=4,internal_loads_per_scenario=3,masks=8,
        batches_total=len(batches),batches_in_continuation=result_control['batches_launched'],
        worker_controller_budget_seconds_consumed=consumed,
        worker_controller_budget_minutes_consumed=consumed/60,budget_seconds_remaining=budget_remaining,
        budget_limit_seconds=7200,automatic_retries=0,failures=0,
        timing_scope='New240 only; controller subprocess execution plus verification, excluding repeated shared audits and reused48.',
        epochs=dict(minimum=min(epochs),mean=statistics.mean(epochs),maximum=max(epochs),
            cells_reaching_maximum50=epochs.count(50),cells_stopping_before50=sum(v<50 for v in epochs)),
        quantization=dict(cells_with_class_changes=sum(r['class_changes_FP32_INT8']>0 for r in matrix),
            summed_class_changes_over_repeated_validation_passes=sum(r['class_changes_FP32_INT8'] for r in matrix),
            cells_with_training_clipping=sum(r['training_clipped_feature_values']>0 for r in matrix),
            cells_with_validation_clipping=sum(r['validation_clipped_feature_values']>0 for r in matrix),
            counts_are_not_unique_windows_or_independent_samples=True),
        per_mask_descriptive=per_mask,
        detector_inventory=e.read(e.PREP/'AUDITORIA_REUTILIZACION.json')['detector_inventory'],
        numeric_reconstruction_of_all240_new_jobs=True,original48_signatures_and_manifests_preserved=True,
        metadata_plan_snapshot_verified_before_and_after=True,
        original_source_result_schema_preserved=True,
        preparation_manifest_sha256=e.sha(e.PREP_MANIFEST),continuation_manifest_sha256=e.sha(control.MANIFEST),
        new_training_fits_by_auditor=0,new_TFLite_invocations_by_auditor=0,
        TensorFlow_imported='tensorflow' in sys.modules,hardware_connected=False,
        outer_predictive_evaluation_executed=False,feature_selection_executed=False,
        manuscript_modified=False,old_artifacts_modified=False,
        next_stage='Complete19 unique normality parameter scopes and60 remaining logical inner rows; classical grids and frozen selection precede outer evaluation.',
        qualifications=['No mask is selected by the global descriptive table.',
            'Each outer selection must use its nine internal scores and detector/cost guards.',
            'Seeds/folds and repeated validation windows are dependent; no IID confidence interval is claimed.',
            'The corpus remains retrospectively explored; this does not create a virgin test.'],
        audit_seconds=time.perf_counter()-started)
    OUT.mkdir(parents=True,exist_ok=False)
    e.write(OUT/'MATRIZ_INTERNA_288.json',matrix)
    e.write(OUT/'RESUMEN_POR_ESCENARIO_MASCARA.json',aggregates)
    e.write(OUT/'CIERRE_MATRIZ_INTERNA.json',report)
    table='\n'.join(f"| {r['candidate']} | {r['mean_macro_f1_INT8']*100:.4f} | {r['minimum_cell_macro_f1_INT8']*100:.4f} |"
        for r in per_mask)
    text=f'''# Cierre de la matriz DNN interna CWRU V2

Estado: `{report['status']}`. Se completaron288/288 celdas:240 ajustes nuevos
y48 reutilizados por firma (32 representantes y16 P0). Cuatro escenarios
exteriores, tres validaciones internas por escenario, ocho máscaras y semillas
11/22/33. No se entrenó ni ejecutó inferencia nueva para esta auditoría.

Se contrastaron todos los cierres/sellos, checkpoint restaurado, historia de
parada temprana, scaler train-only, cuantización, predicciones y métricas de
los240 nuevos. Las48 referencias mantienen sus firmas y artefactos auditados.
El controlador terminó14 lotes adicionales; son{len(batches)} lotes en total.
Todos los240 estados nuevos son PASS, sin reintentos. Tiempo contabilizado:
{consumed:.3f}s ({consumed/60:.2f}min), frente a7200s disponibles. Excluye las
auditorías compartidas entre lotes y los ajustes reutilizados; no equivale a
duración total de reloj. No se amplió ningún límite.

## Resumen descriptivo de las36 celdas de cada máscara

| Máscara | Macro-F1 INT8 medio (%) | Peor celda (%) |
|---|---:|---:|
{table}

La tabla reúne escenarios y semillas dependientes; no es evaluación exterior
ni una selección global. La comparación de cada escenario utilizará sus
nueve resultados internos, las guardias del detector y el costo total ya
validado. Las métricas completas por celda y las medias por escenario están
en los dos JSON de esta carpeta. No se calcularon intervalos IID sobre ventanas.

## Alcance y siguiente etapa

La matriz DNN queda cerrada. El detector conserva36/96 filas lógicas y23/42
ámbitos de parámetros: completar19 ámbitos únicos y60 filas, sin multiplicar
por semillas DNN. Después, completar las búsquedas clásicas, aplicar las
guardias y congelar las decisiones de los cuatro escenarios. Solo entonces
corresponden los reajustes y la evaluación exterior. El piloto temporal RF/SVM
ya cerrado no se repite. No se seleccionaron máscaras ni se editaron el
manuscrito, firmware, dataset o evidencias anteriores.
'''
    (OUT/'RESULTADOS_Y_SIGUIENTE_ETAPA.md').write_text(text,encoding='utf-8')
    files=[p for p in OUT.iterdir() if p.is_file()]
    files += [Path(__file__).resolve(),e.PLAN_FILE,e.PREP_MANIFEST,e.PREP_MANIFEST.with_suffix('.sha256'),
        control.MANIFEST,control.MANIFEST.with_suffix('.sha256'),CAMPAIGN/'CONTRATO_CAMPANA.json',
        e.OUTPUT/'CAMPANA_UNICA.json']
    files += [p for p in receipt.iterdir() if p.is_file()]
    for folder in source_folders:
        files += [p for p in folder.iterdir() if p.is_file()]
    for batch in batches:
        files += [p for p in batch.iterdir() if p.is_file()]
    manifest=OUT/'MANIFEST_CIERRE_MATRIZ_SHA256.json'
    e.write_manifest(manifest,extra=files)
    print(json.dumps(dict(status=report['status'],cells=288,new_fits=240,reused_fits=48,
        new_fits_remaining=0,consumed_minutes=consumed/60,
        manifest_files=len(e.read(manifest)),manifest_sha256=e.sha(manifest),
        TensorFlow_imported='tensorflow' in sys.modules,output=str(OUT)),ensure_ascii=True))


if __name__ == '__main__':
    main()
