"""Cerrar el detector interno ya ejecutado; sin nuevos ajustes o DNN."""
from __future__ import annotations

import json
from pathlib import Path
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'codigo/cwru/ampliacion_v2/detector_interno'))
import ejecutar_detector as detector
e=detector.base
CAMPAIGN=detector.OUTPUT/'20261027T031125_681822Z'
OUT=ROOT/'articulo/ampliacion_cwru/metodologia/cierre_detector_interno_v2_2026-10-26'


def main():
    started=time.perf_counter()
    e.require(not OUT.exists(),'No sobrescribir un cierre anterior')
    plan=detector.validate_campaign(CAMPAIGN)
    manifest=CAMPAIGN/'MANIFEST_CAMPANA_SHA256.json'
    e.verify_manifest(manifest)
    e.require(not list(CAMPAIGN.glob('*.lock')),'Campaña aún activa')
    summary=e.read(CAMPAIGN/'RESUMEN_DETECTOR_INTERNO.json')
    worker=e.read(CAMPAIGN/'ESTADO_WORKER.json')
    e.require(summary['status'] == 'DETECTOR_INTERNAL_COMPLETED'
        and summary['logical_rows'] == 96 and summary['parameter_scopes'] == 42
        and summary['new_parameter_fits'] == 19 and summary['reused_parameter_scopes'] == 23
        and summary['new_unique_evaluations'] == 54 and summary['reused_unique_evaluations'] == 30
        and summary['new_logical_rows'] == 60 and summary['reused_logical_rows'] == 36,
        'Conteos del cierre incompatibles')
    e.require(worker['status'] == 'PASS' and worker['return_code'] == 0 and worker['error'] is None
        and worker['timeout_seconds'] == 300 and not worker['automatic_retry']
        and worker['worker_process_seconds'] == summary['worker_process_seconds'] < 300,
        'Worker fallido o fuera del presupuesto')
    e.require(not any(summary[k] for k in ('DNN_fits','outer_evaluation_executed','selection_executed',
        'STM32_cost_measured','manuscript_modified','automatic_retry'))
        and summary['no_DNN_class_override'], 'Alcance extendido')
    inventory=plan['inventory']
    new_params=[j['param_id'] for j in inventory['parameter_jobs'] if j['param_id'] not in inventory['reused_parameters']]
    new_evals=[j['eval_id'] for j in inventory['evaluation_jobs'] if j['eval_id'] not in inventory['reused_evaluations']]
    e.require({p.name for p in (CAMPAIGN/'parametros').iterdir()} == set(new_params)
        and {p.name for p in (CAMPAIGN/'evaluaciones').iterdir()} == set(new_evals),'Unidades inesperadas o faltantes')
    events=[json.loads(line) for line in (CAMPAIGN/'progreso.jsonl').read_text(encoding='utf-8').splitlines()]
    e.require([event['stage'] for event in events] == ['PARAMETER_PASS']*19+['EVALUATION_PASS']*54
        and [event['param_id'] for event in events[:19]] == new_params
        and [event['eval_id'] for event in events[19:]] == new_evals, 'Traza no corresponde al plan')
    e.require(e.read(CAMPAIGN/'CIERRE_WORKER.json') == dict(status='PASS',new_parameter_fits=19,
        new_unique_evaluations=54,DNN_fits=0,TensorFlow_imported=False),'Cierre del worker distinto')
    print('Verificando parámetros, umbrales, scores y métricas por MAT...',flush=True)
    rows=detector.verify_outputs(plan,CAMPAIGN)
    e.require(rows == e.read(CAMPAIGN/'MATRIZ_DETECTOR_96.json'),'Matriz guardada distinta de su reconstrucción')
    order=inventory['protocol']['candidate_order']
    aggregates=[]
    for outer in range(4):
        for candidate in order:
            selected=[r for r in rows if r['outer_load_hp'] == outer and r['candidate'] == candidate]
            e.require(len(selected) == 3,'Faltan cargas de validación')
            row=dict(outer_load_hp=outer,candidate=candidate,internal_validation_rows=3)
            for metric in ('normal_spec','ir_recall','or_recall','balanced_accuracy'):
                row['mean_'+metric]=statistics.mean(r[metric] for r in selected)
                row['minimum_internal_load_'+metric]=min(r[metric] for r in selected)
            aggregates.append(row)
    # Confirmar alias de representación; no multiplicar su evidencia.
    for outer in range(4):
        for inner in range(4):
            if inner == outer:continue
            aliases=[next(r for r in rows if r['outer_load_hp']==outer
                and r['inner_validation_load']==inner and r['candidate']==candidate) for candidate in ('F14','NF12')]
            e.require(aliases[0]['param_id'] == aliases[1]['param_id']
                and aliases[0]['eval_id'] == aliases[1]['eval_id'] and aliases[0]['metrics'] == aliases[1]['metrics'],
                'Alias F14/NF12 no exacto')
    reference=[r for r in aggregates if r['candidate'] == 'F14']
    F14_rows=[r for r in rows if r['candidate'] == 'F14']
    F14_mean={metric:statistics.mean(r[metric] for r in F14_rows)
        for metric in ('normal_spec','ir_recall','or_recall','balanced_accuracy')}
    by_validation=[dict(inner_validation_load=load,logical_rows=24,
        **{metric:statistics.mean(r[metric] for r in rows if r['inner_validation_load']==load)
           for metric in ('normal_spec','ir_recall','or_recall','balanced_accuracy')}) for load in range(4)]
    report=dict(status='INTERNAL_DETECTOR_COMPLETE_AUDITED_PENDING_CLASSICAL_AND_SELECTION',
        date_local='2026-10-26',timezone='America/Lima',campaign=CAMPAIGN.relative_to(ROOT).as_posix(),
        logical_rows=96,parameter_scopes=42,unique_evaluations=84,
        reused_parameter_scopes=23,new_parameter_fits=19,reused_unique_evaluations=30,new_unique_evaluations=54,
        reused_logical_rows=36,new_logical_rows=60,worker_process_seconds=worker['worker_process_seconds'],
        worker_limit_seconds=300,failed_units=0,automatic_retries=0,
        scores_replayed_for_unique_evaluations=84,new_parameter_scalers_centroids_P95_reconstructed=19,
        all_original_signatures_and_payloads_preserved=True,F14_NF12_aliases_verified=True,
        N_fit_calibration_validation_MAT_scopes_separated=True,refit_after_calibration=False,
        no_DNN_class_override=True,reference_F14_by_outer_internal_mean=reference,
        reference_F14_global_descriptive_mean=F14_mean,
        reference_F14_internal_cells_with_zero_normal_specificity=sum(r['normal_spec']==0 for r in F14_rows),
        descriptive_means_by_inner_validation_load=by_validation,
        primary_guards='Frozen relative F14 margins; absolute0.95 supplement checks are not substituted.',
        operational_PASS_is_not_predictive_acceptance=True,
        source_campaign_manifest_sha256=e.sha(manifest),preparation_manifest_sha256=e.sha(detector.MANIFEST),
        new_detector_fit_normality_calls_by_auditor=0,new_DNN_fits_or_inferences_by_auditor=0,
        existing_scores_replayed_not_new_evidence=True,TensorFlow_imported='tensorflow' in sys.modules,
        hardware_connected=False,manuscript_modified=False,selection_executed=False,outer_evaluation_executed=False,
        next_stage='Prepare frozen RF/SVM inner grids with explicit budget; do not repeat classical timing pilot.',
        qualifications=['The84 unique evaluations expand to96 logical rows; aliases are not independent evidence.',
            'Metrics use per-record rates averaged within class, then equal validation loads.',
            'Preserving a weak reference does not establish adequate absolute detector performance.',
            'No threshold tuning, normality refit or validation-based scaler calibration follows these results.'],
        audit_seconds=time.perf_counter()-started)
    OUT.mkdir(parents=True,exist_ok=False)
    e.write(OUT/'CIERRE_DETECTOR_INTERNO.json',report)
    e.write(OUT/'RESUMEN_POR_ESCENARIO_MASCARA.json',aggregates)
    table='\n'.join(f"| {r['outer_load_hp']} | {100*r['mean_normal_spec']:.2f} | {100*r['mean_ir_recall']:.2f} | {100*r['mean_or_recall']:.2f} | {100*r['mean_balanced_accuracy']:.2f} |" for r in reference)
    text=f'''# Detector interno completo y auditado

Estado: `{report['status']}`. Quedan cerradas96/96 filas lógicas,
42/42 ámbitos de parámetros y84 evaluaciones únicas. Se reutilizaron23
parámetros y30 evaluaciones; el usuario ejecutó19ajustes y54evaluaciones
nuevos. Las60 filas lógicas añadidas incluyen alias, no60scores independientes.

El worker terminó en{worker['worker_process_seconds']:.3f}s, dentro de300s,
sin errores o reintentos. Este tiempo no incluye la verificación del padre.
La auditoría contrastó sellos, las73unidades nuevas, el scaler/centroide/P95
de los19ajustes y los scores, decisiones y métricas por MAT de84evaluaciones.
Se preservaron todos los artefactos y firmas originales. No se entrenó una
DNN, ajustó otra vez el detector o utilizó STM32/TensorFlow para este cierre.

## Referencia F14: promedios de las tres cargas internas por escenario

| Escenario exterior reservado (HP) | Especificidad N (%) | Recall IR (%) | Recall OR (%) | BA (%) |
|---|---:|---:|---:|---:|
{table}

Estas son métricas internas; la carga exterior indicada no se evaluó en ese
escenario. Los valores por máscara y escenario están en el JSON. Las tasas
se calculan por MAT y se promedian por clase y carga, sin ponderar por número
de ventanas. F14/NF12 comparten parámetros/evaluaciones en los12roles.

Limitación observada: F14 tiene una especificidad N media descriptiva de
{100*F14_mean['normal_spec']:.2f}% y BA de{100*F14_mean['balanced_accuracy']:.2f}%,
aunque los recalls IR/OR son100%. En{sum(r['normal_spec']==0 for r in F14_rows)}/12
celdas F14 se rechazan todas las ventanas etiquetadas N. El recall elevado no
debe presentarse sin esta pérdida de aceptación de normales. El JSON incluye
además promedios por carga de validación, sin atribuir una causa a las diferencias.

PASS confirma ejecución válida; no declara que las tasas sean adecuadas para
una aplicación. Las guardias principales comparan con F14 mediante los
márgenes ya fijados: conservar una referencia deficiente no demuestra una
buena detección absoluta. El umbral usa solo N de calibración y no se modifica
después de observar validación. El detector no reemplaza la clase de la DNN,
ni se repite por semilla. No se aplica aquí la guardia absoluta0.95 del
suplemento como si fuera la regla principal.

La DNN interna288/288 y el detector96/96 quedan disponibles. Siguiente etapa:
preparar las búsquedas clásicas RF/SVM y su presupuesto, preservando el piloto
temporal ya cerrado. Después, aplicar las guardias y congelar la selección
antes de los reajustes/evaluación exterior. No se eligieron máscaras ni se
modificaron protocolo, dataset, modelos, firmware o manuscrito.
'''
    (OUT/'RESULTADOS_Y_SIGUIENTE_ETAPA.md').write_text(text,encoding='utf-8')
    files=[p for p in OUT.iterdir() if p.is_file()]
    files += [Path(__file__).resolve(),detector.PLAN,detector.MANIFEST,detector.MANIFEST.with_suffix('.sha256')]
    files += [p for p in CAMPAIGN.rglob('*') if p.is_file()]
    e.write_manifest(OUT/'MANIFEST_CIERRE_DETECTOR_SHA256.json',extra=files)
    print(json.dumps(dict(status=report['status'],logical_rows=96,parameter_scopes=42,unique_evaluations=84,
        worker_process_seconds=worker['worker_process_seconds'],TensorFlow_imported='tensorflow' in sys.modules,
        manifest_sha256=e.sha(OUT/'MANIFEST_CIERRE_DETECTOR_SHA256.json'),
        output=str(OUT)),ensure_ascii=True))


if __name__=='__main__':
    main()
