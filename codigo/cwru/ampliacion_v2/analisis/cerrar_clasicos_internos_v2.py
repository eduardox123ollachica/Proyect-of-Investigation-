"""Auditoría de RF/SVM internos terminados; sin ajustes o selección nuevos."""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import json
import statistics
import sys
import time
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'codigo/cwru/ampliacion_v2/clasicos_internos'))
import ejecutar_clasicos as classical
io=classical.io
CAMPAIGN=classical.OUTPUT/'20261027T060928_938422Z'
OUT=ROOT/'articulo/ampliacion_cwru/metodologia/cierre_clasicos_internos_v2_2026-10-27'


def mean(rows,key):return statistics.mean(r[key] for r in rows)


def main():
    started=time.perf_counter()
    io.require(not OUT.exists(),'No sobrescribir un cierre anterior')
    original_manifest=CAMPAIGN/'MANIFEST_CAMPANA_SHA256.json'
    io.verify_manifest(original_manifest)
    io.require(not list(CAMPAIGN.glob('*.lock')) and not (CAMPAIGN/'SOLICITUD_ACTIVA.json').exists(),
        'Campaña activa o solicitud sin cerrar')
    summary=io.read(CAMPAIGN/'RESUMEN_CAMPANA.json')
    io.require(summary['status'] == 'CLASSICAL_INNER_COMPLETED' and summary['completed_fits'] == 384
        and summary['RF_completed'] == 288 and summary['SVM_completed'] == 96
        and summary['remaining_fits'] == 0,'Campaña no completa')
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    context_snapshot=classical.data.context();context_hash=io.canonical(context_snapshot)
    original_context=classical.data.context;original_read=io.read
    plan_snapshot=original_read(classical.PLAN);plan_hash=io.sha(classical.PLAN)
    def snapshot_read(path):
        return plan_snapshot if Path(path).resolve() == classical.PLAN.resolve() else original_read(path)
    print('Verificando384 modelos, scopes, scalers y predicciones guardadas; fit bloqueado...',flush=True)
    with patch.object(classical.data,'context',return_value=context_snapshot),\
         patch.object(io,'read',side_effect=snapshot_read),\
         patch.object(RandomForestClassifier,'fit',side_effect=RuntimeError('Auditoría sin entrenamiento')) as rf_fit,\
         patch.object(SVC,'fit',side_effect=RuntimeError('Auditoría sin entrenamiento')) as svm_fit:
        plan,done,consumed=classical.inspect_campaign(CAMPAIGN)
        io.require(not rf_fit.called and not svm_fit.called,'Un clasificador intentó entrenarse durante la auditoría')
    io.require(io.canonical(original_context()) == io.canonical(context_snapshot) == context_hash
        and io.sha(classical.PLAN) == plan_hash,'Metadatos o plan cambiados durante la auditoría')
    io.require(len(done) == 384 and abs(consumed-summary['campaign_execution_verification_seconds']) < 1e-6
        and consumed < plan['limits']['campaign_execution_verification_seconds'],'Conteo/presupuesto no reproducible')
    matrix=[];training_groups=defaultdict(list)
    for index,job in enumerate(plan['jobs']):
        result=io.read(CAMPAIGN/job['job_id']/'resultado.json')
        state=io.read(CAMPAIGN/'pasos'/f'{index:03d}.json')
        io.require(state['return_code'] == 0 and state['error'] is None,'Paso no aprobado')
        row=dict(job_id=job['job_id'],family=job['family'],config_index=job['config_index'],seed=job['seed'],
            params=job['params'],outer_load_hp=job['outer_load_hp'],inner_validation_load_hp=job['inner_validation_load_hp'],
            training_windows=result['training_windows'],validation_windows=result['validation_windows'],
            macro_f1=result['metrics']['macro_f1'],accuracy=result['metrics']['accuracy'],metrics=result['metrics'],
            normal_recall=result['metrics']['per_class'][0]['recall'],
            fit_signature=result['fit_signature'],model_sha256=result['model_sha256'],model_bytes=result['model_bytes'],
            support_vectors_dtype=result['support_vectors_dtype'],execution_verification_seconds=state['execution_verification_seconds'],
            result_sha256=io.sha(CAMPAIGN/job['job_id']/'resultado.json'))
        matrix.append(row)
        training_key=io.canonical(dict(family=job['family'],params=job['params'],
            training_ids=job['training_global_window_ids'],versions=plan['tested_versions'],
            normalization=result['normalization']))
        training_groups[training_key].append(row)
    io.require(len(training_groups) == 192 and all(len(group) == 2
        and len({row['model_sha256'] for row in group}) == 1 for group in training_groups.values()),
        'Identidades numéricas entre roles con entrenamiento compartido difieren')
    family_summary=[];by_load=[];by_config=[]
    for family in ('RF','SVM'):
        rows=[r for r in matrix if r['family'] == family]
        scores=[r['macro_f1'] for r in rows]
        family_summary.append(dict(family=family,cells=len(rows),mean_macro_f1=statistics.mean(scores),
            median_macro_f1=statistics.median(scores),minimum_macro_f1=min(scores),maximum_macro_f1=max(scores),
            execution_verification_seconds=sum(r['execution_verification_seconds'] for r in rows),
            distinct_model_hashes=len({r['model_sha256'] for r in rows})))
        for load in range(4):
            selected=[r for r in rows if r['inner_validation_load_hp'] == load]
            by_load.append(dict(family=family,inner_validation_load_hp=load,cells=len(selected),
                mean_macro_f1=mean(selected,'macro_f1'),mean_normal_recall=mean(selected,'normal_recall'),
                cells_with_zero_normal_recall=sum(r['normal_recall']==0 for r in selected)))
        for outer in range(4):
            for config in range(8):
                selected=[r for r in rows if r['outer_load_hp']==outer and r['config_index']==config]
                expected=9 if family == 'RF' else 3
                io.require(len(selected) == expected,'Configuración interna incompleta')
                per_load={str(load):mean([r for r in selected if r['inner_validation_load_hp']==load],'macro_f1')
                    for load in range(4) if load != outer}
                by_config.append(dict(family=family,outer_load_hp=outer,config_index=config,cells=expected,
                    params=selected[0]['params'] if family=='SVM' else {k:v for k,v in selected[0]['params'].items() if k!='random_state'},
                    mean_macro_f1=mean(selected,'macro_f1'),mean_macro_f1_by_internal_load=per_load,
                    worst_load_mean_macro_f1=min(per_load.values()),selected=False))
    rf_seed=[dict(seed=seed,cells=96,mean_macro_f1=mean([r for r in matrix if r['family']=='RF' and r['seed']==seed],'macro_f1'))
        for seed in [11,22,33]]
    worst={family:sorted([r for r in matrix if r['family']==family],key=lambda r:r['macro_f1'])[:3] for family in ('RF','SVM')}
    report=dict(status='CLASSICAL_INNER_MATRIX_COMPLETE_AUDITED_READY_FOR_SELECTION_PREPARATION',
        date_local='2026-10-27',timezone='America/Lima',campaign=CAMPAIGN.relative_to(ROOT).as_posix(),
        completed_fits=384,RF_completed=288,SVM_completed=96,remaining_fits=0,failed_fits=0,automatic_retries=0,
        campaign_execution_verification_seconds=consumed,campaign_execution_verification_minutes=consumed/60,
        budget_seconds_remaining=3600-consumed,budget_seconds_limit=3600,
        family_descriptive=family_summary,validation_load_descriptive=by_load,RF_seed_descriptive=rf_seed,
        worst_cells_by_family=worst,distinct_training_scopes=192,
        identical_model_hashes_for_swapped_roles_confirmed=True,
        model_identity_is_not_independent_physical_evidence=True,
        authenticated_joblib_models_reloaded=384,stored_predictions_replayed=384,
        SVM_train_only_scalers_reconstructed=96,RF_without_scaler=True,SVC_native_predict_rule_preserved=True,
        metadata_plan_snapshot_verified_before_and_after=True,
        new_RF_SVM_training_fits_by_auditor=0,fit_methods_blocked_during_audit=True,
        model_prediction_replays_are_saved_internal_evaluations_not_new_test_data=True,
        TensorFlow_imported='tensorflow' in sys.modules,hardware_connected=False,
        selection_executed=False,outer_evaluation_executed=False,manuscript_modified=False,
        source_campaign_manifest_sha256=io.sha(original_manifest),preparation_manifest_sha256=io.sha(classical.MANIFEST),
        next_stage='Prepare and audit selectors using288DNN+96detector+384classical cells and validated total costs; lock all four scenarios before outer refits/evaluation.',
        qualifications=['Global grid means are descriptive, not performance of selected final procedures.',
            'Each outer scenario must select only with its internal data; outer predictions are not released.',
            'SVM uses one deterministic fit per configuration/fold; it has no three-seed replication.',
            'Training and validation cells are dependent; no IID CI or claim of independent motors is made.',
            'Results, grids and engineering margins remain unchanged after observing the metrics.'],
        audit_seconds=time.perf_counter()-started)
    OUT.mkdir(parents=True,exist_ok=False)
    io.write(OUT/'CIERRE_CLASICOS_INTERNOS.json',report)
    io.write(OUT/'MATRIZ_CLASICOS_384.json',matrix)
    io.write(OUT/'RESUMEN_CONFIGURACIONES_POR_ESCENARIO.json',by_config)
    table='\n'.join(f"| {r['family']} | {r['cells']} | {100*r['mean_macro_f1']:.2f} | {100*r['median_macro_f1']:.2f} | {100*r['minimum_macro_f1']:.2f} |" for r in family_summary)
    loads='\n'.join(f"| {load} | {100*next(r['mean_macro_f1'] for r in by_load if r['family']=='RF' and r['inner_validation_load_hp']==load):.2f} | {100*next(r['mean_macro_f1'] for r in by_load if r['family']=='SVM' and r['inner_validation_load_hp']==load):.2f} |" for load in range(4))
    text=f'''# Búsqueda interna clásica completa y auditada

Estado: `{report['status']}`. Se cerraron384/384 ajustes,288RF y96SVM,
sin fallos o reintentos. Tiempo contabilizado: {consumed:.3f}s
({consumed/60:.2f}min), dentro de3600s. Incluye ejecución y comprobación por
ajuste; no es todo el tiempo calendario. El piloto temporal no se repitió.

La auditoría autenticó y cargó los384modelos propios, reprodujo predicciones,
diagnósticos y métricas, contrastó roles/IDs y reconstruyó los96scalers SVM.
Los métodos fit de RF/SVC se bloquearon durante la revisión. RF conserva datos
sin escalar y SVC su votación nativa. No hubo nuevos entrenamientos, acceso a
datos exteriores por escenario, TensorFlow, hardware o cambios al manuscrito.

## Descriptivos de todas las configuraciones, sin selección

| Familia | Celdas | Macro-F1 medio (%) | Mediana (%) | Peor celda (%) |
|---|---:|---:|---:|---:|
{table}

Estos promedios no son el rendimiento de modelos finales seleccionados.

| Carga de validación interna (HP) | Macro-F1 RF medio (%) | Macro-F1 SVM medio (%) |
|---|---:|---:|
{loads}

La caída en0HP reaparece en ambas familias. En SVM, las24/24celdas de
validación0HP tienen recall N cero; en RF ocurre en18/72. No se atribuye aquí
una causa ni se cambia el grid para mejorar estas cifras. Las tres semillas
RF y los peores casos se conservan en el JSON; no se elige la semilla favorable.

Hay144hashes de modelos RF y48SVM: corresponden a192ámbitos de entrenamiento
compartidos entre roles outer/inner intercambiados. Se ejecutaron los384ajustes
por rol previstos; los modelos de cada par son iguales, sus validaciones y
firmas de ejecución son distintas. No son384adquisiciones independientes.

La DNN interna288/288, detector96/96 y comparadores384/384 quedan completos.
Siguiente etapa: preparar y auditar la selección con las reglas ya fijadas,
costos totales validados y guardias relativas del detector. Congelar decisiones
de los cuatro escenarios antes de reajustar y liberar resultados exteriores.
La baja especificidad absoluta del detector se conserva como limitación.
No se seleccionaron aquí configuraciones o máscaras ni se usaron predicciones
exteriores para modificar reglas.
'''
    (OUT/'RESULTADOS_Y_SIGUIENTE_ETAPA.md').write_text(text,encoding='utf-8')
    files=[p for p in OUT.iterdir() if p.is_file()]
    files += [Path(__file__).resolve(),classical.PLAN,classical.MANIFEST,classical.MANIFEST.with_suffix('.sha256')]
    files += [p for p in CAMPAIGN.rglob('*') if p.is_file()]
    manifest=OUT/'MANIFEST_CIERRE_CLASICOS_INTERNOS_SHA256.json'
    io.write_manifest(manifest,extra=files)
    print(json.dumps(dict(status=report['status'],completed_fits=384,RF=288,SVM=96,
        consumed_minutes=consumed/60,manifest_files=len(io.read(manifest)),manifest_sha256=io.sha(manifest),
        TensorFlow_imported='tensorflow' in sys.modules,output=str(OUT)),ensure_ascii=True))


if __name__ == '__main__':
    main()
