"""Selección interna determinista y congelación, sin ajustes o inferencias.

Consume matrices ya auditadas y costos medidos. Conserva el fallback F14,
incluidos sus incumplimientos, y bloquea ejecución de los reajustes siguientes.
"""
from __future__ import annotations
import datetime
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[4]
SOURCE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'codigo/cwru/ampliacion_v2/busqueda_interna'))
sys.path.insert(0,str(ROOT/'codigo/cwru/ampliacion_v2/protocolo'))
import ejecutar_busqueda_interna as io
from reglas_v2 import select_performance,select_cost,final_epoch_count,component_roles
from seleccion_clasicos import select_classical

METH=ROOT/'articulo/ampliacion_cwru/metodologia'
OUT=METH/'seleccion_congelada_v2_2026-10-27'
CHECKS=METH/'verificacion_seleccion_v2_2026-10-27'
DNN=METH/'cierre_matriz_dnn_interna_v2_2026-10-26'
DET=METH/'cierre_detector_interno_v2_2026-10-26'
CLASSICAL=METH/'cierre_clasicos_internos_v2_2026-10-27'
COST=ROOT/'despliegue_stm32/costo_total_single_rounding_v2_180mhz_2026-10-26'
DET_MATRIX=ROOT/'resultados/cwru/ampliacion_v2/detector_interno_v2_2026-10-26/20261027T031125_681822Z/MATRIZ_DETECTOR_96.json'


def sources():
    dependencies=['protocolo/reglas_v2.py','piloto_clasicos/contrato_piloto.py',
        'busqueda_interna/ejecutar_busqueda_interna.py','representantes/ejecutar_representantes.py']
    files=[*SOURCE.glob('*.py'),*[ROOT/'codigo/cwru/ampliacion_v2'/p for p in dependencies]]
    return {p.relative_to(ROOT).as_posix():io.sha(p) for p in sorted(files)}


def authenticate_input(manifest,paths):
    """Autenticar sello y entradas consumidas, sin releer modelos/RAW ajenos."""
    manifest=Path(manifest)
    digest,name=manifest.with_suffix('.sha256').read_text(encoding='ascii').split()
    io.require(name == manifest.name and digest == io.sha(manifest),'Sello de entrada modificado')
    entries=io.read(manifest)
    verified={manifest.relative_to(ROOT).as_posix():digest,
        manifest.with_suffix('.sha256').relative_to(ROOT).as_posix():io.sha(manifest.with_suffix('.sha256'))}
    for path in paths:
        path=Path(path).resolve()
        io.require(path.is_relative_to(ROOT.resolve()),'Entrada fuera del proyecto')
        relative=path.relative_to(ROOT).as_posix()
        io.require(relative in entries and io.sha(path) == entries[relative],'Entrada modificada o fuera de su cierre: '+relative)
        verified[relative]=entries[relative]
    return verified


def load_inputs():
    specifications=[
        (DNN/'MANIFEST_CIERRE_MATRIZ_SHA256.json',[DNN/'MATRIZ_INTERNA_288.json',DNN/'CIERRE_MATRIZ_INTERNA.json']),
        (DET/'MANIFEST_CIERRE_DETECTOR_SHA256.json',[DET_MATRIX,DET/'CIERRE_DETECTOR_INTERNO.json']),
        (CLASSICAL/'MANIFEST_CIERRE_CLASICOS_INTERNOS_SHA256.json',[
            CLASSICAL/'MATRIZ_CLASICOS_384.json',CLASSICAL/'CIERRE_CLASICOS_INTERNOS.json']),
        (COST/'MANIFEST_COSTOS_TOTALES_SR_VALIDADOS_SHA256.json',[
            COST/'COSTOS_TOTALES_SR_VALIDADOS.json',COST/'CIERRE_COSTOS_TOTALES_SR_VALIDADOS.json']),
        (io.PROTOCOL/'MANIFEST_PROTOCOLO_SHA256.json',[
            io.PROTOCOL/'protocolo_reglas.json',ROOT/'codigo/cwru/ampliacion_v2/protocolo/reglas_v2.py']),
        (METH/'clasicos_internos_v2_2026-10-27/MANIFEST_PREPARACION_CLASICOS_SHA256.json',[
            ROOT/'codigo/cwru/ampliacion_v2/piloto_clasicos/contrato_piloto.py',
            ROOT/'codigo/cwru/ampliacion_v2/busqueda_interna/ejecutar_busqueda_interna.py',
            ROOT/'codigo/cwru/ampliacion_v2/representantes/ejecutar_representantes.py'])]
    provenance={}
    for manifest,paths in specifications:provenance.update(authenticate_input(manifest,paths))
    io.require(io.read(DNN/'CIERRE_MATRIZ_INTERNA.json')['cells'] == 288
        and io.read(DET/'CIERRE_DETECTOR_INTERNO.json')['logical_rows'] == 96
        and io.read(CLASSICAL/'CIERRE_CLASICOS_INTERNOS.json')['completed_fits'] == 384,
        'Las matrices previas no están completas')
    return dict(protocol=io.read(io.PROTOCOL/'protocolo_reglas.json'),
        dnn_rows=io.read(DNN/'MATRIZ_INTERNA_288.json'),detector_rows=io.read(DET_MATRIX),
        classical_rows=io.read(CLASSICAL/'MATRIZ_CLASICOS_384.json'),
        cost_table=io.read(COST/'COSTOS_TOTALES_SR_VALIDADOS.json'),provenance=provenance)


def validate_grids(inputs):
    protocol=inputs['protocol'];catalog=protocol['candidate_order'];seeds=protocol['dnn']['inner_seeds']
    io.require(catalog == ['F14','T6','S8','NF12','NM10','NE12','DE7','FE7'] and seeds == [11,22,33],
        'Catálogo o semillas cambiados')
    expected_dnn={(o,i,m,s) for o in range(4) for i in range(4) if i!=o for m in catalog for s in seeds}
    expected_det={(o,i,m) for o in range(4) for i in range(4) if i!=o for m in catalog}
    dnn=inputs['dnn_rows'];detector=inputs['detector_rows']
    io.require(all(type(r[k]) is int for r in dnn
        for k in ('outer_load_hp','inner_validation_load_hp','seed')), 'Roles/semillas DNN deben ser enteros')
    io.require(all(type(r[k]) is int for r in detector
        for k in ('outer_load_hp','inner_validation_load')), 'Roles detector deben ser enteros')
    keys=[(r['outer_load_hp'],r['inner_validation_load_hp'],r['candidate'],r['seed']) for r in dnn]
    io.require(len(keys)==len(set(keys))==288 and set(keys)==expected_dnn,'Matriz DNN duplicada, ajena o incompleta')
    for row in dnn:
        score=row['macro_f1_INT8']
        io.require(isinstance(score,(int,float)) and not isinstance(score,bool) and math.isfinite(score)
            and 0 <= score <= 1 and type(row['best_epoch']) is int and 1 <= row['best_epoch'] <= 50,
            'Métrica/checkpoint DNN inválido')
    detkeys=[(r['outer_load_hp'],r['inner_validation_load'],r['candidate']) for r in detector]
    io.require(len(detkeys)==len(set(detkeys))==96 and set(detkeys)==expected_det
        and all(r['status']=='PASS' for r in detector),'Matriz detector incompleta/ajena/no aprobada')
    classical=inputs['classical_rows']
    io.require(all(type(r[k]) is int for r in classical
        for k in ('outer_load_hp','inner_validation_load_hp','config_index')), 'Roles/configuración clásicos deben ser enteros')
    expected_classical={(o,i,f,c,s) for o in range(4) for i in range(4) if i!=o
        for f in ('RF','SVM') for c in range(8) for s in (seeds if f=='RF' else [None])}
    ckeys=[(r['outer_load_hp'],r['inner_validation_load_hp'],r['family'],r['config_index'],r['seed']) for r in classical]
    io.require(len(ckeys)==len(set(ckeys))==384 and set(ckeys)==expected_classical,'Grid clásico incompleto, duplicado o ajeno')
    table=inputs['cost_table']
    io.require(table['status']=='VALIDATED_TOTAL_COST' and len(table['rows'])==32,'Costos no validados o incompletos')
    io.require(all(type(r['outer_load']) is int for r in table['rows']),'Roles de costos deben ser enteros')
    costkeys=[(r['outer_load'],r['candidate']) for r in table['rows']]
    io.require(len(set(costkeys))==32 and set(costkeys)=={(o,m) for o in range(4) for m in catalog},
        'Costos duplicados o de otra cuadrícula')
    for row in table['rows']:
        outer=row['outer_load'];mask=row['candidate'];inner=min(set(range(4))-{outer})
        representative=next(r for r in dnn if r['outer_load_hp']==outer
            and r['inner_validation_load_hp']==inner and r['candidate']==mask and r['seed']==11)
        io.require(row == table['costs_by_outer'][str(outer)][mask]
            and row['representative_id']==representative['job_id']
            and row['source_fit_signature']==representative['source_fit_signature']
            and row['model_sha256']==representative['model_sha256']['INT8'],
            'Costo y modelo representante no corresponden a la matriz')


def build_decisions(inputs):
    validate_grids(inputs)
    protocol=inputs['protocol'];catalog=protocol['candidate_order'];seeds=protocol['dnn']['inner_seeds']
    decisions=[]
    for outer in range(4):
        inner_loads=[i for i in range(4) if i!=outer]
        raw_dnn=sorted([r for r in inputs['dnn_rows'] if r['outer_load_hp']==outer],
            key=lambda r:(catalog.index(r['candidate']),r['inner_validation_load_hp'],r['seed']))
        dnn=[dict(candidate=r['candidate'],inner_validation_load=r['inner_validation_load_hp'],seed=r['seed'],
            status='PASS',macro_f1_int8=r['macro_f1_INT8']) for r in raw_dnn]
        det=[r for r in inputs['detector_rows'] if r['outer_load_hp']==outer]
        performance=select_performance(dnn,det,catalog,inner_loads,seeds)
        costs=inputs['cost_table']['costs_by_outer'][str(outer)]
        cost=select_cost(dnn,det,catalog,inner_loads,seeds,costs,outer_load=outer)
        io.require(performance['selected_candidate']==cost['performance_selected_candidate'], 'Selectores no comparten referencia')
        def choice(mask):
            rows=sorted([r for r in raw_dnn if r['candidate']==mask],
                key=lambda r:(r['inner_validation_load_hp'],r['seed']))
            best=[r['best_epoch'] for r in rows]
            return dict(candidate=mask,final_epochs=final_epoch_count(best),best_inner_epochs=best,
                source_inner_job_ids=[r['job_id'] for r in rows],
                mean_macro_f1_int8=performance['candidate_summaries'][mask]['mean_macro_f1_int8'],
                worst_load_mean_macro_f1_int8=performance['candidate_summaries'][mask]['worst_load_mean_macro_f1_int8'])
        baseline=choice('F14')
        performance_choice=choice(performance['selected_candidate'])
        cost_choice=choice(cost['selected_candidate'])
        perf_cost=costs[performance_choice['candidate']];selected_cost=costs[cost_choice['candidate']]
        cost_comparison=dict(reference_F14_mean_ms=costs['F14']['mean_ms'],
            performance_representative_mean_ms=perf_cost['mean_ms'],selected_representative_mean_ms=selected_cost['mean_ms'],
            observed_envelope_reduction_vs_F14=1-selected_cost['upper_ms']/costs['F14']['lower_ms'],
            selected_F1_minus_performance_pp=100*(cost_choice['mean_macro_f1_int8']-performance_choice['mean_macro_f1_int8']),
            same_candidate_as_performance=performance_choice['candidate']==cost_choice['candidate'],
            measured_costs_are_inner_representatives_not_final_outer_firmware=True)
        decisions.append(dict(outer_load_hp=outer,inner_validation_loads=inner_loads,
            baseline=baseline,performance=dict(**performance,choice=performance_choice),
            cost=dict(**cost,choice=cost_choice,comparison=cost_comparison),
            classical={family:select_classical(inputs['classical_rows'],family,outer,protocol) for family in ('RF','SVM')},
            selected_once_for_all_outer_seeds=True,no_outer_predictions_used=True))
    return decisions


def build_refit_plan(decisions,protocol):
    io.require(len(decisions)==4 and {d['outer_load_hp'] for d in decisions}==set(range(4)), 'Se requieren cuatro decisiones')
    seeds=protocol['dnn']['outer_seeds']
    io.require(seeds == [11,22,33,42,44,55,66,77,88,99]
        and protocol['classical_comparators']['RF']['outer_seeds']==seeds, 'Semillas de reajuste distintas')
    dnn={};detectors={};rf=[];svm=[]
    for decision in sorted(decisions,key=lambda d:d['outer_load_hp']):
        outer=decision['outer_load_hp'];roles=component_roles(outer)
        choices={'baseline_F14':decision['baseline'],'performance_only':decision['performance']['choice'],
            'validated_total_cost':decision['cost']['choice']}
        for procedure,choice in choices.items():
            mask=choice['candidate'];epochs=choice['final_epochs']
            for seed in seeds:
                key=(outer,mask,epochs,seed)
                if key not in dnn:dnn[key]=dict(job_id=f'outer{outer}_{mask}_e{epochs}_s{seed}',
                    outer_load_hp=outer,candidate=mask,epochs=epochs,seed=seed,roles=roles,procedures=[])
                dnn[key]['procedures'].append(procedure)
            indices=[i for i in protocol['candidates'][mask] if i not in (4,11)]
            key=(outer,tuple(indices))
            if key not in detectors:detectors[key]=dict(outer_load_hp=outer,indices=indices,roles=roles,
                masks=[],procedures=[],normal_only_fit=True,normal_only_calibration=True,refit_after_calibration=False)
            if mask not in detectors[key]['masks']:detectors[key]['masks'].append(mask)
            detectors[key]['procedures'].append(procedure)
        for seed in seeds:
            config=decision['classical']['RF']
            rf.append(dict(outer_load_hp=outer,config_index=config['selected_config_index'],seed=seed,
                params=dict(config['selected_params'],random_state=seed),roles=roles))
        config=decision['classical']['SVM']
        svm.append(dict(outer_load_hp=outer,config_index=config['selected_config_index'],seed=None,
            params=config['selected_params'],roles=roles))
    return dict(status='DERIVED_REFIT_SCOPES_ONLY_PENDING_EXECUTOR_AND_BUDGET',
        DNN_unique_jobs=list(dnn.values()),RF_jobs=rf,SVM_jobs=svm,detector_unique_scopes=list(detectors.values()),
        counts=dict(DNN_before_dedup=120,DNN_unique=len(dnn),RF=len(rf),SVM=len(svm),detector_unique=len(detectors)),
        canonical_hardware_seed=42,outer_prediction_release_enabled=False,execution_enabled=False,
        outer_early_stopping_allowed=False,seed_ensembles=False)


def freeze():
    io.require(not OUT.exists(),'Decisiones ya congeladas; no sobrescribir o reoptimizar')
    inputs=load_inputs()
    tests=io.read(CHECKS/'PRUEBAS_SELECCION.json')
    io.require(tests['status']=='PASS' and tests['errors']==tests['failures']==0
        and tests['source_sha256']==sources() and tests['new_training_fits']==tests['new_model_predictions']==0
        and tests['TensorFlow_imported'] is False,'Faltan pruebas de las fuentes finales')
    decisions=build_decisions(inputs);refits=build_refit_plan(decisions,inputs['protocol'])
    signed_body=dict(decisions=decisions,input_sha256=inputs['provenance'],selector_source_sha256=sources(),
        protocol_selection=inputs['protocol']['selection'],protocol_classical=inputs['protocol']['classical_comparators'],
        protocol_DNN=inputs['protocol']['dnn'],refit_scopes=refits)
    signature=io.canonical(signed_body)
    lock=dict(status='FOUR_OUTER_SELECTION_DECISIONS_FROZEN_BEFORE_REFITS_AND_OUTER_EVALUATION',
        date_local='2026-10-27',timezone='America/Lima',frozen_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        selection_signature=signature,outer_scenarios=4,decisions=decisions,
        inputs=dict(DNN_cells=288,detector_logical_rows=96,classical_cells=384,validated_cost_rows=32),
        input_sha256=inputs['provenance'],source_sha256=sources(),tests_run=tests['tests_run'],
        cost_goals_met=sum(d['cost']['goal_met'] for d in decisions),
        fallback_scenarios=[d['outer_load_hp'] for d in decisions if not d['cost']['goal_met']],
        fallback_scenarios_outside_predictive_margin=[d['outer_load_hp'] for d in decisions
            if not d['cost']['selected_satisfies_predictive_margin']],
        absolute_detector_quality_guaranteed=False,margins_are_engineering_not_noninferiority_proof=True,
        new_training_fits=0,new_model_predictions=0,TensorFlow_imported='tensorflow' in sys.modules,
        raw_MAT_or_feature_arrays_materialized=False,outer_prediction_files_read=False,
        outer_evaluation_executed=False,refit_execution_enabled=False,manuscript_modified=False,hardware_connected=False,
        historical_operational_flags_preserved=True,
        historical_flags_interpretation='Earlier sealed pending/false flags remain historical; this new artifact records completed internal selection without editing them.')
    OUT.mkdir(parents=True,exist_ok=False)
    io.write(OUT/'DECISIONES_CONGELADAS.json',lock)
    io.write(OUT/'CUERPO_FIRMA_SELECCION.json',signed_body)
    io.write(OUT/'PLAN_REAJUSTES_PENDIENTES.json',refits)
    table='\n'.join(f"| {d['outer_load_hp']} | {d['performance']['choice']['candidate']} | {d['cost']['choice']['candidate']} | {'Sí' if d['cost']['goal_met'] else 'No; respaldo F14'} | {d['classical']['RF']['selected_config_index']} | {d['classical']['SVM']['selected_config_index']} |" for d in decisions)
    epochs='\n'.join(f"| {d['outer_load_hp']} | {d['baseline']['final_epochs']} | {d['performance']['choice']['final_epochs']} | {d['cost']['choice']['final_epochs']} |" for d in decisions)
    text=f'''# Selección congelada antes de los reajustes y evaluación exterior

Firma: `{signature}`. Se consumieron 288 celdas DNN INT8, 96 filas del detector,
384 celdas clásicas y 32 costos totales validados, autenticados contra sus cierres.
No se entrenó o ejecutó inferencia, materializó RAW/características ni leyó
predicciones exteriores. Las cuatro decisiones se congelan juntas.

| Carga exterior reservada (HP) | Máscara por rendimiento | Máscara por costo | Ahorro objetivo frente a F14 | RF config | SVM config |
|---|---|---|---|---:|---:|
{table}

Se reutilizan literalmente select_performance/select_cost/final_epoch_count
del módulo original. Rendimiento usa la media de 9 valores de macro-F1 INT8 internos,
peor carga y orden del catálogo, entre máscaras admisibles para el detector.
Costo conserva guardias predictivas y requiere U(candidata)<=0.8L(F14).
Los intervalos son envolventes observadas, no IC o WCET. Los costos pertenecen
a representantes internos medidos; los pipelines finales aún deben validarse.
RF/SVM eligen media, peor carga y orden del grid con tolerancia 1e-12 por etapa;
RF promedia 3 semillas por carga y SVM no se triplica. No se elige una semilla.

## Excepciones y alcance que deben comunicarse

En el escenario 3 no hay candidata que cumpla simultáneamente margen F1 y
objetivo temporal. La regla previa devuelve F14; selected_satisfies_predictive_margin
es false. Su F1 interno es 84.4429 % frente a 91.5423 % de NF12: pierde 7.0995 puntos,
por lo que NO se declara preservación de F1 ni objetivo cumplido en ese escenario.
El protocolo no se cambia después de observar ese resultado.

Los objetivos temporales frente a F14 se cumplen en 3/4 escenarios internos;
esto no estima una probabilidad de éxito. Solo el escenario 0 añade ahorro
respecto del selector por rendimiento; en los escenarios 1 y 2 ambos eligen la misma máscara y
en el escenario 3 aparece el respaldo. Las guardias relativas no garantizan especificidad
absoluta adecuada: el problema de aceptación de normales del detector permanece.
Ninguna cifra de aquí es rendimiento de evaluación exterior.

## Épocas y trabajo siguiente

| Escenario | F14 base | Selector rendimiento | Selector costo |
|---|---:|---:|---:|
{epochs}

Son medianas enteras de los 9 mejores checkpoints internos por máscara. Se fijan
antes de entrenar con las 3 cargas de desarrollo; no habrá parada temprana con
la carga exterior. Semillas posteriores 11/22/33/42/44/55/66/77/88/99; 42 canónica
para hardware. El plan derivado tiene {refits['counts']['DNN_unique']} reajustes DNN únicos
(en lugar de 120 solicitudes si se duplican procedimientos), 40 RF, 4 SVM y
{refits['counts']['detector_unique']} ámbitos externos del detector, compartidos
entre semillas cuando corresponde. Este archivo delimita el trabajo pendiente;
el ejecutor y presupuesto se prepararán después. La liberación de predicciones exteriores
continúa bloqueada. No se modificaron manuscrito ni evidencias históricas.
'''
    (OUT/'DECISIONES_Y_ALCANCE.md').write_text(text,encoding='utf-8')
    files=[p for p in OUT.iterdir() if p.is_file()]+[ROOT/p for p in sources()]
    files += [CHECKS/'PRUEBAS_SELECCION.json']+[ROOT/p for p in inputs['provenance']]
    manifest=OUT/'MANIFEST_SELECCION_CONGELADA_SHA256.json'
    io.write_manifest(manifest,extra=files)
    io.require(io.canonical(io.read(OUT/'CUERPO_FIRMA_SELECCION.json'))==signature,'Firma no reproducible')
    print(json.dumps(dict(status=lock['status'],selection_signature=signature,outer_scenarios=4,
        cost_goals_met=lock['cost_goals_met'],refit_counts=refits['counts'],new_training_fits=0,
        new_model_predictions=0,manifest_sha256=io.sha(manifest),output=str(OUT)),ensure_ascii=True))


if __name__=='__main__':
    freeze()
