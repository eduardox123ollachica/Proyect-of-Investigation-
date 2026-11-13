"""Descriptivos de la matriz ya auditada; no selecciona ni cambia modelos."""
from __future__ import annotations

import json
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'codigo/cwru/ampliacion_v2/busqueda_interna'))
import ejecutar_busqueda_interna as e

CLOSURE=ROOT/'articulo/ampliacion_cwru/metodologia/cierre_matriz_dnn_interna_v2_2026-10-26'
OUT=ROOT/'articulo/ampliacion_cwru/metodologia/analisis_matriz_interna_v2_2026-10-26'


def mean(rows,key):
    return statistics.mean(r[key] for r in rows)


def main():
    e.require(not OUT.exists(),'No sobrescribir un análisis anterior')
    manifest=CLOSURE/'MANIFEST_CIERRE_MATRIZ_SHA256.json'
    e.verify_manifest(manifest)
    rows=e.read(CLOSURE/'MATRIZ_INTERNA_288.json')
    e.require(len(rows) == 288,'Matriz no completa')
    per_load=[dict(inner_validation_load_hp=load,cells=72,
        mean_macro_f1_FP32=mean([r for r in rows if r['inner_validation_load_hp'] == load],'macro_f1_FP32'),
        mean_macro_f1_INT8=mean([r for r in rows if r['inner_validation_load_hp'] == load],'macro_f1_INT8'))
        for load in range(4)]
    for load in range(4):
        e.require(sum(r['inner_validation_load_hp'] == load for r in rows) == 72,'Carga incompleta')
    deltas=[dict(job_id=r['job_id'],candidate=r['candidate'],outer_load_hp=r['outer_load_hp'],
        inner_validation_load_hp=r['inner_validation_load_hp'],seed=r['seed'],
        macro_f1_FP32=r['macro_f1_FP32'],macro_f1_INT8=r['macro_f1_INT8'],
        delta_INT8_minus_FP32_pp=100*(r['macro_f1_INT8']-r['macro_f1_FP32']),
        class_changes=r['class_changes_FP32_INT8'],validation_clipped_feature_values=r['validation_clipped_feature_values'])
        for r in rows]
    jobs=e.read(e.PLAN_FILE)['jobs']
    e.require([r['job_id'] for r in rows] == [j['job_id'] for j in jobs], 'Orden de matriz distinto del plan')
    feature_values=sum(r['validation_windows']*len(j['feature_indices']) for r,j in zip(rows,jobs))
    clipped=sum(r['validation_clipped_feature_values'] for r in rows)
    report=dict(status='DESCRIPTIVE_INTERNAL_VARIABILITY_RECORDED_NO_SELECTION',
        date_local='2026-10-26',timezone='America/Lima',source_closure_manifest_sha256=e.sha(manifest),
        cells=288,mean_macro_f1_FP32=mean(rows,'macro_f1_FP32'),mean_macro_f1_INT8=mean(rows,'macro_f1_INT8'),
        median_cell_macro_f1_INT8=statistics.median(r['macro_f1_INT8'] for r in rows),
        minimum_cell_macro_f1_INT8=min(r['macro_f1_INT8'] for r in rows),
        per_inner_validation_load=per_load,
        per_outer_scenario_internal_mean=[dict(outer_load_hp=outer,cells=72,
            mean_macro_f1_INT8=mean([r for r in rows if r['outer_load_hp'] == outer],'macro_f1_INT8'))
            for outer in range(4)],
        per_seed_internal_mean=[dict(seed=seed,cells=96,
            mean_macro_f1_INT8=mean([r for r in rows if r['seed'] == seed],'macro_f1_INT8'))
            for seed in [11,22,33]],
        quantization=dict(cells_with_class_changes=sum(r['class_changes_FP32_INT8']>0 for r in rows),
            class_changes_summed_repeated_passes=sum(r['class_changes_FP32_INT8'] for r in rows),
            mean_delta_INT8_minus_FP32_pp=statistics.mean(d['delta_INT8_minus_FP32_pp'] for d in deltas),
            cells_F1_improved=sum(d['delta_INT8_minus_FP32_pp']>0 for d in deltas),
            cells_F1_decreased=sum(d['delta_INT8_minus_FP32_pp']<0 for d in deltas),
            cells_F1_unchanged=sum(d['delta_INT8_minus_FP32_pp']==0 for d in deltas),
            largest_losses=sorted(deltas,key=lambda d:d['delta_INT8_minus_FP32_pp'])[:5],
            largest_gains=sorted(deltas,key=lambda d:d['delta_INT8_minus_FP32_pp'],reverse=True)[:5]),
        clipping=dict(training_feature_values=sum(r['training_clipped_feature_values'] for r in rows),
            validation_feature_values=clipped,validation_feature_evaluations=feature_values,
            percentage_over_repeated_feature_evaluations=100*clipped/feature_values,
            cells_with_validation_clipping=sum(r['validation_clipped_feature_values']>0 for r in rows)),
        cause_identified=False,new_training_fits=0,new_TFLite_invocations=0,
        TensorFlow_imported='tensorflow' in sys.modules,outer_evaluation_executed=False,selection_executed=False,
        qualifications=['Descriptive means over related candidate/fold/seed cells, not final system accuracy.',
            'Internal validation load differs from held-out outer load; outer predictions are not released.',
            'Clipping counts feature values in repeated evaluations, not unique windows.',
            'Poor load0 performance also exists in FP32; quantization alone does not explain it.',
            'These results do not authorize recalibrating with validation, choosing a fortunate seed, or changing frozen margins.'])
    OUT.mkdir(parents=True,exist_ok=False)
    e.write(OUT/'VARIABILIDAD_INTERNA.json',report)
    table='\n'.join(f"| {r['inner_validation_load_hp']} | {100*r['mean_macro_f1_FP32']:.2f} | {100*r['mean_macro_f1_INT8']:.2f} |" for r in per_load)
    worst=report['quantization']['largest_losses'][0]
    text=f'''# Variabilidad observada en la matriz interna

La matriz está completa y técnicamente auditada. Su rendimiento predictivo
varía con el rol, la máscara y la semilla. PASS confirma integridad de la
ejecución; no establece alto rendimiento. La media global descriptiva INT8
de las288 celdas es{report['mean_macro_f1_INT8']*100:.2f}% macro-F1, mediana
{report['median_cell_macro_f1_INT8']*100:.2f}% y mínimo
{report['minimum_cell_macro_f1_INT8']*100:.2f}%. No es la métrica final del sistema.

| Carga de validación interna (HP) | Macro-F1 FP32 medio (%) | Macro-F1 INT8 medio (%) |
|---|---:|---:|
{table}

Cada fila promedia72 celdas relacionadas. La caída al validar con0HP aparece
también en FP32; estos descriptivos no identifican su causa. Debe conservarse
al aplicar las guardias y reportar la evaluación posterior, sin ocultar celdas.

En{report['quantization']['cells_with_class_changes']}/288 celdas cambia alguna clase FP32–INT8. La media del cambio de macro-F1
es{report['quantization']['mean_delta_INT8_minus_FP32_pp']:.4f} puntos porcentuales,
pero hay extremos importantes. `{worst['job_id']}` pasa de
{worst['macro_f1_FP32']*100:.2f}% a{worst['macro_f1_INT8']*100:.2f}%
({worst['delta_INT8_minus_FP32_pp']:.2f} puntos). Se preservan todas las
predicciones y los cinco extremos de mejora/pérdida en el JSON. Esto tampoco
es una discrepancia PC–STM32: aquí se comparan modelos FP32 e INT8 en PC.

Entrenamiento tiene0 valores recortados; validación tiene{clipped} valores
de características recortados en{report['clipping']['cells_with_validation_clipping']} celdas, sobre{feature_values} evaluaciones
de características repetidas. No son ventanas únicas o independientes. No
se recalibra con validación ni se selecciona una semilla favorable. La métrica
de selección sigue siendo INT8 conforme al protocolo.

El siguiente paso es completar el detector para aplicar sus guardias junto
con estas métricas y los costos medidos. Las búsquedas clásicas y decisiones
de selección preceden a los reajustes y la evaluación exterior. Ningún dato
de esta revisión cambia el protocolo ni el manuscrito.
'''
    (OUT/'HALLAZGOS_Y_LIMITES.md').write_text(text,encoding='utf-8')
    e.write_manifest(OUT/'MANIFEST_ANALISIS_SHA256.json',extra=[Path(__file__).resolve(),
        OUT/'VARIABILIDAD_INTERNA.json',OUT/'HALLAZGOS_Y_LIMITES.md',
        CLOSURE/'MATRIZ_INTERNA_288.json',CLOSURE/'CIERRE_MATRIZ_INTERNA.json',
        manifest,manifest.with_suffix('.sha256')])
    print(json.dumps(dict(status=report['status'],mean_macro_f1_INT8=report['mean_macro_f1_INT8'],
        per_inner_validation_load=per_load,clipping=report['clipping'],output=str(OUT)),ensure_ascii=True))


if __name__ == '__main__':
    main()
