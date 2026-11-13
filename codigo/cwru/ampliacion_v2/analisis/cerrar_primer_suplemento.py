"""Audita el primer P1 y sus referencias importadas; no entrena ni infiere."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'codigo/cwru/ampliacion_v2/suplemento'))
from ejecutar_suplemento import inspect_run,read,write,sha,verify_manifest,PREP,DIAG

RUN=ROOT/'resultados/cwru/ampliacion_v2/suplemento_preprocesamiento_normalidad_2026-10-25/20261025T215652_507777Z'
OUT=ROOT/'articulo/ampliacion_cwru/metodologia/cierre_primer_suplemento_2026-10-25'


def require(ok,message):
    if not ok:raise ValueError(message)


def main():
    plan,pending,done,seconds=inspect_run(RUN)
    require(done==['supp_o3_i0_F14_P1_s11'],'Este cierre solo corresponde al primer P1')
    require(not (RUN/'ACTIVO.lock').exists(),'Campana activa')
    require(read(RUN/'REFERENCIAS_IMPORTADAS.json')==plan['imported_references'],'Referencias distintas del plan')
    batch=RUN/'lotes/20261025T215652_507777Z'
    summary=read(batch/'RESUMEN_LOTE.json')
    status=read(batch/'ESTADO_supp_o3_i0_F14_P1_s11.json')
    require(summary['status']=='BATCH_COMPLETED' and summary['new_fits_completed']==1
            and summary['total_DNN_cells_available']==3 and status['status']=='PASS'
            and status['return_code']==0 and status['error'] is None,'Lote fallido o incompleto')
    new=read(RUN/done[0]/'resultado.json')
    original=read(ROOT/plan['imported_references']['supp_o3_i0_F14_P0_s11']['source_result'])
    events=[__import__('json').loads(line) for line in (RUN/done[0]/'progreso.jsonl').read_text(encoding='utf-8').splitlines()]
    start=next(e['elapsed_seconds'] for e in events if e['stage']=='start')
    ready=next(e['elapsed_seconds'] for e in events if e['stage']=='data_ready')
    comparison=dict(FP32_macro_f1=dict(P0=original['metrics']['FP32']['macro_f1'],P1=new['metrics']['FP32']['macro_f1']),
        INT8_macro_f1=dict(P0=original['metrics']['INT8']['macro_f1'],P1=new['metrics']['INT8']['macro_f1']),
        validation_clipped_values=dict(P0=original['int8_details']['input_clipped_values'],P1=new['validation_clipped_values']),
        class_changes=dict(P0=original['FP32_INT8_class_changes'],P1=new['class_changes']),
        normal_recall_INT8=dict(P0=original['metrics']['INT8']['per_class'][0]['recall'],P1=new['metrics']['INT8']['per_class'][0]['recall']))
    report=dict(status='TECHNICAL_GATE_PASS_PREDICTIVE_NEGATIVE_FIRST_PAIR',run=RUN.relative_to(ROOT).as_posix(),
        new_fits_complete=1,imported_cells=2,available_cells=3,total_cells=36,new_fits_remaining=33,
        source_job=done[0],roles=new['roles'],training_windows=799,validation_windows=175,
        epochs_executed=new['epochs_executed'],best_epoch=new['best_epoch'],comparison=comparison,
        clipping_reduction_fraction=1-new['validation_clipped_values']/original['int8_details']['input_clipped_values'],
        timing=dict(process_and_audit_seconds=seconds,worker_to_result_seconds=new['elapsed_worker_seconds'],
            before_worker_directory_event_seconds=start,imports_and_data_interval_seconds=ready-start,
            training_fit_seconds=new['training_fit_seconds'],
            remaining33_linear_at_first_process_seconds=33*seconds,
            remaining33_linear_at_first_process_minutes=33*seconds/60,
            remaining_campaign_limit_seconds=plan['limits']['campaign_process_seconds']-seconds),
        duration_note='Single observed first process; linear projection is not an ETA or confidence bound. Import/data interval is not isolated TensorFlow import.',
        decision='Continue fixed paired design without changing formula, epochs or parameters based on this one pair.',
        detector_block_run=False,outer_evaluation_executed=False,selection_executed=False,STM32_cost_measured=False,
        new_training_fits_during_audit=0,new_model_inferences_during_audit=0,
        next_command=['python3.12.exe','-B',str(ROOT/'codigo/cwru/ampliacion_v2/suplemento/ejecutar_suplemento.py'),
                      '--execute','--resume',str(RUN),'--limit','33'])
    OUT.mkdir(exist_ok=False);write(OUT/'AUDITORIA_PRIMER_P1.json',report)
    rows=['# Auditoría del primer P1','',
        'PASS técnico: firma, roles, medianas train-only, scaler, checkpoint restaurado,',
        'cuantización, métricas, modelos y sellos verificados. Dos referencias importadas',
        'autenticadas con sus firmas originales. Sin entrenar o inferir durante auditoría.','',
        'F14, semilla11, ajuste1/2HP (799ventanas), validación0HP (175).',
        f"Ejecutó {new['epochs_executed']}épocas y restauró la mejor ({new['best_epoch']}).",'',
        '| Métrica | P0 original | P1 compresión |','|---|---:|---:|']
    for label,key in [('F1 macro FP32','FP32_macro_f1'),('F1 macro INT8','INT8_macro_f1'),
                      ('Valores de entrada recortados','validation_clipped_values'),('Cambios FP32–INT8','class_changes')]:
        rows.append(f"| {label} | {comparison[key]['P0']:.6f} | {comparison[key]['P1']:.6f} |")
    rows += ['',
        'La compresión reduce recortes pero empeora el F1 en este primer par. Las28normales',
        'siguen clasificadas OR. Menos recorte no demuestra mejor clasificación. Este dato',
        'no justifica cambiar fórmula o detener selectivamente el diseño fijado por desempeño.',
        'Faltan las demás semillas, máscara y cargas antes de concluir sobre P1.','',
        f"Proceso y auditoría del padre: {seconds:.6f}s; entrenamiento: {new['training_fit_seconds']:.6f}s.",
        f"El intervalo importaciones/datos ocupó {ready-start:.6f}s; no es tiempo aislado de importación TensorFlow.",
        f"Si los33siguientes repitieran este ritmo, sumarían {33*seconds/60:.2f}min más el bloque detector.",
        'Es una proyección lineal de un caso, no duración garantizada. La orientación inicial',
        '15–30min puede quedarse corta si se repite este arranque. Se conserva el límite',
        'global7200s y300s por worker; las pausas humanas no se cuentan.','',
        'Continuar la misma campaña con --resume y --limit33. No repetir el primer P1.',
        'Al completar las34DNNnuevas se ejecuta el bloque detector (1importado+5fits).',
        'Éxito técnico y cumplimiento predictivo se reportarán por separado.',
        'No se cambian protocolos principales, datos, firmware o manuscrito.','']
    (OUT/'AUDITORIA_PRIMER_P1.md').write_text('\n'.join(rows),encoding='utf-8',newline='\n')
    entries=verify_manifest(DIAG/'MANIFEST_CIERRE_SHA256.json')
    entries.update(verify_manifest(PREP/'MANIFEST_PREPARACION_SHA256.json'))
    entries.update(verify_manifest(batch/'MANIFEST_LOTE_SHA256.json'))
    paths=[Path(__file__),OUT/'AUDITORIA_PRIMER_P1.json',OUT/'AUDITORIA_PRIMER_P1.md',
           RUN.parent/'EJECUCION_UNICA.json',PREP/'MANIFEST_PREPARACION_SHA256.json',PREP/'MANIFEST_PREPARACION_SHA256.sha256',
           batch/'MANIFEST_LOTE_SHA256.json',batch/'MANIFEST_LOTE_SHA256.sha256',
           DIAG/'MANIFEST_CIERRE_SHA256.json',DIAG/'MANIFEST_CIERRE_SHA256.sha256']
    entries.update({p.relative_to(ROOT).as_posix():sha(p) for p in paths})
    manifest=OUT/'MANIFEST_CIERRE_SHA256.json';write(manifest,entries)
    manifest.with_suffix('.sha256').write_text(sha(manifest)+'  '+manifest.name+'\n',encoding='ascii')
    verify_manifest(manifest)
    print(f'PASS:3/36celdas,33nuevos pendientes; cierre{len(entries)}recursos; proceso{seconds:.6f}s.')


if __name__=='__main__':main()
