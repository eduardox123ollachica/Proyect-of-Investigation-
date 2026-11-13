"""Cierre del suplemento P0/P1 y CONTROL12/MOM4 desde artefactos existentes."""
from __future__ import annotations
from math import fsum
from pathlib import Path
from statistics import median
import csv
import platform
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'codigo/cwru/ampliacion_v2/suplemento'))
from ejecutar_suplemento import inspect_run,read,write,sha,verify_manifest,PREP,DIAG
from contrato_datos import context,load_data
from preprocesamiento import fit_params,apply
from numerica_representantes import normalize_f32,quantize_int8,detector_metrics,metrics3
from detector_suplemento import detector_jobs,descriptive_guards
from reglas_v2 import weighted_p95

RUN=ROOT/'resultados/cwru/ampliacion_v2/suplemento_preprocesamiento_normalidad_2026-10-25/20261025T215652_507777Z'
OUT=ROOT/'articulo/ampliacion_cwru/metodologia/cierre_suplemento_completo_2026-10-25'


def require(ok,message):
    if not ok:raise ValueError(message)


def average(values):
    values=list(values);return fsum(values)/len(values)


def csv_write(path,rows):
    with path.open('x',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    plan,pending,done,seconds=inspect_run(RUN)
    require(len(done)==34 and not (RUN/'ACTIVO.lock').exists(),'Campana incompleta o activa')
    require(read(RUN/'REFERENCIAS_IMPORTADAS.json')==plan['imported_references'],'Imports diferentes')
    _,metadata,_=context();by_id={int(r['global_window_id']):r for r in metadata}
    batches=sorted((RUN/'lotes').iterdir());statuses=[]
    for batch in batches:
        bp=read(batch/'PLAN_LOTE.json');summary=read(batch/'RESUMEN_LOTE.json')
        for name in bp['job_ids']:
            statuses.append(read(batch/('ESTADO_'+name+'.json')))
        if summary['detector_complete']:statuses.append(read(batch/'ESTADO_DETECTOR_BLOCK.json'))
    require(len(statuses)==35 and all(s['status']=='PASS' and s['return_code']==0 and s['error'] is None for s in statuses),
            'Estado fallido o falta proceso')
    require(abs(fsum(s['process_seconds'] for s in statuses)-seconds)<1e-8,'Tiempos incompatibles')
    require(read(batches[-1]/'RESUMEN_LOTE.json')['status']=='SUPPLEMENT_COMPLETED','Falta cierre completo')
    cells=[];records=[];lookup={}
    for job in plan['jobs']:
        imported=job['job_id'] in plan['imported_references']
        source=(ROOT/plan['imported_references'][job['job_id']]['source_result']) if imported else RUN/job['job_id']/'resultado.json'
        result=read(source);directory=source.parent
        require(result['versions']==plan['tested_versions'],'Versiones incompatibles')
        raw,y,raw_val,y_val,provenance=load_data(job)
        params=fit_params(raw,job['mode']);ix=job['feature_indices']
        with np.load(directory/'scaler_train_only.npz',allow_pickle=False) as scaler:
            normalized=normalize_f32(apply(raw_val,params)[:,ix],scaler['mean32'],scaler['scale32'])
        q,clipped=quantize_int8(normalized,*result['int8_details']['input_quantization'])
        scale,zp=result['int8_details']['input_quantization']
        affine=np.add(np.divide(normalized,np.float32(scale),dtype=np.float32),np.float32(zp),dtype=np.float32)
        rounded=np.rint(affine);clip_window=((rounded < -128)|(rounded > 127)).any(axis=1)
        with np.load(directory/'predicciones_validacion.npz',allow_pickle=False) as saved:
            require(saved['global_window_ids'].tolist()==job['validation_global_window_ids']
                    and np.array_equal(saved['y_true'],y_val) and np.array_equal(saved['int8_inputs'],q), 'NPZ incongruente')
            fp_pred,iq_pred=saved['fp32_predictions'],saved['int8_predictions']
            require(metrics3(y_val,fp_pred)==result['metrics']['FP32'] and metrics3(y_val,iq_pred)==result['metrics']['INT8'],
                    'Metricas no reproducibles')
            changes=int((fp_pred!=iq_pred).sum())
            val_record_ids=[by_id[k]['file_id'] for k in job['validation_global_window_ids']]
            for record in sorted(set(val_record_ids),key=int):
                take=np.asarray([r==record for r in val_record_ids]);label=int(y_val[take][0])
                records.append(dict(cell_id=job['job_id'],candidate=job['candidate'],mode=job['mode'],
                    inner=job['inner_validation_load_hp'],seed=job['seed'],file_id=record,label=label,windows=int(take.sum()),
                    true_class_recall_FP32=float((fp_pred[take]==label).mean()),
                    true_class_recall_INT8=float((iq_pred[take]==label).mean()),clipped_windows=int(clip_window[take].sum())))
        expected_clip=result['int8_details']['input_clipped_values']
        require(clipped==expected_clip,'Recortes mal contados')
        cells.append(dict(cell_id=job['job_id'],candidate=job['candidate'],mode=job['mode'],inner=job['inner_validation_load_hp'],
            seed=job['seed'],imported=imported,source_result=source.relative_to(ROOT).as_posix(),
            source_fit_signature=result['fit_signature'],training_windows=len(job['training_global_window_ids']),
            validation_windows=len(y_val),macro_f1_FP32=result['metrics']['FP32']['macro_f1'],
            macro_f1_INT8=result['metrics']['INT8']['macro_f1'],accuracy_INT8=result['metrics']['INT8']['accuracy'],
            clipped_values=clipped,input_values=int(normalized.size),clipped_windows=int(clip_window.sum()),
            clipped_value_fraction=clipped/normalized.size,clipped_window_fraction=float(clip_window.mean()),
            class_changes=changes,best_epoch=result['best_epoch'],epochs_executed=result['epochs_executed']))
        lookup[job['candidate'],job['mode'],job['inner_validation_load_hp'],job['seed']]=cells[-1]
    require(len(cells)==36 and sum(c['imported'] for c in cells)==2,'Celdas faltantes/duplicadas')
    paired=[];comparisons={};load_rows=[]
    for candidate in ('F14','NF12'):
        for inner in (0,1,2):
            for seed in (11,22,33):
                p0,p1=[lookup[candidate,mode,inner,seed] for mode in ('P0','P1')]
                paired.append(dict(candidate=candidate,inner=inner,seed=seed,
                    P0_F1_INT8=p0['macro_f1_INT8'],P1_F1_INT8=p1['macro_f1_INT8'],
                    delta_INT8=p1['macro_f1_INT8']-p0['macro_f1_INT8'],
                    P0_F1_FP32=p0['macro_f1_FP32'],P1_F1_FP32=p1['macro_f1_FP32'],
                    delta_FP32=p1['macro_f1_FP32']-p0['macro_f1_FP32']))
        subset=[p for p in paired if p['candidate']==candidate]
        by_load={inner:average(p['delta_INT8'] for p in subset if p['inner']==inner) for inner in (0,1,2)}
        gain=average(p['delta_INT8'] for p in subset)
        modes={}
        for mode in ('P0','P1'):
            selected=[c for c in cells if c['candidate']==candidate and c['mode']==mode]
            load_means={inner:average(c['macro_f1_INT8'] for c in selected if c['inner']==inner) for inner in (0,1,2)}
            modes[mode]=dict(mean_F1_INT8=average(c['macro_f1_INT8'] for c in selected),
                mean_F1_FP32=average(c['macro_f1_FP32'] for c in selected),per_load_mean_F1_INT8=load_means,
                worst_load_mean_F1_INT8=min(load_means.values()),
                clipped_values=sum(c['clipped_values'] for c in selected),input_values=sum(c['input_values'] for c in selected),
                clipped_windows=sum(c['clipped_windows'] for c in selected),validation_windows=sum(c['validation_windows'] for c in selected),
                class_changes=sum(c['class_changes'] for c in selected))
            for inner in (0,1,2):
                load_rows.append(dict(candidate=candidate,mode=mode,validation_load_hp=inner,
                    mean_F1_INT8=load_means[inner],mean_F1_FP32=average(c['macro_f1_FP32'] for c in selected if c['inner']==inner)))
        gain_ok=gain>=.01-1e-12;preserve=all(delta>=-.01-1e-12 for delta in by_load.values())
        comparisons[candidate]=dict(modes=modes,mean_paired_INT8_gain=gain,per_load_mean_delta=by_load,
            engineering_gain_at_least_point01=gain_ok,engineering_no_load_loss_over_point01=preserve,
            promising_under_prespecified_rule=gain_ok and preserve,
            favorable_pairs=sum(p['delta_INT8']>1e-12 for p in subset),equal_pairs=sum(abs(p['delta_INT8'])<=1e-12 for p in subset),
            unfavorable_pairs=sum(p['delta_INT8'] < -1e-12 for p in subset),automatically_selected=False)
    det_summary=read(RUN/'detector/RESUMEN_DETECTOR.json');det_rows=[]
    require(det_summary['status']=='PASS' and len(det_summary['results'])==6
            and det_summary['new_fit_calls']==5 and det_summary['imported_reference_fits']==1,'Detector incompleto')
    for det_job in detector_jobs():
        directory=RUN/'detector'/det_job['detector_id'];result=read(directory/'resultado.json')
        data_job=next(j for j in plan['jobs'] if j['job_id']==det_job['data_job_id'])
        require(result==next(r for r in det_summary['results'] if r['detector_id']==det_job['detector_id']),
                'Resultado detector distinto de resumen')
        with np.load(directory/'parametros.npz',allow_pickle=False) as params:
            threshold=np.float32(params['threshold32'].item())
            require(params['indices'].tolist()==det_job['indices']
                    and params['fit_ids'].tolist()==data_job['detector_fit_global_window_ids']
                    and params['calibration_ids'].tolist()==data_job['detector_calibration_global_window_ids'],
                    'Parametros/roles detector distintos')
        with np.load(directory/'calibracion.npz',allow_pickle=False) as saved:
            ids=data_job['detector_calibration_global_window_ids'];recs=[by_id[k]['file_id'] for k in ids]
            require(saved['global_window_ids'].tolist()==ids
                    and weighted_p95(saved['scores'],recs).tobytes()==threshold.tobytes(),'P95 no reproduce')
        with np.load(directory/'validacion.npz',allow_pickle=False) as saved:
            ids=data_job['validation_global_window_ids'];recs=[by_id[k]['file_id'] for k in ids]
            y=np.asarray([int(by_id[k]['label']) for k in ids],dtype=np.int64)
            require(saved['global_window_ids'].tolist()==ids and np.array_equal(saved['y_true'],y)
                    and np.array_equal(saved['rejected'],saved['scores']>threshold),'Decisiones detector distintas')
            metrics=detector_metrics(y,saved['rejected'],recs)
        require(metrics==result['metrics'] and descriptive_guards(metrics)==result['descriptive_guards'],
                'Metricas/guardias detector incongruentes')
        det_rows.append(dict(representation=result['representation'],inner=result['inner'],imported=result['imported_from_reference'],
            normal_spec=metrics['normal_spec'],ir_recall=metrics['ir_recall'],or_recall=metrics['or_recall'],
            balanced_accuracy=metrics['balanced_accuracy'],all_components_at_least_point95=result['descriptive_guards']['passed'],threshold=float(threshold)))
    report=dict(status='TECHNICAL_PASS_PREDEFINED_PREDICTIVE_CRITERIA_NOT_MET',
        campaign=RUN.relative_to(ROOT).as_posix(),total_DNN_cells=36,new_DNN_fits=34,imported_DNN_references=2,
        detector_cases=6,new_detector_fits=5,imported_detector_controls=1,total_processes=35,
        process_and_verification_seconds=seconds,process_and_verification_minutes=seconds/60,
        second_batch_seconds=read(batches[-1]/'RESUMEN_LOTE.json')['process_seconds'],
        detector_block_seconds=next(s['process_seconds'] for s in statuses if s['job_id']=='DETECTOR_BLOCK'),
        comparisons=comparisons,load_summary=load_rows,paired_results=paired,cells=cells,detector_results=det_rows,
        epochs=dict(hit_cap50=sum(c['epochs_executed']==50 for c in cells),stopped_before50=sum(c['epochs_executed']<50 for c in cells),
            median_executed=median(c['epochs_executed'] for c in cells),median_best=median(c['best_epoch'] for c in cells)),
        normality_summary={name:dict(mean_normal_spec=average(r['normal_spec'] for r in det_rows if r['representation']==name),
            mean_or_recall=average(r['or_recall'] for r in det_rows if r['representation']==name),
            mean_balanced_accuracy=average(r['balanced_accuracy'] for r in det_rows if r['representation']==name),
            loads_passing=sum(r['all_components_at_least_point95'] for r in det_rows if r['representation']==name)) for name in ('CONTROL12','MOM4')},
        audit_runtime=dict(python=platform.python_version(),numpy=np.__version__),new_training_fits_during_audit=0,
        new_TFLite_or_DNN_forward_invocations_during_audit=0,normality_scores_recomputed=False,
        parameters_and_quantization_reconstructed_on_authorized_rows=True,
        main_protocol_changed=False,outer_predictive_evaluation=False,mask_or_preprocessing_selected=False,
        full_STM32_cost_measured=False,manuscript_changed=False,
        conclusions=['Neither P1 mask meets the predefined gain criterion; do not promote compression based on lower clipping alone.',
            'MOM4 fails the per-load specificity/IR/OR guard; better mean BA does not establish reliable fault confirmation.',
            'Keep P0 as original reference and the supplementary outcomes as exploratory evidence, not final outer evaluation.',
            'Main total-cost calibration and full predefined mask/seed/load search remain pending; do not copy costs between model versions.'],
        limitations=['One outer development context only; records0/1/2HP already explored, no independent confirmation.',
            'Seeds are initialization replicates, not independent physical acquisitions; windows overlap.',
            'Totals of clipped values/windows repeat evaluations across seeds.',
            'No new runtime inference or independent FlatBuffer parser was used in this audit.'])
    OUT.mkdir(exist_ok=False);write(OUT/'CIERRE_SUPLEMENTO.json',report)
    for name,rows in [('CELDAS_DNN.csv',cells),('COMPARACION_PAREADA.csv',paired),('RESUMEN_POR_CARGA.csv',load_rows),
                      ('RECALL_POR_MAT.csv',records),('RESULTADOS_DETECTOR.csv',det_rows)]:csv_write(OUT/name,rows)
    lines=['# Cierre del suplemento de desarrollo','',
        '## Dictamen','',
        'Ejecución y auditoría técnica completas:34DNNnuevas,2referencias importadas y',
        'bloque detector con5ajustes nuevos y1control importado. Sin fallos o reintentos.',
        f'Tiempo acumulado: {seconds:.6f}s ({seconds/60:.4f}min).',
        'Se verificaron sellos, firmas, roles, parámetros train-only, checkpoints,',
        'modelos, cuantización reconstruida, métricas, P95 y decisiones guardadas.',
        'Ninguna variante P1 alcanza la ganancia prefijada; MOM4 tampoco supera sus',
        'guardias. Los resultados negativos se conservan íntegros.','',
        '## Clasificador: media igual de tres cargas y tres semillas','',
        '| Máscara | F1 INT8 P0 | F1 INT8 P1 | Diferencia en puntos porcentuales | Criterio previo |',
        '|---|---:|---:|---:|---|']
    for name in ('F14','NF12'):
        comp=comparisons[name]
        lines.append(f"| {name} | {comp['modes']['P0']['mean_F1_INT8']:.6f} | {comp['modes']['P1']['mean_F1_INT8']:.6f} | {100*comp['mean_paired_INT8_gain']:+.3f} | No cumple |")
    lines += ['',
        'F14 pierde rendimiento medio y más de0.01en la carga0HP. NF12 preserva las',
        'cargas, pero su ganancia0.007242es inferior a0.01. En0HP los deltas de NF12',
        'para semillas11/22/33 son−0.113086/−0.030800/+0.207146: la media no representa',
        'una mejora consistente. Se reportan los nueve pares, sin seleccionar semillas.',
        'No se cambia el criterio después de observar que otro secundario resulta favorable.','',
        'Por máscara, recortes672→201sobre las mismas evaluaciones: F14 tiene40908',
        'valores y NF1235064. Ventanas afectadas279→180sobre2922evaluaciones por máscara.',
        'Son conteos repetidos por modelos/semillas, no muestras físicas independientes.',
        'F14 cambia71→6clases por cuantización y NF1270→56; reducir recortes o cambios',
        'no garantiza una clasificación mejor. F14 mantiene recallN0=0en ambos modos.','',
        '## Detector: compromisos entre aceptación sana y sensibilidad','',
        '| Validación HP | Detector | Especificidad N | Recall IR | Recall OR | Guardia>=0.95 |',
        '|---|---|---:|---:|---:|---|']
    for row in det_rows:
        lines.append(f"| {row['inner']} | {row['representation']} | {row['normal_spec']:.6f} | {row['ir_recall']:.6f} | {row['or_recall']:.6f} | {'Sí' if row['all_components_at_least_point95'] else 'No'} |")
    lines += ['',
        'MOM4 mejora especificidad media0.333333→0.717570yBA0.666667→0.812329,',
        'pero reduce recallOR1.0→0.814176. Falla la guardia en las tres cargas.',
        'No se adopta automáticamente ni se presenta como confirmación fiable de falla.',
        'CONTROL12también tiene limitaciones fuertes: falla especificidad sana en0y2HP.',
        'No se elevaron umbrales utilizando las normales rechazadas.','',
        '## Épocas, tiempo y continuidad','',
        '19/36alcanzaron50épocas;17pararon antes. Los12casos de validación0HP',
        'restauran épocas1–3. Estos conteos incluyen las dos referencias importadas.',
        f"El bloque detector tardó{report['detector_block_seconds']:.6f}s. El segundo lote{report['second_batch_seconds']:.6f}s.",
        'La proyección lineal de40min del primer arranque fue conservadora: el total',
        'real de procesos y auditorías del padre fue12min9.976s, sin pausas humanas.',
        'La auditoría actual no se suma como si fuera tiempo de entrenamiento.',
        'Se conserva P0 como referencia; el suplemento no selecciona una máscara final.',
        'No se promueve P1 o MOM4 incumpliendo el diseño fijado. Cualquier variante',
        'posterior exige otra justificación y presupuesto, sin continuar ajustando a',
        'estos mismos resultados como si fuesen una prueba independiente.',
        'El plan principal sigue pendiente de calibración del costo total STM32 y de la',
        'búsqueda completa de máscaras/cargas/semillas. Las32referencias originales',
        'permanecen disponibles. No mezclar costos de firmware y métricas de versiones distintas.',
        'La rama de normalidad debe reportarse con su especificidad y sus limitaciones;',
        'la evidencia actual no respalda interpretarla como verificador fiable de falla.',
        'No se alteraron manuscrito, firmware, datos, particiones o modelos anteriores.','',
        '## Archivos y alcance','',
        'CIERRE_SUPLEMENTO.json contiene resultados y trazabilidad. CincoCSV permiten',
        'ver pares, cargas, MAT y detector. Este bloque es desarrollo retrospectivo',
        'con outer3excluido de nuevas predicciones; no es evaluación externa final.',
        'No se calculan p-valores o intervalosIID sobre ventanas solapadas.',
        'Comprobación sin nuevos entrenamientos, forwardDNN o invocacionesTFLite.','']
    (OUT/'CIERRE_SUPLEMENTO.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')
    entries=verify_manifest(DIAG/'MANIFEST_CIERRE_SHA256.json')
    entries.update(verify_manifest(PREP/'MANIFEST_PREPARACION_SHA256.json'))
    for batch in batches:
        entries.update(verify_manifest(batch/'MANIFEST_LOTE_SHA256.json'))
        for name in ('MANIFEST_LOTE_SHA256.json','MANIFEST_LOTE_SHA256.sha256'):
            path=batch/name;entries[path.relative_to(ROOT).as_posix()]=sha(path)
    prior=ROOT/'articulo/ampliacion_cwru/metodologia/cierre_primer_suplemento_2026-10-25/MANIFEST_CIERRE_SHA256.json'
    entries.update(verify_manifest(prior))
    for path in [Path(__file__),prior,prior.with_suffix('.sha256'),PREP/'MANIFEST_PREPARACION_SHA256.json',
                 PREP/'MANIFEST_PREPARACION_SHA256.sha256']+list(OUT.iterdir()):
        entries[path.relative_to(ROOT).as_posix()]=sha(path)
    manifest=OUT/'MANIFEST_CIERRE_SHA256.json';write(manifest,entries)
    manifest.with_suffix('.sha256').write_text(sha(manifest)+'  '+manifest.name+'\n',encoding='ascii')
    verify_manifest(manifest)
    print(f'CIERRE:36celdas+6detectores; {seconds:.6f}s; sello{len(entries)}recursos.0fitsnuevos durante auditoría.')


if __name__=='__main__':main()
