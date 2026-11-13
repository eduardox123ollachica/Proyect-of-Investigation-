"""Cierre del primer representante: solo artefactos, sin ajustar o inferir."""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'codigo/cwru/ampliacion_v2/representantes'))
from ejecutar_representantes import inspect_campaign, read, write, sha, PREP, verify_manifest
from numerica_representantes import integer_argmax, quantize_int8, normalize_f32
from datos_representantes import load_verified_context

CAMPAIGN = ROOT / 'resultados/cwru/ampliacion_v2/representantes_v2_2026-10-25/20261025T171847_957295Z'
OUTPUT = ROOT / 'articulo/ampliacion_cwru/metodologia/cierre_primer_representante_2026-10-25'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def main():
    plan, jobs, done, consumed = inspect_campaign(CAMPAIGN)
    require(done == ['cost_outer0_inner1_F14_seed11'], 'El cierre solo cubre el primer representante')
    require(not (CAMPAIGN / 'EJECUCION_EN_CURSO.lock').exists(), 'Campana aun activa')
    job = jobs[0]
    directory = CAMPAIGN / job['job_id']
    batch = CAMPAIGN / 'lotes/20261025T171847_957295Z'
    status = read(batch / ('ESTADO_' + job['job_id'] + '.json'))
    require(status['status'] == 'PASS' and status['return_code'] == 0 and status['failure'] is None
            and status['result_error'] is None, 'Proceso no aprobado')
    result = read(directory / 'resultado.json')
    provenance = read(directory / 'procedencia.json')
    _, _, rows, inputs = load_verified_context()
    index = {int(row['global_window_id']): row for row in rows}
    require(provenance['train_global_window_ids'] == job['training_global_window_ids']
            and provenance['validation_global_window_ids'] == job['validation_global_window_ids'], 'Procedencia ajena')
    sources = provenance['source_arrays']
    for item in sources:
        require(item['sha256'] == inputs[item['path']] == sha(ROOT / item['path']), 'Cache fuente modificado')
    train_records = sorted({str(index[k]['file_id']) for k in job['training_global_window_ids']}, key=int)
    val_records = sorted({str(index[k]['file_id']) for k in job['validation_global_window_ids']}, key=int)
    require(len(train_records) == 14 and len(val_records) == 7 and set(train_records).isdisjoint(val_records),
            'Registros de ajuste/validacion no disjuntos')
    expected_counts = np.bincount([int(index[k]['label']) for k in job['training_global_window_ids']], minlength=3)
    state = read(directory / 'estado_antes_conversion.json')
    require(state['class_counts'] == expected_counts.tolist()
            and state['class_weights'] == (len(job['training_global_window_ids'])/(3.*expected_counts)).tolist(),
            'Pesos de clase ajenos al ajuste')
    require(result['optimizer_updates'] == result['epochs_executed'] * ((len(job['training_global_window_ids'])+31)//32),
            'Actualizaciones incompatibles con epocas/lotes')
    with np.load(directory / 'predicciones_validacion.npz', allow_pickle=False) as saved:
        require(np.array_equal(integer_argmax(saved['fp32_probabilities']), saved['fp32_predictions'])
                and np.array_equal(integer_argmax(saved['int8_outputs']), saved['int8_predictions']),
                'Clases guardadas difieren de las salidas')
        require(int(np.count_nonzero(saved['fp32_predictions'] != saved['int8_predictions']))
                == result['FP32_INT8_class_changes'], 'Conteo FP32/INT8 inconsistente')
        require(saved['int8_inputs'].dtype == np.int8 and saved['int8_outputs'].dtype == np.int8,
                'Entradas/salidas no INT8')
    with np.load(directory / 'cuantizacion_train.npz', allow_pickle=False) as saved:
        require(saved['global_window_ids'].tolist() == job['training_global_window_ids']
                and saved['int8_inputs'].shape == (812, 14), 'Cuantizacion train incompleta')
    with np.load(directory / 'scaler_train_only.npz', allow_pickle=False) as scaler, \
         np.load(directory / 'banco_costos_referencia.npz', allow_pickle=False) as saved:
        normalized = normalize_f32(saved['raw_features14'][:,job['feature_indices']], scaler['mean32'], scaler['scale32'])
        quantized, clipped = quantize_int8(normalized, *result['int8_details']['input_quantization'])
        require(np.array_equal(normalized, saved['normalized_features']) and np.array_equal(quantized, saved['int8_inputs'])
                and clipped == result['benchmark_int8_details']['input_clipped_values'], 'Banco transformado inconsistente')
    require(result['versions'] == plan['tested_versions'], 'Versiones distintas')
    for key in ('int8_details', 'benchmark_int8_details'):
        details = result[key]
        require(details['input_dtype'] == details['output_dtype'] == 'int8'
                and set(details['tensor_dtypes']) <= {'int8','int32','int64'}
                and details['operators'] == ['FULLY_CONNECTED']*3+['SOFTMAX'], 'Tipos/operadores no full INT8')
    fit_cal = {}
    for name, ids in (('fit',job['detector_fit_global_window_ids']),('calibration',job['detector_calibration_global_window_ids'])):
        fit_cal[name] = dict(windows=len(ids), file_ids=sorted({index[k]['file_id'] for k in ids},key=int),
                            loads_hp=sorted({int(index[k]['load_hp']) for k in ids}))
    report = dict(status='PASS', scope='first_inner_representative_only',
        job_id=job['job_id'], campaign=CAMPAIGN.relative_to(ROOT).as_posix(),
        completed_jobs=done, remaining_jobs=31, resumption_verified_without_execution=True,
        training_windows=812, validation_windows=393, training_records=train_records, validation_records=val_records,
        roles=job['roles'], detector_scopes=fit_cal, best_epoch=result['best_epoch'],
        epochs_executed=result['epochs_executed'], optimizer_updates=result['optimizer_updates'],
        metrics=result['metrics'], detector_metrics=result['detector_metrics'], model_bytes=result['model_bytes'],
        FP32_INT8_class_changes=result['FP32_INT8_class_changes'],
        saturations=dict(training=result['training_saturated_input_values'],validation=result['int8_details']['input_clipped_values'],
                         benchmark=result['benchmark_int8_details']['input_clipped_values']),
        process_and_audit_seconds=consumed, worker_elapsed_seconds=result['elapsed_worker_seconds'],
        linear_remaining_31_seconds=31*consumed, linear_estimate_not_ETA_or_bound=True,
        remaining_campaign_limit_seconds=plan['limits']['campaign_worker_seconds']-consumed,
        benchmark_rows=28, benchmark_repetitions_planned_not_run=3,
        STM32_cost_measured=False, outer_predictive_evaluation_executed=False,
        raw_source_feature_arrays_opened_in_audit=False, cached_feature_file_hashes_checked=True,
        new_training_fits=0, new_model_inferences=0, sklearn_or_tensorflow_imported=False,
        historical_cache_may_contain_excluded_rows=True,
        limitations=['Internal validation for one load/seed/mask, not final independent evaluation.',
            'Operator metadata and saved outputs audited; no fresh TFLite inference or independent FlatBuffer parser.',
            'Source provenance and intended row roles do not constitute an OS access trace.'],
        next_command=['python3.12.exe','-B',str(ROOT/'codigo/cwru/ampliacion_v2/representantes/ejecutar_representantes.py'),
                      '--execute','--resume',str(CAMPAIGN),'--limit','31'])
    OUTPUT.mkdir(exist_ok=False)
    write(OUTPUT / 'AUDITORIA_PRIMER_REPRESENTANTE.json', report)
    (OUTPUT / 'AUDITORIA_PRIMER_REPRESENTANTE.md').write_text(
        '# Primer representante V2 auditado\n\n'
        'Resultado PASS. Se verificaron sellos, firma, IDs/etiquetas, pesos de clase,\n'
        'historia ES, restauración de la época 48, modelos, salidas guardadas,\n'
        'métricas, P95, detector y banco. No se reentrenó ni se ejecutó inferencia.\n\n'
        'F14, semilla 11: ajuste 812 ventanas de 14 MAT (2/3 HP), validación\n'
        '393 ventanas de 7 MAT (1 HP); carga externa 0 HP excluida. Cincuenta\n'
        'épocas, 1300 actualizaciones, mejor época 48 restaurada. No se activó\n'
        'parada temprana antes del límite de 50 épocas.\n\n'
        'FP32 e INT8: exactitud y macro-F1 1.0000 en esta validación interna,\n'
        'matriz diagonal 58/161/174 (N/IR/OR), cero cambios de clase y cero\n'
        'valores de entrada recortados en ajuste, validación o banco. Detector:\n'
        'especificidad N y recalls IR/OR 1.0000 en estos registros. Estos valores\n'
        'no representan el resultado final multisemilla o externo.\n\n'
        'Proceso con comprobación del padre: 14.577924 s; worker hasta resultado:\n'
        '12.232391 s. El entrenamiento con guardado ocupó 2.800226 s. La\n'
        'extrapolación lineal de los 31 es 451.915656 s (7.53 min), sensible\n'
        'a importaciones, otras máscaras y parada temprana: no es una ETA garantizada.\n'
        'Mantener la reserva orientativa de 15–30 min y el límite global de\n'
        '7200 s acumulados. No se midió costo total STM32.\n\n'
        'Se puede continuar la misma campaña con --resume y --limit 31. Los\n'
        'fuentes sellados y el primer resultado se conservan. Véase el JSON\n'
        'para la ruta y el comando completos; la ejecución la inicia el usuario.\n', encoding='utf-8', newline='\n')
    # Sello de cierre solo sobre recursos inmutables; excluye el estado operativo mutable.
    resources = dict(verify_manifest(PREP / 'MANIFEST_PREPARACION_SHA256.json'))
    resources.update(verify_manifest(batch / 'MANIFEST_LOTE_SHA256.json'))
    paths = [Path(__file__), OUTPUT/'AUDITORIA_PRIMER_REPRESENTANTE.json',OUTPUT/'AUDITORIA_PRIMER_REPRESENTANTE.md',
             PREP/'MANIFEST_PREPARACION_SHA256.json',PREP/'MANIFEST_PREPARACION_SHA256.sha256',
             batch/'MANIFEST_LOTE_SHA256.json',batch/'MANIFEST_LOTE_SHA256.sha256',
             ROOT/'resultados/cwru/ampliacion_v2/representantes_v2_2026-10-25/CAMPANA_UNICA.json']
    resources.update({p.relative_to(ROOT).as_posix():sha(p) for p in paths})
    manifest=OUTPUT/'MANIFEST_CIERRE_SHA256.json'
    write(manifest, resources)
    manifest.with_suffix('.sha256').write_text(sha(manifest)+'  '+manifest.name+'\n',encoding='ascii')
    verify_manifest(manifest)
    print('PASS: 1/32; '+str(len(resources))+' recursos del cierre; ninguna ejecucion nueva de modelos.')


if __name__ == '__main__':
    main()
