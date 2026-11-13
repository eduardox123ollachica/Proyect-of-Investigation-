"""Audita y documenta el piloto completado sin entrenar o importar TensorFlow."""
from __future__ import annotations
import argparse
import ast
import csv
import datetime
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / 'codigo/cwru/ampliacion_v2/piloto'
OUT = ROOT / 'articulo/ampliacion_cwru/metodologia/cierre_piloto_tiempo_2026-10-25'
sys.path.insert(0, str(SOURCE))
from presupuesto_piloto import project_dnn_budget


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def verify_manifest(path):
    digest, filename = path.with_suffix('.sha256').read_text(encoding='ascii').split()
    require(filename == path.name and sha(path) == digest, 'Sello distinto: ' + str(path))
    entries = read(path)
    for name, expected in entries.items():
        resource = (ROOT / name).resolve()
        require(resource.is_relative_to(ROOT) and sha(resource) == expected, 'Recurso modificado: ' + name)
    return entries


def audit(run, recovery):
    run = run.resolve()
    require(run.parent == ROOT / 'resultados/cwru/ampliacion_v2/piloto_tiempo_exportacion_v2_2026-10-25',
            'Solo se acepta la revision de exportacion V2')
    run_entries = verify_manifest(run / 'MANIFEST_RESULTADOS_PILOTO_SHA256.json')
    recovery=recovery.resolve()
    require(recovery.parent==ROOT/'resultados/cwru/ampliacion_v2/recuperacion_conversion_t6_2026-10-25',
            'Recuperacion fuera del ambito')
    run_entries.update(verify_manifest(recovery/'MANIFEST_RECUPERACION_SHA256.json'))
    for p in (recovery/'MANIFEST_RECUPERACION_SHA256.json',recovery/'MANIFEST_RECUPERACION_SHA256.sha256'):
        run_entries[p.relative_to(ROOT).as_posix()]=sha(p)
    recovery_status=read(recovery/'ESTADO_RECUPERACION.json')
    require(recovery_status['status']=='PASS' and recovery_status['training_updates_executed']==0,
            'La recuperacion debe aprobar sin entrenamiento')
    original = ROOT / 'resultados/cwru/ampliacion_v2/piloto_tiempo_2026-10-25/20261025T063025_963041Z'
    old_entries = verify_manifest(original / 'MANIFEST_RESULTADOS_PILOTO_SHA256.json')
    old_summary = read(original / 'RESUMEN_PILOTO.json')
    require(old_summary['jobs_started'] == 1 and old_summary['jobs_passed'] == 0,
            'El intento previo debe conservarse como fallo, no como perfil aprobado')
    summary = read(run / 'RESUMEN_PILOTO.json')
    require(summary['status'] == 'PILOT_STOPPED_ON_FIRST_FAILURE' and summary['jobs_started'] == 2
            and summary['jobs_passed']==1
            and summary['mass_training_started'] is False
            and summary['outer_predictive_evaluation_executed'] is False, 'Piloto incompleto o fuera de alcance')
    protocol = ROOT / 'articulo/ampliacion_cwru/metodologia/protocolo_v2_2026-10-25'
    verify_manifest(protocol / 'MANIFEST_PROTOCOLO_SHA256.json')
    plan = read(protocol / 'piloto_plan.json')
    with (ROOT / 'datos/cwru/particiones/ampliacion_v2/por_carga_2026-10-22/indice_ventanas_fuente.csv').open(encoding='utf-8-sig', newline='') as stream:
        index = {int(r['global_window_id']): r for r in csv.DictReader(stream)}
    checked, results = [], []
    for job, status in zip(plan['jobs'], summary['statuses']):
        source_directory=run/job['job_id']
        recovered=job['mask']=='T6'
        if recovered:
            require(status['status']=='FAIL','No ocultar el fallo T6 previo')
            status=dict(status,status='PASS',elapsed_process_seconds=status['elapsed_process_seconds']+
                        recovery_status['elapsed_process_seconds'])
        directory = recovery if recovered else source_directory
        result = read(directory / 'resultado.json')
        results.append(result)
        require(result['status'] == status['status'] == 'PASS' and result['job_id'] == job['job_id']
                and result['mask'] == job['mask'] and result['optimizer_updates'] == 1250
                and result['representative_calibration_rows'] == 799 and result['epochs'] == 50,
                'Identidad/actualizaciones/calibracion de trabajo distintas')
        if recovered:
            history=read(source_directory/'estado_antes_conversion.json')['history']
        else:
            with (directory / 'historial_piloto.csv').open(encoding='utf-8', newline='') as stream:
                history = list(csv.DictReader(stream))
        require(len(history) == 50 and all(int(r['epoch']) == i + 1 and int(r['optimizer_updates']) == 25 * (i + 1)
                and math.isfinite(float(r['loss'])) and math.isfinite(float(r['val_loss'])) for i, r in enumerate(history)),
                'Historial incompleto o actualizaciones alteradas')
        provenance = read(directory / 'procedencia.json')
        require(provenance['train_global_window_ids'] == job['train']['global_window_ids']
                and provenance['validation_global_window_ids'] == job['validation']['global_window_ids']
                and all(int(index[i]['load_hp']) in (1, 2) for i in provenance['train_global_window_ids'])
                and all(int(index[i]['load_hp']) == 0 for i in provenance['validation_global_window_ids']),
                'IDs fuera del piloto')
        require(all(sha(ROOT / a['path']) == a['sha256'] and Path(a['path']).name in
                    ('X_features_train.npy', 'X_features_validation.npy') for a in provenance['source_arrays']),
                'Cache fuente alterado o no autorizado')
        train_labels = np.asarray([int(index[i]['label']) for i in job['train']['global_window_ids']])
        expected_counts=([job['train']['class_window_counts'][k] for k in ('N','IR','OR')]
                         if recovered else result['class_counts'])
        require(np.bincount(train_labels, minlength=3).tolist() == expected_counts, 'Clases entrenamiento distintas')
        d = job['input_dimension']
        with np.load(source_directory / 'scaler_train_only.npz', allow_pickle=False) as scaler:
            require(scaler['mean64'].dtype == scaler['scale64'].dtype == np.float64
                    and scaler['mean32'].dtype == scaler['scale32'].dtype == np.float32
                    and scaler['mean32'].shape == scaler['scale32'].shape == (d,)
                    and np.array_equal(scaler['mean64'].astype(np.float32), scaler['mean32'])
                    and np.array_equal(scaler['scale64'].astype(np.float32), scaler['scale32']), 'Scaler inconsistente')
        with np.load(source_directory / 'checkpoint_antes_conversion.npz', allow_pickle=False) as checkpoint:
            require(len(checkpoint.files)==6 and [checkpoint[k].shape for k in checkpoint.files]==
                    [(d,20),(20,),(20,10),(10,),(10,3),(3,)] and all(np.isfinite(checkpoint[k]).all()
                    for k in checkpoint.files),'Checkpoint incompleto')
            if recovered:
                require(result['freeze_diagnostics']['weights_unchanged'] and
                        result['freeze_diagnostics']['source_checkpoint_sha256']==sha(source_directory/'checkpoint_antes_conversion.npz'),
                        'Pesos recuperados distintos')
            else:
                with np.load(directory/'pesos_piloto_no_confirmatorios.npz',allow_pickle=False) as weights:
                    require(weights.files==checkpoint.files and all(weights[k].tobytes()==checkpoint[k].tobytes()
                            for k in weights.files),'Exportacion cambio pesos')
        with np.load(directory / 'diagnostico_conversion_validacion.npz', allow_pickle=False) as values:
            truth = np.asarray([int(index[i]['label']) for i in job['validation']['global_window_ids']], dtype=np.int64)
            require(np.array_equal(values['y_true'], truth) and np.array_equal(values['validation_global_window_ids'],
                    job['validation']['global_window_ids']), 'Diagnostico con etiquetas/IDs distintos')
            require(all(values[k].shape == (175, 3) and np.isfinite(values[k]).all()
                        for k in ('keras_probabilities','fp32_probabilities','int8_probabilities')), 'Diagnostico no finito')
            scale, zero = result['int8_details']['output_quantization']
            restored = (values['int8_outputs'].astype(np.float32) - np.float32(zero)) * np.float32(scale)
            require(restored.tobytes() == values['int8_probabilities'].tobytes(), 'Dequantizacion registrada inconsistente')
            require(int(np.count_nonzero(np.argmax(values['fp32_probabilities'], axis=1) !=
                        np.argmax(values['int8_probabilities'], axis=1))) == result['FP32_INT8_class_changes'],
                    'Conteo de cambios FP32/INT8 inconsistente')
        for kind, name in (('FP32', 'modelo_fp32.tflite'), ('INT8', 'modelo_int8.tflite')):
            path = directory / name
            require(path.read_bytes()[4:8] == b'TFL3' and sha(path) == result['model_sha256'][kind]
                    and path.stat().st_size == result['model_bytes'][kind], 'Modelo TFLite distinto del informe')
        if recovered:
            freeze=result['freeze_diagnostics']
            require(freeze['weights_unchanged'] and freeze['original_batch1_vs_frozen_batch1_allclose']
                    and freeze['original_batch1_vs_frozen_batch1_class_changes']==0,'Freeze a igual lote fallido')
            freeze_error=freeze['original_batch1_vs_frozen_batch1_max_absolute_error']
        else:
            require(result['freeze_model_weights_unchanged'] is True and result['freeze_validation_class_changes'] == 0
                    and result['freeze_resource_variable_ops_remaining'] == 0,'Freeze fallido')
            freeze_error=result['freeze_validation_probability_max_error']
        require(result['int8_details']['input_dtype'] == result['int8_details']['output_dtype'] == 'int8'
                and set(result['int8_details']['tensor_dtypes']) <= {'int8', 'int32', 'int64'}, 'Contrato de exportacion fallido')
        times = result['timing_nonoverlapping_seconds']
        require(all(math.isfinite(v) and v >= 0 for v in times.values())
                and status['elapsed_process_seconds'] >= sum(times.values()), 'Cronometraje inconsistente')
        checked.append(dict(mask=job['mask'], process_seconds=status['elapsed_process_seconds'],
            process_kind='composite_two_processes_export_recovery' if recovered else 'single_complete_process',
            imports_seconds=times['imports'], training_updates_seconds=times['training_updates'],
            training_epochs_including_diagnostics_seconds=sum(times[k] for k in
                ('training_updates','epoch_training_diagnostics','epoch_validation')),
            FP32_conversion_seconds=times['FP32_conversion'], INT8_conversion_seconds=times['INT8_conversion'],
            inference_seconds=times['inference'], parameters=result['parameters'],
            FP32_bytes=result['model_bytes']['FP32'], INT8_bytes=result['model_bytes']['INT8'],
            freeze_probability_max_error=freeze_error,
            FP32_INT8_class_changes=result['FP32_INT8_class_changes'], versions=result['versions']))
    projection = project_dnn_budget(results)
    gaps={r['mask']:max(0.,r['process_seconds']-sum(p['timing_nonoverlapping_seconds'].values()))
          for r,p in zip(checked,results)}
    projection['profile_kinds']={r['mask']:r['process_kind'] for r in checked}
    projection['T6_composite_includes_repeated_import_and_setup']=True
    projection['range_is_not_a_comparison_of_representation_speed']=True
    projection['observed_process_minus_stages_seconds_by_mask']=gaps
    projection['campaign_actual_training_attempts']=3
    projection['campaign_recovery_training_updates']=0
    for scenario in projection['scenarios'].values():
        scenario['additional_observed_process_gap_scenario_seconds']=408*max(gaps.values())
        scenario['max_profile_plus_observed_gap_scenario_seconds']=(scenario['conservative_max_seconds']+
            scenario['additional_observed_process_gap_scenario_seconds'])
    old_events=[json.loads(line) for line in (original/'pilot_outer3_inner0_F14_seed11/progreso.jsonl').read_text(encoding='utf-8').splitlines()]
    cold_import=next(e['elapsed_seconds'] for e in old_events if e['stage']=='imports_complete')-old_events[0]['elapsed_seconds']
    projection['initial_cold_import_approximate_seconds']=cold_import
    projection['initial_cold_import_is_approximation_from_stage_events']=True
    for scenario in projection['scenarios'].values():
        base=scenario['profiles']['F14']['projected_DNN_seconds']
        scenario['all_fits_like_initial_cold_import_sensitivity_seconds']=(
            base+408*max(0.,cold_import-results[0]['timing_nonoverlapping_seconds']['imports'])+
            408*gaps['F14'])
    def training_ast(path):
        tree = ast.parse(path.read_text(encoding='utf-8'))
        nodes = [n for n in ast.walk(tree) if (isinstance(n, ast.FunctionDef) and n.name == 'training_step')
                 or (isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == 'epoch')]
        return [ast.dump(n, include_attributes=False) for n in nodes]
    require(training_ast(SOURCE / 'worker_piloto_tiempo.py') == training_ast(SOURCE / 'worker_piloto_tiempo_v2.py'),
            'La correccion no debe alterar entrenamiento')
    return dict(status='PILOT_AUDITED_NO_RETRAINING', generated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        run=run.relative_to(ROOT).as_posix(), rows=checked, projection=projection,
        recovery_run=recovery.relative_to(ROOT).as_posix(),
        recovery_training_updates=0,
        T6_freeze_diagnostic=read(recovery/'DIAGNOSTICO_FREEZE.json'),
        observed_campaign_process_seconds=old_summary['observed_worker_process_seconds']+
            summary['observed_worker_process_seconds']+recovery_status['elapsed_process_seconds'],
        run_resources_verified=len(run_entries), failed_attempt_resources_verified=len(old_entries),
        failed_attempt_preserved=True, failed_attempt_process_seconds=old_summary['observed_worker_process_seconds'],
        successful_fits=2, prior_failed_conversion_fit=1, total_training_attempts=3,
        completed_successful_epochs_each=50, completed_successful_updates_each=1250,
        training_loop_unchanged_by_AST=True, source_hashes_rechecked=True,
        no_source_NPY_arrays_loaded_by_auditor=True, TensorFlow_imported_by_auditor=False,
        auditor_preclosure_corrections=['Read historical CSV with utf-8-sig to handle BOM.',
            'Use sealed job class counts for the export-only recovery schema.'],
        inference_reexecuted_by_auditor=False, models_trained_by_auditor=0,
        complete_budget_frozen=False, classical_runtime_pilot_pending=True,
        full_STM32_cost_calibration_pending=True), run_entries, old_entries


def write_outputs(report, run_entries, old_entries):
    require(not OUT.exists(), 'No sobrescribir cierre existente')
    OUT.mkdir()
    result_path = OUT / 'CIERRE_PILOTO.json'
    result_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    lines = ['# Piloto DNN V2: resultados y alcance', '',
        'Piloto de tiempo y conversion, sin seleccion de representacion ni resultados confirmatorios.', '',
        '| Mascara | Tiempo operativo PC (s) | Importacion (s) | 50 epocas con diagnostico (s) | FP32 (s) | INT8 (s) | INT8 (B) |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in report['rows']:
        lines.append(f"| {r['mask']} | {r['process_seconds']:.3f} | {r['imports_seconds']:.3f} | {r['training_epochs_including_diagnostics_seconds']:.3f} | {r['FP32_conversion_seconds']:.3f} | {r['INT8_conversion_seconds']:.3f} | {r['INT8_bytes']} |")
    lines.extend(['', 'Ambos trabajos completaron 799 ventanas de entrenamiento, 175 de diagnostico,',
        '50 epocas, 1250 actualizaciones y calibracion INT8 con las 799 filas de entrenamiento.',
        'Se verificaron historiales, IDs, scaler, checkpoint, pesos, hashes TFLite y salidas registradas.', '',
        '## Incidencia conservada', '',
        f"El primer intento consumio {report['failed_attempt_process_seconds']:.3f} s y fallo en INT8 por READ_VARIABLE.",
        'Se conservo y sello. F14 se repitio una vez por recuperacion tecnica; T6 no se habia iniciado.',
        'En total hubo tres intentos de entrenamiento, dos pilotos completos. El nuevo exportador',
        'congela los pesos sin alterar el bucle de entrenamiento (AST identico) y guarda un checkpoint',
        'antes de convertir. T6 requirio recuperar solo exportacion desde ese checkpoint, con cero',
        'actualizaciones nuevas. Comparando ambas funciones a lote 1 el error fue cero; comparar',
        'lote 175 y lote 1 daba 2.890825e-6, sin cambios de clase. No se relajaron tolerancias.',
        'El tiempo T6 mostrado suma DOS procesos y ambas importaciones/preparaciones; F14 es un',
        'proceso completo. No se comparan como evidencia de que una representacion sea mas rapida.', '',
        '## Proyeccion de planificacion DNN', '',
        '| Epocas por ajuste | Rango de perfiles, etapas (h) | Maximo de perfil + gap observado de proceso (h) |',
        '|---:|---:|---:|'])
    for s in report['projection']['scenarios'].values():
        span=s['range_between_two_pilot_profiles_seconds']
        lines.append(f"| {s['epochs_assumed']} | {span['min']/3600:.3f}--{span['max']/3600:.3f} | {s['max_profile_plus_observed_gap_scenario_seconds']/3600:.3f} |")
    lines.extend(['', 'Los escenarios contemplan 288 ajustes internos y hasta 120 externos (408),',
        'con hasta 816 conversiones antes de reutilizar identicos. Los 32 representantes ya estan',
        'incluidos; pilotos y recuperacion se contabilizan aparte. Son extrapolaciones de dos',
        'perfiles, no intervalos de confianza, WCET, duraciones garantizadas o latencia STM32.',
        'FP32_conversion incluye congelacion y comparacion de 175 inferencias; tratarla como costo',
        'fijo es una aproximacion. El gap agrega arranque, trabajo no atribuido, escritura final y salida.', '',
        'El intento inicial con importacion fria se conserva separado: dos perfiles posteriores no',
        'garantizan que todos los procesos futuros importen igual de rapido. No se usaron resultados',
        'predictivos para elegir perfiles, mascaras, epocas confirmatorias o margenes.', '',
        '**Pendiente:** piloto RF/SVM, costo de ajuste/calibracion del detector, presupuesto global y',
        'medicion del pipeline STM32 completo para los representantes. Busqueda masiva y evaluacion',
        'externa no ejecutadas. Manuscrito, particiones y evidencias historicas se conservan.', '',
        'Evidencias: `' + report['run'] + '`.'])
    cold=report['projection']['initial_cold_import_approximate_seconds']
    cold_hours=report['projection']['scenarios']['epochs_50']['all_fits_like_initial_cold_import_sensitivity_seconds']/3600
    lines.extend(['',f'La primera importacion fue aproximadamente {cold:.3f} s (estimada desde eventos).',
        f'Si se repitiera en cada ajuste, el escenario de sensibilidad de 50 epocas seria {cold_hours:.3f} h DNN.',
        'Es un supuesto de sensibilidad, no una prediccion ni una garantia; demuestra por que el',
        'presupuesto global todavia no se declara congelado.', '',
        '## Diagnostico FP32/INT8', '',
        'Los cambios de clase se conservan como diagnostico de estas instancias de 50 epocas,',
        'sin usarlos para elegir una mascara o modificar el protocolo. No equivalen por si solos',
        'a una perdida de exactitud: se necesitan etiquetas y el protocolo predictivo correspondiente.'])
    for r in report['rows']:
        lines.append(f"- {r['mask']}: {r['FP32_INT8_class_changes']} cambios entre TFLite FP32 e INT8 en 175 ventanas.")
    (OUT / 'CIERRE_PILOTO.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    paths = dict(old_entries)
    paths.update(run_entries)
    for path in [Path(__file__), result_path, OUT / 'CIERRE_PILOTO.md']:
        paths[path.relative_to(ROOT).as_posix()] = sha(path)
    for base in (ROOT / report['run'], ROOT / 'resultados/cwru/ampliacion_v2/piloto_tiempo_2026-10-25/20261025T063025_963041Z'):
        for name in ('MANIFEST_RESULTADOS_PILOTO_SHA256.json', 'MANIFEST_RESULTADOS_PILOTO_SHA256.sha256'):
            path=base/name; paths[path.relative_to(ROOT).as_posix()] = sha(path)
    manifest=OUT/'MANIFEST_CIERRE_PILOTO_SHA256.json'
    manifest.write_text(json.dumps(paths,indent=2)+'\n',encoding='utf-8')
    manifest.with_suffix('.sha256').write_text(sha(manifest)+'  '+manifest.name+'\n',encoding='ascii')
    verify_manifest(manifest)
    return len(paths)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--recovery-dir',type=Path,required=True)
    parser.add_argument('--write',action='store_true')
    args=parser.parse_args()
    report,run_entries,old_entries=audit(args.run_dir,args.recovery_dir)
    count=write_outputs(report,run_entries,old_entries) if args.write else None
    print(json.dumps(dict(status=report['status'],rows=report['rows'],sealed_resources=count),indent=2))
