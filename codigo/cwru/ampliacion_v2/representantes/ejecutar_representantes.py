"""Preparacion, preflight y campana acotada de 32 ajustes internos V2.

Sin --execute: solo lectura. El usuario inicia cada lote; no hay entrenamiento
masivo restante, seleccion por costo ni evaluacion externa en este ejecutor.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
SOURCE = Path(__file__).resolve().parent
PREP = ROOT / 'articulo/ampliacion_cwru/metodologia/representantes_v2_2026-10-25'
OUTPUT = ROOT / 'resultados/cwru/ampliacion_v2/representantes_v2_2026-10-25'
PROTOCOL = ROOT / 'articulo/ampliacion_cwru/metodologia/protocolo_v2_2026-10-25'
ENVIRONMENT = dict(PYTHONHASHSEED='11', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1',
    OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
    TF_NUM_INTRAOP_THREADS='1', TF_NUM_INTEROP_THREADS='1', CUDA_VISIBLE_DEVICES='-1')
ENVIRONMENT.update(TF_ENABLE_ONEDNN_OPTS='0', TF_DETERMINISTIC_OPS='1', TF_CPP_MIN_LOG_LEVEL='2')
LIMITS = dict(worker_seconds=300, campaign_worker_seconds=7200, maximum_fits=32,
              maximum_epochs_per_fit=50, maximum_conversions=64)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def versions():
    return dict(python=platform.python_version(), numpy=importlib.metadata.version('numpy'),
        sklearn=importlib.metadata.version('scikit-learn'), tensorflow=importlib.metadata.version('tensorflow'),
        keras=importlib.metadata.version('keras'))


def verify_manifest(path):
    digest, filename = path.with_suffix('.sha256').read_text(encoding='ascii').split()
    require(filename == path.name and digest == sha(path), 'Sello modificado: ' + str(path))
    entries = read(path)
    require(isinstance(entries, dict) and entries, 'Manifiesto vacio')
    for relative, expected in entries.items():
        target = (ROOT / relative).resolve()
        require(target.is_relative_to(ROOT) and target != ROOT and target.is_file()
                and sha(target) == expected, 'Recurso modificado: ' + relative)
    return entries


def write_manifest(path, directory, extra=()):
    excluded = {path, path.with_suffix('.sha256')}
    paths = [p for p in directory.rglob('*') if p.is_file() and p not in excluded]
    paths += list(extra)
    entries = {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(set(paths))}
    write(path, entries)
    with path.with_suffix('.sha256').open('x', encoding='ascii') as stream:
        stream.write(sha(path) + '  ' + path.name + '\n')
    verify_manifest(path)


def source_hashes():
    dependencies = [ROOT / 'codigo/cwru/ampliacion_v2/protocolo/reglas_v2.py',
                    ROOT / 'codigo/cwru/ampliacion_v2/piloto_clasicos/detector_numerico.py']
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(list(SOURCE.glob('*.py')) + dependencies)}


def fit_signature(job, source, environment, protocol):
    # La validacion forma parte de la firma: ES distingue outer0/1 aun con mismo train.
    content = dict(job=job, sources=source, versions=environment, dnn=protocol['dnn'],
                   numeric=protocol['numeric_contract'], quantization=protocol['quantization'],
                   detector=protocol['detector'], protocol_manifest_sha256=sha(PROTOCOL / 'MANIFEST_PROTOCOLO_SHA256.json'))
    return hashlib.sha256(json.dumps(content, sort_keys=True, separators=(',', ':'),
                                   allow_nan=False).encode('utf-8')).hexdigest()


def prepare():
    from datos_representantes import load_verified_context, build_jobs
    protocol, costs, rows, _ = load_verified_context()
    jobs = build_jobs(protocol, costs, rows)
    tests = read(PREP / 'PRUEBAS_PREPARACION.json')
    synthetic = read(PREP / 'PRUEBA_TENSORFLOW_SINTETICA.json')
    require(tests['status'] == 'PASS' and tests['failures'] == tests['errors'] == tests['skipped'] == 0,
            'Faltan pruebas completas de preparacion')
    require(synthetic['status'] == 'PASS' and synthetic['CWRU_arrays_opened'] is False,
            'Falta prueba sintetica TF sin CWRU')
    source = source_hashes()
    require(tests['source_sha256'] == source and synthetic['source_sha256'] == source,
            'Las fuentes cambiaron desde las pruebas')
    require(tests['versions'] == synthetic['versions'], 'Pruebas en entornos distintos')
    # El presupuesto auditado es antecedente, no una cota temporal garantizada.
    budget = ROOT / 'articulo/ampliacion_cwru/metodologia/cierre_piloto_clasicos_2026-10-25/MANIFEST_CIERRE_CLASICOS_SHA256.json'
    verify_manifest(budget)
    plan = dict(status='PREPARED_FOR_USER_EXECUTION', scope='32_predefined_inner_representatives_only',
        user_starts_training=True, full_protocol_frozen=False, remaining_256_inner_fits_enabled=False,
        outer_predictive_evaluation_enabled=False, cost_selection_enabled=False, manuscript_edits_enabled=False,
        total_STM32_cost_pending=True, fits_included_in_288_inner=True, old_pilot_models_reused=False,
        jobs=jobs, limits=LIMITS, worker_environment=ENVIRONMENT, tested_versions=tests['versions'],
        source_sha256=source, protocol_manifest_sha256=sha(PROTOCOL / 'MANIFEST_PROTOCOLO_SHA256.json'),
        dnn=protocol['dnn'], fit_signatures={j['job_id']: fit_signature(j, source, tests['versions'], protocol) for j in jobs},
        first_execution_recommended_jobs=1, default_execution_jobs=1, automatic_retry=False,
        resume_only_after_complete_matching_jobs=True, failure_blocks_resume=True,
        total_cost_eligibility='requires_later_independent_model_firmware_trace_validation',
        detector_recomputed_for_each_representative=True,
        detector_duplicate_F14_NF12_fits_not_independent_evidence=True)
    write(PREP / 'PLAN_EJECUCION.json', plan)
    extra = [ROOT / name for name in source]
    for path in (PROTOCOL / 'MANIFEST_PROTOCOLO_SHA256.json', budget):
        extra += [path, path.with_suffix('.sha256')]
    write_manifest(PREP / 'MANIFEST_PREPARACION_SHA256.json', PREP, extra)
    print(json.dumps(dict(status='PREPARATION_SEALED', jobs=len(jobs), models_trained_CWRU=0,
                          maximum_worker_seconds=LIMITS['worker_seconds'])))


def preflight():
    from datos_representantes import load_verified_context, build_jobs
    entries = verify_manifest(PREP / 'MANIFEST_PREPARACION_SHA256.json')
    plan = read(PREP / 'PLAN_EJECUCION.json')
    protocol, costs, rows, _ = load_verified_context()
    jobs = build_jobs(protocol, costs, rows)
    require(plan['jobs'] == jobs and len(jobs) == 32 and plan['limits'] == LIMITS
            and plan['source_sha256'] == source_hashes() and plan['scope'] == '32_predefined_inner_representatives_only'
            and plan['worker_environment'] == ENVIRONMENT and plan['dnn'] == protocol['dnn']
            and plan['protocol_manifest_sha256'] == sha(PROTOCOL / 'MANIFEST_PROTOCOLO_SHA256.json')
            and plan['remaining_256_inner_fits_enabled'] is False
            and plan['outer_predictive_evaluation_enabled'] is False and plan['cost_selection_enabled'] is False,
            'Plan, fuentes o alcance distintos')
    expected = {j['job_id']: fit_signature(j, plan['source_sha256'], plan['tested_versions'], protocol) for j in jobs}
    require(expected == plan['fit_signatures'], 'Firma de ajuste distinta')
    return plan, entries


def validate_campaign(campaign):
    campaign = campaign.resolve()
    require(campaign.parent == OUTPUT.resolve() and campaign.is_dir(), 'Campana fuera de su directorio')
    plan, _ = preflight()
    contract = read(campaign / 'CONTRATO_CAMPANA.json')
    require(contract == dict(preparation_manifest_sha256=sha(PREP / 'MANIFEST_PREPARACION_SHA256.json'),
        execution_owner='user_CLI', plan_sha256=sha(PREP / 'PLAN_EJECUCION.json'),
        scope=plan['scope'], maximum_fits=32, automatic_retry=False), 'Contrato de campana distinto')
    return plan, plan['jobs']


def validate_result(result, job, signature=None):
    require(result.get('status') == 'PASS' and result.get('job_id') == job['job_id']
            and result.get('candidate') == job['candidate'] and result.get('seed') == job['seed']
            and result.get('roles') == job['roles'] and result.get('outer_load_hp') == job['outer_load_hp']
            and result.get('inner_validation_load_hp') == job['inner_validation_load_hp'], 'Resultado ajeno o incompleto')
    if signature is not None:
        require(result.get('fit_signature') == signature, 'Firma del resultado distinta')
    require(result.get('training_windows') == len(job['training_global_window_ids'])
            and result.get('validation_windows') == len(job['validation_global_window_ids'])
            and result.get('representative_calibration_rows') == len(job['training_global_window_ids'])
            and result.get('benchmark_rows') == 28 and result.get('parameters') == 20 * job['input_dimension'] + 263,
            'Cantidad/arquitectura distintas')
    require(type(result.get('best_epoch')) is int and type(result.get('epochs_executed')) is int
            and 1 <= result['best_epoch'] <= result['epochs_executed'] <= 50
            and result.get('weights_restored') is True and result.get('outer_evaluation_executed') is False
            and result.get('measured_STM32_total_cost') is None and result.get('remaining_grid_enabled') is False,
            'Checkpoint o alcance incorrecto')
    score = result['metrics']['INT8']['macro_f1']
    require(isinstance(score, (int, float)) and 0 <= score <= 1, 'Macro-F1 no valido')
    return result


def verify_job(directory, job, signature):
    import numpy as np
    from numerica_representantes import metrics3, detector_metrics, EarlyStopping
    from datos_representantes import load_verified_context
    from reglas_v2 import weighted_p95
    entries = verify_manifest(directory / 'MANIFEST_JOB_SHA256.json')
    actual = {p.relative_to(ROOT).as_posix() for p in directory.iterdir()
              if p.is_file() and p.name not in ('MANIFEST_JOB_SHA256.json', 'MANIFEST_JOB_SHA256.sha256')}
    require(set(entries) == actual, 'Archivos ausentes o anadidos al ajuste')
    result = validate_result(read(directory / 'resultado.json'), job, signature)
    state = read(directory / 'estado_antes_conversion.json')
    _, _, rows, _ = load_verified_context()
    by_id = {int(r['global_window_id']): r for r in rows}
    expected_y = np.asarray([int(by_id[k]['label']) for k in job['validation_global_window_ids']], dtype=np.int64)
    expected_records = [str(by_id[k]['file_id']) for k in job['validation_global_window_ids']]
    require(state['weights_restored'] and state['best_epoch'] == result['best_epoch']
            and state['epochs_executed'] == result['epochs_executed'], 'Checkpoint sin restauracion')
    stopper = EarlyStopping()
    for row in state['history']:
        improved, _ = stopper.observe(row['epoch'], row['val_loss'])
        require(improved == row['improved'] and stopper.best_epoch == row['best_epoch']
                and stopper.wait == row['patience_used'], 'Historia ES inconsistente')
    require(stopper.best_epoch == state['best_epoch'] and stopper.best_loss == state['best_validation_loss']
            and stopper.last_epoch == state['epochs_executed'], 'Mejor epoca inconsistente')
    digest = hashlib.sha256()
    with np.load(directory / 'checkpoint_antes_conversion.npz', allow_pickle=False) as restored, \
         np.load(directory / f"checkpoint_epoca_{state['best_epoch']:02d}.npz", allow_pickle=False) as best:
        require(restored.files == best.files and len(restored.files) == 6, 'Checkpoint de pesos incompleto')
        for key in restored.files:
            require(np.array_equal(restored[key], best[key]) and np.isfinite(restored[key]).all(),
                    'Pesos restaurados distintos a mejor epoca')
            digest.update(str(restored[key].shape).encode('ascii'))
            digest.update(restored[key].tobytes())
    require(digest.hexdigest() == state['model_weights_sha256'] == result['freeze']['model_weights_sha256'],
            'Hash de pesos restaurados distinto')
    for family, filename in (('FP32', 'modelo_fp32.tflite'), ('INT8', 'modelo_int8.tflite')):
        require(sha(directory / filename) == result['model_sha256'][family], 'Modelo modificado')
    with np.load(directory / 'predicciones_validacion.npz', allow_pickle=False) as saved:
        require(saved['global_window_ids'].tolist() == job['validation_global_window_ids']
                and np.array_equal(saved['y_true'], expected_y), 'IDs/etiquetas de prediccion distintos del CSV')
        for family, key in (('FP32', 'fp32_predictions'), ('INT8', 'int8_predictions')):
            require(metrics3(saved['y_true'], saved[key]) == result['metrics'][family], 'Metricas no reproducibles')
    with np.load(directory / 'detector_parametros.npz', allow_pickle=False) as saved:
        require(saved['fit_ids'].tolist() == job['detector_fit_global_window_ids']
                and saved['calibration_ids'].tolist() == job['detector_calibration_global_window_ids'], 'Roles detector distintos')
        threshold = np.float32(saved['threshold32'].item())
        require(saved['indices'].tolist() == [i for i in job['feature_indices'] if i not in (4,11)],
                'Variables detector distintas')
    with np.load(directory / 'detector_calibracion.npz', allow_pickle=False) as saved:
        expected_ids = job['detector_calibration_global_window_ids']
        records = [str(by_id[k]['file_id']) for k in expected_ids]
        require(saved['global_window_ids'].tolist() == expected_ids and saved['record_ids'].tolist() == records
                and weighted_p95(saved['scores'], records).tobytes() == threshold.tobytes(), 'P95/IDs inconsistentes')
    with np.load(directory / 'detector_validacion.npz', allow_pickle=False) as saved:
        require(saved['global_window_ids'].tolist() == job['validation_global_window_ids']
                and np.array_equal(saved['y_true'], expected_y)
                and np.array_equal(saved['rejected'], saved['scores'] > threshold)
                and detector_metrics(expected_y, saved['rejected'], expected_records) == result['detector_metrics'],
                'Metricas/decision detector inconsistentes')
    with np.load(directory / 'banco_costos_referencia.npz', allow_pickle=False) as saved:
        require(saved['global_window_ids'].tolist() == job['benchmark_global_window_ids'], 'Banco distinto')
    return result


def inspect_campaign(campaign):
    plan, jobs = validate_campaign(campaign)
    # Toda invocacion anterior debe haber cerrado sus estados y sello.
    batches = sorted((campaign / 'lotes').iterdir())
    consumed = 0.
    for batch in batches:
        verify_manifest(batch / 'MANIFEST_LOTE_SHA256.json')
        summary = read(batch / 'RESUMEN_LOTE.json')
        require(summary['status'] in ('BATCH_COMPLETED', 'CAMPAIGN_COMPLETED'),
                'Existe fallo/interrupcion: preservar y auditar antes de una recuperacion tecnica separada')
        consumed += summary['worker_elapsed_seconds']
    completed = []
    for job in jobs:
        directory = campaign / job['job_id']
        if directory.exists():
            verify_job(directory, job, plan['fit_signatures'][job['job_id']])
            completed.append(job['job_id'])
    require(completed == [j['job_id'] for j in jobs[:len(completed)]], 'Huecos en orden de ejecucion')
    return plan, jobs, completed, consumed


def run_one(campaign, batch, job, environment, timeout):
    job_id = job['job_id']
    started = time.perf_counter()
    process, code, failure = None, None, None
    try:
        with (batch / (job_id + '_stdout.log')).open('x', encoding='utf-8') as out, \
             (batch / (job_id + '_stderr.log')).open('x', encoding='utf-8') as err:
            process = subprocess.Popen([sys.executable, '-B', str(SOURCE / 'worker_representantes.py'),
                '--campaign', str(campaign), '--batch', str(batch), '--job-id', job_id],
                cwd=ROOT, env=environment, stdout=out, stderr=err)
            code = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        failure = 'TIMEOUT'
    except KeyboardInterrupt:
        failure = 'INTERRUPTED'
    except OSError as error:
        failure = 'LAUNCH_ERROR: ' + str(error)
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait(timeout=30)
            code = process.returncode
    result_error = None
    if failure is None and code == 0:
        try:
            plan = read(PREP / 'PLAN_EJECUCION.json')
            verify_job(campaign / job_id, job, plan['fit_signatures'][job_id])
        except Exception as error:
            result_error = type(error).__name__ + ': ' + str(error)
    passed = failure is None and code == 0 and result_error is None
    status = dict(job_id=job_id, status='PASS' if passed else 'FAIL', return_code=code,
                  failure=failure, result_error=result_error, automatic_retry=False,
                  worker_elapsed_seconds=time.perf_counter() - started, timeout_seconds=timeout)
    write(batch / ('ESTADO_' + job_id + '.json'), status)
    return status


def execute(campaign=None, limit=1):
    require(type(limit) is int and 1 <= limit <= 32, '--limit debe estar entre1 y32')
    plan, _ = preflight()
    require(versions() == plan['tested_versions'], 'Usar el entorno Python validado: ' + str(plan['tested_versions']))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    if campaign is None:
        # Evitar lanzar sin querer otra campana y repetir los32 ajustes.
        write(OUTPUT / 'CAMPANA_UNICA.json', dict(run_id=stamp, preparation_sha256=sha(PREP / 'MANIFEST_PREPARACION_SHA256.json')))
        campaign = OUTPUT / stamp
        campaign.mkdir()
        (campaign / 'lotes').mkdir()
        write(campaign / 'CONTRATO_CAMPANA.json', dict(preparation_manifest_sha256=sha(PREP / 'MANIFEST_PREPARACION_SHA256.json'),
            execution_owner='user_CLI', plan_sha256=sha(PREP / 'PLAN_EJECUCION.json'),
            scope=plan['scope'], maximum_fits=32, automatic_retry=False))
    campaign = campaign.resolve()
    # Lock con creacion exclusiva; se retira solo al terminar esta invocacion.
    lock = campaign / 'EJECUCION_EN_CURSO.lock'
    require(campaign.parent == OUTPUT.resolve(), 'Ruta de campana ajena')
    with lock.open('x', encoding='utf-8') as stream:
        stream.write(str(os.getpid()))
    try:
        plan, jobs, done, consumed = inspect_campaign(campaign)
        require(len(done) < 32, 'Campana ya completa; no repetir')
        selected = jobs[len(done):len(done) + limit]
        batch = campaign / 'lotes' / stamp
        batch.mkdir()
        write(batch / 'PLAN_LOTE.json', dict(job_ids=[j['job_id'] for j in selected],
            completed_before=done, consumed_worker_seconds_before=consumed, user_requested_limit=limit))
        print('Campana: ' + str(campaign), flush=True)
        environment = os.environ.copy()
        environment.update(ENVIRONMENT)
        statuses = []
        for job in selected:
            remaining = LIMITS['campaign_worker_seconds'] - consumed
            if remaining <= 0:
                break
            print(f"[{len(done)+len(statuses)+1}/32] INICIO {job['job_id']}; limite {min(300, remaining):.1f} s", flush=True)
            status = run_one(campaign, batch, job, environment, min(LIMITS['worker_seconds'], remaining))
            statuses.append(status)
            consumed += status['worker_elapsed_seconds']
            print(job['job_id'] + ': ' + status['status'], flush=True)
            if status['status'] != 'PASS':
                break
        passed = sum(s['status'] == 'PASS' for s in statuses)
        complete = len(done) + passed == 32
        success = passed == len(selected)
        summary = dict(status='CAMPAIGN_COMPLETED' if complete else 'BATCH_COMPLETED' if success else 'STOPPED_FOR_AUDIT',
            total_completed=len(done) + passed, total_planned=32, jobs_started=len(statuses), jobs_passed=passed,
            worker_elapsed_seconds=sum(s['worker_elapsed_seconds'] for s in statuses),
            campaign_worker_seconds_consumed=consumed, outer_evaluation_executed=False,
            remaining_grid_enabled=False, cost_selection_enabled=False, STM32_total_cost_measured=False)
        write(batch / 'RESUMEN_LOTE.json', summary)
        extra = [campaign / 'CONTRATO_CAMPANA.json', PREP / 'MANIFEST_PREPARACION_SHA256.json',
                 PREP / 'MANIFEST_PREPARACION_SHA256.sha256']
        for job in selected[:len(statuses)]:
            directory = campaign / job['job_id']
            if directory.is_dir():
                extra += [p for p in directory.iterdir() if p.is_file()]
        write_manifest(batch / 'MANIFEST_LOTE_SHA256.json', batch, extra)
        print(f"{summary['status']}: {summary['total_completed']}/32 representantes completos.", flush=True)
        print('Evidencias: ' + str(campaign), flush=True)
        return 0 if success else 1
    finally:
        lock.unlink()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--execute', action='store_true')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--limit', type=int, default=1)
    args = parser.parse_args()
    require(args.execute or (args.resume is None and args.limit == 1), '--resume/--limit requieren --execute')
    if args.prepare:
        prepare()
    elif args.execute:
        raise SystemExit(execute(args.resume, args.limit))
    elif args.resume is None:
        plan, _ = preflight()
        print(json.dumps(dict(status='READ_ONLY_PREFLIGHT_PASS', jobs=len(plan['jobs']), training_started=False,
                              execution_owner='user', default_limit=1, limits=LIMITS)))
