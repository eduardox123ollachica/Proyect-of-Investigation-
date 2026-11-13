"""Orquestador limitado a F14/T6; solo --execute inicia dos workers secuenciales."""
from __future__ import annotations
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
SOURCE = Path(__file__).resolve().parent
PREPARATION = ROOT / 'articulo/ampliacion_cwru/metodologia/piloto_tiempo_exportacion_v2_2026-10-25'
OUTPUT = ROOT / 'resultados/cwru/ampliacion_v2/piloto_tiempo_exportacion_v2_2026-10-25'
WORKER = SOURCE / 'worker_piloto_tiempo_v2.py'
ENVIRONMENT = dict(PYTHONHASHSEED='11', PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='-1',
    TF_DETERMINISTIC_OPS='1', TF_ENABLE_ONEDNN_OPTS='0', TF_NUM_INTRAOP_THREADS='1',
    TF_NUM_INTEROP_THREADS='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
    NUMEXPR_NUM_THREADS='1', TF_CPP_MIN_LOG_LEVEL='2')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def read_worker_result(path):
    if not path.exists():
        return None, 'resultado.json ausente'
    try:
        result = read(path)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        return None, type(error).__name__ + ': ' + str(error)
    if not isinstance(result, dict) or result.get('status') != 'PASS':
        return None, 'resultado.json no contiene un resultado PASS valido'
    return result, None


def write(path, payload):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def verify_preparation():
    manifest = PREPARATION / 'MANIFEST_PREPARACION_PILOTO_SHA256.json'
    digest, filename = manifest.with_suffix('.sha256').read_text(encoding='ascii').split()
    require(filename == manifest.name and digest == sha(manifest), 'Sello de preparacion distinto')
    entries = read(manifest)
    for name, value in entries.items():
        path = (ROOT / name).resolve()
        require(path.is_relative_to(ROOT) and sha(path) == value, 'Preparacion modificada: ' + name)
    contract = read(PREPARATION / 'EJECUCION_PILOTO_ACOTADO.json')
    require(contract['scope'] == 'two_runtime_pilot_jobs_only'
            and contract['pilot_execution_enabled'] is True
            and contract['mass_training_enabled'] is False
            and contract['outer_predictive_evaluation_enabled'] is False
            and contract['maximum_DNN_fits'] == 2 and contract['timeout_per_worker_seconds'] == 300,
            'Limites de ejecucion incompatibles')
    from datos_piloto import load_verified_plan
    plan = load_verified_plan()
    require(contract['source_plan_sha256'] == sha(ROOT / contract['source_plan'])
            and contract['job_ids'] == [j['job_id'] for j in plan['jobs']], 'Plan fuente distinto')
    return plan, contract, entries


def prepare():
    from datos_piloto import load_verified_plan, PROTOCOL
    plan = load_verified_plan()
    require((PREPARATION / 'README.md').is_file(), 'Falta descripcion de alcance')
    tests = read(PREPARATION / 'PRUEBAS_PREPARACION.json')
    require(tests['status'] == 'PASS' and tests['failures'] == tests['errors'] == 0
            and tests['tests_run'] >= 22 and tests['source_AST_passed'] is True,
            'Faltan pruebas puras y revision sintactica aprobadas')
    sources = sorted(SOURCE.glob('*.py'))
    require(all(tests['source_sha256'].get(p.relative_to(ROOT).as_posix()) == sha(p) for p in sources),
            'Fuentes distintas de la preparacion probada')
    contract = dict(scope='two_runtime_pilot_jobs_only', source_plan=(PROTOCOL / 'piloto_plan.json').relative_to(ROOT).as_posix(),
        source_plan_sha256=sha(PROTOCOL / 'piloto_plan.json'),
        historical_plan_pilot_execution_enabled=False, historical_snapshot_unmodified=True, pilot_execution_enabled=True,
        execution_stage_basis='User requested continuing with the bounded runtime pilot on 2026-10-25.',
        job_ids=[j['job_id'] for j in plan['jobs']], maximum_DNN_fits=2,
        maximum_FP32_plus_INT8_conversions=4, timeout_per_worker_seconds=300,
        maximum_worker_seconds_total=600, sequential=True, automatic_retry=False,
        mass_training_enabled=False, outer_predictive_evaluation_enabled=False,
        manuscript_edits_enabled=False, pilot_models_reused_as_confirmatory_results=False)
    contract.update(technical_revision='freeze_variable_values_before_FP32_and_INT8_conversion',
        previous_attempt='resultados/cwru/ampliacion_v2/piloto_tiempo_2026-10-25/20261025T063025_963041Z',
        previous_attempt_preserved=True, previous_attempt_trained_F14_but_failed_INT8=True,
        extra_technical_F14_training_repeat=1,
        reason_for_repeat='The first failed worker did not persist weights before conversion; repeat is a technical recovery, not model selection.',
        cumulative_training_attempts_limit_including_failed_attempt=3,
        new_checkpoint_saved_before_conversion=True)
    contract_path = PREPARATION / 'EJECUCION_PILOTO_ACOTADO.json'
    manifest = PREPARATION / 'MANIFEST_PREPARACION_PILOTO_SHA256.json'
    require(not contract_path.exists() and not manifest.exists() and not manifest.with_suffix('.sha256').exists(),
            'No sobrescribir preparacion existente')
    write(contract_path, contract)
    paths = sources + [p for p in PREPARATION.iterdir() if p.is_file()]
    paths.extend([PROTOCOL / 'MANIFEST_PROTOCOLO_SHA256.json', PROTOCOL / 'MANIFEST_PROTOCOLO_SHA256.sha256'])
    previous = ROOT / contract['previous_attempt'] / 'MANIFEST_RESULTADOS_PILOTO_SHA256.json'
    old_digest, old_name = previous.with_suffix('.sha256').read_text(encoding='ascii').split()
    require(old_name == previous.name and old_digest == sha(previous), 'Sello del fallo previo distinto')
    require(all(sha(ROOT / name) == digest for name, digest in read(previous).items()),
            'Recurso del intento previo modificado')
    paths.extend([previous, previous.with_suffix('.sha256')])
    entries = {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(set(paths))}
    write(manifest, entries)
    with manifest.with_suffix('.sha256').open('x', encoding='ascii') as stream:
        stream.write(sha(manifest) + '  ' + manifest.name + '\n')
    verify_preparation()
    print(json.dumps(dict(status='BOUNDED_PILOT_PREPARED', sealed_resources=len(entries), models_trained=0)))


def execute():
    started = time.perf_counter()
    plan, contract, preparation = verify_preparation()
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    run_dir = OUTPUT / stamp
    write(OUTPUT / 'EJECUCION_UNICA.json', dict(run_id=stamp,
        preparation_manifest_sha256=sha(PREPARATION / 'MANIFEST_PREPARACION_PILOTO_SHA256.json'),
        prevents_automatic_or_accidental_repeat=True))
    run_dir.mkdir(exist_ok=False)
    source_hashes = {name: digest for name, digest in preparation.items()
                     if name.startswith('codigo/cwru/ampliacion_v2/piloto/')}
    authorization = dict(contract, source_sha256=source_hashes, worker_environment=ENVIRONMENT,
                         preparation_manifest_sha256=sha(PREPARATION / 'MANIFEST_PREPARACION_PILOTO_SHA256.json'),
                         run_id=stamp, python_executable=sys.executable, automatic_retry=False)
    write(run_dir / 'EJECUCION_ACOTADA.json', authorization)
    environment = os.environ.copy()
    environment.update(ENVIRONMENT)
    statuses, results = [], []
    for job in plan['jobs']:
        job_id = job['job_id']
        worker_started = time.perf_counter()
        print('INICIO ' + job_id + '; limite 300 s', flush=True)
        command = [sys.executable, '-B', str(WORKER), '--job-id', job_id, '--run-dir', str(run_dir)]
        timed_out, launch_error, return_code = False, None, None
        with (run_dir / (job_id + '_stdout.log')).open('x', encoding='utf-8') as out, \
             (run_dir / (job_id + '_stderr.log')).open('x', encoding='utf-8') as err:
            try:
                process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=out, stderr=err)
                try:
                    return_code = process.wait(timeout=300)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    process.kill()
                    process.wait(timeout=30)
                    return_code = process.returncode
            except OSError as error:
                launch_error = type(error).__name__ + ': ' + str(error)
        elapsed = time.perf_counter() - worker_started
        result_path = run_dir / job_id / 'resultado.json'
        result, result_error = read_worker_result(result_path)
        passed = not timed_out and return_code == 0 and result is not None and result['status'] == 'PASS'
        status = dict(job_id=job_id, status='PASS' if passed else 'TIMEOUT' if timed_out else 'FAIL',
                      return_code=return_code, elapsed_process_seconds=elapsed,
                      timeout_seconds=300, automatic_retry=False,
                      process_launch_error=launch_error, result_read_error=result_error)
        statuses.append(status)
        write(run_dir / ('ESTADO_' + job['mask'] + '.json'), status)
        print(json.dumps(status), flush=True)
        if not passed:
            break
        results.append(result)
    complete = len(results) == 2
    projection_error = None
    if complete:
        try:
            from presupuesto_piloto import project_dnn_budget
            projection = project_dnn_budget(results)
            process_seconds = {r['mask']: s['elapsed_process_seconds'] for r, s in zip(results, statuses)}
            gaps = {r['mask']: max(0., process_seconds[r['mask']] - sum(r['timing_nonoverlapping_seconds'].values()))
                    for r in results}
            projection['observed_process_wall_seconds_by_mask'] = process_seconds
            projection['observed_process_minus_stages_seconds_by_mask'] = gaps
            projection['process_gap_interpretation'] = 'Observed startup, unattributed work, final I/O and exit; copied per fit only as an engineering scenario.'
            for scenario in projection['scenarios'].values():
                scenario['additional_observed_process_gap_scenario_seconds'] = 408 * max(gaps.values())
                scenario['max_profile_plus_observed_gap_scenario_seconds'] = (
                    scenario['conservative_max_seconds'] + scenario['additional_observed_process_gap_scenario_seconds'])
            write(run_dir / 'PROYECCION_PRESUPUESTO_DNN.json', projection)
        except (ValueError, OSError, KeyError) as error:
            complete = False
            projection_error = type(error).__name__ + ': ' + str(error)
    summary = dict(status='PILOT_COMPLETED' if complete else
                   'PILOT_OUTPUT_VALIDATION_FAILED' if projection_error else 'PILOT_STOPPED_ON_FIRST_FAILURE',
        purpose='runtime_and_conversion_only', jobs_requested=2, jobs_started=len(statuses),
        jobs_passed=len(results), statuses=statuses,
        total_elapsed_orchestrator_seconds=time.perf_counter() - started,
        observed_worker_process_seconds=sum(s['elapsed_process_seconds'] for s in statuses),
        maximum_worker_seconds_total=600, models_not_reused_for_selection=True,
        load3_feature_arrays_opened=False, mass_training_started=False,
        outer_predictive_evaluation_executed=False, full_runtime_budget_frozen=False,
        classical_runtime_pilot_pending=True, STM32_full_pipeline_cost_pending=True)
    summary['projection_error'] = projection_error
    write(run_dir / 'RESUMEN_PILOTO.json', summary)
    resources = {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(run_dir.rglob('*')) if p.is_file()}
    resources.update(preparation)
    manifest = run_dir / 'MANIFEST_RESULTADOS_PILOTO_SHA256.json'
    write(manifest, resources)
    with manifest.with_suffix('.sha256').open('x', encoding='ascii') as stream:
        stream.write(sha(manifest) + '  ' + manifest.name + '\n')
    require(all(sha(ROOT / name) == digest for name, digest in resources.items()), 'Recurso distinto al cerrar intento')
    print(json.dumps(dict(summary, evidence_directory=str(run_dir)), ensure_ascii=True), flush=True)
    return 0 if complete else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--execute', action='store_true')
    mode.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    if args.execute:
        raise SystemExit(execute())
    if args.prepare:
        prepare()
        raise SystemExit(0)
    from datos_piloto import load_verified_plan
    prepared = load_verified_plan()
    preparation_verified = False
    if (PREPARATION / 'MANIFEST_PREPARACION_PILOTO_SHA256.json').exists():
        verify_preparation()
        preparation_verified = True
    print(json.dumps(dict(status='READ_ONLY_PLAN_CHECK', jobs=[j['job_id'] for j in prepared['jobs']],
                          preparation_verified=preparation_verified, models_trained=0, execute_required=True)))
