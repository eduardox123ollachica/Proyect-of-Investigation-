"""Ensayo real pequeño de workers con datos exclusivamente sintéticos.

Ejecutar explícitamente --run en el Python del usuario. Cada familia corre
en un proceso nuevo; solo en esa fixture se sustituye contrato.authorize.
No se modifica la autorización de producción, no se abren matrices CWRU y
no se comprueba rendimiento de generalización con estas muestras.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
SOURCE = Path(__file__).resolve().parent
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_npz(path, **arrays):
    import numpy as np
    with Path(path).open('xb') as stream:
        np.savez(stream, **arrays)


def kernel_fidelity():
    """Comparar AST de construcción y paso de actualización, no toda la función.

    La supresión de validación/ES es deliberada. La arquitectura, Adam y
    tf.function/GradientTape deben conservar literalmente el núcleo anterior.
    """
    original = ROOT / 'codigo/cwru/ampliacion_v2/representantes/worker_representantes.py'
    current = SOURCE / 'worker_dnn.py'

    def nodes(path, function_name):
        tree = ast.parse(path.read_text(encoding='utf-8'))
        function = next(node for node in tree.body
                        if isinstance(node, ast.FunctionDef) and node.name == function_name)
        selected = {}
        for node in function.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                if node.targets[0].id in ('dimension', 'penalty', 'model', 'optimizer'):
                    selected[node.targets[0].id] = ast.dump(node, include_attributes=False)
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                call = node.value
                if isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name):
                    if call.func.value.id == 'optimizer' and call.func.attr == 'build':
                        selected['optimizer.build'] = ast.dump(node, include_attributes=False)
            if isinstance(node, ast.FunctionDef) and node.name == 'step':
                selected['step'] = ast.dump(node, include_attributes=False)
        require(set(selected) == {'dimension', 'penalty', 'model', 'optimizer', 'optimizer.build', 'step'},
                'No se identificó todo el núcleo de actualización')
        return selected

    reference, fixed = nodes(original, 'fit_network'), nodes(current, 'fit_fixed_network')
    require(reference == fixed, 'El núcleo AST de actualizaciones difiere del original')
    digest = hashlib.sha256(json.dumps(fixed, sort_keys=True).encode('utf-8')).hexdigest()
    return dict(status='PASS', compared_components=sorted(fixed), canonical_AST_sha256=digest,
                original_source_sha256=sha(original), fixed_source_sha256=sha(current),
                validation_and_early_stopping_deliberately_removed=True)


@contextmanager
def reject_cwru_arrays():
    """Bloquear np.load de datos/ sin reemplazar arreglos con muestras reales."""
    import numpy as np
    original_load = np.load
    data_root = (ROOT / 'datos').resolve()
    accesses = dict(blocked_data_path_requests=0)

    def protected_load(file, *args, **kwargs):
        name = file if isinstance(file, (str, bytes, os.PathLike)) else getattr(file, 'name', None)
        if isinstance(name, (str, bytes, os.PathLike)):
            candidate = Path(os.fsdecode(name)).resolve()
            if candidate == data_root or candidate.is_relative_to(data_root):
                accesses['blocked_data_path_requests'] += 1
                raise ValueError('La prueba sintética prohíbe abrir arrays de datos/')
        return original_load(file, *args, **kwargs)

    with patch.object(np, 'load', protected_load):
        # Probar la guarda antes de cualquier worker, sin acceder al archivo.
        try:
            np.load(data_root / 'cwru' / '__synthetic_guard_probe__.npy')
        except ValueError:
            pass
        else:
            raise AssertionError('La guarda no bloqueó la carga CWRU')
        require(accesses['blocked_data_path_requests'] == 1, 'Sondeo de guarda inesperado')
        yield accesses
        require(accesses['blocked_data_path_requests'] == 1,
                'El flujo intentó abrir arrays CWRU después del sondeo')


def training_provenance(ids, records, loads):
    return dict(global_window_ids=list(ids), selected_loads_hp=list(loads),
        selected_record_ids=sorted(set(records), key=int), shape=[len(ids), 14],
        loaded_dtype='float32', source_dtype='synthetic_float32',
        only_requested_rows_selected_or_materialized=True, synthetic=True,
        source_arrays=[], raw_signals_or_MAT_opened=False)


def create_fixtures(base, protocol, scopes, sourcehashes, environment_versions, worker_environment):
    import numpy as np
    rng = np.random.default_rng(20261027)
    record_ids = np.asarray([str(1001 + index // 2) for index in range(42)], dtype=str)
    labels = np.asarray([(index // 2) % 3 for index in range(42)], dtype=np.int64)
    raw = rng.normal(0., .4, size=(42, 14)).astype(np.float32)
    offsets = np.linspace(.15, 1., 14, dtype=np.float32)
    raw = np.add(raw, np.multiply(labels[:, None].astype(np.float32), offsets, dtype=np.float32), dtype=np.float32)
    raw[:, 13] = np.float32(1.25)  # La regla de varianza cero también se ejercita.
    ids = list(range(42))
    require(np.bincount(labels, minlength=3).tolist() == [14, 14, 14], 'Fixture no balanceada')
    require(len(set(record_ids.tolist())) == 21, 'Fixture requiere 21 MAT sintéticos')
    common_arrays = dict(raw_training=raw, y_training=labels,
                         training_ids=np.asarray(ids, dtype=np.int64), training_records=record_ids)
    jobs = dict(DNN=copy.deepcopy(scopes['DNN_unique_jobs'][0]),
                RF=copy.deepcopy(scopes['RF_jobs'][0]), SVM=copy.deepcopy(scopes['SVM_jobs'][0]),
                detector=copy.deepcopy(scopes['detector_unique_scopes'][0]))
    fixtures = {}
    for family, job in jobs.items():
        folder = base / family
        folder.mkdir(exist_ok=False)
        job['family'] = family
        job['job_id'] = 'synthetic_' + family
        job['training_global_window_ids'] = ids.copy()
        if family != 'detector':
            job['feature_indices'] = (protocol['candidates'][job['candidate']].copy()
                                      if family == 'DNN' else list(range(14)))
            job['input_dimension'] = len(job['feature_indices'])
            if family == 'DNN':
                job['epochs'] = 2
            arrays = common_arrays
            provenance = dict(training=training_provenance(ids, record_ids.tolist(), job['roles']['training_loads']))
        else:
            # Dos MAT normales sintéticos para fit, uno distinto para P95.
            fit_ids, calibration_ids = [0, 1, 18, 19], [30, 31]
            require((labels[fit_ids + calibration_ids] == 0).all(), 'Fixture del detector contiene fallas')
            job['fit_ids'], job['calibration_ids'] = fit_ids, calibration_ids
            arrays = dict(raw_fit=raw[fit_ids], raw_calibration=raw[calibration_ids],
                y_fit=labels[fit_ids], y_calibration=labels[calibration_ids],
                fit_ids=np.asarray(fit_ids, dtype=np.int64), calibration_ids=np.asarray(calibration_ids, dtype=np.int64),
                fit_records=record_ids[fit_ids], calibration_records=record_ids[calibration_ids])
            provenance = dict(fit=training_provenance(fit_ids, record_ids[fit_ids].tolist(),
                job['roles']['detector_fit_normal_loads']),
                calibration=training_provenance(calibration_ids, record_ids[calibration_ids].tolist(),
                job['roles']['detector_calibration_normal_loads']))
        array_path = folder / 'datos_sinteticos.npz'
        save_npz(array_path, **arrays)
        payload = dict(status='AUTHORIZED_FINAL_REFIT_PAYLOAD', family=family, job=job,
            arrays_path=str(array_path.resolve()), arrays_sha256=sha(array_path),
            fit_signature=hashlib.sha256(('synthetic_' + family).encode('ascii')).hexdigest(),
            selection_signature='synthetic_only_not_a_production_authorization',
            source_sha256=sourcehashes, expected_versions=environment_versions,
            provenance=provenance, worker_environment=dict(worker_environment),
            **{key: protocol[key] for key in ('dnn', 'numeric_contract', 'quantization', 'detector')})
        path = folder / 'payload.json'
        write(path, payload)
        fixtures[family] = dict(payload=path, output=folder / 'salida', folder=folder)
    return fixtures


def verify_classical(family, output, arrays):
    import joblib
    import numpy as np
    from numerica_representantes import normalize_f32
    result = read(output / 'RESULTADO.json')
    model = joblib.load(output / 'modelo.joblib')  # Archivo recién generado por esta fixture.
    x = arrays['raw_training']
    if family == 'SVM':
        with np.load(output / 'scaler_train_only.npz', allow_pickle=False) as scaler:
            x = normalize_f32(x, scaler['mean32'], scaler['scale32'])
            require(float(scaler['scale32'][13]) == 1., 'Scaler no conserva regla de varianza cero')
    with np.load(output / 'diagnostico_entrenamiento.npz', allow_pickle=False) as saved:
        actual = model.predict(x)
        require(np.array_equal(actual, saved['predictions']), 'Predicción nativa recargada distinta')
        require(np.array_equal(saved['global_window_ids'], arrays['training_ids']), 'IDs de diagnóstico distintos')
        expected_values = model.predict_proba(x) if family == 'RF' else model.decision_function(x)
        require(np.array_equal(expected_values, saved['values']), 'Valores nativos recargados distintos')
    require(result['status'] == 'PASS' and result['outer_evaluation_executed'] is False
            and result['model_sha256'] == sha(output / 'modelo.joblib'), 'Resultado clásico inválido')
    return dict(status='PASS', native_reload_predictions_exact=True, native_values_exact=True,
                model_sha256=result['model_sha256'], TensorFlow_imported='tensorflow' in sys.modules,
                synthetic_model_fits=1)


def verify_detector(output, arrays):
    import numpy as np
    from detector_numerico import detector_score
    import worker_detector
    with np.load(output / 'parametros.npz', allow_pickle=False) as archive:
        params = {key: archive[key].copy() for key in archive.files}
    params['indices'] = params['indices'].tolist()
    params['threshold32'] = np.float32(params['threshold32'].item())
    actual = detector_score(arrays['raw_calibration'], params)
    with np.load(output / 'calibracion.npz', allow_pickle=False) as saved:
        require(np.array_equal(actual.view(np.uint32), saved['scores'].view(np.uint32)),
                'Scores float32 recargados distintos')
        require(np.array_equal(actual > params['threshold32'], saved['rejected']),
                'Rechazos recargados distintos')
    # Una sola MAT de calibración: P95 observado de dos filas es el máximo.
    require(params['threshold32'].view(np.uint32) == np.max(actual).view(np.uint32),
            'P95 sintético de dos filas no corresponde al observado')
    result = read(output / 'RESULTADO.json')
    require(result['status'] == 'PASS' and result['outer_evaluation_executed'] is False
            and result['parameter_payload_sha256'] == worker_detector.parameter_payload_sha256(output / 'parametros.npz'),
            'Parámetros recargados distintos')
    return dict(status='PASS', float32_reload_scores_exact=True, threshold_observed_P95_verified=True,
                synthetic_parameter_fits=1, refit_after_calibration=False,
                TensorFlow_imported='tensorflow' in sys.modules)


def verify_dnn(output, arrays, payload, family_folder):
    import numpy as np
    import tensorflow as tf
    from tensorflow import keras
    import worker_dnn
    from numerica_representantes import normalize_f32
    result = read(output / 'RESULTADO.json')
    require(result['status'] == 'PASS' and result['epochs_executed'] == result['fixed_epochs'] == 2
            and result['optimizer_updates'] == 4 and result['early_stopping_used'] is False
            and result['validation_used'] is False and result['outer_evaluation_executed'] is False,
            'DNN no completó exactamente dos épocas y cuatro actualizaciones')
    for filename in ('modelo.keras', 'modelo_fp32.tflite', 'modelo_int8.tflite',
                     'checkpoint_antes_conversion.npz', 'estado_antes_conversion.json'):
        require((output / filename).is_file(), 'Artefacto DNN ausente: ' + filename)
    for key, filename in [('Keras', 'modelo.keras'), ('FP32', 'modelo_fp32.tflite'), ('INT8', 'modelo_int8.tflite')]:
        require(result['model_sha256'][key] == sha(output / filename), 'Hash de modelo DNN distinto')
    indices = payload['job']['feature_indices']
    with np.load(output / 'scaler_train_only.npz', allow_pickle=False) as saved:
        mean, scale = saved['mean32'].copy(), saved['scale32'].copy()
        require(float(scale[indices.index(13)]) == 1., 'Scaler DNN no respeta varianza cero')
    x = normalize_f32(arrays['raw_training'][:, indices], mean, scale)
    restored = keras.models.load_model(output / 'modelo.keras', compile=False)
    require(worker_dnn.weights_hash(restored) == result['final_weights_sha256'], 'Pesos Keras recargados distintos')
    probabilities = np.asarray([restored(row[None, :], training=False).numpy()[0] for row in x], dtype=np.float32)
    with np.load(output / 'diagnosticos_train_only.npz', allow_pickle=False) as saved:
        require(np.allclose(probabilities, saved['keras_batch1'], rtol=1e-6, atol=1e-7)
                and np.array_equal(probabilities.argmax(axis=1), saved['keras_batch1'].argmax(axis=1)),
                'Keras recargado no reproduce las predicciones de entrenamiento')
        maximum_error = float(np.max(np.abs(probabilities - saved['keras_batch1'])))
    # Repetir únicamente el ajuste de dos épocas. No se vuelve a ajustar scaler,
    # convertir modelos o usar filas CWRU. Reiniciar estado Keras y las RNG.
    keras.backend.clear_session()
    keras.utils.set_random_seed(payload['job']['seed'])
    repetition = family_folder / 'repeticion_fija'
    repetition.mkdir(exist_ok=False)
    replica, state = worker_dnn.fit_fixed_network(tf, keras, x, arrays['y_training'], payload['dnn'],
        payload['job']['seed'], 2, repetition, lambda *args, **kwargs: None)
    require(state['epochs_executed'] == 2 and state['optimizer_updates'] == 4
            and worker_dnn.weights_hash(replica) == result['final_weights_sha256'],
            'Repetición de dos épocas no es determinista bit a bit')
    return dict(status='PASS', fixed_epochs=2, optimizer_updates=4,
                full_INT8_conversion_verified=True, training_calibration_rows=42,
                native_Keras_reload_weights_exact=True, native_Keras_reload_max_abs_error=maximum_error,
                repeated_fit_weights_exact=True, final_weights_sha256=result['final_weights_sha256'],
                synthetic_model_fits=2, validation_used=False, TensorFlow_imported=True)


def attempt_paths(contract, technical_attempt):
    require(type(technical_attempt) is int and technical_attempt in (1, 2), 'Intento técnico no autorizado')
    if technical_attempt == 1:
        return contract.PREP / 'prueba_sintetica_real', contract.PREP / 'PRUEBA_SINTETICA_REAL.json'
    return contract.PREP / 'prueba_sintetica_real_intento2', contract.PREP / 'PRUEBA_SINTETICA_REAL_INTENTO2.json'


def previous_attempt(contract):
    """Documentar el fallo técnico sin editar ni reutilizar sus modelos."""
    folder, report_path = attempt_paths(contract, 1)
    original = read(report_path)
    require(original['status'] == 'FAILED_PRESERVED_DO_NOT_RETRY'
            and 'TimeoutExpired' in original['failure'], 'El intento 2 requiere el timeout técnico documentado')
    check_path = folder / 'RF' / 'COMPROBACION.json'
    result_path = folder / 'RF' / 'salida' / 'RESULTADO.json'
    check, result = read(check_path), read(result_path)
    require(check['status'] == check['result_status'] == result['status'] == 'PASS'
            and check['synthetic_model_fits'] == 1 and result['family'] == 'RF'
            and result['outer_evaluation_executed'] is False,
            'No se verificó el RF sintético ya completado antes del timeout')
    contract.io.verify_manifest(folder / 'RF' / 'salida' / 'MANIFEST_JOB_SHA256.json')
    snapshot = {str(path.relative_to(ROOT).as_posix()): sha(path)
                for path in sorted([report_path, *[p for p in folder.rglob('*') if p.is_file()]])}
    return dict(attempt=1, report_path=report_path.relative_to(ROOT).as_posix(),
        report_sha256=sha(report_path), status=original['status'],
        reason='RF child exceeded 30 s including cold imports and final replay; RF fit and checks had completed',
        RF_worker_elapsed_seconds=result['worker_elapsed_seconds'], RF_fit_seconds=result['fit_seconds'],
        completed_verified_synthetic_fits=dict(DNN=0, RF=1, SVM=0, detector=0),
        original_report_recorded_synthetic_fits=original['synthetic_fits'],
        original_RF_count_was_zero_because_summary_collection_followed_subprocess_return=True,
        CWRU_arrays_opened=False, CWRU_model_fits=0, evidence_sha256=snapshot,
        reuse_of_prior_synthetic_models=False, automatic_retry=False)


def child(family, payload_path, output, technical_attempt=1):
    import contrato as contract
    require(family in ('DNN', 'RF', 'SVM', 'detector'), 'Familia sintética desconocida')
    base = attempt_paths(contract, technical_attempt)[0].resolve()
    payload_path, output = Path(payload_path).resolve(), Path(output).resolve()
    require(payload_path == base / family / 'payload.json' and output == base / family / 'salida'
            and not output.exists(), 'Fixture de prueba ajena o ya existente')
    payload = read(payload_path)
    require(payload['family'] == family and payload['provenance'].get('training', payload['provenance'].get('fit'))['synthetic'] is True,
            'Solo se permite autorizar payload sintético')
    require(payload['expected_versions'] == contract.versions(), 'Entorno de prueba distinto')
    require(all(os.environ.get(key) == value for key, value in contract.io.ENVIRONMENT.items()), 'Entorno no determinista')
    require(payload['source_sha256'] == contract.sources(), 'Fuentes cambiaron durante la prueba')
    require('tensorflow' not in sys.modules, 'El hijo debe empezar sin TensorFlow')
    array_path = Path(payload['arrays_path']).resolve()
    require(array_path == payload_path.parent / 'datos_sinteticos.npz'
            and sha(array_path) == payload['arrays_sha256'], 'Fixture de arrays alterada')

    def synthetic_authorize(current_payload, current_output, current_family):
        require(Path(current_payload).resolve() == payload_path and Path(current_output).resolve() == output
                and current_family == family and not output.exists(), 'Autorizar solo esta fixture una vez')
        return read(payload_path)

    import numpy as np
    with reject_cwru_arrays() as access, patch.object(contract, 'authorize', synthetic_authorize):
        with np.load(array_path, allow_pickle=False) as archive:
            arrays = {key: archive[key].copy() for key in archive.files}
        if family == 'DNN':
            import worker_dnn
            worker_dnn.run(payload_path, output)
            verification = verify_dnn(output, arrays, payload, payload_path.parent)
        elif family in ('RF', 'SVM'):
            import worker_clasicos
            worker_clasicos.run(payload_path, output)
            verification = verify_classical(family, output, arrays)
        else:
            import worker_detector
            worker_detector.run(payload_path, output)
            verification = verify_detector(output, arrays)
        entries = contract.io.verify_manifest(output / 'MANIFEST_JOB_SHA256.json')
    require(contract.authorize is not synthetic_authorize, 'Parche de autorización no restaurado')
    verification.update(family=family, model_artifact_manifest_entries=len(entries),
        result_status=read(output / 'RESULTADO.json')['status'],
        authorization_patch_scope='this_child_synthetic_fixture_only', production_authorization_modified=False,
        CWRU_arrays_opened=False, blocked_CWRU_guard_probe_requests=access['blocked_data_path_requests'],
        outer_evaluation_executed=False)
    write(payload_path.parent / 'COMPROBACION.json', verification)
    print(json.dumps(dict(status='SYNTHETIC_FAMILY_PASS', family=family), ensure_ascii=True), flush=True)


def run(technical_attempt=1):
    import contrato as contract
    os.environ.update(contract.io.ENVIRONMENT)
    require('tensorflow' not in sys.modules, 'El coordinador sintético no debe importar TensorFlow')
    target, report_path = attempt_paths(contract, technical_attempt)
    require(not target.exists() and not report_path.exists(), 'Prueba existente; no sobrescribir ni repetir automáticamente')
    prior = previous_attempt(contract) if technical_attempt == 2 else None
    source_before, started = contract.sources(), time.perf_counter()
    expected_versions = contract.versions()
    protocol = read(contract.io.PROTOCOL / 'protocolo_reglas.json')
    scopes = read(contract.FROZEN / 'PLAN_REAJUSTES_PENDIENTES.json')
    fidelity = kernel_fidelity()
    contract.PREP.mkdir(parents=True, exist_ok=True)
    target.mkdir(exist_ok=False)
    summaries, executions, failure = {}, {}, None
    try:
        with reject_cwru_arrays() as parent_access:
            fixtures = create_fixtures(target, protocol, scopes, source_before, expected_versions,
                                       contract.io.ENVIRONMENT)
            for family in ('RF', 'SVM', 'detector', 'DNN'):
                fixture = fixtures[family]
                timeout = contract.TIMEOUTS[family]
                command = [sys.executable, '-B', str(Path(__file__).resolve()), '--child', family,
                           '--technical-attempt', str(technical_attempt),
                           '--payload', str(fixture['payload']), '--output', str(fixture['output'])]
                checkpoint = time.perf_counter()
                # Abrir ambos logs antes del proceso: un timeout conserva todo
                # lo emitido, aunque ya exista un RESULTADO/COMPROBACION PASS.
                process, timed_out, process_error = None, False, None
                with (fixture['folder'] / 'stdout.log').open('xb') as stdout, \
                     (fixture['folder'] / 'stderr.log').open('xb') as stderr:
                    try:
                        process = subprocess.run(command, cwd=ROOT, env=dict(os.environ),
                            stdout=stdout, stderr=stderr, timeout=timeout)
                    except subprocess.TimeoutExpired as error:
                        timed_out, process_error = True, repr(error)
                    except Exception as error:
                        process_error = repr(error)
                executions[family] = dict(command=command, timeout_seconds=timeout,
                    elapsed_seconds=time.perf_counter() - checkpoint,
                    returncode=process.returncode if process is not None else None,
                    timed_out=timed_out, process_error=process_error,
                    logs_opened_before_launch=True,
                    subprocess_run_terminates_child_on_timeout=timed_out)
                check_path = fixture['folder'] / 'COMPROBACION.json'
                if check_path.is_file():
                    summaries[family] = read(check_path)
                require(process is not None and process.returncode == 0 and not timed_out,
                        'Falló familia sintética ' + family + '; consultar logs y artefactos conservados')
                require(family in summaries, 'El hijo no persistió su comprobación')
                require(summaries[family]['status'] == 'PASS', 'Comprobación de familia no aprobada')
                print(f'SYNTHETIC_{family}: PASS', flush=True)
        require(parent_access['blocked_data_path_requests'] == 1, 'Acceso CWRU en coordinador')
        require(contract.sources() == source_before, 'Fuentes cambiaron durante el ensayo')
    except Exception:
        failure = traceback.format_exc()
    sources_after = contract.sources()
    # Contar comprobaciones persistidas incluso si el hijo agotó su timeout
    # después de cerrarlas. No inferir cero ajustes de una excepción del padre.
    for family in ('RF', 'SVM', 'detector', 'DNN'):
        check_path = target / family / 'COMPROBACION.json'
        if check_path.is_file():
            summaries[family] = read(check_path)
    if prior is not None:
        require(all(sha(ROOT / path) == digest for path, digest in prior['evidence_sha256'].items()),
                'Se alteraron artefactos del primer intento')
    passed = failure is None and len(summaries) == 4 and sources_after == source_before
    report = dict(status='PASS' if passed else 'FAILED_PRESERVED_DO_NOT_RETRY',
        technical_attempt=technical_attempt, prior_technical_attempt=prior,
        prior_attempt_artifacts_preserved=prior is not None,
        versions=expected_versions, source_sha256=source_before,
        source_sha256_before=source_before, source_sha256_after=sources_after,
        runtime_seconds=time.perf_counter() - started, synthetic_shape=[42, 14],
        synthetic_records=21, synthetic_class_counts=[14, 14, 14], data_generation_seed=20261027,
        family_results=summaries, subprocess_executions=executions, kernel_fidelity=fidelity,
        synthetic_fits={family: (summaries[family].get('synthetic_model_fits',
            summaries[family].get('synthetic_parameter_fits', 0)) if family in summaries
            and summaries[family].get('status') == 'PASS' else 0)
            for family in ('DNN', 'RF', 'SVM', 'detector')},
        synthetic_fit_counts_are_completed_verified_checks=True,
        unverified_partial_fits_may_exist_on_failed_attempt=not passed,
        DNN_replica_included_in_synthetic_fit_count=True, CWRU_arrays_opened=False,
        CWRU_model_fits=0, CWRU_model_predictions=0, outer_evaluation_executed=False,
        production_authorization_modified=False, production_preparation_manifest_required=False,
        fixture_authorization_patch_only=True, automatic_retry=False,
        TensorFlow_imported_in_coordinator='tensorflow' in sys.modules,
        hardware_connected=False, manuscript_modified=False, failure=failure)
    write(report_path, report)
    print(json.dumps(dict(status=report['status'], report=str(report_path), report_sha256=sha(report_path))), flush=True)
    if not passed:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--technical-attempt', type=int, choices=(1, 2), default=1,
                        help='2 solo como nuevo ensayo técnico explícito tras el timeout RF documentado')
    parser.add_argument('--child', choices=('DNN', 'RF', 'SVM', 'detector'), help=argparse.SUPPRESS)
    parser.add_argument('--payload', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--output', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child:
        require(not args.run and args.payload is not None and args.output is not None,
                'Hijo sintético requiere payload/output y no --run')
        child(args.child, args.payload, args.output, args.technical_attempt)
    else:
        require(args.run and args.payload is None and args.output is None,
                'Usar --run explícito para iniciar el ensayo sintético')
        run(args.technical_attempt)


if __name__ == '__main__':
    main()
