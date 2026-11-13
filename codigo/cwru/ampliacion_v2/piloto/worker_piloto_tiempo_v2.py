"""Dos trabajos tecnicos acotados; no es un entrenador de busqueda V2."""
from __future__ import annotations

import time
WORKER_STARTED = time.perf_counter()
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import sys
import traceback

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / 'resultados/cwru/ampliacion_v2/piloto_tiempo_exportacion_v2_2026-10-25'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, payload):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def run(job_id, run_dir):
    run_dir = run_dir.resolve()
    require(run_dir.parent == OUTPUT.resolve(), 'Directorio de ejecucion ajeno al piloto')
    authorization = json.loads((run_dir / 'EJECUCION_ACOTADA.json').read_text(encoding='utf-8'))
    require(authorization['scope'] == 'two_runtime_pilot_jobs_only'
            and authorization['mass_training_enabled'] is False
            and authorization['outer_predictive_evaluation_enabled'] is False
            and authorization['source_sha256'][Path(__file__).relative_to(ROOT).as_posix()] == sha(Path(__file__)),
            'Contrato de ejecucion o fuente distintos')
    require(job_id in authorization['job_ids'] and len(authorization['job_ids']) == 2,
            'Trabajo fuera del piloto')
    for name, digest in authorization['source_sha256'].items():
        path = (ROOT / name).resolve()
        require(path.is_relative_to(ROOT) and sha(path) == digest, 'Fuente modificada antes del trabajo')
    for key, value in authorization['worker_environment'].items():
        require(os.environ.get(key) == value, 'Variable de ejecucion no fijada: ' + key)
    directory = run_dir / job_id
    directory.mkdir(exist_ok=False)
    times = {name: 0. for name in ('imports', 'data_and_scaler', 'graph_compile',
             'training_updates', 'epoch_training_diagnostics', 'epoch_validation',
             'FP32_conversion', 'INT8_conversion', 'inference', 'artifact_IO')}
    stages = directory / 'progreso.jsonl'

    def event(stage, **extra):
        started = time.perf_counter()
        row = dict(stage=stage, elapsed_seconds=time.perf_counter() - WORKER_STARTED, **extra)
        with stages.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
        print(json.dumps(row), flush=True)
        times['artifact_IO'] += time.perf_counter() - started

    def save(callback):
        started = time.perf_counter()
        callback()
        times['artifact_IO'] += time.perf_counter() - started

    event('worker_started')
    started = time.perf_counter()
    import numpy as np
    import sklearn
    from sklearn.preprocessing import StandardScaler
    import tensorflow as tf
    from tensorflow import keras
    from datos_piloto import load_verified_plan, load_job_data
    times['imports'] = time.perf_counter() - started
    event('imports_complete', tensorflow_version=tf.__version__)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.set_visible_devices([], 'GPU')
    random.seed(11)
    np.random.seed(11)
    tf.keras.utils.set_random_seed(11)
    tf.config.experimental.enable_op_determinism()

    started = time.perf_counter()
    plan = load_verified_plan()
    job = next(j for j in plan['jobs'] if j['job_id'] == job_id)
    raw_train, y_train, raw_val, y_val, provenance = load_job_data(job)
    indices = job['feature_indices']
    raw_train, raw_val = raw_train[:, indices], raw_val[:, indices]
    scaler = StandardScaler().fit(raw_train)
    mean32, scale32 = scaler.mean_.astype(np.float32), scaler.scale_.astype(np.float32)
    require(np.isfinite(mean32).all() and np.isfinite(scale32).all() and (scale32 > 0).all(),
            'Parametros de escalado no validos')
    x_train = np.divide(np.subtract(raw_train, mean32, dtype=np.float32), scale32, dtype=np.float32)
    x_val = np.divide(np.subtract(raw_val, mean32, dtype=np.float32), scale32, dtype=np.float32)
    require(np.isfinite(x_train).all() and np.isfinite(x_val).all(), 'Entradas no finitas')
    class_counts = np.bincount(y_train, minlength=3)
    require(len(class_counts) == 3 and (class_counts > 0).all(), 'Clases de entrenamiento incompletas')
    class_weights = len(y_train) / (3. * class_counts.astype(np.float64))
    sample_weights = class_weights[y_train].astype(np.float32)
    times['data_and_scaler'] = time.perf_counter() - started
    save(lambda: write_json(directory / 'procedencia.json', provenance))
    save(lambda: np.savez(directory / 'scaler_train_only.npz', mean64=scaler.mean_,
                         scale64=scaler.scale_, var64=scaler.var_, mean32=mean32, scale32=scale32))
    event('data_ready', train=len(x_train), validation=len(x_val), mask=job['mask'])

    started = time.perf_counter()
    penalty = keras.regularizers.l2(job['L2'])
    model = keras.Sequential([
        keras.layers.Input(shape=(len(indices),), name='features'),
        keras.layers.Dense(20, activation='relu', kernel_regularizer=penalty, name='hidden20'),
        keras.layers.Dense(10, activation='relu', kernel_regularizer=penalty, name='hidden10'),
        keras.layers.Dense(3, activation='softmax', name='classes')], name='cwru_runtime_pilot_' + job['mask'])
    optimizer = keras.optimizers.Adam(learning_rate=job['learning_rate'])
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy')
    optimizer.build(model.trainable_variables)

    @tf.function(input_signature=[tf.TensorSpec([None, len(indices)], tf.float32),
                                  tf.TensorSpec([None], tf.int64), tf.TensorSpec([None], tf.float32)])
    def training_step(batch_x, batch_y, batch_weights):
        with tf.GradientTape() as tape:
            probabilities = model(batch_x, training=True)
            per_example = keras.losses.sparse_categorical_crossentropy(batch_y, probabilities)
            loss = tf.reduce_mean(per_example * batch_weights) + tf.add_n(model.losses)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    def weight_digest():
        digest = hashlib.sha256()
        for weight in model.get_weights():
            digest.update(str(weight.shape).encode('ascii'))
            digest.update(weight.tobytes())
        return digest.hexdigest()

    weights_before_trace = weight_digest()
    iterations_before_trace = int(optimizer.iterations.numpy())
    training_step.get_concrete_function()
    iterations_after_trace = int(optimizer.iterations.numpy())
    require(iterations_before_trace == iterations_after_trace == 0
            and weight_digest() == weights_before_trace, 'El tracing modifico pesos o entreno un lote extra')
    require(model.count_params() == 20 * len(indices) + 263, 'Numero de parametros inesperado')
    times['graph_compile'] = time.perf_counter() - started
    event('graph_ready_without_updates', optimizer_iterations=iterations_after_trace)

    rng = np.random.default_rng(job['seed'])
    history = []
    for epoch in range(50):
        started = time.perf_counter()
        permutation = rng.permutation(len(x_train))
        for offset in range(0, len(permutation), 32):
            ix = permutation[offset:offset + 32]
            loss = float(training_step(tf.convert_to_tensor(x_train[ix]),
                         tf.convert_to_tensor(y_train[ix]), tf.convert_to_tensor(sample_weights[ix])).numpy())
            require(np.isfinite(loss), 'Actualizacion produjo perdida no finita')
        times['training_updates'] += time.perf_counter() - started
        started = time.perf_counter()
        train_probabilities = model(x_train, training=False).numpy()
        train_losses = keras.losses.sparse_categorical_crossentropy(y_train, train_probabilities).numpy()
        regularization = float(tf.add_n(model.losses).numpy())
        train_loss = float(np.mean(train_losses * sample_weights) + regularization)
        times['epoch_training_diagnostics'] += time.perf_counter() - started
        started = time.perf_counter()
        val_probabilities = model(x_val, training=False).numpy()
        val_loss = float(np.mean(keras.losses.sparse_categorical_crossentropy(y_val, val_probabilities).numpy())
                         + regularization)
        times['epoch_validation'] += time.perf_counter() - started
        require(np.isfinite(train_probabilities).all() and np.isfinite(val_probabilities).all()
                and np.isfinite([train_loss, val_loss]).all(), 'Diagnostico no finito')
        history.append(dict(epoch=epoch + 1, loss=train_loss, val_loss=val_loss,
                            optimizer_updates=int(optimizer.iterations.numpy())))
        if (epoch + 1) % 10 == 0:
            event('epoch', epoch=epoch + 1, optimizer_updates=int(optimizer.iterations.numpy()))
    require(int(optimizer.iterations.numpy()) == 1250 and len(history) == 50,
            'El piloto no ejecuto exactamente 50 epocas/1250 actualizaciones')
    require(all(np.isfinite(w).all() for w in model.get_weights()), 'Pesos finales no finitos')

    # Persistir antes de conversion: un fallo del exportador no debe perder
    # el ajuste completado y obligar a entrenarlo otra vez.
    save(lambda: np.savez(directory / 'checkpoint_antes_conversion.npz',
                         **{f'weight_{i:02d}': w for i, w in enumerate(model.get_weights())}))
    save(lambda: write_json(directory / 'estado_antes_conversion.json', dict(
        optimizer_updates=1250, history=history, timing_nonoverlapping_seconds=dict(times),
        model_weights_sha256=weight_digest(), mask=job['mask'], seed=11)))

    started = time.perf_counter()
    @tf.function(input_signature=[tf.TensorSpec([1, len(indices)], tf.float32)])
    def inference_function(inputs):
        return model(inputs, training=False)
    concrete = inference_function.get_concrete_function()
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
    trained_weight_digest = weight_digest()
    frozen = convert_variables_to_constants_v2(concrete, lower_control_flow=False)
    graph = frozen.graph.as_graph_def()
    graph_nodes = list(graph.node) + [node for fun in graph.library.function for node in fun.node_def]
    forbidden = {'ReadVariableOp', 'VarHandleOp', 'VariableV2', 'AssignVariableOp'}
    require(not any(node.op in forbidden for node in graph_nodes), 'Grafo congelado conserva variables')
    frozen_values = np.asarray([frozen(tf.convert_to_tensor(row[np.newaxis, :]))[0].numpy()[0]
                                for row in x_val], dtype=np.float32)
    original_values = model(x_val, training=False).numpy()
    freeze_max_error = float(np.max(np.abs(frozen_values - original_values)))
    require(weight_digest() == trained_weight_digest
            and np.allclose(frozen_values, original_values, rtol=1e-6, atol=1e-7)
            and np.array_equal(np.argmax(frozen_values, axis=1), np.argmax(original_values, axis=1)),
            'Congelar el grafo cambio pesos o predicciones')
    converter_fp32 = tf.lite.TFLiteConverter.from_concrete_functions([frozen])
    fp32_bytes = converter_fp32.convert()
    times['FP32_conversion'] = time.perf_counter() - started
    save(lambda: (directory / 'modelo_fp32.tflite').write_bytes(fp32_bytes))
    event('FP32_conversion_complete', model_bytes=len(fp32_bytes))

    started = time.perf_counter()
    representative_count = 0
    def representative_dataset():
        nonlocal representative_count
        for sample in x_train:
            representative_count += 1
            yield [sample[np.newaxis, :].astype(np.float32)]
    converter_int8 = tf.lite.TFLiteConverter.from_concrete_functions([frozen])
    converter_int8.optimizations = [tf.lite.Optimize.DEFAULT]
    converter_int8.representative_dataset = representative_dataset
    converter_int8.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter_int8.inference_input_type = tf.int8
    converter_int8.inference_output_type = tf.int8
    int8_bytes = converter_int8.convert()
    require(representative_count == 799, 'No se calibraron exactamente las 799 filas de entrenamiento')
    times['INT8_conversion'] = time.perf_counter() - started
    save(lambda: (directory / 'modelo_int8.tflite').write_bytes(int8_bytes))
    event('INT8_conversion_complete', representative_rows=representative_count, model_bytes=len(int8_bytes))

    def evaluate(content, integer):
        interpreter = tf.lite.Interpreter(model_content=content, num_threads=1,
            experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
        interpreter.allocate_tensors()
        inp, out = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
        operations = [item['op_name'] for item in interpreter._get_ops_details()]
        dtypes = sorted({np.dtype(item['dtype']).name for item in interpreter.get_tensor_details()})
        require(list(inp['shape']) == [1, len(indices)] and list(out['shape']) == [1, 3], 'Forma TFLite incorrecta')
        if integer:
            require(inp['dtype'] == out['dtype'] == np.int8 and set(dtypes) <= {'int8', 'int32', 'int64'}
                    and not any('Flex' in op or op in ('DEQUANTIZE', 'DELEGATE') for op in operations),
                    'INT8 contiene fallback float/delegado o tensores no enteros')
        else:
            require(inp['dtype'] == out['dtype'] == np.float32, 'Conversion FP32 con entrada/salida inesperada')
        probabilities, raw_outputs, saturation = [], [], 0
        for sample in x_val:
            batch = sample[np.newaxis, :]
            if integer:
                scale, zero = inp['quantization']
                require(scale > 0, 'Escala de cuantizacion no positiva')
                affine = np.add(np.divide(batch, np.float32(scale), dtype=np.float32), np.float32(zero), dtype=np.float32)
                rounded = np.rint(affine)
                saturation += int(np.count_nonzero((rounded < -128) | (rounded > 127)))
                batch = np.clip(rounded, -128, 127).astype(np.int8)
            interpreter.set_tensor(inp['index'], batch)
            interpreter.invoke()
            raw = interpreter.get_tensor(out['index'])[0]
            raw_outputs.append(raw)
            probability = raw.astype(np.float32)
            if integer:
                scale, zero = out['quantization']
                require(scale > 0, 'Escala de salida no positiva')
                probability = (probability - np.float32(zero)) * np.float32(scale)
            probabilities.append(probability)
        values = np.asarray(probabilities, dtype=np.float32)
        require(np.isfinite(values).all(), 'Salida TFLite no finita')
        detail = dict(input_dtype=np.dtype(inp['dtype']).name, output_dtype=np.dtype(out['dtype']).name,
                      input_quantization=[float(inp['quantization'][0]), int(inp['quantization'][1])],
                      output_quantization=[float(out['quantization'][0]), int(out['quantization'][1])],
                      tensor_dtypes=dtypes, operators=operations, threads=1, default_delegates_disabled=True,
                      validation_saturated_input_values=saturation)
        return values, np.asarray(raw_outputs), detail

    started = time.perf_counter()
    keras_values = model(x_val, training=False).numpy()
    fp32_values, fp32_raw, fp32_details = evaluate(fp32_bytes, False)
    int8_values, int8_raw, int8_details = evaluate(int8_bytes, True)
    times['inference'] = time.perf_counter() - started
    event('inference_diagnostics_complete', validation_windows=len(x_val))

    def save_history():
        with (directory / 'historial_piloto.csv').open('x', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
    save(save_history)
    save(lambda: np.savez(directory / 'pesos_piloto_no_confirmatorios.npz',
                         **{f'weight_{i:02d}': w for i, w in enumerate(model.get_weights())}))
    save(lambda: np.savez(directory / 'diagnostico_conversion_validacion.npz',
                         y_true=y_val, keras_probabilities=keras_values, fp32_probabilities=fp32_values,
                         int8_probabilities=int8_values, int8_outputs=int8_raw,
                         validation_global_window_ids=np.asarray(job['validation']['global_window_ids'], dtype=np.int64)))
    elapsed = time.perf_counter() - WORKER_STARTED
    result = dict(status='PASS', job_id=job_id, mask=job['mask'], seed=11, purpose='runtime_and_conversion_only',
        confirmatory_fit=False, reused_for_selection=False, early_stopping=False,
        training_windows=799, validation_windows=175, epochs=50, batches_per_epoch=25, optimizer_updates=1250,
        architecture=[len(indices), 20, 10, 3], parameters=model.count_params(),
        class_counts=class_counts.astype(int).tolist(), class_weights=class_weights.tolist(),
        tracing_changed_model_weights=False, optimizer_iterations_before_after_trace=[0, 0],
        representative_calibration_rows=representative_count,
        export_method='frozen_concrete_function_from_identical_trained_weights',
        freeze_tensorflow_internal_API_version=tf.__version__,
        freeze_model_weights_unchanged=True, freeze_validation_class_changes=0,
        freeze_validation_probability_max_error=freeze_max_error,
        freeze_resource_variable_ops_remaining=0,
        timing_nonoverlapping_seconds=times,
        derived_training_epochs_seconds=sum(times[k] for k in ('training_updates', 'epoch_training_diagnostics', 'epoch_validation')),
        worker_elapsed_to_result_seconds=elapsed,
        worker_unattributed_before_result_seconds=max(0., elapsed - sum(times.values())),
        final_result_write_and_process_exit_included_in_stage_times=False,
        versions=dict(python=platform.python_version(), tensorflow=tf.__version__, keras=keras.__version__,
                      numpy=np.__version__, sklearn=sklearn.__version__),
        python_executable=sys.executable, worker_environment=authorization['worker_environment'],
        keras_FP32_class_changes=int(np.count_nonzero(np.argmax(keras_values, axis=1) != np.argmax(fp32_values, axis=1))),
        FP32_INT8_class_changes=int(np.count_nonzero(np.argmax(fp32_values, axis=1) != np.argmax(int8_values, axis=1))),
        fp32_details=fp32_details, int8_details=int8_details,
        model_bytes=dict(FP32=len(fp32_bytes), INT8=len(int8_bytes)),
        model_sha256=dict(FP32=hashlib.sha256(fp32_bytes).hexdigest(), INT8=hashlib.sha256(int8_bytes).hexdigest()),
        scaler_fit_scope='training_only_float64_fit_export_float32', normalization='explicit_float32_subtract_then_divide',
        no_load3_arrays_opened=True, no_detector_fitted=True, no_STM32_latency_measured=True,
        validation_used_to_select_checkpoint_or_mask=False)
    write_json(directory / 'resultado.json', result)
    event('PASS', updates=1250)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job-id', required=True,
        choices=['pilot_outer3_inner0_F14_seed11', 'pilot_outer3_inner0_T6_seed11'])
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args.job_id, args.run_dir)
    except Exception as error:
        # Preserva el fallo dentro del intento; nunca reintenta otro trabajo.
        if args.run_dir.resolve().parent == OUTPUT.resolve() and args.run_dir.is_dir():
            failure = args.run_dir / ('FALLO_' + args.job_id + '.json')
            if not failure.exists():
                write_json(failure, dict(status='FAIL', job_id=args.job_id,
                    error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc(),
                    elapsed_seconds=time.perf_counter() - WORKER_STARTED, automatic_retry=False))
        raise
