"""Ajuste interno V2 y exportacion; solo los 32 representantes autorizados.

La importacion de este modulo no importa TensorFlow ni abre datos CWRU.
El entrenamiento real solo entra desde un contrato de campana comprobado.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

START = time.perf_counter()
ROOT = Path(__file__).resolve().parents[4]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def write(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def weights_hash(model):
    digest = hashlib.sha256()
    for weight in model.get_weights():
        digest.update(str(weight.shape).encode('ascii'))
        digest.update(weight.tobytes())
    return digest.hexdigest()


def configure_tf(seed):
    import tensorflow as tf
    from tensorflow import keras
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.set_visible_devices([], 'GPU')
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
    return tf, keras


def fit_network(tf, keras, x, y, validation, y_validation, config, seed, directory, event):
    """ES interno; persistir cada mejora y restaurar antes de exportar."""
    import numpy as np
    from numerica_representantes import EarlyStopping, balanced_weights
    dimension = x.shape[1]
    counts, class_weights, sample_weights = balanced_weights(y)
    penalty = keras.regularizers.l2(config['l2_hidden'])
    model = keras.Sequential([
        keras.layers.Input(shape=(dimension,), name='features'),
        keras.layers.Dense(20, activation='relu', kernel_regularizer=penalty, name='hidden20'),
        keras.layers.Dense(10, activation='relu', kernel_regularizer=penalty, name='hidden10'),
        keras.layers.Dense(3, activation='softmax', name='classes')], name='cwru_v2')
    optimizer = keras.optimizers.Adam(learning_rate=config['learning_rate'])
    optimizer.build(model.trainable_variables)

    @tf.function(input_signature=[tf.TensorSpec([None, dimension], tf.float32),
                                  tf.TensorSpec([None], tf.int64), tf.TensorSpec([None], tf.float32)])
    def step(batch_x, batch_y, batch_weights):
        with tf.GradientTape() as tape:
            probabilities = model(batch_x, training=True)
            losses = keras.losses.sparse_categorical_crossentropy(batch_y, probabilities)
            loss = tf.reduce_mean(losses * batch_weights) + tf.add_n(model.losses)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    before = weights_hash(model)
    step.get_concrete_function()
    require(int(optimizer.iterations.numpy()) == 0 and before == weights_hash(model),
            'Tracing modifico pesos o aplico una actualizacion')
    require(model.count_params() == 20 * dimension + 263, 'Arquitectura inesperada')
    rng = np.random.default_rng(seed)
    es_config = config['early_stopping']
    stopper = EarlyStopping(patience=es_config['patience'], min_delta=es_config['minimum_improvement'])
    best_weights = None
    history = []
    timings = dict(training_updates=0., epoch_training_diagnostics=0., epoch_validation=0.)
    for epoch in range(1, config['max_epochs'] + 1):
        started = time.perf_counter()
        order = rng.permutation(len(x))
        for offset in range(0, len(order), config['batch_size']):
            ix = order[offset:offset + config['batch_size']]
            loss = float(step(tf.convert_to_tensor(x[ix]), tf.convert_to_tensor(y[ix]),
                              tf.convert_to_tensor(sample_weights[ix])).numpy())
            require(np.isfinite(loss), 'Actualizacion no finita')
        timings['training_updates'] += time.perf_counter() - started
        started = time.perf_counter()
        probabilities = model(x, training=False).numpy()
        regularization = float(tf.add_n(model.losses).numpy())
        train_loss = float(np.mean(keras.losses.sparse_categorical_crossentropy(y, probabilities).numpy()
                                   * sample_weights) + regularization)
        timings['epoch_training_diagnostics'] += time.perf_counter() - started
        started = time.perf_counter()
        val_probabilities = model(validation, training=False).numpy()
        val_loss = float(np.mean(keras.losses.sparse_categorical_crossentropy(
            y_validation, val_probabilities).numpy()) + regularization)
        timings['epoch_validation'] += time.perf_counter() - started
        require(np.isfinite(probabilities).all() and np.isfinite(val_probabilities).all()
                and np.isfinite(train_loss), 'Diagnostico de epoca no finito')
        improved, stop = stopper.observe(epoch, val_loss)
        if improved:
            best_weights = [weight.copy() for weight in model.get_weights()]
            np.savez(directory / f'checkpoint_epoca_{epoch:02d}.npz',
                     **{f'weight_{i:02d}': weight for i, weight in enumerate(best_weights)})
        row = dict(epoch=epoch, loss=train_loss, val_loss=val_loss,
                   optimizer_updates=int(optimizer.iterations.numpy()), improved=improved,
                   best_epoch=stopper.best_epoch, patience_used=stopper.wait)
        history.append(row)
        with (directory / 'historial.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
        event('epoch', **row)
        if stop:
            break
    require(best_weights is not None, 'ES no produjo checkpoint')
    model.set_weights(best_weights)
    require(all(np.isfinite(w).all() for w in model.get_weights()), 'Pesos restaurados no finitos')
    expected_updates = len(history) * ((len(x) + config['batch_size'] - 1) // config['batch_size'])
    require(int(optimizer.iterations.numpy()) == expected_updates, 'Cantidad de updates distinta')
    # Primero el checkpoint y el estado. La conversion nunca precede a ellos.
    np.savez(directory / 'checkpoint_antes_conversion.npz',
             **{f'weight_{i:02d}': weight for i, weight in enumerate(model.get_weights())})
    state = dict(best_epoch=stopper.best_epoch, best_validation_loss=stopper.best_loss,
                 epochs_executed=len(history), optimizer_updates=expected_updates,
                 weights_restored=True, model_weights_sha256=weights_hash(model),
                 class_counts=counts.tolist(), class_weights=class_weights.tolist(),
                 training_timing_seconds=timings, history=history,
                 validation_loss='unweighted_sparse_crossentropy_plus_L2', tracing_updates=0)
    write(directory / 'estado_antes_conversion.json', state)
    return model, state


def export_models(tf, model, training, validation, directory, event):
    """Congelar a lote1; full INT8 calibrado con TODO train en su orden."""
    import numpy as np
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
    started = time.perf_counter()
    dimension = training.shape[1]

    @tf.function(input_signature=[tf.TensorSpec([1, dimension], tf.float32)])
    def fn(inputs):
        return model(inputs, training=False)

    concrete = fn.get_concrete_function()
    before = weights_hash(model)
    frozen = convert_variables_to_constants_v2(concrete, lower_control_flow=False)
    graph = frozen.graph.as_graph_def()
    nodes = list(graph.node) + [node for function in graph.library.function for node in function.node_def]
    forbidden = {'ReadVariableOp', 'VarHandleOp', 'VariableV2', 'AssignVariableOp'}
    require(not any(node.op in forbidden for node in nodes), 'Grafo congelado conserva variables')
    original = np.asarray([concrete(tf.convert_to_tensor(row[None, :])).numpy()[0]
                           for row in validation], dtype=np.float32)
    frozen_values = np.asarray([frozen(tf.convert_to_tensor(row[None, :]))[0].numpy()[0]
                                for row in validation], dtype=np.float32)
    require(before == weights_hash(model) and np.isfinite(original).all() and np.isfinite(frozen_values).all()
            and np.allclose(original, frozen_values, rtol=1e-6, atol=1e-7)
            and np.array_equal(original.argmax(axis=1), frozen_values.argmax(axis=1)),
            'Freeze cambio pesos/predicciones a igual batch1')
    freeze = dict(comparison_batch_size=1, rtol=1e-6, atol=1e-7, model_weights_sha256=before,
                  maximum_absolute_error=float(np.max(np.abs(original - frozen_values))),
                  class_changes=0, weights_unchanged=True, resource_variable_ops_remaining=0,
                  tensorflow_internal_API_version=tf.__version__)
    write(directory / 'DIAGNOSTICO_FREEZE.json', freeze)
    converter = tf.lite.TFLiteConverter.from_concrete_functions([frozen])
    fp32 = converter.convert()
    (directory / 'modelo_fp32.tflite').write_bytes(fp32)
    fp32_seconds = time.perf_counter() - started
    event('FP32_exported', bytes=len(fp32))
    started = time.perf_counter()
    visited = []

    def representative():
        for index, row in enumerate(training):
            visited.append(index)
            yield [row[None, :]]

    converter = tf.lite.TFLiteConverter.from_concrete_functions([frozen])
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    integer = converter.convert()
    require(visited == list(range(len(training))), 'Calibracion INT8 incompleta, repetida o desordenada')
    (directory / 'modelo_int8.tflite').write_bytes(integer)
    event('INT8_exported', bytes=len(integer), calibration_rows=len(visited))
    return fp32, integer, original, dict(freeze=freeze, representative_rows=len(visited),
        representative_row_order='ascending_authorized_training_global_window_id',
        FP32_seconds=fp32_seconds, INT8_seconds=time.perf_counter() - started)


def evaluate_lite(tf, content, x, integer):
    import numpy as np
    from numerica_representantes import quantize_int8
    interpreter = tf.lite.Interpreter(model_content=content, num_threads=1,
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
    interpreter.allocate_tensors()
    inp, out = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
    dtypes = sorted({np.dtype(t['dtype']).name for t in interpreter.get_tensor_details()})
    ops = [o['op_name'] for o in interpreter._get_ops_details()]
    require(list(inp['shape']) == [1, x.shape[1]] and list(out['shape']) == [1, 3], 'Formas Lite distintas')
    if integer:
        require(inp['dtype'] == out['dtype'] == np.int8 and set(dtypes) <= {'int8', 'int32', 'int64'}
                and not any('Flex' in op or op in ('DEQUANTIZE', 'DELEGATE', 'READ_VARIABLE', 'VAR_HANDLE') for op in ops),
                'INT8 contiene fallback o variables')
    else:
        require(inp['dtype'] == out['dtype'] == np.float32, 'Entrada/salida FP32 distinta')
    values, raw_values, inputs, saturation = [], [], [], 0
    for row in x:
        batch = row[None, :]
        if integer:
            batch, count = quantize_int8(batch, *inp['quantization'])
            saturation += count
        interpreter.set_tensor(inp['index'], batch)
        interpreter.invoke()
        raw = interpreter.get_tensor(out['index'])[0]
        raw_values.append(raw.copy())
        inputs.append(batch[0].copy())
        probability = raw.astype(np.float32)
        if integer:
            scale, zero = out['quantization']
            require(np.isfinite(scale) and scale > 0, 'Escala de salida invalida')
            probability = np.multiply(np.subtract(probability, np.float32(zero), dtype=np.float32),
                                      np.float32(scale), dtype=np.float32)
        values.append(probability)
    values = np.asarray(values, dtype=np.float32)
    require(np.isfinite(values).all(), 'Salida Lite no finita')
    details = dict(input_dtype=np.dtype(inp['dtype']).name, output_dtype=np.dtype(out['dtype']).name,
                   input_quantization=[float(inp['quantization'][0]), int(inp['quantization'][1])],
                   output_quantization=[float(out['quantization'][0]), int(out['quantization'][1])],
                   tensor_dtypes=dtypes, operators=ops, input_clipped_values=saturation,
                   default_delegates_disabled=True, threads=1)
    return values, np.asarray(raw_values), np.asarray(inputs), details


def run(campaign, batch, job_id):
    from ejecutar_representantes import validate_campaign, write_manifest, validate_result, ENVIRONMENT, versions
    plan, jobs = validate_campaign(campaign)
    job = next((j for j in jobs if j['job_id'] == job_id), None)
    require(job is not None, 'Trabajo fuera de los32 representantes')
    require(batch.resolve().parent == campaign / 'lotes' and (campaign / 'EJECUCION_EN_CURSO.lock').is_file(),
            'Worker requiere lote activo del orquestador')
    batch_plan = json.loads((batch / 'PLAN_LOTE.json').read_text(encoding='utf-8'))
    require(job_id in batch_plan['job_ids'] and not (batch / ('ESTADO_' + job_id + '.json')).exists(),
            'Trabajo no solicitado o ya cerrado')
    require(versions() == plan['tested_versions'], 'Worker con versiones no probadas')
    require(all(os.environ.get(k) == v for k, v in ENVIRONMENT.items()), 'Entorno de worker no fijado')
    directory = campaign / job_id
    directory.mkdir(exist_ok=False)

    def event(stage, **extra):
        row = dict(stage=stage, elapsed_seconds=time.perf_counter() - START, **extra)
        with (directory / 'progreso.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
        print(json.dumps(row), flush=True)

    event('start', job_id=job_id)
    started = time.perf_counter()
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from datos_representantes import load_job_data
    from numerica_representantes import normalize_f32, metrics3, integer_argmax, detector_metrics, quantize_int8
    tf, keras = configure_tf(job['seed'])
    import_seconds = time.perf_counter() - started
    event('imports_complete', versions=versions())
    started = time.perf_counter()
    raw, y, raw_val, y_val, provenance = load_job_data(job)
    write(directory / 'procedencia.json', provenance)
    indices = job['feature_indices']
    scaler = StandardScaler().fit(raw[:, indices])
    mean, scale = scaler.mean_.astype(np.float32), scaler.scale_.astype(np.float32)
    x, val = normalize_f32(raw[:, indices], mean, scale), normalize_f32(raw_val[:, indices], mean, scale)
    np.savez(directory / 'scaler_train_only.npz', mean64=scaler.mean_, scale64=scaler.scale_,
             var64=scaler.var_, mean32=mean, scale32=scale)
    data_seconds = time.perf_counter() - started
    event('data_ready', training=len(x), validation=len(val))
    started = time.perf_counter()
    model, state = fit_network(tf, keras, x, y, val, y_val, plan['dnn'], job['seed'], directory, event)
    fit_seconds = time.perf_counter() - started
    event('best_checkpoint_restored', best_epoch=state['best_epoch'])
    fp32, int8, original, export = export_models(tf, model, x, val, directory, event)
    started = time.perf_counter()
    fp, fp_raw, _, fp_details = evaluate_lite(tf, fp32, val, False)
    iq, iq_raw, iq_inputs, iq_details = evaluate_lite(tf, int8, val, True)
    train_quantized, train_clipped = quantize_int8(x, *iq_details['input_quantization'])
    np.savez(directory / 'cuantizacion_train.npz', global_window_ids=np.asarray(job['training_global_window_ids'],dtype=np.int64),
             int8_inputs=train_quantized)
    fp_pred, iq_pred = integer_argmax(fp), integer_argmax(iq)
    metrics = dict(FP32=metrics3(y_val, fp_pred), INT8=metrics3(y_val, iq_pred))
    np.savez(directory / 'predicciones_validacion.npz', y_true=y_val,
             global_window_ids=np.asarray(job['validation_global_window_ids'], dtype=np.int64),
             keras_batch1=original, fp32_probabilities=fp, int8_probabilities=iq,
             fp32_outputs=fp_raw, int8_outputs=iq_raw, int8_inputs=iq_inputs,
             fp32_predictions=fp_pred, int8_predictions=iq_pred)
    inference_seconds = time.perf_counter() - started
    # El detector no consume errores/clases de la DNN. Solo normales de scopes fijados.
    started = time.perf_counter()
    sys.path.insert(0, str(ROOT / 'codigo/cwru/ampliacion_v2/piloto_clasicos'))
    from detector_numerico import fit_normality, detector_score
    from datos_representantes import load_verified_context
    _, _, rows, _ = load_verified_context()
    by_id = {int(row['global_window_id']): row for row in rows}
    train_pos = {key: pos for pos, key in enumerate(job['training_global_window_ids'])}
    fit_ids, cal_ids = job['detector_fit_global_window_ids'], job['detector_calibration_global_window_ids']
    require(all(y[train_pos[key]] == 0 for key in fit_ids + cal_ids), 'Detector incluyo falla en ajuste/calibracion')
    det_indices = [i for i in indices if i not in (4, 11)]
    params = fit_normality(raw[[train_pos[key] for key in fit_ids]],
                          raw[[train_pos[key] for key in cal_ids]],
                          [str(by_id[key]['file_id']) for key in cal_ids], det_indices)
    cal_scores = detector_score(raw[[train_pos[key] for key in cal_ids]], params)
    np.savez(directory / 'detector_calibracion.npz', global_window_ids=np.asarray(cal_ids,dtype=np.int64),
             record_ids=np.asarray([str(by_id[key]['file_id']) for key in cal_ids]), scores=cal_scores)
    scores = detector_score(raw_val, params)
    rejected = scores > params['threshold32']
    det_metrics = detector_metrics(y_val, rejected,
        [str(by_id[key]['file_id']) for key in job['validation_global_window_ids']])
    np.savez(directory / 'detector_parametros.npz', indices=np.asarray(det_indices, dtype=np.int64),
             mean64=params['mean64'], scale64=params['scale64'], var64=params['var64'],
             mean32=params['mean32'], scale32=params['scale32'], centroid32=params['centroid32'],
             threshold32=params['threshold32'], fit_ids=np.asarray(fit_ids, dtype=np.int64),
             calibration_ids=np.asarray(cal_ids, dtype=np.int64))
    np.savez(directory / 'detector_validacion.npz', global_window_ids=np.asarray(job['validation_global_window_ids']),
             y_true=y_val, scores=scores, rejected=rejected)
    # Banco de exportacion solo del ajuste. Tiempo de PC no es costo ARM.
    bench_ids = job['benchmark_global_window_ids']
    bench_raw = raw[[train_pos[key] for key in bench_ids]]
    bench_x = normalize_f32(bench_raw[:, indices], mean, scale)
    _, bench_outputs, bench_inputs, bench_details = evaluate_lite(tf, int8, bench_x, True)
    bench_scores = detector_score(bench_raw, params)
    np.savez(directory / 'banco_costos_referencia.npz', global_window_ids=np.asarray(bench_ids, dtype=np.int64),
             raw_features14=bench_raw, normalized_features=bench_x, int8_inputs=bench_inputs,
             int8_outputs=bench_outputs, detector_scores=bench_scores,
             outside_normality=bench_scores > params['threshold32'])
    det_seconds = time.perf_counter() - started
    result = dict(status='PASS', job_id=job_id, fit_signature=plan['fit_signatures'][job_id],
        candidate=job['candidate'], outer_load_hp=job['outer_load_hp'],
        inner_validation_load_hp=job['inner_validation_load_hp'], seed=job['seed'],
        purpose='V2_inner_fit_and_predefined_total_cost_representative', included_in_288_inner=True,
        roles=job['roles'], training_windows=len(x), validation_windows=len(val),
        best_epoch=state['best_epoch'], epochs_executed=state['epochs_executed'],
        optimizer_updates=state['optimizer_updates'], weights_restored=True,
        architecture=[len(indices), 20, 10, 3], parameters=model.count_params(),
        representative_calibration_rows=export['representative_rows'], freeze=export['freeze'],
        model_sha256=dict(FP32=hashlib.sha256(fp32).hexdigest(), INT8=hashlib.sha256(int8).hexdigest()),
        model_bytes=dict(FP32=len(fp32), INT8=len(int8)), metrics=metrics, detector_metrics=det_metrics,
        FP32_INT8_class_changes=int(np.count_nonzero(fp_pred != iq_pred)),
        keras_batch1_FP32_class_changes=int(np.count_nonzero(integer_argmax(original) != fp_pred)),
        fp32_details=fp_details, int8_details=iq_details, benchmark_int8_details=bench_details,
        training_saturated_input_values=train_clipped,
        benchmark_rows=len(bench_ids), detector_indices=det_indices,
        detector_refit_after_calibration=False, detector_recomputed_per_representative=True,
        detector_F14_NF12_alias_not_counted_as_independent_evidence=True,
        measured_STM32_total_cost=None, remaining_grid_enabled=False, outer_evaluation_executed=False,
        retrospective_evaluation=True, versions=versions(),
        timings_seconds=dict(imports=import_seconds, data_and_scaler=data_seconds,
            fit_including_checkpoint_IO=fit_seconds, FP32_conversion=export['FP32_seconds'],
            INT8_conversion=export['INT8_seconds'], validation_inference_and_IO=inference_seconds,
            detector_and_benchmark_and_IO=det_seconds), elapsed_worker_seconds=time.perf_counter() - START)
    validate_result(result, job, plan['fit_signatures'][job_id])
    write(directory / 'resultado.json', result)
    event('PASS', best_epoch=state['best_epoch'], macro_f1_int8=metrics['INT8']['macro_f1'])
    write_manifest(directory / 'MANIFEST_JOB_SHA256.json', directory)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--batch', type=Path, required=True)
    parser.add_argument('--job-id', required=True)
    args = parser.parse_args()
    try:
        run(args.campaign.resolve(), args.batch.resolve(), args.job_id)
    except Exception as error:
        # El orquestador registra el estado incluso si falla antes de crear carpeta.
        print(traceback.format_exc(), file=sys.stderr, flush=True)
        raise SystemExit(1)
