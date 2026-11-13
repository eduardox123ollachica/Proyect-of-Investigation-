"""Reajuste DNN final de épocas fijas, sin validación exterior.

Importar este módulo no importa TensorFlow, sklearn ni matrices CWRU.
El worker recibe exclusivamente datos de desarrollo autorizados por la
selección congelada. El entrenamiento real requiere un payload del ejecutor.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[4]
LEGACY = ROOT / 'codigo/cwru/ampliacion_v2/representantes'
if str(LEGACY) not in sys.path:
    sys.path.insert(0, str(LEGACY))
from worker_representantes import configure_tf, export_models, evaluate_lite, weights_hash


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def validate_payload(payload, arrays):
    """Validar el contrato numérico después de la autorización del padre.

    ``arrays`` ya contiene SOLO entrenamiento. Esta comprobación no importa
    TensorFlow, ajusta un estimador ni accede a matrices de otras cargas.
    La procedencia de cada ID/registro la autentica contrato.authorize.
    """
    import numpy as np
    require(payload.get('status') == 'AUTHORIZED_FINAL_REFIT_PAYLOAD'
            and payload.get('family') == 'DNN', 'Payload no es un reajuste DNN autorizado')
    job = payload['job']
    outer, seed, epochs = job['outer_load_hp'], job['seed'], job['epochs']
    require(type(outer) is int and outer in range(4), 'Carga exterior inválida')
    require(type(seed) is int and seed in (11, 22, 33, 42, 44, 55, 66, 77, 88, 99),
            'Semilla final fuera del protocolo')
    require(type(epochs) is int and 1 <= epochs <= 50, 'Épocas finales inválidas')
    roles = job['roles']
    expected_loads = [load for load in range(4) if load != outer]
    require(all(type(value) is int for value in roles['training_loads'])
            and roles['training_loads'] == roles['dnn_fit_loads'] == roles['int8_calibration_loads'] == expected_loads
            and roles['evaluation_loads'] == [outer] and roles['inner_validation_load_hp'] is None,
            'Roles finales incluyen validación interna o carga exterior')
    ids = job['training_global_window_ids']
    require(isinstance(ids, list) and ids and all(type(value) is int and value >= 0 for value in ids)
            and ids == sorted(set(ids)), 'IDs train deben ser enteros únicos ordenados')
    indices = job['feature_indices']
    require(isinstance(indices, list) and indices and all(type(value) is int and 0 <= value < 14 for value in indices)
            and indices == sorted(set(indices)) and job['input_dimension'] == len(indices),
            'Columnas DNN inválidas')
    config = payload['dnn']
    expected = dict(architecture='d-20-10-3', hidden_activation='relu', output_activation='softmax',
        l2_hidden=.001, optimizer='Adam', learning_rate=.005, batch_size=32, max_epochs=50,
        manual_tf_function_batches=True, deterministic_ops=True, seed_ensembles=False)
    require(all(config.get(key) == value for key, value in expected.items()), 'Configuración DNN distinta del protocolo')
    require(config.get('outer_epochs') == 'integer median of nine best inner epochs for that mask; no outer early stopping',
            'Regla de épocas finales distinta')
    require(set(arrays) == {'raw_training', 'y_training', 'training_ids', 'training_records'},
            'NPZ debe contener exclusivamente los cuatro arreglos train')
    raw, y, training_ids, records = (arrays[key] for key in
        ('raw_training', 'y_training', 'training_ids', 'training_records'))
    size = len(ids)
    require(raw.dtype == np.dtype('float32') and raw.shape == (size, 14)
            and np.isfinite(raw).all(), 'Matriz train debe ser float32 finita n por 14')
    require(y.dtype == np.dtype('int64') and y.shape == (size,) and set(y.tolist()) == {0, 1, 2},
            'Etiquetas train deben ser int64 y contener las tres clases')
    require(training_ids.dtype == np.dtype('int64') and training_ids.shape == (size,)
            and training_ids.tolist() == ids, 'IDs del NPZ no corresponden al trabajo')
    require(records.dtype.kind == 'U' and records.shape == (size,)
            and all(value and value.strip() == value for value in records.tolist()),
            'Registro train debe ser identificador Unicode no vacío')
    require(len(set(records.tolist())) == 21, 'Tres cargas de desarrollo requieren 21 MAT')
    provenance = payload['provenance']['training']
    require(provenance['global_window_ids'] == ids
            and provenance['selected_loads_hp'] == expected_loads
            and all(type(value) is int for value in provenance['selected_loads_hp'])
            and provenance['selected_record_ids'] == sorted(set(records.tolist()), key=int)
            and provenance['shape'] == [size, 14]
            and provenance['loaded_dtype'] == 'float32'
            and provenance['only_requested_rows_selected_or_materialized'] is True,
            'Procedencia train no corresponde a IDs, cargas, registros o forma')
    return raw, y, training_ids, records


def fit_fixed_network(tf, keras, x, y, config, seed, epochs, directory, event):
    """Mismas actualizaciones que el ajuste interno; exactamente N épocas.

    No recibe validación, no mantiene/restaura un mejor checkpoint y no puede
    parar por una métrica. Los pesos exportados son los de la última época.
    """
    import numpy as np
    from numerica_representantes import balanced_weights
    require(type(epochs) is int and 1 <= epochs <= config['max_epochs'], 'Épocas finales inválidas')
    require(type(seed) is int, 'Semilla debe ser entero')
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
            'Tracing modificó pesos o aplicó una actualización')
    require(model.count_params() == 20 * dimension + 263, 'Arquitectura distinta de la fijada')
    rng = np.random.default_rng(seed)
    history = []
    timings = dict(training_updates=0., epoch_training_diagnostics=0.)
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        order = rng.permutation(len(x))
        for offset in range(0, len(order), config['batch_size']):
            ix = order[offset:offset + config['batch_size']]
            loss = float(step(tf.convert_to_tensor(x[ix]), tf.convert_to_tensor(y[ix]),
                              tf.convert_to_tensor(sample_weights[ix])).numpy())
            require(np.isfinite(loss), 'Actualización no finita')
        timings['training_updates'] += time.perf_counter() - started
        started = time.perf_counter()
        probabilities = model(x, training=False).numpy()
        regularization = float(tf.add_n(model.losses).numpy())
        train_loss = float(np.mean(keras.losses.sparse_categorical_crossentropy(y, probabilities).numpy()
                                   * sample_weights) + regularization)
        require(np.isfinite(probabilities).all() and np.isfinite(train_loss), 'Diagnóstico train no finito')
        timings['epoch_training_diagnostics'] += time.perf_counter() - started
        row = dict(epoch=epoch, loss=train_loss, optimizer_updates=int(optimizer.iterations.numpy()))
        history.append(row)
        with (directory / 'historial.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
        event('epoch', **row)
    require(len(history) == epochs, 'No se completaron las épocas congeladas')
    require(all(np.isfinite(weight).all() for weight in model.get_weights()), 'Pesos finales no finitos')
    expected_updates = epochs * ((len(x) + config['batch_size'] - 1) // config['batch_size'])
    require(int(optimizer.iterations.numpy()) == expected_updates, 'Cantidad de actualizaciones distinta')
    np.savez(directory / 'checkpoint_antes_conversion.npz',
             **{f'weight_{i:02d}': weight for i, weight in enumerate(model.get_weights())})
    state = dict(final_epoch=epochs, fixed_epochs=epochs, epochs_executed=len(history),
        optimizer_updates=expected_updates, weights_restored=False, early_stopping_used=False,
        validation_used=False, model_weights_sha256=weights_hash(model), class_counts=counts.tolist(),
        class_weights=class_weights.tolist(), training_timing_seconds=timings, history=history,
        training_loss='mean_class_balanced_weighted_sparse_crossentropy_plus_L2', tracing_updates=0)
    write(directory / 'estado_antes_conversion.json', state)
    return model, state


def run(payload_path, output):
    from contrato import authorize
    from ejecutar_representantes import write_manifest, versions
    # Autenticar plan, firma, entorno, fuentes y solicitud activa antes de crear
    # el directorio o importar librerías de entrenamiento.
    payload_path, directory = Path(payload_path).resolve(), Path(output).resolve()
    payload = authorize(payload_path, directory, 'DNN')
    require(directory.is_relative_to(ROOT.resolve()) and directory != ROOT.resolve(), 'Salida fuera del proyecto')
    require(not directory.exists(), 'No sobrescribir ni reintentar un reajuste existente')
    job = payload['job']
    started_worker = time.perf_counter()
    directory.mkdir(parents=False, exist_ok=False)

    def event(stage, **extra):
        row = dict(stage=stage, elapsed_seconds=time.perf_counter() - started_worker, **extra)
        with (directory / 'progreso.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
        print(json.dumps(row), flush=True)

    event('start', job_id=job['job_id'], family='DNN', fixed_epochs=job['epochs'])
    started = time.perf_counter()
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from numerica_representantes import normalize_f32, metrics3, integer_argmax, quantize_int8
    arrays_path = Path(payload['arrays_path']).resolve()
    require(arrays_path.is_relative_to(ROOT.resolve()) and sha(arrays_path) == payload['arrays_sha256'],
            'Arreglo train fuera del proyecto o modificado')
    with np.load(arrays_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    raw, y, training_ids, records = validate_payload(payload, arrays)
    tf, keras = configure_tf(job['seed'])
    import_seconds = time.perf_counter() - started
    require(all(payload['expected_versions'].get(key) == value for key, value in versions().items()),
            'Versiones cambiaron dentro del worker')
    event('imports_complete', versions=versions())
    write(directory / 'procedencia.json', payload['provenance'])
    started = time.perf_counter()
    indices = job['feature_indices']
    scaler = StandardScaler().fit(raw[:, indices])
    mean, scale = scaler.mean_.astype(np.float32), scaler.scale_.astype(np.float32)
    x = normalize_f32(raw[:, indices], mean, scale)
    np.savez(directory / 'scaler_train_only.npz', mean64=scaler.mean_, scale64=scaler.scale_, var64=scaler.var_,
             mean32=mean, scale32=scale, feature_indices=np.asarray(indices, dtype=np.int64))
    data_seconds = time.perf_counter() - started
    event('data_ready', training=len(x), validation=0, external_rows=0)
    started = time.perf_counter()
    model, state = fit_fixed_network(tf, keras, x, y, payload['dnn'], job['seed'], job['epochs'], directory, event)
    fit_seconds = time.perf_counter() - started
    model.save(directory / 'modelo.keras')
    require(weights_hash(model) == state['model_weights_sha256'], 'Guardar Keras alteró pesos')
    event('fixed_epochs_complete', final_epoch=job['epochs'])
    # El segundo x de export_models se usa SOLO para comparar grafo Keras y
    # congelado en lote1. No determina pesos, épocas o hiperparámetros.
    fp32, int8, original, export = export_models(tf, model, x, x, directory, event)
    started = time.perf_counter()
    fp, fp_raw, _, fp_details = evaluate_lite(tf, fp32, x, False)
    iq, iq_raw, iq_inputs, iq_details = evaluate_lite(tf, int8, x, True)
    train_quantized, train_clipped = quantize_int8(x, *iq_details['input_quantization'])
    require(np.array_equal(train_quantized, iq_inputs), 'Cuantización train no coincide con el intérprete')
    fp_pred, iq_pred = integer_argmax(fp), integer_argmax(iq)
    diagnostics = dict(FP32=metrics3(y, fp_pred), INT8=metrics3(y, iq_pred))
    np.savez(directory / 'diagnosticos_train_only.npz', global_window_ids=training_ids, record_ids=records,
             y_true=y, keras_batch1=original, fp32_probabilities=fp, int8_probabilities=iq,
             fp32_outputs=fp_raw, int8_outputs=iq_raw, int8_inputs=iq_inputs,
             fp32_predictions=fp_pred, int8_predictions=iq_pred)
    inference_seconds = time.perf_counter() - started
    require(export['representative_rows'] == len(training_ids)
            and weights_hash(model) == state['model_weights_sha256'],
            'Calibración incompleta o conversión alteró pesos')
    result = dict(status='PASS', family='DNN', job_id=job['job_id'], fit_signature=payload['fit_signature'],
        selection_signature=payload['selection_signature'], purpose='final_fixed_epoch_refit_training_only',
        candidate=job['candidate'], outer_load_hp=job['outer_load_hp'], seed=job['seed'], roles=job['roles'],
        procedures=job['procedures'], training_windows=len(x), training_records=len(set(records.tolist())),
        final_epoch=job['epochs'], fixed_epochs=job['epochs'], epochs_executed=state['epochs_executed'],
        optimizer_updates=state['optimizer_updates'], early_stopping_used=False, validation_used=False,
        weights_restored=False, final_weights_sha256=state['model_weights_sha256'],
        architecture=[len(indices), 20, 10, 3], parameters=model.count_params(),
        class_counts=state['class_counts'], class_weights=state['class_weights'],
        representative_calibration_rows=export['representative_rows'], freeze=export['freeze'],
        model_sha256=dict(Keras=sha(directory / 'modelo.keras'), FP32=hashlib.sha256(fp32).hexdigest(),
                          INT8=hashlib.sha256(int8).hexdigest()),
        model_bytes=dict(Keras=(directory / 'modelo.keras').stat().st_size, FP32=len(fp32), INT8=len(int8)),
        training_only_diagnostics=diagnostics, fp32_details=fp_details, int8_details=iq_details,
        training_saturated_input_values=train_clipped,
        training_FP32_INT8_class_changes=int(np.count_nonzero(fp_pred != iq_pred)),
        training_keras_batch1_FP32_class_changes=int(np.count_nonzero(integer_argmax(original) != fp_pred)),
        outer_rows_selected_or_materialized=False, outer_prediction_files_read=False,
        outer_evaluation_executed=False, predictive_selection_executed=False, detector_fit_executed=False,
        STM32_cost_measured=False, retrospective_evaluation=True, versions=versions(),
        original_training_core_source_sha256=sha(LEGACY / 'worker_representantes.py'),
        worker_source_sha256=sha(Path(__file__)), payload_sha256=sha(payload_path),
        array_sha256=payload['arrays_sha256'], source_sha256=payload['source_sha256'],
        timings_seconds=dict(imports_and_payload_arrays=import_seconds, data_and_scaler=data_seconds,
            fit_and_checkpoint_IO=fit_seconds, FP32_conversion=export['FP32_seconds'],
            INT8_conversion=export['INT8_seconds'], training_only_inference_and_IO=inference_seconds),
        elapsed_worker_seconds=time.perf_counter() - started_worker)
    write(directory / 'RESULTADO.json', result)
    event('PASS', epochs=job['epochs'], outer_evaluation_executed=False)
    write_manifest(directory / 'MANIFEST_JOB_SHA256.json', directory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--payload', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args.payload, args.output)
    except Exception:
        print(traceback.format_exc(), file=sys.stderr, flush=True)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
