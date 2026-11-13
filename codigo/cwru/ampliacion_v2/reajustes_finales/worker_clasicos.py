"""Reajuste RF/SVM congelado con tres cargas de desarrollo, sin test exterior.

Importar este módulo no importa sklearn/joblib ni ajusta estimadores. El
controlador autoriza un payload firmado antes de materializar sus arrays.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
for directory in ('busqueda_interna', 'representantes'):
    path = str(ROOT / 'codigo/cwru/ampliacion_v2' / directory)
    if path not in sys.path:
        sys.path.insert(0, path)
import ejecutar_busqueda_interna as io
from numerica_representantes import balanced_weights, normalize_f32

ARRAY_KEYS = {'raw_training', 'y_training', 'training_ids', 'training_records'}


def versions():
    return dict(python=platform.python_version(), **{
        name: importlib.metadata.version(package)
        for name, package in [('numpy', 'numpy'), ('sklearn', 'scikit-learn'),
                              ('scipy', 'scipy'), ('joblib', 'joblib')]})


def save_npz(path, **arrays):
    with Path(path).open('xb') as stream:
        np.savez(stream, **arrays)


def _records(values, size):
    io.require(isinstance(values, np.ndarray) and values.dtype.kind == 'U'
        and values.shape == (size,) and all(v and v.strip() == v for v in values.tolist()),
        'Identificadores de MAT no canónicos')
    return values.tolist()


def validate_payload(payload, arrays=None):
    """Validaciones numéricas/de scope; no ajusta o predice con modelos.

    ``authorize`` autentica la selección, el trabajo y el NPZ antes de llamar
    aquí. El argumento arrays permite probar las guardas con datos sintéticos.
    """
    io.require(payload.get('status') == 'AUTHORIZED_FINAL_REFIT_PAYLOAD'
        and payload.get('family') in ('RF', 'SVM'), 'Payload clásico no autorizado')
    job = payload['job']; family = payload['family']; outer = job['outer_load_hp']
    io.require(job['family'] == family and type(outer) is int and outer in range(4),
        'Familia o carga exterior inválida')
    roles = job['roles']; loads = [i for i in range(4) if i != outer]
    io.require(all(type(i) is int for i in roles['training_loads'] + roles['evaluation_loads'])
        and type(roles['outer_load_hp']) is int
        and roles['training_loads'] == loads and roles['evaluation_loads'] == [outer]
        and roles['outer_load_hp'] == outer and roles['inner_validation_load_hp'] is None,
        'El reajuste debe usar exactamente las tres cargas de desarrollo')
    ids = job['training_global_window_ids']
    io.require(isinstance(ids, list) and ids and all(type(i) is int and i >= 0 for i in ids)
        and ids == sorted(set(ids)), 'IDs de entrenamiento no canónicos')
    if arrays is None:
        with np.load(payload['arrays_path'], allow_pickle=False) as saved:
            arrays = {key: saved[key].copy() for key in saved.files}
    io.require(set(arrays) == ARRAY_KEYS, 'NPZ clásico debe contener solo entrenamiento')
    raw = arrays['raw_training']; y = arrays['y_training']; size = len(ids)
    io.require(raw.dtype == np.float32 and raw.shape == (size, 14)
        and np.isfinite(raw).all(), 'Reajuste clásico requiere F14 float32 finito')
    io.require(y.dtype == np.int64 and y.shape == (size,) and set(y.tolist()) == {0, 1, 2},
        'Etiquetas de entrenamiento deben contener las tres clases')
    io.require(arrays['training_ids'].dtype == np.int64
        and arrays['training_ids'].shape == (size,)
        and arrays['training_ids'].tolist() == ids, 'NPZ no corresponde a los IDs firmados')
    records = _records(arrays['training_records'], size)
    provenance = payload['provenance']['training']
    io.require(provenance['global_window_ids'] == ids and provenance['selected_loads_hp'] == loads
        and all(type(i) is int for i in provenance['selected_loads_hp'])
        and provenance['selected_record_ids'] == sorted(set(records), key=int)
        and len(set(records)) == 21 and provenance['shape'] == [size, 14]
        and provenance['loaded_dtype'] == 'float32'
        and provenance['only_requested_rows_selected_or_materialized'] is True,
        'Procedencia de entrenamiento incompatible con sus cargas o MAT')
    params = job['params']
    io.require(params['class_weight'] == 'balanced', 'Pesos de clase ajenos al protocolo')
    if family == 'RF':
        io.require(type(job['seed']) is int and job['seed'] in (11, 22, 33, 42, 44, 55, 66, 77, 88, 99)
            and type(params['random_state']) is int and params['random_state'] == job['seed']
            and params['n_jobs'] == 1, 'Semilla RF o paralelismo distintos')
    else:
        io.require(job['seed'] is None and params['kernel'] == 'rbf'
            and params['probability'] is False and params['break_ties'] is False,
            'SVM debe conservar la predicción nativa y su ajuste determinista')
    return arrays


def prepare_training(family, raw):
    io.require(raw.dtype == np.float32 and raw.ndim == 2 and raw.shape[1] == 14
        and len(raw) > 0 and np.isfinite(raw).all(), 'Matriz de entrenamiento inválida')
    if family == 'RF':
        return raw, None
    io.require(family == 'SVM', 'Familia clásica desconocida')
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(raw)
    arrays = dict(mean64=np.asarray(scaler.mean_, dtype=np.float64),
        scale64=np.asarray(scaler.scale_, dtype=np.float64),
        var64=np.asarray(scaler.var_, dtype=np.float64),
        mean32=scaler.mean_.astype(np.float32), scale32=scaler.scale_.astype(np.float32))
    return normalize_f32(raw, arrays['mean32'], arrays['scale32']), arrays


def make_estimator(job):
    if job['family'] == 'RF':
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(**job['params'])
    io.require(job['family'] == 'SVM', 'Familia clásica desconocida')
    from sklearn.svm import SVC
    return SVC(**job['params'])


def training_prediction(model, family, x):
    """Solo diagnóstico técnico de train; SVC.predict mantiene la votación."""
    prediction = model.predict(x)
    values = model.predict_proba(x) if family == 'RF' else model.decision_function(x)
    io.require(prediction.dtype == np.int64 and prediction.shape == (len(x),)
        and set(prediction.tolist()) <= {0, 1, 2} and values.shape == (len(x), 3)
        and np.isfinite(values).all(), 'Diagnóstico de entrenamiento inválido')
    return prediction, values


def run(payload_path, output):
    # La autorización comprueba plan, solicitud activa, hashes, entorno y NPZ.
    from contrato import authorize
    started = time.monotonic()
    header = io.read(payload_path)
    family = header.get('family')
    io.require(family in ('RF', 'SVM'), 'Familia no clásica')
    payload = authorize(payload_path, output, family)
    observed = versions()
    io.require(all(payload['expected_versions'].get(k) == v for k, v in observed.items()),
        'Versiones distintas del entorno preparado')
    arrays = validate_payload(payload)
    output = Path(output).resolve(); output.mkdir(parents=False, exist_ok=False)
    job = payload['job']; raw = arrays['raw_training']; y = arrays['y_training']
    x, scaler = prepare_training(family, raw)
    if scaler is not None:
        save_npz(output / 'scaler_train_only.npz', **scaler)
    counts, weights, _ = balanced_weights(y)
    model = make_estimator(job)
    fit_start = time.monotonic(); model.fit(x, y); fit_seconds = time.monotonic() - fit_start
    io.require(np.array_equal(model.classes_, [0, 1, 2]) and model.n_features_in_ == 14,
        'Clases o dimensión del estimador distintas')
    if family == 'RF':
        io.require(len(model.estimators_) == job['params']['n_estimators'], 'RF incompleto')
    else:
        io.require(model.fit_status_ == 0, 'SVM no convergió')
    prediction_start = time.monotonic()
    predictions, values = training_prediction(model, family, x)
    prediction_seconds = time.monotonic() - prediction_start
    save_npz(output / 'diagnostico_entrenamiento.npz',
        global_window_ids=arrays['training_ids'], y_true=y, predictions=predictions, values=values)
    import joblib
    with (output / 'modelo.joblib').open('xb') as stream:
        joblib.dump(model, stream, compress=3)
    io.write(output / 'procedencia.json', payload['provenance'])
    result = dict(status='PASS', job_id=job['job_id'], family=family,
        outer_load_hp=job['outer_load_hp'], roles=job['roles'], seed=job['seed'],
        config_index=job['config_index'], params=job['params'],
        fit_signature=payload['fit_signature'], selection_signature=payload['selection_signature'],
        source_sha256=payload['source_sha256'], versions=observed,
        training_windows=len(y), training_global_window_ids=job['training_global_window_ids'],
        input_dimension=14, model_classes=model.classes_.tolist(), class_counts=counts.tolist(),
        balanced_class_weights=weights.tolist(),
        normalization='raw_float32' if family == 'RF' else 'train_only_scaler_explicit_float32',
        native_predict_rule=True, diagnostic_scope='training_only_not_generalization_metrics',
        model_sha256=io.sha(output / 'modelo.joblib'), model_bytes=(output / 'modelo.joblib').stat().st_size,
        support_vectors_dtype=str(model.support_vectors_.dtype) if family == 'SVM' else None,
        new_model_fits=1, new_training_prediction_batches=1,
        outer_rows_materialized=False, outer_evaluation_executed=False,
        selection_executed=False, early_stopping=False, TensorFlow_imported='tensorflow' in sys.modules,
        STM32_cost_measured=False, model_joblib_bytes_are_not_Flash_usage=True,
        fit_seconds=fit_seconds, training_prediction_seconds=prediction_seconds,
        worker_elapsed_seconds=time.monotonic() - started)
    io.require(result['TensorFlow_imported'] is False, 'Runtime DNN no permitido en clásicos')
    io.write(output / 'RESULTADO.json', result)
    io.write_manifest(output / 'MANIFEST_JOB_SHA256.json', directory=output)
    print(json.dumps(dict(status='PASS', job_id=job['job_id'], family=family,
        fit_seconds=fit_seconds, outer_evaluation_executed=False)), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--payload', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    run(args.payload, args.output)


if __name__ == '__main__':
    main()
