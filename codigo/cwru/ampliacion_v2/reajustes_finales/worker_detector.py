"""Ajuste final K=1 con normales de desarrollo, sin evaluación exterior.

Reutiliza fit_normality/detector_score del núcleo auditado y no reajusta tras
calibrar el umbral. La importación no importa sklearn ni ejecuta ajustes.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
for directory in ('busqueda_interna', 'piloto_clasicos'):
    path = str(ROOT / 'codigo/cwru/ampliacion_v2' / directory)
    if path not in sys.path:
        sys.path.insert(0, path)
import ejecutar_busqueda_interna as io
from detector_numerico import fit_normality, detector_score

ARRAY_KEYS = {'raw_fit', 'raw_calibration', 'y_fit', 'y_calibration',
              'fit_ids', 'calibration_ids', 'fit_records', 'calibration_records'}
PARAMETER_ARRAYS = ('indices', 'mean64', 'scale64', 'var64', 'mean32', 'scale32',
                    'centroid32', 'threshold32', 'fit_ids', 'calibration_ids')


def versions():
    return dict(python=platform.python_version(), **{
        name: importlib.metadata.version(package)
        for name, package in [('numpy', 'numpy'), ('sklearn', 'scikit-learn')]})


def save_npz(path, **arrays):
    with Path(path).open('xb') as stream:
        np.savez(stream, **arrays)


def parameter_payload_sha256(path):
    """La misma identidad de arrays canónicos del detector interno auditado."""
    digest = hashlib.sha256()
    with np.load(path, allow_pickle=False) as saved:
        io.require(set(saved.files) == set(PARAMETER_ARRAYS), 'Parámetros incompletos')
        for key in sorted(PARAMETER_ARRAYS):
            array = saved[key]
            io.require(array.dtype.kind != 'O', 'No se permiten objetos en parámetros')
            digest.update(key.encode('utf-8'))
            digest.update(str(array.dtype).encode('ascii'))
            digest.update(str(array.shape).encode('ascii'))
            digest.update(array.tobytes(order='C'))
    return digest.hexdigest()


def validate_payload(payload, arrays=None):
    """Comprueba normales y disjunción fit/calibración sin ajustar el núcleo."""
    io.require(payload.get('status') == 'AUTHORIZED_FINAL_REFIT_PAYLOAD'
        and payload.get('family') == 'detector', 'Payload no autorizado del detector')
    job = payload['job']; outer = job['outer_load_hp']; roles = job['roles']
    io.require(job['family'] == 'detector' and type(outer) is int and outer in range(4),
        'Trabajo o carga exterior inválidos')
    loads = [i for i in range(4) if i != outer]
    io.require(all(type(i) is int for key in ('training_loads', 'evaluation_loads',
            'detector_fit_normal_loads', 'detector_calibration_normal_loads') for i in roles[key])
        and type(roles['outer_load_hp']) is int
        and roles['training_loads'] == loads and roles['evaluation_loads'] == [outer]
        and roles['outer_load_hp'] == outer and roles['inner_validation_load_hp'] is None
        and roles['detector_calibration_normal_loads'] == [max(loads)]
        and roles['detector_fit_normal_loads'] == [i for i in loads if i != max(loads)]
        and roles['detector_refit_after_calibration'] is False,
        'Roles de normales distintos del protocolo final')
    indices = job['indices']
    io.require(isinstance(indices, list) and indices
        and all(type(i) is int and 0 <= i < 14 and i not in (4, 11) for i in indices)
        and len(indices) == len(set(indices)), 'Índices efectivos del detector inválidos')
    if arrays is None:
        with np.load(payload['arrays_path'], allow_pickle=False) as saved:
            arrays = {key: saved[key].copy() for key in saved.files}
    io.require(set(arrays) == ARRAY_KEYS, 'NPZ detector debe contener solo fit/calibración')
    records_by_scope = {}
    for scope, raw_key, y_key, role_key in (
        ('fit', 'raw_fit', 'y_fit', 'detector_fit_normal_loads'),
        ('calibration', 'raw_calibration', 'y_calibration', 'detector_calibration_normal_loads')):
        ids_key = scope + '_ids'; records_key = scope + '_records'
        ids = job[ids_key]
        io.require(isinstance(ids, list) and ids and all(type(i) is int and i >= 0 for i in ids)
            and ids == sorted(set(ids)), 'IDs del detector no canónicos: ' + scope)
        size = len(ids); raw = arrays[raw_key]; y = arrays[y_key]
        io.require(raw.dtype == np.float32 and raw.shape == (size, 14) and np.isfinite(raw).all(),
            'Detector requiere matrices completas float32 finitas: ' + scope)
        io.require(y.dtype == np.int64 and y.shape == (size,) and np.all(y == 0),
            'Solo normales pueden ajustar o calibrar: ' + scope)
        io.require(arrays[ids_key].dtype == np.int64 and arrays[ids_key].shape == (size,)
            and arrays[ids_key].tolist() == ids, 'IDs no corresponden al scope firmado: ' + scope)
        records = arrays[records_key]
        io.require(records.dtype.kind == 'U' and records.shape == (size,)
            and all(v and v.strip() == v for v in records.tolist()), 'MAT del detector inválidos')
        records_by_scope[scope] = set(records.tolist())
        expected_records = 2 if scope == 'fit' else 1
        provenance = payload['provenance'][scope]
        io.require(len(records_by_scope[scope]) == expected_records
            and provenance['global_window_ids'] == ids
            and provenance['selected_loads_hp'] == roles[role_key]
            and all(type(i) is int for i in provenance['selected_loads_hp'])
            and provenance['selected_record_ids'] == sorted(records_by_scope[scope], key=int)
            and provenance['shape'] == [size, 14] and provenance['loaded_dtype'] == 'float32'
            and provenance['only_requested_rows_selected_or_materialized'] is True,
            'Procedencia o número de normales distinto: ' + scope)
    io.require(not set(job['fit_ids']) & set(job['calibration_ids'])
        and records_by_scope['fit'].isdisjoint(records_by_scope['calibration']),
        'Ajuste y calibración del detector comparten IDs o MAT')
    return arrays


def run(payload_path, output):
    from contrato import authorize
    started = time.monotonic()
    payload = authorize(payload_path, output, 'detector')
    observed = versions()
    io.require(all(payload['expected_versions'].get(k) == v for k, v in observed.items()),
        'Versiones distintas del detector preparado')
    arrays = validate_payload(payload)
    output = Path(output).resolve(); output.mkdir(parents=False, exist_ok=False)
    job = payload['job']
    fit_start = time.monotonic()
    params = fit_normality(arrays['raw_fit'], arrays['raw_calibration'],
        arrays['calibration_records'].tolist(), job['indices'])
    fit_seconds = time.monotonic() - fit_start
    save_npz(output / 'parametros.npz', indices=np.asarray(params['indices'], dtype=np.int64),
        **{k: params[k] for k in ('mean64', 'scale64', 'var64', 'mean32', 'scale32',
                                  'centroid32', 'threshold32')},
        fit_ids=arrays['fit_ids'], calibration_ids=arrays['calibration_ids'])
    # Se conserva la calibración para reconstruir el P95, sin evaluar la carga exterior.
    scores = detector_score(arrays['raw_calibration'], params)
    save_npz(output / 'calibracion.npz', global_window_ids=arrays['calibration_ids'],
        record_ids=arrays['calibration_records'], scores=scores,
        rejected=scores > params['threshold32'])
    io.write(output / 'procedencia.json', payload['provenance'])
    result = dict(status='PASS', job_id=job['job_id'], family='detector',
        outer_load_hp=job['outer_load_hp'], roles=job['roles'], indices=job['indices'],
        masks=job['masks'], procedures=job['procedures'],
        fit_signature=payload['fit_signature'], selection_signature=payload['selection_signature'],
        source_sha256=payload['source_sha256'], versions=observed,
        fit_ids=job['fit_ids'], calibration_ids=job['calibration_ids'],
        fit_rows=len(arrays['fit_ids']), calibration_rows=len(arrays['calibration_ids']),
        fit_normal_records=2, calibration_normal_records=1,
        threshold_bits=np.float32(params['threshold32']).tobytes().hex(),
        threshold_float32=float(params['threshold32']),
        parameter_payload_sha256=parameter_payload_sha256(output / 'parametros.npz'),
        new_parameter_fits=1, normal_only_fit=True, normal_only_calibration=True,
        refit_after_calibration=False, dnn_scaler_reused=False, class_override=False,
        outer_rows_materialized=False, outer_evaluation_executed=False, selection_executed=False,
        TensorFlow_imported='tensorflow' in sys.modules, STM32_cost_measured=False,
        fit_seconds=fit_seconds, worker_elapsed_seconds=time.monotonic() - started)
    io.require(result['TensorFlow_imported'] is False, 'Runtime DNN no permitido en detector')
    io.write(output / 'RESULTADO.json', result)
    io.write_manifest(output / 'MANIFEST_JOB_SHA256.json', directory=output)
    print(json.dumps(dict(status='PASS', job_id=job['job_id'], family='detector',
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
