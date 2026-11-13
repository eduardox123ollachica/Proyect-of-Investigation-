"""Datos del suplemento de desarrollo P0/P1: 36 ajustes, outer3 excluido.

No ajusta estimadores ni transforma magnitudes. Deriva los roles desde el indice
V2 sellado y materializa solo ajuste/validacion interna de cargas0/1/2. Nunca
abre el cache test historico (3HP), RAW, MAT o etiquetas NPY. La comprobacion de
hashes/mapas de archivos no equivale a una traza de accesos del sistema operativo.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
REPRESENTATIVES = ROOT / 'codigo/cwru/ampliacion_v2/representantes'
if str(REPRESENTATIVES) not in sys.path:
    sys.path.insert(0, str(REPRESENTATIVES))
from datos_representantes import (
    EXPECTED_PROTOCOL_SHA256, FEATURE_BASE, PARTITIONS_REL,
    _validate_index, bounded_path, component_roles, load_verified_context, sha256,
)

MODES = ('P0', 'P1')
MASKS = ('F14', 'NF12')
SEEDS = (11, 22, 33)
INNER_LOADS = (0, 1, 2)
OUTER_LOAD = 3
ALLOWED_FEATURES = {FEATURE_BASE + 'X_features_' + name + '.npy'
                    for name in ('train', 'validation')}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def context():
    """Retorna (protocolo, filas CSV, hashes) tras verificar sello/metadatos."""
    protocol, _, rows, hashes = load_verified_context()
    require(protocol['dnn']['inner_seeds'] == list(SEEDS)
            and protocol['dnn']['max_epochs'] == 50
            and protocol['scope']['loads_hp'] == [0, 1, 2, 3], 'Protocolo V2 incompatible')
    return protocol, rows, hashes


def _select(rows, loads, normal_only=False):
    return [r for r in rows if int(r['load_hp']) in loads
            and (not normal_only or r['class'] == 'N')]


def _expected_job(protocol, rows, inner, candidate, mode, seed):
    require(type(inner) is int and inner in INNER_LOADS, 'Validacion fuera de desarrollo0/1/2')
    require(candidate in MASKS and mode in MODES and type(seed) is int and seed in SEEDS,
            'Mascara, modo o semilla fuera del suplemento')
    roles = component_roles(OUTER_LOAD, inner)
    train = _select(rows, roles['training_loads'])
    validation = _select(rows, [inner])
    fit = _select(rows, roles['detector_fit_normal_loads'], True)
    calibration = _select(rows, roles['detector_calibration_normal_loads'], True)
    indices = protocol['candidates'][candidate]
    require(indices == (list(range(14)) if candidate == 'F14' else [0,1,2,3,5,6,7,8,9,10,12,13]),
            'Indices originales modificados')
    return dict(job_id=f'supp_o3_i{inner}_{candidate}_{mode}_s{seed}', candidate=candidate, mode=mode,
                feature_indices=indices.copy(), input_dimension=len(indices), outer_load_hp=OUTER_LOAD,
                inner_validation_load_hp=inner, seed=seed, roles=roles,
                training_global_window_ids=[int(r['global_window_id']) for r in train],
                validation_global_window_ids=[int(r['global_window_id']) for r in validation],
                detector_fit_global_window_ids=[int(r['global_window_id']) for r in fit],
                detector_calibration_global_window_ids=[int(r['global_window_id']) for r in calibration],
                training_loads=roles['training_loads'].copy(), validation_loads=[inner])


def validate_job(job, rows=None, protocol=None):
    """Exige igualdad exacta al alcance derivado; firmas/limites fuera del job.

    Devuelve (filas de ajuste, filas de validacion), ambas en orden global.
    """
    if rows is None or protocol is None:
        require(rows is None and protocol is None, 'Indicar juntos protocolo y metadatos')
        protocol, rows, _ = context()
    _validate_index(rows)
    require(isinstance(job, dict), 'Job debe ser un diccionario')
    expected = _expected_job(protocol, rows, job.get('inner_validation_load_hp'),
                             job.get('candidate'), job.get('mode'), job.get('seed'))
    require(job == expected, 'Job distinto del alcance de desarrollo sellado')
    train_ids, val_ids = job['training_global_window_ids'], job['validation_global_window_ids']
    fit_ids, cal_ids = job['detector_fit_global_window_ids'], job['detector_calibration_global_window_ids']
    require(not set(train_ids) & set(val_ids) and not set(fit_ids) & set(cal_ids)
            and set(fit_ids + cal_ids) <= set(train_ids), 'Scopes solapados')
    train, validation = [rows[k] for k in train_ids], [rows[k] for k in val_ids]
    require({r['file_id'] for r in train}.isdisjoint(r['file_id'] for r in validation)
            and len({r['file_id'] for r in train}) == 14
            and len({r['file_id'] for r in validation}) == 7, 'MAT mezclado o incompleto')
    require(all(int(r['load_hp']) != OUTER_LOAD and r['source_features'] in ALLOWED_FEATURES
                and r['source_split_v1'] in ('train', 'validation') for r in train + validation),
            'Fila de3HP o cache test fuera del suplemento')
    require(all(rows[k]['class'] == 'N' for k in fit_ids + cal_ids), 'Detector incluye fallas')
    return train, validation


def jobs():
    """36 casos: inner, mascara, semilla, modo. Los dos primeros son la puerta."""
    protocol, rows, _ = context()
    result = [_expected_job(protocol, rows, inner, candidate, mode, seed)
              for inner in INNER_LOADS for candidate in MASKS for seed in SEEDS for mode in MODES]
    require(len(result) == len({j['job_id'] for j in result}) == 36, 'Cantidad de jobs incorrecta')
    for job in result:
        validate_job(job, rows, protocol)
    return result


def load_data(job):
    """Retorna raw14f32,y_train,raw_val14f32,y_val,procedencia; sin preprocesar.

    P0/P1 no cambian las filas o columnas fuente. La seleccion de mascara y la
    transformacion train-only se realizan despues, por el worker del suplemento.
    """
    protocol, rows, hashes = context()
    train, validation = validate_job(job, rows, protocol)
    relatives = sorted({r['source_features'] for r in train + validation})
    require(set(relatives) == ALLOWED_FEATURES, 'Fuentes distintas de train/validation historicos')
    arrays, sources = {}, []
    for relative in relatives:
        path = bounded_path(ROOT, relative)
        require(relative in hashes and sha256(path) == hashes[relative], 'Cache modificado: ' + relative)
        array = np.load(path, allow_pickle=False, mmap_mode='r')
        selected_source_rows = [r for r in rows if r['source_features'] == relative]
        require(array.dtype == np.float64 and array.shape == (len(selected_source_rows), 14),
                'Dtype/forma del cache distintos')
        arrays[relative] = array
        sources.append(dict(path=relative, sha256=hashes[relative], shape=list(array.shape),
                            dtype='float64', mmap_mode='r', contains_outer3_rows=False))

    def materialize(selected):
        raw = np.asarray([arrays[r['source_features']][int(r['source_row'])] for r in selected], dtype=np.float64)
        require(np.isfinite(raw).all(), 'Filas fuente seleccionadas no finitas')
        with np.errstate(over='ignore', invalid='ignore'):
            x = raw.astype(np.float32)
        require(np.isfinite(x).all(), 'Cast float32 no finito')
        return x, np.asarray([int(r['label']) for r in selected], dtype=np.int64)

    x, y = materialize(train)
    val, val_y = materialize(validation)
    provenance = dict(job_id=job['job_id'], source_arrays=sources,
        source_index_sha256=sha256(ROOT / PARTITIONS_REL / 'indice_ventanas_fuente.csv'),
        protocol_manifest_sha256=EXPECTED_PROTOCOL_SHA256,
        train_global_window_ids=job['training_global_window_ids'],
        validation_global_window_ids=job['validation_global_window_ids'],
        detector_fit_global_window_ids=job['detector_fit_global_window_ids'],
        detector_calibration_global_window_ids=job['detector_calibration_global_window_ids'],
        train_file_ids=sorted({r['file_id'] for r in train}, key=int),
        validation_file_ids=sorted({r['file_id'] for r in validation}, key=int),
        roles=job['roles'], train_shape=list(x.shape), validation_shape=list(val.shape),
        source_dtype='float64', loaded_dtype='float32', label_dtype='int64',
        label_source='sealed CSV metadata, no label NPY opened',
        arrays_mapped=relatives, outer3_arrays_opened=False, outer_rows_selected_or_materialized=False,
        raw_signals_or_MAT_opened=False, mapping_or_hashing_is_not_OS_access_trace=True,
        preprocessing_applied=False)
    return x, y, val, val_y, provenance
