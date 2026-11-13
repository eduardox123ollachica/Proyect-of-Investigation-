"""Lectura acotada del piloto V2; sin TensorFlow, entrenamiento ni escrituras.

Solo las dos matrices historicas de caracteristicas autorizadas por el plan
pueden abrirse. Los IDs y etiquetas proceden de metadatos sellados. Verificar
archivos no convierte este piloto retrospectivo en una evaluacion confirmatoria.
"""
from __future__ import annotations

from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
PROTOCOL = ROOT / 'articulo/ampliacion_cwru/metodologia/protocolo_v2_2026-10-25'
PARTITIONS = ROOT / 'datos/cwru/particiones/ampliacion_v2/por_carga_2026-10-22'
FEATURE_BASE = 'datos/cwru/procesado/dual_12khz_w4096/features_14/'
ALLOWED_FEATURES = {FEATURE_BASE + 'X_features_train.npy', FEATURE_BASE + 'X_features_validation.npy'}
MASKS = {'F14': list(range(14)), 'T6': [0, 1, 2, 7, 8, 9]}
LABELS = {'N': 0, 'IR': 1, 'OR': 2}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def bounded_feature_path(relative, root=ROOT):
    require(isinstance(relative, str) and relative in ALLOWED_FEATURES,
            'Matriz no autorizada: solo train y validation del plan')
    path = (root / relative).resolve()
    require(path.is_relative_to(root.resolve()) and path != root.resolve(), 'Ruta fuera del proyecto')
    return path


def _bounded_metadata(relative):
    require(isinstance(relative, str), 'Ruta de metadatos invalida')
    path = (ROOT / relative).resolve()
    require(path != ROOT and path.is_relative_to(ROOT), 'Recurso fuera del proyecto')
    require(path.suffix.lower() in ('.csv', '.json', '.md', '.py'), 'Recurso ajeno al sello de protocolo')
    return path


def load_verified_plan():
    """Verifica el sello y sus treinta recursos; devuelve el plan historico."""
    manifest = PROTOCOL / 'MANIFEST_PROTOCOLO_SHA256.json'
    words = manifest.with_suffix('.sha256').read_text(encoding='ascii').split()
    require(len(words) == 2 and words[1] == manifest.name and words[0] == sha256(manifest),
            'Sello de protocolo no coincide')
    entries = read_json(manifest)
    require(isinstance(entries, dict) and len(entries) == 30, 'Se esperaban treinta recursos sellados')
    # El sello incluye un .sha256 del cierre historico de costos.
    for name, digest in entries.items():
        path = (ROOT / name).resolve()
        require(path != ROOT and path.is_relative_to(ROOT), 'Ruta sellada fuera del proyecto')
        require(path.suffix.lower() in ('.csv', '.json', '.md', '.py', '.sha256', '.txt'), 'Extension sellada inesperada')
        require(sha256(path) == digest, 'Recurso del protocolo modificado: ' + name)
    plan_path = PROTOCOL / 'piloto_plan.json'
    require(plan_path.relative_to(ROOT).as_posix() in entries, 'Plan fuera del sello')
    plan = read_json(plan_path)
    require(plan['status'] == 'PILOT_PREPARED_NOT_EXECUTED' and plan['mass_training_enabled'] is False
            and plan['outer_predictive_evaluation_enabled'] is False
            and plan['maximum_DNN_fits'] == 2 and plan['maximum_model_conversions'] == 4,
            'Plan de piloto incompatible')
    require(len(plan['jobs']) == 2 and [j['mask'] for j in plan['jobs']] == ['F14', 'T6'],
            'Se requieren exactamente los dos trabajos previstos')
    return plan


def _integer(value, name):
    require(type(value) is int, name + ' debe ser entero')
    return value


def _scope_ids(scope, expected_loads, expected_windows):
    require(isinstance(scope, dict) and scope['loads_hp'] == expected_loads
            and scope['normal_only'] is False and scope['windows'] == expected_windows,
            'Scope de cargas/cantidad alterado')
    ids = scope['global_window_ids']
    require(isinstance(ids, list) and len(ids) == expected_windows, 'Cantidad de IDs invalida')
    ids = [_integer(v, 'global_window_id') for v in ids]
    require(len(set(ids)) == len(ids) and ids == sorted(ids), 'IDs duplicados o desordenados')
    return ids


def validate_job_scope(job, rows):
    """Valida metadata pura. Retorna filas train/val en el orden global fijado.

    Requiere todos y solo los IDs de cargas 1/2 para ajuste y 0 para validacion.
    Los metadatos pueden incluir 3 HP, pero sus matrices nunca se abren.
    """
    require(isinstance(job, dict) and job['mask'] in MASKS, 'Mascara fuera del piloto')
    require(job['feature_indices'] == MASKS[job['mask']]
            and job['input_dimension'] == len(MASKS[job['mask']]), 'Indices de mascara alterados')
    require(job['seed'] == 11 and type(job['seed']) is int and job['fixed_epochs'] == 50
            and job['early_stopping'] is False and job['outer_load_hp'] == 3
            and job['inner_validation_load_hp'] == 0 and job['timeout_worker_seconds'] == 300,
            'Semilla, cargas o presupuesto alterados')
    require(job['runtime_and_finiteness_diagnostics_only'] is True
            and job['metrics_used_for_selection'] is False and job['reuse_as_predictive_V2_fit'] is False,
            'No convertir el piloto en resultado predictivo')
    train_ids = _scope_ids(job['train'], [1, 2], 799)
    val_ids = _scope_ids(job['validation'], [0], 175)
    require(set(train_ids).isdisjoint(val_ids), 'Validacion mezclada con ajuste')
    require(job['representative_int8_calibration'] == job['train'], 'Calibracion INT8 fuera de entrenamiento')
    require(set(job['authorized_feature_source_paths']) == ALLOWED_FEATURES
            and len(job['authorized_feature_source_paths']) == 2, 'Fuentes autorizadas alteradas')
    index = {}
    for row in rows:
        identifier = int(row['global_window_id'])
        require(identifier not in index, 'ID global duplicado en metadatos')
        require(row['class'] in LABELS and int(row['label']) == LABELS[row['class']], 'Etiqueta inconsistente')
        require(int(row['load_hp']) in (0, 1, 2, 3), 'Carga no definida')
        index[identifier] = row
    train_expected = sorted(k for k, r in index.items() if int(r['load_hp']) in (1, 2))
    val_expected = sorted(k for k, r in index.items() if int(r['load_hp']) == 0)
    require(train_ids == train_expected and val_ids == val_expected, 'IDs distintos del alcance autorizado')
    chosen = []
    for name, ids, loads in (('train', train_ids, (1, 2)), ('validation', val_ids, (0,))):
        selected = [index[k] for k in ids]
        require(all(int(r['load_hp']) in loads and r['source_features'] in ALLOWED_FEATURES
                    and r['source_split_v1'] in ('train', 'validation') and int(r['source_row']) >= 0
                    and r['source_features'] == FEATURE_BASE + 'X_features_' + r['source_split_v1'] + '.npy'
                    for r in selected), 'Fila, matriz o carga ajena al piloto')
        pairs = [(r['source_features'], int(r['source_row'])) for r in selected]
        require(len(set(pairs)) == len(pairs), 'Fila de matriz repetida con IDs distintos')
        require(sorted({str(r['file_id']) for r in selected}, key=int) == job[name]['file_ids'],
                'Lista MAT distinta del scope')
        counts = Counter(r['class'] for r in selected)
        require({k: counts[k] for k in LABELS} == job[name]['class_window_counts']
                and set(counts) == set(LABELS), 'Conteo de clases distinto')
        chosen.append(selected)
    require({r['file_id'] for r in chosen[0]}.isdisjoint(r['file_id'] for r in chosen[1]),
            'MAT compartido entre ajuste y validacion')
    return tuple(chosen)


def load_job_data(job):
    """Carga dos caches autorizados y devuelve f32 sin escalar, y/procedencia.

    La seleccion de columnas y el ajuste del scaler son responsabilidad del
    worker. Nunca abre etiquetas .npy: las reconstruye del indice sellado.
    """
    plan = load_verified_plan()
    canonical = next((j for j in plan['jobs'] if j['job_id'] == job.get('job_id')), None)
    require(canonical is not None and json.dumps(job, sort_keys=True) == json.dumps(canonical, sort_keys=True),
            'Trabajo distinto del plan sellado')
    rows = read_csv(PARTITIONS / 'indice_ventanas_fuente.csv')
    require(len(rows) == 1380, 'Indice historico debe contener 1380 ventanas')
    train_rows, val_rows = validate_job_scope(job, rows)
    hashes = {r['path']: r['sha256'] for r in read_csv(PARTITIONS / 'entradas_sha256.csv')}
    arrays, sources = {}, []
    for relative in sorted(ALLOWED_FEATURES):
        path = bounded_feature_path(relative)
        require(relative in hashes and sha256(path) == hashes[relative], 'Cache de features modificado: ' + relative)
        array = np.load(path, allow_pickle=False)
        require(array.ndim == 2 and array.shape[1] == 14 and array.dtype == np.float64
                and np.isfinite(array).all(), 'Cache no finito o tipo/forma inesperado')
        expected_rows = [r for r in rows if r['source_features'] == relative]
        require(len(expected_rows) == len(array)
                and sorted(int(r['source_row']) for r in expected_rows) == list(range(len(array))),
                'Filas del cache no coinciden con metadata')
        arrays[relative] = array
        sources.append(dict(path=relative, sha256=hashes[relative], shape=list(array.shape), dtype=str(array.dtype)))
    def materialize(selected):
        x = np.asarray([arrays[r['source_features']][int(r['source_row'])] for r in selected], dtype=np.float32)
        y = np.asarray([int(r['label']) for r in selected], dtype=np.int64)
        require(np.isfinite(x).all(), 'Cast float32 produjo no finitos')
        return x, y
    xtrain, ytrain = materialize(train_rows)
    xval, yval = materialize(val_rows)
    provenance = dict(source_arrays=sources, source_index_sha256=sha256(PARTITIONS / 'indice_ventanas_fuente.csv'),
        protocol_manifest_sha256=sha256(PROTOCOL / 'MANIFEST_PROTOCOLO_SHA256.json'),
        job_id=job['job_id'], train_global_window_ids=job['train']['global_window_ids'],
        validation_global_window_ids=job['validation']['global_window_ids'],
        train_shape=list(xtrain.shape), validation_shape=list(xval.shape), loaded_dtype='float32',
        label_dtype='int64', label_source='sealed CSV metadata, no label arrays opened',
        arrays_opened=[s['path'] for s in sources], load3_arrays_opened=False,
        raw_signals_or_MAT_opened=False)
    return xtrain, ytrain, xval, yval, provenance
