"""Datos de los 32 representantes de costo V2, con roles y entradas selladas.

No entrena ni escribe. Los cachés NPY son almacenamiento histórico y pueden
contener filas de cargas excluidas. Se verifica su hash y se mapean en lectura;
solo se seleccionan/materializan las filas de ajuste y validación interna del
trabajo. Esto no equivale a una traza de todos los accesos del sistema operativo.
No se leen señales RAW, MAT, etiquetas NPY o predicciones externas.
"""
from __future__ import annotations

from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
PROTOCOL_REL = Path('articulo/ampliacion_cwru/metodologia/protocolo_v2_2026-10-25')
PARTITIONS_REL = Path('datos/cwru/particiones/ampliacion_v2/por_carga_2026-10-22')
EXPECTED_PROTOCOL_SHA256 = 'ce556659026f9eaa06b7619e6f3409b244ddeafc9fe534f40619b65911af68e0'
FEATURE_BASE = 'datos/cwru/procesado/dual_12khz_w4096/features_14/'
ALLOWED_FEATURES = {FEATURE_BASE + 'X_features_' + split + '.npy'
                    for split in ('train', 'validation', 'test')}
LABELS = {'N': 0, 'IR': 1, 'OR': 2}
ORDER = ['F14', 'T6', 'S8', 'NF12', 'NM10', 'NE12', 'DE7', 'FE7']

_rules_dir = ROOT / 'codigo/cwru/ampliacion_v2/protocolo'
if str(_rules_dir) not in sys.path:
    sys.path.insert(0, str(_rules_dir))
from reglas_v2 import component_roles


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def bounded_path(root, relative):
    require(isinstance(relative, (str, Path)), 'Ruta relativa inválida')
    root = Path(root).resolve()
    value = Path(relative)
    require(not value.is_absolute(), 'No se admite ruta absoluta')
    target = (root / value).resolve()
    require(target != root and target.is_relative_to(root), 'Ruta fuera del proyecto')
    return target


def _validate_index(rows):
    require(isinstance(rows, list) and len(rows) == 1380, 'Índice debe tener 1380 ventanas')
    require([int(r['global_window_id']) for r in rows] == list(range(1380)),
            'IDs globales incompletos, repetidos o desordenados')
    records, pairs = {}, set()
    for row in rows:
        load, label = int(row['load_hp']), int(row['label'])
        require(load in range(4) and row['class'] in LABELS and LABELS[row['class']] == label,
                'Carga o etiqueta inválida')
        require(row['source_split_v1'] in ('train', 'validation', 'test') and
                row['source_features'] == FEATURE_BASE + 'X_features_' + row['source_split_v1'] + '.npy',
                'Fuente de características ajena al catálogo')
        source_row = int(row['source_row'])
        require(source_row >= 0, 'Fila fuente negativa')
        pair = row['source_features'], source_row
        require(pair not in pairs, 'Fila fuente repetida con distintos IDs')
        pairs.add(pair)
        require(int(row['file_id']) > 0 and int(row['window_index_in_file']) >= 0
                and row['de_key'] != row['fe_key'], 'Registro o canales inválidos')
        key = str(row['file_id'])
        signature = (row['file'], load, row['class'], label, row['de_key'], row['fe_key'])
        require(key not in records or records[key] == signature, 'Un MAT tiene metadatos contradictorios')
        records[key] = signature
    require(len(records) == 28, 'Se requieren 28 MAT')
    for load in range(4):
        require(Counter(v[2] for v in records.values() if v[1] == load) == Counter(N=1, IR=3, OR=3),
                'Composición de registros por carga alterada')
    for source in ALLOWED_FEATURES:
        source_rows = [int(r['source_row']) for r in rows if r['source_features'] == source]
        require(source_rows and sorted(source_rows) == list(range(len(source_rows))),
                'Correspondencia caché/índice incompleta')


def _scope(rows, loads, normal_only=False):
    selected = [r for r in rows if int(r['load_hp']) in loads and (not normal_only or r['class'] == 'N')]
    return dict(loads_hp=list(loads), normal_only=normal_only,
                file_ids=sorted({str(r['file_id']) for r in selected}, key=int),
                global_window_ids=[int(r['global_window_id']) for r in selected],
                windows=len(selected), class_window_counts={c: sum(r['class'] == c for r in selected) for c in LABELS})


def _expected_job(protocol, rows, outer, name):
    inner = min(set(range(4)) - {outer})
    roles = component_roles(outer, inner)
    train = _scope(rows, roles['training_loads'])
    val = _scope(rows, [inner])
    fit = _scope(rows, roles['detector_fit_normal_loads'], True)
    cal = _scope(rows, roles['detector_calibration_normal_loads'], True)
    train_rows = [r for r in rows if int(r['load_hp']) in roles['training_loads']]
    by_record = {}
    for row in train_rows:
        by_record.setdefault(str(row['file_id']), []).append(row)
    require(len(by_record) == 14, 'Banco de costo debe usar catorce MAT')
    bank = []
    for record in sorted(by_record, key=int):
        ordered = sorted(by_record[record], key=lambda r: int(r['window_index_in_file']))
        require(len(ordered) > 1, 'Banco necesita primera y última ventana distintas')
        require(len({int(r['window_index_in_file']) for r in ordered}) == len(ordered),
                'Índice de ventana repetido dentro del registro')
        bank.extend([int(ordered[0]['global_window_id']), int(ordered[-1]['global_window_id'])])
    paths = sorted({r['source_features'] for r in rows
                    if int(r['load_hp']) in roles['training_loads'] + [inner]})
    indices = protocol['candidates'][name]
    require(isinstance(indices, list) and indices and all(type(v) is int and 0 <= v < 14 for v in indices)
            and len(set(indices)) == len(indices), 'Índices de máscara inválidos')
    return dict(job_id=f'cost_outer{outer}_inner{inner}_{name}_seed11', candidate=name,
                feature_indices=indices.copy(), input_dimension=len(indices), outer_load_hp=outer,
                inner_validation_load_hp=inner, seed=11, roles=roles,
                training_global_window_ids=train['global_window_ids'],
                validation_global_window_ids=val['global_window_ids'],
                benchmark_global_window_ids=bank, benchmark_loads=roles['training_loads'],
                benchmark_file_ids=sorted(by_record, key=int), benchmark_windows=28, repetitions=3,
                detector_fit_global_window_ids=fit['global_window_ids'],
                detector_calibration_global_window_ids=cal['global_window_ids'],
                detector_fit=fit, detector_calibration=cal,
                authorized_feature_source_paths=paths, included_in_inner_search=True)


def validate_job_scope(job, rows, protocol):
    """Valida metadata y compara exactamente con el trabajo derivado del índice.

    Devuelve (filas_ajuste, filas_validación), ordenadas por ID global.
    """
    _validate_index(rows)
    require(isinstance(job, dict) and job.get('candidate') in ORDER, 'Candidata no autorizada')
    outer = job.get('outer_load_hp')
    require(type(outer) is int and outer in range(4), 'Carga externa inválida')
    expected = _expected_job(protocol, rows, outer, job['candidate'])
    require(job == expected, 'Trabajo distinto de su alcance V2 derivado')
    train_ids, val_ids = job['training_global_window_ids'], job['validation_global_window_ids']
    fit_ids, cal_ids = job['detector_fit_global_window_ids'], job['detector_calibration_global_window_ids']
    require(not set(train_ids) & set(val_ids) and not set(fit_ids) & set(cal_ids)
            and set(fit_ids + cal_ids) <= set(train_ids), 'Roles de ajuste/calibración solapados')
    train_rows, val_rows = [rows[i] for i in train_ids], [rows[i] for i in val_ids]
    require(not {r['file_id'] for r in train_rows} & {r['file_id'] for r in val_rows}, 'MAT compartido con validación')
    require(all(int(rows[i]['load_hp']) != outer for i in train_ids + val_ids)
            and all(rows[i]['class'] == 'N' for i in fit_ids + cal_ids), 'Fuga externa o detector no normal')
    require(len(set(job['benchmark_global_window_ids'])) == 28
            and set(job['benchmark_global_window_ids']) <= set(train_ids), 'Banco fuera de ajuste')
    return train_rows, val_rows


def build_jobs(protocol, cost_plan, rows):
    """Deriva los32 trabajos y contrasta todo el plan de costo sellado."""
    _validate_index(rows)
    require(protocol.get('candidate_order') == ORDER and protocol.get('primary_cost_objective') == 'total_pipeline'
            and protocol.get('dnn', {}).get('max_epochs') == 50
            and protocol['dnn']['inner_seeds'] == [11, 22, 33], 'Protocolo incompatible')
    require(cost_plan.get('status') == 'TOTAL_COST_CALIBRATION_PLANNED_NOT_EXECUTED'
            and cost_plan.get('cost_scope') == 'full_pipeline'
            and cost_plan.get('DNN_fits_included_in_inner_288') is True
            and cost_plan.get('observations_per_candidate_per_outer') == 84,
            'Plan de costo incompatible')
    representatives = cost_plan.get('representatives')
    require(isinstance(representatives, list) and len(representatives) == 32,
            'Se requieren exactamente 32 representantes')
    jobs = []
    for outer in range(4):
        for name in ORDER:
            job = _expected_job(protocol, rows, outer, name)
            old = dict(representative_id=job['job_id'], outer_load_hp=outer,
                       inner_validation_load_hp=job['inner_validation_load_hp'], candidate=name,
                       feature_indices=job['feature_indices'], seed=11, included_in_inner_search=True,
                       training_global_window_ids=job['training_global_window_ids'],
                       benchmark_loads=job['benchmark_loads'], benchmark_global_window_ids=job['benchmark_global_window_ids'],
                       benchmark_file_ids=job['benchmark_file_ids'], benchmark_windows=28, repetitions=3,
                       detector_fit=job['detector_fit'], detector_calibration=job['detector_calibration'],
                       cost_status='PENDING_NOT_MEASURED', mean_ms=None, lower_ms=None, upper_ms=None)
            require(representatives[len(jobs)] == old, 'Representante sellado no coincide con roles/IDs/banco')
            validate_job_scope(job, rows, protocol)
            jobs.append(job)
    require(len({j['job_id'] for j in jobs}) == 32, 'IDs de trabajos repetidos')
    return jobs


def load_verified_context(root=ROOT):
    """Verifica el sello de30 recursos y los metadatos; no abre matrices."""
    root = Path(root).resolve()
    directory = root / PROTOCOL_REL
    manifest = directory / 'MANIFEST_PROTOCOLO_SHA256.json'
    words = manifest.with_suffix('.sha256').read_text(encoding='ascii').split()
    require(words == [EXPECTED_PROTOCOL_SHA256, manifest.name] and sha256(manifest) == EXPECTED_PROTOCOL_SHA256,
            'Sello de protocolo distinto del verificado el25 de octubre')
    entries = read_json(manifest)
    require(isinstance(entries, dict) and len(entries) == 30, 'Se esperaban treinta recursos de protocolo')
    for relative, digest in entries.items():
        path = bounded_path(root, relative)
        require(path.suffix.lower() in ('.csv', '.json', '.md', '.py', '.sha256', '.txt'),
                'Tipo de recurso inesperado en protocolo')
        require(sha256(path) == digest, 'Recurso sellado modificado: ' + relative)
    protocol = read_json(directory / 'protocolo_reglas.json')
    cost_plan = read_json(directory / 'calibracion_costo_total_plan.json')
    parts = root / PARTITIONS_REL
    rows = read_csv(parts / 'indice_ventanas_fuente.csv')
    source_rows = read_csv(parts / 'entradas_sha256.csv')
    hashes = {row['path']: row['sha256'] for row in source_rows}
    require(len(hashes) == len(source_rows) == 104 and ALLOWED_FEATURES <= set(hashes),
            'Catálogo de hashes de entrada incompleto o duplicado')
    build_jobs(protocol, cost_plan, rows)
    return protocol, cost_plan, rows, hashes


def load_job_data(job, root=ROOT):
    """Devuelve (raw_train14f32,y_train,raw_val14f32,y_val,procedencia)."""
    root = Path(root).resolve()
    protocol, cost_plan, rows, hashes = load_verified_context(root)
    canonical = next((j for j in build_jobs(protocol, cost_plan, rows) if j['job_id'] == job.get('job_id')), None)
    require(canonical is not None and job == canonical, 'Trabajo ajeno al plan de32 representantes')
    train_rows, val_rows = validate_job_scope(job, rows, protocol)
    arrays, sources = {}, []
    for relative in job['authorized_feature_source_paths']:
        require(relative in ALLOWED_FEATURES, 'Matriz fuente no autorizada')
        path = bounded_path(root, relative)
        require(sha256(path) == hashes[relative], 'Caché de características modificado: ' + relative)
        array = np.load(path, allow_pickle=False, mmap_mode='r')
        source_rows = [r for r in rows if r['source_features'] == relative]
        require(array.dtype == np.float64 and array.shape == (len(source_rows), 14),
                'Forma o dtype del caché distintos')
        # No se exploran valores de las filas excluidas para comprobar finitud.
        arrays[relative] = array
        sources.append(dict(path=relative, sha256=hashes[relative], shape=list(array.shape),
                            dtype=str(array.dtype), mmap_mode='r',
                            contains_excluded_outer_rows=any(int(r['load_hp']) == job['outer_load_hp']
                                                           for r in source_rows)))
    def materialize(selected):
        raw = np.asarray([arrays[r['source_features']][int(r['source_row'])] for r in selected], dtype=np.float64)
        require(np.isfinite(raw).all(), 'Filas seleccionadas contienen no finitos float64')
        with np.errstate(over='ignore', invalid='ignore'):
            x = raw.astype(np.float32)
        require(np.isfinite(x).all(), 'Cast float32 de filas seleccionadas no finito')
        return x, np.asarray([int(r['label']) for r in selected], dtype=np.int64)
    xtrain, ytrain = materialize(train_rows)
    xval, yval = materialize(val_rows)
    provenance = dict(job_id=job['job_id'], source_arrays=sources,
        source_index_sha256=sha256(root / PARTITIONS_REL / 'indice_ventanas_fuente.csv'),
        protocol_manifest_sha256=EXPECTED_PROTOCOL_SHA256,
        train_global_window_ids=job['training_global_window_ids'],
        validation_global_window_ids=job['validation_global_window_ids'],
        detector_fit_global_window_ids=job['detector_fit_global_window_ids'],
        detector_calibration_global_window_ids=job['detector_calibration_global_window_ids'],
        benchmark_global_window_ids=job['benchmark_global_window_ids'],
        train_file_ids=sorted({r['file_id'] for r in train_rows}, key=int),
        validation_file_ids=sorted({r['file_id'] for r in val_rows}, key=int),
        roles=job['roles'], train_shape=list(xtrain.shape), validation_shape=list(xval.shape),
        loaded_dtype='float32', source_dtype='float64', label_dtype='int64',
        label_source='sealed CSV metadata, no label arrays opened',
        arrays_mapped=[s['path'] for s in sources], outer_rows_selected_or_materialized=False,
        mapping_or_file_hashing_is_not_OS_access_trace=True,
        raw_signals_or_MAT_opened=False)
    return xtrain, ytrain, xval, yval, provenance
