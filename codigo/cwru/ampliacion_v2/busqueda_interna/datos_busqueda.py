"""Roles y carga de filas autorizadas para los288 ajustes DNN internos V2.

No importa TensorFlow ni entrena. El almacenamiento NPY historico puede
contener cargas excluidas: solo se materializan train/validacion interna.
"""
from __future__ import annotations
import copy
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
LEGACY = ROOT/'codigo/cwru/ampliacion_v2/representantes'
if str(LEGACY) not in sys.path:
    sys.path.insert(0, str(LEGACY))
import datos_representantes as old
from reglas_v2 import component_roles


def require(ok, message):
    if not ok:
        raise ValueError(message)


def job_for(protocol, rows, outer, inner, candidate, seed):
    require(type(outer) is int and outer in range(4) and type(inner) is int
            and inner in set(range(4))-{outer}, 'Roles externos/internos no validos')
    require(candidate in protocol['candidate_order'] and type(seed) is int
            and seed in protocol['dnn']['inner_seeds'], 'Candidata o semilla no prevista')
    roles = component_roles(outer, inner)
    train = old._scope(rows, roles['training_loads'])
    validation = old._scope(rows, [inner])
    fitting = old._scope(rows, roles['detector_fit_normal_loads'], True)
    calibration = old._scope(rows, roles['detector_calibration_normal_loads'], True)
    by_record = {}
    for row in rows:
        if int(row['load_hp']) in roles['training_loads']:
            by_record.setdefault(str(row['file_id']), []).append(row)
    require(len(by_record) == 14, 'Las dos cargas de entrenamiento deben tener14MAT')
    bank = []
    for record in sorted(by_record, key=int):
        ordered = sorted(by_record[record], key=lambda row:int(row['window_index_in_file']))
        require(len(ordered) > 1, 'Registro sin primera/ultima ventana distintas')
        bank.extend([int(ordered[0]['global_window_id']), int(ordered[-1]['global_window_id'])])
    representative = inner == min(set(range(4))-{outer}) and seed == 11
    prefix = 'cost' if representative else 'inner'
    indices = copy.deepcopy(protocol['candidates'][candidate])
    job = dict(job_id=f'{prefix}_outer{outer}_inner{inner}_{candidate}_seed{seed}',candidate=candidate,
        feature_indices=indices,input_dimension=len(indices),outer_load_hp=outer,
        inner_validation_load_hp=inner,seed=seed,roles=roles,
        training_global_window_ids=train['global_window_ids'],
        validation_global_window_ids=validation['global_window_ids'],
        benchmark_global_window_ids=bank,benchmark_loads=roles['training_loads'],
        benchmark_file_ids=sorted(by_record,key=int),benchmark_windows=28,repetitions=3,
        detector_fit_global_window_ids=fitting['global_window_ids'],
        detector_calibration_global_window_ids=calibration['global_window_ids'],
        detector_fit=fitting,detector_calibration=calibration,
        authorized_feature_source_paths=sorted({row['source_features'] for row in rows
            if int(row['load_hp']) in roles['training_loads']+[inner]}),included_in_inner_search=True)
    if representative:
        require(job == old._expected_job(protocol,rows,outer,candidate), 'Representante no reproduce contrato original')
    return job


def build_jobs(protocol, rows):
    old._validate_index(rows)
    require(protocol['dnn']['inner_seeds'] == [11,22,33] and protocol['dnn']['max_epochs'] == 50
            and protocol['dnn']['early_stopping'] == dict(patience=10,minimum_improvement=1e-12,restore_best_weights=True),
            'El contrato de entrenamiento cambio')
    jobs = [job_for(protocol,rows,outer,inner,name,seed) for outer in range(4)
            for inner in range(4) if inner != outer
            for name in protocol['candidate_order'] for seed in protocol['dnn']['inner_seeds']]
    require(len(jobs) == 288 and len({j['job_id'] for j in jobs}) == 288, 'Plan interno incompleto o repetido')
    return jobs


def validate_scope(job, rows, protocol):
    require(job == job_for(protocol,rows,job['outer_load_hp'],job['inner_validation_load_hp'],
                           job['candidate'],job['seed']), 'Trabajo distinto de su rol canonico')
    train_ids, val_ids = job['training_global_window_ids'],job['validation_global_window_ids']
    require(not set(train_ids)&set(val_ids) and len(set(train_ids)) == len(train_ids)
            and len(set(val_ids)) == len(val_ids), 'IDs repetidos o train/val solapados')
    train_rows,val_rows = [rows[i] for i in train_ids],[rows[i] for i in val_ids]
    require(not {r['file_id'] for r in train_rows}&{r['file_id'] for r in val_rows}
            and all(int(r['load_hp']) != job['outer_load_hp'] for r in train_rows+val_rows),
            'Fuga de MAT o carga externa')
    fit,cal = job['detector_fit_global_window_ids'],job['detector_calibration_global_window_ids']
    require(not set(fit)&set(cal) and set(fit+cal)<=set(train_ids)
            and all(rows[i]['class'] == 'N' for i in fit+cal), 'Roles normales invalidos')
    return train_rows,val_rows


def context():
    protocol, _, rows, hashes = old.load_verified_context()
    return protocol,rows,hashes


def load_job_data(job, root=ROOT):
    root = Path(root).resolve()
    require(root == ROOT.resolve(), 'Workspace distinto del preparado')
    protocol,rows,hashes = context()
    train_rows,val_rows = validate_scope(job,rows,protocol)
    arrays,sources = {},[]
    for relative in job['authorized_feature_source_paths']:
        require(relative in old.ALLOWED_FEATURES, 'Matriz de entrada no autorizada')
        path = old.bounded_path(root,relative)
        require(old.sha256(path) == hashes[relative], 'Caracteristicas fuente modificadas')
        array = np.load(path,allow_pickle=False,mmap_mode='r')
        source_rows = [r for r in rows if r['source_features'] == relative]
        require(array.dtype == np.float64 and array.shape == (len(source_rows),14), 'Cache con forma/dtype distintos')
        arrays[relative] = array
        sources.append(dict(path=relative,sha256=hashes[relative],shape=list(array.shape),dtype=str(array.dtype),
                            mmap_mode='r',contains_excluded_outer_rows=any(
                                int(r['load_hp']) == job['outer_load_hp'] for r in source_rows)))
    def materialize(selected):
        raw = np.asarray([arrays[r['source_features']][int(r['source_row'])] for r in selected],dtype=np.float64)
        require(np.isfinite(raw).all(), 'Filas autorizadas no finitas')
        with np.errstate(over='ignore',invalid='ignore'):
            values = raw.astype(np.float32)
        require(np.isfinite(values).all(), 'Cast de filas autorizadas no finito')
        return values,np.asarray([int(r['label']) for r in selected],dtype=np.int64)
    train,y_train = materialize(train_rows)
    val,y_val = materialize(val_rows)
    provenance = dict(job_id=job['job_id'],source_arrays=sources,
        source_index_sha256=old.sha256(root/old.PARTITIONS_REL/'indice_ventanas_fuente.csv'),
        protocol_manifest_sha256=old.EXPECTED_PROTOCOL_SHA256,
        train_global_window_ids=job['training_global_window_ids'],validation_global_window_ids=job['validation_global_window_ids'],
        train_file_ids=sorted({r['file_id'] for r in train_rows},key=int),
        validation_file_ids=sorted({r['file_id'] for r in val_rows},key=int),roles=job['roles'],
        train_shape=list(train.shape),validation_shape=list(val.shape),loaded_dtype='float32',source_dtype='float64',
        label_source='sealed CSV metadata; label arrays not opened',
        outer_rows_selected_or_materialized=False,mapping_or_hashing_is_not_OS_access_trace=True,
        raw_signals_or_MAT_opened=False,detector_fit_executed=False)
    return train,y_train,val,y_val,provenance
