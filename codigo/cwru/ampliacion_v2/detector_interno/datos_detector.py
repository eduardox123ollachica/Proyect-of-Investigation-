"""Inventario y filas autorizadas del detector interno CWRU V2.

Preparar el inventario solo comprueba fuentes y resultados existentes. No
ajusta parámetros, ejecuta el scorer ni genera evaluaciones nuevas. F14 y
NF12 comparten parámetros/evaluaciones cuando sus índices efectivos son iguales.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[4]
INNER=ROOT/'codigo/cwru/ampliacion_v2/busqueda_interna'
if str(INNER) not in sys.path:
    sys.path.insert(0,str(INNER))
import datos_busqueda as data
import ejecutar_busqueda_interna as io
from numerica_representantes import detector_metrics
from reglas_v2 import weighted_p95

NUMERIC=ROOT/'codigo/cwru/ampliacion_v2/piloto_clasicos/detector_numerico.py'
SUPP_CLOSURE=ROOT/'articulo/ampliacion_cwru/metodologia/cierre_suplemento_completo_2026-10-25/MANIFEST_CIERRE_SHA256.json'
PARAMETER_ARRAYS=('indices','mean64','scale64','var64','mean32','scale32','centroid32',
                  'threshold32','fit_ids','calibration_ids')


def require(ok,message):
    if not ok:
        raise ValueError(message)


def _path(value):
    value=Path(value)
    target=(ROOT/value).resolve() if not value.is_absolute() else value.resolve()
    require(target != ROOT.resolve() and target.is_relative_to(ROOT.resolve()),'Ruta fuera del proyecto')
    return target


def parameter_payload_sha256(path):
    """Hash de arrays canónicos, independiente de ZIP/fechas de un NPZ."""
    digest=hashlib.sha256()
    with np.load(_path(path),allow_pickle=False) as saved:
        require(set(saved.files) == set(PARAMETER_ARRAYS),'Payload de parámetros incompleto o ajeno')
        for key in sorted(PARAMETER_ARRAYS):
            array=saved[key]
            require(array.dtype.kind != 'O','No admitir objetos en parámetros')
            digest.update(key.encode('utf-8'))
            digest.update(str(array.dtype).encode('ascii'))
            digest.update(str(array.shape).encode('ascii'))
            digest.update(array.tobytes(order='C'))
    return digest.hexdigest()


def load_params(path):
    """Leer parámetros existentes; no ajustar scaler, centroide o umbral."""
    with np.load(_path(path),allow_pickle=False) as saved:
        require(set(saved.files) == set(PARAMETER_ARRAYS),'Esquema de parámetros no canónico')
        values={key:saved[key].copy() for key in saved.files}
    indices=values['indices']
    require(indices.dtype == np.int64 and indices.ndim == 1 and len(indices)>0
        and len(set(indices.tolist())) == len(indices)
        and all(0 <= int(i) < 14 and int(i) not in (4,11) for i in indices), 'Índices detector inválidos')
    size=len(indices)
    for key in ('mean64','scale64','var64','mean32','scale32','centroid32'):
        expected=np.float64 if key.endswith('64') else np.float32
        require(values[key].dtype == expected and values[key].shape == (size,)
            and np.isfinite(values[key]).all(),'Vector de parámetros inválido: '+key)
    require((values['scale64']>0).all() and (values['scale32']>0).all()
        and (values['var64']>=0).all(),'Escala/varianza de parámetros inválida')
    require(np.array_equal(values['mean32'].view(np.uint32),values['mean64'].astype(np.float32).view(np.uint32))
        and np.array_equal(values['scale32'].view(np.uint32),values['scale64'].astype(np.float32).view(np.uint32)),
        'Exportación del scaler no corresponde a float32')
    threshold=values['threshold32']
    require(threshold.dtype == np.float32 and threshold.shape == ()
        and np.isfinite(threshold) and threshold >= 0,'Umbral no escalar float32 finito')
    for key in ('fit_ids','calibration_ids'):
        ids=values[key]
        require(ids.dtype == np.int64 and ids.ndim == 1 and len(ids)>0
            and ids.tolist() == sorted(set(ids.tolist())) and (ids>=0).all(),'IDs de parámetros inválidos: '+key)
    require(not set(values['fit_ids'].tolist())&set(values['calibration_ids'].tolist()),'Ajuste y calibración solapados')
    values['indices']=indices.tolist()
    values['threshold32']=np.float32(threshold.item())
    return values


def load_rows(ids,context=None):
    """Materializar solo IDs dados; el llamador contrasta su scope con el plan.

    Los cachés históricos pueden contener otras cargas. Hash/mmap del archivo
    no son una traza de accesos del SO; no se inspeccionan valores excluidos.
    ``context`` permite una instantánea previamente verificada por el llamador.
    """
    protocol,rows,hashes=data.context() if context is None else context
    require(isinstance(ids,(list,tuple)) and ids and all(type(i) is int for i in ids)
        and list(ids) == sorted(set(ids)) and all(0 <= i < len(rows) for i in ids),
        'IDs solicitados deben ser enteros únicos ordenados del índice')
    require(all(int(rows[i]['global_window_id']) == i for i in ids),'Índice no corresponde a sus IDs')
    selected=[rows[i] for i in ids]
    arrays={};sources=[]
    for relative in sorted({r['source_features'] for r in selected}):
        require(relative in data.old.ALLOWED_FEATURES and relative in hashes,'Matriz no autorizada')
        path=data.old.bounded_path(ROOT,relative)
        require(data.old.sha256(path) == hashes[relative],'Caché modificado: '+relative)
        array=np.load(path,allow_pickle=False,mmap_mode='r')
        source_rows=[r for r in rows if r['source_features'] == relative]
        require(array.dtype == np.float64 and array.shape == (len(source_rows),14), 'Caché con forma/dtype distinto')
        arrays[relative]=array
        sources.append(dict(path=relative,sha256=hashes[relative],shape=list(array.shape),dtype='float64',mmap_mode='r'))
    raw=np.asarray([arrays[r['source_features']][int(r['source_row'])] for r in selected],dtype=np.float64)
    require(np.isfinite(raw).all(),'Valores seleccionados no finitos')
    with np.errstate(over='ignore',invalid='ignore'):
        values=raw.astype(np.float32)
    require(np.isfinite(values).all(),'Cast de filas solicitadas no finito')
    labels=np.asarray([int(r['label']) for r in selected],dtype=np.int64)
    records=[str(r['file_id']) for r in selected]
    provenance=dict(global_window_ids=list(ids),source_arrays=sources,
        source_index_sha256=data.old.sha256(ROOT/data.old.PARTITIONS_REL/'indice_ventanas_fuente.csv'),
        protocol_manifest_sha256=data.old.EXPECTED_PROTOCOL_SHA256,
        selected_loads_hp=sorted({int(r['load_hp']) for r in selected}),
        selected_record_ids=sorted(set(records),key=int),shape=list(values.shape),loaded_dtype='float32',
        source_dtype='float64',label_source='sealed CSV metadata; label arrays not opened',
        only_requested_rows_selected_or_materialized=True,mapping_or_hashing_is_not_OS_access_trace=True,
        raw_signals_or_MAT_opened=False,detector_fit_executed=False,detector_score_executed=False)
    return values,labels,records,provenance


def _normal_scopes(job,rows):
    fit=job['detector_fit_global_window_ids'];cal=job['detector_calibration_global_window_ids']
    val=job['validation_global_window_ids']
    require(not set(fit)&set(cal) and not set(fit+cal)&set(val), 'Scopes detector solapados')
    require(all(rows[i]['class']=='N' and int(rows[i]['label'])==0 for i in fit+cal), 'Ajuste/calibración no normal')
    require(len({rows[i]['file_id'] for i in fit}) == len({rows[i]['file_id'] for i in cal}) == 1,
        'Se requiere un MAT normal distinto por ajuste/calibración')
    require({rows[i]['file_id'] for i in fit}.isdisjoint({rows[i]['file_id'] for i in cal+val})
        and {rows[i]['file_id'] for i in cal}.isdisjoint({rows[i]['file_id'] for i in val}), 'MAT compartidos entre papeles')
    return fit,cal,val


def _reference(folder,source_family,source_job_id,row,param,rows,original_signature,manifest):
    names=('detector_parametros.npz','detector_calibracion.npz','detector_validacion.npz') \
        if source_family == 'representantes' else ('parametros.npz','calibracion.npz','validacion.npz')
    params_path,cal_path,val_path=[folder/name for name in names]
    result_path=folder/'resultado.json'
    result=io.read(result_path)
    params=load_params(params_path)
    require(params['indices'] == param['indices'] and params['fit_ids'].tolist() == param['fit_ids']
        and params['calibration_ids'].tolist() == param['calibration_ids'], 'Parámetros reutilizados de otro ámbito')
    require(result['status']=='PASS','Resultado histórico no aprobado')
    if source_family == 'representantes':
        require(result['fit_signature'] == original_signature and result['roles'] == row['roles']
            and result['outer_load_hp'] == row['outer_load_hp']
            and result['inner_validation_load_hp'] == row['inner_validation_load_hp']
            and result['candidate'] == row['candidate'] and not result['detector_refit_after_calibration'],
            'Firma o roles REP incompatibles')
        metrics=result['detector_metrics']
    else:
        require(result['representation']=='CONTROL12' and result['outer_load_hp']==3
            and result['inner']==row['inner_validation_load_hp'] and result['indices']==row['indices']
            and result['source_parameters_sha256']==io.sha(params_path)
            and not result['refit_after_calibration'], 'Control12 suplementario incompatible')
        metrics=result['metrics']
    with np.load(cal_path,allow_pickle=False) as cal:
        require(cal['global_window_ids'].tolist()==param['calibration_ids'] and cal['scores'].dtype==np.float32
            and cal['scores'].shape==(len(param['calibration_ids']),) and np.isfinite(cal['scores']).all(),
            'Calibración histórica inválida')
        rec=[str(rows[i]['file_id']) for i in param['calibration_ids']]
        require(cal['record_ids'].tolist()==rec
            and weighted_p95(cal['scores'],rec).tobytes()==params['threshold32'].tobytes(), 'P95 histórico distinto')
    with np.load(val_path,allow_pickle=False) as val:
        ids=row['validation_ids']
        labels=np.asarray([int(rows[i]['label']) for i in ids],dtype=np.int64)
        records=[str(rows[i]['file_id']) for i in ids]
        require(val['global_window_ids'].tolist()==ids and np.array_equal(val['y_true'],labels)
            and val['scores'].dtype==np.float32 and val['scores'].shape==(len(ids),)
            and np.isfinite(val['scores']).all() and val['rejected'].dtype==np.bool_
            and np.array_equal(val['rejected'],val['scores']>params['threshold32']), 'Validación histórica incompatible')
        require(detector_metrics(labels,val['rejected'],records)==metrics,'Métricas históricas incompatibles')
    files=[params_path,cal_path,val_path,result_path,manifest,manifest.with_suffix('.sha256')]
    provenance=folder/'procedencia.json'
    if provenance.is_file():files.append(provenance)
    paths={p.relative_to(ROOT).as_posix():io.sha(p) for p in files}
    return dict(params_path=params_path.relative_to(ROOT).as_posix(),
        calibration_path=cal_path.relative_to(ROOT).as_posix(),validation_path=val_path.relative_to(ROOT).as_posix(),
        result_path=result_path.relative_to(ROOT).as_posix(),
        provenance_path=provenance.relative_to(ROOT).as_posix() if provenance.is_file() else None,
        source_folder=folder.relative_to(ROOT).as_posix(),source_family=source_family,source_job_id=source_job_id,
        original_signature=original_signature,original_signature_available=original_signature is not None,
        original_signature_rewritten=False,source_sha256=paths,
        parameter_payload_sha256=parameter_payload_sha256(params_path),metrics=metrics,
        source_reference_link=result.get('reference_link'),new_fit_calls=0,new_score_calls=0)


def build_inventory():
    """42 parámetros /84 evaluaciones únicas /96 filas lógicas, sin fit nuevo."""
    protocol,rows,hashes=data.context()
    inner_plan=io.read(io.PLAN_FILE)
    io.verify_manifest(io.PREP_MANIFEST)
    tested_versions=inner_plan['tested_versions']
    detector_versions={key:tested_versions[key] for key in ('python','numpy','sklearn')}
    index_hash=data.old.sha256(ROOT/data.old.PARTITIONS_REL/'indice_ventanas_fuente.csv')
    numeric_sources={p.relative_to(ROOT).as_posix():io.sha(p) for p in
        (NUMERIC,ROOT/'codigo/cwru/ampliacion_v2/protocolo/reglas_v2.py')}
    parameter_jobs={};evaluation_jobs={};logical_jobs=[]
    for outer in range(4):
        for inner in range(4):
            if inner==outer:continue
            for candidate in protocol['candidate_order']:
                job=data.job_for(protocol,rows,outer,inner,candidate,11)
                data.validate_scope(job,rows,protocol)
                fit,cal,val=_normal_scopes(job,rows)
                indices=[i for i in protocol['candidates'][candidate] if i not in (4,11)]
                paths=sorted({rows[i]['source_features'] for i in fit+cal})
                contract=dict(indices=indices,fit_ids=fit,calibration_ids=cal,
                    numeric_contract=protocol['numeric_contract'],detector_policy=protocol['detector'],
                    source_features_sha256={p:hashes[p] for p in paths},
                    source_index_sha256=index_hash,numeric_source_sha256=numeric_sources,versions=detector_versions,
                    protocol_manifest_sha256=data.old.EXPECTED_PROTOCOL_SHA256)
                signature=io.canonical(contract);param_id='detp_'+signature[:20]
                parameter=dict(param_id=param_id,parameter_signature=signature,indices=indices,
                    fit_ids=fit,calibration_ids=cal,fit_loads=job['roles']['detector_fit_normal_loads'],
                    calibration_loads=job['roles']['detector_calibration_normal_loads'],
                    authorized_feature_source_paths=paths,parameter_contract=contract)
                if param_id in parameter_jobs:
                    require(parameter_jobs[param_id]==parameter,'Colisión de ámbito/firma de parámetros')
                else:parameter_jobs[param_id]=parameter
                val_paths=sorted({rows[i]['source_features'] for i in val})
                eval_signature=io.canonical(dict(param_id=param_id,parameter_signature=signature,
                    validation_ids=val,source_features_sha256={p:hashes[p] for p in val_paths},
                    source_index_sha256=index_hash,metric_source_sha256=io.sha(io.REP/'numerica_representantes.py')))
                eval_id='dete_'+eval_signature[:20]
                row_id=f'detr_outer{outer}_inner{inner}_{candidate}'
                evaluation=dict(eval_id=eval_id,evaluation_signature=eval_signature,param_id=param_id,
                    validation_ids=val,indices=indices,row_source_paths=val_paths,logical_row_ids=[])
                if eval_id not in evaluation_jobs:evaluation_jobs[eval_id]=evaluation
                else:
                    require(all(evaluation_jobs[eval_id][k]==evaluation[k] for k in evaluation if k!='logical_row_ids'),
                        'Colisión de ámbito/firma de evaluación')
                evaluation_jobs[eval_id]['logical_row_ids'].append(row_id)
                row=dict(row_id=row_id,param_id=param_id,eval_id=eval_id,candidate=candidate,
                    outer_load_hp=outer,inner_validation_load_hp=inner,roles=job['roles'],fit_ids=fit,
                    calibration_ids=cal,validation_ids=val,indices=indices,
                    row_source_paths=val_paths,data_job=job)
                row['signature']=io.canonical(row)
                logical_jobs.append(row)
    require(len(parameter_jobs)==42 and len(evaluation_jobs)==84 and len(logical_jobs)==96,'Inventario interno no completo')
    by_row={(r['outer_load_hp'],r['inner_validation_load_hp'],r['candidate']):r for r in logical_jobs}
    rep_plan=io.read(ROOT/'articulo/ampliacion_cwru/metodologia/representantes_v2_2026-10-25/PLAN_EJECUCION.json')
    reused_parameters={};reused_evaluations={};reused_rows={}
    def add(row,reference):
        require(row['row_id'] not in reused_rows,'Fila histórica repetida')
        pid,eid=row['param_id'],row['eval_id']
        if pid in reused_parameters:
            require(reused_parameters[pid]['parameter_payload_sha256']==reference['parameter_payload_sha256'],
                'Parámetros históricos compartidos con payload distinto')
        else:reused_parameters[pid]=reference
        if eid in reused_evaluations:
            old_ref=reused_evaluations[eid]
            require(old_ref['metrics']==reference['metrics'],'Alias de evaluación con métricas distintas')
            with np.load(ROOT/old_ref['validation_path'],allow_pickle=False) as old_val, \
                 np.load(ROOT/reference['validation_path'],allow_pickle=False) as new_val:
                require(np.array_equal(old_val['scores'].view(np.uint32),new_val['scores'].view(np.uint32))
                    and np.array_equal(old_val['rejected'],new_val['rejected']), 'Alias de evaluación no bit-exacto')
        else:reused_evaluations[eid]=reference
        reused_rows[row['row_id']]=reference
    for original in rep_plan['jobs']:
        key=(original['outer_load_hp'],original['inner_validation_load_hp'],original['candidate'])
        row=by_row[key];folder=io.REP_RUN/original['job_id']
        manifest=folder/'MANIFEST_JOB_SHA256.json'
        io.verify_manifest(manifest,exact_directory=folder)
        signature=rep_plan['fit_signatures'][original['job_id']]
        ref=_reference(folder,'representantes',original['job_id'],row,parameter_jobs[row['param_id']],rows,signature,manifest)
        add(row,ref)
    io.verify_manifest(SUPP_CLOSURE)
    for inner in (1,2):
        folder=io.SUPP_RUN/'detector'/f'detector_o3_i{inner}_CONTROL12'
        for candidate in ('F14','NF12'):
            row=by_row[3,inner,candidate]
            ref=_reference(folder,'suplemento_CONTROL12',folder.name,row,parameter_jobs[row['param_id']],rows,None,SUPP_CLOSURE)
            add(row,ref)
    require(len(reused_parameters)==23 and len(reused_evaluations)==30 and len(reused_rows)==36,
        'Reutilización distinta de23 parámetros/30 evaluaciones/36 filas')
    dependencies=(Path(__file__).resolve(),Path(data.__file__).resolve(),Path(data.old.__file__).resolve(),
        Path(io.__file__).resolve(),NUMERIC,
        ROOT/'codigo/cwru/ampliacion_v2/protocolo/reglas_v2.py',io.REP/'numerica_representantes.py',
        io.REP/'worker_representantes.py',ROOT/'codigo/cwru/ampliacion_v2/suplemento/detector_suplemento.py')
    return dict(protocol=protocol,tested_versions=tested_versions,input_hashes=hashes,
        parameter_jobs=list(parameter_jobs.values()),evaluation_jobs=list(evaluation_jobs.values()),
        logical_jobs=logical_jobs,reused_parameters=reused_parameters,reused_evaluations=reused_evaluations,
        reused_rows=reused_rows,source_sha256={p.relative_to(ROOT).as_posix():io.sha(p) for p in dependencies},
        counts=dict(parameters_total=42,parameters_reused=23,parameters_new=19,
            evaluations_total=84,evaluations_reused=30,evaluations_new=54,
            logical_rows_total=96,logical_rows_reused=36,logical_rows_new=60),
        detector_fit_calls=0,detector_score_calls=0,outer_evaluation_executed=False,
        TensorFlow_imported='tensorflow' in sys.modules)
