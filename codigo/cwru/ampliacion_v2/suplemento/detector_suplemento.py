"""Comparador secundario N12/MOM4 del suplemento; sin ejecucion al importar.

El orquestador, iniciado por el usuario, llama ``run_detector_block``. Este
modulo nunca selecciona detector, altera el protocolo o ajusta una DNN. Usa
features originales float32; P0/P1 del clasificador no cambian estos datos.
"""
from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
import platform
import sys

import numpy as np

from contrato_datos import ROOT, context, jobs, load_data

REPRESENTATIVES = ROOT / 'codigo/cwru/ampliacion_v2/representantes'
NUMERIC_SOURCE = ROOT / 'codigo/cwru/ampliacion_v2/piloto_clasicos/detector_numerico.py'
for path in (REPRESENTATIVES, NUMERIC_SOURCE.parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
from detector_numerico import fit_normality, detector_score
from numerica_representantes import detector_metrics
from ejecutar_representantes import validate_campaign, verify_job, read, sha

CONTROL12 = [0,1,2,3,5,6,7,8,9,10,12,13]
MOM4 = [1,2,8,9]
CAMPAIGN = ROOT / 'resultados/cwru/ampliacion_v2/representantes_v2_2026-10-25/20261025T171847_957295Z'
MINIMUM_RATE = .95


def require(ok, message):
    if not ok:
        raise ValueError(message)


def _write(path, payload):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def _npz(path, **arrays):
    with path.open('xb') as stream:
        np.savez(stream, **arrays)


def detector_jobs():
    """Seis trabajos, sin semillas: inner0/1/2 por CONTROL12 y MOM4."""
    return [dict(detector_id=f'detector_o3_i{inner}_{name}', inner=inner,
                 indices=indices.copy(), representation=name,
                 data_job_id=f'supp_o3_i{inner}_F14_P0_s11', outer_load_hp=3,
                 data_space='raw_features_float32', repeated_per_DNN_seed=False,
                 import_original_reference=(inner == 0 and name == 'CONTROL12'))
            for inner in (0,1,2) for name,indices in (('CONTROL12',CONTROL12),('MOM4',MOM4))]


def descriptive_guards(metrics):
    """Criterios por carga de ingenieria; no seleccion ni garantia estadistica."""
    fields = ('normal_spec','ir_recall','or_recall')
    for field in fields:
        value = metrics[field]
        require(isinstance(value,(int,float)) and not isinstance(value,bool)
                and np.isfinite(value) and 0 <= value <= 1, 'Tasa invalida: ' + field)
    checks = {field:bool(metrics[field] >= MINIMUM_RATE) for field in fields}
    return dict(minimum_each_rate=MINIMUM_RATE, checks=checks, passed=all(checks.values()),
                scope='one_internal_validation_load_descriptive_engineering_only',
                statistical_guarantee=False, automatic_detector_selection=False)


def _import_reference(data_job):
    """Autentica el ajuste original antes de reutilizar sus parametros."""
    old_plan, old_jobs = validate_campaign(CAMPAIGN)
    original = next(j for j in old_jobs if j['job_id'] == 'cost_outer3_inner0_F14_seed11')
    for key in ('detector_fit_global_window_ids','detector_calibration_global_window_ids',
                'validation_global_window_ids','training_global_window_ids'):
        require(data_job[key] == original[key], 'IDs distintos de la referencia: ' + key)
    require(data_job['roles'] == original['roles'] and data_job['candidate'] == 'F14'
            and data_job['mode'] == 'P0' and data_job['seed'] == 11,
            'Scope distinto del control original')
    relative = NUMERIC_SOURCE.relative_to(ROOT).as_posix()
    require(old_plan['source_sha256'][relative] == sha(NUMERIC_SOURCE),
            'Aritmetica del detector distinta de la fuente firmada')
    directory = CAMPAIGN / original['job_id']
    result = verify_job(directory, original, old_plan['fit_signatures'][original['job_id']])
    params_file = directory / 'detector_parametros.npz'
    with np.load(params_file, allow_pickle=False) as saved:
        params = {key:saved[key].copy() for key in saved.files}
    require(params['indices'].tolist() == CONTROL12
            and params['fit_ids'].tolist() == data_job['detector_fit_global_window_ids']
            and params['calibration_ids'].tolist() == data_job['detector_calibration_global_window_ids'],
            'Parametros originales no corresponden al control')
    params['indices'] = params['indices'].tolist()
    params['threshold32'] = np.float32(params['threshold32'].item())
    link = dict(imported_from_reference=True, original_job_id=original['job_id'],
                original_fit_signature=result['fit_signature'],
                original_job_manifest_sha256=sha(directory/'MANIFEST_JOB_SHA256.json'),
                original_parameters_path=params_file.relative_to(ROOT).as_posix(),
                original_parameters_sha256=sha(params_file),
                numeric_source_sha256=sha(NUMERIC_SOURCE), original_signature_rewritten=False,
                new_fit_calls=0)
    return params, link, params_file


def run_detector_block(output_dir, event_callback=None):
    """Usuario CLI: 1 ajuste importado y5 nuevos, en directorio nuevo.

    El padre controla autorizacion, limites, entorno, proceso separado y sellos
    de campana. Un fallo se propaga sin reintentar o reemplazar la referencia.
    No ejecutar este bloque como prueba de preparacion.
    """
    output_dir = Path(output_dir).resolve()
    allowed = (ROOT/'resultados/cwru/ampliacion_v2').resolve()
    require(output_dir != allowed and output_dir.is_relative_to(allowed), 'Salida fuera de resultados V2')
    output_dir.mkdir(exist_ok=False)
    protocol, metadata, _ = context()
    by_id = {int(row['global_window_id']):row for row in metadata}
    data_jobs = {job['job_id']:job for job in jobs()}
    results = []
    loaded = {}

    def event(stage, **extra):
        row = dict(stage=stage, **extra)
        with (output_dir/'progreso.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
        if event_callback is not None:
            event_callback(stage, **extra)

    for detector_job in detector_jobs():
        detector_id = detector_job['detector_id']
        directory = output_dir/detector_id
        directory.mkdir(exist_ok=False)
        data_job = data_jobs[detector_job['data_job_id']]
        inner = detector_job['inner']
        event('detector_start', detector_id=detector_id)
        if inner not in loaded:
            loaded[inner] = load_data(data_job)
        raw, y_train, raw_val, y_val, provenance = loaded[inner]
        positions = {key:pos for pos,key in enumerate(data_job['training_global_window_ids'])}
        fit_ids = data_job['detector_fit_global_window_ids']
        cal_ids = data_job['detector_calibration_global_window_ids']
        val_ids = data_job['validation_global_window_ids']
        require(set(fit_ids).isdisjoint(cal_ids) and set(fit_ids+cal_ids) <= set(positions)
                and all(y_train[positions[key]] == 0 for key in fit_ids+cal_ids),
                'Detector fuera de entrenamiento normal autorizado')
        fit = raw[[positions[key] for key in fit_ids]]
        calibration = raw[[positions[key] for key in cal_ids]]
        cal_records = [str(by_id[key]['file_id']) for key in cal_ids]
        val_records = [str(by_id[key]['file_id']) for key in val_ids]
        if detector_job['import_original_reference']:
            params, link, original_file = _import_reference(data_job)
            # Copia exacta; se conserva la firma de origen como referencia.
            with (directory/'parametros.npz').open('xb') as stream:
                stream.write(original_file.read_bytes())
            require(sha(directory/'parametros.npz') == link['original_parameters_sha256'],
                    'La copia de parametros no es exacta')
        else:
            params = fit_normality(fit,calibration,cal_records,detector_job['indices'])
            link = dict(imported_from_reference=False,new_fit_calls=1,
                        numeric_source_sha256=sha(NUMERIC_SOURCE))
            _npz(directory/'parametros.npz',indices=np.asarray(params['indices'],dtype=np.int64),
                 mean64=params['mean64'],scale64=params['scale64'],var64=params['var64'],
                 mean32=params['mean32'],scale32=params['scale32'],centroid32=params['centroid32'],
                 threshold32=params['threshold32'],fit_ids=np.asarray(fit_ids,dtype=np.int64),
                 calibration_ids=np.asarray(cal_ids,dtype=np.int64))
        scores_cal = detector_score(calibration,params)
        scores_val = detector_score(raw_val,params)
        rejected = scores_val > params['threshold32']
        metrics = detector_metrics(y_val,rejected,val_records)
        if detector_job['import_original_reference']:
            old_directory = CAMPAIGN/'cost_outer3_inner0_F14_seed11'
            with np.load(old_directory/'detector_validacion.npz',allow_pickle=False) as saved:
                require(np.array_equal(scores_val.view(np.uint32),saved['scores'].view(np.uint32))
                        and np.array_equal(rejected,saved['rejected'])
                        and np.array_equal(y_val,saved['y_true']), 'Replay original distinto')
            require(metrics == read(old_directory/'resultado.json')['detector_metrics'],
                    'Metricas importadas distintas')
        _npz(directory/'calibracion.npz',global_window_ids=np.asarray(cal_ids,dtype=np.int64),
             record_ids=np.asarray(cal_records),scores=scores_cal)
        _npz(directory/'validacion.npz',global_window_ids=np.asarray(val_ids,dtype=np.int64),
             record_ids=np.asarray(val_records),y_true=y_val,scores=scores_val,rejected=rejected)
        _write(directory/'procedencia.json',dict(data_provenance=provenance,data_job_id=data_job['job_id'],
            detector_job=detector_job,reference_link=link,fit_ids=fit_ids,calibration_ids=cal_ids,
            validation_ids=val_ids,data_space='raw_features_float32',DNN_compression_applied=False,
            fit_and_calibration_normal_only=True,refit_after_calibration=False,
            threshold_policy=protocol['detector']['threshold'],rejection='score_strictly_greater_than_threshold'))
        result = dict(status='PASS',detector_id=detector_id,representation=detector_job['representation'],
            inner=inner,outer_load_hp=3,indices=detector_job['indices'],data_job_id=data_job['job_id'],
            metrics=metrics,descriptive_guards=descriptive_guards(metrics),reference_link=link,
            imported_from_reference=link['imported_from_reference'],new_fit_calls=link['new_fit_calls'],
            fit_windows=len(fit_ids),calibration_windows=len(cal_ids),validation_windows=len(val_ids),
            threshold=float(params['threshold32']),selected_as_primary=False,
            source_parameters_sha256=sha(directory/'parametros.npz'),
            outer_predictive_evaluation_executed=False,refit_after_calibration=False)
        _write(directory/'resultado.json',result)
        results.append(result)
        event('detector_PASS',detector_id=detector_id,normal_spec=metrics['normal_spec'])
    require(len(results)==6 and sum(r['new_fit_calls'] for r in results)==5
            and sum(r['imported_from_reference'] for r in results)==1,'Conteo de ajustes incorrecto')
    summary = dict(status='PASS',scope='secondary_normality_development_only',results=results,
        new_fit_calls=5,imported_reference_fits=1,maximum_representations=2,internal_folds=3,
        repeated_per_DNN_seed=False,primary_selector_changed=False,automatic_detector_selection=False,
        outer_predictive_evaluation_executed=False,calibration_data_used_for_fit=False,
        versions=dict(python=platform.python_version(),numpy=np.__version__,
                      sklearn=importlib.metadata.version('scikit-learn')))
    _write(output_dir/'RESUMEN_DETECTOR.json',summary)
    return summary


if __name__ == '__main__':
    import argparse
    from ejecutar_suplemento import validate_run, versions, ENVIRONMENT
    import os
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    target=parser.parse_args().output_dir.resolve()
    plan=validate_run(target.parent)
    require(target.name=='detector' and (target.parent/'ACTIVO.lock').is_file(),'Bloque sin campana activa')
    require(versions()==plan['tested_versions'] and all(os.environ.get(k)==v for k,v in ENVIRONMENT.items()),
            'Entorno del bloque distinto del probado')
    require(all((target.parent/j['job_id']/'resultado.json').is_file() for j in plan['jobs']
                if j['job_id'] not in plan['imported_references']),'Faltan ajustes DNN anteriores')
    run_detector_block(target,lambda stage,**extra:print(json.dumps(dict(stage=stage,**extra)),flush=True))
