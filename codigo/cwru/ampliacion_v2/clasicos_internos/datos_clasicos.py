"""Plan y filas autorizadas para la búsqueda interna clásica CWRU V2.

Construir/importar este módulo no ajusta modelos, ejecuta inferencias o abre
TensorFlow. Reutiliza el orden de configuraciones del piloto, sin importar sus
ajustes no confirmatorios. RF usa tres semillas y SVM un ajuste determinista.
"""
from __future__ import annotations

import copy
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[4]
for directory in ('busqueda_interna','detector_interno','piloto_clasicos'):
    path=str(ROOT/'codigo/cwru/ampliacion_v2'/directory)
    if path not in sys.path:
        sys.path.insert(0,path)

import datos_busqueda as inner
import datos_detector as detector_data
import ejecutar_busqueda_interna as io
import contrato_piloto as pilot_contract


def require(ok,message):
    if not ok:
        raise ValueError(message)


def context():
    """Protocolo, metadatos y hashes validados de los cachés históricos."""
    return inner.context()


def _context_arguments(protocol,rows):
    if protocol is None or rows is None:
        observed_protocol,observed_rows,_=context()
        protocol=observed_protocol if protocol is None else protocol
        rows=observed_rows if rows is None else rows
    return protocol,rows


def _templates(protocol):
    classical=protocol['classical_comparators']
    require(classical['RF']['feature_mask'] == classical['SVM']['feature_mask'] == 'F14',
        'Los comparadores deben usar la representación F14')
    require(classical['RF']['inner_seeds'] == [11,22,33], 'Semillas RF distintas del protocolo')
    require(classical['SVM']['deterministic_one_fit_per_configuration_fold'] is True,
        'SVM debe ajustarse una sola vez por configuración y pliegue')
    templates=[job for job in pilot_contract.build_jobs(protocol) if job['family'] in ('RF','SVM')]
    require(len(templates) == 16 and [(j['family'],j['config_index']) for j in templates] ==
        [(family,index) for family in ('RF','SVM') for index in range(8)], 'Orden de grid no canónico')
    return templates


def _job_for(protocol,rows,outer,inner_load,template,seed):
    family=template['family']
    require(family in ('RF','SVM'), 'Familia clásica no válida')
    require((family == 'RF' and type(seed) is int and seed in [11,22,33])
        or (family == 'SVM' and seed is None), 'Semilla incompatible con la familia')
    # Este contenedor aporta únicamente roles/IDs. Su seed11 no controla RF/SVM.
    data_job=inner.job_for(protocol,rows,outer,inner_load,'F14',11)
    params=copy.deepcopy(template['params'])
    if family == 'RF':
        params['random_state']=seed
    config=template['config_index']
    seed_text=str(seed) if seed is not None else 'none'
    return dict(job_id=f'clasico_outer{outer}_inner{inner_load}_{family}_c{config:02d}_s{seed_text}',
        family=family,config_index=config,seed=seed,params=params,
        outer_load_hp=outer,inner_validation_load_hp=inner_load,feature_mask='F14',
        roles=copy.deepcopy(data_job['roles']),
        training_global_window_ids=copy.deepcopy(data_job['training_global_window_ids']),
        validation_global_window_ids=copy.deepcopy(data_job['validation_global_window_ids']),
        authorized_feature_source_paths=copy.deepcopy(data_job['authorized_feature_source_paths']),
        data_job=data_job,timeout_seconds=60 if family == 'RF' else 30)


def build_jobs(protocol=None,rows=None):
    """384 celdas: cuatro outer × tres inner × (8RF×3semillas +8SVM)."""
    protocol,rows=_context_arguments(protocol,rows)
    inner.old._validate_index(rows)
    templates=_templates(protocol)
    jobs=[]
    for outer in range(4):
        for inner_load in range(4):
            if inner_load == outer:
                continue
            for template in templates:
                seeds=protocol['classical_comparators']['RF']['inner_seeds'] if template['family']=='RF' else [None]
                for seed in seeds:
                    jobs.append(_job_for(protocol,rows,outer,inner_load,template,seed))
    require(len(jobs) == len({job['job_id'] for job in jobs}) == 384
        and sum(job['family']=='RF' for job in jobs) == 288
        and sum(job['family']=='SVM' for job in jobs) == 96, 'Grid clásico incompleto o duplicado')
    return jobs


def validate_scope(job,protocol,rows):
    """Rechazar parámetros o IDs alterados antes de materializar muestras."""
    require(isinstance(job,dict),'Trabajo clásico no es un diccionario')
    template=next((template for template in _templates(protocol)
        if template['family']==job.get('family') and template['config_index']==job.get('config_index')),None)
    require(template is not None,'Familia o configuración no pertenece al grid')
    expected=_job_for(protocol,rows,job['outer_load_hp'],job['inner_validation_load_hp'],template,job['seed'])
    require(io.canonical(job) == io.canonical(expected),
        'Trabajo distinto de su configuración/rol canónico')
    training,validation=inner.validate_scope(job['data_job'],rows,protocol)
    train_ids=job['training_global_window_ids'];validation_ids=job['validation_global_window_ids']
    require(not set(train_ids)&set(validation_ids)
        and len(train_ids)==len(set(train_ids)) and len(validation_ids)==len(set(validation_ids)),
        'IDs repetidos o compartidos entre entrenamiento y validación')
    require({row['file_id'] for row in training}.isdisjoint({row['file_id'] for row in validation})
        and all(int(row['load_hp']) != job['outer_load_hp'] for row in training+validation),
        'Fuga de MAT o de carga exterior')
    require(sorted({int(row['load_hp']) for row in training}) == job['roles']['training_loads']
        and {int(row['load_hp']) for row in validation} == {job['inner_validation_load_hp']},
        'Cargas distintas de los roles establecidos')
    return training,validation


def load_job_data(job,context=None):
    """Leer únicamente train/validación del trabajo, sin transformar o ajustar."""
    selected_context=globals()['context']() if context is None else context
    protocol,rows,_=selected_context
    validate_scope(job,protocol,rows)
    raw,y,train_records,train_provenance=detector_data.load_rows(
        job['training_global_window_ids'],context=selected_context)
    validation,y_validation,validation_records,validation_provenance=detector_data.load_rows(
        job['validation_global_window_ids'],context=selected_context)
    require(raw.shape == (len(job['training_global_window_ids']),14)
        and validation.shape == (len(job['validation_global_window_ids']),14), 'Forma clásica no válida')
    require(set(train_records).isdisjoint(validation_records),'MAT compartido en muestras cargadas')
    provenance=dict(job_id=job['job_id'],family=job['family'],config_index=job['config_index'],seed=job['seed'],
        roles=job['roles'],feature_mask='F14',raw_dtype='float32',label_dtype='int64',
        source_dtype='float64',training=train_provenance,validation=validation_provenance,
        training_global_window_ids=job['training_global_window_ids'],
        validation_global_window_ids=job['validation_global_window_ids'],
        outer_rows_selected_or_materialized=False,raw_signals_or_MAT_opened=False,
        normalization='raw_float32' if job['family']=='RF' else 'train_only_scaler_explicit_float32_in_worker',
        scaler_fitted_by_this_loader=False,model_fitted_by_this_loader=False,
        prediction_executed_by_this_loader=False,detector_fit_or_score_executed=False,
        historical_seed11_data_container_not_classifier_seed=True)
    return raw,y,validation,y_validation,provenance


def training_signature(job,protocol,inputhashes,versions,sourcehashes):
    """Identidad del ajuste; no hace equivalentes los modelos del piloto."""
    paths=job['authorized_feature_source_paths']
    require(paths and all(path in inputhashes for path in paths), 'Faltan hashes de entradas autorizadas')
    require(isinstance(versions,dict) and versions and isinstance(sourcehashes,dict) and sourcehashes,
        'Se requieren versiones y fuentes explícitas')
    body=dict(job=job,classical_comparators=protocol['classical_comparators'],
        numeric_contract=protocol['numeric_contract'],
        feature_sources_sha256={path:inputhashes[path] for path in paths},
        source_sha256=sourcehashes,versions=versions,
        input_representation='F14/raw_cache_float64_cast_float32',labels_dtype='int64',
        preprocessing='raw_float32' if job['family']=='RF' else 'train_only_scaler_explicit_float32',
        classical_predictions_are_not_INT8=True,outer_evaluation_enabled=False,
        pilot_models_are_not_confirmatory_references=True)
    return io.canonical(body)
