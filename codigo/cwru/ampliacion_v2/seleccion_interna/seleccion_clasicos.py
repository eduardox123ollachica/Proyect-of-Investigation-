"""Selección pura RF/SVM con las celdas internas ya auditadas.

No abre datos, ajusta modelos, ejecuta predicciones o importa TensorFlow.
La media da igual peso a las tres cargas internas y a sus semillas RF.
Los empates siguen media, peor carga y orden original del grid.
"""
from __future__ import annotations

from copy import deepcopy
from math import fsum, isfinite
from numbers import Integral, Real
from pathlib import Path
import json
import sys

PILOT_CODE=Path(__file__).resolve().parents[1]/'piloto_clasicos'
if str(PILOT_CODE) not in sys.path:
    sys.path.insert(0,str(PILOT_CODE))
from contrato_piloto import build_jobs as build_pilot_jobs


def _require(ok,message):
    if not ok:
        raise ValueError(message)


def _integer(value,name):
    _require(isinstance(value,Integral) and not isinstance(value,bool),name+' debe ser entero')
    return int(value)


def _rate(value,name):
    _require(isinstance(value,Real) and not isinstance(value,bool),name+' debe ser numérico')
    value=float(value)
    _require(isfinite(value) and 0 <= value <= 1,name+' debe ser finito entre cero y uno')
    return value


def _canonical(value):
    try:
        return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)
    except (TypeError,ValueError) as error:
        raise ValueError('Parámetros no serializables o no finitos') from error


def select_classical(rows,family,outer,protocol):
    """Elegir una configuración sin utilizar resultados de la carga externa.

    ``rows`` contiene filas de la matriz clásica auditada (o el subconjunto
    correspondiente). Se requieren las ocho configuraciones completas del
    escenario/familia: nueve celdas por configuración RF o tres para SVM.
    RF conserva una única configuración común a las semillas externas; por
    eso ``selected_params`` no incluye ``random_state``. SVM no se repite por
    semilla. Los parámetros deben coincidir con el catálogo original.
    """
    _require(family in ('RF','SVM'),'Familia clásica fuera de RF/SVM')
    outer=_integer(outer,'outer')
    _require(outer in range(4),'Escenario exterior fuera de0..3')
    _require(isinstance(protocol,dict),'Protocolo malformado')
    try:
        epsilon=protocol['selection']['numeric_comparison_tolerance']
        rf_seeds=protocol['classical_comparators']['RF']['inner_seeds']
        deterministic=protocol['classical_comparators']['SVM']['deterministic_one_fit_per_configuration_fold']
    except KeyError as error:
        raise ValueError('Campo del protocolo ausente: '+str(error)) from error
    _require(isinstance(epsilon,Real) and not isinstance(epsilon,bool)
        and isfinite(float(epsilon)) and float(epsilon)==1e-12,'Tolerancia numérica distinta de la fijada')
    epsilon=float(epsilon)
    _require(rf_seeds == [11,22,33] and all(type(s) is int for s in rf_seeds)
        and deterministic is True,'Política de semillas clásica distinta de la fijada')
    catalogue=[job for job in build_pilot_jobs(protocol) if job['family']==family]
    _require(len(catalogue)==8 and [job['config_index'] for job in catalogue]==list(range(8)),
        'Catálogo clásico incompleto o desordenado')
    templates={job['config_index']:job for job in catalogue}
    inner_loads=[load for load in range(4) if load != outer]
    seeds=rf_seeds if family=='RF' else [None]
    expected={(config,load,seed) for config in range(8) for load in inner_loads for seed in seeds}
    _require(isinstance(rows,(list,tuple)),'Las filas deben ser una lista o tupla')
    observed={};identifiers=set()
    for row in rows:
        _require(isinstance(row,dict),'Fila clásica malformada')
        try:
            observed_family=row['family']
            observed_outer=_integer(row['outer_load_hp'],'outer_load_hp')
            observed_inner=_integer(row['inner_validation_load_hp'],'inner_validation_load_hp')
            _require(observed_family in ('RF','SVM') and observed_outer in range(4)
                and observed_inner in set(range(4))-{observed_outer},'Familia o papeles de fila inválidos')
            if observed_family != family or observed_outer != outer:
                continue
            config=_integer(row['config_index'],'config_index')
            _require(config in templates,'Configuración fuera del grid')
            seed=row['seed']
            if family=='RF':
                seed=_integer(seed,'seed')
                _require(seed in rf_seeds,'Semilla RF no prevista')
            else:
                _require(seed is None,'SVM no debe repetirse por semilla')
            key=(config,observed_inner,seed)
            _require(key in expected and key not in observed,'Celda clásica ajena o duplicada')
            _require(row.get('status','PASS')=='PASS','Celda no aprobada: no omitirla del promedio')
            params=deepcopy(templates[config]['params'])
            if family=='RF':
                params['random_state']=seed
            _require(_canonical(row['params'])==_canonical(params),'Parámetros distintos del catálogo y semilla')
            score=_rate(row['macro_f1'],'macro_f1')
            if 'metrics' in row:
                _require(isinstance(row['metrics'],dict)
                    and _rate(row['metrics']['macro_f1'],'metrics.macro_f1')==score,
                    'Macro-F1 resumido contradice las métricas')
            identifier=row['job_id']
            _require(isinstance(identifier,str) and identifier and identifier not in identifiers,
                'Identificador clásico vacío o repetido')
            identifiers.add(identifier)
            observed[key]=dict(score=score,job_id=identifier)
        except KeyError as error:
            raise ValueError('Campo de fila clásica ausente: '+str(error)) from error
    _require(set(observed)==expected,'Matriz clásica incompleta para este escenario y familia')
    summaries=[]
    for config in range(8):
        per_load={str(load):fsum(observed[config,load,seed]['score'] for seed in seeds)/len(seeds)
            for load in inner_loads}
        params=deepcopy(templates[config]['params'])
        if family=='RF':
            del params['random_state']
        summaries.append(dict(config_index=config,params=params,cells=len(inner_loads)*len(seeds),
            mean_macro_f1=fsum(per_load.values())/len(inner_loads),
            worst_load_mean_macro_f1=min(per_load.values()),
            mean_macro_f1_by_internal_load=per_load,
            source_job_ids=[observed[config,load,seed]['job_id'] for load in inner_loads for seed in seeds]))
    maximum_mean=max(row['mean_macro_f1'] for row in summaries)
    mean_tied=[row for row in summaries if row['mean_macro_f1'] >= maximum_mean-epsilon]
    maximum_worst=max(row['worst_load_mean_macro_f1'] for row in mean_tied)
    final_tied=[row for row in mean_tied if row['worst_load_mean_macro_f1'] >= maximum_worst-epsilon]
    selected=min(final_tied,key=lambda row:row['config_index'])
    return dict(family=family,outer_load_hp=outer,selected_config_index=selected['config_index'],
        selected_params=deepcopy(selected['params']),mean_macro_f1=selected['mean_macro_f1'],
        worst_load_mean_macro_f1=selected['worst_load_mean_macro_f1'],
        config_summaries=summaries,
        tie_diagnostics=dict(numeric_tolerance=epsilon,maximum_mean_macro_f1=maximum_mean,
            mean_tied_config_indices=[row['config_index'] for row in mean_tied],
            maximum_worst_load_among_mean_ties=maximum_worst,
            worst_load_tied_config_indices=[row['config_index'] for row in final_tied],
            grid_order_rule='minimum_original_config_index'),
        inner_seeds=list(seeds),latency_used_for_selection=False,
        detector_used_for_classical_selection=False,probability_ensemble_used=False,
        outer_seed_assignment_pending=(family=='RF'))
