"""Contrato verificable de particiones CWRU. Sin entrenamiento ni escritura."""
from __future__ import annotations
from collections import Counter
import numpy as np

LOADS = (0,1,2,3)
LABELS = {'N':0,'IR':1,'OR':2}


def validate_records(records):
    if len(records)!=28 or len({str(r['file_id']) for r in records})!=28 or len({r['file'] for r in records})!=28:
        raise ValueError('Se requieren 28 registros/IDs/nombres únicos')
    for load in LOADS:
        rr=[r for r in records if int(r['load_hp'])==load]
        if Counter(r['class'] for r in rr)!=Counter({'N':1,'IR':3,'OR':3}):
            raise ValueError(f'Composición inválida en carga {load}')
    if any(int(r['label'])!=LABELS[r['class']] or r['de_key']==r['fe_key'] for r in records):
        raise ValueError('Etiquetas o claves de canal incompatibles')


def validate_config(config):
    expected={'output_sampling_rate_hz':12000,'window_size_samples':4096,
              'hop_size_samples':2048,'overlap_fraction':0.5,'channel_order':['DE','FE']}
    for k,v in expected.items():
        if config.get(k)!=v:raise ValueError(f'Configuración distinta del contrato: {k}')


def make_roles(records, outer_load, inner_load=None):
    validate_records(records)
    if outer_load not in LOADS or (inner_load is not None and (inner_load not in LOADS or inner_load==outer_load)):
        raise ValueError('Carga externa/interna inválida')
    result=[]
    for r in records:
        load=int(r['load_hp'])
        if inner_load is None:
            role='outer_test' if load==outer_load else 'development'
        else:
            role='outer_excluded' if load==outer_load else 'inner_validation' if load==inner_load else 'inner_train'
        result.append(dict(r,role=role))
    return result


def validate_roles(records, outer_load, inner_load=None):
    expected=make_roles(records,outer_load,inner_load)
    if any(r.get('role')!=e['role'] for r,e in zip(records,expected)):
        raise ValueError('Contaminación o papel incorrecto en partición')


def guard_fit_scope(index_rows, requested_global_ids, allowed_training_file_ids):
    """Falla si un ajuste pide ventanas ajenas al entrenamiento autorizado.

    El consumidor debe invocar esta guarda antes de fit; no inspecciona por sí
    misma modelos o archivos que un futuro entrenamiento use por otras vías.
    """
    index={int(r['global_window_id']):r for r in index_rows}
    if len(index)!=len(index_rows):raise ValueError('Índices globales duplicados')
    requested=list(map(int,requested_global_ids));allowed=set(map(str,allowed_training_file_ids))
    if len(set(requested))!=len(requested):raise ValueError('Solicitud con índices repetidos')
    for i in requested:
        if i not in index or str(index[i]['file_id']) not in allowed:
            raise ValueError(f'Fila no autorizada para ajuste: {i}')
    return requested


def select_training_rows(features, index_rows, requested_global_ids, allowed_training_file_ids):
    requested=guard_fit_scope(index_rows,requested_global_ids,allowed_training_file_ids)
    if len(features)!=len(index_rows):raise ValueError('Matriz e índice no corresponden')
    if any(int(r['global_window_id'])!=i for i,r in enumerate(index_rows)):
        raise ValueError('La matriz requiere índices globales contiguos en el mismo orden')
    return np.asarray(features)[requested]
