"""Métricas puras de la evaluación exterior CWRU V2.

No abre datos, ajusta modelos o ejecuta inferencias. El llamador autentica
modelos/roles/IDs. Las medias entre cargas y semillas son descriptivas del
corpus fijo; las ventanas solapadas y semillas no son adquisiciones nuevas.
"""
from __future__ import annotations

from math import fsum, isfinite
from numbers import Integral, Real
from pathlib import Path
from statistics import stdev
import sys

import numpy as np

_LEGACY=Path(__file__).resolve().parents[1]/'representantes'
if str(_LEGACY) not in sys.path:
    sys.path.insert(0,str(_LEGACY))
from numerica_representantes import metrics3, detector_metrics, integer_argmax, quantize_int8

OUTER_SEEDS=(11,22,33,42,44,55,66,77,88,99)
_EPS=1e-12


def _require(ok,message):
    if not ok:
        raise ValueError(message)


def _int(value,name):
    _require(isinstance(value,Integral) and not isinstance(value,(bool,np.bool_)),name+' debe ser entero')
    return int(value)


def _rate(value,name):
    _require(isinstance(value,Real) and not isinstance(value,(bool,np.bool_)),name+' debe ser real')
    result=float(value)
    _require(isfinite(result) and 0<=result<=1,name+' debe estar entre cero y uno')
    return result


def _labels(values,name):
    if isinstance(values,(list,tuple)):
        _require(not any(isinstance(value,(bool,np.bool_)) for value in values),
            name+' no admite booleanos como etiquetas')
    array=np.asarray(values)
    _require(array.ndim==1 and len(array)>0 and array.dtype.kind in 'iu'
        and ((array>=0)&(array<=2)).all(),name+' debe contener etiquetas enteras 0, 1 o 2')
    return array.astype(np.int64,copy=False)


def _records(record_ids,y):
    _require(not isinstance(record_ids,(str,bytes)),'Se requiere un ID MAT por ventana')
    identifiers=list(record_ids)
    _require(len(identifiers)==len(y),'Número de MAT distinto de las ventanas')
    normalized=[]
    for value in identifiers:
        _require(isinstance(value,(str,Integral)) and not isinstance(value,(bool,np.bool_))
            and str(value) and str(value).strip()==str(value),'ID MAT inválido')
        normalized.append(str(value))
    groups={}
    for index,value in enumerate(normalized):groups.setdefault(value,[]).append(index)
    for value,offsets in groups.items():
        _require(len(set(y[offsets].tolist()))==1,'MAT con más de una clase: '+value)
    return normalized,groups


def supervised(y,pred,record_ids):
    """Métricas de tres clases y recall de la clase verdadera de cada MAT.

    El macro-F1 principal se calcula sobre las ventanas de la carga completa,
    con etiquetas [0,1,2] y zero_division=0. No se calcula macro-F1 de un MAT
    que contiene únicamente una clase. Las matrices tienen filas verdaderas.
    """
    truth,prediction=_labels(y,'y'),_labels(pred,'pred')
    _require(truth.shape==prediction.shape,'Etiquetas y predicciones incompatibles')
    _,groups=_records(record_ids,truth)
    records=[]
    for identifier in sorted(groups):
        offsets=groups[identifier];label=int(truth[offsets[0]])
        correct=int(np.count_nonzero(prediction[offsets]==label))
        records.append(dict(record_id=identifier,label=label,windows=len(offsets),correct_windows=correct,
            true_class_recall=correct/len(offsets),
            predicted_class_counts=np.bincount(prediction[offsets],minlength=3).tolist()))
    return dict(**metrics3(truth,prediction),per_record_true_class_recall=records,MAT_count=len(records))


def detector(y,rejected,record_ids):
    """Tasas de rechazo/aceptación ponderadas igualmente por MAT dentro de clase."""
    truth=_labels(y,'y');_records(record_ids,truth)
    return detector_metrics(truth,rejected,record_ids)


def _qparams(values,name):
    _require(isinstance(values,(list,tuple)) and len(values)==2,name+' requiere scale y zero_point')
    scale,zero=values
    _require(isinstance(scale,Real) and not isinstance(scale,(bool,np.bool_))
        and isfinite(float(scale)) and float(scale)>0,name+' scale debe ser positiva finita')
    with np.errstate(over='ignore',under='ignore'):
        scale32=np.float32(scale)
    zero=_int(zero,name+' zero_point')
    _require(np.isfinite(scale32) and scale32>0 and -128<=zero<=127,'Parámetros INT8 inválidos: '+name)
    return scale32,zero


def _probabilities(array,size,name):
    value=np.asarray(array)
    _require(value.dtype==np.float32 and value.shape==(size,3) and np.isfinite(value).all(),
        name+' requiere matriz float32 finita n por tres')
    return value


def quantization_report(z,input_int8,raw_int8_outputs,fp_probs,int_probs,y,record_ids,
                        inputq,outputq,global_window_ids=None):
    """Auditar cuantización y comparar salidas FP32/INT8 de las mismas ventanas.

    Se reconstruyen exactamente la cuantización affine float32 nearest-even
    y la des-cuantización de salida. Un valor en un extremo INT8 no demuestra
    clipping: este último se cuenta antes del clip, después del redondeo.
    Las diferencias de métricas tienen signo INT8 menos FP32.
    """
    truth=_labels(y,'y');size=len(truth);records,_=_records(record_ids,truth)
    normalized=np.asarray(z);inputs=np.asarray(input_int8);outputs=np.asarray(raw_int8_outputs)
    _require(normalized.dtype==np.float32 and normalized.ndim==2 and normalized.shape[0]==size
        and normalized.shape[1]>0 and np.isfinite(normalized).all(),'z debe ser float32 finito n por d')
    _require(inputs.dtype==np.int8 and inputs.shape==normalized.shape,'Entradas INT8 con forma/dtype distinto')
    _require(outputs.dtype==np.int8 and outputs.shape==(size,3),'Salidas INT8 con forma/dtype distinto')
    fp=_probabilities(fp_probs,size,'FP32');integer=_probabilities(int_probs,size,'INT8 des-cuantizado')
    input_scale,input_zero=_qparams(inputq,'entrada')
    output_scale,output_zero=_qparams(outputq,'salida')
    expected,clipped_total=quantize_int8(normalized,float(input_scale),input_zero)
    _require(np.array_equal(expected,inputs),'La entrada INT8 no coincide con la cuantización fijada')
    with np.errstate(over='ignore',invalid='ignore',under='ignore'):
        rounded=np.rint(np.add(np.divide(normalized,input_scale,dtype=np.float32),np.float32(input_zero),dtype=np.float32))
        dequantized=np.multiply(np.subtract(outputs.astype(np.float32),np.float32(output_zero),dtype=np.float32),
            output_scale,dtype=np.float32)
    _require(np.isfinite(rounded).all() and np.isfinite(dequantized).all(),
        'Cuantización o des-cuantización no finita')
    _require(np.array_equal(integer,dequantized),'Probabilidades INT8 no corresponden a la salida des-cuantizada')
    clipping=(rounded < -128)|(rounded > 127)
    _require(int(np.count_nonzero(clipping))==clipped_total,'Conteo de clipping incoherente')
    ids=None
    if global_window_ids is not None:
        ids=[_int(value,'global_window_id') for value in global_window_ids]
        _require(len(ids)==size and ids==sorted(set(ids)) and min(ids)>=0,
            'IDs globales deben ser enteros únicos ordenados y no negativos')
    fp_pred,integer_pred=integer_argmax(fp),integer_argmax(integer)
    _require(np.array_equal(integer_pred,integer_argmax(outputs)),
        'Argmax INT8 crudo y des-cuantizado no coinciden')
    changed=np.flatnonzero(fp_pred!=integer_pred).tolist()
    changed_rows=[dict(row_index=index,global_window_id=ids[index] if ids is not None else None,
        record_id=records[index],true_label=int(truth[index]),FP32_prediction=int(fp_pred[index]),
        INT8_prediction=int(integer_pred[index])) for index in changed]
    by_record={value:0 for value in sorted(set(records))}
    for index in changed:by_record[records[index]]+=1
    fp_metrics,integer_metrics=metrics3(truth,fp_pred),metrics3(truth,integer_pred)
    input_endpoint=(inputs==-128)|(inputs==127)
    output_endpoint=(outputs==-128)|(outputs==127)
    ties=np.sum(outputs==outputs.max(axis=1,keepdims=True),axis=1)>1
    clipped_by_window=np.count_nonzero(clipping,axis=1)
    difference=np.abs(np.subtract(fp,integer,dtype=np.float32))
    return dict(windows=size,features=normalized.shape[1],input_clipped_feature_values=clipped_total,
        input_clipped_by_feature=np.count_nonzero(clipping,axis=0).tolist(),
        input_clipped_by_window=clipped_by_window.tolist(),
        input_clipped_window_count=int(np.count_nonzero(clipped_by_window)),
        input_clipped_feature_value_fraction=clipped_total/normalized.size,
        input_clipped_global_window_ids=[ids[index] for index in np.flatnonzero(clipped_by_window)] if ids is not None else None,
        input_endpoint_value_count=int(np.count_nonzero(input_endpoint)),
        input_endpoint_values_without_clipping=int(np.count_nonzero(input_endpoint & ~clipping)),
        output_endpoint_value_count=int(np.count_nonzero(output_endpoint)),
        endpoint_values_are_not_equivalent_to_clipping=True,
        INT8_output_tied_argmax_window_count=int(np.count_nonzero(ties)),
        argmax_tie_rule='lowest class index',class_changes=len(changed),class_change_fraction=len(changed)/size,
        class_change_rows=changed_rows,class_changes_per_record=by_record,
        FP32_correct_INT8_incorrect=int(np.count_nonzero((fp_pred==truth)&(integer_pred!=truth))),
        FP32_incorrect_INT8_correct=int(np.count_nonzero((fp_pred!=truth)&(integer_pred==truth))),
        metrics_FP32=fp_metrics,metrics_INT8=integer_metrics,
        INT8_minus_FP32_accuracy=integer_metrics['accuracy']-fp_metrics['accuracy'],
        INT8_minus_FP32_macro_f1=integer_metrics['macro_f1']-fp_metrics['macro_f1'],
        INT8_minus_FP32_accuracy_pp=100*(integer_metrics['accuracy']-fp_metrics['accuracy']),
        INT8_minus_FP32_macro_f1_pp=100*(integer_metrics['macro_f1']-fp_metrics['macro_f1']),
        maximum_absolute_probability_difference=float(difference.max()),
        mean_absolute_probability_difference=float(difference.mean(dtype=np.float64)),
        quantization_verified=True,output_dequantization_verified=True,
        no_statistical_noninferiority_claim=True)


def _metric_row(row):
    _require(isinstance(row,dict),'Fila de métricas debe ser diccionario')
    outer=_int(row['outer_load_hp'],'outer_load_hp')
    _require(outer in range(4),'Carga exterior fuera de 0..3')
    seed=row['seed']
    if seed is not None:
        seed=_int(seed,'seed');_require(seed in OUTER_SEEDS,'Semilla fuera de las diez fijadas')
    for key in ('procedure','precision'):
        _require(isinstance(row[key],str) and row[key] and row[key].strip()==row[key],key+' inválido')
    family=row['procedure'].casefold()
    if family in ('rf','svm'):
        _require(row['precision']=='native','RF/SVM conservan su predicción nativa')
        _require(seed is None if family=='svm' else seed in OUTER_SEEDS,
            'SVM requiere seed=None; RF requiere las diez semillas fijadas')
    elif row['procedure'] in ('baseline_F14','performance_only','validated_total_cost'):
        _require(row['precision'] in ('FP32','INT8') and seed in OUTER_SEEDS,
            'Cada procedimiento DNN requiere precisión y semillas fijadas')
    normalized=dict(procedure=row['procedure'],precision=row['precision'],outer_load_hp=outer,seed=seed,
        accuracy=_rate(row['accuracy'],'accuracy'),macro_f1=_rate(row['macro_f1'],'macro_f1'))
    classes=row['per_class']
    _require(isinstance(classes,list) and len(classes)==3,'Se requieren tres métricas de clase')
    per_class=[]
    for expected,value in enumerate(classes):
        _require(_int(value['label'],'label')==expected,'Clases fuera de orden [0,1,2]')
        support=_int(value['support'],'support');_require(support>=0,'Soporte negativo')
        per_class.append(dict(label=expected,support=support,**{key:_rate(value[key],key)
            for key in ('precision','recall','f1')}))
    _require(sum(value['support'] for value in per_class)>0,'Soportes de clase vacíos')
    _require(abs(fsum(value['f1'] for value in per_class)/3-normalized['macro_f1'])<=_EPS,
        'Macro-F1 no corresponde a las tres clases')
    normalized['per_class']=per_class
    return normalized


def _complete_groups(cells):
    rows=[_metric_row(row) for row in cells]
    _require(rows,'No hay filas de evaluación para agregar')
    groups={};seen=set()
    for row in rows:
        key=(row['procedure'],row['precision'],row['outer_load_hp'],row['seed'])
        _require(key not in seen,'Fila por carga/semilla duplicada');seen.add(key)
        groups.setdefault(key[:2],[]).append(row)
    for group in groups.values():
        seeds={r['seed'] for r in group}
        expected_seeds=(None,) if seeds=={None} else OUTER_SEEDS
        _require(seeds==set(expected_seeds),'Semillas incompletas o SVM repetido como ensayos estocásticos')
        _require({(r['outer_load_hp'],r['seed']) for r in group}==
            {(outer,seed) for outer in range(4) for seed in expected_seeds},'Faltan cargas/semillas de evaluación')
        group.sort(key=lambda r:(r['outer_load_hp'],-1 if r['seed'] is None else r['seed']))
    for outer in range(4):
        supports={tuple(value['support'] for value in row['per_class'])
            for row in rows if row['outer_load_hp']==outer}
        _require(len(supports)==1,'Los procedimientos/semillas no evaluaron los mismos soportes por carga')
    return groups


def _mean(values):
    values=list(values);_require(values,'Media sin valores')
    return fsum(values)/len(values)


def _summary(values):
    values=list(values)
    return dict(count=len(values),mean=_mean(values),sample_standard_deviation=stdev(values) if len(values)>1 else None,
        minimum=min(values),maximum=max(values),ddof=1)


def aggregate_rows(cells):
    """Agregación descriptiva: cada carga y semilla tiene el mismo peso.

    Una fila por (procedure,precision,outer_load_hp,seed). Requiere accuracy,
    macro_f1 y per_class de metrics3. Los grupos estocásticos deben contener
    las 4×10 celdas fijadas; SVM usa cuatro celdas con seed=None, sin réplicas.
    No promedia probabilidades ni calcula intervalos/p-valores sobre ventanas.
    """
    groups=_complete_groups(cells);summaries=[]
    for (procedure,precision),group in sorted(groups.items()):
        seeds=sorted({r['seed'] for r in group},key=lambda seed:-1 if seed is None else seed)
        loads=[];by_seed=[]
        for outer in range(4):
            subset=[r for r in group if r['outer_load_hp']==outer]
            loads.append(dict(outer_load_hp=outer,seed_count=len(subset),
                accuracy_across_seeds=_summary(r['accuracy'] for r in subset),
                macro_f1_across_seeds=_summary(r['macro_f1'] for r in subset)))
        for seed in seeds:
            subset=[r for r in group if r['seed']==seed]
            by_seed.append(dict(seed=seed,mean_accuracy_across_loads=_mean(r['accuracy'] for r in subset),
                mean_macro_f1_across_loads=_mean(r['macro_f1'] for r in subset),
                worst_load_macro_f1=min(r['macro_f1'] for r in subset)))
        per_class=[dict(label=label,**{key:_mean(r['per_class'][label][key] for r in group)
            for key in ('precision','recall','f1')}) for label in range(3)]
        summaries.append(dict(procedure=procedure,precision=precision,cells=len(group),seeds=seeds,
            mean_accuracy=_mean(item['accuracy_across_seeds']['mean'] for item in loads),
            mean_macro_f1=_mean(item['macro_f1_across_seeds']['mean'] for item in loads),
            worst_load_mean_macro_f1=min(item['macro_f1_across_seeds']['mean'] for item in loads),
            minimum_cell_macro_f1=min(r['macro_f1'] for r in group),maximum_cell_macro_f1=max(r['macro_f1'] for r in group),
            per_class_equal_load_seed_means=per_class,per_load=loads,per_seed=by_seed,
            accuracy_of_four_load_means_across_seeds=_summary(r['mean_accuracy_across_loads'] for r in by_seed),
            macro_f1_of_four_load_means_across_seeds=_summary(r['mean_macro_f1_across_loads'] for r in by_seed),
            deterministic_single_fit_per_load=seeds==[None]))
    return dict(status='COMPLETE_EQUAL_LOAD_AND_SEED_DESCRIPTIVE_AGGREGATION',groups=summaries,
        window_count_weighting_used=False,probability_ensemble_used=False,confidence_intervals_computed=False,
        seeds_are_independent_physical_acquisitions=False,overlapping_windows_are_IID=False,
        sample_SD_interpretation='Descriptive variation across fixed seeds on the same corpus; per-load SD conditions on a fixed heldout load. Not a population uncertainty estimate.')


def paired_differences(cells,left,right,precision='INT8'):
    """Diferencias left−right sobre las mismas 4 cargas y 10 semillas.

    No aplica pruebas de hipótesis o no inferioridad; los márgenes fijados son
    criterios de ingeniería. Un grupo determinista también puede compararse
    si ambos usan seed=None; nunca se replica diez veces para fabricar pares.
    """
    _require(isinstance(left,str) and isinstance(right,str) and left!=right,'Se requieren procedimientos distintos')
    groups=_complete_groups(cells)
    _require((left,precision) in groups and (right,precision) in groups,'Faltan procedimientos a comparar')
    a={(r['outer_load_hp'],r['seed']):r for r in groups[left,precision]}
    b={(r['outer_load_hp'],r['seed']):r for r in groups[right,precision]}
    _require(set(a)==set(b),'Los procedimientos no comparten cargas y semillas exactas')
    differences=[]
    for outer,seed in sorted(a,key=lambda key:(key[0],-1 if key[1] is None else key[1])):
        first,second=a[outer,seed],b[outer,seed]
        differences.append(dict(outer_load_hp=outer,seed=seed,
            macro_f1_difference_pp=100*(first['macro_f1']-second['macro_f1']),
            accuracy_difference_pp=100*(first['accuracy']-second['accuracy'])))
    seeds=sorted({row['seed'] for row in differences},key=lambda seed:-1 if seed is None else seed)
    per_seed=[dict(seed=seed,mean_macro_f1_difference_pp=_mean(r['macro_f1_difference_pp'] for r in differences if r['seed']==seed),
        mean_accuracy_difference_pp=_mean(r['accuracy_difference_pp'] for r in differences if r['seed']==seed)) for seed in seeds]
    per_load=[dict(outer_load_hp=outer,mean_macro_f1_difference_pp=_mean(r['macro_f1_difference_pp'] for r in differences if r['outer_load_hp']==outer),
        mean_accuracy_difference_pp=_mean(r['accuracy_difference_pp'] for r in differences if r['outer_load_hp']==outer)) for outer in range(4)]
    return dict(left=left,right=right,precision=precision,sign='left minus right',pairs=len(differences),
        mean_macro_f1_difference_pp=_mean(r['mean_macro_f1_difference_pp'] for r in per_load),
        mean_accuracy_difference_pp=_mean(r['mean_accuracy_difference_pp'] for r in per_load),
        four_load_mean_macro_f1_differences_across_seeds=_summary(r['mean_macro_f1_difference_pp'] for r in per_seed),
        differences=differences,per_seed=per_seed,per_load=per_load,
        confidence_interval_computed=False,p_value_computed=False,noninferiority_claimed=False,
        physical_acquisition_independence_claimed=False)
