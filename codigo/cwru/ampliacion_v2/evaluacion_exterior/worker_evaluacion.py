"""Inferencia exterior de modelos sellados, sin entrenar o recalibrar.

Un proceso evalúa una carga reservada y conserva predicciones por ventana.
No publica métricas en stdout, decide máscaras o reemplaza la clase DNN por
la decisión del detector. Importar este módulo no importa runtimes de ML.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[4]
BASE=ROOT/'codigo/cwru/ampliacion_v2'
for directory in ('busqueda_interna','detector_interno','representantes','piloto_clasicos'):
    path=str(BASE/directory)
    if path not in sys.path:sys.path.insert(0,path)


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path,value):
    with Path(path).open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n')


def save_npz(path,**arrays):
    import numpy as np
    with Path(path).open('xb') as stream:np.savez(stream,**arrays)


def versions():
    return dict(python=platform.python_version(),**{name:importlib.metadata.version(package)
        for name,package in [('numpy','numpy'),('sklearn','scikit-learn'),('tensorflow','tensorflow'),
            ('keras','keras'),('scipy','scipy'),('joblib','joblib')]})


def bounded(relative):
    value=Path(relative)
    require(not value.is_absolute() and value.parts and '..' not in value.parts,'Ruta de modelo no relativa o insegura')
    path=(ROOT/value).resolve()
    require(path!=ROOT.resolve() and path.is_relative_to(ROOT.resolve()),'Modelo fuera del proyecto')
    return path


def source_snapshot(sourcehashes):
    require(isinstance(sourcehashes,dict) and sourcehashes,'Fuentes ausentes')
    actual={relative:sha(bounded(relative)) for relative in sorted(sourcehashes)}
    require(actual==sourcehashes,'Fuentes distintas de la preparación autorizada')
    return actual


def validate_job(job):
    """La autorización autentica el inventario completo; aquí se contrastan roles."""
    outer=job['outer_load_hp']
    require(type(outer) is int and outer in range(4),'Carga reservada inválida')
    require(job['job_id']==f'outer_eval_{outer}','ID de evaluación no canónico')
    ids=job['heldout_global_window_ids']
    require(isinstance(ids,list) and ids and all(type(i) is int and i>=0 for i in ids)
        and ids==sorted(set(ids)),'IDs exteriores no canónicos')
    records=job['heldout_MAT']
    require(isinstance(records,list) and len(records)==len(set(records))==7
        and all(isinstance(record,str) and record and record.strip()==record for record in records),
        'La carga reservada requiere siete MAT identificados')
    models=job['models']
    require(isinstance(models,list) and models and len({m['job_id'] for m in models})==len(models),
        'Inventario vacío o repetido')
    loads=[load for load in range(4) if load!=outer]
    for model in models:
        require(model['family'] in ('DNN','RF','SVM','detector') and model['outer_load_hp']==outer
            and type(model['outer_load_hp']) is int,'Modelo de otra familia/carga')
        identifier=model['job_id']
        require(isinstance(identifier,str) and identifier and Path(identifier).name==identifier
            and identifier not in ('.','..'),'ID de modelo no seguro')
        roles=model['roles']
        require(roles['outer_load_hp']==outer and type(roles['outer_load_hp']) is int
            and roles['inner_validation_load_hp'] is None
            and all(type(v) is int for v in roles['training_loads']+roles['evaluation_loads'])
            and roles['training_loads']==loads and roles['evaluation_loads']==[outer],
            'El modelo no excluyó la carga evaluada durante el reajuste')
        paths=model['paths'];digests=model['artifact_sha256']
        require(isinstance(paths,dict) and paths and set(paths)==set(digests),'Artefactos de modelo incompletos')
        expected_artifacts={'DNN':{'modelo.keras','modelo_fp32.tflite','modelo_int8.tflite','scaler_train_only.npz'},
            'RF':{'modelo.joblib'},'SVM':{'modelo.joblib','scaler_train_only.npz'},
            'detector':{'parametros.npz','calibracion.npz'}}
        require(set(paths)==expected_artifacts[model['family']],'Artefactos ajenos a la familia del modelo')
        for name,relative in paths.items():
            require(Path(name).name==name and bounded(relative).name==name,'Nombre/ruta de artefacto no corresponde')
        if model['family']=='DNN':
            indices=model['indices']
            require(isinstance(indices,list) and indices and indices==sorted(set(indices))
                and all(type(i) is int and 0<=i<14 for i in indices),'Columnas DNN inválidas')
            require(type(model['seed']) is int and model['seed'] in (11,22,33,42,44,55,66,77,88,99),
                'Semilla DNN no predefinida')
        elif model['family']=='detector':
            require(isinstance(model['indices'],list) and model['indices']
                and all(type(i) is int and 0<=i<14 and i not in (4,11) for i in model['indices'])
                and model['indices']==sorted(set(model['indices'])),'Columnas detector inválidas')
            require(roles['detector_fit_normal_loads']==[load for load in loads if load!=max(loads)]
                and roles['detector_calibration_normal_loads']==[max(loads)]
                and all(type(v) is int for v in roles['detector_fit_normal_loads']+roles['detector_calibration_normal_loads'])
                and roles['detector_refit_after_calibration'] is False,'Roles del detector alterados')
        elif model['family']=='RF':
            require(type(model['seed']) is int and model['seed'] in (11,22,33,42,44,55,66,77,88,99),
                'Semilla RF no predefinida')
        else:
            require(model['seed'] is None,'SVM no debe repetirse por semillas')
    return models


def model_snapshot(models):
    values={}
    for model in models:
        for name,relative in model['paths'].items():
            actual=sha(bounded(relative))
            require(actual==model['artifact_sha256'][name],'Modelo/scaler modificado: '+relative)
            values[relative]=actual
        path=bounded(model['result_path'])
        actual=sha(path)
        require(actual==model['result_sha256'],'Resultado de reajuste modificado')
        values[model['result_path']]=actual
    return values


def load_heldout(job):
    """Materializar exclusivamente la carga exterior que autoriza este worker."""
    import datos_busqueda as data
    from datos_detector import load_rows
    context=data.context();rows=context[1]
    ids=job['heldout_global_window_ids']
    require(all(i<len(rows) and int(rows[i]['global_window_id'])==i
        and int(rows[i]['load_hp'])==job['outer_load_hp'] for i in ids),'IDs de una carga ajena')
    require(ids==[int(r['global_window_id']) for r in rows if int(r['load_hp'])==job['outer_load_hp']],
        'La evaluación requiere todas las ventanas de la carga reservada')
    require({str(rows[i]['file_id']) for i in ids}==set(job['heldout_MAT']),'MAT exteriores distintos del mapa')
    return load_rows(ids,context=context)


def validate_heldout(job,raw,y,records,provenance):
    import numpy as np
    size=len(job['heldout_global_window_ids'])
    require(raw.dtype==np.float32 and raw.shape==(size,14) and np.isfinite(raw).all(),
        'Entradas exteriores requieren F14 float32 finito')
    require(y.dtype==np.int64 and y.shape==(size,) and set(y.tolist())=={0,1,2},
        'Etiquetas exteriores deben contener las tres clases')
    require(len(records)==size and set(records)==set(job['heldout_MAT'])
        and all(isinstance(r,str) and r for r in records),'Registros de entrada incompatibles')
    require(provenance['global_window_ids']==job['heldout_global_window_ids']
        and provenance['selected_loads_hp']==[job['outer_load_hp']]
        and set(provenance['selected_record_ids'])==set(records)
        and provenance['shape']==[size,14] and provenance['loaded_dtype']=='float32'
        and provenance['only_requested_rows_selected_or_materialized'] is True,
        'Procedencia de carga reservada no corresponde')


def load_scaler(path,indices,with_indices):
    """Leer/exportar estadísticas fijadas; nunca recalcularlas con evaluación."""
    import numpy as np
    keys={'mean64','scale64','var64','mean32','scale32'}|({'feature_indices'} if with_indices else set())
    with np.load(path,allow_pickle=False) as saved:
        require(set(saved.files)==keys,'Scaler con esquema diferente')
        values={key:saved[key].copy() for key in saved.files}
    for key in ('mean64','scale64','var64','mean32','scale32'):
        a=values[key];dtype=np.float64 if key.endswith('64') else np.float32
        require(a.dtype==dtype and a.shape==(len(indices),) and np.isfinite(a).all(),'Estadísticas guardadas no válidas')
    require((values['scale64']>0).all() and (values['scale32']>0).all() and (values['var64']>=0).all(),
        'Escala/varianza no válida')
    require(np.array_equal(values['mean32'],values['mean64'].astype(np.float32))
        and np.array_equal(values['scale32'],values['scale64'].astype(np.float32)),'Exportación float32 distinta')
    if with_indices:
        require(values['feature_indices'].dtype==np.int64 and values['feature_indices'].tolist()==indices,
            'Columnas del scaler DNN diferentes')
    return values


def block_training():
    """Guardas del proceso: cualquier ajuste accidental falla explícitamente."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.cluster import KMeans
    blocked=[]
    def forbidden(*args,**kwargs):raise RuntimeError('Evaluación exterior: entrenar o recalibrar está prohibido')
    for cls,names in [(StandardScaler,('fit','fit_transform','partial_fit')),(RandomForestClassifier,('fit',)),
        (SVC,('fit',)),(KMeans,('fit','fit_predict'))]:
        for name in names:
            if hasattr(cls,name):setattr(cls,name,forbidden);blocked.append(cls.__name__+'.'+name)
    return blocked,forbidden


def evaluate_models(plan,job,directory,raw,y,records,provenance):
    """Inferir una vez por modelo único; guardar alias sin duplicar cómputo."""
    import numpy as np
    from numerica_representantes import normalize_f32,integer_argmax
    from metricas_evaluacion import supervised,detector as detector_metrics,quantization_report
    validate_heldout(job,raw,y,records,provenance)
    models=validate_job(job);source_before=source_snapshot(plan['source_sha256'])
    before=model_snapshot(models);blocked,forbidden=block_training()
    ids=np.asarray(job['heldout_global_window_ids'],dtype=np.int64);record_array=np.asarray(records,dtype=str)
    write(directory/'procedencia.json',provenance)
    save_npz(directory/'entrada_evaluacion.npz',raw_features14=raw,y_true=y,global_window_ids=ids,record_ids=record_array)
    results=[];tf=None
    counts=dict(DNN_models=0,DNN_FP32_invocations=0,DNN_INT8_invocations=0,
        RF_models=0,RF_prediction_batches=0,RF_predicted_rows=0,SVM_models=0,SVM_prediction_batches=0,
        SVM_predicted_rows=0,detector_models=0,detector_score_batches=0,detector_scored_rows=0)
    for model in models:
        started=time.monotonic();family=model['family'];identifier=model['job_id']
        print(json.dumps(dict(stage='infer_model',model_job_id=identifier,family=family)),flush=True)
        filename=identifier+'_predicciones.npz'
        base=dict(model_job_id=identifier,family=family,candidate=model.get('candidate'),seed=model.get('seed'),
            config_index=model.get('config_index'),outer_load_hp=job['outer_load_hp'],
            prediction_file=filename,model_source_sha256={p:before[p] for p in model['paths'].values()},
            fit_signature=model['fit_signature'],selection_signature=model['selection_signature'],
            prediction_scope='heldout_load_for_this_final_refit',windows=len(y))
        paths={name:bounded(relative) for name,relative in model['paths'].items()}
        if family=='DNN':
            if tf is None:
                import tensorflow as tf
                # Solo intérpretes Lite: los métodos de entrenamiento no son una ruta permitida.
                tf.keras.Model.fit=forbidden;tf.keras.Model.train_on_batch=forbidden
                tf.lite.TFLiteConverter.convert=forbidden
                blocked.extend(['keras.Model.fit','keras.Model.train_on_batch','lite.TFLiteConverter.convert'])
            from worker_representantes import evaluate_lite
            indices=model['indices'];scaler=load_scaler(paths['scaler_train_only.npz'],indices,True)
            z=normalize_f32(raw[:,indices],scaler['mean32'],scaler['scale32'])
            fp,fp_outputs,fp_inputs,fp_details=evaluate_lite(tf,paths['modelo_fp32.tflite'].read_bytes(),z,False)
            iq,iq_outputs,iq_inputs,iq_details=evaluate_lite(tf,paths['modelo_int8.tflite'].read_bytes(),z,True)
            require(np.array_equal(fp_inputs,z),'El modelo FP32 recibió valores diferentes del scaler fijado')
            original=read(bounded(model['result_path']))
            for mode,current in [('fp32_details',fp_details),('int8_details',iq_details)]:
                require({k:v for k,v in current.items() if k!='input_clipped_values'}==
                    {k:v for k,v in original[mode].items() if k!='input_clipped_values'},
                    'Contrato del intérprete distinto del modelo reajustado')
            fp_pred=integer_argmax(fp);iq_pred=integer_argmax(iq)
            quant=quantization_report(z,iq_inputs,iq_outputs,fp,iq,y,records,
                iq_details['input_quantization'],iq_details['output_quantization'],global_window_ids=ids.tolist())
            require(quant['input_clipped_feature_values']==iq_details['input_clipped_values'],
                'El conteo de saturación difiere del intérprete')
            save_npz(directory/filename,global_window_ids=ids,y_true=y,record_ids=record_array,
                normalized_features=z,fp32_inputs=fp_inputs,int8_inputs=iq_inputs,
                fp32_outputs=fp_outputs,int8_outputs=iq_outputs,fp32_probabilities=fp,int8_probabilities=iq,
                fp32_predictions=fp_pred,int8_predictions=iq_pred,
                input_quantization=np.asarray(iq_details['input_quantization'],dtype=np.float64),
                output_quantization=np.asarray(iq_details['output_quantization'],dtype=np.float64),
                input_clipped_by_feature=np.asarray(quant['input_clipped_by_feature'],dtype=np.int64),
                input_clipped_by_window=np.asarray(quant['input_clipped_by_window'],dtype=np.int64))
            base.update(metrics=dict(FP32=supervised(y,fp_pred,records),INT8=supervised(y,iq_pred,records)),
                quantization_report=quant,fp32_details=fp_details,int8_details=iq_details,
                procedures=model.get('procedures'),epochs=model['epochs'],indices=indices)
            counts['DNN_models']+=1;counts['DNN_FP32_invocations']+=len(y);counts['DNN_INT8_invocations']+=len(y)
        elif family in ('RF','SVM'):
            import joblib
            estimator=joblib.load(paths['modelo.joblib'])
            original=read(bounded(model['result_path']))
            require(np.array_equal(estimator.classes_,[0,1,2]) and estimator.n_features_in_==14
                and all(estimator.get_params(deep=False).get(k)==v for k,v in original['params'].items()),
                'Estimador clásico incompatible con el reajuste sellado')
            if family=='RF':
                z=raw
                require(len(estimator.estimators_)==original['params']['n_estimators'],'RF incompleto')
            else:
                require(estimator.fit_status_==0,'SVM guardado no convergió')
                scaler=load_scaler(paths['scaler_train_only.npz'],list(range(14)),False)
                z=normalize_f32(raw,scaler['mean32'],scaler['scale32'])
            prediction=estimator.predict(z)
            values=estimator.predict_proba(z) if family=='RF' else estimator.decision_function(z)
            require(prediction.dtype==np.int64 and prediction.shape==(len(y),)
                and set(prediction.tolist())<={0,1,2} and values.shape==(len(y),3) and np.isfinite(values).all(),
                'Salidas clásicas no válidas')
            save_npz(directory/filename,global_window_ids=ids,y_true=y,record_ids=record_array,
                model_features=z,predictions=prediction,values=values)
            base.update(metrics=supervised(y,prediction,records),native_predict_rule=True,
                values_kind='predict_proba' if family=='RF' else 'decision_function_not_argmax_classifier',
                input_representation='raw_float32_F14' if family=='RF' else 'stored_train_only_scaler_explicit_float32_F14')
            counts[family+'_models']+=1;counts[family+'_prediction_batches']+=1
            counts[family+'_predicted_rows']+=len(y)
        else:
            from datos_detector import load_params
            from detector_numerico import detector_score
            params=load_params(paths['parametros.npz'])
            require(params['indices']==model['indices'],'Índices del detector diferentes del inventario')
            scores=detector_score(raw,params);rejected=scores>params['threshold32']
            require(scores.dtype==np.float32 and scores.shape==(len(y),) and np.isfinite(scores).all(),
                'Scores del detector no válidos')
            save_npz(directory/filename,global_window_ids=ids,y_true=y,record_ids=record_array,
                scores=scores,rejected=rejected,threshold32=np.asarray(params['threshold32'],dtype=np.float32))
            base.update(metrics=detector_metrics(y,rejected,records),threshold_float32=float(params['threshold32']),
                threshold_bits=np.float32(params['threshold32']).tobytes().hex(),masks=model['masks'],
                procedures=model.get('procedures'),indices=params['indices'],class_override=False,
                strict_rejection_rule='score > stored_threshold; equality accepted')
            counts['detector_models']+=1;counts['detector_score_batches']+=1
            counts['detector_scored_rows']+=len(y)
        base['prediction_sha256']=sha(directory/filename)
        base['prediction_seconds']=time.monotonic()-started
        results.append(base)
    after=model_snapshot(models);source_after=source_snapshot(plan['source_sha256'])
    require(before==after and source_before==source_after,'La evaluación modificó fuentes o modelos')
    if 'expected_counts' in job:
        require({family:counts[family+'_models'] for family in ('DNN','RF','SVM','detector')}==job['expected_counts'],
            'Conteo de modelos distinto del plan')
    if 'expected_inference_counts' in job:
        require({key:counts[key] for key in job['expected_inference_counts']}==job['expected_inference_counts'],
            'Número de inferencias distinto del plan')
    return dict(model_results=results,inference_counts=counts,model_hashes_before=before,model_hashes_after=after,
        model_sha256_before=before,model_sha256_after=after,
        source_sha256_before=source_before,source_sha256_after=source_after,training_blocked_methods=blocked)


def run(campaign,outer_load):
    from contrato_evaluacion import authorize
    from ejecutar_busqueda_interna import write_manifest
    campaign=Path(campaign).resolve();started=time.monotonic()
    plan,job=authorize(campaign,outer_load)
    require(type(outer_load) is int and outer_load==job['outer_load_hp'],'Carga/solicitud incompatibles')
    validate_job(job)
    require(versions()==plan['expected_versions'],'Versiones diferentes del entorno preparado')
    require(all(os.environ.get(k)==v for k,v in plan['worker_environment'].items()),'Variables del worker no fijadas')
    directory=campaign/job['job_id']
    require(not directory.exists() and directory.is_relative_to(ROOT.resolve()),'No sobrescribir evaluación existente')
    directory.mkdir(parents=False,exist_ok=False)
    raw,y,records,provenance=load_heldout(job)
    evaluated=evaluate_models(plan,job,directory,raw,y,records,provenance)
    result=dict(status='PASS',job_id=job['job_id'],outer_load_hp=outer_load,
        selection_signature=plan['selection_signature'],versions=versions(),
        evaluation_signature=plan['evaluation_signatures'][job['job_id']],source_sha256=plan['source_sha256'],
        provenance=provenance,input_snapshot_sha256=sha(directory/'entrada_evaluacion.npz'),
        evaluated_windows=len(y),evaluated_MAT=len(set(records)),heldout_global_window_ids=job['heldout_global_window_ids'],
        new_training_fits=0,scalers_refitted=False,INT8_recalibrated=False,detector_threshold_refitted=False,
        training_executed=False,parameter_fit_executed=False,threshold_calibrated=False,model_conversion_executed=False,
        mask_or_seed_selection_executed=False,seed_ensembles=False,detector_class_override=False,
        channels=['DE','FE'],sensor_positions_not_axes=True,STM32_cost_measured=False,
        predictions_are_for_previously_explored_retrospective_dataset=True,
        metrics_printed_to_stdout=False,partial_results_are_not_rule_change_input=True,
        elapsed_worker_seconds=time.monotonic()-started,**evaluated)
    write(directory/'RESULTADO.json',result)
    write_manifest(directory/'MANIFEST_OUTER_SHA256.json',directory=directory)
    print(json.dumps(dict(stage='outer_inference_complete',job_id=job['job_id'],status='PASS',metrics_printed=False)),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--outer-load',type=int,required=True)
    args=parser.parse_args()
    try:run(args.campaign,args.outer_load)
    except Exception:print(traceback.format_exc(),file=sys.stderr,flush=True);raise SystemExit(1)


if __name__=='__main__':main()
