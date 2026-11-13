"""Descriptivos de reajustes terminados; solo JSON/logs, sin ML o escrituras.

Las métricas disponibles son de entrenamiento. No constituyen evaluación
exterior ni estiman generalización. Los tamaños TFLite no son Flash de firmware.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct

ROOT=Path(__file__).resolve().parents[4]
FAMILIES=('DNN','RF','SVM','detector')
EXPECTED_COUNTS=dict(DNN=80,RF=40,SVM=4,detector=7)


def _require(condition,message):
    if not condition:
        raise ValueError(message)


def _number(value,minimum=0,maximum=None):
    _require(type(value) in (int,float) and math.isfinite(value) and value>=minimum
        and (maximum is None or value<=maximum),'Contador/métrica no finito o fuera de rango')
    return float(value)


def _integer(value,minimum=0,maximum=None):
    _require(type(value) is int and value>=minimum and (maximum is None or value<=maximum),
        'Contador entero inválido')
    return value


def _describe(values):
    values=list(values)
    _require(bool(values),'No resumir una familia vacía')
    for value in values:_number(value)
    return dict(count=len(values),minimum=min(values),maximum=max(values),
        mean=statistics.mean(values),median=statistics.median(values),sum=math.fsum(values))


def summarize(plan,campaign):
    """Recoger metadatos de 131 trabajos; no ejecutar ajustes o inferencias."""
    campaign=Path(campaign).resolve()
    _require(campaign.is_dir() and campaign.is_relative_to(ROOT.resolve()),'Campaña fuera del proyecto')
    jobs=plan['jobs']
    _require(len(jobs)==len({job['job_id'] for job in jobs})==131
        and dict(Counter(job['family'] for job in jobs))==EXPECTED_COUNTS,'Inventario no corresponde a 131 reajustes')
    source_sha256={}

    def record(path):
        path=Path(path).resolve()
        _require(path.is_relative_to(ROOT.resolve()) and path.is_file(),'Recurso fuera del proyecto o ausente')
        digest=hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
        source_sha256[path.relative_to(ROOT).as_posix()]=digest.hexdigest()
        return path

    def read(path):
        return json.loads(record(path).read_text(encoding='utf-8-sig'))

    results=[];times=defaultdict(list);stderr_contents=[]
    for index,job in enumerate(jobs):
        job_id=job['job_id'];family=job['family']
        _require(type(job_id) is str and Path(job_id).name==job_id,'ID de trabajo no canónico')
        state=read(campaign/'pasos'/f'{index:03d}.json')
        result=read(campaign/job_id/'RESULTADO.json')
        _require(state['status']==result['status']=='PASS' and state['job_id']==result['job_id']==job_id
            and state['family']==result['family']==family and state['return_code']==0
            and state['automatic_retry'] is False,'Trabajo no cerrado PASS o ajeno al plan')
        seconds=_number(state['execution_verification_seconds'])
        _require(state['timeout_seconds']==job['timeout_seconds'],'Watchdog diferente del plan')
        _require(result['outer_load_hp']==job['outer_load_hp'] and result['roles']==job['roles']
            and result['outer_evaluation_executed'] is False,'Resultado pertenece a otro ámbito o evalúa exterior')
        times[family].append(seconds)
        text=record(campaign/(job_id+'_stderr.log')).read_text(encoding='utf-8-sig')
        stderr_contents.append((job_id,family,text))
        results.append(result)

    dnn=[r for r in results if r['family']=='DNN']
    groups=defaultdict(list);masks=defaultdict(list)
    for r in dnn:
        n=_integer(r['training_windows'],1)
        _integer(r['training_FP32_INT8_class_changes'],0,n)
        _integer(r['training_keras_batch1_FP32_class_changes'],0,n)
        _integer(r['training_saturated_input_values'],0,n*r['architecture'][0])
        for key in ('accuracy','macro_f1'):_number(r['training_only_diagnostics']['INT8'][key],0,1)
        _require(r['int8_details']['input_dtype']==r['int8_details']['output_dtype']=='int8'
            and set(r['int8_details']['tensor_dtypes'])<=set(('int8','int32','int64'))
            and r['freeze']['weights_unchanged'] is True,'INT8 no completo o conversión altera pesos')
        _number(r['freeze']['maximum_absolute_error'])
        _integer(r['freeze']['class_changes'],0,n)
        for name in ('Keras','FP32','INT8'):_integer(r['model_bytes'][name],1)
        groups[r['outer_load_hp'],r['candidate']].append(r)
        masks[r['candidate']].append(r)
    per_scope=[]
    for (outer,mask),rows in sorted(groups.items()):
        _require(len(rows)==10 and sorted(r['seed'] for r in rows)==[11,22,33,42,44,55,66,77,88,99],
            'Ámbito DNN sin diez semillas completas')
        per_scope.append(dict(outer_load_hp=outer,candidate=mask,fits=len(rows),
            training_INT8_accuracy=_describe(r['training_only_diagnostics']['INT8']['accuracy'] for r in rows),
            training_INT8_macro_f1=_describe(r['training_only_diagnostics']['INT8']['macro_f1'] for r in rows),
            FP32_INT8_training_class_changes=sum(r['training_FP32_INT8_class_changes'] for r in rows),
            clipped_training_input_feature_values=sum(r['training_saturated_input_values'] for r in rows)))
    sizes=[]
    for mask,rows in sorted(masks.items()):
        observed={tuple(r['model_bytes'][key] for key in ('Keras','FP32','INT8')) for r in rows}
        _require(len(observed)==1,'Tamaño de contenedor varía dentro de una máscara; no ocultar su dispersión')
        keras,fp32,int8=next(iter(observed))
        sizes.append(dict(candidate=mask,fits=len(rows),Keras_bytes=keras,FP32_tflite_bytes=fp32,
            INT8_tflite_bytes=int8,INT8_minus_FP32_bytes=int8-fp32,INT8_container_smaller=int8<fp32))

    detectors=[]
    for r in results:
        if r['family']!='detector':continue
        threshold=_number(r['threshold_float32'])
        encoded=bytes.fromhex(r['threshold_bits'])
        _require(len(encoded)==4 and struct.unpack('<f',encoded)[0]==threshold,'Umbral no coincide con bits float32')
        _require(r['normal_only_fit'] is r['normal_only_calibration'] is True
            and r['fit_normal_records']==2 and r['calibration_normal_records']==1
            and r['refit_after_calibration'] is r['class_override'] is r['dnn_scaler_reused'] is False,
            'Política del detector diferente del protocolo')
        _integer(r['fit_rows'],1);_integer(r['calibration_rows'],1)
        detectors.append({key:r[key] for key in ('job_id','outer_load_hp','indices','masks','procedures',
            'threshold_float32','threshold_bits','fit_rows','calibration_rows','fit_normal_records','calibration_normal_records')})

    classical=[]
    for family in ('RF','SVM'):
        for outer in range(4):
            rows=[r for r in results if r['family']==family and r['outer_load_hp']==outer]
            _require(len(rows)==(10 if family=='RF' else 1),'Ámbito clásico incompleto')
            configs={r['config_index'] for r in rows}
            params=[{k:v for k,v in r['params'].items() if k!='random_state'} for r in rows]
            _require(len(configs)==1 and all(p==params[0] for p in params)
                and all(r['native_predict_rule'] is True for r in rows),'Comparador alterado entre semillas')
            if family=='RF':
                _require(sorted(r['seed'] for r in rows)==[11,22,33,42,44,55,66,77,88,99],
                    'Faltan semillas RF finales')
            else:_require(rows[0]['seed'] is None,'SVM repetido por semilla')
            for row in rows:_integer(row['model_bytes'],1)
            classical.append(dict(family=family,outer_load_hp=outer,fits=len(rows),
                config_index=next(iter(configs)),params_without_random_state=params[0],
                model_joblib_bytes=_describe(row['model_bytes'] for row in rows),
                joblib_container_is_not_embedded_Flash=True,native_predict_rule=True))

    patterns=dict(tensorflow_GPU_native_Windows='TensorFlow GPU support is not available on native Windows',
        converter_trackable_deprecation='Please consider providing the trackable_obj argument',
        converter_quantized_statistics='Statistics for quantized inputs were expected',
        tflite_interpreter_deprecation='Warning: tf.lite.Interpreter is deprecated',
        SVM_probability_deprecation='The `probability` parameter was deprecated')
    warnings={key:dict(files=sum(needle in text for _,_,text in stderr_contents),
        occurrences=sum(text.count(needle) for _,_,text in stderr_contents)) for key,needle in patterns.items()}
    errors=[dict(job_id=job_id,pattern=needle) for job_id,_,text in stderr_contents
        for needle in ('Traceback','ConvergenceWarning','ERROR','NaN') if needle in text]
    record(Path(__file__))
    return dict(status='FINAL_REFITS_METADATA_SUMMARY_ONLY_NO_OUTER_EVALUATION',
        counts=dict(EXPECTED_COUNTS,total=131),
        timing_seconds_by_family={family:_describe(times[family]) for family in FAMILIES},
        total_execution_verification_seconds=math.fsum(value for family in FAMILIES for value in times[family]),
        timing_scope='PC per-job preparation, subprocess and parent verification; shared preflight/replay excluded; not STM32 latency.',
        training_DNN_diagnostics=dict(fits=80,training_prediction_rows_repeated=sum(r['training_windows'] for r in dnn),
            INT8_accuracy=_describe(r['training_only_diagnostics']['INT8']['accuracy'] for r in dnn),
            INT8_macro_f1=_describe(r['training_only_diagnostics']['INT8']['macro_f1'] for r in dnn),
            FP32_INT8_training_class_changes=sum(r['training_FP32_INT8_class_changes'] for r in dnn),
            fits_with_FP32_INT8_training_class_changes=sum(r['training_FP32_INT8_class_changes']>0 for r in dnn),
            clipped_training_input_feature_values=sum(r['training_saturated_input_values'] for r in dnn),
            fits_with_training_input_clipping=sum(r['training_saturated_input_values']>0 for r in dnn),
            Keras_FP32_training_class_changes=sum(r['training_keras_batch1_FP32_class_changes'] for r in dnn),
            graph_freeze_class_changes=sum(r['freeze']['class_changes'] for r in dnn),
            graph_freeze_maximum_absolute_error=max(r['freeze']['maximum_absolute_error'] for r in dnn),
            by_outer_and_mask=per_scope),
        DNN_model_container_sizes=sizes,INT8_container_savings_are_not_universal=any(not r['INT8_container_smaller'] for r in sizes),
        classical_configurations=classical,detector_parameters=detectors,
        detector_policy=dict(normal_only_fit=True,normal_only_calibration=True,refit_after_calibration=False,
            class_override=False,threshold_is_observed_float32_P95=True,
            future_normal_acceptance_or_absolute_specificity_guaranteed=False),
        log_diagnostics=dict(empty_stderr_files=sum(not text for _,_,text in stderr_contents),
            nonempty_stderr_files=sum(bool(text) for _,_,text in stderr_contents),known_warnings=warnings,
            detected_error_patterns=errors,known_warnings_do_not_override_saved_tensor_operator_checks=True),
        interpretation=dict(metrics_are_training_only=True,overlapping_repeated_rows_are_not_independent_samples=True,
            no_generalization_or_selector_superiority_conclusion=True,model_bytes_are_not_firmware_Flash=True,
            no_external_detector_quality_conclusion=True),
        new_training_fits=0,new_model_predictions=0,outer_evaluation_executed=False,
        source_sha256=source_sha256)
