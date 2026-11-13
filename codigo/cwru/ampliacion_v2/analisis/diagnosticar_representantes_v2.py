"""Diagnostico reproducible de los32 representantes; sin entrenamiento.

Solo selecciona filas autorizadas por cada trabajo. Reproduce la aritmetica
del detector y hace forward NumPy de pesos ya guardados en validacion interna.
No invoca TensorFlow/TFLite, no convierte, modifica parametros o evalua outer.
"""
from __future__ import annotations

from collections import Counter
import csv
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'codigo/cwru/ampliacion_v2/representantes'))
from ejecutar_representantes import read,write,sha,verify_manifest,preflight,PREP
from datos_representantes import load_job_data,load_verified_context
from numerica_representantes import normalize_f32,quantize_int8,metrics3
sys.path.insert(0,str(ROOT/'codigo/cwru/ampliacion_v2/piloto_clasicos'))
from detector_numerico import detector_score
from reglas_v2 import weighted_p95

CAMPAIGN=ROOT/'resultados/cwru/ampliacion_v2/representantes_v2_2026-10-25/20261025T171847_957295Z'
CLOSURE=ROOT/'articulo/ampliacion_cwru/metodologia/cierre_representantes_v2_2026-10-25'
OUTPUT=ROOT/'articulo/ampliacion_cwru/metodologia/diagnostico_representantes_v2_2026-10-25'
FEATURE_FILE=ROOT/'datos/cwru/procesado/dual_12khz_w4096/features_14/feature_names.txt'
FEATURES=[line.split(': ',1)[1].strip() for line in FEATURE_FILE.read_text(encoding='utf-8-sig').splitlines()]


def require(ok,message):
    if not ok:raise ValueError(message)


def forward(x,weights):
    """Forward flotante de diagnostico, sin actualizar pesos ni crear optimizador."""
    require(x.dtype==np.float32 and x.ndim==2 and len(weights)==6,'Entrada/pesos incorrectos')
    value=x
    for layer in range(3):
        kernel,bias=weights[2*layer:2*layer+2]
        require(kernel.dtype==bias.dtype==np.float32 and kernel.shape[0]==value.shape[1]
                and bias.shape==(kernel.shape[1],),'Pesos con forma/dtype incorrectos')
        value=np.add(np.matmul(value,kernel),bias,dtype=np.float32)
        if layer<2:value=np.maximum(value,np.float32(0))
    logits=value
    shifted=np.subtract(logits,logits.max(axis=1,keepdims=True),dtype=np.float32)
    exp=np.exp(shifted,dtype=np.float32)
    probabilities=np.divide(exp,np.sum(exp,axis=1,keepdims=True,dtype=np.float32),dtype=np.float32)
    require(np.isfinite(probabilities).all(),'Forward no finito')
    return probabilities,logits


def write_csv(path,rows):
    require(rows,'CSV vacio')
    with path.open('x',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    entries=verify_manifest(CLOSURE/'MANIFEST_CIERRE_SHA256.json')
    plan,_=preflight()
    _,_,metadata,input_hashes=load_verified_context()
    by_id={int(row['global_window_id']):row for row in metadata}
    job_rows=[];feature_rows=[];record_rows=[];det_rows=[];normal_rows=[];window_rows=[]
    forward_scenarios={};global_seen={};global_clips=np.zeros(14,dtype=np.int64)
    replay_max=0.;n_diagnostic_forward_rows=0
    for job in plan['jobs']:
        job_id=job['job_id'];directory=CAMPAIGN/job_id;indices=job['feature_indices']
        result=read(directory/'resultado.json');state=read(directory/'estado_antes_conversion.json')
        raw,y,raw_val,y_val,provenance=load_job_data(job)
        for ids,array in ((job['training_global_window_ids'],raw),(job['validation_global_window_ids'],raw_val)):
            for key,row in zip(ids,array):
                if key in global_seen:require(np.array_equal(global_seen[key],row),'Una fila cambia entre scopes')
                else:global_seen[key]=row.copy()
        with np.load(directory/'scaler_train_only.npz',allow_pickle=False) as scaler:
            mean,scale=scaler['mean32'],scaler['scale32']
            check_mean=np.mean(raw[:,indices],axis=0,dtype=np.float64)
            check_var=np.var(raw[:,indices].astype(np.float64),axis=0)
            require(np.allclose(check_mean,scaler['mean64'],rtol=1e-12,atol=1e-12)
                    and np.allclose(check_var,scaler['var64'],rtol=1e-12,atol=1e-12),
                    'Estadisticos de scaler no reproducibles sobre train')
            x=normalize_f32(raw_val[:,indices],mean,scale)
        q,clipped_count=quantize_int8(x,*result['int8_details']['input_quantization'])
        quant_scale,zp=result['int8_details']['input_quantization']
        quant_scale=np.float32(quant_scale)
        affine=np.add(np.divide(x,quant_scale,dtype=np.float32),np.float32(zp),dtype=np.float32)
        rounded=np.rint(affine)
        clip=(rounded < -128)|(rounded > 127);clip_windows=clip.any(axis=1)
        with np.load(directory/'predicciones_validacion.npz',allow_pickle=False) as saved:
            require(np.array_equal(q,saved['int8_inputs']) and np.array_equal(y_val,saved['y_true'])
                    and clipped_count==result['int8_details']['input_clipped_values'],'Cuantizacion/etiquetas difieren')
            fp_pred=saved['fp32_predictions'].copy();int_pred=saved['int8_predictions'].copy()
            reference=saved['keras_batch1'].copy()
        with np.load(directory/'checkpoint_antes_conversion.npz',allow_pickle=False) as saved:
            weights=[saved[f'weight_{i:02d}'].copy() for i in range(6)]
        original,logits=forward(x,weights)
        original_pred=np.argmax(original,axis=1)
        error=float(np.max(np.abs(original-reference)));replay_max=max(replay_max,error)
        require(np.array_equal(original_pred,np.argmax(reference,axis=1))
                and np.allclose(original,reference,rtol=2e-5,atol=2e-6),
                'Forward NumPy no reproduce referencia Keras con suficiente fidelidad')
        low=np.multiply(np.float32(-128-zp),quant_scale,dtype=np.float32)
        high=np.multiply(np.float32(127-zp),quant_scale,dtype=np.float32)
        limited=np.clip(x,low,high).astype(np.float32)
        dequant=np.multiply(np.subtract(q.astype(np.float32),np.float32(zp),dtype=np.float32),quant_scale,dtype=np.float32)
        limited_prob,_=forward(limited,weights);dequant_prob,_=forward(dequant,weights)
        limited_pred=limited_prob.argmax(axis=1);dequant_pred=dequant_prob.argmax(axis=1)
        n_diagnostic_forward_rows+=3*len(x)
        scenarios=dict(original_float_weights=metrics3(y_val,original_pred),
            continuous_input_range_limited_float_weights=metrics3(y_val,limited_pred),
            input_quantized_dequantized_float_weights=metrics3(y_val,dequant_pred),
            actual_saved_full_INT8=result['metrics']['INT8'])
        forward_scenarios[job_id]=scenarios
        # Separar CE y L2 solo en el checkpoint restaurado; no se infiere la curva completa.
        logits64=logits.astype(np.float64);maximum=logits64.max(axis=1)
        logsum=maximum+np.log(np.exp(logits64-maximum[:,None]).sum(axis=1))
        ce=float(np.mean(logsum-logits64[np.arange(len(y_val)),y_val]))
        l2=float(.001*sum(np.sum(weights[k].astype(np.float64)**2) for k in (0,2)))
        require(abs(ce+l2-state['best_validation_loss'])<2e-5,'CE+L2 no reproduce perdida best')
        changes=fp_pred!=int_pred
        for column,index in enumerate(indices):
            count=int(clip[:,column].sum());global_clips[index]+=count
            feature_rows.append(dict(job_id=job_id,outer=job['outer_load_hp'],inner=job['inner_validation_load_hp'],
                candidate=job['candidate'],feature_index=index,feature=FEATURES[index],
                clipped_values=count,validation_z_min=float(x[:,column].min()),validation_z_max=float(x[:,column].max()),
                representable_z_min=float(low),representable_z_max=float(high)))
        vals=job['validation_global_window_ids'];record_ids=[by_id[k]['file_id'] for k in vals]
        for record in sorted(set(record_ids),key=int):
            selected=np.asarray([v==record for v in record_ids]);label=int(y_val[selected][0])
            record_rows.append(dict(job_id=job_id,file_id=record,class_label=label,windows=int(selected.sum()),
                clipped_windows=int(clip_windows[selected].sum()),FP32_errors=int((fp_pred[selected]!=label).sum()),
                INT8_errors=int((int_pred[selected]!=label).sum()),FP32_INT8_changes=int(changes[selected].sum()),
                FP32_pred_N=int((fp_pred[selected]==0).sum()),FP32_pred_IR=int((fp_pred[selected]==1).sum()),
                FP32_pred_OR=int((fp_pred[selected]==2).sum())))
        for pos,key in enumerate(vals):
            window_rows.append(dict(job_id=job_id,global_window_id=key,file_id=record_ids[pos],label=int(y_val[pos]),
                clipped_input=bool(clip_windows[pos]),saved_FP32_class=int(fp_pred[pos]),saved_INT8_class=int(int_pred[pos]),
                numpy_original_class=int(original_pred[pos]),numpy_limited_class=int(limited_pred[pos]),
                numpy_input_quant_dequant_class=int(dequant_pred[pos])))
        with np.load(directory/'detector_parametros.npz',allow_pickle=False) as saved:
            params={key:saved[key].copy() for key in saved.files}
        params['indices']=params['indices'].tolist();params['threshold32']=np.float32(params['threshold32'].item())
        scores=detector_score(raw_val,params)
        with np.load(directory/'detector_validacion.npz',allow_pickle=False) as saved:
            require(np.array_equal(scores.view(np.uint32),saved['scores'].view(np.uint32)), 'Score detector no exacto')
        positions={key:pos for pos,key in enumerate(job['training_global_window_ids'])}
        fit_ids,cal_ids=job['detector_fit_global_window_ids'],job['detector_calibration_global_window_ids']
        fit=raw[[positions[k] for k in fit_ids]];cal=raw[[positions[k] for k in cal_ids]]
        di=params['indices']
        require(np.allclose(np.mean(fit[:,di],axis=0,dtype=np.float64),params['mean64'],rtol=1e-12,atol=1e-12)
                and np.allclose(np.var(fit[:,di].astype(np.float64),axis=0),params['var64'],rtol=1e-12,atol=1e-12),
                'Scaler normal-only fuera del ajuste')
        zfit=normalize_f32(fit[:,di],params['mean32'],params['scale32'])
        centroid=np.mean(zfit,axis=0,dtype=np.float64).astype(np.float32)
        require(np.array_equal(centroid,params['centroid32']),'Centroide no reproduce')
        cal_scores=detector_score(cal,params)
        threshold=weighted_p95(cal_scores,[by_id[k]['file_id'] for k in cal_ids])
        require(threshold.tobytes()==params['threshold32'].tobytes(),'P95 no reproduce')
        normal=y_val==0
        z=normalize_f32(raw_val[:,di],params['mean32'],params['scale32'])
        residual=np.subtract(z,params['centroid32'],dtype=np.float32)
        terms=np.multiply(residual,residual,dtype=np.float32)
        mean_terms=np.mean(terms[normal],axis=0,dtype=np.float64)
        order=np.argsort(mean_terms)[::-1]
        contributions=[dict(feature_index=di[p],feature=FEATURES[di[p]],
            mean_squared_contribution=float(mean_terms[p]),fraction_of_mean_sum=float(mean_terms[p]/mean_terms.sum()))
            for p in order]
        det_rows.append(dict(job_id=job_id,candidate=job['candidate'],outer=job['outer_load_hp'],
            inner=job['inner_validation_load_hp'],normal_fit_ids=fit_ids,normal_calibration_ids=cal_ids,
            threshold=float(threshold),normal_windows=int(normal.sum()),normal_accepted=int((scores[normal]<=threshold).sum()),
            normal_score_min=float(scores[normal].min()),normal_score_median=float(np.median(scores[normal])),
            normal_score_max=float(scores[normal].max()),normal_median_to_threshold=float(np.median(scores[normal])/threshold),
            calibration_normal_accepted=int((cal_scores<=threshold).sum()),calibration_windows=len(cal_scores),
            feature_contributions=contributions))
        job_rows.append(dict(job_id=job_id,candidate=job['candidate'],outer=job['outer_load_hp'],inner=job['inner_validation_load_hp'],
            validation_rows=len(vals),clipped_values=clipped_count,clipped_windows=int(clip_windows.sum()),
            class_changes=int(changes.sum()),changes_with_clipping=int((changes&clip_windows).sum()),
            changes_without_clipping=int((changes&~clip_windows).sum()),
            numpy_replay_probability_max_error=error,best_crossentropy_without_L2=ce,best_L2=l2,
            numpy_original_vs_saved_FP32_class_changes=int((original_pred!=fp_pred).sum()),
            numpy_original_macro_f1=scenarios['original_float_weights']['macro_f1'],
            numpy_range_limited_macro_f1=scenarios['continuous_input_range_limited_float_weights']['macro_f1'],
            numpy_input_quant_dequant_macro_f1=scenarios['input_quantized_dequantized_float_weights']['macro_f1'],
            saved_INT8_macro_f1=result['metrics']['INT8']['macro_f1']))
        print(job_id+': q exacta; detector exacto; replay NumPy validado',flush=True)
    require(len(global_seen)==1380,'Resumen de registros no cubre datos ya autorizados')
    for record in sorted({row['file_id'] for row in metadata},key=int):
        ids=[k for k,row in by_id.items() if row['file_id']==record]
        values=np.asarray([global_seen[k] for k in ids]);first=by_id[ids[0]]
        if first['class']=='N':
            for i,name in enumerate(FEATURES):
                normal_rows.append(dict(file_id=record,load_hp=int(first['load_hp']),windows=len(ids),
                    feature_index=i,feature=name,minimum=float(values[:,i].min()),
                    median=float(np.median(values[:,i])),maximum=float(values[:,i].max())))
    report=dict(status='DIAGNOSTIC_COMPLETE_WITH_WARNINGS',scope='authorized_inner_validation_and_training_rows',
        campaign=CAMPAIGN.relative_to(ROOT).as_posix(),source_closure_sha256=sha(CLOSURE/'MANIFEST_CIERRE_SHA256.json'),
        quantization_input_replay_exact_jobs=32,normality_score_replay_bit_exact_jobs=32,
        numpy_DNN_reference_class_agreement_jobs=32,numpy_probability_max_error=replay_max,
        new_training_fits=0,new_TFLite_invocations=0,diagnostic_NumPy_forward_rows=n_diagnostic_forward_rows,
        diagnostic_normality_replay_jobs=32,new_quantization_conversions=0,outer_predictive_evaluation=False,
        clipped_values_per_original_feature=global_clips.tolist(),
        clipped_values_total=int(global_clips.sum()),changes_without_input_clipping=sum(j['changes_without_clipping'] for j in job_rows),
        sources_materialized_per_job_with_original_role_guards=True,normal_record_summaries=normal_rows,
        jobs=job_rows,detector=det_rows,forward_scenarios=forward_scenarios,
        limitations=['Diagnostics use validation already inspected; future revisions are development, not independent confirmation.',
            'Only one normal MAT per load; load and record/acquisition effects cannot be separated.',
            'NumPy counterfactuals use unchanged float weights; cannot isolate quantized hidden activations or ARM kernels.',
            'Continuous clipping and quantize/dequantize are diagnostic interventions, not new deployed models.',
            'Fs documented, not measured here; dominant-frequency differences do not prove incorrect Fs.',
            'Counts aggregate overlapping windows and repeated model evaluations, not independent samples.'],
        no_parameters_modified=True,no_models_selected=True,no_data_deleted_or_relabelled=True,
        manuscript_edited=False,full_STM32_cost_pending=True)
    OUTPUT.mkdir(exist_ok=False)
    write(OUTPUT/'DIAGNOSTICO.json',report)
    write_csv(OUTPUT/'RESUMEN_MODELOS.csv',job_rows)
    write_csv(OUTPUT/'RECORTES_POR_CARACTERISTICA.csv',feature_rows)
    write_csv(OUTPUT/'ERRORES_POR_REGISTRO.csv',record_rows)
    write_csv(OUTPUT/'REFERENCIAS_POR_VENTANA.csv',window_rows)
    write_csv(OUTPUT/'NORMALES_POR_REGISTRO.csv',normal_rows)
    write(OUTPUT/'FUENTES_PRIMARIAS.json',[
        dict(url='https://developers.google.com/edge/litert/conversion/tensorflow/quantization/post_training_integer_quant',
             supports='Representative data estimate dynamic ranges; quantization includes weights and activations, not only inputs.'),
        dict(url='https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html',
             supports='Train statistics reused to transform new data; sensitive to outliers; zero variance scaling policy.'),
        dict(url='https://engineering.case.edu/bearingdatacenter/apparatus-and-procedures',
             supports='DE and FE sensor positions, acquisition procedure and documented12k/48k configurations.')])
    # Captura final: ninguna entrada/material sellado debe cambiar durante el diagnostico.
    verify_manifest(CLOSURE/'MANIFEST_CIERRE_SHA256.json')
    for relative in {source['path'] for job in plan['jobs'] for source in read(CAMPAIGN/job['job_id']/'procedencia.json')['source_arrays']}:
        require(sha(ROOT/relative)==input_hashes[relative],'Fuente modificada durante diagnostico')
        entries[relative]=input_hashes[relative]
    paths=[Path(__file__),FEATURE_FILE,CLOSURE/'MANIFEST_CIERRE_SHA256.json',CLOSURE/'MANIFEST_CIERRE_SHA256.sha256']+list(OUTPUT.iterdir())
    entries.update({p.relative_to(ROOT).as_posix():sha(p) for p in paths})
    manifest=OUTPUT/'MANIFEST_DIAGNOSTICO_SHA256.json';write(manifest,entries)
    manifest.with_suffix('.sha256').write_text(sha(manifest)+'  '+manifest.name+'\n',encoding='ascii')
    verify_manifest(manifest)
    print(f'PASS:32 q y detector; {n_diagnostic_forward_rows} filasforward NumPy,0fits; sello{len(entries)}recursos.')


if __name__=='__main__':main()
