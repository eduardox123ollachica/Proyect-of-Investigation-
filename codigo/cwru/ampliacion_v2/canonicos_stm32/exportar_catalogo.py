"""Exportar ocho modelos canónicos fijados, sin banco RAW o ejecución ARM.

Importar no escribe, ajusta o infiere. Conserva el ABI numérico de 297 bytes;
el índice del catálogo y mask_id son identidades distintas.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import struct
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[4]
BASE=ROOT/'codigo/cwru/ampliacion_v2'
for directory in ('costo_total','busqueda_interna'):
    sys.path.insert(0,str(BASE/directory))
from exportar_recursos_total import serialize_params,bits32,cfloat,fnv1a,padded,CANONICAL_KEYS,ORDER
import ejecutar_busqueda_interna as io

OUT=ROOT/'despliegue_stm32/canonicos_42_180mhz_2026-10-27/recursos'
CLOSED=ROOT/'articulo/ampliacion_cwru/metodologia/cierre_evaluacion_exterior_v2_2026-10-27'
REFIT=ROOT/'articulo/ampliacion_cwru/metodologia/cierre_reajustes_finales_v2_2026-10-27'
REFIT_PLAN=ROOT/'articulo/ampliacion_cwru/metodologia/reajustes_finales_v2_2026-10-27/PLAN_REAJUSTES.json'
CLOSED_SHA256='35c80f337c4856b55d3edcbadbff475ac30f0520044da1fa0e25a5d0cc7459f5'
REFIT_SHA256='5d90a9dc4e91b6d9295acd593878e31bf48c5b9e6117cb910ed52a618986bc8b'
SELECTION_SIGNATURE='c577339ac0c9450f99d40044850063e7090b19d5e81502eca0378c46e1b2c32f'
MODEL_NAMES=('outer0_F14_e45_s42','outer0_T6_e47_s42','outer1_F14_e14_s42','outer1_FE7_e27_s42',
    'outer2_F14_e19_s42','outer2_T6_e32_s42','outer3_F14_e9_s42','outer3_NF12_e47_s42')
MASK_IDS=(0,1,0,7,0,1,0,3)
SCALER_KEYS={'mean64','scale64','var64','mean32','scale32','feature_indices'}
DETECTOR_KEYS={'indices','mean64','scale64','var64','mean32','scale32','centroid32','threshold32','fit_ids','calibration_ids'}


def require(ok,message):
    if not ok:raise ValueError(message)


def bounded(path):
    path=Path(path);target=(ROOT/path).resolve() if not path.is_absolute() else path.resolve()
    require(target!=ROOT.resolve() and target.is_relative_to(ROOT.resolve()),'Recurso fuera del proyecto')
    return target


def _checked(path,sources,expected=None):
    path=bounded(path);digest=io.sha(path)
    require(expected is None or digest==expected,'Entrada modificada: '+str(path))
    sources[path.relative_to(ROOT).as_posix()]=digest
    return path


def _read(path,sources,expected=None):return io.read(_checked(path,sources,expected))


def _manifest(path,expected,sources):
    path=_checked(path,sources,expected)
    _checked(path.with_suffix('.sha256'),sources)
    io.verify_manifest(path)


def validate_metadata(canonical,mapping,inventory,refit_plan):
    require(canonical['status']=='PENDING_SEPARATE_HARDWARE_VALIDATION'
        and canonical['unique_models']==8 and canonical['seed']==42
        and canonical['seed_fixed_before_outer_evaluation'] is True
        and canonical['hardware_execution_enabled'] is False,'Canonización no fijada antes de evaluar')
    models=canonical['models'];by_id={m['job_id']:m for m in inventory};trained={j['job_id']:j for j in refit_plan['jobs']}
    require(len(models)==8 and len(inventory)==len(by_id)==131
        and len(refit_plan['jobs'])==len(trained)==131,'Inventario duplicado o incompleto')
    require([fold['outer_load_hp'] for fold in mapping['outer_scenarios']]==[0,1,2,3]
        and mapping['selection_signature']==SELECTION_SIGNATURE,'Mapa de cuatro decisiones distinto')
    derived=[];detectors={};procedures={}
    for fold in mapping['outer_scenarios']:
        require(type(fold['outer_load_hp']) is int,'Carga no entera')
        for name in ('baseline_F14','performance_only','validated_total_cost'):
            choice=fold['DNN_procedures'][name];identifier=choice['canonical_hardware_job_id']
            require(identifier==choice['DNN_by_seed']['42'],'Modelo no corresponde a semilla canónica')
            if identifier not in derived:derived.append(identifier)
            if identifier in detectors:require(detectors[identifier]==choice['detector_job_id'],'Alias con detectores distintos')
            detectors[identifier]=choice['detector_job_id'];procedures.setdefault(identifier,[]).append(name)
    require(tuple(derived)==MODEL_NAMES and tuple(m['model_job_id'] for m in models)==MODEL_NAMES,
        'Los ocho modelos no son los fijados; no seleccionar por métricas exteriores')
    for model in models:
        original=by_id[model['model_job_id']];job=trained[model['model_job_id']]
        require(type(model['seed']) is int and model['seed']==42
            and model['seed_selected_from_outer_performance'] is False
            and model['pending_hardware_validation'] is True,'Semilla elegida o hardware ya declarado validado')
        for key in ('outer_load_hp','candidate','seed','epochs','indices','paths','artifact_sha256','fit_signature'):
            require(io.canonical(model[key])==io.canonical(original[key]),'Canon distinto del inventario: '+key)
        require(type(model['outer_load_hp']) is int and original['family']=='DNN'
            and job['seed']==42 and job['candidate']==model['candidate'] and job['epochs']==model['epochs'],
            'Identidad del reajuste distinta')
        require(model['procedures']==procedures[model['model_job_id']],'Alias de procedimiento distinto')
        detector=by_id[detectors[model['model_job_id']]];djob=trained[detector['job_id']]
        require(detector['family']=='detector' and detector['outer_load_hp']==model['outer_load_hp']
            and model['candidate'] in detector['masks'] and djob['refit_after_calibration'] is False,
            'Detector no corresponde al modelo')
        require(detector['indices']==[i for i in model['indices'] if i not in (4,11)],'Índices detector distintos')
    return by_id,trained,detectors,procedures


def load_context():
    sources={}
    _manifest(CLOSED/'MANIFEST_CIERRE_EVALUACION_EXTERIOR_SHA256.json',CLOSED_SHA256,sources)
    _manifest(REFIT/'MANIFEST_CIERRE_REAJUSTES_FINALES_SHA256.json',REFIT_SHA256,sources)
    closure=_read(CLOSED/'CIERRE_EVALUACION_EXTERIOR.json',sources)
    require(closure['status']=='OUTER_EVALUATION_COMPLETE_AUDITED_PENDING_FINAL_HARDWARE_AND_REPORTING'
        and closure['loads_completed']==4 and closure['failures']==closure['retries']==0
        and closure['selection_signature']==SELECTION_SIGNATURE,'Cierre exterior no auditado')
    canonical=_read(CLOSED/'MODELOS_CANONICOS_42_PENDIENTES_STM32.json',sources)
    mapping=_read(REFIT/'MAPA_PARA_EVALUACION_EXTERIOR.json',sources)
    inventory=_read(REFIT/'INVENTARIO_MODELOS_FINALES_131.json',sources)
    refit_plan=_read(REFIT_PLAN,sources)
    validate_metadata(canonical,mapping,inventory,refit_plan)
    return dict(canonical=canonical,mapping=mapping,inventory=inventory,refit_plan=refit_plan,sources=sources)


def _statistics(saved,dimension):
    for key in ('mean64','scale64','var64','mean32','scale32'):
        value=saved[key];dtype=np.float64 if key.endswith('64') else np.float32
        require(value.dtype==dtype and value.shape==(dimension,) and np.isfinite(value).all(),
            'Estadística inválida: '+key)
    require((saved['scale64']>0).all() and (saved['scale32']>0).all() and (saved['var64']>=0).all(),
        'Escala/varianza inválida')
    require(np.array_equal(saved['mean32'].view(np.uint32),saved['mean64'].astype(np.float32).view(np.uint32))
        and np.array_equal(saved['scale32'].view(np.uint32),saved['scale64'].astype(np.float32).view(np.uint32)),
        'Exportación float32 de estadísticas distinta')


def _arrays(path,keys,sources,expected):
    with np.load(_checked(path,sources,expected),allow_pickle=False) as saved:
        require(set(saved.files)==keys,'NPZ incompleto o con campos ajenos')
        return {key:saved[key].copy() for key in saved.files}


def canonical_bytes(rows):
    return json.dumps([{key:row[key] for key in CANONICAL_KEYS} for row in rows],ensure_ascii=False,
        sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')


def build_catalog(context=None):
    context=load_context() if context is None else context
    canonical,mapping,inventory,refit_plan=[context[k] for k in ('canonical','mapping','inventory','refit_plan')]
    by_id,trained,detectors,procedures=validate_metadata(canonical,mapping,inventory,refit_plan)
    sources=dict(context['sources']);rows=[];models=[];c_arrays=[]
    for slot,model in enumerate(canonical['models']):
        identifier=model['model_job_id'];record=by_id[identifier];job=trained[identifier]
        result=_read(record['result_path'],sources,record['result_sha256'])
        require(result['status']=='PASS' and result['fit_signature']==record['fit_signature']
            and result['seed']==42 and result['epochs_executed']==model['epochs']
            and result['roles']==job['roles'] and result['early_stopping_used'] is False
            and result['validation_used'] is False and result['weights_restored'] is False,'Reajuste DNN distinto')
        source_model=_checked(model['paths']['modelo_int8.tflite'],sources,model['artifact_sha256']['modelo_int8.tflite'])
        blob=source_model.read_bytes()
        require(len(blob)==result['model_bytes']['INT8'] and io.sha(source_model)==result['model_sha256']['INT8'],
            'Modelo INT8 modificado o de tamaño distinto')
        details=result['int8_details']
        require(details['input_dtype']==details['output_dtype']=='int8'
            and set(details['tensor_dtypes'])<={'int8','int32','int64'}
            and details['operators']==['FULLY_CONNECTED']*3+['SOFTMAX'],'Modelo no es full INT8 fijado')
        indices=model['indices'];d=len(indices);mask_id=ORDER.index(model['candidate'])
        require(mask_id==MASK_IDS[slot],'Máscara confundida con índice del catálogo')
        scaler=_arrays(model['paths']['scaler_train_only.npz'],SCALER_KEYS,sources,model['artifact_sha256']['scaler_train_only.npz'])
        _statistics(scaler,d)
        require(scaler['feature_indices'].dtype==np.int64 and scaler['feature_indices'].tolist()==indices,
            'Scaler corresponde a otras columnas')
        detector=by_id[detectors[identifier]];detjob=trained[detector['job_id']]
        dresult=_read(detector['result_path'],sources,detector['result_sha256'])
        params=_arrays(detector['paths']['parametros.npz'],DETECTOR_KEYS,sources,detector['artifact_sha256']['parametros.npz'])
        di=params['indices'].tolist();nd=len(di);_statistics(params,nd)
        require(params['indices'].dtype==np.int64 and di==detector['indices']
            and params['fit_ids'].dtype==params['calibration_ids'].dtype==np.int64
            and params['fit_ids'].tolist()==detjob['fit_ids'] and params['calibration_ids'].tolist()==detjob['calibration_ids']
            and not set(detjob['fit_ids'])&set(detjob['calibration_ids']),'Ámbitos del detector distintos')
        require(params['centroid32'].dtype==np.float32 and params['centroid32'].shape==(nd,)
            and np.isfinite(params['centroid32']).all() and params['threshold32'].dtype==np.float32
            and params['threshold32'].shape==() and np.isfinite(params['threshold32']) and params['threshold32']>=0,
            'Centroide/umbral inválido')
        require(dresult['status']=='PASS' and dresult['fit_signature']==detector['fit_signature']
            and dresult['normal_only_fit'] is True and dresult['normal_only_calibration'] is True
            and dresult['refit_after_calibration'] is False and dresult['class_override'] is False,'Detector no congelado')
        threshold=np.float32(params['threshold32'].item());qscale,qzero=details['input_quantization']
        require(details['output_quantization']==[1/256,-128],'Cuantización de salida incompatible con firmware')
        packed=serialize_params(d,nd,mask_id,indices,di,scaler['mean32'],scaler['scale32'],
            params['mean32'],params['scale32'],params['centroid32'],threshold,qscale,qzero)
        scope=dict(outer_load_hp=model['outer_load_hp'],inner_validation_load_hp=None,seed=42,epochs=model['epochs'],
            training_loads=job['roles']['training_loads'],training_global_window_ids=job['training_global_window_ids'],
            procedures=procedures[identifier],selection_signature=SELECTION_SIGNATURE,
            seed_fixed_before_outer_evaluation=True,seed_selected_from_outer_performance=False,
            detector_job_id=detector['job_id'],detector_fit_signature=detector['fit_signature'],
            detector_parameter_payload_sha256=dresult['parameter_payload_sha256'],
            detector_parameters_sha256=detector['artifact_sha256']['parametros.npz'],
            detector_fit_global_window_ids=detjob['fit_ids'],detector_calibration_global_window_ids=detjob['calibration_ids'],
            detector_fit_normal_loads=detjob['roles']['detector_fit_normal_loads'],
            detector_calibration_normal_loads=detjob['roles']['detector_calibration_normal_loads'],
            detector_refit_after_calibration=False,detector_class_override=False,RAW_bank_selected=False)
        row=dict(id=slot,representative_id=identifier,candidate=model['candidate'],outer_load_hp=model['outer_load_hp'],
            seed=42,epochs=model['epochs'],inner_validation_load_hp=None,mask_id=mask_id,dims=d,det_dims=nd,
            feature_mask=f'{sum(1<<i for i in indices):04x}',model_path='models/'+identifier+'.tflite',model_bytes=len(blob),
            model_sha256=io.sha(source_model),source_fit_signature=record['fit_signature'],fit_signature=record['fit_signature'],
            model_fnv=f'{fnv1a(blob):08x}',params_fnv=f'{fnv1a(packed):08x}',params_bytes_hex=packed.hex(),
            feature_indices=indices,detector_indices=di,dnn_mean32_bits=bits32(scaler['mean32']),dnn_scale32_bits=bits32(scaler['scale32']),
            detector_mean32_bits=bits32(params['mean32']),detector_scale32_bits=bits32(params['scale32']),centroid32_bits=bits32(params['centroid32']),
            threshold_bits=bits32([threshold])[0],input_quantization=dict(scale_bits=bits32([qscale])[0],zero_point=int(qzero)),
            output_quantization=dict(scale_bits=bits32([details['output_quantization'][0]])[0],zero_point=-128),
            detector_job_id=detector['job_id'],c_model_symbol='cwru_c42_'+identifier,scope=scope)
        arrays=dict(indices=padded(indices,14,0,np.uint8),det_indices=padded(di,12,0,np.uint8),
            mean=padded(scaler['mean32'],14),scale=padded(scaler['scale32'],14,1.),
            det_mean=padded(params['mean32'],12),det_scale=padded(params['scale32'],12,1.),
            centroid=padded(params['centroid32'],12),threshold=threshold,qscale=np.float32(qscale),qzero=int(qzero))
        rows.append(row);models.append(blob);c_arrays.append(arrays)
    canonical_blob=canonical_bytes(rows)
    catalog=dict(schema_version=2,status='CANONICAL_42_CATALOG_EXPORTED_PENDING_HARDWARE_VALIDATION',count=8,
        seed=42,representatives=rows,catalog_sha256=hashlib.sha256(canonical_blob).hexdigest(),
        catalog_fnv=f'{fnv1a(b"".join(struct.pack("<II",int(r["model_fnv"],16),int(r["params_fnv"],16)) for r in rows)):08x}',
        canonical_catalog_path='CATALOGO_CANONICO.json',canonical_keys=list(CANONICAL_KEYS),
        canonical_json='UTF8 array, sorted keys, compact separators, no trailing newline',
        catalog_fnv_serialization='8 pairs model_fnv_u32le,params_fnv_u32le in catalog ID order',
        params_serialization=dict(bytes=297,endianness='little',padding='indices/means/centroid0, scales1',
            floats='mean14,scale14,det_mean12,det_scale12,centroid12,threshold,qin_scale',tail='qin_zero int32'),
        validation_policy=dict(same_input_INT8_output_required_exact=True,
            downstream_reference='PC replay using published ARM feature bits and matching frozen model/parameters',
            cached_PC_comparison_reported_separately=True,cached_PC_end_to_end_bit_exact_claimed=False,
            RAW_bank_exported=False,hardware_execution_enabled=False,physical_validation_completed=False,
            cost_validated=False,training_fits=0,TFLite_invocations=0,seed_fixed_before_outer_evaluation=True),
        source_outer_closure_manifest_sha256=CLOSED_SHA256,source_final_refit_manifest_sha256=REFIT_SHA256)
    for p in (Path(__file__),BASE/'costo_total/exportar_recursos_total.py'):_checked(p,sources)
    return catalog,models,c_arrays,sources


def render_header():
    return '''#ifndef CWRU_TOTAL_CATALOG_H_
#define CWRU_TOTAL_CATALOG_H_
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
#define CWRUTV_COUNT 8U
typedef struct {
 const char *id;
 const unsigned char *model;
 unsigned int model_size;
 uint8_t mask_id,dims,det_dims;
 uint8_t indices[14],det_indices[12];
 float mean[14],scale[14],det_mean[12],det_scale[12],centroid[12];
 float threshold,qin_scale;
 int32_t qin_zero;
 uint32_t model_fnv,params_fnv;
 const char *model_sha256,*source_fit_signature;
} CWRUTV_Descriptor;
extern const CWRUTV_Descriptor cwru_total_catalog[CWRUTV_COUNT];
extern const char cwru_total_catalog_sha256[];
extern const uint32_t cwru_total_catalog_fnv;
#ifdef __cplusplus
}
#endif
#endif
'''


def render_cpp(catalog,models,c_arrays):
    lines=['#include "cwru_total_catalog.h"','']
    for row,blob in zip(catalog['representatives'],models):
        lines.append(f'alignas(16) static const unsigned char {row["c_model_symbol"]}[] = {{')
        for offset in range(0,len(blob),16):lines.append(' '+','.join(f'0x{v:02x}' for v in blob[offset:offset+16])+',')
        lines.extend(['};',''])
    lines.append('const CWRUTV_Descriptor cwru_total_catalog[CWRUTV_COUNT] = {')
    for row,a in zip(catalog['representatives'],c_arrays):
        lines.extend([' {',f'  "{row["representative_id"]}",{row["c_model_symbol"]},{row["model_bytes"]}U,',
            f'  {row["mask_id"]}U,{row["dims"]}U,{row["det_dims"]}U,'])
        for key in ('indices','det_indices'):lines.append('  {'+','.join(str(int(v))+'U' for v in a[key])+'},')
        for key in ('mean','scale','det_mean','det_scale','centroid'):lines.append('  {'+','.join(cfloat(v) for v in a[key])+'},')
        lines.extend([f'  {cfloat(a["threshold"])},{cfloat(a["qscale"])},{a["qzero"]},',
            f'  0x{row["model_fnv"]}U,0x{row["params_fnv"]}U,"{row["model_sha256"]}","{row["source_fit_signature"]}"',' },'])
    lines.extend(['};',f'const char cwru_total_catalog_sha256[] = "{catalog["catalog_sha256"]}";',
        f'const uint32_t cwru_total_catalog_fnv = 0x{catalog["catalog_fnv"]}U;',''])
    return '\n'.join(lines)


def verify_resources(output):
    """Autenticar el paquete local de ocho modelos, sin dependencias del banco RAW."""
    output=bounded(output);manifest=output/'MANIFEST_RECURSOS_SHA256.json'
    digest,name=manifest.with_suffix('.sha256').read_text(encoding='ascii').split()
    require(name==manifest.name and digest==io.sha(manifest),'Sello local modificado')
    entries=io.read(manifest)
    actual={p.relative_to(output).as_posix() for p in output.rglob('*') if p.is_file()
        and p not in (manifest,manifest.with_suffix('.sha256'))}
    require(set(entries)==actual,'Recursos añadidos o ausentes')
    for relative,expected in entries.items():
        path=(output/relative).resolve()
        require(path.is_relative_to(output.resolve()) and io.sha(path)==expected,'Recurso local modificado')
    catalog=io.read(output/'CATALOGO_TOTAL.json');rows=catalog['representatives']
    require(catalog['count']==len(rows)==8 and [r['id'] for r in rows]==list(range(8))
        and tuple(r['mask_id'] for r in rows)==MASK_IDS
        and tuple(r['representative_id'] for r in rows)==MODEL_NAMES
        and all(type(r['seed']) is int and r['seed']==42 for r in rows),'Catálogo canónico alterado')
    require((output/'CATALOGO_CANONICO.json').read_bytes()==canonical_bytes(rows)
        and io.sha(output/'CATALOGO_CANONICO.json')==catalog['catalog_sha256'],'Identidad canónica alterada')
    fnv_blob=b''
    for row in rows:
        path=(output/row['model_path']).resolve()
        require(path.is_relative_to(output.resolve()) and io.sha(path)==row['model_sha256']
            and path.stat().st_size==row['model_bytes'] and fnv1a(path.read_bytes())==int(row['model_fnv'],16),
            'Modelo local distinto')
        def from_bits(values):
            require(all(isinstance(v,str) and len(v)==8 for v in values),'Formato de bits no uint32hex')
            return np.asarray([int(v,16) for v in values],dtype='<u4').view('<f4')
        packed=serialize_params(row['dims'],row['det_dims'],row['mask_id'],row['feature_indices'],row['detector_indices'],
            from_bits(row['dnn_mean32_bits']),from_bits(row['dnn_scale32_bits']),from_bits(row['detector_mean32_bits']),
            from_bits(row['detector_scale32_bits']),from_bits(row['centroid32_bits']),
            from_bits([row['threshold_bits']])[0],from_bits([row['input_quantization']['scale_bits']])[0],
            row['input_quantization']['zero_point'])
        require(packed.hex()==row['params_bytes_hex'] and f'{fnv1a(packed):08x}'==row['params_fnv'],'Parámetros locales distintos')
        fnv_blob+=struct.pack('<II',int(row['model_fnv'],16),int(row['params_fnv'],16))
    require(f'{fnv1a(fnv_blob):08x}'==catalog['catalog_fnv'],'FNV de catálogo distinto')
    return catalog


def export_resources(output=OUT):
    output=bounded(output);require(not output.exists(),'No sobrescribir recursos existentes')
    catalog,models,arrays,sources=build_catalog()
    for path,digest in sources.items():require(io.sha(ROOT/path)==digest,'Fuente cambió durante exportación')
    output.mkdir(parents=True,exist_ok=False);(output/'models').mkdir()
    for row,blob in zip(catalog['representatives'],models):
        with (output/row['model_path']).open('xb') as stream:stream.write(blob)
        require(io.sha(output/row['model_path'])==row['model_sha256'],'Modelo copiado distinto')
    with (output/'CATALOGO_CANONICO.json').open('xb') as stream:stream.write(canonical_bytes(catalog['representatives']))
    io.write(output/'CATALOGO_TOTAL.json',catalog)
    io.write(output/'FUENTES_SHA256.json',dict(sorted(sources.items())))
    for name,text in [('cwru_total_catalog.h',render_header()),('cwru_total_catalog.cpp',render_cpp(catalog,models,arrays))]:
        with (output/name).open('x',encoding='utf-8',newline='\n') as stream:stream.write(text)
    entries={p.relative_to(output).as_posix():io.sha(p) for p in sorted(output.rglob('*')) if p.is_file()}
    manifest=output/'MANIFEST_RECURSOS_SHA256.json';io.write(manifest,entries)
    with manifest.with_suffix('.sha256').open('x',encoding='ascii') as stream:
        stream.write(io.sha(manifest)+'  '+manifest.name+'\n')
    require(io.sha(output/'CATALOGO_CANONICO.json')==catalog['catalog_sha256'],'Catálogo canónico distinto tras guardar')
    for path,digest in entries.items():require(io.sha(output/path)==digest,'Recurso exportado modificado')
    verify_resources(output)
    return dict(status=catalog['status'],count=8,models_copied=8,RAW_windows_exported=0,resources=len(entries),
        catalog_sha256=catalog['catalog_sha256'],catalog_fnv=catalog['catalog_fnv'],output=str(output),
        training_fits=0,TFLite_invocations=0,serial_opened=False,hardware_flashed=False)


if __name__=='__main__':
    print(json.dumps(export_resources(),ensure_ascii=True))
