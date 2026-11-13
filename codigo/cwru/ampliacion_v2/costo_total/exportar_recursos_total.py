"""Exporta los32 representantes P0 y sus bancos, sin entrenar o inferir.

Importar el modulo no escribe. ``export_resources`` crea exclusivamente una
carpeta nueva; nunca modifica resultados, fuentes o sellos historicos.
Los clientes verifican catalogo, modelos, parametros y payloads por separado.
"""
from __future__ import annotations

import hashlib
import json
from numbers import Integral
from pathlib import Path
import struct
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT/'despliegue_stm32/costo_total_representantes_v2_180mhz_2026-10-25/recursos'
CAMPAIGN = ROOT/'resultados/cwru/ampliacion_v2/representantes_v2_2026-10-25/20261025T171847_957295Z'
REP_PREP = ROOT/'articulo/ampliacion_cwru/metodologia/representantes_v2_2026-10-25'
CLOSURE = ROOT/'articulo/ampliacion_cwru/metodologia/cierre_representantes_v2_2026-10-25'
PHYSICAL = ROOT/'despliegue_stm32/integracion_pipeline_180mhz_2026-10-23'
PARAM_BYTES = 297
CANONICAL_KEYS = ('id','representative_id','model_sha256','source_fit_signature','params_bytes_hex','scope')
ORDER = ('F14','T6','S8','NF12','NM10','NE12','DE7','FE7')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fnv1a(data):
    value = 2166136261
    for byte in data:
        value = ((value ^ byte) * 16777619) & 0xffffffff
    return value


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def bits32(values):
    return [f'{int(v):08x}' for v in np.asarray(values,dtype='<f4').view('<u4').ravel()]


def cfloat(value):
    value = np.float32(value)
    require(np.isfinite(value), 'Constante C no finita')
    return float(value).hex()+'f'


def padded(values, length, fill=0., dtype=np.float32):
    array = np.asarray(values, dtype=dtype)
    require(array.ndim == 1 and len(array) <= length, 'Vector demasiado largo')
    result = np.full(length, fill, dtype=dtype)
    result[:len(array)] = array
    return result


def serialize_params(dims, det_dims, mask_id, indices, det_indices, mean, scale,
                     det_mean, det_scale, centroid, threshold, qin_scale, qin_zero):
    """297 bytes, little-endian, sin padding ABI de struct C.

    u8(d,nd,mask), indices14,det_indices12; mean14,scale14,det_mean12,
    det_scale12,centroid12,threshold,qscale float32; zero_point int32.
    Valores omitidos: indices/means/centroid0, escalas1.
    """
    require(type(dims) is int and 1 <= dims <= 14 and type(det_dims) is int and 1 <= det_dims <= 12
            and type(mask_id) is int and 0 <= mask_id < 8, 'Dimensiones/mascara invalidas')
    require(len(indices) == len(mean) == len(scale) == dims
            and len(det_indices) == len(det_mean) == len(det_scale) == len(centroid) == det_dims,
            'Longitudes de parametros incompatibles')
    require(len(set(indices)) == dims and len(set(det_indices)) == det_dims
            and all(isinstance(v,Integral) and not isinstance(v,(bool,np.bool_)) and 0 <= int(v) < 14
                    for v in list(indices)+list(det_indices)),
            'Indices invalidos')
    vectors = [padded(mean,14),padded(scale,14,1.),padded(det_mean,12),padded(det_scale,12,1.),padded(centroid,12)]
    require(all(np.isfinite(a).all() for a in vectors) and (vectors[1]>0).all() and (vectors[3]>0).all(),
            'Parametro no finito o escala no positiva')
    tail = np.asarray([threshold,qin_scale],dtype='<f4')
    require(np.isfinite(tail).all() and tail[0]>=0 and tail[1]>0
            and isinstance(qin_zero,Integral) and not isinstance(qin_zero,(bool,np.bool_)) and -128 <= qin_zero <= 127,
            'Umbral/cuantizacion invalidos')
    result = struct.pack('<BBB',dims,det_dims,mask_id)
    result += padded(indices,14,0,np.uint8).tobytes()+padded(det_indices,12,0,np.uint8).tobytes()
    result += b''.join(np.asarray(a,dtype='<f4').tobytes() for a in vectors)
    result += tail.tobytes()+struct.pack('<i',int(qin_zero))
    require(len(result) == PARAM_BYTES, 'Serializacion de parametros distinta de297bytes')
    return result


def export_resources():
    require(not OUT.exists(), 'No sobrescribir recursos existentes')
    sys.path.insert(0,str(ROOT/'codigo/cwru/ampliacion_v2/representantes'))
    from datos_representantes import load_verified_context, build_jobs
    from numerica_representantes import normalize_f32, quantize_int8
    sources = {}

    def checked(path, expected=None):
        path = Path(path).resolve()
        require(path.is_relative_to(ROOT) and path != ROOT and path.is_file(), 'Entrada fuera del proyecto')
        digest = sha(path)
        require(expected is None or digest == expected, 'Entrada modificada: '+str(path))
        sources[path.relative_to(ROOT).as_posix()] = digest
        return path

    def checked_read(path, expected=None):
        return read(checked(path,expected))

    def verified_manifest(path):
        path = checked(path)
        digest,name = checked(path.with_suffix('.sha256')).read_text(encoding='ascii').split()
        require(name == path.name and digest == sha(path), 'Sello invalido: '+str(path))
        entries = read(path)
        require(isinstance(entries,dict) and entries, 'Manifiesto vacio')
        for relative,expected in entries.items():
            target = (ROOT/relative).resolve()
            checked(target,expected)
        return entries

    verified_manifest(CLOSURE/'MANIFEST_CIERRE_SHA256.json')
    verified_manifest(REP_PREP/'MANIFEST_PREPARACION_SHA256.json')
    plan = checked_read(REP_PREP/'PLAN_EJECUCION.json')
    protocol,cost_plan,metadata,input_hashes = load_verified_context()
    jobs = build_jobs(protocol,cost_plan,metadata)
    require(plan['jobs'] == jobs and len(jobs) == 32, 'Plan original de32 representantes distinto')
    index = {int(row['global_window_id']):row for row in metadata}
    scoped_banks = [jobs[outer*8]['benchmark_global_window_ids'] for outer in range(4)]
    global_ids = sorted(set(key for ids in scoped_banks for key in ids))
    require(len(global_ids) == 42 and len(set.union(*(set(ids) for ids in scoped_banks))) == 42,
            'Se esperan42 payloads unicos, no112')
    require(all(len(ids) == len(set(ids)) == 28 for ids in scoped_banks), 'Banco porouter incompleto')
    raw_arrays,feature_arrays = {},{}
    for key,target_name in (('source_windows','raw'),('source_features','features')):
        arrays = raw_arrays if target_name == 'raw' else feature_arrays
        for relative in sorted({index[gid][key] for gid in global_ids}):
            require(relative in input_hashes, 'Fuente sin hash autorizado')
            array = np.load(checked(ROOT/relative,input_hashes[relative]),allow_pickle=False,mmap_mode='r')
            count = sum(row[key] == relative for row in metadata)
            if target_name == 'raw':
                require(array.dtype == np.float32 and array.shape == (count,4096,2), 'RAW shape/dtype distinto')
            else:
                require(array.dtype == np.float64 and array.shape == (count,14), 'Features shape/dtype distinto')
            arrays[relative] = array

    physical_entries = verified_manifest(PHYSICAL/'MANIFEST_RESULTADOS_FISICOS_PIPELINE_COMPLETA_SHA256.json')
    physical_report = checked_read(PHYSICAL/'RESULTADOS_FISICOS_PIPELINE_COMPLETA.json')
    require(physical_report['status'] == 'PHYSICAL_PIPELINE_SUITE_AUDITED'
            and physical_report['completed_executions'] == 126
            and physical_report['all_intermediates_and_outputs_bit_exact'], 'Referencia fisica incompleta')
    physical_ref_path = PHYSICAL/'Validacion_PC/REFERENCIA_PIPELINE.json'
    require(physical_ref_path.relative_to(ROOT).as_posix() in physical_entries,
            'Referencia ARM no sellada en cierre fisico')
    physical_ref = checked_read(physical_ref_path)
    old_cases = {int(c['metadata']['global_window_id']):c for c in physical_ref['cases']}
    windows = [];window_rows = [];payload_hashes = set();position = {}
    for pos,gid in enumerate(global_ids):
        row = index[gid]
        window = np.ascontiguousarray(raw_arrays[row['source_windows']][int(row['source_row'])],dtype='<f4')
        original_features = np.asarray(feature_arrays[row['source_features']][int(row['source_row'])],dtype=np.float32)
        require(window.shape == (4096,2) and original_features.shape == (14,)
                and np.isfinite(window).all() and np.isfinite(original_features).all(), 'Fila no finita/incompleta')
        payload = window.tobytes(order='C');payload_sha = hashlib.sha256(payload).hexdigest()
        require(payload_sha not in payload_hashes, 'Payload repetido con IDs distintos')
        payload_hashes.add(payload_sha);position[gid]=pos;windows.append(window)
        old = old_cases.get(gid)
        arm_bits = None
        if old is not None:
            require(old['payload_sha256'] == payload_sha and old['checksum_fnv1a'] == f'{fnv1a(payload):08x}',
                    'Referencia ARM corresponde a otro payload')
            arm_bits = old['features_bits']
            require(len(arm_bits) == 14, 'Referencia ARM sin14caracteristicas')
        authorized = [rep_id for rep_id,job in enumerate(jobs) if gid in job['benchmark_global_window_ids']]
        window_rows.append(dict(global_id=gid,npz_index=pos,payload_sha256=payload_sha,
            checksum_fnv1a=f'{fnv1a(payload):08x}',features_pc_cache14=original_features.astype(float).tolist(),
            features_pc_cache14_bits=bits32(original_features),arm_features14_bits=arm_bits,
            authorized_rep_ids=authorized,metadata=row))
    require(sum(r['arm_features14_bits'] is not None for r in window_rows) == 28,
            'Se esperaban28 referencias ARM y14 ventanas nuevas de3HP')

    descriptors=[];c_arrays=[];models=[];references={}
    for rep_id,job in enumerate(jobs):
        directory = CAMPAIGN/job['job_id']
        result = checked_read(directory/'resultado.json')
        require(result['status']=='PASS' and result['fit_signature']==plan['fit_signatures'][job['job_id']]
                and result['roles']==job['roles'] and result['seed']==11, 'Ajuste no autenticado')
        model_path = checked(directory/'modelo_int8.tflite',result['model_sha256']['INT8'])
        model = model_path.read_bytes()
        require(len(model)==result['model_bytes']['INT8'], 'Tamano del modelo distinto')
        with np.load(checked(directory/'scaler_train_only.npz'),allow_pickle=False) as saved:
            mean,scale = saved['mean32'].copy(),saved['scale32'].copy()
        with np.load(checked(directory/'detector_parametros.npz'),allow_pickle=False) as saved:
            di = saved['indices'].astype(int).tolist()
            det_mean,det_scale,centroid = (saved[k].copy() for k in ('mean32','scale32','centroid32'))
            threshold = np.float32(saved['threshold32'].item())
            require(saved['fit_ids'].tolist()==job['detector_fit_global_window_ids']
                    and saved['calibration_ids'].tolist()==job['detector_calibration_global_window_ids'],
                    'Roles del detector distintos')
        indices=job['feature_indices'];dims=len(indices);nd=len(di);mask_id=ORDER.index(job['candidate'])
        require(di==[i for i in indices if i not in (4,11)] and (mean.shape==scale.shape==(dims,))
                and det_mean.shape==det_scale.shape==centroid.shape==(nd,), 'Indices/dimensiones distintos')
        qscale,qzero=result['int8_details']['input_quantization']
        params_blob=serialize_params(dims,nd,mask_id,indices,di,mean,scale,det_mean,det_scale,centroid,threshold,qscale,qzero)
        scope={key:job[key] for key in ('outer_load_hp','inner_validation_load_hp','training_global_window_ids',
            'validation_global_window_ids','detector_fit_global_window_ids','detector_calibration_global_window_ids',
            'benchmark_global_window_ids')}
        row=dict(id=rep_id,representative_id=job['job_id'],candidate=job['candidate'],
            outer_load_hp=job['outer_load_hp'],inner_validation_load_hp=job['inner_validation_load_hp'],
            mask_id=mask_id,dims=dims,det_dims=nd,feature_mask=f'{sum(1<<i for i in indices):04x}',
            model_path='models/'+job['job_id']+'.tflite',model_bytes=len(model),model_sha256=result['model_sha256']['INT8'],
            source_fit_signature=result['fit_signature'],fit_signature=result['fit_signature'],
            model_fnv=f'{fnv1a(model):08x}',params_fnv=f'{fnv1a(params_blob):08x}',params_bytes_hex=params_blob.hex(),
            feature_indices=indices,detector_indices=di,dnn_mean32_bits=bits32(mean),dnn_scale32_bits=bits32(scale),
            detector_mean32_bits=bits32(det_mean),detector_scale32_bits=bits32(det_scale),centroid32_bits=bits32(centroid),
            threshold_bits=bits32([threshold])[0],
            input_quantization=dict(scale_bits=bits32([qscale])[0],zero_point=int(qzero)),
            output_quantization=dict(scale_bits=bits32([result['int8_details']['output_quantization'][0]])[0],
                                     zero_point=int(result['int8_details']['output_quantization'][1])),scope=scope)
        descriptors.append(row);models.append(model)
        c_arrays.append(dict(mean=padded(mean,14),scale=padded(scale,14,1.),det_mean=padded(det_mean,12),
            det_scale=padded(det_scale,12,1.),centroid=padded(centroid,12),indices=padded(indices,14,0,np.uint8),
            det_indices=padded(di,12,0,np.uint8),threshold=threshold,qscale=np.float32(qscale),qzero=int(qzero)))
        with np.load(checked(directory/'banco_costos_referencia.npz'),allow_pickle=False) as saved:
            ids = saved['global_window_ids'].tolist()
            require(ids==job['benchmark_global_window_ids'] and len(ids)==28, 'Orden del banco distinto')
            expected_features=np.asarray([window_rows[position[gid]]['features_pc_cache14'] for gid in ids],dtype=np.float32)
            require(np.array_equal(saved['raw_features14'].view(np.uint32),expected_features.view(np.uint32)),
                    'Features del banco distintas de fuente auditada')
            normalized=normalize_f32(expected_features[:,indices],mean,scale)
            quantized,_=quantize_int8(normalized,qscale,qzero)
            require(np.array_equal(normalized.view(np.uint32),saved['normalized_features'].view(np.uint32))
                    and np.array_equal(quantized,saved['int8_inputs']) and saved['int8_outputs'].dtype==np.int8
                    and saved['int8_outputs'].shape==(28,3), 'Referencia PC no consistente')
            require(np.array_equal(saved['outside_normality'],saved['detector_scores']>threshold), 'Rechazo PC inconsistente')
            references[str(rep_id)]={str(gid):dict(raw_features14_bits=bits32(saved['raw_features14'][pos]),
                normalized_features_bits=bits32(saved['normalized_features'][pos]),
                int8_inputs=saved['int8_inputs'][pos].astype(int).tolist(),
                int8_outputs=saved['int8_outputs'][pos].astype(int).tolist(),
                detector_score_bits=bits32([saved['detector_scores'][pos]])[0],
                outside_normality=bool(saved['outside_normality'][pos])) for pos,gid in enumerate(ids)}

    canonical=[{key:row[key] for key in CANONICAL_KEYS} for row in descriptors]
    canonical_bytes=json.dumps(canonical,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')
    catalog_sha=hashlib.sha256(canonical_bytes).hexdigest()
    catalog_blob=b''.join(struct.pack('<II',int(row['model_fnv'],16),int(row['params_fnv'],16)) for row in descriptors)
    catalog_fnv=fnv1a(catalog_blob)
    policy=dict(full_feature_BASE_outside_measured_total=True,feature_rtol=.001,feature_atol=.00001,
        exact_dominant_bins=True,selected_kernel_vs_BASE='bit_exact_float32',inactive_slots_positive_zero=True,
        downstream_reference='PC_replay_from_published_ARM_features_using_corresponding_model_and_params',
        cached_PC_feature_reference_reported_separately=True,same_input_INT8_output_required_exact=True,
        no_claim_of_PC_cache_end_to_end_bit_exactness_from_ARM_feature_replay=True,
        timing_excludes=['UART','parser','checksums','model_initialization_or_switch','diagnostic_BASE'],
        cost_validated=False,confidence_interval_or_WCET_claimed=False)
    catalog=dict(schema_version=1,status='EXPORTED_NOT_ARM_VALIDATED',count=32,representatives=descriptors,
        catalog_sha256=catalog_sha,catalog_fnv=f'{catalog_fnv:08x}',canonical_catalog_path='CATALOGO_CANONICO.json',
        canonical_keys=list(CANONICAL_KEYS),canonical_json='UTF8 array, sorted keys, compact separators, no trailing newline',
        catalog_fnv_serialization='32 pairs model_fnv_u32le,params_fnv_u32le in representative ID order',
        params_serialization=dict(bytes=297,header='dims,det_dims,mask_id uint8',indices='14+12 uint8 padded0',
            floats='mean14,scale14,det_mean12,det_scale12,centroid12,threshold,qin_scale float32 little-endian',
            float_padding='means and centroid0; scales1',tail='qin_zero int32 little-endian'),validation_policy=policy)
    banks=[]
    for outer in range(4):
        sample=jobs[outer*8];ids=sample['benchmark_global_window_ids']
        require(all(j['benchmark_global_window_ids']==ids for j in jobs[outer*8:(outer+1)*8]), 'Bancos cambian por mascara')
        banks.append(dict(outer_load_hp=outer,representative_ids=list(range(outer*8,(outer+1)*8)),
            global_ids=ids,npz_indices=[position[gid] for gid in ids],training_loads=sample['benchmark_loads'],
            inner_validation_load_hp=sample['inner_validation_load_hp'],repetitions=3))
    bank=dict(schema_version=1,status='PC_BANK_READY_NOT_ARM_VALIDATED',windows_npz='ventanas_banco.npz',
        windows=window_rows,banks=banks,reference_by_rep_gid=references,repetitions=3,unique_windows=42,
        scoped_bank_window_occurrences=112,total_model_window_pairs=896,total_planned_RUN=2688,
        known_physical_feature_windows=28,new_load3_physical_feature_windows=14,validation_policy=policy,
        serialization='windows float32 little-endian C-order4096x2, DE/FE interleaved,32768bytes per payload',
        independent_physical_acquisitions_claimed=False,predictive_outer_evaluation=False)
    checked(Path(__file__))
    for relative,digest in list(sources.items()):
        require(sha(ROOT/relative)==digest,'Fuente cambio durante exportacion: '+relative)
    OUT.mkdir(parents=True,exist_ok=False);(OUT/'models').mkdir()
    for row,model in zip(descriptors,models):
        with (OUT/row['model_path']).open('xb') as stream:stream.write(model)
        require(sha(OUT/row['model_path'])==row['model_sha256'],'Modelo copiado distinto')
    with (OUT/'ventanas_banco.npz').open('xb') as stream:
        np.savez(stream,global_window_ids=np.asarray(global_ids,dtype='<i8'),windows=np.asarray(windows,dtype='<f4'))
    bank['windows_npz_sha256']=sha(OUT/'ventanas_banco.npz')
    with (OUT/'CATALOGO_CANONICO.json').open('xb') as stream:stream.write(canonical_bytes)
    write(OUT/'CATALOGO_TOTAL.json',catalog);write(OUT/'BANCO_TOTAL.json',bank)
    header='''#ifndef CWRU_TOTAL_CATALOG_H_\n#define CWRU_TOTAL_CATALOG_H_\n#include <stdint.h>\n#ifdef __cplusplus\nextern "C" {\n#endif\n#define CWRUTV_COUNT 32U\ntypedef struct {\n const char *id;\n const unsigned char *model;\n unsigned int model_size;\n uint8_t mask_id,dims,det_dims;\n uint8_t indices[14],det_indices[12];\n float mean[14],scale[14],det_mean[12],det_scale[12],centroid[12];\n float threshold,qin_scale;\n int32_t qin_zero;\n uint32_t model_fnv,params_fnv;\n const char *model_sha256,*source_fit_signature;\n} CWRUTV_Descriptor;\nextern const CWRUTV_Descriptor cwru_total_catalog[CWRUTV_COUNT];\nextern const char cwru_total_catalog_sha256[];\nextern const uint32_t cwru_total_catalog_fnv;\n#ifdef __cplusplus\n}\n#endif\n#endif\n'''
    lines=['#include "cwru_total_catalog.h"','']
    for rep_id,model in enumerate(models):
        lines.append(f'alignas(16) static const unsigned char cwru_tv_model_{rep_id:02d}[] = {{')
        for offset in range(0,len(model),16):
            lines.append(' '+','.join(f'0x{b:02x}' for b in model[offset:offset+16])+',')
        lines.extend(['};',''])
    lines.append('const CWRUTV_Descriptor cwru_total_catalog[CWRUTV_COUNT] = {')
    for row,arrays in zip(descriptors,c_arrays):
        rep_id=row['id'];lines.append(' {')
        lines.append(f'  "{row["representative_id"]}",cwru_tv_model_{rep_id:02d},{row["model_bytes"]}U,')
        lines.append(f'  {row["mask_id"]}U,{row["dims"]}U,{row["det_dims"]}U,')
        for key in ('indices','det_indices'):lines.append('  {'+','.join(str(int(v))+'U' for v in arrays[key])+'},')
        for key in ('mean','scale','det_mean','det_scale','centroid'):
            lines.append('  {'+','.join(cfloat(v) for v in arrays[key])+'},')
        lines.append(f'  {cfloat(arrays["threshold"])},{cfloat(arrays["qscale"])},{arrays["qzero"]},')
        lines.append(f'  0x{row["model_fnv"]}U,0x{row["params_fnv"]}U,"{row["model_sha256"]}","{row["source_fit_signature"]}"')
        lines.append(' },')
    lines.extend(['};',f'const char cwru_total_catalog_sha256[] = "{catalog_sha}";',
                  f'const uint32_t cwru_total_catalog_fnv = 0x{catalog_fnv:08x}U;',''])
    for name,text in (('cwru_total_catalog.h',header),('cwru_total_catalog.cpp','\n'.join(lines))):
        with (OUT/name).open('x',encoding='utf-8',newline='\n') as stream:stream.write(text)
    write(OUT/'FUENTES_SHA256.json',dict(sorted(sources.items())))
    resource_manifest={path.relative_to(OUT).as_posix():sha(path) for path in sorted(OUT.rglob('*')) if path.is_file()}
    manifest=OUT/'MANIFEST_RECURSOS_SHA256.json';write(manifest,resource_manifest)
    with manifest.with_suffix('.sha256').open('x',encoding='ascii') as stream:
        stream.write(sha(manifest)+'  '+manifest.name+'\n')
    require(sha(OUT/'CATALOGO_CANONICO.json')==catalog_sha,'Hash canonico distinto tras escribir')
    for relative,digest in resource_manifest.items():require(sha(OUT/relative)==digest,'Recurso exportado distinto')
    return dict(status='EXPORTED_NOT_ARM_VALIDATED',resources=len(resource_manifest),representatives=32,
                unique_windows=42,total_planned_RUN=2688,catalog_sha256=catalog_sha,catalog_fnv=f'{catalog_fnv:08x}',
                output=str(OUT),training_fits=0,TFLite_invocations=0)


if __name__=='__main__':
    print(json.dumps(export_resources(),ensure_ascii=False))
