"""Bancos canónicos escalonados: RAW parcial y POST exterior completo.

No entrena, convierte ni invoca clasificadores. Los resultados esperados se
copian de inferencias previamente selladas. Importar no abre datos ni escribe.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import exportar_catalogo as e

for directory in ('representantes','detector_interno','piloto_clasicos'):
    sys.path.insert(0,str(e.BASE/directory))
from datos_representantes import load_verified_context,PARTITIONS_REL
from numerica_representantes import normalize_f32,quantize_int8
from datos_detector import load_params
from detector_numerico import detector_score

OUT=e.OUT.parent/'bancos'
MANIFEST='MANIFEST_BANCOS_SHA256.json'


def require(ok,message):e.require(ok,message)


def same(a,b):
    return a.dtype==b.dtype and a.shape==b.shape and a.tobytes(order='C')==b.tobytes(order='C')


def build_plan(metadata,catalog):
    """Selección mecánica por identidad y orden, sin abrir valores o métricas."""
    require(len(metadata)==1380 and [int(r['global_window_id']) for r in metadata]==list(range(1380)),
        'Índice global incompleto o desordenado')
    by_record={}
    for row in metadata:by_record.setdefault(str(row['file_id']),[]).append(row)
    require(len(by_record)==28,'Se requieren 28 MAT')
    ids=[]
    for record in sorted(by_record,key=int):
        rows=sorted(by_record[record],key=lambda r:int(r['window_index_in_file']))
        require(len(rows)>1 and len({int(r['window_index_in_file']) for r in rows})==len(rows),
            'Primera/última ventana no son distintas')
        require(len({int(r['load_hp']) for r in rows})==1,'MAT con cargas contradictorias')
        ids.extend([int(rows[0]['global_window_id']),int(rows[-1]['global_window_id'])])
    require(len(ids)==len(set(ids))==56,'Banco RAW requiere 56 ventanas únicas')
    index={int(r['global_window_id']):r for r in metadata};models=[]
    rows=catalog['representatives']
    require(tuple(r['representative_id'] for r in rows)==e.MODEL_NAMES
        and [r['id'] for r in rows]==list(range(8)),'Catálogo no es el canon fijado')
    for row in rows:
        outer=row['outer_load_hp'];raw=[gid for gid in ids if int(index[gid]['load_hp'])!=outer]
        post=[int(r['global_window_id']) for r in metadata if int(r['load_hp'])==outer]
        require(row['seed']==42 and len(raw)==42 and len(set(str(index[g]['file_id']) for g in raw))==21,
            'RAW fuera de tres cargas de desarrollo')
        require(set(raw)<=set(row['scope']['training_global_window_ids']) and not set(raw)&set(post),
            'Roles RAW/POST solapados por modelo')
        models.append(dict(id=row['id'],model_job_id=row['representative_id'],outer_load_hp=outer,
            feature_indices=row['feature_indices'],detector_job_id=row['detector_job_id'],
            raw_global_window_ids=raw,post_global_window_ids=post,reference_prefix=f'm{row["id"]}_'))
    require(sum(len(r['post_global_window_ids']) for r in models)==2760,'POST debe cubrir dos modelos por ventana')
    return dict(status='STAGED_CANONICAL_BANKS_PREPARED_PENDING_HARDWARE',seed=42,
        catalog_sha256=catalog['catalog_sha256'],catalog_fnv=catalog['catalog_fnv'],
        RAW_global_window_ids=ids,models=models,
        counts=dict(RAW_unique_windows=56,RAW_model_window_pairs=336,RAW_primary_RUN=1008,
            RAW_PROFILE=336,RAW_BASE=56,POST_unique_heldout_windows=1380,POST_model_window_pairs=2760),
        selection_rule='First and last indexed window of each of 28 MAT; each model uses only its three development loads.',
        RAW_repetitions=3,PROFILE_repetitions=1,
        policy=dict(RAW_feature_rtol=0.001,RAW_feature_atol=0.00001,dominant_bin_indices=[4,11],
            dominant_bin_exact=True,mask_vs_BASE_bit_exact=True,inactive_slots_positive_zero=True,
            same_ARM_feature_vector_downstream_PC_replay_bit_exact=True,
            cache_PC_vs_RAW_end_to_end_differences_reported_separately=True,
            POST_all_reserved_feature_vectors=True,RAW_exhaustive_reserved_coverage=False,
            POST_timing_is_full_pipeline_cost=False,old_timing_imports_allowed=False,
            training_fits=0,classifier_invocations=0,models_or_parameters_changed=False,
            RAW_references_are_training_diagnostics=True,RAW_is_partial_development_coverage=True))


def _checked(path,sources,expected=None):return e._checked(path,sources,expected)


def _read(path,sources,expected=None):return e._read(path,sources,expected)


def _npz(path,sources,expected=None):
    with np.load(_checked(path,sources,expected),allow_pickle=False) as a:
        return {k:a[k].copy() for k in a.files}


def _bits(values):return e.bits32(values)


def _from_bits(values):return np.asarray([int(v,16) for v in values],dtype='<u4').view('<f4')


def _verify_reference(raw,z,iq,outputs,pred,row):
    indices=row['feature_indices']
    expected=normalize_f32(raw[:,indices],_from_bits(row['dnn_mean32_bits']),_from_bits(row['dnn_scale32_bits']))
    require(same(z,expected),'Normalización guardada distinta del scaler congelado')
    expected_iq,_=quantize_int8(z,_from_bits([row['input_quantization']['scale_bits']])[0],
        row['input_quantization']['zero_point'])
    require(same(iq,expected_iq),'INT8 de referencia distinto de la regla congelada')
    require(outputs.dtype==np.int8 and outputs.shape==(len(raw),3) and pred.dtype==np.int64
        and pred.shape==(len(raw),) and np.array_equal(np.argmax(outputs,axis=1),pred),
        'Salidas o argmax de referencia inválidos')


def build_banks(context=None):
    context=e.load_context() if context is None else context
    catalog,_,_,sources=e.build_catalog(context)
    protocol,cost,metadata,hashes=load_verified_context()
    sources=dict(sources)
    for name in ('indice_ventanas_fuente.csv','entradas_sha256.csv'):
        _checked(e.ROOT/PARTITIONS_REL/name,sources)
    plan=build_plan(metadata,catalog);index={int(r['global_window_id']):r for r in metadata}
    canonical={r['model_job_id']:r for r in context['canonical']['models']}
    inventory={r['job_id']:r for r in context['inventory']}
    checked_manifests=set()
    def check_job(record):
        path=record['manifest_path']
        if path not in checked_manifests:
            e._manifest(e.ROOT/path,record['manifest_sha256'],sources);checked_manifests.add(path)
    raw_arrays={};feature_arrays={}
    for field,target in [('source_windows',raw_arrays),('source_features',feature_arrays)]:
        for relative in sorted({index[g][field] for g in plan['RAW_global_window_ids']}):
            require(relative in hashes,'Fuente sin hash autorizado')
            a=np.load(_checked(e.ROOT/relative,sources,hashes[relative]),allow_pickle=False,mmap_mode='r')
            count=sum(r[field]==relative for r in metadata)
            shape=(count,4096,2) if field=='source_windows' else (count,14)
            dtype=np.float32 if field=='source_windows' else np.float64
            require(a.dtype==dtype and a.shape==shape,'Fuente RAW/features con forma o tipo distintos')
            target[relative]=a
    windows=[];features=[];rawrows=[];seen=set()
    for position,gid in enumerate(plan['RAW_global_window_ids']):
        row=index[gid];window=np.ascontiguousarray(raw_arrays[row['source_windows']][int(row['source_row'])],dtype='<f4')
        feature=np.asarray(feature_arrays[row['source_features']][int(row['source_row'])],dtype=np.float32)
        require(np.isfinite(window).all() and np.isfinite(feature).all(),'Banco RAW no finito')
        payload=window.tobytes(order='C');digest=hashlib.sha256(payload).hexdigest()
        require(digest not in seen,'RAW duplicado con IDs distintos');seen.add(digest)
        windows.append(window);features.append(feature)
        rawrows.append(dict(global_id=gid,npz_index=position,payload_sha256=digest,
            checksum_fnv1a=f'{e.fnv1a(payload):08x}',features_pc_cache14_bits=_bits(feature),metadata=row,
            authorized_model_ids=[m['id'] for m in plan['models'] if gid in m['raw_global_window_ids']]))
    raw=dict(global_window_ids=np.asarray(plan['RAW_global_window_ids'],dtype=np.int64),
        windows=np.asarray(windows,dtype=np.float32),features_pc_cache14=np.asarray(features,dtype=np.float32))
    rawrefs={};postrefs={};positions={int(g):i for i,g in enumerate(raw['global_window_ids'])}
    closure=_read(e.CLOSED/'CIERRE_EVALUACION_EXTERIOR.json',sources)
    campaign=e.bounded(closure['campaign']);cm=campaign/'MANIFEST_CAMPANA_EVALUACION_EXTERIOR_SHA256.json'
    if not cm.exists():
        candidates=[p for p in campaign.glob('MANIFEST*.json') if e.io.sha(p)==closure['campaign_manifest_sha256']]
        require(len(candidates)==1,'No se encuentra manifiesto exterior cerrado');cm=candidates[0]
    e._manifest(cm,closure['campaign_manifest_sha256'],sources)
    folds={}
    for outer in range(4):
        folder=campaign/f'outer_eval_{outer}'
        result=_read(folder/'RESULTADO.json',sources)
        require(result['status']=='PASS' and result['outer_load_hp']==outer,'Resultado exterior incompleto')
        snapshot=_npz(folder/'entrada_evaluacion.npz',sources,result['input_snapshot_sha256'])
        expected=[int(r['global_window_id']) for r in metadata if int(r['load_hp'])==outer]
        require(snapshot['global_window_ids'].dtype==np.int64 and snapshot['global_window_ids'].tolist()==expected
            and snapshot['raw_features14'].dtype==np.float32 and snapshot['raw_features14'].shape==(len(expected),14)
            and np.isfinite(snapshot['raw_features14']).all(),'Snapshot exterior no corresponde al índice')
        require(snapshot['y_true'].tolist()==[int(index[g]['label']) for g in expected]
            and snapshot['record_ids'].tolist()==[str(index[g]['file_id']) for g in expected],'Etiquetas/MAT exteriores distintos')
        folds[outer]=(folder,result,snapshot)
    for model,row in zip(plan['models'],catalog['representatives']):
        record=inventory[row['representative_id']];check_job(record)
        diag=_npz((e.ROOT/record['result_path']).parent/'diagnosticos_train_only.npz',sources)
        ids=diag['global_window_ids'];require(ids.dtype==np.int64 and len(ids)==len(set(ids.tolist())),'IDs diagnóstico duplicados')
        require(ids.tolist()==row['scope']['training_global_window_ids'],'Diagnóstico no es train_only fijado')
        locations={int(g):i for i,g in enumerate(ids)};selected=[locations[g] for g in model['raw_global_window_ids']]
        require(diag['record_ids'][selected].tolist()==[str(index[g]['file_id']) for g in model['raw_global_window_ids']]
            and diag['y_true'][selected].tolist()==[int(index[g]['label']) for g in model['raw_global_window_ids']],
            'MAT o etiquetas de diagnóstico RAW distintos')
        cache=np.asarray([raw['features_pc_cache14'][positions[g]] for g in model['raw_global_window_ids']],dtype=np.float32)
        z=normalize_f32(cache[:,row['feature_indices']],_from_bits(row['dnn_mean32_bits']),_from_bits(row['dnn_scale32_bits']))
        iq=diag['int8_inputs'][selected];outputs=diag['int8_outputs'][selected];pred=diag['int8_predictions'][selected]
        _verify_reference(cache,z,iq,outputs,pred,row)
        drecord=inventory[row['detector_job_id']];check_job(drecord)
        params_path=_checked(e.ROOT/drecord['paths']['parametros.npz'],sources,drecord['artifact_sha256']['parametros.npz'])
        params=load_params(params_path);scores=detector_score(cache,params)
        prefix=model['reference_prefix']
        def add(target,items):target.update({prefix+k:v for k,v in items.items()})
        add(rawrefs,dict(global_window_ids=np.asarray(model['raw_global_window_ids'],dtype=np.int64),
            features14=cache,normalized_features=z,int8_inputs=iq,int8_outputs=outputs,int8_predictions=pred,
            detector_scores=scores,detector_rejected=scores>params['threshold32']))
        folder,result,snapshot=folds[row['outer_load_hp']]
        evaluated={r['model_job_id']:r for r in result['model_results']};pr=evaluated[row['representative_id']]
        dr=evaluated[row['detector_job_id']]
        require(pr['fit_signature']==row['source_fit_signature'] and dr['fit_signature']==drecord['fit_signature'],
            'Predicciones de otros modelos')
        prediction=_npz(folder/pr['prediction_file'],sources,pr['prediction_sha256'])
        detector=_npz(folder/dr['prediction_file'],sources,dr['prediction_sha256'])
        for saved in (prediction,detector):
            require(same(saved['global_window_ids'],snapshot['global_window_ids'])
                and same(saved['y_true'],snapshot['y_true']) and same(saved['record_ids'],snapshot['record_ids']),
                'Filas de predicciones exteriores distintas')
        _verify_reference(snapshot['raw_features14'],prediction['normalized_features'],prediction['int8_inputs'],
            prediction['int8_outputs'],prediction['int8_predictions'],row)
        expected_scores=detector_score(snapshot['raw_features14'],params)
        require(same(detector['scores'],expected_scores) and same(detector['rejected'],expected_scores>params['threshold32']),
            'Detector exterior distinto de parámetros congelados')
        add(postrefs,dict(global_window_ids=snapshot['global_window_ids'],features14=snapshot['raw_features14'],
            y_true=snapshot['y_true'],record_ids=snapshot['record_ids'],normalized_features=prediction['normalized_features'],
            int8_inputs=prediction['int8_inputs'],int8_outputs=prediction['int8_outputs'],
            int8_predictions=prediction['int8_predictions'],detector_scores=detector['scores'],detector_rejected=detector['rejected']))
        payloads=[np.ascontiguousarray(v,dtype='<f4').tobytes() for v in snapshot['raw_features14']]
        model.update(RAW_reference_source='diagnosticos_train_only.npz; no new classifier invocation',
            POST_reference_source=(folder/pr['prediction_file']).relative_to(e.ROOT).as_posix(),
            POST_detector_reference_source=(folder/dr['prediction_file']).relative_to(e.ROOT).as_posix(),
            POST_payload_sha256=[hashlib.sha256(p).hexdigest() for p in payloads],
            POST_payload_fnv1a=[f'{e.fnv1a(p):08x}' for p in payloads],
            RAW_rows=42,POST_rows=len(snapshot['global_window_ids']))
    for path in [Path(__file__),e.ROOT/'codigo/cwru/ampliacion_v2/representantes/numerica_representantes.py',
        e.ROOT/'codigo/cwru/ampliacion_v2/piloto_clasicos/detector_numerico.py',
        e.ROOT/'codigo/cwru/ampliacion_v2/detector_interno/datos_detector.py']:_checked(path,sources)
    plan['RAW_windows']=rawrows
    plan['array_files']={'RAW':'banco_raw.npz','RAW_references':'referencia_raw.npz','POST_references':'referencia_post.npz'}
    plan['array_keys']={'banco_raw.npz':sorted(raw),'referencia_raw.npz':sorted(rawrefs),
        'referencia_post.npz':sorted(postrefs)}
    return plan,raw,rawrefs,postrefs,sources


def verify(output=OUT):
    output=e.bounded(output);manifest=output/MANIFEST
    require(manifest.with_suffix('.sha256').read_text(encoding='ascii').split()==[e.io.sha(manifest),manifest.name],
        'Sello banco modificado')
    entries=e.io.read(manifest)
    actual={p.relative_to(output).as_posix() for p in output.rglob('*') if p.is_file()
        and p not in (manifest,manifest.with_suffix('.sha256'))}
    require(set(entries)==actual,'Recursos del banco añadidos o ausentes')
    for relative,digest in entries.items():
        path=(output/relative).resolve();require(path.is_relative_to(output.resolve()) and e.io.sha(path)==digest,
            'Banco modificado: '+relative)
    plan=e.io.read(output/'BANCOS_VALIDACION.json')
    require(plan['counts']['RAW_unique_windows']==56 and plan['counts']['POST_model_window_pairs']==2760
        and plan['policy']['classifier_invocations']==0 and plan['policy']['RAW_exhaustive_reserved_coverage'] is False,
        'Alcance escalonado cambiado')
    with np.load(output/'banco_raw.npz',allow_pickle=False) as a:
        require(set(a.files)=={'global_window_ids','windows','features_pc_cache14'}
            and a['global_window_ids'].tolist()==plan['RAW_global_window_ids']
            and a['windows'].dtype==np.float32 and a['windows'].shape==(56,4096,2),'RAW incompleto')
        for row in plan['RAW_windows']:
            payload=a['windows'][row['npz_index']].astype('<f4',copy=False).tobytes()
            require(hashlib.sha256(payload).hexdigest()==row['payload_sha256']
                and f'{e.fnv1a(payload):08x}'==row['checksum_fnv1a'],'Payload RAW distinto')
    for key,name in [('raw_global_window_ids','referencia_raw.npz'),('post_global_window_ids','referencia_post.npz')]:
        with np.load(output/name,allow_pickle=False) as a:
            require(set(a.files)==set(plan['array_keys'][name]),'Campos de referencias añadidos o ausentes')
            for model in plan['models']:
                prefix=model['reference_prefix'];ids=a[prefix+'global_window_ids'];n=len(ids)
                require(ids.tolist()==model[key] and a[prefix+'features14'].dtype==np.float32
                    and a[prefix+'features14'].shape==(n,14) and a[prefix+'int8_outputs'].shape==(n,3),
                    'Referencias incompletas')
                if name=='referencia_post.npz':
                    payloads=[np.ascontiguousarray(v,dtype='<f4').tobytes() for v in a[prefix+'features14']]
                    require(model['POST_payload_sha256']==[hashlib.sha256(p).hexdigest() for p in payloads]
                        and model['POST_payload_fnv1a']==[f'{e.fnv1a(p):08x}' for p in payloads],
                        'Payload POST distinto de características guardadas')
    return plan


def prepare(output=OUT):
    output=e.bounded(output);require(not output.exists(),'No sobrescribir bancos existentes')
    plan,raw,rawrefs,postrefs,sources=build_banks()
    for relative,digest in sources.items():require(e.io.sha(e.ROOT/relative)==digest,'Fuente cambió durante preparación')
    output.mkdir(parents=True,exist_ok=False)
    for name,arrays in [('banco_raw.npz',raw),('referencia_raw.npz',rawrefs),('referencia_post.npz',postrefs)]:
        with (output/name).open('xb') as stream:np.savez(stream,**arrays)
    e.io.write(output/'BANCOS_VALIDACION.json',plan);e.io.write(output/'FUENTES_SHA256.json',dict(sorted(sources.items())))
    manifest=output/MANIFEST
    entries={p.relative_to(output).as_posix():e.io.sha(p) for p in sorted(output.iterdir()) if p.is_file()}
    e.io.write(manifest,entries)
    with manifest.with_suffix('.sha256').open('x',encoding='ascii') as stream:stream.write(e.io.sha(manifest)+'  '+manifest.name+'\n')
    verify(output)
    return dict(status=plan['status'],output=str(output),**plan['counts'],classifier_invocations=0,training_fits=0,
        serial_opened=False,hardware_flashed=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare',action='store_true');group.add_argument('--verify',action='store_true')
    args=parser.parse_args()
    print(json.dumps(prepare() if args.prepare else dict(status='BANKS_INTEGRITY_PASS',counts=verify()['counts']),ensure_ascii=True))
