"""Recupera SOLO exportacion T6 desde checkpoint: cero entrenamiento."""
from __future__ import annotations
import time
START = time.perf_counter()
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[4]
PREP = ROOT / 'articulo/ampliacion_cwru/metodologia/recuperacion_conversion_t6_2026-10-25'
OUTPUT = ROOT / 'resultados/cwru/ampliacion_v2/recuperacion_conversion_t6_2026-10-25'
PRIOR = ROOT / 'resultados/cwru/ampliacion_v2/piloto_tiempo_exportacion_v2_2026-10-25/20261025T063627_547883Z'
JOB = 'pilot_outer3_inner0_T6_seed11'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def verify(path):
    digest, name = path.with_suffix('.sha256').read_text(encoding='ascii').split()
    require(name == path.name and sha(path) == digest, 'Sello distinto')
    for name, digest in read(path).items():
        target = (ROOT / name).resolve()
        require(target.is_relative_to(ROOT) and sha(target) == digest, 'Recurso modificado: ' + name)


def prepare():
    verify(PRIOR / 'MANIFEST_RESULTADOS_PILOTO_SHA256.json')
    contract = dict(scope='T6_checkpoint_export_only', maximum_training_updates=0, maximum_model_conversions=2,
        timeout_worker_seconds=300, mass_training_enabled=False, outer_predictive_evaluation_enabled=False,
        prior_run=PRIOR.relative_to(ROOT).as_posix(), prior_manifest_sha256=sha(PRIOR/'MANIFEST_RESULTADOS_PILOTO_SHA256.json'),
        source_sha256=sha(Path(__file__)), no_F14_rerun=True, no_T6_retraining=True,
        fix='Compare unfrozen and frozen ConcreteFunctions with identical batch size 1, preserving prior tolerance.',
        automatic_retry=False)
    require(not (PREP / 'CONTRATO_RECUPERACION.json').exists(), 'No sobrescribir recuperacion')
    write(PREP / 'CONTRATO_RECUPERACION.json', contract)
    paths = [Path(__file__), PREP/'README.md', PREP/'CONTRATO_RECUPERACION.json',
             PRIOR/'MANIFEST_RESULTADOS_PILOTO_SHA256.json', PRIOR/'MANIFEST_RESULTADOS_PILOTO_SHA256.sha256']
    manifest=PREP/'MANIFEST_PREPARACION_SHA256.json'
    write(manifest,{p.relative_to(ROOT).as_posix():sha(p) for p in paths})
    manifest.with_suffix('.sha256').write_text(sha(manifest)+'  '+manifest.name+'\n',encoding='ascii')
    verify(manifest)


def worker(directory):
    require(directory.resolve().parent == OUTPUT.resolve(), 'Directorio ajeno a recuperacion')
    verify(PREP/'MANIFEST_PREPARACION_SHA256.json')
    verify(PRIOR/'MANIFEST_RESULTADOS_PILOTO_SHA256.json')
    contract=read(PREP/'CONTRATO_RECUPERACION.json')
    require(contract['maximum_training_updates']==0 and contract['source_sha256']==sha(Path(__file__)), 'Contrato invalido')
    timing={k:0. for k in ('imports','data_and_scaler','graph_compile','training_updates',
        'epoch_training_diagnostics','epoch_validation','FP32_conversion','INT8_conversion','inference','artifact_IO')}
    started=time.perf_counter()
    import numpy as np
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
    from datos_piloto import load_verified_plan,load_job_data
    timing['imports']=time.perf_counter()-started
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.set_visible_devices([], 'GPU')
    tf.keras.utils.set_random_seed(11)
    tf.config.experimental.enable_op_determinism()
    old=PRIOR/JOB
    state=read(old/'estado_antes_conversion.json')
    require(state['optimizer_updates']==1250 and len(state['history'])==50, 'Checkpoint incompleto')
    started=time.perf_counter()
    job=next(j for j in load_verified_plan()['jobs'] if j['job_id']==JOB)
    raw,y_train,val,y_val,provenance=load_job_data(job)
    with np.load(old/'scaler_train_only.npz',allow_pickle=False) as scaler:
        mean,scale=scaler['mean32'],scaler['scale32']
        x=np.divide(np.subtract(raw[:,job['feature_indices']],mean,dtype=np.float32),scale,dtype=np.float32)
        v=np.divide(np.subtract(val[:,job['feature_indices']],mean,dtype=np.float32),scale,dtype=np.float32)
    timing['data_and_scaler']=time.perf_counter()-started
    started=time.perf_counter()
    model=keras.Sequential([keras.layers.Input(shape=(6,),name='features'),
        keras.layers.Dense(20,activation='relu',kernel_regularizer=keras.regularizers.l2(.001),name='hidden20'),
        keras.layers.Dense(10,activation='relu',kernel_regularizer=keras.regularizers.l2(.001),name='hidden10'),
        keras.layers.Dense(3,activation='softmax',name='classes')],name='cwru_runtime_pilot_T6')
    with np.load(old/'checkpoint_antes_conversion.npz',allow_pickle=False) as checkpoint:
        weights=[checkpoint[k] for k in checkpoint.files]
    model.set_weights(weights)
    def weights_hash():
        digest=hashlib.sha256()
        for w in model.get_weights():
            digest.update(str(w.shape).encode('ascii')); digest.update(w.tobytes())
        return digest.hexdigest()
    require(weights_hash()==state['model_weights_sha256'],'Pesos restaurados distintos')
    timing['graph_compile']=time.perf_counter()-started
    started=time.perf_counter()
    @tf.function(input_signature=[tf.TensorSpec([1,6],tf.float32)])
    def fn(inputs):
        return model(inputs,training=False)
    concrete=fn.get_concrete_function()
    frozen=convert_variables_to_constants_v2(concrete,lower_control_flow=False)
    nodes=list(frozen.graph.as_graph_def().node)
    require(not any(n.op in {'ReadVariableOp','VarHandleOp','VariableV2','AssignVariableOp'} for n in nodes), 'Variables remanentes')
    original_rows=np.asarray([concrete(tf.convert_to_tensor(row[np.newaxis,:])).numpy()[0] for row in v],dtype=np.float32)
    frozen_rows=np.asarray([frozen(tf.convert_to_tensor(row[np.newaxis,:]))[0].numpy()[0] for row in v],dtype=np.float32)
    batch_rows=model(v,training=False).numpy()
    diagnostic=dict(weights_unchanged=weights_hash()==state['model_weights_sha256'],
        original_batch1_vs_frozen_batch1_max_absolute_error=float(np.max(np.abs(original_rows-frozen_rows))),
        original_batch1_vs_frozen_batch1_allclose=bool(np.allclose(original_rows,frozen_rows,rtol=1e-6,atol=1e-7)),
        original_batch1_vs_frozen_batch1_class_changes=int(np.count_nonzero(np.argmax(original_rows,axis=1)!=np.argmax(frozen_rows,axis=1))),
        original_batch175_vs_frozen_batch1_max_absolute_error=float(np.max(np.abs(batch_rows-frozen_rows))),
        original_batch175_vs_frozen_batch1_allclose=bool(np.allclose(batch_rows,frozen_rows,rtol=1e-6,atol=1e-7)),
        original_batch175_vs_frozen_batch1_class_changes=int(np.count_nonzero(np.argmax(batch_rows,axis=1)!=np.argmax(frozen_rows,axis=1))),
        tolerance_unchanged=dict(rtol=1e-6,atol=1e-7), source_checkpoint_sha256=sha(old/'checkpoint_antes_conversion.npz'))
    write(directory/'DIAGNOSTICO_FREEZE.json',diagnostic)
    require(diagnostic['weights_unchanged'] and diagnostic['original_batch1_vs_frozen_batch1_allclose']
            and diagnostic['original_batch1_vs_frozen_batch1_class_changes']==0,'Freeze difiere a igual batch')
    conv=tf.lite.TFLiteConverter.from_concrete_functions([frozen])
    fp32=conv.convert()
    timing['FP32_conversion']=time.perf_counter()-started
    started=time.perf_counter(); count=0
    def representative():
        nonlocal count
        for row in x:
            count+=1; yield [row[np.newaxis,:]]
    conv=tf.lite.TFLiteConverter.from_concrete_functions([frozen])
    conv.optimizations=[tf.lite.Optimize.DEFAULT]; conv.representative_dataset=representative
    conv.target_spec.supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type=tf.int8; conv.inference_output_type=tf.int8
    integer=conv.convert()
    timing['INT8_conversion']=time.perf_counter()-started
    require(count==799,'Calibracion fuera de entrenamiento')
    def evaluate(content,int8):
        interpreter=tf.lite.Interpreter(model_content=content,num_threads=1,
            experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
        interpreter.allocate_tensors(); inp=interpreter.get_input_details()[0]; out=interpreter.get_output_details()[0]
        types=sorted({np.dtype(t['dtype']).name for t in interpreter.get_tensor_details()})
        ops=[o['op_name'] for o in interpreter._get_ops_details()]
        if int8:
            require(inp['dtype']==out['dtype']==np.int8 and set(types)<={'int8','int32','int64'}
                    and not any('Flex'in op or op in ('DEQUANTIZE','DELEGATE','READ_VARIABLE') for op in ops),'Fallback no permitido')
        values=[]; raw=[]; saturation=0
        for row in v:
            batch=row[np.newaxis,:]
            if int8:
                s,z=inp['quantization']
                q=np.rint(np.add(np.divide(batch,np.float32(s),dtype=np.float32),np.float32(z),dtype=np.float32))
                saturation+=int(np.count_nonzero((q < -128)|(q > 127))); batch=np.clip(q,-128,127).astype(np.int8)
            interpreter.set_tensor(inp['index'],batch); interpreter.invoke(); a=interpreter.get_tensor(out['index'])[0]
            raw.append(a); a=a.astype(np.float32)
            if int8:
                s,z=out['quantization']; a=(a-np.float32(z))*np.float32(s)
            values.append(a)
        values=np.asarray(values,dtype=np.float32); require(np.isfinite(values).all(),'Salida no finita')
        return values,np.asarray(raw),dict(input_dtype=np.dtype(inp['dtype']).name,output_dtype=np.dtype(out['dtype']).name,
            input_quantization=list(inp['quantization']),output_quantization=list(out['quantization']),
            tensor_dtypes=types,operators=ops,validation_saturated_input_values=saturation)
    started=time.perf_counter()
    fp,_,fp_detail=evaluate(fp32,False); iq,ir,iq_detail=evaluate(integer,True)
    timing['inference']=time.perf_counter()-started
    started=time.perf_counter()
    (directory/'modelo_fp32.tflite').write_bytes(fp32); (directory/'modelo_int8.tflite').write_bytes(integer)
    np.savez(directory/'diagnostico_conversion_validacion.npz',y_true=y_val,
        keras_probabilities=batch_rows,keras_batch1_probabilities=original_rows,
        fp32_probabilities=fp,int8_probabilities=iq,int8_outputs=ir,
        validation_global_window_ids=np.asarray(job['validation']['global_window_ids'],dtype=np.int64))
    write(directory/'procedencia.json',provenance)
    timing['artifact_IO']=time.perf_counter()-started
    combined={k:state['timing_nonoverlapping_seconds'][k]+timing[k] for k in timing}
    result=dict(status='PASS',mask='T6',job_id=JOB,training_windows=799,validation_windows=175,
        epochs=50,batches_per_epoch=25,optimizer_updates=1250,training_updates_in_this_recovery=0,
        timing_nonoverlapping_seconds=combined,recovery_only_stage_seconds=timing,
        profile_kind='composite_training_checkpoint_plus_export_recovery_including_both_imports',
        extra_import_and_setup_included=True,source_checkpoint_run=PRIOR.relative_to(ROOT).as_posix(),
        representative_calibration_rows=count,parameters=model.count_params(),
        model_bytes=dict(FP32=len(fp32),INT8=len(integer)),
        model_sha256=dict(FP32=hashlib.sha256(fp32).hexdigest(),INT8=hashlib.sha256(integer).hexdigest()),
        fp32_details=fp_detail,int8_details=iq_detail,freeze_diagnostics=diagnostic,
        FP32_INT8_class_changes=int(np.count_nonzero(np.argmax(fp,axis=1)!=np.argmax(iq,axis=1))),
        no_load3_arrays_opened=True,no_detector_fitted=True,confirmatory_fit=False,
        tensorflow_version=tf.__version__,versions=dict(tensorflow=tf.__version__,numpy=np.__version__),
        elapsed_recovery_worker_to_result_seconds=time.perf_counter()-START)
    write(directory/'resultado.json',result)


def execute():
    from ejecutar_piloto_tiempo_v2 import ENVIRONMENT
    verify(PREP/'MANIFEST_PREPARACION_SHA256.json')
    OUTPUT.mkdir(parents=True,exist_ok=True)
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    write(OUTPUT/'EJECUCION_UNICA.json',dict(run_id=stamp,training_updates_authorized=0))
    directory=OUTPUT/stamp; directory.mkdir()
    env=os.environ.copy(); env.update(ENVIRONMENT)
    started=time.perf_counter(); timedout=False
    with (directory/'stdout.log').open('x',encoding='utf-8') as out,(directory/'stderr.log').open('x',encoding='utf-8') as err:
        process=subprocess.Popen([sys.executable,'-B',str(Path(__file__)),'--worker-dir',str(directory)],cwd=ROOT,env=env,stdout=out,stderr=err)
        try: code=process.wait(timeout=300)
        except subprocess.TimeoutExpired:
            timedout=True; process.kill(); process.wait(); code=process.returncode
    passed=code==0 and (directory/'resultado.json').is_file()
    write(directory/'ESTADO_RECUPERACION.json',dict(status='PASS'if passed else'TIMEOUT'if timedout else'FAIL',
        return_code=code,elapsed_process_seconds=time.perf_counter()-started,training_updates_executed=0,
        timeout_seconds=300,automatic_retry=False))
    manifest=directory/'MANIFEST_RECUPERACION_SHA256.json'
    entries={p.relative_to(ROOT).as_posix():sha(p) for p in directory.iterdir() if p.is_file()}
    entries.update(read(PREP/'MANIFEST_PREPARACION_SHA256.json'))
    entries[(PREP/'MANIFEST_PREPARACION_SHA256.json').relative_to(ROOT).as_posix()]=sha(PREP/'MANIFEST_PREPARACION_SHA256.json')
    entries[(PREP/'MANIFEST_PREPARACION_SHA256.sha256').relative_to(ROOT).as_posix()]=sha(PREP/'MANIFEST_PREPARACION_SHA256.sha256')
    write(manifest,entries); manifest.with_suffix('.sha256').write_text(sha(manifest)+'  '+manifest.name+'\n',encoding='ascii'); verify(manifest)
    print(json.dumps(dict(status='PASS'if passed else'FAIL',training_updates_executed=0,evidence_directory=str(directory))))
    return 0 if passed else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare',action='store_true'); mode.add_argument('--execute',action='store_true')
    mode.add_argument('--worker-dir',type=Path)
    args=parser.parse_args()
    if args.prepare: prepare()
    elif args.execute: raise SystemExit(execute())
    else:
        try: worker(args.worker_dir)
        except Exception as error:
            if args.worker_dir.resolve().parent==OUTPUT.resolve():
                write(args.worker_dir/'FALLO.json',dict(error=str(error),traceback=traceback.format_exc(),training_updates_executed=0))
            raise
