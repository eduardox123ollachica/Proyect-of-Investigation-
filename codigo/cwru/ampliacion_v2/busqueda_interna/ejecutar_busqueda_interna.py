"""Ejecutor acotado de la busqueda DNN interna V2; entrenar solo con --run.

Importa resultados auditados por firma original y equivalencia de roles.
No hace seleccion, ajuste externo, fit de detector, RF/SVM o hardware.
"""
from __future__ import annotations
import argparse
import datetime
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
SOURCE = Path(__file__).resolve().parent
PREP = ROOT/'articulo/ampliacion_cwru/metodologia/busqueda_interna_v2_2026-10-26'
OUTPUT = ROOT/'resultados/cwru/ampliacion_v2/busqueda_interna_v2_2026-10-26'
REP = ROOT/'codigo/cwru/ampliacion_v2/representantes'
SUPP = ROOT/'codigo/cwru/ampliacion_v2/suplemento'
REP_RUN = ROOT/'resultados/cwru/ampliacion_v2/representantes_v2_2026-10-25/20261025T171847_957295Z'
SUPP_RUN = ROOT/'resultados/cwru/ampliacion_v2/suplemento_preprocesamiento_normalidad_2026-10-25/20261025T215652_507777Z'
PROTOCOL = ROOT/'articulo/ampliacion_cwru/metodologia/protocolo_v2_2026-10-25'
PLAN_FILE = PREP/'PLAN_EJECUCION.json'
PREP_MANIFEST = PREP/'MANIFEST_EJECUTOR_INTERNO_SHA256.json'
COST_OUT = ROOT/'despliegue_stm32/costo_total_single_rounding_v2_180mhz_2026-10-26'
if str(REP) not in sys.path:
    sys.path.insert(0,str(REP))
import ejecutar_representantes as legacy
ENVIRONMENT = dict(legacy.ENVIRONMENT)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path,value):
    with Path(path).open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n')


def canonical(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),
                                    allow_nan=False).encode('utf-8')).hexdigest()


def confined(base,relative):
    relative = Path(relative)
    require(not relative.is_absolute() and relative.parts and '..' not in relative.parts,'Ruta relativa no valida')
    target = (base/relative).resolve()
    require(target != base.resolve() and target.is_relative_to(base.resolve()),'Ruta fuera del destino autorizado')
    return target


def verify_manifest(path, exact_directory=None):
    digest,name = Path(path).with_suffix('.sha256').read_text(encoding='ascii').split()
    require(name == Path(path).name and digest == sha(path),'Sello distinto: '+str(path))
    entries = read(path)
    require(isinstance(entries,dict) and entries,'Manifiesto vacio')
    for relative,expected in entries.items():
        require(sha(confined(ROOT,relative)) == expected,'Archivo modificado: '+relative)
    if exact_directory is not None:
        excluded = {Path(path),Path(path).with_suffix('.sha256')}
        actual = {p.relative_to(ROOT).as_posix() for p in Path(exact_directory).iterdir()
                  if p.is_file() and p not in excluded}
        require(set(entries) == actual,'Ajuste con recursos faltantes o anadidos')
    return entries


def write_manifest(path, directory=None, extra=()):
    files = set(extra)
    if directory is not None:
        files.update(p for p in Path(directory).rglob('*') if p.is_file()
                     and p not in (Path(path),Path(path).with_suffix('.sha256')))
    entries = {p.relative_to(ROOT).as_posix():sha(p) for p in sorted(files)}
    write(path,entries)
    with Path(path).with_suffix('.sha256').open('x',encoding='ascii') as stream:
        stream.write(sha(path)+'  '+Path(path).name+'\n')
    verify_manifest(path)


def source_hashes():
    files = list(SOURCE.glob('*.py')) + list(REP.glob('*.py'))
    files += [ROOT/'codigo/cwru/ampliacion_v2/protocolo/reglas_v2.py']
    return {p.relative_to(ROOT).as_posix():sha(p) for p in sorted(files)}


def semantic_signature(job, protocol, versions, input_hashes):
    keys = ('outer_load_hp','inner_validation_load_hp','candidate','feature_indices','input_dimension',
            'seed','roles','training_global_window_ids','validation_global_window_ids')
    return canonical(dict(job={k:job[k] for k in keys},dnn=protocol['dnn'],
        numeric=protocol['numeric_contract'],quantization=protocol['quantization'],versions=versions,
        source_features_sha256={k:input_hashes[k] for k in job['authorized_feature_source_paths']},
        frozen_training_core_sha256=sha(REP/'worker_representantes.py')))


def execution_signature(job, sources, versions, protocol):
    return canonical(dict(job=job,sources=sources,versions=versions,dnn=protocol['dnn'],
        numeric=protocol['numeric_contract'],quantization=protocol['quantization'],
        protocol_manifest_sha256=sha(PROTOCOL/'MANIFEST_PROTOCOLO_SHA256.json')))


def import_result(job, source_job, directory, result, family, expected_signature, protocol, versions, inputs):
    require(family in ('representantes','suplemento_P0')
            and (family != 'suplemento_P0' or source_job.get('mode') == 'P0'), 'Familia de reutilizacion no autorizada')
    semantic_keys = ('feature_indices','input_dimension','seed','roles','training_global_window_ids',
                     'validation_global_window_ids','outer_load_hp','inner_validation_load_hp','candidate')
    require(all(job[k] == source_job[k] for k in semantic_keys),'Resultado con scope semantico distinto')
    require(result['fit_signature'] == expected_signature and result['versions'] == versions,'Firma original o entorno distintos')
    return dict(job_id=job['job_id'],family=family,source_job_id=source_job['job_id'],
        source_directory=directory.relative_to(ROOT).as_posix(),
        source_result_sha256=sha(directory/'resultado.json'),source_manifest_sha256=sha(directory/'MANIFEST_JOB_SHA256.json'),
        source_fit_signature=expected_signature,semantic_signature=semantic_signature(job,protocol,versions,inputs),
        model_sha256=result['model_sha256'],best_epoch=result['best_epoch'],epochs_executed=result['epochs_executed'],
        metric_macro_f1_int8=result['metrics']['INT8']['macro_f1'],new_training_fits=0,
        signature_interpretation='Original signature verified; new wrapper signature is separate, not claimed textually identical')


def imported_references(jobs,protocol,inputs):
    rep_plan,rep_jobs,done,_ = legacy.inspect_campaign(REP_RUN)
    require(len(done) == 32,'Representantes originales incompletos')
    versions = rep_plan['tested_versions']
    mapping = {}
    by_key = {(j['outer_load_hp'],j['inner_validation_load_hp'],j['candidate'],j['seed']):j for j in jobs}
    for source_job in rep_jobs:
        key = (source_job['outer_load_hp'],source_job['inner_validation_load_hp'],source_job['candidate'],source_job['seed'])
        job = by_key[key]
        require(job == source_job,'Representante debe conservar exactamente su job original')
        directory = REP_RUN/source_job['job_id']
        result = legacy.verify_job(directory,source_job,rep_plan['fit_signatures'][source_job['job_id']])
        mapping[job['job_id']] = import_result(job,source_job,directory,result,'representantes',
            rep_plan['fit_signatures'][source_job['job_id']],protocol,versions,inputs)
    if str(SUPP) not in sys.path:
        sys.path.insert(0,str(SUPP))
    import ejecutar_suplemento as supplement
    supplementary = supplement.preflight()
    require(supplementary['tested_versions'] == versions,'Entorno del suplemento distinto')
    extra = 0
    for source_job in supplementary['jobs']:
        if source_job['mode'] != 'P0' or source_job['job_id'] in supplementary['imported_references']:
            continue
        key = (source_job['outer_load_hp'],source_job['inner_validation_load_hp'],source_job['candidate'],source_job['seed'])
        job = by_key[key]
        require(job['job_id'] not in mapping,'Resultado importado repetido')
        directory = SUPP_RUN/source_job['job_id']
        result = supplement.verify_new_job(directory,source_job,supplementary)
        mapping[job['job_id']] = import_result(job,source_job,directory,result,'suplemento_P0',
            supplementary['fit_signatures'][source_job['job_id']],protocol,versions,inputs)
        extra += 1
    require(len(mapping) == 48 and extra == 16,'Inventario debe contener32+16P0 distintos')
    return mapping,versions


def prepare():
    from datos_busqueda import context,build_jobs
    protocol,rows,inputs = context()
    jobs = build_jobs(protocol,rows)
    require(not PLAN_FILE.exists() and not PREP_MANIFEST.exists(),'No sobrescribir preparacion')
    PREP.mkdir(parents=True,exist_ok=True)
    imported,versions = imported_references(jobs,protocol,inputs)
    budget_path = PREP/'PRESUPUESTO_Y_LIMITES.json'
    budget = read(budget_path)
    require(budget['status'] == 'BUDGET_FROZEN_FOR_INNER_DNN_ONLY','Falta presupuesto congelado')
    limits = budget['limits']
    require(limits['worker_seconds'] == 300 and limits['max_new_jobs'] == 16
            and limits['maximum_new_fits_total'] == len(jobs)-len(imported),'Presupuesto incompatible con inventario')
    costs_manifest = COST_OUT/'MANIFEST_COSTOS_TOTALES_SR_VALIDADOS_SHA256.json'
    verify_manifest(costs_manifest)
    require(read(COST_OUT/'COSTOS_TOTALES_SR_VALIDADOS.json')['status'] == 'VALIDATED_TOTAL_COST','Costos no cerrados')
    sources = source_hashes()
    tests_path = PREP/'PRUEBAS_PREPARACION.json'
    tests = read(tests_path)
    require(tests['status'] == 'PASS' and tests['failures'] == tests['errors'] == 0
            and tests['source_sha256'] == sources and tests['simulation_only'] is True
            and tests['real_training_fits'] == tests['real_TFLite_invocations'] == 0,
            'Faltan pruebas de las fuentes finales, sin entrenamiento real')
    signatures = {j['job_id']:execution_signature(j,sources,versions,protocol) for j in jobs}
    semantic = {j['job_id']:semantic_signature(j,protocol,versions,inputs) for j in jobs}
    plan = dict(status='PREPARED_FOR_USER_INNER_DNN_EXECUTION',scope='inner_DNN_only_no_outer_evaluation_or_selection',
        jobs=jobs,logical_cells=288,imported=imported,imported_cells=len(imported),new_fits=len(jobs)-len(imported),
        tested_versions=versions,source_sha256=sources,limits=limits,
        budget_sha256=sha(budget_path),dnn=protocol['dnn'],worker_environment=ENVIRONMENT,
        fit_signatures=signatures,semantic_signatures=semantic,
        protocol_manifest_sha256=sha(PROTOCOL/'MANIFEST_PROTOCOLO_SHA256.json'),
        validated_cost_table_sha256=sha(COST_OUT/'COSTOS_TOTALES_SR_VALIDADOS.json'),
        detector_fits_enabled=False,classical_fits_enabled=False,outer_evaluation_enabled=False,
        predictive_selection_enabled=False,manuscript_edits_enabled=False,automatic_retry=False,
        concurrent_workers=1,maximum_epochs=50,early_stopping_preserved=True,
        user_starts_training=True,default_new_jobs_per_invocation=1,default_batch_seconds=600)
    write(PLAN_FILE,plan)
    write_manifest(PREP_MANIFEST,extra=[PLAN_FILE,budget_path,tests_path,*[ROOT/p for p in sources],
        PROTOCOL/'MANIFEST_PROTOCOLO_SHA256.json',PROTOCOL/'MANIFEST_PROTOCOLO_SHA256.sha256',
        COST_OUT/'COSTOS_TOTALES_SR_VALIDADOS.json',costs_manifest,costs_manifest.with_suffix('.sha256')])
    print(json.dumps(dict(status='PREPARED_FOR_USER_INNER_DNN_EXECUTION',logical_cells=288,
                         imported_cells=len(imported),new_fits=plan['new_fits'],training_fits=0),ensure_ascii=True))


def preflight(check_environment=False, verify_imports=True):
    from datos_busqueda import context,build_jobs
    verify_manifest(PREP_MANIFEST)
    plan = read(PLAN_FILE)
    protocol,rows,inputs = context()
    require(plan['jobs'] == build_jobs(protocol,rows) and plan['source_sha256'] == source_hashes()
            and plan['worker_environment'] == ENVIRONMENT and plan['dnn'] == protocol['dnn'], 'Plan o fuentes cambiaron')
    require(plan['budget_sha256'] == sha(PREP/'PRESUPUESTO_Y_LIMITES.json')
            and plan['limits'] == read(PREP/'PRESUPUESTO_Y_LIMITES.json')['limits'],'Presupuesto cambiado')
    require(plan['logical_cells'] == 288 and plan['imported_cells'] == 48 and plan['new_fits'] == 240
            and len(plan['imported']) == 48,'Inventario distinto del congelado')
    signatures = {j['job_id']:execution_signature(j,plan['source_sha256'],plan['tested_versions'],protocol) for j in plan['jobs']}
    require(signatures == plan['fit_signatures'],'Firmas nuevas distintas')
    require(all(not plan[key] for key in ('detector_fits_enabled','classical_fits_enabled','outer_evaluation_enabled',
                 'predictive_selection_enabled','manuscript_edits_enabled','automatic_retry')), 'Alcance extendido')
    if verify_imports:
        for imported in plan['imported'].values():
            directory = confined(ROOT,imported['source_directory'])
            verify_manifest(directory/'MANIFEST_JOB_SHA256.json',exact_directory=directory)
            require(sha(directory/'resultado.json') == imported['source_result_sha256']
                    and sha(directory/'MANIFEST_JOB_SHA256.json') == imported['source_manifest_sha256']
                    and read(directory/'resultado.json')['fit_signature'] == imported['source_fit_signature'],
                    'Importacion original alterada')
    if check_environment:
        require(legacy.versions() == plan['tested_versions'],'Usar el entorno Python exacto de los ajustes verificados')
    return plan


def validate_campaign(campaign, cheap=False):
    campaign = Path(campaign).resolve()
    require(campaign.parent == OUTPUT.resolve() and campaign.is_dir(),'Campana fuera de su directorio')
    require(read(OUTPUT/'CAMPANA_UNICA.json') == dict(run_id=campaign.name,
        preparation_sha256=sha(PREP_MANIFEST)), 'Campana distinta del marcador unico; no reiniciar presupuesto')
    plan = preflight(verify_imports=not cheap)
    require(read(campaign/'CONTRATO_CAMPANA.json') == dict(scope=plan['scope'],execution_owner='user_CLI',
        preparation_manifest_sha256=sha(PREP_MANIFEST),plan_sha256=sha(PLAN_FILE),new_fits=240,
        imported_cells=48,automatic_retry=False),'Contrato de campana distinto')
    return plan


def validate_result(result,job,signature,semantic_signature):
    require(result.get('status') == 'PASS' and result.get('job_id') == job['job_id']
            and result.get('fit_signature') == signature and result.get('roles') == job['roles']
            and result.get('seed') == job['seed'] and result.get('candidate') == job['candidate']
            and result.get('outer_load_hp') == job['outer_load_hp']
            and result.get('inner_validation_load_hp') == job['inner_validation_load_hp']
            and result.get('semantic_signature') == semantic_signature, 'Resultado ajeno o incompleto')
    require(result.get('training_windows') == len(job['training_global_window_ids'])
            and result.get('validation_windows') == len(job['validation_global_window_ids'])
            and result.get('representative_calibration_rows') == len(job['training_global_window_ids'])
            and result.get('parameters') == 20*job['input_dimension']+263
            and result.get('architecture') == [job['input_dimension'],20,10,3], 'Scope o arquitectura distintos')
    require(type(result.get('epochs_executed')) is int and type(result.get('best_epoch')) is int
            and 1 <= result['best_epoch'] <= result['epochs_executed'] <= 50
            and result.get('weights_restored') is True and result.get('detector_fit_executed') is False
            and result.get('outer_evaluation_executed') is False and result.get('predictive_selection_executed') is False,
            'Checkpoint o alcance incorrectos')
    require(0 <= result['metrics']['INT8']['macro_f1'] <= 1,'Macro-F1 invalido')
    return result


def verify_job(directory,job,signature,semantic_signature=None):
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from numerica_representantes import EarlyStopping,metrics3,normalize_f32,quantize_int8,integer_argmax
    from datos_busqueda import load_job_data
    plan=read(PLAN_FILE)
    if semantic_signature is None:
        semantic_signature = plan['semantic_signatures'][job['job_id']]
    verify_manifest(directory/'MANIFEST_JOB_SHA256.json',exact_directory=directory)
    result = validate_result(read(directory/'resultado.json'),job,signature,semantic_signature)
    require(result['versions'] == plan['tested_versions']
            and result['original_training_core_source_sha256'] == sha(REP/'worker_representantes.py')
            and result['preparation_manifest_sha256'] == sha(PREP_MANIFEST), 'Versiones, core o preparacion distintos')
    state = read(directory/'estado_antes_conversion.json')
    stopper = EarlyStopping()
    updates_per_epoch=(len(job['training_global_window_ids'])+31)//32
    for row in state['history']:
        improved,_ = stopper.observe(row['epoch'],row['val_loss'])
        require(row['improved'] == improved and row['best_epoch'] == stopper.best_epoch
                and row['patience_used'] == stopper.wait
                and row['optimizer_updates'] == row['epoch']*updates_per_epoch,'Historia de checkpoint inconsistente')
    require(stopper.best_epoch == state['best_epoch'] == result['best_epoch']
            and stopper.last_epoch == state['epochs_executed'] == result['epochs_executed']
            and len(state['history']) == state['epochs_executed']
            and stopper.best_loss == state['best_validation_loss']
            and state['optimizer_updates'] == result['optimizer_updates'] == len(state['history'])*updates_per_epoch
            and (state['epochs_executed'] == 50 or stopper.stopped)
            and state['weights_restored'], 'Mejor epoca, updates o restauracion inconsistentes')
    digest = hashlib.sha256()
    with np.load(directory/'checkpoint_antes_conversion.npz',allow_pickle=False) as restored, \
         np.load(directory/f"checkpoint_epoca_{state['best_epoch']:02d}.npz",allow_pickle=False) as best:
        require(restored.files == best.files and len(restored.files) == 6,'Checkpoint incompleto')
        shapes=[(job['input_dimension'],20),(20,),(20,10),(10,),(10,3),(3,)]
        for i,key in enumerate(restored.files):
            require(restored[key].shape == shapes[i] and restored[key].dtype == np.float32
                    and np.array_equal(restored[key],best[key]) and np.isfinite(restored[key]).all(),'Pesos no restaurados o forma distinta')
            digest.update(str(restored[key].shape).encode('ascii'));digest.update(restored[key].tobytes())
    require(digest.hexdigest() == state['model_weights_sha256'] == result['freeze']['model_weights_sha256'],'Hash pesos distinto')
    require(read(directory/'DIAGNOSTICO_FREEZE.json') == result['freeze']
            and result['freeze']['weights_unchanged'] and result['freeze']['class_changes'] == 0
            and result['freeze']['resource_variable_ops_remaining'] == 0, 'Diagnostico de congelacion distinto')
    raw,_,raw_val,y,provenance = load_job_data(job)
    require(read(directory/'procedencia.json') == provenance,'Procedencia de datos distinta')
    scaler=StandardScaler().fit(raw[:,job['feature_indices']])
    with np.load(directory/'scaler_train_only.npz',allow_pickle=False) as saved_scaler:
        require(all(np.array_equal(saved_scaler[key],value) for key,value in (
            ('mean64',scaler.mean_),('scale64',scaler.scale_),('var64',scaler.var_),
            ('mean32',scaler.mean_.astype(np.float32)),('scale32',scaler.scale_.astype(np.float32)))),
            'Scaler no reproduce entrenamiento autorizado')
    x=normalize_f32(raw[:,job['feature_indices']],scaler.mean_.astype(np.float32),scaler.scale_.astype(np.float32))
    val=normalize_f32(raw_val[:,job['feature_indices']],scaler.mean_.astype(np.float32),scaler.scale_.astype(np.float32))
    details=result['int8_details']
    require(details['input_dtype'] == details['output_dtype'] == 'int8'
            and set(details['tensor_dtypes']) <= {'int8','int32','int64'}
            and details['operators'] == ['FULLY_CONNECTED']*3+['SOFTMAX']
            and details['default_delegates_disabled'] and details['threads'] == 1,
            'Contrato INT8 distinto')
    expected_qtrain,train_clipped=quantize_int8(x,*details['input_quantization'])
    expected_qval,val_clipped=quantize_int8(val,*details['input_quantization'])
    require(train_clipped == result['training_saturated_input_values']
            and val_clipped == details['input_clipped_values'],'Saturaciones inconsistentes')
    with np.load(directory/'cuantizacion_train.npz',allow_pickle=False) as saved_train:
        require(saved_train['global_window_ids'].tolist() == job['training_global_window_ids']
                and saved_train['global_window_ids'].dtype == np.int64
                and saved_train['int8_inputs'].dtype == np.int8
                and np.array_equal(saved_train['int8_inputs'],expected_qtrain),'Cuantizacion train distinta')
    with np.load(directory/'predicciones_validacion.npz',allow_pickle=False) as saved:
        require(saved['global_window_ids'].tolist() == job['validation_global_window_ids']
                and saved['global_window_ids'].dtype == np.int64 and saved['y_true'].dtype == np.int64
                and np.array_equal(saved['y_true'],y),'IDs/etiquetas distintos')
        require(saved['fp32_outputs'].dtype == np.float32 and saved['int8_outputs'].dtype == np.int8
                and saved['fp32_outputs'].shape == saved['int8_outputs'].shape == (len(y),3)
                and np.isfinite(saved['fp32_outputs']).all()
                and saved['int8_inputs'].dtype == np.int8
                and np.array_equal(saved['int8_inputs'],expected_qval)
                and np.array_equal(saved['fp32_probabilities'],saved['fp32_outputs']),
                'Entradas/salidas guardadas incompatibles')
        scale,zero=details['output_quantization']
        require(np.isfinite(scale) and scale > 0,'Escala de salida invalida')
        probabilities=np.multiply(np.subtract(saved['int8_outputs'].astype(np.float32),np.float32(zero),dtype=np.float32),
                                  np.float32(scale),dtype=np.float32)
        require(np.array_equal(saved['int8_probabilities'],probabilities)
                and np.array_equal(saved['fp32_predictions'],integer_argmax(saved['fp32_outputs']))
                and np.array_equal(saved['int8_predictions'],integer_argmax(saved['int8_outputs']))
                and result['FP32_INT8_class_changes'] == int(np.count_nonzero(saved['fp32_predictions'] != saved['int8_predictions']))
                and result['keras_batch1_FP32_class_changes'] == int(np.count_nonzero(
                    integer_argmax(saved['keras_batch1']) != saved['fp32_predictions'])), 'Predicciones contradicen las salidas')
        for family,key in (('FP32','fp32_predictions'),('INT8','int8_predictions')):
            require(metrics3(y,saved[key]) == result['metrics'][family],'Metricas no reproducibles')
    for family,filename in (('FP32','modelo_fp32.tflite'),('INT8','modelo_int8.tflite')):
        require(sha(directory/filename) == result['model_sha256'][family]
                and (directory/filename).stat().st_size == result['model_bytes'][family], 'Modelo exportado modificado')
    return result


def inspect_campaign(campaign):
    plan = validate_campaign(campaign)
    consumed = 0.
    for batch in sorted((campaign/'lotes').iterdir()):
        verify_manifest(batch/'MANIFEST_LOTE_SHA256.json')
        summary = read(batch/'RESUMEN_LOTE.json')
        require(summary['status'] in ('BATCH_COMPLETED','BATCH_BUDGET_BOUNDARY','CAMPAIGN_COMPLETED'),
                'Fallo/interrupcion previa: conservar y auditar antes de recuperacion tecnica separada')
        consumed += summary['worker_elapsed_seconds']
    pending_order = [j for j in plan['jobs'] if j['job_id'] not in plan['imported']]
    done = []
    for job in pending_order:
        directory = campaign/job['job_id']
        if directory.exists():
            verify_job(directory,job,plan['fit_signatures'][job['job_id']]);done.append(job['job_id'])
    require(done == [j['job_id'] for j in pending_order[:len(done)]],'Huecos o trabajos completados fuera de orden')
    return plan,pending_order,done,consumed


def run_one(campaign,batch,job,environment,timeout,signature):
    started = time.monotonic()
    process,code,failure = None,None,None
    try:
        with (batch/(job['job_id']+'_stdout.log')).open('x',encoding='utf-8') as out, \
             (batch/(job['job_id']+'_stderr.log')).open('x',encoding='utf-8') as err:
            process = subprocess.Popen([sys.executable,'-B',str(SOURCE/'worker_busqueda_interna.py'),
                '--campaign',str(campaign),'--batch',str(batch),'--job-id',job['job_id']],cwd=ROOT,
                env=environment,stdout=out,stderr=err)
            code = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        failure = 'TIMEOUT'
    except KeyboardInterrupt:
        failure = 'INTERRUPTED'
    except OSError as error:
        failure = 'LAUNCH_ERROR: '+str(error)
    finally:
        if process is not None and process.poll() is None:
            process.kill();process.wait(timeout=30);code=process.returncode
    verification_error = None
    if failure is None and code == 0:
        try:
            verify_job(campaign/job['job_id'],job,signature)
        except Exception as error:
            verification_error = type(error).__name__+': '+str(error)
    status = dict(job_id=job['job_id'],status='PASS' if failure is None and code == 0 and verification_error is None else 'FAIL',
        return_code=code,failure=failure,result_error=verification_error,automatic_retry=False,
        worker_elapsed_seconds=time.monotonic()-started,timeout_seconds=timeout)
    write(batch/('ESTADO_'+job['job_id']+'.json'),status)
    return status


def can_start(worker_seconds,batch_remaining,campaign_remaining):
    return batch_remaining >= worker_seconds and campaign_remaining >= worker_seconds


def execute(campaign=None,max_new_jobs=1,max_seconds=600):
    plan = preflight(check_environment=True)
    limits = plan['limits']
    require(type(max_new_jobs) is int and 1 <= max_new_jobs <= limits['max_new_jobs'],'Lote fuera del maximo16')
    require(type(max_seconds) is int and limits['worker_seconds'] <= max_seconds <= limits['max_batch_seconds'],
            'Presupuesto de lote fuera de los limites congelados')
    OUTPUT.mkdir(parents=True,exist_ok=True)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    if campaign is None:
        write(OUTPUT/'CAMPANA_UNICA.json',dict(run_id=stamp,preparation_sha256=sha(PREP_MANIFEST)))
        campaign = OUTPUT/stamp;campaign.mkdir();(campaign/'lotes').mkdir()
        write(campaign/'CONTRATO_CAMPANA.json',dict(scope=plan['scope'],execution_owner='user_CLI',
            preparation_manifest_sha256=sha(PREP_MANIFEST),plan_sha256=sha(PLAN_FILE),new_fits=240,
            imported_cells=48,automatic_retry=False))
    campaign = Path(campaign).resolve()
    require(campaign.parent == OUTPUT.resolve(),'Campana ajena al directorio previsto')
    lock = campaign/'EJECUCION_EN_CURSO.lock'
    with lock.open('x',encoding='utf-8') as stream:
        stream.write(str(os.getpid()))
    try:
        plan,jobs,done,consumed = inspect_campaign(campaign)
        require(len(done) < len(jobs),'Campana ya completada; no repetir')
        selected = jobs[len(done):len(done)+max_new_jobs]
        batch = campaign/'lotes'/stamp;batch.mkdir()
        write(batch/'PLAN_LOTE.json',dict(job_ids=[j['job_id'] for j in selected],completed_before=done,
            consumed_worker_seconds_before=consumed,max_new_jobs=max_new_jobs,max_seconds=max_seconds))
        print('Campana: '+str(campaign),flush=True)
        print(f'Importadas verificadas:48/288; pendientes nuevos antes:{len(jobs)-len(done)}',flush=True)
        environment = os.environ.copy();environment.update(ENVIRONMENT)
        statuses = [];boundary=False;batch_start=time.monotonic()
        for job in selected:
            remaining = max_seconds-(time.monotonic()-batch_start)
            campaign_remaining = limits['campaign_worker_seconds']-consumed
            if not can_start(limits['worker_seconds'],remaining,campaign_remaining):
                boundary=True;break
            print(f"[{len(done)+len(statuses)+1}/{len(jobs)} nuevos] INICIO {job['job_id']}; limite300s",flush=True)
            status = run_one(campaign,batch,job,environment,limits['worker_seconds'],plan['fit_signatures'][job['job_id']])
            statuses.append(status);consumed+=status['worker_elapsed_seconds']
            print(job['job_id']+': '+status['status'],flush=True)
            if status['status'] != 'PASS':
                break
        passed = sum(s['status'] == 'PASS' for s in statuses)
        success = passed == len(statuses)
        complete = len(done)+passed == len(jobs)
        summary = dict(status='CAMPAIGN_COMPLETED' if complete else 'BATCH_BUDGET_BOUNDARY' if boundary and success
            else 'BATCH_COMPLETED' if success and len(statuses) == len(selected) else 'STOPPED_FOR_AUDIT',
            completed_new_fits=len(done)+passed,planned_new_fits=len(jobs),imported_cells=48,
            available_logical_cells=48+len(done)+passed,total_logical_cells=288,
            jobs_started=len(statuses),jobs_passed=passed,worker_elapsed_seconds=sum(s['worker_elapsed_seconds'] for s in statuses),
            campaign_worker_seconds_consumed=consumed,wall_seconds=time.monotonic()-batch_start,
            budget_boundary_without_starting_another_job=boundary,outer_evaluation_executed=False,
            detector_fits_executed=False,predictive_selection_executed=False)
        write(batch/'RESUMEN_LOTE.json',summary)
        extra = [campaign/'CONTRATO_CAMPANA.json',PREP_MANIFEST,PREP_MANIFEST.with_suffix('.sha256')]
        for job in selected[:len(statuses)]:
            directory = campaign/job['job_id']
            if directory.is_dir():
                extra += [p for p in directory.iterdir() if p.is_file()]
        write_manifest(batch/'MANIFEST_LOTE_SHA256.json',batch,extra)
        print(f"{summary['status']}: {summary['available_logical_cells']}/288 celdas disponibles.",flush=True)
        print('Evidencias: '+str(campaign),flush=True)
        return 0 if summary['status'] != 'STOPPED_FOR_AUDIT' else 1
    finally:
        lock.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare',action='store_true');mode.add_argument('--preflight',action='store_true');mode.add_argument('--run',action='store_true')
    parser.add_argument('--campaign',type=Path)
    parser.add_argument('--max-new-jobs',type=int,default=1)
    parser.add_argument('--max-seconds',type=int,default=600)
    parser.add_argument('--check-environment',action='store_true')
    args=parser.parse_args()
    require(args.run or (args.campaign is None and args.max_new_jobs == 1 and args.max_seconds == 600),
            '--campaign/limites requieren --run')
    if args.prepare:
        prepare();return 0
    if args.preflight:
        plan=preflight(check_environment=args.check_environment)
        print(json.dumps(dict(status='READ_ONLY_PREFLIGHT_PASS',logical_cells=288,imported_cells=48,new_fits=240,
            training_started=False,TensorFlow_imported='tensorflow' in sys.modules,limits=plan['limits']),ensure_ascii=True));return 0
    return execute(args.campaign,args.max_new_jobs,args.max_seconds)


if __name__=='__main__':
    raise SystemExit(main())
