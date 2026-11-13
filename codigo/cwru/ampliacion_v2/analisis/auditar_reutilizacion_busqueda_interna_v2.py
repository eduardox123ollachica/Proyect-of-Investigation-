"""Audita 48 ajustes internos reutilizables sin entrenar ni invocar TFLite.

Conserva las firmas REP/SUPP y contrasta cada ajuste con su papel en la matriz
interna fija. Los alias semanticos no sustituyen las firmas originales.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
PREP = ROOT/'articulo/ampliacion_cwru/metodologia/busqueda_interna_v2_2026-10-26'
REPORT = PREP/'AUDITORIA_REUTILIZACION.json'
REP_SOURCE = ROOT/'codigo/cwru/ampliacion_v2/representantes'
SUP_SOURCE = ROOT/'codigo/cwru/ampliacion_v2/suplemento'
REP_RUN = ROOT/'resultados/cwru/ampliacion_v2/representantes_v2_2026-10-25/20261025T171847_957295Z'
SUP_RUN = ROOT/'resultados/cwru/ampliacion_v2/suplemento_preprocesamiento_normalidad_2026-10-25/20261025T215652_507777Z'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')).hexdigest()


def main():
    started = time.perf_counter()
    require(not REPORT.exists(), 'No sobrescribir auditoria')
    sys.path[:0] = [str(SUP_SOURCE), str(REP_SOURCE), str(ROOT/'codigo/cwru/ampliacion_v2/protocolo')]
    import numpy as np
    import ejecutar_representantes as rep
    import ejecutar_suplemento as sup
    from datos_representantes import load_verified_context
    from reglas_v2 import component_roles, weighted_p95
    from numerica_representantes import detector_metrics
    from preprocesamiento import apply, fit_params

    protocol, _, metadata, _ = load_verified_context()
    by_id = {int(r['global_window_id']):r for r in metadata}
    old_plan, _ = rep.preflight()
    supp_plan = sup.preflight()
    require(old_plan['tested_versions'] == supp_plan['tested_versions']
            and old_plan['dnn'] == supp_plan['dnn'] == protocol['dnn'], 'Entorno o configuracion de origen distintos')
    require(protocol['dnn']['max_epochs'] == 50
            and protocol['dnn']['early_stopping'] == dict(patience=10,minimum_improvement=1e-12,restore_best_weights=True),
            'No cambiar ES establecido por maximo de50 epocas fijas')
    require(old_plan['worker_environment'] == supp_plan['environment'], 'Entornos de worker distintos')
    closure = ROOT/'articulo/ampliacion_cwru/metodologia/cierre_suplemento_completo_2026-10-25/MANIFEST_CIERRE_SHA256.json'
    rep.verify_manifest(closure)
    # El modulo P0 devuelve copia exacta incluso para signos/cero. La comprobacion
    # guardada de cada ajuste contrasta tambien sus entradas reales cuantizadas.
    sample = np.asarray([[0.,-1.,2.,3.,4.,5.,6.,7.,8.,9.,10.,11.,12.,13.]],dtype=np.float32)
    require(np.array_equal(apply(sample,fit_params(sample,'P0')).view(np.uint32),sample.view(np.uint32)),
            'P0 no es identidad float32')
    core = REP_SOURCE/'worker_representantes.py'
    require('from worker_representantes import configure_tf,fit_network,export_models,evaluate_lite'
            in (SUP_SOURCE/'worker_suplemento.py').read_text(encoding='utf-8'), 'Suplemento no reutiliza el nucleo original')
    require(old_plan['source_sha256'][core.relative_to(ROOT).as_posix()] == sha(core), 'Nucleo original alterado')
    require(supp_plan['source_training_core'] == 'unchanged_sealed_representatives_worker_functions',
            'Procedencia del nucleo suplementario distinta')

    links = []
    keys = set()
    def add(family, job, folder, result, signature):
        outer, inner, candidate, seed = (job[k] for k in
            ('outer_load_hp','inner_validation_load_hp','candidate','seed'))
        key = outer,inner,candidate,seed
        require(key not in keys and outer in range(4) and inner in set(range(4))-{outer}
                and candidate in protocol['candidate_order'] and seed in protocol['dnn']['inner_seeds'],
                'Celda fuera de la matriz o duplicada')
        roles = component_roles(outer,inner)
        train_ids = [int(r['global_window_id']) for r in metadata if int(r['load_hp']) in roles['training_loads']]
        val_ids = [int(r['global_window_id']) for r in metadata if int(r['load_hp']) == inner]
        require(job['roles'] == roles and job['training_global_window_ids'] == train_ids
                and job['validation_global_window_ids'] == val_ids
                and job['feature_indices'] == protocol['candidates'][candidate], 'Semantica de entrada distinta')
        for name, filename in (('FP32','modelo_fp32.tflite'),('INT8','modelo_int8.tflite')):
            require(result['model_sha256'][name] == sha(folder/filename), 'Modelo cambiado')
        semantic = dict(outer_load_hp=outer,inner_validation_load_hp=inner,candidate=candidate,seed=seed,
            roles=roles,training_global_window_ids=train_ids,validation_global_window_ids=val_ids,
            feature_indices=job['feature_indices'],preprocessing='P0_identity_raw_features_float32',
            dnn=protocol['dnn'],numeric_contract=protocol['numeric_contract'],
            quantization=protocol['quantization'],versions=old_plan['tested_versions'],
            training_core_sha256=sha(core),worker_environment=old_plan['worker_environment'])
        links.append(dict(source_family=family,source_job_id=job['job_id'],
            main_cell=dict(outer_load_hp=outer,inner_validation_load_hp=inner,candidate=candidate,seed=seed),
            source_directory=folder.relative_to(ROOT).as_posix(),
            source_result_sha256=sha(folder/'resultado.json'),
            source_job_manifest_sha256=sha(folder/'MANIFEST_JOB_SHA256.json'),
            source_fit_signature=signature,source_signature_verified=True,
            semantic_equivalence_signature=digest(semantic),semantic_contract=semantic,
            model_sha256=result['model_sha256'],versions=result['versions'],
            epochs_executed=result['epochs_executed'],best_epoch=result['best_epoch'],
            weights_restored=result['weights_restored'],new_training_fits=0,
            original_signature_rewritten=False))
        keys.add(key)

    for job in old_plan['jobs']:
        signature = rep.fit_signature(job,old_plan['source_sha256'],old_plan['tested_versions'],protocol)
        require(signature == old_plan['fit_signatures'][job['job_id']], 'Firma REP no reproduce')
        folder = REP_RUN/job['job_id']
        result = rep.verify_job(folder,job,signature)
        add('REP',job,folder,result,signature)
    for job in supp_plan['jobs']:
        if job['mode'] != 'P0' or job['job_id'] in supp_plan['imported_references']:
            continue
        signature = sup.signature(job,supp_plan['source_sha256'],supp_plan['tested_versions'],protocol)
        require(signature == supp_plan['fit_signatures'][job['job_id']], 'Firma SUPP no reproduce')
        folder = SUP_RUN/job['job_id']
        result = sup.verify_new_job(folder,job,supp_plan)
        require(result['mode'] == 'P0', 'No reutilizar P1 como flujo principal')
        add('SUPP_P0',job,folder,result,signature)
    require(len(links) == 48 and Counter(r['source_family'] for r in links) == Counter(REP=32,SUPP_P0=16),
            'Inventario DNN incompleto')

    def parameter_payload(path):
        h = hashlib.sha256()
        with np.load(path,allow_pickle=False) as saved:
            for key in sorted(saved.files):
                arr = saved[key]
                h.update(key.encode());h.update(str(arr.dtype).encode());h.update(str(arr.shape).encode());h.update(arr.tobytes())
        return h.hexdigest()
    parameter_hashes = {parameter_payload(REP_RUN/j['job_id']/'detector_parametros.npz') for j in old_plan['jobs']}
    require(len(parameter_hashes) == 21, 'Parametros detector originales distintos de inventario previsto')
    detector_links = []
    for inner in (1,2):
        folder = SUP_RUN/'detector'/f'detector_o3_i{inner}_CONTROL12'
        result, provenance = read(folder/'resultado.json'), read(folder/'procedencia.json')
        require(result['status'] == 'PASS' and not result['refit_after_calibration']
                and result['indices'] == [i for i in range(14) if i not in (4,11)]
                and result['source_parameters_sha256'] == sha(folder/'parametros.npz'), 'Control12 distinto')
        with np.load(folder/'parametros.npz',allow_pickle=False) as saved:
            fit_ids, cal_ids = saved['fit_ids'].tolist(),saved['calibration_ids'].tolist()
            threshold = np.float32(saved['threshold32'].item())
            require(fit_ids == provenance['fit_ids'] and cal_ids == provenance['calibration_ids']
                    and set(fit_ids).isdisjoint(cal_ids)
                    and all(by_id[k]['class'] == 'N' and int(by_id[k]['load_hp']) not in (3,inner)
                            for k in fit_ids+cal_ids), 'Roles CONTROL12 invalidos')
        with np.load(folder/'calibracion.npz',allow_pickle=False) as saved:
            require(saved['global_window_ids'].tolist() == cal_ids
                    and weighted_p95(saved['scores'],[by_id[k]['file_id'] for k in cal_ids]).tobytes() == threshold.tobytes(),
                    'P95 CONTROL12 distinto')
        with np.load(folder/'validacion.npz',allow_pickle=False) as saved:
            ids = saved['global_window_ids'].tolist()
            expected_y = np.asarray([int(by_id[k]['label']) for k in ids])
            require(all(int(by_id[k]['load_hp']) == inner for k in ids)
                    and np.array_equal(saved['y_true'],expected_y)
                    and np.array_equal(saved['rejected'],saved['scores'] > threshold)
                    and detector_metrics(expected_y,saved['rejected'],[by_id[k]['file_id'] for k in ids]) == result['metrics'],
                    'Validacion CONTROL12 distinta')
        parameter_hashes.add(parameter_payload(folder/'parametros.npz'))
        detector_links.append(dict(source_directory=folder.relative_to(ROOT).as_posix(),outer_load_hp=3,
            inner_validation_load_hp=inner,aliases=['F14','NF12'],indices=result['indices'],
            source_result_sha256=sha(folder/'resultado.json'),source_parameters_sha256=sha(folder/'parametros.npz'),
            semantic_equivalence='Raw float32 CONTROL12 equals both F14 and NF12 detector indices; dominant frequencies excluded',
            new_detector_fits=0))
    require(len(parameter_hashes) == 23, 'Conteo de parametros reutilizables distinto')
    require('tensorflow' not in sys.modules, 'La auditoria no debe importar TensorFlow')
    script_paths = [Path(__file__).resolve(),core,SUP_SOURCE/'worker_suplemento.py',
        SUP_SOURCE/'preprocesamiento.py',REP_SOURCE/'ejecutar_representantes.py',
        SUP_SOURCE/'ejecutar_suplemento.py',SUP_SOURCE/'contrato_datos.py',REP_SOURCE/'datos_representantes.py']
    report = dict(status='PASS_48_INNER_DNN_FITS_REUSABLE_WITH_ORIGINAL_SIGNATURES',
        date_local='2026-10-26',timezone='America/Lima',planned_inner_DNN_cells=288,
        reusable_DNN_fits=48,reusable_representative_fits=32,reusable_supplement_P0_fits=16,
        new_DNN_fits_required=240,main_seeds=[11,22,33],links=links,
        available_DNN_rows_by_outer={str(i):sum(r['main_cell']['outer_load_hp'] == i for r in links) for i in range(4)},
        required_DNN_rows_per_outer=72,required_detector_rows_per_outer=24,
        source_tested_versions=old_plan['tested_versions'],live_worker_versions_checked=False,
        original_source_signatures_preserved=True,semantic_signature_is_not_original_source_signature=True,
        both_source_families_use_same_frozen_training_core=True,P1_reused=False,MOM4_reused=False,
        early_stopping=protocol['dnn']['early_stopping'],maximum_epochs=50,
        detector_inventory=dict(logical_rows_planned=96,logical_rows_available=36,logical_rows_remaining=60,
            source_representative_rows=32,extra_CONTROL12_alias_rows=4,
            distinct_training_parameter_scopes_planned=42,parameter_payloads_available=23,
            distinct_training_parameter_scopes_remaining=19,
            derivation='Six possible two-load training pairs times seven unique detector index sets; F14/NF12 alias.',
            extra_source_links=detector_links,counts_not_independent_experimental_repetitions=True),
        new_training_fits=0,new_TFLite_invocations=0,new_detector_fit_calls=0,
        source_feature_matrices_read_for_SUPP_preprocessing_checks=True,
        outer_feature_rows_materialized=False,raw_or_MAT_opened=False,
        TensorFlow_imported=False,hardware_connected=False,source_files_modified=False,
        source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in script_paths},
        upstream_manifests_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in
            (rep.PREP/'MANIFEST_PREPARACION_SHA256.json',sup.PREP/'MANIFEST_PREPARACION_SHA256.json',closure)},
        selection_executed=False,outer_evaluation_executed=False,
        limitations=['All data were explored previously; this reuse does not create prospective evidence.',
            'Original signatures authenticate old execution wrappers; semantic aliases explicitly map to the fixed main matrix.',
            'Detector inventory is separate from DNN execution and does not train missing detector scopes.',
            'Current worker package versions must be checked by execution preflight before training.'],
        seconds=time.perf_counter()-started)
    PREP.mkdir(parents=True,exist_ok=True)
    with REPORT.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(report,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({k:report[k] for k in ('status','reusable_DNN_fits','new_DNN_fits_required',
        'available_DNN_rows_by_outer','detector_inventory','seconds')},ensure_ascii=True))


if __name__ == '__main__':
    main()
