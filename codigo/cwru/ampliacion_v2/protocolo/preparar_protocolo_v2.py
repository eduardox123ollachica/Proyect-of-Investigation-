"""Preflight y planes V2 desde metadatos; no abre matrices ni entrena.

Por defecto solo lee. --write crea roles, piloto, preflight y un sello nuevos
sin sobrescribir fuentes, particiones o resultados anteriores.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

from reglas_v2 import component_roles
import reglas_v2 as rules

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / 'articulo/ampliacion_cwru/metodologia/protocolo_v2_2026-10-25'
PARTS = ROOT / 'datos/cwru/particiones/ampliacion_v2/por_carga_2026-10-22'
COSTS = ROOT / 'despliegue_stm32/costos_mascaras_180mhz_2026-10-24'
DATA_CONTRACT = ROOT / 'articulo/ampliacion_cwru/metodologia/protocolo_v2_particiones_2026-10-22.json'
LOADS = (0, 1, 2, 3)
LABELS = {'N': 0, 'IR': 1, 'OR': 2}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def bounded(base, relative):
    path = (base / relative).resolve()
    require(path != base.resolve() and path.is_relative_to(base.resolve()), 'Ruta fuera del ambito')
    return path


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def check_sources():
    manifest = PARTS / 'MANIFEST_SHA256.csv'
    rows = read_csv(manifest)
    require(len(rows) == 13 and len({r['path'] for r in rows}) == 13, 'Sello de trece particiones esperado')
    inputs = [manifest]
    for row in rows:
        path = bounded(PARTS, row['path'])
        require(path.suffix.lower() in ('.json', '.csv', '.md', '.txt'), 'Solo metadatos en el preflight')
        require(sha(path) == row['sha256'], 'Particion modificada: ' + row['path'])
        inputs.append(path)
    require(any(r['path'] == 'indice_ventanas_fuente.csv' for r in rows), 'Indice fuera del sello')
    contract = read_json(DATA_CONTRACT)
    require(contract['partitions']['scheme'] == 'nested_leave_one_load_out'
            and contract['partitions']['outer_folds'] == 4
            and contract['partitions']['inner_folds_per_outer'] == 3
            and contract['mass_training_enabled'] is False
            and contract['outer_predictive_evaluation_enabled'] is False, 'Contrato de datos distinto')
    inputs.append(DATA_CONTRACT)
    closure_manifest = COSTS / 'MANIFEST_CIERRE_COSTOS_EXTRACCION_SHA256.json'
    closure_seal = closure_manifest.with_suffix('.sha256')
    digest, filename = closure_seal.read_text(encoding='ascii').split()
    require(filename == closure_manifest.name and sha(closure_manifest) == digest, 'Sello de costos distinto')
    closure_path = COSTS / 'CIERRE_COSTOS_EXTRACCION.json'
    entries = read_json(closure_manifest)
    require(entries.get(closure_path.relative_to(ROOT).as_posix()) == sha(closure_path), 'Cierre de costos fuera del sello')
    closure = read_json(closure_path)
    require(closure['status'] == 'EXTRACTION_COSTS_CONSOLIDATED_NOT_FULL_PIPELINE_COST'
            and closure['catalog_fnv1a'] == 'fa1040f2'
            and closure['selector_total_cost_ready'] is False
            and closure['full_pipeline_cost_ms'] is None
            and closure['DNN_cost_ms'] is None and closure['detector_cost_ms'] is None,
            'No sustituir tiempos desconocidos por cero o por costos de extraccion')
    inputs.extend((closure_path, closure_manifest, closure_seal))
    return inputs


def check_index():
    rows = read_csv(PARTS / 'indice_ventanas_fuente.csv')
    require(len(rows) == 1380 and [int(r['global_window_id']) for r in rows] == list(range(1380)),
            'Se requieren los 1380 IDs globales contiguos')
    records = {}
    for row in rows:
        load = int(row['load_hp'])
        require(load in LOADS and row['class'] in LABELS and int(row['label']) == LABELS[row['class']],
                'Carga o etiqueta distinta')
        require(row['de_key'] != row['fe_key'] and int(row['source_row']) >= 0, 'Canales o fila invalidos')
        require(row['source_split_v1'] in ('train', 'validation', 'test'), 'Origen historico invalido')
        feature = bounded(ROOT, row['source_features'])
        require(feature.suffix == '.npy' and feature.name == 'X_features_' + row['source_split_v1'] + '.npy',
                'Referencia de caracteristicas distinta; no se abre la matriz')
        signature = (row['file'], load, row['class'], int(row['label']), row['de_key'], row['fe_key'])
        key = str(row['file_id'])
        require(key not in records or records[key] == signature, 'Registro con metadatos inconsistentes')
        records[key] = signature
    require(len(records) == 28, 'Se requieren 28 MAT')
    for load in LOADS:
        require(Counter(s[2] for s in records.values() if s[1] == load) == Counter(N=1, IR=3, OR=3),
                'Composicion por carga distinta')
    return rows


def expected_load_roles(outer, inner=None):
    development = sorted(set(LOADS) - {outer})
    training = [load for load in development if load != inner]
    calibration = max(training)
    return dict(outer_load_hp=outer, inner_validation_load_hp=inner,
        development_loads=development, training_loads=training,
        dnn_fit_loads=training, int8_calibration_loads=training,
        detector_fit_normal_loads=[load for load in training if load != calibration],
        detector_calibration_normal_loads=[calibration], evaluation_loads=[outer if inner is None else inner],
        outer_excluded_loads=[outer], detector_refit_after_calibration=False)


def scope(rows, loads, normal_only=False):
    chosen = [r for r in rows if int(r['load_hp']) in loads and (not normal_only or r['class'] == 'N')]
    return dict(loads_hp=list(loads), normal_only=normal_only,
        file_ids=sorted({str(r['file_id']) for r in chosen}, key=int),
        global_window_ids=[int(r['global_window_id']) for r in chosen], windows=len(chosen),
        class_window_counts={name: sum(r['class'] == name for r in chosen) for name in LABELS})


def build_roles(rows):
    plans = []
    for outer in LOADS:
        development = sorted(set(LOADS) - {outer})
        for inner in [None, *development]:
            expected = expected_load_roles(outer, inner)
            require(component_roles(outer, inner) == expected, 'Helper y verificacion independiente no coinciden')
            components = dict(dnn_fit=scope(rows, expected['dnn_fit_loads']),
                int8_calibration=scope(rows, expected['int8_calibration_loads']),
                detector_fit=scope(rows, expected['detector_fit_normal_loads'], True),
                detector_calibration=scope(rows, expected['detector_calibration_normal_loads'], True),
                evaluation=scope(rows, expected['evaluation_loads']),
                outer_excluded=scope(rows, [outer]))
            file_sets = {name: set(value['file_ids']) for name, value in components.items()}
            eval_ids = file_sets['evaluation']
            excluded = file_sets['outer_excluded']
            for name in ('dnn_fit', 'int8_calibration', 'detector_fit', 'detector_calibration'):
                require(file_sets[name] and not file_sets[name] & (eval_ids | excluded), 'Fuga de evaluacion: ' + name)
            require(not file_sets['detector_fit'] & file_sets['detector_calibration'], 'Ajuste y calibracion detector solapados')
            require(file_sets['detector_fit'] | file_sets['detector_calibration'] <= file_sets['dnn_fit'],
                    'Roles del detector fuera del entrenamiento permitido')
            require(components['int8_calibration']['global_window_ids'] == components['dnn_fit']['global_window_ids'],
                    'Calibracion INT8 debe usar todo y solo el entrenamiento')
            plans.append(dict(stage='outer_refit' if inner is None else 'inner_fit', **expected, components=components,
                detector_calibration_may_belong_to_DNN_training=True,
                outer_excluded_means_excluded_from_fitting_not_from_final_evaluation=True))
    require(len(plans) == 16 and sum(r['stage'] == 'inner_fit' for r in plans) == 12, 'Cantidad de pliegues distinta')
    return dict(schema_version=1, status='COMPONENT_ROLES_PREPARED_NO_FIT', labels=LABELS,
        unique_windows=1380, records=28, outer_folds=4, inner_folds=12,
        detector_calibration_rule='highest_allowed_training_load_normal_record',
        detector_refit_after_calibration=False, physical_independence_claimed=False, plans=plans)


def build_pilot(rows, roles):
    plan = next(r for r in roles['plans'] if r['outer_load_hp'] == 3 and r['inner_validation_load_hp'] == 0)
    train = plan['components']['dnn_fit']
    validation = plan['components']['evaluation']
    require(train['loads_hp'] == [1, 2] and train['windows'] == 799 and validation['loads_hp'] == [0]
            and validation['windows'] == 175, 'Tamano de piloto distinto')
    ids = set(train['global_window_ids']) | set(validation['global_window_ids'])
    chosen = [r for r in rows if int(r['global_window_id']) in ids]
    paths = sorted({r['source_features'] for r in chosen})
    require(all(int(r['load_hp']) != 3 and r['source_split_v1'] in ('train', 'validation') for r in chosen)
            and not any(Path(p).name == 'X_features_test.npy' for p in paths), 'El piloto no puede abrir 3 HP')
    jobs = []
    for name, indices in (('F14', list(range(14))), ('T6', [0, 1, 2, 7, 8, 9])):
        jobs.append(dict(job_id=f'pilot_outer3_inner0_{name}_seed11', model='DNN', mask=name,
            feature_indices=indices, input_dimension=len(indices), seed=11,
            outer_load_hp=3, inner_validation_load_hp=0,
            train=train, validation=validation, representative_int8_calibration=train,
            authorized_feature_source_paths=paths,
            detector_fit_role_only=plan['components']['detector_fit'],
            detector_calibration_role_only=plan['components']['detector_calibration'],
            detector_fitted_in_runtime_pilot=False, fixed_epochs=50, early_stopping=False,
            batch_size=32, optimizer='Adam', learning_rate=0.005, L2=0.001,
            hidden_units=[20, 10], output_units=3, timeout_worker_seconds=300,
            runtime_and_finiteness_diagnostics_only=True, metrics_used_for_selection=False,
            reuse_as_predictive_V2_fit=False, model_conversions=['TFLite_FP32', 'TFLite_INT8']))
    return dict(schema_version=1, status='PILOT_PREPARED_NOT_EXECUTED', pilot_execution_enabled=False,
        mass_training_enabled=False, outer_predictive_evaluation_enabled=False,
        jobs=jobs, maximum_DNN_fits=2, maximum_model_conversions=4, sequential_workers=True,
        maximum_worker_wall_seconds_total=600, orchestrator_io_outside_worker_cap=True,
        runtime_budget_available=False, execute_TensorFlow_now=False,
        no_outer_3HP_predictive_arrays_authorized=True, classical_runtime_pilot_pending=True,
        pilot_normalization='fit scaler on authorized train; export parameters f32; explicit f32 subtract/divide for all inputs',
        no_training_or_conversion_executed=True)


def check_rule_agreement(protocol):
    selection = protocol['selection']
    require(selection['DNN_F1_loss_margin'] == rules.F1_LOSS_MARGIN
            and selection['detector_mean_BA_loss_margin'] == rules.DETECTOR_BA_LOSS_MARGIN
            and selection['detector_mean_specificity_IR_OR_loss_margin_each'] == rules.DETECTOR_COMPONENT_LOSS_MARGIN
            and selection['minimum_total_cost_reduction_vs_F14'] == rules.TARGET_TOTAL_REDUCTION
            and selection['numeric_comparison_tolerance'] == rules._EPS,
            'Documento y reglas ejecutables usan margenes distintos')
    require(protocol['dnn']['max_epochs'] == rules.MAX_EPOCHS
            and protocol['dnn']['inner_seeds'] == [11, 22, 33]
            and protocol['detector']['refit_after_threshold_calibration'] is False
            and protocol['pilot_execution_enabled'] is False,
            'Reglas de entrenamiento/calibracion distintas de la preparacion')
    closure = read_json(COSTS / 'CIERRE_COSTOS_EXTRACCION.json')
    catalogue = {r['name']: [i for i in range(14) if int(r['dnn_feature_mask'], 16) & (1 << i)]
                 for r in closure['candidates']}
    require(protocol['candidates'] == catalogue
            and protocol['candidate_order'] == [r['name'] for r in closure['candidates']],
            'Catalogo distinto de los kernels medidos')


def build_cost_plan(rows, roles, protocol):
    representatives = []
    for outer in LOADS:
        validation = min(set(LOADS) - {outer})
        role = next(r for r in roles['plans'] if r['outer_load_hp'] == outer
                    and r['inner_validation_load_hp'] == validation)
        train_ids = set(role['components']['dnn_fit']['global_window_ids'])
        train_rows = [r for r in rows if int(r['global_window_id']) in train_ids]
        by_record = {}
        for row in train_rows:
            by_record.setdefault(row['file_id'], []).append(row)
        require(len(by_record) == 14, 'Banco temporal debe tener catorce MAT de ajuste')
        bank = []
        for record in sorted(by_record, key=int):
            ordered = sorted(by_record[record], key=lambda r: int(r['window_index_in_file']))
            require(len(ordered) >= 2, 'Registro sin primera y ultima ventana distintas')
            bank.extend([int(ordered[0]['global_window_id']), int(ordered[-1]['global_window_id'])])
        require(len(set(bank)) == 28 and set(bank) <= train_ids, 'Banco temporal invalido')
        for name in protocol['candidate_order']:
            representatives.append(dict(
                representative_id=f'cost_outer{outer}_inner{validation}_{name}_seed11',
                outer_load_hp=outer, inner_validation_load_hp=validation,
                candidate=name, feature_indices=protocol['candidates'][name], seed=11,
                included_in_inner_search=True, training_global_window_ids=sorted(train_ids),
                benchmark_loads=role['training_loads'], benchmark_global_window_ids=bank,
                benchmark_file_ids=sorted(by_record, key=int), benchmark_windows=28, repetitions=3,
                detector_fit=role['components']['detector_fit'],
                detector_calibration=role['components']['detector_calibration'],
                cost_status='PENDING_NOT_MEASURED', mean_ms=None, lower_ms=None, upper_ms=None))
    require(len(representatives) == 32, 'Se requieren treinta y dos representantes')
    return dict(status='TOTAL_COST_CALIBRATION_PLANNED_NOT_EXECUTED', cost_scope='full_pipeline',
        representatives=representatives, DNN_fits_included_in_inner_288=True,
        observations_per_candidate_per_outer=84, timing_includes_UART=False,
        lower_is_observed_min=True, upper_is_observed_max=True,
        confidence_interval_or_WCET_claimed=False, all_components_pending=True)


def prepare():
    inputs = check_sources()
    protocol_path = OUT / 'protocolo_reglas.json'
    protocol = read_json(protocol_path)
    require(protocol['primary_cost_objective'] == 'total_pipeline'
            and protocol['mass_training_enabled'] is False
            and protocol['outer_predictive_evaluation_enabled'] is False
            and protocol['full_protocol_frozen'] is False, 'Guardias de protocolo incompatibles')
    check_rule_agreement(protocol)
    rows = check_index()
    roles = build_roles(rows)
    pilot = build_pilot(rows, roles)
    cost_plan = build_cost_plan(rows, roles, protocol)
    budget = dict(DNN_inner_max_fits=288, predefined_cost_representative_fits_included_in_inner=32,
        DNN_remaining_inner_fits_after_cost_representatives=256, DNN_outer_max_fits=120,
        DNN_max_fits_before_identical_artifact_reuse=408, DNN_FP32_plus_INT8_max_conversions=816,
        random_forest_inner_max_fits=288, random_forest_outer_max_fits=40, random_forest_total_max_fits=328,
        SVM_inner_max_fits=96, SVM_outer_max_fits=4, SVM_total_max_fits=100,
        detector_naive_inner_fits=96, detector_naive_outer_max_fits=12,
        detector_repeated_per_DNN_seed=False,
        engineering_pilot_additional_DNN_fits=2, engineering_pilot_additional_conversions=4,
        classical_runtime_pilot_pending=True, measured_runtime_budget_available=False,
        total_pipeline_cost_table_available=False, mass_training_authorized=False)
    checks = dict(partition_metadata_files_hash_verified=13, metadata_windows=1380, metadata_records=28,
        component_plans=16, inner_component_plans=12, outer_component_plans=4,
        helper_roles_match_independent_logic=True, fit_calibration_evaluation_record_guards_passed=True,
        protocol_margins_and_catalogue_match_executable_rules=True,
        total_cost_representatives=32, cost_benchmark_windows_per_representative=28,
        cost_benchmark_training_only_verified=True,
        detector_fit_and_calibration_records_disjoint=True, INT8_calibration_training_only=True,
        pilot_train_windows=799, pilot_validation_windows=175, pilot_masks=['F14', 'T6'],
        pilot_load3_exclusion_verified=True, source_NPY_NPZ_MAT_arrays_opened=False,
        predictive_feature_or_label_arrays_loaded=False, outer_metadata_read_for_scope_verification=True,
        cost_closure_manifest_and_JSON_hash_verified=True,
        cost_closure_source_resources_reverified=False, total_pipeline_cost_not_imputed=True,
        TensorFlow_imported=False, models_trained=0, conversions_executed=0,
        firmware_changed=False, serial_port_opened=False, manuscript_modified=False)
    preflight = dict(schema_version=1, status='PREFLIGHT_PASSED_NO_TRAINING', checks=checks,
        planning_budget=budget, protocol_sha256=sha(protocol_path),
        source_metadata_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in inputs},
        limitations=['Metadata disjointness does not prove independent physical bearings or motors.',
                     'This preflight does not audit a future training implementation that bypasses these guards.',
                     'The measured total-cost table and runtime budget remain unavailable; mass training stays disabled.'])
    return roles, pilot, cost_plan, preflight, inputs


def write_outputs(roles, pilot, cost_plan, preflight, inputs):
    names = ('roles_componentes.json', 'piloto_plan.json', 'calibracion_costo_total_plan.json', 'preflight.json')
    manifest = OUT / 'MANIFEST_PROTOCOLO_SHA256.json'
    seal = manifest.with_suffix('.sha256')
    require(not any((OUT / name).exists() for name in names) and not manifest.exists() and not seal.exists(),
            'No sobrescribir preparacion existente')
    sources = [Path(__file__), Path(__file__).with_name('reglas_v2.py'), Path(__file__).with_name('test_reglas_v2.py')]
    require(all(p.is_file() for p in sources), 'Falta helper o pruebas del contrato')
    require(any(p.is_file() and p.suffix.lower() == '.md' for p in OUT.iterdir()), 'Falta documento de decisiones')
    test_report = read_json(OUT / 'PRUEBAS_REGLAS.json')
    require(test_report['status'] == 'PASS' and test_report['failures'] == 0
            and test_report['errors'] == 0 and test_report['tests_run'] >= 25,
            'Falta informe aprobado de reglas deterministas')
    require(all(test_report['source_sha256'].get(p.relative_to(ROOT).as_posix()) == sha(p)
                for p in sources[1:]), 'Codigo de reglas/pruebas distinto del informe aprobado')
    preflight['checks']['deterministic_rule_tests_passed'] = test_report['tests_run']
    for name, value in zip(names, (roles, pilot, cost_plan, preflight)):
        with (OUT / name).open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
    paths = set(inputs + sources)
    paths.update(p for p in OUT.iterdir() if p.is_file() and p.suffix.lower() in ('.json', '.md')
                 and p.name != manifest.name)
    require(any(p.suffix.lower() == '.md' and p.parent == OUT for p in paths), 'Falta documento de decisiones')
    entries = {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(paths)}
    with manifest.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(entries, stream, indent=2)
        stream.write('\n')
    with seal.open('x', encoding='ascii') as stream:
        stream.write(sha(manifest) + '  ' + manifest.name + '\n')
    require(all(sha(bounded(ROOT, name)) == digest for name, digest in entries.items()), 'Recurso distinto al cerrar el sello')
    return len(entries)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    roles, pilot, cost_plan, preflight, inputs = prepare()
    if args.write:
        count = write_outputs(roles, pilot, cost_plan, preflight, inputs)
        print(json.dumps(dict(status=preflight['status'], resources_sealed=count,
            component_plans=16, pilot_jobs=2, models_trained=0, total_cost_ready=False)))
    else:
        print(json.dumps(dict(status=preflight['status'], checks=preflight['checks'],
            planning_budget=preflight['planning_budget'], component_plans=16,
            pilot_job_ids=[job['job_id'] for job in pilot['jobs']]), ensure_ascii=True, indent=2))
