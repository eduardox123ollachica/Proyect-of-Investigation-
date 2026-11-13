"""Sella la preparacion TotalV2 desde evidencias existentes, sin repetir ensayos.

Importar no escribe. La ejecucion crea exclusivamente un cierre y su manifiesto;
no compila, entrena, abre COM, programa la placa ni escribe el proyecto D:.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[4]
SOURCE_CODE = ROOT / 'codigo/cwru/ampliacion_v2/costo_total'
OUT = ROOT / 'despliegue_stm32/costo_total_representantes_v2_180mhz_2026-10-25'
METHOD = ROOT / 'articulo/ampliacion_cwru/metodologia/calibracion_costo_total_v2_2026-10-25/ALCANCE_Y_DISENO.md'
CLOSURE = OUT / 'CIERRE_PREPARACION_TOTAL.json'
MANIFEST = OUT / 'MANIFEST_PREPARACION_TOTAL_SHA256.json'
SEAL = MANIFEST.with_suffix('.sha256')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def load_preparer():
    spec = importlib.util.spec_from_file_location('preparar_total_cierre_solo_lectura', SOURCE_CODE / 'preparar_total.py')
    require(spec is not None and spec.loader is not None, 'Preparador no importable')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_count(path):
    tree = ast.parse(Path(path).read_text(encoding='utf-8-sig'))
    return sum(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith('test_')
               for node in ast.walk(tree))


def include_path(path, prepared):
    """Excluir salidas posteriores sin excluir los quince ELF del respaldo."""
    resolved = path.resolve()
    require(resolved.is_relative_to(ROOT.resolve()), 'Archivo fuera del proyecto: ' + str(path))
    parts = path.relative_to(ROOT).parts
    if '__pycache__' in parts or path.suffix in ('.pyc', '.pyo'):
        return False
    if path.is_relative_to(OUT):
        relative = path.relative_to(OUT)
        if relative.parts[0] in ('workspace_aislado', 'intentos', 'resultados', 'logs'):
            return False
        if relative.parts[:2] == ('Validacion_PC', 'resultados'):
            return False
        if path.name.startswith(('CUBEIDE_TOTAL_INTENTO_', 'COMPILACION_TOTAL')) or path.suffix == '.log':
            return False
        project = Path(prepared['project'])
        if path.is_relative_to(project):
            inside = path.relative_to(project)
            configuration_names = set(prepared['existing_elf_sha256']) | {prepared['configuration']}
            if inside.parts[0] in configuration_names | {'.git', '.metadata'}:
                return False
    return path not in (MANIFEST, SEAL, CLOSURE)


def main():
    started = time.perf_counter()
    require(not any(path.exists() for path in (CLOSURE, MANIFEST, SEAL)), 'No sobrescribir cierre o sello existente')
    require(not list(OUT.glob('COMPILACION_TOTAL*.json')) and not list(OUT.glob('CUBEIDE_TOTAL_INTENTO_*.log')),
            'La compilacion ya tiene evidencias; este cierre debe anteceder al primer build')
    prep = read(OUT / 'PREPARACION.json')
    syntax = read(OUT / 'SINTAXIS_TOTAL.json')
    prep_tests = read(OUT / 'PRUEBAS_PREPARADOR_TOTAL.json')
    pc = read(OUT / 'VERIFICACIONES_PC_TOTAL.json')
    source_audit = read(OUT / 'AUDITORIA_FUENTES_ANTES_BUILD.json')
    require(prep['status'] == 'PREPARED_NOT_BUILT_NOT_ARM_VALIDATED'
            and prep['configuration'] == 'Release_180MHz_TotalV2'
            and prep['previous_fifteen_configurations_preserved'], 'Estado de preparacion incompatible')
    require(all(prep[field] is False for field in ('hardware_flashed', 'serial_port_opened', 'd_files_modified', 'selector_costs_enabled'))
            and prep['physical_validation_pending'] is True, 'Preparacion con acciones o costos no autorizados')

    print('Verificando inventario original D:, copia, quince ELF y recursos; solo lectura.', flush=True)
    preparer = load_preparer()
    preparer.verify_prepared(prep)
    origin_names = preparer.configs(preparer.SOURCE / '.cproject')[2]
    copy_names = preparer.configs(preparer.PROJECT / '.cproject')[2]
    require(len(origin_names) == len(prep['existing_elf_sha256']) == 15
            and len(copy_names) == 16 and preparer.same_configs(preparer.PROJECT / '.cproject'),
            'Recuento o contenido de configuraciones distinto')
    require(len(prep['source_inventory_before']) == 1867, 'Inventario original distinto de1867 archivos')
    require(prep_tests['status'] == 'PASS' and prep_tests['returncode'] == 0
            and prep_tests['simulation_only'] is True and 'Ran 5 tests' in prep_tests['output']
            and prep_tests['output'].rstrip().endswith('OK'), 'Informe de cinco pruebas del preparador incompatible')
    counts = {name: test_count(SOURCE_CODE / name) for name in ('test_preparar_total.py', 'test_validar_total.py')}
    require(counts == {'test_preparar_total.py': 5, 'test_validar_total.py': 21}, 'Fuentes de pruebas distintas de5+21')
    tests = pc['unittest']
    require(pc['status'] == 'PC_CHECKS_PASSED_NOT_ARM_VALIDATED' and tests['tests'] == sum(counts.values()) == 26
            and tests['failures'] == tests['errors'] == 0, 'Informe de26 pruebas incompatible')
    require(syntax['status'] == 'PASS_SYNTAX_AND_CONTROL_PREPROCESSING' and syntax['simulation_only'] is True
            and len(syntax['syntax']) == 6
            and all(row['returncode'] == 0 and row['output'] == '' for row in syntax['syntax'].values())
            and len(syntax['old_preprocessed']) == 4
            and all(row['identical'] is True and row['before_sha256'] == row['after_sha256']
                    for row in syntax['old_preprocessed'].values()), 'Sintaxis o preprocesados incompatibles')
    require(all(syntax[field] is False for field in ('serial_port_opened', 'hardware_flashed', 'd_files_modified')),
            'Sintaxis con acciones fisicas declaradas')
    bank_check, smoke = pc['bank_preflight'], pc['TFLite_backend_smoke']
    require(bank_check['status'] == 'PASS' and bank_check['resources'] == 39 and bank_check['representatives'] == 32
            and bank_check['unique_windows'] == 42 and bank_check['model_window_pairs'] == bank_check['exact_PC_replays'] == 896
            and bank_check['TFLite_invocations'] == 0, 'Preflight PC incompatible')
    require(smoke['status'] == 'PASS' and smoke['representatives'] == smoke['TFLite_invocations'] == smoke['exact_INT8_outputs'] == 32,
            'Smoke TFLite incompatible')
    require(pc['new_training_fits'] == 0 and all(pc[field] is False for field in (
        'serial_port_opened', 'hardware_flashed', 'D_files_modified', 'new_physical_timings_obtained', 'selector_costs_enabled')),
        'Comprobaciones PC con acciones o costos no previstos')
    require(source_audit['status'] == 'SOURCE_AUDIT_AND_ARM_SYNTAX_PASS_NOT_BUILT_OR_PHYSICALLY_VALIDATED'
            and len(source_audit['checks']) == 4 and all(row['exit_code'] == 0 for row in source_audit['checks']),
            'Auditoria preliminar de fuentes incompatible')
    for row in source_audit['files']:
        require(sha(row['path']) == row['sha256'], 'Fuente de auditoria preliminar cambio: ' + row['path'])

    resource_entries = read(preparer.RESOURCES / 'MANIFEST_RECURSOS_SHA256.json')
    catalog = read(preparer.RESOURCES / 'CATALOGO_TOTAL.json')
    bank = read(preparer.RESOURCES / 'BANCO_TOTAL.json')
    require(resource_entries == prep['resources'] and len(resource_entries) == 39
            and catalog['catalog_sha256'] == pc['catalog_sha256'] and catalog['catalog_fnv'] == pc['catalog_fnv']
            and catalog['count'] == 32 and bank['total_planned_RUN'] == 2688
            and bank['total_model_window_pairs'] == 896, 'Catalogo o banco inconsistente con comprobaciones')
    require(METHOD.is_file() and (OUT / 'LEEME_COSTO_TOTAL_V2.md').is_file(), 'Falta documentacion final')
    paths = sorted(set(path for folder in (OUT, SOURCE_CODE) for path in folder.rglob('*')
                       if path.is_file() and include_path(path, prep)) | {Path(__file__).resolve(), METHOD})
    require(all(path.resolve().is_relative_to(ROOT.resolve()) for path in paths), 'Sello incluiria fuente externa')
    entries = {path.relative_to(ROOT).as_posix(): sha(path) for path in paths}
    require(len(entries) == len(paths), 'Rutas duplicadas en sello')
    report = dict(
        schema_version=1, status='PREPARED_VERIFIED_PENDING_USER_BUILD', date_local='2026-10-25', timezone='America/Lima',
        configuration=prep['configuration'], original_source_inventory_verified=1867,
        original_configurations_verified=15, isolated_copy_configurations=16,
        prior_configurations_semantically_unchanged=True, original_ELF_hashes_verified=15,
        D_files_modified=False, hardware_flashed=False, serial_port_opened=False,
        training_fits_executed=0, physical_RUN_executed=0, build_executed=False,
        tests=dict(total=26, preparer=5, client=21, failures=0, errors=0, rerun_by_closure=False),
        syntax=dict(units=6, failures=0, control_preprocessing_comparisons=4, all_controls_identical=True, rerun_by_closure=False),
        PC_validation=dict(resource_files=39, representatives=32, unique_RAW_windows=42,
                           exact_replays=896, TFLite_smoke_invocations=32, exact_TFLite_smoke_outputs=32,
                           rerun_by_closure=False),
        original_source_audit=dict(syntax_checks=4, source_hashes_verified=10),
        catalog_sha256=catalog['catalog_sha256'], catalog_fnv=catalog['catalog_fnv'],
        resource_manifest_sha256=sha(preparer.RESOURCES / 'MANIFEST_RECURSOS_SHA256.json'),
        planned_physical=dict(model_window_pairs=896, primary_RUN=2688, primary_RUN_per_representative=84,
                              diagnostic_PROFILE=896, independent_BASE=42, gate_primary_RUN=3, gate_PROFILE=1,
                              gate_rejections=17, new_ARM_feature_windows=14),
        selector_costs_enabled=False, physical_validation_pending=True,
        manifest_entries_including_this_closure=len(entries)+1,
        manifest_path=MANIFEST.relative_to(ROOT).as_posix(),
        manifest_scope='Rutas relativas a la raiz: todos los archivos preparados OUT, codigo costo_total, este auditor y metodo; sin rutas D directas.',
        excluded_future_outputs=['workspace_aislado', 'intentos', 'resultados', 'Validacion_PC/resultados',
                                 'logs', 'CUBEIDE_TOTAL_INTENTO_*.log', 'COMPILACION_TOTAL*.json',
                                 'directorios de compilacion dentro de la copia aislada', '__pycache__', '*.pyc', '*.pyo'],
        integrity_check_seconds_before_writing=time.perf_counter()-started,
        limitations=[
            'Los informes existentes se auditan sin repetir sus pruebas, inferencias ni preflight.',
            'Sintaxis y preprocesado no sustituyen compilacion, enlace y auditoria de ELF/mapa.',
            'Memoria estatica y arena real de los32 modelos siguen pendientes de build y placa.',
            'No hay medidas nuevas STM32 ni costos habilitados para el selector.',
            'Los SHA256 de modelo/catalogo que publica la placa son declarados; vincular con ELF y programador.',
            'La Flash del banco con32 modelos no equivale a memoria del despliegue final con un modelo.',
            'La reserva stack4096 B no es una nueva medicion de stack.',
            'La evaluacion predictiva continua retrospectiva; no se acredita independencia de motor o rodamiento.'
        ],
        next_step='El usuario inicia build aislado con preparar_total.py build --timeout 1800; revisar log/ELF/map/controles antes de instalar y medir gate.'
    )
    for path in paths:
        require(sha(path) == entries[path.relative_to(ROOT).as_posix()], 'Archivo cambio durante cierre: ' + str(path))
    write(CLOSURE, report)
    entries[CLOSURE.relative_to(ROOT).as_posix()] = sha(CLOSURE)
    write(MANIFEST, dict(sorted(entries.items())))
    digest = sha(MANIFEST)
    with SEAL.open('x', encoding='ascii', newline='\n') as stream:
        stream.write(digest + '  ' + MANIFEST.name + '\n')
    require(sha(MANIFEST) == SEAL.read_text(encoding='ascii').split()[0], 'Sello final distinto')
    for relative, expected in read(MANIFEST).items():
        require(sha(ROOT / relative) == expected, 'Recurso distinto tras escribir sello: ' + relative)
    print(json.dumps(dict(status=report['status'], resources=len(entries), manifest_sha256=digest,
                          closure_sha256=sha(CLOSURE), elapsed_seconds=time.perf_counter()-started), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
