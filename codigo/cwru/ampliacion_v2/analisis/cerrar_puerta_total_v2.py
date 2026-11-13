"""Cierra la puerta TotalV2 auditada sin repetir medidas ni inferencias.

Importar no escribe. La ejecución crea exclusivamente un cierre y su sello.
No modifica la evidencia original, los clientes, los sellos anteriores ni D:.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / 'despliegue_stm32/costo_total_representantes_v2_180mhz_2026-10-25'
GATE = OUT / 'Validacion_PC/resultados/20261026T040903_561072Z'
RESOURCES = OUT / 'recursos'
CLIENT = OUT / 'Validacion_PC/validar_total.py'
SOURCE_CLIENT = ROOT / 'codigo/cwru/ampliacion_v2/costo_total/validar_total.py'
NUMERIC_SOURCE = Path(__file__).with_name('auditar_numerica_gate_total_v2.py')
METHOD = ROOT / 'articulo/ampliacion_cwru/metodologia/calibracion_costo_total_v2_2026-10-25/VALIDACION_PUERTA_TOTAL_V2.md'
GUIDE = OUT / 'CAMPANA_COMPLETA_TOTAL_V2.md'
CLOSURE = OUT / 'CIERRE_PUERTA_TOTAL_V2.json'
MANIFEST = OUT / 'MANIFEST_PUERTA_TOTAL_V2_SHA256.json'
SEAL = MANIFEST.with_suffix('.sha256')
ELF_SHA = '85e8d7e2d1848833117fd4704493710cda0058fecedef3b379a80e9560171b1c'
GATE_MANIFEST_SHA = 'e64a86c398bcee2cffd297112bfba580b9926b137d4228af8002f323d90aa0e6'
BUILD_MANIFEST_SHA = 'ed8b770a98833175198d454add321124c2e645f408bf91ef8710fc7574ac5a7e'
CONFIG = 'Release_180MHz_TotalV2'
PROJECT_NAME = 'Prueba_2_CWRU_Extraccion'


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


def manifest_header(path, expected=None):
    digest, name = path.with_suffix('.sha256').read_text(encoding='ascii').split()
    require(name == path.name and sha(path) == digest, 'Sello inválido: ' + str(path))
    require(expected is None or digest == expected, 'No es el sello revisado: ' + str(path))
    entries = read(path)
    require(isinstance(entries, dict) and entries, 'Manifiesto vacío')
    return digest, entries


def verify_local_manifest(folder, name, count, expected=None):
    path = folder / name
    digest, entries = manifest_header(path, expected)
    require(len(entries) == count, 'Número de recursos distinto: ' + str(path))
    for relative, declared in entries.items():
        target = (folder / relative).resolve()
        require(target.is_relative_to(folder.resolve()) and target.is_file() and sha(target) == declared,
                'Recurso local modificado: ' + relative)
    return digest, entries


def load_client():
    spec = importlib.util.spec_from_file_location('total_gate_closure_readonly', CLIENT)
    require(spec is not None and spec.loader is not None, 'Cliente no importable')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    started = time.perf_counter()
    require(not any(path.exists() for path in (CLOSURE, MANIFEST, SEAL)), 'No sobrescribir cierre o sello')
    gate_digest, gate_entries = verify_local_manifest(GATE, 'MANIFEST_EVIDENCIAS_SHA256.json', 12, GATE_MANIFEST_SHA)
    resource_digest, resource_entries = verify_local_manifest(RESOURCES, 'MANIFEST_RECURSOS_SHA256.json', 39)
    trace = read(OUT / 'AUDITORIA_TRAZA_GATE_TOTAL.json')
    numeric = read(OUT / 'AUDITORIA_NUMERICA_GATE_TOTAL.json')
    estimate = read(OUT / 'ESTIMACION_DURACION_CAMPANA_TOTAL.json')
    report = read(GATE / 'RESULTADO_TOTAL.json')
    plan = read(GATE / 'PLAN_EJECUTADO.json')
    require(trace['status'] == 'PASS_GATE_TRACE_REPLAYED_NO_HARDWARE'
            and numeric['status'] == 'PASS_NUMERIC_TIMING_GATE_AUDIT_NOT_TOTAL_COST_VALIDATED'
            and trace['gate_manifest_sha256'] == numeric['gate_manifest_sha256'] == gate_digest,
            'Auditorías de otra puerta o no aprobadas')
    require(trace['trace_rows'] == 450 and trace['transactions'] == 225 and trace['measurement_rows'] == 24
            and trace['completed_primary_RUN'] == numeric['primary_RUN'] == 3
            and trace['completed_PROFILE'] == numeric['PROFILE'] == 1
            and trace['completed_BASE'] == numeric['BASE'] == 1
            and trace['negative_rejections'] == 17 and trace['valid_extra_RUN'] == 0,
            'Conteos auditados incompatibles')
    require(trace['RAW']['global_id'] == numeric['global_window_id'] == 568
            and trace['RAW']['values'] == 8192 and trace['RAW']['frames'] == 171
            and trace['RAW']['bit_exact_to_authorized_bank'] is True
            and trace['all_sequences_timestamps_and_TX_RX_pairs_valid'] is True
            and trace['pages_bind_run_id_model_checksum'] is True
            and trace['pages_bit_identical_across_three_RUN_and_PROFILE'] is True
            and trace['three_invalid_DATA_frames_atomic_at_count4'] is True
            and trace['final_run_counter'] == 4 and trace['final_result_valid'] is False,
            'La auditoría de trazas no establece la puerta prevista')
    require(sha(GATE / 'TRAZA_UART.jsonl') == trace['trace_sha256']
            and sha(GATE / 'MEDICIONES.jsonl') == trace['measurements_sha256'], 'Traza o mediciones cambiaron')
    for name, declared in numeric['data_files_sha256'].items():
        require(Path(name).name == name and sha(GATE / name) == declared, 'Fuente numérica modificada: ' + name)
    require(sha(NUMERIC_SOURCE) == numeric['script_sha256'], 'Auditor numérico modificado')
    require(numeric['same_INT8_input_output_exact'] is True
            and numeric['float32_ARM_feature_replay_all_intermediates_exact'] is True
            and numeric['only_first_model_runtime_allocation_verified'] is True
            and numeric['arena_used_bytes'] == trace['SELECT']['arena_used_bytes'] == 1188
            and numeric['arena_capacity_bytes'] == 16384
            and numeric['PC_cache_end_to_end_bit_exact_claimed'] is False
            and numeric['total_cost_validated'] is False and numeric['selector_enabled'] is False,
            'Alcance numérico de la puerta incompatible')
    require(numeric['primary_statistics']['count'] == 3
            and math.isclose(numeric['primary_statistics']['mean_ms'], 84.2701962962963, rel_tol=0, abs_tol=1e-12)
            and report['timing_statistics'][0]['mean_ms'] == numeric['primary_statistics']['mean_ms']
            and numeric['profile']['excluded_from_primary_statistics'] is True,
            'Estadísticas principales o exclusión del perfil incompatibles')
    require(estimate['status'] == 'RUNTIME_PLANNING_ESTIMATE_NOT_MEASURED_FULL_CAMPAIGN'
            and estimate['recommended_max_seconds'] == 1800
            and estimate['recommended_user_expectation_minutes'] == [15, 25]
            and estimate['selector_costs_enabled'] is False
            and estimate['new_physical_RUN'] == estimate['TFLite_invocations'] == 0,
            'Estimación no corresponde al presupuesto previsto')
    for name, declared in estimate['sources_sha256'].items():
        require(Path(name).name == name and sha(GATE / name) == declared, 'Fuente de estimación modificada')

    require(sha(CLIENT) == sha(SOURCE_CLIENT) == sha(GATE / 'validar_total_ejecutado.py') == report['client_sha256'],
            'Cliente actual distinto al de la puerta')
    client = load_client()
    catalog, bank, windows, _ = client.load_resources(RESOURCES)
    client.verify_gate(GATE, catalog, ELF_SHA, resource_digest)
    schedule = client.schedule(bank, True)
    require(len(schedule) == 42 and sum(len(ids) for _, ids in schedule) == 896,
            'Calendario de campaña completa incompatible')
    require(plan['mode'] == 'gate' and plan['schedule'] == [[568, [0]]]
            and report['resource_manifest_sha256'] == plan['resource_manifest_sha256'] == resource_digest,
            'Plan de puerta o identidad de recursos incompatibles')

    upstream_manifest = OUT / 'MANIFEST_BUILD_INSTALACION_TOTAL_SHA256.json'
    build_digest, upstream_entries = manifest_header(upstream_manifest, BUILD_MANIFEST_SHA)
    build_closure = read(OUT / 'CIERRE_BUILD_INSTALACION_TOTAL.json')
    built = read(OUT / 'COMPILACION_TOTAL.json')
    installed = read(OUT / 'APLICACION_TOTAL_EN_D.json')
    require(build_closure['status'] == 'BUILD_AUDITED_INSTALLED_PENDING_PHYSICAL_GATE'
            and built['status'] == 'PASS_PENDING_ELF_MANUAL_REVIEW_AND_ARM'
            and installed['status'] == 'PASS_INSTALLED_PENDING_ARM'
            and build_closure['elf_sha256'] == built['configurations'][CONFIG]['elf_sha256']
            == installed['elf_sha256'] == report['firmware_elf_sha256'] == ELF_SHA,
            'Cadena de compilación o instalación incompatible')
    name = PROJECT_NAME + '.elf'
    elf_paths = {
        'declared_gate': GATE / 'firmware_declarado.elf',
        'isolated_build': OUT / 'copia_aislada' / PROJECT_NAME / CONFIG / name,
        'archived_build': OUT / 'intentos' / str(built['attempt']) / 'compilacion' / CONFIG / name,
        'installed_D': Path('D:/STM32IDE') / PROJECT_NAME / CONFIG / name,
    }
    for label, path in elf_paths.items():
        require(sha(path) == ELF_SHA, 'ELF cambió: ' + label)
    require(report['firmware_loaded_verified_by_programmer_report'] is False
            and report['programmer_report_attached'] is False, 'Revisar nueva prueba de programador antes de cerrar')

    paths = {GATE / relative for relative in gate_entries}
    paths.update((GATE / 'MANIFEST_EVIDENCIAS_SHA256.json', GATE / 'MANIFEST_EVIDENCIAS_SHA256.sha256'))
    paths.update(RESOURCES / relative for relative in resource_entries)
    paths.update((RESOURCES / 'MANIFEST_RECURSOS_SHA256.json', RESOURCES / 'MANIFEST_RECURSOS_SHA256.sha256'))
    paths.update((CLIENT, SOURCE_CLIENT, NUMERIC_SOURCE, Path(__file__).resolve(), METHOD, GUIDE,
                  OUT / 'AUDITORIA_TRAZA_GATE_TOTAL.json', OUT / 'AUDITORIA_NUMERICA_GATE_TOTAL.json',
                  OUT / 'ESTIMACION_DURACION_CAMPANA_TOTAL.json', upstream_manifest,
                  upstream_manifest.with_suffix('.sha256'), OUT / 'CIERRE_BUILD_INSTALACION_TOTAL.json',
                  OUT / 'COMPILACION_TOTAL.json', OUT / 'AUDITORIA_ELF_TOTAL.json',
                  OUT / 'AUDITORIA_MAQUINA_TOTAL.json', OUT / 'APLICACION_TOTAL_EN_D.json',
                  elf_paths['isolated_build'], elf_paths['archived_build']))
    require(all(path.is_file() and path.resolve().is_relative_to(ROOT.resolve()) for path in paths),
            'Archivo requerido ausente o fuera del workspace')
    entries = {path.relative_to(ROOT).as_posix(): sha(path) for path in sorted(paths)}
    checked_upstream = []
    for relative, current in entries.items():
        if relative in upstream_entries:
            require(current == upstream_entries[relative], 'Recurso relevante cambió respecto al cierre build: ' + relative)
            checked_upstream.append(relative)

    user_python = 'C:/Users/LENOVO/AppData/Local/Microsoft/WindowsApps/python3.12.exe'
    argv = [str(Path(user_python)), '-B', str(CLIENT), '--full', '--port', 'COM4', '--firmware-elf',
            str(elf_paths['installed_D']), '--build-report', str(OUT / 'COMPILACION_TOTAL.json'),
            '--gate-evidence', str(GATE), '--max-seconds', '1800']
    guide_text = GUIDE.read_text(encoding='utf-8')
    require(guide_text.count('```powershell') == 1, 'Comando PowerShell ambiguo en guía')
    powershell = guide_text.split('```powershell', 1)[1].split('```', 1)[0].strip()
    require('--full --port COM4' in powershell and '--max-seconds 1800' in powershell
            and '20261026T040903_561072Z' in powershell, 'Comando documentado distinto del previsto')
    closure = dict(
        schema_version=1, status='GATE_AUDITED_READY_FOR_USER_FULL_CAMPAIGN',
        date_local='2026-10-25', timezone='America/Lima', gate=GATE.relative_to(ROOT).as_posix(),
        gate_manifest_sha256=gate_digest, gate_resources_verified=12,
        resource_manifest_sha256=resource_digest, exported_resources_verified=39,
        ancestor_build_manifest_sha256=build_digest,
        ancestor_strategy='Verify ancestor seal SHA and current relevant inherited files; do not rehash unrelated2377 resources.',
        current_relevant_ancestor_files_verified=len(checked_upstream),
        catalog_sha256=catalog['catalog_sha256'], client_sha256=sha(CLIENT),
        declared_ELF_sha256=ELF_SHA, four_ELF_copies_identical=True,
        ELF_copy_paths={label:str(path) for label,path in elf_paths.items()},
        firmware_identity_on_board_verified_by_programmer_report=False,
        audited_gate=dict(primary_RUN=3, diagnostic_PROFILE=1, BASE=1, protocol_rejections=17,
                          trace_rows=450, transactions=225, RAW_float32_values=8192,
                          primary_mean_ms=numeric['primary_statistics']['mean_ms'],
                          primary_min_ms=numeric['primary_min_ms'], primary_max_ms=numeric['primary_max_ms'],
                          first_model_arena_used_bytes=1188, reserved_arena_bytes=16384,
                          arena_of_other31_models_verified=False,
                          same_input_INT8_output_exact=True, ARM_feature_replay_exact=True,
                          PC_cache_end_to_end_bit_exact_claimed=False),
        planned_full_campaign=dict(unique_RAW_windows=42, model_window_pairs=896, main_RUN=2688,
                                   main_RUN_per_representative=84, diagnostic_PROFILE=896, BASE=42,
                                   protocol_rejections=17, expected_status='FULL_ARM_MEASUREMENTS_READY_FOR_AUDIT',
                                   estimated_user_duration_minutes=[15,25], budget_seconds=1800,
                                   command_argv=argv, powershell_command_exact_from_guide=powershell,
                                   executed_by_closure=False, user_personally_launches=True),
        total_cost_validated=False, selector_enabled=False,
        serial_port_opened=False, hardware_flashed=False, new_physical_RUN=0,
        TFLite_invocations=0, training_fits=0, audits_rerun=False,
        mutable_ESTADO_included=False, prior_manifests_modified=False,
        manifest_entries_including_closure=len(entries)+1,
        manifest_scope='Root-relative gate originals, exported resources, clients, reports, relevant ancestors, ELF copies, new audit source and final documentation; no D paths in manifest.',
        seconds_before_writing=time.perf_counter()-started,
        limitations=['The gate physically exercised one model/window; full32-model cost calibration remains pending.',
                     'Copied and declared ELF identity is not readback verification of all board Flash.',
                     'Three observed timings do not establish84-sample per-representative envelopes, CI or WCET.',
                     'Duration forecast is planning information and is not measured full-campaign runtime.'])
    write(CLOSURE, closure)
    entries[CLOSURE.relative_to(ROOT).as_posix()] = sha(CLOSURE)
    write(MANIFEST, dict(sorted(entries.items())))
    with SEAL.open('x', encoding='ascii') as stream:
        stream.write(sha(MANIFEST) + '  ' + MANIFEST.name + '\n')
    for relative, expected in entries.items():
        require(sha(ROOT / relative) == expected, 'Recurso cambió durante cierre: ' + relative)
    print(json.dumps(dict(status=closure['status'], resources_verified=len(entries),
                          manifest_sha256=sha(MANIFEST), closure_sha256=sha(CLOSURE),
                          ancestor_relevant_files_verified=len(checked_upstream),
                          seconds=time.perf_counter()-started), ensure_ascii=False))


if __name__ == '__main__':
    main()
