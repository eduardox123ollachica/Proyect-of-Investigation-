"""Cierra la puerta SR auditada y prepara la campaña del usuario; sin placa/TF.

Solo crea cierre y sello nuevos; verifica evidencias existentes en lectura.
No repite mediciones, entrenamientos, inferencias ni compilaciones.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT/'despliegue_stm32/costo_total_single_rounding_v2_180mhz_2026-10-26'
GATE = OUT/'Validacion_PC/resultados/20261026T143621_946818Z'
RESOURCES = OUT/'recursos'
CLIENT = OUT/'Validacion_PC/validar_total_sr.py'
SOURCE_CLIENT = ROOT/'codigo/cwru/ampliacion_v2/costo_total_single_rounding/validar_total_sr.py'
METHOD = ROOT/'articulo/ampliacion_cwru/metodologia/calibracion_costo_total_v2_2026-10-25/VALIDACION_PUERTA_SR_2026-10-26.md'
GUIDE = OUT/'CAMPANA_COMPLETA_TOTAL_SR.md'
CLOSURE = OUT/'CIERRE_PUERTA_TOTAL_SR.json'
MANIFEST = OUT/'MANIFEST_PUERTA_TOTAL_SR_SHA256.json'
SEAL = MANIFEST.with_suffix('.sha256')
ELF_SHA = 'c0c3853d2b0e90bb20fe00be13bcf107d186088406be131b729a605ded44ecc7'
BUILD_SEAL = 'e7d2d215b80db22b0cb8fa8a1ee42fadb18821c6eaca07383eb6be98a69a447a'
GATE_SEAL = '0ab587d5fccfbd901215de8feda60f00d5cb545e7df3081c277ee4ca3b166849'
CONFIG = 'Release_180MHz_TotalV2_SR'
PROJECT = 'Prueba_2_CWRU_Extraccion'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def verify_manifest(folder, name, base, expected=None):
    path = folder/name
    digest, declared = path.with_suffix('.sha256').read_text(encoding='ascii').split()
    require(declared == name and sha(path) == digest and (expected is None or digest == expected),
            'Sello diferente: '+name)
    entries = read(path)
    require(isinstance(entries, dict) and entries, 'Manifiesto vacio')
    for relative, declared_sha in entries.items():
        target = (base/relative).resolve()
        require(target.is_relative_to(base.resolve()) and target.is_file() and sha(target) == declared_sha,
                'Evidencia modificada: '+relative)
    return digest, entries


def main():
    started = time.perf_counter()
    require(not any(p.exists() for p in (CLOSURE, MANIFEST, SEAL)), 'No sobrescribir cierre')
    gate_sha, gate_files = verify_manifest(GATE, 'MANIFEST_EVIDENCIAS_SHA256.json', GATE, GATE_SEAL)
    resource_sha, resource_files = verify_manifest(RESOURCES, 'MANIFEST_RECURSOS_SHA256.json', RESOURCES)
    build_sha, build_files = verify_manifest(OUT, 'MANIFEST_BUILD_INSTALACION_SR_SHA256.json', ROOT, BUILD_SEAL)
    numeric_sha, numeric_files = verify_manifest(OUT, 'MANIFEST_NUMERICA_PUERTA_SR_SHA256.json', ROOT)
    require(len(gate_files) == 12 and len(resource_files) == 39 and len(build_files) == 136,
            'Conteos de manifiestos incompatibles')
    trace = read(OUT/'AUDITORIA_TRAZA_PUERTA_SR.json')
    numeric = read(OUT/'AUDITORIA_NUMERICA_PUERTA_SR.json')
    estimate = read(OUT/'ESTIMACION_DURACION_CAMPANA_SR.json')
    report = read(GATE/'RESULTADO_TOTAL.json')
    require(trace['status'] == 'PASS_SR_GATE_TRACE_REPLAYED_NO_HARDWARE'
            and numeric['status'] == 'PASS_SR_NUMERIC_TIMING_GATE_AUDIT_NOT_TOTAL_COST_VALIDATED'
            and trace['gate_manifest_sha256'] == numeric['gate_manifest_sha256'] == gate_sha,
            'Auditorias de otra puerta o no aprobadas')
    require((trace['trace_rows'], trace['transactions'], trace['measurement_rows']) == (834, 417, 30)
            and trace['entire_trace_consumed'] and trace['all_sequences_timestamps_and_TX_RX_pairs_valid']
            and trace['final_run_counter'] == 8 and not trace['final_result_valid']
            and trace['negative_rejections'] == 17 and trace['valid_extra_RUN'] == 0,
            'Traza incompleta o estado final distinto')
    require((trace['completed_primary_RUN'], trace['completed_PROFILE'], trace['completed_BASE']) == (6, 2, 2)
            and (numeric['primary_RUN'], numeric['PROFILE'], numeric['BASE']) == (6, 2, 2)
            and numeric['UART_results_reconstructed'] == 8, 'Conteos incompatibles')
    require([row['global_id'] for row in trace['RAW']] == [568, 232]
            and all(row['frames'] == 171 and row['values'] == 8192
                    and row['bit_exact_to_authorized_bank'] and row['bit_exact_to_source_matrix_row']
                    for row in trace['RAW'])
            and trace['benchmark_scope_load_authorizations_verified']
            and trace['model_and_parameter_fingerprints_verified'], 'RAW o autorizaciones no comprobadas')
    require(trace['trace_sha256'] == sha(GATE/'TRAZA_UART.jsonl')
            and trace['measurements_sha256'] == sha(GATE/'MEDICIONES.jsonl'), 'Traza cambiada')
    cases = numeric['cases']
    require([c['rep_id'] for c in cases] == [0, 22]
            and all(c['same_INT8_input_output_exact'] and c['float32_intermediates_exact']
                    and c['primary_statistics']['count'] == 3 and c['arena_used_bytes'] == 1188
                    and c['profile']['excluded'] for c in cases), 'Numerica o alcance incompatible')
    require(numeric['regression_case_previous_failure_resolved_observed']
            and cases[1]['input_INT8'] == [-45, -15, -38, -36, -69, -39, -42]
            and cases[1]['output_INT8'] == [41, -100, -69]
            and numeric['previous_regression_ARM_output'] == [42, -101, -69]
            and cases[1]['FC_single_logits'] == [33, -29, -3], 'Regresion no confirmada')
    for case in cases:
        stat = next(s for s in report['timing_statistics'] if s['rep_id'] == case['rep_id'])
        require(math.isclose(stat['mean_ms'], case['primary_statistics']['mean_ms'], abs_tol=1e-12),
                'Media distinta del informe original')
    for rel, expected in {**numeric['helper_sha256'], **numeric['data_sha256']}.items():
        require(sha(ROOT/rel) == expected, 'Dependencia numerica modificada: '+rel)
    script_reports = {
        'auditar_traza_gate_total_sr_v2.py': trace,
        'auditar_numerica_gate_total_sr_v2.py': numeric,
        'estimar_duracion_total_sr_v2.py': estimate,
    }
    for name, row in script_reports.items():
        require(sha(Path(__file__).with_name(name)) == row['script_sha256'], 'Auditor distinto: '+name)
    require(estimate['status'] == 'RUNTIME_PLANNING_ESTIMATE_NOT_MEASURED_FULL_CAMPAIGN'
            and estimate['recommended_user_expectation_minutes'] == [15, 25]
            and estimate['recommended_max_seconds'] == 1800
            and estimate['new_physical_RUN'] == estimate['TFLite_invocations'] == 0,
            'Estimacion o presupuesto incompatibles')
    for name, expected in estimate['sources_sha256'].items():
        require(Path(name).name == name and sha(GATE/name) == expected, 'Fuente de estimacion modificada')
    require(sha(CLIENT) == sha(SOURCE_CLIENT) == sha(GATE/'validar_total_ejecutado.py') == report['client_sha256'],
            'Cliente ejecutado o actual distinto')
    spec = importlib.util.spec_from_file_location('sr_closure_client_readonly', CLIENT)
    client = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(client)
    cat, bank, _, _ = client.load_resources(RESOURCES)
    client.verify_gate(GATE, cat, ELF_SHA, resource_sha)
    schedule = client.schedule(bank, True)
    require(len(schedule) == 42 and sum(len(ids) for _, ids in schedule) == 896, 'Calendario full incompatible')
    elf_paths = [GATE/'firmware_declarado.elf', OUT/'copia_aislada'/PROJECT/CONFIG/(PROJECT+'.elf'),
                 OUT/'intentos/1/compilacion'/CONFIG/(PROJECT+'.elf'),
                 Path('D:/STM32IDE')/PROJECT/CONFIG/(PROJECT+'.elf')]
    require(all(sha(p) == ELF_SHA for p in elf_paths), 'ELF declarado/construido/instalado diferente')
    require(not report['programmer_report_attached'] and not report['firmware_loaded_verified_by_programmer_report'],
            'Revisar evidencia nueva del programador antes de cerrar')
    guide = GUIDE.read_text(encoding='utf-8')
    require(guide.count('```powershell') == 1, 'Comando de guia ambiguo')
    command = guide.split('```powershell', 1)[1].split('```', 1)[0].strip()
    require('--full --port COM4' in command and '--max-seconds 1800' in command
            and '20261026T143621_946818Z' in command and CONFIG in command
            and 'validar_total_sr.py' in command and 'COMPILACION_SR.json' in command,
            'Comando documentado incompatible')
    files = {GATE/name for name in gate_files} | {RESOURCES/name for name in resource_files}
    files.update(ROOT/name for name in numeric_files)
    files.update({Path(__file__).resolve(), CLIENT, SOURCE_CLIENT, METHOD, GUIDE})
    files.update(Path(__file__).with_name(name) for name in script_reports)
    files.update(OUT/name for name in (
        'AUDITORIA_TRAZA_PUERTA_SR.json', 'AUDITORIA_NUMERICA_PUERTA_SR.json',
        'ESTIMACION_DURACION_CAMPANA_SR.json', 'MANIFEST_NUMERICA_PUERTA_SR_SHA256.json',
        'MANIFEST_NUMERICA_PUERTA_SR_SHA256.sha256', 'CIERRE_BUILD_INSTALACION_SR.json',
        'MANIFEST_BUILD_INSTALACION_SR_SHA256.json', 'MANIFEST_BUILD_INSTALACION_SR_SHA256.sha256'))
    files.update({GATE/'MANIFEST_EVIDENCIAS_SHA256.json', GATE/'MANIFEST_EVIDENCIAS_SHA256.sha256',
                  RESOURCES/'MANIFEST_RECURSOS_SHA256.json', RESOURCES/'MANIFEST_RECURSOS_SHA256.sha256'})
    require(all(p.is_file() and p.resolve().is_relative_to(ROOT.resolve()) for p in files), 'Recurso externo o ausente')
    entries = {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(files)}
    closure = dict(status='SR_GATE_AUDITED_READY_FOR_USER_FULL_CAMPAIGN',
        date_local='2026-10-26', timezone='America/Lima', gate=GATE.relative_to(ROOT).as_posix(),
        gate_manifest_sha256=gate_sha, build_manifest_sha256=build_sha, numeric_manifest_sha256=numeric_sha,
        gate_resources_verified=12, build_resources_verified=136, exported_resources_verified=39,
        client_sha256=sha(CLIENT), elf_sha256=ELF_SHA, four_ELF_copies_identical=True,
        trace_rows=834, transactions=417, measurement_rows=30,
        gate_counts=dict(primary_RUN=6, PROFILE=2, BASE=2, negative_rejections=17),
        cases=[{k:c[k] for k in ('rep_id', 'global_window_id', 'input_INT8', 'output_INT8',
                               'primary_statistics', 'arena_used_bytes')} for c in cases],
        regression_previous_failure_resolved_for_tested_case=True,
        independent_internal_ARM_activations_captured=False,
        cached_PC_end_to_end_bit_exactness_claimed=False,
        full_image_identity_verified_by_programmer_report=False,
        planned_full_campaign=dict(unique_RAW_windows=42, model_window_pairs=896,
            primary_RUN=2688, primary_RUN_per_representative=84, PROFILE=896, BASE=42,
            negative_rejections=17, estimated_minutes=[15, 25], budget_seconds=1800,
            expected_status='FULL_ARM_MEASUREMENTS_READY_FOR_AUDIT',
            powershell_command_from_guide=command, user_personally_launches=True, launched_here=False),
        old_firmware_or_gate_timings_imported_into_full_table=False,
        total_cost_validated=False, selector_enabled=False,
        new_physical_RUN=0, TFLite_invocations=0, training_fits=0, builds=0,
        serial_port_opened=False, hardware_flashed=False, D_files_modified=False,
        manuscript_modified=False, original_captures_and_seals_preserved=True,
        manifest_entries_including_closure=len(entries)+1, seconds_before_write=time.perf_counter()-started,
        limitations=['Two cases do not establish all32-model equivalence or complete costs.',
            'Full board Flash identity lacks programmer verification evidence.',
            'Three principal samples per case do not replace84 per representative or establish WCET.'])
    write(CLOSURE, closure)
    entries[CLOSURE.relative_to(ROOT).as_posix()] = sha(CLOSURE)
    write(MANIFEST, dict(sorted(entries.items())))
    with SEAL.open('x', encoding='ascii') as stream:
        stream.write(sha(MANIFEST)+'  '+MANIFEST.name+'\n')
    for relative, expected in entries.items():
        require(sha(ROOT/relative) == expected, 'Cambio despues del cierre: '+relative)
    print(json.dumps(dict(status=closure['status'], files=len(entries), manifest_sha256=sha(MANIFEST),
        seconds=time.perf_counter()-started), ensure_ascii=True))


if __name__ == '__main__':
    main()
