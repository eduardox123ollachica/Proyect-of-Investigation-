"""Reconstruye el gate Canon42 desde UART, sin serial ni inferencia PC.

La auditoría se escribe aparte; nunca modifica las evidencias originales.
Solo admite un RAW cuyo INT8 sea el del NPZ guardado: en este gate real
todos lo cumplen, de modo que no hace falta invocar un clasificador.
"""
from __future__ import annotations

from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT/'despliegue_stm32/canonicos_42_180mhz_2026-10-27'
GATE = OUT/'Validacion_PC/resultados/20261028T050129_905025Z'
SOURCE = ROOT/'codigo/cwru/ampliacion_v2/canonicos_validacion/validar_canonicos_gate.py'
REPORT = OUT/'AUDITORIA_GATE_CANON42.json'
EXPECTED_JOINT = 'b1cbbc5d78ee3268cef2b2e412b9f83a4a139ca5756940ff8ff82dddb467c238'


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def load_client():
    spec = importlib.util.spec_from_file_location('audit_gate_canon_client', SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def strict_manifest(folder, name, allowed_extra=()):
    """Incluye todas las subcarpetas; rechaza archivos añadidos o ausentes."""
    folder = Path(folder).resolve()
    path = folder/name
    companion = path.with_suffix('.sha256')
    require(companion.read_text(encoding='ascii').split() == [sha(path), name], 'Sello inválido')
    entries = read(path)
    actual = {p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file()
        and p not in (path, companion)}
    require(set(entries) | set(allowed_extra) == actual, 'Inventario recursivo incompleto o con archivos adicionales')
    for relative, expected in entries.items():
        target = (folder/relative).resolve()
        require(target.is_relative_to(folder) and sha(target) == expected, 'Evidencia alterada: '+relative)
    return entries


def read_jsonl(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding='utf-8-sig').splitlines()]
    require([r['seq'] for r in rows] == list(range(1, len(rows)+1)), 'Secuencias JSONL discontinuas')
    return rows


def audit_programmer_logs(g, programming):
    """Contrasta comandos, identidad y stdout originales con la carga sellada."""
    source = g.PACKAGE/'cargar_firmware_canon42.py'
    spec = importlib.util.spec_from_file_location('audit_canon_programmer_pure_helpers', source)
    loader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader)
    folder = Path(programming['evidence_folder'])
    report = read(folder/'RESULTADO_CARGA.json')
    phases = report['phases']
    require(len(phases) == 2 and all(loader.command_ok(phase) for phase in phases),
        'La carga no contiene dos fases CLI satisfactorias')
    require(phases[0]['command'] == loader.connection_command(folder/'programmer_identidad.log')
        and phases[1]['command'] == loader.program_command(folder/'programmer_carga.log'),
        'Los comandos CLI no corresponden al ELF auditado o la sonda fijada')
    for phase, stdout, detail in zip(phases, ('stdout_identidad.txt', 'stdout_carga.txt'),
            ('programmer_identidad.log', 'programmer_carga.log')):
        text = (folder/stdout).read_bytes().decode('utf-8')
        require(text == phase['text'], 'Stdout no coincide con la fase CLI')
        require(loader.identity(text)['approved'], 'La identidad registrada por SWD no coincide')
        raw = (folder/detail).read_bytes()
        log = raw.decode('utf-16' if raw[:2] in (b'\xff\xfe', b'\xfe\xff') else 'utf-8-sig', errors='replace')
        require(loader.identity(log)['approved'], 'El log detallado no identifica la sonda/MCU esperado')
    require(report['program_identity'] == loader.identity(phases[1]['text'])
        and 'Download verified successfully' in phases[1]['text'] and report['reset_requested']
        and report['reset_command_succeeded'] and report['download_verification_marker'],
        'El programador no confirma la descarga y reset del ELF')
    require(report['CLI_sha256'] == loader.CLI_SHA == sha(loader.CLI), 'Programador instalado diferente')
    require(0 < report['seconds'] <= report['hardware_budget_seconds'] == 120,
        'Carga fuera de su presupuesto')
    return dict(phases=2, fixed_probe_serial=loader.PROBE_SN, expected_device_id='0x421',
        programmer_executable_sha256=loader.CLI_SHA, verification_marker_confirmed=True,
        reset_confirmed=True, programming_seconds=report['seconds'])


class ReplayLink:
    """Reproduce únicamente líneas archivadas, sin abrir un puerto."""
    def __init__(self, trace):
        self.trace, self.index = trace, 0
        stamps = [datetime.fromisoformat(row['utc']) for row in trace]
        require(all(a <= b for a, b in zip(stamps, stamps[1:])), 'UTC de UART no monótono')

    def ask(self, command, prefix, allow_error=False):
        require(self.index+1 < len(self.trace), 'Traza UART incompleta')
        tx, rx = self.trace[self.index:self.index+2]
        require(tx['direction'] == 'TX' and tx['command'] == command,
            'TX diferente del protocolo/banco en seq '+str(tx['seq']))
        require(rx['direction'] == 'RX' and rx['line'].startswith(prefix),
            'RX diferente del protocolo esperado en seq '+str(rx['seq']))
        require(allow_error or not rx['line'].startswith('CWRUTOTAL_ERROR,'), 'Error UART no autorizado')
        self.index += 2
        return rx['line']


class SavedOnlyOracle:
    invocations = 0

    def expected(self, rid, quantized, ref):
        require(quantized == ref['int8_inputs'].astype(int).tolist(),
            'El gate necesita un oráculo diferente de la referencia guardada')
        return ref['int8_outputs'].astype(int).tolist(), 'saved_same_INT8_input'


def reconstruct(g, h, cat, bank, arrays, trace, events):
    plan = g.make_plan(cat, bank, arrays['postrefs'])
    link = ReplayLink(trace)
    info = g.parse_info(h, link.ask('CWRUTOTAL_INFO', 'CWRUTOTAL_INFO,'), cat)
    require(link.ask('CWRUTOTAL_POST_INFO', 'CWRUTOTAL_POST_INFO,') == 'CWRUTOTAL_POST_INFO,1,14,56,0',
        'POST_INFO incompatible')
    last = info['last_run_id']
    for rid in range(8):
        h.validate_model(link.ask(f'CWRUTOTAL_MODEL,{rid}', 'CWRUTOTAL_MODEL,'), cat['representatives'][rid])
    records, runs, posts, bases, selections = [], [], [], [], []
    def emit(kind, result):
        records.append(dict(seq=len(records)+1, type=kind, **result))
    by_id = {row['global_id']: row for row in bank['RAW_windows']}
    oracle = SavedOnlyOracle()
    for gid, ids in plan['RAW_schedule']:
        row = dict(by_id[gid], arm_features14_bits=None)
        h.transmit(link, arrays['raw']['windows'][row['npz_index']], row)
        base = h.baseline(link, row)
        bases.append(base); emit('BASE', base)
        for rid in ids:
            rep, model = cat['representatives'][rid], bank['models'][rid]
            selection = h.parse_selected(link.ask(f'CWRUTOTAL_SELECT,{rid}', 'CWRUTOTAL_SELECTED,'), rep)
            selections.append(selection); emit('SELECT', dict(global_id=gid, **selection))
            ref, previous = g.reference(arrays['rawrefs'], model, gid), None
            for repetition, mode in [(1, 0), (2, 0), (3, 0), (1, 1)]:
                command = 'CWRUTOTAL_PROFILE' if mode else 'CWRUTOTAL_RUN'
                result = h.parse_result(link.ask(command+','+row['checksum_fnv1a'], 'CWRUTOTAL_RESULT,'),
                    rep, row, last, mode, selection['arena_used'])
                last = result['run_id']
                pages = h.get_pages(link, result, rep)
                comparison = g.compare_raw(h, result, pages, base, rep, ref, oracle)
                require(previous is None or previous == pages, 'RUN/PROFILE no conservan los intermedios')
                previous = pages
                result.update(repetition=repetition, status='PASS', pages=pages, comparison=comparison)
                runs.append(result); emit('PROFILE' if mode else 'RUN', result)
    for case in plan['POST_cases']:
        rid, gid = case['model_id'], case['global_id']
        rep = cat['representatives'][rid]
        ref = g.reference(arrays['postrefs'], bank['models'][rid], gid)
        selection = h.parse_selected(link.ask(f'CWRUTOTAL_SELECT,{rid}', 'CWRUTOTAL_SELECTED,'), rep)
        selections.append(selection); emit('SELECT_POST', dict(global_id=gid, **selection))
        command, checksum = g.post_command(h, ref['features14'])
        result = g.parse_post_result(h, link.ask(command, 'CWRUTOTAL_POST_RESULT,'),
            rep, checksum, last, selection['arena_used'])
        last = result['run_id']; pages = h.get_pages(link, result, rep)
        comparison = g.compare_post(h, result, pages, rep, ref)
        result.update(global_id=gid, label=case['label'], pages=pages, comparison=comparison, status='PASS')
        posts.append(result); emit('POST', result)
    rejected = g.negatives(h, link, posts[-1], ref['features14'])
    for item in rejected:
        emit('NEGATIVE_POST', item)
    final = g.parse_info(h, link.ask('CWRUTOTAL_INFO', 'CWRUTOTAL_INFO,'), cat)
    require(final['last_run_id'] == last and not final['result_valid'], 'Rechazos alteraron publicación o contador')
    require(link.index == len(trace), 'Traza contiene intercambios no consumidos')
    require(records == events, 'MEDICIONES no coincide exactamente con la reconstrucción UART/NPZ')
    require(info['last_run_id'] == 0 and last == 56, 'Secuencia global de 56 resultados diferente')
    return dict(plan=plan, firmware_info=info, firmware_info_final=final,
        RAW_timing_statistics=h.statistics(runs, cat['representatives']),
        RAW_primary=[row for row in runs if row['mode'] == 0],
        PROFILE=[row for row in runs if row['mode'] == 1], POST=posts, BASE=bases,
        selections=selections, negatives=rejected, events=records,
        UART_pairs=link.index//2, PC_same_input_classifier_invocations=0)


def audit(folder=GATE):
    folder = Path(folder).resolve()
    require(folder == GATE.resolve(), 'Esta auditoría se refiere al gate canónico fijado')
    entries = strict_manifest(folder, 'MANIFEST_EVIDENCIAS_SHA256.json')
    original_hashes = {path: sha(folder/path) for path in entries}
    g = load_client()
    h, cat, bank, arrays = g.load_resources()
    args = SimpleNamespace(firmware_elf=OUT/'copia_aislada/Prueba_2_CWRU_Extraccion'/g.CONFIG/'Prueba_2_CWRU_Extraccion.elf',
        build_report=OUT/'COMPILACION_CANON42.json', elf_audit=OUT/'AUDITORIA_ELF_CANON42.json')
    joint = g.joint_manifest(args)
    require(joint == EXPECTED_JOINT, 'El cliente no pertenece al sello conjunto ejecutado')
    provenance = g.authenticate_build(args, cat)
    programming = g.authenticate_programming(joint)
    programmer_logs = audit_programmer_logs(g, programming)
    report = read(folder/'RESULTADO_GATE.json')
    require(report['status'] == g.GATE_STATUS and report['mode'] == 'gate', 'Gate no terminó con PASS')
    require(report['firmware_elf_sha256'] == g.ELF_SHA and report['catalog_sha256'] == g.CATALOG_SHA
        and report['bank_manifest_sha256'] == g.BANK_MANIFEST_SHA
        and report['joint_client_manifest_sha256'] == joint
        and report['programming_evidence'] == programming, 'Procedencia del resultado divergente')
    require(sha(folder/'firmware_declarado.elf') == g.ELF_SHA
        and sha(folder/'COMPILACION_CANON42.json') == provenance['build_report_sha256'] == report['build_report_sha256']
        and sha(folder/'AUDITORIA_ELF_CANON42.json') == provenance['elf_audit_sha256'] == report['elf_audit_sha256']
        and sha(folder/'validar_canonicos_gate_ejecutado.py') == sha(SOURCE) == report['client_sha256']
        and sha(folder/g.HELPER_NAME) == report['helper_sha256'] == g.HELPER_SHA,
        'Capturas ejecutadas no corresponden a originales autenticados')
    for relative, digest in programming['files_sha256'].items():
        archived = folder/'carga_verificada'/Path(relative).name
        require(sha(archived) == digest == sha(ROOT/relative), 'Copia de programación divergente')
    strict_manifest(Path(programming['evidence_folder']), 'MANIFEST_CARGA_SHA256.json')
    strict_manifest(folder/'carga_verificada', 'MANIFEST_CARGA_SHA256.json', ['INTENTO_CARGA_CANON42.json'])
    # El marcador se archivó fuera del manifiesto local original de la carga.
    # Por eso se valida ese inventario sobre su carpeta original, mientras la
    # copia archivada completa queda cubierta por el manifiesto recursivo gate.
    trace, events = read_jsonl(folder/'TRAZA_UART.jsonl'), read_jsonl(folder/'MEDICIONES.jsonl')
    rebuilt = reconstruct(g, h, cat, bank, arrays, trace, events)
    counters = dict(completed_RAW_RUN=len(rebuilt['RAW_primary']), completed_RAW_PROFILE=len(rebuilt['PROFILE']),
        completed_RAW_BASE=len(rebuilt['BASE']), completed_POST=len(rebuilt['POST']),
        POST_negative_rejections=len(rebuilt['negatives']))
    require(list(counters.values()) == [24, 8, 2, 24, 3]
        and all(report[key] == value for key, value in counters.items()), 'Contadores divergentes')
    for key in ('firmware_info', 'firmware_info_final', 'RAW_timing_statistics', 'PC_same_input_classifier_invocations'):
        require(report[key] == rebuilt[key], 'Resumen reconstruido diferente: '+key)
    executed = read(folder/'PLAN_EJECUTADO.json')
    require(all(executed[key] == value for key, value in rebuilt['plan'].items()), 'Plan ejecutado diferente')
    require(executed['client_sha256'] == report['client_sha256'] and executed['helper_sha256'] == g.HELPER_SHA,
        'Plan no corresponde al código ejecutado')
    require(report['wall_seconds'] > 0 and report['wall_seconds'] <= executed['max_seconds'] == 600,
        'Duración fuera del presupuesto fijado')
    require(not report['full_validation_completed'] and not report['RAW_exhaustive_reserved_coverage']
        and not report['POST_exhaustive_reserved_coverage'] and not report['POST_timing_is_full_pipeline']
        and not report['cached_PC_end_to_end_exactness_claimed'], 'Alcance excesivo en el resumen')
    require(all(sha(folder/path) == digest for path, digest in original_hashes.items()), 'Evidencia cambió durante auditoría')
    unique_pairs = [row for row in rebuilt['RAW_primary'] if row['repetition'] == 1]
    return dict(status='PASS_GATE_CANON42_AUDITED_PENDING_FULL', date_local='2026-10-28', timezone='America/Lima',
        gate_folder=str(folder), gate_evidence_manifest_sha256=sha(folder/'MANIFEST_EVIDENCIAS_SHA256.json'),
        gate_report_sha256=sha(folder/'RESULTADO_GATE.json'), verified_events_sha256=canonical_sha(events),
        firmware_elf_sha256=g.ELF_SHA, catalog_sha256=g.CATALOG_SHA, bank_manifest_sha256=g.BANK_MANIFEST_SHA,
        joint_client_manifest_sha256=joint, client_sha256=report['client_sha256'], helper_sha256=g.HELPER_SHA,
        programming_evidence=programming, programmer_logs=programmer_logs,
        counters=counters, UART_pairs=rebuilt['UART_pairs'], event_count=len(events),
        UTC_monotonic=True, RAW_payloads_exact_bank=True, all_same_input_INT8_outputs_exact_saved=True,
        all_intermediates_bit_exact=True, all_selected_features_bit_exact_BASE=True,
        all_POST_saved_external_predictions_exact=True, PROFILE_pages_equal_RUN=True,
        PC_same_input_classifier_invocations=0, RAW_timing_statistics=rebuilt['RAW_timing_statistics'],
        RAW_unique_window_ids=[b['global_id'] for b in rebuilt['BASE']], RAW_model_ids=list(range(8)),
        POST_model_ids=list(range(8)), POST_labels_per_model=[0, 1, 2],
        arena_used_by_model={str(rid): sorted({s['arena_used'] for s in rebuilt['selections'] if s['rep_id'] == rid})
            for rid in range(8)},
        RAW_PC_cache_comparison=[dict(rep_id=row['rep_id'], global_id=row['global_id'],
            **row['comparison']['cached_PC_differences']) for row in unique_pairs],
        RAW_max_abs_feature_error_PC_cache=max(b['max_abs_error_PC_cache'] for b in rebuilt['BASE']),
        gate_wall_seconds=report['wall_seconds'], training_fits=0, classifier_invocations=0,
        serial_port_opened=False, hardware_flashed=False, D_files_modified=False,
        audit_script_sha256=sha(__file__), evidence_files=len(entries),
        limitations=['Gate is partial: two RAW windows/eight model-window pairs and 24 POST cases.',
            'Full staged bank remains pending; RAW heldout extraction is not exhaustive.',
            'HAL clock is declared, not externally measured; no WCET or runtime stack maximum established.',
            'PC cached features differ numerically from RAW ARM features; output exactness here uses the same INT8 input.'])


def main():
    require(not REPORT.exists(), 'No sobrescribir auditoría existente')
    result = audit()
    with REPORT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps({key: result[key] for key in ('status', 'counters', 'UART_pairs',
        'arena_used_by_model', 'gate_wall_seconds', 'gate_evidence_manifest_sha256')}, ensure_ascii=True))


if __name__ == '__main__':
    main()
