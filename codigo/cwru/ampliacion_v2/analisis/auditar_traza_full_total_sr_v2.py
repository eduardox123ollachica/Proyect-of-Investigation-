"""Reproduce toda la traza SR completa, sin placa, TensorFlow o compilacion.

Solo crea un informe nuevo. No altera capturas, sellos ni el proyecto D:.
Las funciones de protocolo se aplican al texto UART ya guardado.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT/'despliegue_stm32/costo_total_single_rounding_v2_180mhz_2026-10-26'
RUN = OUT/'Validacion_PC/resultados/20261026T144540_396879Z'
GATE = OUT/'Validacion_PC/resultados/20261026T143621_946818Z'
CLIENT = OUT/'Validacion_PC/validar_total_sr.py'
SOURCE = ROOT/'codigo/cwru/ampliacion_v2/costo_total_single_rounding/validar_total_sr.py'
HELPER = Path(__file__).with_name('auditar_traza_gate_total_sr_v2.py')
REPORT = OUT/'AUDITORIA_TRAZA_CAMPANA_SR.json'
ELF_SHA = 'c0c3853d2b0e90bb20fe00be13bcf107d186088406be131b729a605ded44ecc7'
CLIENT_SHA = '631303e144b432a7144054150ed6466d8a8b10d1f179484329bc2e2b666d91fb'
BUILD_SHA = '94ca669647e4b9924730febe6513c338277ae401f5cfabf008f4f3b69af7f974'
HELPER_SHA = '55a25f6c421e9b0a84a768481742f894eeca9d0e02f30ee3f3fa071240efc45c'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, 'Modulo no importable')
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def main():
    started = time.perf_counter()
    require(not REPORT.exists(), 'No sobrescribir auditoria')
    require(sha(HELPER) == HELPER_SHA, 'Helper de traza modificado')
    h = load(HELPER, 'sr_full_readonly_trace_helpers')
    run_digest, run_entries = h.local_manifest(RUN, 'MANIFEST_EVIDENCIAS_SHA256.json')
    require(len(run_entries) == 12, 'Captura no contiene doce recursos previstos')
    resource_digest, resources = h.local_manifest(OUT/'recursos', 'MANIFEST_RECURSOS_SHA256.json')
    require(len(resources) == 39, 'Banco exportado inesperado')
    require(sha(CLIENT) == sha(SOURCE) == sha(RUN/'validar_total_ejecutado.py') == CLIENT_SHA,
            'Cliente distinto del auditado')
    require(sha(RUN/'firmware_declarado.elf') == ELF_SHA
            and sha(RUN/'build_report.json') == sha(OUT/'COMPILACION_SR.json') == BUILD_SHA,
            'ELF o build distintos de SR')
    for name in ('CATALOGO_TOTAL.json', 'CATALOGO_CANONICO.json', 'BANCO_TOTAL.json',
                 'MANIFEST_RECURSOS_SHA256.json', 'MANIFEST_RECURSOS_SHA256.sha256'):
        require(sha(RUN/name) == sha(OUT/'recursos'/name), 'Recurso de captura distinto: '+name)
    c = load(CLIENT, 'sr_full_saved_client_parsers')
    cat, bank, windows, _ = c.load_resources(OUT/'recursos')
    c.verify_gate(GATE, cat, ELF_SHA, resource_digest)
    report, plan = read(RUN/'RESULTADO_TOTAL.json'), read(RUN/'PLAN_EJECUTADO.json')
    expected_plan = [[gid, ids] for gid, ids in c.schedule(bank, True)]
    require(report['status'] == 'FULL_ARM_MEASUREMENTS_READY_FOR_AUDIT'
            and report['mode'] == plan['mode'] == 'full'
            and plan['schedule'] == expected_plan
            and len(expected_plan) == 42 and sum(len(ids) for _, ids in expected_plan) == 896,
            'Calendario o estado de campana incompatible')
    require(plan['gate_evidence'] == str(GATE)
            and (plan['expected_primary_RUN'], plan['expected_PROFILE'], plan['expected_BASE']) == (2688, 896, 42)
            and plan['max_seconds'] == 1800 and report['wall_seconds'] < 1800,
            'Puerta o limites de ejecucion diferentes')
    trace, events = h.jsonlines(RUN/'TRAZA_UART.jsonl'), h.jsonlines(RUN/'MEDICIONES.jsonl')
    require(len(trace)%2 == 0 and [r['seq'] for r in trace] == list(range(1, len(trace)+1))
            and [r['seq'] for r in events] == list(range(1, len(events)+1)), 'Secuencias no consecutivas')
    dates = [datetime.fromisoformat(row['utc']) for row in trace]
    require(all(date.utcoffset().total_seconds() == 0 for date in dates)
            and all(a <= b for a, b in zip(dates, dates[1:])), 'Fechas no UTC o no monotonas')
    link = h.Replay(trace)
    event_pos = 0

    def event(kind, fields):
        nonlocal event_pos
        require(event_pos < len(events), 'Eventos agotados')
        row = events[event_pos]
        require(row['type'] == kind and all(row[k] == value for k, value in fields.items()),
                'Evento distinto de la publicacion UART: '+kind)
        event_pos += 1
        return row

    initial = c.parse_info(link.ask('CWRUTOTAL_INFO', 'CWRUTOTAL_INFO,'), cat)
    require(initial == report['firmware_info'] and initial['last_run_id'] == 8
            and not initial['result_valid'], 'INFO inicial distinto del estado posterior a la puerta')
    last = initial['last_run_id']
    raw_reports, selections, measured = [], [], []
    model_seen, source_matrices = set(), {}
    counts = Counter()
    per_rep_gids = {rep['id']: [] for rep in cat['representatives']}
    rows = {row['global_id']: row for row in bank['windows']}
    for gid, ids in expected_plan:
        row = rows[gid]
        window = windows[row['npz_index']]
        metadata = row['metadata']
        source = (ROOT/metadata['source_windows']).resolve()
        require(source.is_relative_to(ROOT.resolve()), 'Matriz fuente externa')
        if source not in source_matrices:
            source_matrices[source] = np.load(source, mmap_mode='r', allow_pickle=False)
        original = source_matrices[source]
        require(np.asarray(original[int(metadata['source_row'])], dtype='<f4').tobytes() == window.tobytes(),
                'RAW distinto de su fila fuente')
        before = link.pos
        c.transmit(link, window, row)
        frames = trace[before:link.pos:2]
        data_frames = [r['command'] for r in frames if r['command'].startswith('CWRUFEAT_DATA,')]
        reconstructed = np.asarray([value for command in data_frames for value in command.split(',')[2:]], dtype='<f4')
        require(len(data_frames) == 171 and reconstructed.shape == (8192,)
                and reconstructed.tobytes() == window.tobytes(), 'RAW UART no conserva float32 del banco')
        raw_reports.append(dict(global_id=gid, npz_index=row['npz_index'], frames=171, values=8192,
            checksum_fnv1a=row['checksum_fnv1a'], payload_sha256=row['payload_sha256'], metadata=metadata,
            bit_exact_to_authorized_bank=True, bit_exact_to_source_matrix_row=True,
            authorized_rep_ids=ids))
        base = c.baseline(link, row)
        event('BASE', base)
        counts['BASE'] += 1
        for rid in ids:
            rep = cat['representatives'][rid]
            scope = rep['scope']
            require(rid in row['authorized_rep_ids'] and gid in scope['benchmark_global_window_ids']
                    and gid in scope['training_global_window_ids']
                    and gid not in scope['validation_global_window_ids']
                    and int(metadata['load_hp']) not in (rep['outer_load_hp'], scope['inner_validation_load_hp']),
                    'Ventana temporal fuera del entrenamiento interior autorizado')
            if rid not in model_seen:
                c.validate_model(link.ask(f'CWRUTOTAL_MODEL,{rid}', 'CWRUTOTAL_MODEL,'), rep)
                model_seen.add(rid)
            selected = c.parse_selected(link.ask(f'CWRUTOTAL_SELECT,{rid}', 'CWRUTOTAL_SELECTED,'), rep)
            require(not selected['reused'] and selected['arena_used'] > 0
                    and selected['arena_used'] <= 16384, 'SELECT no asigna arena valida')
            event('SELECT', dict(global_id=gid, **selected))
            selections.append(dict(global_id=gid, **selected))
            per_rep_gids[rid].append(gid)
            first_pages = None
            for repetition, mode in ((1, 0), (2, 0), (3, 0), (1, 1)):
                command = 'CWRUTOTAL_PROFILE' if mode else 'CWRUTOTAL_RUN'
                result = c.parse_result(link.ask(command+','+row['checksum_fnv1a'], 'CWRUTOTAL_RESULT,'),
                                        rep, row, last, mode, selected['arena_used'])
                last = result['run_id']
                pages = c.get_pages(link, result, rep)
                require(first_pages is None or pages == first_pages, 'Repeticion/PROFILE cambia paginas')
                first_pages = pages
                features = [base['features_bits'][i] if i in rep['feature_indices'] else '00000000' for i in range(14)]
                require(pages['features_bits'] == features, 'Features distintas de BASE o slots +0')
                replay = c.replay(pages['features_bits'], rep)
                require(all(pages[k] == value for k, value in replay.items())
                        and result['score_bits'] == pages['score_bits'] and result['rejected'] == pages['rejected'],
                        'Intermedios float32 o distancia distintos de paginas')
                result.update(repetition=repetition, status='PASS', pages=pages)
                kind = 'PROFILE' if mode else 'RUN'
                saved = event(kind, result)
                measured.append(result)
                counts[kind] += 1
                require(saved['comparison']['ARM_features_replay_all_intermediates_bit_exact']
                        and saved['comparison']['selected_features_bit_exact_BASE']
                        and saved['comparison']['inactive_features_positive_zero']
                        and saved['comparison']['same_INT8_input_output_exact'],
                        'Declaracion de comparacion incompatible')
    final_rid = expected_plan[-1][1][-1]
    rejects, support = c.negatives(link, measured[-1], cat['representatives'][final_rid])
    event('NEGATIVE_SUPPORT_SELECT', support)
    for rejection in rejects:
        event('NEGATIVE', rejection)
    final = c.parse_info(link.ask('CWRUTOTAL_INFO', 'CWRUTOTAL_INFO,'), cat)
    require(final == report['firmware_info_final']
            and final['last_run_id'] == last == initial['last_run_id']+3584 == 3592
            and final['selected_id'] == final_rid == 23 and not final['result_valid'], 'Estado final distinto')
    require(link.pos == len(trace) and event_pos == len(events) == 4540, 'Transacciones/eventos sin consumir')
    require((counts['RUN'], counts['PROFILE'], counts['BASE'], len(rejects)) == (2688, 896, 42, 17)
            and len(model_seen) == 32 and len(selections) == 896,
            'Conteos de campana incompletos')
    require([r['global_id'] for r in raw_reports] == report['physical_unique_window_ids']
            and len({r['global_id'] for r in raw_reports}) == 42, 'Banco temporal distinto')
    stats = c.statistics(measured, cat['representatives'])
    require(stats == report['timing_statistics'], 'Estadisticas no reproducibles desde UART')
    per_rep = []
    for rep in cat['representatives']:
        rid = rep['id']
        arena = [r['arena_used'] for r in selections if r['rep_id'] == rid]
        runs = [r for r in measured if r['rep_id'] == rid and r['mode'] == 0]
        profiles = [r for r in measured if r['rep_id'] == rid and r['mode'] == 1]
        require(len(runs) == 84 and len(profiles) == len(arena) == 28
                and len(set(per_rep_gids[rid])) == 28
                and set(per_rep_gids[rid]) == set(rep['scope']['benchmark_global_window_ids']),
                'Banco de representante incompleto')
        per_rep.append(dict(rep_id=rid, representative_id=rep['representative_id'], model_sha256=rep['model_sha256'],
            params_fnv=rep['params_fnv'], unique_windows=28, primary_RUN=84, PROFILE=28, selections=28,
            arena_used_min_bytes=min(arena), arena_used_max_bytes=max(arena),
            initialization_cycles_min=min(r['cycles'] for r in selections if r['rep_id'] == rid),
            initialization_cycles_max=max(r['cycles'] for r in selections if r['rep_id'] == rid),
            benchmark_all_training_only=True, global_window_ids=per_rep_gids[rid]))
    machine = read(OUT/'AUDITORIA_MAQUINA_SR.json')
    require(machine['status'] == 'PASS_SR_MACHINE_ROUNDING_REVIEW_PENDING_ARM'
            and machine['elf_sha256'] == ELF_SHA and machine['primary']['read_count'] == 2
            and machine['profile']['read_count'] == 10, 'Alcance temporal de otro ELF')
    durations = [(datetime.fromisoformat(rx['utc'])-datetime.fromisoformat(tx['utc'])).total_seconds()
                 for tx, rx in link.transactions]
    result = dict(status='PASS_SR_FULL_TRACE_REPLAYED_NO_HARDWARE', date_local='2026-10-26', timezone='America/Lima',
        run=RUN.relative_to(ROOT).as_posix(), run_manifest_sha256=run_digest, run_resources_verified=12,
        resource_manifest_sha256=resource_digest, exported_resources_verified=39,
        gate_manifest_sha256=sha(GATE/'MANIFEST_EVIDENCIAS_SHA256.json'),
        client_sha256=CLIENT_SHA, firmware_elf_sha256=ELF_SHA, build_report_sha256=BUILD_SHA,
        trace_sha256=sha(RUN/'TRAZA_UART.jsonl'), measurements_sha256=sha(RUN/'MEDICIONES.jsonl'),
        trace_rows=len(trace), transactions=len(link.transactions), measurement_rows=len(events),
        all_sequences_timestamps_and_TX_RX_pairs_valid=True, entire_trace_consumed=True,
        completed_primary_RUN=2688, completed_PROFILE=896, completed_BASE=42, negative_rejections=17,
        valid_extra_RUN=0, unique_RAW_windows=42, RAW_frames=42*171, RAW_float32_values=42*8192,
        RAW=raw_reports, model_queries=32, model_selections=896, per_representative=per_rep,
        all32_model_parameter_fingerprints_verified=True, initial_run_counter=initial['last_run_id'],
        final_run_counter=3592, final_result_valid=False, final_selected_id=23,
        pages_bind_run_id_model_checksum=True, pages_bit_identical_across_three_RUN_and_PROFILE=True,
        three_invalid_DATA_frames_atomic_at_count4=True, malformed_query_preserves_publication=True,
        invalid_RUN_and_BEGIN_invalidate_publication=True, benchmark_scope_load_authorizations_verified=True,
        all896_model_window_pairs_are_inner_training_only=True,
        historical_source_split_names_not_used_as_claim_of_new_independent_acquisitions=True,
        timing_statistics_exact_to_trace=stats, PROFILE_excluded_from_primary_statistics=True,
        initialization_UART_BASE_and_SELECT_excluded_from_primary_cost=True,
        max_observed_transaction_seconds=max(durations), wall_seconds_reported=report['wall_seconds'],
        timing_scope_machine_audit_sha256=sha(OUT/'AUDITORIA_MAQUINA_SR.json'),
        firmware_loaded_verified_by_programmer_report=False, programmer_report_attached=report['programmer_report_attached'],
        physical_full_campaign_complete=True, total_cost_validated=False, selector_enabled=False,
        data_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in
            (RUN/'TRAZA_UART.jsonl', RUN/'MEDICIONES.jsonl', RUN/'RESULTADO_TOTAL.json', RUN/'PLAN_EJECUTADO.json')},
        script_sha256=sha(__file__), helper_sha256={HELPER.relative_to(ROOT).as_posix():HELPER_SHA},
        TFLite_invocations=0, new_physical_RUN=0, D_files_modified=False, serial_port_opened=False,
        hardware_flashed=False, seconds=time.perf_counter()-started,
        limitations=['UART fingerprints and declared ELF do not cryptographically verify all board Flash.',
            'Neural arithmetic is independently audited separately; this audit reconstructs protocol and float preprocessing.',
            'Timing excludes UART, initialization, BASE and PROFILE and is not a WCET guarantee.',
            'Only the32 specified models on the42 authorized benchmark windows are covered.'])
    with REPORT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({key:result[key] for key in ('status','run_manifest_sha256','trace_rows','transactions',
        'measurement_rows','completed_primary_RUN','completed_PROFILE','completed_BASE','seconds')}, ensure_ascii=True))


if __name__ == '__main__':
    main()
