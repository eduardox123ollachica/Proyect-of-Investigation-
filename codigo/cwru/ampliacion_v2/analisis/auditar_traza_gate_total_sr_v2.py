"""Reproduce el protocolo de la puerta SR sobre su traza, sin placa ni TF.

Importar no escribe. El programa crea exclusivamente un informe nuevo;
conserva las capturas, clientes, modelos, sellos, firmware y proyecto D:.
"""
from __future__ import annotations

from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT/'despliegue_stm32/costo_total_single_rounding_v2_180mhz_2026-10-26'
GATE = OUT/'Validacion_PC/resultados/20261026T143621_946818Z'
CLIENT = OUT/'Validacion_PC/validar_total_sr.py'
SOURCE = ROOT/'codigo/cwru/ampliacion_v2/costo_total_single_rounding/validar_total_sr.py'
REPORT = OUT/'AUDITORIA_TRAZA_PUERTA_SR.json'
ELF_SHA = 'c0c3853d2b0e90bb20fe00be13bcf107d186088406be131b729a605ded44ecc7'
CLIENT_SHA = '631303e144b432a7144054150ed6466d8a8b10d1f179484329bc2e2b666d91fb'
BUILD_SHA = '94ca669647e4b9924730febe6513c338277ae401f5cfabf008f4f3b69af7f974'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def jsonlines(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]


def local_manifest(folder, name):
    manifest = folder/name
    digest, declared = manifest.with_suffix('.sha256').read_text(encoding='ascii').split()
    require(declared == name and sha(manifest) == digest, 'Sello invalido: '+name)
    entries = read(manifest)
    for relative, expected in entries.items():
        target = (folder/relative).resolve()
        require(target.is_relative_to(folder.resolve()) and target.is_file() and sha(target) == expected,
                'Entrada de manifiesto modificada: '+relative)
    return digest, entries


class Replay:
    def __init__(self, trace):
        self.trace = trace
        self.pos = 0
        self.transactions = []

    def ask(self, command, prefix, allow_error=False):
        require(self.pos+1 < len(self.trace), 'Traza agotada')
        tx, rx = self.trace[self.pos:self.pos+2]
        require(tx['direction'] == 'TX' and rx['direction'] == 'RX'
                and tx['command'] == command, 'Comando distinto al protocolo reproducido')
        line = rx['line']
        require(len(command.encode('ascii'))+2 < 1024 and len(line.encode('ascii'))+2 < 512,
                'Longitud de trama fuera del contrato')
        require(line.startswith(prefix) and (allow_error or '_ERROR,' not in line),
                'Respuesta/prefijo inesperado: '+line)
        self.transactions.append((tx, rx))
        self.pos += 2
        return line


def main():
    started = time.perf_counter()
    require(not REPORT.exists(), 'No sobrescribir auditoria')
    gate_digest, entries = local_manifest(GATE, 'MANIFEST_EVIDENCIAS_SHA256.json')
    require(len(entries) == 12, 'Puerta no contiene los doce recursos previstos')
    resource_digest, resources = local_manifest(OUT/'recursos', 'MANIFEST_RECURSOS_SHA256.json')
    require(len(resources) == 39, 'Banco de recursos inesperado')
    require(sha(CLIENT) == sha(SOURCE) == sha(GATE/'validar_total_ejecutado.py') == CLIENT_SHA,
            'Cliente distinto del auditado')
    require(sha(GATE/'firmware_declarado.elf') == ELF_SHA
            and sha(GATE/'build_report.json') == sha(OUT/'COMPILACION_SR.json') == BUILD_SHA,
            'ELF o build distintos de SR')
    for name in ('CATALOGO_TOTAL.json', 'CATALOGO_CANONICO.json', 'BANCO_TOTAL.json',
                 'MANIFEST_RECURSOS_SHA256.json', 'MANIFEST_RECURSOS_SHA256.sha256'):
        require(sha(GATE/name) == sha(OUT/'recursos'/name), 'Recurso de captura distinto del original')
    spec = importlib.util.spec_from_file_location('sr_client_trace_readonly', CLIENT)
    require(spec and spec.loader, 'Cliente no importable')
    c = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(c)
    cat, bank, windows, _ = c.load_resources(OUT/'recursos')
    c.verify_gate(GATE, cat, ELF_SHA, resource_digest)
    report, plan = read(GATE/'RESULTADO_TOTAL.json'), read(GATE/'PLAN_EJECUTADO.json')
    expected_plan = [[568, [0]], [232, [22]]]
    require(plan['schedule'] == expected_plan and report['gate_cases'] == expected_plan,
            'Calendario distinto de control y regresion')
    trace, events = jsonlines(GATE/'TRAZA_UART.jsonl'), jsonlines(GATE/'MEDICIONES.jsonl')
    require(len(trace)%2 == 0 and [r['seq'] for r in trace] == list(range(1, len(trace)+1))
            and [r['seq'] for r in events] == list(range(1, len(events)+1)), 'Secuencias no consecutivas')
    dates = [datetime.fromisoformat(r['utc']) for r in trace]
    require(all(d.utcoffset().total_seconds() == 0 for d in dates)
            and all(a <= b for a, b in zip(dates, dates[1:])), 'Fechas no UTC/monotonas')
    link = Replay(trace)
    event_pos = 0

    def event(kind, fields):
        nonlocal event_pos
        require(event_pos < len(events), 'Eventos agotados')
        row = events[event_pos]
        require(row['type'] == kind and all(row[k] == v for k, v in fields.items()),
                'Evento no corresponde a la publicacion UART: '+kind)
        event_pos += 1
        return row

    initial = c.parse_info(link.ask('CWRUTOTAL_INFO', 'CWRUTOTAL_INFO,'), cat)
    require(initial == report['firmware_info'] and initial['last_run_id'] == 0
            and not initial['result_valid'], 'INFO inicial incompatible')
    last = initial['last_run_id']
    raw_reports, base_rows, selected_rows, measured = [], [], [], []
    rows = {r['global_id']: r for r in bank['windows']}
    for gid, ids in expected_plan:
        row = rows[gid]
        window = windows[row['npz_index']]
        metadata = row['metadata']
        # Contraste independiente contra la fila de la matriz fuente, sin entrenar.
        source = (ROOT/metadata['source_windows']).resolve()
        require(source.is_relative_to(ROOT.resolve()), 'Matriz fuente fuera del proyecto')
        original = np.load(source, mmap_mode='r', allow_pickle=False)
        require(np.asarray(original[int(metadata['source_row'])], dtype='<f4').tobytes() == window.tobytes(),
                'RAW distinto de su fila fuente')
        before = link.pos
        c.transmit(link, window, row)
        frames = trace[before:link.pos:2]
        data_frames = [r['command'] for r in frames if r['command'].startswith('CWRUFEAT_DATA,')]
        reconstructed = np.asarray([value for command in data_frames for value in command.split(',')[2:]], dtype='<f4')
        require(len(data_frames) == 171 and reconstructed.shape == (8192,)
                and reconstructed.tobytes() == window.tobytes(), 'RAW UART no conserva los 8192 float32')
        raw_reports.append(dict(global_id=gid, npz_index=row['npz_index'], frames=171, values=8192,
            checksum_fnv1a=row['checksum_fnv1a'], payload_sha256=row['payload_sha256'],
            metadata=metadata, bit_exact_to_authorized_bank=True, bit_exact_to_source_matrix_row=True))
        base = c.baseline(link, row)
        event('BASE', base)
        base_rows.append(base)
        for rid in ids:
            rep = cat['representatives'][rid]
            require(rid in row['authorized_rep_ids'] and gid in rep['scope']['benchmark_global_window_ids']
                    and int(metadata['load_hp']) != rep['outer_load_hp'], 'Uso de carga exterior no autorizado')
            c.validate_model(link.ask(f'CWRUTOTAL_MODEL,{rid}', 'CWRUTOTAL_MODEL,'), rep)
            selected = c.parse_selected(link.ask(f'CWRUTOTAL_SELECT,{rid}', 'CWRUTOTAL_SELECTED,'), rep)
            require(not selected['reused'] and selected['arena_used'] == 1188, 'SELECT inicial inesperado')
            event('SELECT', dict(global_id=gid, **selected))
            selected_rows.append(selected)
            first_pages = None
            for repetition, mode in ((1, 0), (2, 0), (3, 0), (1, 1)):
                command = 'CWRUTOTAL_PROFILE' if mode else 'CWRUTOTAL_RUN'
                result = c.parse_result(link.ask(command+','+row['checksum_fnv1a'], 'CWRUTOTAL_RESULT,'),
                                        rep, row, last, mode, selected['arena_used'])
                last = result['run_id']
                pages = c.get_pages(link, result, rep)
                require(first_pages is None or pages == first_pages, 'Repeticion/PROFILE cambia paginas')
                first_pages = pages
                expected_features = [base['features_bits'][i] if i in rep['feature_indices'] else '00000000'
                                     for i in range(14)]
                require(pages['features_bits'] == expected_features, 'Features no exactas a BASE/slots +0')
                float_replay = c.replay(pages['features_bits'], rep)
                require(all(pages[k] == v for k, v in float_replay.items())
                        and result['score_bits'] == pages['score_bits'] and result['rejected'] == pages['rejected'],
                        'Intermedios float32 o distancia no corresponden a paginas')
                result.update(repetition=repetition, status='PASS', pages=pages)
                saved = event('PROFILE' if mode else 'RUN', result)
                measured.append(result)
                require(saved['comparison']['ARM_features_replay_all_intermediates_bit_exact']
                        and saved['comparison']['selected_features_bit_exact_BASE'], 'Comparacion almacenada incompatible')
    rejects, support = c.negatives(link, measured[-1], cat['representatives'][22])
    event('NEGATIVE_SUPPORT_SELECT', support)
    for rejection in rejects:
        event('NEGATIVE', rejection)
    final = c.parse_info(link.ask('CWRUTOTAL_INFO', 'CWRUTOTAL_INFO,'), cat)
    require(final == report['firmware_info_final'] and final['last_run_id'] == last == 8
            and final['selected_id'] == 22 and not final['result_valid'], 'Negativas alteraron contador/publicacion')
    require(link.pos == len(trace) and event_pos == len(events) == 30, 'Quedan transacciones/eventos sin auditar')
    require(len(measured) == 8 and sum(r['mode'] == 0 for r in measured) == 6
            and sum(r['mode'] == 1 for r in measured) == 2 and len(rejects) == 17, 'Conteos distintos')
    require(c.statistics(measured, cat['representatives']) == report['timing_statistics'], 'Estadisticas no reproducibles')
    machine = read(OUT/'AUDITORIA_MAQUINA_SR.json')
    require(machine['status'] == 'PASS_SR_MACHINE_ROUNDING_REVIEW_PENDING_ARM'
            and machine['elf_sha256'] == ELF_SHA and machine['primary']['read_count'] == 2
            and machine['profile']['read_count'] == 10, 'Alcance temporal de otro ELF')
    durations = [(datetime.fromisoformat(rx['utc'])-datetime.fromisoformat(tx['utc'])).total_seconds()
                 for tx, rx in link.transactions]
    result = dict(status='PASS_SR_GATE_TRACE_REPLAYED_NO_HARDWARE', date_local='2026-10-26', timezone='America/Lima',
        gate=GATE.relative_to(ROOT).as_posix(), gate_manifest_sha256=gate_digest, gate_resources_verified=12,
        resource_manifest_sha256=resource_digest, exported_resources_verified=39, client_sha256=CLIENT_SHA,
        firmware_elf_sha256=ELF_SHA, build_report_sha256=BUILD_SHA,
        trace_sha256=sha(GATE/'TRAZA_UART.jsonl'), measurements_sha256=sha(GATE/'MEDICIONES.jsonl'),
        trace_rows=len(trace), transactions=len(link.transactions), measurement_rows=len(events),
        all_sequences_timestamps_and_TX_RX_pairs_valid=True, entire_trace_consumed=True,
        completed_primary_RUN=6, completed_PROFILE=2, completed_BASE=2, negative_rejections=17,
        valid_extra_RUN=0, RAW=raw_reports, SELECT=selected_rows, final_run_counter=8, final_result_valid=False,
        pages_bind_run_id_model_checksum=True, pages_bit_identical_across_three_RUN_and_PROFILE=True,
        three_invalid_DATA_frames_atomic_at_count4=True, malformed_query_preserves_publication=True,
        invalid_RUN_and_BEGIN_invalidate_publication=True, model_and_parameter_fingerprints_verified=True,
        benchmark_scope_load_authorizations_verified=True,
        historical_source_split_names_not_used_as_claim_of_new_independent_acquisitions=True,
        timing_statistics_exact_to_trace=report['timing_statistics'], PROFILE_excluded_from_primary_statistics=True,
        max_observed_transaction_seconds=max(durations), wall_seconds_reported=report['wall_seconds'],
        timing_scope_machine_audit_sha256=sha(OUT/'AUDITORIA_MAQUINA_SR.json'),
        UART_BASE_SELECT_PROFILE_excluded_from_primary_cost=True,
        firmware_loaded_verified_by_programmer_report=False, programmer_report_attached=report['programmer_report_attached'],
        physical_full_campaign_complete=False, total_cost_validated=False, selector_enabled=False,
        script_sha256=sha(__file__), TFLite_invocations=0, new_physical_RUN=0, D_files_modified=False,
        serial_port_opened=False, hardware_flashed=False, seconds=time.perf_counter()-started,
        limitations=['Only the two predefined model/window gate cases were physically exercised.',
            'The declared ELF and UART fingerprints do not verify the full board Flash image.',
            'Float preprocessing replay and protocol audit do not replace independent neural numerical review.',
            'Timing excludes UART and initialization by audited ELF design; observed timings are not WCET.'])
    with REPORT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({k:result[k] for k in ('status', 'gate_manifest_sha256', 'trace_rows', 'transactions',
        'measurement_rows', 'completed_primary_RUN', 'completed_PROFILE', 'completed_BASE', 'seconds')}, ensure_ascii=True))


if __name__ == '__main__':
    main()
