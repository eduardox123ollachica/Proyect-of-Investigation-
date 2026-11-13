"""Diagnóstico de siete grupos compartidos del extractor F14 EntM.

--index 0: un envío y tres PIPE_RUN; --all: 42 envíos, 126 PIPE_RUN y
ocho rechazos, sin RUN auxiliares. --self-test usa resultados sintéticos:
no ejecuta C/ARM, no abre COM ni mide tiempos físicos. Los grupos fusionados
no identifican costos marginales de características ni ahorros de máscaras.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BLOCKS = ('sumas_rms_media', 'momentos_centrado', 'correcciones_momentos',
          'fft', 'energia_dominante_cache', 'entropia', 'salida')
RESOURCES = {'validar_coste.py', 'validar_pipeline.py', 'REFERENCIA_PIPELINE.json',
             'ventanas_desarrollo.npz', 'metadata_ventanas.csv', 'GRAFO_DEPENDENCIAS.json'}
HCLK = 180000000
REPETITIONS = 3


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(folder):
    manifest = json.loads((folder / 'VALIDACION_MANIFEST.json').read_text(encoding='utf-8-sig'))
    require(set(manifest) == RESOURCES, 'Manifiesto de seis recursos requerido')
    # Authenticate the frozen imported code before executing it.
    for name, expected in manifest.items():
        require(digest(folder / name) == expected, 'Recurso modificado: ' + name)
    spec = importlib.util.spec_from_file_location('frozen_pipeline', folder / 'validar_pipeline.py')
    pc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pc)
    ref, windows, metadata = pc.load_bundle(folder)
    graph = json.loads((folder / 'GRAFO_DEPENDENCIAS.json').read_text(encoding='utf-8-sig'))
    require(graph.get('selector_costs_enabled') is False,
            'Este perfil no autoriza costos de máscaras para el selector')
    return pc, ref, windows


def exchange(port, command, prefix, trace, timeout=20., allow_error=False):
    """COST-aware exchange; frozen PIPE client remains unmodified."""
    payload = (command + '\r\n').encode('ascii')
    require(len(payload) < 1024, 'Trama TX fuera de buffer')
    trace.append(dict(direction='TX', command=command, bytes=len(payload),
                      utc=datetime.now(timezone.utc).isoformat()))
    require(port.write(payload) == len(payload), 'Escritura UART parcial')
    port.flush()
    deadline = time.monotonic() + timeout
    last = []
    while time.monotonic() < deadline:
        raw = port.readline()
        if not raw:
            continue
        line = raw.decode('ascii', errors='strict').rstrip('\r\n')
        if not line:
            continue
        trace.append(dict(direction='RX', line=line, bytes=len(raw),
                          utc=datetime.now(timezone.utc).isoformat()))
        last.append(line)
        require(raw.endswith(b'\r\n') and len(raw) < 512, 'Respuesta truncada o fuera de buffer')
        if '_ERROR,' in line:
            if allow_error:
                return line
            raise ValueError(line)
        if line.startswith(prefix):
            return line
    raise TimeoutError(f'No llegó {prefix}; últimas respuestas={last[-5:]}')


def parse_info(pc, line):
    f = line.split(',')
    require(len(f) == 11 and f[0] == 'CWRUCOST_INFO', 'INFO de costos incompleto')
    v = [pc.decimal(x) for x in f[1:]]
    require(v[:7] == [1, 7, 2, HCLK, 1, 2, 1] and v[8:] == [0, 64]
            and 0 < v[7] < HCLK, 'INFO/reloj/DWT/EntM/caché incompatibles')
    return dict(protocol=1, blocks=7, channels=2, hclk_hz=HCLK, dwt_enabled=True,
                entropy_mode=2, power_cache=True, paired_read_cycles_min=v[7],
                overhead_subtracted=False, paired_read_calibration_samples=64,
                nominal_hal_clock_not_externally_measured=True)


def cost_info(pc, port, trace, timeout):
    return parse_info(pc, exchange(port, 'CWRUCOST_INFO', 'CWRUCOST_INFO,', trace, timeout))


def parse_profile(pc, line, result):
    f = line.split(',')
    require(len(f) == 19 and f[0] == 'CWRUCOST_GET', 'GET de costos incompleto')
    require(pc.decimal(f[1], 1) == result['id'] and
            pc.hex_text(f[2], 8) == result['checksum_fnv1a'], 'Perfil de otro ID/checksum')
    require(f[3:5] == ['7', '2'], 'Dimensiones de perfil incompatibles')
    cycles = [pc.decimal(x, 1) for x in f[5:]]
    summed = sum(cycles)
    require(summed <= result['extract_cycles'], 'Suma perfil excede extracción envolvente')
    residual = result['extract_cycles'] - summed
    return dict(id=result['id'], checksum_fnv1a=result['checksum_fnv1a'],
                cycles_by_channel={'DE': cycles[:7], 'FE': cycles[7:]},
                measured_group_sum_cycles=summed, extraction_envelope_cycles=result['extract_cycles'],
                extraction_unassigned_residual_cycles=residual,
                measured_group_sum_ms=summed / 180000,
                extraction_unassigned_residual_ms=residual / 180000,
                group_order=list(BLOCKS), overhead_subtracted=False,
                marginal_feature_costs_identifiable=False)


def profile(pc, port, trace, result, timeout):
    line = exchange(port, f'CWRUCOST_GET,{result["id"]}', 'CWRUCOST_GET,', trace, timeout)
    return line, parse_profile(pc, line, result)


def negatives(pc, port, trace, result, timeout, report):
    last_id = result['id']
    tests = [
        ('stale_id', f'CWRUCOST_GET,{last_id + 1}', 'STALE_ID'),
        ('unknown_command', 'CWRUCOST_BAD', 'UNKNOWN'),
        ('info_extra', 'CWRUCOST_INFO,extra', 'FORMAT'),
        ('zero_id', 'CWRUCOST_GET,0', 'FORMAT'),
        ('extra_get_field', f'CWRUCOST_GET,{last_id},extra', 'FORMAT'),
        ('nonnumeric_id', 'CWRUCOST_GET,abc', 'FORMAT'),
        ('bare_get', 'CWRUCOST_GET', 'FORMAT'),
    ]
    for name, command, reason in tests:
        row = dict(name=name, command=command, expected_reason=reason, status='ERROR')
        report['negative_tests'].append(row)
        row['response'] = exchange(port, command, 'CWRUCOST_ERROR,', trace, timeout, allow_error=True)
        require(row['response'] == f'CWRUCOST_ERROR,{reason},-1', 'Rechazo incorrecto: ' + row['response'])
        row['status'] = 'PASS'
    _, still_valid = profile(pc, port, trace, result, timeout)
    require(still_valid == report['results'][-1]['profile'], 'GET inválidos alteraron el perfil válido')
    report['invalid_queries_preserved_publication'] = True
    pc.begin(port, timeout, trace)
    row = dict(name='begin_invalidates', command=f'CWRUCOST_GET,{last_id}',
               expected_reason='NO_RESULT', status='ERROR')
    report['negative_tests'].append(row)
    row['response'] = exchange(port, row['command'], 'CWRUCOST_ERROR,', trace, timeout, allow_error=True)
    require(row['response'] == 'CWRUCOST_ERROR,NO_RESULT,-1', 'BEGIN no invalidó perfil')
    row['status'] = 'PASS'


def diagnostic_statistics(results):
    rows = [r for r in results if r.get('status') == 'PASS' and 'profile' in r]
    if not rows:
        return None

    def stats(values):
        a = np.asarray(values, dtype=np.float64)
        return dict(count=len(a), min_cycles=int(a.min()), max_cycles=int(a.max()),
                    mean_cycles=float(a.mean()), mean_ms=float(a.mean() / 180000),
                    median_ms=float(np.median(a) / 180000), p95_ms=float(np.percentile(a, 95) / 180000))

    by_channel = {ch: {name: stats([r['profile']['cycles_by_channel'][ch][i] for r in rows])
                       for i, name in enumerate(BLOCKS)} for ch in ('DE', 'FE')}
    combined = {}
    for i, name in enumerate(BLOCKS):
        v = [r['profile']['cycles_by_channel']['DE'][i] +
             r['profile']['cycles_by_channel']['FE'][i] for r in rows]
        combined[name] = stats(v)
    denominator = sum(v['mean_cycles'] for v in combined.values())
    for value in combined.values():
        value['share_of_measured_groups_mean_percent'] = 100 * value['mean_cycles'] / denominator
    return dict(groups_by_channel=by_channel, groups_DE_plus_FE=combined,
                extraction_envelope=stats([r['observed']['extract_cycles'] for r in rows]),
                pipeline_total_instrumented=stats([r['observed']['total_cycles'] for r in rows]),
                extraction_unassigned_residual=stats([r['profile']['extraction_unassigned_residual_cycles'] for r in rows]),
                timing_is_diagnostic=True, canonical_latency_replaced=False,
                operational_latency_limit_applied=False, no_candidate_mask_latency_predictions=True,
                shared_fused_groups_are_not_marginal_feature_costs=True)


def session(pc, port, ref, windows, selected, report, trace, timeout=20., all_cases=False):
    report['pipeline_before'] = pc.info(port, ref, timeout, trace)
    report['initialization'] = pc.initialize(port, ref, timeout, trace)
    report['reuse'] = pc.initialize(port, ref, timeout, trace)
    require(report['reuse']['reused'] and report['reuse']['nn_init_cycles'] == 0, 'INIT no reutilizó la DNN')
    report['pipeline_info'] = pc.info(port, ref, timeout, trace)
    require(report['pipeline_info']['nn_status'] == 0 and
            report['pipeline_info']['arena_used'] == ref['expected_arena_used'], 'DNN no inicializada')
    report['cost_info'] = cost_info(pc, port, trace, timeout)
    last_id = report['pipeline_info']['last_id']
    for case in selected:
        pc.transmit(port, windows[case['index']], case['checksum_fnv1a'], timeout, trace)
        report['positive_window_transmissions'] += 1
        previous = None
        for repetition in range(1, REPETITIONS + 1):
            row = dict(index=case['index'], name=case['name'], metadata=case['metadata'],
                       checksum_fnv1a=case['checksum_fnv1a'], repetition=repetition, status='ERROR')
            report['results'].append(row)
            last_id = pc.execute(port, case, ref, last_id, timeout, trace, row)
            # pc.execute establishes numeric compatibility only. The diagnostic
            # profile must also pass before the row may count as a completed run.
            row['status'] = 'ERROR'
            stable = previous is None or previous == row['observed_pages']
            row['repeated_numeric_outputs_stable'] = stable
            require(stable, 'La instrumentación alteró resultados entre repeticiones')
            previous = row['observed_pages']
            line = exchange(port, f'CWRUCOST_GET,{last_id}', 'CWRUCOST_GET,', trace, timeout)
            row['profile_line'] = line
            row['profile'] = parse_profile(pc, line, row['observed'])
            row['status'] = 'PASS'
        v = row['profile']['cycles_by_channel']
        major = int(np.argmax(np.asarray(v['DE']) + np.asarray(v['FE'])))
        print(f'{case["name"]}: PASS; 3 RUN; extracción instrumentada última='
              f'{row["observed"]["extract_cycles"]/180000:.6f} ms; mayor grupo={BLOCKS[major]}', flush=True)
    if all_cases:
        negatives(pc, port, trace, report['results'][-1]['observed'], timeout, report)
    report['pipeline_after'] = pc.info(port, ref, timeout, trace)
    report['cost_info_after'] = cost_info(pc, port, trace, timeout)
    require(report['pipeline_after']['last_id'] == last_id, 'Contador final incoherente')
    require(report['pipeline_after']['result_valid'] == (not all_cases),
            'Publicacion final incoherente tras GET o BEGIN')
    require(report['cost_info_after']['hclk_hz'] == report['cost_info']['hclk_hz'], 'Reloj cambió')
    require(len(report['results']) == len(selected) * 3 and
            len(report['negative_tests']) == (8 if all_cases else 0), 'Suite incompleta')
    report['diagnostic_statistics'] = diagnostic_statistics(report['results'])
    report['status'] = 'PASS'


def fake_transport(pc, ref, windows):
    class Fixture(pc.FakeTransport):
        """Synthetic COST pages paired with the frozen synthetic PIPE result."""
        def write(self, payload):
            command = payload.decode('ascii').removesuffix('\r\n')
            if not command.startswith('CWRUCOST_'):
                return super().write(payload)
            f = command.split(',')
            error = lambda reason: f'CWRUCOST_ERROR,{reason},-1'
            if command == 'CWRUCOST_INFO':
                reply = 'CWRUCOST_INFO,1,7,2,180000000,1,2,1,13,0,64'
            elif f[0] == 'CWRUCOST_INFO':
                reply = error('FORMAT')
            elif f[0] == 'CWRUCOST_GET':
                if len(f) != 2 or re.fullmatch('[0-9]+', f[1]) is None or not 0 < int(f[1]) <= 0xffffffff:
                    reply = error('FORMAT')
                elif self.result is None:
                    reply = error('NO_RESULT')
                elif int(f[1]) != self.last_id:
                    reply = error('STALE_ID')
                else:
                    counts = [500000, 1200000, 10000, 300000, 400000, 6000000, 1000] * 2
                    reply = f'CWRUCOST_GET,{self.last_id},{self.result["checksum_fnv1a"]},7,2,' + ','.join(map(str, counts))
            else:
                reply = error('UNKNOWN')
            self.queue.append((reply + '\r\n').encode('ascii'))
            return len(payload)
    return Fixture(ref, windows)


def parser_tests(pc, report):
    row = report['results'][0]
    tests = []
    for name, mutate in [
            ('wrong_id', lambda f: f.__setitem__(1, str(row['observed']['id'] + 1))),
            ('wrong_checksum', lambda f: f.__setitem__(2, '00000000')),
            ('truncation', lambda f: f.pop()),
            ('zero_cycles', lambda f: f.__setitem__(5, '0')),
            ('sum_exceeds_envelope', lambda f: f.__setitem__(5, str(row['observed']['extract_cycles']))),
            ('wrong_blocks', lambda f: f.__setitem__(3, '6'))]:
        f = row['profile_line'].split(',')
        mutate(f)
        try:
            parse_profile(pc, ','.join(f), row['observed'])
        except ValueError:
            tests.append(dict(name=name, status='PASS'))
        else:
            raise RuntimeError('Parser aceptó corrupción: ' + name)
    for name, index, value in [('clock', 4, '16000000'), ('mode', 6, '1'),
                               ('cache', 7, '0'), ('subtraction', 9, '1')]:
        f = 'CWRUCOST_INFO,1,7,2,180000000,1,2,1,13,0,64'.split(',')
        f[index] = value
        try:
            parse_info(pc, ','.join(f))
        except ValueError:
            tests.append(dict(name='info_' + name, status='PASS'))
        else:
            raise RuntimeError('INFO aceptó corrupción: ' + name)
    return tests


def save(folder, report, trace):
    report.update(utc_finished=datetime.now(timezone.utc).isoformat(),
                  completed_runs=sum(r.get('status') == 'PASS' for r in report['results']),
                  attempted_runs=len(report['results']),
                  negative_tests_passed=sum(r.get('status') == 'PASS' for r in report['negative_tests']),
                  client_sha256=digest(Path(__file__)),
                  pipeline_run_commands_sent=sum(t['direction'] == 'TX' and
                      t['command'].startswith('CWRUPIPE_RUN,') for t in trace))
    report['diagnostic_statistics'] = diagnostic_statistics(report['results'])
    with (folder / 'resultados.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    with (folder / 'uart_trace.jsonl').open('x', encoding='utf-8', newline='\n') as stream:
        for row in trace:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    values = {name: digest(folder / name) for name in ('resultados.json', 'uart_trace.jsonl')}
    with (folder / 'MANIFEST_SHA256.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(values, stream, indent=2)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port')
    parser.add_argument('--self-test', action='store_true')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--index', type=int)
    group.add_argument('--all', action='store_true')
    parser.add_argument('--repeat', type=int, default=3)
    parser.add_argument('--timeout', type=float, default=20.)
    parser.add_argument('--bundle-dir', type=Path, default=HERE)
    parser.add_argument('--output-root', type=Path)
    args = parser.parse_args()
    if args.repeat != 3 or args.timeout <= 0 or (not args.self_test and not args.port):
        parser.error('Se requiere --port, tres repeticiones y timeout positivo')
    if args.index is not None and not 0 <= args.index < 42:
        parser.error('Índice fuera de0..41')
    all_cases = args.all or args.self_test
    folder = (args.output_root or args.bundle_dir / 'resultados') / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    folder.mkdir(parents=True, exist_ok=False)
    report = dict(status='ERROR', mode='all' if all_cases else 'single', results=[], negative_tests=[],
                  simulation_only=args.self_test, firmware_executed=False, serial_port_opened=False,
                  training_executed=False, model_conversion_executed=False, threshold_refitted=False,
                  outer_test_loaded=False, protocol_frozen=False, new_predictive_metrics_computed=False,
                  wcet_claim=False, real_time_guarantee=False, canonical_latency_replaced=False,
                  selector_costs_enabled=False, marginal_feature_costs_identifiable=False,
                  operational_latency_limit_applied=False, group_order=list(BLOCKS), repetitions=3,
                  positive_expected_runs=126 if all_cases else 3, expected_negative_tests=8 if all_cases else 0,
                  expected_auxiliary_runs=0, positive_window_transmissions=0,
                  overhead_subtracted=False, measurement='Seven fused shared groups, two sensor positions DE/FE',
                  exclusions=['UART', 'acquisition', 'offline FIR 48-to-12kHz', 'initialization'],
                  utc_started=datetime.now(timezone.utc).isoformat(), python=sys.version, numpy=np.__version__)
    trace = []
    try:
        pc, ref, windows = load(args.bundle_dir)
        for name, filename in [('reference_sha256', 'REFERENCIA_PIPELINE.json'),
                               ('bundle_manifest_sha256', 'VALIDACION_MANIFEST.json'),
                               ('dependency_graph_sha256', 'GRAFO_DEPENDENCIAS.json')]:
            report[name] = digest(args.bundle_dir / filename)
        selected = ref['cases'] if all_cases else [ref['cases'][0 if args.index is None else args.index]]
        if args.self_test:
            report['synthetic_cycle_counts_not_hardware_measurement'] = True
            report['self_test_executes_c_arm'] = False
            session(pc, fake_transport(pc, ref, windows), ref, windows, selected, report, trace,
                    timeout=args.timeout, all_cases=True)
            report['self_test_parser_corruption_checks'] = parser_tests(pc, report)
            print('SIMULACIÓN EXPLÍCITA: sin C/ARM, COM, DWT ni tiempos físicos.', flush=True)
        else:
            import serial
            with serial.Serial(args.port, 115200, timeout=.2, write_timeout=args.timeout) as port:
                report['serial_port_opened'] = True
                time.sleep(.2)
                port.reset_input_buffer()
                report['firmware_executed'] = True
                session(pc, port, ref, windows, selected, report, trace,
                        timeout=args.timeout, all_cases=args.all)
    except Exception as error:
        report['status'] = 'ERROR'
        report['error'] = f'{type(error).__name__}: {error}'
        print('ERROR:', report['error'], file=sys.stderr)
    finally:
        save(folder, report, trace)
        print('Evidencias:', folder, flush=True)
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
