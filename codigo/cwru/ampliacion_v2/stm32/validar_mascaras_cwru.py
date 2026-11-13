"""Compatibilidad y tiempo de extractores CWRU especializados por mascara.

Primera prueba: --index 0 hace tres BASE EntM historicos y tres RUN F14
nuevos. --all-masks amplia el indice a los diez kernels; --all usa las 42
ventanas de desarrollo. --synthetic usa cuatro estimulos separados y compara
con BASE ARM, sin atribuirles una referencia absoluta PC. --self-test simula
transporte y resultados; no ejecuta C/ARM ni mide tiempos de hardware.
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
HCLK = 180000000
REPETITIONS = 3
RESOURCES = {'validar_mascaras.py', 'validar_pipeline.py', 'REFERENCIA_PIPELINE.json',
             'ventanas_desarrollo.npz', 'metadata_ventanas.csv', 'CATALOGO_MASCARAS.json'}
NAMES = ('F14', 'T6', 'S8', 'NF12', 'NM10', 'NE12', 'DE7', 'FE7', 'FIX_DE13', 'FIX_FE13')
MASKS = (0x3fff, 0x0387, 0x3c78, 0x37ef, 0x3cf9, 0x1fbf, 0x007f, 0x3f80, 0x37ff, 0x3fef)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(folder):
    manifest = json.loads((folder / 'VALIDACION_MANIFEST.json').read_text(encoding='utf-8-sig'))
    require(set(manifest) == RESOURCES, 'Manifiesto de seis recursos requerido')
    # Verify all imported code and references before executing the frozen module.
    for name, digest in manifest.items():
        require(sha(folder / name) == digest, 'Recurso modificado: ' + name)
    spec = importlib.util.spec_from_file_location('frozen_pipeline', folder / 'validar_pipeline.py')
    pc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pc)
    ref, windows, metadata = pc.load_bundle(folder)
    catalog = json.loads((folder / 'CATALOGO_MASCARAS.json').read_text(encoding='utf-8-sig'))
    expected = [(i, name, f'{mask:04x}', [j for j in range(14) if mask & (1 << j)])
                for i, (name, mask) in enumerate(zip(NAMES, MASKS))]
    actual = [(r['id'], r['name'], r['feature_mask'], r['indices']) for r in catalog['kernels']]
    catalog_fnv = pc.fnv(np.asarray(MASKS, dtype='<u2').tobytes())
    require(actual == expected and catalog['catalog_fnv1a'] == catalog_fnv == 'fa1040f2', 'Catalogo incompatible')
    require(catalog.get('selector_costs_enabled') is False, 'El ensayo no habilita seleccion por costo')
    return pc, ref, windows, catalog


def exchange(port, command, prefix, trace, timeout=20., allow_error=False):
    payload = (command + '\r\n').encode('ascii')
    require(len(payload) < 1024, 'TX excede buffer1024')
    trace.append(dict(direction='TX', command=command, bytes=len(payload), utc=datetime.now(timezone.utc).isoformat()))
    require(port.write(payload) == len(payload), 'Escritura UART parcial')
    port.flush()
    deadline, last = time.monotonic() + timeout, []
    while time.monotonic() < deadline:
        raw = port.readline()
        if not raw:
            continue
        line = raw.decode('ascii', errors='strict').rstrip('\r\n')
        if not line:
            continue
        trace.append(dict(direction='RX', line=line, bytes=len(raw), utc=datetime.now(timezone.utc).isoformat()))
        last.append(line)
        require(raw.endswith(b'\r\n') and len(raw) < 512, 'Respuesta truncada o fuera de buffer512')
        if '_ERROR,' in line:
            if allow_error:
                return line
            raise ValueError(line)
        if line.startswith(prefix):
            return line
    raise TimeoutError(f'No llego {prefix}; ultimas respuestas={last[-5:]}')


def parse_info(pc, line, catalog):
    f = line.split(',')
    require(len(f) == 15 and f[0] == 'CWRUMASK_INFO', 'INFO incompleto')
    require([pc.decimal(x) for x in f[1:8]] == [1, 12000, 4096, 2, 14, HCLK, 1], 'INFO/reloj/DWT incompatibles')
    fpscr = int(pc.hex_text(f[8], 8), 16)
    require(not fpscr & ((3 << 22) | (1 << 24)), 'Se requiere redondeo al par y FTZ desactivado')
    require([pc.decimal(x) for x in f[9:12]] == [10, 2, 1]
            and pc.hex_text(f[12], 8) == catalog['catalog_fnv1a'], 'INFO kernels/EntM/cache/catalogo incompatibles')
    valid, last_id = pc.decimal(f[13], 0, 1), pc.decimal(f[14])
    return dict(protocol=1, sample_rate_hz=12000, samples=4096, channels=2, output_slots=14,
                hclk_hz=HCLK, dwt_enabled=True, fpscr=f'{fpscr:08x}', rounding='nearest_even',
                flush_to_zero=False, kernels=10, entropy_mode=2, power_cache=True,
                catalog_fnv1a=f[12].lower(), base_valid=bool(valid), last_id=last_id,
                nominal_hal_clock_not_externally_measured=True)


def info(pc, port, catalog, timeout, trace):
    return parse_info(pc, exchange(port, 'CWRUMASK_INFO', 'CWRUMASK_INFO,', trace, timeout), catalog)


def parse_base(pc, line, case):
    f = line.split(',')
    require(len(f) == 4 and f[0] == 'CWRUMASK_BASE', 'BASE incompleto')
    require(pc.hex_text(f[1], 8) == case['checksum_fnv1a'], 'BASE de otra ventana')
    cycles, bits = pc.decimal(f[2], 1), pc.float_words(f[3], 14)
    result = dict(checksum_fnv1a=f[1].lower(), cycles=cycles, milliseconds=cycles / 180000, features_bits=bits)
    if not case.get('synthetic', False):
        require(bits == case['features_bits'], 'BASE diferente de la referencia EntM sellada')
        result['original_float64_comparison'] = pc.compare_float64(bits, case)
        require(result['original_float64_comparison']['passed'], 'BASE no cumple las tolerancias PC originales')
        result['bit_exact_to_frozen_real_reference'] = True
    else:
        result.update(compatibility_reference_is_canonical_arm_base=True,
                      synthetic_absolute_pc_features_or_bins_claimed=False)
    return result


def parse_result(pc, line, case, kernel, last_id, baseline):
    f = line.split(',')
    require(len(f) == 8 and f[0] == 'CWRUMASK_RESULT', 'RESULT incompleto')
    result_id = pc.decimal(f[1], 1)
    require(result_id == last_id + 1 and pc.hex_text(f[2], 8) == case['checksum_fnv1a'], 'RESULT ID/checksum incorrectos')
    require(pc.decimal(f[3], 0, 9) == kernel['id'] and
            pc.hex_text(f[4], 4) == kernel['feature_mask'], 'Kernel/mascara incorrectos')
    cycles, exact = pc.decimal(f[5], 1), pc.hex_text(f[6], 4)
    bits = pc.float_words(f[7], 14)
    require(exact == kernel['feature_mask'], 'El firmware detecto diferencia en slots seleccionados')
    indices = set(kernel['indices'])
    require(all(word == (baseline[j] if j in indices else '00000000') for j, word in enumerate(bits)),
            'Valores seleccionados distintos o slots omitidos no nulos')
    return dict(id=result_id, checksum_fnv1a=f[2].lower(), kernel_id=kernel['id'],
                kernel_name=kernel['name'], feature_mask=f[4].lower(), cycles=cycles,
                milliseconds=cycles / 180000, exact_mask=exact, features_bits=bits,
                selected_values_bit_exact=True, unselected_slots_positive_zero=True)


def synthetic_cases(pc):
    windows = []
    zeros = np.zeros((4096, 2), dtype='<f4')
    windows.append(('zeros', zeros))
    constant = zeros.copy(); constant[:, 0] = .25; constant[:, 1] = -.5
    windows.append(('constant', constant))
    alternating = zeros.copy()
    alternating[:, 0] = np.where(np.arange(4096) % 2 == 0, .125, -.125)
    alternating[:, 1] = np.where(np.arange(4096) % 2 == 0, .25, -.25)
    windows.append(('nyquist', alternating))
    impulse = zeros.copy(); impulse[0, 0] = .25; impulse[2048, 1] = -.5
    windows.append(('impulse', impulse))
    result = []
    for index, (name, window) in enumerate(windows):
        raw = window.tobytes()
        result.append(dict(index=index, name='synthetic_' + name, synthetic=True, window=window,
                           checksum_fnv1a=pc.fnv(raw), payload_sha256=hashlib.sha256(raw).hexdigest(),
                           metadata=dict(stimulus=name, data_origin='deterministic_stimulus_not_CWRU_record')))
    return result


def timing_statistics(rows, base_rows):
    passed = [r for r in rows if r.get('status') == 'PASS']
    if not passed:
        return None

    def stats(values):
        a = np.asarray(values, dtype=np.float64)
        return dict(count=len(a), min_cycles=int(a.min()), max_cycles=int(a.max()), mean_cycles=float(a.mean()),
                    mean_ms=float(a.mean() / 180000), median_ms=float(np.median(a) / 180000),
                    p95_ms=float(np.percentile(a, 95) / 180000),
                    sample_std_ms=float(a.std(ddof=1) / 180000) if len(a) > 1 else None)

    f14 = {(r['name'], r['repetition']): r['observed']['cycles'] for r in passed if r['kernel_id'] == 0}
    result = {}
    for kernel_id, name in enumerate(NAMES):
        current = [r for r in passed if r['kernel_id'] == kernel_id]
        if not current:
            continue
        value = stats([r['observed']['cycles'] for r in current])
        pairs = [(r['observed']['cycles'], f14[(r['name'], r['repetition'])]) for r in current
                 if (r['name'], r['repetition']) in f14]
        value.update(paired_new_f14_count=len(pairs),
                     mean_paired_ratio_to_new_f14=float(np.mean([a / b for a, b in pairs])) if pairs else None,
                     mean_paired_extraction_reduction_fraction=float(np.mean([1 - a / b for a, b in pairs])) if pairs else None)
        result[name] = value
    baselines = [r['observed']['cycles'] for r in base_rows if r.get('status') == 'PASS']
    return dict(kernels=result, canonical_entm_control=stats(baselines) if baselines else None,
                comparison_denominator='new_specialized_F14_same_harness_paired_by_window_and_repetition',
                historical_instrumented_profile_used_as_denominator=False, timing_scope='feature_extraction_only',
                whole_pipeline_savings_claimed=False, ram_savings_claimed=False, per_kernel_deployed_flash_claimed=False,
                selector_costs_enabled=False, samples_are_repeated_development_windows_not_independent_motors=True)


def negatives(pc, port, trace, checksum, timeout, report):
    wrong_checksum = f'{int(checksum, 16) ^ 1:08x}'
    tests = [('unknown', 'CWRUMASK_BAD', 'UNKNOWN'), ('info_extra', 'CWRUMASK_INFO,extra', 'FORMAT'),
             ('invalid_kernel', f'CWRUMASK_RUN,999,{checksum}', 'MASK_ID'),
             ('nonnumeric_kernel', f'CWRUMASK_RUN,abc,{checksum}', 'FORMAT'),
             ('extra_run_field', f'CWRUMASK_RUN,0,{checksum},extra', 'FORMAT'),
             ('wrong_checksum', f'CWRUMASK_RUN,0,{wrong_checksum}', 'CHECKSUM'),
             ('malformed_base', 'CWRUMASK_BASE,1234567', 'FORMAT'),
             ('missing_base', f'CWRUMASK_RUN,0,{checksum}', 'NO_BASE')]

    def reject(name, command, reason):
        row = dict(name=name, command=command, expected_reason=reason, status='ERROR')
        report['negative_tests'].append(row)
        row['response'] = exchange(port, command, 'CWRUMASK_ERROR,', trace, timeout, allow_error=True)
        require(row['response'] == f'CWRUMASK_ERROR,{reason},-1', 'Rechazo incorrecto: ' + row['response'])
        row['status'] = 'PASS'

    for test in tests:
        reject(*test)
    pc.begin(port, timeout, trace)
    reject('incomplete_run', f'CWRUMASK_RUN,0,{checksum}', 'INCOMPLETE')
    reject('incomplete_base', f'CWRUMASK_BASE,{checksum}', 'INCOMPLETE')


def session(pc, port, catalog, cases, kernels, report, trace, timeout=20., run_negatives=False):
    report['firmware_before'] = info(pc, port, catalog, timeout, trace)
    last_id = report['firmware_before']['last_id']
    for case in cases:
        pc.transmit(port, case['window'], case['checksum_fnv1a'], timeout, trace)
        report['positive_window_transmissions'] += 1
        previous_base = None
        for repetition in range(1, 4):
            row = dict(name=case['name'], index=case['index'], metadata=case['metadata'],
                       synthetic=case.get('synthetic', False), repetition=repetition,
                       checksum_fnv1a=case['checksum_fnv1a'], status='ERROR')
            report['base_results'].append(row)
            row['response_line'] = exchange(port, 'CWRUMASK_BASE,' + case['checksum_fnv1a'], 'CWRUMASK_BASE,', trace, timeout)
            row['observed'] = parse_base(pc, row['response_line'], case)
            baseline = row['observed']['features_bits']
            require(previous_base is None or baseline == previous_base, 'BASE inestable entre repeticiones')
            previous_base = baseline
            row.update(status='PASS', repeated_outputs_stable=True)
        for kernel in kernels:
            previous = None
            for repetition in range(1, 4):
                row = dict(name=case['name'], index=case['index'], metadata=case['metadata'],
                           synthetic=case.get('synthetic', False), repetition=repetition,
                           kernel_id=kernel['id'], kernel_name=kernel['name'],
                           checksum_fnv1a=case['checksum_fnv1a'], status='ERROR')
                report['results'].append(row)
                row['response_line'] = exchange(port, f'CWRUMASK_RUN,{kernel["id"]},{case["checksum_fnv1a"]}',
                                                'CWRUMASK_RESULT,', trace, timeout)
                row['observed'] = parse_result(pc, row['response_line'], case, kernel, last_id, baseline)
                last_id = row['observed']['id']
                current = row['observed']['features_bits']
                require(previous is None or current == previous, 'Mascara inestable entre repeticiones')
                previous = current
                row.update(status='PASS', repeated_outputs_stable=True)
            print(f'{case["name"]}/{kernel["name"]}: PASS; 3 RUN; ultima={row["observed"]["milliseconds"]:.6f} ms', flush=True)
    if run_negatives:
        negatives(pc, port, trace, cases[-1]['checksum_fnv1a'], timeout, report)
    report['firmware_after'] = info(pc, port, catalog, timeout, trace)
    require(report['firmware_after']['last_id'] == last_id and
            report['firmware_after']['base_valid'] == (not run_negatives), 'Estado final incoherente')
    stable = set(report['firmware_before']) - {'fpscr', 'last_id', 'base_valid'}
    require(all(report['firmware_before'][k] == report['firmware_after'][k] for k in stable), 'Entorno cambio durante prueba')
    require(len(report['base_results']) == len(cases) * 3 and len(report['results']) == len(cases) * len(kernels) * 3
            and len(report['negative_tests']) == (10 if run_negatives else 0), 'Suite incompleta')
    report['statistics'] = timing_statistics(report['results'], report['base_results'])
    report['status'] = 'PASS'


def fake_transport(pc, ref, windows, catalog, cases):
    lookup = {c['checksum_fnv1a']: c for c in cases}

    class Fixture(pc.FakeTransport):
        """Explicit result fixture: no numerical C/ARM implementation runs here."""
        def __init__(self):
            super().__init__(ref, windows)
            self.base_bits = None
            self.mask_last_id = 0

        def write(self, payload):
            command = payload.decode('ascii').removesuffix('\r\n')
            if not command.startswith('CWRUMASK_'):
                if command == 'CWRUFEAT_BEGIN' or command.startswith('CWRUFEAT_DATA,') or command == 'CWRUFEAT_RUN':
                    self.base_bits = None
                return super().write(payload)
            f = command.split(',')
            error = lambda reason: f'CWRUMASK_ERROR,{reason},-1'
            checksum = pc.fnv(np.asarray(self.received, dtype='<f4').tobytes())
            if f[0] == 'CWRUMASK_BASE':
                self.base_bits = None
            if command == 'CWRUMASK_INFO':
                reply = f'CWRUMASK_INFO,1,12000,4096,2,14,180000000,1,00000000,10,2,1,{catalog["catalog_fnv1a"]},{int(self.base_bits is not None)},{self.mask_last_id}'
            elif f[0] == 'CWRUMASK_INFO':
                reply = error('FORMAT')
            elif f[0] == 'CWRUMASK_BASE':
                if len(f) != 2 or re.fullmatch('[0-9a-fA-F]{8}', f[1]) is None:
                    reply = error('FORMAT')
                elif not self.active or len(self.received) != 8192:
                    reply = error('INCOMPLETE')
                elif f[1].lower() != checksum:
                    reply = error('CHECKSUM')
                else:
                    case = lookup[checksum]
                    if case.get('synthetic', False):
                        values = np.arange(14, dtype='<f4') / np.float32(32)
                        values[[4, 11]] = np.float32(12000 / 4096)
                        self.base_bits = pc.words(values)
                    else:
                        self.base_bits = case['features_bits']
                    reply = f'CWRUMASK_BASE,{checksum},18000000,{pc.raw_hex(self.base_bits)}'
            elif f[0] == 'CWRUMASK_RUN':
                if len(f) != 3 or re.fullmatch('[0-9]+', f[1]) is None or int(f[1]) > 0xffffffff or re.fullmatch('[0-9a-fA-F]{8}', f[2]) is None:
                    reply = error('FORMAT')
                elif int(f[1]) >= 10:
                    reply = error('MASK_ID')
                elif not self.active or len(self.received) != 8192:
                    reply = error('INCOMPLETE')
                elif f[2].lower() != checksum:
                    reply = error('CHECKSUM')
                elif self.base_bits is None:
                    reply = error('NO_BASE')
                else:
                    kernel = catalog['kernels'][int(f[1])]
                    bits = [self.base_bits[j] if j in kernel['indices'] else '00000000' for j in range(14)]
                    cycles = (18000000, 7200000, 10500000, 17999000, 12000000, 11000000, 9000000, 9000000, 17999500, 17999500)[kernel['id']]
                    self.mask_last_id += 1
                    reply = f'CWRUMASK_RESULT,{self.mask_last_id},{checksum},{kernel["id"]},{kernel["feature_mask"]},{cycles},{kernel["feature_mask"]},{pc.raw_hex(bits)}'
            else:
                reply = error('UNKNOWN')
            self.queue.append((reply + '\r\n').encode('ascii'))
            return len(payload)
    return Fixture()


def corruption_tests(pc, catalog, cases, report):
    tests = []
    case, row = cases[0], report['results'][0]
    baseline = report['base_results'][0]['observed']['features_bits']
    last_id = row['observed']['id'] - 1
    for name, mutate in [('wrong_id', lambda f: f.__setitem__(1, str(last_id + 2))),
                         ('checksum', lambda f: f.__setitem__(2, '00000000')),
                         ('kernel', lambda f: f.__setitem__(3, '1')),
                         ('mask', lambda f: f.__setitem__(4, '0387')),
                         ('zero_cycles', lambda f: f.__setitem__(5, '0')),
                         ('exact_mask', lambda f: f.__setitem__(6, '0000')),
                         ('truncated', lambda f: f.pop()),
                         ('feature_value', lambda f: f.__setitem__(7, '0000803f' + f[7][8:]))]:
        f = row['response_line'].split(','); mutate(f)
        try:
            parse_result(pc, ','.join(f), case, catalog['kernels'][0], last_id, baseline)
        except ValueError:
            tests.append(dict(name=name, status='PASS'))
        else:
            raise RuntimeError('Parser acepto corrupcion: ' + name)
    reduced = next(r for r in report['results'] if r['kernel_id'] == 1)
    f = reduced['response_line'].split(',')
    bits = [baseline[j] if j in catalog['kernels'][1]['indices'] else '00000000' for j in range(14)]
    bits[3] = '3f800000'; f[7] = pc.raw_hex(bits)
    try:
        parse_result(pc, ','.join(f), case, catalog['kernels'][1], reduced['observed']['id'] - 1, baseline)
    except ValueError:
        tests.append(dict(name='unselected_slot_nonzero', status='PASS'))
    else:
        raise RuntimeError('Parser acepto slot omitido distinto de cero')
    for name, mutate in [('checksum', lambda f: f.__setitem__(1, '00000000')),
                         ('zero_cycles', lambda f: f.__setitem__(2, '0')),
                         ('truncation', lambda f: f.pop()),
                         ('nonfinite_value', lambda f: f.__setitem__(3, '0000807f' + f[3][8:]))]:
        f = report['base_results'][0]['response_line'].split(','); mutate(f)
        try:
            parse_base(pc, ','.join(f), case)
        except ValueError:
            tests.append(dict(name='base_' + name, status='PASS'))
        else:
            raise RuntimeError('BASE acepto corrupcion: ' + name)
    for name, index, value in [('clock', 6, '16000000'), ('ftz', 8, '01000000'),
                               ('mode', 10, '1'), ('catalog', 12, '00000000')]:
        f = f'CWRUMASK_INFO,1,12000,4096,2,14,180000000,1,00000000,10,2,1,{catalog["catalog_fnv1a"]},0,0'.split(',')
        f[index] = value
        try:
            parse_info(pc, ','.join(f), catalog)
        except ValueError:
            tests.append(dict(name='info_' + name, status='PASS'))
        else:
            raise RuntimeError('INFO acepto corrupcion: ' + name)
    return tests


def save(folder, report, trace):
    report.update(utc_finished=datetime.now(timezone.utc).isoformat(), client_sha256=sha(Path(__file__)),
                  completed_runs=sum(r.get('status') == 'PASS' for r in report['results']),
                  attempted_runs=len(report['results']),
                  completed_base_extractions=sum(r.get('status') == 'PASS' for r in report['base_results']),
                  attempted_base_extractions=len(report['base_results']),
                  negative_tests_passed=sum(r.get('status') == 'PASS' for r in report['negative_tests']),
                  candidate_run_commands_sent=sum(r['direction'] == 'TX' and r['command'].startswith('CWRUMASK_RUN,') for r in trace),
                  base_commands_sent=sum(r['direction'] == 'TX' and r['command'].startswith('CWRUMASK_BASE,') for r in trace),
                  numpy=np.__version__, python=sys.version)
    report['statistics'] = timing_statistics(report['results'], report['base_results'])
    for name, value in [('resultados.json', report), ('uart_trace.jsonl', trace)]:
        with (folder / name).open('x', encoding='utf-8', newline='\n') as stream:
            if name.endswith('.jsonl'):
                for row in value:
                    stream.write(json.dumps(row, ensure_ascii=False) + '\n')
            else:
                json.dump(value, stream, ensure_ascii=False, indent=2); stream.write('\n')
    with (folder / 'MANIFEST_SHA256.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump({name: sha(folder / name) for name in ('resultados.json', 'uart_trace.jsonl')}, stream, indent=2)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port')
    parser.add_argument('--self-test', action='store_true')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--index', type=int)
    group.add_argument('--all', action='store_true')
    group.add_argument('--synthetic', action='store_true')
    parser.add_argument('--all-masks', action='store_true')
    parser.add_argument('--repeat', type=int, default=3)
    parser.add_argument('--timeout', type=float, default=20.)
    parser.add_argument('--bundle-dir', type=Path, default=HERE)
    parser.add_argument('--output-root', type=Path)
    args = parser.parse_args()
    if args.repeat != 3 or args.timeout <= 0 or (not args.self_test and not args.port):
        parser.error('Se requiere --port, tres repeticiones y timeout positivo')
    if args.index is not None and not 0 <= args.index < 42:
        parser.error('Indice fuera de0..41')
    full = args.all or args.synthetic or args.self_test
    all_masks = args.all_masks or full
    mode = 'synthetic' if args.synthetic else ('all' if args.all or args.self_test else 'single')
    folder = (args.output_root or args.bundle_dir / 'resultados') / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    folder.mkdir(parents=True, exist_ok=False)
    report = dict(status='ERROR', mode=mode, repetitions=3, results=[], base_results=[], negative_tests=[],
                  simulation_only=args.self_test, firmware_executed=False, serial_port_opened=False,
                  training_executed=False, model_conversion_executed=False, detector_refitted=False,
                  dnn_executed=False, detector_executed=False, outer_test_loaded=False, protocol_frozen=False,
                  new_predictive_metrics_computed=False, selector_costs_enabled=False, whole_pipeline_savings_claimed=False,
                  ram_savings_claimed=False, per_kernel_deployed_flash_claimed=False,
                  wcet_claim=False, real_time_guarantee=False, canonical_latency_replaced=False,
                  sample_origin='deterministic_stimuli' if args.synthetic else '42_fixed_development_windows_0_1_2HP',
                  timing_scope='feature_extraction_only', output_clearing_and_comparison_outside_timing=True,
                  positive_window_transmissions=0, expected_negative_tests=10 if full else 0,
                  no_auxiliary_candidate_runs=True, nominal_hal_hclk_hz=HCLK,
                  utc_started=datetime.now(timezone.utc).isoformat())
    trace = []
    try:
        pc, ref, windows, catalog = load(args.bundle_dir)
        for key, name in [('reference_sha256', 'REFERENCIA_PIPELINE.json'),
                          ('bundle_manifest_sha256', 'VALIDACION_MANIFEST.json'),
                          ('catalog_sha256', 'CATALOGO_MASCARAS.json')]:
            report[key] = sha(args.bundle_dir / name)
        if args.synthetic:
            cases = synthetic_cases(pc)
        else:
            selected = ref['cases'] if mode == 'all' else [ref['cases'][0 if args.index is None else args.index]]
            cases = [dict(c, synthetic=False, window=windows[c['index']]) for c in selected]
        kernels = catalog['kernels'] if all_masks else catalog['kernels'][:1]
        report.update(selected_kernel_ids=[k['id'] for k in kernels], selected_kernel_names=[k['name'] for k in kernels],
                      windows_expected=len(cases), positive_expected_base_extractions=3 * len(cases),
                      positive_expected_runs=3 * len(cases) * len(kernels),
                      synthetic_absolute_pc_features_or_bins_claimed=False)
        if args.self_test:
            report.update(synthetic_cycle_counts_not_hardware_measurement=True, self_test_executes_c_arm=False)
            session(pc, fake_transport(pc, ref, windows, catalog, cases), catalog, cases, kernels, report, trace,
                    timeout=args.timeout, run_negatives=True)
            report['self_test_parser_corruption_checks'] = corruption_tests(pc, catalog, cases, report)
            print('SIMULACION EXPLICITA: sin C/ARM, COM, DWT ni tiempos fisicos.', flush=True)
        else:
            import serial
            with serial.Serial(args.port, 115200, timeout=.2, write_timeout=args.timeout) as port:
                report['serial_port_opened'] = True
                time.sleep(.2); port.reset_input_buffer()
                report['firmware_executed'] = True
                session(pc, port, catalog, cases, kernels, report, trace, timeout=args.timeout, run_negatives=full)
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
