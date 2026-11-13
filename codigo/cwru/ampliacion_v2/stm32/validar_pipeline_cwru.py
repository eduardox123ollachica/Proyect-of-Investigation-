"""Cliente CWRUPIPE v1 autocontenido (NumPy; pyserial sólo en prueba física).

Bundle: este cliente, REFERENCIA_PIPELINE.json, ventanas_desarrollo.npz,
metadata_ventanas.csv y VALIDACION_MANIFEST.json. --index 0 ejecuta tres RUN
con un único envío; --all ejecuta 42x3 y 24 negativas con dos RUN auxiliares.
--self-test simula transporte y resultados: NO ejecuta C ARM ni mide hardware.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
MODEL_SHA = 'cabc6e967b91691287c0cd8b899958d6784f0712b9bc956f68565273c9652596'
INPUT_SHA = '24275e363e29573e7a158d44b00099d2224be9a783ac87654f2e1575fc86f1a5'
REPETITIONS = 3
NEGATIVE_COUNT = 24
HOP_MS = 170.666667
LIMIT_MS = 136.533333


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fnv(data):
    value = 2166136261
    for byte in data:
        value = ((value ^ byte) * 16777619) & 0xffffffff
    return f'{value:08x}'


def words(values):
    return [f'{int(v):08x}' for v in np.asarray(values, dtype='<f4').view('<u4').ravel()]


def raw_hex(bits):
    return np.asarray([int(v, 16) for v in bits], dtype='<u4').tobytes().hex()


def hex_text(text, digits):
    require(re.fullmatch('[0-9a-fA-F]{' + str(digits) + '}', text) is not None, 'Campo hexadecimal inválido')
    return text.lower()


def decimal(text, minimum=0, maximum=0xffffffff):
    require(re.fullmatch('[0-9]+', text) is not None, 'Entero decimal inválido')
    value = int(text)
    require(minimum <= value <= maximum, 'Entero fuera de rango')
    return value


def float_words(text, count):
    raw = bytes.fromhex(hex_text(text, count * 8))
    values = np.frombuffer(raw, dtype='<f4')
    require(np.isfinite(values).all(), 'Intermedio no finito')
    return words(values)


def load_bundle(folder):
    manifest = json.loads((folder / 'VALIDACION_MANIFEST.json').read_text(encoding='utf-8-sig'))
    required = {'REFERENCIA_PIPELINE.json', 'ventanas_desarrollo.npz', 'metadata_ventanas.csv', 'validar_pipeline.py'}
    require(required.issubset(manifest), 'Manifiesto no sella los cuatro archivos del cliente')
    for name, digest in manifest.items():
        require(Path(name).name == name and sha(folder / name) == digest, 'Recurso modificado: ' + name)
    ref = json.loads((folder / 'REFERENCIA_PIPELINE.json').read_text(encoding='utf-8-sig'))
    require(ref['schema_version'] == 1 and ref['status'] == 'PC_REFERENCE_READY_NOT_ARM_INTEGRATED' and
            len(ref['cases']) == 42 and [c['index'] for c in ref['cases']] == list(range(42)), 'Referencia incompatible')
    require(ref['model_sha256'] == MODEL_SHA and ref['input_sha256'] == INPUT_SHA and
            ref['model_fnv1a'] == 'a3f03713' and ref['detector_fnv1a'] == '555ede8e' and
            ref['threshold_bits'] == '439af637' and ref['model_bytes'] == 4064 and ref['arena_capacity'] == 16384,
            'Artefactos diferentes del contrato')
    require(ref['policy']['rtol'] == .001 and ref['policy']['atol'] == .00001 and
            ref['policy']['repetitions'] == REPETITIONS and ref['policy']['operational_limit_ms'] == LIMIT_MS and
            ref['policy']['hop_ms'] == HOP_MS, 'Criterio modificado')
    require(sha(folder / 'ventanas_desarrollo.npz') == ref['windows_sha256'] and
            sha(folder / 'metadata_ventanas.csv') == ref['metadata_sha256'], 'Bundle diferente de la referencia')
    with np.load(folder / 'ventanas_desarrollo.npz', allow_pickle=False) as data:
        windows, original = data['windows'].copy(), data['features'].copy()
    with (folder / 'metadata_ventanas.csv').open(encoding='utf-8-sig', newline='') as stream:
        metadata = list(csv.DictReader(stream))
    validate_cases(ref, windows, metadata, original)
    return ref, windows, metadata


def validate_cases(ref, windows, metadata, original=None):
    require(windows.shape == (42, 4096, 2) and len(metadata) == 42 and np.isfinite(windows).all(), '42 ventanas finitas requeridas')
    for case in ref['cases']:
        index = case['index']
        payload = np.asarray(windows[index], dtype='<f4').tobytes()
        require(case['metadata'] == metadata[index] and int(metadata[index]['load_hp']) in (0, 1, 2) and
                int(metadata[index]['bundle_index']) == index and fnv(payload) == case['checksum_fnv1a'] and
                hashlib.sha256(payload).hexdigest() == case['payload_sha256'] == metadata[index]['window_payload_sha256'],
                'Ventana/metadatos/checksum no pareados')
        f = np.frombuffer(bytes.fromhex(raw_hex(case['features_bits'])), dtype='<f4')
        require(len(f) == 14 and np.isfinite(f).all(), '14 características finitas requeridas')
        require(len(case['input_expected']['quantized']) == 14 and
                case['dnn_expected']['output_int8'] == case['dnn_pc_primary']['output_int8'] and
                case['dnn_expected']['predicted_class'] == case['dnn_pc_primary']['predicted_class'], 'Referencia DNN no exacta')
        if original is not None:
            require(np.array_equal(original[index], np.asarray(case['features_float64_expected'])), 'Referencia float64 original distinta')
        comparison = compare_float64(case['features_bits'], case)
        require(comparison['passed'], 'Referencia EntM no pasa contraste float64 original')


def exchange(port, command, prefix, timeout, trace, allow_error=False):
    payload = (command + '\r\n').encode('ascii')
    require(len(payload) < 1024, 'Trama TX excede RX1024')
    trace.append(dict(direction='TX', command=command, bytes=len(payload), utc=datetime.now(timezone.utc).isoformat()))
    if port.write(payload) != len(payload):
        raise IOError('Escritura UART incompleta')
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
        trace.append(dict(direction='RX', line=line, bytes=len(raw), utc=datetime.now(timezone.utc).isoformat()))
        last.append(line)
        require(raw.endswith(b'\r\n') and len(raw) < 512, 'Respuesta sin CRLF o excede buffer main512')
        if line.startswith(('CWRUPIPE_ERROR,', 'CWRUFEAT_ERROR,')):
            if allow_error:
                return line
            raise RuntimeError(line)
        if line.startswith(prefix):
            return line
    raise TimeoutError(f'No llegó {prefix}; últimas respuestas={last[-5:]}')


def info(port, ref, timeout, trace):
    f = exchange(port, 'CWRUPIPE_INFO', 'CWRUPIPE_INFO,', timeout, trace).split(',')
    require(len(f) == 20 and tuple(map(int, f[1:8])) == (1, 12000, 4096, 2, 14, 180000000, 1), 'INFO/reloj/DWT incompatibles')
    fpscr = int(hex_text(f[8], 8), 16)
    require(not fpscr & ((3 << 22) | (1 << 24)), 'FPSCR requiere redondeo al par y FTZ desactivado')
    require(f[9:16] == ['4064', 'a3f03713', MODEL_SHA, INPUT_SHA, '555ede8e', '439af637', '16384'],
            'INFO: huellas, parámetros o arena diferentes')
    arena, valid, last_id = decimal(f[16], 0, 16384), decimal(f[18], 0, 1), decimal(f[19])
    require(not valid or last_id > 0, 'Publicación válida con ID cero')
    return dict(protocol=1, hclk_hz=180000000, dwt_enabled=True, fpscr=f'{fpscr:08x}',
        rounding='nearest_even', flush_to_zero=False, model_fnv1a=f[10], model_sha256_declared=f[11],
        input_sha256_declared=f[12], detector_fnv1a=f[13], threshold_bits=f[14], arena_capacity=16384,
        arena_used=arena, nn_status=int(f[17]), result_valid=bool(valid), last_id=last_id,
        nominal_hal_clock_not_externally_measured=True, full_flash_read=False)


def initialize(port, ref, timeout, trace):
    f = exchange(port, 'CWRUPIPE_INIT', 'CWRUPIPE_READY,', timeout, trace).split(',')
    require(len(f) == 5, 'INIT incompleto')
    cycles, arena, reused, extractor = map(decimal, f[1:])
    require(arena == ref['expected_arena_used'] and reused in (0, 1) and extractor == 1 and
            (cycles == 0 if reused else cycles > 0), 'INIT: ciclos, arena o estado incompatibles')
    return dict(nn_init_cycles=cycles, arena_used=arena, reused=bool(reused), extractor_initialized=True,
                dnn_initialization_measured_now=not bool(reused), nn_init_ms=cycles / 180000 if not reused else None,
                extractor_initialization_timed=False, excluded_from_window_total=True)


def frames(window, chunk=48):
    values = np.asarray(window, dtype='<f4')
    require(values.shape == (4096, 2) and np.isfinite(values).all() and 1 <= chunk <= 48, 'Ventana o chunk inválidos')
    values = values.ravel()
    for offset in range(0, 8192, chunk):
        block = values[offset:offset + chunk]
        command = f'CWRUFEAT_DATA,{offset},' + ','.join(format(float(v), '.9g') for v in block)
        require(len(command) + 2 < 1024 and np.asarray(command.split(',')[2:], dtype='<f4').tobytes() == block.tobytes(),
                'Serialización no conserva float32 o excede RX1024')
        yield command, offset + len(block)


def begin(port, timeout, trace):
    line = exchange(port, 'CWRUFEAT_BEGIN', 'CWRUFEAT_READY,', timeout, trace)
    require(line == 'CWRUFEAT_READY,8192', 'BEGIN incompatible')


def check(port, count, checksum, timeout, trace):
    line = exchange(port, 'CWRUFEAT_CHECK', 'CWRUFEAT_CHECK,', timeout, trace)
    require(line == f'CWRUFEAT_CHECK,{count},{checksum}', 'CHECK: commit o checksum diferentes')
    return line


def transmit(port, window, checksum, timeout, trace):
    begin(port, timeout, trace)
    for command, count in frames(window):
        response = exchange(port, command, 'CWRUFEAT_ACK,', timeout, trace)
        require(response == f'CWRUFEAT_ACK,{count}', 'ACK de offset incorrecto')
    check(port, 8192, checksum, timeout, trace)


def parse_result(line, case, ref, last_id):
    f = line.split(',')
    require(len(f) == 17 and f[0] == 'CWRUPIPE_RESULT', 'RESULT incompleto')
    result_id = decimal(f[1], 1)
    require(result_id == last_id + 1 and hex_text(f[2], 8) == case['checksum_fnv1a'], 'ID/checksum de RUN incorrecto')
    total, extract, inp, nn_call, invoke, det = [decimal(v, 1) for v in f[3:9]]
    cls = decimal(f[9], 0, 2)
    require(all(re.fullmatch('-?[0-9]+', v) is not None for v in f[10:13]), 'Códigos de salida mal formados')
    q = list(map(int, f[10:13]))
    score = hex_text(f[13], 8)
    rejected, arena, extractions = decimal(f[14], 0, 1), decimal(f[15], 1, 16384), decimal(f[16], 1)
    require(all(-128 <= v <= 127 for v in q) and cls == int(np.argmax(q)) and
            total >= extract + inp + nn_call + det and invoke <= nn_call and extractions == 1 and
            arena == ref['expected_arena_used'], 'RESULT: tiempos, clase, códigos, arena o extracciones inválidos')
    score_value = np.frombuffer(bytes.fromhex(raw_hex([score])), dtype='<f4')[0]
    require(np.isfinite(score_value) and score_value >= 0, 'Score inválido')
    return dict(id=result_id, checksum_fnv1a=f[2].lower(), total_cycles=total, extract_cycles=extract,
        input_cycles=inp, nn_call_cycles=nn_call, nn_invoke_cycles=invoke, detector_cycles=det,
        instrumentation_control_residual_cycles=total - extract - inp - nn_call - det,
        dnn_expected_comparison=dict(output_int8=q, predicted_class=cls), score_bits=score,
        rejected=bool(rejected), arena_used=arena, extractions=extractions)


def page_fields(port, page, result, length, timeout, trace):
    f = exchange(port, f'CWRUPIPE_{page},{result["id"]}', f'CWRUPIPE_{page},', timeout, trace).split(',')
    require(len(f) == length and decimal(f[1], 1) == result['id'] and hex_text(f[2], 8) == result['checksum_fnv1a'],
            'Página de otra ejecución o checksum distinto: ' + page)
    return f


def get_pages(port, result, timeout, trace):
    f = page_fields(port, 'FEATURES', result, 4, timeout, trace)
    features = float_words(f[3], 14)
    f = page_fields(port, 'INPUT', result, 8, timeout, trace)
    z, affine = float_words(f[3], 14), float_words(f[4], 14)
    q = np.frombuffer(bytes.fromhex(hex_text(f[5], 28)), dtype=np.int8).astype(int).tolist()
    low, high = int(hex_text(f[6], 4), 16), int(hex_text(f[7], 4), 16)
    require(low < (1 << 14) and high < (1 << 14), 'Máscaras fuera de catorce entradas')
    inp = dict(standardized_bits=z, affine_bits=affine, quantized=q, saturated_low=low, saturated_high=high)
    f = page_fields(port, 'DETECTOR', result, 8, timeout, trace)
    det = dict(z_bits=float_words(f[3], 12), term_bits=float_words(f[4], 12), score_bits=hex_text(f[5], 8),
               rejected=bool(decimal(f[6], 0, 1)), cluster=decimal(f[7], 0, 0))
    require(det['score_bits'] == result['score_bits'] and det['rejected'] == result['rejected'], 'Resumen/página detector distintos')
    return dict(features_bits=features, input_expected=inp, detector_expected=det,
                dnn_expected=result['dnn_expected_comparison'])


def compare_float64(feature_bits, case):
    observed = np.frombuffer(bytes.fromhex(raw_hex(feature_bits)), dtype='<f4').astype(np.float64)
    original = np.asarray(case['features_float64_expected'], dtype=np.float64)
    close = np.isclose(observed, original, rtol=.001, atol=.00001)
    bins = np.rint(observed[[4, 11]] / (12000 / 4096)).astype(int).tolist()
    bins_match = [a == b for a, b in zip(bins, case['dominant_bins_expected'])]
    return dict(passed=bool(close.all() and all(bins_match)), per_feature_pass=close.tolist(),
        dominant_bins_observed=bins, dominant_bins_exact=bins_match, rtol=.001, atol=.00001,
        maximum_absolute_error=float(np.abs(observed - original).max()))


def execute(port, case, ref, last_id, timeout, trace, row):
    # Preserve even a partial summary/page in the row if a later parse or comparison fails.
    line = exchange(port, 'CWRUPIPE_RUN,' + case['checksum_fnv1a'], 'CWRUPIPE_RESULT,', timeout, trace)
    row['summary_line'] = line
    result = parse_result(line, case, ref, last_id)
    row['observed'] = result
    pages = get_pages(port, result, timeout, trace)
    row['observed_pages'] = pages
    exact = {k: pages[k] == case[k] for k in ('features_bits', 'input_expected', 'dnn_expected', 'detector_expected')}
    old = compare_float64(pages['features_bits'], case)
    row.update(bit_exact=exact, original_float64_comparison=old, reference={k: case[k] for k in exact})
    row['status'] = 'PASS' if all(exact.values()) and old['passed'] else 'FAIL'
    require(row['status'] == 'PASS', 'Discrepancia integrada: ' + case['name'] + '; preservar evidencia sin relajar criterio')
    return result['id']


def negatives(port, ref, windows, last_id, timeout, trace, rejected, support):
    """Exactly 24 rejection commands; supporting CHECK/INIT and two RUN are explicit."""
    case = ref['cases'][0]
    stale_id = last_id

    def reject(name, cmd, namespace, reason, detail):
        row = dict(name=name, command=cmd, expected_reason=reason, expected_detail=detail, status='ERROR')
        rejected.append(row)
        line = exchange(port, cmd, namespace + '_ERROR,', timeout, trace, allow_error=True)
        row['response'] = line
        row['status'] = 'PASS' if line == f'{namespace}_ERROR,{reason},{detail}' else 'FAIL'
        require(row['status'] == 'PASS', 'Rechazo incorrecto: ' + line)

    wrong_id = 1 if last_id != 1 else 2
    for page in ('FEATURES', 'INPUT', 'DETECTOR'):
        reject('wrong_id_' + page.lower(), f'CWRUPIPE_{page},{wrong_id}', 'CWRUPIPE', 'STALE_ID', -1)
    # Wrong-ID queries must leave the current publication intact.
    current = dict(id=last_id, checksum_fnv1a=ref['cases'][-1]['checksum_fnv1a'])
    f = page_fields(port, 'FEATURES', current, 4, timeout, trace)
    require(float_words(f[3], 14) == ref['cases'][-1]['features_bits'], 'GET inválido alteró publicación válida')
    reject('unknown_command', 'CWRUPIPE_BAD', 'CWRUPIPE', 'UNKNOWN', -1)
    # BEGIN must invalidate a publication that is still valid at this point.
    # Malformed RUN also invalidates it, so test those only after BEGIN's queries.
    begin(port, timeout, trace)
    for page in ('FEATURES', 'INPUT', 'DETECTOR'):
        reject('begin_invalidates_' + page.lower(), f'CWRUPIPE_{page},{stale_id}', 'CWRUPIPE', 'NO_RESULT', -1)
    reject('short_run_checksum', 'CWRUPIPE_RUN,1234567', 'CWRUPIPE', 'FORMAT', -1)
    reject('extra_run_field', 'CWRUPIPE_RUN,12345678,0', 'CWRUPIPE', 'FORMAT', -1)
    reject('zero_query_id', 'CWRUPIPE_FEATURES,0', 'CWRUPIPE', 'FORMAT', -1)
    reject('incomplete_run', 'CWRUPIPE_RUN,' + case['checksum_fnv1a'], 'CWRUPIPE', 'INCOMPLETE', 0)
    reject('get_after_failed_run', f'CWRUPIPE_FEATURES,{stale_id}', 'CWRUPIPE', 'NO_RESULT', -1)
    require(exchange(port, 'CWRUFEAT_DATA,0,1,2,3,4', 'CWRUFEAT_ACK,', timeout, trace) == 'CWRUFEAT_ACK,4', 'ACK parcial inválido')
    partial_checksum = fnv(np.asarray([1, 2, 3, 4], dtype='<f4').tobytes())
    check(port, 4, partial_checksum, timeout, trace)
    atomic = [('offset', 'CWRUFEAT_DATA,3,5', 'OFFSET', 4),
        ('bad_offset', 'CWRUFEAT_DATA,-1,5', 'BAD_OFFSET', -1),
        ('invalid_token', 'CWRUFEAT_DATA,4,5,g', 'BAD_VALUE', 1),
        ('nonfinite', 'CWRUFEAT_DATA,4,5,nan', 'BAD_VALUE', 1),
        ('empty_value', 'CWRUFEAT_DATA,4,5,', 'EMPTY_VALUE', 1),
        ('frame_values', 'CWRUFEAT_DATA,4,' + ','.join(['5'] * 49), 'FRAME_VALUES', 48)]
    for name, cmd, reason, detail in atomic:
        reject('atomic_' + name, cmd, 'CWRUFEAT', reason, detail)
        rejected[-1]['unchanged_check'] = check(port, 4, partial_checksum, timeout, trace)
        rejected[-1]['atomic_no_commit_verified'] = True
    transmit(port, windows[0], case['checksum_fnv1a'], timeout, trace)
    row = dict(name=case['name'], purpose='lifecycle_support_before_wrong_checksum', status='ERROR')
    support.append(row)
    last_id = execute(port, case, ref, last_id, timeout, trace, row)
    previous_id = last_id
    wrong_checksum = f'{int(case["checksum_fnv1a"], 16) ^ 1:08x}'
    reject('wrong_complete_checksum', 'CWRUPIPE_RUN,' + wrong_checksum, 'CWRUPIPE', 'CHECKSUM', -1)
    reject('get_after_checksum_failure', f'CWRUPIPE_FEATURES,{previous_id}', 'CWRUPIPE', 'NO_RESULT', -1)
    # A failed RUN invalidates publication but preserves the complete input and its ID counter.
    check(port, 8192, case['checksum_fnv1a'], timeout, trace)
    row = dict(name=case['name'], purpose='lifecycle_support_after_failed_run', status='ERROR')
    support.append(row)
    last_id = execute(port, case, ref, last_id, timeout, trace, row)
    reject('stale_after_new_run', f'CWRUPIPE_FEATURES,{previous_id}', 'CWRUPIPE', 'STALE_ID', -1)
    reject('info_extra_field', 'CWRUPIPE_INFO,extra', 'CWRUPIPE', 'FORMAT', -1)
    reject('init_extra_field', 'CWRUPIPE_INIT,extra', 'CWRUPIPE', 'FORMAT', -1)
    reject('query_extra_field', f'CWRUPIPE_FEATURES,{last_id},extra', 'CWRUPIPE', 'FORMAT', -1)
    current = dict(id=last_id, checksum_fnv1a=case['checksum_fnv1a'])
    require(float_words(page_fields(port, 'FEATURES', current, 4, timeout, trace)[3], 14) == case['features_bits'],
            'Consultas/formato inválidos alteraron una publicación válida')
    require(len(rejected) == NEGATIVE_COUNT and len(support) == 2, 'Catálogo de lifecycle diferente del previo')
    return last_id


def statistics(results):
    completed = [r['observed'] for r in results if 'observed' in r and r.get('status') == 'PASS']
    if not completed:
        return None
    summary = {}
    for name in ('total', 'extract', 'input', 'nn_call', 'nn_invoke', 'detector', 'instrumentation_control_residual'):
        a = np.asarray([r[name + '_cycles'] for r in completed], dtype=np.float64)
        summary[name] = dict(count=len(a), min_cycles=int(a.min()), max_cycles=int(a.max()), mean_cycles=float(a.mean()),
            median_ms=float(np.median(a) / 180000), p95_ms=float(np.percentile(a, 95) / 180000),
            maximum_ms=float(a.max() / 180000), mean_ms=float(a.mean() / 180000),
            sample_std_ms=float(a.std(ddof=1) / 180000) if len(a) > 1 else None)
    maximum = summary['total']['maximum_ms']
    return dict(blocks=summary, invoke_is_subset_of_nn_call_not_summed_twice=True, hop_ms=HOP_MS,
        maximum_observed_ms=maximum, provisional_operational_limit_ms=LIMIT_MS, required_margin_fraction=.20,
        observed_margin_fraction=1 - maximum / HOP_MS, meets_provisional_operational_limit=maximum <= LIMIT_MS,
        meets_nominal_hop=maximum <= HOP_MS, not_wcet=True, not_real_time_guarantee=True,
        excludes_uart_acquisition_initialization_and_offline_fir=True)


def run_session(port, ref, windows, selected, timeout, trace, report, all_cases=False):
    report['info_before_init'] = info(port, ref, timeout, trace)
    report['initialization'] = initialize(port, ref, timeout, trace)
    report['initialization_reuse_check'] = initialize(port, ref, timeout, trace)
    require(report['initialization_reuse_check']['reused'] and report['initialization_reuse_check']['nn_init_cycles'] == 0,
            'La segunda inicialización no reutiliza la instancia')
    report['firmware_info'] = info(port, ref, timeout, trace)
    require(report['firmware_info']['nn_status'] == 0 and report['firmware_info']['arena_used'] == ref['expected_arena_used'],
            'DNN no está inicializada después de INIT')
    last_id = report['firmware_info']['last_id']
    for case in selected:
        transmit(port, windows[case['index']], case['checksum_fnv1a'], timeout, trace)
        report['positive_window_transmissions'] += 1
        previous = None
        for repetition in range(1, REPETITIONS + 1):
            row = dict(name=case['name'], index=case['index'], metadata=case['metadata'], repetition=repetition,
                       checksum_fnv1a=case['checksum_fnv1a'], status='ERROR')
            report['results'].append(row)
            last_id = execute(port, case, ref, last_id, timeout, trace, row)
            stable = previous is None or row['observed_pages'] == previous
            row['repeated_outputs_and_intermediates_stable'] = stable
            if not stable:
                row['status'] = 'FAIL'
            require(stable, 'Repeticiones inestables')
            previous = row['observed_pages']
        print(f'{case["name"]}: PASS; tres extracciones/RUN; total último={row["observed"]["total_cycles"]/180000:.6f} ms', flush=True)
    if all_cases:
        last_id = negatives(port, ref, windows, last_id, timeout, trace, report['negative_tests'], report['lifecycle_support_results'])
        report['lifecycle_full_window_transmissions'] = 1
    report['firmware_info_after'] = info(port, ref, timeout, trace)
    require(report['firmware_info_after']['last_id'] == last_id, 'INFO final no conserva ID')
    report['timing'] = statistics(report['results'])
    require(report['timing']['meets_provisional_operational_limit'], 'Máximo observado supera límite operativo provisional de136.533333ms')
    report['status'] = 'PASS'


class FakeTransport:
    """Explicit transport/result fixture; no C extractor, ARM/TFLM or DWT execution."""
    def __init__(self, ref, windows):
        self.ref, self.windows = ref, windows
        self.received = []
        self.queue = []
        self.result = None
        self.last_id = 0
        self.initialized = False
        self.active = False
        self.mutate_page_id = False

    def write(self, payload):
        command = payload.decode('ascii').removesuffix('\r\n')
        f = command.split(',')
        error = lambda why, detail=-1: f'CWRUPIPE_ERROR,{why},{detail}'
        if f[0] == 'CWRUPIPE_RUN':
            self.result = None
        if f[0] in ('CWRUPIPE_INFO', 'CWRUPIPE_INIT') and len(f) != 1:
            reply = error('FORMAT')
        elif command == 'CWRUPIPE_INFO':
            reply = f'CWRUPIPE_INFO,1,12000,4096,2,14,180000000,1,00000000,4064,a3f03713,{MODEL_SHA},{INPUT_SHA},555ede8e,439af637,16384,{self.ref["expected_arena_used"] if self.initialized else 0},{0 if self.initialized else -1},{int(self.result is not None)},{self.last_id}'
        elif command == 'CWRUPIPE_INIT':
            reused = self.initialized
            self.initialized = True
            reply = f'CWRUPIPE_READY,{0 if reused else 50000},{self.ref["expected_arena_used"]},{int(reused)},1'
        elif command == 'CWRUFEAT_BEGIN':
            self.received, self.result, self.active = [], None, True
            reply = 'CWRUFEAT_READY,8192'
        elif f[0] == 'CWRUFEAT_DATA':
            self.result = None
            reply = None
            if not self.active:
                reply = 'CWRUFEAT_ERROR,NO_BEGIN,-1'
            elif len(f) < 2 or not re.fullmatch('[0-9]+', f[1]):
                reply = 'CWRUFEAT_ERROR,BAD_OFFSET,-1'
            elif int(f[1]) != len(self.received):
                reply = f'CWRUFEAT_ERROR,OFFSET,{len(self.received)}'
            values = []
            if reply is None:
                for i, text in enumerate(f[2:]):
                    if i >= 48:
                        reply = 'CWRUFEAT_ERROR,FRAME_VALUES,48'; break
                    if not text:
                        reply = f'CWRUFEAT_ERROR,EMPTY_VALUE,{i}'; break
                    try:
                        value = np.float32(text)
                        if not np.isfinite(value):
                            raise ValueError('nonfinite')
                    except ValueError:
                        reply = f'CWRUFEAT_ERROR,BAD_VALUE,{i}'; break
                    values.append(value)
            if reply is None:
                if not values or len(self.received) + len(values) > 8192:
                    reply = f'CWRUFEAT_ERROR,VALUE_COUNT,{len(values)}'
                else:
                    self.received.extend(values)
                    reply = f'CWRUFEAT_ACK,{len(self.received)}'
        elif command == 'CWRUFEAT_CHECK':
            digest = fnv(np.asarray(self.received, dtype='<f4').tobytes())
            reply = f'CWRUFEAT_CHECK,{len(self.received)},{digest}'
        elif f[0] == 'CWRUPIPE_RUN':
            if len(f) != 2 or re.fullmatch('[0-9a-fA-F]{8}', f[1]) is None:
                reply = error('FORMAT')
            elif len(self.received) != 8192:
                reply = error('INCOMPLETE', len(self.received))
            elif fnv(np.asarray(self.received, dtype='<f4').tobytes()) != f[1].lower():
                reply = error('CHECKSUM')
            elif not self.initialized:
                reply = error('NOT_INITIALIZED', -1)
            else:
                case = next(c for c in self.ref['cases'] if c['checksum_fnv1a'] == f[1].lower())
                self.last_id += 1
                self.result = case
                q = case['dnn_expected']['output_int8']
                reply = f'CWRUPIPE_RESULT,{self.last_id},{f[1]},18020000,18000000,1800,11000,10500,1800,{case["dnn_expected"]["predicted_class"]},{q[0]},{q[1]},{q[2]},{case["detector_expected"]["score_bits"]},{int(case["detector_expected"]["rejected"])},{self.ref["expected_arena_used"]},1'
        elif f[0] in ('CWRUPIPE_FEATURES', 'CWRUPIPE_INPUT', 'CWRUPIPE_DETECTOR'):
            if len(f) != 2 or re.fullmatch('[0-9]+', f[1]) is None or int(f[1]) == 0:
                reply = error('FORMAT')
            elif self.result is None:
                reply = error('NO_RESULT')
            elif int(f[1]) != self.last_id:
                reply = error('STALE_ID')
            else:
                case = self.result
                prefix = f'{f[0]},{self.last_id + int(self.mutate_page_id)},{case["checksum_fnv1a"]}'
                if f[0] == 'CWRUPIPE_FEATURES':
                    reply = prefix + ',' + raw_hex(case['features_bits'])
                elif f[0] == 'CWRUPIPE_INPUT':
                    v = case['input_expected']
                    reply = f'{prefix},{raw_hex(v["standardized_bits"])},{raw_hex(v["affine_bits"])},{np.asarray(v["quantized"], dtype=np.int8).tobytes().hex()},{v["saturated_low"]:04x},{v["saturated_high"]:04x}'
                else:
                    v = case['detector_expected']
                    reply = f'{prefix},{raw_hex(v["z_bits"])},{raw_hex(v["term_bits"])},{v["score_bits"]},{int(v["rejected"])},0'
        else:
            reply = error('UNKNOWN')
        self.queue.append((reply + '\r\n').encode('ascii'))
        return len(payload)

    def flush(self):
        pass

    def readline(self):
        return self.queue.pop(0) if self.queue else b''


def new_report(mode):
    return dict(status='ERROR', mode=mode, repetitions=3, results=[], negative_tests=[], lifecycle_support_results=[],
        positive_window_transmissions=0, lifecycle_full_window_transmissions=0, expected_negative_tests=NEGATIVE_COUNT if mode == 'all' else 0,
        positive_expected_executions=126 if mode == 'all' else 3, expected_lifecycle_support_executions=2 if mode == 'all' else 0,
        training_executed=False, model_conversion_executed=False, threshold_refitted=False, outer_test_loaded=False,
        real_time_guarantee=False, wcet_claim=False)


def save_evidence(folder, report, trace):
    report['completed_executions'] = sum(r.get('status') == 'PASS' for r in report['results'])
    report['attempted_positive_executions'] = len(report['results'])
    report['completed_lifecycle_support_executions'] = sum(r.get('status') == 'PASS' for r in report['lifecycle_support_results'])
    report['total_successful_pipeline_executions'] = report['completed_executions'] + report['completed_lifecycle_support_executions']
    report['negative_tests_passed'] = sum(r.get('status') == 'PASS' for r in report['negative_tests'])
    report['pipeline_run_commands_sent'] = sum(r['direction'] == 'TX' and r['command'].startswith('CWRUPIPE_RUN,') for r in trace)
    report['activity_counts'] = {name: sum(r['direction'] == 'TX' and
        (r['command'] == command if exact else r['command'].startswith(command)) for r in trace)
        for name, command, exact in [('begin_commands', 'CWRUFEAT_BEGIN', True),
            ('data_commands', 'CWRUFEAT_DATA,', False), ('check_commands', 'CWRUFEAT_CHECK', True),
            ('run_commands', 'CWRUPIPE_RUN,', False), ('init_commands', 'CWRUPIPE_INIT', True),
            ('info_commands', 'CWRUPIPE_INFO', True)]}
    report['timing'] = statistics(report['results'])
    report.update(client_sha256=sha(Path(__file__)), numpy=np.__version__, python=sys.version,
                  utc_finished=datetime.now(timezone.utc).isoformat())
    with (folder / 'resultados.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2); stream.write('\n')
    with (folder / 'uart_trace.jsonl').open('x', encoding='utf-8', newline='\n') as stream:
        for row in trace:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    manifest = {name: sha(folder / name) for name in ('resultados.json', 'uart_trace.jsonl')}
    with (folder / 'MANIFEST_SHA256.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(manifest, stream, indent=2); stream.write('\n')


def self_test(ref, windows, metadata, folder, report, trace):
    validate_cases(ref, windows, metadata)
    fake = FakeTransport(ref, windows)
    report.update(mode='all', simulation_only=True, firmware_executed=False, serial_port_opened=False,
                  self_test_executes_c_arm=False, synthetic_cycle_counts_not_hardware_measurement=True,
                  expected_negative_tests=24, positive_expected_executions=126, expected_lifecycle_support_executions=2)
    run_session(fake, ref, windows, ref['cases'], 1., trace, report, all_cases=True)
    require(report['negative_tests'] and len(report['negative_tests']) == 24 and
            len(report['results']) == 126 and len(report['lifecycle_support_results']) == 2, 'Self-test incompleto')
    # Reject corrupted IDs, timings, output codes, truncation and mismatching feature bins.
    case = ref['cases'][0]
    tests = []
    line = report['results'][0]['summary_line']
    for name, mutate in [('zero_total', lambda f: f.__setitem__(3, '0')),
        ('sum_exceeds_total', lambda f: f.__setitem__(3, '1')),
        ('invoke_exceeds_call', lambda f: f.__setitem__(7, '999999999')),
        ('wrong_extractions', lambda f: f.__setitem__(16, '2')),
        ('wrong_checksum', lambda f: f.__setitem__(2, '00000000')),
        ('stale_summary_id', lambda f: f.__setitem__(1, '2')),
        ('out_of_range_q', lambda f: f.__setitem__(10, '128')),
        ('truncated_summary', lambda f: f.pop())]:
        f = line.split(','); mutate(f)
        try:
            parse_result(','.join(f), case, ref, 0)
        except ValueError:
            tests.append(dict(name=name, status='PASS'))
        else:
            raise RuntimeError('Parser aceptó corrupción: ' + name)
    fake.mutate_page_id = True
    try:
        page_fields(fake, 'FEATURES', dict(id=fake.last_id, checksum_fnv1a=case['checksum_fnv1a']), 4, 1., trace)
    except ValueError:
        tests.append(dict(name='mismatched_page_id', status='PASS'))
    else:
        raise RuntimeError('Aceptó página de otra ejecución')
    corrupted = np.frombuffer(bytes.fromhex(raw_hex(case['features_bits'])), dtype='<f4').copy()
    corrupted[4] += np.float32(12000 / 4096)
    require(not compare_float64(words(corrupted), case)['passed'], 'No detectó bin dominante diferente')
    tests.append(dict(name='dominant_bin_corruption', status='PASS'))
    report['self_test_parser_corruption_checks'] = tests
    print('SIMULACIÓN EXPLÍCITA: transporte/fixtures; sin C ARM, TFLM ARM, COM o tiempos físicos.', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--port')
    p.add_argument('--self-test', action='store_true')
    g = p.add_mutually_exclusive_group()
    g.add_argument('--index', type=int)
    g.add_argument('--all', action='store_true')
    p.add_argument('--repeat', type=int, default=3)
    p.add_argument('--timeout', type=float, default=20.)
    p.add_argument('--bundle-dir', type=Path, default=HERE)
    p.add_argument('--output-root', type=Path)
    args = p.parse_args()
    if args.repeat != 3 or args.timeout <= 0 or (not args.self_test and not args.port):
        p.error('Se requiere --port en prueba física, --repeat3 y timeout positivo')
    if args.index is not None and not 0 <= args.index < 42:
        p.error('Índice fuera de0..41')
    folder = (args.output_root or args.bundle_dir / 'resultados') / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    folder.mkdir(parents=True, exist_ok=False)
    report, trace = new_report('all' if args.all or args.self_test else 'single_development_index'), []
    report.update(utc_started=datetime.now(timezone.utc).isoformat(), port=args.port if not args.self_test else None,
                  simulation_only=args.self_test, firmware_executed=False, serial_port_opened=False)
    try:
        ref, windows, metadata = load_bundle(args.bundle_dir.resolve())
        report.update(reference_sha256=sha(args.bundle_dir / 'REFERENCIA_PIPELINE.json'),
            bundle_manifest_sha256=sha(args.bundle_dir / 'VALIDACION_MANIFEST.json'))
        if args.self_test:
            self_test(ref, windows, metadata, folder, report, trace)
        else:
            import serial
            selected = ref['cases'] if args.all else [ref['cases'][0 if args.index is None else args.index]]
            with serial.Serial(args.port, 115200, bytesize=8, parity='N', stopbits=1, timeout=.2, write_timeout=args.timeout) as port:
                report['serial_port_opened'] = True
                time.sleep(.2); port.reset_input_buffer()
                report['firmware_executed'] = True
                run_session(port, ref, windows, selected, args.timeout, trace, report, all_cases=args.all)
    except Exception as exc:
        report['status'] = 'ERROR'
        report['error'] = f'{type(exc).__name__}: {exc}'
        print('ERROR:', report['error'], file=sys.stderr)
    finally:
        save_evidence(folder, report, trace)
        print('Evidencias:', folder, flush=True)
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
