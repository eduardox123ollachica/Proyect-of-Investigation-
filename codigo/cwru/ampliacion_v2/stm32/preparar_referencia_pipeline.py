"""Fija 42 referencias integradas desde evidencias aisladas; no usa TF ni COM."""
from __future__ import annotations
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
STAGES = ROOT / 'despliegue_stm32'
OUT = STAGES / 'integracion_pipeline_180mhz_2026-10-23'
INPUT = STAGES / 'escalado_cuantizacion_180mhz_2026-10-23'
DNN = STAGES / 'dnn_int8_180mhz_2026-10-23'
DET = STAGES / 'detector_normalidad_180mhz_2026-10-23'
ENTM = STAGES / 'mejoras_entropia_180mhz_2026-10-23'
SOURCES = {}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked(path, expected=None):
    digest = sha(path)
    require(expected is None or digest == expected, 'SHA256 diferente: ' + str(path))
    SOURCES[path.relative_to(ROOT).as_posix()] = digest
    return path


def read(path, expected=None):
    return json.loads(checked(path, expected).read_text(encoding='utf-8-sig'))


def manifest(path, base):
    values = read(path)
    for name, digest in values.items():
        checked(base / name.replace('\\', '/'), digest)
    return values


def bits(values):
    return [f'{int(v):08x}' for v in np.asarray(values, dtype='<f4').view('<u4').ravel()]


def raw(words):
    return np.asarray([int(v, 16) for v in words], dtype='<u4').tobytes().hex()


def fnv(data):
    value = 2166136261
    for byte in data:
        value = ((value ^ byte) * 16777619) & 0xffffffff
    return f'{value:08x}'


def numerical(observed):
    return {k: v for k, v in observed.items() if k not in ('cycles', 'invoke_cycles', 'call_cycles', 'arena_used')}


def physical(stage, report_name, expected_report, count, trace_key):
    audit = read(stage / report_name, expected_report)
    require(audit['passed_executions'] == count, 'Auditoría física incompleta')
    folder = stage / 'Validacion_PC/resultados' / audit['run']
    # Examine the recorded manifest before using either the parsed or raw data.
    manifest(folder / 'MANIFEST_SHA256.json', folder)
    result = read(folder / 'resultados.json', audit['results_sha256'])
    trace_path = checked(folder / 'uart_trace.jsonl', audit[trace_key])
    trace = [json.loads(line) for line in trace_path.read_text(encoding='utf-8').splitlines()]
    require(result['status'] == 'PASS' and result['completed_executions'] == count and
            len(result['results']) == count and all(r['status'] == 'PASS' for r in result['results']),
            'Resultados físicos incompletos')
    return result, trace


def responses(trace, prefix):
    return [r['line'].split(',') for r in trace if r['direction'] == 'RX' and r['line'].startswith(prefix)]


def main():
    require(not (OUT / 'REFERENCIA_PIPELINE.json').exists() and not (OUT / 'ENTRADAS_REFERENCIA_SHA256.json').exists(),
            'No se sobrescribe una referencia previamente fijada')
    # Historical manifests are checked before their referenced data are consumed.
    manifest(INPUT / 'auditoria_pc/MANIFEST_SHA256.json', INPUT / 'auditoria_pc')
    manifest(INPUT / 'auditoria_pc/ENTRADAS_SHA256.json', ROOT)
    manifest(DNN / 'REFERENCIA_MANIFEST_SHA256.json', DNN)
    manifest(DNN / 'MANIFEST_RESULTADOS_FISICOS_DNN_SHA256.json', ROOT)
    manifest(DET / 'MANIFEST_RESULTADOS_FISICOS_DETECTOR_SHA256.json', ROOT)
    case_source = read(INPUT / 'auditoria_pc/casos_validacion.json',
                       'b219d95260a8f452787c3eb30528e268a84e0e0bfd305750731c36044b154b83')
    dnn_ref = read(DNN / 'REFERENCIA_DNN.json', 'b1184cf30f067095ff546b216bdc10b86dd05f0e0abbffb1bfb51bfea21dfede')
    detector_ref = read(DET / 'REFERENCIA_DETECTOR.json', 'c77632037675f4a5a0ef9b9b6f0f467b38d66740cc3a76ec4586a1efff074848')
    input_run, input_trace = physical(INPUT, 'RESULTADOS_FISICOS_ENTRADA_COMPLETA.json',
        '50eca1e01a7fb25a4f3c4b9c079242b42488ca6d7b08fa7e4cdf7c925d17f3e4', 756, 'trace_sha256')
    dnn_run, dnn_trace = physical(DNN, 'RESULTADOS_FISICOS_DNN_COMPLETA.json',
        '50212c21212316240d5ce3075783ac600e8f35acbb5e144fef6f47feaa51604b', 243, 'uart_trace_sha256')
    det_run, det_trace = physical(DET, 'RESULTADOS_FISICOS_DETECTOR_COMPLETA.json',
        '8fcb851bfd828eff2ce49ac7c9213319471d02a2a39606a3a2866f3b300fcda5', 495, 'trace_sha256')
    # Check all physical raw responses against the parsed observations, not only reference declarations.
    ir = responses(input_trace, 'CWRUIN_RESULT,')
    dr = responses(dnn_trace, 'CWRUNN_RESULT,')
    tr = responses(det_trace, 'CWRUND_RESULT,')
    require((len(ir), len(dr), len(tr)) == (756, 243, 495), 'Cantidad de respuestas UART distinta')
    for row, f in zip(input_run['results'], ir):
        o = row['observed']
        require(len(f) == 48 and f[2] == fnv(bytes.fromhex(raw(row['input_bits']))) and
                list(map(int, f[3:6])) == [o['cycles'], o['saturated_low'], o['saturated_high']] and
                f[6:20] == o['standardized_bits'] and f[20:34] == o['affine_bits'] and
                list(map(int, f[34:])) == o['quantized'], 'UART físico INPUT no corresponde al JSON')
    for row, f in zip(dnn_run['results'], dr):
        o = row['observed']
        require(len(f) == 9 and f[1] == fnv(bytes.fromhex(row['input_hex'])) and
                list(map(int, f[2:])) == [o['invoke_cycles'], o['call_cycles'], o['predicted_class'],
                    *o['output_int8'], o['arena_used']], 'UART físico DNN no corresponde al JSON')
    for row, f in zip(det_run['results'], tr):
        o = row['observed']
        require(len(f) == 10 and f[2] == fnv(bytes.fromhex(row['input_hex'])) and int(f[3]) == o['cycles'] and
                f[4] == o['score_bits'] and int(f[5]) == int(o['rejected']) and int(f[6]) == o['cluster'] and
                f[8] == raw(o['z_bits']) and f[9] == raw(o['term_bits']), 'UART físico detector no corresponde al JSON')
    vector_dir = ENTM / 'Validacion_Mixta'
    npz_path = checked(vector_dir / 'ventanas_desarrollo.npz',
                       'ff09491a486716b7176986cc504138a726e003389cb93b657565a3c422c234cb')
    metadata_path = checked(vector_dir / 'metadata_ventanas.csv',
                            '362ed9979c4370be7bddc95a28ea8a8234f19459ccf72ee3bceaf0c62ce03459')
    with np.load(npz_path, allow_pickle=False) as data:
        windows, original = data['windows'].copy(), data['features'].copy()
    with metadata_path.open(encoding='utf-8-sig', newline='') as stream:
        metadata = list(csv.DictReader(stream))
    require(windows.shape == (42, 4096, 2) and original.shape == (42, 14) and len(metadata) == 42,
            'Vectores de desarrollo incompletos')
    require(set(int(m['load_hp']) for m in metadata) == {0, 1, 2} and len(set(m['file_id'] for m in metadata)) == 21,
            'Metadatos no corresponden a 21 MAT de desarrollo 0/1/2 HP')
    entm_folder = vector_dir / 'resultados/20261023T221815_802310Z'
    entm_rows = read(entm_folder / 'resultados.json')
    entm_trace_path = checked(entm_folder / 'uart_trace.jsonl')
    entm_trace = [json.loads(s) for s in entm_trace_path.read_text(encoding='utf-8').splitlines()]
    features_uart = responses(entm_trace, 'CWRUFEAT_RESULT,')
    checks_uart = responses(entm_trace, 'CWRUFEAT_CHECK,')
    require(len(entm_rows) == len(features_uart) == len(checks_uart) == 42, 'Extracción física incompleta')
    by_input = {c['name']: c for c in case_source}
    by_dnn = {c['name']: c for c in dnn_ref['cases']}
    by_det = {c['name']: c for c in detector_ref['cases']}
    cases = []
    for index in range(42):
        name = f'entm_{index:03d}'
        words = by_input[name]['input_bits']
        physical_features = np.asarray(features_uart[index][2:], dtype='<f4')
        payload = np.asarray(windows[index], dtype='<f4').tobytes(order='C')
        payload_sha = hashlib.sha256(payload).hexdigest()
        checksum = fnv(payload)
        require(bits(physical_features) == words and int(features_uart[index][1]) == entm_rows[index]['cycles'] and
                checks_uart[index] == ['CWRUFEAT_CHECK', '8192', checksum] and
                payload_sha == metadata[index]['window_payload_sha256'] == entm_rows[index]['window_payload_sha256'] and
                int(metadata[index]['bundle_index']) == entm_rows[index]['index'] == index,
                'Extracción/ventana física no pareada: ' + name)
        close = np.isclose(physical_features.astype(np.float64), original[index], rtol=.001, atol=.00001)
        bins = np.rint(physical_features[[4, 11]] / (12000 / 4096)).astype(int)
        expected_bins = np.rint(original[index, [4, 11]] / (12000 / 4096)).astype(int)
        require(close.all() and np.array_equal(bins, expected_bins), 'Contraste float64 original fallido: ' + name)
        input_rows = [r for r in input_run['results'] if r['name'] == name]
        dnn_rows = [r for r in dnn_run['results'] if r['name'] == name]
        det_rows = [r for r in det_run['results'] if r['name'] == name]
        require(all(len(r) == 3 for r in (input_rows, dnn_rows, det_rows)), 'Tres repeticiones aisladas requeridas')
        input_expected = numerical(input_rows[0]['observed'])
        dnn_expected = numerical(dnn_rows[0]['observed'])
        det_expected = numerical(det_rows[0]['observed'])
        qhex = np.asarray(input_expected['quantized'], dtype=np.int8).tobytes().hex()
        require(all(r['input_bits'] == words and numerical(r['observed']) == input_expected == r['reference'] for r in input_rows),
                'INPUT no exacta/estable: ' + name)
        require(qhex == by_dnn[name]['input_hex'] and all(r['input_hex'] == qhex and numerical(r['observed']) == dnn_expected
                and r['comparison']['output_bit_exact'] for r in dnn_rows) and
                dnn_expected['output_int8'] == by_dnn[name]['primary']['output_int8'] and
                dnn_expected['predicted_class'] == by_dnn[name]['primary']['predicted_class'], 'DNN desarrollo no exacta: ' + name)
        require(by_det[name]['input_bits'] == words and all(r['input_hex'] == raw(words) and
                numerical(r['observed']) == det_expected == r['reference'] for r in det_rows) and
                det_expected == by_det[name]['expected'], 'Detector no exacto/estable: ' + name)
        cases.append(dict(index=index, name=name, metadata=metadata[index], checksum_fnv1a=checksum,
            payload_sha256=payload_sha, features_bits=words, features_float64_expected=original[index].astype(float).tolist(),
            dominant_bins_expected=expected_bins.tolist(), input_expected=input_expected, dnn_expected=dnn_expected,
            dnn_pc_primary=by_dnn[name]['primary'], dnn_pc_secondary=by_dnn[name]['secondary_reference'],
            detector_expected=det_expected))
    for p in (OUT / 'CONTRATO_PREVIO_PIPELINE.md', OUT / 'ESPECIFICACION_UART_PREVIA.md', Path(__file__)):
        checked(p)
    annex = OUT / 'ANEXO_CICLO_VIDA_PREVIO.md'
    if annex.exists():
        checked(annex)
    report = dict(schema_version=1, status='PC_REFERENCE_READY_NOT_ARM_INTEGRATED',
        prepared_utc=datetime.now(timezone.utc).isoformat(), cases=cases, development_cases=42,
        model_sha256=dnn_ref['model_sha256'], model_fnv1a=dnn_ref['model_fnv1a'], model_bytes=4064,
        input_sha256='24275e363e29573e7a158d44b00099d2224be9a783ac87654f2e1575fc86f1a5',
        detector_fnv1a=detector_ref['params']['params_fnv1a'], threshold_bits=detector_ref['params']['threshold_bits'],
        expected_arena_used=dnn_run['initialization']['arena_used'], arena_capacity=16384,
        windows_sha256=sha(npz_path), metadata_sha256=sha(metadata_path),
        policy=dict(repetitions=3, exact_intermediates=True, exact_dnn_outputs_against_isolated=True,
            rtol=.001, atol=.00001, exact_dominant_bins=True, hop_ms=170.666667,
            provisional_margin_fraction=.20, operational_limit_ms=136.533333,
            excluded_costs=['UART/parser/checksum', 'initialization', 'diagnostic_record_clear', 'acquisition', 'offline_FIR'],
            wcet_claim=False, real_time_guarantee=False),
        evidence=dict(input_executions_verified=756, dnn_executions_verified=243,
            dnn_development_exact_executions=126, detector_executions_verified=495,
            raw_uart_replay_verified=True, physical_entm_features_verified=42),
        training_executed=False, tensorflow_inference_executed=False, model_conversion_executed=False,
        outer_test_loaded=False, serial_port_opened=False, firmware_executed=False,
        python=sys.version, numpy=np.__version__)
    for name, data in [('REFERENCIA_PIPELINE.json', report), ('ENTRADAS_REFERENCIA_SHA256.json', SOURCES)]:
        with (OUT / name).open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
    print(json.dumps(dict(status=report['status'], cases=len(cases), sources=len(SOURCES), evidence=report['evidence']), indent=2))


if __name__ == '__main__':
    main()
