"""Auditoría de trazas CWRUPIPE existentes; sin COM, firmware o entrenamiento.

Por defecto solo lee. --write-report crea informe y sello físicos nuevos,
separados del sello de preparación, sin sobrescribir resultados anteriores.
La reproducción devuelve exclusivamente las respuestas UART ya registradas.
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from auditar_validacion_o3 import RecordedTransport, sha

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / 'despliegue_stm32/integracion_pipeline_180mhz_2026-10-23'
CLIENT = OUT / 'Validacion_PC'


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def bounded(base, name):
    path = (base / name).resolve()
    require(path != base.resolve() and path.is_relative_to(base.resolve()), 'Ruta fuera del ámbito de auditoría')
    return path


def audit(run_name):
    require(Path(run_name).name == run_name and run_name not in ('.', '..'), 'Nombre de ejecución inválido')
    folder = bounded(CLIENT / 'resultados', run_name)
    digest, name = (OUT / 'MANIFEST_SHA256.sha256').read_text(encoding='ascii').split()
    require(name == 'MANIFEST_SHA256.json' and sha(OUT / name) == digest, 'Sello de preparación diferente')
    preparation = json.loads((OUT / name).read_text(encoding='utf-8'))
    for relative, expected in preparation.items():
        require(sha(bounded(ROOT, relative)) == expected, 'Recurso de preparación modificado: ' + relative)
    manifest = json.loads((folder / 'MANIFEST_SHA256.json').read_text(encoding='utf-8'))
    require(set(manifest) == {'resultados.json', 'uart_trace.jsonl'}, 'Manifiesto físico inesperado')
    for name, expected in manifest.items():
        require(sha(folder / name) == expected, 'Resultado físico modificado: ' + name)
    run = json.loads((folder / 'resultados.json').read_text(encoding='utf-8'))
    require(run['status'] == 'PASS' and not run['simulation_only'] and run['firmware_executed']
            and run['serial_port_opened'], 'No es una prueba física aprobada')
    require(not any(run[key] for key in ('training_executed', 'model_conversion_executed',
            'threshold_refitted', 'outer_test_loaded', 'real_time_guarantee', 'wcet_claim')), 'Alcance incompatible')
    require(run['mode'] in ('single_development_index', 'all') and run['repetitions'] == 3, 'Modo/repeticiones diferentes')
    for key, path in (('client_sha256', CLIENT / 'validar_pipeline.py'),
                      ('reference_sha256', CLIENT / 'REFERENCIA_PIPELINE.json'),
                      ('bundle_manifest_sha256', CLIENT / 'VALIDACION_MANIFEST.json')):
        require(run[key] == sha(path), 'Recurso distinto del usado durante la prueba: ' + key)
    spec = importlib.util.spec_from_file_location('sealed_pipeline_client', CLIENT / 'validar_pipeline.py')
    client = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(client)
    ref, windows, metadata = client.load_bundle(CLIENT)
    all_cases = run['mode'] == 'all'
    require(bool(run['results']), 'No hay ejecuciones guardadas')
    index = None if all_cases else run['results'][0]['index']
    require(index is None or (isinstance(index, int) and 0 <= index < 42), 'Índice individual inválido')
    selected = ref['cases'] if all_cases else [ref['cases'][index]]
    require(len(run['results']) == len(selected) * 3, 'Número de recorridos incorrecto')
    records = [json.loads(line) for line in (folder / 'uart_trace.jsonl').read_text(encoding='utf-8').splitlines()]
    require(len(records) % 2 == 0, 'Número de registros UART impar')
    for number, row in enumerate(records):
        direction = 'RX' if number % 2 else 'TX'
        require(row['direction'] == direction, 'Orden TX/RX inesperado')
        value = row['command'] if direction == 'TX' else row['line']
        length = len((value + '\r\n').encode('ascii'))
        require(length == row['bytes'] and 0 < length < (1024 if direction == 'TX' else 512),
                'Longitud UART incorrecta o fuera del buffer')
    transport, replay = RecordedTransport(records), client.new_report(run['mode'])
    with contextlib.redirect_stdout(io.StringIO()):
        client.run_session(transport, ref, windows, selected, 1., [], replay, all_cases=all_cases)
    require(transport.position == len(records), 'Tramas sin asociar al protocolo previsto')
    # Se vuelven a analizar las tramas y se reconstruyen los resultados; no se
    # confía únicamente en el PASS escrito en el JSON del cliente.
    for key, value in replay.items():
        require(run[key] == value, 'Resultado guardado distinto de la reproducción: ' + key)
    main_count, support_count = len(replay['results']), len(replay['lifecycle_support_results'])
    negative_count = len(replay['negative_tests'])
    counters = dict(completed_executions=main_count, attempted_positive_executions=main_count,
        completed_lifecycle_support_executions=support_count,
        total_successful_pipeline_executions=main_count + support_count, negative_tests_passed=negative_count,
        pipeline_run_commands_sent=sum(r['direction'] == 'TX' and r['command'].startswith('CWRUPIPE_RUN,') for r in records))
    counters['activity_counts'] = {name: sum(r['direction'] == 'TX' and
        (r['command'] == command if exact else r['command'].startswith(command)) for r in records)
        for name, command, exact in [('begin_commands', 'CWRUFEAT_BEGIN', True),
            ('data_commands', 'CWRUFEAT_DATA,', False), ('check_commands', 'CWRUFEAT_CHECK', True),
            ('run_commands', 'CWRUPIPE_RUN,', False), ('init_commands', 'CWRUPIPE_INIT', True),
            ('info_commands', 'CWRUPIPE_INFO', True)]}
    require(all(run[key] == value for key, value in counters.items()), 'Contadores de actividad incoherentes')
    before, init, info, after = [replay[k] for k in ('info_before_init', 'initialization', 'firmware_info', 'firmware_info_after')]
    require(before['nn_status'] == (0 if init['reused'] else -100)
            and before['arena_used'] == (ref['expected_arena_used'] if init['reused'] else 0), 'Estado de arranque incompatible')
    stable_keys = set(info) - {'fpscr', 'result_valid', 'last_id'}
    require(all(info[key] == after[key] for key in stable_keys), 'Entorno o parámetros cambiaron durante la prueba')
    built = json.loads((OUT / 'COMPILACION_PIPELINE.json').read_text(encoding='utf-8'))['configurations']['Release_180MHz_Pipeline']
    local_elf = Path('D:/STM32IDE/Prueba_2_CWRU_Extraccion/Release_180MHz_Pipeline/Prueba_2_CWRU_Extraccion.elf')
    local_hash = sha(local_elf) if local_elf.is_file() else None
    require(local_hash == built['elf_sha256'], 'ELF local ausente o distinto del preparado: requiere revisión')
    label = 'COMPLETA' if all_cases else f'INDICE_{index}'
    report = dict(status='PHYSICAL_PIPELINE_SUITE_AUDITED' if all_cases else 'FIRST_PIPELINE_CASE_AUDITED',
        audited_utc=datetime.now(timezone.utc).isoformat(), run=run_name, mode=run['mode'], development_index=index,
        windows=len(selected), mat_files=len({c['metadata']['file_id'] for c in selected}),
        loads_hp=sorted({int(c['metadata']['load_hp']) for c in selected}), repetitions_per_window=3,
        **counters, expected_negative_tests=24 if all_cases else 0,
        expected_support_executions=2 if all_cases else 0,
        one_extraction_per_run=True, main_extractions=main_count, all_intermediates_and_outputs_bit_exact=True,
        all_repeated_outputs_stable=True, original_float64_feature_comparisons_passed=14 * main_count,
        dominant_bins_exact=2 * main_count, initialization=init,
        initialization_reuse_check=replay['initialization_reuse_check'], firmware_info=info, firmware_info_after=after,
        timing=replay['timing'], raw_records_verified=len(records), recorded_input_output_replay_verified=True,
        maximum_tx_bytes=max(r['bytes'] for r in records if r['direction'] == 'TX'),
        maximum_rx_bytes=max(r['bytes'] for r in records if r['direction'] == 'RX'),
        results_sha256=sha(folder / 'resultados.json'), trace_sha256=sha(folder / 'uart_trace.jsonl'),
        run_manifest_sha256=sha(folder / 'MANIFEST_SHA256.json'), preparation_manifest_sha256=digest,
        preparation_files_verified=len(preparation), client_sha256=sha(CLIENT / 'validar_pipeline.py'),
        reference_sha256=sha(CLIENT / 'REFERENCIA_PIPELINE.json'), auditor_sha256=sha(Path(__file__)),
        recorded_transport_source_sha256=sha(Path(__file__).with_name('auditar_validacion_o3.py')),
        prepared_elf_sha256=built['elf_sha256'], local_elf_sha256=local_hash, local_elf_matches_prepared_build=True,
        prepared_flash_bytes=built['flash_load_bytes'],
        prepared_static_ram_with_linker_reservations_bytes=built['ram_static_with_linker_reservations_bytes'],
        audit_python=sys.version, audit_numpy=np.__version__, full_device_flash_read=False,
        serial_port_opened_during_audit=False, physical_experiment_repeated=False, firmware_changed_during_audit=False,
        training_executed_during_audit=False, predictive_metrics_computed=False,
        limitations=[
            'First development window is not the full 42-window validation.' if not all_cases else
            '42 fixed development windows do not establish predictive generalization or the V2 study.',
            'One extraction per RUN, three RUN per window; client stdout wording does not mean three extractions per RUN.',
            'Total excludes UART/parser/checksum, initialization, diagnostic clearing, acquisition and offline FIR.',
            'Invoke is a subset of NN_CALL, not an additional term in the total.',
            'Nominal HAL HCLK; observed maxima are not WCET or autonomous acquisition proof.',
            'Physical peak stack and shared-operation cost profile remain pending.',
            'Declared hashes and local ELF identity are not a complete device Flash readback.'])
    return report, label, folder


def write_report(report, label, folder):
    target = OUT / f'RESULTADOS_FISICOS_PIPELINE_{label}.json'
    human = OUT / f'RESULTADOS_VALIDACION_PIPELINE_{label}.md'
    manifest_path = OUT / f'MANIFEST_RESULTADOS_FISICOS_PIPELINE_{label}_SHA256.json'
    seal_path = manifest_path.with_suffix('.sha256')
    require(not any(path.exists() for path in (target, human, manifest_path, seal_path)), 'No sobrescribir informes físicos')
    with target.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    timing = report['timing']['blocks']
    lines = [f'# Auditoría física del pipeline: {label}', '', f"Ejecución: `{report['run']}`.", '',
        f"Se verificaron {report['raw_records_verified']} registros TX/RX y {report['completed_executions']} recorridos principales.",
        'Las características, entrada de la DNN, salida/clase e intermedios/score/decisión del detector coinciden exactamente',
        'con la referencia aislada fijada. Una extracción por RUN; tres RUN por ventana.', '',
        'Tiempos de los recorridos principales a HCLK HAL nominal de 180 MHz:', '',
        '| Bloque | Media (ms) | Máximo observado (ms) |', '|---|---:|---:|']
    for name in ('total', 'extract', 'input', 'nn_call', 'nn_invoke', 'detector', 'instrumentation_control_residual'):
        lines.append(f"| {name} | {timing[name]['mean_ms']:.9f} | {timing[name]['maximum_ms']:.9f} |")
    lines.extend(['', 'Invoke está incluido en nn_call y no se suma otra vez.', '',
        f"Inicialización DNN medida ahora: {report['initialization']['dnn_initialization_measured_now']}; "
        f"ciclos: {report['initialization']['nn_init_cycles']}. El segundo INIT reutilizó la instancia.",
        f"Arena usada: {report['firmware_info']['arena_used']} B; reserva: {report['firmware_info']['arena_capacity']} B.", '',
        f"Rechazos verificados: {report['negative_tests_passed']}; RUN auxiliares: {report['completed_lifecycle_support_executions']}.",
        'La auditoría solo reprodujo respuestas guardadas: no abrió COM ni repitió el experimento físico.', '',
        '## Alcance y pendientes', ''])
    lines.extend('- ' + value for value in report['limitations'])
    lines.extend(['', 'El criterio provisional de margen del 20% se conserva, sin ajustar tolerancias tras la placa.',
        'El sello de preparación y los resultados UART originales no se modificaron.', ''])
    with human.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write('\n'.join(lines))
    paths = [Path(__file__), Path(__file__).with_name('auditar_validacion_o3.py'), target, human,
        OUT / 'MANIFEST_SHA256.json', OUT / 'MANIFEST_SHA256.sha256', CLIENT / 'validar_pipeline.py',
        CLIENT / 'REFERENCIA_PIPELINE.json', CLIENT / 'VALIDACION_MANIFEST.json',
        folder / 'resultados.json', folder / 'uart_trace.jsonl', folder / 'MANIFEST_SHA256.json']
    manifest = {path.relative_to(ROOT).as_posix(): sha(path) for path in paths}
    with manifest_path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(manifest, stream, indent=2)
        stream.write('\n')
    with seal_path.open('x', encoding='ascii') as stream:
        stream.write(sha(manifest_path) + '  ' + manifest_path.name + '\n')
    require(all(sha(bounded(ROOT, name)) == expected for name, expected in manifest.items()), 'Sello físico diferente al cerrar')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--write-report', action='store_true')
    args = parser.parse_args()
    report, label, folder = audit(args.run)
    if args.write_report:
        write_report(report, label, folder)
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == '__main__':
    main()
