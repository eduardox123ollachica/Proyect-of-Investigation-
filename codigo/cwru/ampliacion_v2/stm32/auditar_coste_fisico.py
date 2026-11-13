"""Audita CWRUCOST por reproduccion de respuestas existentes, sin COM.

La opcion --write-report crea informes y un sello fisico independiente.
No modifica firmware, clientes, referencias o sellos anteriores.
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

import preparar_coste_cwru as stage
from auditar_validacion_o3 import RecordedTransport, sha

ROOT, OUT, CLIENT = stage.ROOT, stage.OUT, stage.CLIENT


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def bounded(base, name):
    path = (base / name).resolve()
    require(path != base.resolve() and path.is_relative_to(base.resolve()), 'Ruta fuera de la auditoria')
    return path


def audit(run_name):
    require(Path(run_name).name == run_name and run_name not in ('.', '..'), 'Nombre de ejecucion invalido')
    folder = bounded(CLIENT / 'resultados', run_name)
    with contextlib.redirect_stdout(io.StringIO()):
        stage.verify()
    manifest = json.loads((folder / 'MANIFEST_SHA256.json').read_text(encoding='utf-8'))
    require(set(manifest) == {'resultados.json', 'uart_trace.jsonl'}, 'Manifiesto fisico inesperado')
    for name, expected in manifest.items():
        require(sha(folder / name) == expected, 'Resultado fisico modificado: ' + name)
    run = json.loads((folder / 'resultados.json').read_text(encoding='utf-8'))
    require(run['status'] == 'PASS' and not run['simulation_only'] and run['firmware_executed']
            and run['serial_port_opened'], 'No es una prueba fisica aprobada')
    forbidden = ('training_executed', 'model_conversion_executed', 'threshold_refitted',
                 'outer_test_loaded', 'protocol_frozen', 'new_predictive_metrics_computed',
                 'wcet_claim', 'real_time_guarantee', 'canonical_latency_replaced',
                 'selector_costs_enabled', 'marginal_feature_costs_identifiable',
                 'operational_latency_limit_applied', 'overhead_subtracted')
    require(not any(run[key] for key in forbidden), 'Alcance distinto del contrato')
    for key, name in (('client_sha256', 'validar_coste.py'),
                      ('reference_sha256', 'REFERENCIA_PIPELINE.json'),
                      ('bundle_manifest_sha256', 'VALIDACION_MANIFEST.json'),
                      ('dependency_graph_sha256', 'GRAFO_DEPENDENCIAS.json')):
        require(run[key] == sha(CLIENT / name), 'Recurso diferente del usado en placa: ' + name)
    spec = importlib.util.spec_from_file_location('sealed_cost_client', CLIENT / 'validar_coste.py')
    client = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(client)
    pc, ref, windows = client.load(CLIENT)
    require(run['mode'] in ('single', 'all') and run['repetitions'] == 3, 'Modo/repeticiones distintos')
    require(run['results'], 'Suite vacia')
    all_cases = run['mode'] == 'all'
    index = None if all_cases else run['results'][0]['index']
    require(index is None or isinstance(index, int) and 0 <= index < 42, 'Indice invalido')
    selected = ref['cases'] if all_cases else [ref['cases'][index]]
    expected_runs, expected_negatives = 3 * len(selected), 8 if all_cases else 0
    require(run['group_order'] == list(client.BLOCKS) and run['expected_auxiliary_runs'] == 0
            and run['positive_expected_runs'] == expected_runs
            and run['expected_negative_tests'] == expected_negatives, 'Catalogo diferente')
    records = [json.loads(line) for line in (folder / 'uart_trace.jsonl').read_text(encoding='utf-8').splitlines()]
    require(len(records) % 2 == 0, 'Numero de registros UART impar')
    for number, row in enumerate(records):
        direction = 'RX' if number % 2 else 'TX'
        require(row['direction'] == direction, 'Orden TX/RX diferente')
        value = row['command'] if direction == 'TX' else row['line']
        length = len((value + '\r\n').encode('ascii'))
        require(length == row['bytes'] and 0 < length < (1024 if direction == 'TX' else 512),
                'Longitud UART incorrecta o fuera del buffer')
    replay = dict(results=[], negative_tests=[], positive_window_transmissions=0)
    transport = RecordedTransport(records)
    with contextlib.redirect_stdout(io.StringIO()):
        client.session(pc, transport, ref, windows, selected, replay, [], all_cases=all_cases)
    require(transport.position == len(records), 'Tramas no asociadas a la suite')
    for key, value in replay.items():
        require(run[key] == value, 'JSON distinto de las respuestas registradas: ' + key)
    run_commands = sum(row['direction'] == 'TX' and row['command'].startswith('CWRUPIPE_RUN,') for row in records)
    require(run['completed_runs'] == run['attempted_runs'] == len(run['results']) == run_commands
            == run['pipeline_run_commands_sent'] == expected_runs, 'Numero de RUN incoherente')
    require(len(run['negative_tests']) == run['negative_tests_passed'] == expected_negatives,
            'Numero de rechazos incoherente')
    before, init, info, after = (replay[k] for k in ('pipeline_before', 'initialization', 'pipeline_info', 'pipeline_after'))
    require(before['nn_status'] == (0 if init['reused'] else -100)
            and before['arena_used'] == (ref['expected_arena_used'] if init['reused'] else 0), 'Estado de arranque incoherente')
    stable_keys = set(info) - {'fpscr', 'result_valid', 'last_id'}
    require(all(info[key] == after[key] for key in stable_keys), 'Entorno o parametros cambiaron')
    cost_before, cost_after = replay['cost_info'], replay['cost_info_after']
    stable_cost = set(cost_before) - {'paired_read_cycles_min'}
    require(all(cost_before[k] == cost_after[k] for k in stable_cost), 'Configuracion de perfil cambio')
    built = json.loads((OUT / 'COMPILACION_COSTE.json').read_text(encoding='utf-8'))['configurations'][stage.CONFIG]
    elf = stage.SOURCE / stage.CONFIG / (stage.NAME + '.elf')
    require(elf.is_file() and sha(elf) == built['elf_sha256'], 'ELF local ausente o diferente del preparado')
    installed = json.loads((OUT / 'APLICACION_EN_D.json').read_text(encoding='utf-8'))
    accepted_launch_reserializations = []
    for name, expected in installed['installed_sha256'].items():
        actual = sha(stage.SOURCE / name)
        if actual != expected:
            require(name.endswith('.launch') and stage.prior.launch_semantics(stage.SOURCE / name)
                    == stage.prior.launch_semantics(stage.OVERLAY / name), 'Fuente instalada diferente: ' + name)
            accepted_launch_reserializations.append(dict(path=name, old_sha256=expected,
                                                         current_sha256=actual, semantically_equal=True))
    prep = json.loads((OUT / 'PREPARACION.json').read_text(encoding='utf-8'))
    require(stage.same_configs(stage.SOURCE / '.cproject') and all(
        sha(stage.SOURCE / n / (stage.NAME + '.elf')) == d for n, d in prep['existing_elf_sha256'].items()),
        'Configuracion o ELF anterior cambio')
    rows = replay['results']
    residuals = [row['profile']['extraction_unassigned_residual_cycles'] for row in rows]
    report = dict(status='PHYSICAL_COST_SUITE_AUDITED' if all_cases else 'FIRST_COST_CASE_AUDITED',
        audited_utc=datetime.now(timezone.utc).isoformat(), run=run_name, mode=run['mode'], development_index=index,
        windows=len(selected), mat_files=len({c['metadata']['file_id'] for c in selected}),
        loads_hp=sorted({int(c['metadata']['load_hp']) for c in selected}), repetitions_per_window=3,
        passed_main_runs=expected_runs, negative_tests_passed=expected_negatives,
        auxiliary_pipeline_runs=0, pipeline_run_commands_sent=run_commands, one_extraction_per_run=True,
        all_intermediates_and_outputs_bit_exact=True, all_repeated_numeric_outputs_stable=True,
        original_float64_comparisons_passed=14 * expected_runs, dominant_bins_exact=2 * expected_runs,
        profile_counter_values_verified=14 * expected_runs, profile_id_checksum_exact=True,
        all_profile_sums_within_extraction=True, residual_cycles_range=[min(residuals), max(residuals)],
        initialization=init, reuse=replay['reuse'], pipeline_info=info, pipeline_info_after=after,
        cost_info=cost_before, cost_info_after=cost_after, diagnostic_statistics=replay['diagnostic_statistics'],
        raw_records_verified=len(records), recorded_input_output_replay_verified=True,
        positive_window_transmissions=replay['positive_window_transmissions'],
        maximum_tx_bytes=max(r['bytes'] for r in records if r['direction'] == 'TX'),
        maximum_rx_bytes=max(r['bytes'] for r in records if r['direction'] == 'RX'),
        first_result_id=rows[0]['observed']['id'], last_result_id=rows[-1]['observed']['id'],
        results_sha256=sha(folder / 'resultados.json'), trace_sha256=sha(folder / 'uart_trace.jsonl'),
        run_manifest_sha256=sha(folder / 'MANIFEST_SHA256.json'),
        preparation_manifest_sha256=sha(OUT / 'MANIFEST_SHA256.json'),
        preparation_files_verified=len(json.loads((OUT / 'MANIFEST_SHA256.json').read_text(encoding='utf-8'))),
        auditor_sha256=sha(Path(__file__)), recorded_transport_sha256=sha(Path(__file__).with_name('auditar_validacion_o3.py')),
        prepared_elf_sha256=built['elf_sha256'], local_elf_matches_prepared=True, full_device_flash_read=False,
        accepted_launch_reserializations=accepted_launch_reserializations,
        previous_thirteen_configurations_and_elf_preserved=True,
        prepared_flash_bytes=built['flash_load_bytes'],
        prepared_static_ram_with_linker_reservations_bytes=built['ram_static_with_linker_reservations_bytes'],
        audit_python=sys.version, audit_numpy=np.__version__, serial_port_opened_during_audit=False,
        physical_experiment_repeated=False, firmware_changed_during_audit=False, training_executed=False,
        predictive_metrics_computed=False, selector_costs_enabled=False, marginal_feature_costs_identifiable=False,
        reduced_mask_latency_predictions=False, canonical_latency_replaced=False, overhead_subtracted=False,
        limitations=[
            'First development window is not the full 42-window profile.' if not all_cases else
            '42 fixed development windows are not independent motors or predictive V2 evaluation.',
            'Intervals between compiled markers contain fused shared operations, not marginal feature costs.',
            'Compiler scheduling and profiling can alter attribution and latency; this is a diagnostic build.',
            'Profile reading calibration is orientative and not subtracted; interrupts remain possible.',
            'Residual is unassigned, not a separately identified operator.',
            'No canonical latency replacement, operational latency pass/fail, WCET or autonomous acquisition claim.',
            'Dropping DNN columns alone does not remove computation or buffers; fixed detector needs feature union.',
            'Candidate kernels or validated cost model and V2 protocol/training remain pending.',
            'Local ELF identity and declared artifact IDs are not a complete device Flash readback.'])
    return report, ('COMPLETA' if all_cases else f'INDICE_{index}'), folder


def write_report(report, label, folder):
    target = OUT / f'RESULTADOS_FISICOS_COSTE_{label}.json'
    human = OUT / f'RESULTADOS_VALIDACION_COSTE_{label}.md'
    manifest = OUT / f'MANIFEST_RESULTADOS_FISICOS_COSTE_{label}_SHA256.json'
    seal = manifest.with_suffix('.sha256')
    require(not any(p.exists() for p in (target, human, manifest, seal)), 'No sobrescribir informes fisicos')
    with target.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    stats = report['diagnostic_statistics']
    lines = [f'# Perfil fisico compartido: {label}', '', f"Ejecucion `{report['run']}`.", '',
        f"Aprobados {report['passed_main_runs']} RUN, {report['negative_tests_passed']} rechazos y cero RUN auxiliares.",
        f"Se reprodujeron {report['raw_records_verified']} registros TX/RX guardados, sin abrir COM ni repetir hardware.",
        'Caracteristicas, escalado/cuantizacion, DNN y detector exactos frente a las referencias previas.', '',
        'Promedios DE+FE, HCLK HAL nominal de 180 MHz. Los porcentajes usan la suma de grupos, sin residual.', '',
        '| Grupo fusionado | Media (ms) | Porcentaje de grupos |', '|---|---:|---:|']
    for name, values in stats['groups_DE_plus_FE'].items():
        lines.append(f"| {name} | {values['mean_ms']:.9f} | {values['share_of_measured_groups_mean_percent']:.4f} % |")
    lines.extend(['', f"Extraccion instrumentada media: {stats['extraction_envelope']['mean_ms']:.9f} ms.",
        f"Total integrado instrumentado medio: {stats['pipeline_total_instrumented']['mean_ms']:.9f} ms.",
        f"Residual medio del extractor: {stats['extraction_unassigned_residual']['mean_ms']:.9f} ms; "
        f"rango de ciclos: {report['residual_cycles_range']}.",
        f"Calibracion inicial/final de lecturas: {report['cost_info']['paired_read_cycles_min']}/"
        f"{report['cost_info_after']['paired_read_cycles_min']} ciclos; no se sustrae.", '',
        'La entropia es un grupo exclusivo de esa salida, pero este perfil solo no mide el ahorro real al retirarla.',
        'No se descuentan bloques fusionados para predecir latencias de otras mascaras.', '',
        '## Alcance', ''])
    lines.extend('- ' + value for value in report['limitations'])
    lines.extend(['', 'Se conservan los resultados y sellos anteriores. El maximo previo del stack no se vuelve a medir aqui.',
        'Corresponde ampliar a las mismas 42 ventanas tras aprobar el indice individual.' if label != 'COMPLETA' else
        'El perfil completo queda auditado; aun se requiere validar costos de candidatos antes del protocolo V2.', ''])
    with human.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write('\n'.join(lines))
    paths = [Path(__file__), Path(__file__).with_name('auditar_validacion_o3.py'), target, human,
             OUT / 'MANIFEST_SHA256.json', OUT / 'MANIFEST_SHA256.sha256', OUT / 'COMPILACION_COSTE.json',
             OUT / 'GRAFO_DEPENDENCIAS.json', CLIENT / 'VALIDACION_MANIFEST.json',
             *(CLIENT / name for name in stage.CLIENT_NAMES),
             folder / 'resultados.json', folder / 'uart_trace.jsonl', folder / 'MANIFEST_SHA256.json']
    entries = {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}
    with manifest.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(entries, stream, indent=2)
        stream.write('\n')
    with seal.open('x', encoding='ascii') as stream:
        stream.write(sha(manifest) + '  ' + manifest.name + '\n')
    require(all(sha(bounded(ROOT, n)) == d for n, d in entries.items()), 'Sello fisico diferente al cerrar')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--write-report', action='store_true')
    args = parser.parse_args()
    report, label, folder = audit(args.run)
    if args.write_report:
        write_report(report, label, folder)
    print(json.dumps(report, ensure_ascii=True, indent=2))
