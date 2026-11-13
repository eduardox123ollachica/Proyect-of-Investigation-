"""Audita trazas CWRUMASK guardadas, sin COM ni repeticion de hardware.

Por defecto solo lee. --write-report crea informes y un sello fisico nuevo;
no modifica firmware, clientes, referencias, resultados o sellos anteriores.
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

import preparar_mascaras_cwru as stage
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
    preparation = json.loads((OUT / 'MANIFEST_SHA256.json').read_text(encoding='utf-8'))
    require(len(preparation) == 9757, 'Sello de preparacion distinto del banco revisado')
    manifest = json.loads((folder / 'MANIFEST_SHA256.json').read_text(encoding='utf-8'))
    require(set(manifest) == {'resultados.json', 'uart_trace.jsonl'}, 'Manifiesto fisico inesperado')
    for name, expected in manifest.items():
        require(sha(folder / name) == expected, 'Resultado fisico modificado: ' + name)
    run = json.loads((folder / 'resultados.json').read_text(encoding='utf-8'))
    require(run['status'] == 'PASS' and not run['simulation_only'] and run['firmware_executed']
            and run['serial_port_opened'], 'No es una prueba fisica aprobada')
    forbidden = ('training_executed', 'model_conversion_executed', 'detector_refitted',
                 'dnn_executed', 'detector_executed', 'outer_test_loaded', 'protocol_frozen',
                 'new_predictive_metrics_computed', 'selector_costs_enabled', 'whole_pipeline_savings_claimed',
                 'ram_savings_claimed', 'per_kernel_deployed_flash_claimed', 'wcet_claim',
                 'real_time_guarantee', 'canonical_latency_replaced',
                 'synthetic_absolute_pc_features_or_bins_claimed')
    require(all(run[key] is False for key in forbidden), 'Alcance distinto del contrato')
    require(run['timing_scope'] == 'feature_extraction_only' and run['repetitions'] == 3 and
            run['output_clearing_and_comparison_outside_timing'] is True and
            run['no_auxiliary_candidate_runs'] is True and run['nominal_hal_hclk_hz'] == 180000000,
            'Configuracion de medicion distinta')
    for key, name in (('client_sha256', 'validar_mascaras.py'),
                      ('reference_sha256', 'REFERENCIA_PIPELINE.json'),
                      ('bundle_manifest_sha256', 'VALIDACION_MANIFEST.json'),
                      ('catalog_sha256', 'CATALOGO_MASCARAS.json')):
        require(run[key] == sha(CLIENT / name), 'Recurso diferente del usado en placa: ' + name)
    spec = importlib.util.spec_from_file_location('sealed_mask_client', CLIENT / 'validar_mascaras.py')
    client = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(client)
    pc, ref, windows, catalog = client.load(CLIENT)
    mode = run['mode']
    require(mode in ('single', 'all', 'synthetic'), 'Modo no admitido')
    require(run['results'] and run['base_results'], 'Suite sin extracciones')
    kernel_ids = run['selected_kernel_ids']
    require(kernel_ids in ([0], list(range(10))), 'Seleccion de kernels fuera del catalogo previsto')
    if mode != 'single':
        require(kernel_ids == list(range(10)), 'La suite completa/esquina debe incluir los diez kernels')
    kernels = [catalog['kernels'][i] for i in kernel_ids]
    require(run['selected_kernel_names'] == [k['name'] for k in kernels], 'IDs y nombres no coinciden')
    index = None
    if mode == 'synthetic':
        cases = client.synthetic_cases(pc)
        require(run['sample_origin'] == 'deterministic_stimuli', 'Origen sintetico distinto')
        label, status = 'ESQUINA', 'PHYSICAL_MASKS_CORNER_SUITE_AUDITED'
    else:
        if mode == 'single':
            index = run['results'][0]['index']
            require(isinstance(index, int) and 0 <= index < 42, 'Indice invalido')
        selected = ref['cases'] if mode == 'all' else [ref['cases'][index]]
        cases = [dict(c, synthetic=False, window=windows[c['index']]) for c in selected]
        require(run['sample_origin'] == '42_fixed_development_windows_0_1_2HP', 'Origen real distinto')
        if mode == 'all':
            label, status = 'COMPLETA', 'PHYSICAL_MASKS_SUITE_AUDITED'
        else:
            suffix = 'F14' if kernel_ids == [0] else 'TODAS'
            label = f'INDICE_{index}_{suffix}'
            status = 'FIRST_MASK_F14_CASE_AUDITED' if kernel_ids == [0] else 'FIRST_ALL_MASKS_CASE_AUDITED'
    negative_suite = mode != 'single'
    expected_base = len(cases) * 3
    expected_runs = expected_base * len(kernels)
    expected_negatives = 10 if negative_suite else 0
    require(run['windows_expected'] == len(cases) and run['positive_expected_base_extractions'] == expected_base
            and run['positive_expected_runs'] == expected_runs and run['expected_negative_tests'] == expected_negatives,
            'Catalogo de ejecuciones distinto')
    records = [json.loads(line) for line in (folder / 'uart_trace.jsonl').read_text(encoding='utf-8').splitlines()]
    require(records and len(records) % 2 == 0, 'Numero de registros UART vacio o impar')
    for number, row in enumerate(records):
        direction = 'RX' if number % 2 else 'TX'
        require(row['direction'] == direction, 'Orden TX/RX distinto')
        value = row['command'] if direction == 'TX' else row['line']
        length = len((value + '\r\n').encode('ascii'))
        require(length == row['bytes'] and 0 < length < (1024 if direction == 'TX' else 512),
                'Longitud UART incorrecta o fuera del buffer')
    replay = dict(results=[], base_results=[], negative_tests=[], positive_window_transmissions=0)
    transport = RecordedTransport(records)
    with contextlib.redirect_stdout(io.StringIO()):
        client.session(pc, transport, catalog, cases, kernels, replay, [], timeout=1., run_negatives=negative_suite)
    require(transport.position == len(records), 'Tramas no asociadas a la suite')
    for key, value in replay.items():
        require(run[key] == value, 'JSON distinto de las respuestas registradas: ' + key)
    require(run['completed_runs'] == run['attempted_runs'] == len(run['results']) == expected_runs,
            'Numero de RUN aprobados distinto')
    require(run['completed_base_extractions'] == run['attempted_base_extractions'] == len(run['base_results'])
            == expected_base, 'Numero de BASE aprobados distinto')
    require(run['negative_tests_passed'] == len(run['negative_tests']) == expected_negatives,
            'Numero de rechazos distinto')
    sent_runs = sum(row['direction'] == 'TX' and row['command'].startswith('CWRUMASK_RUN,') for row in records)
    sent_base = sum(row['direction'] == 'TX' and row['command'].startswith('CWRUMASK_BASE,') for row in records)
    require(sent_runs == run['candidate_run_commands_sent'] == expected_runs + (6 if negative_suite else 0),
            'Numero de comandos RUN distinto; separar negativos de extracciones')
    require(sent_base == run['base_commands_sent'] == expected_base + (2 if negative_suite else 0),
            'Numero de comandos BASE distinto; separar negativos de extracciones')
    successful_run_replies = sum(row['direction'] == 'RX' and row['line'].startswith('CWRUMASK_RESULT,') for row in records)
    successful_base_replies = sum(row['direction'] == 'RX' and row['line'].startswith('CWRUMASK_BASE,') for row in records)
    require(successful_run_replies == expected_runs and successful_base_replies == expected_base,
            'Respuestas de extraccion no asociadas a los resultados')
    require(replay['positive_window_transmissions'] == len(cases), 'Numero de transferencias distinto')
    built = json.loads((OUT / 'COMPILACION_MASCARAS.json').read_text(encoding='utf-8'))['configurations'][stage.CONFIG]
    elf = stage.SOURCE / stage.CONFIG / (stage.NAME + '.elf')
    require(elf.is_file() and sha(elf) == built['elf_sha256'], 'ELF local ausente o distinto del preparado')
    installed = json.loads((OUT / 'APLICACION_EN_D.json').read_text(encoding='utf-8'))
    accepted_launch_reserializations = []
    for name, expected in installed['installed_sha256'].items():
        actual = sha(stage.SOURCE / name)
        if actual != expected:
            require(name.endswith('.launch') and stage.prior.launch_semantics(stage.SOURCE / name)
                    == stage.prior.launch_semantics(stage.OVERLAY / name), 'Fuente instalada distinta: ' + name)
            accepted_launch_reserializations.append(dict(path=name, old_sha256=expected,
                                                         current_sha256=actual, semantically_equal=True))
    prep = json.loads((OUT / 'PREPARACION.json').read_text(encoding='utf-8'))
    require(stage.same_configs(stage.SOURCE / '.cproject') and all(
        sha(stage.SOURCE / n / (stage.NAME + '.elf')) == d for n, d in prep['existing_elf_sha256'].items()),
        'Configuracion o ELF anterior cambio')
    original_float64_comparisons = 0 if mode == 'synthetic' else expected_base * 14
    selected_values = len(cases) * 3 * sum(len(k['indices']) for k in kernels)
    unselected_zero_values = 14 * expected_runs - selected_values
    stats = replay['statistics']
    f14_mean = stats['kernels']['F14']['mean_ms']
    old_mean = stats['canonical_entm_control']['mean_ms']
    real_cases = [] if mode == 'synthetic' else cases
    real_by_name = {case['name']: case for case in real_cases}
    selected_float64_count = 0
    selected_bin_count = 0
    # Contraste directo adicional: no se cuentan los slots de diagnostico +0
    # como caracteristicas calculadas, ni se imponen bins PC a sinteticos.
    for row in run['results'] if real_cases else []:
        case = real_by_name[row['name']]
        kernel = catalog['kernels'][row['kernel_id']]
        indices = np.asarray(kernel['indices'], dtype=int)
        values = np.frombuffer(bytes.fromhex(pc.raw_hex(row['observed']['features_bits'])), dtype='<f4').astype(np.float64)
        target = np.asarray(case['features_float64_expected'], dtype=np.float64)
        require(np.isclose(values[indices], target[indices], rtol=.001, atol=.00001).all(),
                'Caracteristica seleccionada no cumple el criterio float64 original')
        selected_float64_count += len(indices)
        for axis, feature in enumerate((4, 11)):
            if feature in kernel['indices']:
                observed_bin = int(np.rint(values[feature] / (12000 / 4096)))
                require(observed_bin == case['dominant_bins_expected'][axis], 'Bin seleccionado distinto de PC')
                selected_bin_count += 1
    report = dict(status=status, audited_utc=datetime.now(timezone.utc).isoformat(), run=run_name, mode=mode,
        development_index=index, windows=len(cases), synthetic_stimuli=mode == 'synthetic',
        stimulus_names=[c['metadata']['stimulus'] for c in cases] if mode == 'synthetic' else [],
        mat_files=len({c['metadata']['file_id'] for c in real_cases}),
        loads_hp=sorted({int(c['metadata']['load_hp']) for c in real_cases}),
        repetitions=3, selected_kernel_ids=kernel_ids, selected_kernel_names=run['selected_kernel_names'],
        passed_candidate_runs=expected_runs, passed_historical_base_extractions=expected_base,
        total_successful_extractions=expected_runs + expected_base, negative_tests_passed=expected_negatives,
        candidate_run_commands_sent=sent_runs, base_commands_sent=sent_base,
        negative_run_commands_without_extraction=6 if negative_suite else 0,
        negative_base_commands_without_extraction=2 if negative_suite else 0,
        auxiliary_successful_candidate_runs=0, all_selected_features_bit_exact=True,
        all_unselected_slots_positive_zero=True, all_repeated_numeric_outputs_stable=True,
        historical_baseline_feature_values_checked=14 * expected_base,
        selected_feature_values_checked=selected_values, unselected_zero_values_checked=unselected_zero_values,
        original_float64_direct_baseline_comparisons_passed=original_float64_comparisons,
        original_float64_direct_selected_comparisons_passed=selected_float64_count,
        original_float64_direct_total_comparisons_passed=original_float64_comparisons + selected_float64_count,
        dominant_bins_exact_on_real_baselines=0 if mode == 'synthetic' else 2 * expected_base,
        dominant_bins_exact_on_real_selected_outputs=selected_bin_count,
        selected_real_features_exact_to_float64_checked_baselines=mode != 'synthetic',
        synthetic_absolute_pc_features_or_bins_claimed=False,
        statistics=stats, mean_new_F14_minus_historical_EntM_ms=f14_mean - old_mean,
        temporal_difference_is_descriptive_not_predictive_or_causal_claim=True,
        firmware_before=replay['firmware_before'], firmware_after=replay['firmware_after'],
        raw_records_verified=len(records), recorded_input_output_replay_verified=True,
        positive_window_transmissions=len(cases),
        maximum_tx_bytes=max(r['bytes'] for r in records if r['direction'] == 'TX'),
        maximum_rx_bytes=max(r['bytes'] for r in records if r['direction'] == 'RX'),
        first_result_id=run['results'][0]['observed']['id'], last_result_id=run['results'][-1]['observed']['id'],
        results_sha256=sha(folder / 'resultados.json'), trace_sha256=sha(folder / 'uart_trace.jsonl'),
        run_manifest_sha256=sha(folder / 'MANIFEST_SHA256.json'),
        preparation_manifest_sha256=sha(OUT / 'MANIFEST_SHA256.json'), preparation_files_verified=len(preparation),
        auditor_sha256=sha(Path(__file__)), recorded_transport_sha256=sha(Path(__file__).with_name('auditar_validacion_o3.py')),
        prepared_elf_sha256=built['elf_sha256'], local_elf_matches_prepared=True, full_device_flash_read=False,
        accepted_launch_reserializations=accepted_launch_reserializations,
        previous_fourteen_configurations_and_elf_preserved=True,
        prepared_harness_flash_bytes=built['flash_load_bytes'],
        prepared_harness_static_ram_with_linker_reservations_bytes=built['ram_static_with_linker_reservations_bytes'],
        audit_python=sys.version, audit_numpy=np.__version__, serial_port_opened_during_audit=False,
        physical_experiment_repeated=False, firmware_changed_during_audit=False, training_executed=False,
        dnn_executed=False, detector_executed=False, predictive_metrics_computed=False,
        selector_costs_enabled=False, whole_pipeline_savings_claimed=False, ram_savings_claimed=False,
        per_kernel_deployed_flash_claimed=False, canonical_latency_replaced=False,
        limitations=[
            'Synthetic stimuli test compatibility with canonical ARM BASE, not absolute PC bins in tied spectra.' if mode == 'synthetic' else
            ('One development window is not the full 42-window masked-extractor suite.' if mode == 'single' else
             '42 fixed development windows and their repetitions are not independent motors or predictive V2 evaluation.'),
            'Temporal denominator is specialized F14 in this same harness, paired by window and repetition.',
            'Historical EntM BASE is measured separately; a small difference on one window is descriptive only.',
            'Fixed mask execution order is not a randomized or interleaved timing experiment.',
            'Times cover extraction API and selected writes; clearing, comparison and UART are outside DWT.',
            'This stage executes neither DNN nor detector and does not measure final pipeline savings.',
            'Fixed detector requires the union of inputs; shared-mask detector still requires refit and validation.',
            'Harness contains ten kernels and retains input/FFT buffers, so per-deployment Flash/RAM savings are not measured.',
            'Interrupts can occur; observed timings are not WCET or autonomous acquisition proof.',
            'Local ELF identity and declared catalog ID are not a complete device Flash readback.'])
    return report, label, folder


def next_step(report):
    if report['mode'] == 'single' and report['selected_kernel_ids'] == [0]:
        return ('Siguiente puerta: mismo indice con los diez kernels mediante --index '
                f"{report['development_index']} --all-masks, tras revisar este informe.")
    if report['mode'] == 'single':
        return 'Siguiente puerta: las mismas 42 ventanas de desarrollo con --all, tras revisar este informe.'
    if report['mode'] == 'all':
        return 'Siguiente puerta: suite de esquina separada con --synthetic; despues revisar costos y restricciones V2.'
    return 'Suite de esquina auditada; revisar junto con las 42 ventanas antes de congelar costos, reglas y presupuesto V2.'


def write_report(report, label, folder):
    target = OUT / f'RESULTADOS_FISICOS_MASCARAS_{label}.json'
    human = OUT / f'RESULTADOS_VALIDACION_MASCARAS_{label}.md'
    manifest = OUT / f'MANIFEST_RESULTADOS_FISICOS_MASCARAS_{label}_SHA256.json'
    seal = manifest.with_suffix('.sha256')
    require(not any(p.exists() for p in (target, human, manifest, seal)), 'No sobrescribir informes fisicos')
    with target.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    stats = report['statistics']
    lines = [f'# Auditoria fisica de kernels por mascara: {label}', '', f"Ejecucion `{report['run']}`.", '',
        f"Aprobados {report['passed_historical_base_extractions']} BASE historicos, "
        f"{report['passed_candidate_runs']} RUN de kernels y {report['negative_tests_passed']} rechazos.",
        f"Verificados {report['raw_records_verified']} registros TX/RX guardados, sin abrir COM ni repetir hardware.",
        'Valores seleccionados exactos frente a BASE; slots omitidos +0; repeticiones numericamente estables.', '',
        'Tiempos de extraccion a HCLK HAL nominal de 180 MHz:', '',
        '| Implementacion | Media (ms) | Maximo observado (ms) | P95 (ms) |', '|---|---:|---:|---:|']
    old = stats['canonical_entm_control']
    lines.append(f"| EntM historico BASE | {old['mean_ms']:.9f} | {old['max_cycles']/180000:.9f} | {old['p95_ms']:.9f} |")
    for name, value in stats['kernels'].items():
        lines.append(f"| {name} nuevo | {value['mean_ms']:.9f} | {value['max_cycles']/180000:.9f} | {value['p95_ms']:.9f} |")
    lines.extend(['', f"Diferencia descriptiva F14 nuevo menos BASE historico: "
        f"{report['mean_new_F14_minus_historical_EntM_ms']:+.9f} ms.",
        'Los ratios del cliente usan el F14 nuevo del mismo banco, emparejado por ventana y repeticion.',
        'El perfil instrumentado anterior no es el denominador. Esta diferencia no es una mejora predictiva ni una prueba causal.', '',
        f"Comparaciones directas BASE frente a PC float64: {report['original_float64_direct_baseline_comparisons_passed']}; "
        f"bins reales exactos: {report['dominant_bins_exact_on_real_baselines']}.",
        f"Comparaciones directas de slots seleccionados frente a PC float64: "
        f"{report['original_float64_direct_selected_comparisons_passed']}; "
        f"bins seleccionados exactos: {report['dominant_bins_exact_on_real_selected_outputs']}.",
        f"Valores seleccionados comprobados: {report['selected_feature_values_checked']}; "
        f"slots omitidos verificados en +0: {report['unselected_zero_values_checked']}.",
        f"Comandos enviados RUN/BASE: {report['candidate_run_commands_sent']}/{report['base_commands_sent']}; "
        f"extracciones aprobadas RUN/BASE: {report['passed_candidate_runs']}/{report['passed_historical_base_extractions']}.",
        'Los comandos negativos no ejecutan extracciones; no se agregan a la distribucion temporal.', '',
        '## Alcance y siguiente paso', ''])
    lines.extend('- ' + value for value in report['limitations'])
    lines.extend(['', next_step(report), '',
        'Datos originales, fuentes y sellos anteriores permanecen intactos. Se crea un sello fisico separado.', ''])
    with human.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write('\n'.join(lines))
    paths = [Path(__file__), Path(__file__).with_name('auditar_validacion_o3.py'), target, human,
             OUT / 'MANIFEST_SHA256.json', OUT / 'MANIFEST_SHA256.sha256', OUT / 'COMPILACION_MASCARAS.json',
             OUT / 'CATALOGO_MASCARAS.json', CLIENT / 'VALIDACION_MANIFEST.json',
             *(CLIENT / name for name in stage.CLIENT_NAMES),
             folder / 'resultados.json', folder / 'uart_trace.jsonl', folder / 'MANIFEST_SHA256.json']
    entries = {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}
    with manifest.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(entries, stream, indent=2)
        stream.write('\n')
    with seal.open('x', encoding='ascii') as stream:
        stream.write(sha(manifest) + '  ' + manifest.name + '\n')
    require(all(sha(bounded(ROOT, name)) == digest for name, digest in entries.items()), 'Sello fisico diferente al cerrar')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--write-report', action='store_true')
    args = parser.parse_args()
    report, label, folder = audit(args.run)
    if args.write_report:
        write_report(report, label, folder)
    print(json.dumps(report, ensure_ascii=True, indent=2))
