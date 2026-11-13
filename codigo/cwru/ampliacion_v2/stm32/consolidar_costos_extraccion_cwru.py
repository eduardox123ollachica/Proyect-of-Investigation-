"""Consolida costos de extraccion ya auditados; no mide, entrena ni abre COM.

Por defecto solo lee y muestra JSON. --write crea un cierre y un sello nuevos
sin sobrescribir archivos. Los tiempos de esquina no se usan como costos CWRU.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / 'despliegue_stm32/costos_mascaras_180mhz_2026-10-24'
EXPECTED = (('F14', 0x3fff), ('T6', 0x0387), ('S8', 0x3c78), ('NF12', 0x37ef),
            ('NM10', 0x3cf9), ('NE12', 0x1fbf), ('DE7', 0x007f), ('FE7', 0x3f80),
            ('FIX_DE13', 0x37ff), ('FIX_FE13', 0x3fef))


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def bounded(name):
    path = (ROOT / name).resolve()
    require(path != ROOT.resolve() and path.is_relative_to(ROOT.resolve()), 'Ruta fuera del proyecto')
    return path


def physical_manifest(label):
    manifest = OUT / f'MANIFEST_RESULTADOS_FISICOS_MASCARAS_{label}_SHA256.json'
    seal = manifest.with_suffix('.sha256')
    digest, name = seal.read_text(encoding='ascii').split()
    require(name == manifest.name and sha(manifest) == digest, 'Sello fisico distinto: ' + label)
    entries = read(manifest)
    require(isinstance(entries, dict) and len(entries) == 18, 'Se esperaban 18 recursos fisicos: ' + label)
    for relative, expected in entries.items():
        require(sha(bounded(relative)) == expected, 'Recurso fisico modificado: ' + relative)
    return entries, manifest, seal


def checked_report(label, entries):
    path = OUT / f'RESULTADOS_FISICOS_MASCARAS_{label}.json'
    require(entries.get(path.relative_to(ROOT).as_posix()) == sha(path), 'Informe fuera del sello')
    report = read(path)
    real = label == 'COMPLETA'
    require(report['status'] == ('PHYSICAL_MASKS_SUITE_AUDITED' if real else 'PHYSICAL_MASKS_CORNER_SUITE_AUDITED')
            and report['mode'] == ('all' if real else 'synthetic'), 'Estado fisico distinto: ' + label)
    require(report['windows'] == (42 if real else 4) and report['synthetic_stimuli'] == (not real)
            and report['passed_candidate_runs'] == (1260 if real else 120)
            and report['passed_historical_base_extractions'] == (126 if real else 12)
            and report['negative_tests_passed'] == 10 and report['repetitions'] == 3,
            'Cobertura fisica distinta: ' + label)
    require(report['candidate_run_commands_sent'] == report['passed_candidate_runs'] + 6
            and report['base_commands_sent'] == report['passed_historical_base_extractions'] + 2,
            'Comandos negativos confundidos con extracciones')
    require(report['selected_kernel_ids'] == list(range(10)) and
            report['selected_kernel_names'] == [name for name, _ in EXPECTED] and
            report['all_selected_features_bit_exact'] and report['all_unselected_slots_positive_zero']
            and report['all_repeated_numeric_outputs_stable'] and report['local_elf_matches_prepared'],
            'Compatibilidad de kernels no auditada')
    require(report['preparation_files_verified'] == 9757 and
            report['synthetic_absolute_pc_features_or_bins_claimed'] is False, 'Alcance de auditoria distinto')
    for key in ('serial_port_opened_during_audit', 'physical_experiment_repeated', 'firmware_changed_during_audit',
                'training_executed', 'dnn_executed', 'detector_executed', 'predictive_metrics_computed',
                'selector_costs_enabled', 'whole_pipeline_savings_claimed', 'ram_savings_claimed',
                'per_kernel_deployed_flash_claimed', 'canonical_latency_replaced'):
        require(report[key] is False, 'Alcance no permitido: ' + key)
    for info in (report['firmware_before'], report['firmware_after']):
        require(info['hclk_hz'] == 180000000 and info['catalog_fnv1a'] == 'fa1040f2' and info['dwt_enabled']
                and info['rounding'] == 'nearest_even' and not info['flush_to_zero'], 'Entorno diferente')
    if real:
        require(report['mat_files'] == 21 and report['loads_hp'] == [0, 1, 2], 'Corpus real diferente')
        require(report['original_float64_direct_total_comparisons_passed'] == 14616, 'Contraste float64 incompleto')
    else:
        require(report['stimulus_names'] == ['zeros', 'constant', 'nyquist', 'impulse'] and
                report['mat_files'] == 0 and report['loads_hp'] == [] and
                report['original_float64_direct_total_comparisons_passed'] == 0 and
                report['dominant_bins_exact_on_real_baselines'] == 0 and
                report['dominant_bins_exact_on_real_selected_outputs'] == 0,
                'Sinteticos mezclados con referencias absolutas PC/CWRU')
    return report, path


def extraction_stats(value, clock):
    require(value['count'] == 126 and value['paired_new_f14_count'] == 126, 'Tiempos sin cobertura emparejada')
    for name in ('mean_ms', 'p95_ms', 'sample_std_ms', 'mean_paired_ratio_to_new_f14',
                 'mean_paired_extraction_reduction_fraction'):
        require(math.isfinite(value[name]), 'Estadistica no finita')
    require(0 < value['min_cycles'] <= value['mean_cycles'] <= value['max_cycles'] <= 0xffffffff,
            'Ciclos invalidos')
    require(math.isclose(value['mean_ms'], value['mean_cycles'] * 1000 / clock, rel_tol=0, abs_tol=1e-10)
            and math.isclose(1 - value['mean_paired_ratio_to_new_f14'],
                             value['mean_paired_extraction_reduction_fraction'], rel_tol=0, abs_tol=1e-12),
            'Conversion de ciclos o ratio diferente')
    return dict(observations=126, mean_ms=value['mean_ms'], min_ms=value['min_cycles'] * 1000 / clock,
                max_observed_ms=value['max_cycles'] * 1000 / clock, p95_ms=value['p95_ms'],
                sample_std_ms=value['sample_std_ms'],
                mean_paired_ratio_to_new_F14=value['mean_paired_ratio_to_new_f14'],
                mean_paired_extraction_reduction_fraction=value['mean_paired_extraction_reduction_fraction'])


def consolidate():
    source_entries = {}
    manifests = []
    reports = {}
    for label in ('COMPLETA', 'ESQUINA'):
        entries, manifest, seal = physical_manifest(label)
        for name, digest in entries.items():
            require(name not in source_entries or source_entries[name] == digest, 'Entrada compartida con hashes distintos')
        source_entries.update(entries)
        source_entries.update({manifest.relative_to(ROOT).as_posix(): sha(manifest),
                               seal.relative_to(ROOT).as_posix(): sha(seal)})
        report, path = checked_report(label, entries)
        reports[label] = report
        manifests.append(dict(label=label, report=path.relative_to(ROOT).as_posix(), report_sha256=sha(path),
                              physical_manifest=manifest.relative_to(ROOT).as_posix(), physical_manifest_sha256=sha(manifest)))
    full, corner = reports['COMPLETA'], reports['ESQUINA']
    for key in ('preparation_manifest_sha256', 'prepared_elf_sha256', 'prepared_harness_flash_bytes',
                'prepared_harness_static_ram_with_linker_reservations_bytes', 'auditor_sha256'):
        require(full[key] == corner[key], 'Auditorias de paquetes/ELF diferentes: ' + key)
    require(full['prepared_harness_flash_bytes'] == 107952 and
            full['prepared_harness_static_ram_with_linker_reservations_bytes'] == 71228,
            'Recursos del banco diferentes')
    catalog_path = OUT / 'CATALOGO_MASCARAS.json'
    require(source_entries.get(catalog_path.relative_to(ROOT).as_posix()) == sha(catalog_path), 'Catalogo no sellado')
    catalog = read(catalog_path)
    rows = catalog['kernels']
    require(len(rows) == 10 and catalog['catalog_fnv1a'] == 'fa1040f2'
            and catalog['fixed_detector_feature_mask'] == '37ef' and catalog['selector_costs_enabled'] is False,
            'Catalogo distinto')
    fnv = 2166136261
    for index, ((name, mask), row) in enumerate(zip(EXPECTED, rows)):
        require(row['id'] == index and row['name'] == name and row['feature_mask'] == f'{mask:04x}' and
                row['indices'] == [i for i in range(14) if mask & (1 << i)], 'Mascara distinta')
        for byte in mask.to_bytes(2, 'little'):
            fnv = ((fnv ^ byte) * 16777619) & 0xffffffff
    require(f'{fnv:08x}' == catalog['catalog_fnv1a'], 'FNV del catalogo diferente')
    require(full['statistics']['comparison_denominator'] ==
            'new_specialized_F14_same_harness_paired_by_window_and_repetition' and
            full['statistics']['historical_instrumented_profile_used_as_denominator'] is False and
            full['statistics']['timing_scope'] == 'feature_extraction_only', 'Denominador temporal distinto')
    stats = {name: extraction_stats(full['statistics']['kernels'][name], 180000000) for name, _ in EXPECTED}
    require(stats['F14']['mean_paired_ratio_to_new_F14'] == 1., 'Referencia F14 distinta')
    by_mask = {mask: index for index, (_, mask) in enumerate(EXPECTED)}
    candidate_rows = []
    for candidate in rows[:8]:
        mask = int(candidate['feature_mask'], 16)
        union = mask | 0x37ef
        fixed_id = by_mask[union]
        fixed = rows[fixed_id]
        require(catalog['candidate_to_fixed_union_kernel_id'][candidate['name']] == fixed_id,
                'Union con detector fijo distinta')
        shared_detector_mask = mask & ~(1 << 4) & ~(1 << 11)
        candidate_rows.append(dict(name=candidate['name'], dnn_input_dimension=len(candidate['indices']),
            dnn_feature_mask=candidate['feature_mask'], shared_extraction_kernel=candidate['name'],
            shared_required_feature_mask=candidate['feature_mask'],
            shared_detector_future_feature_mask=f'{shared_detector_mask:04x}', shared_detector_refit_pending=True,
            shared_extraction_statistics=stats[candidate['name']],
            fixed_detector_feature_mask='37ef', fixed_union_feature_mask=f'{union:04x}',
            fixed_union_kernel=fixed['name'], fixed_extraction_dimension=len(fixed['indices']),
            fixed_extraction_statistics=stats[fixed['name']],
            full_pipeline_cost_ms=None, DNN_cost_ms=None, detector_cost_ms=None,
            per_candidate_deployed_flash_bytes=None, per_candidate_deployed_ram_bytes=None))
    result = dict(schema_version=1, status='EXTRACTION_COSTS_CONSOLIDATED_NOT_FULL_PIPELINE_COST',
        generated_utc=datetime.now(timezone.utc).isoformat(), generator_sha256=sha(Path(__file__)),
        scope='measured extraction kernels, eight candidate representations and fixed-detector unions',
        clock=dict(hclk_hal_nominal_hz=180000000, externally_measured=False),
        signal=dict(sample_rate_hz=12000, samples=4096, sensor_positions=['DE', 'FE']),
        real_cost_evidence=dict(windows=42, mat_files=21, loads_hp=[0, 1, 2], repetitions=3,
            run=full['run'], passed_base_extractions=126, passed_mask_extractions=1260,
            negative_tests_passed=10, observations_per_kernel=126,
            repeated_windows_are_not_independent_motors=True),
        corner_compatibility_evidence=dict(run=corner['run'], stimuli=corner['stimulus_names'],
            passed_base_extractions=12, passed_mask_extractions=120, negative_tests_passed=10,
            selected_values_bit_exact=True, absolute_PC_bins_claimed=False, timings_used_for_costs=False),
        catalog_fnv1a=catalog['catalog_fnv1a'], candidates=candidate_rows,
        fixed_detector_max_observed_extraction_reduction_fraction=max(
            r['fixed_extraction_statistics']['mean_paired_extraction_reduction_fraction'] for r in candidate_rows),
        canonical_EntM_control_from_real_suite=full['statistics']['canonical_entm_control'],
        harness_only_resources=dict(flash_bytes=107952, static_ram_with_linker_reservations_bytes=71228,
            per_candidate_deployment_resources_measured=False, buffers_reduced=False),
        full_pipeline_cost_ms=None, DNN_cost_ms=None, detector_cost_ms=None,
        selector_total_cost_ready=False, selector_costs_enabled=False, training_enabled=False,
        protocol_v2_frozen=False, shared_detector_refit_pending=True, predictive_metrics_computed=False,
        manuscript_modified=False, serial_port_opened=False, physical_experiments_repeated=False,
        initial_preparation_full_reverification_performed=False,
        physical_auditors_preparation_files_verified=9757, source_audits=manifests,
        source_evidence_sha256=source_entries,
        limitations=[
            'Only the 42 real development windows provide the timing costs; corner stimuli provide compatibility evidence separately.',
            'Ratios are paired by window/repetition against specialized F14 in the same harness, not the instrumented profile.',
            'Fixed execution order and repeated windows do not support independent-sample confidence intervals or causal claims.',
            'All negative measured reductions are preserved, without clipping or imposing monotonic cost versus input count.',
            'A shared-mask detector needs refitting and predictive evaluation; the fixed detector needs the union of feature inputs.',
            'DNN, detector and whole-pipeline times for refitted models remain unavailable, not zero.',
            'Multi-kernel harness Flash and unchanged buffers are not per-deployment Flash/RAM savings.',
            'Timing excludes clearing, comparison, UART, acquisition, initialization and offline FIR; observed maxima are not WCET.',
            'The predictive protocol, constraints, pilot training time and total execution budget still require freeze.'])
    return result


def write_closure(result):
    target = OUT / 'CIERRE_COSTOS_EXTRACCION.json'
    human = OUT / 'CIERRE_COSTOS_EXTRACCION.md'
    manifest = OUT / 'MANIFEST_CIERRE_COSTOS_EXTRACCION_SHA256.json'
    seal = manifest.with_suffix('.sha256')
    require(not any(p.exists() for p in (target, human, manifest, seal)), 'No sobrescribir el cierre existente')
    with target.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    lines = ['# Cierre de costos medidos de extraccion', '',
        'Se consolidaron las 42 ventanas de desarrollo (21 MAT, cargas 0/1/2 HP), tres repeticiones por kernel.',
        'Los cuatro estimulos de esquina respaldan compatibilidad y permanecen separados de estos tiempos.', '',
        'Ambas condiciones siguientes se refieren solo a extraccion. La condicion compartida es una representacion',
        'que el futuro detector debera reajustar y validar; no es otro detector ya entrenado.', '',
        '| Candidata | Entradas DNN | Extraccion S (ms) | Union con detector fijo | Extraccion union (ms) | Reduccion S frente F14 (%) |',
        '|---|---:|---:|---|---:|---:|']
    for row in result['candidates']:
        shared, fixed = row['shared_extraction_statistics'], row['fixed_extraction_statistics']
        lines.append(f"| {row['name']} | {row['dnn_input_dimension']} | {shared['mean_ms']:.6f} | "
                     f"{row['fixed_union_kernel']} | {fixed['mean_ms']:.6f} | "
                     f"{100 * shared['mean_paired_extraction_reduction_fraction']:.4f} |")
    if result['fixed_detector_max_observed_extraction_reduction_fraction'] <= 0:
        lines.extend(['', 'En estas condiciones las uniones del detector fijo no mostraron ahorro de extraccion frente a F14.',
            'Se conservan las pequenas regresiones de NF12/FIX; no se ajustan los resultados para forzar monotonia.'])
    lines.extend(['', 'Los porcentajes son medias de ratios emparejados por ventana y repeticion. Se uso el nuevo F14 del mismo',
        'banco como denominador. El JSON conserva media, minimo, maximo observado, P95 y desviacion descriptiva.', '',
        'No hay aun un costo total de pipeline para los modelos reajustados: DNN, detector y total se registran como null,',
        'no como cero. No se habilita seleccion por costo, se congela el protocolo o se entrena en esta consolidacion.', '',
        'La Flash 107952 B y RAM estatica con reservas 71228 B pertenecen al banco de diez kernels. Los buffers se mantienen;',
        'no son mediciones de ahorro de memoria de cada despliegue final.', '',
        '## Evidencia y limites', '',
        'Suite real: 126 BASE, 1260 RUN y diez rechazos. Esquina: doce BASE, 120 RUN y diez rechazos, sin mezclar tiempos.',
        'Se verificaron los dos sellos fisicos y sus 18 recursos cada uno. Este escritor no repite el ensayo ni la auditoria',
        'integral de 9757 recursos ya realizada por los auditores fisicos. Los hashes de informes, trazas y referencias quedan enlazados.', ''])
    lines.extend('- ' + value for value in result['limitations'])
    lines.extend(['', 'Siguiente trabajo: fijar restricciones predictivas y del detector, regla de seleccion, grids y presupuesto',
                  'con un piloto exclusivamente de desarrollo, antes del entrenamiento multisemilla V2.', ''])
    with human.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write('\n'.join(lines))
    entries = dict(result['source_evidence_sha256'])
    for path in (Path(__file__), target, human):
        entries[path.relative_to(ROOT).as_posix()] = sha(path)
    with manifest.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(entries, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    with seal.open('x', encoding='ascii') as stream:
        stream.write(sha(manifest) + '  ' + manifest.name + '\n')
    require(all(sha(bounded(name)) == digest for name, digest in entries.items()), 'Cierre diferente al sellar')
    return len(entries)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    result = consolidate()
    if args.write:
        count = write_closure(result)
        print(json.dumps(dict(status=result['status'], closure_resources_verified=count,
                             selector_total_cost_ready=False, training_enabled=False)))
    else:
        print(json.dumps(result, ensure_ascii=True, indent=2))
