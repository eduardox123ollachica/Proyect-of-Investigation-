"""Audita aritmetica y tiempos FULL SR desde evidencias existentes, sin TF/COM.

Las FC se recalculan enteramente desde pesos TFLite. Softmax se contrasta
con el cache sellado solo despues de comprobar identidad de la entrada.
No se capturan ni se presuponen activaciones internas de STM32.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT/'despliegue_stm32/costo_total_single_rounding_v2_180mhz_2026-10-26'
RUN = OUT/'Validacion_PC/resultados/20261026T144540_396879Z'
REPORT = OUT/'AUDITORIA_NUMERICA_CAMPANA_SR.json'
REPLAYS = OUT/'REPLAY_FC_CAMPANA_SR.jsonl'
MANIFEST = OUT/'MANIFEST_NUMERICA_CAMPANA_SR_SHA256.json'
ELF_SHA = 'c0c3853d2b0e90bb20fe00be13bcf107d186088406be131b729a605ded44ecc7'
STAGES = ('extract_cycles', 'input_cycles', 'nn_cycles', 'detector_cycles', 'residual_cycles')


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
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def close(a, b):
    return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)


def canonical_sha(value):
    data = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('ascii')
    return hashlib.sha256(data).hexdigest()


def ulps(a, b):
    def ordered(value):
        n = int(value, 16)
        return 0x80000000-n if n&0x80000000 else 0x80000000+n
    return abs(ordered(a)-ordered(b))


def main():
    started = time.perf_counter()
    require(not any(p.exists() for p in (REPORT, REPLAYS, MANIFEST, MANIFEST.with_suffix('.sha256'))),
            'No sobrescribir auditoria')
    prior_path = Path(__file__).with_name('auditar_numerica_gate_total_v2.py')
    saved_path = Path(__file__).with_name('auditar_numerica_gate_total_sr_v2.py')
    dense_path = Path(__file__).with_name('reproducir_fc_entera_total_v2.py')
    prior = load(prior_path, 'full_sr_independent_float')
    saved = load(saved_path, 'full_sr_saved_uart')
    sys.path.insert(0, str(Path(__file__).parent))
    dense = load(dense_path, 'full_sr_integer_dense')
    client = load(RUN/'validar_total_ejecutado.py', 'full_sr_executed_client_parsers')
    gate_manifest = prior.manifest(RUN/'MANIFEST_EVIDENCIAS_SHA256.json', RUN)
    resources = prior.manifest(OUT/'recursos/MANIFEST_RECURSOS_SHA256.json', OUT/'recursos')
    report, plan = read(RUN/'RESULTADO_TOTAL.json'), read(RUN/'PLAN_EJECUTADO.json')
    cat, bank = read(RUN/'CATALOGO_TOTAL.json'), read(RUN/'BANCO_TOTAL.json')
    require(report['status'] == 'FULL_ARM_MEASUREMENTS_READY_FOR_AUDIT'
            and report['mode'] == plan['mode'] == 'full'
            and (report['completed_primary_RUN'], report['completed_PROFILE'], report['completed_BASE'],
                 report['negative_rejections']) == (2688, 896, 42, 17), 'Campana incompleta')
    require(report['firmware_elf_sha256'] == sha(RUN/'firmware_declarado.elf') == ELF_SHA, 'ELF diferente')
    require(report['client_sha256'] == sha(RUN/'validar_total_ejecutado.py') == sha(OUT/'Validacion_PC/validar_total_sr.py'),
            'Cliente diferente')
    for name in ('CATALOGO_TOTAL.json', 'BANCO_TOTAL.json', 'CATALOGO_CANONICO.json'):
        require(sha(RUN/name) == sha(OUT/'recursos'/name), 'Referencia diferente: '+name)
    require(plan['schedule'] == [[gid, list(ids)] for gid, ids in client.schedule(bank, True)],
            'Calendario distinto de bancos autorizados')
    events = [json.loads(line) for line in (RUN/'MEDICIONES.jsonl').read_text(encoding='utf-8').splitlines()]
    trace = [json.loads(line) for line in (RUN/'TRAZA_UART.jsonl').read_text(encoding='utf-8').splitlines()]
    require([e['seq'] for e in events] == list(range(1, len(events)+1)), 'Secuencia de mediciones')
    link = saved.SavedLink(trace)
    pubs = [e for e in events if e['type'] in ('RUN', 'PROFILE')]
    results = [e['line'] for e in trace if e['direction'] == 'RX' and e['line'].startswith('CWRUTOTAL_RESULT,')]
    first = report['firmware_info']['last_run_id']
    require(len(pubs) == len(results) == 3584
            and [e['run_id'] for e in pubs] == list(range(first+1, first+3585)), 'Publicaciones incompletas')
    for info in (report['firmware_info'], report['firmware_info_final']):
        require(info['hclk_hz'] == 180000000 and not int(info['fpscr'], 16)&((3<<22)|(1<<24))
                and info['paired_read_cycles_min'] == 5 and info['calibration_samples'] == 64
                and not info['overhead_subtracted'], 'Reloj, FPSCR o calibracion distintos')
    bases = {e['global_id']:e for e in events if e['type'] == 'BASE'}
    selections = {(e['global_id'], e['rep_id']):e for e in events if e['type'] == 'SELECT'}
    groups = defaultdict(list)
    for index, event in enumerate(pubs):
        groups[(event['global_id'], event['rep_id'])].append((index, event))
    require(len(bases) == 42 and len(selections) == len(groups) == 896, 'Bancos incompletos')
    expected_pairs = [(gid, rid) for gid, ids in plan['schedule'] for rid in ids]
    require(list(groups) == expected_pairs and set(selections) == set(expected_pairs), 'Orden o pares diferentes')
    row_by_id = {r['global_id']:r for r in bank['windows']}
    base_differences = []
    feature_error = [dict(index=i, differing_windows=0, maximum_absolute_error=0., maximum_ULP=0) for i in range(14)]
    for gid, base in bases.items():
        row = row_by_id[gid]
        require(all(base[k] == v for k, v in client.baseline(link, row).items()), 'BASE no reproduce UART')
        require(row['arm_features14_bits'] is None or base['features_bits'] == row['arm_features14_bits'],
                'BASE distinta del historico ARM cuando existe')
        a, b = prior.floats(base['features_bits']), prior.floats(row['features_pc_cache14_bits'])
        require(np.allclose(a, b, rtol=.001, atol=.00001) and np.array_equal(a[[4, 11]], b[[4, 11]]),
                'BASE no cumple tolerancias congeladas')
        differences = []
        for i, (ab, bb) in enumerate(zip(base['features_bits'], row['features_pc_cache14_bits'])):
            error = abs(float(a[i])-float(b[i]))
            feature_error[i]['maximum_absolute_error'] = max(error, feature_error[i]['maximum_absolute_error'])
            feature_error[i]['maximum_ULP'] = max(ulps(ab, bb), feature_error[i]['maximum_ULP'])
            if ab != bb:
                feature_error[i]['differing_windows'] += 1
                differences.append(dict(index=i, ARM_bits=ab, PC_bits=bb, absolute_error=error, ULP=ulps(ab, bb)))
        base_differences.append(dict(global_id=gid, differences=differences))
    models = {}
    for rep in cat['representatives']:
        model = (OUT/'recursos'/rep['model_path']).read_bytes()
        require(hashlib.sha256(model).hexdigest() == rep['model_sha256'] and client.fnv(model) == rep['model_fnv'],
                'Modelo no autentico')
        params = client.params_bytes(rep)
        require(params.hex() == rep['params_bytes_hex'] and client.fnv(params) == rep['params_fnv'], 'Parametros diferentes')
        tensors, ops = dense.parse_model(model)
        require(len(ops) == 3 and [len(tensors[o['output']]['shape']) for o in ops] == [2, 2, 2]
                and [tensors[o['output']]['shape'][-1] for o in ops] == [20, 10, 3], 'Topologia distinta')
        models[rep['id']] = (tensors, ops)
    cache_comparison_counts = Counter()
    pair_rows, all_fc_replays = [], []
    max_abs_acc = 0
    for gid, rid in expected_pairs:
        rep, row = cat['representatives'][rid], row_by_id[gid]
        base, selection = bases[gid], selections[(gid, rid)]
        group = groups[(gid, rid)]
        require([e['type'] for _, e in group] == ['RUN', 'RUN', 'RUN', 'PROFILE']
                and [e['repetition'] for _, e in group] == [1, 2, 3, 1], 'Orden de repeticiones/perfil distinto')
        require(selection['arena_used'] == 1188 and selection['cycles'] > 0 and selection['excluded_from_primary_cost'],
                'Asignacion/inicializacion distinta')
        selected = [base['features_bits'][i] if i in rep['feature_indices'] else '00000000' for i in range(14)]
        replay = prior.independent_replay(selected, rep)
        ref = bank['reference_by_rep_gid'][str(rid)][str(gid)]
        require(replay['quantized'] == ref['int8_inputs'], 'Q distinta del cache: exige oracle Softmax nuevo')
        tensors, ops = models[rid]
        x, layers = replay['quantized'], []
        for op in ops:
            layer = dense.dense(x, op, tensors, 'single')
            layers.append(layer)
            max_abs_acc = max(max_abs_acc, *(abs(v) for v in layer['accumulators']))
            x = layer['output']
        fc_summary = dict(global_id=gid, rep_id=rid, model_sha256=rep['model_sha256'], input_INT8=replay['quantized'],
            layer_outputs=[l['output'] for l in layers], logits=x, all_layer_details_sha256=canonical_sha(layers),
            cached_PC_output_INT8=ref['int8_outputs'])
        all_fc_replays.append(fc_summary)
        output = ref['int8_outputs']
        cls = int(np.argmax(output))
        comparison = dict(feature_bit_differences=sum(a != b for a, b in zip(base['features_bits'], ref['raw_features14_bits'])),
            quantized_value_differences=0, output_value_differences=0, class_differs=False,
            detector_score_bit_differs=replay['score_bits'] != ref['detector_score_bits'],
            detector_rejection_differs=replay['rejected'] != ref['outside_normality'])
        require(not comparison['detector_rejection_differs'], 'Cambio de decision detector contra cache')
        cache_comparison_counts[json.dumps(comparison, sort_keys=True)] += 1
        for publication_index, e in group:
            mode = int(e['type'] == 'PROFILE')
            parsed = client.parse_result(results[publication_index], rep, row, e['run_id']-1, mode, selection['arena_used'])
            require(all(e[k] == v for k, v in parsed.items()), 'RESULT no reproduce UART')
            pages = client.get_pages(link, parsed, rep)
            require(e['pages'] == pages and pages['features_bits'] == selected
                    and all(pages[k] == v for k, v in replay.items()), 'Intermedios/replay float32 diferentes')
            require(e['output_int8'] == output and e['predicted_class'] == cls
                    and e['score_bits'] == replay['score_bits'] and e['rejected'] == replay['rejected'],
                    'Salida/clase/detector distintos del oracle para identica Q')
            require(e['comparison']['cached_PC'] == comparison and e['status'] == 'PASS'
                    and close(e['milliseconds'], e['total_cycles']/180000), 'Comparacion/tiempo almacenado incorrecto')
            require(sum(e[k] for k in STAGES) == (e['total_cycles'] if mode else 0), 'Perfil no cierra')
        pair_rows.append(dict(global_id=gid, rep_id=rid, quantized_input_exact_to_cache=True,
            output_INT8=output, predicted_class=cls, rejected=replay['rejected'], score_bits=replay['score_bits'],
            threshold_bits=rep['threshold_bits'], saturation_low=replay['saturated_low'], saturation_high=replay['saturated_high'],
            cache_comparison=comparison, FC_logits=x))
    representatives = []
    for rep in cat['representatives']:
        rid = rep['id']
        main = [e for e in pubs if e['rep_id'] == rid and e['type'] == 'RUN']
        profile = [e for e in pubs if e['rep_id'] == rid and e['type'] == 'PROFILE']
        sels = [e for (gid, selected_rid), e in selections.items() if selected_rid == rid]
        require(len(main) == 84 and len(profile) == len(sels) == 28
                and {e['global_id'] for e in main} == set(rep['scope']['benchmark_global_window_ids']), '84/28 banco incompleto')
        cycles = np.asarray([e['total_cycles'] for e in main], dtype=np.float64)
        stats = dict(count=84, min_cycles=int(cycles.min()), max_cycles=int(cycles.max()), mean_cycles=float(cycles.mean()),
            mean_ms=float(cycles.mean()/180000), median_ms=float(np.median(cycles)/180000),
            p95_ms=float(np.percentile(cycles, 95)/180000), sample_std_ms=float(cycles.std(ddof=1)/180000))
        declared = next(s for s in report['timing_statistics'] if s['rep_id'] == rid)
        require(all(close(value, declared[key]) for key, value in stats.items()), 'Estadistica principal distinta')
        require(all(not declared[k] for k in ('PROFILE_included', 'BASE_included', 'initialization_included',
                                             'UART_included', 'WCET_claimed', 'selector_enabled')), 'Alcance principal alterado')
        profile_stats = {key:dict(mean_cycles=float(np.mean([e[key] for e in profile])),
            min_cycles=min(e[key] for e in profile), max_cycles=max(e[key] for e in profile)) for key in ('total_cycles', *STAGES)}
        representatives.append(dict(rep_id=rid, representative_id=rep['representative_id'], candidate=rep['candidate'],
            outer_load_hp=rep['outer_load_hp'], primary_statistics=stats, unique_windows=28, primary_RUN=84, PROFILE=28,
            arena_used_bytes=sorted({e['arena_used'] for e in sels}), arena_reserved_bytes=16384,
            initializations_or_selects=28, reused_selects=sum(e['reused'] for e in sels),
            profile_statistics_excluded_from_primary=profile_stats, model_sha256=rep['model_sha256'], params_fnv=rep['params_fnv']))
    require(len(representatives) == 32 and not report['selector_enabled'] and not report['total_cost_validated'], 'Alcance excedido')
    regression = next(p for p in pair_rows if p['global_id'] == 232 and p['rep_id'] == 22)
    require(regression['output_INT8'] == [41, -100, -69] and regression['FC_logits'] == [33, -29, -3], 'Regresion no permanece corregida')
    with REPLAYS.open('x', encoding='utf-8', newline='\n') as stream:
        for row in all_fc_replays:
            stream.write(json.dumps(row, separators=(',', ':'), allow_nan=False)+'\n')
    result = dict(status='PASS_SR_FULL_NUMERIC_AUDIT_NOT_PREDICTIVE_EVALUATION',
        date_local='2026-10-26', timezone='America/Lima', run=RUN.relative_to(ROOT).as_posix(),
        capture_manifest_sha256=sha(RUN/'MANIFEST_EVIDENCIAS_SHA256.json'), capture_resources=len(gate_manifest),
        resource_manifest_sha256=sha(OUT/'recursos/MANIFEST_RECURSOS_SHA256.json'), resource_count=len(resources),
        elf_sha256=ELF_SHA, client_sha256=report['client_sha256'], script_sha256=sha(__file__),
        helper_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in (prior_path, saved_path, dense_path,
            Path(__file__).with_name('auditar_elf_total_v2.py'))},
        data_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in (RUN/'MEDICIONES.jsonl', RUN/'TRAZA_UART.jsonl',
            RUN/'RESULTADO_TOTAL.json', RUN/'PLAN_EJECUTADO.json', RUN/'BANCO_TOTAL.json', RUN/'CATALOGO_TOTAL.json', REPLAYS)},
        primary_RUN=2688, PROFILE=896, BASE=42, model_window_pairs=896, representatives=representatives,
        UART_results_reconstructed=3584, UART_pages_reconstructed=10752,
        independent_float32_replays=896, all_intermediates_exact_across_3584_publications=True,
        same_INT8_input_output_exact_across_3584_publications=True,
        quantized_input_differences_to_original_cache=0, output_differences_to_original_cache=0,
        class_differences_to_original_cache=0, normality_rejection_differences_to_original_cache=0,
        pair_cache_comparison_distribution=[dict(comparison=json.loads(k), pairs=v) for k, v in sorted(cache_comparison_counts.items())],
        BASE_feature_error_by_slot=feature_error, BASE_feature_differences=base_differences,
        BASE_historical_ARM_references_exact=sum(row_by_id[g]['arm_features14_bits'] is not None for g in bases),
        BASE_new_ARM_windows=sum(row_by_id[g]['arm_features14_bits'] is None for g in bases),
        FC_models_parsed=32, FC_layers_computed=2688, FC_neurons_computed=29568,
        FC_replay_file=REPLAYS.relative_to(ROOT).as_posix(), maximum_absolute_FC_accumulator=max_abs_acc,
        FC_internal_activations_captured_on_board=False,
        Softmax_oracle='Original sealed PC outputs only after independently recomputing identical INT8 inputs for all896 pairs',
        Softmax_independently_replayed_here=False, regression_DE7_232=regression,
        arena_all32_models_bytes=1188, arena_reserved_bytes=16384, runtime_stack_remeasured=False,
        primary_excludes=['PROFILE', 'BASE', 'SELECT/initialization', 'UART'], overhead_subtracted=False,
        wall_seconds_recorded=report['wall_seconds'], nominal_HCLK_HAL_Hz=180000000, WCET_claimed=False,
        predictive_outer_evaluation=False, motor_or_bearing_independence_claimed=False,
        cached_PC_end_to_end_bit_exactness_claimed=False, full_Flash_identity_verified_by_programmer_report=False,
        total_cost_table_emitted_here=False, selector_enabled=False,
        new_TFLite_invocations=0, new_training_fits=0, new_physical_RUN=0, builds_executed=0, serial_opened=False,
        D_files_modified=False, original_capture_modified=False, seconds=time.perf_counter()-started,
        limitations=['The neural output oracle is the sealed PC cache for independently verified identical quantized inputs.',
            'Offline FC replays are not internal activation measurements from the board.',
            'Flash identity has no programmer-report/readback proof attached.',
            'The32 trained representative models calibrate computational costs and do not complete multiseed predictive evaluation.',
            'Observed maxima and percentiles are not WCET or independent-acquisition confidence intervals.'])
    save(REPORT, result)
    paths = {Path(__file__).resolve(), REPORT, REPLAYS, prior_path, saved_path, dense_path,
        Path(__file__).with_name('auditar_elf_total_v2.py'), RUN/'MANIFEST_EVIDENCIAS_SHA256.json',
        RUN/'MANIFEST_EVIDENCIAS_SHA256.sha256', OUT/'recursos/MANIFEST_RECURSOS_SHA256.json',
        OUT/'recursos/MANIFEST_RECURSOS_SHA256.sha256'}
    paths.update(ROOT/p for p in result['data_sha256'])
    save(MANIFEST, {p.relative_to(ROOT).as_posix():sha(p) for p in sorted(paths)})
    with MANIFEST.with_suffix('.sha256').open('x', encoding='ascii') as stream:
        stream.write(sha(MANIFEST)+'  '+MANIFEST.name+'\n')
    prior.manifest(MANIFEST, ROOT)
    print(json.dumps(dict(status=result['status'], pairs=896, publications=3584,
        representatives=32, FC_layers=2688, FC_neurons=29568, manifest_sha256=sha(MANIFEST),
        seconds=time.perf_counter()-started), ensure_ascii=True))


if __name__ == '__main__':
    main()
