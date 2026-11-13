"""Audita la puerta SR existente: aritmetica, UART y tiempos; sin TF ni COM."""
from __future__ import annotations
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
RUN = OUT/'Validacion_PC/resultados/20261026T143621_946818Z'
DIAG = ROOT/'despliegue_stm32/costo_total_representantes_v2_180mhz_2026-10-25/diagnostico_divergencia_20261026T042424Z'
REPORT = OUT/'AUDITORIA_NUMERICA_PUERTA_SR.json'
HCLK = 180000000


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


class SavedLink:
    """Solo consulta texto ya capturado; nunca importa serial ni abre puertos."""
    def __init__(self, trace):
        self.responses = {}
        pending = None
        for event in trace:
            if event['direction'] == 'TX':
                require(pending is None, 'TX sin respuesta previa')
                pending = event['command']
            else:
                require(event['direction'] == 'RX' and pending is not None, 'RX sin TX')
                self.responses.setdefault(pending, []).append(event['line'])
                pending = None
        require(pending is None, 'TX final sin respuesta')

    def ask(self, command, prefix):
        matches = [line for line in self.responses.get(command, []) if line.startswith(prefix)]
        require(len(set(matches)) == 1, 'Respuesta positiva ausente/ambigua: '+command)
        return matches[0]


def main():
    started = time.perf_counter()
    require(not REPORT.exists(), 'No sobrescribir auditoria')
    prior_path = Path(__file__).with_name('auditar_numerica_gate_total_v2.py')
    prior = load(prior_path, 'sr_gate_independent_float_replay')
    # El modulo FC importa el lector FlatBuffer vecino; no ejecuta su main.
    sys.path.insert(0, str(Path(__file__).parent))
    dense_path = Path(__file__).with_name('reproducir_fc_entera_total_v2.py')
    dense = load(dense_path, 'sr_gate_integer_dense_replay')
    client_path = RUN/'validar_total_ejecutado.py'
    client = load(client_path, 'sr_saved_gate_parsers')
    gate_manifest = prior.manifest(RUN/'MANIFEST_EVIDENCIAS_SHA256.json', RUN)
    resources = prior.manifest(OUT/'recursos/MANIFEST_RECURSOS_SHA256.json', OUT/'recursos')
    prior.manifest(DIAG/'MANIFEST_DIAGNOSTICO_ROUNDING_SHA256.json', ROOT)
    result, plan = read(RUN/'RESULTADO_TOTAL.json'), read(RUN/'PLAN_EJECUTADO.json')
    cat, bank = read(RUN/'CATALOGO_TOTAL.json'), read(RUN/'BANCO_TOTAL.json')
    require(result['status'] == 'SR_GATE_PASS_NOT_TOTAL_COST_VALIDATED'
            and result['mode'] == plan['mode'] == 'gate'
            and plan['schedule'] == [[568, [0]], [232, [22]]]
            and (result['completed_primary_RUN'], result['completed_PROFILE'], result['completed_BASE'],
                 result['negative_rejections']) == (6, 2, 2, 17), 'Puerta SR incompleta')
    require(result['firmware_elf_sha256'] == sha(RUN/'firmware_declarado.elf') ==
            'c0c3853d2b0e90bb20fe00be13bcf107d186088406be131b729a605ded44ecc7', 'ELF diferente')
    require(result['client_sha256'] == sha(client_path) == sha(OUT/'Validacion_PC/validar_total_sr.py'),
            'Cliente ejecutado diferente')
    for name in ('CATALOGO_TOTAL.json', 'BANCO_TOTAL.json', 'CATALOGO_CANONICO.json'):
        require(sha(RUN/name) == sha(OUT/'recursos'/name), 'Copia de referencia distinta: '+name)
    entries = [json.loads(line) for line in (RUN/'MEDICIONES.jsonl').read_text(encoding='utf-8').splitlines()]
    trace = [json.loads(line) for line in (RUN/'TRAZA_UART.jsonl').read_text(encoding='utf-8').splitlines()]
    require([e['seq'] for e in entries] == list(range(1, len(entries)+1)), 'Secuencia de mediciones')
    require([e['seq'] for e in trace] == list(range(1, len(trace)+1)), 'Secuencia UART')
    link = SavedLink(trace)
    publications = [e for e in entries if e['type'] in ('RUN', 'PROFILE')]
    rx_results = [e['line'] for e in trace if e['direction'] == 'RX' and e['line'].startswith('CWRUTOTAL_RESULT,')]
    require(len(publications) == len(rx_results) == 8
            and [e['run_id'] for e in publications] == list(range(1, 9)), 'Publicaciones incompletas')
    for info in (result['firmware_info'], result['firmware_info_final']):
        require(info['hclk_hz'] == HCLK and not int(info['fpscr'], 16)&((3<<22)|(1<<24))
                and info['paired_read_cycles_min'] == 5 and info['calibration_samples'] == 64
                and not info['overhead_subtracted'], 'Reloj, FPSCR o calibracion distintos')
    cases = []
    stage_keys = ('extract_cycles', 'input_cycles', 'nn_cycles', 'detector_cycles', 'residual_cycles')
    for gid, rid in ((568, 0), (232, 22)):
        rep = cat['representatives'][rid]
        row = next(r for r in bank['windows'] if r['global_id'] == gid)
        ref = bank['reference_by_rep_gid'][str(rid)][str(gid)]
        bases = [e for e in entries if e['type'] == 'BASE' and e['global_id'] == gid]
        selections = [e for e in entries if e['type'] == 'SELECT' and e['rep_id'] == rid]
        pubs = [e for e in publications if e['rep_id'] == rid]
        require(len(bases) == len(selections) == 1 and len(pubs) == 4, 'Caso incompleto')
        base, selection = bases[0], selections[0]
        require(all(base[k] == v for k, v in client.baseline(link, row).items()), 'BASE no reproduce UART')
        require(base['features_bits'] == row['arm_features14_bits'], 'BASE difiere del historico ARM')
        require(selection['arena_used'] == 1188 and not selection['reused']
                and selection['cycles'] > 0 and selection['excluded_from_primary_cost'], 'Inicializacion distinta')
        selected = [base['features_bits'][i] if i in rep['feature_indices'] else '00000000' for i in range(14)]
        replay = prior.independent_replay(selected, rep)
        require(replay['quantized'] == ref['int8_inputs'], 'Entrada ARM no exacta al cache sellado')
        model_path = OUT/'recursos'/rep['model_path']
        model = model_path.read_bytes()
        require(hashlib.sha256(model).hexdigest() == rep['model_sha256'], 'Modelo cambiado')
        tensors, ops = dense.parse_model(model)
        layers = []
        x = replay['quantized']
        for i, op in enumerate(ops):
            layer = dense.dense(x, op, tensors, 'single')
            layer['layer'] = i
            layers.append(layer)
            x = layer['output']
        cached_output = ref['int8_outputs']
        expected_class = int(np.argmax(cached_output))
        for e in pubs:
            mode = 1 if e['type'] == 'PROFILE' else 0
            parsed = client.parse_result(rx_results[e['run_id']-1], rep, row, e['run_id']-1, mode, selection['arena_used'])
            require(all(e[k] == v for k, v in parsed.items()), 'RESULT no reproduce UART')
            pages = client.get_pages(link, parsed, rep)
            require(e['pages'] == pages and pages['features_bits'] == selected
                    and all(pages[k] == v for k, v in replay.items()), 'Paginas/replay float32 distintos')
            require(e['output_int8'] == cached_output and e['predicted_class'] == expected_class
                    and e['score_bits'] == replay['score_bits'] and e['rejected'] == replay['rejected'],
                    'Salida INT8/clase/detector distintos del oracle sellado')
            require(e['status'] == 'PASS' and close(e['milliseconds'], e['total_cycles']/180000), 'Estado/tiempo distintos')
            comparison = dict(feature_bit_differences=sum(a != b for a, b in zip(base['features_bits'], ref['raw_features14_bits'])),
                quantized_value_differences=0, output_value_differences=0, class_differs=False,
                detector_score_bit_differs=e['score_bits'] != ref['detector_score_bits'],
                detector_rejection_differs=e['rejected'] != ref['outside_normality'])
            require(e['comparison']['cached_PC'] == comparison, 'Comparacion al cache incorrecta')
        runs = [e for e in pubs if e['type'] == 'RUN']
        profile = next(e for e in pubs if e['type'] == 'PROFILE')
        require([e['repetition'] for e in runs] == [1, 2, 3]
                and all(e['mode'] == 0 and all(e[k] == 0 for k in stage_keys) for e in runs)
                and profile['mode'] == 1 and sum(profile[k] for k in stage_keys) == profile['total_cycles'],
                'Instrumentacion principal/perfil incorrecta')
        cycles = np.asarray([e['total_cycles'] for e in runs], dtype=np.float64)
        stats = dict(count=3, min_cycles=int(cycles.min()), max_cycles=int(cycles.max()), mean_cycles=float(cycles.mean()),
            mean_ms=float(cycles.mean()/180000), median_ms=float(np.median(cycles)/180000),
            p95_ms=float(np.percentile(cycles, 95)/180000), sample_std_ms=float(cycles.std(ddof=1)/180000))
        declared = next(r for r in result['timing_statistics'] if r['rep_id'] == rid)
        require(all(close(value, declared[key]) for key, value in stats.items()), 'Estadistica no reproduce registros')
        differences = [dict(index=i, ARM_bits=a, PC_cache_bits=b,
            absolute_error=float(abs(float(prior.floats([a])[0])-float(prior.floats([b])[0]))))
            for i, (a, b) in enumerate(zip(base['features_bits'], row['features_pc_cache14_bits'])) if a != b]
        cases.append(dict(rep_id=rid, representative_id=rep['representative_id'], global_window_id=gid,
            primary_statistics=stats, primary_min_ms=float(cycles.min()/180000), primary_max_ms=float(cycles.max()/180000),
            initialization_cycles=selection['cycles'], initialization_excluded=True, arena_used_bytes=selection['arena_used'],
            model_sha256=rep['model_sha256'], input_INT8=replay['quantized'], output_INT8=cached_output,
            predicted_class=expected_class, score_bits=replay['score_bits'], outside_normality=replay['rejected'],
            threshold_bits=rep['threshold_bits'], same_INT8_input_output_exact=True,
            UART_publications_reconstructed=4, float32_intermediates_exact=True,
            feature_PC_cache_differences=differences, cache_comparison=comparison,
            FC_single_integer_layers=layers, FC_single_logits=x,
            Softmax_oracle='Original sealed PC cache; recomputed ARM INT8 input is exactly the same',
            Softmax_independently_replayed_here=False,
            profile=dict(total_cycles=profile['total_cycles'], stages={k: profile[k] for k in stage_keys}, excluded=True)))
    regression = cases[1]
    causal = read(DIAG/'REPLAY_FC_ENTERAS.json')
    soft = read(DIAG/'COMPARACION_SOFTMAX_AISLADO.json')
    failed = read(DIAG/'CASO_FALLO.json')
    require(regression['input_INT8'] == causal['input_INT8'] == failed['input_INT8']
            and regression['model_sha256'] == causal['model_sha256'] == soft['original_model_sha256']
            and regression['FC_single_integer_layers'] == causal['rounding_modes']['single']['layers']
            and regression['FC_single_logits'] == soft['results']['SINGLE']['logits']
            and regression['output_INT8'] == soft['results']['SINGLE']['output_INT8'] == failed['PC_cache_output_INT8']
            and regression['output_INT8'] != failed['ARM_output_INT8'], 'Regresion no corresponde al diagnostico')
    require(not result['selector_enabled'] and not result['total_cost_validated'], 'Alcance de puerta excedido')
    report = dict(status='PASS_SR_NUMERIC_TIMING_GATE_AUDIT_NOT_TOTAL_COST_VALIDATED',
        run=str(RUN), date_local='2026-10-26', timezone='America/Lima',
        gate_manifest_resources=len(gate_manifest), resource_count=len(resources),
        gate_manifest_sha256=sha(RUN/'MANIFEST_EVIDENCIAS_SHA256.json'),
        elf_sha256=result['firmware_elf_sha256'], client_sha256=sha(client_path), script_sha256=sha(__file__),
        helper_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in
            (prior_path, dense_path, Path(__file__).with_name('auditar_elf_total_v2.py'))},
        data_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in
            (RUN/'MEDICIONES.jsonl', RUN/'TRAZA_UART.jsonl', RUN/'RESULTADO_TOTAL.json',
             DIAG/'REPLAY_FC_ENTERAS.json', DIAG/'COMPARACION_SOFTMAX_AISLADO.json', DIAG/'CASO_FALLO.json')},
        primary_RUN=6, PROFILE=2, BASE=2, UART_results_reconstructed=8, cases=cases,
        regression_case_previous_failure_resolved_observed=True,
        previous_regression_ARM_output=failed['ARM_output_INT8'], current_regression_ARM_output=regression['output_INT8'],
        diagnostic_Softmax_reused_only_for_identical_regression_model_and_logits=True,
        FC_integer_layers_computed=6, FC_neurons_computed=66,
        FC_internal_activations_captured_on_board=False,
        wall_seconds_recorded=result['wall_seconds'], overhead_subtracted=False,
        total_cost_validated=False, selector_enabled=False, WCET_claimed=False,
        new_TFLite_invocations=0, new_training_fits=0, new_physical_runs=0, serial_opened=False,
        limitations=['Only two representatives and two windows are covered; full banks and other30 models pending.',
            'FC activations are offline integer replays, not internal activation captures from STM32.',
            'Softmax is compared with sealed PC outputs for identical input; prior isolated diagnostic additionally checks DE7/232 only.',
            'No programmer report supplied: declared/copied ELF is not cryptographic proof of all board Flash.',
            'Three principal times per representative do not replace the planned84 samples per representative.'],
        seconds=time.perf_counter()-started)
    save(REPORT, report)
    paths = {Path(__file__).resolve(), REPORT, prior_path, dense_path,
        Path(__file__).with_name('auditar_elf_total_v2.py'), RUN/'MANIFEST_EVIDENCIAS_SHA256.json',
        RUN/'MANIFEST_EVIDENCIAS_SHA256.sha256', DIAG/'MANIFEST_DIAGNOSTICO_ROUNDING_SHA256.json',
        DIAG/'MANIFEST_DIAGNOSTICO_ROUNDING_SHA256.sha256'}
    paths.update(ROOT/p for p in report['data_sha256'])
    seal = OUT/'MANIFEST_NUMERICA_PUERTA_SR_SHA256.json'
    save(seal, {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(paths)})
    with seal.with_suffix('.sha256').open('x', encoding='ascii') as stream:
        stream.write(sha(seal)+'  '+seal.name+'\n')
    prior.manifest(seal, ROOT)
    print(json.dumps(dict(status=report['status'], cases=[dict(rep_id=c['rep_id'], output_INT8=c['output_INT8'],
        mean_ms=c['primary_statistics']['mean_ms']) for c in cases], manifest_sha256=sha(seal)), ensure_ascii=True))


if __name__ == '__main__':
    main()
