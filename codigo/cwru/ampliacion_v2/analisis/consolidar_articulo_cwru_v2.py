"""Tablas y figuras CWRU desde cierres auditados; sin entrenar o inferir.

--generate crea solamente productos nuevos de redacción. --check compara
las cifras exportadas con los mismos cierres y verifica su manifiesto.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT/'articulo/ampliacion_cwru/metodologia'
DEST = ROOT/'articulo/ampliacion_cwru/redaccion/consolidacion_2026-10-28'
EXTERIOR = BASE/'cierre_evaluacion_exterior_v2_2026-10-27'
FISICA = BASE/'validacion_fisica_canonicos_completa_2026-10-28'
SELECCION = BASE/'seleccion_congelada_v2_2026-10-27'
PROTOCOLO = BASE/'protocolo_v2_2026-10-25'
NAMES = {'baseline_F14': 'F14', 'performance_only': 'Selección por rendimiento',
    'validated_total_cost': 'Selección por costo total', 'RF': 'RF', 'SVM': 'SVM'}
ORDER = list(NAMES)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def verify_manifest(path, expected=None):
    path = Path(path)
    digest = sha(path)
    require(path.with_suffix('.sha256').read_text(encoding='ascii').split() == [digest, path.name], 'Sello inválido')
    require(expected is None or digest == expected, 'Sello diferente del cierre aprobado')
    entries = read(path)
    for relative, value in entries.items():
        target = (ROOT/relative).resolve()
        require(target.is_relative_to(ROOT.resolve()) and target.is_file() and sha(target) == value,
            'Fuente cerrada modificada: '+relative)
    return digest


def build():
    manifests = [EXTERIOR/'MANIFEST_CIERRE_EVALUACION_EXTERIOR_SHA256.json',
        FISICA/'MANIFEST_CIERRE_SHA256.json', SELECCION/'MANIFEST_SELECCION_CONGELADA_SHA256.json',
        PROTOCOLO/'MANIFEST_PROTOCOLO_SHA256.json']
    for p in manifests:
        verify_manifest(p, '1edee37afa1c8cc42a2cb6d4307992765b6169029a459fbf5952dae24d99843e' if p.parent == FISICA else None)
    paths = [EXTERIOR/'METRICAS_Y_LIMITES.json', EXTERIOR/'CIERRE_EVALUACION_EXTERIOR.json',
        FISICA/'RESUMEN_VALIDACION_FISICA.json', FISICA/'CIERRE_VALIDACION_FISICA.json',
        SELECCION/'DECISIONES_CONGELADAS.json', PROTOCOLO/'protocolo_reglas.json',
        ROOT/'despliegue_stm32/canonicos_42_180mhz_2026-10-27/AUDITORIA_ELF_CANON42.json']
    metrics, outer, physical, physical_close, choices, rules, elf = map(read, paths)
    require(outer['loads_completed'] == 4 and outer['heldout_windows'] == 1380, 'Evaluación exterior incompleta')
    require(physical_close['physical_audit_status'] == 'PASS_CANON42_STAGED_FULL_AUDITED', 'Auditoría física pendiente')
    require(outer['selection_signature'] == choices['selection_signature'], 'Selecciones no coincidentes')
    predictive = []
    for name in ORDER:
        precision = 'native' if name in ('RF', 'SVM') else 'INT8'
        group = next(g for g in metrics['groups'] if g['procedure'] == name and g['precision'] == precision)
        predictive.append(dict(procedure=name, display=NAMES[name], precision=precision, logical_cells=group['cells'],
            seed_count=len(group['seeds']), macro_f1_percent=100*group['mean_macro_f1'],
            accuracy_percent=100*group['mean_accuracy'],
            per_load=[dict(load_hp=v['outer_load_hp'], seed_count=v['seed_count'],
                macro_f1_percent=100*v['macro_f1_across_seeds']['mean'],
                macro_f1_seed_sd_pp=(100*v['macro_f1_across_seeds']['sample_standard_deviation']
                    if v['macro_f1_across_seeds']['sample_standard_deviation'] is not None else None),
                accuracy_percent=100*v['accuracy_across_seeds']['mean']) for v in group['per_load']]))
    selection_rows, method_loads, comparisons = [], [], []
    for row in choices['decisions']:
        fold = row['outer_load_hp']
        first, second = physical['latencies'][2*fold:2*fold+2]
        perf, cost = row['performance']['selected_candidate'], row['cost']['selected_candidate']
        require(first['candidate'] == 'F14' and first['outer_load_hp'] == fold, 'Referencia física diferente')
        require({perf, cost} <= {'F14', second['candidate']}, 'Instancia física seleccionada ausente')
        get_time = lambda mask: first['mean_ms'] if mask == 'F14' else second['mean_ms']
        post = physical['POST_seed42'][2*fold]
        method_loads.append(dict(outer_load_hp=fold, MAT=7, windows=post['windows'],
            N_IR_OR_windows=post['detector_class_counts'], development_loads_hp=[n for n in range(4) if n != fold]))
        selection_rows.append(dict(outer_load_hp=fold, performance_mask=perf, cost_mask=cost,
            epochs_F14=row['baseline']['final_epochs'],
            epochs_performance=row['performance']['choice']['final_epochs'],
            epochs_cost=row['cost']['choice']['final_epochs'], cost_goal_met=(fold not in choices['fallback_scenarios']),
            fallback=(fold in choices['fallback_scenarios']),
            selected_satisfies_predictive_margin=row['cost']['selected_satisfies_predictive_margin']))
        comparisons.append(dict(outer_load_hp=fold, F14_mean_ms_seed42=get_time('F14'),
            performance_mean_ms_seed42=get_time(perf), cost_mean_ms_seed42=get_time(cost),
            cost_reduction_vs_F14_percent=100*(1-get_time(cost)/get_time('F14')),
            cost_reduction_vs_performance_percent=100*(1-get_time(cost)/get_time(perf))))
    paired = []
    for p in metrics['paired']:
        require(not p['confidence_interval_computed'] and not p['p_value_computed']
            and not p['noninferiority_claimed'], 'Inferencia estadística no esperada')
        stats = p['four_load_mean_macro_f1_differences_across_seeds']
        paired.append(dict(left=p['left'], right=p['right'], pairs=p['pairs'],
            mean_macro_f1_difference_pp=p['mean_macro_f1_difference_pp'],
            mean_accuracy_difference_pp=p['mean_accuracy_difference_pp'],
            seed_averaged_over_four_loads_sd_pp=stats['sample_standard_deviation'],
            seed_averaged_over_four_loads_min_pp=stats['minimum'],
            seed_averaged_over_four_loads_max_pp=stats['maximum'], confidence_interval_computed=False))
    detector = [dict(outer_load_hp=r['outer_load_hp'], specificity_N_percent=100*r['normal_spec'],
        recall_IR_percent=100*r['ir_recall'], recall_OR_percent=100*r['or_recall'],
        binary_balanced_accuracy_percent=100*r['balanced_accuracy'])
        for r in metrics['detector_cells'] if r['procedure'] == 'baseline_F14']
    require(len(detector) == 4 and all(r['normal_spec'] == detector[r['outer_load_hp']]['specificity_N_percent']/100
        for r in metrics['detector_cells']), 'Detector difiere entre procedimientos; revisar agrupación')
    quant = {k: v for k, v in metrics['quantization'].items() if not isinstance(v, (list, dict))}
    require(quant['class_changes'] == 482 and quant['unique_model_window_pairs'] == 27600, 'Cuantización distinta')
    require(sum(row['windows'] for row in method_loads) == 1380 and sum(row['MAT'] for row in method_loads) == 28,
        'Conteos de corpus distintos')
    require([r['cost_mask'] for r in selection_rows] == ['T6', 'FE7', 'T6', 'F14'], 'Selección congelada distinta')
    return dict(status='CWRU_ARTICLE_CONSOLIDATION_READY_FOR_STRUCTURAL_REWRITE',
        date_local='2026-10-28', timezone='America/Lima',
        source_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in paths+manifests},
        scope=copy.deepcopy(rules['scope']), method_loads=method_loads,
        candidate_indices=rules['candidates'], selection=selection_rows,
        predictive_equal_load_seed_means=predictive, paired_descriptive=paired,
        physical_seed42_latencies=physical['latencies'], physical_procedure_comparison=comparisons,
        physical_coverage=physical['coverage'], physical_RAW_PC_ARM=physical['RAW_PC_ARM'],
        detector_per_load=detector, detector_scope=metrics['detector_scope'], quantization=quant,
        resource_scope=dict(firmware='eight-model verification harness', Flash_bytes=elf['memory']['flash_load_bytes'],
            static_RAM_including_reservations_bytes=elf['memory']['ram_static_including_linker_reservations_bytes'],
            arena_capacity_bytes=elf['memory']['arena_capacity_bytes'],
            arena_observed_bytes=physical['arena_used_bytes_by_model']['0'][0],
            stack_reserved_bytes=elf['memory']['stack_reservation_bytes'], heap_reserved_bytes=elf['memory']['heap_reservation_bytes'],
            runtime_stack_maximum_measured=False, RF_SVM_STM32_costs_measured=False),
        interpretation=dict(primary_CWRU=True, Arciniegas_antecedent=True, algorithmic_priority_claimed=False,
            no_universal_accuracy_or_time_improvement=True, descriptive_seed_variability_not_physical_independence=True,
            physical_seed42_not_the_80_model_population=True, figures_are_descriptive=True),
        original_manuscript_sha256=sha(ROOT/'articulo/manuscrito/articulo_ieee_borrador.tex'),
        new_training_fits=0, new_classifier_invocations=0, serial_port_opened=False, manuscript_modified=False)


def table(header, rows):
    return '\n'.join(['| '+' | '.join(header)+' |', '| '+' | '.join(['---']*len(header))+' |',
        *['| '+' | '.join(str(x) for x in row)+' |' for row in rows]])


def render_tables(d):
    s = ['# Tablas consolidadas para la ampliación CWRU', '',
        'Catálogo para elegir y distribuir entre texto principal y suplemento. '
        'Las tablas predictivas usan diez semillas salvo SVM; los costos físicos corresponden a semilla 42.', '',
        '## T1. Corpus y evaluación exterior por carga', '',
        table(['Carga exterior (HP)', 'MAT', 'Ventanas N / IR / OR', 'Total ventanas', 'Cargas de desarrollo (HP)'],
            [[r['outer_load_hp'], r['MAT'], ' / '.join(map(str, r['N_IR_OR_windows'])), r['windows'],
                ', '.join(map(str, r['development_loads_hp']))] for r in d['method_loads']]), '',
        '28 MAT, DE/FE como posiciones de sensor, 12 kHz, N=4096 y salto=2048 (50 %). '
        'Cuatro pliegues exteriores y tres interiores por pliegue. Cada MAT conserva juntos canales y ventanas. '
        'Evaluación retrospectiva, sin independencia física entre motores demostrada.', '',
        '## T2. Selección congelada y épocas finales', '',
        table(['Carga (HP)', 'Rendimiento', 'Costo', 'Épocas F14 / rendimiento / costo', 'Objetivo temporal interno', 'Margen predictivo interno del elegido por costo'],
            [[r['outer_load_hp'], r['performance_mask'], r['cost_mask']+(' (respaldo)' if r['fallback'] else ''),
                f"{r['epochs_F14']} / {r['epochs_performance']} / {r['epochs_cost']}",
                'Cumplido' if r['cost_goal_met'] else 'No cumplido',
                'Cumplido' if r['selected_satisfies_predictive_margin'] else 'No cumplido'] for r in d['selection']]), '',
        'Objetivo temporal interno U(candidata) ≤ 0.8 L(F14); L/U son extremos observados. '
        'El margen predictivo es una tolerancia de ingeniería, no una prueba de no inferioridad. '
        'El respaldo de 3 HP no cumplió el margen interno ni el objetivo temporal.', '',
        '## T3. Desempeño exterior: igual peso de cuatro cargas', '',
        table(['Procedimiento', 'Precisión', 'Macro-F1 (%)', 'Exactitud (%)', 'Celdas lógicas carga–semilla'],
            [[r['display'], r['precision'], f"{r['macro_f1_percent']:.4f}", f"{r['accuracy_percent']:.4f}", r['logical_cells']]
                for r in d['predictive_equal_load_seed_means']]), '',
        'DNN y RF: diez semillas por carga. SVM: un ajuste determinista por carga. '
        'Las equivalencias entre procedimientos reutilizan modelos; las celdas no son adquisiciones independientes. '
        'RF/SVM no cuentan con latencia ni memoria medidas en STM32.', '',
        '## T4. Macro-F1 exterior por carga: media ± desviación entre semillas (%)', '',
        table(['Procedimiento', '0 HP', '1 HP', '2 HP', '3 HP'],
            [[r['display'], *[f"{v['macro_f1_percent']:.4f}" +
                (f" ± {v['macro_f1_seed_sd_pp']:.4f}" if v['seed_count'] > 1 else '') for v in r['per_load']]]
                for r in d['predictive_equal_load_seed_means']]), '',
        'La desviación caracteriza la variación entre entrenamientos con semillas distintas en este corpus. '
        'No es un intervalo de confianza entre motores.', '',
        '## T5. Costo físico de ocho instancias canónicas de semilla 42', '',
        table(['Carga (HP)', 'Máscara', 'Media (ms)', 'Mediana (ms)', 'p95 (ms)', 'Mín.–máx. observado (ms)'],
            [[r['outer_load_hp'], r['candidate'], f"{r['mean_ms']:.4f}", f"{r['median_ms']:.4f}", f"{r['p95_ms']:.4f}",
                f"{r['min_ms']:.4f}–{r['max_ms']:.4f}"] for r in d['physical_seed42_latencies']]), '',
        '126 RUN por modelo = 42 ventanas de desarrollo × tres repeticiones. '
        'Cómputo posterior a recepción: excluye UART, adquisición, FIR offline, inicialización/SELECT, BASE, PROFILE y POST. '
        'El reloj HAL es de 180 MHz, no medido externamente; los extremos no son WCET.', '',
        '## T6. Ahorro físico del procedimiento por costo: solo semilla 42', '',
        table(['Carga (HP)', 'F14 (ms)', 'Rendimiento (ms)', 'Costo (ms)', 'Ahorro vs. F14 (%)', 'Ahorro vs. rendimiento (%)'],
            [[r['outer_load_hp'], f"{r['F14_mean_ms_seed42']:.4f}", f"{r['performance_mean_ms_seed42']:.4f}",
                f"{r['cost_mean_ms_seed42']:.4f}", f"{r['cost_reduction_vs_F14_percent']:.4f}",
                f"{r['cost_reduction_vs_performance_percent']:.4f}"] for r in d['physical_procedure_comparison']]), '',
        'En 0 HP la selección por costo añade un ahorro que cumple el objetivo interno del 20 % frente a F14; '
        'en 1 y 2 HP ambos selectores usan la misma máscara. En 3 HP, F14 es un respaldo: '
        'su diferencia de tiempo del 0.4819 % frente a NF12 es incidental y no cumple el objetivo del 20 %. '
        'No corresponde transferir estos tiempos a las diez semillas.', '',
        '## T7. Detector de normalidad: evaluación exterior por carga', '',
        table(['Carga (HP)', 'Especificidad NORMAL (%)', 'Recall IR (%)', 'Recall OR (%)', 'BA binaria (%)'],
            [[r['outer_load_hp'], f"{r['specificity_N_percent']:.1f}", f"{r['recall_IR_percent']:.1f}",
                f"{r['recall_OR_percent']:.1f}", f"{r['binary_balanced_accuracy_percent']:.1f}"] for r in d['detector_per_load']]), '',
        'Mismo resultado por procedimiento. Especificidad media por carga: 50 %; rechazo de normales ponderado por ventana: '
        '86/202 = 42.5743 %. Son denominadores diferentes. El detector no modifica la clase de la DNN y '
        'no se interpreta como confirmador fiable de falla ante cambio de carga.', '',
        '## T8. Cuantización y cobertura física', '',
        table(['Comprobación', 'Resultado', 'Alcance'], [
            ['Cambios FP32→INT8', '482 / 27600 pares; 41 / 80 modelos', 'Comparación predictiva PC multisemilla'],
            ['Cambios con / sin clipping', '273 / 209', 'No establece causalidad del clipping'],
            ['Valores recortados', '2390 / 304290', 'Valores de características, no ventanas independientes'],
            ['RAW', '56 ventanas, 336 pares, 1008 RUN', 'Banco parcial de desarrollo, 42 ventanas por modelo'],
            ['POST', '2760 pares sobre 1380 ventanas', 'Desde características PC guardadas; sin extracción RAW'],
            ['Coincidencia INT8 RAW', '336 / 336 pares', 'Misma entrada; extracción flotante con tolerancia'],
            ['Flash / RAM estática', '185388 B / 88460 B', 'Banco de ocho modelos, incluye reservas'],
            ['Arena observada / capacidad', '1188 B / 16384 B', 'Arena única reutilizada'],
            ['Stack reservado / máximo medido', '4096 B / no medido', 'Reserva no equivale a máximo de ejecución']]), '',
        '## S1. Diferencias pareadas descriptivas', '',
        table(['Contraste', 'Δmacro-F1 (pp)', 'Δexactitud (pp)', 'DE entre semillas (pp)', 'Rango entre semillas (pp)'],
            [[NAMES[r['left']]+' − '+NAMES[r['right']], f"{r['mean_macro_f1_difference_pp']:+.4f}",
                f"{r['mean_accuracy_difference_pp']:+.4f}", f"{r['seed_averaged_over_four_loads_sd_pp']:.4f}",
                f"[{r['seed_averaged_over_four_loads_min_pp']:+.4f}; {r['seed_averaged_over_four_loads_max_pp']:+.4f}]"]
                for r in d['paired_descriptive']]), '',
        'Cada observación de la dispersión ya promedia las cuatro cargas de una semilla. '
        'No se calcularon intervalos de confianza o valores p ni se demostró no inferioridad.', '',
        '## S2. Máscaras candidatas', '',
        table(['Máscara', 'Dimensión', 'Índices de las 14 características'],
            [[name, len(indices), ', '.join(map(str, indices))] for name, indices in d['candidate_indices'].items()]), '',
        'Orden por posición: RMS, asimetría, curtosis, energía, frecuencia dominante, amplitud dominante y entropía; '
        'primero DE (0–6), después FE (7–13). El costo incluye la unión de operaciones necesarias para DNN y detector.', '']
    return '\n'.join(s)


def figures(d):
    import os
    import sys
    os.environ['MPLCONFIGDIR'] = str(ROOT/'.codex_tmp/cwru_mpl_2026-10-28')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
        'savefig.facecolor': 'white', 'pdf.fonttype': 42})
    colors = ['#245b9e', '#d17616', '#13816c', '#7953a0', '#555555']
    marks = ['o', 's', 'D', '^', 'x']
    fig, ax = plt.subplots(figsize=(7.2, 4.7))
    for row, color, mark in zip(d['predictive_equal_load_seed_means'], colors, marks):
        ax.plot([v['load_hp'] for v in row['per_load']], [v['macro_f1_percent'] for v in row['per_load']],
            marker=mark, color=color, label=row['display'], linewidth=1.6, markersize=5)
    ax.set(xticks=range(4), xlabel='Carga exterior reservada (HP)', ylabel='Macro-F1 medio (%)', ylim=(0, 104))
    ax.grid(axis='y', alpha=.2)
    ax.set_title('CWRU: desempeño exterior por carga', fontsize=12)
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.18), ncol=2, frameon=False, fontsize=9)
    fig.subplots_adjust(left=.10, right=.98, top=.90, bottom=.29)
    for ext in ('png', 'pdf'):
        fig.savefig(DEST/('macrof1_por_carga.'+ext), dpi=300, metadata={'CreationDate': None} if ext == 'pdf' else None)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.2, 4.7))
    rows = d['physical_seed42_latencies']
    x = np.arange(4)
    for which, color, label in [(0, '#245b9e', 'F14'), (1, '#13816c', 'Otra instancia canónica')]:
        chosen = rows[which::2]
        mean = np.asarray([r['mean_ms'] for r in chosen])
        low = mean-np.asarray([r['min_ms'] for r in chosen])
        high = np.asarray([r['max_ms'] for r in chosen])-mean
        bars = ax.bar(x+(which-.5)*.34, mean, .32, color=color, label=label,
            yerr=np.asarray([low, high]), error_kw={'linewidth': .8, 'capsize': 3})
        for b, v in zip(bars, mean):
            ax.text(b.get_x()+b.get_width()/2, v+1.7, f'{v:.2f}', ha='center', fontsize=9)
    ax.set(xticks=x, xticklabels=['0 HP\nF14 / T6', '1 HP\nF14 / FE7', '2 HP\nF14 / T6', '3 HP\nF14 / NF12'],
        ylabel='Tiempo principal por ventana (ms)', ylim=(0, 103))
    ax.set_title('STM32: ocho instancias canónicas, semilla 42', fontsize=12)
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.19), ncol=2, frameon=False)
    ax.grid(axis='y', alpha=.2);ax.set_axisbelow(True)
    fig.subplots_adjust(left=.11, right=.98, top=.90, bottom=.28)
    for ext in ('png', 'pdf'):
        fig.savefig(DEST/('latencia_canonicos.'+ext), dpi=300, metadata={'CreationDate': None} if ext == 'pdf' else None)
    plt.close(fig)
    environment = dict(python=sys.version, matplotlib=matplotlib.__version__,
        numpy=np.__version__, matplotlib_source=str(Path(matplotlib.__file__).resolve()),
        backend='Agg', raw_results_modified=False, new_inferences=0)
    env_path = DEST/'ENTORNO_GRAFICOS.json'
    if env_path.exists():
        require(read(env_path) == environment, 'Cambió el entorno de gráficos')
    else:
        save(env_path, environment)


def generate():
    targets = ['CONSOLIDACION_ARTICULO.json', 'TABLAS_CONSOLIDADAS.md', 'macrof1_por_carga.png',
        'macrof1_por_carga.pdf', 'latencia_canonicos.png', 'latencia_canonicos.pdf']
    require(DEST.is_dir() and not any((DEST/name).exists() for name in targets), 'No sobrescribir consolidación')
    # La disponibilidad gráfica se comprueba antes de crear las tablas.
    import matplotlib
    import numpy
    d = build()
    save(DEST/targets[0], d)
    with (DEST/targets[1]).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(render_tables(d))
    figures(d)
    print(json.dumps(dict(status=d['status'], new_training_fits=0, new_classifier_invocations=0,
        manuscript_modified=False, destination=str(DEST)), ensure_ascii=True))


def figures_only():
    names = ['macrof1_por_carga.png', 'macrof1_por_carga.pdf', 'latencia_canonicos.png',
        'latencia_canonicos.pdf', 'ENTORNO_GRAFICOS.json']
    require(not any((DEST/name).exists() for name in names), 'No sobrescribir figuras')
    current = build()
    require(read(DEST/'CONSOLIDACION_ARTICULO.json') == current, 'Cifras distintas antes de graficar')
    require((DEST/'TABLAS_CONSOLIDADAS.md').read_text(encoding='utf-8') == render_tables(current), 'Tabla alterada')
    figures(current)
    print('ARTICLE_FIGURES_CREATED_FROM_VERIFIED_CONSOLIDATION')


def check():
    current = build()
    require(read(DEST/'CONSOLIDACION_ARTICULO.json') == current, 'Cifras consolidadas no coinciden con cierres')
    require((DEST/'TABLAS_CONSOLIDADAS.md').read_text(encoding='utf-8') == render_tables(current), 'Tabla alterada')
    digest = verify_manifest(DEST/'MANIFEST_CONSOLIDACION_SHA256.json')
    print(json.dumps(dict(status='PASS_ARTICLE_CONSOLIDATION_SOURCES_NUMBERS_AND_SEAL',
        manifest_sha256=digest, new_training_fits=0, new_classifier_invocations=0), ensure_ascii=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--generate', action='store_true')
    group.add_argument('--figures', action='store_true')
    group.add_argument('--check', action='store_true')
    args = parser.parse_args()
    generate() if args.generate else figures_only() if args.figures else check()
