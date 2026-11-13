"""Consolida salidas guardadas de Canon42; no ejecuta modelos ni usa la placa.

Exige la auditoría de trazas aprobada y escribe en una carpeta nueva.
No reescribe evidencias o documentación histórica sellada.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT/'despliegue_stm32/canonicos_42_180mhz_2026-10-27'
FULL = OUT/'Validacion_Completa_PC/resultados/20261028T162240_469517Z'
DEST = ROOT/'articulo/ampliacion_cwru/metodologia/validacion_fisica_canonicos_completa_2026-10-28'
AUDIT = OUT/'AUDITORIA_FULL_CANON42.json'
EXPECTED_AUDIT_SHA = '0a17d5b980269f99c392a57619526432709747d839a5005dc21959940a64f690'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def metrics(matrix):
    matrix = np.asarray(matrix, dtype=np.int64)
    denominator = matrix.sum(axis=0) + matrix.sum(axis=1)
    f1 = np.divide(2*np.diag(matrix), denominator, out=np.zeros(3), where=denominator != 0)
    return dict(confusion_matrix=matrix.tolist(), windows=int(matrix.sum()),
        correct=int(np.trace(matrix)), accuracy_percent=float(100*np.trace(matrix)/matrix.sum()),
        macro_f1_percent=float(100*np.mean(f1)))


def build_summary():
    assert sha(AUDIT) == EXPECTED_AUDIT_SHA, 'Auditoría distinta de la aprobada'
    audit = read(AUDIT)
    assert audit['status'] == 'PASS_CANON42_STAGED_FULL_AUDITED'
    assert sha(FULL/'MANIFEST_EVIDENCIAS_SHA256.json') == audit['full_evidence_manifest_sha256']
    entries = read(FULL/'MANIFEST_EVIDENCIAS_SHA256.json')
    for relative, expected in entries.items():
        target = (FULL/relative).resolve()
        assert target.is_relative_to(FULL.resolve()) and sha(target) == expected
    events = [json.loads(line) for line in (FULL/'MEDICIONES.jsonl').read_text(encoding='utf-8').splitlines()]
    event_hash = hashlib.sha256(json.dumps(events, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    assert event_hash == audit['verified_events_sha256']
    cat = read(OUT/'recursos/CATALOGO_TOTAL.json')
    bank = read(OUT/'bancos/BANCOS_VALIDACION.json')
    canonical = read(ROOT/'articulo/ampliacion_cwru/metodologia/cierre_evaluacion_exterior_v2_2026-10-27/MODELOS_CANONICOS_42_PENDIENTES_STM32.json')
    original = {row['model_job_id']: row for row in canonical['models']}
    runs = [row for row in events if row['type'] == 'RUN']
    profiles = [row for row in events if row['type'] == 'PROFILE']
    posts = [row for row in events if row['type'] == 'POST']
    bases = [row for row in events if row['type'] == 'BASE']
    latencies, prediction_rows = [], []
    for rep in cat['representatives']:
        rid = rep['id']
        selected = [r for r in runs if r['rep_id'] == rid]
        times = np.asarray([r['total_cycles'] for r in selected], dtype=float)/180000
        assert len(times) == 126
        row = dict(id=rid, model=rep['representative_id'], candidate=rep['candidate'],
            outer_load_hp=rep['outer_load_hp'], seed=42, windows=42, RUN=126,
            mean_ms=float(times.mean()), median_ms=float(np.median(times)),
            p95_ms=float(np.percentile(times, 95)), min_ms=float(times.min()), max_ms=float(times.max()),
            sample_std_ms=float(times.std(ddof=1)), CV_percent=float(100*times.std(ddof=1)/times.mean()))
        prof = [r for r in profiles if r['rep_id'] == rid]
        sums = {key: float(np.mean([r[key] for r in prof])) for key in
            ['total_cycles', 'extract_cycles', 'input_cycles', 'nn_cycles', 'detector_cycles', 'residual_cycles']}
        row['PROFILE'] = dict(count=len(prof), mean_ms={k: v/180000 for k, v in sums.items()},
            extraction_percent=100*sums['extract_cycles']/sums['total_cycles'])
        latencies.append(row)
        cm = np.zeros((3, 3), dtype=np.int64)
        rejected = np.zeros(3, dtype=np.int64)
        by_class = np.zeros(3, dtype=np.int64)
        for item in posts:
            if item['rep_id'] == rid:
                cm[item['label'], item['predicted_class']] += 1
                rejected[item['label']] += int(item['rejected'])
                by_class[item['label']] += 1
        result = metrics(cm)
        saved = original[rep['representative_id']]['metrics']['INT8']
        assert result['confusion_matrix'] == saved['confusion_matrix'], 'Matriz distinta del exterior cerrado'
        assert abs(result['accuracy_percent']/100 - saved['accuracy']) < 1e-12
        assert abs(result['macro_f1_percent']/100 - saved['macro_f1']) < 1e-12
        prediction_rows.append(dict(id=rid, model=rep['representative_id'], **result,
            saved_outer_INT8_seed42_exact=True, detector_rejected_by_class=rejected.tolist(),
            detector_class_counts=by_class.tolist(), normal_rejection_percent=float(100*rejected[0]/by_class[0])))
    comparisons = [dict(outer_load_hp=fold, baseline_model=latencies[2*fold]['model'],
        alternative_model=latencies[2*fold+1]['model'],
        mean_time_reduction_percent=100*(1-latencies[2*fold+1]['mean_ms']/latencies[2*fold]['mean_ms']))
        for fold in range(4)]
    by_gid = {r['global_id']: r for r in bank['RAW_windows']}
    feature_error = []
    with np.load(OUT/'bancos/banco_raw.npz', allow_pickle=False) as raw:
        for row in bases:
            arm = np.asarray([int(v, 16) for v in row['features_bits']], dtype='<u4').view('<f4').astype(float)
            pc = raw['features_pc_cache14'][by_gid[row['global_id']]['npz_index']].astype(float)
            diff = np.abs(arm-pc)
            assert np.allclose(arm, pc, rtol=.001, atol=.00001) and np.array_equal(arm[[4, 11]], pc[[4, 11]])
            for idx in range(14):
                feature_error.append(dict(global_id=row['global_id'], feature_index=idx,
                    PC=float(pc[idx]), ARM=float(arm[idx]), absolute_error=float(diff[idx]),
                    relative_error=float(diff[idx]/abs(pc[idx])) if pc[idx] != 0 else None,
                    tolerance_fraction=float(diff[idx]/(.00001+.001*abs(pc[idx])))))
    pairs = [r['comparison']['cached_PC_differences'] for r in runs if r['repetition'] == 1]
    assert len(pairs) == 336 and all(r['comparison']['oracle'] == 'saved_same_INT8_input' for r in runs)
    hop_ms = 2048/12000*1000
    largest = max(r['max_ms'] for r in latencies)
    return dict(status='CANON42_PHYSICAL_STAGED_VALIDATION_COMPLETE_AUDITED_PENDING_REPORTING',
        date_local='2026-10-28', timezone='America/Lima', source_audit_sha256=sha(AUDIT),
        coverage=audit['coverage'], wall_seconds=audit['wall_seconds'],
        imported_counts=audit['imported_counts'], new_counts=audit['new_completed'],
        event_origins=dict(Counter(r['measurement_source'] for r in events)), latencies=latencies,
        same_fold_time_comparison=comparisons, POST_seed42=prediction_rows,
        RAW_PC_ARM=dict(pairs=336, pairs_with_feature_bit_difference=sum(r['features'] > 0 for r in pairs),
            feature_bit_differences_sum=sum(r['features'] for r in pairs),
            input_INT8_differences_sum=sum(r['int8_values'] for r in pairs),
            output_INT8_differences_sum=sum(r['outputs'] for r in pairs),
            class_differences=sum(r['class_differs'] for r in pairs),
            detector_decision_differences=sum(r['detector_rejection_differs'] for r in pairs),
            detector_score_bit_differences=sum(r['detector_score_differs'] for r in pairs),
            max_absolute_error=max(feature_error, key=lambda r: r['absolute_error']),
            max_relative_error_nonzero_reference=max((r for r in feature_error if r['relative_error'] is not None), key=lambda r: r['relative_error']),
            max_tolerance_fraction=max(r['tolerance_fraction'] for r in feature_error),
            dominant_bins_exact=True),
        timing_scope=dict(hop_ms=hop_ms, max_observed_ms=largest, observed_margin_ms=hop_ms-largest,
            max_observed_hop_percent=100*largest/hop_ms, WCET_claimed=False,
            autonomous_acquisition_validated=False, RAW_heldout_exhaustive=False),
        arena_used_bytes_by_model=audit['arena_used_by_model'], classifier_invocations=0,
        training_fits=0, serial_port_opened=False, hardware_flashed=False, manuscript_modified=False)


def markdown(s):
    lines = ['# Validación física canónica CWRU: resultados auditados', '',
        '28 de octubre de 2026, America/Lima.', '',
        'La validación escalonada acordada quedó completa y auditada para los ocho modelos INT8 de semilla 42. '
        'La ejecución nueva duró 1005.016 s (16 min 45.016 s). Se conservaron los modelos, parámetros y decisiones congeladas.', '',
        '## Cobertura y equivalencia', '',
        'El total reúne 56 ventanas RAW únicas, 336 pares modelo–ventana, 1008 RUN, 336 PROFILE y 56 BASE. '
        'Cada modelo tiene 42 ventanas de desarrollo con tres RUN y un PROFILE por ventana. '
        'POST cubre los 2760 pares correspondientes a las 1380 ventanas exteriores, desde sus características PC guardadas.', '',
        'Del gate se reutilizaron 24 RUN, ocho PROFILE, dos BASE y 24 POST. La ejecución nueva añadió '
        '984 RUN, 328 PROFILE, 54 BASE y 2736 POST; 55 transmisiones RAW. '
        'La reconstrucción verifica 28836 pares UART nuevos y 7315 eventos, incluidos 93 importados del gate. '
        'El contador inicial informado fue cero, en una sesión distinta de la del gate; el nuevo registro conserva su origen y termina en 4048.', '',
        'En los 336 pares RAW, la entrada INT8, la salida, la clase y la decisión de rechazo coincidieron '
        'con PC. En los 2760 POST, las matrices y métricas coinciden con las salidas INT8 exteriores guardadas de semilla 42. '
        'No se invocaron nuevos clasificadores PC para estas comprobaciones. PASS expresa equivalencia funcional; '
        'no implica que toda predicción sea correcta frente a su etiqueta.', '',
        '## Latencia principal medida', '',
        'Solo se usan los 126 RUN por modelo; UART, parser, checksums, inicialización/SELECT, BASE, PROFILE y POST quedan excluidos. '
        'Los percentiles son descriptivos del banco. El reloj informado por HAL es de 180 MHz; no se midió externamente.', '',
        '| Carga exterior | Modelo | Media (ms) | Mediana (ms) | p95 (ms) | Máximo observado (ms) |',
        '|---|---|---:|---:|---:|---:|']
    for r in s['latencies']:
        lines.append(f"| {r['outer_load_hp']} HP | {r['candidate']} | {r['mean_ms']:.4f} | {r['median_ms']:.4f} | {r['p95_ms']:.4f} | {r['max_ms']:.4f} |")
    lines += ['', 'Las reducciones medias frente a F14 de la misma carga exterior son 52.9313 % para T6 en 0 HP, '
        '50.0677 % para FE7 en 1 HP y 52.9131 % para T6 en 2 HP. NF12 en 3 HP es 0.4842 % más lento que F14. '
        'El selector por costo conserva el respaldo F14 en esa última carga; no se atribuye ahorro a ese respaldo.', '',
        'La extracción ocupa 99.8353–99.9107 % del total instrumentado. El salto de 2048 muestras a 12 kHz equivale a '
        f"{s['timing_scope']['hop_ms']:.6f} ms. El máximo observado ({s['timing_scope']['max_observed_ms']:.6f} ms) "
        f"deja {s['timing_scope']['observed_margin_ms']:.6f} ms en ese intervalo teórico. "
        'Esta comparación no demuestra adquisición continua ni WCET.', '',
        '## Diferencias numéricas y memoria', '',
        '330 de los 336 pares presentan alguna característica PC–ARM distinta en bits; hay 804 diferencias de características '
        'y 158 distancias de detector distintas en bits. Los bins dominantes coinciden en las 56 BASE. '
        'El máximo error absoluto es 1.0 en energías del orden de millones. Por ejemplo, gid1206 registra '
        '8530657 en PC y 8530656 en ARM: error relativo 1.17224×10⁻⁷. El máximo error relativo no nulo es '
        '1.99461×10⁻⁷ y el mayor error utiliza 0.0199448 % de la tolerancia admitida. '
        'Se documenta tolerancia en características y equivalencia posterior; no se afirma identidad bit a bit de toda la extracción.', '',
        'La arena observada es 1188 B por modelo, con capacidad reservada única de 16 KiB. '
        'Los 185388 B de Flash y 88460 B de RAM estática son del banco con ocho modelos y reservas de heap/stack. '
        'No describen un firmware productivo individual. No se midió el stack máximo de este firmware.', '',
        '## Resultados predictivos reproducidos: solo semilla 42', '',
        '| Carga exterior | Modelo | Aciertos/ventanas | Exactitud (%) | Macro-F1 (%) | Rechazo NORMAL (%) |',
        '|---|---|---:|---:|---:|---:|']
    for model, r in zip(s['latencies'], s['POST_seed42']):
        lines.append(f"| {model['outer_load_hp']} HP | {model['candidate']} | {r['correct']}/{r['windows']} | {r['accuracy_percent']:.4f} | {r['macro_f1_percent']:.4f} | {r['normal_rejection_percent']:.1f} |")
    lines += ['', 'Estas ocho instancias fijas no sustituyen el resumen de diez semillas de la evaluación exterior. '
        'Las limitaciones predictivas permanecen: generalización débil en 0 HP, deterioro de NF12 en 3 HP y '
        'rechazo de todas las ventanas normales en 0 y 3 HP. El detector reproduce el mismo problema de normalidad de PC; '
        'no es un confirmador fiable de falla frente al cambio de carga. No se recalibraron umbrales con estos resultados.', '',
        '## Alcance cerrado y siguiente paso', '',
        'Quedó cerrada la equivalencia física en el alcance escalonado elegido: RAW parcial de desarrollo y POST '
        'exhaustivo desde características exteriores. Se conservaron los reportes y sellos históricos, incluidos sus campos '
        'de pendiente; su resolución está en la auditoría posterior AUDITORIA_FULL_CANON42.json.', '',
        'El siguiente paso es consolidar evaluación exterior, costo físico, comparación con baselines y limitaciones '
        'para preparar la redacción de CWRU. El manuscrito permanece sin cambios en este cierre. No se requieren '
        'nuevos entrenamientos o repetir esta campaña para trasladar sus resultados a la documentación. '
        'Adquisición autónoma, energía, WCET, stack máximo y RAW exhaustivo exterior quedan fuera de lo validado.', '']
    return '\n'.join(lines)


def main():
    assert not DEST.exists(), 'No sobrescribir un cierre existente'
    summary = build_summary()
    DEST.mkdir()
    save(DEST/'RESUMEN_VALIDACION_FISICA.json', summary)
    with (DEST/'RESULTADOS_Y_LIMITES.md').open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(markdown(summary))
    print(json.dumps(dict(status=summary['status'], destination=str(DEST), coverage=summary['coverage']), ensure_ascii=True))


if __name__ == '__main__':
    main()
