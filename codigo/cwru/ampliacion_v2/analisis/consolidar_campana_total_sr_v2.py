"""Consolida costos SR auditados, pendientes de identidad Flash; sin hardware/TF.

No habilita el selector ni modifica el protocolo sellado o la captura original.
Produce una tabla provisional nueva, un resumen metodologico y su manifiesto.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT/'despliegue_stm32/costo_total_single_rounding_v2_180mhz_2026-10-26'
RUN = OUT/'Validacion_PC/resultados/20261026T144540_396879Z'
METHOD = ROOT/'articulo/ampliacion_cwru/metodologia/cierre_campana_total_sr_2026-10-26'
RULES = ROOT/'articulo/ampliacion_cwru/metodologia/protocolo_v2_2026-10-25'
TABLE = OUT/'COSTOS_TOTALES_SR_PROVISIONALES.json'
REPORT = OUT/'CIERRE_AUDITORIA_CAMPANA_SR.json'
MANIFEST = OUT/'MANIFEST_AUDITORIA_CAMPANA_SR_SHA256.json'
ELF_SHA = 'c0c3853d2b0e90bb20fe00be13bcf107d186088406be131b729a605ded44ecc7'
ORDER = ['F14','T6','S8','NF12','NM10','NE12','DE7','FE7']
EXPECTED_LOADS = {0:[2,3],1:[2,3],2:[1,3],3:[1,2]}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, data):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def sealed(path, base):
    digest, name = path.with_suffix('.sha256').read_text(encoding='ascii').split()
    require(name == path.name and sha(path) == digest, 'Manifiesto invalido')
    entries = read(path)
    for rel, expected in entries.items():
        target = (base/rel).resolve()
        require(target.is_relative_to(base.resolve()) and sha(target) == expected,
                'Recurso cambiado: '+rel)
    return digest, entries


def main():
    started = time.perf_counter()
    require(not any(p.exists() for p in (TABLE, REPORT, MANIFEST, METHOD)), 'No sobrescribir consolidacion')
    capture_sha, capture = sealed(RUN/'MANIFEST_EVIDENCIAS_SHA256.json', RUN)
    resource_sha, resources = sealed(OUT/'recursos/MANIFEST_RECURSOS_SHA256.json', OUT/'recursos')
    trace = read(OUT/'AUDITORIA_TRAZA_CAMPANA_SR.json')
    numeric = read(OUT/'AUDITORIA_NUMERICA_CAMPANA_SR.json')
    result = read(RUN/'RESULTADO_TOTAL.json')
    cat = read(RUN/'CATALOGO_TOTAL.json')
    bank_rows = {row['global_id']:row for row in read(RUN/'BANCO_TOTAL.json')['windows']}
    protocol = read(RULES/'protocolo_reglas.json')
    require(result['status'] == 'FULL_ARM_MEASUREMENTS_READY_FOR_AUDIT'
            and (result['completed_primary_RUN'], result['completed_PROFILE'], result['completed_BASE'],
                 result['negative_rejections']) == (2688,896,42,17), 'Campana incompleta')
    require(trace['status'] == 'PASS_SR_FULL_TRACE_REPLAYED_NO_HARDWARE'
            and numeric['status'] == 'PASS_SR_FULL_NUMERIC_AUDIT_NOT_PREDICTIVE_EVALUATION',
            'Faltan auditorias aprobadas')
    require(numeric['capture_manifest_sha256'] == capture_sha
            and numeric['elf_sha256'] == result['firmware_elf_sha256'] == ELF_SHA,
            'Auditorias de otra captura o ELF')
    for name, row in [('auditar_traza_full_total_sr_v2.py', trace),
                      ('auditar_numerica_full_total_sr_v2.py', numeric)]:
        require(sha(Path(__file__).with_name(name)) == row['script_sha256'], 'Script de auditoria cambiado')
        for key in ('helper_sha256', 'data_sha256'):
            for rel, expected in row.get(key, {}).items():
                require(sha(ROOT/rel) == expected, 'Dependencia de auditoria cambiada: '+rel)
    require(numeric['same_INT8_input_output_exact_across_3584_publications']
            and numeric['all_intermediates_exact_across_3584_publications']
            and all(numeric[key] == 0 for key in ('quantized_input_differences_to_original_cache',
                'output_differences_to_original_cache','class_differences_to_original_cache',
                'normality_rejection_differences_to_original_cache')), 'Numerica no cerrada')
    require(not result['total_cost_validated'] and not result['selector_enabled']
            and not result['programmer_report_attached'], 'Revisar nueva evidencia antes de consolidar')
    require(protocol['total_cost_calibration']['benchmark_repetitions'] == 3
            and protocol['total_cost_calibration']['selector_cost_status_required'] == 'VALIDATED_TOTAL_COST',
            'Contrato de costos diferente')
    events = [json.loads(line) for line in (RUN/'MEDICIONES.jsonl').read_text(encoding='utf-8').splitlines()]
    rows = []
    by_outer = {str(i):{} for i in range(4)}
    for rep in cat['representatives']:
        rid, outer, name = rep['id'], rep['outer_load_hp'], rep['candidate']
        pubs = [e for e in events if e['type'] == 'RUN' and e['rep_id'] == rid]
        ids = rep['scope']['benchmark_global_window_ids']
        loads = sorted({int(bank_rows[gid]['metadata']['load_hp']) for gid in ids})
        require(sorted(loads) == EXPECTED_LOADS[outer] and len(ids) == 28
                and len(pubs) == 84 and {e['global_id'] for e in pubs} == set(ids), 'Banco temporal incorrecto')
        require(all(sum(e['global_id'] == gid for e in pubs) == 3 for gid in ids), 'Repeticiones incompletas')
        cycles = [e['total_cycles'] for e in pubs]
        mean = sum(cycles)/84/180000
        lower, upper = min(cycles)/180000, max(cycles)/180000
        measured = next(r for r in numeric['representatives'] if r['rep_id'] == rid)
        require(math.isclose(measured['primary_statistics']['mean_ms'], mean, rel_tol=0, abs_tol=1e-12)
                and lower == measured['primary_statistics']['min_cycles']/180000
                and upper == measured['primary_statistics']['max_cycles']/180000, 'Estadisticas distintas')
        row = dict(status='PENDING_FIRMWARE_IMAGE_VERIFICATION', cost_scope='full_pipeline', is_proxy=False,
            outer_load=outer, candidate=name, representative_id=rep['representative_id'], rep_id=rid,
            inner_validation_load=rep['inner_validation_load_hp'], seed=11,
            benchmark_loads=sorted(loads), benchmark_global_window_ids=ids,
            observations=84, unique_windows=28, repetitions=3,
            mean_ms=mean, lower_ms=lower, upper_ms=upper,
            min_cycles=min(cycles), max_cycles=max(cycles), mean_cycles=sum(cycles)/84,
            median_ms=measured['primary_statistics']['median_ms'], p95_ms=measured['primary_statistics']['p95_ms'],
            sample_std_ms=measured['primary_statistics']['sample_std_ms'],
            model_sha256=rep['model_sha256'], source_fit_signature=rep['source_fit_signature'],
            params_fnv=rep['params_fnv'], arena_used_bytes=1188,
            envelope_interpretation='observed_min_max_not_CI_or_WCET',
            firmware_elf_sha256=ELF_SHA, numerical_and_protocol_audits_passed=True,
            Flash_readback_verified=False, cost_selection_enabled=False)
        rows.append(row)
        by_outer[str(outer)][name] = row
    require(len(rows) == 32 and all(list(group) == ORDER for group in by_outer.values()), 'Catalogo incompleto')
    for group in by_outer.values():
        for row in group.values():
            row['meets_20percent_timing_condition_only'] = row['upper_ms'] <= .8*group['F14']['lower_ms']
            row['predictive_and_detector_eligibility_not_evaluated_here'] = True
    table = dict(status='AUDITED_TOTAL_COST_PENDING_FLASH_IDENTITY', date_local='2026-10-26', timezone='America/Lima',
        capture=RUN.relative_to(ROOT).as_posix(), capture_manifest_sha256=capture_sha,
        resource_manifest_sha256=resource_sha, firmware_elf_sha256=ELF_SHA,
        clock_HAL_Hz=180000000, clock_externally_measured=False, rows=rows, costs_by_outer=by_outer,
        selector_cost_status_required='VALIDATED_TOTAL_COST', cost_selection_enabled=False,
        qualification='Measured and audited costs; protocol firmware provenance proof pending. Not selectable yet.',
        units='ms', interval='empirical_min_max', full_scope_includes=['features','scaling_quantization',
            'input_copy_Invoke_output_copy_argmax','normality','integration_residual'],
        excludes=['UART','BASE','PROFILE','SELECT_initialization'],
        overhead_subtracted=False, WCET_claimed=False, motor_independence_claimed=False,
        predictive_outer_evaluation=False, manuscript_modified=False)
    write(TABLE, table)
    METHOD.mkdir(parents=True, exist_ok=False)
    lines = ['# Auditoría de la campaña completa de costo total SR', '',
        'La campaña `20261026T144540_396879Z` completó 2688 RUN, 896 PROFILE, 42 BASE y 17 rechazos. '
        'Duró 981.047 s (16 min 21 s). Las auditorías de protocolo y numérica aprobaron sin repetir el ensayo.', '',
        'Se reconstruyeron 45210 filas UART y 22605 transacciones. Las 42 ventanas transmitidas '
        'contienen 344064 valores float32, exactos al banco y a las filas fuente. Cada uno de los '
        '32 representantes tiene 28 ventanas de sus dos cargas de entrenamiento y 84 RUN principales. '
        'Se excluyen las cargas externa y de validación interna de su banco.', '',
        'En las 3584 publicaciones, el replay desde las características ARM reproduce escalado, '
        'cuantización y detector. Las entradas INT8, salidas, clases y decisiones coinciden con '
        'las referencias PC selladas. Las características float32 originales pueden diferir dentro '
        'de las tolerancias fijadas; no se afirma identidad binaria de todo el preprocesamiento. '
        'Se calcularon además 2688 capas FC enteras offline (29568 neuronas); no son activaciones internas capturadas en la placa.', '',
        '## Tiempos principales por escenario', '',
        'Medias en ms, con 84 observaciones por celda. El escenario identifica la carga externa '
        'excluida, no una medición sobre esa carga. El JSON conserva mínimos, máximos, mediana, '
        'P95, desviación y bancos exactos por representante.', '',
        '| Máscara | Externo 0 HP | Externo 1 HP | Externo 2 HP | Externo 3 HP | Condición temporal 20 % en los cuatro |',
        '|---|---:|---:|---:|---:|:---:|']
    for name in ORDER:
        values = [by_outer[str(i)][name] for i in range(4)]
        lines.append('| '+name+' | '+' | '.join(f"{r['mean_ms']:.6f}" for r in values)+' | '+
                     ('Sí' if all(r['meets_20percent_timing_condition_only'] for r in values) else 'No')+' |')
    lines += ['', 'La regla congelada utiliza L=mínimo y U=máximo observado, con '
        '`U_candidato <= 0.8 * L_F14`. La columna temporal no aplica las guardias de F1 y detector '
        'y no selecciona un modelo. P95 y media son descriptores, no sustituyen esa regla. '
        'NF12 no redujo el tiempo respecto de F14: quitar características no asegura un ahorro proporcional.', '',
        '## Memoria y alcance', '',
        'Los 32 modelos informaron 1188 B usados de una arena reservada de 16384 B. '
        'La Flash de 280276 B y RAM con reservas de 88460 B corresponden al banco de 32 modelos. '
        'Esta campaña no vuelve a medir el máximo de stack ni la ocupación de un despliegue final.', '',
        '## Evidencia pendiente antes de habilitar los costos', '',
        'El diseño previo `calibracion_costo_total_v2_2026-10-25/ALCANCE_Y_DISENO.md` exige '
        'vincular el ELF, la compilación y la verificación del programador antes del cierre. '
        'La captura no adjunta ese registro y no se localizó un log vinculable. El ELF declarado '
        'y los fingerprints de modelos y parámetros sí coinciden, pero no verifican todos los '
        'bytes cargables del programa en Flash.', '',
        'Se prepara una lectura posterior por SWD para contrastar los segmentos Flash del ELF '
        'auditado sin reprogramar. Será evidencia posterior de correspondencia y no un log '
        'histórico contemporáneo a la campaña. Hasta revisarla, la tabla tiene estado '
        '`AUDITED_TOTAL_COST_PENDING_FLASH_IDENTITY`, y sus filas no son `VALIDATED_TOTAL_COST`. '
        'El selector permanece deshabilitado.', '',
        '## Continuidad', '',
        'No se modifica el protocolo sellado ni se rellenan sus campos históricos. Los 32 ajustes '
        'de calibración forman parte de los 288 internos previstos y podrán reutilizarse si '
        'coinciden las firmas. La selección también requiere completar las tres semillas '
        'y las tres validaciones internas por escenario; no procede elegir una máscara solo '
        'por estos tiempos. Antes del resto de entrenamientos se preparará el ejecutor '
        'reanudable y se congelarán sus límites, entregando el comando al usuario. '
        'El piloto RF/SVM ya está cerrado y no se repite.', '',
        'Las capturas y sellos anteriores se conservan. No se ejecutaron nuevos entrenamientos, '
        'inferencias TFLite ni mediciones físicas durante esta auditoría; el manuscrito no se modificó.', '']
    doc = METHOD/'RESULTADOS_Y_PENDIENTE_FLASH.md'
    with doc.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write('\n'.join(lines))
    paths = {RUN/name for name in capture} | {OUT/'recursos'/name for name in resources}
    paths.update({TABLE, doc, Path(__file__).resolve(), RUN/'MANIFEST_EVIDENCIAS_SHA256.json',
        RUN/'MANIFEST_EVIDENCIAS_SHA256.sha256', OUT/'recursos/MANIFEST_RECURSOS_SHA256.json',
        OUT/'recursos/MANIFEST_RECURSOS_SHA256.sha256', OUT/'AUDITORIA_TRAZA_CAMPANA_SR.json',
        OUT/'AUDITORIA_NUMERICA_CAMPANA_SR.json', RULES/'protocolo_reglas.json',
        RULES/'DECISIONES_Y_FUNDAMENTO.md', ROOT/'codigo/cwru/ampliacion_v2/protocolo/reglas_v2.py',
        OUT/'AUDITORIA_ELF_SR.json', OUT/'AUDITORIA_MAQUINA_SR.json'})
    for row in (trace, numeric):
        for key in ('helper_sha256','data_sha256'):
            paths.update(ROOT/name for name in row.get(key, {}))
    paths.update(Path(__file__).with_name(name) for name in
        ('auditar_traza_full_total_sr_v2.py','auditar_numerica_full_total_sr_v2.py'))
    num_manifest = OUT/'MANIFEST_NUMERICA_CAMPANA_SR_SHA256.json'
    if num_manifest.exists():
        _, num_entries = sealed(num_manifest, ROOT)
        paths.update(ROOT/name for name in num_entries)
        paths.update({num_manifest,num_manifest.with_suffix('.sha256')})
    entries = {p.relative_to(ROOT).as_posix():sha(p) for p in sorted(paths)}
    closure = dict(status='FULL_SR_AUDITED_COSTS_PENDING_FLASH_IDENTITY', date_local='2026-10-26',
        timezone='America/Lima', capture_manifest_sha256=capture_sha, audited_total_cost_rows=32,
        primary_RUN=2688, PROFILE=896, BASE=42, negative_rejections=17,
        model_window_pairs=896, unique_RAW_windows=42, observations_per_representative=84,
        wall_seconds=result['wall_seconds'], all32_runtime_arena_used_bytes=1188,
        all_same_input_INT8_outputs_exact=True, data_roles_and_trace_verified=True,
        selector_enabled=False, remaining_gate='Firmware image provenance/readback',
        old_firmware_or_gate_measurements_imported=False, old_seals_modified=False,
        new_training_fits=0,new_TFLite_invocations=0,new_physical_RUN=0,
        D_files_modified=False, serial_port_opened=False, hardware_flashed=False,
        manuscript_modified=False, manifest_entries_including_closure=len(entries)+1,
        seconds_before_write=time.perf_counter()-started)
    write(REPORT, closure)
    entries[REPORT.relative_to(ROOT).as_posix()] = sha(REPORT)
    write(MANIFEST, dict(sorted(entries.items())))
    with MANIFEST.with_suffix('.sha256').open('x', encoding='ascii') as stream:
        stream.write(sha(MANIFEST)+'  '+MANIFEST.name+'\n')
    for rel, expected in entries.items():
        require(sha(ROOT/rel) == expected, 'Cambio durante cierre: '+rel)
    print(json.dumps(dict(status=closure['status'], files=len(entries),manifest_sha256=sha(MANIFEST),
        seconds=time.perf_counter()-started), ensure_ascii=True))


if __name__ == '__main__':
    main()
