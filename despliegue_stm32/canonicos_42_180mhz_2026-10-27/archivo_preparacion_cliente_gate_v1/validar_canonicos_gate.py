"""Gate UART canónico escalonado; preparar o probar, nunca programar Flash.

Comprueba 24 POST (primera ventana N/IR/OR de cada modelo), 2 RAW con 8
pares modelo-ventana, 24 RUN, 8 PROFILE y 2 BASE. No habilita el lote completo.
POST usa predicciones guardadas. RAW solo invoca Lite si la entrada INT8 ARM
difiere del caché, como oráculo para la misma entrada, con caché de consultas.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import struct
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p/'codigo/cwru/ampliacion_v2').is_dir()
    and (p/'despliegue_stm32').is_dir())
OUT = ROOT/'despliegue_stm32/canonicos_42_180mhz_2026-10-27'
PACKAGE = OUT/'Validacion_PC'
RESOURCES = OUT/'recursos'
BANKS = OUT/'bancos'
LEGACY = ROOT/'codigo/cwru/ampliacion_v2/costo_total_single_rounding/validar_total_sr.py'
HELPER_NAME = 'protocolo_total_sr_legacy.py'
HELPER_SHA = '631303e144b432a7144054150ed6466d8a8b10d1f179484329bc2e2b666d91fb'
CATALOG_SHA = 'e7a0c848ccb293c3dd95c9e150e8e6909f6ae6b82c3ec90e6332faebbc2a6fdd'
BANK_MANIFEST_SHA = 'b43fd7ef371aaf844dc91861eed3e0a420d4420c90ae0802591dc1ff8b7df173'
PREPARATION_SHA = '514c54360b77343ae3c2dd872941edf0fdad22e2216931ba63d03915472fa369'
ELF_SHA = 'c2de62a6e73a7f0f4cc2f4d25f5cbef718bde28eb93e05d0273871337ce84a46'
CONFIG = 'Release_180MHz_Canon42_SR'
GATE_STATUS = 'CANON42_STAGED_GATE_PASS_NOT_FULL_VALIDATED'


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def helper():
    path = HERE/HELPER_NAME if (HERE/HELPER_NAME).is_file() else LEGACY
    require(sha(path) == HELPER_SHA, 'Helper de protocolo modificado')
    spec = importlib.util.spec_from_file_location('canon42_sr_protocol', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def manifest(folder, name, expected=None):
    folder = Path(folder).resolve()
    path = folder/name
    digest, filename = path.with_suffix('.sha256').read_text(encoding='ascii').split()
    require(filename == name and sha(path) == digest and (expected is None or digest == expected),
        'Sello alterado: '+str(path))
    entries = read(path)
    for relative, value in entries.items():
        target = (folder/relative).resolve()
        require(target.is_relative_to(folder) and target.is_file() and sha(target) == value,
            'Entrada alterada o insegura: '+relative)
    return entries


def load_resources():
    h = helper()
    resources = manifest(RESOURCES, 'MANIFEST_RECURSOS_SHA256.json')
    manifest(BANKS, 'MANIFEST_BANCOS_SHA256.json', BANK_MANIFEST_SHA)
    cat, bank = read(RESOURCES/'CATALOGO_TOTAL.json'), read(BANKS/'BANCOS_VALIDACION.json')
    require(cat['count'] == len(cat['representatives']) == 8 and cat['seed'] == bank['seed'] == 42
        and cat['catalog_sha256'] == bank['catalog_sha256'] == CATALOG_SHA, 'No es el banco canónico')
    canonical = [{k: row[k] for k in h.CANONICAL_KEYS} for row in cat['representatives']]
    blob = json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(',', ':'),
        allow_nan=False).encode()
    require(hashlib.sha256(blob).hexdigest() == CATALOG_SHA
        and (RESOURCES/'CATALOGO_CANONICO.json').read_bytes() == blob, 'Identidad del catálogo alterada')
    aggregate = b''
    for rid, row in enumerate(cat['representatives']):
        require(row['id'] == rid and row['seed'] == 42 and row['outer_load_hp'] == rid//2,
            'Modelo/carga/semilla distintos')
        model = (RESOURCES/row['model_path']).resolve()
        require(model.is_relative_to(RESOURCES.resolve()) and row['model_path'] in resources
            and sha(model) == row['model_sha256'] and h.fnv(model.read_bytes()) == row['model_fnv']
            and model.stat().st_size == row['model_bytes'], 'Modelo alterado')
        packed = h.params_bytes(row)
        require(len(packed) == 297 and packed.hex() == row['params_bytes_hex']
            and h.fnv(packed) == row['params_fnv'], 'Parámetros alterados')
        aggregate += struct.pack('<II', int(row['model_fnv'], 16), int(row['params_fnv'], 16))
    require(h.fnv(aggregate) == cat['catalog_fnv'], 'FNV de catálogo alterado')
    arrays = {}
    for key, name in [('raw', 'banco_raw.npz'), ('rawrefs', 'referencia_raw.npz'),
                      ('postrefs', 'referencia_post.npz')]:
        with np.load(BANKS/name, allow_pickle=False) as archive:
            arrays[key] = {k: archive[k].copy() for k in archive.files}
    require(arrays['raw']['windows'].shape == (56, 4096, 2)
        and arrays['raw']['windows'].dtype == np.float32, 'RAW incompleto')
    return h, cat, bank, arrays


def make_plan(cat, bank, postrefs):
    """Mecánico: primera RAW por carga y primera POST de cada etiqueta."""
    by_load = {}
    for row in bank['RAW_windows']:
        by_load.setdefault(int(row['metadata']['load_hp']), row)
    require(set(by_load) == {0, 1, 2, 3}, 'Faltan cargas RAW')
    first = by_load[0]
    first_ids = list(first['authorized_model_ids'])
    second = by_load[1]
    second_ids = [rid for rid in range(8) if rid not in first_ids]
    require(len(first_ids) == 6 and len(second_ids) == 2
        and set(second_ids) <= set(second['authorized_model_ids']), 'Cobertura RAW del gate inválida')
    raw = [[first['global_id'], first_ids], [second['global_id'], second_ids]]
    post = []
    for model, rep in zip(bank['models'], cat['representatives']):
        rid = rep['id']; prefix = model['reference_prefix']
        ids, y = postrefs[prefix+'global_window_ids'], postrefs[prefix+'y_true']
        require(ids.tolist() == model['post_global_window_ids'] and set(y.tolist()) == {0, 1, 2},
            'POST sin cobertura de tres etiquetas')
        for label in (0, 1, 2):
            position = int(np.flatnonzero(y == label)[0])
            post.append(dict(model_id=rid, global_id=int(ids[position]), label=label,
                reference_index=position))
    require(len(post) == 24, 'Gate debe tener 24 POST')
    return dict(mode='gate', RAW_schedule=raw, POST_cases=post,
        expected_RAW_primary_RUN=24, expected_RAW_PROFILE=8, expected_RAW_BASE=2,
        expected_POST=24, expected_POST_negative_rejections=3,
        RAW_coverage='partial development coverage; eight model-window pairs',
        POST_coverage='three mechanical labelled cases per model; not all reserved windows',
        selection_uses_accuracy_or_confidence=False, full_validation_enabled=False)


def authenticate_build(args, cat):
    firmware = Path(args.firmware_elf).resolve()
    build_path, audit_path = Path(args.build_report).resolve(), Path(args.elf_audit).resolve()
    build, audit = read(build_path), read(audit_path)
    require(sha(OUT/'MANIFEST_PREPARACION_PROYECTO_SHA256.json') == PREPARATION_SHA,
        'Preparación distinta')
    require(sha(firmware) == ELF_SHA == build['elf_sha256'] == audit['firmware_elf_sha256']
        and build['status'] == 'PASS_BUILD_PENDING_ELF_REVIEW_AND_ARM'
        and build['configuration'] == CONFIG and build['returncode'] == 0
        and build['zero_errors_and_warnings'] is True and build['integrity_after_build'] is True
        and build['D_source_unchanged_verified'] is True and not build['timed_out'], 'Build no autenticado')
    require(audit['status'] == 'PASS_ELF_CANON42_VERIFIED_PENDING_ARM'
        and audit['catalog_sha256'] == cat['catalog_sha256'] and audit['model_count'] == 8,
        'ELF no auditado para estos modelos')
    require([(r['id'], r['model_sha256'], r['params_fnv']) for r in audit['models']]
        == [(r['id'], r['model_sha256'], r['params_fnv']) for r in cat['representatives']],
        'Inventario ELF distinto del catálogo')
    return dict(firmware=firmware, build=build_path, audit=audit_path,
        build_report_sha256=sha(build_path), elf_audit_sha256=sha(audit_path))


def joint_manifest(args):
    name = 'MANIFEST_CLIENTES_CANON42_SHA256.json'
    path = OUT/name
    digest, filename = path.with_suffix('.sha256').read_text(encoding='ascii').split()
    require(filename == name and sha(path) == digest, 'Sello conjunto de clientes alterado')
    entries = read(path)
    required = [Path(__file__), HERE/HELPER_NAME if (HERE/HELPER_NAME).is_file() else LEGACY,
        Path(args.firmware_elf), Path(args.build_report), Path(args.elf_audit),
        RESOURCES/'MANIFEST_RECURSOS_SHA256.json', BANKS/'MANIFEST_BANCOS_SHA256.json']
    for item in required:
        require(item.resolve().is_relative_to(ROOT.resolve())
            and item.resolve().relative_to(ROOT.resolve()).as_posix() in entries,
            'Sello conjunto incompleto: '+str(item))
    for relative, expected in entries.items():
        target = (ROOT/relative).resolve()
        require(target.is_relative_to(ROOT.resolve()) and target.is_file() and sha(target) == expected,
            'Entrada conjunta alterada: '+relative)
    return digest


def parse_info(h, line, cat):
    f = line.split(',')
    require(len(f) == 17 and f[0] == 'CWRUTOTAL_INFO'
        and [h.dec(v) for v in f[1:8]] == [2, 8, 12000, 4096, 2, 180000000, 1],
        'INFO: protocolo, número de modelos, reloj o DWT distintos')
    require(not int(h.hext(f[8], 8), 16) & ((3 << 22) | (1 << 24)), 'FPSCR incompatible')
    require(h.hext(f[9], 8) == cat['catalog_fnv'] and h.hext(f[10], 64) == CATALOG_SHA
        and f[15:] == ['64', '0'], 'Catálogo o calibración distintos')
    selected, valid, last = h.signed(f[11], -1, 7), h.dec(f[12], 0, 1), h.dec(f[13])
    require(not valid or last > 0, 'Publicación sin ID')
    return dict(protocol=2, count=8, hclk_hz=180000000, DWT_enabled=True,
        fpscr=f[8].lower(), selected_id=selected, result_valid=bool(valid), last_run_id=last,
        paired_read_cycles_min=h.dec(f[14], 1), overhead_subtracted=False,
        nominal_HAL_clock_not_externally_measured=True)


def post_command(h, features):
    values = np.ascontiguousarray(features, dtype='<f4')
    require(values.shape == (14,) and np.isfinite(values).all(), 'POST requiere 14 binary32 finitos')
    payload = values.tobytes()
    return 'CWRUTOTAL_POST,'+h.fnv(payload)+','+payload.hex(), h.fnv(payload)


def parse_post_result(h, line, rep, checksum, last, arena):
    f = line.split(','); expected = 1 if last == 0xffffffff else last+1
    require(len(f) == 19 and f[0] == 'CWRUTOTAL_POST_RESULT'
        and h.dec(f[1], 1) == expected and h.dec(f[2], 0, 7) == rep['id']
        and h.hext(f[3], 8) == checksum and f[4] == '0', 'POST RESULT despareado')
    require(h.dec(f[5], 1) > 0 and f[6:11] == ['0']*5 and f[18] == '0',
        'POST no puede declarar extracción o costos RAW')
    output = [h.signed(v) for v in f[12:15]]; cls = h.dec(f[11], 0, 2)
    require(cls == int(np.argmax(output)) and h.dec(f[17], 1, 16384) == arena,
        'POST clase o arena inválida')
    score = h.hext(f[15], 8); require(h.from_bits([score])[0] >= 0, 'POST score inválido')
    return dict(run_id=expected, rep_id=rep['id'], checksum=checksum, mode='POST',
        post_cycles=h.dec(f[5], 1), output_int8=output, predicted_class=cls,
        score_bits=score, rejected=bool(h.dec(f[16], 0, 1)), arena_used=arena, extractions=0,
        timing_is_full_pipeline=False)


def reference(arrays, model, gid):
    prefix = model['reference_prefix']; ids = arrays[prefix+'global_window_ids']
    positions = np.flatnonzero(ids == gid)
    require(len(positions) == 1, 'Referencia ausente o duplicada')
    i = int(positions[0])
    return {key[len(prefix):]: value[i] for key, value in arrays.items() if key.startswith(prefix)}


def compare_intermediates(h, result, pages, rep, expected_features):
    expected = [h.bits(expected_features)[i] if i in rep['feature_indices'] else '00000000' for i in range(14)]
    require(pages['features_bits'] == expected, 'Features activas distintas o inactivas no +0')
    replay = h.replay(pages['features_bits'], rep)
    require(all(pages[k] == value for k, value in replay.items()), 'Intermedios distintos del replay float32')
    require(result['score_bits'] == pages['score_bits'] and result['rejected'] == pages['rejected'],
        'Publicación del detector despareada')
    return replay


def compare_post(h, result, pages, rep, ref):
    compare_intermediates(h, result, pages, rep, ref['features14'])
    require(pages['standardized_bits'] == h.bits(ref['normalized_features'])
        and pages['quantized'] == ref['int8_inputs'].astype(int).tolist()
        and result['output_int8'] == ref['int8_outputs'].astype(int).tolist()
        and result['predicted_class'] == int(ref['int8_predictions'])
        and result['score_bits'] == h.bits([ref['detector_scores']])[0]
        and result['rejected'] == bool(ref['detector_rejected']), 'POST distinto de referencia exterior guardada')
    return dict(saved_external_reference_exact=True, classifier_PC_invocations=0,
        RAW_extraction_validated=False, all_intermediates_exact=True)


class RawOracle:
    def __init__(self, h, reps):
        self.h, self.reps, self.lite, self.cache, self.invocations = h, reps, None, {}, 0

    def expected(self, rid, quantized, ref):
        if quantized == ref['int8_inputs'].astype(int).tolist():
            return ref['int8_outputs'].astype(int).tolist(), 'saved_same_INT8_input'
        key = (rid, tuple(quantized))
        if key not in self.cache:
            if self.lite is None:
                self.lite = self.h.LiteModels(RESOURCES, self.reps)
            self.cache[key] = self.lite.predict(rid, quantized); self.invocations += 1
        return self.cache[key], 'PC_replay_same_ARM_INT8_input'


def compare_raw(h, result, pages, base, rep, ref, oracle):
    compare_intermediates(h, result, pages, rep, h.from_bits(base['features_bits']))
    expected, source = oracle.expected(rep['id'], pages['quantized'], ref)
    require(result['output_int8'] == expected, 'Salida INT8 distinta para la MISMA entrada')
    return dict(same_input_INT8_output_exact=True, oracle=source,
        all_intermediates_exact=True, selected_features_exact_BASE=True,
        cached_PC_end_to_end_exact_claimed=False,
        cached_PC_differences=dict(features=sum(a != b for a, b in zip(base['features_bits'], h.bits(ref['features14']))),
            int8_values=sum(a != b for a, b in zip(pages['quantized'], ref['int8_inputs'].tolist())),
            outputs=sum(a != b for a, b in zip(result['output_int8'], ref['int8_outputs'].tolist())),
            class_differs=result['predicted_class'] != int(ref['int8_predictions']),
            detector_score_differs=result['score_bits'] != h.bits([ref['detector_scores']])[0],
            detector_rejection_differs=result['rejected'] != bool(ref['detector_rejected'])))


def negatives(h, link, result, features):
    command, checksum = post_command(h, features)
    payload = command.rsplit(',', 1)[1]
    bad = np.asarray(features, dtype='<f4').copy(); bad[0] = np.nan
    nan_payload = bad.tobytes()
    cases = [('CWRUTOTAL_POST', 'FORMAT', -1),
        ('CWRUTOTAL_POST,'+f'{int(checksum, 16)^1:08x}'+','+payload, 'CHECKSUM', -1),
        ('CWRUTOTAL_POST,'+h.fnv(nan_payload)+','+nan_payload.hex(), 'NONFINITE', 0)]
    rows = []
    for command, reason, detail in cases:
        response = link.ask(command, 'CWRUTOTAL_ERROR,', allow_error=True)
        require(response == f'CWRUTOTAL_ERROR,{reason},{detail}', 'Rechazo POST incorrecto')
        invalid = link.ask(f'CWRUTOTAL_FEATURES,{result["run_id"]}', 'CWRUTOTAL_ERROR,', allow_error=True)
        require(invalid == 'CWRUTOTAL_ERROR,NO_RESULT,-1', 'POST inválido conserva publicación anterior')
        rows.append(dict(command=command, response=response, old_result_invalidated=True))
    return rows


def prepare(args):
    h, cat, bank, arrays = load_resources(); provenance = authenticate_build(args, cat)
    plan = make_plan(cat, bank, arrays['postrefs'])
    require(not PACKAGE.exists(), 'No sobrescribir entrega del cliente')
    PACKAGE.mkdir()
    shutil.copy2(__file__, PACKAGE/'validar_canonicos_gate.py')
    shutil.copy2(LEGACY, PACKAGE/HELPER_NAME)
    write(PACKAGE/'PLAN_GATE.json', plan)
    write(PACKAGE/'PROCEDENCIA_CLIENTE.json', dict(status='CLIENT_READY_PENDING_USER_FLASH_AND_GATE',
        firmware_elf_sha256=ELF_SHA, build_report_sha256=provenance['build_report_sha256'],
        elf_audit_sha256=provenance['elf_audit_sha256'], catalog_sha256=CATALOG_SHA,
        gate_only=True, hardware_flashed=False, serial_opened=False, training_fits=0))
    entries = {p.name: sha(p) for p in PACKAGE.iterdir() if p.is_file()}
    mp = PACKAGE/'MANIFEST_CLIENTE_GATE_SHA256.json'; write(mp, entries)
    with mp.with_suffix('.sha256').open('x', encoding='ascii') as stream:
        stream.write(sha(mp)+'  '+mp.name+'\n')
    print(json.dumps(dict(status='CLIENT_READY_PENDING_USER_FLASH_AND_GATE', package=str(PACKAGE),
        POST_cases=24, RAW_pairs=8, serial_opened=False), ensure_ascii=True))


def check(args):
    joint_sha = joint_manifest(args)
    if HERE.resolve() == PACKAGE.resolve():
        manifest(PACKAGE, 'MANIFEST_CLIENTE_GATE_SHA256.json')
    h, cat, bank, arrays = load_resources()
    provenance = authenticate_build(args, cat)
    plan = make_plan(cat, bank, arrays['postrefs'])
    if HERE.resolve() == PACKAGE.resolve():
        require(plan == read(PACKAGE/'PLAN_GATE.json'), 'Plan entregado alterado')
    print(json.dumps(dict(status='CANON42_GATE_PREFLIGHT_PASS_NO_SERIAL_OPENED',
        POST_cases=len(plan['POST_cases']), RAW_pairs=sum(len(ids) for gid, ids in plan['RAW_schedule']),
        RAW_schedule=plan['RAW_schedule'], firmware_elf_sha256=ELF_SHA,
        elf_audit_sha256=provenance['elf_audit_sha256'], joint_client_manifest_sha256=joint_sha, serial_opened=False,
        hardware_flashed=False, full_validation_enabled=False), ensure_ascii=True))
    return plan


def run(args):
    started = time.monotonic()
    require(args.gate and not args.prepare and args.port, 'Se requiere --gate y --port')
    require(60 <= args.max_seconds <= 600 and 1 <= args.timeout <= 30, 'Presupuesto gate fuera de límites')
    require(HERE.resolve() == PACKAGE.resolve(), 'Ejecutar la copia entregada en Validacion_PC')
    joint_sha = joint_manifest(args)
    manifest(PACKAGE, 'MANIFEST_CLIENTE_GATE_SHA256.json')
    h, cat, bank, arrays = load_resources(); provenance = authenticate_build(args, cat)
    plan = make_plan(cat, bank, arrays['postrefs'])
    require(plan == read(PACKAGE/'PLAN_GATE.json'), 'El gate cambió respecto de la entrega')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output = PACKAGE/'resultados'/stamp; output.mkdir(parents=True, exist_ok=False)
    for path, name in [(provenance['firmware'], 'firmware_declarado.elf'),
        (provenance['build'], 'COMPILACION_CANON42.json'), (provenance['audit'], 'AUDITORIA_ELF_CANON42.json'),
        (Path(__file__), 'validar_canonicos_gate_ejecutado.py'), (HERE/HELPER_NAME, HELPER_NAME)]:
        shutil.copy2(path, output/name)
    write(output/'PLAN_EJECUTADO.json', dict(**plan, firmware_elf_sha256=ELF_SHA,
        resource_manifest_sha256=sha(RESOURCES/'MANIFEST_RECURSOS_SHA256.json'),
        bank_manifest_sha256=BANK_MANIFEST_SHA, max_seconds=args.max_seconds,
        timeout_per_command=args.timeout, python=sys.version, numpy_version=np.__version__,
        helper_sha256=HELPER_SHA, client_sha256=sha(__file__)))
    trace, events = h.Trace(output/'TRAZA_UART.jsonl'), h.Trace(output/'MEDICIONES.jsonl')
    report = dict(status='FAILED_INCOMPLETE', mode='gate', firmware_elf_sha256=ELF_SHA,
        catalog_sha256=CATALOG_SHA, bank_manifest_sha256=BANK_MANIFEST_SHA,
        client_sha256=sha(__file__), helper_sha256=HELPER_SHA,
        build_report_sha256=provenance['build_report_sha256'], elf_audit_sha256=provenance['elf_audit_sha256'],
        joint_client_manifest_sha256=joint_sha,
        completed_RAW_RUN=0, completed_RAW_PROFILE=0, completed_RAW_BASE=0, completed_POST=0,
        POST_negative_rejections=0, training_fits=0, full_validation_completed=False,
        RAW_exhaustive_reserved_coverage=False, POST_exhaustive_reserved_coverage=False,
        POST_timing_is_full_pipeline=False, selector_enabled=False,
        firmware_loaded_verified_by_UART=False,
        firmware_identity_limitation='Declared ELF audited; UART authenticates catalog and parameters, not all Flash bytes.',
        excluded_from_RAW_primary_cost=['UART', 'parser', 'checksums', 'SELECT', 'BASE', 'PROFILE', 'POST'],
        cached_PC_end_to_end_exactness_claimed=False)
    runs, bases, posts = [], [], []; oracle = RawOracle(h, cat['representatives'])
    print('Evidencias: '+str(output), flush=True)
    try:
        import serial
        with serial.Serial(args.port, args.baud, timeout=.10, write_timeout=args.timeout) as port:
            link = h.Link(port, trace, args.timeout, deadline=started+args.max_seconds)
            info = parse_info(h, link.ask('CWRUTOTAL_INFO', 'CWRUTOTAL_INFO,'), cat)
            require(link.ask('CWRUTOTAL_POST_INFO', 'CWRUTOTAL_POST_INFO,') == 'CWRUTOTAL_POST_INFO,1,14,56,0',
                'POST_INFO incompatible')
            report['firmware_info'] = info; last = info['last_run_id']
            for rid in range(8):
                h.validate_model(link.ask(f'CWRUTOTAL_MODEL,{rid}', 'CWRUTOTAL_MODEL,'), cat['representatives'][rid])
            by_id = {row['global_id']: row for row in bank['RAW_windows']}
            for gid, ids in plan['RAW_schedule']:
                row = dict(by_id[gid], arm_features14_bits=None)
                h.transmit(link, arrays['raw']['windows'][row['npz_index']], row)
                base = h.baseline(link, row); bases.append(base); report['completed_RAW_BASE'] += 1
                events.append(dict(type='BASE', **base))
                for rid in ids:
                    rep, model = cat['representatives'][rid], bank['models'][rid]
                    selection = h.parse_selected(link.ask(f'CWRUTOTAL_SELECT,{rid}', 'CWRUTOTAL_SELECTED,'), rep)
                    events.append(dict(type='SELECT', global_id=gid, **selection))
                    ref = reference(arrays['rawrefs'], model, gid); previous = None
                    for repetition, mode in [(1, 0), (2, 0), (3, 0), (1, 1)]:
                        command = 'CWRUTOTAL_PROFILE' if mode else 'CWRUTOTAL_RUN'
                        result = h.parse_result(link.ask(command+','+row['checksum_fnv1a'], 'CWRUTOTAL_RESULT,'),
                            rep, row, last, mode, selection['arena_used'])
                        last = result['run_id']; pages = h.get_pages(link, result, rep)
                        comparison = compare_raw(h, result, pages, base, rep, ref, oracle)
                        require(previous is None or pages == previous, 'RUN/PROFILE cambian los intermedios')
                        previous = pages
                        result.update(repetition=repetition, status='PASS', pages=pages, comparison=comparison)
                        runs.append(result); events.append(dict(type='PROFILE' if mode else 'RUN', **result))
                        report['completed_RAW_PROFILE' if mode else 'completed_RAW_RUN'] += 1
                    print(f'{rep["representative_id"]}/RAW gid{gid}: PASS; 3 RUN + PROFILE', flush=True)
            for case in plan['POST_cases']:
                rid, gid = case['model_id'], case['global_id']; rep = cat['representatives'][rid]
                ref = reference(arrays['postrefs'], bank['models'][rid], gid)
                selection = h.parse_selected(link.ask(f'CWRUTOTAL_SELECT,{rid}', 'CWRUTOTAL_SELECTED,'), rep)
                events.append(dict(type='SELECT_POST', global_id=gid, **selection))
                command, checksum = post_command(h, ref['features14'])
                result = parse_post_result(h, link.ask(command, 'CWRUTOTAL_POST_RESULT,'),
                    rep, checksum, last, selection['arena_used'])
                last = result['run_id']; pages = h.get_pages(link, result, rep)
                comparison = compare_post(h, result, pages, rep, ref)
                result.update(global_id=gid, label=case['label'], pages=pages, comparison=comparison, status='PASS')
                posts.append(result); events.append(dict(type='POST', **result)); report['completed_POST'] += 1
                print(f'{rep["representative_id"]}/POST gid{gid} etiqueta{case["label"]}: PASS', flush=True)
            rejected = negatives(h, link, posts[-1], ref['features14'])
            for item in rejected: events.append(dict(type='NEGATIVE_POST', **item))
            report['POST_negative_rejections'] = len(rejected)
            final = parse_info(h, link.ask('CWRUTOTAL_INFO', 'CWRUTOTAL_INFO,'), cat)
            require(final['last_run_id'] == last and not final['result_valid'], 'Rechazos alteran contador/publicación')
            report['firmware_info_final'] = final
        require([report[k] for k in ('completed_RAW_RUN', 'completed_RAW_PROFILE', 'completed_RAW_BASE',
            'completed_POST', 'POST_negative_rejections')] == [24, 8, 2, 24, 3], 'Gate incompleto')
        require(sha(provenance['firmware']) == ELF_SHA and sha(provenance['build']) == provenance['build_report_sha256']
            and sha(provenance['audit']) == provenance['elf_audit_sha256'], 'Procedencia cambió durante gate')
        require(joint_manifest(args) == joint_sha, 'Sello conjunto cambió durante gate')
        report['status'] = GATE_STATUS
    except BaseException as exc:
        report['error'] = dict(type=type(exc).__name__, message=str(exc)); raise
    finally:
        report.update(wall_seconds=time.monotonic()-started, RAW_timing_statistics=h.statistics(runs, cat['representatives']),
            PC_same_input_classifier_invocations=oracle.invocations, RAW_unique_window_ids=[r['global_id'] for r in bases],
            RAW_model_ids=sorted({r['rep_id'] for r in runs}), POST_model_ids=sorted({r['rep_id'] for r in posts}))
        trace.close(); events.close(); write(output/'RESULTADO_GATE.json', report)
        entries = {p.name: sha(p) for p in output.iterdir() if p.is_file()}
        mp = output/'MANIFEST_EVIDENCIAS_SHA256.json'; write(mp, entries)
        with mp.with_suffix('.sha256').open('x', encoding='ascii') as stream:
            stream.write(sha(mp)+'  '+mp.name+'\n')
        print(report['status']+'; evidencias: '+str(output), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare', action='store_true'); group.add_argument('--gate', action='store_true')
    group.add_argument('--check', action='store_true')
    parser.add_argument('--port'); parser.add_argument('--baud', type=int, default=115200)
    parser.add_argument('--timeout', type=float, default=15.)
    parser.add_argument('--max-seconds', type=int, default=600)
    parser.add_argument('--firmware-elf', type=Path, default=OUT/'copia_aislada/Prueba_2_CWRU_Extraccion'/CONFIG/'Prueba_2_CWRU_Extraccion.elf')
    parser.add_argument('--build-report', type=Path, default=OUT/'COMPILACION_CANON42.json')
    parser.add_argument('--elf-audit', type=Path, default=OUT/'AUDITORIA_ELF_CANON42.json')
    args = parser.parse_args()
    if args.prepare:
        prepare(args)
    elif args.check:
        check(args)
    else:
        run(args)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('INTERRUPTED', file=sys.stderr); raise SystemExit(130)
    except Exception as exc:
        print('ERROR: '+str(exc), file=sys.stderr); raise SystemExit(1)
