"""Cierra identidad Flash por secciones/SREC y valida costos SR ya medidos.

Conserva el readback inicialmente FAILED y todos los sellos anteriores.
No ejecuta hardware, entrenamiento, TFLite, build ni seleccion predictiva.
"""
from __future__ import annotations
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT/'despliegue_stm32/costo_total_single_rounding_v2_180mhz_2026-10-26'
READBACK = OUT/'lectura_flash/20261026T151832_154874Z'
TABLE = OUT/'COSTOS_TOTALES_SR_VALIDADOS.json'
REPORT = OUT/'CIERRE_COSTOS_TOTALES_SR_VALIDADOS.json'
MANIFEST = OUT/'MANIFEST_COSTOS_TOTALES_SR_VALIDADOS_SHA256.json'
METHOD = ROOT/'articulo/ampliacion_cwru/metodologia/cierre_campana_total_sr_2026-10-26/VALIDACION_FINAL_IMAGEN_Y_COSTOS_SR.md'
ELF_SHA = 'c0c3853d2b0e90bb20fe00be13bcf107d186088406be131b729a605ded44ecc7'


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


def save(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def sealed(path, base, expected=None):
    digest, name = path.with_suffix('.sha256').read_text(encoding='ascii').split()
    require(name == path.name and sha(path) == digest and (expected is None or digest == expected),
            'Sello invalido: '+str(path))
    entries = read(path)
    for rel, declared in entries.items():
        target = (base/rel).resolve()
        require(target.is_relative_to(base.resolve()) and sha(target) == declared, 'Recurso modificado: '+rel)
    return digest, entries


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, 'Modulo no importable')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    started = time.perf_counter()
    require(not any(p.exists() for p in (TABLE,REPORT,MANIFEST,MANIFEST.with_suffix('.sha256'))), 'No sobrescribir cierre')
    prior_sha, prior_files = sealed(OUT/'MANIFEST_AUDITORIA_CAMPANA_SR_SHA256.json', ROOT,
        '2ce73b6e8e45c101c4a54ac736436d2413f07835bd9ad6b3f5914653c4ac21ea')
    read_sha, read_files = sealed(READBACK/'MANIFEST_LECTURA_FLASH_SHA256.json', READBACK,
        '1d886565e1cde57e1252b74d92cf7bf45258e654573ef01bf01398d81fd8d0f3')
    image_sha, image_files = sealed(OUT/'MANIFEST_IMAGEN_PROGRAMABLE_SR_SHA256.json', ROOT,
        'aa8e8bb6f901ba13f2df25942ae8d04155f4da2f167c1b6e03975b50da55345f')
    require(len(prior_files) == 76 and len(read_files) == 7, 'Cadena de evidencia incompleta')
    acquisition = read(OUT/'AUDITORIA_CAPTURA_FLASH_SR.json')
    section = read(OUT/'AUDITORIA_RELLENO_ELF_FLASH_SR.json')
    image = read(OUT/'AUDITORIA_IMAGEN_PROGRAMABLE_SR.json')
    tests = read(OUT/'PRUEBAS_COMPARACION_SECCIONES_FLASH_SR.json')
    original = read(READBACK/'RESULTADO_LECTURA_FLASH_SR.json')
    require(acquisition['status'] == 'PASS_SR_READBACK_CAPTURE_LOG_VERIFIED_COMPARISON_SEPARATE'
            and acquisition['readback_manifest_sha256'] == read_sha
            and acquisition['read_success_confirmed'] and acquisition['CLI_returncode'] == 0
            and acquisition['five_ELF_copies_exact'] and acquisition['readback_after_campaign'], 'Captura no autenticada')
    require(section['status'] == 'PASS_ALL_PROGRAMMABLE_SECTIONS_EXACT_ONLY_INTER_SECTION_PADDING_DIFFERS'
            and section['capture_manifest_sha256'] == read_sha and section['comparison']['match']
            and section['comparison']['compared_bytes'] == 280276 and section['comparison']['mismatch_count'] == 0
            and section['comparison']['numeric_tolerance'] == 0
            and not section['any_payload_tolerance_added'] and not section['hardcoded_ignored_address_exception'],
            'Comparacion estricta de secciones no aprobada')
    require(image['status'] == 'PASS_PROGRAMMABLE_SREC_IHEX_BYTES_EXACT_TO_EXISTING_READBACK'
            and image['readback_manifest_sha256'] == read_sha and image['expected_programmable_bytes'] == 280276
            and image['programmable_mismatches'] == 0 and image['excluded_alignment_bytes'] == 12
            and image['same_programmable_addresses_and_values_in_both_formats'], 'Conversion independiente no aprobada')
    require(all(row['elf_sha256'] == ELF_SHA for row in (acquisition,section,image,original)), 'ELF distinto')
    require(original['status'] == 'FAILED_FLASH_READBACK' and original['comparison']['mismatch_count'] == 12,
            'No preservar el fallo original alteraria el historial')
    sources = {
        'auditar_captura_flash_sr_v2.py':acquisition,
        'auditar_relleno_flash_total_sr_v2.py':section,
        'verificar_imagen_srec_sr_v2.py':image,
    }
    for name, row in sources.items():
        require(sha(Path(__file__).with_name(name)) == row['script_sha256'], 'Auditoria de otro codigo: '+name)
        for key in ('data_sha256','evidence_sha256'):
            for rel, expected in row.get(key, {}).items():
                require(sha(ROOT/rel) == expected, 'Dependencia cambiada: '+rel)
    comparator_path = Path(__file__).with_name('auditar_relleno_flash_total_sr_v2.py')
    test_path = Path(__file__).with_name('test_comparacion_secciones_flash_sr_v2.py')
    require(tests['status'] == 'PASS_STRICT_SECTION_PAYLOAD_GUARDS' and tests['tests'] == 5
            and tests['failures'] == tests['errors'] == 0 and tests['simulation_only']
            and tests['comparison_source_sha256'] == sha(comparator_path)
            and tests['test_source_sha256'] == sha(test_path), 'Guardas estrictas no comprobadas')
    # Revalidacion offline del payload; no se repite la lectura ni las pruebas.
    comparator = load(comparator_path, 'sr_final_section_comparator_readonly')
    reader = load(Path(__file__).with_name('auditar_elf_total_v2.py'), 'sr_final_elf_reader_readonly')
    elf = reader.ELF32(READBACK/'firmware_auditado_sr.elf')
    dump = (READBACK/'flash_512KiB.bin').read_bytes()
    sections = comparator.programmable_sections(elf)
    compared = comparator.compare_sections(dump, sections)
    require(compared == section['comparison'], 'Comparacion final de secciones diferente')
    formats = load(Path(__file__).with_name('verificar_imagen_srec_sr_v2.py'), 'sr_final_format_parsers_readonly')
    srec, _ = formats.parse_srec((OUT/'diagnostico_padding_flash/firmware_auditado_sr.srec').read_text(encoding='ascii'))
    ihex, _ = formats.parse_ihex((OUT/'diagnostico_padding_flash/firmware_auditado_sr.hex').read_text(encoding='ascii'))
    expected = {s['address']+i:value for s in sections for i,value in enumerate(s['data'])}
    require(srec == ihex == expected and all(dump[a-0x08000000] == v for a,v in expected.items()),
            'Secciones y formatos no especifican la misma imagen')
    prior = read(OUT/'COSTOS_TOTALES_SR_PROVISIONALES.json')
    require(prior['status'] == 'AUDITED_TOTAL_COST_PENDING_FLASH_IDENTITY' and len(prior['rows']) == 32,
            'Falta tabla provisional auditada')
    table = copy.deepcopy(prior)
    proof = dict(status='PROGRAMMABLE_FLASH_IMAGE_BYTE_EXACT_TO_AUDITED_ELF',
        readback=READBACK.relative_to(ROOT).as_posix(), readback_manifest_sha256=read_sha,
        readback_dump_sha256=sha(READBACK/'flash_512KiB.bin'), elf_sha256=ELF_SHA,
        compared_programmable_bytes=280276, mismatches=0, numeric_tolerance=0,
        section_count=7, independent_formats=['SREC','Intel_HEX'],
        inter_section_alignment_bytes_not_programmed=12,
        comparison_scope='all allocated file-backed Flash sections and matching emitted SREC/IHEX bytes',
        readback_is_after_campaign=True, historical_download_log_claimed=False,
        exact_historical_image_at_every_campaign_instant_proved_by_readback_alone=False,
        unused_Flash_or_dynamic_RAM_identity_claimed=False)
    for row in table['rows']:
        require(row['observations'] == 84 and 0 < row['lower_ms'] <= row['mean_ms'] <= row['upper_ms'],
                'Fila temporal no valida')
        row.update(status='VALIDATED_TOTAL_COST', Flash_readback_verified=True,
            firmware_readback_is_after_campaign=True, validated_as_cost_input=True,
            cost_selection_enabled=False, firmware_image_comparison_scope=proof['comparison_scope'])
    table['costs_by_outer'] = {str(i):{row['candidate']:row for row in table['rows'] if row['outer_load'] == i}
                               for i in range(4)}
    require(all(len(group) == 8 for group in table['costs_by_outer'].values()), 'Falta candidata o escenario')
    table.update(status='VALIDATED_TOTAL_COST', cost_table_validated=True,
        qualification='Audited representative full-pipeline costs with subsequent programmable-image readback verification.',
        firmware_identity_proof=proof, source_provisional_table_sha256=sha(OUT/'COSTOS_TOTALES_SR_PROVISIONALES.json'),
        cost_selection_enabled=False, predictive_selection_executed=False,
        full_inner_predictive_matrix_complete=False)
    save(TABLE,table)
    paths = {ROOT/rel for rel in prior_files} | {ROOT/rel for rel in image_files} | {READBACK/rel for rel in read_files}
    paths.update({TABLE,METHOD,Path(__file__).resolve(),test_path,
        OUT/'AUDITORIA_CAPTURA_FLASH_SR.json',OUT/'AUDITORIA_RELLENO_ELF_FLASH_SR.json',
        OUT/'AUDITORIA_IMAGEN_PROGRAMABLE_SR.json',OUT/'PRUEBAS_COMPARACION_SECCIONES_FLASH_SR.json'})
    paths.update(Path(__file__).with_name(name) for name in sources)
    for row in (acquisition,section):
        for key in ('data_sha256','evidence_sha256'):
            paths.update(ROOT/rel for rel in row.get(key,{}))
    for folder,name in ((OUT,'MANIFEST_AUDITORIA_CAMPANA_SR_SHA256.json'),
                        (OUT,'MANIFEST_IMAGEN_PROGRAMABLE_SR_SHA256.json'),
                        (READBACK,'MANIFEST_LECTURA_FLASH_SHA256.json')):
        paths.update({folder/name,(folder/name).with_suffix('.sha256')})
    require(all(p.is_file() and p.resolve().is_relative_to(ROOT.resolve()) for p in paths), 'Fuente ausente o externa')
    entries = {p.relative_to(ROOT).as_posix():sha(p) for p in sorted(paths)}
    closure = dict(status='TOTAL_COST_CALIBRATION_VALIDATED_PENDING_PREDICTIVE_SEARCH',
        date_local='2026-10-26', timezone='America/Lima', cost_table=TABLE.relative_to(ROOT).as_posix(),
        cost_table_sha256=sha(TABLE), validated_rows=32, primary_RUN=2688, PROFILE=896, BASE=42,
        observations_per_representative=84, firmware_identity_proof=proof,
        original_readback_status_preserved='FAILED_FLASH_READBACK',
        original_failure_classification='PT_LOAD inter-section padding checked as if it were programmed payload',
        original_failure_required_firmware_change=False, prior_seals_preserved=True,
        inherited_campaign_manifest_sha256=prior_sha, image_manifest_sha256=image_sha,
        five_payload_guard_tests_pass=True, cost_rule_changed=False,
        prediction_margins_changed=False, any_payload_tolerance_added=False,
        selector_enabled=False, predictive_selection_executed=False,
        next_step='Prepare remaining inner-fit executor, matching-signature reuse, and freeze execution budget before user starts training.',
        expected_remaining_inner_DNN_fits_if_32_reused=256,
        new_physical_readbacks=0, new_physical_RUN=0, training_fits=0,
        TFLite_invocations=0, builds=0, Flash_programming_or_erasing=False,
        serial_COM_opened=False, D_files_modified=False, manuscript_modified=False,
        new_objcopy_diagnostic_conversions=2, manifest_entries_including_closure=len(entries)+1,
        seconds_before_write=time.perf_counter()-started,
        limitations=['Readback establishes later programmable-image correspondence, not a contemporaneous download log.',
            'Calibration applies to the32 predefined seed11 representatives and their benchmark scope.',
            'Full predictive inner and outer multiseed evaluation remains pending.',
            'Empirical time envelopes are not confidence intervals or WCET guarantees.'])
    save(REPORT,closure)
    entries[REPORT.relative_to(ROOT).as_posix()] = sha(REPORT)
    save(MANIFEST,dict(sorted(entries.items())))
    with MANIFEST.with_suffix('.sha256').open('x',encoding='ascii') as stream:
        stream.write(sha(MANIFEST)+'  '+MANIFEST.name+'\n')
    for rel, expected_sha in entries.items():
        require(sha(ROOT/rel) == expected_sha, 'Cambio despues del cierre: '+rel)
    print(json.dumps(dict(status=closure['status'], validated_rows=32, files=len(entries),
        manifest_sha256=sha(MANIFEST),seconds=time.perf_counter()-started),ensure_ascii=True))


if __name__ == '__main__':
    main()
