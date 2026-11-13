"""Sella el verificador de lectura Flash probado; no conecta con la placa."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT/'despliegue_stm32/costo_total_single_rounding_v2_180mhz_2026-10-26'
SOURCE = ROOT/'codigo/cwru/ampliacion_v2/stm32/verificar_flash_total_sr_v2.py'
TEST = SOURCE.with_name('test_verificar_flash_total_sr_v2.py')
REPORT = OUT/'CIERRE_PREPARACION_LECTURA_FLASH_SR.json'
MANIFEST = OUT/'MANIFEST_PREPARACION_LECTURA_FLASH_SR_SHA256.json'


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


def main():
    require(not any(p.exists() for p in (REPORT, MANIFEST, MANIFEST.with_suffix('.sha256'))), 'No sobrescribir cierre')
    tests = read(OUT/'PRUEBAS_VERIFICADOR_FLASH_SR.json')
    preflight = read(OUT/'PREFLIGHT_VERIFICADOR_FLASH_SR.json')
    help_report = read(OUT/'AYUDA_PROGRAMADOR_OFFLINE_SR.json')
    require(tests['status'] == 'PASS_SR_FLASH_VERIFIER_SYNTHETIC_TESTS'
            and tests['tests'] == 6 and tests['failures'] == tests['errors'] == tests['exitcode'] == 0
            and tests['source_sha256'] == sha(SOURCE) and tests['test_source_sha256'] == sha(TEST)
            and tests['real_device_connections'] == 0, 'Pruebas no corresponden al codigo actual')
    require(preflight['status'] == 'PREFLIGHT_PASS_READY_FOR_USER_READBACK'
            and preflight['verifier_sha256'] == sha(SOURCE)
            and not preflight['hardware_connected'] and not preflight['readback_executed']
            and preflight['compared_ELF_bytes'] == 280288 and preflight['read_size_bytes'] == 524288,
            'Preflight de otro codigo o alcance')
    require(help_report['status'] == 'PASS_LOCAL_OFFLINE_HELP_NO_DEVICE'
            and help_report['returncode'] == 0 and not help_report['hardware_connected']
            and sha(OUT/'AYUDA_PROGRAMADOR_OFFLINE_SR.txt') == help_report['stdout_sha256']
            and sha(Path(preflight['CLI'])) == preflight['CLI_sha256'] == help_report['CLI_sha256'],
            'Ayuda CLI o ejecutable distintos')
    argv = preflight['command_template_only_not_executed']
    require(argv[1:9] == ['-c','port=SWD','mode=HOTPLUG','freq=4000',
                         'sn=0672FF505487525067162526','-u','0x08000000','0x80000']
            and argv[10] == '-log' and len(argv) == 12, 'Comando distinto de lectura fija')
    paths = {Path(__file__).resolve(), SOURCE, TEST,
        OUT/'PRUEBAS_VERIFICADOR_FLASH_SR.json',OUT/'PREFLIGHT_VERIFICADOR_FLASH_SR.json',
        OUT/'AYUDA_PROGRAMADOR_OFFLINE_SR.json',OUT/'AYUDA_PROGRAMADOR_OFFLINE_SR.txt',
        OUT/'VERIFICACION_FLASH_SR.md',OUT/'CIERRE_AUDITORIA_CAMPANA_SR.json',
        OUT/'MANIFEST_AUDITORIA_CAMPANA_SR_SHA256.json',OUT/'MANIFEST_AUDITORIA_CAMPANA_SR_SHA256.sha256',
        ROOT/'codigo/cwru/ampliacion_v2/analisis/auditar_elf_total_v2.py'}
    require(all(p.is_file() and p.resolve().is_relative_to(ROOT.resolve()) for p in paths), 'Archivo ausente o externo')
    entries = {p.relative_to(ROOT).as_posix():sha(p) for p in sorted(paths)}
    report = dict(status='FLASH_READBACK_VERIFIER_PREPARED_PENDING_USER_EXECUTION',
        date_local='2026-10-26', timezone='America/Lima', tests=6, preflight_pass=True,
        source_sha256=sha(SOURCE), source_campaign_manifest_sha256=preflight['source_campaign_manifest_sha256'],
        command_is_fixed_read_only=True, budget_seconds=120, read_bytes=524288,
        compare_ELF_load_bytes=280288, verification_is_after_campaign=True,
        contemporaneous_programming_log_claimed=False, hardware_connected=False,
        Flash_programming_or_erasing=False, serial_COM_opened=False, new_physical_RUN=0,
        training_fits=0, selector_enabled=False, costs_promoted=False,
        original_capture_or_prior_seals_modified=False, manifest_entries_including_closure=len(entries)+1)
    save(REPORT, report)
    entries[REPORT.relative_to(ROOT).as_posix()] = sha(REPORT)
    save(MANIFEST, dict(sorted(entries.items())))
    with MANIFEST.with_suffix('.sha256').open('x', encoding='ascii') as stream:
        stream.write(sha(MANIFEST)+'  '+MANIFEST.name+'\n')
    for name, expected in entries.items():
        require(sha(ROOT/name) == expected, 'Cambio durante el cierre: '+name)
    print(json.dumps(dict(status=report['status'], files=len(entries), manifest_sha256=sha(MANIFEST)), ensure_ascii=True))


if __name__ == '__main__':
    main()
