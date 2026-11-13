"""Audita logs y procedencia de una lectura Flash ya realizada; sin hardware.

No decide la validez de los bytes de relleno ELF: conserva el fallo original
y separa la adquisicion satisfactoria de la comparacion posterior.
"""
from __future__ import annotations
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re
import shlex
import time

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT/'despliegue_stm32/costo_total_single_rounding_v2_180mhz_2026-10-26'
READBACK = OUT/'lectura_flash/20261026T151832_154874Z'
CAMPAIGN = OUT/'Validacion_PC/resultados/20261026T144540_396879Z'
REPORT = OUT/'AUDITORIA_CAPTURA_FLASH_SR.json'
ELF_SHA = 'c0c3853d2b0e90bb20fe00be13bcf107d186088406be131b729a605ded44ecc7'
CLI_SHA = '269db404ed3f2cdd7cfcccc257ab0a907f2364c3d44e39edab6b007cb5f668ed'
SN = '0672FF505487525067162526'
NAME = 'Prueba_2_CWRU_Extraccion'
CONFIG = 'Release_180MHz_TotalV2_SR'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sealed(folder, filename):
    path = folder/filename
    digest, name = path.with_suffix('.sha256').read_text(encoding='ascii').split()
    require(name == path.name and sha(path) == digest, 'Sello invalido')
    entries = read(path)
    for relative, expected in entries.items():
        target = (folder/relative).resolve()
        require(target.is_relative_to(folder.resolve()) and target.is_file() and sha(target) == expected,
                'Captura original modificada: '+relative)
    return digest, entries


def main():
    started = time.perf_counter()
    require(not REPORT.exists(), 'No sobrescribir auditoria')
    dump_seal, dump_entries = sealed(READBACK, 'MANIFEST_LECTURA_FLASH_SHA256.json')
    campaign_seal, campaign_entries = sealed(CAMPAIGN, 'MANIFEST_EVIDENCIAS_SHA256.json')
    require(len(dump_entries) == 7 and len(campaign_entries) == 12, 'Conteo de recursos distinto')
    original = read(READBACK/'RESULTADO_LECTURA_FLASH_SR.json')
    plan = read(READBACK/'PLAN_LECTURA_FLASH_SR.json')
    require(original['status'] == 'FAILED_FLASH_READBACK'
            and original['comparison']['mismatch_count'] == 12 and not original['comparison']['match'],
            'El fallo original debe conservarse como evidencia')
    require(original['execution']['returncode'] == 0 and not original['execution']['timed_out']
            and not original['execution']['interrupted'], 'La adquisicion no termino correctamente')
    cli = Path(plan['CLI'])
    require(sha(cli) == CLI_SHA == plan['CLI_sha256'] == original['CLI_sha256'], 'CLI distinto')
    expected_argv = [str(cli), '-c', 'port=SWD', 'mode=HOTPLUG', 'freq=4000', 'sn='+SN,
        '-u','0x08000000','0x80000',str(READBACK/'flash_512KiB.bin'),'-log',str(READBACK/'programador.log')]
    require(original['command_argv'] == plan['command_argv'] == expected_argv, 'Comando no es lectura fija')
    require(not any(original[key] for key in ('Flash_programming_or_erasing','reset_requested',
        'serial_COM_opened','automatic_retry','selector_enabled','total_cost_promoted_here')),
        'Declaracion de acciones inesperada')
    stdout = (READBACK/'stdout_programador.txt').read_text(encoding='utf-8', errors='replace')
    detail = (READBACK/'programador.log').read_text(encoding='utf-8', errors='replace')
    require((READBACK/'stderr_programador.txt').stat().st_size == 0, 'stderr no vacio')
    require(re.search(r'STM32CubeProgrammer\s+v2\.23\.0\b', stdout)
            and re.search(r'ST-LINK\s+SN\s*:\s*'+SN+r'\b', detail)
            and 'Board       : NUCLEO-F446RE' in detail
            and re.search(r'Device ID\s*:\s*0x421\b', detail)
            and re.search(r'Device name\s*:\s*STM32F446', detail), 'Version/sonda/dispositivo distintos')
    command_lines = re.findall(r'Command line:\s*([^\r\n]+)', detail)
    require(len(command_lines) == 1, 'No existe un unico comando registrado')
    tokens = [token.strip('"') for token in shlex.split(command_lines[0], posix=False)]
    require([token.casefold() for token in tokens] == [token.casefold() for token in expected_argv[1:]],
            'El comando del log no corresponde al plan')
    require('Connect mode: Hot Plug' in stdout and 'Data read successfully' in stdout
            and 'Data read successfully' in detail and 'Disconnected' in detail
            and re.search(r'Size\s*:\s*524288 Bytes', detail)
            and re.search(r'r ap 0 @0x08000000 0x00080000 bytes', detail), 'Lectura completa no acreditada')
    require(not re.search(r'Erasing|Download in Progress|Memory Programming|File download complete', detail, re.I),
            'El log contiene operaciones diferentes de la lectura prevista')
    dump = READBACK/'flash_512KiB.bin'
    require(dump.stat().st_size == 524288 and sha(dump) == original['dump_sha256'], 'Dump incompleto o distinto')
    elf_paths = {
        'copied_readback': READBACK/'firmware_auditado_sr.elf',
        'declared_full_campaign': CAMPAIGN/'firmware_declarado.elf',
        'isolated_build': OUT/'copia_aislada'/NAME/CONFIG/(NAME+'.elf'),
        'archived_build': OUT/'intentos/1/compilacion'/CONFIG/(NAME+'.elf'),
        'installed_D': Path('D:/STM32IDE')/NAME/CONFIG/(NAME+'.elf'),
    }
    require(all(sha(path) == ELF_SHA for path in elf_paths.values()), 'Una copia ELF cambio')
    require(original['elf_sha256'] == plan['elf_sha256'] == ELF_SHA
            and original['source_campaign_manifest_sha256'] == plan['source_campaign_manifest_sha256'] == campaign_seal,
            'La lectura esta vinculada a otra campaña')
    lines = (CAMPAIGN/'TRAZA_UART.jsonl').read_text(encoding='utf-8').splitlines()
    first, last = json.loads(lines[0]), json.loads(lines[-1])
    campaign_end = datetime.fromisoformat(last['utc'])
    collection_start = datetime.fromisoformat(original['date_utc'])
    require(collection_start > campaign_end and last['seq'] == 45210, 'Cronologia no posterior a campaña completa')
    local = collection_start.astimezone(timezone(timedelta(hours=-5)))
    first_log_time = re.search(r'^(\d{2}:\d{2}:\d{2}):(\d{3})', detail)
    require(first_log_time, 'Falta hora del log')
    log_local = datetime.fromisoformat(local.date().isoformat()+'T'+first_log_time[1]+'.'+first_log_time[2]).replace(tzinfo=local.tzinfo)
    require(abs((log_local-local).total_seconds()) < 2, 'Hora de log no corresponde a UTC menos cinco')
    source_script = ROOT/'codigo/cwru/ampliacion_v2/stm32/verificar_flash_total_sr_v2.py'
    require(sha(source_script) == original['script_sha256'], 'Verificador original cambio')
    report = dict(status='PASS_SR_READBACK_CAPTURE_LOG_VERIFIED_COMPARISON_SEPARATE',
        date_local='2026-10-26', timezone='America/Lima', readback=READBACK.relative_to(ROOT).as_posix(),
        original_status_preserved='FAILED_FLASH_READBACK', original_mismatches_preserved=12,
        readback_manifest_sha256=dump_seal, readback_resources_verified=7,
        campaign_manifest_sha256=campaign_seal, campaign_resources_verified=12,
        CLI_version='2.23.0', CLI_sha256=CLI_SHA, fixed_read_command_exact_to_plan_and_log=True,
        STLINK_SN=SN, board='NUCLEO-F446RE', device_id='0x421', connect_mode='HOTPLUG',
        CLI_returncode=0, stderr_bytes=0, full_dump_bytes=524288, dump_sha256=sha(dump),
        CLI_process_seconds=original['execution']['seconds'], completed_device_read_seconds=4.360,
        read_success_confirmed=True, readback_after_campaign=True,
        campaign_first_UART_UTC=first['utc'], campaign_last_UART_UTC=last['utc'],
        readback_start_UTC=collection_start.isoformat(), readback_start_Lima=local.isoformat(),
        log_first_local_time=first_log_time[0], seconds_after_campaign_UART_end=(collection_start-campaign_end).total_seconds(),
        log_date_inferred_from_capture_UTC=True,
        reset_mode_text='Software reset', explicit_reset_command_requested=False,
        reset_mode_text_does_not_establish_a_reset_was_issued=True,
        five_ELF_copies_exact=True, elf_sha256=ELF_SHA,
        ELF_copy_paths={k:str(v) for k,v in elf_paths.items()},
        original_script_sha256=sha(source_script),
        script_sha256=sha(__file__),
        data_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in
            [READBACK/name for name in dump_entries]+[CAMPAIGN/'MANIFEST_EVIDENCIAS_SHA256.json',
             READBACK/'MANIFEST_LECTURA_FLASH_SHA256.json', source_script]},
        hardware_connected_by_audit=False, new_device_reads=0, new_physical_RUN=0,
        Flash_programming_or_erasing=False, reset_requested_by_audit=False, serial_COM_opened=False,
        training_fits=0, TFLite_invocations=0, builds_executed=0, D_files_modified=False,
        selector_enabled=False, cost_table_promoted=False,
        limitations=['Acquisition proof is separate from deciding which ELF bytes form the programmed image.',
            'This is a later readback, not a contemporaneous historical programming log.',
            'Reset mode is a reported connection setting; no reset was requested by the recorded CLI command.',
            'The later readback alone does not prove the exact program present at each historical campaign instant.'],
        seconds=time.perf_counter()-started)
    with REPORT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({k:report[k] for k in ('status','readback_manifest_sha256','five_ELF_copies_exact',
        'seconds_after_campaign_UART_end','seconds')}, ensure_ascii=True))


if __name__ == '__main__':
    main()
