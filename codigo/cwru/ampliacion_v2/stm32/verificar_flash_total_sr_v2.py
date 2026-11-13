"""Lee Flash por SWD y la contrasta con el ELF SR; nunca programa ni borra.

--preflight solo verifica archivos y muestra el plan, sin conectar hardware.
--readback lo inicia personalmente el usuario, despues de terminar el debugger.
La lectura es posterior a la campaña; no fabrica un log historico de carga.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT/'despliegue_stm32/costo_total_single_rounding_v2_180mhz_2026-10-26'
CAMPAIGN = OUT/'Validacion_PC/resultados/20261026T144540_396879Z'
PROJECT = 'Prueba_2_CWRU_Extraccion'
CONFIG = 'Release_180MHz_TotalV2_SR'
INSTALLED_ELF = Path('D:/STM32IDE')/PROJECT/CONFIG/(PROJECT+'.elf')
CLI = Path('D:/STM32CubeIDE_2.2.0/STM32CubeIDE/plugins/com.st.stm32cube.ide.mcu.externaltools.cubeprogrammer.win32_2.2.500.202603051304/tools/bin/STM32_Programmer_CLI.exe')
CLI_SHA = '269db404ed3f2cdd7cfcccc257ab0a907f2364c3d44e39edab6b007cb5f668ed'
CLI_VERSION = '2.23.0'
ELF_SHA = 'c0c3853d2b0e90bb20fe00be13bcf107d186088406be131b729a605ded44ecc7'
CAMPAIGN_SEAL = 'df72445555afa114ec4974fe931de60970164308e07c24b3935cf47ed12439b2'
READER = ROOT/'codigo/cwru/ampliacion_v2/analisis/auditar_elf_total_v2.py'
READER_SHA = '67d6c41c45ab3afb21d568fa7147fe80cc6d62670bcaec6ccf7731710cdfbf0b'
STLINK_SN = '0672FF505487525067162526'
FLASH_START = 0x08000000
FLASH_SIZE = 0x80000


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


def checked_capture(folder):
    folder = Path(folder).resolve()
    base = (OUT/'lectura_flash').resolve()
    require(OUT.resolve().is_relative_to(ROOT.resolve()) and base.is_relative_to(OUT.resolve())
            and folder.parent == base and re.fullmatch(r'\d{8}T\d{6}_\d{6}Z', folder.name),
            'La captura debe ser un directorio UTC nuevo bajo OUT/lectura_flash')
    return folder


def readback_command(folder):
    folder = checked_capture(folder)
    # Esta lista es fija: no acepta argumentos adicionales del usuario.
    return [str(CLI), '-c', 'port=SWD', 'mode=HOTPLUG', 'freq=4000', 'sn='+STLINK_SN,
            '-u', '0x08000000', '0x80000', str(folder/'flash_512KiB.bin'),
            '-log', str(folder/'programador.log')]


def flash_segments(elf):
    result = []
    for segment in elf.segments:
        if segment['type'] != 1 or not segment['size']:
            continue
        start, size = segment['physical'], segment['size']
        end = start+size
        overlaps = start < FLASH_START+FLASH_SIZE and end > FLASH_START
        require(not overlaps or FLASH_START <= start < end <= FLASH_START+FLASH_SIZE,
                'Segmento ELF cruza el limite Flash')
        if overlaps:
            result.append(dict(address=start, size=size,
                               virtual_address=segment['address'],
                               data=elf.range(segment['offset'], size)))
    result.sort(key=lambda row: row['address'])
    require(result and all(a['address']+a['size'] <= b['address']
                           for a, b in zip(result, result[1:])), 'Segmentos vacios o superpuestos')
    return result


def compare_dump(data, segments):
    require(len(data) == FLASH_SIZE, 'La lectura debe contener exactamente512KiB')
    reports = []
    mismatches = []
    mismatch_count = 0
    for segment in segments:
        offset = segment['address']-FLASH_START
        actual = data[offset:offset+segment['size']]
        expected = segment['data']
        require(len(expected) == segment['size'] == len(actual), 'Rango de comparacion incompleto')
        positions = [i for i, (a, b) in enumerate(zip(actual, expected)) if a != b]
        mismatch_count += len(positions)
        mismatches.extend(dict(address=f'0x{segment["address"]+i:08x}',
            expected=expected[i], actual=actual[i]) for i in positions[:max(0, 32-len(mismatches))])
        reports.append(dict(address=f'0x{segment["address"]:08x}',
            virtual_address=f'0x{segment["virtual_address"]:08x}', bytes=segment['size'],
            expected_sha256=hashlib.sha256(expected).hexdigest(),
            actual_sha256=hashlib.sha256(actual).hexdigest(), mismatch_count=len(positions)))
    compared = sum(s['size'] for s in segments)
    return dict(match=mismatch_count == 0, loaded_segments=reports, compared_bytes=compared,
        mismatch_count=mismatch_count, first_mismatches=mismatches,
        read_bytes=len(data), bytes_not_specified_by_ELF=len(data)-compared,
        gaps_and_tail_required_to_equal_FF=False, BSS_or_stack_RAM_compared=False)


def check_programmer_log(stdout, detail):
    require(detail.strip(), 'Falta el log detallado solicitado al programador')
    text = stdout+'\n'+detail
    require(re.search(r'STM32CubeProgrammer\s+v'+re.escape(CLI_VERSION)+r'\b', text), 'Version del programador distinta')
    require(re.search(r'ST-LINK\s+SN\s*:\s*'+STLINK_SN+r'\b', text), 'ST-LINK distinto del autorizado')
    require(re.search(r'Board\s*:\s*NUCLEO-F446RE\b', text, re.I)
            and re.search(r'Device\s+ID\s*:\s*0x421\b', text, re.I)
            and re.search(r'Device\s+name\s*:\s*STM32F446', text, re.I), 'Dispositivo distinto de Nucleo-F446RE')
    require(re.search(r'(?:Data|Memory)\s+read\s+successfully|Upload(?:ing)?[^\n]*(?:complete|success)', text, re.I),
            'El programador no confirma lectura completada')
    return dict(programmer_version=CLI_VERSION, STLINK_SN=STLINK_SN,
                board='NUCLEO-F446RE', device_id='0x421', read_success_confirmed=True)


def run_bounded(argv, timeout):
    require(10 <= timeout <= 120, 'Presupuesto permitido:10a120s')
    started = time.monotonic()
    process = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    timed_out = interrupted = False
    try:
        stdout, stderr = process.communicate(timeout=timeout-5)
    except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
        timed_out = isinstance(error, subprocess.TimeoutExpired)
        interrupted = isinstance(error, KeyboardInterrupt)
        process.terminate()
        try:
            stdout, stderr = process.communicate(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate(timeout=2)
    return dict(returncode=process.returncode, timed_out=timed_out, interrupted=interrupted,
                stdout=stdout, stderr=stderr, seconds=time.monotonic()-started)


def preflight():
    require(ROOT.resolve() == Path('C:/Users/LENOVO/Desktop/Proyecto Investigación').resolve(), 'Workspace inesperado')
    require(sha(CLI) == CLI_SHA and sha(READER) == READER_SHA, 'CLI o lector ELF distintos de los revisados')
    manifest = CAMPAIGN/'MANIFEST_EVIDENCIAS_SHA256.json'
    digest, name = manifest.with_suffix('.sha256').read_text(encoding='ascii').split()
    require(name == manifest.name and digest == sha(manifest) == CAMPAIGN_SEAL, 'Sello de campaña distinto')
    entries = read(manifest)
    require(len(entries) == 12, 'Campaña sellada incompleta')
    for relative, expected in entries.items():
        target = (CAMPAIGN/relative).resolve()
        require(target.is_relative_to(CAMPAIGN.resolve()) and sha(target) == expected, 'Captura alterada: '+relative)
    campaign = read(CAMPAIGN/'RESULTADO_TOTAL.json')
    require(campaign['status'] == 'FULL_ARM_MEASUREMENTS_READY_FOR_AUDIT'
            and (campaign['completed_primary_RUN'], campaign['completed_PROFILE'], campaign['completed_BASE'],
                 campaign['negative_rejections']) == (2688, 896, 42, 17), 'Campaña completa no disponible')
    original = CAMPAIGN/'firmware_declarado.elf'
    require(sha(original) == sha(INSTALLED_ELF) == campaign['firmware_elf_sha256'] == ELF_SHA,
            'ELF original o instalado distinto del auditado')
    spec = importlib.util.spec_from_file_location('sr_flash_readonly_elf_reader', READER)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    segments = flash_segments(helper.ELF32(original))
    require(len(segments) == 2 and sum(s['size'] for s in segments) == 280288,
            'Segmentos Flash distintos del ELF SR auditado')
    info = dict(status='PREFLIGHT_PASS_READY_FOR_USER_READBACK', CLI=str(CLI), CLI_sha256=CLI_SHA,
        CLI_version_help_verified=CLI_VERSION, elf_sha256=ELF_SHA,
        source_campaign=str(CAMPAIGN), source_campaign_manifest_sha256=digest,
        compared_ELF_bytes=280288, read_size_bytes=FLASH_SIZE, device_id_required='0x421',
        STLINK_SN=STLINK_SN, original_capture_modified=False, hardware_connected=False,
        readback_executed=False, Flash_programming_or_erasing=False)
    return info, segments


def readback(timeout):
    info, segments = preflight()
    folder = checked_capture(OUT/'lectura_flash'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    folder.mkdir(parents=True, exist_ok=False)
    argv = readback_command(folder)
    report = dict(status='FAILED_FLASH_READBACK', date_utc=datetime.now(timezone.utc).isoformat(),
        configuration=CONFIG, elf_sha256=ELF_SHA, command_argv=argv, requested_budget_seconds=timeout,
        active_process_timeout_seconds=timeout-5, termination_grace_seconds=5,
        script_sha256=sha(__file__), reader_sha256=READER_SHA, CLI_sha256=CLI_SHA,
        source_campaign=str(CAMPAIGN), source_campaign_manifest_sha256=CAMPAIGN_SEAL,
        verification_is_after_campaign=True, contemporaneous_programming_log_claimed=False,
        readback_proves_campaign_time_image_by_itself=False,
        full_Flash_dump_bytes=FLASH_SIZE, comparison_scope='ELF PT_LOAD physical Flash bytes, including initialized RAM load image',
        Flash_programming_or_erasing=False, reset_requested=False, serial_COM_opened=False,
        automatic_retry=False, selector_enabled=False, total_cost_promoted_here=False,
        unused_Flash_bytes_content_known_from_ELF=False)
    save(folder/'PLAN_LECTURA_FLASH_SR.json', dict(**info, command_argv=argv, budget_seconds=timeout))
    copied = folder/'firmware_auditado_sr.elf'
    with copied.open('xb') as stream:
        stream.write((CAMPAIGN/'firmware_declarado.elf').read_bytes())
    require(sha(copied) == ELF_SHA, 'Copia ELF distinta')
    print('Evidencias: '+str(folder), flush=True)
    exitcode = 1
    try:
        result = run_bounded(argv, timeout)
        for filename, data in (('stdout_programador.txt', result.pop('stdout')),
                               ('stderr_programador.txt', result.pop('stderr'))):
            with (folder/filename).open('xb') as stream:
                stream.write(data)
        report['execution'] = result
        require(not result['timed_out'] and not result['interrupted'] and result['returncode'] == 0,
                'El programador termino con fallo, interrupcion o timeout; captura conservada')
        dump, log = folder/'flash_512KiB.bin', folder/'programador.log'
        report['device'] = check_programmer_log((folder/'stdout_programador.txt').read_text(encoding='utf-8', errors='replace'),
                                               log.read_text(encoding='utf-8', errors='replace'))
        report['dump_sha256'] = sha(dump)
        report['comparison'] = compare_dump(dump.read_bytes(), segments)
        require(report['comparison']['match'], 'La Flash leida no coincide con el ELF SR auditado')
        require(sha(INSTALLED_ELF) == sha(copied) == ELF_SHA, 'ELF cambio durante lectura')
        report['status'] = 'PASS_FLASH_READBACK_MATCHES_AUDITED_SR_ELF'
        exitcode = 0
    except BaseException as error:
        report['error'] = dict(type=type(error).__name__, message=str(error))
    finally:
        save(folder/'RESULTADO_LECTURA_FLASH_SR.json', report)
        files = {p.name: sha(p) for p in sorted(folder.iterdir()) if p.is_file()}
        manifest = folder/'MANIFEST_LECTURA_FLASH_SHA256.json'
        save(manifest, files)
        with manifest.with_suffix('.sha256').open('x', encoding='ascii') as stream:
            stream.write(sha(manifest)+'  '+manifest.name+'\n')
        print(report['status']+'; evidencias: '+str(folder), flush=True)
    return exitcode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--preflight', action='store_true')
    mode.add_argument('--readback', action='store_true')
    parser.add_argument('--timeout', type=int, default=120)
    args = parser.parse_args()
    require(10 <= args.timeout <= 120, '--timeout debe estar entre10y120s')
    if args.preflight:
        info, _ = preflight()
        planned = OUT/'lectura_flash'/'20990101T000000_000000Z'
        info['command_template_only_not_executed'] = readback_command(planned)
        print(json.dumps(info, ensure_ascii=True, indent=2))
        return 0
    return readback(args.timeout)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as error:
        print('ERROR: '+str(error), file=sys.stderr)
        raise SystemExit(1)
