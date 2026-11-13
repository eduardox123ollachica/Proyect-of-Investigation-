"""Carga explícita por el usuario del ELF Canon42 auditado; --check es offline.

No modifica D:, compila o abre COM. --program usa un único intento de carga
SWD y un presupuesto global de 120 s, precedido por lectura de identidad.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time


def find_root():
    marker = Path('codigo/cwru/ampliacion_v2/canonicos_stm32/preparar_proyecto.py')
    for parent in Path(__file__).resolve().parents:
        if (parent/marker).is_file():
            return parent
    raise ValueError('No se encuentra la raíz del proyecto')


ROOT = find_root()
OUT = ROOT/'despliegue_stm32/canonicos_42_180mhz_2026-10-27'
CONFIG = 'Release_180MHz_Canon42_SR'
ELF = OUT/'copia_aislada/Prueba_2_CWRU_Extraccion'/CONFIG/'Prueba_2_CWRU_Extraccion.elf'
MAP = ELF.with_suffix('.map')
AUDIT = OUT/'AUDITORIA_ELF_CANON42.json'
BUILD = OUT/'COMPILACION_CANON42.json'
MANIFEST = OUT/'MANIFEST_CLIENTES_CANON42_SHA256.json'
MARKER = OUT/'INTENTO_CARGA_CANON42.json'
CLI = Path('D:/STM32CubeIDE_2.2.0/STM32CubeIDE/plugins/'
    'com.st.stm32cube.ide.mcu.externaltools.cubeprogrammer.win32_2.2.500.202603051304/'
    'tools/bin/STM32_Programmer_CLI.exe')
CLI_SHA = '269db404ed3f2cdd7cfcccc257ab0a907f2364c3d44e39edab6b007cb5f668ed'
ELF_SHA = 'c2de62a6e73a7f0f4cc2f4d25f5cbef718bde28eb93e05d0273871337ce84a46'
BUILD_SHA = '17044183489abacb3fa5098366e2bdc8cef495e8fc4eb48700f0f71badf3b860'
CATALOG_SHA = 'e7a0c848ccb293c3dd95c9e150e8e6909f6ae6b82c3ec90e6332faebbc2a6fdd'
PROBE_SN = '0672FF505487525067162526'
MAX_SECONDS = 120


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


def relative(path):
    path = Path(path).resolve()
    require(path.is_relative_to(ROOT.resolve()), 'Archivo fuera del proyecto')
    return path.relative_to(ROOT.resolve()).as_posix()


def verify_manifest():
    digest, name = MANIFEST.with_suffix('.sha256').read_text(encoding='ascii').split()
    require(name == MANIFEST.name and digest == sha(MANIFEST), 'Sello de clientes modificado')
    entries = read(MANIFEST)
    required = {relative(p) for p in (Path(__file__), AUDIT, BUILD, ELF, MAP,
        OUT/'MANIFEST_PREPARACION_PROYECTO_SHA256.json',
        OUT/'recursos/MANIFEST_RECURSOS_SHA256.json', OUT/'bancos/MANIFEST_BANCOS_SHA256.json')}
    require(isinstance(entries, dict) and required <= set(entries), 'Sello de clientes incompleto')
    for name, expected in entries.items():
        rel = Path(name)
        require(not rel.is_absolute() and rel.parts and '..' not in rel.parts, 'Ruta de manifiesto insegura')
        target = (ROOT/rel).resolve()
        require(target.is_relative_to(ROOT.resolve()) and target != ROOT.resolve(), 'Ruta fuera del proyecto')
        require(sha(target) == expected, 'Archivo sellado modificado: '+name)
    return digest


def check():
    """Solo archivos: no crea carpetas/registros ni ejecuta herramientas."""
    manifest_sha = verify_manifest()
    require(sha(CLI) == CLI_SHA, 'Programador instalado distinto del verificado')
    require(sha(ELF) == ELF_SHA, 'ELF distinto del auditado')
    require(sha(BUILD) == BUILD_SHA, 'Informe de compilación distinto del verificado')
    build, audit = read(BUILD), read(AUDIT)
    require(build['status'] == 'PASS_BUILD_PENDING_ELF_REVIEW_AND_ARM'
        and build['returncode'] == 0 and not build['timed_out']
        and build['zero_errors_and_warnings'] and build['integrity_after_build']
        and build['D_source_unchanged_verified'] and build['elf_sha256'] == ELF_SHA
        and build['map_sha256'] == sha(MAP), 'Compilación no aprobada o alterada')
    require(audit['status'] == 'PASS_ELF_CANON42_VERIFIED_PENDING_ARM'
        and audit['firmware_elf_sha256'] == ELF_SHA and audit['build_report_sha256'] == BUILD_SHA
        and audit['configuration'] == CONFIG and audit['catalog_sha256'] == CATALOG_SHA
        and audit['model_count'] == 8 and audit['zero_errors_and_warnings']
        and audit['source_integrity_after_build'], 'Auditoría ELF no aprobada o distinta')
    return dict(status='CANON42_OFFLINE_CHECK_PASS_NOT_PROGRAMMED',
        manifest_sha256=manifest_sha, audit_sha256=sha(AUDIT), ELF_sha256=ELF_SHA,
        ELF_path=str(ELF), CLI_path=str(CLI), CLI_sha256=CLI_SHA,
        fixed_STLINK_serial=PROBE_SN, expected_device_id='0x421',
        expected_device='STM32F446xx', expected_board='NUCLEO-F446RE',
        hardware_budget_seconds=MAX_SECONDS, connection_mode='SWD Under Reset / hardware reset',
        automatic_retry=False, COM_opened=False, hardware_contacted=False,
        hardware_flashed=False, D_files_modified=False)


def connection_command(logpath):
    return [str(CLI), '-c', 'port=SWD', 'sn='+PROBE_SN, 'mode=UR', 'reset=HWrst', '-log', str(logpath)]


def program_command(logpath):
    return [str(CLI), '-c', 'port=SWD', 'sn='+PROBE_SN, 'mode=UR', 'reset=HWrst',
        '-d', str(ELF), '-v', '-rst', '-log', str(logpath)]


def identity(text):
    serials = re.findall(r'ST[- ]LINK\s+SN\s*:\s*(\S+)', text, re.I)
    device_ids = re.findall(r'Device\s+ID\s*:\s*(0x[0-9a-f]+)', text, re.I)
    devices = re.findall(r'Device\s+name\s*:\s*([^\r\n]+)', text, re.I)
    boards = re.findall(r'\bBoard\s*:\s*([^\r\n]+)', text, re.I)
    capacities = re.findall(r'(?:NVM|Flash)\s+size\s*:\s*(\d+)\s*K(?:Bytes?|B)\b', text, re.I)
    normal_board = lambda value: re.sub(r'[^A-Z0-9]', '', value.upper())
    valid = bool(serials and device_ids and devices)
    valid = valid and all(value == PROBE_SN for value in serials)
    valid = valid and all(value.lower() == '0x421' for value in device_ids)
    valid = valid and all(re.fullmatch(r'STM32F446(?:xx|RE(?:Tx)?)', value.strip(), re.I) for value in devices)
    valid = valid and all(normal_board(value) == 'NUCLEOF446RE' for value in boards)
    valid = valid and all(int(value) == 512 for value in capacities)
    return dict(approved=bool(valid), STLINK_serials=serials, device_ids=device_ids,
        device_names=[value.strip() for value in devices], board_names=[value.strip() for value in boards],
        capacity_KiB=[int(value) for value in capacities],
        board_name_explicitly_reported=bool(boards), capacity_explicitly_reported=bool(capacities))


def run_cli(command, transcript, timeout):
    """Una llamada; sin reintentos. El hijo queda oculto y se cierra al timeout."""
    require(timeout > 0, 'Presupuesto agotado antes de iniciar el programador')
    started = time.monotonic()
    process = None
    raw = b''
    code, error, timed_out = None, None, False
    cleanup_error = None
    def stop():
        nonlocal code, raw, cleanup_error
        if process is None:
            return
        try:
            if process.poll() is None:
                process.kill()
            raw, _ = process.communicate(timeout=5)
            code = process.returncode
        except Exception as exc:
            cleanup_error = type(exc).__name__+': '+str(exc)
    try:
        process = subprocess.Popen(command, cwd=OUT, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        raw, _ = process.communicate(timeout=timeout)
        code = process.returncode
    except subprocess.TimeoutExpired:
        timed_out, error = True, 'TIMEOUT'
        stop()
    except KeyboardInterrupt:
        error = 'INTERRUPTED'
        stop()
    except Exception as exc:
        error = type(exc).__name__+': '+str(exc)
        stop()
    text = raw.decode('utf-8', errors='replace')
    with Path(transcript).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(text)
    return dict(command=command, process_created=process is not None, returncode=code,
        timed_out=timed_out, error=error, cleanup_error=cleanup_error,
        seconds=time.monotonic()-started, text=text)


def command_ok(result):
    return (result['returncode'] == 0 and result['error'] is None and not result['timed_out']
        and not re.search(r'(?im)^\s*(?:Error\s*:|.*\bfailed\b)', result['text']))


def seal_evidence(folder):
    """Sello local posterior al resultado, sin incluirse a sí mismo."""
    folder = Path(folder).resolve()
    require(folder.is_relative_to(OUT.resolve()) and folder != OUT.resolve(), 'Evidencias fuera de la entrega')
    manifest = folder/'MANIFEST_CARGA_SHA256.json'
    require(not manifest.exists() and not manifest.with_suffix('.sha256').exists(), 'Sello local ya existente')
    entries = {p.name: sha(p) for p in sorted(folder.iterdir()) if p.is_file()}
    require('RESULTADO_CARGA.json' in entries, 'Resultado ausente antes del sellado')
    write(manifest, entries)
    with manifest.with_suffix('.sha256').open('x', encoding='ascii') as stream:
        stream.write(sha(manifest)+'  '+manifest.name+'\n')
    require(all(sha(folder/name) == digest for name, digest in entries.items()), 'Evidencias modificadas al sellar')
    return sha(manifest)


def program():
    proof = check()
    require(not MARKER.exists(), 'Ya existe un intento de carga: conservar evidencias; no reintentar automáticamente')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    folder = OUT/'programacion_flash'/stamp
    require(folder.resolve().is_relative_to(OUT.resolve()), 'Destino de registros fuera de la entrega')
    folder.mkdir(parents=True, exist_ok=False)
    write(MARKER, dict(attempt=1, evidence_path=str(folder), offline_check=proof,
        hardware_budget_seconds=MAX_SECONDS, automatic_retry=False, execution_owner='user_CLI'))
    print('CANON42: comprobar identidad SWD; después una carga y verificación, sin COM.', flush=True)
    started = time.monotonic()
    phases = []
    status, reason = 'FAIL_CANON42_PROGRAMMING', None
    download_verified = False
    programmed_identity = None
    integrity = False
    try:
        first = run_cli(connection_command(folder/'programmer_identidad.log'),
            folder/'stdout_identidad.txt', MAX_SECONDS-(time.monotonic()-started))
        phases.append(first)
        first['identity'] = identity(first['text'])
        require(command_ok(first) and first['identity']['approved'], 'Lectura SWD sin identidad aprobada; no se inició carga')
        # Revalidar los bytes y sellos justo antes de enviar la orden de Flash.
        check()
        second = run_cli(program_command(folder/'programmer_carga.log'),
            folder/'stdout_carga.txt', MAX_SECONDS-(time.monotonic()-started))
        phases.append(second)
        programmed_identity = identity(second['text'])
        download_verified = bool(re.search(r'Download\s+verified\s+successfully', second['text'], re.I))
        require(command_ok(second) and programmed_identity['approved'] and download_verified,
            'Carga/verificación sin confirmación completa; conservar estado y revisar')
        check()
        integrity = True
        status = 'PASS_CANON42_PROGRAMMED_VERIFIED_RESET_PENDING_UART_GATE'
    except Exception as exc:
        reason = type(exc).__name__+': '+str(exc)
    except KeyboardInterrupt:
        reason = 'INTERRUPTED'
    program_started = len(phases) > 1 and phases[1]['process_created']
    verified = program_started and download_verified and command_ok(phases[1])
    reset_success = bool(program_started and command_ok(phases[1]))
    report = dict(status=status, attempt=1, reason=reason, evidence_path=str(folder),
        ELF_sha256=ELF_SHA, CLI_sha256=CLI_SHA, manifest_sha256=proof['manifest_sha256'],
        request_marker_sha256=sha(MARKER),
        seconds=time.monotonic()-started, hardware_budget_seconds=MAX_SECONDS,
        phases=phases, program_identity=programmed_identity,
        download_verification_marker=download_verified, programming_command_started=program_started,
        flash_state='VERIFIED_PROGRAMMED' if verified else ('MAY_BE_MODIFIED' if program_started else 'NO_PROGRAM_COMMAND_STARTED'),
        hardware_flashed=bool(verified), reset_requested=program_started,
        reset_command_succeeded=reset_success,
        reset_text_reported=bool(program_started and re.search(r'reset', phases[1]['text'], re.I)),
        runtime_boot_confirmed=False, UART_gate_pending=True, integrity_after_program=integrity,
        automatic_retry=False, COM_opened=False, D_files_modified=False, full_build_executed=False)
    write(folder/'RESULTADO_CARGA.json', report)
    evidence_sha = seal_evidence(folder)
    print(json.dumps(dict(status=status, reason=reason, evidence_path=str(folder),
        evidence_manifest_sha256=evidence_sha), ensure_ascii=True), flush=True)
    return 0 if status.startswith('PASS_') else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check', action='store_true')
    mode.add_argument('--program', action='store_true')
    args = parser.parse_args()
    if args.check:
        print(json.dumps(check(), ensure_ascii=True, indent=2))
        return 0
    return program()


if __name__ == '__main__':
    raise SystemExit(main())
