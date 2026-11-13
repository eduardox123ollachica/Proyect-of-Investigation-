"""Contrasta el dump existente con SREC/IHEX derivados del ELF SR, sin placa.

Solo convierte formato mediante objcopy y crea evidencia nueva en el workspace.
No compila ni modifica originales, firmware, capturas previas, D: o hardware.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT/'despliegue_stm32/costo_total_single_rounding_v2_180mhz_2026-10-26'
CAPTURE = OUT/'lectura_flash/20261026T151832_154874Z'
DIAG = OUT/'diagnostico_padding_flash'
REPORT = OUT/'AUDITORIA_IMAGEN_PROGRAMABLE_SR.json'
OBJCOPY = Path('D:/STM32CubeIDE_2.2.0/STM32CubeIDE/plugins/com.st.stm32cube.ide.mcu.externaltools.gnu-tools-for-stm32.14.3.rel1.win32_1.0.100.202602081740/tools/bin/arm-none-eabi-objcopy.exe')
ELF_SHA = 'c0c3853d2b0e90bb20fe00be13bcf107d186088406be131b729a605ded44ecc7'
FLASH_START = 0x08000000
FLASH_SIZE = 0x80000
GAP = range(0x080001c4,0x080001d0)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, value):
    with path.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n')


def put(memory, address, payload):
    for i, value in enumerate(payload):
        key = address+i
        require(FLASH_START <= key < FLASH_START+FLASH_SIZE, 'Dato fuera de Flash')
        require(key not in memory, 'Direccion de datos repetida')
        memory[key] = value


def parse_srec(text):
    memory = {}
    records = {}
    data_count = 0
    end_address = None
    for line in text.splitlines():
        require(line.startswith('S') and line[1] in '0123456789', 'Cabecera SREC invalida')
        kind = int(line[1]); raw = bytes.fromhex(line[2:])
        require(raw and raw[0] == len(raw)-1 and sum(raw)&0xff == 0xff, 'Checksum/tamano SREC invalido')
        address_bytes = {0:2,1:2,2:3,3:4,5:2,6:3,7:4,8:3,9:2}.get(kind)
        require(address_bytes is not None and len(raw) >= address_bytes+2, 'Tipo SREC no admitido')
        address = int.from_bytes(raw[1:address_bytes+1],'big')
        payload = raw[address_bytes+1:-1]
        records[str(kind)] = records.get(str(kind),0)+1
        if kind in (1,2,3):
            put(memory,address,payload); data_count += 1
        elif kind in (5,6):
            require(not payload and address == data_count, 'Conteo SREC distinto')
        elif kind in (7,8,9):
            require(not payload and end_address is None, 'Cierre SREC invalido')
            end_address = address
    require(memory and end_address is not None, 'SREC incompleto')
    return memory,dict(record_counts=records,data_records=data_count,entry_point=f'0x{end_address:08x}')


def parse_ihex(text):
    memory = {}; upper = 0; eof = False; records = {}; entry = None
    for line in text.splitlines():
        require(not eof and line.startswith(':'), 'Cabecera IHEX o datos tras EOF invalidos')
        raw = bytes.fromhex(line[1:])
        require(len(raw) >= 5 and raw[0] == len(raw)-5 and sum(raw)&0xff == 0, 'Checksum/tamano IHEX invalido')
        count = raw[0]; address = int.from_bytes(raw[1:3],'big'); kind = raw[3]; payload = raw[4:-1]
        records[str(kind)] = records.get(str(kind),0)+1
        if kind == 0:
            put(memory,upper+address,payload)
        elif kind == 1:
            require(count == address == 0, 'EOF IHEX invalido'); eof = True
        elif kind in (2,4):
            require(count == 2 and address == 0, 'Direccion extendida IHEX invalida')
            upper = int.from_bytes(payload,'big') << (4 if kind == 2 else 16)
        elif kind == 5:
            require(count == 4 and address == 0 and entry is None, 'Entry point IHEX invalido')
            entry = int.from_bytes(payload,'big')
        else:
            raise ValueError('Tipo IHEX no esperado: '+str(kind))
    require(memory and eof, 'IHEX incompleto')
    return memory,dict(record_counts=records,entry_point=f'0x{entry:08x}' if entry is not None else None)


def contiguous_ranges(addresses):
    result = []
    for address in sorted(addresses):
        if not result or address != result[-1][1]:
            result.append([address,address+1])
        else:
            result[-1][1] += 1
    return [dict(start=f'0x{a:08x}',end_exclusive=f'0x{b:08x}',bytes=b-a) for a,b in result]


def main():
    started = time.perf_counter()
    require(not REPORT.exists() and not DIAG.exists(), 'No sobrescribir diagnostico')
    require(DIAG.resolve().is_relative_to(ROOT.resolve()) and OUT.is_dir(), 'Destino fuera del workspace')
    manifest = CAPTURE/'MANIFEST_LECTURA_FLASH_SHA256.json'
    digest, name = manifest.with_suffix('.sha256').read_text(encoding='ascii').split()
    require(name == manifest.name and sha(manifest) == digest, 'Sello de lectura diferente')
    for rel, expected in read(manifest).items():
        source = (CAPTURE/rel).resolve()
        require(source.is_relative_to(CAPTURE.resolve()) and sha(source) == expected, 'Captura modificada')
    elf = CAPTURE/'firmware_auditado_sr.elf'
    dump_path = CAPTURE/'flash_512KiB.bin'
    require(sha(elf) == ELF_SHA and len(dump_path.read_bytes()) == FLASH_SIZE, 'ELF/dump invalidos')
    prior = read(CAPTURE/'RESULTADO_LECTURA_FLASH_SR.json')
    require(prior['execution']['returncode'] == 0 and prior['device']['read_success_confirmed']
            and prior['comparison']['mismatch_count'] == 12, 'No corresponde a lectura con12 diferencias')
    require({int(row['address'],16) for row in prior['comparison']['first_mismatches']} == set(GAP), 'Rango de diferencias distinto')
    DIAG.mkdir(exist_ok=False)
    results = {}
    memories = {}
    for fmt, suffix, parser in (('srec','.srec',parse_srec),('ihex','.hex',parse_ihex)):
        target = DIAG/('firmware_auditado_sr'+suffix)
        argv = [str(OBJCOPY),'-O',fmt,str(elf),str(target)]
        finished = subprocess.run(argv,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=30,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        record = dict(command_argv=argv,returncode=finished.returncode,
            stdout=finished.stdout.decode('utf-8',errors='replace'),stderr=finished.stderr.decode('utf-8',errors='replace'))
        save(DIAG/('CONVERSION_'+fmt.upper()+'.json'),record)
        require(finished.returncode == 0 and target.is_file(), 'Objcopy no completo conversion: '+fmt)
        memory, details = parser(target.read_text(encoding='ascii'))
        require(len(memory) == 280276, 'Tamano de datos programables inesperado')
        dump = dump_path.read_bytes()
        differences = [address for address,value in memory.items() if dump[address-FLASH_START] != value]
        require(not differences, 'Diferencias en bytes realmente emitidos por objcopy')
        require(not set(GAP)&memory.keys(), 'Los12bytes se emitieron como datos')
        memories[fmt] = memory
        results[fmt] = dict(**record,**details,output=str(target.relative_to(ROOT)),output_sha256=sha(target),
            programmable_bytes=len(memory),all_emitted_bytes_exact_to_readback=True,
            mismatch_count=0,address_ranges=contiguous_ranges(memory),
            alignment_12bytes_omitted_from_format=True,
            ordered_payload_sha256=hashlib.sha256(bytes(memory[a] for a in sorted(memory))).hexdigest())
    require(memories['srec'] == memories['ihex'], 'SREC e IHEX no representan iguales datos')
    gap_detail = [dict(address=f'0x{a:08x}',dump_value=dump_path.read_bytes()[a-FLASH_START],
        encoded_in_SREC=False,encoded_in_IHEX=False) for a in GAP]
    require(all(row['dump_value'] == 255 for row in gap_detail), 'Relleno leido no esFF')
    report = dict(status='PASS_PROGRAMMABLE_SREC_IHEX_BYTES_EXACT_TO_EXISTING_READBACK',
        date_local='2026-10-26',timezone='America/Lima',elf_sha256=ELF_SHA,
        readback_dump_sha256=sha(dump_path),readback_manifest_sha256=digest,
        objcopy_executable=str(OBJCOPY),objcopy_sha256=sha(OBJCOPY),formats=results,
        same_programmable_addresses_and_values_in_both_formats=True,
        expected_programmable_bytes=280276,programmable_mismatches=0,
        excluded_alignment_bytes=12,alignment_gap=gap_detail,
        explanation='Objcopy SREC/IHEX data records omit12ELFsegment alignment bytes; the readback exactly matches every emitted byte.',
        diagnostic_conversion_is_current_not_historical_download_record=True,
        does_not_prove_historical_programming_file_was_this_generated_SREC=True,
        archived_failed_verification_unchanged=True,new_physical_readbacks=0,hardware_connected=False,
        training_fits=0,builds_executed=0,Flash_programming_or_erasing=False,D_files_modified=False,
        selector_enabled=False,costs_promoted_here=False,
        script_sha256=sha(__file__),seconds=time.perf_counter()-started,
        limitations=['Readback was captured after the computational campaign.',
            'These format files are freshly generated diagnostic artifacts, not recovered IDE download files.',
            'No tolerance was applied to emitted code, constants, vector or initialized-data bytes.'])
    save(REPORT,report)
    paths = [Path(__file__).resolve(),REPORT,*[p for p in DIAG.iterdir() if p.is_file()],
        manifest,manifest.with_suffix('.sha256')]
    paths.extend(CAPTURE/rel for rel in read(manifest))
    sealed = OUT/'MANIFEST_IMAGEN_PROGRAMABLE_SR_SHA256.json'
    save(sealed,{p.relative_to(ROOT).as_posix():sha(p) for p in sorted(paths)})
    with sealed.with_suffix('.sha256').open('x',encoding='ascii') as stream:
        stream.write(sha(sealed)+'  '+sealed.name+'\n')
    print(json.dumps(dict(status=report['status'],programmable_bytes=280276,mismatches=0,
        omitted_alignment_bytes=12,manifest_sha256=sha(sealed)),ensure_ascii=True))


if __name__ == '__main__':
    main()
