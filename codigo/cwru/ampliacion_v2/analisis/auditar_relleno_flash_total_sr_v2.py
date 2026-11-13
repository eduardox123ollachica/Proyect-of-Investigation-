"""Clasifica el relleno ELF y compara secciones programables sin usar hardware.

El readback y su dictamen original se conservan. Solo se generan un informe
nuevo y su evidencia, con comparacion estricta en todos los bytes de seccion.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import re
import time

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT/'despliegue_stm32/costo_total_single_rounding_v2_180mhz_2026-10-26'
CAPTURE = OUT/'lectura_flash/20261026T151832_154874Z'
PROJECT = OUT/'copia_aislada/Prueba_2_CWRU_Extraccion'
BUILD = PROJECT/'Release_180MHz_TotalV2_SR'
READER = Path(__file__).with_name('auditar_elf_total_v2.py')
REPORT = OUT/'AUDITORIA_RELLENO_ELF_FLASH_SR.json'
FLASH_START = 0x08000000
FLASH_SIZE = 0x80000
ELF_SHA = 'c0c3853d2b0e90bb20fe00be13bcf107d186088406be131b729a605ded44ecc7'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def programmable_sections(elf):
    """Mapea VMA/offset a LMA por PT_LOAD; incluye .data y todo padding interno.

    Solo se excluyen SHT_NOBITS, secciones vacias y bytes sin una seccion
    SHF_ALLOC respaldada por archivo. No hay excepciones por valor o direccion.
    """
    result = []
    for section in elf.sections:
        if not section['size'] or not section['flags'] & 2 or section['type'] == 8:
            continue
        matches = []
        for segment in elf.segments:
            if segment['type'] != 1 or not segment['size']:
                continue
            delta = section['offset']-segment['offset']
            if 0 <= delta and delta+section['size'] <= segment['size'] and \
                    segment['address']+delta == section['address']:
                matches.append(segment)
        require(len(matches) == 1, 'Seccion asignada no tiene PT_LOAD unico: '+section['name'])
        segment = matches[0]
        lma = segment['physical']+section['offset']-segment['offset']
        end = lma+section['size']
        overlap = lma < FLASH_START+FLASH_SIZE and end > FLASH_START
        require(not overlap or FLASH_START <= lma < end <= FLASH_START+FLASH_SIZE,
                'Seccion cruza limite Flash')
        if overlap:
            payload = elf.range(section['offset'], section['size'])
            require(len(payload) == section['size'], 'Contenido ELF incompleto')
            result.append(dict(name=section['name'], address=lma,
                virtual_address=section['address'], size=section['size'],
                file_offset=section['offset'], alignment=section['alignment'],
                flags=section['flags'], type=section['type'], data=payload))
    result.sort(key=lambda item:item['address'])
    require(result and all(a['address']+a['size'] <= b['address'] for a,b in zip(result,result[1:])),
            'Secciones Flash vacias o superpuestas')
    return result


def compare_sections(dump, sections):
    require(len(dump) == FLASH_SIZE, 'Lectura Flash incompleta: se requieren512KiB')
    comparisons = []
    first = []
    mismatch_count = 0
    for section in sections:
        offset = section['address']-FLASH_START
        require(0 <= offset < offset+section['size'] <= len(dump), 'Seccion fuera del dump')
        actual = dump[offset:offset+section['size']]
        expected = section['data']
        require(len(expected) == section['size'], 'Payload de seccion incompleto')
        positions = [i for i,(a,b) in enumerate(zip(actual,expected)) if a != b]
        mismatch_count += len(positions)
        first.extend(dict(address=f'0x{section["address"]+i:08x}', section=section['name'],
            expected=expected[i], actual=actual[i]) for i in positions[:max(0,32-len(first))])
        comparisons.append(dict(name=section['name'], address=f'0x{section["address"]:08x}',
            virtual_address=f'0x{section["virtual_address"]:08x}', bytes=section['size'],
            expected_sha256=hashlib.sha256(expected).hexdigest(),
            actual_sha256=hashlib.sha256(actual).hexdigest(), mismatches=len(positions)))
    return dict(match=mismatch_count == 0, compared_bytes=sum(s['size'] for s in sections),
        mismatch_count=mismatch_count, first_mismatches=first, sections=comparisons,
        numeric_tolerance=0, excluded_file_backed_section_bytes=0,
        inter_section_bytes_not_asserted_from_ELF=True)


def main():
    started = time.perf_counter()
    require(not REPORT.exists(), 'No sobrescribir informe')
    manifest = CAPTURE/'MANIFEST_LECTURA_FLASH_SHA256.json'
    digest,name = manifest.with_suffix('.sha256').read_text(encoding='ascii').split()
    require(name == manifest.name and sha(manifest) == digest, 'Sello de captura distinto')
    for rel,expected in read(manifest).items():
        target=(CAPTURE/rel).resolve()
        require(target.is_relative_to(CAPTURE.resolve()) and sha(target) == expected, 'Captura modificada')
    raw = read(CAPTURE/'RESULTADO_LECTURA_FLASH_SR.json')
    require(raw['status'] == 'FAILED_FLASH_READBACK' and raw['execution']['returncode'] == 0
            and raw['device']['read_success_confirmed'] and raw['comparison']['mismatch_count'] == 12,
            'No corresponde al fallo de comparacion documentado')
    elf_path = CAPTURE/'firmware_auditado_sr.elf'
    dump_path = CAPTURE/'flash_512KiB.bin'
    require(sha(elf_path) == ELF_SHA and sha(dump_path) == raw['dump_sha256'], 'ELF/dump distinto')
    spec=importlib.util.spec_from_file_location('padding_sr_elf_reader',READER)
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    elf=helper.ELF32(elf_path)
    sections=programmable_sections(elf)
    dump=dump_path.read_bytes()
    comparison=compare_sections(dump,sections)
    require(comparison['match'] and comparison['compared_bytes'] == 280276 and len(sections) == 7,
            'Contenido asignado difiere o cobertura inesperada')
    covered = set()
    for section in sections:
        covered.update(range(section['address'], section['address']+section['size']))
    segment_addresses = set()
    mismatch_addresses = []
    for segment in elf.segments:
        if segment['type'] == 1 and segment['size'] and FLASH_START <= segment['physical'] < FLASH_START+FLASH_SIZE:
            payload=elf.range(segment['offset'],segment['size'])
            segment_addresses.update(range(segment['physical'],segment['physical']+segment['size']))
            offset=segment['physical']-FLASH_START
            mismatch_addresses.extend(segment['physical']+i for i,(a,b) in enumerate(zip(dump[offset:offset+len(payload)],payload)) if a != b)
    gap=sorted(segment_addresses-covered)
    require(gap == mismatch_addresses == list(range(0x080001c4,0x080001d0)), 'Diferencias no limitadas al relleno')
    first_segment=next(s for s in elf.segments if s['type']==1 and s['physical']==FLASH_START and s['size'])
    require(elf.range(first_segment['offset']+gap[0]-FLASH_START,12)==b'\x00'*12
            and dump[gap[0]-FLASH_START:gap[-1]+1-FLASH_START]==b'\xff'*12,
            'Valores de relleno distintos de los reportados')
    vector=next(s for s in sections if s['name']=='.isr_vector')
    text=next(s for s in sections if s['name']=='.text')
    require(vector['address']+vector['size'] == gap[0] and text['address'] == gap[-1]+1
            and text['alignment'] == 16, 'Limites/alineacion no corresponden')
    map_path=BUILD/'Prueba_2_CWRU_Extraccion.map'
    map_lines=map_path.read_text(encoding='utf-8-sig').splitlines()
    map_evidence=[dict(line=i+1,text=line) for i,line in enumerate(map_lines)
        if re.match(r'^\.(isr_vector|text|data)\s',line)]
    require(any('.isr_vector' in row['text'] and '0x1c4' in row['text'] for row in map_evidence)
            and any('.text' in row['text'] and '0x080001d0' in row['text'] for row in map_evidence), 'Mapa distinto')
    linker=PROJECT/'STM32F446RETX_FLASH.ld'
    startup=PROJECT/'Startup/startup_stm32f446retx.s'
    require('KEEP(*(.isr_vector))' in linker.read_text(encoding='utf-8-sig')
            and '} >RAM AT> FLASH' in linker.read_text(encoding='utf-8-sig'), 'Linker layout diferente')
    vector_symbol=elf.symbol('g_pfnVectors')
    require(vector_symbol['address']==FLASH_START and vector_symbol['size']==452, 'Tabla vectorial distinta')
    symbols_in_gap=[s for s in elf.symbols if s['section_index'] != 0 and gap[0] <= s['address'] < gap[-1]+1]
    listing_path=BUILD/'Prueba_2_CWRU_Extraccion.list'
    instruction_lines=[line for line in listing_path.read_text(encoding='utf-8-sig').splitlines()
        if re.match(r'^\s*[0-9a-f]+:\t[0-9a-f ]+\s*\t',line)]
    explicit_refs=[line for line in instruction_lines if re.search(r'\b(?:0x)?0*80001c[4-9a-f]\b',line.split('\t')[-1],re.I)]
    result=dict(status='PASS_ALL_PROGRAMMABLE_SECTIONS_EXACT_ONLY_INTER_SECTION_PADDING_DIFFERS',
        date_local='2026-10-26',timezone='America/Lima',capture=CAPTURE.relative_to(ROOT).as_posix(),
        capture_manifest_sha256=digest,original_failed_status_preserved=True,elf_sha256=ELF_SHA,
        dump_sha256=sha(dump_path),comparison=comparison,PT_LOAD_file_bytes=len(segment_addresses),
        programmable_section_bytes=len(covered),section_count=len(sections),
        inter_section_gap=dict(start='0x080001c4',end_exclusive='0x080001d0',bytes=12,
            ELF_file_byte=0,readback_byte=255,previous_section='.isr_vector',next_section='.text',
            next_section_alignment=16,covered_by_any_SHF_ALLOC_file_backed_section=False,
            covered_by_any_ELF_section=any(s['offset'] <= 0x11c4 < s['offset']+s['size'] for s in elf.sections if s['type']!=8),
            symbols_defined_in_gap=symbols_in_gap,explicit_disassembly_targets_in_gap=explicit_refs),
        vector_table=dict(bytes=452,entries_32bit=113,whole_section_exact=True),map_evidence=map_evidence,
        original_comparator_scope='All PT_LOAD file bytes, including inter-section padding',
        corrected_comparator_scope='All SHF_ALLOC file-backed sections mapped to physical Flash LMA, including initialized RAM load image',
        any_payload_tolerance_added=False,hardcoded_ignored_address_exception=False,
        inference='Original failure arises from requiring ELF inter-section padding zeros in a region that has no programmable section.',
        limitations=['No claim that arbitrary dynamic code can never read the gap; section membership and explicit references do not prove universal unreachability.',
            'This report classifies the existing capture and is not a new physical readback.',
            'Readback occurred after the timing campaign and does not by itself establish historical image identity at every earlier instant.',
            'Independent S-record conversion evidence is produced separately.'],
        primary_sources=[dict(title='ELF gABI, Sections',url='https://gabi.xinuos.com/elf/03-sheader.html',
            support='Sections may leave inactive file space; alignment padding between sections is an example and its contents are unspecified.'),
            dict(title='GNU binutils objcopy',url='https://www.sourceware.org/binutils/docs/binutils/objcopy.html',
            support='objcopy supports S-record output and exposes gap filling as an explicit operation on section load addresses.')],
        script_sha256=sha(__file__),reader_sha256=sha(READER),
        evidence_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in
            (elf_path,dump_path,manifest,manifest.with_suffix('.sha256'),map_path,linker,startup,listing_path)},
        new_hardware_connections=0,new_builds=0,new_training_fits=0,new_TFLite_invocations=0,
        D_files_modified=False,original_capture_or_old_seals_modified=False,seconds=time.perf_counter()-started)
    with REPORT.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(status=result['status'],section_bytes=len(covered),gap_bytes=len(gap),
        section_mismatches=comparison['mismatch_count']),ensure_ascii=True))


if __name__=='__main__':
    main()
