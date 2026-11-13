"""Guardas de exactitud de payload Flash; solo datos sinteticos en memoria."""
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import auditar_relleno_flash_total_sr_v2 as m


def fixture():
    data=b'VECT'+b'\0'*12+b'CODE'+b'CONS'+b'INIT'
    segments=[dict(type=1,offset=0,address=m.FLASH_START,physical=m.FLASH_START,size=24),
              dict(type=1,offset=24,address=0x20000000,physical=m.FLASH_START+32,size=4)]
    sections=[dict(name='.isr_vector',type=1,flags=2,address=m.FLASH_START,offset=0,size=4,alignment=1),
              dict(name='.text',type=1,flags=6,address=m.FLASH_START+16,offset=16,size=4,alignment=16),
              dict(name='.rodata',type=1,flags=2,address=m.FLASH_START+20,offset=20,size=4,alignment=4),
              dict(name='.data',type=1,flags=3,address=0x20000000,offset=24,size=4,alignment=4),
              dict(name='.bss',type=8,flags=3,address=0x20000004,offset=28,size=64,alignment=4)]
    elf=SimpleNamespace(segments=segments,sections=sections,range=lambda p,n:data[p:p+n])
    dump=bytearray(b'\xff'*m.FLASH_SIZE)
    dump[0:4],dump[16:24],dump[32:36]=b'VECT',b'CODECONS',b'INIT'
    return elf,dump


class SectionComparisonGuards(unittest.TestCase):
    def test_only_gap_difference_is_accepted_and_all_payload_bytes_remain_required(self):
        elf,dump=fixture();sections=m.programmable_sections(elf)
        report=m.compare_sections(dump,sections)
        self.assertTrue(report['match']);self.assertEqual(report['compared_bytes'],16)
        self.assertEqual(len(sections),4)
        self.assertEqual(sections[-1]['address'],m.FLASH_START+32)
        self.assertEqual(sections[-1]['virtual_address'],0x20000000)

    def test_one_code_byte_change_fails(self):
        elf,dump=fixture();dump[16]^=1
        report=m.compare_sections(dump,m.programmable_sections(elf))
        self.assertFalse(report['match']);self.assertEqual(report['mismatch_count'],1)
        self.assertEqual(report['first_mismatches'][0]['section'],'.text')

    def test_gap_becomes_required_if_it_has_an_allocated_file_backed_section(self):
        elf,dump=fixture()
        elf.sections.append(dict(name='.required_zeros',type=1,flags=2,address=m.FLASH_START+4,
                                 offset=4,size=12,alignment=4))
        report=m.compare_sections(dump,m.programmable_sections(elf))
        self.assertFalse(report['match']);self.assertEqual(report['mismatch_count'],12)
        self.assertEqual(report['compared_bytes'],28)

    def test_one_vector_byte_change_fails(self):
        elf,dump=fixture();dump[0]^=1
        report=m.compare_sections(dump,m.programmable_sections(elf))
        self.assertFalse(report['match']);self.assertEqual(report['mismatch_count'],1)
        self.assertEqual(report['first_mismatches'][0]['section'],'.isr_vector')

    def test_initialized_RAM_load_byte_change_and_incomplete_read_fail(self):
        elf,dump=fixture();dump[32]^=1
        sections=m.programmable_sections(elf);report=m.compare_sections(dump,sections)
        self.assertFalse(report['match']);self.assertEqual(report['mismatch_count'],1)
        self.assertEqual(report['first_mismatches'][0]['section'],'.data')
        with self.assertRaises(ValueError):m.compare_sections(dump[:-1],sections)


if __name__=='__main__':
    output=io.StringIO()
    result=unittest.TextTestRunner(stream=output,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(SectionComparisonGuards))
    report=dict(status='PASS_STRICT_SECTION_PAYLOAD_GUARDS' if result.wasSuccessful() else 'FAIL',
        tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),output=output.getvalue(),
        comparison_source_sha256=m.sha(Path(m.__file__)),test_source_sha256=m.sha(__file__),
        simulation_only=True,hardware_connections=0,old_files_modified=False)
    path=m.OUT/'PRUEBAS_COMPARACION_SECCIONES_FLASH_SR.json'
    with path.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(report,stream,ensure_ascii=False,indent=2);stream.write('\n')
    print(json.dumps({key:report[key] for key in ('status','tests','failures','errors')},ensure_ascii=True))
    raise SystemExit(0 if result.wasSuccessful() else 1)
