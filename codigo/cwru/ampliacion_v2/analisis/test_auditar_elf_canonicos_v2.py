"""Controles sobre bytes ELF reales; las mutaciones solo existen en memoria."""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import struct
import unittest

PATH=Path(__file__).with_name('auditar_elf_canonicos_v2.py')
spec=importlib.util.spec_from_file_location('canon42_elf_test_subject',PATH)
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)


@unittest.skipUnless(a.ELF.exists(),'Se requiere el ELF canónico compilado por el usuario')
class LinkedELFControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.elf=a.reader.ELF32(a.ELF)
        cls.catalog=a.read(a.OUT/'recursos/CATALOGO_TOTAL.json')
        cls.original_sha=a.sha(a.ELF)

    @classmethod
    def tearDownClass(cls):
        if a.sha(a.ELF)!=cls.original_sha:raise AssertionError('Pruebas modificaron el ELF real')

    def mutated(self,address):
        elf=copy.copy(self.elf)
        sections=[s for s in elf.sections if s['flags']&2 and s['type']!=8 and
            s['address']<=address<s['address']+s['size']]
        self.assertEqual(len(sections),1)
        section=sections[0];offset=section['offset']+address-section['address']
        data=bytearray(elf.data);data[offset]^=1;elf.data=bytes(data)
        return elf

    def test_actual_eight_models_and_memory_are_valid(self):
        models=a.audit_models(self.elf,self.catalog,a.OUT/'recursos')
        memory=a.audit_memory(self.elf)
        self.assertEqual(len(models),8)
        self.assertEqual(sum(m['model_bytes'] for m in models),28688)
        self.assertEqual(memory['flash_load_bytes'],185388)
        self.assertEqual(memory['ram_static_including_linker_reservations_bytes'],88460)
        self.assertEqual(memory['flash_program_address_span_bytes'],185408)

    def test_one_changed_linked_parameter_bit_is_rejected(self):
        address=self.elf.symbol('cwru_total_catalog')['address']+44
        with self.assertRaisesRegex(ValueError,'Parámetros enlazados distintos'):
            a.audit_models(self.mutated(address),self.catalog,a.OUT/'recursos')

    def test_one_changed_linked_model_bit_is_rejected(self):
        descriptor=self.elf.address(self.elf.symbol('cwru_total_catalog')['address'],328)
        _,address,size=struct.unpack_from('<III',descriptor)
        with self.assertRaisesRegex(ValueError,'Modelo enlazado distinto'):
            a.audit_models(self.mutated(address+size-1),self.catalog,a.OUT/'recursos')

    def test_missing_or_extra_descriptor_cannot_be_reported_as_eight(self):
        cat=copy.deepcopy(self.catalog);cat['count']=9
        with self.assertRaisesRegex(ValueError,'exactamente ocho'):
            a.audit_models(self.elf,cat,a.OUT/'recursos')

    def test_changed_buffer_allocation_is_rejected(self):
        elf=copy.copy(self.elf);elf.symbols=copy.deepcopy(self.elf.symbols)
        next(s for s in elf.symbols if s['name']=='g_window')['size']=32764
        with self.assertRaisesRegex(ValueError,'Buffer distinto'):
            a.audit_memory(elf)


if __name__=='__main__':unittest.main()
