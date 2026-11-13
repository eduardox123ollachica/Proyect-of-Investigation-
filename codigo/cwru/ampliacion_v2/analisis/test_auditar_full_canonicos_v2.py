"""Pruebas de reconstrucción e integridad FULL; no abren COM ni infieren."""
from __future__ import annotations

import copy
from types import SimpleNamespace
import unittest

import auditar_full_canonicos_v2 as x


class FullAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.f=x.load_full()
        args=SimpleNamespace(firmware_elf=x.OUT/'copia_aislada/Prueba_2_CWRU_Extraccion/Release_180MHz_Canon42_SR/Prueba_2_CWRU_Extraccion.elf',
            build_report=x.OUT/'COMPILACION_CANON42.json',elf_audit=x.OUT/'AUDITORIA_ELF_CANON42.json')
        cls.context=cls.f.load_context(args)
        cls.g,cls.h,cls.cat,cls.bank,cls.arrays,cls.provenance,cls.programming,cls.imports,cls.plan=cls.context
        cls.trace=x.a.read_jsonl(x.FOLDER/'TRAZA_UART_NUEVA.jsonl')
        cls.events=x.a.read_jsonl(x.FOLDER/'MEDICIONES.jsonl')
        cls.report=x.read(x.FOLDER/'RESULTADO_COMPLETO.json')
        cls.rebuilt=cls.replay()

    @classmethod
    def replay(cls,trace=None,events=None):
        return x.reconstruct(cls.f,cls.g,cls.h,cls.cat,cls.bank,cls.arrays,cls.imports,cls.plan,
            cls.trace if trace is None else trace,cls.events if events is None else events)

    def validate(self,report):
        x.validate_report(self.f,self.g,report,self.rebuilt,self.plan,x.EXPECTED_FULL_MANIFEST,
            self.programming,self.imports)

    def test_real_FULL_reconstruction_and_report(self):
        self.validate(self.report)
        self.assertEqual(self.rebuilt['coverage'],dict(RAW_RUN=1008,RAW_PROFILE=336,RAW_BASE=56,
            RAW_pairs=336,POST=2760,POST_unique=1380))
        self.assertEqual(self.rebuilt['UART_pairs'],28836)
        self.assertEqual(len(self.rebuilt['events']),7315)
        self.assertEqual(self.rebuilt['firmware_info']['last_run_id'],0)
        self.assertEqual(self.rebuilt['firmware_info_final']['last_run_id'],4048)

    def test_RAW_payload_alteration_rejected_against_frozen_bank(self):
        trace=list(self.trace)
        i=next(i for i,r in enumerate(trace) if r.get('command','').startswith('CWRUFEAT_DATA,'))
        trace[i]=dict(trace[i],command=trace[i]['command']+',0')
        with self.assertRaisesRegex(ValueError,'TX diferente'):self.replay(trace=trace)

    def test_POST_output_corruption_rejected(self):
        trace=list(self.trace)
        i=next(i for i,r in enumerate(trace) if r.get('line','').startswith('CWRUTOTAL_POST_RESULT,'))
        parts=trace[i]['line'].split(',');parts[3]='00000000'
        trace[i]=dict(trace[i],line=','.join(parts))
        with self.assertRaises(ValueError):self.replay(trace=trace)

    def test_firmware_clock_change_rejected(self):
        trace=list(self.trace)
        trace[1]=dict(trace[1],line=trace[1]['line'].replace(',180000000,',',16000000,'))
        with self.assertRaisesRegex(ValueError,'INFO'):self.replay(trace=trace)

    def test_imported_origin_cannot_be_relabelled_as_new(self):
        events=list(self.events);events[0]=dict(events[0],measurement_source='FULL_NEW_UART')
        with self.assertRaisesRegex(ValueError,'MEDICIONES diverge'):self.replay(events=events)

    def test_historical_BASE_reuse_is_required(self):
        i=next(i for i,r in enumerate(self.events) if r['type']=='REUSE_BASE')
        self.assertEqual(self.events[i]['global_id'],175)
        events=list(self.events);events[i]=dict(events[i],new_extraction_performed=True)
        with self.assertRaisesRegex(ValueError,'MEDICIONES diverge'):self.replay(events=events)

    def test_missing_final_UART_confirmation_rejected(self):
        with self.assertRaisesRegex(ValueError,'Traza UART incompleta'):self.replay(trace=self.trace[:-2])

    def test_false_exhaustive_RAW_claim_rejected(self):
        report=copy.deepcopy(self.report);report['RAW_exhaustive_reserved_coverage']=True
        with self.assertRaisesRegex(ValueError,'alcance excesiva'):self.validate(report)

    def test_changed_timing_summary_rejected(self):
        report=copy.deepcopy(self.report);report['RAW_timing_statistics'][0]['mean_ms']+=1
        with self.assertRaisesRegex(ValueError,'Resumen reconstruido diferente'):self.validate(report)

    def test_imported_and_new_counts_cannot_be_merged_twice(self):
        report=copy.deepcopy(self.report);report['total_completed']['RAW_RUN']+=24
        with self.assertRaisesRegex(ValueError,'Conteos reportados divergentes'):self.validate(report)


if __name__=='__main__':unittest.main()
