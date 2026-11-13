"""Rechaza alteraciones de capturas archivadas; no COM/ML o escritura original."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import auditar_gate_canonicos_v2 as a


class ArchivedGate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.g = a.load_client()
        cls.h, cls.cat, cls.bank, cls.arrays = cls.g.load_resources()
        cls.trace = a.read_jsonl(a.GATE/'TRAZA_UART.jsonl')
        cls.events = a.read_jsonl(a.GATE/'MEDICIONES.jsonl')

    def replay(self, trace=None, events=None):
        return a.reconstruct(self.g, self.h, self.cat, self.bank, self.arrays,
            self.trace if trace is None else trace, self.events if events is None else events)

    def test_real_trace_reconstructs_every_exchange_without_classifier(self):
        rebuilt = self.replay()
        self.assertEqual(rebuilt['UART_pairs'], 621)
        self.assertEqual(len(rebuilt['events']), 93)
        self.assertEqual(rebuilt['firmware_info_final']['last_run_id'], 56)
        self.assertEqual(rebuilt['PC_same_input_classifier_invocations'], 0)

    def test_changed_RAW_transmission_is_rejected_against_bank(self):
        trace = copy.deepcopy(self.trace)
        target = next(row for row in trace if row.get('command', '').startswith('CWRUFEAT_DATA,'))
        values = target['command'].split(','); values[2] = '1.0'; target['command'] = ','.join(values)
        with self.assertRaisesRegex(ValueError, 'TX diferente'):
            self.replay(trace=trace)

    def test_changed_POST_output_is_rejected_against_saved_prediction(self):
        trace = copy.deepcopy(self.trace)
        target = next(row for row in trace if row.get('line', '').startswith('CWRUTOTAL_POST_RESULT,'))
        values = target['line'].split(',')
        index = 12+(int(values[11])+1)%3
        values[index] = '-127' if values[index] == '-128' else '-128'
        target['line'] = ','.join(values)
        with self.assertRaisesRegex(ValueError, 'POST distinto'):
            self.replay(trace=trace)

    def test_measurement_annotation_cannot_override_UART_intermediate(self):
        events = copy.deepcopy(self.events)
        target = next(row for row in events if row['type'] == 'RUN')
        target['pages']['score_bits'] = '00000000'
        with self.assertRaisesRegex(ValueError, 'MEDICIONES no coincide'):
            self.replay(events=events)

    def test_failed_stale_result_rejection_cannot_be_reported_PASS(self):
        trace = copy.deepcopy(self.trace)
        target = next(row for row in trace if row.get('line') == 'CWRUTOTAL_ERROR,NO_RESULT,-1')
        target['line'] = 'CWRUTOTAL_ERROR,FORMAT,-1'
        with self.assertRaisesRegex(ValueError, 'conserva publicación'):
            self.replay(trace=trace)

    def test_nested_file_tamper_or_extra_file_is_rejected(self):
        base = a.ROOT/'tmp'; base.mkdir(exist_ok=True)
        folder = Path(tempfile.mkdtemp(prefix='audit_gate_manifest_', dir=base)).resolve()
        self.assertTrue(folder.is_relative_to(base.resolve()) and folder != base.resolve())
        try:
            child = folder/'nested/log.txt'; child.parent.mkdir(); child.write_text('original', encoding='utf-8')
            manifest = folder/'MANIFEST.json'
            manifest.write_text(json.dumps({'nested/log.txt': a.sha(child)}), encoding='utf-8')
            manifest.with_suffix('.sha256').write_text(a.sha(manifest)+'  MANIFEST.json\n', encoding='ascii')
            a.strict_manifest(folder, manifest.name)
            child.write_text('changed', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Evidencia alterada'):
                a.strict_manifest(folder, manifest.name)
            child.write_text('original', encoding='utf-8'); (folder/'extra.txt').write_text('extra', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Inventario recursivo'):
                a.strict_manifest(folder, manifest.name)
        finally:
            if folder != base.resolve() and folder.is_relative_to(base.resolve()):
                shutil.rmtree(folder)


if __name__ == '__main__':
    unittest.main()
