"""Contratos de gate y comparación contra referencias reales; no abre COM."""
from __future__ import annotations

import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import shutil
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

import validar_canonicos_gate as g


class GateData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.h, cls.cat, cls.bank, cls.arrays = g.load_resources()
        cls.plan = g.make_plan(cls.cat, cls.bank, cls.arrays['postrefs'])

    def test_mechanical_plan_covers_all_models_three_labels_and_only_authorized_RAW(self):
        self.assertEqual(len(self.plan['POST_cases']), 24)
        by_id = {row['global_id']: row for row in self.bank['RAW_windows']}
        seen = []
        self.assertEqual(len(self.plan['RAW_schedule']), 2)
        self.assertEqual([int(by_id[gid]['metadata']['load_hp']) for gid, ids in self.plan['RAW_schedule']], [0, 1])
        for gid, ids in self.plan['RAW_schedule']:
            self.assertTrue(set(ids) <= set(by_id[gid]['authorized_model_ids']))
            for rid in ids:
                self.assertNotEqual(int(by_id[gid]['metadata']['load_hp']), self.cat['representatives'][rid]['outer_load_hp'])
            seen.extend(ids)
        self.assertEqual(sorted(seen), list(range(8)))
        for rid in range(8):
            cases = [r for r in self.plan['POST_cases'] if r['model_id'] == rid]
            self.assertEqual([r['label'] for r in cases], [0, 1, 2])
            prefix = self.bank['models'][rid]['reference_prefix']
            ids, y = self.arrays['postrefs'][prefix+'global_window_ids'], self.arrays['postrefs'][prefix+'y_true']
            for case in cases:
                self.assertEqual(case['reference_index'], int(np.flatnonzero(y == case['label'])[0]))
                self.assertEqual(case['global_id'], ids[case['reference_index']])
        self.assertFalse(self.plan['selection_uses_accuracy_or_confidence'])
        self.assertFalse(self.plan['full_validation_enabled'])

    def test_POST_is_exact_saved_output_for_all_24_selected_cases_without_classifier(self):
        for case in self.plan['POST_cases']:
            rep = self.cat['representatives'][case['model_id']]
            ref = g.reference(self.arrays['postrefs'], self.bank['models'][rep['id']], case['global_id'])
            f = ref['features14']; features = [self.h.bits(f)[i] if i in rep['feature_indices'] else '00000000' for i in range(14)]
            pages = dict(features_bits=features, **self.h.replay(features, rep))
            result = dict(output_int8=ref['int8_outputs'].astype(int).tolist(),
                predicted_class=int(ref['int8_predictions']), score_bits=self.h.bits([ref['detector_scores']])[0],
                rejected=bool(ref['detector_rejected']))
            checked = g.compare_post(self.h, result, pages, rep, ref)
            self.assertTrue(checked['saved_external_reference_exact'])
            self.assertEqual(checked['classifier_PC_invocations'], 0)
            changed = copy.deepcopy(result); changed['output_int8'][0] = -128 if changed['output_int8'][0] != -128 else 127
            with self.assertRaisesRegex(ValueError, 'POST distinto'):
                g.compare_post(self.h, changed, pages, rep, ref)

    def test_info_requires_eight_protocol2_180MHz_and_safe_FPSCR(self):
        base = f'CWRUTOTAL_INFO,2,8,12000,4096,2,180000000,1,00000000,{self.cat["catalog_fnv"]},{g.CATALOG_SHA},-1,0,0,13,64,0'
        self.assertEqual(g.parse_info(self.h, base, self.cat)['count'], 8)
        for old, new in [('INFO,2,8', 'INFO,1,32'), ('180000000', '16000000'),
                         ('00000000', '01000000'), ('00000000', '00400000')]:
            with self.assertRaises(ValueError):
                g.parse_info(self.h, base.replace(old, new), self.cat)

    def test_POST_binary_serialization_preserves_float32_bits_and_rejects_nonfinite(self):
        values = np.arange(14, dtype=np.float32)/np.float32(7)
        values[0] = np.float32(-0.0); values[1] = np.float32(1e-40)
        command, checksum = g.post_command(self.h, values)
        self.assertEqual(len(command), 136)
        self.assertEqual(bytes.fromhex(command.rsplit(',', 1)[1]), values.astype('<f4').tobytes())
        self.assertEqual(checksum, self.h.fnv(values.astype('<f4').tobytes()))
        for invalid in (values[:13], np.full(14, np.nan, dtype=np.float32), np.full(14, np.inf, dtype=np.float32)):
            with self.assertRaises(ValueError):
                g.post_command(self.h, invalid)

    def test_POST_parser_zero_extractions_counter_wrap_pairing_and_cost_separation(self):
        rep = self.cat['representatives'][0]
        line = 'CWRUTOTAL_POST_RESULT,1,0,12345678,0,100,0,0,0,0,0,0,127,-128,-128,00000000,0,1188,0'
        result = g.parse_post_result(self.h, line, rep, '12345678', 0xffffffff, 1188)
        self.assertEqual(result['run_id'], 1)
        self.assertEqual(result['extractions'], 0)
        self.assertFalse(result['timing_is_full_pipeline'])
        for index, changed in [(0, 'CWRUTOTAL_RESULT'), (1, '2'), (2, '1'), (3, '12345679'),
                               (6, '1'), (11, '2'), (17, '1200'), (18, '1')]:
            fields = line.split(','); fields[index] = changed
            with self.assertRaises(ValueError):
                g.parse_post_result(self.h, ','.join(fields), rep, '12345678', 0xffffffff, 1188)

    def test_RAW_oracle_uses_saved_output_for_same_INT8_and_caches_changed_inputs(self):
        fake_lite = Mock(); fake_lite.predict.return_value = [0, 1, 2]
        fake_h = SimpleNamespace(LiteModels=Mock(return_value=fake_lite))
        oracle = g.RawOracle(fake_h, self.cat['representatives'])
        ref = dict(int8_inputs=np.asarray([1, 2], dtype=np.int8), int8_outputs=np.asarray([3, 4, 5], dtype=np.int8))
        result, kind = oracle.expected(0, [1, 2], ref)
        self.assertEqual(result, [3, 4, 5]); self.assertEqual(kind, 'saved_same_INT8_input')
        fake_h.LiteModels.assert_not_called(); self.assertEqual(oracle.invocations, 0)
        self.assertEqual(oracle.expected(0, [1, 3], ref)[0], [0, 1, 2])
        self.assertEqual(oracle.expected(0, [1, 3], ref)[0], [0, 1, 2])
        fake_lite.predict.assert_called_once_with(0, [1, 3]); self.assertEqual(oracle.invocations, 1)

    def test_POST_invalid_cases_have_real_wrong_FNV_and_NaN_and_invalidate_publication(self):
        class FakeLink:
            def __init__(self): self.calls = []; self.rejects = 0
            def ask(inner, command, prefix, allow_error=False):
                inner.calls.append(command)
                if command.startswith('CWRUTOTAL_FEATURES,'):
                    return 'CWRUTOTAL_ERROR,NO_RESULT,-1'
                inner.rejects += 1
                if inner.rejects == 1: return 'CWRUTOTAL_ERROR,FORMAT,-1'
                supplied, payload = command.split(',')[1:]
                binary = bytes.fromhex(payload)
                if inner.rejects == 2:
                    self.assertNotEqual(supplied, self.h.fnv(binary))
                    return 'CWRUTOTAL_ERROR,CHECKSUM,-1'
                self.assertEqual(supplied, self.h.fnv(binary))
                self.assertTrue(np.isnan(np.frombuffer(binary, dtype='<f4')[0]))
                return 'CWRUTOTAL_ERROR,NONFINITE,0'
        link = FakeLink()
        rows = g.negatives(self.h, link, {'run_id': 42}, np.arange(14, dtype=np.float32))
        self.assertEqual(len(rows), 3); self.assertEqual(len(link.calls), 6)
        self.assertTrue(all(row['old_result_invalidated'] for row in rows))

    def test_actual_build_and_ELF_correspond_to_expected_gate_identity(self):
        args = SimpleNamespace(firmware_elf=g.OUT/'copia_aislada/Prueba_2_CWRU_Extraccion'/g.CONFIG/'Prueba_2_CWRU_Extraccion.elf',
            build_report=g.OUT/'COMPILACION_CANON42.json', elf_audit=g.OUT/'AUDITORIA_ELF_CANON42.json')
        proof = g.authenticate_build(args, self.cat)
        self.assertEqual(g.sha(proof['firmware']), g.ELF_SHA)
        original = g.read
        def bad(path):
            value = original(path)
            if Path(path).resolve() == args.elf_audit.resolve(): value['status'] = 'FAILED'
            return value
        with patch.object(g, 'read', side_effect=bad), self.assertRaisesRegex(ValueError, 'ELF no auditado'):
            g.authenticate_build(args, self.cat)


class JointSeal(unittest.TestCase):
    def test_joint_manifest_requires_client_helpers_ELF_and_reports_and_rejects_changed_file(self):
        base = (g.ROOT/'tmp').resolve(); base.mkdir(exist_ok=True)
        folder = Path(tempfile.mkdtemp(prefix='canon42_joint_', dir=base)).resolve()
        self.assertTrue(folder.is_relative_to(base))
        try:
            out = folder/'out'; out.mkdir(); here = folder/'client'; here.mkdir()
            resources = folder/'resources'; resources.mkdir(); banks = folder/'banks'; banks.mkdir()
            own = folder/'client.py'; own.write_text('# fixture\n', encoding='utf-8')
            files = [own, here/g.HELPER_NAME, out/'fw.elf', out/'build.json', out/'audit.json',
                resources/'MANIFEST_RECURSOS_SHA256.json', banks/'MANIFEST_BANCOS_SHA256.json']
            for path in files[1:]: path.write_bytes(b'fixture')
            args = SimpleNamespace(firmware_elf=files[2], build_report=files[3], elf_audit=files[4])
            mp = out/'MANIFEST_CLIENTES_CANON42_SHA256.json'
            mp.write_text(json.dumps({p.relative_to(folder).as_posix(): g.sha(p) for p in files}), encoding='utf-8')
            mp.with_suffix('.sha256').write_text(g.sha(mp)+'  '+mp.name+'\n', encoding='ascii')
            with patch.multiple(g, ROOT=folder, OUT=out, HERE=here, RESOURCES=resources, BANKS=banks, __file__=str(own)):
                self.assertEqual(g.joint_manifest(args), g.sha(mp))
                files[3].write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError, 'Entrada conjunta alterada'):
                    g.joint_manifest(args)
        finally:
            if folder != base and folder.is_relative_to(base): shutil.rmtree(folder)


class ProgrammingChain(unittest.TestCase):
    def test_gate_seal_covers_nested_programming_evidence_and_detects_its_modification(self):
        base = (g.ROOT/'tmp').resolve(); base.mkdir(exist_ok=True)
        folder = Path(tempfile.mkdtemp(prefix='canon42_nested_seal_', dir=base)).resolve()
        try:
            nested = folder/'carga_verificada'; nested.mkdir()
            (folder/'RESULTADO_GATE.json').write_text('{"fixture": true}', encoding='utf-8')
            report = nested/'RESULTADO_CARGA.json'; report.write_text('{"status": "fixture"}', encoding='utf-8')
            (nested/'stdout_carga.txt').write_text('fixture programmer transcript', encoding='utf-8')
            entries = g.seal_evidence(folder)
            self.assertEqual(set(entries), {'RESULTADO_GATE.json', 'carga_verificada/RESULTADO_CARGA.json',
                'carga_verificada/stdout_carga.txt'})
            self.assertEqual(g.manifest(folder, 'MANIFEST_EVIDENCIAS_SHA256.json'), entries)
            report.write_text('modified after gate', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Entrada alterada'):
                g.manifest(folder, 'MANIFEST_EVIDENCIAS_SHA256.json')
        finally:
            if folder != base and folder.is_relative_to(base): shutil.rmtree(folder)

    def test_missing_failed_or_different_programming_is_rejected_and_matching_chain_passes(self):
        base = (g.ROOT/'tmp').resolve(); base.mkdir(exist_ok=True)
        folder = Path(tempfile.mkdtemp(prefix='canon42_program_', dir=base)).resolve()
        try:
            out = folder/'out'; out.mkdir()
            evidence = out/'programacion_flash'/'fixture'; evidence.mkdir(parents=True)
            joint = 'a'*64; cli = 'b'*64
            marker_path = out/'INTENTO_CARGA_CANON42.json'
            with patch.multiple(g, ROOT=folder, OUT=out):
                with self.assertRaisesRegex(ValueError, 'Falta la carga'):
                    g.authenticate_programming(joint)
                marker = dict(attempt=1, automatic_retry=False, evidence_path=str(evidence),
                    offline_check=dict(manifest_sha256=joint, ELF_sha256=g.ELF_SHA, CLI_sha256=cli))
                marker_path.write_text(json.dumps(marker), encoding='utf-8')
                report = dict(status='PASS_CANON42_PROGRAMMED_VERIFIED_RESET_PENDING_UART_GATE',
                    attempt=1, automatic_retry=False, ELF_sha256=g.ELF_SHA, manifest_sha256=joint,
                    CLI_sha256=cli, request_marker_sha256=g.sha(marker_path), evidence_path=str(evidence),
                    hardware_flashed=True, flash_state='VERIFIED_PROGRAMMED', reset_command_succeeded=True,
                    integrity_after_program=True, download_verification_marker=True)
                result = evidence/'RESULTADO_CARGA.json'; log = evidence/'stdout_carga.txt'
                log.write_text('Download verified successfully', encoding='utf-8')
                def publish(value):
                    result.write_text(json.dumps(value), encoding='utf-8')
                    mp = evidence/'MANIFEST_CARGA_SHA256.json'
                    mp.write_text(json.dumps({p.name: g.sha(p) for p in (result, log)}), encoding='utf-8')
                    mp.with_suffix('.sha256').write_text(g.sha(mp)+'  '+mp.name+'\n', encoding='ascii')
                publish(report)
                proof = g.authenticate_programming(joint)
                self.assertEqual(proof['programmer_report_sha256'], g.sha(result))
                self.assertEqual(len(proof['files_sha256']), 5)
                self.assertFalse(proof['runtime_boot_confirmed'])
                for key, value in [('status', 'FAILED'), ('ELF_sha256', 'c'*64), ('manifest_sha256', 'd'*64),
                                   ('flash_state', 'MAY_BE_MODIFIED'), ('hardware_flashed', False),
                                   ('reset_command_succeeded', False), ('integrity_after_program', False),
                                   ('download_verification_marker', False), ('automatic_retry', True)]:
                    bad = dict(report); bad[key] = value; publish(bad)
                    with self.assertRaises(ValueError):
                        g.authenticate_programming(joint)
                publish(report)
                log.write_text('tampered after load', encoding='utf-8')
                with self.assertRaisesRegex(ValueError, 'Entrada alterada'):
                    g.authenticate_programming(joint)
        finally:
            if folder != base and folder.is_relative_to(base): shutil.rmtree(folder)

    def test_offline_check_does_not_require_or_open_hardware_programming(self):
        args = SimpleNamespace(firmware_elf=g.OUT/'copia_aislada/Prueba_2_CWRU_Extraccion'/g.CONFIG/'Prueba_2_CWRU_Extraccion.elf',
            build_report=g.OUT/'COMPILACION_CANON42.json', elf_audit=g.OUT/'AUDITORIA_ELF_CANON42.json')
        with patch.object(g, 'joint_manifest', return_value='a'*64), \
             patch.object(g, 'authenticate_programming', side_effect=AssertionError('hardware prerequisite called')) as program, \
             redirect_stdout(io.StringIO()) as printed:
            plan = g.check(args)
        program.assert_not_called()
        self.assertEqual(len(plan['POST_cases']), 24)
        self.assertIn('NO_SERIAL_OPENED', printed.getvalue())


if __name__ == '__main__':
    unittest.main()
