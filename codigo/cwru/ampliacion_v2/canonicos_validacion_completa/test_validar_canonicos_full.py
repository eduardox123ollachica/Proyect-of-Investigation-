"""Integridad y cobertura de FULL con evidencias reales; sin COM ni modelos ML."""
from __future__ import annotations

import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import validar_canonicos_full as f

TEMP_PARENT = (f.ROOT/'tmp').resolve()
assert TEMP_PARENT.is_relative_to(f.ROOT.resolve())


class FullData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.args = SimpleNamespace(
            firmware_elf=f.OUT/'copia_aislada/Prueba_2_CWRU_Extraccion/Release_180MHz_Canon42_SR/Prueba_2_CWRU_Extraccion.elf',
            build_report=f.OUT/'COMPILACION_CANON42.json',
            elf_audit=f.OUT/'AUDITORIA_ELF_CANON42.json')
        cls.context = f.load_context(cls.args)
        cls.g, cls.h, cls.cat, cls.bank, cls.arrays, cls.provenance, cls.programming, cls.imports, cls.plan = cls.context

    def test_real_context_imports_audited_gate_without_classifier(self):
        self.assertEqual(self.imports['report']['PC_same_input_classifier_invocations'], 0)
        self.assertEqual(len(self.imports['bases']), 2)
        self.assertEqual(len(self.imports['runs']), 32)
        self.assertEqual(len(self.imports['posts']), 24)
        self.assertEqual(self.plan['new_counts'], dict(RAW_transmissions=55,
            RAW_model_window_pairs=328, RAW_RUN=984, RAW_PROFILE=328, RAW_BASE=54, POST=2736))
        self.assertFalse(self.plan['already_completed_cases_reexecuted'])
        self.assertFalse(self.plan['old_representative_timings_imported'])

    def test_programming_authentication_keeps_original_joint_seal(self):
        with patch.object(self.g, 'authenticate_programming', wraps=self.g.authenticate_programming) as auth, \
                patch.object(f, 'gate_module', return_value=self.g):
            f.load_context(self.args)
        auth.assert_called_once_with(f.OLD_JOINT_SHA)

    def test_reused_BASE_gid175_and_finished_gid0_are_not_reexecuted(self):
        scheduled = {row['global_id']: row for row in self.plan['RAW_schedule']}
        self.assertNotIn(0, scheduled)
        self.assertFalse(scheduled[175]['new_BASE'])
        self.assertEqual(scheduled[175]['BASE_origin'], 'AUDITED_GATE_IMPORT')
        self.assertEqual(scheduled[175]['model_ids'], [4, 5, 6, 7])
        self.assertEqual(sum(row['new_BASE'] for row in scheduled.values()), 54)
        remaining = {(row['global_id'], rid) for row in scheduled.values() for rid in row['model_ids']}
        self.assertFalse(remaining & self.imports['raw_pairs'])
        remaining_post = {(row['model_id'], row['global_id']) for row in self.plan['POST_cases']}
        self.assertFalse(remaining_post & self.imports['post_keys'])

    def assembled_coverage(self):
        runs = copy.deepcopy(self.imports['runs'])
        bases = copy.deepcopy(self.imports['bases'])
        posts = copy.deepcopy(self.imports['posts'])
        for row in self.plan['RAW_schedule']:
            gid = row['global_id']
            if row['new_BASE']:
                bases.append(dict(global_id=gid))
            for rid in row['model_ids']:
                runs.extend(dict(global_id=gid, rep_id=rid, mode=mode, repetition=rep)
                    for mode, rep in [(0, 1), (0, 2), (0, 3), (1, 1)])
        posts.extend(dict(rep_id=row['model_id'], global_id=row['global_id'])
            for row in self.plan['POST_cases'])
        return runs, bases, posts

    def test_gate_plus_new_plan_covers_exact_frozen_bank(self):
        self.assertEqual(f.coverage(*self.assembled_coverage(), self.bank), dict(
            RAW_RUN=1008, RAW_PROFILE=336, RAW_BASE=56, RAW_pairs=336, POST=2760, POST_unique=1380))

    def test_coverage_rejects_repeated_imported_RAW_and_POST(self):
        runs, bases, posts = self.assembled_coverage()
        runs[-1] = copy.deepcopy(runs[0])
        with self.assertRaisesRegex(ValueError, 'RAW incompleto o duplicado'):
            f.coverage(runs, bases, posts, self.bank)
        runs, bases, posts = self.assembled_coverage()
        posts[-1] = copy.deepcopy(posts[0])
        with self.assertRaisesRegex(ValueError, 'POST incompleto o duplicado'):
            f.coverage(runs, bases, posts, self.bank)

    def test_coverage_rejects_substitution_outside_frozen_bank(self):
        runs, bases, posts = self.assembled_coverage()
        source_gid, rid = runs[-1]['global_id'], runs[-1]['rep_id']
        for row in runs:
            if (row['global_id'], row['rep_id']) == (source_gid, rid):
                row['global_id'] = 100000
        with self.assertRaisesRegex(ValueError, 'IDs distintos'):
            f.coverage(runs, bases, posts, self.bank)

    def replay_imports(self, changed_events):
        original_read_text = Path.read_text
        def replacement(path, *args, **kwargs):
            if path == f.GATE_FOLDER/'MEDICIONES.jsonl':
                return '\n'.join(json.dumps(row) for row in changed_events)
            return original_read_text(path, *args, **kwargs)
        # El sello se prueba por separado; aquí se prueba que incluso una captura
        # decodificada manipulada no puede importarse como resultado válido.
        with patch.object(self.g, 'manifest', return_value=self.imports['entries']), \
                patch.object(Path, 'read_text', replacement):
            return f.load_imports(self.g, self.cat, self.bank, self.arrays, audit_required=False)

    def test_import_rejects_duplicate_RAW_with_unchanged_counts(self):
        events = copy.deepcopy(self.imports['events'])
        indices = [i for i, row in enumerate(events) if row['type'] == 'RUN']
        events[indices[-1]] = copy.deepcopy(events[indices[0]])
        with self.assertRaisesRegex(ValueError, 'duplicado'):
            self.replay_imports(events)

    def test_import_rejects_tampered_RAW_intermediate(self):
        events = copy.deepcopy(self.imports['events'])
        row = next(row for row in events if row['type'] == 'RUN')
        row['pages']['features_bits'][0] = '00000000'
        with self.assertRaisesRegex(ValueError, 'Features activas distintas'):
            self.replay_imports(events)

    def test_import_rejects_tampered_POST_saved_output(self):
        events = copy.deepcopy(self.imports['events'])
        row = next(row for row in events if row['type'] == 'POST')
        row['output_int8'][0] += 1
        with self.assertRaisesRegex(ValueError, 'POST distinto'):
            self.replay_imports(events)

    def test_import_origin_keeps_original_event_sequence(self):
        event = self.imports['events'][0]
        tagged = f.tag_import(event)
        self.assertEqual(tagged['gate_event_seq'], event['seq'])
        self.assertNotIn('seq', tagged)
        self.assertEqual(tagged['measurement_source'], 'AUDITED_GATE_IMPORT')
        self.assertTrue(tagged['included_without_reexecution'])
        self.assertIn('seq', event)

    def test_missing_prepare_source_does_not_leave_partial_package(self):
        with tempfile.TemporaryDirectory(dir=TEMP_PARENT, prefix='canon42_full_test_') as temp:
            here = Path(temp); package = here/'not_created'
            with patch.multiple(f, HERE=here, PACKAGE=package), \
                    patch.object(f, 'load_context', return_value=self.context):
                with self.assertRaisesRegex(ValueError, 'Fuente FULL ausente'):
                    f.prepare(self.args)
            self.assertFalse(package.exists())

    def test_serial_failure_is_archived_without_real_serial_and_blocks_retry(self):
        with tempfile.TemporaryDirectory(dir=TEMP_PARENT, prefix='canon42_full_test_') as temp:
            package = Path(temp)/'package'; package.mkdir()
            shutil.copyfile(f.GATE_ORIGINAL, package/f.GATE_NAME)
            shutil.copyfile(f.OUT/'Validacion_PC'/f.HELPER_NAME, package/f.HELPER_NAME)
            f.write(package/'PLAN_COMPLETO.json', self.plan)
            marker = package/'INTENTO_VALIDACION_COMPLETA.json'
            args = SimpleNamespace(**vars(self.args), full=True, port='SIMULATED_ONLY',
                max_seconds=1800, timeout=15., baud=115200)
            fake_serial = SimpleNamespace(Serial=Mock(side_effect=OSError('simulated serial failure')))
            def tiny_gate_copy(source, target):
                target.mkdir(); (target/'gate-proof.txt').write_text('fake copy only for failure path')
            with patch.multiple(f, HERE=package, PACKAGE=package, ATTEMPT=marker), \
                    patch.object(f, 'own_manifest', return_value='test-seal'), \
                    patch.object(f, 'load_context', return_value=self.context), \
                    patch.object(f.shutil, 'copytree', side_effect=tiny_gate_copy), \
                    patch.dict(sys.modules, serial=fake_serial), redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(OSError, 'simulated serial failure'):
                    f.run(args)
                report_path = next((package/'resultados').glob('*/RESULTADO_COMPLETO.json'))
                report = f.read(report_path)
                self.assertEqual(report['status'], 'FAILED_INCOMPLETE')
                self.assertFalse(report['full_validation_completed'])
                self.assertEqual(report['new_completed']['RAW_transmissions'], 0)
                self.assertEqual(report['imported_counts'], f.IMPORTED_COUNTS)
                evidence = f.read(report_path.parent/'MANIFEST_EVIDENCIAS_SHA256.json')
                self.assertIn('gate_importado/gate-proof.txt', evidence)
                self.assertTrue(marker.is_file())
                with self.assertRaisesRegex(ValueError, 'Ya hubo un intento FULL'):
                    f.run(args)
                fake_serial.Serial.assert_called_once()


class RootManifest(unittest.TestCase):
    def make_manifest(self, root, entries):
        manifest = root/'manifest.json'; f.write(manifest, entries)
        manifest.with_suffix('.sha256').write_text(f.sha(manifest)+'  '+manifest.name+'\n', encoding='ascii')
        return manifest

    def test_valid_manifest_and_pinned_digest_reject_modified_content(self):
        with tempfile.TemporaryDirectory(dir=TEMP_PARENT, prefix='canon42_full_test_') as temp:
            root = Path(temp); source = root/'source'; source.write_bytes(b'original')
            manifest = self.make_manifest(root, {'source': f.sha(source)})
            with patch.object(f, 'ROOT', root):
                entries, digest = f.verify_root_manifest(manifest)
                self.assertEqual(entries, {'source': f.sha(source)})
                source.write_bytes(b'altered')
                with self.assertRaisesRegex(ValueError, 'Archivo sellado modificado'):
                    f.verify_root_manifest(manifest, digest)

    def test_recomputed_manifest_is_rejected_against_fixed_original_digest(self):
        with tempfile.TemporaryDirectory(dir=TEMP_PARENT, prefix='canon42_full_test_') as temp:
            root = Path(temp); source = root/'source'; source.write_bytes(b'original')
            manifest = self.make_manifest(root, {'source': f.sha(source)})
            digest = f.sha(manifest)
            source.write_bytes(b'altered')
            manifest.write_text(json.dumps({'source': f.sha(source)}), encoding='utf-8')
            manifest.with_suffix('.sha256').write_text(f.sha(manifest)+'  '+manifest.name+'\n', encoding='ascii')
            with patch.object(f, 'ROOT', root), self.assertRaisesRegex(ValueError, 'Sello de procedencia modificado'):
                f.verify_root_manifest(manifest, digest)

    def test_manifest_cannot_escape_project_root(self):
        with tempfile.TemporaryDirectory(dir=TEMP_PARENT, prefix='canon42_full_test_') as temp:
            parent = Path(temp); root = parent/'root'; root.mkdir()
            outside = parent/'outside'; outside.write_bytes(b'outside')
            manifest = self.make_manifest(root, {'../outside': f.sha(outside)})
            with patch.object(f, 'ROOT', root), self.assertRaisesRegex(ValueError, 'Archivo sellado modificado'):
                f.verify_root_manifest(manifest)


if __name__ == '__main__':
    unittest.main()
