"""Pruebas de contratos de reajuste; sin fit/predict ni valores CWRU reales."""
from __future__ import annotations

import copy
from contextlib import contextmanager
import importlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch, Mock

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SOURCE = Path(__file__).resolve().parent
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))
import contrato as contract
import worker_clasicos as classical
import worker_detector as detector
import worker_dnn as dnn
import ejecutar_reajustes as runner


@contextmanager
def fixture():
    base = (ROOT / 'tmp').resolve(); base.mkdir(exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix='final_refits_test_', dir=base)).resolve()
    try:
        yield folder
    finally:
        if folder.parent != base or not folder.name.startswith('final_refits_test_'):
            raise ValueError('Temporal fuera del proyecto autorizado')
        shutil.rmtree(folder)


def arrays_for(job, rows):
    """Datos inventados, usando solo IDs/etiquetas/MAT del índice sellado."""
    if job['family'] != 'detector':
        ids = job['training_global_window_ids']; n = len(ids)
        return dict(raw_training=np.arange(n * 14, dtype=np.float32).reshape(n, 14) / np.float32(100),
            y_training=np.asarray([int(rows[i]['label']) for i in ids], dtype=np.int64),
            training_ids=np.asarray(ids, dtype=np.int64),
            training_records=np.asarray([str(rows[i]['file_id']) for i in ids]))
    result = {}
    for scope, raw_key, y_key in [('fit', 'raw_fit', 'y_fit'),
                                ('calibration', 'raw_calibration', 'y_calibration')]:
        ids = job[scope + '_ids']; n = len(ids)
        result[raw_key] = np.arange(n * 14, dtype=np.float32).reshape(n, 14) / np.float32(100)
        result[y_key] = np.zeros(n, dtype=np.int64)
        result[scope + '_ids'] = np.asarray(ids, dtype=np.int64)
        result[scope + '_records'] = np.asarray([str(rows[i]['file_id']) for i in ids])
    return result


def provenance_for(job, rows):
    def scope(ids):
        return dict(global_window_ids=ids, selected_loads_hp=sorted({int(rows[i]['load_hp']) for i in ids}),
            selected_record_ids=sorted({str(rows[i]['file_id']) for i in ids}, key=int),
            shape=[len(ids), 14], loaded_dtype='float32', only_requested_rows_selected_or_materialized=True)
    if job['family'] != 'detector':
        return dict(training=scope(job['training_global_window_ids']))
    return {name: scope(job[name + '_ids']) for name in ('fit', 'calibration')}


def payload_for(job, protocol, rows):
    return dict(status='AUTHORIZED_FINAL_REFIT_PAYLOAD', family=job['family'], job=copy.deepcopy(job),
        provenance=provenance_for(job, rows), **{key: copy.deepcopy(protocol[key])
        for key in ('dnn', 'numeric_contract', 'quantization', 'detector')})


class RefitInventory(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Autenticar JSON/CSV/código no debe abrir NPZ/NPY ni leer sus valores.
        with patch.object(np, 'load', side_effect=AssertionError('Preparar inventario no lee arrays')):
            cls.scopes, cls.protocol, cls.rows, cls.hashes = contract.frozen_contract()
            cls.jobs = contract.build_jobs(cls.scopes, cls.protocol, cls.rows)

    def test_all_131_scopes_are_derived_without_reading_feature_values(self):
        self.assertEqual(len(self.jobs), 131)
        self.assertEqual(len({job['job_id'] for job in self.jobs}), 131)
        self.assertEqual({family: sum(j['family'] == family for j in self.jobs)
            for family in ('DNN', 'RF', 'SVM', 'detector')}, dict(DNN=80, RF=40, SVM=4, detector=7))
        for job in self.jobs:
            ids = job['training_global_window_ids']; outer = job['outer_load_hp']
            self.assertEqual(ids, sorted(set(ids)))
            self.assertNotIn(outer, {int(self.rows[i]['load_hp']) for i in ids})
            self.assertEqual(len({self.rows[i]['file_id'] for i in ids}), 21)
            self.assertEqual(job['roles']['evaluation_loads'], [outer])
            self.assertEqual(job['timeout_seconds'], contract.TIMEOUTS[job['family']])
            self.assertNotIn('validation_global_window_ids', job)

    def test_fixed_epochs_and_ten_seeds_preserve_the_frozen_choices(self):
        expected = {(0, 'F14'): 45, (0, 'T6'): 47, (1, 'F14'): 14, (1, 'FE7'): 27,
                    (2, 'F14'): 19, (2, 'T6'): 32, (3, 'F14'): 9, (3, 'NF12'): 47}
        grouped = {}
        for job in self.jobs:
            if job['family'] != 'DNN':
                continue
            key = job['outer_load_hp'], job['candidate']
            self.assertEqual(job['epochs'], expected[key])
            self.assertEqual(job['feature_indices'], self.protocol['candidates'][job['candidate']])
            grouped.setdefault(key, set()).add(job['seed'])
        self.assertEqual(set(grouped), set(expected))
        self.assertTrue(all(seeds == set(self.protocol['dnn']['outer_seeds']) for seeds in grouped.values()))
        self.assertEqual(sum(len(j['procedures']) for j in self.jobs if j['family'] == 'DNN'), 120)

    def test_detector_uses_two_fit_MAT_and_one_calibration_without_outer_or_faults(self):
        for job in [j for j in self.jobs if j['family'] == 'detector']:
            fit, cal = job['fit_ids'], job['calibration_ids']
            self.assertFalse(set(fit) & set(cal))
            self.assertTrue(set(fit + cal) <= set(job['training_global_window_ids']))
            self.assertTrue(all(int(self.rows[i]['label']) == 0 for i in fit + cal))
            self.assertEqual(len({self.rows[i]['file_id'] for i in fit}), 2)
            self.assertEqual(len({self.rows[i]['file_id'] for i in cal}), 1)
            self.assertNotIn(4, job['indices']); self.assertNotIn(11, job['indices'])
        aliases = [j for j in self.jobs if j['family'] == 'detector' and j['outer_load_hp'] == 3]
        self.assertEqual(len(aliases), 1)
        self.assertEqual(set(aliases[0]['masks']), {'F14', 'NF12'})

    def test_foreign_roles_or_missing_frozen_jobs_cannot_silently_change_inventory(self):
        changed = copy.deepcopy(self.scopes)
        changed['DNN_unique_jobs'][0]['roles']['training_loads'].append(0)
        with self.assertRaises(ValueError):
            contract.build_jobs(changed, self.protocol, self.rows)
        changed = copy.deepcopy(self.scopes); changed['RF_jobs'].pop()
        with self.assertRaises(ValueError):
            contract.build_jobs(changed, self.protocol, self.rows)

    def test_fit_identity_binds_selection_sources_versions_and_exact_scope(self):
        job = self.jobs[0]; versions = {'python': 'test'}; sources = {'source.py': 'a' * 64}
        baseline = contract.fit_signature(job, self.protocol, self.hashes, sources, versions)
        self.assertEqual(baseline, contract.fit_signature(copy.deepcopy(job), self.protocol,
            self.hashes, dict(sources), dict(versions)))
        for variant in ('epochs', 'seed'):
            changed = copy.deepcopy(job); changed[variant] += 1
            self.assertNotEqual(baseline, contract.fit_signature(changed, self.protocol,
                self.hashes, sources, versions))
        self.assertNotEqual(baseline, contract.fit_signature(job, self.protocol,
            self.hashes, sources, {'python': 'different'}))

    def test_materialize_requests_only_exact_authorized_ids_and_never_other_values(self):
        import datos_detector as loader
        for family in ('DNN', 'detector'):
            job = next(j for j in self.jobs if j['family'] == family)
            calls = []
            def fake_load(ids, context=None):
                calls.append(list(ids))
                self.assertTrue(all(int(self.rows[i]['load_hp']) != job['outer_load_hp'] for i in ids))
                y = np.asarray([int(self.rows[i]['label']) for i in ids], dtype=np.int64)
                records = [str(self.rows[i]['file_id']) for i in ids]
                return np.zeros((len(ids), 14), dtype=np.float32), y, records, {'global_window_ids': ids}
            with fixture() as folder, patch.object(loader, 'load_rows', side_effect=fake_load):
                contract.materialize(job, folder, (self.protocol, self.rows, self.hashes))
                with np.load(folder / 'datos.npz', allow_pickle=False) as saved:
                    self.assertEqual(set(saved.files), classical.ARRAY_KEYS if family == 'DNN' else detector.ARRAY_KEYS)
            expected = [job['training_global_window_ids']] if family == 'DNN' else [job['fit_ids'], job['calibration_ids']]
            self.assertEqual(calls, expected)


class WorkerGuards(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scopes, cls.protocol, cls.rows, _ = contract.frozen_contract()
        cls.jobs = contract.build_jobs(cls.scopes, cls.protocol, cls.rows)

    def sample(self, family):
        job = next(j for j in self.jobs if j['family'] == family)
        return payload_for(job, self.protocol, self.rows), arrays_for(job, self.rows)

    def test_all_families_accept_synthetic_data_matching_real_metadata_without_fitting(self):
        for family, module in [('DNN', dnn), ('RF', classical), ('SVM', classical), ('detector', detector)]:
            payload, arrays = self.sample(family)
            module.validate_payload(payload, arrays)
        self.assertNotIn('tensorflow', sys.modules)
        self.assertNotIn('sklearn', sys.modules)
        self.assertNotIn('joblib', sys.modules)

    def test_nonfinite_wrong_dtypes_labels_ids_and_extra_arrays_are_rejected(self):
        for family, module in [('DNN', dnn), ('RF', classical), ('SVM', classical), ('detector', detector)]:
            payload, arrays = self.sample(family)
            raw_key = 'raw_fit' if family == 'detector' else 'raw_training'
            label_key = 'y_fit' if family == 'detector' else 'y_training'
            ids_key = 'fit_ids' if family == 'detector' else 'training_ids'
            mutations = []
            bad = copy.deepcopy(arrays); bad[raw_key][0, 0] = np.nan; mutations.append(bad)
            bad = copy.deepcopy(arrays); bad[raw_key] = bad[raw_key].astype(np.float64); mutations.append(bad)
            bad = copy.deepcopy(arrays); bad[label_key][0] = 1 if family == 'detector' else 3; mutations.append(bad)
            bad = copy.deepcopy(arrays); bad[ids_key][0] = -1; mutations.append(bad)
            bad = copy.deepcopy(arrays); bad['raw_outer_test'] = np.zeros((1, 14), np.float32); mutations.append(bad)
            for index, changed in enumerate(mutations):
                with self.subTest(family=family, mutation=index), self.assertRaises(ValueError):
                    module.validate_payload(payload, changed)

    def test_outer_load_cannot_enter_worker_training_roles(self):
        for family, module in [('DNN', dnn), ('RF', classical), ('SVM', classical), ('detector', detector)]:
            payload, arrays = self.sample(family)
            payload['job']['roles']['training_loads'].append(payload['job']['outer_load_hp'])
            with self.subTest(family=family), self.assertRaises(ValueError):
                module.validate_payload(payload, arrays)

    def test_boolean_roles_are_not_accepted_as_integer_loads(self):
        for family, module in [('DNN', dnn), ('RF', classical), ('SVM', classical), ('detector', detector)]:
            payload, arrays = self.sample(family)
            payload['job']['roles']['training_loads'][0] = True
            with self.subTest(family=family), self.assertRaises(ValueError):
                module.validate_payload(payload, arrays)

    def test_DNN_rejects_unfixed_epochs_missing_classes_or_invalid_seed(self):
        for key, value in [('epochs', 0), ('epochs', 51), ('epochs', True), ('seed', 123), ('seed', True)]:
            payload, arrays = self.sample('DNN'); payload['job'][key] = value
            with self.subTest(field=key, value=value), self.assertRaises(ValueError):
                dnn.validate_payload(payload, arrays)
        payload, arrays = self.sample('DNN'); arrays['y_training'][:] = 0
        with self.assertRaises(ValueError):
            dnn.validate_payload(payload, arrays)

    def test_detector_rejects_overlap_faults_frequencies_and_calibration_refit(self):
        payload, arrays = self.sample('detector')
        payload['job']['indices'].append(4)
        with self.assertRaises(ValueError):
            detector.validate_payload(payload, arrays)
        payload, arrays = self.sample('detector')
        payload['job']['roles']['detector_refit_after_calibration'] = True
        with self.assertRaises(ValueError):
            detector.validate_payload(payload, arrays)
        payload, arrays = self.sample('detector')
        arrays['calibration_records'][:] = arrays['fit_records'][0]
        with self.assertRaises(ValueError):
            detector.validate_payload(payload, arrays)

    def test_classical_preprocessing_and_native_SVM_rule_cannot_be_replaced(self):
        payload, arrays = self.sample('RF'); payload['job']['params']['n_jobs'] = -1
        with self.assertRaises(ValueError):
            classical.validate_payload(payload, arrays)
        payload, arrays = self.sample('SVM'); payload['job']['params']['probability'] = True
        with self.assertRaises(ValueError):
            classical.validate_payload(payload, arrays)


class WorkerAuthorization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scopes, cls.protocol, cls.rows, _ = contract.frozen_contract()
        cls.job = next(j for j in contract.build_jobs(cls.scopes, cls.protocol, cls.rows) if j['family'] == 'RF')

    @contextmanager
    def authorized_fixture(self):
        with fixture() as folder:
            output_root = folder / 'campaigns'; campaign = output_root / 'run001'
            request_folder = campaign / 'solicitudes' / self.job['job_id']
            request_folder.mkdir(parents=True)
            manifest = folder / 'MANIFEST.json'; manifest.write_text('{}\n', encoding='utf-8')
            plan_path = folder / 'PLAN.json'; plan_path.write_text('{}\n', encoding='utf-8')
            data_path = request_folder / 'datos.npz'; data_path.write_bytes(b'synthetic-signed-input')
            payload_path = request_folder / 'payload.json'
            output = campaign / self.job['job_id']; signature = 'a' * 64
            plan = dict(jobs=[self.job], fit_signatures={self.job['job_id']: signature},
                source_sha256={'synthetic.py': 'b' * 64}, expected_versions={'python': 'synthetic'},
                worker_environment={})
            payload = payload_for(self.job, self.protocol, self.rows)
            payload.update(arrays_path=str(data_path), arrays_sha256=contract.io.sha(data_path),
                fit_signature=signature, selection_signature=contract.SELECTION_SIGNATURE,
                source_sha256=plan['source_sha256'], expected_versions=plan['expected_versions'], worker_environment={})
            contract.io.write(payload_path, payload)
            with patch.object(contract, 'OUTPUT', output_root), patch.object(contract, 'MANIFEST', manifest), \
                 patch.object(contract, 'PLAN', plan_path), patch.object(contract, 'preflight', return_value=plan):
                contract.io.write(output_root / 'CAMPANA_UNICA.json',
                    dict(run_id=campaign.name, preparation_sha256=contract.io.sha(manifest)))
                contract.io.write(campaign / 'CONTRATO.json', contract.campaign_contract(plan))
                active = campaign / 'ACTIVO.lock'; active.write_text('test-parent-active\n', encoding='utf-8')
                request = dict(job_id=self.job['job_id'], fit_signature=signature,
                    payload_path=payload_path.relative_to(ROOT).as_posix())
                request_path = campaign / 'SOLICITUD_ACTIVA.json'; contract.io.write(request_path, request)
                yield payload_path, output, payload, data_path, request_path, active

    def test_matching_request_is_authorized_without_importing_or_running_models(self):
        with self.authorized_fixture() as (path, output, payload, *_):
            self.assertEqual(contract.authorize(path, output, 'RF'), payload)

    def test_missing_request_or_tampered_arrays_block_before_worker_fit(self):
        with self.authorized_fixture() as (path, output, _, data, request, _):
            request.unlink()
            with self.assertRaises((ValueError, FileNotFoundError)):
                contract.authorize(path, output, 'RF')
        with self.authorized_fixture() as (path, output, _, data, *_):
            data.write_bytes(b'tampered')
            with self.assertRaisesRegex(ValueError, 'Arrays'):
                contract.authorize(path, output, 'RF')

    def test_wrong_family_changed_fit_signature_or_existing_output_are_rejected(self):
        with self.authorized_fixture() as (path, output, payload, *_):
            with self.assertRaises(ValueError):
                contract.authorize(path, output, 'SVM')
            changed = copy.deepcopy(payload); changed['fit_signature'] = 'c' * 64
            path.write_text(json.dumps(changed), encoding='utf-8')
            with self.assertRaises(ValueError):
                contract.authorize(path, output, 'RF')
        with self.authorized_fixture() as (path, output, *_):
            output.mkdir()
            with self.assertRaises(ValueError):
                contract.authorize(path, output, 'RF')

    def test_worker_requires_active_parent_lock(self):
        with self.authorized_fixture() as (path, output, _, _, _, active):
            active.unlink()
            with self.assertRaises((ValueError, FileNotFoundError)):
                contract.authorize(path, output, 'RF')


class CampaignSafety(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scopes, cls.protocol, cls.rows, cls.hashes = contract.frozen_contract()
        cls.jobs = contract.build_jobs(cls.scopes, cls.protocol, cls.rows)

    @contextmanager
    def campaign(self, jobs=None):
        with fixture() as folder:
            output = folder / 'campaigns'; campaign = output / 'run001'
            (campaign / 'pasos').mkdir(parents=True)
            (campaign / 'solicitudes').mkdir()
            (campaign / 'lotes').mkdir()
            manifest = folder / 'MANIFEST.json'; manifest.write_text('{}\n', encoding='utf-8')
            manifest.with_suffix('.sha256').write_text('synthetic seal\n', encoding='ascii')
            selected = self.jobs if jobs is None else jobs
            plan = dict(jobs=selected, worker_environment={}, source_sha256={'test.py': 'a' * 64},
                expected_versions={'python': 'synthetic'},
                fit_signatures={j['job_id']: 'b' * 64 for j in selected})
            fixed_contract = {'synthetic_campaign': True}
            contract.io.write(campaign / 'CONTRATO.json', fixed_contract)
            contract.io.write(output / 'CAMPANA_UNICA.json',
                dict(run_id=campaign.name, preparation_sha256=contract.io.sha(manifest)))
            with patch.object(contract, 'OUTPUT', output), patch.object(contract, 'MANIFEST', manifest), \
                 patch.object(contract, 'campaign_contract', return_value=fixed_contract), \
                 patch.object(contract, 'preflight', return_value=plan), \
                 patch.object(contract.io, 'verify_manifest', return_value={}), \
                 patch.object(contract.io, 'write_manifest'), \
                 patch.object(contract.inner, 'context', return_value=(self.protocol, self.rows, self.hashes)):
                yield campaign, plan

    def write_step(self, campaign, index, job, seconds=10., status='PASS'):
        state = dict(status=status, job_id=job['job_id'], automatic_retry=False,
            timeout_seconds=job['timeout_seconds'], execution_verification_seconds=seconds)
        (campaign / 'pasos' / f'{index:03d}.json').write_text(json.dumps(state), encoding='utf-8')
        return state

    def test_timeout_budget_boundary_reserves_the_entire_worker_timeout(self):
        self.assertTrue(runner.can_start(300, 420., 720))
        self.assertFalse(runner.can_start(300, 420.000001, 720))
        self.assertFalse(runner.can_start(300, 720., 720))
        self.assertTrue(runner.can_start(30, 0., 30))

    def test_completed_prefix_is_replayed_and_consumed_time_not_reset(self):
        with self.campaign(self.jobs[:3]) as (campaign, plan), \
             patch.object(runner, 'verify_job', return_value={'status': 'PASS'}) as verifier:
            self.write_step(campaign, 0, plan['jobs'][0], 12.25)
            self.write_step(campaign, 1, plan['jobs'][1], 19.75)
            done, seconds = runner.inspect_campaign(campaign, plan)
            self.assertEqual(done, [j['job_id'] for j in plan['jobs'][:2]])
            self.assertEqual(seconds, 32.)
            self.assertEqual(verifier.call_count, 2)

    def test_gap_failed_state_or_partial_folder_cannot_be_skipped(self):
        for variant in ('gap', 'failed', 'partial', 'nan_time'):
            with self.subTest(variant=variant), self.campaign(self.jobs[:3]) as (campaign, plan), \
                 patch.object(runner, 'verify_job'):
                if variant == 'gap':
                    self.write_step(campaign, 1, plan['jobs'][1])
                elif variant == 'failed':
                    self.write_step(campaign, 0, plan['jobs'][0], status='FAIL')
                elif variant == 'partial':
                    (campaign / plan['jobs'][0]['job_id']).mkdir()
                else:
                    self.write_step(campaign, 0, plan['jobs'][0], seconds=float('nan'))
                with self.assertRaises(ValueError):
                    runner.inspect_campaign(campaign, plan)

    def test_worker_timeout_kills_process_keeps_logs_and_does_not_retry(self):
        with self.campaign(self.jobs[:1]) as (campaign, plan):
            process = Mock()
            process.wait.side_effect = [subprocess.TimeoutExpired('synthetic-worker', 300), -9]
            process.poll.return_value = None; process.returncode = -9
            with patch.object(runner, 'payload_for', return_value={'synthetic': True}), \
                 patch.object(runner.subprocess, 'Popen', return_value=process) as popen, \
                 patch.object(runner, 'verify_job') as verifier:
                state = runner.launch_one(campaign, plan['jobs'][0], plan, None)
            self.assertEqual(state['status'], 'FAIL'); self.assertEqual(state['error'], 'TIMEOUT')
            self.assertFalse(state['automatic_retry'])
            popen.assert_called_once(); process.kill.assert_called_once(); verifier.assert_not_called()
            self.assertFalse((campaign / 'SOLICITUD_ACTIVA.json').exists())
            self.assertTrue((campaign / (plan['jobs'][0]['job_id'] + '_stdout.log')).is_file())
            self.assertTrue((campaign / 'solicitudes' / plan['jobs'][0]['job_id'] / 'payload.json').is_file())

    def test_worker_launch_failure_preserves_request_and_stops_for_audit(self):
        with self.campaign(self.jobs[:1]) as (campaign, plan), \
             patch.object(runner, 'payload_for', return_value={'synthetic': True}), \
             patch.object(runner.subprocess, 'Popen', side_effect=OSError('synthetic failure')) as popen:
            state = runner.launch_one(campaign, plan['jobs'][0], plan, None)
            self.assertEqual(state['status'], 'FAIL'); self.assertIn('OSError', state['error'])
            self.assertFalse(state['automatic_retry']); popen.assert_called_once()
            self.assertTrue((campaign / 'solicitudes' / plan['jobs'][0]['job_id'] / 'payload.json').is_file())
            self.assertFalse((campaign / 'SOLICITUD_ACTIVA.json').exists())

    def test_first_failed_job_stops_queue_and_next_invocation_cannot_retry_it(self):
        with self.campaign() as (campaign, plan), \
             patch.object(runner, 'inspect_campaign', return_value=([], 0.)), \
             patch.object(runner, 'launch_one', return_value=dict(status='FAIL',
                job_id=plan['jobs'][0]['job_id'], family='DNN', automatic_retry=False,
                execution_verification_seconds=3., timeout_seconds=300, error='SYNTHETIC_FAIL')) as launch:
            self.assertEqual(runner.execute(max_new_jobs=131, max_seconds=7200), 1)
            launch.assert_called_once()
            summary_file = next((campaign / 'lotes').glob('*/RESUMEN_LOTE.json'))
            summary = contract.io.read(summary_file)
            self.assertEqual(summary['status'], 'STOPPED_FOR_AUDIT')
            self.assertEqual(summary['completed_jobs'], 0)
            self.assertFalse(summary['outer_prediction_release_enabled'])
            with self.assertRaises(ValueError):
                runner.execute(max_new_jobs=131, max_seconds=7200)
            launch.assert_called_once()
            self.assertFalse((campaign / 'ACTIVO.lock').exists())

    def test_active_request_or_existing_lock_blocks_another_execution(self):
        for variant in ('request', 'lock'):
            with self.subTest(variant=variant), self.campaign() as (campaign, _), \
                 patch.object(runner, 'launch_one') as launch:
                marker = campaign / ('SOLICITUD_ACTIVA.json' if variant == 'request' else 'ACTIVO.lock')
                marker.write_text('{}\n', encoding='utf-8')
                with self.assertRaises((ValueError, FileExistsError)):
                    runner.execute(max_new_jobs=1, max_seconds=600)
                launch.assert_not_called()


class PreparedPlanSafety(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.context = contract.frozen_contract()
        cls.scopes, cls.protocol, cls.rows, cls.hashes = cls.context
        cls.jobs = contract.build_jobs(cls.scopes, cls.protocol, cls.rows)

    def plan(self):
        source_hashes = {'source.py': 'a' * 64}; versions = {'python': 'synthetic'}
        return dict(jobs=self.jobs, source_sha256=source_hashes,
            selection_signature=contract.SELECTION_SIGNATURE, limits=copy.deepcopy(contract.LIMITS),
            worker_environment=contract.io.ENVIRONMENT, outer_evaluation_enabled=False,
            manuscript_edits_enabled=False, automatic_retry=False, early_stopping=False,
            expected_versions=versions, fit_signatures={j['job_id']: contract.fit_signature(j,
                self.protocol, self.hashes, source_hashes, versions) for j in self.jobs})

    def test_preflight_rejects_changed_sources_limits_or_fit_signature(self):
        original = self.plan()
        for variant in ('source', 'limits', 'fit_signature', 'outer_release', 'environment'):
            changed = copy.deepcopy(original)
            if variant == 'source': changed['source_sha256']['source.py'] = 'b' * 64
            elif variant == 'limits': changed['limits']['max_new_jobs'] = 132
            elif variant == 'fit_signature': changed['fit_signatures'][self.jobs[0]['job_id']] = 'c' * 64
            elif variant == 'outer_release': changed['outer_evaluation_enabled'] = True
            with self.subTest(variant=variant), \
                 patch.object(contract.io, 'verify_manifest'), \
                 patch.object(contract.io, 'read', return_value=changed), \
                 patch.object(contract, 'frozen_contract', return_value=self.context), \
                 patch.object(contract, 'sources', return_value=original['source_sha256']), \
                 patch.object(contract, 'versions', return_value={'python': 'different'}):
                with self.assertRaises(ValueError):
                    contract.preflight(environment=variant == 'environment')
if __name__ == '__main__':
    unittest.main(verbosity=2)
