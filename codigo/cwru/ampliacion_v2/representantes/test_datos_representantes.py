"""Pruebas de roles/lectura: metadatos reales y matrices sintéticas, sin fit."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

import datos_representantes as data


class TrackingArray:
    """Caché sintético que registra exactamente las filas materializadas."""
    def __init__(self, size, bad_rows=()):
        self.dtype = np.dtype(np.float64)
        self.shape = (size, 14)
        self.rows_read = []
        self.bad_rows = set(bad_rows)

    def __getitem__(self, key):
        if not isinstance(key, int):
            raise AssertionError('Solo se autoriza seleccionar filas individuales')
        self.rows_read.append(key)
        result = np.full(14, key + .25, dtype=np.float64)
        if key in self.bad_rows:
            result[0] = np.nan
        return result


class RepresentativeDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol, cls.cost_plan, cls.rows, cls.hashes = data.load_verified_context()
        cls.jobs = data.build_jobs(cls.protocol, cls.cost_plan, cls.rows)

    def test_all_32_roles_exclude_outer_and_do_not_duplicate_work(self):
        self.assertEqual(len(self.jobs), 32)
        self.assertEqual(len({j['job_id'] for j in self.jobs}), 32)
        for job in self.jobs:
            tr, va = data.validate_job_scope(job, self.rows, self.protocol)
            outer, inner = job['outer_load_hp'], job['inner_validation_load_hp']
            self.assertEqual(inner, min(set(range(4)) - {outer}))
            self.assertEqual({int(r['load_hp']) for r in tr}, set(range(4)) - {outer, inner})
            self.assertEqual({int(r['load_hp']) for r in va}, {inner})
            self.assertFalse({r['file_id'] for r in tr} & {r['file_id'] for r in va})

    def test_expected_fold_sizes_are_independent_known_totals(self):
        self.assertEqual([(len(j['training_global_window_ids']), len(j['validation_global_window_ids']))
                          for j in self.jobs[::8]], [(812, 393), (812, 175), (799, 175), (799, 175)])

    def test_benchmark_first_last_and_identical_across_masks(self):
        for outer in range(4):
            jobs = self.jobs[outer * 8:(outer + 1) * 8]
            self.assertTrue(all(j['benchmark_global_window_ids'] == jobs[0]['benchmark_global_window_ids']
                                for j in jobs))
            bank = jobs[0]['benchmark_global_window_ids']
            self.assertEqual(len(bank), 28)
            for first, last in zip(bank[::2], bank[1::2]):
                rows = [r for r in self.rows if r['file_id'] == self.rows[first]['file_id']]
                self.assertEqual(self.rows[last]['file_id'], self.rows[first]['file_id'])
                self.assertEqual(int(self.rows[first]['window_index_in_file']),
                                 min(int(r['window_index_in_file']) for r in rows))
                self.assertEqual(int(self.rows[last]['window_index_in_file']),
                                 max(int(r['window_index_in_file']) for r in rows))

    def test_detector_fits_and_calibrates_disjoint_normal_records(self):
        for job in self.jobs:
            fit = [self.rows[i] for i in job['detector_fit_global_window_ids']]
            cal = [self.rows[i] for i in job['detector_calibration_global_window_ids']]
            self.assertEqual({r['class'] for r in fit + cal}, {'N'})
            self.assertEqual(len({r['file_id'] for r in fit}), 1)
            self.assertEqual(len({r['file_id'] for r in cal}), 1)
            self.assertFalse({r['file_id'] for r in fit} & {r['file_id'] for r in cal})
            self.assertEqual({int(r['load_hp']) for r in cal}, {max(job['roles']['training_loads'])})

    def test_reject_outer_row_inserted_in_training(self):
        job = deepcopy(self.jobs[0])
        job['training_global_window_ids'][0] = 0
        with self.assertRaises(ValueError):
            data.validate_job_scope(job, self.rows, self.protocol)

    def test_reject_validation_row_used_for_benchmark(self):
        job = deepcopy(self.jobs[0])
        job['benchmark_global_window_ids'][0] = job['validation_global_window_ids'][0]
        with self.assertRaises(ValueError):
            data.validate_job_scope(job, self.rows, self.protocol)

    def test_reject_training_record_used_for_detector_calibration_and_fit(self):
        job = deepcopy(self.jobs[0])
        job['detector_calibration_global_window_ids'] = job['detector_fit_global_window_ids'].copy()
        with self.assertRaises(ValueError):
            data.validate_job_scope(job, self.rows, self.protocol)

    def test_reject_changed_mask_seed_or_order(self):
        for key, value in [('seed', 42), ('feature_indices', [0, 1, 2]),
                           ('training_global_window_ids', list(reversed(self.jobs[0]['training_global_window_ids'])))]:
            job = deepcopy(self.jobs[0])
            job[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                data.validate_job_scope(job, self.rows, self.protocol)

    def test_reject_changed_or_missing_representative_plan(self):
        for change in ('missing', 'order', 'cost'):
            cost = deepcopy(self.cost_plan)
            if change == 'missing':
                cost['representatives'].pop()
            elif change == 'order':
                cost['representatives'][0], cost['representatives'][1] = cost['representatives'][1], cost['representatives'][0]
            else:
                cost['representatives'][0]['mean_ms'] = 0.
            with self.subTest(change=change), self.assertRaises(ValueError):
                data.build_jobs(self.protocol, cost, self.rows)

    def test_reject_duplicate_source_pairs_or_inconsistent_mat(self):
        for change in ('pair', 'mat'):
            rows = deepcopy(self.rows)
            if change == 'pair':
                rows[1]['source_row'] = rows[0]['source_row']
            else:
                rows[1]['load_hp'] = '1'
            with self.subTest(change=change), self.assertRaises(ValueError):
                data.build_jobs(self.protocol, self.cost_plan, rows)

    def test_reject_escaping_and_absolute_paths(self):
        for relative in ('../fuera.npy', data.ROOT / 'foo.npy'):
            with self.subTest(relative=str(relative)), self.assertRaises(ValueError):
                data.bounded_path(data.ROOT, relative)

    def test_reject_changed_protocol_hash(self):
        with patch.object(data, 'sha256', return_value='0' * 64), self.assertRaises(ValueError):
            data.load_verified_context()

    def _load_synthetic(self, job, bad=None):
        fake = {}
        for relative in job['authorized_feature_source_paths']:
            rows = [r for r in self.rows if r['source_features'] == relative]
            fake[relative] = TrackingArray(len(rows), (bad or {}).get(relative, ()))
        def loader(path, **kwargs):
            self.assertEqual(kwargs, dict(allow_pickle=False, mmap_mode='r'))
            return fake[Path(path).relative_to(data.ROOT).as_posix()]
        with patch.object(data, 'load_verified_context', return_value=(self.protocol, self.cost_plan, self.rows, self.hashes)), \
                patch.object(data, 'sha256', side_effect=lambda p: self.hashes.get(Path(p).relative_to(data.ROOT).as_posix(), 'index')), \
                patch.object(data.np, 'load', side_effect=loader):
            result = data.load_job_data(job)
        return result, fake

    def test_loader_selects_only_authorized_rows_for_all_four_outer_folds(self):
        for job in self.jobs[::8]:
            result, fake = self._load_synthetic(job)
            xtr, ytr, xva, yva, provenance = result
            self.assertEqual(xtr.dtype, np.float32)
            self.assertEqual(xva.dtype, np.float32)
            self.assertEqual(ytr.dtype, np.int64)
            self.assertEqual(yva.dtype, np.int64)
            selected = [self.rows[i] for i in job['training_global_window_ids'] + job['validation_global_window_ids']]
            for relative, array in fake.items():
                expected = [int(r['source_row']) for r in selected if r['source_features'] == relative]
                self.assertEqual(array.rows_read, expected)
                self.assertFalse(set(array.rows_read) & {int(r['source_row']) for r in self.rows
                                  if r['source_features'] == relative and int(r['load_hp']) == job['outer_load_hp']})
            self.assertFalse(provenance['outer_rows_selected_or_materialized'])
            self.assertTrue(provenance['mapping_or_file_hashing_is_not_OS_access_trace'])

    def test_loader_rejects_selected_nonfinite(self):
        job = self.jobs[0]
        row = self.rows[job['training_global_window_ids'][0]]
        with self.assertRaises(ValueError):
            self._load_synthetic(job, {row['source_features']: [int(row['source_row'])]})

    def test_loader_does_not_scan_unselected_outer_values(self):
        job = self.jobs[0]
        row = next(r for r in self.rows if int(r['load_hp']) == job['outer_load_hp'])
        result, _ = self._load_synthetic(job, {row['source_features']: [int(row['source_row'])]})
        self.assertTrue(np.isfinite(result[0]).all())
        self.assertTrue(np.isfinite(result[2]).all())


if __name__ == '__main__':
    unittest.main()
