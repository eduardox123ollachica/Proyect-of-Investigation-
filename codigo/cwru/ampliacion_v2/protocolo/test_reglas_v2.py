"""Pruebas de contrato con datos sinteticos; no entrena ni abre CWRU."""
from __future__ import annotations

import copy
import unittest

import numpy as np

from reglas_v2 import (component_roles, final_epoch_count, select_cost,
                       select_performance, weighted_p95)

CANDIDATES = ['F14', 'T6', 'S8', 'NF12', 'NM10', 'NE12', 'DE7', 'FE7']
INNER = [0, 1, 2]
SEEDS = [11, 22, 33]


def fixture():
    dnn = [dict(candidate=name, inner_validation_load=load, seed=seed,
                status='PASS', macro_f1_int8=.94 if name != 'F14' else .95)
           for name in CANDIDATES for load in INNER for seed in SEEDS]
    detector = [dict(candidate=name, inner_validation_load=load, status='PASS',
                     normal_spec=.98, ir_recall=.96, or_recall=.96)
                for name in CANDIDATES for load in INNER]
    costs = {name: dict(status='VALIDATED_TOTAL_COST', mean_ms=100., lower_ms=99., upper_ms=101.,
                        outer_load=3, benchmark_loads=[1, 2]) for name in CANDIDATES}
    return dnn, detector, costs


def score(dnn, name, value):
    for row in dnn:
        if row['candidate'] == name:
            row['macro_f1_int8'] = value


class RolesAndCalibrationTests(unittest.TestCase):
    def test_inner_roles(self):
        r = component_roles(3, 0)
        self.assertEqual(r['dnn_fit_loads'], [1, 2])
        self.assertEqual(r['int8_calibration_loads'], [1, 2])
        self.assertEqual(r['detector_fit_normal_loads'], [1])
        self.assertEqual(r['detector_calibration_normal_loads'], [2])
        self.assertEqual(r['evaluation_loads'], [0])
        self.assertFalse(r['detector_refit_after_calibration'])

    def test_outer_roles(self):
        r = component_roles(3)
        self.assertEqual(r['dnn_fit_loads'], [0, 1, 2])
        self.assertEqual(r['detector_fit_normal_loads'], [0, 1])
        self.assertEqual(r['detector_calibration_normal_loads'], [2])
        self.assertEqual(r['evaluation_loads'], [3])

    def test_all_roles_exclude_evaluation_from_fit_and_calibration(self):
        for outer in range(4):
            for inner in [None] + [v for v in range(4) if v != outer]:
                r = component_roles(outer, inner)
                for key in ('dnn_fit_loads', 'int8_calibration_loads',
                            'detector_fit_normal_loads', 'detector_calibration_normal_loads'):
                    self.assertTrue(set(r[key]).isdisjoint(r['evaluation_loads']))
                self.assertTrue(set(r['detector_fit_normal_loads']).isdisjoint(r['detector_calibration_normal_loads']))

    def test_illegal_roles_rejected(self):
        for args in ((3, 3), (4, None), (True, None), (0, 4), (0, '1')):
            with self.subTest(args=args), self.assertRaises(ValueError):
                component_roles(*args)

    def test_p95_observed_boundary_and_strict_tie(self):
        t = weighted_p95(list(range(20)), ['a'] * 20)
        self.assertIsInstance(t, np.float32)
        self.assertEqual(float(t), 18.)
        self.assertFalse(np.float32(18.) > t)
        self.assertTrue(np.float32(19.) > t)

    def test_records_have_equal_mass(self):
        # La cuantila por ventana seria 0; el segundo registro tiene masa 1/2.
        self.assertEqual(float(weighted_p95([0.] * 100 + [10.], ['a'] * 100 + ['b'])), 10.)

    def test_p95_permutation_and_ties(self):
        scores = [1., 1., 2., 3., 9.]
        ids = ['a', 'a', 'a', 'b', 'b']
        first = weighted_p95(scores, ids)
        second = weighted_p95(scores[::-1], ids[::-1])
        self.assertEqual(first.tobytes(), second.tobytes())
        self.assertEqual(float(weighted_p95([2.] * 7, ['a'] * 7)), 2.)

    def test_invalid_calibration_rejected(self):
        pairs = [([], []), ([1.], []), ([-1.], ['a']), ([np.nan], ['a']),
                 ([np.inf], ['a']), ([1e100], ['a']), ([[1.]], ['a']),
                 ([1.], [None]), ([1.], ['']), ([True], ['a'])]
        for scores, ids in pairs:
            with self.subTest(scores=scores, ids=ids), self.assertRaises(ValueError):
                weighted_p95(scores, ids)

    def test_epoch_median_uses_nine_internal_checkpoints(self):
        self.assertEqual(final_epoch_count([1, 2, 3, 4, 5, 6, 7, 8, 50]), 5)
        for values in ([2] * 8, [0] * 9, [51] * 9, [True] * 9):
            with self.subTest(values=values), self.assertRaises(ValueError):
                final_epoch_count(values)


class SelectionTests(unittest.TestCase):
    def test_performance_uses_mean_not_ensemble_or_cost(self):
        dnn, detector, _ = fixture()
        score(dnn, 'T6', .96)
        answer = select_performance(dnn, detector, CANDIDATES, INNER, SEEDS)
        self.assertEqual(answer['selected_candidate'], 'T6')
        self.assertFalse(answer['probability_ensemble_used'])
        self.assertFalse(answer['latency_or_dimension_used_for_performance_tie'])

    def test_catalog_breaks_exact_performance_tie(self):
        dnn, detector, _ = fixture()
        for name in CANDIDATES:
            score(dnn, name, .95)
        self.assertEqual(select_performance(dnn, detector, CANDIDATES, INNER, SEEDS)['selected_candidate'], 'F14')

    def test_worst_load_breaks_mean_tie(self):
        dnn, detector, _ = fixture()
        score(dnn, 'F14', .94)
        score(dnn, 'T6', .95)
        score(dnn, 'S8', .95)
        for row in dnn:
            if row['candidate'] == 'T6':
                row['macro_f1_int8'] = {0: .93, 1: .96, 2: .96}[row['inner_validation_load']]
        self.assertEqual(select_performance(dnn, detector, CANDIDATES, INNER, SEEDS)['selected_candidate'], 'S8')

    def test_detector_constraint_is_shared_by_both_procedures(self):
        dnn, detector, costs = fixture()
        score(dnn, 'T6', .99)
        for row in detector:
            if row['candidate'] == 'T6':
                row['normal_spec'] = .90
        costs['T6'].update(mean_ms=30., lower_ms=29., upper_ms=31.)
        self.assertEqual(select_performance(dnn, detector, CANDIDATES, INNER, SEEDS)['selected_candidate'], 'F14')
        result = select_cost(dnn, detector, CANDIDATES, INNER, SEEDS, costs, outer_load=3)
        self.assertNotIn('T6', result['predictive_eligible_candidates'])

    def test_numeric_tie_uses_worst_load_without_reordering_real_differences(self):
        dnn, detector, _ = fixture()
        score(dnn, 'T6', .95)
        score(dnn, 'S8', .95 + 5e-13)
        # Inside numerical tolerance, the catalogue resolves identical worst
        # loads. A genuine score difference beyond tolerance remains ordered.
        self.assertEqual(select_performance(dnn, detector, CANDIDATES, INNER, SEEDS)['selected_candidate'], 'F14')
        score(dnn, 'S8', .95 + 3e-12)
        self.assertEqual(select_performance(dnn, detector, CANDIDATES, INNER, SEEDS)['selected_candidate'], 'S8')

    def test_cost_benchmark_contract_for_every_outer_fold(self):
        for outer in range(4):
            dnn, detector, costs = fixture()
            loads = sorted(set(range(4)) - {outer})
            mapping = dict(zip(INNER, loads))
            for row in dnn + detector:
                row['inner_validation_load'] = mapping[row['inner_validation_load']]
            benchmark = loads[1:]
            for row in costs.values():
                row.update(outer_load=outer, benchmark_loads=benchmark)
            self.assertFalse(select_cost(dnn, detector, CANDIDATES, loads, SEEDS, costs, outer_load=outer)['goal_met'])
            costs['T6']['benchmark_loads'] = loads[:2]
            with self.assertRaises(ValueError):
                select_cost(dnn, detector, CANDIDATES, loads, SEEDS, costs, outer_load=outer)

    def test_incomplete_duplicate_failed_or_outside_rows_rejected(self):
        dnn, detector, _ = fixture()
        variants = [dnn[:-1], dnn + [dnn[0]], copy.deepcopy(dnn), copy.deepcopy(dnn)]
        variants[2][0]['status'] = 'ERROR'
        variants[3][0]['inner_validation_load'] = 3
        for rows in variants:
            with self.assertRaises(ValueError):
                select_performance(rows, detector, CANDIDATES, INNER, SEEDS)
        with self.assertRaises(ValueError):
            select_performance(dnn, detector[:-1], CANDIDATES, INNER, SEEDS)

    def test_input_order_does_not_change_selection(self):
        dnn, detector, _ = fixture()
        score(dnn, 'T6', .96)
        a = select_performance(dnn, detector, CANDIDATES, INNER, SEEDS)
        b = select_performance(dnn[::-1], detector[::-1], CANDIDATES, INNER[::-1], SEEDS[::-1])
        self.assertEqual(a, b)

    def test_lower_total_cost_candidate_selected(self):
        dnn, detector, costs = fixture()
        score(dnn, 'T6', .955)
        score(dnn, 'S8', .96)
        costs['T6'].update(mean_ms=50., lower_ms=49., upper_ms=51.)
        costs['S8'].update(mean_ms=60., lower_ms=59., upper_ms=61.)
        answer = select_cost(dnn, detector, CANDIDATES, INNER, SEEDS, costs, outer_load=3)
        self.assertEqual(answer['selected_candidate'], 'T6')
        self.assertEqual(answer['performance_selected_candidate'], 'S8')
        self.assertTrue(answer['goal_met'])

    def test_no_target_falls_back_to_reference(self):
        dnn, detector, costs = fixture()
        score(dnn, 'S8', .96)
        answer = select_cost(dnn, detector, CANDIDATES, INNER, SEEDS, costs, outer_load=3)
        self.assertEqual(answer['selected_candidate'], 'F14')
        self.assertFalse(answer['goal_met'])
        self.assertEqual(answer['reason'], 'no_candidate_meets_total_cost_target')

    def test_missing_proxy_null_or_mislabelled_total_cost_rejected(self):
        dnn, detector, original = fixture()
        variants = [copy.deepcopy(original) for _ in range(5)]
        del variants[0]['T6']
        variants[1]['T6']['mean_ms'] = None
        variants[2]['T6']['is_proxy'] = True
        variants[3]['T6']['cost_scope'] = 'feature_extraction_only'
        variants[4]['T6']['status'] = 'UNVALIDATED'
        for costs in variants:
            with self.assertRaises(ValueError):
                select_cost(dnn, detector, CANDIDATES, INNER, SEEDS, costs, outer_load=3)

    def test_outer_benchmark_leakage_rejected(self):
        dnn, detector, costs = fixture()
        costs['T6']['benchmark_loads'] = [1, 3]
        with self.assertRaises(ValueError):
            select_cost(dnn, detector, CANDIDATES, INNER, SEEDS, costs, outer_load=3)
        costs['T6']['benchmark_loads'] = [1, 2]
        costs['T6']['outer_load'] = 2
        with self.assertRaises(ValueError):
            select_cost(dnn, detector, CANDIDATES, INNER, SEEDS, costs, outer_load=3)

    def test_cost_benchmark_excludes_representative_validation_load(self):
        dnn, detector, costs = fixture()
        # Sigue dentro de desarrollo, pero 0 es la validacion del representante.
        costs['T6']['benchmark_loads'] = [0, 1]
        with self.assertRaises(ValueError):
            select_cost(dnn, detector, CANDIDATES, INNER, SEEDS, costs, outer_load=3)

    def test_invalid_envelope_rejected(self):
        dnn, detector, costs = fixture()
        for value in (-1., 0., np.nan, 101.):
            changed = copy.deepcopy(costs)
            changed['T6']['lower_ms'] = value
            with self.assertRaises(ValueError):
                select_cost(dnn, detector, CANDIDATES, INNER, SEEDS, changed, outer_load=3)

    def test_overlapping_envelopes_use_performance_tie_break(self):
        dnn, detector, costs = fixture()
        score(dnn, 'T6', .955)
        score(dnn, 'S8', .96)
        costs['T6'].update(mean_ms=50., lower_ms=49., upper_ms=51.)
        costs['S8'].update(mean_ms=53., lower_ms=50., upper_ms=55.)
        answer = select_cost(dnn, detector, CANDIDATES, INNER, SEEDS, costs, outer_load=3)
        self.assertEqual(answer['overlapping_cost_shortlist'], ['T6', 'S8'])
        self.assertEqual(answer['selected_candidate'], 'S8')

    def test_margin_also_preserves_performance_comparator(self):
        dnn, detector, costs = fixture()
        score(dnn, 'S8', .98)
        score(dnn, 'T6', .955)
        costs['T6'].update(mean_ms=50., lower_ms=49., upper_ms=51.)
        answer = select_cost(dnn, detector, CANDIDATES, INNER, SEEDS, costs, outer_load=3)
        self.assertNotIn('T6', answer['predictive_eligible_candidates'])
        self.assertEqual(answer['selected_candidate'], 'F14')


if __name__ == '__main__':
    unittest.main(verbosity=2)
