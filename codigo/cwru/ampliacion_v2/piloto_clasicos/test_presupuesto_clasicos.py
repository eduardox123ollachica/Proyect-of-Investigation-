"""Pruebas sinteticas del presupuesto, sin sklearn, TensorFlow o archivos."""
import copy
import math
import unittest

from presupuesto_clasicos import (CLASSICAL_STAGES, DETECTOR_CANDIDATES,
                                  project_classical_budget)


def fixture():
    rows = [dict(status='PASS', family=family, config_index=i, job_id=f'{family}_{i}',
        training_windows=799, validation_windows=175,
        timing_nonoverlapping_seconds={name: 1. + i / 10. for name in CLASSICAL_STAGES},
        process_elapsed_seconds=5. * (1. + i / 10.) + .5)
        for family in ('RF', 'SVM') for i in range(8)]
    detector = dict(status='PASS', fit_windows=58, calibration_windows=58,
        validation_windows=175, unique_representations=7,
        timing_nonoverlapping_seconds=dict(imports=1., data_and_scaler=2., artifact_IO=.5),
        rows=[dict(candidate=name, rep=rep, status='PASS', fit_calibration_seconds=.1,
                   inference_seconds=.05) for name in DETECTOR_CANDIDATES for rep in range(3)],
        process_elapsed_seconds=3.5 + 21 * .15 + .2)
    return rows, detector


class ClassicalBudgetTests(unittest.TestCase):
    def test_counts_do_not_repeat_SVM_or_detector_by_DNN_seed(self):
        result = project_classical_budget(*fixture())
        self.assertEqual(result['RF']['projected_total_fits'], 328)
        self.assertEqual(result['SVM']['projected_total_fits'], 100)
        self.assertEqual(result['detector']['projected_total_fits_before_cache'], 108)
        self.assertFalse(result['detector']['repeated_per_DNN_seed'])
        self.assertEqual(result['detector']['pilot_fit_calibration_repetitions'], 21)
        self.assertFalse(result['DNN_included'])
        self.assertEqual(result['pilot_scope']['additional_RF_fits'], 8)

    def test_totals_preserve_seconds_and_sum_every_stage_once(self):
        r = project_classical_budget(*fixture())
        for family, count, outer_count in (('RF', 36, 40), ('SVM', 12, 4)):
            f = r[family]
            inner = math.fsum(p['inner']['seconds_per_fit'] * count for p in f['configuration_profiles'].values())
            outer = max(p['outer']['seconds_per_fit'] for p in f['configuration_profiles'].values()) * outer_count
            self.assertAlmostEqual(f['projected_stage_seconds'], inner + outer)
            self.assertEqual(f['outer_sensitivity_source_config_index'], 7)
        total = math.fsum(r[k]['projected_stage_seconds'] for k in ('RF', 'SVM', 'detector'))
        self.assertAlmostEqual(r['aggregate']['projected_stage_seconds'], total)
        self.assertAlmostEqual(r['aggregate']['projected_stage_hours'], total / 3600.)
        self.assertGreater(r['aggregate']['projected_with_observed_process_gap_seconds'], total)

    def test_scaling_factors_are_declared_sensitivity_assumptions(self):
        r = project_classical_budget(*fixture())
        for outer, n in ((False, 812), (True, 1205)):
            label = 'outer' if outer else 'inner'
            rf = r['RF']['configuration_profiles']['0'][label]
            svm = r['SVM']['configuration_profiles']['0'][label]
            self.assertAlmostEqual(rf['scale_factors']['fit'], n * math.log(n) / (799 * math.log(799)))
            self.assertAlmostEqual(svm['scale_factors']['fit'], (n / 799) ** 2)
            self.assertAlmostEqual(rf['scale_factors']['inference'], 406 / 175)
            self.assertAlmostEqual(svm['scale_factors']['inference'], (406 / 175)*(n/799))
            self.assertEqual(rf['scale_factors']['imports'], 1.)
        self.assertFalse(r['guaranteed_ETA'])
        self.assertFalse(r['full_budget_frozen'])
        self.assertTrue(r['cold_import_strategy_pending'])

    def test_detector_copies_group_overhead_per_fit_without_dividing_or_seed_factor(self):
        r = project_classical_budget(*fixture())['detector']
        self.assertEqual(r['grouped_overhead_copied_per_fit_seconds'], 3.5)
        inner = 3.5 + .1 + .05 * 406 / 175
        outer = 3.5 + .2 + .05 * 406 / 175
        self.assertAlmostEqual(r['projected_stage_seconds'], 96 * inner + 12 * outer)
        self.assertAlmostEqual(r['observed_process_gap_scenario_seconds'], 108 * .2)
        self.assertFalse(r['overhead_divided_by_representations_or_repetitions'])
        self.assertFalse(r['cache_savings_claimed'])
        self.assertEqual(r['candidate_aliases'], {'NF12': 'F14'})

    def test_missing_wall_time_remains_unknown_instead_of_zero(self):
        rows, detector = fixture()
        del rows[0]['process_elapsed_seconds']
        r = project_classical_budget(rows, detector)
        self.assertIsNone(r['RF']['observed_process_gap_scenario_seconds'])
        self.assertIsNone(r['aggregate']['projected_with_observed_process_gap_seconds'])
        self.assertGreater(r['aggregate']['projected_stage_seconds'], 0.)

    def test_failed_missing_duplicate_or_wrong_scopes_rejected(self):
        base, detector = fixture()
        variants = [base[:-1], base + [base[0]]]
        for key, value in (('family', 'DNN'), ('config_index', True), ('config_index', 8),
                ('training_windows', 800), ('validation_windows', 406), ('status', 'FAIL')):
            rows = copy.deepcopy(base); rows[0][key] = value; variants.append(rows)
        rows = copy.deepcopy(base); rows[1]['config_index'] = 0; variants.append(rows)
        rows = copy.deepcopy(base); rows[1]['job_id'] = rows[0]['job_id']; variants.append(rows)
        for rows in variants:
            with self.assertRaises(ValueError):
                project_classical_budget(rows, detector)
        for key, value in (('unique_representations', 8), ('fit_windows', 116), ('calibration_windows', 57)):
            d = copy.deepcopy(detector); d[key] = value
            with self.assertRaises(ValueError):
                project_classical_budget(base, d)
        for edit in ('missing', 'duplicate', 'failed'):
            d = copy.deepcopy(detector)
            if edit == 'missing': d['rows'].pop()
            elif edit == 'duplicate': d['rows'][1] = d['rows'][0]
            else: d['rows'][0]['status'] = 'FAIL'
            with self.assertRaises(ValueError):
                project_classical_budget(base, d)

    def test_invalid_or_overlapping_times_and_short_wall_rejected(self):
        rows, detector = fixture()
        for value in (-1., float('nan'), float('inf'), None, True):
            changed = copy.deepcopy(rows); changed[0]['timing_nonoverlapping_seconds']['fit'] = value
            with self.assertRaises(ValueError):
                project_classical_budget(changed, detector)
        changed = copy.deepcopy(rows); changed[0]['timing_nonoverlapping_seconds']['total'] = 10.
        with self.assertRaises(ValueError): project_classical_budget(changed, detector)
        changed = copy.deepcopy(rows); changed[0]['process_elapsed_seconds'] = 1.
        with self.assertRaises(ValueError): project_classical_budget(changed, detector)
        d = copy.deepcopy(detector); d['process_elapsed_seconds'] = 1.
        with self.assertRaises(ValueError): project_classical_budget(rows, d)

    def test_input_order_and_predictive_metrics_do_not_change_projection(self):
        rows, detector = fixture()
        baseline = project_classical_budget(rows, detector)
        rows[0].update(macro_f1=.1, accuracy=.2)
        detector['rows'][0].update(normal_specificity=.2)
        detector['rows'].reverse()
        self.assertEqual(project_classical_budget(rows[::-1], detector), baseline)


if __name__ == '__main__':
    unittest.main(verbosity=2)
