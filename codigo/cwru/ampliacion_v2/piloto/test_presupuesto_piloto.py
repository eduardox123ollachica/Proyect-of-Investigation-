"""Pruebas puras del presupuesto: sin archivos, CWRU o TensorFlow."""
import copy
import math
import unittest

from presupuesto_piloto import STAGES, project_dnn_budget


def fixture():
    return [dict(mask=mask, status='PASS', training_windows=799,
        validation_windows=175, epochs=50, batches_per_epoch=25,
        optimizer_updates=1250,
        timing_nonoverlapping_seconds={name: factor for name in STAGES})
        for mask, factor in (('F14', 2.), ('T6', 1.))]


class BudgetTests(unittest.TestCase):
    def test_counts_do_not_double_count_representatives_or_pilot(self):
        result = project_dnn_budget(fixture())
        self.assertEqual(result['maximum_DNN_fits_before_identical_reuse'], 408)
        self.assertEqual(result['maximum_FP32_plus_INT8_conversions'], 816)
        self.assertEqual(result['additional_cost_representative_fits'], 0)
        self.assertEqual(result['cost_representatives_included_in_inner_fits'], 32)
        self.assertEqual(result['pilot_scope']['additional_DNN_fits'], 2)
        self.assertEqual(result['pilot_scope']['additional_conversions'], 4)
        self.assertFalse(result['pilot_scope']['included_in_projected_408'])

    def test_all_stages_sum_once_and_conserve_seconds_units(self):
        result = project_dnn_budget(fixture())
        profile = result['scenarios']['epochs_50']['profiles']['T6']
        for name, fits in (('inner', 288), ('outer', 120)):
            block = profile[name]
            self.assertAlmostEqual(block['seconds_per_fit'], math.fsum(block['stage_seconds_per_fit'].values()))
            self.assertAlmostEqual(block['projected_seconds_all_fits'], block['seconds_per_fit'] * fits)
            self.assertAlmostEqual(block['projected_seconds_all_fits'], math.fsum(block['projected_stage_seconds_all_fits'].values()))
        self.assertAlmostEqual(profile['projected_DNN_seconds'], profile['inner']['projected_seconds_all_fits'] + profile['outer']['projected_seconds_all_fits'])
        self.assertAlmostEqual(profile['projected_DNN_hours'], profile['projected_DNN_seconds'] / 3600.)
        self.assertEqual(result['units'], 'seconds')

    def test_epoch_scenario_changes_only_epoch_dependent_stages(self):
        result = project_dnn_budget(fixture())
        short = result['scenarios']['epochs_25']['profiles']['T6']
        full = result['scenarios']['epochs_50']['profiles']['T6']
        epoch_stages = {'training_updates', 'epoch_training_diagnostics', 'epoch_validation'}
        for block in ('inner', 'outer'):
            for stage in STAGES:
                ratio = .5 if stage in epoch_stages else 1.
                self.assertAlmostEqual(short[block]['stage_seconds_per_fit'][stage], full[block]['stage_seconds_per_fit'][stage] * ratio)
        self.assertLess(short['projected_DNN_seconds'], full['projected_DNN_seconds'])
        self.assertGreater(short['projected_DNN_seconds'], .5 * full['projected_DNN_seconds'])

    def test_training_and_inference_scope_scaling(self):
        profile = project_dnn_budget(fixture())['scenarios']['epochs_50']['profiles']['T6']
        self.assertAlmostEqual(profile['inner']['timing_scale_factors']['training_updates'], 26 / 25)
        self.assertAlmostEqual(profile['outer']['timing_scale_factors']['training_updates'], 38 / 25)
        self.assertEqual(profile['outer']['stage_seconds_per_fit']['epoch_validation'], 0.)
        for block, rows in (('inner', 812), ('outer', 1205)):
            self.assertAlmostEqual(profile[block]['timing_scale_factors']['INT8_conversion'], rows / 799)
            self.assertAlmostEqual(profile[block]['timing_scale_factors']['inference'], 406 / 175)
        self.assertEqual(profile['inner']['optimizer_updates_assumed'], 1300)
        self.assertEqual(profile['outer']['optimizer_updates_assumed'], 1900)

    def test_two_complete_profiles_define_range_not_statistical_interval(self):
        result = project_dnn_budget(fixture())
        for scenario in result['scenarios'].values():
            a = scenario['profiles']['F14']['projected_DNN_seconds']
            b = scenario['profiles']['T6']['projected_DNN_seconds']
            self.assertAlmostEqual(a, 2 * b)
            self.assertEqual(scenario['range_between_two_pilot_profiles_seconds'], dict(min=b, max=a))
            self.assertEqual(scenario['conservative_max_seconds'], a)
            self.assertFalse(scenario['confidence_interval'])
            self.assertFalse(scenario['guaranteed_ETA'])
            self.assertFalse(scenario['WCET'])
        self.assertTrue(result['classical_runtime_budget_pending'])
        self.assertFalse(result['full_budget_frozen'])
        self.assertFalse(result['STM32_latency_measured'])

    def test_bad_scope_missing_duplicate_or_failed_profiles_rejected(self):
        invalid = [[], fixture()[:1], fixture() + [fixture()[0]]]
        for key, value in (('training_windows', 800), ('validation_windows', 406),
                ('epochs', 25), ('batches_per_epoch', 26), ('optimizer_updates', 1251),
                ('status', 'FAIL'), ('mask', 'NM10'), ('epochs', True)):
            rows = fixture()
            rows[0][key] = value
            invalid.append(rows)
        rows = fixture()
        rows[1]['mask'] = 'F14'
        invalid.append(rows)
        for rows in invalid:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                project_dnn_budget(rows)

    def test_invalid_or_overlapping_timing_schema_rejected(self):
        invalid = []
        for value in (-1., float('nan'), float('inf'), True, None):
            rows = fixture()
            rows[0]['timing_nonoverlapping_seconds']['imports'] = value
            invalid.append(rows)
        rows = fixture()
        del rows[0]['timing_nonoverlapping_seconds']['imports']
        invalid.append(rows)
        rows = fixture()
        rows[0]['timing_nonoverlapping_seconds']['total'] = 10.
        invalid.append(rows)
        rows = fixture()
        rows[0]['timing_nonoverlapping_seconds'] = dict.fromkeys(STAGES, 0.)
        invalid.append(rows)
        for rows in invalid:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                project_dnn_budget(rows)

    def test_order_and_irrelevant_predictive_metrics_do_not_change_budget(self):
        rows = fixture()
        original = copy.deepcopy(rows)
        baseline = project_dnn_budget(rows)
        rows[0].update(macro_f1=0., accuracy=0.)
        rows[1].update(macro_f1=1., accuracy=1.)
        self.assertEqual(project_dnn_budget(rows[::-1]), baseline)
        self.assertEqual(original, fixture())


if __name__ == '__main__':
    unittest.main(verbosity=2)
