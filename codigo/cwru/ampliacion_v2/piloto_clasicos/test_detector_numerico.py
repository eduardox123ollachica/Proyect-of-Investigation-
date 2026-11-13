"""Pruebas sinteticas; no abre CWRU, archivos o puertos ni importa TensorFlow."""
from __future__ import annotations

import importlib.util
import unittest

import numpy as np

from detector_numerico import detector_score, fit_normality


def matrix(values, columns=(0,)):
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 1:
        values = values[:, None]
    result = np.zeros((len(values), 14), dtype=np.float32)
    result[:, list(columns)] = values
    return result


def parameters(indices=(0, 1)):
    n = len(indices)
    return dict(indices=list(indices), mean32=np.zeros(n, dtype=np.float32),
                scale32=np.ones(n, dtype=np.float32), centroid32=np.zeros(n, dtype=np.float32))


class ScoreTests(unittest.TestCase):
    def test_known_float32_scores(self):
        params = parameters()
        params['scale32'][1] = 2.
        params['centroid32'][1] = 1.
        scores = detector_score(matrix([[3., 4.], [-1., 2.]], (0, 1)), params)
        self.assertEqual(scores.dtype, np.float32)
        np.testing.assert_array_equal(scores, np.asarray([10., 1.], dtype=np.float32))

    def test_strict_threshold_ties_are_accepted(self):
        params = parameters((0,))
        params['threshold32'] = np.float32(1.)
        scores = detector_score(matrix([0., 1., 2.]), params)
        self.assertEqual((scores > params['threshold32']).tolist(), [False, False, True])

    def test_unselected_finite_columns_do_not_change_score(self):
        data = matrix([1., 2., 3.])
        baseline = detector_score(data, parameters((0,)))
        data[:, 1:] = np.finfo(np.float32).max
        self.assertEqual(detector_score(data, parameters((0,))).tobytes(), baseline.tobytes())

    def test_nonfinite_in_any_input_column_is_rejected(self):
        for column in (0, 4, 11):
            for value in (np.nan, np.inf, -np.inf):
                data = matrix([1.])
                data[0, column] = value
                with self.subTest(column=column, value=value), self.assertRaises(ValueError):
                    detector_score(data, parameters((0,)))

    def test_overflow_in_normalization_is_rejected(self):
        for mean, scale in ((-np.finfo(np.float32).max, 1.), (0., 1e-30)):
            params = parameters((0,))
            params['mean32'][0], params['scale32'][0] = mean, scale
            with self.subTest(mean=mean, scale=scale), self.assertRaises(ValueError):
                detector_score(matrix([np.finfo(np.float32).max]), params)

    def test_square_overflow_is_rejected(self):
        with self.assertRaises(ValueError):
            detector_score(matrix([1e20]), parameters((0,)))

    def test_sequential_sum_overflow_is_rejected(self):
        value = np.sqrt(np.float32(2e38), dtype=np.float32)
        with self.assertRaises(ValueError):
            detector_score(matrix([[value, value]], (0, 1)), parameters())

    def test_invalid_parameter_types_dimensions_scales_and_threshold(self):
        invalid = []
        for key, value in (('mean32', np.zeros(1, dtype=np.float32)),
                           ('scale32', np.ones(2, dtype=np.float64)),
                           ('centroid32', np.asarray([0., np.nan], dtype=np.float32)),
                           ('scale32', np.asarray([1., 0.], dtype=np.float32)),
                           ('threshold32', np.float64(1.)), ('threshold32', np.float32(-1.))):
            params = parameters()
            params[key] = value
            invalid.append(params)
        for params in invalid:
            with self.subTest(params=params), self.assertRaises(ValueError):
                detector_score(matrix([[1., 2.]], (0, 1)), params)

    def test_indices_must_be_unique_integer_allowed_features(self):
        for indices in ([], [0, 0], [4], [11], [-1], [14], [True], [0.], '0'):
            params = parameters((0,))
            params['indices'] = indices
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                detector_score(matrix([1.]), params)

    def test_input_shape_and_dtype_are_strict(self):
        for data in (np.zeros((1, 14), dtype=np.float64), np.zeros((0, 14), dtype=np.float32),
                     np.zeros(14, dtype=np.float32), np.zeros((2, 13), dtype=np.float32), [[0.] * 14]):
            with self.subTest(data=data), self.assertRaises(ValueError):
                detector_score(data, parameters((0,)))

    def test_invalid_fit_scope_and_record_ids_rejected_before_sklearn(self):
        fit = matrix([-1., 1.])
        cal = matrix([0., 2.])
        for records in ([], ['a'], ['a', None], ['a', True], ['a', ' b'], 'aa'):
            with self.subTest(records=records), self.assertRaises(ValueError):
                fit_normality(fit, cal, records, [0])
        for data in (fit.astype(np.float64), np.zeros((0,14), dtype=np.float32)):
            with self.assertRaises(ValueError):
                fit_normality(data, cal, ['a','a'], [0])


@unittest.skipUnless(importlib.util.find_spec('sklearn') is not None, 'StandardScaler requiere sklearn en este runtime')
class FitTests(unittest.TestCase):
    def test_scaler_and_p95_match_known_problem_without_refit(self):
        fit = matrix([-1., 1.])
        cal = matrix(np.arange(20, dtype=np.float32))
        params = fit_normality(fit, cal, ['a'] * 20, [0])
        np.testing.assert_array_equal(params['mean64'], [0.])
        np.testing.assert_array_equal(params['scale64'], [1.])
        np.testing.assert_array_equal(params['centroid32'], [0.])
        self.assertEqual(params['threshold32'], np.float32(324.))
        self.assertFalse(params['refit_after_calibration'])
        self.assertEqual(params['fit_rows'], 2)

    def test_zero_variance_uses_scaler_unit_scale(self):
        params = fit_normality(matrix([3., 3., 3.]), matrix([3., 4.]), ['a', 'a'], [0])
        self.assertEqual(params['scale64'][0], 1.)
        self.assertEqual(params['scale32'][0], np.float32(1.))
        self.assertEqual(params['threshold32'], np.float32(1.))

    def test_centroid_is_mean_after_float32_transform_not_forced_zero(self):
        values = np.asarray([0.001230153371579945, 0.2987455427646637, -0.27413785457611084], dtype=np.float32)
        fit = matrix(values)
        params = fit_normality(fit, matrix([0.]), ['a'], [0])
        z = np.divide(np.subtract(fit[:, [0]], params['mean32'], dtype=np.float32),
                      params['scale32'], dtype=np.float32)
        expected = np.mean(z, axis=0, dtype=np.float64).astype(np.float32)
        self.assertNotEqual(expected[0], np.float32(0.))
        self.assertEqual(params['centroid32'].tobytes(), expected.tobytes())

    def test_calibration_gives_equal_total_mass_to_each_record(self):
        params = fit_normality(matrix([-1., 1.]), matrix([0.] * 100 + [2.]), ['a'] * 100 + ['b'], [0])
        self.assertEqual(params['threshold32'], np.float32(4.))

    def test_fit_preserves_inputs_and_repeats_bit_exactly(self):
        fit = matrix([-1., 1., 3.])
        cal = matrix([0., 2.])
        fit_copy, cal_copy = fit.copy(), cal.copy()
        first = fit_normality(fit, cal, ['a', 'b'], [0])
        second = fit_normality(fit, cal, ['a', 'b'], [0])
        self.assertEqual(fit.tobytes(), fit_copy.tobytes())
        self.assertEqual(cal.tobytes(), cal_copy.tobytes())
        for name in ('mean64','scale64','var64','mean32','scale32','centroid32','threshold32'):
            self.assertEqual(first[name].tobytes(), second[name].tobytes())
        self.assertEqual(first['mean64'].dtype, np.float64)
        self.assertEqual(first['scale32'].dtype, np.float32)
        self.assertIsInstance(first['threshold32'], np.float32)


if __name__ == '__main__':
    unittest.main(verbosity=2)
