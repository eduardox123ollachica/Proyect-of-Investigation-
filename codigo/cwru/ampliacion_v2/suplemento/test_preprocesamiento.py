"""Pruebas sinteticas del preprocesamiento: ningun dato/modelo CWRU se abre."""
from copy import deepcopy
import unittest

import numpy as np

from preprocesamiento import INDICES, apply, fit_params


def matrix(rows=5):
    result = np.arange(rows * 14, dtype=np.float32).reshape(rows, 14) / np.float32(10)
    result[:, 1] = np.float32(-2)
    result[:, 2] = np.float32(-.5)
    result[:, 8] = np.float32(-1)
    result[:, 9] = np.float32(-1.25)
    return result


class ParameterTests(unittest.TestCase):
    def test_identity_parameters_and_copy_preserve_bits(self):
        raw = matrix()
        raw[0, 0] = np.float32(-2)
        raw[0, 4] = np.float32(-0.)
        before = raw.copy()
        params = fit_params(raw, 'P0')
        self.assertEqual(params['indices'], [0, 3, 5, 7, 10, 12])
        self.assertEqual(params['scales64'].dtype, np.float64)
        self.assertEqual(params['scales32'].dtype, np.float32)
        self.assertTrue(np.array_equal(params['scales64'], np.ones(6)))
        transformed = apply(raw, params)
        self.assertEqual(transformed.dtype, np.float32)
        self.assertFalse(np.shares_memory(raw, transformed))
        self.assertTrue(np.array_equal(raw.view(np.uint32), transformed.view(np.uint32)))
        self.assertTrue(np.array_equal(raw, before))

    def test_median_uses_only_strictly_positive_training_values(self):
        raw = matrix()
        for index in INDICES:
            raw[:, index] = [0, 0, 0, 2, 10]
        before = raw.copy()
        params = fit_params(raw, 'P1')
        self.assertTrue(np.array_equal(params['scales64'], np.full(6, 6., dtype=np.float64)))
        self.assertTrue(np.array_equal(params['scales32'], np.full(6, 6., dtype=np.float32)))
        self.assertTrue(np.array_equal(raw, before))

    def test_even_median_calculated_after_float64_cast(self):
        raw = matrix(2)
        largest = np.finfo(np.float32).max
        for index in INDICES:
            raw[:, index] = [largest, largest]
        params = fit_params(raw, 'P1')
        self.assertTrue(np.isfinite(params['scales64']).all())
        self.assertTrue(np.array_equal(params['scales32'], np.full(6, largest, dtype=np.float32)))

    def test_all_zero_training_columns_use_fixed_one(self):
        raw = matrix()
        raw[:, INDICES] = 0
        params = fit_params(raw, 'P1')
        self.assertTrue(np.array_equal(params['scales64'], np.ones(6)))
        self.assertTrue(np.array_equal(apply(raw, params), raw))
        future = raw.copy()
        future[:, INDICES] = 3
        transformed = apply(future, params)
        self.assertTrue(np.allclose(transformed[:, INDICES], np.log(np.float32(4)), rtol=1e-7))

    def test_validation_values_never_refit_training_scales(self):
        train = matrix()
        train[:, INDICES] = np.asarray([1, 2, 4, 8, 16], dtype=np.float32)[:, None]
        validation = matrix(3)
        validation[:, INDICES] = 1000000
        params = fit_params(train, 'P1')
        before = deepcopy(params)
        transformed = apply(validation, params)
        self.assertTrue(np.array_equal(params['scales64'], before['scales64']))
        self.assertTrue(np.array_equal(params['scales32'], before['scales32']))
        expected = np.log1p(np.float32(1000000) / np.float32(4), dtype=np.float32)
        self.assertTrue(np.array_equal(transformed[:, INDICES], np.full((3, 6), expected, dtype=np.float32)))
        self.assertNotEqual(float(params['scales32'][0]), float(fit_params(validation, 'P1')['scales32'][0]))

    def test_reject_dtype_shape_empty_and_nonfinite(self):
        raw = matrix()
        bad = [raw.astype(np.float64), raw.tolist(), raw[:, :13], raw.reshape(-1), raw[:0]]
        for value in (float('nan'), float('inf'), -float('inf')):
            changed = raw.copy(); changed[1, 4] = value; bad.append(changed)
        for item in bad:
            for mode in ('P0', 'P1'):
                with self.subTest(mode=mode, item=type(item)), self.assertRaises(ValueError):
                    fit_params(item, mode)

    def test_reject_invalid_mode(self):
        for mode in ('p0', 'log', '', None, 1, True):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                fit_params(matrix(), mode)

    def test_negative_magnitudes_stop_P1_fit_and_apply(self):
        for index in INDICES:
            raw = matrix()
            params = fit_params(raw, 'P1')
            raw[0, index] = np.float32(-1e-6)
            with self.subTest(index=index), self.assertRaises(ValueError):
                fit_params(raw, 'P1')
            with self.subTest(index=index), self.assertRaises(ValueError):
                apply(raw, params)


class TransformTests(unittest.TestCase):
    def test_explicit_f32_division_log1p_and_no_mutation(self):
        raw = matrix()
        params = fit_params(raw, 'P1')
        before, scales_before = raw.copy(), params['scales32'].copy()
        expected = np.log1p(np.divide(raw[:, INDICES], params['scales32'], dtype=np.float32), dtype=np.float32)
        result = apply(raw, params)
        self.assertEqual(result.dtype, np.float32)
        self.assertTrue(np.array_equal(result[:, INDICES].view(np.uint32), expected.view(np.uint32)))
        self.assertTrue(np.array_equal(raw, before))
        self.assertTrue(np.array_equal(params['scales32'], scales_before))

    def test_untransformed_frequencies_and_shapes_are_bit_exact(self):
        raw = matrix()
        raw[:, 4] = [4145.5078125, 360.3515625, 1025.390625, 2.9296875, -0.]
        raw[:, 11] = raw[::-1, 4]
        result = apply(raw, fit_params(raw, 'P1'))
        untouched = [i for i in range(14) if i not in INDICES]
        self.assertTrue(np.array_equal(raw[:, untouched].view(np.uint32), result[:, untouched].view(np.uint32)))
        self.assertTrue((result[:, [1, 2, 8, 9]] < 0).all())

    def test_monotone_sublinear_not_bounded(self):
        train = matrix(3)
        train[:, INDICES] = 1
        params = fit_params(train, 'P1')
        raw = matrix(5)
        raw[:, INDICES] = np.asarray([0, 1, 10, 1000, 1e20], dtype=np.float32)[:, None]
        output = apply(raw, params)[:, 0]
        self.assertTrue((np.diff(output) > 0).all())
        self.assertEqual(float(output[0]), 0.)
        self.assertTrue((output[1:] < raw[1:, 0]).all())
        self.assertGreater(float(output[-1]), 40.)  # No fixed clipping range.

    def test_division_overflow_stops_without_log_or_silent_repair(self):
        train = matrix(3)
        tiny = np.nextafter(np.float32(0), np.float32(1))
        train[:, INDICES] = tiny
        params = fit_params(train, 'P1')
        raw = matrix(2)
        raw[:, INDICES] = np.finfo(np.float32).max
        before = raw.copy()
        with self.assertRaisesRegex(ValueError, 'Division'):
            apply(raw, params)
        self.assertTrue(np.array_equal(raw, before))

    def test_reject_nonfinite_input_in_untransformed_column(self):
        train = matrix()
        params = fit_params(train, 'P1')
        raw = train.copy(); raw[0, 1] = np.nan
        with self.assertRaises(ValueError):
            apply(raw, params)

    def test_reject_parameter_tampering(self):
        valid = fit_params(matrix(), 'P1')
        changed = []
        item = deepcopy(valid); item['indices'] = INDICES[::-1]; changed.append(item)
        item = deepcopy(valid); item['indices'][0] = False; changed.append(item)
        item = deepcopy(valid); item['scales64'] = item['scales64'].astype(np.float32); changed.append(item)
        item = deepcopy(valid); item['scales32'] = item['scales32'].astype(np.float64); changed.append(item)
        item = deepcopy(valid); item['scales32'] = item['scales32'][:5]; changed.append(item)
        item = deepcopy(valid); item['scales32'][0] *= np.float32(2); changed.append(item)
        item = deepcopy(valid); item['scales64'][0] = np.inf; changed.append(item)
        item = deepcopy(valid); item['scales32'][0] = 0; changed.append(item)
        item = deepcopy(valid); item['mode'] = 'P0'; changed.append(item)
        for params in changed:
            with self.subTest(params=params), self.assertRaises(ValueError):
                apply(matrix(), params)


if __name__ == '__main__':
    unittest.main()
