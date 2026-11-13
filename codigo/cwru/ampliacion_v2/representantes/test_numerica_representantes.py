"""Pruebas sinteticas del contrato numerico; sin archivos, TF ni ajustes."""
import unittest

import numpy as np

from numerica_representantes import (EarlyStopping, balanced_weights, detector_metrics,
                                     integer_argmax, metrics3, normalize_f32, quantize_int8)


class EarlyStoppingTests(unittest.TestCase):
    def test_first_observation_and_patience_exact(self):
        stop = EarlyStopping(patience=2)
        self.assertEqual(stop.observe(1, 1.), (True, False))
        self.assertEqual(stop.observe(2, 1.), (False, False))
        self.assertEqual(stop.observe(3, 1.1), (False, True))
        self.assertEqual((stop.best_epoch, stop.best_loss, stop.wait), (1, 1., 2))
        with self.assertRaises(ValueError):
            stop.observe(4, .1)

    def test_improvement_resets_patience_and_updates_best(self):
        stop = EarlyStopping(patience=2)
        for epoch, loss in enumerate([2., 2., 1., 1.], 1):
            stop.observe(epoch, loss)
        self.assertEqual((stop.best_epoch, stop.best_loss, stop.wait), (3, 1., 1))
        self.assertFalse(stop.stopped)

    def test_strict_min_delta_boundary(self):
        stop = EarlyStopping(patience=3, min_delta=.125)
        self.assertEqual(stop.observe(1, 1.), (True, False))
        self.assertEqual(stop.observe(2, .875), (False, False))
        self.assertEqual(stop.observe(3, .75), (True, False))
        self.assertEqual(stop.best_epoch, 3)

    def test_default_min_delta(self):
        stop = EarlyStopping()
        self.assertEqual(stop.min_delta, 1e-12)
        stop.observe(1, 1.)
        self.assertEqual(stop.observe(2, 1. - 5e-13), (False, False))
        self.assertEqual(stop.observe(3, 1. - 2e-12), (True, False))

    def test_out_of_order_epoch_rejected_without_state_change(self):
        stop = EarlyStopping()
        for invalid in [0, 2, True, 1.]:
            with self.assertRaises(ValueError):
                stop.observe(invalid, 1.)
        self.assertEqual(stop.last_epoch, 0)
        stop.observe(1, 1.)
        for invalid in [1, 3, -1]:
            with self.assertRaises(ValueError):
                stop.observe(invalid, .1)
        self.assertEqual((stop.last_epoch, stop.best_epoch), (1, 1))

    def test_invalid_loss_and_constructor(self):
        for loss in [float('nan'), float('inf'), -float('inf'), True, '1']:
            with self.assertRaises(ValueError):
                EarlyStopping().observe(1, loss)
        for kwargs in [dict(patience=0), dict(patience=True), dict(patience=2.),
                       dict(min_delta=-1), dict(min_delta=float('nan'))]:
            with self.assertRaises(ValueError):
                EarlyStopping(**kwargs)


class WeightTests(unittest.TestCase):
    def test_imbalanced_exact_weights_and_dtypes(self):
        y = np.array([0, 1, 1, 2, 2, 2], dtype=np.int64)
        counts, weights, samples = balanced_weights(y)
        np.testing.assert_array_equal(counts, [1, 2, 3])
        np.testing.assert_array_equal(weights, [2., 1., 2. / 3.])
        np.testing.assert_array_equal(samples, weights[y].astype(np.float32))
        self.assertEqual((counts.dtype, weights.dtype, samples.dtype),
                         (np.dtype('int64'), np.dtype('float64'), np.dtype('float32')))
        self.assertAlmostEqual(float(samples.astype(np.float64).mean()), 1., places=7)

    def test_missing_class_and_invalid_labels_stop(self):
        for values in [[], [0, 0, 1], [-1, 1, 2], [0, 1, 3], [0., 1., 2.],
                       [True, False], [[0, 1, 2]]]:
            with self.assertRaises(ValueError):
                balanced_weights(values)


class TransformTests(unittest.TestCase):
    def test_normalization_explicit_f32_and_no_mutation(self):
        source = np.array([[100000001., 2.], [3., -1.]], dtype=np.float64)
        before = source.copy()
        means, scales = np.array([100000000., 1.]), np.array([3., 2.])
        result = normalize_f32(source, means, scales)
        expected = np.divide(np.subtract(source.astype(np.float32), means.astype(np.float32),
                                        dtype=np.float32), scales.astype(np.float32), dtype=np.float32)
        self.assertEqual(result.dtype, np.float32)
        self.assertEqual(result.tobytes(), expected.tobytes())
        self.assertEqual(result[0, 0], 0.)
        np.testing.assert_array_equal(source, before)

    def test_normalization_rejects_bad_shapes_scale_and_nonfinite(self):
        for x, mean, scale in [([[1., 2.]], [0.], [1.]), ([[1.]], [0.], [0.]),
                               ([[1.]], [0.], [-1.]), ([[float('nan')]], [0.], [1.]),
                               ([[1.]], [float('inf')], [1.]), ([1.], [0.], [1.]),
                               ([[1e100]], [0.], [1.]), ([[3e38]], [-3e38], [1.])]:
            with self.assertRaises(ValueError):
                normalize_f32(x, mean, scale)

    def test_quantization_ties_even_and_zero_point(self):
        result, clipped = quantize_int8(np.array([-2.5, -1.5, -.5, .5, 1.5, 2.5]), 1., 0)
        np.testing.assert_array_equal(result, [-2, -2, 0, 0, 2, 2])
        self.assertEqual(result.dtype, np.int8)
        self.assertEqual(clipped, 0)
        shifted, _ = quantize_int8([.5, 1.5, 2.5], 1., -1)
        np.testing.assert_array_equal(shifted, [0, 0, 2])

    def test_saturation_count_after_rounding(self):
        result, clipped = quantize_int8([-129., -128.5, -128., 127., 127.4, 127.5, 128.], 1., 0)
        np.testing.assert_array_equal(result, [-128, -128, -128, 127, 127, 127, 127])
        self.assertEqual(clipped, 3)

    def test_quantization_rejects_invalid_and_nonfinite_intermediates(self):
        for x, scale, zero in [([1.], 0., 0), ([1.], -1., 0), ([1.], 1e-100, 0),
                                ([1.], 1., 128), ([1.], 1., -129), ([1.], 1., True),
                                ([1.], 1., 0.), ([float('inf')], 1., 0),
                                ([3e38], 1e-10, 0), ([1.], float('nan'), 0)]:
            with self.assertRaises(ValueError):
                quantize_int8(x, scale, zero)


class ClassifierMetricsTests(unittest.TestCase):
    def test_argmax_int8_ties_select_lowest_index(self):
        values = np.array([[127, 127, -128], [-128, -128, -128], [0, 1, 1]], dtype=np.int8)
        predicted = integer_argmax(values)
        np.testing.assert_array_equal(predicted, [0, 0, 1])
        self.assertEqual(predicted.dtype, np.int64)

    def test_argmax_invalid_shape_or_nonfinite(self):
        for value in [[1., 2., 3.], [[0., float('nan'), 1.]], [[1., 2.]], [], [[True, False, True]]]:
            with self.assertRaises(ValueError):
                integer_argmax(value)

    def test_confusion_orientation_and_macro_f1(self):
        result = metrics3([0, 0, 1, 1, 2], [0, 1, 1, 2, 2])
        self.assertEqual(result['confusion_matrix'], [[1, 1, 0], [0, 1, 1], [0, 0, 1]])
        self.assertEqual(result['accuracy'], .6)
        self.assertAlmostEqual(result['macro_f1'], (2. / 3. + .5 + 2. / 3.) / 3.)
        self.assertEqual(result['per_class'][0]['precision'], 1.)
        self.assertEqual(result['per_class'][0]['recall'], .5)

    def test_absent_classes_zero_division_and_fixed_label_order(self):
        result = metrics3([0, 0], [0, 0])
        self.assertEqual(result['accuracy'], 1.)
        self.assertEqual(result['macro_f1'], 1. / 3.)
        self.assertEqual([r['label'] for r in result['per_class']], [0, 1, 2])
        self.assertEqual(result['per_class'][2], dict(label=2, precision=0., recall=0., f1=0., support=0))

    def test_metrics_invalid_labels_and_lengths(self):
        for y, pred in [([], []), ([0], [0, 1]), ([0], [3]), ([0.], [0]), ([True], [0])]:
            with self.assertRaises(ValueError):
                metrics3(y, pred)


class DetectorMetricsTests(unittest.TestCase):
    def test_equal_record_mass_not_pooled_windows(self):
        # N small: acceptance0; N large: acceptance1 => specificity.5, not.75.
        y = [0, 0, 0, 0, 1, 2]
        rejected = np.array([True, False, False, False, True, False])
        ids = ['Nsmall', 'Nlarge', 'Nlarge', 'Nlarge', 'IR', 'OR']
        result = detector_metrics(y, rejected, ids)
        self.assertEqual((result['normal_spec'], result['ir_recall'], result['or_recall']), (.5, 1., 0.))
        self.assertEqual(result['balanced_accuracy'], .5)
        self.assertEqual(result['BA'], .5)
        reverse = detector_metrics(list(reversed(y)), rejected[::-1], list(reversed(ids)))
        self.assertEqual(result, reverse)

    def test_invalid_record_labels_ids_and_decisions(self):
        for y, rejected, ids in [([0, 1, 2], [False, True, True], ['a', 'a', 'b']),
                                ([0, 1, 2], [False, True, True], ['a', 'b']),
                                ([0, 1, 2], [False, True, True], ['a', '', 'b']),
                                ([0, 1, 2], [False, True, True], ['a', True, 'b']),
                                ([0, 1], [False, True], ['a', 'b'])]:
            with self.assertRaises(ValueError):
                detector_metrics(y, np.asarray(rejected, dtype=bool), ids)
        with self.assertRaises(ValueError):
            detector_metrics([0, 1, 2], [0, 1, 1], ['a', 'b', 'c'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
