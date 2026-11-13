"""Cierre de resultados parciales: sin procesos, archivos ni TensorFlow."""
import json
import unittest
from unittest.mock import Mock, patch

from ejecutar_piloto_tiempo import read_worker_result


class WorkerResultTests(unittest.TestCase):
    def test_missing_result_is_failure(self):
        result, error = read_worker_result(Mock(exists=Mock(return_value=False)))
        self.assertIsNone(result)
        self.assertIn('ausente', error)

    def test_partial_json_is_preserved_as_failure(self):
        path = Mock(exists=Mock(return_value=True))
        with patch('ejecutar_piloto_tiempo.read', side_effect=json.JSONDecodeError('interrupted', '{', 1)):
            result, error = read_worker_result(path)
        self.assertIsNone(result)
        self.assertIn('JSONDecodeError', error)

    def test_unreadable_result_is_failure(self):
        with patch('ejecutar_piloto_tiempo.read', side_effect=OSError('locked')):
            result, error = read_worker_result(Mock(exists=Mock(return_value=True)))
        self.assertIsNone(result)
        self.assertIn('OSError', error)

    def test_invalid_payload_is_failure(self):
        for value in (None, [], {'status': 'FAIL'}, {'status': 'RUNNING'}):
            with self.subTest(value=value), patch('ejecutar_piloto_tiempo.read', return_value=value):
                result, error = read_worker_result(Mock(exists=Mock(return_value=True)))
                self.assertIsNone(result)
                self.assertIsNotNone(error)

    def test_valid_payload_survives_without_mutation(self):
        payload = {'status': 'PASS', 'optimizer_updates': 1250}
        with patch('ejecutar_piloto_tiempo.read', return_value=payload):
            result, error = read_worker_result(Mock(exists=Mock(return_value=True)))
        self.assertIs(result, payload)
        self.assertIsNone(error)


if __name__ == '__main__':
    unittest.main(verbosity=2)
