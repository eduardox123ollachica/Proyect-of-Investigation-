"""Controles de cierre de la revision de exportacion, sin entrenar."""
import json
import unittest
from unittest.mock import Mock, patch
from ejecutar_piloto_tiempo_v2 import read_worker_result


class WorkerRevisionResultTests(unittest.TestCase):
    def test_missing_result(self):
        self.assertIsNone(read_worker_result(Mock(exists=Mock(return_value=False)))[0])

    def test_partial_json(self):
        with patch('ejecutar_piloto_tiempo_v2.read', side_effect=json.JSONDecodeError('cut', '{', 1)):
            result, error = read_worker_result(Mock(exists=Mock(return_value=True)))
        self.assertIsNone(result)
        self.assertIn('JSONDecodeError', error)

    def test_unreadable_result(self):
        with patch('ejecutar_piloto_tiempo_v2.read', side_effect=OSError('locked')):
            self.assertIsNone(read_worker_result(Mock(exists=Mock(return_value=True)))[0])

    def test_invalid_result(self):
        for value in (None, [], {'status':'FAIL'}):
            with patch('ejecutar_piloto_tiempo_v2.read', return_value=value):
                self.assertIsNone(read_worker_result(Mock(exists=Mock(return_value=True)))[0])

    def test_valid_result(self):
        payload={'status':'PASS'}
        with patch('ejecutar_piloto_tiempo_v2.read', return_value=payload):
            result,error=read_worker_result(Mock(exists=Mock(return_value=True)))
        self.assertIs(result,payload)
        self.assertIsNone(error)


if __name__ == '__main__':
    unittest.main(verbosity=2)
