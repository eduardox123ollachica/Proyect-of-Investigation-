"""Pruebas de cierre y parada de procesos con dobles; no ejecuta workers."""
import json
import unittest
from unittest.mock import MagicMock,Mock,patch
import subprocess

from ejecutar_piloto_clasicos import read_worker_result,run_one


class ExecutionTests(unittest.TestCase):
    def test_partial_json_rejected(self):
        with patch('ejecutar_piloto_clasicos.read',side_effect=json.JSONDecodeError('cut','{',1)):
            result,error=read_worker_result(Mock(),dict(job_id='RF_00',family='RF'))
        self.assertIsNone(result);self.assertIn('JSONDecodeError',error)

    def test_missing_result_rejected(self):
        with patch('ejecutar_piloto_clasicos.read',side_effect=FileNotFoundError('missing')):
            self.assertIsNone(read_worker_result(Mock(),dict(job_id='RF_00',family='RF'))[0])

    def test_invalid_status_rejected(self):
        with patch('ejecutar_piloto_clasicos.read',return_value=dict(status='FAIL')):
            self.assertIsNone(read_worker_result(Mock(),dict(job_id='RF_00',family='RF'))[0])

    def exercise(self,effect=None,launch_error=None):
        job=dict(job_id='RF_00',family='RF',timeout_seconds=60)
        process=Mock(returncode=-9)
        process.poll.return_value=None
        process.wait.side_effect=[effect,0] if effect is not None else None
        if effect is None:process.wait.return_value=0
        with patch('ejecutar_piloto_clasicos.subprocess.Popen',side_effect=launch_error,return_value=process), \
             patch('ejecutar_piloto_clasicos.read_worker_result',return_value=(dict(status='PASS'),None)), \
             patch('ejecutar_piloto_clasicos.write'):
            status,result=run_one(MagicMock(),job,{})
        return status,result,process

    def test_successful_process_is_retained(self):
        status,result,process=self.exercise()
        self.assertEqual(status['status'],'PASS');self.assertIsNotNone(result)
        process.kill.assert_not_called()

    def test_timeout_kills_child_and_is_not_success(self):
        status,_,process=self.exercise(subprocess.TimeoutExpired(['worker'],60))
        self.assertEqual(status['status'],'TIMEOUT');process.kill.assert_called_once()
        self.assertFalse(status['automatic_retry'])

    def test_keyboard_interrupt_kills_child(self):
        status,_,process=self.exercise(KeyboardInterrupt())
        self.assertEqual(status['status'],'INTERRUPTED');process.kill.assert_called_once()

    def test_launch_failure_is_preserved(self):
        status,_,process=self.exercise(launch_error=OSError('denied'))
        self.assertEqual(status['status'],'FAIL');self.assertIn('OSError',status['process_launch_error'])
        process.kill.assert_not_called()


if __name__=='__main__':unittest.main(verbosity=2)
