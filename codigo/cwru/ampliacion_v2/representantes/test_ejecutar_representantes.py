"""Pruebas de puertas, preservacion y firma; no crean procesos de entrenamiento."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import ejecutar_representantes as runner
from datos_representantes import build_jobs, load_verified_context


def temporary_directory():
    base=runner.ROOT/'resultados/cwru/ampliacion_v2/pruebas_temporales_representantes'
    base.mkdir(parents=True,exist_ok=True)
    return tempfile.TemporaryDirectory(dir=base)


class ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol, cls.costs, cls.rows, _ = load_verified_context()
        cls.jobs = build_jobs(cls.protocol, cls.costs, cls.rows)

    def result(self):
        j = self.jobs[0]
        return dict(status='PASS', job_id=j['job_id'], candidate=j['candidate'], seed=j['seed'],
            roles=j['roles'], outer_load_hp=j['outer_load_hp'], inner_validation_load_hp=j['inner_validation_load_hp'],
            fit_signature='signature', training_windows=len(j['training_global_window_ids']),
            validation_windows=len(j['validation_global_window_ids']),
            representative_calibration_rows=len(j['training_global_window_ids']), benchmark_rows=28,
            parameters=20*j['input_dimension']+263, best_epoch=3, epochs_executed=13,
            weights_restored=True, outer_evaluation_executed=False, measured_STM32_total_cost=None,
            remaining_grid_enabled=False, metrics={'INT8':{'macro_f1':.8}})

    def test_signature_includes_validation(self):
        job = self.jobs[0]
        modified = copy.deepcopy(job)
        modified['validation_global_window_ids'] = modified['validation_global_window_ids'][::-1]
        args = ({'source':'hash'}, {'tensorflow':'version'}, self.protocol)
        self.assertNotEqual(runner.fit_signature(job,*args), runner.fit_signature(modified,*args))

    def test_signature_includes_sources_versions_and_seed(self):
        job = self.jobs[0]
        initial = runner.fit_signature(job, {'source':'hash'}, {'tensorflow':'v'}, self.protocol)
        for sources, versions in (({'source':'changed'},{'tensorflow':'v'}), ({'source':'hash'},{'tensorflow':'v2'})):
            self.assertNotEqual(initial, runner.fit_signature(job,sources,versions,self.protocol))
        modified = dict(job,seed=22)
        self.assertNotEqual(initial,runner.fit_signature(modified,{'source':'hash'},{'tensorflow':'v'},self.protocol))

    def test_result_scope_guard(self):
        result = self.result()
        runner.validate_result(result, self.jobs[0], 'signature')
        for key,value in (('outer_evaluation_executed',True),('measured_STM32_total_cost',1.),
                          ('remaining_grid_enabled',True),('weights_restored',False),('best_epoch',51),
                          ('representative_calibration_rows',1),('fit_signature','wrong')):
            bad = copy.deepcopy(result);bad[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):
                runner.validate_result(bad,self.jobs[0],'signature')

    def test_nan_metric_rejected(self):
        bad = self.result();bad['metrics']['INT8']['macro_f1']=float('nan')
        with self.assertRaises(ValueError):runner.validate_result(bad,self.jobs[0])

    def test_exclusive_json_preserves_original(self):
        with temporary_directory() as tmp:
            path=Path(tmp)/'result.json'
            runner.write(path,{'value':1})
            with self.assertRaises(FileExistsError):runner.write(path,{'value':2})
            self.assertEqual(json.loads(path.read_text()),{'value':1})

    def test_limits_do_not_enable_full_campaign(self):
        self.assertEqual(runner.LIMITS['maximum_fits'],32)
        self.assertEqual(runner.LIMITS['maximum_conversions'],64)
        self.assertEqual(runner.LIMITS['worker_seconds'],300)
        self.assertEqual(runner.LIMITS['campaign_worker_seconds'],7200)

    def test_failed_process_stops_without_result_read_or_retry(self):
        class Process:
            returncode=1
            def wait(self,timeout):return 1
            def poll(self):return 1
        with temporary_directory() as tmp:
            directory=Path(tmp)
            with patch.object(runner.subprocess,'Popen',return_value=Process()) as popen, \
                 patch.object(runner,'verify_job',side_effect=AssertionError('must not read failed result')):
                status=runner.run_one(directory,directory,self.jobs[0],{},300)
            self.assertEqual(status['status'],'FAIL')
            self.assertFalse(status['automatic_retry'])
            self.assertEqual(popen.call_count,1)

    def test_timeout_kills_child(self):
        class Process:
            returncode=None
            def __init__(self):self.killed=False
            def wait(self,timeout):
                if not self.killed:raise runner.subprocess.TimeoutExpired('synthetic',timeout)
                self.returncode=-1;return -1
            def poll(self):return self.returncode
            def kill(self):self.killed=True
        process=Process()
        with temporary_directory() as tmp:
            directory=Path(tmp)
            with patch.object(runner.subprocess,'Popen',return_value=process):
                status=runner.run_one(directory,directory,self.jobs[0],{},.001)
        self.assertTrue(process.killed)
        self.assertEqual(status['failure'],'TIMEOUT')

    def test_root_outside_campaign_rejected(self):
        with self.assertRaises(ValueError):runner.validate_campaign(runner.ROOT)


if __name__=='__main__':unittest.main()
