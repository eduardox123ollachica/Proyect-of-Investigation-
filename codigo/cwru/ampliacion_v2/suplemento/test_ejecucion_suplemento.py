"""Puertas del suplemento: sin procesos reales de entrenamiento."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import ejecutar_suplemento as run


def tempdir():
    parent=run.ROOT/'resultados/cwru/ampliacion_v2/pruebas_temporales_suplemento'
    parent.mkdir(parents=True,exist_ok=True)
    return tempfile.TemporaryDirectory(dir=parent)


class SupplementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.items=run.jobs();cls.protocol,_,_=run.context()

    def result(self):
        j=self.items[1]
        return dict(status='PASS',job_id=j['job_id'],candidate=j['candidate'],mode=j['mode'],seed=j['seed'],
            roles=j['roles'],fit_signature='sig',training_windows=len(j['training_global_window_ids']),
            validation_windows=len(j['validation_global_window_ids']),
            representative_calibration_rows=len(j['training_global_window_ids']),parameters=20*j['input_dimension']+263,
            best_epoch=1,epochs_executed=11,weights_restored=True,outer_predictive_evaluation=False,
            mask_or_preprocessing_selected=False,STM32_cost_measured=False,metrics={'INT8':{'macro_f1':.5}})

    def test_result_guards(self):
        valid=self.result();run.validate_result(valid,self.items[1],'sig')
        for key,value in [('mode','P0'),('fit_signature','other'),('outer_predictive_evaluation',True),
                          ('STM32_cost_measured',True),('mask_or_preprocessing_selected',True),('best_epoch',51),
                          ('representative_calibration_rows',1)]:
            bad=copy.deepcopy(valid);bad[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):run.validate_result(bad,self.items[1],'sig')

    def test_nan_f1_rejected(self):
        bad=self.result();bad['metrics']['INT8']['macro_f1']=float('nan')
        with self.assertRaises(ValueError):run.validate_result(bad,self.items[1],'sig')

    def test_signature_contains_mode_seed_scope_and_sources(self):
        j=self.items[0];base=run.signature(j,{'a':'b'},{'v':'1'},self.protocol)
        self.assertNotEqual(base,run.signature(self.items[1],{'a':'b'},{'v':'1'},self.protocol))
        for key,value in [('seed',22),('validation_global_window_ids',j['validation_global_window_ids'][::-1])]:
            bad=dict(j,**{key:value})
            self.assertNotEqual(base,run.signature(bad,{'a':'b'},{'v':'1'},self.protocol))
        self.assertNotEqual(base,run.signature(j,{'a':'c'},{'v':'1'},self.protocol))

    def test_counts_no_main_mass_run(self):
        self.assertEqual(run.LIMITS['maximum_new_DNN_fits'],34)
        self.assertEqual(run.LIMITS['maximum_new_conversions'],68)
        self.assertEqual(run.LIMITS['maximum_new_detector_fits'],5)
        self.assertEqual(run.LIMITS['campaign_process_seconds'],7200)

    def test_original_two_references_imported_readonly(self):
        mappings=run.imported_references()
        self.assertEqual(set(mappings),{'supp_o3_i0_F14_P0_s11','supp_o3_i0_NF12_P0_s11'})
        self.assertTrue(all(m['new_training_fits']==0 and len(m['source_fit_signature'])==64 for m in mappings.values()))

    def test_create_exclusive_preserves_existing_file(self):
        with tempdir() as tmp:
            p=Path(tmp)/'a.json';run.write(p,{'a':1})
            with self.assertRaises(FileExistsError):run.write(p,{'a':2})
            self.assertEqual(run.read(p),{'a':1})

    def test_child_failure_does_not_retry(self):
        class Process:
            returncode=1
            def wait(self,timeout):return 1
            def poll(self):return 1
        with tempdir() as tmp:
            directory=Path(tmp)
            with patch.object(run.subprocess,'Popen',return_value=Process()) as popen:
                status=run.process_one(directory,directory,'synthetic',5)
            self.assertEqual(popen.call_count,1);self.assertEqual(status['status'],'FAIL')

    def test_child_timeout_killed(self):
        class Process:
            returncode=None
            def __init__(self):self.killed=False
            def poll(self):return self.returncode
            def kill(self):self.killed=True;self.returncode=-1
            def wait(self,timeout):
                if not self.killed:raise run.subprocess.TimeoutExpired('synthetic',timeout)
                return -1
        process=Process()
        with tempdir() as tmp:
            directory=Path(tmp)
            with patch.object(run.subprocess,'Popen',return_value=process):
                status=run.process_one(directory,directory,'synthetic',.001)
        self.assertTrue(process.killed);self.assertEqual(status['error'],'TIMEOUT')


if __name__=='__main__':unittest.main()
