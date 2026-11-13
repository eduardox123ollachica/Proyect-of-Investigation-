"""Guardas del detector interno; solo inventario existente y datos sintéticos.

No ajusta parámetros con datos CWRU ni importa TensorFlow. Los ajustes
sintéticos explícitos no son experimentos nuevos del corpus.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import ejecutar_detector as runner
import datos_detector as d
from datos_busqueda import context
from detector_numerico import detector_score, fit_normality
from reglas_v2 import weighted_p95


@contextmanager
def fixture():
    base=(d.ROOT/'tmp').resolve()
    base.mkdir(exist_ok=True)
    folder=Path(tempfile.mkdtemp(prefix='detector_internal_test_',dir=base)).resolve()
    try:
        yield folder
    finally:
        if folder.parent != base or not folder.name.startswith('detector_internal_test_'):
            raise ValueError('Fixture fuera del temporal previsto')
        shutil.rmtree(folder)


def synthetic_payload():
    return dict(indices=np.asarray([0,1],dtype=np.int64),mean64=np.zeros(2,dtype=np.float64),
        scale64=np.ones(2,dtype=np.float64),var64=np.ones(2,dtype=np.float64),
        mean32=np.zeros(2,dtype=np.float32),scale32=np.ones(2,dtype=np.float32),
        centroid32=np.zeros(2,dtype=np.float32),threshold32=np.float32(4),
        fit_ids=np.asarray([1,2],dtype=np.int64),calibration_ids=np.asarray([3,4],dtype=np.int64))


class InventoryGuards(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.context=context()
        import detector_numerico as numeric
        # El inventario solo autentica artefactos previos: ni siquiera debe
        # invocar un ajuste numérico sobre sus muestras.
        with patch.object(numeric,'fit_normality',side_effect=AssertionError('No real detector fit during inventory')):
            cls.inventory=d.build_inventory()
        cls.rows=cls.context[1]
        cls.by_id={int(row['global_window_id']):row for row in cls.rows}

    def test_complete_inventory_and_reuse_counts(self):
        inventory=self.inventory
        for field,count in (('logical_jobs',96),('parameter_jobs',42),('evaluation_jobs',84),
            ('reused_parameters',23),('reused_rows',36),('reused_evaluations',30)):
            with self.subTest(field=field):
                self.assertEqual(len(inventory[field]),count)
        self.assertEqual(len(inventory['parameter_jobs'])-len(inventory['reused_parameters']),19)
        self.assertEqual(len(inventory['evaluation_jobs'])-len(inventory['reused_evaluations']),54)
        self.assertEqual(len(inventory['logical_jobs'])-len(inventory['reused_rows']),60)

    def test_parameter_scopes_are_normal_only_disjoint_and_seed_independent(self):
        identifiers=set()
        for job in self.inventory['parameter_jobs']:
            fit,cal=job['fit_ids'],job['calibration_ids']
            self.assertTrue(fit and cal)
            self.assertFalse(set(fit)&set(cal))
            self.assertTrue(all(self.by_id[i]['class']=='N' for i in fit+cal))
            fit_loads={int(self.by_id[i]['load_hp']) for i in fit}
            cal_loads={int(self.by_id[i]['load_hp']) for i in cal}
            self.assertEqual(len(fit_loads),1)
            self.assertEqual(len(cal_loads),1)
            self.assertLess(next(iter(fit_loads)),next(iter(cal_loads)))
            self.assertFalse({self.by_id[i]['file_id'] for i in fit}&{self.by_id[i]['file_id'] for i in cal})
            self.assertFalse(set(job['indices'])&{4,11})
            self.assertNotIn('seed',job)
            identifiers.add(job['param_id'])
        self.assertEqual(len(identifiers),42)

    def test_logical_rows_exclude_outer_and_validation_from_parameter_fitting(self):
        for job in self.inventory['logical_jobs']:
            fit,cal,val=job['fit_ids'],job['calibration_ids'],job['validation_ids']
            self.assertFalse(set(fit+cal)&set(val))
            self.assertFalse({self.by_id[i]['file_id'] for i in fit+cal}&{self.by_id[i]['file_id'] for i in val})
            self.assertTrue(all(int(self.by_id[i]['load_hp']) != job['outer_load_hp'] for i in fit+cal+val))
            self.assertTrue(all(int(self.by_id[i]['load_hp']) == job['inner_validation_load_hp'] for i in val))
            self.assertEqual({self.by_id[i]['class'] for i in val},{'N','IR','OR'})

    def test_F14_NF12_alias_both_parameters_and_evaluation_but_keep_logical_rows(self):
        lookup={(j['outer_load_hp'],j['inner_validation_load_hp'],j['candidate']):j
            for j in self.inventory['logical_jobs']}
        for outer in range(4):
            for inner in range(4):
                if outer==inner:continue
                full,nofreq=lookup[outer,inner,'F14'],lookup[outer,inner,'NF12']
                self.assertNotEqual(full['row_id'],nofreq['row_id'])
                self.assertEqual(full['indices'],nofreq['indices'])
                self.assertEqual(full['param_id'],nofreq['param_id'])
                self.assertEqual(full['eval_id'],nofreq['eval_id'])

    def test_same_fit_cal_pair_reuses_parameters_across_outer_inner_roles(self):
        lookup={(j['outer_load_hp'],j['inner_validation_load_hp'],j['candidate']):j
            for j in self.inventory['logical_jobs']}
        for outer in range(4):
            for inner in range(4):
                if outer==inner:continue
                first,swapped=lookup[outer,inner,'T6'],lookup[inner,outer,'T6']
                self.assertEqual(first['param_id'],swapped['param_id'])
                self.assertNotEqual(first['eval_id'],swapped['eval_id'])
                self.assertNotEqual(first['validation_ids'],swapped['validation_ids'])

    def test_payload_identity_ignores_zip_encoding_but_detects_single_parameter_bit(self):
        with fixture() as folder:
            first,encoded,changed=folder/'first.npz',folder/'compressed.npz',folder/'changed.npz'
            values=synthetic_payload()
            np.savez(first,**values)
            np.savez_compressed(encoded,**dict(reversed(list(values.items()))))
            self.assertNotEqual(first.read_bytes(),encoded.read_bytes())
            self.assertEqual(d.parameter_payload_sha256(first),d.parameter_payload_sha256(encoded))
            values['mean32'][0]=np.nextafter(np.float32(0),np.float32(1),dtype=np.float32)
            np.savez(changed,**values)
            self.assertNotEqual(d.parameter_payload_sha256(first),d.parameter_payload_sha256(changed))

    def test_loader_materializes_only_requested_rows_and_keeps_float32_contract(self):
        ids=sorted([self.inventory['parameter_jobs'][0]['fit_ids'][i] for i in (0,1)])
        allowed=set(ids)
        rows_by_source={path:[row for row in self.rows if row['source_features']==path]
            for path in {self.by_id[i]['source_features'] for i in ids}}
        visited=[]
        class ScopedArray:
            dtype=np.dtype('float64')
            def __init__(self,source):
                self.shape=(len(rows_by_source[source]),14)
                self.lookup={int(row['source_row']):int(row['global_window_id']) for row in rows_by_source[source]}
            def __getitem__(self,index):
                if isinstance(index,tuple):index=index[0]
                def one(position):
                    gid=self.lookup[int(position)]
                    if gid not in allowed:raise AssertionError('Unauthorized row materialized')
                    visited.append(gid)
                    return np.arange(14,dtype=np.float64)+gid
                if isinstance(index,(int,np.integer)):return one(index)
                return np.asarray([one(position) for position in index],dtype=np.float64)
        def fake_load(path,**kwargs):
            self.assertEqual(kwargs,dict(allow_pickle=False,mmap_mode='r'))
            source=Path(path).resolve().relative_to(d.ROOT.resolve()).as_posix()
            self.assertIn(source,rows_by_source)
            return ScopedArray(source)
        with patch.object(d.np,'load',side_effect=fake_load):
            raw,y,records,_=d.load_rows(ids,context=self.context)
        self.assertEqual(visited,ids)
        self.assertEqual(raw.dtype,np.float32)
        self.assertEqual(y.dtype,np.int64)
        self.assertEqual(records,[str(self.by_id[i]['file_id']) for i in ids])
        np.testing.assert_array_equal(raw,np.asarray([np.arange(14)+i for i in ids],dtype=np.float32))
        self.assertEqual(y.tolist(),[int(self.by_id[i]['label']) for i in ids])

    def test_loader_rejects_duplicate_out_of_range_or_boolean_ids_before_arrays(self):
        for ids in ([],[0,0],[-1],[len(self.rows)],[True],['0']):
            with self.subTest(ids=ids),patch.object(d.np,'load',side_effect=AssertionError('Array must not open')):
                with self.assertRaises((ValueError,TypeError)):
                    d.load_rows(ids,context=self.context)

    def test_inventory_and_imports_never_load_tensorflow(self):
        self.assertNotIn('tensorflow',sys.modules)
        self.assertNotIn('keras',sys.modules)


class NumericIntegrationGuards(unittest.TestCase):
    def test_worker_uses_the_original_numeric_core_functions(self):
        self.assertIs(runner.fit_normality,fit_normality)
        self.assertIs(runner.detector_score,detector_score)

    def test_float32_sequential_sum_is_not_promoted_or_reassociated(self):
        params=dict(indices=[0,1,2],mean32=np.zeros(3,dtype=np.float32),
            scale32=np.ones(3,dtype=np.float32),centroid32=np.zeros(3,dtype=np.float32),
            threshold32=np.float32(16777216))
        raw=np.zeros((1,14),dtype=np.float32)
        raw[0,:3]=[4096,1,1]
        scores=detector_score(raw,params)
        self.assertEqual(scores.dtype,np.float32)
        self.assertEqual(scores[0],np.float32(16777216))
        promoted=np.float32(np.sum(raw[0,:3].astype(np.float64)**2,dtype=np.float64))
        self.assertNotEqual(scores[0],promoted)
        self.assertFalse(bool((scores > params['threshold32'])[0]))

    def test_equal_record_mass_P95_and_strict_threshold_tie(self):
        scores=np.asarray([0]*100+[4],dtype=np.float32)
        threshold=weighted_p95(scores,['long']*100+['short'])
        self.assertEqual(threshold,np.float32(4))
        probes=np.asarray([0,4,np.nextafter(np.float32(4),np.float32(5))],dtype=np.float32)
        self.assertEqual((probes>threshold).tolist(),[False,False,True])


class ExecutionGuards(unittest.TestCase):
    def test_preflight_does_not_adjust_or_score_detector(self):
        plan=dict(source_sha256=runner.sources(),limits=runner.LIMITS,
            inventory=dict(reused_parameters={},reused_evaluations={}))
        with patch.object(runner.base,'verify_manifest'),patch.object(runner.base,'read',return_value=plan), \
             patch.object(runner.base,'preflight'), \
             patch.object(runner,'fit_normality',side_effect=AssertionError('No preflight fit')) as fit, \
             patch.object(runner,'detector_score',side_effect=AssertionError('No preflight evaluation')) as score:
            self.assertEqual(runner.preflight(),plan)
        fit.assert_not_called();score.assert_not_called()

    def test_timeout_kills_only_launched_worker_without_retry(self):
        class FakeProcess:
            returncode=None
            def __init__(self):self.waits=0;self.killed=False
            def wait(self,timeout):
                self.waits+=1
                if self.waits==1:raise subprocess.TimeoutExpired('synthetic_worker',timeout)
                self.returncode=-9;return -9
            def poll(self):return self.returncode
            def kill(self):self.killed=True
        process=FakeProcess()
        with fixture() as folder,patch.object(runner.subprocess,'Popen',return_value=process) as launch:
            status=runner.launch_worker(folder,timeout=300)
        self.assertEqual(status['status'],'FAIL')
        self.assertEqual(status['error'],'TIMEOUT')
        self.assertTrue(process.killed)
        self.assertEqual(launch.call_count,1)
        self.assertFalse(status['automatic_retry'])

    def test_worker_requires_active_parent_before_any_fit(self):
        with fixture() as folder,patch.object(runner,'validate_campaign',return_value={}), \
             patch.object(runner,'fit_normality',side_effect=AssertionError('No unauthorized fit')) as fit:
            with self.assertRaisesRegex(ValueError,'padre activo'):
                runner.worker(folder)
        fit.assert_not_called()

    def test_worker_rejects_fault_labels_in_fit_before_adjusting(self):
        job=dict(param_id='synthetic_parameter',fit_ids=[0],calibration_ids=[1],indices=[0])
        plan=dict(inventory=dict(parameter_jobs=[job],reused_parameters={}))
        metadata=[dict(global_window_id=str(i)) for i in range(2)]
        values=np.zeros((1,14),dtype=np.float32)
        fit_rows=(values,np.asarray([1],dtype=np.int64),['fit_record'],{})
        cal_rows=(values,np.asarray([0],dtype=np.int64),['cal_record'],{})
        import datos_busqueda
        with fixture() as folder:
            (folder/'ACTIVO.lock').write_text('synthetic_parent',encoding='utf-8')
            (folder/'parametros').mkdir()
            with patch.object(runner,'validate_campaign',return_value=plan), \
                 patch.dict(runner.os.environ,runner.base.ENVIRONMENT), \
                 patch.object(datos_busqueda,'context',return_value=({},metadata,{})), \
                 patch.object(d,'load_rows',side_effect=[fit_rows,cal_rows]), \
                 patch.object(runner,'fit_normality',side_effect=AssertionError('No fitting fault data')) as fit:
                with self.assertRaisesRegex(ValueError,'no normales'):
                    runner.worker(folder)
            fit.assert_not_called()


if __name__=='__main__':
    unittest.main(verbosity=2)
