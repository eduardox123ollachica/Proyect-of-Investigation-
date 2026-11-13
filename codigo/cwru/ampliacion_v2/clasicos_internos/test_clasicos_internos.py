"""Guardas de comparadores internos; sin fit o inferencias sobre CWRU.

Los únicos ajustes permitidos en esta suite usan arrays sintéticos pequeños.
Las lecturas del corpus se limitan a sus contratos e índices ya sellados.
"""
from __future__ import annotations

import copy
from contextlib import contextmanager
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import datos_clasicos as data
from datos_busqueda import context
from contrato_piloto import build_jobs as pilot_grid
import presupuesto_clasicos_internos as budget_source
import ejecutar_clasicos as runner


@contextmanager
def fixture():
    base=(data.ROOT/'tmp').resolve()
    base.mkdir(exist_ok=True)
    folder=Path(tempfile.mkdtemp(prefix='classical_internal_test_',dir=base)).resolve()
    try:
        yield folder
    finally:
        if folder.parent != base or not folder.name.startswith('classical_internal_test_'):
            raise ValueError('Fixture fuera de su temporal autorizado')
        shutil.rmtree(folder)


class ClassicalDataGuards(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol,cls.rows,cls.hashes=context()
        cls.jobs=data.build_jobs(cls.protocol,cls.rows)

    def test_matrix_has_288_RF_and_96_single_fit_SVM_cells(self):
        self.assertEqual(len(self.jobs),384)
        self.assertEqual(len({j['job_id'] for j in self.jobs}),384)
        self.assertEqual(sum(j['family']=='RF' for j in self.jobs),288)
        self.assertEqual(sum(j['family']=='SVM' for j in self.jobs),96)
        for outer in range(4):
            for inner in range(4):
                if inner == outer:continue
                current=[j for j in self.jobs if j['outer_load_hp']==outer and j['inner_validation_load_hp']==inner]
                self.assertEqual(len(current),32)
                self.assertEqual({j['config_index'] for j in current},set(range(8)))
                self.assertEqual({j['seed'] for j in current if j['family']=='RF'},{11,22,33})
                self.assertTrue(all(j['seed'] is None for j in current if j['family']=='SVM'))

    def test_grid_params_and_order_preserve_original_pilot_catalogue(self):
        catalogue={(j['family'],j['config_index']):j for j in pilot_grid(self.protocol) if j['family'] in ('RF','SVM')}
        for job in self.jobs:
            with self.subTest(job=job['job_id']):
                expected=copy.deepcopy(catalogue[job['family'],job['config_index']]['params'])
                if job['family']=='RF':expected['random_state']=job['seed']
                self.assertEqual(job['params'],expected)
                self.assertEqual(job['feature_mask'],'F14')
        first=[j for j in self.jobs if j['outer_load_hp']==0 and j['inner_validation_load_hp']==1]
        self.assertEqual([(j['family'],j['config_index'],j['seed']) for j in first],
            [('RF',config,seed) for config in range(8) for seed in (11,22,33)]
            +[('SVM',config,None) for config in range(8)])

    def test_every_job_excludes_outer_and_has_disjoint_MAT_and_window_roles(self):
        for job in self.jobs:
            data.validate_scope(job,self.protocol,self.rows)
            train,val=job['training_global_window_ids'],job['validation_global_window_ids']
            self.assertFalse(set(train)&set(val))
            self.assertTrue(all(int(self.rows[i]['load_hp']) != job['outer_load_hp'] for i in train+val))
            self.assertFalse({self.rows[i]['file_id'] for i in train}&{self.rows[i]['file_id'] for i in val})
            self.assertEqual({int(self.rows[i]['load_hp']) for i in train},set(job['roles']['training_loads']))
            self.assertEqual({int(self.rows[i]['load_hp']) for i in val},{job['inner_validation_load_hp']})
            self.assertEqual(len({self.rows[i]['file_id'] for i in train}),14)
            self.assertEqual(len({self.rows[i]['file_id'] for i in val}),7)

    def test_tampered_seed_params_or_role_is_rejected(self):
        rf=next(j for j in self.jobs if j['family']=='RF')
        svm=next(j for j in self.jobs if j['family']=='SVM')
        altered=[]
        job=copy.deepcopy(rf);job['seed']=44;altered.append(job)
        job=copy.deepcopy(rf);job['params']['n_estimators']=200;altered.append(job)
        job=copy.deepcopy(rf);job['training_global_window_ids']=job['validation_global_window_ids'];altered.append(job)
        job=copy.deepcopy(rf);job['outer_load_hp']=job['inner_validation_load_hp'];altered.append(job)
        job=copy.deepcopy(svm);job['seed']=11;altered.append(job)
        for job in altered:
            with self.subTest(job=job['job_id']),self.assertRaises(ValueError):
                data.validate_scope(job,self.protocol,self.rows)

    def test_importing_data_and_building_jobs_does_not_import_TensorFlow(self):
        self.assertNotIn('tensorflow',sys.modules)
        self.assertNotIn('keras',sys.modules)

    def test_loader_requests_only_authorized_training_and_validation_rows(self):
        job=next(j for j in self.jobs if j['family']=='SVM' and j['outer_load_hp']==3)
        expected=[job['training_global_window_ids'],job['validation_global_window_ids']]
        observed=[]
        def fake_load_rows(ids,context=None):
            self.assertEqual(ids,expected[len(observed)])
            self.assertIs(context[0],self.protocol)
            self.assertTrue(all(int(self.rows[i]['load_hp']) != 3 for i in ids))
            observed.append(list(ids))
            values=np.asarray([[i/100+column for column in range(14)] for i in ids],dtype=np.float32)
            y=np.asarray([int(self.rows[i]['label']) for i in ids],dtype=np.int64)
            records=[str(self.rows[i]['file_id']) for i in ids]
            return values,y,records,dict(synthetic_fixture=True,ids=list(ids))
        with patch.object(data.detector_data,'load_rows',side_effect=fake_load_rows):
            x,y,val,y_val,provenance=data.load_job_data(job,context=(self.protocol,self.rows,self.hashes))
        self.assertEqual(observed,expected)
        self.assertEqual(x.shape,(len(expected[0]),14));self.assertEqual(val.shape,(len(expected[1]),14))
        self.assertEqual(x.dtype,np.float32);self.assertEqual(val.dtype,np.float32)
        self.assertEqual(y.dtype,np.int64);self.assertEqual(y_val.dtype,np.int64)
        self.assertFalse(provenance['scaler_fitted_by_this_loader'])
        self.assertFalse(provenance['model_fitted_by_this_loader'])
        self.assertFalse(provenance['prediction_executed_by_this_loader'])

    def test_training_signature_binds_roles_params_versions_sources_and_inputs(self):
        job=self.jobs[0]
        versions=dict(python='synthetic',numpy='synthetic',sklearn='synthetic')
        sources={'synthetic_worker.py':'0'*64}
        baseline=data.training_signature(job,self.protocol,self.hashes,versions,sources)
        different=copy.deepcopy(job);different['seed']=22;different['params']['random_state']=22
        self.assertNotEqual(baseline,data.training_signature(different,self.protocol,self.hashes,versions,sources))
        different=copy.deepcopy(job);different['validation_global_window_ids']=different['validation_global_window_ids'][1:]
        self.assertNotEqual(baseline,data.training_signature(different,self.protocol,self.hashes,versions,sources))
        changed=dict(self.hashes);changed[job['authorized_feature_source_paths'][0]]='0'*64
        self.assertNotEqual(baseline,data.training_signature(job,self.protocol,changed,versions,sources))
        self.assertNotEqual(baseline,data.training_signature(job,self.protocol,self.hashes,
            dict(versions,sklearn='changed'),sources))
        self.assertNotEqual(baseline,data.training_signature(job,self.protocol,self.hashes,versions,
            {'synthetic_worker.py':'1'*64}))


class ClassicalBudgetGuards(unittest.TestCase):
    def test_budget_uses_audited_timing_only_and_preserves_scope_and_limits(self):
        budget=budget_source.build_budget()
        self.assertEqual(budget['planned_inner_cells'],384)
        self.assertEqual(budget['RF_inner_fits'],288)
        self.assertEqual(budget['SVM_inner_fits'],96)
        self.assertEqual(budget['reused_confirmatory_fits'],0)
        self.assertFalse(budget['pilot_models_reused'])
        self.assertTrue(budget['pilot_not_repeated_for_timing'])
        self.assertFalse(budget['model_fit_deduplication_across_swapped_validation_outer_roles_enabled'])
        self.assertEqual(budget['limits'],dict(RF_worker_seconds=60,SVM_worker_seconds=30,
            campaign_execution_verification_seconds=3600,maximum_new_fits=384,
            concurrent_workers=1,automatic_retry=False))
        self.assertEqual(len(budget['configuration_profiles']),16)
        self.assertEqual(budget['new_fits_during_budget_preparation'],0)
        self.assertEqual(budget['new_inferences_during_budget_preparation'],0)
        self.assertFalse(budget['TensorFlow_imported'])
        self.assertLess(budget['forecast']['all_family_jobs_at_observed_family_max_seconds'],3600)


class ClassicalNumericalGuards(unittest.TestCase):
    def test_RF_receives_raw_float32_without_scaler_or_inplace_changes(self):
        train=np.arange(84,dtype=np.float32).reshape(6,14)
        val=np.arange(42,dtype=np.float32).reshape(3,14)+np.float32(1000)
        before_train=train.copy();before_val=val.copy()
        x,v,scaler=runner.prepare_inputs('RF',train,val)
        self.assertIsNone(scaler)
        self.assertTrue(np.array_equal(x.view(np.uint32),before_train.view(np.uint32)))
        self.assertTrue(np.array_equal(v.view(np.uint32),before_val.view(np.uint32)))
        self.assertTrue(np.array_equal(train,before_train));self.assertTrue(np.array_equal(val,before_val))

    def test_SVM_scaler_uses_only_train_and_materializes_float32_operations(self):
        train=np.arange(84,dtype=np.float32).reshape(6,14)*np.float32(.5)
        val=np.arange(42,dtype=np.float32).reshape(3,14)+np.float32(100000)
        x,v,scaler=runner.prepare_inputs('SVM',train,val)
        expected_mean=np.mean(train,axis=0,dtype=np.float64)
        expected_var=np.var(train,axis=0,dtype=np.float64)
        self.assertTrue(np.array_equal(scaler['mean64'],expected_mean))
        self.assertTrue(np.array_equal(scaler['var64'],expected_var))
        self.assertTrue(np.array_equal(scaler['scale64'],np.sqrt(expected_var)))
        self.assertEqual(scaler['mean32'].dtype,np.float32);self.assertEqual(scaler['scale32'].dtype,np.float32)
        expected_x=np.divide(np.subtract(train,scaler['mean32'],dtype=np.float32),scaler['scale32'],dtype=np.float32)
        expected_v=np.divide(np.subtract(val,scaler['mean32'],dtype=np.float32),scaler['scale32'],dtype=np.float32)
        self.assertEqual(x.dtype,np.float32);self.assertEqual(v.dtype,np.float32)
        self.assertTrue(np.array_equal(x.view(np.uint32),expected_x.view(np.uint32)))
        self.assertTrue(np.array_equal(v.view(np.uint32),expected_v.view(np.uint32)))
        self.assertFalse(np.array_equal(scaler['mean64'],np.mean(np.concatenate([train,val]),axis=0)))

    def test_invalid_nonfinite_or_wrong_input_representation_is_rejected(self):
        valid=np.ones((3,14),dtype=np.float32)
        for invalid in (valid.astype(np.float64),valid[:,:13],np.full((3,14),np.nan,dtype=np.float32)):
            with self.subTest(dtype=invalid.dtype,shape=invalid.shape),self.assertRaises(ValueError):
                runner.prepare_inputs('SVM',invalid,valid)

    def test_SVM_predictions_preserve_native_vote_even_if_decision_argmax_differs(self):
        class FakeSVM:
            def predict(self,values):return np.asarray([2,1],dtype=np.int64)
            def decision_function(self,values):return np.asarray([[9.,0.,1.],[5.,2.,1.]],dtype=np.float64)
        native,diagnostics=runner.predict_estimator(FakeSVM(),'SVM',np.zeros((2,14),dtype=np.float32))
        self.assertEqual(native.tolist(),[2,1])
        self.assertEqual(np.argmax(diagnostics,axis=1).tolist(),[0,0])

    def test_tiny_synthetic_estimators_roundtrip_joblib_and_native_predictions(self):
        import joblib
        protocol=context()[0]
        templates=pilot_grid(protocol)
        rng=np.random.default_rng(31)
        labels=np.repeat(np.arange(3,dtype=np.int64),12)
        raw=(rng.normal(0,.15,size=(36,14))+(labels[:,None]-1)*3).astype(np.float32)
        val=(rng.normal(0,.15,size=(6,14))+(np.repeat(np.arange(3),2)[:,None]-1)*3).astype(np.float32)
        for family in ('RF','SVM'):
            job=next(j for j in templates if j['family']==family and j['config_index']==0)
            with self.subTest(family=family),fixture() as folder:
                x,v,_=runner.prepare_inputs(family,raw,val)
                model=runner.make_estimator(job)
                model.fit(x,labels)
                self.assertEqual(model.classes_.tolist(),[0,1,2])
                self.assertTrue(all(model.get_params()[k] == value for k,value in job['params'].items()))
                pred,values=runner.predict_estimator(model,family,v)
                self.assertEqual(pred.shape,(6,));self.assertEqual(pred.dtype,np.int64)
                self.assertEqual(values.shape,(6,3))
                if family == 'RF':self.assertEqual(len(model.estimators_),100)
                else:
                    self.assertEqual(model.fit_status_,0)
                    self.assertEqual(model.support_vectors_.dtype,np.float64)
                path=folder/'synthetic_model.joblib'
                joblib.dump(model,path,compress=3)
                loaded=joblib.load(path)
                after_pred,after_values=runner.predict_estimator(loaded,family,v)
                self.assertTrue(np.array_equal(pred,after_pred))
                self.assertTrue(np.array_equal(values,after_values))
        self.assertNotIn('tensorflow',sys.modules)


class ClassicalExecutionGuards(unittest.TestCase):
    def test_preflight_cannot_fit_or_predict_a_classifier(self):
        limits=dict(RF_worker_seconds=60,SVM_worker_seconds=30,campaign_execution_verification_seconds=3600,
            maximum_new_fits=384,concurrent_workers=1,automatic_retry=False)
        plan=dict(source_sha256={'synthetic_source':'hash'},limits=limits,budget_sha256='budget-hash',
            logical_cells=384,RF_fits=288,SVM_fits=96,jobs=[{}]*384,imported_fits=0,
            selection_enabled=False,outer_evaluation_enabled=False,DNN_enabled=False,detector_enabled=False,
            hardware_enabled=False,automatic_retry=False,worker_environment=runner.io.ENVIRONMENT)
        def fake_read(path):return plan if Path(path)==runner.PLAN else dict(limits=limits)
        with patch.object(runner.io,'verify_manifest',return_value={}),patch.object(runner.io,'read',side_effect=fake_read),\
             patch.object(runner.io,'sha',return_value='budget-hash'),patch.object(runner,'sources',return_value=plan['source_sha256']),\
             patch.object(runner,'make_estimator',side_effect=AssertionError('No estimator during preflight')) as estimator,\
             patch.object(runner.data,'load_job_data',side_effect=AssertionError('No data matrix during preflight')) as loader,\
             patch.object(runner,'predict_estimator',side_effect=AssertionError('No predictions during preflight')) as predict:
            self.assertIs(runner.preflight(environment=False,deep=False),plan)
            estimator.assert_not_called();loader.assert_not_called();predict.assert_not_called()

    def test_budget_needs_whole_family_timeout_and_does_not_reset_consumed_time(self):
        self.assertTrue(runner.can_start(60,3540,3600))
        self.assertFalse(runner.can_start(60,3540.1,3600))
        self.assertTrue(runner.can_start(30,3570,3600))
        self.assertFalse(runner.can_start(30,3570.1,3600))

    def test_timeout_kills_only_the_launched_worker_without_retry(self):
        class FakeProcess:
            returncode=None
            def __init__(self):self.waits=0;self.killed=False
            def wait(self,timeout):
                self.waits+=1
                if self.waits == 1:raise subprocess.TimeoutExpired('synthetic',timeout)
                self.returncode=-9;return -9
            def poll(self):return self.returncode
            def kill(self):self.killed=True
        process=FakeProcess();job=dict(job_id='synthetic_RF')
        plan=dict(fit_signatures={'synthetic_RF':'signature'},worker_environment={})
        with fixture() as folder,patch.object(runner.subprocess,'Popen',return_value=process) as launch:
            status=runner.launch_one(folder,job,plan,60)
            self.assertEqual(status['status'],'FAIL');self.assertEqual(status['error'],'TIMEOUT')
            self.assertFalse(status['automatic_retry']);self.assertTrue(process.killed)
            self.assertEqual(launch.call_count,1)
            self.assertFalse((folder/'SOLICITUD_ACTIVA.json').exists())

    def test_zero_exit_does_not_accept_missing_or_invalid_model_artifacts(self):
        class FakeProcess:
            returncode=0
            def wait(self,timeout):return 0
            def poll(self):return 0
        job=dict(job_id='synthetic_SVM')
        plan=dict(fit_signatures={'synthetic_SVM':'signature'},worker_environment={})
        with fixture() as folder,patch.object(runner.subprocess,'Popen',return_value=FakeProcess()),\
             patch.object(runner,'verify_job',side_effect=ValueError('Invalid saved estimator')):
            status=runner.launch_one(folder,job,plan,30)
            self.assertEqual(status['status'],'FAIL')
            self.assertIn('Invalid saved estimator',status['error'])
            self.assertFalse(status['automatic_retry'])

    def test_unique_marker_rejects_copied_contract_and_budget_reset(self):
        with fixture() as folder:
            valid=folder/'valid';copied=folder/'copied';valid.mkdir();copied.mkdir()
            manifest=folder/'preparation.json';manifest.write_bytes(b'synthetic preparation')
            planfile=folder/'plan.json';planfile.write_bytes(b'synthetic plan')
            runner.io.write(folder/'CAMPANA_UNICA.json',dict(run_id='valid',preparation_sha256=runner.io.sha(manifest)))
            contract=dict(preparation_sha256=runner.io.sha(manifest),plan_sha256=runner.io.sha(planfile),
                scope='classical_inner_only',execution_owner='user_CLI')
            runner.io.write(valid/'CONTRATO.json',contract);runner.io.write(copied/'CONTRATO.json',contract)
            with patch.object(runner,'OUTPUT',folder),patch.object(runner,'MANIFEST',manifest),\
                 patch.object(runner,'PLAN',planfile),patch.object(runner,'preflight',return_value={'synthetic':True}):
                self.assertEqual(runner.validate_campaign(valid),{'synthetic':True})
                with self.assertRaises(ValueError):runner.validate_campaign(copied)

    def test_resume_checks_complete_prefix_and_preserves_accumulated_budget(self):
        jobs=[dict(job_id=f'synthetic_{i}',timeout_seconds=60) for i in range(3)]
        with fixture() as folder:
            (folder/'pasos').mkdir()
            for index,seconds in enumerate((12.5,8.25)):
                (folder/jobs[index]['job_id']).mkdir()
                runner.io.write(folder/'pasos'/f'{index:03d}.json',dict(status='PASS',job_id=jobs[index]['job_id'],
                    automatic_retry=False,execution_verification_seconds=seconds,timeout_seconds=60))
            plan=dict(jobs=jobs)
            with patch.object(runner,'validate_campaign',return_value=plan),\
                 patch.object(runner,'verify_job',return_value={}) as verify:
                observed,done,consumed=runner.inspect_campaign(folder)
            self.assertEqual(done,['synthetic_0','synthetic_1']);self.assertEqual(consumed,20.75)
            self.assertEqual(verify.call_count,2)

    def test_failed_or_interrupted_step_blocks_resume(self):
        with fixture() as folder:
            (folder/'pasos').mkdir()
            runner.io.write(folder/'pasos/000.json',dict(status='FAIL',job_id='synthetic_0',automatic_retry=False))
            with patch.object(runner,'validate_campaign',return_value=dict(jobs=[dict(job_id='synthetic_0',timeout_seconds=60)])),\
                 patch.object(runner,'verify_job',side_effect=AssertionError('No model replay after failure')):
                with self.assertRaises(ValueError):runner.inspect_campaign(folder)

    def test_hole_or_unclosed_job_cannot_be_silently_skipped(self):
        jobs=[dict(job_id='synthetic_0',timeout_seconds=60),dict(job_id='synthetic_1',timeout_seconds=60)]
        with fixture() as folder:
            (folder/'pasos').mkdir();(folder/'synthetic_1').mkdir()
            runner.io.write(folder/'pasos/001.json',dict(status='PASS',job_id='synthetic_1',automatic_retry=False,
                execution_verification_seconds=1.,timeout_seconds=60))
            with patch.object(runner,'validate_campaign',return_value=dict(jobs=jobs)),patch.object(runner,'verify_job',return_value={}):
                with self.assertRaises(ValueError):runner.inspect_campaign(folder)
        with fixture() as folder:
            (folder/'pasos').mkdir();(folder/'synthetic_0').mkdir()
            with patch.object(runner,'validate_campaign',return_value=dict(jobs=jobs)):
                with self.assertRaises(ValueError):runner.inspect_campaign(folder)

    def test_worker_rejects_missing_parent_or_wrong_request_before_loading_data(self):
        job=dict(job_id='synthetic_RF')
        plan=dict(jobs=[job],fit_signatures={'synthetic_RF':'signature'})
        with fixture() as folder,patch.object(runner,'validate_campaign',return_value=plan),\
             patch.object(runner.data,'load_job_data',side_effect=AssertionError('Dataset must not be opened')) as load:
            with self.assertRaises(ValueError):runner.worker(folder,'synthetic_RF')
            (folder/'ACTIVO.lock').write_bytes(b'synthetic parent')
            runner.io.write(folder/'SOLICITUD_ACTIVA.json',dict(job_id='another_job',fit_signature='signature'))
            with self.assertRaises(ValueError):runner.worker(folder,'synthetic_RF')
            load.assert_not_called()


if __name__ == '__main__':
    unittest.main(verbosity=2)
