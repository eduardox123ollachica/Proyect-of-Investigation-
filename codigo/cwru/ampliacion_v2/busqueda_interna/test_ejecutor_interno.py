"""Guardas de scope, reanudacion y procesos; no TF, fits ni inferencias reales."""
from __future__ import annotations
import copy
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

import datos_busqueda as d
import ejecutar_busqueda_interna as e
import worker_busqueda_interna as w


@contextmanager
def fixture():
    base=(e.ROOT/'tmp').resolve();base.mkdir(exist_ok=True)
    directory=Path(tempfile.mkdtemp(prefix='inner_v2_test_',dir=base)).resolve()
    try:
        yield directory
    finally:
        e.require(directory.parent == base and directory.name.startswith('inner_v2_test_')
                  and directory.is_relative_to(base),'Fixture fuera del temporal autorizado')
        shutil.rmtree(directory)


def reseal_fixture(directory):
    """Volver a sellar una alteracion deliberada, solo dentro del fixture."""
    base=(e.ROOT/'tmp').resolve()
    e.require(directory.resolve().is_relative_to(base),'Solo resellar fixtures temporales')
    manifest=directory/'MANIFEST_JOB_SHA256.json'
    for path in (manifest,manifest.with_suffix('.sha256')):
        if path.exists():path.unlink()
    e.write_manifest(manifest,directory)


@contextmanager
def historical_result_fixture(job,protocol,inputs):
    """Adaptar metadatos de un ajuste ya realizado, sin entrenar ni inferir.

    Los modelos, checkpoints, scaler y predicciones siguen siendo los originales.
    La adaptacion de procedencia y firma reproduce el nuevo contenedor y se
    limita a una copia temporal; no representa una ejecucion nueva del worker.
    """
    original=e.REP_RUN/job['job_id']
    e.verify_manifest(original/'MANIFEST_JOB_SHA256.json',exact_directory=original)
    with fixture() as root:
        directory=root/'adapted_result';directory.mkdir()
        for path in original.iterdir():
            if path.is_file() and path.name not in ('MANIFEST_JOB_SHA256.json','MANIFEST_JOB_SHA256.sha256'):
                shutil.copyfile(path,directory/path.name)
        result=e.read(directory/'resultado.json')
        semantic=e.semantic_signature(job,protocol,result['versions'],inputs)
        signature=e.canonical(dict(synthetic_wrapper=True,job=job))
        preparation=root/'PREPARATION';preparation.write_bytes(b'synthetic preparation, no real fit')
        plan=root/'PLAN.json'
        e.write(plan,dict(tested_versions=result['versions'],semantic_signatures={job['job_id']:semantic}))
        result.update(fit_signature=signature,semantic_signature=semantic,
            original_training_core_source_sha256=e.sha(e.REP/'worker_representantes.py'),
            preparation_manifest_sha256=e.sha(preparation),detector_fit_executed=False,
            outer_evaluation_executed=False,predictive_selection_executed=False)
        (directory/'resultado.json').write_text(json.dumps(result,allow_nan=False),encoding='utf-8')
        provenance=d.load_job_data(job)[-1]
        (directory/'procedencia.json').write_text(json.dumps(provenance,allow_nan=False),encoding='utf-8')
        reseal_fixture(directory)
        with patch.object(e,'PLAN_FILE',plan),patch.object(e,'PREP_MANIFEST',preparation):
            yield directory,signature


def alter_npz(path,key,change):
    with np.load(path,allow_pickle=False) as saved:
        values={name:saved[name].copy() for name in saved.files}
    values[key]=change(values[key])
    np.savez(path,**values)


class InternalGuards(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol,cls.rows,cls.inputs=d.context()
        cls.jobs=d.build_jobs(cls.protocol,cls.rows)

    def test_plan_has_all_288_cells_and_no_outer_or_record_overlap(self):
        self.assertEqual(len(self.jobs),288)
        self.assertEqual(len({j['job_id'] for j in self.jobs}),288)
        for job in self.jobs:
            train,val=d.validate_scope(job,self.rows,self.protocol)
            self.assertTrue(all(int(r['load_hp']) != job['outer_load_hp'] for r in train+val))
            self.assertFalse({r['file_id'] for r in train}&{r['file_id'] for r in val})
        for outer in range(4):
            current=[j for j in self.jobs if j['outer_load_hp'] == outer]
            self.assertEqual(len(current),72)
            self.assertEqual({j['seed'] for j in current},{11,22,33})

    def test_original_32_job_dictionaries_are_preserved_exactly(self):
        original=d.old.build_jobs(self.protocol,d.old.load_verified_context()[1],self.rows)
        by_id={j['job_id']:j for j in self.jobs}
        self.assertEqual(len(original),32)
        self.assertTrue(all(by_id[j['job_id']] == j for j in original))

    def test_tampered_training_ids_seed_or_outer_role_are_rejected(self):
        original=self.jobs[0]
        for key,value in [('seed',44),('training_global_window_ids',original['validation_global_window_ids']),
                          ('outer_load_hp',original['inner_validation_load_hp'])]:
            job=copy.deepcopy(original);job[key]=value
            with self.assertRaises(ValueError):
                d.validate_scope(job,self.rows,self.protocol)

    def test_semantic_signature_binds_validation_seed_numeric_and_input_hash(self):
        job=self.jobs[0];versions={'python':'synthetic'}
        baseline=e.semantic_signature(job,self.protocol,versions,self.inputs)
        changed=copy.deepcopy(job);changed['seed']=22
        self.assertNotEqual(baseline,e.semantic_signature(changed,self.protocol,versions,self.inputs))
        changed=copy.deepcopy(job);changed['validation_global_window_ids']=job['validation_global_window_ids'][1:]
        self.assertNotEqual(baseline,e.semantic_signature(changed,self.protocol,versions,self.inputs))
        protocol=copy.deepcopy(self.protocol);protocol['numeric_contract']['quantization']='wrong'
        self.assertNotEqual(baseline,e.semantic_signature(job,protocol,versions,self.inputs))
        inputs=dict(self.inputs);inputs[job['authorized_feature_source_paths'][0]]='0'*64
        self.assertNotEqual(baseline,e.semantic_signature(job,self.protocol,versions,inputs))

    def test_loader_never_materializes_outer_rows_or_unapproved_arrays(self):
        job=next(j for j in self.jobs if j['outer_load_hp'] == 3)
        requested=[]
        allowed=set(job['training_global_window_ids']+job['validation_global_window_ids'])
        by_source={source:[r for r in self.rows if r['source_features'] == source]
                   for source in job['authorized_feature_source_paths']}
        class ScopedArray:
            dtype=np.dtype('float64')
            def __init__(self,source):
                self.source=source;self.shape=(len(by_source[source]),14)
                self.lookup={int(r['source_row']):int(r['global_window_id']) for r in by_source[source]}
            def __getitem__(self,index):
                gid=self.lookup[index]
                if gid not in allowed:
                    raise AssertionError('Excluded outer row materialized')
                requested.append(gid)
                return np.arange(14,dtype=np.float64)+index
        old_sha=d.old.sha256
        def fake_sha(path):
            relative=Path(path).resolve().relative_to(d.ROOT.resolve()).as_posix()
            return self.inputs[relative] if relative in by_source else old_sha(path)
        def fake_load(path,**kwargs):
            self.assertEqual(kwargs,dict(allow_pickle=False,mmap_mode='r'))
            return ScopedArray(Path(path).resolve().relative_to(d.ROOT.resolve()).as_posix())
        with patch.object(d,'context',return_value=(self.protocol,self.rows,self.inputs)), \
             patch.object(d.old,'sha256',side_effect=fake_sha),patch.object(d.np,'load',side_effect=fake_load):
            train,y,val,yval,provenance=d.load_job_data(job)
        self.assertEqual(requested,job['training_global_window_ids']+job['validation_global_window_ids'])
        self.assertEqual(train.dtype,np.float32);self.assertEqual(val.dtype,np.float32)
        self.assertFalse(provenance['outer_rows_selected_or_materialized'])
        self.assertFalse(provenance['detector_fit_executed'])

    def test_manifest_rejects_byte_change_and_unlisted_extra_file(self):
        with fixture() as directory:
            artifact=directory/'model.bin';artifact.write_bytes(b'model')
            manifest=directory/'MANIFEST.json'
            e.write_manifest(manifest,directory)
            e.verify_manifest(manifest,exact_directory=directory)
            (directory/'extra').write_bytes(b'extra')
            with self.assertRaises(ValueError):
                e.verify_manifest(manifest,exact_directory=directory)
            artifact.write_bytes(b'changed')
            with self.assertRaises(ValueError):
                e.verify_manifest(manifest)

    def test_path_confinement_rejects_traversal_and_absolute_paths(self):
        with fixture() as directory:
            self.assertEqual(e.confined(directory,'nested/file'),(directory/'nested/file').resolve())
            for bad in ('../outside','',str(directory/'absolute')):
                with self.assertRaises(ValueError):e.confined(directory,bad)

    def test_budget_boundary_does_not_start_truncated_worker(self):
        self.assertFalse(e.can_start(300,299,7200))
        self.assertFalse(e.can_start(300,1800,299))
        self.assertTrue(e.can_start(300,300,300))

    def test_unique_campaign_marker_rejects_copied_contract_budget_reset(self):
        with fixture() as directory:
            valid=directory/'valid';copied=directory/'copied'
            valid.mkdir();copied.mkdir()
            manifest=directory/'PREPARATION';manifest.write_bytes(b'fixed preparation')
            plan=dict(scope='synthetic_inner_scope')
            e.write(directory/'CAMPANA_UNICA.json',dict(run_id='valid',preparation_sha256=e.sha(manifest)))
            contract=dict(scope=plan['scope'],execution_owner='user_CLI',preparation_manifest_sha256=e.sha(manifest),
                          plan_sha256='synthetic_plan_hash',new_fits=240,imported_cells=48,automatic_retry=False)
            e.write(valid/'CONTRATO_CAMPANA.json',contract);e.write(copied/'CONTRATO_CAMPANA.json',contract)
            original_sha=e.sha
            def synthetic_sha(path):
                return 'synthetic_plan_hash' if Path(path) == e.PLAN_FILE else original_sha(path)
            with patch.object(e,'OUTPUT',directory),patch.object(e,'PREP_MANIFEST',manifest), \
                 patch.object(e,'preflight',return_value=plan),patch.object(e,'sha',side_effect=synthetic_sha):
                self.assertEqual(e.validate_campaign(valid),plan)
                with self.assertRaisesRegex(ValueError,'marcador unico'):
                    e.validate_campaign(copied)

    def test_timeout_kills_only_launched_worker_and_does_not_retry(self):
        class FakeProcess:
            returncode=None
            def __init__(self):self.killed=False;self.waits=0
            def wait(self,timeout):
                self.waits+=1
                if self.waits == 1:raise subprocess.TimeoutExpired('synthetic',timeout)
                self.returncode=-9;return -9
            def poll(self):return self.returncode
            def kill(self):self.killed=True
        process=FakeProcess()
        with fixture() as directory,patch.object(e.subprocess,'Popen',return_value=process) as launch:
            result=e.run_one(directory,directory,self.jobs[0],{},300,'synthetic')
            self.assertEqual(result['status'],'FAIL');self.assertEqual(result['failure'],'TIMEOUT')
            self.assertTrue(process.killed);self.assertEqual(launch.call_count,1)
            self.assertFalse(result['automatic_retry'])
            self.assertTrue((directory/('ESTADO_'+self.jobs[0]['job_id']+'.json')).is_file())

    def test_zero_exit_does_not_accept_missing_or_invalid_artifacts(self):
        class FakeProcess:
            returncode=0
            def wait(self,timeout):return 0
            def poll(self):return 0
        with fixture() as directory,patch.object(e.subprocess,'Popen',return_value=FakeProcess()), \
             patch.object(e,'verify_job',side_effect=ValueError('missing model')):
            result=e.run_one(directory,directory,self.jobs[0],{},300,'synthetic')
            self.assertEqual(result['status'],'FAIL')
            self.assertIn('missing model',result['result_error'])

    def test_failed_batch_blocks_resume(self):
        with fixture() as directory:
            (directory/'lotes/first').mkdir(parents=True)
            batch=directory/'lotes/first'
            e.write(batch/'RESUMEN_LOTE.json',dict(status='STOPPED_FOR_AUDIT',worker_elapsed_seconds=1))
            e.write_manifest(batch/'MANIFEST_LOTE_SHA256.json',batch)
            with patch.object(e,'validate_campaign',return_value=dict(jobs=[],imported={})), \
                 self.assertRaisesRegex(ValueError,'Fallo/interrupcion'):
                e.inspect_campaign(directory)

    def test_worker_reuses_original_functions_and_does_not_import_TensorFlow(self):
        import worker_representantes as core
        self.assertIs(w.fit_network,core.fit_network)
        self.assertIs(w.configure_tf,core.configure_tf)
        self.assertIs(w.export_models,core.export_models)
        self.assertIs(w.evaluate_lite,core.evaluate_lite)
        self.assertNotIn('tensorflow',sys.modules)

    def test_worker_rejects_reused_job_before_training(self):
        job=self.jobs[0]
        with fixture() as directory,patch.object(e,'validate_campaign',return_value=dict(jobs=[job],imported={job['job_id']:{}})), \
             patch.object(w,'configure_tf',side_effect=AssertionError('TF must not execute')) as configure:
            with self.assertRaisesRegex(ValueError,'repetir una celda importada'):
                w.run(directory,directory,job['job_id'])
            configure.assert_not_called()

    def test_import_adapter_rejects_nonidentity_preprocessing(self):
        job=self.jobs[0];source=copy.deepcopy(job);source['mode']='P1'
        with fixture() as directory,self.assertRaisesRegex(ValueError,'Familia'):
            e.import_result(job,source,directory,{},'suplemento_P0','bad',self.protocol,{},self.inputs)

    def test_existing_real_artifacts_pass_new_result_verifier_without_inference(self):
        job=next(j for j in self.jobs if j['job_id']=='cost_outer0_inner1_F14_seed11')
        with historical_result_fixture(job,self.protocol,self.inputs) as (directory,signature):
            result=e.verify_job(directory,job,signature)
            self.assertEqual(result['best_epoch'],48)
            self.assertEqual(result['epochs_executed'],50)
            self.assertEqual(result['metrics']['INT8']['macro_f1'],1.0)
            self.assertNotIn('tensorflow',sys.modules)

    def test_result_metadata_tampering_rejected_even_after_resealing(self):
        job=next(j for j in self.jobs if j['job_id']=='cost_outer0_inner1_F14_seed11')
        for field,value in [('seed',22),('semantic_signature','wrong'),
                            ('original_training_core_source_sha256','0'*64),
                            ('optimizer_updates',1299)]:
            with self.subTest(field=field),historical_result_fixture(job,self.protocol,self.inputs) as (directory,signature):
                result=e.read(directory/'resultado.json');result[field]=value
                (directory/'resultado.json').write_text(json.dumps(result,allow_nan=False),encoding='utf-8')
                reseal_fixture(directory)
                with self.assertRaises(ValueError):e.verify_job(directory,job,signature)

    def test_artifact_tampering_rejected_even_after_resealing(self):
        job=next(j for j in self.jobs if j['job_id']=='cost_outer0_inner1_F14_seed11')
        changes=[
            ('checkpoint_antes_conversion.npz','w0',lambda a:a[:-1], 'Pesos no restaurados'),
            ('scaler_train_only.npz','mean32',lambda a:a+np.float32(1), 'Scaler no reproduce'),
            ('predicciones_validacion.npz','int8_predictions',lambda a:(a+1)%3, 'Predicciones contradicen'),
            ('cuantizacion_train.npz','int8_inputs',lambda a:np.roll(a,1,axis=0), 'Cuantizacion train distinta')]
        for filename,key,change,message in changes:
            with self.subTest(artifact=filename),historical_result_fixture(job,self.protocol,self.inputs) as (directory,signature):
                if filename=='checkpoint_antes_conversion.npz':
                    with np.load(directory/filename,allow_pickle=False) as saved:key=saved.files[0]
                alter_npz(directory/filename,key,change)
                reseal_fixture(directory)
                with self.assertRaisesRegex(ValueError,message):e.verify_job(directory,job,signature)


if __name__=='__main__':
    unittest.main(verbosity=2)
