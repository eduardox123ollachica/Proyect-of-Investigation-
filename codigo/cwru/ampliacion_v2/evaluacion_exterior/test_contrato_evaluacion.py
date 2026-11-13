"""Pruebas operativas de evaluación; sin ML ni materialización de valores CWRU."""
from __future__ import annotations

import copy
from contextlib import contextmanager
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock,patch

import numpy as np
import contrato_evaluacion as c

_CONTEXT=None
_JOBS=None


def actual_context():
    global _CONTEXT,_JOBS
    if _CONTEXT is None:
        with patch.object(np,'load',side_effect=AssertionError('No cargar valores de características')) as loader:
            _CONTEXT=c.closed_context()
            _JOBS=c.build_jobs(_CONTEXT[0],_CONTEXT[1],_CONTEXT[3],_CONTEXT[4],_CONTEXT[5])
            loader.assert_not_called()
    return _CONTEXT


def build_changed(context):
    protocol,rows,_,mapping,inventory,refit_plan=context
    return c.build_jobs(protocol,rows,mapping,inventory,refit_plan)


@contextmanager
def fixture():
    base=(c.ROOT/'tmp').resolve();base.mkdir(exist_ok=True)
    folder=Path(tempfile.mkdtemp(prefix='outer_eval_contract_test_',dir=base)).resolve()
    try:yield folder
    finally:
        if folder.parent!=base or not folder.name.startswith('outer_eval_contract_test_'):
            raise ValueError('Fixture fuera del temporal autorizado')
        shutil.rmtree(folder)


@contextmanager
def authorized_fixture():
    with fixture() as folder:
        output=folder/'campaigns';output.mkdir();campaign=output/'run001';campaign.mkdir()
        prep=folder/'preparation';prep.mkdir()
        manifest=prep/'MANIFEST.json';manifest.write_text('{}\n',encoding='utf-8')
        plan_path=prep/'PLAN.json';plan_path.write_text('{}\n',encoding='utf-8')
        jobs=[dict(job_id=f'outer_eval_{outer}',outer_load_hp=outer,timeout_seconds=300) for outer in range(4)]
        plan=dict(jobs=jobs,evaluation_signatures={job['job_id']:'s'+str(i) for i,job in enumerate(jobs)},
            worker_environment={'SYNTHETIC_OUTER_EVAL_TEST':'1'},limits=copy.deepcopy(c.LIMITS))
        with patch.object(c,'OUTPUT',output),patch.object(c,'MANIFEST',manifest),patch.object(c,'PLAN',plan_path),\
             patch.object(c,'preflight',return_value=plan),patch.dict(c.os.environ,plan['worker_environment']):
            c.io.write(campaign/'CONTRATO.json',c.campaign_contract(plan))
            c.io.write(output/'CAMPANA_UNICA.json',dict(run_id=campaign.name,preparation_sha256=c.io.sha(manifest)))
            (campaign/'ACTIVO.lock').write_text('synthetic-only\n',encoding='utf-8')
            c.io.write(campaign/'SOLICITUD_ACTIVA.json',dict(job_id='outer_eval_0',outer_load_hp=0,evaluation_signature='s0'))
            yield campaign,plan


class ClosedInventoryAndRoles(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.context=actual_context();cls.jobs=_JOBS

    def test_closed_context_and_inventory_do_not_load_feature_values_or_ML(self):
        self.assertEqual(len(self.context[1]),1380)
        self.assertEqual(len(self.context[4]),131)
        self.assertNotIn('tensorflow',sys.modules)
        self.assertNotIn('sklearn',sys.modules)
        with patch.object(np,'load',side_effect=AssertionError('No materializar reserva')) as loader:
            self.assertEqual(build_changed(self.context),self.jobs)
            loader.assert_not_called()

    def test_four_jobs_cover_175_393_406_406_windows_once_with_55200_lite_invocations(self):
        self.assertEqual([job['job_id'] for job in self.jobs],[f'outer_eval_{i}' for i in range(4)])
        self.assertEqual([job['heldout_windows'] for job in self.jobs],[175,393,406,406])
        heldout=[i for job in self.jobs for i in job['heldout_global_window_ids']]
        self.assertEqual(len(heldout),len(set(heldout)))
        self.assertEqual(sorted(heldout),list(range(1380)))
        self.assertEqual(sum(job['expected_counts']['DNN'] for job in self.jobs),80)
        self.assertEqual(sum(job['expected_counts']['RF'] for job in self.jobs),40)
        self.assertEqual(sum(job['expected_counts']['SVM'] for job in self.jobs),4)
        self.assertEqual(sum(job['expected_counts']['detector'] for job in self.jobs),7)
        counts=[job['expected_inference_counts'] for job in self.jobs]
        self.assertEqual(sum(r['DNN_FP32_invocations'] for r in counts),27600)
        self.assertEqual(sum(r['DNN_INT8_invocations'] for r in counts),27600)
        self.assertEqual(sum(r['RF_predicted_rows'] for r in counts),13800)
        self.assertEqual(sum(r['SVM_predicted_rows'] for r in counts),1380)
        self.assertEqual(sum(r['detector_scored_rows'] for r in counts),2354)

    def test_whole_MAT_and_load_roles_keep_heldout_disjoint_from_training_and_calibration(self):
        by_id={job['job_id']:job for job in self.context[5]['jobs']}
        rows=self.context[1]
        for job in self.jobs:
            outer=job['outer_load_hp'];ids=set(job['heldout_global_window_ids'])
            self.assertEqual(len(job['heldout_MAT']),7)
            self.assertTrue(all(int(rows[i]['load_hp'])==outer for i in ids))
            self.assertEqual(ids,{int(r['global_window_id']) for r in rows if int(r['load_hp'])==outer})
            for model in job['models']:
                training=by_id[model['job_id']]
                self.assertTrue(ids.isdisjoint(training['training_global_window_ids']))
                self.assertNotIn(outer,training['roles']['training_loads'])
                if model['family']=='detector':
                    self.assertTrue(ids.isdisjoint(training['fit_ids']+training['calibration_ids']))

    def test_procedure_aliases_preserve_frozen_masks_and_canonical_seed_42(self):
        perf=['F14','FE7','T6','NF12'];cost=['T6','FE7','T6','F14']
        for outer,job in enumerate(self.jobs):
            mapping=job['procedure_mapping']
            self.assertEqual(mapping['baseline_F14']['candidate'],'F14')
            self.assertEqual(mapping['performance_only']['candidate'],perf[outer])
            self.assertEqual(mapping['validated_total_cost']['candidate'],cost[outer])
            for procedure in mapping.values():
                self.assertEqual(list(procedure['DNN_by_seed']),['11','22','33','42','44','55','66','77','88','99'])
                self.assertEqual(procedure['canonical_hardware_job_id'],procedure['DNN_by_seed']['42'])
            self.assertEqual(len(job['models']),33 if outer<3 else 32)
        outer3=self.jobs[3]['procedure_mapping']
        self.assertEqual(outer3['baseline_F14']['detector_job_id'],outer3['performance_only']['detector_job_id'])
        self.assertEqual(outer3['baseline_F14']['DNN_by_seed'],outer3['validated_total_cost']['DNN_by_seed'])

    def test_missing_foreign_or_reordered_folds_and_changed_heldout_ids_are_rejected(self):
        for mutation in ('missing_fold','reordered_folds','changed_id','changed_MAT','missing_model'):
            changed=copy.deepcopy(self.context)
            if mutation=='missing_fold':changed[3]['outer_scenarios'].pop()
            elif mutation=='reordered_folds':changed[3]['outer_scenarios'].reverse()
            elif mutation=='changed_id':changed[3]['outer_scenarios'][0]['heldout_global_window_ids'][0]=1379
            elif mutation=='changed_MAT':changed[3]['outer_scenarios'][0]['heldout_MAT'][0]='not-a-MAT'
            else:changed[4].pop()
            with self.subTest(mutation=mutation),self.assertRaises((ValueError,KeyError)):
                build_changed(changed)

    def test_seed_grid_and_model_alias_tampering_cannot_silently_change_inference_counts(self):
        for mutation in ('missing_seed','reordered_seed','wrong_model','canonical_seed','wrong_detector','RF_seed','SVM_alias'):
            changed=copy.deepcopy(self.context);fold=changed[3]['outer_scenarios'][0]
            procedure=fold['DNN_procedures']['baseline_F14']
            if mutation=='missing_seed':procedure['DNN_by_seed'].pop('99')
            elif mutation=='reordered_seed':procedure['DNN_by_seed']=dict(reversed(list(procedure['DNN_by_seed'].items())))
            elif mutation=='wrong_model':procedure['DNN_by_seed']['11']=procedure['DNN_by_seed']['22']
            elif mutation=='canonical_seed':procedure['canonical_hardware_job_id']=procedure['DNN_by_seed']['11']
            elif mutation=='wrong_detector':procedure['detector_job_id']=procedure['DNN_by_seed']['11']
            elif mutation=='RF_seed':fold['RF_by_seed']['11']=fold['RF_by_seed']['22']
            else:fold['SVM_job_id']=fold['RF_by_seed']['11']
            with self.subTest(mutation=mutation),self.assertRaises((ValueError,KeyError)):
                build_changed(changed)

    def test_changed_artifact_hash_is_rejected_without_loading_the_estimator(self):
        changed=copy.deepcopy(self.context)
        model=changed[4][0];key=next(iter(model['artifact_sha256']))
        model['artifact_sha256'][key]='0'*64
        with self.assertRaises(ValueError):build_changed(changed)

    def test_outer_role_seed_and_training_overlap_are_rejected_before_evaluation(self):
        for mutation in ('bool_outer','bool_model_outer','bool_training_role','bool_seed','training_overlap','detector_calibration_overlap','duplicate_refit_job','empty_artifacts'):
            changed=copy.deepcopy(self.context)
            if mutation=='bool_outer':changed[3]['outer_scenarios'][0]['outer_load_hp']=False
            elif mutation=='bool_model_outer':
                model=next(r for r in changed[4] if r['outer_load_hp']==0);model['outer_load_hp']=False
            elif mutation=='bool_training_role':
                model=next(r for r in changed[4] if r['outer_load_hp']==0)
                model['roles']['training_loads'][0]=True
            elif mutation=='bool_seed':
                model=next(r for r in changed[4] if r['family']=='DNN');model['seed']=True
            elif mutation=='training_overlap':changed[5]['jobs'][0]['training_global_window_ids'].append(changed[3]['outer_scenarios'][0]['heldout_global_window_ids'][0])
            elif mutation=='detector_calibration_overlap':
                model=next(r for r in changed[5]['jobs'] if r['family']=='detector' and r['outer_load_hp']==0)
                model['calibration_ids'].append(changed[3]['outer_scenarios'][0]['heldout_global_window_ids'][0])
            elif mutation=='duplicate_refit_job':changed[5]['jobs'].append(copy.deepcopy(changed[5]['jobs'][0]))
            else:
                changed[4][0]['paths']={};changed[4][0]['artifact_sha256']={}
            with self.subTest(mutation=mutation),self.assertRaises((ValueError,KeyError)):
                build_changed(changed)

    def test_evaluation_signature_binds_inputs_models_sources_versions_and_selection(self):
        hashes=self.context[2];job=copy.deepcopy(self.jobs[0])
        sources={'synthetic-source.py':'a'*64};versions={'python':'synthetic'}
        signature=c.signature(job,sources,versions,hashes)
        changed=copy.deepcopy(job);changed['heldout_global_window_ids'][0]=1379
        self.assertNotEqual(signature,c.signature(changed,sources,versions,hashes))
        self.assertNotEqual(signature,c.signature(job,{'synthetic-source.py':'b'*64},versions,hashes))
        self.assertNotEqual(signature,c.signature(job,sources,{'python':'changed'},hashes))
        changed_hashes=dict(hashes);changed_hashes[job['authorized_feature_source_paths'][0]]='0'*64
        self.assertNotEqual(signature,c.signature(job,sources,versions,changed_hashes))


class ActiveRequestAuthorization(unittest.TestCase):
    def test_matching_request_is_authorized_without_reading_values(self):
        with authorized_fixture() as (campaign,plan),patch.object(np,'load',side_effect=AssertionError('No arrays')) as loader:
            returned,job=c.authorize(campaign,0)
            self.assertIs(returned,plan);self.assertEqual(job['job_id'],'outer_eval_0');loader.assert_not_called()

    def test_missing_lock_request_or_changed_signature_block_before_any_worker(self):
        for mutation in ('lock','request','signature','request_outer','environment','marker','contract'):
            with self.subTest(mutation=mutation),authorized_fixture() as (campaign,plan):
                if mutation=='lock':(campaign/'ACTIVO.lock').unlink()
                elif mutation=='request':(campaign/'SOLICITUD_ACTIVA.json').unlink()
                elif mutation in ('signature','request_outer'):
                    path=campaign/'SOLICITUD_ACTIVA.json';value=c.io.read(path)
                    value['evaluation_signature' if mutation=='signature' else 'outer_load_hp']='changed'
                    path.write_text(json.dumps(value),encoding='utf-8')
                elif mutation=='environment':c.os.environ['SYNTHETIC_OUTER_EVAL_TEST']='changed'
                elif mutation=='marker':(c.OUTPUT/'CAMPANA_UNICA.json').write_text('{}',encoding='utf-8')
                else:(campaign/'CONTRATO.json').write_text('{}',encoding='utf-8')
                with self.assertRaises((ValueError,FileNotFoundError)):
                    c.authorize(campaign,0)

    def test_existing_output_or_noninteger_outer_cannot_be_reexecuted(self):
        with authorized_fixture() as (campaign,_):
            (campaign/'outer_eval_0').mkdir()
            with self.assertRaises(ValueError):c.authorize(campaign,0)
        for outer in (True,False,-1,4,'0',None):
            with self.subTest(outer=outer),authorized_fixture() as (campaign,_),self.assertRaises(ValueError):
                c.authorize(campaign,outer)


class FrozenPreflightGuards(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.context=actual_context();cls.jobs=_JOBS

    @contextmanager
    def prepared(self):
        sourcehashes={'synthetic.py':'a'*64};versions={'python':'synthetic'}
        plan=dict(jobs=copy.deepcopy(self.jobs),source_sha256=sourcehashes,limits=copy.deepcopy(c.LIMITS),
            worker_environment=copy.deepcopy(c.io.ENVIRONMENT),expected_versions=versions,
            selection_signature=c.refit.SELECTION_SIGNATURE,final_refit_manifest_sha256=c.CLOSED_SHA256,
            outer_evaluation_enabled=True,training_enabled=False,calibration_enabled=False,partial_metric_release_enabled=False)
        plan['evaluation_signatures']={job['job_id']:c.signature(job,sourcehashes,versions,self.context[2]) for job in plan['jobs']}
        with patch.object(c.io,'verify_manifest',return_value={}),patch.object(c.io,'read',return_value=plan),\
             patch.object(c,'closed_context',return_value=self.context),patch.object(c,'build_jobs',return_value=self.jobs),\
             patch.object(c,'sources',return_value=sourcehashes),patch.object(c.refit,'versions',return_value=versions):
            yield plan

    def test_preflight_never_trains_calibrates_or_releases_partial_metrics(self):
        with self.prepared() as plan,patch.object(np,'load',side_effect=AssertionError('No arrays')) as loader:
            self.assertIs(c.preflight(environment=True),plan)
            loader.assert_not_called()
        for field,value in [('training_enabled',True),('calibration_enabled',True),
            ('partial_metric_release_enabled',True),('outer_evaluation_enabled',False)]:
            with self.subTest(field=field),self.prepared() as plan:
                plan[field]=value
                with self.assertRaises(ValueError):c.preflight()

    def test_failed_incomplete_or_other_selection_closure_cannot_enable_evaluation(self):
        closure=c.io.read(c.CLOSED/'CIERRE_REAJUSTES_FINALES.json')
        protocol,rows,hashes,mapping,inventory,refit_plan=self.context
        for mutation in ('status','completed_jobs','failures','remaining_jobs','selection_signature','closure_sha'):
            changed=copy.deepcopy(closure)
            if mutation=='status':changed['status']='INCOMPLETE'
            elif mutation=='completed_jobs':changed['completed_jobs']=130
            elif mutation=='failures':changed['failures']=1
            elif mutation=='remaining_jobs':changed['remaining_jobs']=1
            elif mutation=='selection_signature':changed['selection_signature']='0'*64
            values={str(c.CLOSED/'CIERRE_REAJUSTES_FINALES.json'):changed,
                str(c.CLOSED/'MAPA_PARA_EVALUACION_EXTERIOR.json'):mapping,
                str(c.CLOSED/'INVENTARIO_MODELOS_FINALES_131.json'):inventory,str(c.refit.PLAN):refit_plan}
            with self.subTest(mutation=mutation),patch.object(c.io,'sha',return_value='wrong' if mutation=='closure_sha' else c.CLOSED_SHA256),\
                 patch.object(c.io,'verify_manifest',return_value={}),\
                 patch.object(c.io,'read',side_effect=lambda path:values[str(path)]),\
                 patch.object(c.refit,'frozen_contract',return_value=(None,protocol,rows,hashes)),self.assertRaises(ValueError):
                c.closed_context()

    def test_preflight_rejects_source_limit_signature_scope_and_environment_changes(self):
        for mutation in ('source','limit','signature','scope','worker_environment','versions','selection','refit_closure'):
            with self.subTest(mutation=mutation),self.prepared() as plan:
                if mutation=='source':plan['source_sha256']={'synthetic.py':'changed'}
                elif mutation=='limit':plan['limits']['worker_seconds']+=1
                elif mutation=='signature':plan['evaluation_signatures']['outer_eval_0']='changed'
                elif mutation=='scope':plan['jobs'][0]['heldout_global_window_ids'][0]=1379
                elif mutation=='worker_environment':plan['worker_environment']['OMP_NUM_THREADS']='2'
                elif mutation=='selection':plan['selection_signature']='0'*64
                elif mutation=='refit_closure':plan['final_refit_manifest_sha256']='0'*64
                else:
                    with patch.object(c.refit,'versions',return_value={'python':'different'}),self.assertRaises(ValueError):
                        c.preflight(environment=True)
                    continue
                with self.assertRaises(ValueError):c.preflight()


class ControlledQueueBudget(unittest.TestCase):
    def test_another_worker_requires_its_full_watchdog_remaining(self):
        import ejecutar_evaluacion as execution
        self.assertTrue(execution.can_start(300,1500,1800))
        self.assertFalse(execution.can_start(300,1500.001,1800))
        self.assertFalse(execution.can_start(300,1800,1800))

    def test_inspection_reuses_only_a_complete_prefix_and_carries_time_without_prediction(self):
        import ejecutar_evaluacion as execution
        with authorized_fixture() as (campaign,plan),patch.object(c.io,'verify_manifest',return_value={}),\
             patch.object(execution,'verify_fold',return_value={'status':'PASS'}) as verify:
            steps=campaign/'pasos';steps.mkdir()
            for index,seconds in enumerate((4.,7.5)):
                c.io.write(steps/f'{index:02d}.json',dict(status='PASS',job_id=plan['jobs'][index]['job_id'],
                    timeout_seconds=300,automatic_retry=False,execution_verification_seconds=seconds))
            done,consumed=execution.inspect_campaign(campaign,plan)
            self.assertEqual(done,['outer_eval_0','outer_eval_1']);self.assertEqual(consumed,11.5)
            self.assertEqual(verify.call_count,2)

    def test_gap_failed_nonfinite_partial_or_foreign_steps_cannot_be_resumed(self):
        import ejecutar_evaluacion as execution
        for mutation in ('gap','failed','nonfinite','negative','partial','foreign'):
            with self.subTest(mutation=mutation),authorized_fixture() as (campaign,plan),\
                 patch.object(c.io,'verify_manifest',return_value={}),\
                 patch.object(execution,'verify_fold',return_value={'status':'PASS'}):
                steps=campaign/'pasos';steps.mkdir()
                state=dict(status='PASS',job_id='outer_eval_0',timeout_seconds=300,automatic_retry=False,
                    execution_verification_seconds=1.)
                if mutation=='gap':
                    state['job_id']='outer_eval_1';path=steps/'01.json'
                elif mutation=='foreign':path=steps/'99.json'
                else:path=steps/'00.json'
                if mutation=='failed':state['status']='FAIL'
                elif mutation=='nonfinite':state['execution_verification_seconds']=float('nan')
                elif mutation=='negative':state['execution_verification_seconds']=-1.
                if mutation=='partial':(campaign/'outer_eval_0').mkdir()
                else:path.write_text(json.dumps(state),encoding='utf-8')
                with self.assertRaises(ValueError):execution.inspect_campaign(campaign,plan)

    def test_release_requires_all_four_verified_folds_and_cannot_overwrite_a_result(self):
        import ejecutar_evaluacion as execution
        with authorized_fixture() as (campaign,plan),\
             patch.object(execution,'inspect_campaign',return_value=(['outer_eval_0','outer_eval_1','outer_eval_2'],3.)),\
             patch.object(execution,'aggregate_results') as aggregate:
            with self.assertRaises(ValueError):execution.release_results(campaign,plan)
            aggregate.assert_not_called()
            self.assertFalse((campaign/'RESULTADOS_EXTERIORES.json').exists())
        with authorized_fixture() as (campaign,plan),\
             patch.object(execution,'inspect_campaign',return_value=([j['job_id'] for j in plan['jobs']],4.)),\
             patch.object(execution,'aggregate_results',return_value={'synthetic_only':True}) as aggregate:
            self.assertEqual(execution.release_results(campaign,plan),{'synthetic_only':True})
            before=(campaign/'RESULTADOS_EXTERIORES.json').read_bytes()
            with self.assertRaises((ValueError,FileExistsError)):execution.release_results(campaign,plan)
            self.assertEqual((campaign/'RESULTADOS_EXTERIORES.json').read_bytes(),before)

    def test_timeout_kills_mock_process_preserves_logs_and_does_not_verify_or_retry(self):
        import ejecutar_evaluacion as execution
        with authorized_fixture() as (campaign,plan):
            (campaign/'SOLICITUD_ACTIVA.json').unlink()
            process=Mock();process.wait.side_effect=[subprocess.TimeoutExpired('synthetic',300),-9]
            process.poll.return_value=None;process.returncode=-9
            with patch.object(execution.subprocess,'Popen',return_value=process) as launch,\
                 patch.object(execution,'verify_fold') as verify:
                state=execution.launch_one(campaign,plan['jobs'][0],plan)
            self.assertEqual(state['status'],'FAIL');self.assertEqual(state['error'],'TIMEOUT')
            self.assertFalse(state['automatic_retry']);self.assertEqual(launch.call_count,1)
            process.kill.assert_called_once();verify.assert_not_called()
            self.assertTrue((campaign/'outer_eval_0_stdout.log').exists())
            self.assertTrue((campaign/'outer_eval_0_stderr.log').exists())
            self.assertFalse((campaign/'SOLICITUD_ACTIVA.json').exists())

    def test_launch_error_nonzero_or_old_logs_cannot_trigger_predictions_or_overwrites(self):
        import ejecutar_evaluacion as execution
        for mutation in ('launch_error','nonzero','old_log'):
            with self.subTest(mutation=mutation),authorized_fixture() as (campaign,plan):
                (campaign/'SOLICITUD_ACTIVA.json').unlink()
                process=Mock();process.wait.return_value=2;process.poll.return_value=2;process.returncode=2
                if mutation=='old_log':(campaign/'outer_eval_0_stdout.log').write_bytes(b'preserve-synthetic\n')
                options={'side_effect':OSError('synthetic launch failure')} if mutation=='launch_error' else {'return_value':process}
                with patch.object(execution.subprocess,'Popen',**options) as launch,\
                     patch.object(execution,'verify_fold') as verify:
                    state=execution.launch_one(campaign,plan['jobs'][0],plan)
                self.assertEqual(state['status'],'FAIL');self.assertFalse(state['automatic_retry'])
                verify.assert_not_called()
                if mutation=='old_log':
                    launch.assert_not_called()
                    self.assertEqual((campaign/'outer_eval_0_stdout.log').read_bytes(),b'preserve-synthetic\n')
                else:self.assertEqual(launch.call_count,1)

    @contextmanager
    def queue(self):
        import ejecutar_evaluacion as execution
        with fixture() as folder:
            prep=folder/'preparation';prep.mkdir();output=folder/'campaigns'
            manifest=prep/'MANIFEST.json';manifest.write_text('{}\n',encoding='utf-8')
            plan_path=prep/'PLAN.json';plan_path.write_text('{}\n',encoding='utf-8')
            jobs=[dict(job_id=f'outer_eval_{outer}',outer_load_hp=outer,timeout_seconds=300) for outer in range(4)]
            plan=dict(jobs=jobs,evaluation_signatures={j['job_id']:'s'+str(i) for i,j in enumerate(jobs)},
                worker_environment={'SYNTHETIC_ONLY':'1'},limits=copy.deepcopy(c.LIMITS))
            with patch.object(c,'OUTPUT',output),patch.object(c,'PLAN',plan_path),patch.object(c,'MANIFEST',manifest),\
                 patch.object(c,'preflight',return_value=plan),patch.object(c.io,'write_manifest',return_value=None):
                yield execution,output,plan

    def test_queue_releases_once_after_four_successes_and_refuses_a_second_campaign(self):
        events=[]
        with self.queue() as (execution,output,plan):
            def launch(campaign,job,current):
                events.append(job['job_id'])
                return dict(status='PASS',job_id=job['job_id'],timeout_seconds=300,automatic_retry=False,
                    execution_verification_seconds=2.)
            def release(campaign,current):
                self.assertEqual(events,[j['job_id'] for j in plan['jobs']])
                self.assertEqual(len(list((campaign/'pasos').glob('[0-9][0-9].json'))),4)
                return {'synthetic_only':True}
            with patch.object(execution,'launch_one',side_effect=launch) as runner,\
                 patch.object(execution,'release_results',side_effect=release) as releaser:
                self.assertEqual(execution.execute(),0)
                self.assertEqual(runner.call_count,4);releaser.assert_called_once()
                campaign=next(p for p in output.iterdir() if p.is_dir())
                summary=c.io.read(campaign/'RESUMEN_CAMPANA.json')
                self.assertTrue(summary['results_released']);self.assertEqual(summary['completed_loads'],4)
                self.assertEqual(summary['campaign_execution_verification_seconds'],8.)
                with self.assertRaises(ValueError):execution.execute()
                self.assertEqual(runner.call_count,4)

    def test_first_failed_fold_stops_queue_and_never_releases_partial_metrics(self):
        with self.queue() as (execution,output,plan):
            states=[dict(status=status,job_id=plan['jobs'][index]['job_id'],timeout_seconds=300,automatic_retry=False,
                execution_verification_seconds=2.) for index,status in enumerate(('PASS','FAIL'))]
            with patch.object(execution,'launch_one',side_effect=states) as runner,\
                 patch.object(execution,'release_results') as release,patch.object(execution,'aggregate_results') as aggregate:
                self.assertEqual(execution.execute(),1)
                self.assertEqual(runner.call_count,2);release.assert_not_called();aggregate.assert_not_called()
                campaign=next(p for p in output.iterdir() if p.is_dir())
                summary=c.io.read(campaign/'RESUMEN_CAMPANA.json')
                self.assertEqual(summary['completed_loads'],1);self.assertFalse(summary['results_released'])
                self.assertEqual(summary['status'],'STOPPED_FOR_AUDIT_NO_FINAL_RESULTS')
                self.assertFalse((campaign/'RESULTADOS_EXTERIORES.json').exists())
                before=(campaign/'RESUMEN_CAMPANA.json').read_bytes()
                with self.assertRaises(ValueError):execution.execute()
                self.assertEqual((campaign/'RESUMEN_CAMPANA.json').read_bytes(),before)
                self.assertEqual(runner.call_count,2)

    def test_budget_boundary_stops_before_next_launch_without_releasing_partial_metrics(self):
        with self.queue() as (execution,output,plan):
            plan['limits']['campaign_execution_verification_seconds']=300
            state=dict(status='PASS',job_id='outer_eval_0',timeout_seconds=300,automatic_retry=False,
                execution_verification_seconds=1.)
            with patch.object(execution,'launch_one',return_value=state) as runner,\
                 patch.object(execution,'release_results') as release:
                self.assertEqual(execution.execute(),1)
                runner.assert_called_once();release.assert_not_called()
                campaign=next(p for p in output.iterdir() if p.is_dir())
                summary=c.io.read(campaign/'RESUMEN_CAMPANA.json')
                self.assertEqual(summary['completed_loads'],1);self.assertIn('BUDGET_BOUNDARY',summary['error'])
                self.assertFalse(summary['results_released'])


class SavedQuantizationParameterTypes(unittest.TestCase):
    def test_saved_float64_quantization_pairs_are_compared_to_typed_JSON_before_pure_audit(self):
        """NPZ stores zero as float64; quantization_report must receive JSON int."""
        import ejecutar_evaluacion as execution
        import worker_evaluacion as worker
        from metricas_evaluacion import supervised,quantization_report
        from numerica_representantes import quantize_int8,integer_argmax
        for mismatch in (False,True):
            with self.subTest(mismatch=mismatch),fixture() as campaign:
                folder=campaign/'outer_eval_0';folder.mkdir()
                raw=np.zeros((3,14),dtype=np.float32)
                raw[:,:2]=np.asarray([[-.25,.125],[.25,-.125],[.375,.5]],dtype=np.float32)
                z=raw[:,:2];y=np.asarray([0,1,2],dtype=np.int64);ids=np.asarray([0,1,2],dtype=np.int64)
                records=np.asarray(['synthetic-N','synthetic-IR','synthetic-OR'])
                inputq=[.125,-9];outputq=[1/256,-128]
                iq_inputs,_=quantize_int8(z,*inputq)
                outputs=np.full((3,3),-128,dtype=np.int8);np.fill_diagonal(outputs,127)
                iq=np.multiply(np.subtract(outputs.astype(np.float32),np.float32(outputq[1]),dtype=np.float32),
                    np.float32(outputq[0]),dtype=np.float32)
                fp=np.eye(3,dtype=np.float32)
                model=dict(job_id='synthetic_DNN',family='DNN',seed=11,fit_signature='synthetic-fit',indices=[0,1],
                    paths={'scaler_train_only.npz':'synthetic/scaler.npz'})
                job=dict(job_id='outer_eval_0',outer_load_hp=0,heldout_global_window_ids=ids.tolist(),models=[model],
                    expected_inference_counts=dict(DNN_FP32_invocations=3,DNN_INT8_invocations=3))
                plan=dict(selection_signature='synthetic-selection',evaluation_signatures={'outer_eval_0':'synthetic-evaluation'},
                    expected_versions={'python':'synthetic'},source_sha256={'synthetic.py':'synthetic-hash'})
                np.savez(folder/'entrada_evaluacion.npz',raw_features14=raw,y_true=y,global_window_ids=ids,record_ids=records)
                saved_inputq=np.asarray(inputq,dtype=np.float64)
                if mismatch:saved_inputq[0]=.25
                np.savez(folder/'synthetic_DNN_predicciones.npz',global_window_ids=ids,y_true=y,record_ids=records,
                    normalized_features=z,fp32_inputs=z,fp32_outputs=fp,fp32_probabilities=fp,int8_probabilities=iq,
                    int8_inputs=iq_inputs,int8_outputs=outputs,fp32_predictions=integer_argmax(fp),
                    int8_predictions=integer_argmax(iq),input_quantization=saved_inputq,
                    output_quantization=np.asarray(outputq,dtype=np.float64))
                row=dict(model_job_id=model['job_id'],family='DNN',seed=11,fit_signature='synthetic-fit',
                    prediction_file='synthetic_DNN_predicciones.npz',prediction_sha256=c.io.sha(folder/'synthetic_DNN_predicciones.npz'),
                    metrics=dict(FP32=supervised(y,integer_argmax(fp),records.tolist()),
                        INT8=supervised(y,integer_argmax(iq),records.tolist())),
                    quantization_report=quantization_report(z,iq_inputs,outputs,fp,iq,y,records.tolist(),inputq,outputq,ids.tolist()),
                    int8_details=dict(input_quantization=inputq,output_quantization=outputq))
                result=dict(status='PASS',job_id=job['job_id'],outer_load_hp=0,
                    selection_signature=plan['selection_signature'],evaluation_signature='synthetic-evaluation',
                    versions=plan['expected_versions'],source_sha256=plan['source_sha256'],
                    input_snapshot_sha256=c.io.sha(folder/'entrada_evaluacion.npz'),provenance={},
                    model_hashes_before={},model_hashes_after={},source_sha256_before=plan['source_sha256'],
                    source_sha256_after=plan['source_sha256'],model_results=[row],inference_counts=job['expected_inference_counts'])
                result.update({key:False for key in ('training_executed','parameter_fit_executed','threshold_calibrated',
                    'model_conversion_executed','scalers_refitted','INT8_recalibrated','detector_threshold_refitted',
                    'mask_or_seed_selection_executed','seed_ensembles','detector_class_override')})
                c.io.write(folder/'RESULTADO.json',result)
                scaler=dict(mean32=np.zeros(2,dtype=np.float32),scale32=np.ones(2,dtype=np.float32))
                with patch.object(c.io,'verify_manifest',return_value={}),\
                     patch.object(worker,'validate_heldout'),patch.object(worker,'model_snapshot',return_value={}),\
                     patch.object(worker,'load_scaler',return_value=scaler):
                    if mismatch:
                        with self.assertRaises(ValueError):execution.verify_fold(campaign,job,plan)
                    else:self.assertEqual(execution.verify_fold(campaign,job,plan),result)


if __name__=='__main__':unittest.main(verbosity=2)
