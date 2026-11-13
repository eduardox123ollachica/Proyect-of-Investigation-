"""Pruebas del controlador con callbacks sintéticos; sin fit, TF o hardware."""
from __future__ import annotations

import copy
from contextlib import contextmanager
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import continuar_restantes as controller


class SimulatedCampaign:
    def __init__(self,completed=17,consumed=225.843,step_cap=None,return_code=0):
        self.campaign=Path('synthetic_campaign')
        self.jobs=[dict(job_id=f'new_{i:03d}') for i in range(240)]
        self.plan=dict(limits=dict(max_new_jobs=16,max_batch_seconds=1800,
            campaign_worker_seconds=7200,worker_seconds=300,maximum_new_fits_total=240,
            default_first_batch_seconds=600),
            imported={f'imported_{i:03d}':{} for i in range(48)},
            logical_cells=288,imported_cells=48,new_fits=240)
        self.completed=completed
        self.consumed=consumed
        self.step_cap=step_cap
        self.return_code=return_code
        self.calls=[]
        self.started=[]
        self.events=[]

    def inspect(self,campaign):
        if campaign != self.campaign:
            raise AssertionError('Se intentó otra campaña')
        return copy.deepcopy(self.plan),copy.deepcopy(self.jobs),[
            j['job_id'] for j in self.jobs[:self.completed]],self.consumed

    def execute(self,campaign,max_new_jobs,max_seconds):
        if campaign != self.campaign:
            raise AssertionError('Se intentó otra campaña')
        self.calls.append((campaign,max_new_jobs,max_seconds))
        if self.return_code:
            return self.return_code
        count=min(max_new_jobs,len(self.jobs)-self.completed)
        if self.step_cap is not None:
            count=min(count,self.step_cap)
        self.started += [j['job_id'] for j in self.jobs[self.completed:self.completed+count]]
        self.completed += count
        self.consumed += count*10
        return 0

    def run(self):
        return controller.drive(self.campaign,self.inspect,self.execute,self.events.append)


@contextmanager
def snapshot_fixture():
    base=(controller.ROOT/'tmp').resolve();base.mkdir(exist_ok=True)
    folder=Path(tempfile.mkdtemp(prefix='continuation_test_',dir=base)).resolve()
    simulation=SimulatedCampaign(completed=2,consumed=20)
    campaign=folder/'campaign';campaign.mkdir();(campaign/'lotes/batch1').mkdir(parents=True)
    for job in simulation.jobs[:2]:(campaign/job['job_id']).mkdir()
    plan=copy.deepcopy(simulation.plan);plan['jobs']=copy.deepcopy(simulation.jobs)
    values={'PLAN_LOTE.json':dict(completed_before=[],job_ids=[j['job_id'] for j in simulation.jobs[:2]]),
        'RESUMEN_LOTE.json':dict(status='BATCH_COMPLETED',jobs_started=2,jobs_passed=2,
            worker_elapsed_seconds=20,completed_new_fits=2,available_logical_cells=50,
            campaign_worker_seconds_consumed=20)}
    try:
        with patch.object(controller.e,'validate_campaign',return_value=plan), \
             patch.object(controller.e,'verify_manifest') as verify, \
             patch.object(controller.e,'read',side_effect=lambda p:copy.deepcopy(values[Path(p).name])):
            yield campaign,values,verify
    finally:
        controller.e.require(folder.parent == base and folder.name.startswith('continuation_test_')
            and folder.is_relative_to(base),'Fixture fuera del temporal autorizado')
        shutil.rmtree(folder)


class ContinuationGuards(unittest.TestCase):
    def test_all_223_remaining_complete_in_fourteen_bounded_batches(self):
        simulation=SimulatedCampaign()
        result=simulation.run()
        self.assertEqual(result['status'],'CAMPAIGN_COMPLETED')
        self.assertEqual(simulation.completed,240)
        self.assertEqual(len(simulation.calls),14)
        self.assertEqual([c[1] for c in simulation.calls],[16]*13+[15])
        self.assertTrue(all(c[2] == 1800 for c in simulation.calls))

    def test_resume_only_pending_jobs_and_preserve_existing_seventeen(self):
        simulation=SimulatedCampaign()
        simulation.run()
        self.assertEqual(simulation.started,[j['job_id'] for j in simulation.jobs[17:]])
        self.assertEqual(len(set(simulation.started)),223)
        self.assertFalse(set(simulation.started)&{j['job_id'] for j in simulation.jobs[:17]})

    def test_failing_batch_stops_without_automatic_retry(self):
        simulation=SimulatedCampaign(return_code=1)
        result=simulation.run()
        self.assertNotEqual(result['status'],'CAMPAIGN_COMPLETED')
        self.assertEqual(len(simulation.calls),1)
        self.assertEqual(simulation.completed,17)
        self.assertEqual(simulation.started,[])

    def test_zero_progress_stops_instead_of_looping(self):
        simulation=SimulatedCampaign(step_cap=0)
        result=simulation.run()
        self.assertNotEqual(result['status'],'CAMPAIGN_COMPLETED')
        self.assertEqual(len(simulation.calls),1)
        self.assertEqual(simulation.completed,17)

    def test_insufficient_cumulative_budget_never_starts_batch(self):
        simulation=SimulatedCampaign(consumed=7000)
        result=simulation.run()
        self.assertNotEqual(result['status'],'CAMPAIGN_COMPLETED')
        self.assertEqual(simulation.calls,[])
        self.assertEqual(simulation.completed,17)
        self.assertEqual(simulation.consumed,7000)

    def test_partial_clean_batch_can_continue_without_resetting_budget(self):
        simulation=SimulatedCampaign(completed=235,consumed=500,step_cap=2)
        result=simulation.run()
        self.assertEqual(result['status'],'CAMPAIGN_COMPLETED')
        self.assertEqual(simulation.completed,240)
        self.assertEqual([c[1] for c in simulation.calls],[5,3,1])
        self.assertEqual(simulation.consumed,550)

    def test_cumulative_budget_is_not_reset_between_partial_batches(self):
        simulation=SimulatedCampaign(consumed=6870,step_cap=1)
        result=simulation.run()
        self.assertEqual(result['status'],'CAMPAIGN_BUDGET_BOUNDARY')
        self.assertEqual(len(simulation.calls),4)
        self.assertEqual(simulation.completed,21)
        self.assertEqual(simulation.consumed,6910)

    def test_completed_jobs_must_be_canonical_prefix(self):
        simulation=SimulatedCampaign()
        def changed_inspect(campaign):
            plan,jobs,done,consumed=simulation.inspect(campaign)
            done[0],done[1]=done[1],done[0]
            return plan,jobs,done,consumed
        with self.assertRaises(ValueError):
            controller.drive(simulation.campaign,changed_inspect,simulation.execute,simulation.events.append)
        self.assertEqual(simulation.calls,[])

    def test_already_completed_campaign_does_not_retrain(self):
        simulation=SimulatedCampaign(completed=240)
        result=simulation.run()
        self.assertEqual(result['status'],'CAMPAIGN_COMPLETED')
        self.assertEqual(simulation.calls,[])

    def test_callback_imports_do_not_load_tensorflow(self):
        SimulatedCampaign(completed=240).run()
        self.assertNotIn('tensorflow',sys.modules)
        self.assertNotIn('keras',sys.modules)

    def test_closed_snapshot_verifies_seal_sequence_counts_and_budget(self):
        with snapshot_fixture() as (campaign,values,verify):
            _,_,done,consumed=controller.closed_snapshot(campaign)
            self.assertEqual(done,['new_000','new_001'])
            self.assertEqual(consumed,20)
            verify.assert_called_once_with(campaign/'lotes/batch1/MANIFEST_LOTE_SHA256.json')

    def test_closed_snapshot_rejects_tampered_budget_and_job_sequence(self):
        with snapshot_fixture() as (campaign,values,_):
            values['RESUMEN_LOTE.json']['campaign_worker_seconds_consumed']=21
            with self.assertRaises(ValueError):controller.closed_snapshot(campaign)
        with snapshot_fixture() as (campaign,values,_):
            values['PLAN_LOTE.json']['job_ids'].reverse()
            with self.assertRaises(ValueError):controller.closed_snapshot(campaign)

    def test_closed_snapshot_rejects_unclosed_job_or_failed_batch(self):
        with snapshot_fixture() as (campaign,values,_):
            (campaign/'new_002').mkdir()
            with self.assertRaises(ValueError):controller.closed_snapshot(campaign)
        with snapshot_fixture() as (campaign,values,_):
            values['RESUMEN_LOTE.json']['status']='STOPPED_FOR_AUDIT'
            with self.assertRaises(ValueError):controller.closed_snapshot(campaign)


if __name__ == '__main__':
    unittest.main(verbosity=2)
