"""Roles y guardias del bloque secundario; sin datos fuente/ajustes reales."""
from copy import deepcopy
import unittest

import detector_suplemento as detector
from contrato_datos import context,jobs


class DetectorPlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol,cls.rows,_=context()
        cls.data_jobs={job['job_id']:job for job in jobs()}

    def test_exactly_six_unique_jobs_no_DNN_seed_repetition(self):
        plan=detector.detector_jobs()
        self.assertEqual(len(plan),6)
        self.assertEqual(len({j['detector_id'] for j in plan}),6)
        self.assertEqual([(j['inner'],j['representation']) for j in plan],
                         [(i,n) for i in range(3) for n in ('CONTROL12','MOM4')])
        self.assertTrue(all(not j['repeated_per_DNN_seed'] for j in plan))

    def test_only_one_reference_import_no_fit_fallback(self):
        imported=[j for j in detector.detector_jobs() if j['import_original_reference']]
        self.assertEqual(len(imported),1)
        self.assertEqual((imported[0]['inner'],imported[0]['representation']),(0,'CONTROL12'))

    def test_indices_exact_and_shared_with_F14_NF12(self):
        for job in detector.detector_jobs():
            expected=detector.CONTROL12 if job['representation']=='CONTROL12' else detector.MOM4
            self.assertEqual(job['indices'],expected)
            self.assertTrue(set(job['indices']) <= set(self.protocol['candidates']['F14']))
            self.assertTrue(set(job['indices']) <= set(self.protocol['candidates']['NF12']))
            self.assertFalse(set(job['indices']) & {4,11})
        modified=detector.detector_jobs();modified[0]['indices'].append(4)
        self.assertEqual(detector.detector_jobs()[0]['indices'],detector.CONTROL12)

    def test_fixed_data_jobs_and_raw_space(self):
        for job in detector.detector_jobs():
            data=self.data_jobs[job['data_job_id']]
            self.assertEqual((data['mode'],data['candidate'],data['seed']),('P0','F14',11))
            self.assertEqual(data['inner_validation_load_hp'],job['inner'])
            self.assertEqual(job['data_space'],'raw_features_float32')

    def test_fit_calibration_validation_records_disjoint_and_no_outer(self):
        for job in detector.detector_jobs():
            data=self.data_jobs[job['data_job_id']]
            scopes=[data[key] for key in ('detector_fit_global_window_ids',
                                         'detector_calibration_global_window_ids','validation_global_window_ids')]
            records=[{self.rows[k]['file_id'] for k in ids} for ids in scopes]
            self.assertTrue(records[0].isdisjoint(records[1]))
            self.assertTrue(records[0].isdisjoint(records[2]))
            self.assertTrue(records[1].isdisjoint(records[2]))
            self.assertTrue(all(int(self.rows[k]['load_hp']) != 3 for ids in scopes for k in ids))
            self.assertTrue(all(self.rows[k]['class']=='N' for ids in scopes[:2] for k in ids))

    def test_normal_roles_by_inner(self):
        expected={0:([1],[2]),1:([0],[2]),2:([0],[1])}
        for job in detector.detector_jobs():
            data=self.data_jobs[job['data_job_id']]
            self.assertEqual((data['roles']['detector_fit_normal_loads'],
                              data['roles']['detector_calibration_normal_loads']),expected[job['inner']])

    def test_descriptive_guard_boundary_and_each_component(self):
        metrics=dict(normal_spec=.95,ir_recall=.95,or_recall=.95)
        result=detector.descriptive_guards(metrics)
        self.assertTrue(result['passed'])
        self.assertFalse(result['automatic_detector_selection'])
        self.assertFalse(result['statistical_guarantee'])
        for key in metrics:
            bad=dict(metrics);bad[key]=.9499999
            self.assertFalse(detector.descriptive_guards(bad)['passed'])

    def test_guard_rejects_nonfinite_out_of_range_and_boolean(self):
        for value in (float('nan'),float('inf'),-1,1.1,True):
            with self.subTest(value=value),self.assertRaises(ValueError):
                detector.descriptive_guards(dict(normal_spec=value,ir_recall=1.,or_recall=1.))


if __name__=='__main__':unittest.main()
