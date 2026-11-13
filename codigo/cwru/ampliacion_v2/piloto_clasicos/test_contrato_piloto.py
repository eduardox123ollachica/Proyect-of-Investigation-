"""Pruebas puras de cuadrículas y guardias; no accede a CWRU o sklearn."""
import copy
import unittest
from contrato_piloto import build_jobs,validate_result


def protocol():
    return dict(candidate_order=['F14','T6','S8','NF12','NM10','NE12','DE7','FE7'],
        candidates=dict(F14=list(range(14)),T6=[0,1,2,7,8,9],S8=[3,4,5,6,10,11,12,13],
            NF12=[0,1,2,3,5,6,7,8,9,10,12,13],NM10=[0,3,4,5,6,7,10,11,12,13],
            NE12=[0,1,2,3,4,5,7,8,9,10,11,12],DE7=list(range(7)),FE7=list(range(7,14))),
        classical_comparators=dict(RF=dict(n_estimators=[100,300],max_depth=[None,12],
            min_samples_leaf=[1,3],n_jobs=1,max_features='sqrt',criterion='gini',bootstrap=True,
            class_weight='balanced'),SVM=dict(C=[.1,1,10,100],gamma=['scale',.01],probability=False)))


class ContractTests(unittest.TestCase):
    def test_grid_and_process_limits(self):
        jobs=build_jobs(protocol())
        self.assertEqual(len(jobs),17)
        self.assertEqual(sum(j['timeout_seconds'] for j in jobs),780)
        self.assertEqual(len({j['job_id'] for j in jobs}),17)
        self.assertEqual([j['family'] for j in jobs].count('RF'),8)
        self.assertEqual([j['family'] for j in jobs].count('SVM'),8)

    def test_rf_configurations_cover_full_grid(self):
        rf=[j['params'] for j in build_jobs(protocol()) if j['family']=='RF']
        self.assertEqual(len({(r['n_estimators'],r['max_depth'],r['min_samples_leaf']) for r in rf}),8)
        self.assertTrue(all(r['random_state']==11 and r['n_jobs']==1 for r in rf))

    def test_svm_is_not_repeated_by_seeds(self):
        rows=[j['params'] for j in build_jobs(protocol()) if j['family']=='SVM']
        self.assertEqual(len({(r['C'],r['gamma']) for r in rows}),8)
        self.assertTrue(all(r['probability'] is False and 'random_state' not in r for r in rows))

    def test_detector_alias_and_excluded_frequencies(self):
        detector=build_jobs(protocol())[-1]
        self.assertEqual(detector['aliases'],{'NF12':'F14'})
        self.assertEqual(detector['maximum_unique_fit_calls'],21)
        self.assertEqual(detector['indices_by_candidate']['NF12'],detector['indices_by_candidate']['F14'])
        self.assertTrue(all(not set(v)&{4,11} for v in detector['indices_by_candidate'].values()))

    def test_changed_grid_or_catalogue_is_rejected(self):
        for field in ('RF','SVM','order'):
            p=protocol()
            if field=='RF':p['classical_comparators']['RF']['n_estimators']=[10]
            elif field=='SVM':p['classical_comparators']['SVM']['C']=[1]
            else:p['candidate_order'].reverse()
            with self.assertRaises(ValueError):build_jobs(p)

    def test_invalid_result_scope_is_rejected(self):
        job=build_jobs(protocol())[0]
        baseline=dict(status='PASS',job_id=job['job_id'],family='RF',training_windows=799,validation_windows=175,
            no_load3_arrays_opened=True,metrics_used_for_selection=False,confirmatory_fit=False,
            config_index=job['config_index'],requested_params=job['params'])
        self.assertIs(validate_result(baseline,job),baseline)
        for key,value in [('status','FAIL'),('training_windows',800),('no_load3_arrays_opened',False),
                          ('confirmatory_fit',True),('config_index',99)]:
            row=copy.deepcopy(baseline);row[key]=value
            with self.assertRaises(ValueError):validate_result(row,job)


if __name__=='__main__':unittest.main(verbosity=2)
