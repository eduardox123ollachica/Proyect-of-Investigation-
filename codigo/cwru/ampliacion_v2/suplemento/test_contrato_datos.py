"""Pruebas de scope y lectura del suplemento, sin ajustes o TensorFlow."""
from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

import contrato_datos as data


class TrackingArray:
    def __init__(self, size, bad=()):
        self.dtype=np.dtype(np.float64)
        self.shape=(size,14)
        self.read=[]
        self.bad=set(bad)

    def __getitem__(self,index):
        if not isinstance(index,int):raise AssertionError('Solo materializar filas autorizadas')
        self.read.append(index)
        row=np.full(14,index+.25,dtype=np.float64)
        if index in self.bad:row[0]=np.nan
        return row


class SupplementDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol,cls.rows,cls.hashes=data.context()
        with patch.object(data,'context',return_value=(cls.protocol,cls.rows,cls.hashes)):
            cls.jobs=data.jobs()

    def test_36_unique_jobs_and_two_initial_gates(self):
        self.assertEqual(len(self.jobs),36)
        self.assertEqual(len({j['job_id'] for j in self.jobs}),36)
        self.assertEqual([j['job_id'] for j in self.jobs[:2]],
            ['supp_o3_i0_F14_P0_s11','supp_o3_i0_F14_P1_s11'])

    def test_all_factorial_cells_present(self):
        observed={(j['inner_validation_load_hp'],j['candidate'],j['mode'],j['seed']) for j in self.jobs}
        self.assertEqual(observed,{(i,m,p,s) for i in (0,1,2) for m in ('F14','NF12')
                                  for p in ('P0','P1') for s in (11,22,33)})

    def test_known_independent_counts_for_each_inner(self):
        counts={j['inner_validation_load_hp']:(len(j['training_global_window_ids']),
            len(j['validation_global_window_ids'])) for j in self.jobs}
        self.assertEqual(counts,{0:(799,175),1:(581,393),2:(568,406)})

    def test_all_scopes_mat_disjoint_and_outer3_excluded(self):
        for job in self.jobs:
            train,val=data.validate_job(job,self.rows,self.protocol)
            self.assertEqual({int(r['load_hp']) for r in train},set((0,1,2))-{job['inner_validation_load_hp']})
            self.assertEqual({int(r['load_hp']) for r in val},{job['inner_validation_load_hp']})
            self.assertFalse({r['file_id'] for r in train}&{r['file_id'] for r in val})
            self.assertTrue(all(r['source_features'] in data.ALLOWED_FEATURES for r in train+val))

    def test_modes_masks_seeds_do_not_change_rows(self):
        for inner in (0,1,2):
            group=[j for j in self.jobs if j['inner_validation_load_hp']==inner]
            for key in ('training_global_window_ids','validation_global_window_ids',
                        'detector_fit_global_window_ids','detector_calibration_global_window_ids'):
                self.assertTrue(all(j[key]==group[0][key] for j in group))

    def test_detector_uses_normal_disjoint_records_training_only(self):
        for job in self.jobs:
            fit,cal=job['detector_fit_global_window_ids'],job['detector_calibration_global_window_ids']
            self.assertFalse(set(fit)&set(cal))
            self.assertTrue(set(fit+cal)<=set(job['training_global_window_ids']))
            self.assertTrue(all(self.rows[i]['class']=='N' for i in fit+cal))
            self.assertEqual({int(self.rows[i]['load_hp']) for i in cal},{max(job['training_loads'])})

    def test_reject_outer_inserted_in_train_or_val(self):
        outer=next(int(r['global_window_id']) for r in self.rows if int(r['load_hp'])==3)
        for key in ('training_global_window_ids','validation_global_window_ids'):
            job=deepcopy(self.jobs[0]);job[key][0]=outer
            with self.subTest(key=key),self.assertRaises(ValueError):data.validate_job(job,self.rows,self.protocol)

    def test_reject_modified_mode_seed_mask_or_unknown_keys(self):
        for key,value in [('mode','P2'),('seed',42),('seed',True),('feature_indices',[0,1]),
                          ('candidate','T6'),('outer_load_hp',2),('fit_signature','outside-job')]:
            job=deepcopy(self.jobs[0]);job[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):data.validate_job(job,self.rows,self.protocol)

    def test_reject_reordered_or_duplicate_ids(self):
        for change in ('order','duplicate'):
            job=deepcopy(self.jobs[0]);ids=job['training_global_window_ids']
            if change=='order':ids.reverse()
            else:ids[1]=ids[0]
            with self.subTest(change=change),self.assertRaises(ValueError):data.validate_job(job,self.rows,self.protocol)

    def test_reject_fit_used_as_detector_calibration(self):
        job=deepcopy(self.jobs[0]);job['detector_calibration_global_window_ids']=job['detector_fit_global_window_ids']
        with self.assertRaises(ValueError):data.validate_job(job,self.rows,self.protocol)

    def _synthetic_load(self,job,bad=None,tamper_hash=False):
        arrays={}
        for relative in data.ALLOWED_FEATURES:
            count=sum(r['source_features']==relative for r in self.rows)
            arrays[relative]=TrackingArray(count,(bad or {}).get(relative,()))
        calls=[]
        def np_loader(path,**kwargs):
            relative=Path(path).relative_to(data.ROOT).as_posix();calls.append(relative)
            self.assertEqual(kwargs,dict(allow_pickle=False,mmap_mode='r'))
            self.assertIn(relative,data.ALLOWED_FEATURES)
            self.assertNotIn('test.npy',relative)
            return arrays[relative]
        def hash_value(path):
            relative=Path(path).relative_to(data.ROOT).as_posix()
            return 'bad' if tamper_hash else self.hashes.get(relative,'index')
        with patch.object(data,'context',return_value=(self.protocol,self.rows,self.hashes)), \
             patch.object(data,'sha256',side_effect=hash_value),patch.object(data.np,'load',side_effect=np_loader):
            result=data.load_data(job)
        return result,arrays,calls

    def test_loader_three_inner_folds_exact_row_selection_and_types(self):
        for inner in (0,1,2):
            job=next(j for j in self.jobs if j['inner_validation_load_hp']==inner)
            (x,y,val,val_y,provenance),arrays,calls=self._synthetic_load(job)
            self.assertEqual(x.shape,(len(job['training_global_window_ids']),14))
            self.assertEqual(val.shape,(len(job['validation_global_window_ids']),14))
            self.assertEqual((x.dtype,val.dtype,y.dtype,val_y.dtype),(np.float32,np.float32,np.int64,np.int64))
            selected=[self.rows[k] for k in job['training_global_window_ids']+job['validation_global_window_ids']]
            for relative,array in arrays.items():
                self.assertEqual(array.read,[int(r['source_row']) for r in selected if r['source_features']==relative])
            self.assertEqual(set(calls),data.ALLOWED_FEATURES)
            self.assertFalse(provenance['outer3_arrays_opened'])
            self.assertFalse(provenance['preprocessing_applied'])
            self.assertFalse(provenance['outer_rows_selected_or_materialized'])
            self.assertTrue(provenance['mapping_or_hashing_is_not_OS_access_trace'])

    def test_loader_selected_nonfinite_stops(self):
        row=self.rows[self.jobs[0]['training_global_window_ids'][0]]
        with self.assertRaises(ValueError):self._synthetic_load(self.jobs[0],{row['source_features']:[int(row['source_row'])]})

    def test_loader_rejects_changed_cache_hash_before_numpy_load(self):
        with self.assertRaises(ValueError):self._synthetic_load(self.jobs[0],tamper_hash=True)

    def test_does_not_read_incompatible_test_cache(self):
        # The fake source catalog intentionally omits the cache for outer3.
        job=self.jobs[-1]
        result,_,calls=self._synthetic_load(job)
        self.assertEqual(len(calls),2)
        self.assertEqual(len(result[0]),568)


if __name__=='__main__':unittest.main()
