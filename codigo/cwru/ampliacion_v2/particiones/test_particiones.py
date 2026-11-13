"""Pruebas positivas/negativas con registros sintéticos, sin tocar datos reales."""
import unittest
import numpy as np
from contrato_particiones import (make_roles,validate_roles,validate_records,validate_config,
                                 guard_fit_scope,select_training_rows)


def fixture():
    return [dict(file_id=str(7*load+j),file=f'synthetic_{load}_{j}.mat',load_hp=load,
                 **{'class':cls},label={'N':0,'IR':1,'OR':2}[cls],de_key=f'DE_{load}_{j}',fe_key=f'FE_{load}_{j}')
            for load in range(4) for j,cls in enumerate(['N','IR','IR','IR','OR','OR','OR'])]


class PartitionTests(unittest.TestCase):
    def test_outer_coverage_once(self):
        records=fixture();seen=[]
        for outer in range(4):
            rows=make_roles(records,outer);validate_roles(rows,outer)
            seen.extend(r['file_id'] for r in rows if r['role']=='outer_test')
            self.assertEqual(sum(r['role']=='development' for r in rows),21)
        self.assertEqual(sorted(seen),sorted(r['file_id'] for r in records))

    def test_inner_counts_and_validation_coverage(self):
        records=fixture()
        for outer in range(4):
            validation=[]
            for inner in [x for x in range(4) if x!=outer]:
                rows=make_roles(records,outer,inner);validate_roles(rows,outer,inner)
                self.assertEqual(sum(r['role']=='inner_train' for r in rows),14)
                self.assertEqual(sum(r['role']=='inner_validation' for r in rows),7)
                self.assertEqual(sum(r['role']=='outer_excluded' for r in rows),7)
                validation.extend(r['file_id'] for r in rows if r['role']=='inner_validation')
            expected=[r['file_id'] for r in records if r['load_hp']!=outer]
            self.assertEqual(sorted(validation),sorted(expected))

    def test_same_outer_inner_load_rejected(self):
        with self.assertRaises(ValueError):make_roles(fixture(),2,2)

    def test_relabelled_outer_record_as_training_rejected(self):
        rows=make_roles(fixture(),0,1);rows[0]['role']='inner_train'
        with self.assertRaises(ValueError):validate_roles(rows,0,1)

    def test_duplicate_record_rejected(self):
        rows=fixture();rows[1]=dict(rows[0])
        with self.assertRaises(ValueError):validate_records(rows)

    def test_changed_overlap_rejected(self):
        cfg=dict(output_sampling_rate_hz=12000,window_size_samples=4096,hop_size_samples=2048,
                 overlap_fraction=.5,channel_order=['DE','FE'])
        validate_config(cfg);cfg['hop_size_samples']=4096
        with self.assertRaises(ValueError):validate_config(cfg)

    def test_validation_or_external_fit_rejected(self):
        rows=make_roles(fixture(),3,2)
        index=[dict(global_window_id=i,file_id=r['file_id']) for i,r in enumerate(rows)]
        allowed=[r['file_id'] for r in rows if r['role']=='inner_train']
        for role in ('inner_validation','outer_excluded'):
            bad=next(i for i,r in enumerate(rows) if r['role']==role)
            with self.assertRaises(ValueError):guard_fit_scope(index,[bad],allowed)

    def test_duplicate_index_rejected(self):
        with self.assertRaises(ValueError):guard_fit_scope([dict(global_window_id=0,file_id='a')]*2,[0],['a'])

    def test_unknown_or_repeated_request_rejected(self):
        index=[dict(global_window_id=0,file_id='a')]
        with self.assertRaises(ValueError):guard_fit_scope(index,[5],['a'])
        with self.assertRaises(ValueError):guard_fit_scope(index,[0,0],['a'])

    def test_external_sentinel_cannot_change_training_view(self):
        index=[dict(global_window_id=i,file_id=str(i)) for i in range(3)]
        x=np.array([[1.0],[3.0],[10.0]])
        a=select_training_rows(x,index,[0,1],['0','1']).mean()
        x[2]=1e12
        b=select_training_rows(x,index,[0,1],['0','1']).mean()
        self.assertEqual(a,b);self.assertEqual(a,2.0)


if __name__=='__main__':unittest.main(verbosity=2)
