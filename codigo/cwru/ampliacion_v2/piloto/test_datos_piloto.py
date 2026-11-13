"""Guardias del piloto con metadatos sinteticos; sin arrays CWRU ni TF."""
from __future__ import annotations

from collections import Counter
import copy
from pathlib import Path
import unittest

from datos_piloto import (ALLOWED_FEATURES, FEATURE_BASE, MASKS,
                          bounded_feature_path, validate_job_scope)


def fixture(mask='F14'):
    rows = []
    source_offsets = {'train': 0, 'validation': 0, 'test': 0}
    for load, count in ((0, 175), (1, 393), (2, 406), (3, 406)):
        split = ('train', 'train', 'validation', 'test')[load]
        for i in range(count):
            c = ('N', 'IR', 'OR')[i % 3]
            rows.append(dict(global_window_id=str(len(rows)), load_hp=str(load),
                source_split_v1=split, source_row=str(source_offsets[split]),
                source_features=FEATURE_BASE+'X_features_'+split+'.npy',
                file_id=str(load*10+(i%7)), label=str(('N','IR','OR').index(c)), **{'class': c}))
            source_offsets[split] += 1
    def scope(loads):
        selected = [r for r in rows if int(r['load_hp']) in loads]
        counts=Counter(r['class'] for r in selected)
        return dict(loads_hp=loads, normal_only=False, windows=len(selected),
            global_window_ids=[int(r['global_window_id']) for r in selected],
            file_ids=sorted({r['file_id'] for r in selected}, key=int),
            class_window_counts={k:counts[k] for k in ('N','IR','OR')})
    train=scope([1,2])
    job=dict(job_id='synthetic_'+mask, mask=mask, feature_indices=MASKS[mask].copy(),
        input_dimension=len(MASKS[mask]), seed=11, fixed_epochs=50, early_stopping=False,
        outer_load_hp=3, inner_validation_load_hp=0, timeout_worker_seconds=300,
        runtime_and_finiteness_diagnostics_only=True, metrics_used_for_selection=False,
        reuse_as_predictive_V2_fit=False, train=train, validation=scope([0]),
        representative_int8_calibration=copy.deepcopy(train),
        authorized_feature_source_paths=sorted(ALLOWED_FEATURES))
    return job, rows


class PilotScopeTests(unittest.TestCase):
    def test_both_masks_accept_exact_scopes(self):
        for mask in MASKS:
            train, val=validate_job_scope(*fixture(mask))
            self.assertEqual((len(train),len(val)),(799,175))

    def test_no_load3_rows_are_materialized(self):
        train,val=validate_job_scope(*fixture())
        self.assertTrue(all(int(r['load_hp'])!=3 for r in train+val))

    def test_duplicate_scope_id_rejected(self):
        job,rows=fixture(); job['train']['global_window_ids'][1]=job['train']['global_window_ids'][0]
        with self.assertRaises(ValueError): validate_job_scope(job,rows)

    def test_validation_id_in_train_rejected(self):
        job,rows=fixture(); job['train']['global_window_ids'][0]=0
        with self.assertRaises(ValueError): validate_job_scope(job,rows)

    def test_outer_id_in_train_rejected(self):
        job,rows=fixture(); job['train']['global_window_ids'][-1]=1379
        with self.assertRaises(ValueError): validate_job_scope(job,rows)

    def test_wrong_mask_rejected(self):
        job,rows=fixture(); job['feature_indices']=[0,1]
        with self.assertRaises(ValueError): validate_job_scope(job,rows)

    def test_extra_epoch_or_seed_rejected(self):
        for key,value in (('fixed_epochs',51),('seed',42),('early_stopping',True),('timeout_worker_seconds',301)):
            job,rows=fixture(); job[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError): validate_job_scope(job,rows)

    def test_calibration_not_training_rejected(self):
        job,rows=fixture(); job['representative_int8_calibration']=copy.deepcopy(job['validation'])
        with self.assertRaises(ValueError): validate_job_scope(job,rows)

    def test_duplicate_metadata_ids_rejected(self):
        job,rows=fixture(); rows.append(copy.deepcopy(rows[0]))
        with self.assertRaises(ValueError): validate_job_scope(job,rows)

    def test_reused_source_row_rejected(self):
        job,rows=fixture(); rows[1]['source_row']=rows[0]['source_row']
        with self.assertRaises(ValueError): validate_job_scope(job,rows)

    def test_wrong_label_rejected(self):
        job,rows=fixture(); rows[0]['label']='2'
        with self.assertRaises(ValueError): validate_job_scope(job,rows)

    def test_test_source_and_wrong_count_rejected(self):
        job,rows=fixture(); rows[0]['source_features']=FEATURE_BASE+'X_features_test.npy'
        with self.assertRaises(ValueError): validate_job_scope(job,rows)
        job,rows=fixture(); job['validation']['windows']=176
        with self.assertRaises(ValueError): validate_job_scope(job,rows)

    def test_path_allowlist_and_escape(self):
        # Solo resolucion de rutas, sin crear o borrar directorios temporales.
        synthetic_root = Path.cwd() / '__synthetic_pilot_scope__'
        for path in ALLOWED_FEATURES:
            self.assertTrue(bounded_feature_path(path,synthetic_root).is_relative_to(synthetic_root.resolve()))
        for path in ('../X_features_train.npy',FEATURE_BASE+'X_features_test.npy','C:/other.npy',FEATURE_BASE+'y_train.npy'):
            with self.subTest(path=path),self.assertRaises(ValueError): bounded_feature_path(path,synthetic_root)

    def test_changed_class_and_file_counts_rejected(self):
        job,rows=fixture(); job['train']['class_window_counts']['N']+=1
        with self.assertRaises(ValueError): validate_job_scope(job,rows)
        job,rows=fixture(); job['train']['file_ids'].pop()
        with self.assertRaises(ValueError): validate_job_scope(job,rows)


if __name__=='__main__': unittest.main(verbosity=2)
