"""Contrato de alcance, referencias selladas y protección de los bancos."""
from __future__ import annotations
import copy
import csv
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import preparar_bancos as b


class BankContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.existed=b.OUT.exists()
        cls.context=b.e.load_context()
        cls.catalog,*_=b.e.build_catalog(cls.context)
        cls.plan,cls.raw,cls.rawrefs,cls.postrefs,cls.sources=b.build_banks(cls.context)
        with (b.e.ROOT/b.PARTITIONS_REL/'indice_ventanas_fuente.csv').open(encoding='utf-8-sig',newline='') as f:
            cls.metadata=list(csv.DictReader(f))

    @classmethod
    def tearDownClass(cls):
        if b.OUT.exists()!=cls.existed:raise AssertionError('Prueba creó salida real')
        for relative,digest in cls.sources.items():
            if b.e.io.sha(b.e.ROOT/relative)!=digest:raise AssertionError('Fuente modificada por prueba '+relative)

    def test_56_unique_first_last_windows_and_six_models_per_payload(self):
        ids=self.plan['RAW_global_window_ids']
        self.assertEqual(len(ids),len(set(ids)))
        self.assertEqual(len(ids),56)
        for row in self.plan['RAW_windows']:
            record=[r for r in self.metadata if r['file_id']==row['metadata']['file_id']]
            expected={min(int(r['window_index_in_file']) for r in record),max(int(r['window_index_in_file']) for r in record)}
            self.assertIn(int(row['metadata']['window_index_in_file']),expected)
            self.assertEqual(len(row['authorized_model_ids']),6)
        self.assertEqual(self.raw['windows'].shape,(56,4096,2))
        self.assertEqual(self.raw['windows'].dtype,np.float32)

    def test_each_model_has_42_development_RAW_and_all_its_heldout_POST(self):
        byid={int(r['global_window_id']):r for r in self.metadata}
        for model in self.plan['models']:
            outer=model['outer_load_hp'];raw=model['raw_global_window_ids'];post=model['post_global_window_ids']
            self.assertEqual(len(raw),42)
            self.assertEqual(len({byid[g]['file_id'] for g in raw}),21)
            self.assertEqual({int(byid[g]['load_hp']) for g in raw},set(range(4))-{outer})
            self.assertEqual(post,[int(r['global_window_id']) for r in self.metadata if int(r['load_hp'])==outer])
            self.assertFalse(set(raw)&set(post))
        self.assertEqual(sum(len(m['post_global_window_ids']) for m in self.plan['models']),2760)

    def test_references_include_masked_dimensions_and_original_int8_outputs(self):
        for model,row in zip(self.plan['models'],self.catalog['representatives']):
            prefix=model['reference_prefix'];d=row['dims']
            for arrays,n in ((self.rawrefs,42),(self.postrefs,len(model['post_global_window_ids']))):
                self.assertEqual(arrays[prefix+'features14'].shape,(n,14))
                self.assertEqual(arrays[prefix+'normalized_features'].shape,(n,d))
                self.assertEqual(arrays[prefix+'int8_inputs'].shape,(n,d))
                self.assertEqual(arrays[prefix+'int8_outputs'].dtype,np.int8)
                self.assertTrue(np.array_equal(arrays[prefix+'int8_predictions'],
                    np.argmax(arrays[prefix+'int8_outputs'],axis=1)))
                self.assertEqual(arrays[prefix+'detector_scores'].dtype,np.float32)
                threshold=b._from_bits([row['threshold_bits']])[0]
                self.assertTrue(np.array_equal(arrays[prefix+'detector_rejected'],arrays[prefix+'detector_scores']>threshold))

    def test_partial_RAW_scope_and_POST_timing_limits_are_explicit(self):
        policy=self.plan['policy']
        self.assertFalse(policy['RAW_exhaustive_reserved_coverage'])
        self.assertFalse(policy['POST_timing_is_full_pipeline_cost'])
        self.assertFalse(policy['old_timing_imports_allowed'])
        self.assertTrue(policy['RAW_references_are_training_diagnostics'])
        self.assertEqual(policy['classifier_invocations'],0)
        self.assertEqual(policy['training_fits'],0)
        self.assertNotIn('tensorflow',sys.modules)
        self.assertNotIn('sklearn',sys.modules)

    def test_duplicate_window_or_reordered_global_id_rejected(self):
        rows=copy.deepcopy(self.metadata);rows[0]['global_window_id']='1'
        with self.assertRaisesRegex(ValueError,'Índice global'):b.build_plan(rows,self.catalog)
        rows=copy.deepcopy(self.metadata)
        same_record=[r for r in rows if r['file_id']==rows[0]['file_id']]
        same_record[-1]['window_index_in_file']=same_record[0]['window_index_in_file']
        with self.assertRaisesRegex(ValueError,'Primera/última'):b.build_plan(rows,self.catalog)

    def test_foreign_catalog_or_outer_in_training_scope_rejected(self):
        cat=copy.deepcopy(self.catalog);cat['representatives'][0]['seed']=44
        with self.assertRaisesRegex(ValueError,'RAW fuera'):b.build_plan(self.metadata,cat)
        cat=copy.deepcopy(self.catalog);cat['representatives'][0]['scope']['training_global_window_ids']=[]
        with self.assertRaisesRegex(ValueError,'Roles RAW/POST'):b.build_plan(self.metadata,cat)

    def test_int8_or_normalized_reference_changes_fail_closed(self):
        row=self.catalog['representatives'][0];p='m0_';r=self.rawrefs
        z=r[p+'normalized_features'].copy();z[0,0]=np.nextafter(z[0,0],np.float32(np.inf))
        with self.assertRaisesRegex(ValueError,'Normalización'):
            b._verify_reference(r[p+'features14'],z,r[p+'int8_inputs'],r[p+'int8_outputs'],r[p+'int8_predictions'],row)
        iq=r[p+'int8_inputs'].copy();iq[0,0]^=1
        with self.assertRaisesRegex(ValueError,'INT8 de referencia'):
            b._verify_reference(r[p+'features14'],r[p+'normalized_features'],iq,r[p+'int8_outputs'],r[p+'int8_predictions'],row)

    def test_bit_comparison_handles_noncontiguous_and_preserves_signed_zero(self):
        a=np.arange(24,dtype=np.float32).reshape(4,6)[:,::2]
        self.assertTrue(b.same(a,a.copy()))
        self.assertFalse(b.same(np.asarray([0.],dtype=np.float32),np.asarray([-0.],dtype=np.float32)))

    def test_exclusive_write_verify_mutation_and_path_escape(self):
        temp=Path(tempfile.mkdtemp(prefix='canonbanks_test_',dir=b.e.ROOT/'tmp')).resolve()
        try:
            output=temp/'bancos'
            material=(self.plan,self.raw,self.rawrefs,self.postrefs,self.sources)
            with patch.object(b,'build_banks',return_value=material):
                report=b.prepare(output)
                self.assertEqual(report['POST_model_window_pairs'],2760)
                self.assertEqual(b.verify(output)['counts'],self.plan['counts'])
                with self.assertRaisesRegex(ValueError,'No sobrescribir'):b.prepare(output)
            target=output/'referencia_post.npz';original=target.read_bytes();target.write_bytes(original+b'changed')
            with self.assertRaisesRegex(ValueError,'Banco modificado'):b.verify(output)
            target.write_bytes(original)
            (output/'extra.txt').write_text('unexpected',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'añadidos o ausentes'):b.verify(output)
            with self.assertRaisesRegex(ValueError,'fuera del proyecto'):b.prepare(b.e.ROOT.parent/'bad_bank')
        finally:
            if not temp.is_relative_to((b.e.ROOT/'tmp').resolve()) or not temp.name.startswith('canonbanks_test_'):
                raise AssertionError('No se borra temporal fuera de tmp')
            shutil.rmtree(temp)


if __name__=='__main__':unittest.main(verbosity=2)
