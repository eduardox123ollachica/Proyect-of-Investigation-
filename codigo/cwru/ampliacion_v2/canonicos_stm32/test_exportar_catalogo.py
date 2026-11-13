"""Pruebas sin ajuste/inferencia/RAW: ABI, canon fijado y exportación en fixture."""
from __future__ import annotations
import copy
import hashlib
import importlib.util
from pathlib import Path
import shutil
import struct
import sys
import tempfile
import unittest

import numpy as np
import exportar_catalogo as e


class CatalogContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.context=e.load_context()
        cls.cat,cls.models,cls.arrays,cls.sources=e.build_catalog(cls.context)
        cls.real_output_existed=e.OUT.exists()

    @classmethod
    def tearDownClass(cls):
        for path,digest in cls.sources.items():
            if e.io.sha(e.ROOT/path)!=digest:raise AssertionError('Prueba modificó fuente '+path)
        if e.OUT.exists()!=cls.real_output_existed:raise AssertionError('Prueba creó la salida real')

    def test_eight_frozen_seed42_models_and_mask_ids_are_not_catalog_indices(self):
        rows=self.cat['representatives']
        self.assertEqual(len(rows),8)
        self.assertEqual([r['id'] for r in rows],list(range(8)))
        self.assertEqual(tuple(r['mask_id'] for r in rows),(0,1,0,7,0,1,0,3))
        self.assertNotEqual([r['mask_id'] for r in rows],[i%8 for i in range(8)])
        self.assertEqual(tuple(r['representative_id'] for r in rows),e.MODEL_NAMES)
        self.assertTrue(all(r['seed']==42 for r in rows))

    def test_param_wire_abi_is_297_bytes_with_padding_and_int32_zero(self):
        for row,a in zip(self.cat['representatives'],self.arrays):
            blob=bytes.fromhex(row['params_bytes_hex']);d=row['dims'];nd=row['det_dims']
            self.assertEqual(len(blob),297)
            self.assertEqual(struct.unpack_from('<BBB',blob),(d,nd,row['mask_id']))
            self.assertEqual(list(blob[3:3+d]),row['feature_indices'])
            self.assertEqual(blob[3+d:17],b'\0'*(14-d))
            self.assertEqual(list(blob[17:17+nd]),row['detector_indices'])
            self.assertEqual(blob[17+nd:29],b'\0'*(12-nd))
            values=np.frombuffer(blob[29:293],dtype='<f4')
            self.assertEqual(len(values),66)
            self.assertTrue(np.array_equal(values[:14],a['mean']))
            self.assertTrue(np.array_equal(values[14:28],a['scale']))
            self.assertTrue(np.array_equal(values[28:40],a['det_mean']))
            self.assertTrue(np.array_equal(values[40:52],a['det_scale']))
            self.assertTrue(np.array_equal(values[52:64],a['centroid']))
            self.assertEqual(values[64].tobytes(),a['threshold'].tobytes())
            self.assertEqual(values[65].tobytes(),a['qscale'].tobytes())
            self.assertEqual(struct.unpack_from('<i',blob,293)[0],a['qzero'])
            self.assertTrue(np.array_equal(a['scale'][d:],np.ones(14-d,dtype=np.float32)))
            self.assertTrue(np.array_equal(a['det_scale'][nd:],np.ones(12-nd,dtype=np.float32)))

    def test_existing_UART_parameter_serializer_matches_all_eight_descriptors(self):
        path=e.ROOT/'codigo/cwru/ampliacion_v2/costo_total_single_rounding/validar_total_sr.py'
        spec=importlib.util.spec_from_file_location('old_uart_param_codec',path)
        client=importlib.util.module_from_spec(spec);spec.loader.exec_module(client)
        for row in self.cat['representatives']:
            blob=client.params_bytes(row)
            self.assertEqual(blob.hex(),row['params_bytes_hex'])
            self.assertEqual(client.fnv(blob),row['params_fnv'])

    def test_threshold_bits_are_uint32_hex_not_little_endian_byte_hex(self):
        row=self.cat['representatives'][0]
        self.assertEqual(row['threshold_bits'],'42baffa1')
        self.assertNotEqual(row['threshold_bits'],'a1ffba42')
        blob=bytes.fromhex(row['params_bytes_hex'])
        self.assertEqual(blob[285:289].hex(),'a1ffba42')
        self.assertEqual(f'{struct.unpack_from("<I",blob,285)[0]:08x}',row['threshold_bits'])

    def test_model_bytes_and_both_fingerprints_match_sources_without_conversion(self):
        self.assertEqual(sum(map(len,self.models)),28688)
        for row,blob in zip(self.cat['representatives'],self.models):
            self.assertEqual(hashlib.sha256(blob).hexdigest(),row['model_sha256'])
            self.assertEqual(f'{e.fnv1a(blob):08x}',row['model_fnv'])
            self.assertEqual(f'{e.fnv1a(bytes.fromhex(row["params_bytes_hex"])):08x}',row['params_fnv'])

    def test_canonical_sha_binds_detector_provenance_and_scope_beyond_numeric_FNV(self):
        original=hashlib.sha256(e.canonical_bytes(self.cat['representatives'])).hexdigest()
        self.assertEqual(original,self.cat['catalog_sha256'])
        changed=copy.deepcopy(self.cat['representatives'])
        changed[0]['scope']['detector_fit_signature']='0'*64
        self.assertNotEqual(hashlib.sha256(e.canonical_bytes(changed)).hexdigest(),original)
        self.assertEqual(changed[0]['params_fnv'],self.cat['representatives'][0]['params_fnv'])
        packed=b''.join(struct.pack('<II',int(r['model_fnv'],16),int(r['params_fnv'],16)) for r in changed)
        self.assertEqual(f'{e.fnv1a(packed):08x}',self.cat['catalog_fnv'])

    def test_seed_selected_after_evaluation_or_unfrozen_context_is_rejected(self):
        for field,value in [('seed',11),('seed_fixed_before_outer_evaluation',False),('hardware_execution_enabled',True)]:
            context=copy.deepcopy(self.context);context['canonical'][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):
                e.validate_metadata(*[context[k] for k in ('canonical','mapping','inventory','refit_plan')])
        for field,value in [('seed',88),('seed_selected_from_outer_performance',True),('pending_hardware_validation',False)]:
            context=copy.deepcopy(self.context);context['canonical']['models'][0][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):
                e.validate_metadata(*[context[k] for k in ('canonical','mapping','inventory','refit_plan')])

    def test_metrics_are_not_used_to_choose_or_reorder_the_canonical_models(self):
        context=copy.deepcopy(self.context)
        for row in context['canonical']['models']:row['metrics']={'fake_favorable_score':123456.}
        updated,_,_,_=e.build_catalog(context)
        self.assertEqual(updated['catalog_sha256'],self.cat['catalog_sha256'])
        self.assertEqual(updated['catalog_fnv'],self.cat['catalog_fnv'])

    def test_duplicate_or_missing_model_aliases_are_rejected(self):
        for mutation in ('duplicate_canonical','missing_inventory','duplicate_refit','wrong_detector','wrong_alias'):
            context=copy.deepcopy(self.context)
            if mutation=='duplicate_canonical':context['canonical']['models'][-1]=context['canonical']['models'][0]
            elif mutation=='missing_inventory':context['inventory'].pop()
            elif mutation=='duplicate_refit':context['refit_plan']['jobs'].append(context['refit_plan']['jobs'][0])
            elif mutation=='wrong_detector':context['mapping']['outer_scenarios'][0]['DNN_procedures']['baseline_F14']['detector_job_id']='outer3_detector_06'
            else:context['canonical']['models'][0]['procedures']=['validated_total_cost']
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                e.validate_metadata(*[context[k] for k in ('canonical','mapping','inventory','refit_plan')])

    def test_artifact_fingerprint_tamper_is_rejected_before_export(self):
        context=copy.deepcopy(self.context)
        model=context['canonical']['models'][0];model['artifact_sha256']['modelo_int8.tflite']='0'*64
        original=next(r for r in context['inventory'] if r['job_id']==model['model_job_id'])
        original['artifact_sha256']['modelo_int8.tflite']='0'*64
        with self.assertRaises(ValueError):e.build_catalog(context)

    def test_invalid_numeric_parameters_fail_without_repair(self):
        row=self.cat['representatives'][1];a=self.arrays[1]
        args=[row['dims'],row['det_dims'],row['mask_id'],row['feature_indices'],row['detector_indices'],
            a['mean'][:row['dims']],a['scale'][:row['dims']],a['det_mean'][:row['det_dims']],
            a['det_scale'][:row['det_dims']],a['centroid'][:row['det_dims']],a['threshold'],a['qscale'],a['qzero']]
        for index,value in [(0,True),(2,8),(10,np.nan),(10,-1.),(11,0.),(11,np.inf),(12,True),(12,128)]:
            changed=copy.deepcopy(args);changed[index]=value
            with self.subTest(index=index,value=value),self.assertRaises(ValueError):e.serialize_params(*changed)
        changed=copy.deepcopy(args);changed[6][0]=0.
        with self.assertRaises(ValueError):e.serialize_params(*changed)

    def test_C_header_and_symbols_keep_eight_catalog_and_exact_jobs(self):
        header=e.render_header();cpp=e.render_cpp(self.cat,self.models,self.arrays)
        self.assertIn('#define CWRUTV_COUNT 8U',header)
        self.assertNotIn('#define CWRUTV_COUNT 32U',header)
        self.assertEqual(cpp.count('alignas(16) static const unsigned char'),8)
        for row in self.cat['representatives']:
            self.assertIn('cwru_c42_'+row['representative_id'],cpp)
            self.assertIn('"'+row['representative_id']+'"',cpp)
        self.assertIn('CWRUTV_Descriptor',header)

    def test_export_fixture_has_no_RAW_bank_preserves_models_and_rejects_tamper_and_overwrite(self):
        base=(e.ROOT/'tmp').resolve();base.mkdir(exist_ok=True)
        folder=Path(tempfile.mkdtemp(prefix='canon42_catalog_test_',dir=base)).resolve()
        try:
            output=folder/'recursos';report=e.export_resources(output)
            self.assertEqual(report['count'],8);self.assertEqual(report['RAW_windows_exported'],0)
            self.assertEqual(report['training_fits'],0);self.assertEqual(report['TFLite_invocations'],0)
            self.assertFalse((output/'BANCO_TOTAL.json').exists());self.assertFalse(list(output.glob('*.npz')))
            verified=e.verify_resources(output)
            self.assertEqual(verified['catalog_sha256'],self.cat['catalog_sha256'])
            with self.assertRaises(ValueError):e.export_resources(output)
            first=output/self.cat['representatives'][0]['model_path'];blob=bytearray(first.read_bytes());blob[-1]^=1
            first.write_bytes(blob)
            with self.assertRaises(ValueError):e.verify_resources(output)
        finally:
            if folder.parent!=base or not folder.name.startswith('canon42_catalog_test_'):raise ValueError('Fixture no confinada')
            shutil.rmtree(folder)

    def test_paths_cannot_escape_workspace_and_import_does_not_load_ML(self):
        with self.assertRaises(ValueError):e.bounded(e.ROOT.parent/'outside')
        with self.assertRaises(ValueError):e.bounded(e.ROOT)
        self.assertNotIn('tensorflow',sys.modules);self.assertNotIn('sklearn',sys.modules)


if __name__=='__main__':unittest.main(verbosity=2)
