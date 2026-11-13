"""Pruebas de adaptación histórica y caché del cierre; sin fit/inferencia."""
from __future__ import annotations

import copy
import io
import json
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import patch

import numpy as np
import cerrar_matriz_dnn_interna_v2 as close
import datos_busqueda as data

e=close.e


class ClosureCompatibility(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan=e.read(e.PLAN_FILE)
        cls.by_id={j['job_id']:j for j in cls.plan['jobs']}

    def test_all_sixteen_supplement_P0_results_map_to_canonical_roles_without_mutation(self):
        count=0
        for job_id,reference in self.plan['imported'].items():
            if reference['family'] != 'suplemento_P0':continue
            count+=1
            path=e.ROOT/reference['source_directory']/'resultado.json'
            before=e.sha(path);source=e.read(path);original=copy.deepcopy(source)
            mapped=close.compatible_result(source,'suplemento_P0')
            job=self.by_id[job_id]
            self.assertEqual(mapped['outer_load_hp'],job['outer_load_hp'])
            self.assertEqual(mapped['inner_validation_load_hp'],job['inner_validation_load_hp'])
            self.assertEqual(mapped['FP32_INT8_class_changes'],source['class_changes'])
            self.assertEqual(mapped['training_saturated_input_values'],source['training_clipped_values'])
            self.assertEqual(mapped['roles'],job['roles'])
            self.assertEqual(mapped['fit_signature'],reference['source_fit_signature'])
            self.assertEqual(source,original)
            self.assertEqual(e.sha(path),before)
        self.assertEqual(count,16)

    def test_all_thirtytwo_representatives_keep_original_result_and_signature(self):
        count=0
        for job_id,reference in self.plan['imported'].items():
            if reference['family'] != 'representantes':continue
            count+=1
            source=e.read(e.ROOT/reference['source_directory']/'resultado.json')
            mapped=close.compatible_result(source,'representantes')
            self.assertIs(mapped,source)
            self.assertEqual(mapped['fit_signature'],reference['source_fit_signature'])
            self.assertEqual(mapped['roles'],self.by_id[job_id]['roles'])
        self.assertEqual(count,32)

    def test_nonidentity_P1_result_is_rejected(self):
        reference=next(r for r in self.plan['imported'].values() if r['family']=='suplemento_P0')
        source=e.read(e.ROOT/reference['source_directory']/'resultado.json');source['mode']='P1'
        with self.assertRaisesRegex(ValueError,'otro preprocesamiento'):
            close.compatible_result(source,'suplemento_P0')

    def test_each_contradictory_alias_is_rejected(self):
        reference=next(r for r in self.plan['imported'].values() if r['family']=='suplemento_P0')
        original=e.read(e.ROOT/reference['source_directory']/'resultado.json')
        aliases=[('outer_load_hp','outer'),('inner_validation_load_hp','inner'),
            ('FP32_INT8_class_changes','class_changes'),('training_saturated_input_values','training_clipped_values')]
        for target,source in aliases:
            with self.subTest(alias=target):
                changed=copy.deepcopy(original);changed[target]=changed[source]+1
                with self.assertRaisesRegex(ValueError,'contradictorio'):
                    close.compatible_result(changed,'suplemento_P0')

    def test_all_288_results_supply_closure_schema_and_canonical_roles(self):
        required=('candidate','outer_load_hp','inner_validation_load_hp','seed','roles',
            'training_windows','validation_windows','metrics','best_epoch','epochs_executed',
            'FP32_INT8_class_changes','training_saturated_input_values','int8_details','model_sha256','fit_signature')
        for job in self.plan['jobs']:
            imported=self.plan['imported'].get(job['job_id'])
            folder=e.ROOT/imported['source_directory'] if imported else close.CAMPAIGN/job['job_id']
            source=e.read(folder/'resultado.json')
            mapped=close.compatible_result(source,imported['family'] if imported else 'new_inner_campaign')
            with self.subTest(job=job['job_id']):
                self.assertTrue(all(k in mapped for k in required))
                self.assertTrue(all(mapped[k]==job[k] for k in ('candidate','outer_load_hp','inner_validation_load_hp','seed','roles')))
                self.assertIn('input_clipped_values',mapped['int8_details'])

    def test_snapshot_preserves_one_authorized_job_arrays_provenance_and_plan(self):
        original_context=data.context;original_read=e.read
        snapshot=original_context();context_digest=e.canonical(snapshot)
        plan_snapshot=original_read(e.PLAN_FILE);plan_digest=e.sha(e.PLAN_FILE)
        job=next(j for j in self.plan['jobs'] if j['job_id']=='inner_outer0_inner1_F14_seed22')
        original=data.load_job_data(job)
        def cached_read(path):
            return plan_snapshot if Path(path).resolve()==e.PLAN_FILE.resolve() else original_read(path)
        with patch.object(data,'context',return_value=snapshot),patch.object(e,'read',side_effect=cached_read):
            cached=data.load_job_data(job)
            self.assertIs(e.read(e.PLAN_FILE),plan_snapshot)
            self.assertEqual(e.read(close.CAMPAIGN/job['job_id']/'resultado.json')['job_id'],job['job_id'])
        for expected,actual in zip(original[:4],cached[:4]):
            self.assertEqual(expected.dtype,actual.dtype)
            self.assertEqual(expected.shape,actual.shape)
            self.assertEqual(expected.tobytes(),actual.tobytes())
        self.assertEqual(original[4],cached[4])
        self.assertFalse(cached[4]['outer_rows_selected_or_materialized'])
        self.assertEqual(e.canonical(snapshot),context_digest)
        self.assertEqual(e.canonical(original_context()),context_digest)
        self.assertEqual(e.sha(e.PLAN_FILE),plan_digest)

    def test_auditor_imports_no_ML_runtime(self):
        self.assertNotIn('tensorflow',sys.modules)
        self.assertNotIn('keras',sys.modules)


if __name__=='__main__':
    source_files=[Path(__file__).resolve(),Path(close.__file__).resolve(),
        Path(__file__).with_name('analizar_variabilidad_matriz_interna_v2.py')]
    before={p.relative_to(e.ROOT).as_posix():e.sha(p) for p in source_files}
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(ClosureCompatibility)
    names=unittest.defaultTestLoader.getTestCaseNames(ClosureCompatibility)
    stream=io.StringIO();started=time.perf_counter()
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    after={p.relative_to(e.ROOT).as_posix():e.sha(p) for p in source_files}
    report=dict(status='PASS' if result.wasSuccessful() and before==after else 'FAIL',
        tests_run=result.testsRun,test_names=names,failures=len(result.failures),errors=len(result.errors),
        seconds=time.perf_counter()-started,source_sha256=after,source_sha256_before=before,
        source_sha256_after=after,sources_unchanged_during_tests=before==after,
        real_training_fits=0,real_TFLite_invocations=0,TensorFlow_imported='tensorflow' in sys.modules,
        runtime_import_blocked_by_auditor=True,outer_evaluation_executed=False,
        scope='Historical result adapter on16P0+32REP/all288schema; metadata cache equivalence on one real authorized job.',
        python_executable=sys.executable,test_output=stream.getvalue())
    output=e.ROOT/'articulo/ampliacion_cwru/metodologia/comprobaciones_cierre_matriz_interna_v2_2026-10-26'
    output.mkdir(parents=True,exist_ok=True)
    e.write(output/'PRUEBAS_CIERRE.json',report)
    print(stream.getvalue())
    print(json.dumps({k:report[k] for k in ('status','tests_run','failures','errors','seconds','TensorFlow_imported')}))
    raise SystemExit(0 if report['status']=='PASS' else 1)
