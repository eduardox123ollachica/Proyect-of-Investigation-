"""Pruebas de selección pura; sin ajustes, inferencias o evaluación exterior."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import contextmanager
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'codigo/cwru/ampliacion_v2/piloto_clasicos'))
sys.path.insert(0,str(ROOT/'codigo/cwru/ampliacion_v2/protocolo'))
from contrato_piloto import build_jobs as pilot_jobs
import reglas_v2 as original_rules
from seleccion_clasicos import select_classical
import seleccionar as selector


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def actual_inputs():
    return dict(protocol=read(ROOT/'articulo/ampliacion_cwru/metodologia/protocolo_v2_2026-10-25/protocolo_reglas.json'),
        dnn_rows=read(ROOT/'articulo/ampliacion_cwru/metodologia/cierre_matriz_dnn_interna_v2_2026-10-26/MATRIZ_INTERNA_288.json'),
        detector_rows=read(ROOT/'resultados/cwru/ampliacion_v2/detector_interno_v2_2026-10-26/20261027T031125_681822Z/MATRIZ_DETECTOR_96.json'),
        classical_rows=read(ROOT/'articulo/ampliacion_cwru/metodologia/cierre_clasicos_internos_v2_2026-10-27/MATRIZ_CLASICOS_384.json'),
        cost_table=read(ROOT/'despliegue_stm32/costo_total_single_rounding_v2_180mhz_2026-10-26/COSTOS_TOTALES_SR_VALIDADOS.json'))


def synthetic_classical(protocol,family='RF',outer=3):
    rows=[]
    for template in pilot_jobs(protocol):
        if template['family'] != family:continue
        for load in range(4):
            if load==outer:continue
            for seed in ([11,22,33] if family=='RF' else [None]):
                params=copy.deepcopy(template['params'])
                if family=='RF':params['random_state']=seed
                rows.append(dict(job_id=f'synthetic_{family}_{template["config_index"]}_{load}_{seed}',
                    family=family,config_index=template['config_index'],seed=seed,params=params,
                    outer_load_hp=outer,inner_validation_load_hp=load,macro_f1=.8,validation_windows=10))
    return rows


def profile(rows,index,values):
    for row in rows:
        if row['config_index']==index:
            row['macro_f1']=values[row['inner_validation_load_hp']] if isinstance(values,dict) else values


def original_rule_inputs(inputs,outer):
    dnn=[dict(candidate=r['candidate'],inner_validation_load=r['inner_validation_load_hp'],
        seed=r['seed'],status='PASS',macro_f1_int8=r['macro_f1_INT8'])
        for r in inputs['dnn_rows'] if r['outer_load_hp']==outer]
    detector=[dict(candidate=r['candidate'],inner_validation_load=r['inner_validation_load'],
        status=r['status'],normal_spec=r['normal_spec'],ir_recall=r['ir_recall'],or_recall=r['or_recall'])
        for r in inputs['detector_rows'] if r['outer_load_hp']==outer]
    return dnn,detector,inputs['cost_table']['costs_by_outer'][str(outer)]


@contextmanager
def fixture():
    base=(ROOT/'tmp').resolve();base.mkdir(exist_ok=True)
    folder=Path(tempfile.mkdtemp(prefix='freeze_selection_test_',dir=base)).resolve()
    try:yield folder
    finally:
        if folder.parent != base or not folder.name.startswith('freeze_selection_test_'):
            raise ValueError('Fixture fuera del temporal autorizado')
        shutil.rmtree(folder)


class ClassicalSelectionGuards(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inputs=actual_inputs()
        cls.protocol=cls.inputs['protocol']

    def test_all_actual_classical_decisions_match_independent_expected_grid_choices(self):
        for family,expected in [('RF',[5,5,0,4]),('SVM',[5,7,4,5])]:
            for outer,config in enumerate(expected):
                with self.subTest(family=family,outer=outer):
                    result=select_classical(self.inputs['classical_rows'],family,outer,self.protocol)
                    self.assertEqual(result['selected_config_index'],config)
                    self.assertNotIn('random_state',result['selected_params'])

    def test_missing_or_duplicated_cell_is_not_omitted_from_average(self):
        rows=synthetic_classical(self.protocol)
        for changed in (rows[:-1],rows+[copy.deepcopy(rows[-1])]):
            with self.assertRaises(ValueError):select_classical(changed,'RF',3,self.protocol)

    def test_invalid_seed_configuration_parameters_and_outer_overlap_rejected(self):
        for field,value in [('seed',44),('seed',True),('config_index',8),('config_index',True),
            ('inner_validation_load_hp',3),('macro_f1',float('nan')),('macro_f1',True),('macro_f1',1.01)]:
            rows=synthetic_classical(self.protocol);rows[0][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):
                select_classical(rows,'RF',3,self.protocol)
        rows=synthetic_classical(self.protocol);rows[0]['params']['class_weight']=None
        with self.assertRaises(ValueError):select_classical(rows,'RF',3,self.protocol)
        rows=synthetic_classical(self.protocol,'SVM');rows[0]['seed']=11
        with self.assertRaises(ValueError):select_classical(rows,'SVM',3,self.protocol)

    def test_RF_uses_all_three_seeds_instead_of_choosing_favorable_seed(self):
        rows=synthetic_classical(self.protocol)
        for row in rows:
            if row['config_index']==0:row['macro_f1']=.99 if row['seed']==22 else .80
        profile(rows,1,.90)
        self.assertEqual(select_classical(rows,'RF',3,self.protocol)['selected_config_index'],1)

    def test_equal_validation_load_weight_is_not_pooled_by_window_count(self):
        rows=synthetic_classical(self.protocol)
        profile(rows,0,{0:.80,1:.95,2:.95});profile(rows,1,.91)
        for row in rows:row['validation_windows']=1 if row['inner_validation_load_hp']==0 else 10000
        self.assertEqual(select_classical(rows,'RF',3,self.protocol)['selected_config_index'],1)

    def test_numeric_ties_use_maximum_envelope_not_transitive_pairwise_chain(self):
        rows=synthetic_classical(self.protocol)
        profile(rows,0,.9);profile(rows,1,.9+.75e-12);profile(rows,2,.9+1.5e-12)
        self.assertEqual(select_classical(rows,'RF',3,self.protocol)['selected_config_index'],1)

    def test_worst_load_then_grid_order_break_real_tie(self):
        rows=synthetic_classical(self.protocol)
        profile(rows,0,{0:.8,1:.95,2:.95})
        profile(rows,1,{0:.85,1:.925,2:.925});profile(rows,2,{0:.85,1:.925,2:.925})
        self.assertEqual(select_classical(rows,'RF',3,self.protocol)['selected_config_index'],1)

    def test_input_row_permutation_cannot_change_classical_choice_or_summary(self):
        rows=copy.deepcopy(self.inputs['classical_rows']);random.Random(810).shuffle(rows)
        for family in ('RF','SVM'):
            for outer in range(4):
                self.assertEqual(select_classical(rows,family,outer,self.protocol),
                    select_classical(self.inputs['classical_rows'],family,outer,self.protocol))

    def test_no_training_or_runtime_ML_imported_by_pure_selection(self):
        select_classical(self.inputs['classical_rows'],'SVM',0,self.protocol)
        self.assertNotIn('tensorflow',sys.modules)
        self.assertNotIn('sklearn',sys.modules)


class ExistingRuleIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inputs=actual_inputs()
        cls.protocol=cls.inputs['protocol']

    def test_actual_DNN_performance_and_cost_choices_follow_frozen_rules(self):
        for outer,(performance,cost) in enumerate(zip(['F14','FE7','T6','NF12'],['T6','FE7','T6','F14'])):
            dnn,detector,costs=original_rule_inputs(self.inputs,outer)
            inner=[load for load in range(4) if load!=outer]
            p=original_rules.select_performance(dnn,detector,self.protocol['candidate_order'],inner,[11,22,33])
            c=original_rules.select_cost(dnn,detector,self.protocol['candidate_order'],inner,[11,22,33],costs,outer_load=outer)
            with self.subTest(outer=outer):
                self.assertEqual(p['selected_candidate'],performance)
                self.assertEqual(c['selected_candidate'],cost)
                self.assertEqual(c['goal_met'],outer!=3)

    def test_outer3_fallback_retains_failed_predictive_margin_instead_of_claiming_preservation(self):
        dnn,detector,costs=original_rule_inputs(self.inputs,3)
        c=original_rules.select_cost(dnn,detector,self.protocol['candidate_order'],[0,1,2],[11,22,33],costs,outer_load=3)
        self.assertEqual(c['selected_candidate'],'F14')
        self.assertFalse(c['goal_met'])
        self.assertFalse(c['selected_satisfies_predictive_margin'])
        self.assertEqual(c['target_eligible_candidates'],[])
        self.assertEqual(c['predictive_eligible_candidates'],['NF12'])
        self.assertGreater(c['macro_f1_threshold'],c['candidate_summaries']['F14']['mean_macro_f1_int8'])

    def test_final_epoch_median_is_from_nine_internal_checkpoints_for_that_mask(self):
        for outer,mask,expected in [(0,'F14',45),(0,'T6',47),(1,'F14',14),(1,'FE7',27),
            (2,'F14',19),(2,'T6',32),(3,'F14',9),(3,'NF12',47)]:
            epochs=[r['best_epoch'] for r in self.inputs['dnn_rows'] if r['outer_load_hp']==outer and r['candidate']==mask]
            with self.subTest(outer=outer,mask=mask):
                self.assertEqual(len(epochs),9)
                self.assertEqual(original_rules.final_epoch_count(epochs),expected)

    def test_unknown_or_proxy_costs_and_missing_internal_cells_are_rejected(self):
        dnn,detector,costs=original_rule_inputs(self.inputs,0)
        incomplete=dict(costs);del incomplete['FE7']
        proxy=copy.deepcopy(costs);proxy['T6']['is_proxy']=True
        wrong_scope=copy.deepcopy(costs);wrong_scope['T6']['benchmark_loads']=[0,3]
        for current_dnn,current_detector,current_costs in [(dnn,detector,incomplete),(dnn,detector,proxy),
            (dnn,detector,wrong_scope),(dnn[:-1],detector,costs),(dnn,detector[:-1],costs)]:
            with self.assertRaises(ValueError):
                original_rules.select_cost(current_dnn,current_detector,self.protocol['candidate_order'],[1,2,3],
                    [11,22,33],current_costs,outer_load=0)


class FrozenDecisionIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inputs=selector.load_inputs()
        cls.decisions=selector.build_decisions(cls.inputs)

    def test_wrapper_reuses_original_DNN_rule_functions_literally(self):
        self.assertIs(selector.select_performance,original_rules.select_performance)
        self.assertIs(selector.select_cost,original_rules.select_cost)
        self.assertIs(selector.final_epoch_count,original_rules.final_epoch_count)

    def test_fresh_isolated_process_imports_selector_without_preloaded_rules(self):
        code='''import json,sys
assert 'reglas_v2' not in sys.modules
sys.path.insert(0,sys.argv[1])
import seleccionar
inputs=seleccionar.load_inputs()
decisions=seleccionar.build_decisions(inputs)
plan=seleccionar.build_refit_plan(decisions,inputs['protocol'])
assert 'tensorflow' not in sys.modules and 'sklearn' not in sys.modules
assert not plan['execution_enabled'] and not plan['outer_prediction_release_enabled']
print(json.dumps({'scenarios':len(decisions),'counts':plan['counts']}))
'''
        result=subprocess.run([sys.executable,'-I','-B','-c',code,str(selector.SOURCE)],
            check=True,capture_output=True,text=True,timeout=60,cwd=ROOT)
        self.assertEqual(json.loads(result.stdout),dict(scenarios=4,
            counts=dict(DNN_before_dedup=120,DNN_unique=80,RF=40,SVM=4,detector_unique=7)))

    def test_boolean_roles_cannot_alias_integer_roles_in_any_consumed_grid(self):
        cases=[('dnn_rows','outer_load_hp'),('dnn_rows','inner_validation_load_hp'),('dnn_rows','seed'),
            ('detector_rows','outer_load_hp'),('detector_rows','inner_validation_load'),
            ('classical_rows','outer_load_hp'),('classical_rows','inner_validation_load_hp'),
            ('classical_rows','config_index'),('cost_table','outer_load')]
        for grid,field in cases:
            changed=copy.deepcopy(self.inputs)
            rows=changed[grid]['rows'] if grid=='cost_table' else changed[grid]
            rows[0][field]=bool(rows[0][field])
            with self.subTest(grid=grid,field=field),self.assertRaisesRegex(ValueError,'enteros'):
                selector.validate_grids(changed)

    def test_four_decisions_preserve_choices_and_cost_fallback_limitations(self):
        self.assertEqual([d['outer_load_hp'] for d in self.decisions],[0,1,2,3])
        self.assertEqual([d['performance']['choice']['candidate'] for d in self.decisions],['F14','FE7','T6','NF12'])
        self.assertEqual([d['cost']['choice']['candidate'] for d in self.decisions],['T6','FE7','T6','F14'])
        self.assertEqual(sum(d['cost']['goal_met'] for d in self.decisions),3)
        self.assertFalse(self.decisions[3]['cost']['selected_satisfies_predictive_margin'])
        self.assertAlmostEqual(self.decisions[3]['cost']['comparison']['selected_F1_minus_performance_pp'],-7.099467221134997)
        for d in self.decisions:
            self.assertTrue(d['selected_once_for_all_outer_seeds'])
            self.assertTrue(d['no_outer_predictions_used'])

    def test_complete_decisions_are_deterministic_under_input_permutation(self):
        shuffled=copy.deepcopy(self.inputs)
        rng=random.Random(1275)
        for key in ('dnn_rows','detector_rows','classical_rows'):rng.shuffle(shuffled[key])
        rng.shuffle(shuffled['cost_table']['rows'])
        self.assertEqual(selector.build_decisions(shuffled),self.decisions)

    def test_incomplete_or_foreign_grids_and_model_cost_mismatch_are_rejected(self):
        for key in ('dnn_rows','detector_rows','classical_rows'):
            changed=copy.deepcopy(self.inputs);changed[key].pop()
            with self.subTest(key=key),self.assertRaises(ValueError):selector.build_decisions(changed)
        changed=copy.deepcopy(self.inputs);changed['dnn_rows'][0]['outer_load_hp']=1
        with self.assertRaises(ValueError):selector.build_decisions(changed)
        changed=copy.deepcopy(self.inputs)
        changed['cost_table']['rows'][0]['model_sha256']='0'*64
        with self.assertRaises(ValueError):selector.build_decisions(changed)

    def test_derived_refits_deduplicate_only_identical_training_and_never_release_outer_predictions(self):
        plan=selector.build_refit_plan(self.decisions,self.inputs['protocol'])
        self.assertEqual(plan['counts'],dict(DNN_before_dedup=120,DNN_unique=80,RF=40,SVM=4,detector_unique=7))
        self.assertEqual(len(plan['DNN_unique_jobs']),80)
        self.assertEqual(len(plan['RF_jobs']),40)
        self.assertEqual(len(plan['SVM_jobs']),4)
        self.assertEqual(len(plan['detector_unique_scopes']),7)
        self.assertFalse(plan['execution_enabled'])
        self.assertFalse(plan['outer_prediction_release_enabled'])
        self.assertFalse(plan['outer_early_stopping_allowed'])
        self.assertFalse(plan['seed_ensembles'])
        self.assertEqual(plan['canonical_hardware_seed'],42)
        for outer in range(4):
            for family in ('DNN_unique_jobs','RF_jobs'):
                seeds={j['seed'] for j in plan[family] if j['outer_load_hp']==outer}
                self.assertEqual(seeds,set(self.inputs['protocol']['dnn']['outer_seeds']))
            self.assertEqual(sum(j['outer_load_hp']==outer for j in plan['SVM_jobs']),1)
        identities=[(j['outer_load_hp'],j['candidate'],j['epochs'],j['seed']) for j in plan['DNN_unique_jobs']]
        self.assertEqual(len(identities),len(set(identities)))
        for job in plan['DNN_unique_jobs']+plan['RF_jobs']+plan['SVM_jobs']+plan['detector_unique_scopes']:
            roles=job['roles'];outer=job['outer_load_hp']
            self.assertNotIn(outer,roles['training_loads'])
            self.assertTrue(set(roles['detector_fit_normal_loads']).isdisjoint(roles['detector_calibration_normal_loads']))
            self.assertEqual(roles['evaluation_loads'],[outer])
        for job in plan['RF_jobs']:
            self.assertEqual(job['params']['random_state'],job['seed'])
            selected=self.decisions[job['outer_load_hp']]['classical']['RF']
            self.assertEqual(job['config_index'],selected['selected_config_index'])
            self.assertEqual({k:v for k,v in job['params'].items() if k!='random_state'},selected['selected_params'])
        self.assertTrue(all(j['seed'] is None for j in plan['SVM_jobs']))
        o3=[j for j in plan['detector_unique_scopes'] if j['outer_load_hp']==3]
        self.assertEqual(len(o3),1)
        self.assertEqual(set(o3[0]['masks']),{'F14','NF12'})

    def test_authentication_rejects_changed_consumed_byte_or_manifest_seal(self):
        with fixture() as folder:
            artifact=folder/'matrix.json';artifact.write_bytes(b'{"valid":true}\n')
            manifest=folder/'MANIFEST.json'
            selector.io.write_manifest(manifest,directory=folder)
            selector.authenticate_input(manifest,[artifact])
            undeclared=folder/'other.json';undeclared.write_bytes(b'{}')
            with self.assertRaises(ValueError):selector.authenticate_input(manifest,[undeclared])
            artifact.write_bytes(b'{"valid":false}\n')
            with self.assertRaises(ValueError):selector.authenticate_input(manifest,[artifact])
            artifact.write_bytes(b'{"valid":true}\n')
            manifest.with_suffix('.sha256').write_text('0'*64+'  '+manifest.name+'\n',encoding='ascii')
            with self.assertRaises(ValueError):selector.authenticate_input(manifest,[artifact])

    def test_existing_selection_cannot_be_overwritten_or_recomputed(self):
        with fixture() as existing,patch.object(selector,'OUT',existing),\
             patch.object(selector,'load_inputs',side_effect=AssertionError('No reoptimizar una selección congelada')) as loader:
            with self.assertRaisesRegex(ValueError,'congeladas'):
                selector.freeze()
            loader.assert_not_called()

    def test_selection_does_not_mutate_inputs_or_engineering_margins(self):
        original=copy.deepcopy(self.inputs)
        selector.build_decisions(self.inputs)
        self.assertEqual(self.inputs,original)
        self.assertEqual(self.inputs['protocol']['selection']['DNN_F1_loss_margin'],.01)
        self.assertEqual(self.inputs['protocol']['selection']['minimum_total_cost_reduction_vs_F14'],.2)
        self.assertNotIn('sklearn',sys.modules)
        self.assertNotIn('tensorflow',sys.modules)


if __name__=='__main__':
    unittest.main(verbosity=2)
