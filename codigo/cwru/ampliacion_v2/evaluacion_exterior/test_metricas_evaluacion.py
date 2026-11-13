"""Pruebas sintéticas de métricas; no datos, modelos o inferencias CWRU."""
from __future__ import annotations
import copy
import math
import random
import sys
import unittest

import numpy as np

import metricas_evaluacion as metrics


def synthetic_rows(procedure='baseline_F14',precision='INT8',seeds=metrics.OUTER_SEEDS,offset=0.):
    values=[]
    for outer in range(4):
        for index,seed in enumerate(seeds):
            score=.2*(outer+1)+index*.001+offset
            support=10000 if outer==0 else outer
            values.append(dict(procedure=procedure,precision=precision,outer_load_hp=outer,seed=seed,
                accuracy=score,macro_f1=score,
                per_class=[dict(label=label,support=support,precision=score,recall=score,f1=score)
                    for label in range(3)]))
    return values


def quantization_fixture():
    z=np.asarray([[-128.,127.],[-128.6,127.6],[-128.5,126.5],[-127.5,127.5]],dtype=np.float32)
    q=np.asarray([[-128,127],[-128,127],[-128,126],[-128,127]],dtype=np.int8)
    output=np.asarray([[-128,127,-128],[0,0,-128],[-128,-128,127],[127,-128,-128]],dtype=np.int8)
    integer=np.multiply(np.subtract(output.astype(np.float32),np.float32(-128),dtype=np.float32),
        np.float32(1/256),dtype=np.float32)
    fp=np.asarray([[.9,.05,.05],[.1,.8,.1],[.1,.2,.7],[.2,.7,.1]],dtype=np.float32)
    return dict(z=z,input_int8=q,raw_int8_outputs=output,fp_probs=fp,int_probs=integer,
        y=np.asarray([0,1,2,0],dtype=np.int64),record_ids=['N_a','IR_a','OR_a','N_b'],
        inputq=(1.,0),outputq=(1/256,-128),global_window_ids=[100,101,102,103])


class SupervisedMetrics(unittest.TestCase):
    def test_known_confusion_precision_recall_f1_and_true_class_mat_recall(self):
        result=metrics.supervised([0,0,1,1,2,2],[0,1,1,2,2,2],['N','N','IR','IR','OR','OR'])
        self.assertEqual(result['confusion_matrix'],[[1,1,0],[0,1,1],[0,0,2]])
        self.assertAlmostEqual(result['accuracy'],2/3)
        self.assertAlmostEqual(result['macro_f1'],(2/3+.5+.8)/3)
        self.assertEqual(result['per_class'][0],dict(label=0,precision=1.,recall=.5,f1=2/3,support=2))
        per_record={r['record_id']:r for r in result['per_record_true_class_recall']}
        self.assertEqual([per_record[k]['true_class_recall'] for k in ('N','IR','OR')],[.5,.5,1.])
        self.assertEqual(per_record['IR']['predicted_class_counts'],[0,1,1])

    def test_single_class_mat_uses_true_class_recall_without_mat_macro_f1(self):
        result=metrics.supervised([1,1],[1,1],['one','one'])
        self.assertEqual(result['accuracy'],1.)
        self.assertEqual(result['macro_f1'],1/3)
        self.assertEqual(result['per_record_true_class_recall'][0]['true_class_recall'],1.)
        self.assertNotIn('macro_f1',result['per_record_true_class_recall'][0])

    def test_absent_class_zero_denominators_are_zero(self):
        result=metrics.supervised([0,0],[0,0],['N','N'])
        for label in (1,2):
            self.assertEqual(result['per_class'][label],dict(label=label,precision=0.,recall=0.,f1=0.,support=0))

    def test_invalid_labels_records_shapes_and_mixed_mat_classes_are_rejected(self):
        cases=[([True,False],[0,1],['a','b']),([0,True],[0,1],['a','b']),
            ([0,1.],[0,1],['a','b']),([0,3],[0,1],['a','b']),([],[],[]),
            ([0,1],[0],['a','b']),([0,1],[0,1],['same','same']),
            ([0,1],[0,1],['a',True]),([0,1],[0,1],['a',' bad '])]
        for y,pred,records in cases:
            with self.subTest(y=y,pred=pred,records=records),self.assertRaises(ValueError):
                metrics.supervised(y,pred,records)

    def test_row_order_does_not_change_metrics_or_per_mat_results(self):
        y=np.asarray([0,0,1,1,2,2],dtype=np.int64);pred=np.asarray([0,1,1,2,2,2],dtype=np.int64)
        records=np.asarray(['N','N','IR','IR','OR','OR'])
        ix=[5,2,0,4,1,3]
        self.assertEqual(metrics.supervised(y,pred,records),metrics.supervised(y[ix],pred[ix],records[ix]))


class DetectorMetrics(unittest.TestCase):
    def test_mat_equal_weighting_differs_from_pooled_window_weighting(self):
        y=np.asarray([0]*5+[1]*3+[2]*2,dtype=np.int64)
        rejected=np.asarray([False,True,True,True,False,True,False,True,False,True],dtype=np.bool_)
        records=['N_big']*4+['N_small']+['IR_a']*2+['IR_b']+['OR']*2
        result=metrics.detector(y,rejected,records)
        self.assertEqual(result['normal_spec'],.625)
        self.assertNotEqual(result['normal_spec'],2/5)
        self.assertEqual(result['ir_recall'],.75)
        self.assertEqual(result['or_recall'],.5)
        self.assertEqual(result['balanced_accuracy'],.625)

    def test_detector_requires_boolean_rejections_and_all_classes(self):
        with self.assertRaises(ValueError):metrics.detector([0,1,2],[0,1,1],['N','IR','OR'])
        with self.assertRaises(ValueError):metrics.detector([0,1],np.asarray([False,True]),['N','IR'])


class QuantizationMetrics(unittest.TestCase):
    def test_clipping_is_counted_after_rounding_and_distinguished_from_endpoints(self):
        result=metrics.quantization_report(**quantization_fixture())
        self.assertEqual(result['input_clipped_feature_values'],3)
        self.assertEqual(result['input_clipped_by_feature'],[1,2])
        self.assertEqual(result['input_clipped_by_window'],[0,2,0,1])
        self.assertEqual(result['input_clipped_window_count'],2)
        self.assertEqual(result['input_clipped_global_window_ids'],[101,103])
        self.assertEqual(result['input_endpoint_value_count'],7)
        self.assertEqual(result['input_endpoint_values_without_clipping'],4)
        self.assertEqual(result['output_endpoint_value_count'],10)
        self.assertEqual(result['input_clipped_feature_value_fraction'],3/8)

    def test_tied_argmax_uses_lowest_class_and_differences_keep_sign_and_ids(self):
        result=metrics.quantization_report(**quantization_fixture())
        self.assertEqual(result['INT8_output_tied_argmax_window_count'],1)
        self.assertEqual(result['class_changes'],3)
        self.assertEqual([r['global_window_id'] for r in result['class_change_rows']],[100,101,103])
        self.assertEqual(result['class_change_rows'][1]['INT8_prediction'],0)
        self.assertEqual(result['FP32_correct_INT8_incorrect'],2)
        self.assertEqual(result['FP32_incorrect_INT8_correct'],1)
        self.assertEqual(result['INT8_minus_FP32_accuracy_pp'],-25.)
        self.assertAlmostEqual(result['INT8_minus_FP32_macro_f1_pp'],100*(.5-7/9))
        self.assertTrue(result['no_statistical_noninferiority_claim'])

    def test_quantization_must_match_actual_int8_inputs_and_dequantized_outputs(self):
        changed=quantization_fixture();changed['input_int8']=changed['input_int8'].copy();changed['input_int8'][0,0]=-127
        with self.assertRaises(ValueError):metrics.quantization_report(**changed)
        changed=quantization_fixture();changed['int_probs']=changed['int_probs'].copy();changed['int_probs'][0,0]=.1
        with self.assertRaises(ValueError):metrics.quantization_report(**changed)

    def test_nonfinite_wrong_dtypes_bad_qparams_or_duplicate_ids_are_rejected(self):
        changes=[('z',np.ones((4,2),dtype=np.float64)),('z',np.full((4,2),np.nan,dtype=np.float32)),
            ('input_int8',np.ones((4,2),dtype=np.int16)),('raw_int8_outputs',np.ones((4,3),dtype=np.float32)),
            ('fp_probs',np.full((4,3),np.inf,dtype=np.float32)),('inputq',(0.,0)),('inputq',(1.,True)),
            ('outputq',(float('nan'),0)),('outputq',(1.,128)),('global_window_ids',[100,100,102,103]),
            ('global_window_ids',[100,101,True,103])]
        for key,value in changes:
            kwargs=quantization_fixture();kwargs[key]=value
            with self.subTest(key=key,value=str(value)),self.assertRaises(ValueError):metrics.quantization_report(**kwargs)

    def test_report_does_not_mutate_arrays_and_ids_are_optional(self):
        kwargs=quantization_fixture();before={k:v.copy() for k,v in kwargs.items() if isinstance(v,np.ndarray)}
        result=metrics.quantization_report(**kwargs)
        for key,array in before.items():self.assertTrue(np.array_equal(array,kwargs[key]))
        kwargs['global_window_ids']=None
        result=metrics.quantization_report(**kwargs)
        self.assertIsNone(result['input_clipped_global_window_ids'])
        self.assertIsNone(result['class_change_rows'][0]['global_window_id'])


class AggregationMetrics(unittest.TestCase):
    def test_equal_load_and_seed_weighting_is_not_support_weighting(self):
        result=metrics.aggregate_rows(synthetic_rows())
        group=result['groups'][0]
        self.assertEqual(group['cells'],40)
        self.assertAlmostEqual(group['mean_macro_f1'],.5045)
        self.assertAlmostEqual(group['worst_load_mean_macro_f1'],.2045)
        self.assertAlmostEqual(group['minimum_cell_macro_f1'],.2)
        self.assertAlmostEqual(group['maximum_cell_macro_f1'],.809)
        self.assertAlmostEqual(group['per_seed'][0]['mean_macro_f1_across_loads'],.5)
        self.assertAlmostEqual(group['per_seed'][-1]['mean_macro_f1_across_loads'],.509)
        self.assertEqual(group['macro_f1_of_four_load_means_across_seeds']['ddof'],1)
        self.assertEqual(group['macro_f1_of_four_load_means_across_seeds']['count'],10)
        self.assertFalse(result['window_count_weighting_used'])
        self.assertFalse(result['probability_ensemble_used'])
        self.assertFalse(result['confidence_intervals_computed'])

    def test_svm_is_four_deterministic_rows_and_rf_is_forty_rows(self):
        rows=synthetic_rows('RF','native')+synthetic_rows('SVM','native',(None,))
        groups={row['procedure']:row for row in metrics.aggregate_rows(rows)['groups']}
        self.assertEqual(groups['SVM']['cells'],4)
        self.assertEqual(groups['SVM']['seeds'],[None])
        self.assertTrue(groups['SVM']['deterministic_single_fit_per_load'])
        self.assertIsNone(groups['SVM']['macro_f1_of_four_load_means_across_seeds']['sample_standard_deviation'])
        self.assertEqual(groups['RF']['cells'],40)
        self.assertFalse(groups['RF']['deterministic_single_fit_per_load'])
        with self.assertRaises(ValueError):metrics.aggregate_rows(synthetic_rows('SVM','native'))
        with self.assertRaises(ValueError):metrics.aggregate_rows(synthetic_rows('RF','native',(None,)))

    def test_missing_duplicate_foreign_or_mismatched_support_cells_are_rejected(self):
        rows=synthetic_rows()
        variants=[rows[:-1],rows+[copy.deepcopy(rows[0])]]
        for key,value in [('outer_load_hp',True),('seed',44.5),('macro_f1',float('nan')),
            ('accuracy',True),('precision','native')]:
            variant=copy.deepcopy(rows);variant[0][key]=value;variants.append(variant)
        variant=copy.deepcopy(rows);variant[0]['per_class'][0]['support']+=1;variants.append(variant)
        variant=copy.deepcopy(rows);variant[0]['per_class'][0]['f1']=.9;variants.append(variant)
        for index,variant in enumerate(variants):
            with self.subTest(index=index),self.assertRaises(ValueError):metrics.aggregate_rows(variant)

    def test_permutation_preserves_full_summaries_and_inputs(self):
        rows=synthetic_rows()+synthetic_rows('RF','native')
        before=copy.deepcopy(rows);expected=metrics.aggregate_rows(rows)
        shuffled=copy.deepcopy(rows);random.Random(71).shuffle(shuffled)
        self.assertEqual(metrics.aggregate_rows(shuffled),expected)
        self.assertEqual(rows,before)

    def test_paired_differences_match_exact_load_and_seed_without_confidence_claim(self):
        rows=synthetic_rows('baseline_F14')+synthetic_rows('performance_only',offset=.05)
        result=metrics.paired_differences(rows,'performance_only','baseline_F14')
        self.assertEqual(result['pairs'],40)
        self.assertAlmostEqual(result['mean_macro_f1_difference_pp'],5.)
        self.assertEqual(len(result['per_seed']),10)
        self.assertEqual(len(result['per_load']),4)
        self.assertFalse(result['confidence_interval_computed'])
        self.assertFalse(result['p_value_computed'])
        self.assertFalse(result['noninferiority_claimed'])
        self.assertFalse(result['physical_acquisition_independence_claimed'])

    def test_pairs_cannot_invent_seed_replicates_for_deterministic_comparator(self):
        rows=synthetic_rows('A')+synthetic_rows('B',seeds=(None,))
        with self.assertRaises(ValueError):metrics.paired_differences(rows,'A','B')
        with self.assertRaises(ValueError):metrics.paired_differences(synthetic_rows('A'),'A','A')
        with self.assertRaises(ValueError):metrics.paired_differences(synthetic_rows('A'),'A','missing')

    def test_deterministic_pairs_are_allowed_without_replication(self):
        rows=synthetic_rows('A',seeds=(None,))+synthetic_rows('B',seeds=(None,),offset=.025)
        result=metrics.paired_differences(rows,'A','B')
        self.assertEqual(result['pairs'],4)
        self.assertAlmostEqual(result['mean_macro_f1_difference_pp'],-2.5)
        self.assertEqual(len(result['per_seed']),1)

    def test_module_does_not_import_training_or_inference_runtimes(self):
        self.assertNotIn('tensorflow',sys.modules)
        self.assertNotIn('sklearn',sys.modules)
        self.assertNotIn('joblib',sys.modules)


if __name__=='__main__':unittest.main(verbosity=2)
