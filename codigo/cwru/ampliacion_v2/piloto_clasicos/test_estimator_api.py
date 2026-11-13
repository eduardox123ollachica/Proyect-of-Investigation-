"""Comprueba configuraciones con sklearn sin fit de RF o SVM."""
import importlib.util
import unittest
from contrato_piloto import build_jobs
from test_contrato_piloto import protocol


@unittest.skipUnless(importlib.util.find_spec('sklearn') is not None,'Requiere sklearn del Python de usuario')
class EstimatorAPITests(unittest.TestCase):
    def test_all_rf_configurations_are_valid(self):
        from sklearn.ensemble import RandomForestClassifier
        for job in build_jobs(protocol()):
            if job['family']=='RF':
                model=RandomForestClassifier(**job['params'])
                model._validate_params()
                self.assertEqual(model.n_jobs,1)

    def test_all_svm_configurations_are_valid(self):
        from sklearn.svm import SVC
        for job in build_jobs(protocol()):
            if job['family']=='SVM':
                model=SVC(**job['params'])
                model._validate_params()
                self.assertFalse(model.probability)


if __name__=='__main__':unittest.main(verbosity=2)
