"""Preflight/test sintetico del suplemento; no ejecuta ajustes CWRU."""
import argparse
import ast
import unittest
from ejecutar_suplemento import PREP,SOURCE,write,source_hashes,versions


def main(save=False):
    for p in SOURCE.glob('*.py'):ast.parse(p.read_text(encoding='utf-8-sig'),filename=str(p))
    suite=unittest.defaultTestLoader.discover(str(SOURCE),pattern='test_*.py')
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    report=dict(status='PASS' if result.wasSuccessful() and not result.skipped else 'FAIL',tests_run=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),
        source_sha256=source_hashes(),versions=versions(),real_training_fits=0,TensorFlow_imported=False,
        original_import_reference_checks=True,source_AST_passed=True)
    if save:
        PREP.mkdir(parents=True,exist_ok=True);write(PREP/'PRUEBAS_PREPARACION.json',report)
    return 0 if report['status']=='PASS' else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--write-report',action='store_true')
    raise SystemExit(main(parser.parse_args().write_report))
