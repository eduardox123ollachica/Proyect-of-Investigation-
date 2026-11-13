"""Suite sin TensorFlow ni ajustes CWRU. --write-report crea informe nuevo."""
import argparse
import ast
from pathlib import Path
import unittest

from ejecutar_representantes import PREP, SOURCE, ROOT, source_hashes, versions, write


def main(write_report=False):
    for path in SOURCE.glob('*.py'):
        ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
    suite=unittest.defaultTestLoader.discover(str(SOURCE),pattern='test_*.py')
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    report=dict(status='PASS' if result.wasSuccessful() and not result.skipped else 'FAIL',
        tests_run=result.testsRun, failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),
        versions=versions(),source_sha256=source_hashes(),source_AST_passed=True,
        CWRU_fits=0,TensorFlow_imported=False,real_subprocess_training_launched=False,
        metadata_and_hash_validation=True)
    if write_report:
        PREP.mkdir(parents=True,exist_ok=True)
        write(PREP/'PRUEBAS_PREPARACION.json',report)
    return 0 if report['status']=='PASS' else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-report',action='store_true')
    raise SystemExit(main(parser.parse_args().write_report))
