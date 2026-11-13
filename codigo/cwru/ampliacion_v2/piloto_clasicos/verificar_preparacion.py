"""Registra pruebas sinteticas completas; no inicia el piloto CWRU."""
from __future__ import annotations
import ast
import datetime
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import platform
import sys
import unittest

ROOT=Path(__file__).resolve().parents[4]
SOURCE=Path(__file__).resolve().parent
OUT=ROOT/'articulo/ampliacion_cwru/metodologia/piloto_clasicos_2026-10-25/PRUEBAS_PREPARACION.json'


def main():
    sources=sorted(SOURCE.glob('*.py'))
    for path in sources:ast.parse(path.read_text(encoding='utf-8'),filename=str(path))
    suite=unittest.defaultTestLoader.discover(str(SOURCE),pattern='test_*.py')
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    complete=result.wasSuccessful() and len(result.skipped)==0
    report=dict(status='PASS' if complete else 'FAIL_OR_INCOMPLETE',
        generated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),tests_run=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),log=stream.getvalue(),
        source_AST_passed=True,source_sha256={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        environment=dict(python=platform.python_version(),numpy=importlib.metadata.version('numpy'),
                         sklearn=importlib.metadata.version('scikit-learn')),
        data='synthetic_only',real_RF_SVM_fits=0,source_NPY_MAT_arrays_opened=False,
        TensorFlow_imported=any(n=='tensorflow' or n.startswith('tensorflow.') for n in sys.modules),
        prior_bundled_run=dict(tests_run=39,failures=0,errors=0,skipped=7,
            reason='Bundled Python lacks sklearn; final report requires all synthetic tests without skips.'))
    with OUT.open('x',encoding='utf-8',newline='\n') as output:
        json.dump(report,output,indent=2,ensure_ascii=False);output.write('\n')
    print(stream.getvalue())
    print(json.dumps({k:report[k] for k in ('status','tests_run','failures','errors','skipped','real_RF_SVM_fits','TensorFlow_imported')}))
    return 0 if complete else 1


if __name__=='__main__':raise SystemExit(main())
