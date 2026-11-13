"""Valida una auditoría guardada, sus entradas y las pruebas sintéticas."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import unittest
from audit_core import file_hash, verify_manifest
import test_auditoria_cwru


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[4])
    parser.add_argument('--directory',type=Path,required=True)
    args=parser.parse_args();root=args.root.resolve();out=(root/args.directory).resolve()
    rows=verify_manifest(out)
    listed={r['path'] for r in rows}|{'MANIFEST_SHA256.csv'}
    actual={p.relative_to(out).as_posix() for p in out.rglob('*') if p.is_file()}
    if listed!=actual:raise ValueError(f'Archivos no incluidos o ausentes: {listed ^ actual}')
    with (out/'entradas_sha256.csv').open(encoding='utf-8-sig',newline='') as f:inputs=list(csv.DictReader(f))
    for row in inputs:
        p=(root/row['path']).resolve()
        if not p.is_relative_to(root) or not p.is_file() or file_hash(p)!=row['sha256']:
            raise ValueError(f"Entrada ausente o alterada: {row['path']}")
    with (out/'registro_pruebas.csv').open(encoding='utf-8-sig',newline='') as f:checks=list(csv.DictReader(f))
    if any(r['status']=='FAIL' for r in checks):raise ValueError('La auditoría contiene FAIL')
    summary=json.loads((out/'resumen_auditoria.json').read_text(encoding='utf-8'))
    if not summary['original_inputs_unchanged']:raise ValueError('La auditoría detectó cambios en las entradas')
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(test_auditoria_cwru))
    if not result.wasSuccessful():raise SystemExit(1)
    print(json.dumps(dict(status='VALIDADO',outputs=len(rows),inputs=len(inputs),
                         synthetic_tests=result.testsRun,recorded_checks=len(checks),
                         warnings_retained=summary['test_status_counts']),ensure_ascii=True,indent=2))


if __name__=='__main__':main()
