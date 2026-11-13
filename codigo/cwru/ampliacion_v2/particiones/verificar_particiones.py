"""Verifica integridad y semántica de manifiestos, sin reescribir el paquete."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
import unittest
import test_particiones


def rows(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[4])
    p.add_argument('--directory',type=Path,required=True)
    a=p.parse_args();root=a.root.resolve();directory=(root/a.directory).resolve()
    outputs=rows(directory/'MANIFEST_SHA256.csv');inputs=rows(directory/'entradas_sha256.csv')
    for manifest,base in ((outputs,directory),(inputs,root)):
        for r in manifest:
            target=(base/r['path']).resolve()
            if not target.is_relative_to(base) or not target.is_file() or sha(target)!=r['sha256']:
                raise ValueError('Ruta ausente/alterada/fuera de alcance: '+r['path'])
    names={r['path'] for r in outputs}|{'MANIFEST_SHA256.csv'}
    if names!={f.relative_to(directory).as_posix() for f in directory.rglob('*') if f.is_file()}:
        raise ValueError('Salidas fuera del manifiesto')
    index=rows(directory/'indice_ventanas_fuente.csv')
    if [int(r['global_window_id']) for r in index]!=list(range(1380)):raise ValueError('Índice incompleto/no contiguo')
    global_index={int(r['global_window_id']):r for r in index}
    outer=rows(directory/'manifiesto_registros_externo.csv');inner=rows(directory/'manifiesto_registros_interno.csv')
    ow=rows(directory/'indices_ventanas_externo.csv');iw=rows(directory/'indices_ventanas_interno.csv')
    if (len(outer),len(inner),len(ow),len(iw))!=(112,336,5520,16560):raise ValueError('Tamaños inesperados')
    seen=[]
    for held in range(4):
        record_rows=[r for r in outer if int(r['outer_load_hp'])==held]
        if len({r['file_id'] for r in record_rows})!=28:raise ValueError('MAT repetidos/faltantes')
        for r in record_rows:
            if r['role']!=('outer_test' if int(r['load_hp'])==held else 'development'):raise ValueError('Papel externo inválido')
        role_map={r['file_id']:r['role'] for r in record_rows}
        windows=[r for r in ow if int(r['outer_load_hp'])==held]
        if sorted(int(r['global_window_id']) for r in windows)!=list(range(1380)):raise ValueError('Cobertura externa incorrecta')
        for r in windows:
            original=global_index[int(r['global_window_id'])]
            if r['file_id']!=original['file_id'] or r['role']!=role_map[original['file_id']]:raise ValueError('Canales/ventanas de un MAT separados')
            if r['role']=='outer_test':seen.append(int(r['global_window_id']))
        for valid in [l for l in range(4) if l!=held]:
            records=[r for r in inner if int(r['outer_load_hp'])==held and int(r['inner_validation_load_hp'])==valid]
            if len({r['file_id'] for r in records})!=28:raise ValueError('MAT internos repetidos/faltantes')
            counts=Counter(r['role'] for r in records)
            if counts!=Counter({'inner_train':14,'inner_validation':7,'outer_excluded':7}):raise ValueError('Papeles internos incompletos')
            for r in records:
                load=int(r['load_hp']);expected='outer_excluded' if load==held else 'inner_validation' if load==valid else 'inner_train'
                if r['role']!=expected:raise ValueError('Contaminación interna/externa')
            mapping={r['file_id']:r['role'] for r in records}
            windows=[r for r in iw if int(r['outer_load_hp'])==held and int(r['inner_validation_load_hp'])==valid]
            if sorted(int(r['global_window_id']) for r in windows)!=list(range(1380)):raise ValueError('Cobertura interna incorrecta')
            for r in windows:
                original=global_index[int(r['global_window_id'])]
                if r['file_id']!=original['file_id'] or r['role']!=mapping[original['file_id']]:raise ValueError('Ventana interna mal asignada')
    if sorted(seen)!=list(range(1380)):raise ValueError('Cada ventana debe estar exactamente una vez en prueba externa')
    if any(r['status']!='PASS' for r in rows(directory/'pruebas.csv')):raise ValueError('Pruebas de datos no aprobadas')
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(test_particiones))
    if not result.wasSuccessful():raise SystemExit(1)
    print(json.dumps(dict(status='VALIDADO',outputs=len(outputs),inputs=len(inputs),outer_folds=4,
                         inner_folds=12,unique_windows=1380,unit_tests=result.testsRun,
                         model_training_performed=False),ensure_ascii=True,indent=2))


if __name__=='__main__':main()
