"""Materializa manifiestos V2 por carga, conservando originales y auditoría."""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
import sys
import unittest
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from contrato_particiones import (LOADS,LABELS,make_roles,validate_roles,validate_records,
                                 validate_config,guard_fit_scope)


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def csv_read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def csv_write(p,rows):
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)


def json_write(p,value):
    p.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def verify_audit(root,audit):
    outputs=csv_read(audit/'MANIFEST_SHA256.csv')
    for row in outputs:
        p=(audit/row['path']).resolve()
        if not p.is_relative_to(audit.resolve()) or not p.is_file() or sha(p)!=row['sha256']:
            raise ValueError('Salida de auditoría ausente o alterada: '+row['path'])
    inputs=csv_read(audit/'entradas_sha256.csv')
    for row in inputs:
        p=(root/row['path']).resolve()
        if not p.is_relative_to(root) or not p.is_file() or sha(p)!=row['sha256']:
            raise ValueError('Entrada histórica ausente o alterada: '+row['path'])
    return inputs,len(outputs)


def build_index(root,records,base):
    by_file={r['file']:r for r in records};rows=[]
    for split in ('train','validation','test'):
        metadata=csv_read(base/f'metadata_{split}.csv')
        features=np.load(base/'features_14'/f'X_features_{split}.npy',mmap_mode='r')
        labels=np.load(base/f'y_{split}.npy')
        feature_labels=np.load(base/'features_14'/f'y_{split}.npy')
        if features.shape!=(len(metadata),14) or len(labels)!=len(metadata) or not np.array_equal(labels,feature_labels):
            raise ValueError('Etiquetas/formas incompatibles con metadata: '+split)
        seen=Counter()
        for i,m in enumerate(metadata):
            r=by_file[m['file']];j=seen[m['file']];seen[m['file']]+=1
            for key in ('file_id','class','de_key','fe_key'):
                if str(r[key])!=str(m[key]):raise ValueError('Metadata incompatible en '+key)
            if int(m['load_hp'])!=int(r['load_hp']) or int(m['label'])!=LABELS[r['class']] or labels[i]!=int(m['label']):
                raise ValueError('Carga/clase/etiqueta incompatible')
            start=int(m['start_sample_12khz_valid'])
            if start!=j*2048 or int(m['end_sample_12khz_valid_exclusive'])!=start+4096:
                raise ValueError('Inicio/fin de ventana incompatible')
            rows.append(dict(global_window_id=len(rows),source_split_v1=split,source_row=i,
                         source_features=(base/'features_14'/f'X_features_{split}.npy').relative_to(root).as_posix(),
                         source_windows=(base/f'X_{split}.npy').relative_to(root).as_posix(),
                         file=m['file'],file_id=m['file_id'],load_hp=int(m['load_hp']),
                         **{'class':m['class']},label=int(m['label']),
                         window_index_in_file=j,start_sample_12khz_valid=start,
                         end_sample_12khz_valid_exclusive=start+4096,de_key=m['de_key'],fe_key=m['fe_key']))
        for file,n in seen.items():
            if n!=int(by_file[file]['expected_windows']):raise ValueError('Conteo por MAT distinto de auditoría')
    if len(rows)!=1380 or {r['file'] for r in rows}!=set(by_file):raise ValueError('Cobertura de ventanas incompleta')
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[4])
    parser.add_argument('--output',type=Path,default=Path('datos/cwru/particiones/ampliacion_v2/por_carga_2026-10-22'))
    args=parser.parse_args();root=args.root.resolve();out=(root/args.output).resolve()
    allowed=(root/'datos/cwru/particiones/ampliacion_v2').resolve()
    if out.exists() or not out.is_relative_to(allowed) or out==allowed:
        raise ValueError('La salida debe ser una subcarpeta NUEVA de particiones/ampliacion_v2')
    protocol_path=root/'articulo/ampliacion_cwru/metodologia/protocolo_v2_particiones_2026-10-22.json'
    protocol=json.loads(protocol_path.read_text(encoding='utf-8'))
    if not protocol['partition_generation_enabled'] or protocol['mass_training_enabled'] or protocol['outer_predictive_evaluation_enabled']:
        raise ValueError('Se requiere contrato sólo de preparación, sin ejecución predictiva')
    audit=root/protocol['audit']['directory'];inputs,audit_outputs=verify_audit(root,audit)
    records=csv_read(root/protocol['audit']['inventory']);validate_records(records)
    base=root/protocol['preprocessing']['reuse_directory']
    config=json.loads((base/'configuracion.json').read_text(encoding='utf-8'));validate_config(config)
    paths={root/r['path'] for r in inputs}
    paths.update([protocol_path,audit/'MANIFEST_SHA256.csv',audit/'inventario_archivos.csv',base/'configuracion.json'])
    paths.update(p for p in Path(__file__).resolve().parent.glob('*.py'))
    paths.add(protocol_path.with_name('ACTUALIZACION_DATOS_Y_PARTICIONES_2026-10-22.md'))
    hashes={p:sha(p) for p in sorted(paths)}
    index=build_index(root,records,base)
    outer_records=[];inner_records=[];outer_indices=[];inner_indices=[];summaries=[];ledger=[];checks=[]
    def record_rows(assigned,outer,inner=None):
        return [dict(outer_load_hp=outer,inner_validation_load_hp='' if inner is None else inner,
                     file_id=r['file_id'],file=r['file'],load_hp=int(r['load_hp']),
                     **{'class':r['class']},label=int(r['label']),role=r['role'],
                     windows=int(r['expected_windows']),de_key=r['de_key'],fe_key=r['fe_key'],
                     raw_path=r['path'],raw_sha256=r['sha256'],
                     physical_bearing_id_verified=False,
                     quality_note=r['literature_warning']) for r in assigned]
    def window_rows(assigned,outer,inner=None):
        roles={str(r['file_id']):r['role'] for r in assigned}
        return [dict(outer_load_hp=outer,inner_validation_load_hp='' if inner is None else inner,
                     global_window_id=r['global_window_id'],file_id=r['file_id'],role=roles[str(r['file_id'])]) for r in index]
    def add_summary(assigned,outer,inner=None):
        for role in sorted({r['role'] for r in assigned}):
            rr=[r for r in assigned if r['role']==role]
            summaries.append(dict(outer_load_hp=outer,inner_validation_load_hp='' if inner is None else inner,
                                  role=role,files=len(rr),windows=sum(int(r['expected_windows']) for r in rr),
                                  N=sum(int(r['expected_windows']) for r in rr if r['class']=='N'),
                                  IR=sum(int(r['expected_windows']) for r in rr if r['class']=='IR'),
                                  OR=sum(int(r['expected_windows']) for r in rr if r['class']=='OR')))
    for outer in LOADS:
        assigned=make_roles(records,outer);validate_roles(assigned,outer)
        outer_records.extend(record_rows(assigned,outer));outer_indices.extend(window_rows(assigned,outer));add_summary(assigned,outer)
        dev_files=[r['file_id'] for r in assigned if r['role']=='development']
        test_files=[r['file_id'] for r in assigned if r['role']=='outer_test']
        dev_indices=[r['global_window_id'] for r in index if r['file_id'] in dev_files]
        test_indices=[r['global_window_id'] for r in index if r['file_id'] in test_files]
        guard_fit_scope(index,dev_indices,dev_files)
        assert not(set(dev_indices)&set(test_indices)) and len(dev_indices)+len(test_indices)==1380
        ledger.append(dict(outer_load_hp=outer,stage='outer_refit',inner_validation_load_hp=None,
                           allowed_fit_file_ids=dev_files,forbidden_file_ids=test_files,
                           allowed_fit_rows=len(dev_indices),outer_test_rows=len(test_indices),
                           scaler_classweights_int8_fit_scope='development only; detector restricted further to N',
                           training_executed=False))
        checks.append(dict(id=f'PT_OUTER_{outer}',status='PASS',observed=f'{len(dev_files)}/{len(test_files)} MAT; {len(dev_indices)}/{len(test_indices)} ventanas',
                           criterion='Desarrollo/prueba disjuntos por MAT y ventana; cobertura total'))
        for inner in [x for x in LOADS if x!=outer]:
            assigned=make_roles(records,outer,inner);validate_roles(assigned,outer,inner)
            inner_records.extend(record_rows(assigned,outer,inner));inner_indices.extend(window_rows(assigned,outer,inner));add_summary(assigned,outer,inner)
            train_files=[r['file_id'] for r in assigned if r['role']=='inner_train']
            val_files=[r['file_id'] for r in assigned if r['role']=='inner_validation']
            train_indices=[r['global_window_id'] for r in index if r['file_id'] in train_files]
            guard_fit_scope(index,train_indices,train_files)
            for bad in [r['global_window_id'] for r in index if r['file_id'] in test_files+val_files][:1]:
                try:guard_fit_scope(index,[bad],train_files)
                except ValueError:pass
                else:raise AssertionError('Guarda no detectó contaminación')
            ledger.append(dict(outer_load_hp=outer,stage='inner_fit',inner_validation_load_hp=inner,
                               allowed_fit_file_ids=train_files,forbidden_file_ids=val_files+test_files,
                               allowed_fit_rows=len(train_indices),outer_test_rows=len(test_indices),
                               scaler_classweights_int8_fit_scope='inner_train only; detector restricted further to N',
                               training_executed=False))
            checks.append(dict(id=f'PT_INNER_{outer}_{inner}',status='PASS',observed=f'{len(train_files)}/{len(val_files)}/{len(test_files)} MAT',
                               criterion='Train/validation/excluido externo disjuntos; guarda rechaza fila no autorizada'))
    outer_test_ids=[r['global_window_id'] for r in outer_indices if r['role']=='outer_test']
    assert sorted(outer_test_ids)==list(range(1380))
    checks.append(dict(id='PT_COVERAGE',status='PASS',observed='1380/1380 una vez',criterion='Cada ventana figura exactamente una vez en la prueba externa rotatoria'))
    import test_particiones
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(test_particiones))
    if not result.wasSuccessful():raise RuntimeError(stream.getvalue())
    checks.append(dict(id='PT_UNIT',status='PASS',observed=f'{result.testsRun} pruebas',criterion='Casos positivos y negativos sintéticos aprobados'))
    changed=[p.relative_to(root).as_posix() for p,h in hashes.items() if sha(p)!=h]
    if changed:raise ValueError('Entradas modificadas: '+str(changed))
    checks.append(dict(id='PT_INTEGRITY',status='PASS',observed=f'{len(hashes)} entradas intactas; {audit_outputs} salidas auditoría comprobadas',criterion='SHA256 antes/después y sellos históricos válidos'))
    out.mkdir(parents=True)
    csv_write(out/'indice_ventanas_fuente.csv',index)
    csv_write(out/'manifiesto_registros_externo.csv',outer_records)
    csv_write(out/'manifiesto_registros_interno.csv',inner_records)
    csv_write(out/'indices_ventanas_externo.csv',outer_indices)
    csv_write(out/'indices_ventanas_interno.csv',inner_indices)
    csv_write(out/'resumen_particiones.csv',summaries)
    csv_write(out/'pruebas.csv',checks)
    csv_write(out/'entradas_sha256.csv',[dict(path=p.relative_to(root).as_posix(),sha256=h) for p,h in hashes.items()])
    json_write(out/'alcance_ajustes_por_pliegue.json',ledger)
    json_write(out/'contrato_datos_y_particiones.json',protocol)
    (out/'pruebas_unitarias.txt').write_text(stream.getvalue(),encoding='utf-8')
    summary=dict(protocol_id=protocol['protocol_id'],created_utc=datetime.now(timezone.utc).isoformat(),
                 status='PARTICIONES_PREPARADAS_Y_VERIFICADAS_SIN_ENTRENAMIENTO',outer_folds=4,inner_folds_total=12,
                 records=28,unique_windows=1380,outer_manifest_rows=len(outer_records),inner_manifest_rows=len(inner_records),
                 outer_index_rows=len(outer_indices),inner_index_rows=len(inner_indices),
                 synthetic_tests=result.testsRun,data_checks=len(checks),inputs_verified_unchanged=len(hashes),
                 audit_outputs_verified=audit_outputs,no_arrays_copied_or_rewritten=True,no_models_trained=True,
                 no_new_predictive_metrics=True,overlap_fraction=.5,window_size=4096,fs=12000,
                 physical_independence_claimed=False,full_experimental_protocol_frozen=False,
                 python=sys.version,numpy=np.__version__)
    json_write(out/'resumen.json',summary)
    outer_summary=[r for r in summaries if r['inner_validation_load_hp']=='']
    lines=[]
    for load in LOADS:
        dev=next(r for r in outer_summary if r['outer_load_hp']==load and r['role']=='development')
        test=next(r for r in outer_summary if r['outer_load_hp']==load and r['role']=='outer_test')
        lines.append(f"| {load} HP | {dev['files']} | {dev['windows']} | {test['files']} | {test['windows']} |")
    (out/'README_PARTICIONES.md').write_text('''# Particiones CWRU V2 por carga

Estado: preparación validada, sin entrenamiento ni métricas nuevas.
Se conserva el 50% de solapamiento, N=4096 y 12 kHz. No se movieron ni copiaron
MAT, NPY o modelos. No se modificaron los documentos/sellos de la auditoría.

| Carga de prueba | MAT desarrollo | Ventanas desarrollo | MAT prueba | Ventanas prueba |
|---|---:|---:|---:|---:|
'''+ '\n'.join(lines)+'''

Cada conjunto de desarrollo se divide tres veces: dos cargas para entrenamiento
y una para validación. La carga externa queda excluida en los tres casos. Tras
seleccionar sólo dentro de desarrollo, se reajustaría con sus tres cargas; no
se ha hecho ese ajuste. Hay 4 pliegues externos y 12 internos.

## Cómo usar las tablas

- `manifiesto_registros_externo.csv`: filtrar outer_load_hp. Papeles development/outer_test.
- `manifiesto_registros_interno.csv`: filtrar outer_load_hp e inner_validation_load_hp.
  Papeles inner_train/inner_validation/outer_excluded.
- `indice_ventanas_fuente.csv`: identificador global de ventana y ruta/fila NPY
  original. Contiene 1380 ventanas únicas; ambos canales permanecen juntos.
- `indices_ventanas_externo.csv` / `indices_ventanas_interno.csv`: asignan esos
  identificadores a un papel por pliegue, no crean observaciones nuevas.
- `alcance_ajustes_por_pliegue.json`: allowlist de MAT para cada futuro fit.
- `contrato_datos_y_particiones.json`: snapshot exacto de las reglas operativas.

El orden global concatena las ubicaciones históricas train, validation, test.
**Esos nombres históricos no determinan el papel V2**: sólo indican dónde leer.
Las características no están escaladas. Cada nuevo fit requiere un scaler nuevo,
pesos de clase y calibración INT8 usando únicamente su allowlist. El detector
requiere, además, restringir su ajuste a N. No reutilizar modelos o scaler V1
como estimadores de pliegues que contienen datos vistos por ellos.

La guarda `guard_fit_scope` debe integrarse antes de los futuros ajustes. Estas
pruebas validan el contrato y sus índices; no certifican entrenamientos aún no
implementados ni pueden impedir una lectura externa que ignore deliberadamente
la guarda. La carga externa no determina early stopping, umbral, máscaras ni
hiperparámetros, y no cambia la regla de selección de posteriores pliegues.

## Límites

No hay separación verificada por rodamiento/motor. Las cargas son condiciones de
un banco; no réplicas físicas independientes. La evaluación es retrospectiva:
el corpus ya fue explorado. Las 1380 ventanas no son 1380 adquisiciones. Los
cuatro pliegues externos y las semillas tampoco son bancos independientes.
Se conserva la advertencia de diseño histórico del FIR con cargas 0/1.

No se excluyeron 201/213/238 por canales equivalentes ni 177/214/215 por
advertencias de adquisición. Cualquier sensibilidad adicional debe fijarse
antes de entrenar y sin selección según resultados externos.

## Reproducción

Desde la raíz, con Python y NumPy:

```powershell
python -B codigo/cwru/ampliacion_v2/particiones/test_particiones.py
python -B codigo/cwru/ampliacion_v2/particiones/verificar_particiones.py --directory datos/cwru/particiones/ampliacion_v2/por_carga_2026-10-22
python -B codigo/cwru/ampliacion_v2/particiones/generar_particiones.py --output datos/cwru/particiones/ampliacion_v2/repeticion
```

Una repetición exige directorio nuevo. `entradas_sha256.csv` identifica entradas
y código; `MANIFEST_SHA256.csv` sella todas las salidas salvo a sí mismo. No
regenerar un sello para ocultar discrepancias. El protocolo experimental completo
sigue pendiente de costo STM32, contribución, reglas de selección y presupuesto.
''',encoding='utf-8')
    files=sorted(p for p in out.iterdir() if p.is_file())
    csv_write(out/'MANIFEST_SHA256.csv',[dict(path=p.name,sha256=sha(p)) for p in files])
    print(json.dumps(summary,ensure_ascii=True,indent=2))


if __name__=='__main__':main()
