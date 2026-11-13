"""Auditoría determinista CWRU, sin entrenamiento ni modificación de entradas.

Ejecutar desde la raíz: python -B codigo/cwru/ampliacion_v2/auditoria/auditar_dataset_cwru.py
Requiere NumPy, SciPy y Matplotlib. --output debe ser un directorio nuevo.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import platform
import sys
import time
import unittest
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
from scipy.io import loadmat
from scipy.signal import freqz

from audit_core import (affine_relation, assert_metadata, assert_windows, centered_corr,
                        choose_channel, file_hash, independent_decimation,
                        independent_features, numeric_hash, scan_exact_blocks)

SPLITS = ("train", "validation", "test")
SOURCE_URLS = {
    "official_description": "https://engineering.case.edu/bearingdatacenter/download-data-file",
    "official_faults": "https://engineering.case.edu/bearingdatacenter/48k-drive-end-bearing-fault-data",
    "official_normal": "https://engineering.case.edu/bearingdatacenter/normal-baseline-data",
    "official_apparatus": "https://engineering.case.edu/bearingdatacenter/apparatus-and-procedures",
    "smith_randall_doi": "https://doi.org/10.1016/j.ymssp.2015.04.021",
    "smith_randall_author_copy": "https://www.researchgate.net/publication/276248554_Rolling_Element_Bearing_Diagnostics_Using_the_Case_Western_Reserve_University_Data_A_Benchmark_Study",
}
LITERATURE_FLAGS = {
    177: "Ruido eléctrico DE/FE, Smith y Randall (2015), Tabla 3; no atribución automática de toda muestra a ruido.",
    201: "DE/FE equivalentes salvo escala, Smith y Randall (2015), Tabla 3; comprobación local independiente.",
    213: "DE/FE equivalentes salvo escala, Smith y Randall (2015), Tabla 3; comprobación local independiente.",
    238: "DE/FE equivalentes salvo escala, Smith y Randall (2015), Tabla 3; comprobación local independiente.",
    214: "Segmentos recortados en DE, Smith y Randall (2015), Tabla 3; no se establece aquí umbral físico de saturación.",
    215: "Segmentos recortados en DE, Smith y Randall (2015), Tabla 3; no se establece aquí umbral físico de saturación.",
}


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields=None):
    if fields is None:
        fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        w.writerows(rows)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def finite(value):
    return float(value) if np.isfinite(value) else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[4])
    parser.add_argument("--output", type=Path, default=Path("articulo/ampliacion_cwru/auditoria/revision_2026-10-22_validada"))
    args = parser.parse_args()
    root = args.root.resolve()
    out = (root/args.output).resolve()
    permitted = (root/"articulo/ampliacion_cwru/auditoria").resolve()
    if not out.is_relative_to(permitted) or out == permitted or out.exists():
        raise ValueError("La salida debe ser una subcarpeta NUEVA de articulo/ampliacion_cwru/auditoria")
    out.mkdir(parents=True)
    started = time.perf_counter()
    tests = []

    def check(code, condition, observed, criterion, artifact, failure="FAIL"):
        tests.append(dict(id=code, status="PASS" if condition else failure,
                          observed=str(observed), criterion=criterion, evidence=artifact))

    raw = root/"datos/cwru/raw"
    processed = root/"datos/cwru/procesado/dual_12khz_w4096"
    features_dir = processed/"features_14"
    filt = root/"resultados/cwru/preprocesamiento_12khz/filtro_antialias"
    manifest_path = root/"datos/cwru/particiones/principal_por_carga/manifest_particion.csv"
    manifest = read_csv(manifest_path)
    raw_paths = sorted(raw.rglob("*.mat"))
    inputs = set(raw_paths + [manifest_path])
    for directory in (processed, filt, root/"codigo/cwru/preprocesamiento",
                      root/"codigo/cwru/ampliacion_v2/auditoria",
                      root/"modelos/cwru/canonico_seed42",
                      root/"documentacion/cwru"):
        inputs.update(p for p in directory.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    for p in (root/"articulo/ampliacion_cwru/PLAN_MAESTRO_AMPLIACION_CWRU.md",
              root/"articulo/ampliacion_cwru/protocolo_v2_propuesto.json",
              root/"resultados/cwru/generalizacion_carga_diametro.csv"):
        if p.exists(): inputs.add(p)
    inputs.update((root/"resultados/cwru/modelo_features_14/evaluacion_final_test_load3").glob("*.json"))
    print(f"1/7 Huellas de {len(inputs)} entradas", flush=True)
    before = {p: file_hash(p) for p in sorted(inputs)}
    write_csv(out/"entradas_sha256.csv", [dict(path=p.relative_to(root).as_posix(), bytes=p.stat().st_size,
              sha256=h) for p,h in before.items()])
    check("CW01",len(raw_paths)==28 and len(manifest)==28 and len({r['file'] for r in manifest})==28,
          f"MAT={len(raw_paths)}; filas manifiesto={len(manifest)}", "28 archivos únicos y 28 filas", "inventario_archivos.csv")

    records, signals, vectors, variables, quality = [], [], [], [], []
    labels = {"N":0,"IR":1,"OR":2}
    actual_ids = {"N":{0:[97],1:[98],2:[99],3:[100]},
                  "IR":{0:[109,174,213],1:[110,175,214],2:[111,176,215],3:[112,177,217]},
                  "OR":{0:[135,201,238],1:[136,202,239],2:[137,203,240],3:[138,204,241]}}
    print("2/7 Inventario MAT y selección explícita", flush=True)
    for r in manifest:
        matches = [p for p in raw_paths if p.name == r["file"]]
        if len(matches)!=1: raise ValueError(f"Ruta ambigua: {r['file']}")
        p = matches[0]
        d = loadmat(p)
        n, fid, load = int(r["samples_per_channel"]), int(r["file_id"]), int(r["load_hp"])
        de, fe = choose_channel(d,r["de_key"],n), choose_channel(d,r["fe_key"],n)
        signals.append(np.column_stack((de,fe)))
        prefix = r["de_key"].split("_DE_time")[0]
        rpm_key = prefix+"RPM"
        rpm = float(d[rpm_key].squeeze()) if rpm_key in d else None
        valid_len = (n+3)//4-44
        count = max(0,(valid_len-4096)//2048+1)
        relation = affine_relation(de,fe)
        row = dict(r, path=p.relative_to(root).as_posix(), bytes=p.stat().st_size,
                   sha256=before[p], duration_s_at_documented_48khz=n/48000,
                   rpm_measured=rpm, rpm_manifest_approx=int(r["rpm"]),
                   rpm_abs_difference=None if rpm is None else abs(rpm-int(r["rpm"])),
                   source_fs_hz_documented=48000, source_fs_embedded_in_mat=False,
                   channel_semantics="DE y FE: posiciones, no ejes",
                   source_configuration_proxy=f"{r['class']}_{float(r['diameter_in']):.3f}"+("_6oclock" if r['class']=="OR" else ""),
                   physical_bearing_id_verified=False, valid_samples_12khz=valid_len,
                   expected_windows=count, de_fe_r=relation["correlation"],
                   de_fe_affine_identical=relation["affine_match"],
                   fe_over_de_scale=relation["scale"], de_fe_max_residual=relation["max_abs_residual"],
                   literature_warning=LITERATURE_FLAGS.get(fid,""))
        records.append(row)
        for key,a in d.items():
            if key.startswith("__"): continue
            a = np.asarray(a)
            entry = dict(file=r['file'],file_id=fid,split=r['split'],variable=key,
                         shape="x".join(map(str,a.shape)),dtype=str(a.dtype),elements=a.size,
                         selected=key in (r['de_key'],r['fe_key']),finite=bool(np.isfinite(a).all()),
                         minimum=float(a.min()),maximum=float(a.max()))
            variables.append(entry)
            if a.size>1000 and a.ndim==2 and min(a.shape)==1:
                vectors.append(dict(file=r['file'],file_id=fid,split=r['split'],variable=key,
                                    selected=entry['selected'],array=a.ravel(),hash=numeric_hash(a.ravel())))
            elif a.ndim==2 and a.shape[1]==2 and a.shape[0]>1000:
                for col in (0,1):
                    v=a[:,col]
                    vectors.append(dict(file=r['file'],file_id=fid,split=r['split'],variable=f"{key}[:,{col}]",
                                        selected=False,array=v,hash=numeric_hash(v)))
        for channel,a in (("DE",de),("FE",fe)):
            q=dict(file=r['file'],file_id=fid,split=r['split'],channel=channel,
                   samples=n,mean=float(a.mean()),std_population=float(a.std()),
                   rms=float(np.sqrt(np.mean(a*a))),minimum=float(a.min()),maximum=float(a.max()),
                   exact_minimum_count=int(np.count_nonzero(a==a.min())),
                   exact_maximum_count=int(np.count_nonzero(a==a.max())),
                   exact_adjacent_equal_pairs=int(np.count_nonzero(a[1:]==a[:-1])))
            blocks=np.array_split(a,10)
            br=np.array([np.sqrt(np.mean(b*b)) for b in blocks])
            q.update(rms_ten_blocks_min=float(br.min()),rms_ten_blocks_max=float(br.max()),
                     rms_ten_blocks_max_min_ratio=float(br.max()/br.min()))
            for lag in (1,4,16,32,256,4096):q[f'autocorr_lag_{lag}_raw_samples']=finite(centered_corr(a[:-lag],a[lag:]))
            quality.append(q)
    write_csv(out/"inventario_archivos.csv",records)
    write_csv(out/"inventario_variables_mat.csv",variables)
    write_csv(out/"calidad_y_autocorrelacion.csv",quality)
    check("CW02",set(p.name for p in raw_paths)==set(r['file'] for r in manifest),len(records),
          "Manifiesto y archivos MAT coinciden; ninguna ruta ambigua", "inventario_archivos.csv")
    check("CW03",all(int(r['file_id']) in actual_ids[r['class']][int(r['load_hp'])] and
          float(r['diameter_in'])==(0.0 if r['class']=='N' else
              (.007,.014,.021)[actual_ids[r['class']][int(r['load_hp'])].index(int(r['file_id']))]) and
          r['de_key']==f"X{173 if int(r['file_id'])==174 else int(r['file_id']):03d}_DE_time" and
          r['fe_key']==f"X{173 if int(r['file_id'])==174 else int(r['file_id']):03d}_FE_time" and
          int(r['label'])==labels[r['class']] for r in records), "IDs, clases, diámetros y claves contrastados con tablas oficiales",
          "IDs, etiquetas y cargas oficiales esperadas", "inventario_archivos.csv; fuentes_y_alcance.md")
    check("CW04",all(v['finite'] for v in variables),len(variables),"Todas las variables numéricas finitas", "inventario_variables_mat.csv")
    selected_affine=[r for r in records if r['de_fe_affine_identical']]
    check("CW05",not selected_affine,[r['file_id'] for r in selected_affine],
          "Los dos canales no deben suponerse independientes si son copias afines", "inventario_archivos.csv; equivalencias_afines.csv", "WARN")

    print("3/7 Copias completas, afines y bloques exactos en todas las posiciones", flush=True)
    duplicates, affine = [], []
    for i,a in enumerate(vectors):
        for b in vectors[i+1:]:
            if len(a['array']) != len(b['array']):continue
            base={f'{key}_{side}':v[key] for side,v in (("a",a),("b",b))
                  for key in ("file","file_id","split","variable","selected")}
            base['cross_split']=a['split']!=b['split']
            base['samples']=len(a['array'])
            if a['hash']==b['hash'] and np.array_equal(a['array'],b['array']):
                duplicates.append(dict(base,sha256_numeric=a['hash']))
            rel=affine_relation(a['array'],b['array'])
            if rel['affine_match']:affine.append(dict(base,**rel))
    write_csv(out/"duplicados_variables_completas.csv",duplicates,
              list(duplicates[0]) if duplicates else ['file_a','file_b','selected_a','selected_b','cross_split'])
    write_csv(out/"equivalencias_afines.csv",affine)
    block_reports={}
    for mode,column in (("DE",0),("FE",1),("DE_FE",None)):
        print(f"  Escaneo {mode}: bloques >=16 muestras originales; todos los desfases",flush=True)
        arrays=[s if column is None else s[:,column] for s in signals]
        matches,scan=scan_exact_blocks(records,arrays,16)
        block_reports[mode]=scan
        write_csv(out/f"segmentos_exactos_{mode}.csv",matches,
                  ['file_a','file_b','split_a','split_b','start_a','start_b','length_samples','cross_split'])
        check(f"CW06_{mode}",scan['coverage_complete'] and not any(x['cross_split'] for x in matches),scan,
              "Cobertura completa y sin segmentos exactos cruzados >=16 muestras, a igualdad de canal",
              f"segmentos_exactos_{mode}.csv", "WARN")
    write_json(out/"cobertura_busqueda_duplicados.json",block_reports)
    cross_selected_affine=[x for x in affine if x['cross_split'] and x['selected_a'] and x['selected_b']]
    check("CW07",not cross_selected_affine,len(cross_selected_affine),
          "Sin copias afines COMPLETAS entre variables seleccionadas de distintos conjuntos",
          "equivalencias_afines.csv")
    auxiliary_cross=[x for x in affine if x['cross_split'] and not(x['selected_a'] and x['selected_b'])]
    check("CW08",not auxiliary_cross,len(auxiliary_cross),
          "Variables auxiliares cruzadas presentes: la lectura debe ser por clave explícita",
          "equivalencias_afines.csv", "WARN")
    # Libera las vistas auxiliares antes del recálculo de ventanas.
    del vectors

    print("4/7 Recálculo independiente de FIR, ventanas y 14 características", flush=True)
    taps=np.load(filt/"coeficientes_fir_kaiser_177_float32.npy")
    specification=json.loads((filt/"especificacion_filtro.json").read_text(encoding="utf-8"))
    coefficients_hash=hashlib.sha256(taps.tobytes()).hexdigest()
    frequencies,response=freqz(taps.astype(np.float64),worN=262144,fs=48000)
    db=20*np.log10(np.maximum(np.abs(response),np.finfo(float).tiny))
    ripple=float(np.ptp(db[frequencies<=5000]));stop=float(db[frequencies>=6000].max())
    fir=dict(num_taps=len(taps),dtype=str(taps.dtype),coefficients_sha256=coefficients_hash,
             passband_ripple_db=ripple,stopband_max_db=stop,input_fs_hz=48000,output_fs_hz=12000,
             group_delay_raw_samples=88,group_delay_output_samples=22,edge_trim_each_side_output=22,
             response_grid_points=262144,passband_edge_hz=5000,stopband_edge_hz=6000)
    write_json(out/"verificacion_fir.json",fir)
    check("CW09",len(taps)==177 and taps.dtype==np.float32 and
          coefficients_hash==specification['sha256_float32_coefficients'] and ripple<=0.1 and stop<=-60,
          fir,"177 coeficientes float32, hash canónico, ripple<=0.1dB y rechazo>=60dB", "verificacion_fir.json")
    loaded={}
    for split in SPLITS:
        x=np.load(processed/f'X_{split}.npy',mmap_mode='r')
        y=np.load(processed/f'y_{split}.npy')
        f=np.load(features_dir/f'X_features_{split}.npy')
        fy=np.load(features_dir/f'y_{split}.npy')
        m=read_csv(processed/f'metadata_{split}.csv')
        fm=read_csv(features_dir/f'metadata_features_{split}.csv')
        if not (len(x)==len(y)==len(f)==len(m)==len(fm) and np.array_equal(y,fy)):
            raise ValueError(f"Longitudes o etiquetas incompatibles: {split}")
        if x.shape[1:]!=(4096,2) or x.dtype!=np.float32 or f.shape[1]!=14 or not np.isfinite(x).all() or not np.isfinite(f).all():
            raise ValueError(f"Forma/tipo/valores inválidos: {split}")
        loaded[split]=(x,y,f,m,fm)
    feature_names=json.loads((features_dir/'configuracion_features.json').read_text(encoding='utf-8'))['feature_order']
    checks_by_file=[];stats=[]; window_hashes=[]; overlap_checked=0;decimal_roundings=0;max_decimal_error=0.0
    for r,sig in zip(records,signals):
        split=r['split'];x,y,f,m,fm=loaded[split]
        indices=[i for i,row in enumerate(m) if row['file']==r['file']]
        if len(indices)!=r['expected_windows']:raise ValueError(f"Número de ventanas distinto: {r['file']}")
        independent=np.column_stack([independent_decimation(sig[:,j],taps) for j in (0,1)])
        max_window_error,max_feature_error,max_feature_rel=0.0,0.0,0.0
        max_metadata_feature_error=0.0
        for j,i in enumerate(indices):
            row=m[i];start=j*2048
            if (int(row['start_sample_12khz_valid'])!=start or
                int(row['end_sample_12khz_valid_exclusive'])!=start+4096 or
                int(row['start_sample_12khz_aligned'])!=start+22 or
                int(row['end_sample_12khz_aligned_exclusive'])!=start+4118 or
                abs(float(row['start_time_s'])-(start+22)/12000)>1e-12 or
                abs(float(row['end_time_s'])-(start+4118)/12000)>1e-12 or
                int(row['window_index_in_file'])!=j or int(row['file_id'])!=int(r['file_id']) or
                int(row['label'])!=int(r['label']) or y[i]!=int(r['label']) or
                int(row['load_hp'])!=int(r['load_hp']) or row['class']!=r['class'] or
                row['de_key']!=r['de_key'] or row['fe_key']!=r['fe_key'] or row['split']!=split):
                raise ValueError(f"Metadata incompatible: {r['file']}, ventana {j}")
            assert_windows(independent,x[i],start)
            max_window_error=max(max_window_error,float(np.max(np.abs(independent[start:start+4096]-x[i]))))
            expected=independent_features(x[i])
            np.testing.assert_allclose(expected,f[i],rtol=1e-9,atol=1e-8)
            error=np.abs(expected-f[i]);max_feature_error=max(max_feature_error,float(error.max()))
            max_feature_rel=max(max_feature_rel,float(np.max(error/np.maximum(np.abs(f[i]),1e-12))))
            meta_f=np.array([float(fm[i][name]) for name in feature_names])
            np.testing.assert_allclose(f[i],meta_f,rtol=1e-12,atol=1e-8)
            max_metadata_feature_error=max(max_metadata_feature_error,float(np.max(np.abs(f[i]-meta_f))))
            assert_metadata(row,fm[i])
            for k in ('start_time_s','end_time_s','diameter_in'):
                if fm[i][k]!=row[k]:
                    decimal_roundings+=1
                    max_decimal_error=max(max_decimal_error,abs(float(fm[i][k])-float(row[k])))
            if j:
                np.testing.assert_array_equal(x[indices[j-1],2048:],x[i,:2048]);overlap_checked+=1
            window_hashes.append(dict(split=split,file=r['file'],window_index=j,sha256_numeric=numeric_hash(x[i])))
        checks_by_file.append(dict(file=r['file'],file_id=r['file_id'],split=split,windows=len(indices),
                                   max_abs_window_error=max_window_error,max_abs_feature_error=max_feature_error,
                                   max_relative_feature_error=max_feature_rel,
                                   max_abs_feature_metadata_error=max_metadata_feature_error,
                                   metadata_valid=True,channels_explicit_verified=True))
        for k,name in enumerate(feature_names):
            values=f[indices,k]
            stat=dict(file=r['file'],file_id=r['file_id'],split=split,feature=name,
                      windows=len(indices),minimum=float(values.min()),maximum=float(values.max()),
                      mean=float(values.mean()),std_population=float(values.std()))
            for lag in (1,2,4):stat[f'within_file_feature_autocorr_lag_{lag}_windows']=finite(centered_corr(values[:-lag],values[lag:])) if len(values)>lag+1 else None
            stats.append(stat)
        print(f"  {r['file_id']}: {len(indices)} ventanas verificadas",flush=True)
    write_csv(out/"recalculo_por_archivo.csv",checks_by_file)
    write_csv(out/"estadisticas_caracteristicas.csv",stats)
    write_csv(out/"huellas_ventanas.csv",window_hashes)
    check('CW23',max_decimal_error<=1e-12,f'{decimal_roundings} representaciones decimales diferentes; error máximo={max_decimal_error}',
          'Metadata base/enriquecida idéntica salvo representación float con diferencia<=1e-12', 'recalculo_por_archivo.csv')
    n_windows=sum(r['windows'] for r in checks_by_file)
    check("CW10",n_windows==1380,n_windows,"Todas las 1380 ventanas reproducidas desde MAT seleccionados; rtol1e-6, atol1e-7", "recalculo_por_archivo.csv")
    check("CW11",n_windows*14==19320,n_windows*14,"19320 valores de características; fórmulas independientes; rtol1e-9, atol1e-8", "recalculo_por_archivo.csv")
    check("CW12",overlap_checked==n_windows-len(records),overlap_checked,"Solapamiento exacto 2048 muestras sólo dentro de archivo", "recalculo_por_archivo.csv")
    by_hash=defaultdict(list)
    for row in window_hashes:by_hash[row['sha256_numeric']].append(row)
    duplicate_windows=[v for v in by_hash.values() if len(v)>1]
    write_json(out/'ventanas_duplicadas.json',duplicate_windows)
    check("CW13",not duplicate_windows,len(duplicate_windows),"Ninguna ventana procesada completa repetida (dos canales)","ventanas_duplicadas.json")
    legacy_hash_checks=[]
    for hp in sorted(processed.rglob('HASHES.sha256')):
        for line in hp.read_text(encoding='utf-8-sig').splitlines():
            if not line.strip():continue
            digest,name=line.split(maxsplit=1);name=name.lstrip('* ')
            p=hp.parent/name
            actual=file_hash(p)
            legacy_hash_checks.append(dict(manifest=hp.relative_to(root).as_posix(),file=name,
                                           matches=actual==digest,expected=digest,actual=actual))
    write_csv(out/'comprobacion_hashes_historicos.csv',legacy_hash_checks)
    check("CW14",bool(legacy_hash_checks) and all(r['matches'] for r in legacy_hash_checks),len(legacy_hash_checks),
          "Todos los HASHES.sha256 históricos contrastados", "comprobacion_hashes_historicos.csv")

    print("5/7 Agrupación, exposición histórica y pruebas negativas", flush=True)
    grouped=[]
    for split in SPLITS:
        for cls in ('N','IR','OR'):
            rr=[r for r in records if r['split']==split and r['class']==cls]
            grouped.append(dict(split=split,**{'class':cls},files=len(rr),windows=sum(r['expected_windows'] for r in rr),
                                raw_duration_s=sum(r['duration_s_at_documented_48khz'] for r in rr)))
    write_csv(out/'distribucion_por_clase_y_split.csv',grouped)
    proxies=defaultdict(set)
    for r in records:proxies[r['source_configuration_proxy']].add(r['split'])
    write_csv(out/'grupos_y_limitaciones.csv',[dict(source_configuration_proxy=k,splits=';'.join(sorted(v)),
              physical_bearing_identity='NO_VERIFICADA',independence_scope='por registro/carga; no por motor o rodamiento nuevo') for k,v in sorted(proxies.items())])
    check('CW15',all(int(r['load_hp']) in ({0,1} if r['split']=='train' else {2} if r['split']=='validation' else {3}) for r in records),
          '14 train/7 validation/7 test; cargas 0,1/2/3',"Separación por MAT y por carga antes del ventaneo",'inventario_archivos.csv')
    check('CW16',False,'7 configuraciones nominales; identidad física no verificable; configuraciones compartidas entre cargas',
          'No afirmar independencia entre motores/rodamientos a partir de archivos distintos', 'grupos_y_limitaciones.csv','LIMITATION')
    exposure=dict(all_28_records_previously_used=True,load3_test_previously_evaluated=True,
                  primary_evidence='documentacion/cwru/evaluacion_final_test_cwru.md',
                  additional_legacy_evidence='resultados/cwru/generalizacion_carga_diametro.csv',
                  v2_design_status='Evaluación retrospectiva, con reglas futuras congeladas antes de ejecutar; no test virgen',
                  no_new_training_in_this_audit=True)
    write_json(out/'exposicion_historica.json',exposure)
    check('CW17',False,'El conjunto 3HP ya tiene resultados y fue inspeccionado; nuevo split no borra la exposición',
          'No llamar ciego/no observado al nuevo test reutilizado', 'exposicion_historica.json','LIMITATION')
    check('CW18',False,'Fs documentada=48000 Hz; no hay reloj/timestamps/fs embebidos que permitan medición independiente',
          'Separar frecuencia documentada de frecuencia medida', 'inventario_variables_mat.csv; fuentes_y_alcance.md','LIMITATION')
    check('CW19',False,'No se descargaron ni compararon bytes oficiales en esta ejecución; procedencia por IDs y claves, no autenticación remota',
          'Hash local no equivale a hash del distribuidor', 'entradas_sha256.csv; fuentes_y_alcance.md','LIMITATION')
    warnings=[dict(file_id=k,source=SOURCE_URLS['smith_randall_doi'],warning=v,status='NO_CORREGIDO_NO_EXCLUIDO') for k,v in LITERATURE_FLAGS.items()]
    write_csv(out/'advertencias_bibliograficas.csv',warnings)
    check('CW20',False,'Ruido eléctrico reportado en 177; recorte de DE reportado en 214 y 215',
          'Documentar calidad antes de entrenar; no exclusiones por desempeño', 'advertencias_bibliograficas.csv','WARN')
    import test_auditoria_cwru
    stream=io.StringIO()
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(test_auditoria_cwru))
    (out/'pruebas_unitarias.txt').write_text(stream.getvalue(),encoding='utf-8')
    check('CW21',result.wasSuccessful(),f'{result.testsRun} tests; errores={len(result.errors)}; fallos={len(result.failures)}',
          'Canales erróneos, desfase, copias, FIR, momentos, constante y alteración de manifiesto detectados', 'pruebas_unitarias.txt')

    print("6/7 Figuras de diagnóstico (no son figuras finales del artículo)", flush=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9})
    proxy_names=sorted(proxies)
    matrix=np.zeros((len(proxy_names),4),dtype=int)
    for r in records:matrix[proxy_names.index(r['source_configuration_proxy']),int(r['load_hp'])]=r['expected_windows']
    fig,ax=plt.subplots(figsize=(7.2,4.2),constrained_layout=True)
    ax.imshow(matrix,cmap='Blues',aspect='auto')
    ax.set(xticks=range(4),xticklabels=['0 HP\ntrain','1 HP\ntrain','2 HP\nvalidation','3 HP\ntest'],
           yticks=range(len(proxy_names)),yticklabels=proxy_names,title='Ventanas por configuración nominal y carga (no IDs físicos)')
    for j in range(len(proxy_names)):
        for k in range(4):ax.text(k,j,str(matrix[j,k]),ha='center',va='center')
    fig.savefig(out/'distribucion_ventanas.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(8.5,5.3),constrained_layout=True)
    a=axes[0];a.bar(range(len(records)),[r['de_fe_r'] for r in records],color=['#b91c1c' if r['de_fe_affine_identical'] else '#2563eb' for r in records])
    a.set(xticks=range(len(records)),xticklabels=[r['file_id'] for r in records],ylim=(-1.05,1.05),
          ylabel='Correlación DE–FE',title='Rojo: copia afín confirmada con todas las muestras, no sólo correlación alta')
    a.tick_params(axis='x',labelsize=7)
    b=axes[1];b.plot(frequencies/1000,db);b.set(xlim=(0,12),ylim=(-120,5),xlabel='Frecuencia (kHz)',ylabel='Respuesta FIR (dB)')
    b.axvline(5,color='gray',ls=':');b.axvline(6,color='gray',ls=':');b.axhline(-60,color='red',ls='--');b.grid(alpha=.25)
    fig.savefig(out/'canales_y_respuesta_fir.png',dpi=180);plt.close(fig)
    # Ejemplo seleccionado por ID publicado, no por acierto/error de un clasificador.
    idx=next(i for i,r in enumerate(records) if int(r['file_id'])==213)
    de,fe=signals[idx][:4096].T;scale=records[idx]['fe_over_de_scale']
    fig,axes=plt.subplots(2,1,figsize=(7.2,4.8),constrained_layout=True)
    axes[0].plot(np.arange(4096)/48000,de,label='DE');axes[0].plot(np.arange(4096)/48000,fe/scale,'--',label='FE / escala',lw=.8)
    axes[0].set(title='Registro 213: superposición tras corregir escala',ylabel='Amplitud MAT',xlabel='Tiempo documentado (s)');axes[0].legend()
    axes[1].plot(np.arange(4096)/48000,fe-scale*de);axes[1].set(xlabel='Tiempo documentado (s)',ylabel='FE − escala × DE',title='Residuo; comprobar CSV para el registro completo')
    fig.savefig(out/'evidencia_canales_213.png',dpi=180);plt.close(fig)

    print("7/7 Integridad final e informe", flush=True)
    changed=[p.relative_to(root).as_posix() for p,h in before.items() if not p.exists() or file_hash(p)!=h]
    check('CW22',not changed,changed or f'{len(before)} entradas sin cambios',
          'SHA256 idéntico antes y después; originales y V1 intactos', 'entradas_sha256.csv')
    write_json(out/'integridad_antes_despues.json',dict(inputs_checked=len(before),changed=changed,unchanged=not changed))
    write_csv(out/'registro_pruebas.csv',tests)
    summary=dict(status='AUDITADO_CON_ADVERTENCIAS' if not any(t['status']=='FAIL' for t in tests) else 'FALLOS_DE_VALIDACION',
                 created_utc=datetime.now(timezone.utc).isoformat(),duration_seconds=time.perf_counter()-started,
                 raw_mat_files=len(records),raw_bytes=sum(r['bytes'] for r in records),variables=len(variables),
                 windows=n_windows,feature_values=n_windows*14,overlaps_checked=overlap_checked,
                 split_window_counts={s:len(loaded[s][0]) for s in SPLITS},
                 selected_de_fe_affine_duplicate_ids=[int(r['file_id']) for r in selected_affine],
                 exact_complete_variable_pairs=len(duplicates),affine_complete_variable_pairs=len(affine),
                 cross_split_affine_selected_pairs=len(cross_selected_affine),
                 cross_split_affine_auxiliary_pairs=len(auxiliary_cross),block_scans=block_reports,
                 max_window_abs_error=max(r['max_abs_window_error'] for r in checks_by_file),
                 max_feature_abs_error=max(r['max_abs_feature_error'] for r in checks_by_file),
                 test_status_counts=dict(Counter(t['status'] for t in tests)),
                 metadata_decimal_representation_differences=decimal_roundings,
                 metadata_decimal_max_abs_error=max_decimal_error,
                 original_inputs_unchanged=not changed,no_training_performed=True,
                 no_dataset_or_manuscript_modified=True)
    write_json(out/'resumen_auditoria.json',summary)
    write_json(out/'entorno.json',dict(python=sys.version,executable=sys.executable,platform=platform.platform(),
               numpy=np.__version__,scipy=scipy.__version__,matplotlib=matplotlib.__version__,
               source_root=str(root),commands='Ver README_REPRODUCIBILIDAD.md'))
    write_json(out/'fuentes.json',SOURCE_URLS)
    (out/'fuentes_y_alcance.md').write_text('''# Fuentes y alcance probatorio

Consulta: 2026-10-22. Fuentes primarias, sin redistribuir sus textos.

El centro oficial CWRU identifica DE, FE y BA como posiciones de acelerómetros,
no ejes; sus tablas permiten contrastar IDs, diámetros, cargas y RPM aproximadas.
Los archivos seleccionados pertenecen a normal y fallas del rodamiento DE, OR a
las 6. No confundir canal FE con la colección de fallas del rodamiento FE.

Smith y Randall (2015), Mechanical Systems and Signal Processing 64–65, 100–131,
DOI 10.1016/j.ymssp.2015.04.021: Tabla 3 advierte canales equivalentes salvo escala
(incluidos 201, 213 y 238), ruido eléctrico en 177 y recorte en 214/215 DE.
Las Tablas A1/A3 respaldan 48 kHz en normal y canales DE/FE de esta colección.
La equivalencia afín se confirmó aquí con las muestras locales completas; las
otras advertencias permanecen bibliográficas, sin diagnosticar saturación
instrumental ni cuantificar ruido eléctrico mediante un umbral inventado.

En fuentes.json están los enlaces oficiales, DOI y copia del autor consultada.
No se compararon los bytes locales con una nueva descarga oficial. Los hashes
prueban identidad local, no autoría/licencia. Tampoco se demuestra sincronía
instrumental por la mera igualdad de longitudes, ni identidad de rodamientos.
No se realizan inferencias clínicas/legales ni se conceden permisos de uso.
''',encoding='utf-8')
    (out/'README_REPRODUCIBILIDAD.md').write_text('''# Reproducibilidad de esta auditoría

Desde la raíz del proyecto, con Python 3.12, NumPy, SciPy y Matplotlib:

```powershell
python -B codigo/cwru/ampliacion_v2/auditoria/test_auditoria_cwru.py
python -B codigo/cwru/ampliacion_v2/auditoria/verificar_auditoria_cwru.py --directory articulo/ampliacion_cwru/auditoria/revision_2026-10-22_validada
python -B codigo/cwru/ampliacion_v2/auditoria/auditar_dataset_cwru.py --output articulo/ampliacion_cwru/auditoria/revision_repetida
```

La repetición exige un nombre nuevo: no sobrescribe una auditoría anterior.
El verificador comprueba hashes de entradas y salidas y ejecuta las pruebas
sintéticas, pero no repite el barrido completo de señales. El tercer comando sí.
No requiere TensorFlow, conexión al STM32 ni reentrenamiento.

La ejecución original usó Python del runtime de Codex y los paquetes científicos
Python312 ya instalados por el usuario, sin instalar dependencias. Sus versiones
exactas y el ejecutable están registrados en entorno.json. Si python no encuentra
SciPy, usar ese intérprete científico o incorporar su directorio site-packages
antes de importar bibliotecas, como se hizo en esta ejecución.

Tolerancias: ventanas rtol=1e-6/atol=1e-7; características rtol=1e-9/atol=1e-8.
El FIR se recalcula con convolución directa, no llamando al generador canónico.
Las características usan momentos con corrección de sesgo y FFT por ventana,
no llaman al extractor canónico. El barrido exacto revisa todos los inicios de
bloques de 16 muestras en DE, FE y par DE/FE; todo candidato se confirma valor
por valor. No demuestra ausencia de parentesco, copias escaladas parciales,
similitud aproximada ni independencia física. Las copias afines se comparan
sólo a registro completo e igual longitud, incluidas columnas auxiliares.

MANIFEST_SHA256.csv cubre todas las salidas excepto a sí mismo. Los códigos y
entradas están cubiertos por entradas_sha256.csv. Una edición invalida el sello;
no regenerarlo para ocultar una discrepancia. Las figuras son diagnósticas y
no se insertaron en el manuscrito. Ningún archivo canónico fue modificado.
''',encoding='utf-8')
    (out/'INFORME_AUDITORIA_CWRU.md').write_text(build_report(summary,records,tests,checks_by_file,grouped,fir),encoding='utf-8')
    seal=[]
    for p in sorted(out.rglob('*')):
        if p.is_file():seal.append(dict(path=p.relative_to(out).as_posix(),bytes=p.stat().st_size,sha256=file_hash(p)))
    write_csv(out/'MANIFEST_SHA256.csv',seal)
    print(json.dumps(summary,ensure_ascii=True,indent=2),flush=True)
    if any(t['status']=='FAIL' for t in tests):raise SystemExit(2)


def build_report(s,records,tests,recalculated,grouped,fir):
    table='\n'.join(f"| {r['split']} | {r['class']} | {r['files']} | {r['windows']} | {r['raw_duration_s']:.4f} |" for r in grouped)
    limits='\n'.join(f"- `{t['id']}` ({t['status']}): {t['observed']}. Evidencia: `{t['evidence']}`." for t in tests if t['status']!='PASS')
    short=min(records,key=lambda r:int(r['samples_per_channel']))
    return f'''# Auditoría CWRU — 22 de octubre de 2026

## Dictamen ejecutivo

El subconjunto local es utilizable para una evaluación retrospectiva, controlada
y por carga, pero no está libre de problemas de adquisición ni demuestra
generalización a motores/rodamientos nuevos. El pipeline V1 seleccionó los canales
correctos: se verificaron todas sus ventanas y características desde los MAT.
No se entrenó ningún modelo, no se alteraron datos, particiones, resultados V1 ni
el manuscrito. Estado: **{s['status']}**.

## Inventario y balance

{s['raw_mat_files']} MAT, {s['raw_bytes']:,} bytes, {s['variables']} variables numéricas.
Tres clases N/IR/OR, cuatro cargas 0–3 HP, diámetros 0.007/0.014/0.021 pulgadas,
OR centrado a las 6. No es toda la base CWRU: no incluye clase bola ni todos los
diámetros/posiciones/colecciones. DE y FE son posiciones, no ejes X/Y/Z.

| Subconjunto histórico | Clase | MAT | Ventanas | Duración original (s, Fs documentada) |
|---|---|---:|---:|---:|
{table}

Fs fuente documentada 48 kHz, remuestreo a 12 kHz, N=4096, hop=2048,
341.333 ms/ventana y 170.667 ms entre decisiones. 1380 ventanas no equivalen a
1380 ensayos independientes. El archivo {short['file_id']} tiene
{short['duration_s_at_documented_48khz']:.4f} s y {short['expected_windows']} ventanas;
es mucho más corto que la mayoría: registrar el desbalance, no duplicarlo.

## Hallazgos confirmados

1. **Canales redundantes:** 201, 213 y 238 (carga 0, train) tienen FE ≈
   0.984848484848485 × DE. Residuo completo máximo inferior a 1e-15. Se verificó
   con todas las muestras, no sólo Pearson. No afirmar diversidad de dos sensores
   para estos registros. Ver `equivalencias_afines.csv` y `evidencia_canales_213.png`.
2. **Variables auxiliares peligrosas:** 99 incluye copias X098; 217 incluye X215.
   175 contiene una matriz X217 y otra señal que son equivalentes, tras escala,
   a señales del test 217. Existen {s['cross_split_affine_auxiliary_pairs']} pares
   afines cruzados con al menos una variable auxiliar. No constituyen fuga en V1:
   sus claves seleccionadas excluyen estos extras, confirmado por el recálculo.
3. **Identificador interno:** 174 usa X173; está declarado en el manifiesto.
   RPM interna y RPM tabulada aproximada se conservan por separado; no son un
   error de clase por diferir unos pocos RPM. 98/99 no contienen RPM seleccionada.
4. **Búsqueda de copias seleccionadas:** barrido exhaustivo en todos los inicios de
   bloques exactos de 16 muestras, por DE, FE y par. Consultar los tres CSV y
   `cobertura_busqueda_duplicados.json`: no extender ausencia de coincidencias a
   una afirmación de independencia. Comparación afín de registros completos:
   {s['cross_split_affine_selected_pairs']} pares cruzados seleccionados.
5. **Integridad y reconstrucción:** {s['windows']} ventanas, {s['feature_values']}
   características, {s['overlaps_checked']} uniones solapadas verificadas.
   Error absoluto máximo de ventanas {s['max_window_abs_error']:.8g}; de
   características {s['max_feature_abs_error']:.8g}. Se aplica tolerancia
   absoluta y relativa, no igualdad decimal arbitraria.
6. **Filtro:** 177 taps float32, ripple {fir['passband_ripple_db']:.6f} dB a 0–5 kHz,
   máximo en rechazo >=6 kHz {fir['stopband_max_db']:.6f} dB. Compensación de 22
   muestras a 12 kHz y retirada de 22 muestras de cada extremo reconstruidas.
   Es respuesta numérica del filtro, no una medición de antialiasing analógico.

## Advertencias externas, no correcciones silenciosas

Smith y Randall (2015), DOI 10.1016/j.ymssp.2015.04.021, Tabla 3, documentan
problemas de adquisición para registros de esta selección; la redundancia afín
se confirmó localmente. Ruido eléctrico reportado en 177 y recorte reportado en
214/215 DE permanecen advertencias bibliográficas: no se estableció aquí un umbral
instrumental de saturación ni se borraron muestras. Ver `fuentes_y_alcance.md`,
`advertencias_bibliograficas.csv` y `fuentes.json`.

## Límites y decisiones pendientes

{limits}

Las siete configuraciones nominales clase–diámetro son grupos descriptivos,
**no siete rodamientos físicos demostrados**. La separación por archivos/carga
evita mezclar ventanas del mismo MAT, pero no prueba independencia de montaje,
defecto o rodamiento entre cargas. Tampoco hay un test nuevo: 3 HP ya fue evaluado.
Autocorrelaciones son descriptivas; no se calculan p-valores tratando ventanas
solapadas como réplicas independientes.

## Siguiente paso recomendado, antes de reentrenar

1. Congelar la selección explícita de claves y estos hashes; conservar V1 intacta.
2. Resolver el alcance de canales: mantener DE/FE con limitación declarada y
   comparación de sensibilidad DE-solo, o establecer DE-solo como referencia
   principal y DE/FE como ampliación. No fabricar un tercer eje, ni eliminar
   automáticamente los registros redundantes por favorecer métricas.
3. Predefinir tratamiento de advertencias 177/214/215: resultado principal con
   datos intactos y, si se justifica, sensibilidad secundaria de calidad definida
   antes del entrenamiento. Sin corregir según aciertos/errores de test.
4. Fijar evaluación externa por carga y selección interna separada; declarar
   retrospectividad, usar métricas por archivo/carga y no prometer motores nuevos.
5. Sólo después habilitar los experimentos V2 del plan maestro. Esta auditoría
   no autoriza por sí misma nuevas exclusiones, cambios de etiquetas o filtros.

## Cómo comprobarlo después

Leer `registro_pruebas.csv` (IDs estables), `resumen_auditoria.json` y
`README_REPRODUCIBILIDAD.md`. Las {sum(t['status']=='PASS' for t in tests)} pruebas
PASS no anulan las advertencias/limitaciones. Los nueve tests sintéticos incluyen
casos negativos: clave auxiliar, canales invertidos, inicio incorrecto y archivo
alterado. `MANIFEST_SHA256.csv` sella las salidas; `entradas_sha256.csv` identifica
los datos y códigos, con comprobación antes/después sin cambios.
'''


if __name__=='__main__':
    main()
