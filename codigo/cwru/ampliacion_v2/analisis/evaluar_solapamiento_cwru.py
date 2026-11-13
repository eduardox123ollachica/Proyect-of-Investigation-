"""Reevaluación estructural del solapamiento; no regenera datos ni entrena modelos.

Desde la raíz: python -B codigo/cwru/ampliacion_v2/analisis/evaluar_solapamiento_cwru.py
Requiere NumPy. Cada repetición usa --output con un directorio nuevo.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from datetime import datetime, timezone
import numpy as np


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()


def read_csv(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def write_csv(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)


def count(n,hop,size=4096):
    return max(0,(n-size)//hop+1)


def corr(a,b):
    a=a-a.mean();b=b-b.mean();denom=np.sqrt(np.sum(a*a)*np.sum(b*b))
    return float(np.sum(a*b)/denom) if denom>0 else None


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[4])
    parser.add_argument('--output',type=Path,default=Path('articulo/ampliacion_cwru/analisis_solapamiento/revision_2026-10-22'))
    args=parser.parse_args();root=args.root.resolve();out=(root/args.output).resolve()
    allowed=root/'articulo/ampliacion_cwru/analisis_solapamiento'
    if out.exists() or not out.is_relative_to(allowed.resolve()) or out==allowed.resolve():
        raise ValueError('Se requiere una subcarpeta nueva de articulo/ampliacion_cwru/analisis_solapamiento')
    out.mkdir(parents=True)
    inventory=root/'articulo/ampliacion_cwru/auditoria/revision_2026-10-22_validada/inventario_archivos.csv'
    data=root/'datos/cwru/procesado/dual_12khz_w4096'
    fd=data/'features_14'
    inputs=[inventory,Path(__file__).resolve(),fd/'configuracion_features.json']
    for split in ('train','validation','test'):
        inputs.extend([fd/f'X_features_{split}.npy',data/f'metadata_{split}.csv'])
    hashes={p:sha(p) for p in inputs}
    rows=read_csv(inventory)
    names=json.loads((fd/'configuracion_features.json').read_text(encoding='utf-8'))['feature_order']
    profiles=[('50pct',2048,50,0),('25pct',3072,25,0),('0pct',4096,0,0),('0pct_gap44',4140,0,44)]
    by_file=[];totals=[];by_class=[]
    for name,hop,overlap,gap in profiles:
        for r in rows:
            by_file.append(dict(profile=name,hop_samples=hop,overlap_percent=overlap,gap_output_samples=gap,
                                file=r['file'],file_id=r['file_id'],split=r['split'],label=r['class'],
                                valid_samples=int(r['valid_samples_12khz']),
                                windows=count(int(r['valid_samples_12khz']),hop),
                                shared_output_samples_adjacent=max(0,4096-hop),
                                shared_raw_fir_support_samples_adjacent=max(0,4*(4096-1)+177-4*hop)))
        counts={s:sum(r['windows'] for r in by_file if r['profile']==name and r['split']==s)
                for s in ('train','validation','test')}
        totals.append(dict(profile=name,hop_samples=hop,overlap_percent=overlap,gap_output_samples=gap,
                           decision_stride_ms=1000*hop/12000,**counts,total=sum(counts.values())))
        for s in ('train','validation','test'):
            for cls in ('N','IR','OR'):
                by_class.append(dict(profile=name,split=s,label=cls,
                     windows=sum(r['windows'] for r in by_file if r['profile']==name and r['split']==s and r['label']==cls)))
    write_csv(out/'recuento_por_archivo.csv',by_file)
    write_csv(out/'recuento_por_clase.csv',by_class)
    write_csv(out/'comparacion_recuentos.csv',totals)
    correlations=[];checks=[];zero_count=0;fifty_count=0
    for split in ('train','validation','test'):
        features=np.load(fd/f'X_features_{split}.npy')
        metadata=read_csv(data/f'metadata_{split}.csv')
        assert len(features)==len(metadata)
        for r in [r for r in rows if r['split']==split]:
            ix=[i for i,m in enumerate(metadata) if m['file']==r['file']]
            assert len(ix)==count(int(r['valid_samples_12khz']),2048)
            selected=ix[::2];zero_count+=len(selected);fifty_count+=len(ix)
            assert len(selected)==count(int(r['valid_samples_12khz']),4096)
            starts=[int(metadata[i]['start_sample_12khz_valid']) for i in selected]
            assert starts==list(range(0,len(selected)*4096,4096))
            checks.append(dict(id='OV01_'+r['file_id'],status='PASS',file=r['file'],
                               observed=len(selected),criterion='Selección 0,2,4... equivale exactamente al hop4096; inicios disjuntos'))
            for profile,indices in [('50pct',ix),('0pct',selected)]:
                values=features[indices]
                for k,name in enumerate(names):
                    v=values[:,k]
                    value=corr(v[:-1],v[1:]) if len(v)>=3 else None
                    correlations.append(dict(profile=profile,file_id=r['file_id'],file=r['file'],split=split,
                          feature=name,windows=len(v),adjacent_pairs=max(0,len(v)-1),
                          pearson_lag1=value,absolute_pearson_lag1=None if value is None else abs(value),
                          included_in_descriptive_summary=len(v)-1>=10 and value is not None))
    write_csv(out/'autocorrelacion_por_archivo.csv',correlations)
    aggregate=[]
    for profile in ('50pct','0pct'):
        for feature in names:
            values=[r['absolute_pearson_lag1'] for r in correlations if r['profile']==profile and
                    r['feature']==feature and r['included_in_descriptive_summary']]
            assert values
            aggregate.append(dict(profile=profile,feature=feature,files_included=len(values),
                                   median_absolute_pearson=float(np.median(values)),
                                   q25_absolute_pearson=float(np.quantile(values,.25)),
                                   q75_absolute_pearson=float(np.quantile(values,.75)),
                                   maximum_absolute_pearson=float(np.max(values))))
    write_csv(out/'autocorrelacion_resumen.csv',aggregate)
    assert count(4095,4096)==0 and count(4096,4096)==1 and count(8192,4096)==2
    assert fifty_count==1380 and zero_count==691
    # Soporte inclusivo de FIR177 y diezmado4: gap43 aún comparte una muestra.
    assert 4*(4096-1)+177-4*(4096+43)==1
    assert max(0,4*(4096-1)+177-4*(4096+44))==0
    checks.extend([dict(id='OV02',status='PASS',file='sintetico',observed='0/1/2',criterion='Conteo en longitudes límite 4095,4096,8192'),
                   dict(id='OV03',status='PASS',file='todos',observed='1380 -> 691',criterion='Concordancia del subconjunto sin solapamiento y fórmula independiente'),
                   dict(id='OV04',status='PASS',file='soporte_fir',observed='gap43=1; gap44=0 muestras compartidas',criterion='Cálculo de soporte FIR177 original, decimación4')])
    changed=[p.relative_to(root).as_posix() for p,h in hashes.items() if sha(p)!=h]
    assert not changed
    checks.append(dict(id='OV05',status='PASS',file='entradas',observed=len(hashes),criterion='SHA256 de todas las entradas antes/después sin cambios'))
    write_csv(out/'pruebas.csv',checks)
    write_csv(out/'entradas_sha256.csv',[dict(path=p.relative_to(root).as_posix(),sha256=h) for p,h in hashes.items()])
    overview={p:float(np.median([r['median_absolute_pearson'] for r in aggregate if r['profile']==p])) for p in ('50pct','0pct')}
    summary=dict(created_utc=datetime.now(timezone.utc).isoformat(),profiles=totals,
                 median_across_14_feature_medians_absolute_lag1=overview,
                 correlation_scope='Descriptivo por archivo; >=10 pares; mediana igual peso por archivo. No prueba de independencia.',
                 unit_of_physical_replication_unchanged=True,
                 zero_overlap_via_existing_windows_every_other_within_file=True,
                 no_dataset_regeneration=True,no_training=True,no_manuscript_change=True,
                 inputs_unchanged=True,checks_passed=len(checks),numpy=np.__version__,python=sys.version)
    (out/'resumen.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    table='\n'.join(f"| {r['overlap_percent']} %"+(' + margen44' if r['gap_output_samples'] else '')+
                    f" | {r['hop_samples']} | {r['train']} | {r['validation']} | {r['test']} | {r['total']} |" for r in totals)
    (out/'INFORME_SOLAPAMIENTO.md').write_text(f'''# Reevaluación del solapamiento CWRU — 22-10-2026

## Resultado y alcance

Quitar el solapamiento evita repetir muestras de salida entre ventanas adyacentes,
pero no demuestra independencia estadística ni crea motores/rodamientos nuevos.
La separación entre archivos/cargas debe seguir realizándose antes del ventaneo.
El 50% actual no produce por sí mismo fuga entre train/test: sus MAT ya son disjuntos.
El cambio se justifica por menor redundancia y claridad de evaluación, no para
reparar una mezcla entre conjuntos que esta auditoría no detectó.

Recuentos exactos desde longitudes válidas a 12 kHz; N=4096, mismo inicio y FIR:

| Solapamiento | Hop | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|---:|
{table}

El perfil 0% se comprobó como selección 0,2,4... de las ventanas existentes de
cada archivo, sin generar arrays nuevos ni entrenar. El registro174 pasa de6 a3
ventanas: quitar redundancia también disminuye ejemplos, sin alterar el número
de adquisiciones. No ajustar N sólo para inflar el recuento sin justificación física.

## Correlación descriptiva, no certificado de independencia

Se reutilizaron las14 características guardadas, verificadas en la auditoría,
y se calculó Pearson de cada característica entre ventanas consecutivas dentro
del mismo MAT. Con 0% el desplazamiento físico se duplica; la comparación cambia
simultáneamente coincidencia de muestras y separación temporal.

Se guardaron todas las estimaciones, pero los resúmenes exigen >=10 pares por
archivo para reducir la inestabilidad de secuencias muy cortas; ese umbral es una
regla descriptiva, no garantía de precisión. Cada archivo pesa igual; no se
concatenaron archivos ni se trataron ventanas como réplicas físicas independientes.
No hay p-valores, selección por accuracy, ni un supuesto tamaño muestral efectivo.
Correlación cero tampoco demostraría independencia o ausencia de parentesco.

Mediana de las14 medianas por característica de |r|: 50% = {overview['50pct']:.4f};
0% = {overview['0pct']:.4f}. Es un resumen exploratorio, no una métrica universal
del dataset: revisar `autocorrelacion_resumen.csv` por característica y los casos
individuales, que pueden subir o bajar por periodicidad/variabilidad finita.

## Matiz del filtro

Con el FIR177 aplicado antes de diezmar por4, ventanas de salida sin solapamiento
todavía pueden depender de173 muestras originales compartidas en sus soportes.
Un margen de44 muestras a12 kHz elimina ese cruce de soporte para este FIR finito.
No elimina la correlación de la maquinaria ni la dependencia de un mismo registro.
Es un cálculo del soporte, no una prueba de ruido blanco ni una recomendación de
usar44 como distancia de decorrelación. Entre MAT distintos no se comparte filtro.
El perfil con margen sólo tiene recuentos teóricos; no se extrajeron sus características.

## Propuesta, aún no adoptada ni ejecutada

Para una referencia principal sencilla, evaluar0% en train/validation/test y
reservar50% como sensibilidad predefinida, conservando los mismos grupos por carga,
semillas y reglas de ajuste. Otra posibilidad es entrenamiento50% y evaluación0%,
pero respondería una pregunta distinta y debe documentarse. No escoger el esquema
según cuál dé mayor métrica de prueba. Con0% hay691 ventanas, no691 ensayos nuevos.

El tiempo de una ventana sigue siendo341.333ms. La cadencia pasa de170.667ms a
341.333ms; esto no duplica el tiempo de inferencia de una ventana. La evaluación
de muestras no solapadas puede coexistir con operación desplegada a50% si se
declara y se evalúa separadamente la cadencia de decisiones.

No recomendaría repartir aleatoriamente ventanas sin solapamiento del mismo MAT
entre train/test para aumentar datos. No resuelve dependencia de fuente y reduciría
el alcance de la prueba a segmentos de registros ya representados en entrenamiento.

## Fundamento y limitaciones externas

Vieira et al., *Towards a more realistic evaluation of machine learning models
for bearing fault diagnosis*, MSSP258 (2026),114640,
https://doi.org/10.1016/j.ymssp.2026.114640 (texto de autor: https://arxiv.org/abs/2509.22267),
contrasta separación por segmentos/condiciones con separación por rodamiento.
Respalda distinguir control temporal de independencia física; no establece
los recuentos ni los valores de correlación locales, calculados aquí.
FIR/decimación: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.upfirdn.html.

## Validación y reproducción

{len(checks)} comprobaciones aprobadas. Entradas intactas, sin regeneración de datos,
sin entrenamiento ni cambios de manuscrito/protocolo. `entradas_sha256.csv` fija
código y archivos usados; `MANIFEST_SHA256.csv` sella las salidas excepto a sí mismo.
Cada repetición exige un directorio nuevo:

```powershell
python -B codigo/cwru/ampliacion_v2/analisis/evaluar_solapamiento_cwru.py --output articulo/ampliacion_cwru/analisis_solapamiento/repeticion
```

Se requiere NumPy. En esta ejecución se usó el intérprete/runtime y los paquetes
científicos ya instalados registrados en la auditoría principal; no se instaló nada.
La adopción de0% o un margen queda pendiente: este informe no sustituye la decisión
del usuario ni modifica las referencias históricas V1.
''',encoding='utf-8')
    outputs=[p for p in sorted(out.iterdir()) if p.is_file()]
    write_csv(out/'MANIFEST_SHA256.csv',[dict(path=p.name,sha256=sha(p)) for p in outputs])
    print(json.dumps(summary,ensure_ascii=True,indent=2))


if __name__=='__main__':main()
