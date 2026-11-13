"""Exporta 42 ventanas de desarrollo (0/1/2 HP), nunca la carga externa 3 HP."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'despliegue_stm32/cwru_extraccion_prueba2/Validacion_PC'
VIEWS=ROOT/'datos/cwru/particiones/ampliacion_v2/por_carga_2026-10-22'


def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def main():
    if (OUT/'ventanas_desarrollo.npz').exists():raise FileExistsError('No sobrescribir vectores')
    index=read(VIEWS/'indice_ventanas_fuente.csv')
    allowed={r['global_window_id'] for r in read(VIEWS/'indices_ventanas_externo.csv')
             if r['outer_load_hp']=='3' and r['role']=='development'}
    selected=[]
    for file_id in sorted({r['file_id'] for r in index if r['global_window_id'] in allowed},key=int):
        candidates=[r for r in index if r['file_id']==file_id and r['global_window_id'] in allowed]
        candidates.sort(key=lambda r:int(r['window_index_in_file']))
        selected.extend([candidates[0],candidates[-1]])
    assert len(selected)==42 and all(int(r['load_hp']) in (0,1,2) for r in selected)
    cache={};windows=[];features=[];metadata=[];sources={}
    for i,r in enumerate(selected):
        for key in ('source_windows','source_features'):
            p=ROOT/r[key]
            if p not in cache:cache[p]=np.load(p,mmap_mode='r');sources[r[key]]=sha(p)
        row=int(r['source_row'])
        w=np.array(cache[ROOT/r['source_windows']][row],dtype=np.float32)
        f=np.array(cache[ROOT/r['source_features']][row],dtype=np.float64)
        assert w.shape==(4096,2) and f.shape==(14,) and np.isfinite(w).all() and np.isfinite(f).all()
        windows.append(w);features.append(f)
        metadata.append(dict(bundle_index=i,**r,window_payload_sha256=hashlib.sha256(w.astype('<f4').tobytes()).hexdigest()))
    np.savez_compressed(OUT/'ventanas_desarrollo.npz',windows=np.stack(windows),features=np.stack(features))
    with (OUT/'metadata_ventanas.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,list(metadata[0]));w.writeheader();w.writerows(metadata)
    with np.load(OUT/'ventanas_desarrollo.npz') as data:
        np.testing.assert_array_equal(data['windows'],np.stack(windows))
        np.testing.assert_array_equal(data['features'],np.stack(features))
    (OUT/'PROCEDENCIA_VECTORES.json').write_text(json.dumps(dict(status='DEVELOPMENT_VECTORS_NOT_HARDWARE_RESULTS',
       outer_reserved_load_hp=3,loads_used=[0,1,2],windows=42,mat_files=21,
       selection='Primera y última ventana de cada MAT de desarrollo; sin selección por métricas',
       sampling_rate_hz=12000,window_samples=4096,channels=['DE','FE'],
       preprocessing='Ya filtradas y remuestreadas offline; el STM32 no ejecuta el FIR 48->12 kHz',
       expected_features='float64 auditadas; comparar con salida float32/CMSIS',
       source_sha256=sources,npz_sha256=sha(OUT/'ventanas_desarrollo.npz')),
       ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('42 ventanas exportadas y verificadas; ninguna de la carga 3 HP')


if __name__=='__main__':main()
