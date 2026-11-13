"""ZIP final de archivos sellados, sin modificar el paquete ni el original."""
from pathlib import Path
import csv
import hashlib
import json
import zipfile

ROOT=Path(__file__).resolve().parents[4]
BASE=ROOT/'despliegue_stm32/cwru_extraccion_prueba2'
ZIP=ROOT/'despliegue_stm32/Prueba_2_CWRU_Extraccion_lista.zip'


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def main():
    if ZIP.exists():raise FileExistsError('No sobrescribir ZIP existente')
    with (BASE/'MANIFEST_SHA256.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    for row in rows:
        p=(BASE/row['path']).resolve()
        if not p.is_relative_to(BASE) or sha(p)!=row['sha256']:raise ValueError('Sello inválido: '+row['path'])
    files=[BASE/r['path'] for r in rows]+[BASE/'MANIFEST_SHA256.csv']
    with zipfile.ZipFile(ZIP,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files:z.write(p,arcname=BASE.name+'/'+p.relative_to(BASE).as_posix())
    with zipfile.ZipFile(ZIP) as z:
        bad=z.testzip()
        if bad:raise ValueError('CRC incorrecto: '+bad)
        assert len(z.namelist())==len(files)
    summary=dict(zip=ZIP.name,bytes=ZIP.stat().st_size,sha256=sha(ZIP),files=len(files),
                 crc_test='PASS',package_manifest_verified=True,hardware_flashed=False)
    ZIP.with_suffix('.sha256.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
