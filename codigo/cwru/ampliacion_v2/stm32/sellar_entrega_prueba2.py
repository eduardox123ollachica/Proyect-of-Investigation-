"""Cierra evidencia de la entrega: original intacto, tests, memoria y sello."""
from pathlib import Path
import csv
import hashlib
import io
import importlib.util
import json
import unittest
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[4]
BASE=ROOT/'despliegue_stm32/cwru_extraccion_prueba2'
PROJECT=BASE/'Prueba_2_CWRU_Extraccion'
ORIGINAL=Path(r'D:\STM32IDE\Prueba 2')


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def write_csv(p,rows):
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    if (BASE/'MANIFEST_SHA256.csv').exists():raise FileExistsError('No sobrescribir un sello existente')
    with (BASE/'ORIGINAL_D_SHA256.csv').open(encoding='utf-8-sig',newline='') as f:original=list(csv.DictReader(f))
    all_original={p.relative_to(ORIGINAL).as_posix() for p in ORIGINAL.rglob('*') if p.is_file()}
    if all_original!={r['path'] for r in original}:raise ValueError('Cambió el inventario original de D:')
    changes=[]
    for r in original:
        if sha(ORIGINAL/r['path'])!=r['sha256']:raise ValueError('Original modificado: '+r['path'])
        target=PROJECT/r['path']
        changes.append(dict(path=r['path'],status='OMITIDO_EN_COPIA' if not target.exists() else
                       'COPIADO_SIN_CAMBIO' if sha(target)==r['sha256'] else 'MODIFICADO_EN_COPIA',
                       original_sha256=r['sha256'],copy_sha256=sha(target) if target.exists() else ''))
    known={r['path'] for r in original}
    for p in sorted(PROJECT.rglob('*')):
        if p.is_file() and p.relative_to(PROJECT).as_posix() not in known:
            changes.append(dict(path=p.relative_to(PROJECT).as_posix(),status='NUEVO_EN_COPIA',original_sha256='',copy_sha256=sha(p)))
    write_csv(BASE/'CAMBIOS_RESPECTO_ORIGINAL.csv',changes)
    canonical=ROOT/'modelos/cwru/canonico_seed42/firmware/cwru_feature_extraction.c'
    assert sha(canonical)==sha(PROJECT/'Src/cwru_feature_extraction.c')
    config=ET.parse(PROJECT/'.cproject');ids=[]
    for c in config.findall('./storageModule/cconfiguration'):
        name=c.find('storageModule').get('name')
        assert len(c.findall('.//sourceEntries/entry'))==6
        for tool in c.findall('.//tool'):
            if tool.get('superClass','').endswith('tool.c.compiler'):
                include=next(e for e in tool.findall('option') if e.get('superClass','').endswith('.includepaths'))
                assert len(include)>=6
            if tool.get('superClass','').endswith(('tool.c.linker','tool.cpp.linker')):
                assert any('_printf_float' in e.get('value','') for e in tool.iter('listOptionValue'))
    source=(PROJECT/'Src/main.c').read_text(encoding='utf-8')
    assert 'discard_overlong_frame' in source and '#if CWRU_EXTRACTION_ONLY' in source
    spec=importlib.util.spec_from_file_location('client_cwru',BASE/'Validacion_PC/validar_extraccion.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(module.TransportTests))
    assert result.wasSuccessful()
    (BASE/'PRUEBAS_TRANSPORTE.txt').write_text(stream.getvalue()+'\nSimulación del transporte; no prueba de hardware.\n',encoding='utf-8')
    reports=json.loads((BASE/'COMPILACION_VERIFICADA.json').read_text(encoding='utf-8'))
    memory=[]
    for r in reports:
        name=r['configuration'];text=(BASE/'Build_verificado'/name/'Prueba_2_CWRU_Extraccion.map').read_text(errors='replace')
        import re
        sections={}
        for section in ('.data','.bss','._user_heap_stack'):
            found=re.search(r'^'+re.escape(section)+r'\s+(?:\n\s*)?(0x[0-9a-f]+)\s+(0x[0-9a-f]+)',text,re.M)
            if found:sections[section]=int(found.group(2),16)
        assert r['flash_load_bytes']<512*1024 and r['ram_static_including_linker_reservations_bytes']<128*1024
        memory.append(dict(configuration=name,map_sections_bytes=sections,
                           runtime_high_watermark_measured=False,static_link_fits=True))
    summary=dict(status='PREPARADO_COMPILADO_NO_PROGRAMADO',original_d_files_verified_unchanged=len(original),
                 original_d_path=str(ORIGINAL),project_name=PROJECT.name,transport_tests_passed=result.testsRun,
                 extractor_c_identical_to_canonical=True,
                 source_change_counts={s:sum(r['status']==s for r in changes) for s in sorted({r['status'] for r in changes})},
                 profiles=reports,memory=memory,development_vectors=42,
                 external_load3_included=False,firmware_flashed=False,pc_stm32_equivalence_measured=False)
    (BASE/'RESUMEN_VERIFICACION.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    # Copias de los scripts de preparación: trazabilidad, no ejecución automática.
    import shutil
    scripts=BASE/'Herramientas_preparacion';scripts.mkdir(exist_ok=True)
    for p in Path(__file__).parent.glob('*.py'):shutil.copy2(p,scripts/p.name)
    files=sorted(p for p in BASE.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    write_csv(BASE/'MANIFEST_SHA256.csv',[dict(path=p.relative_to(BASE).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in files])
    print(json.dumps(summary,ensure_ascii=True,indent=2))


if __name__=='__main__':main()
