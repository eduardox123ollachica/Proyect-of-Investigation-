"""Copia de trabajo autónoma; nunca escribe en D: ni sobrescribe una entrega."""
from pathlib import Path
import csv
import hashlib
import shutil
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[4]
SOURCE=Path(r'D:\STM32IDE\Prueba 2')
PACKAGE=ROOT/'despliegue_stm32/cwru_extraccion_prueba2'
TARGET=PACKAGE/'Prueba_2_CWRU_Extraccion'
NAME=TARGET.name


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def main():
    if PACKAGE.exists():raise FileExistsError('No se sobrescribe una entrega existente')
    PACKAGE.mkdir(parents=True)
    source_files=sorted(p for p in SOURCE.rglob('*') if p.is_file())
    with (PACKAGE/'ORIGINAL_D_SHA256.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,['path','bytes','sha256']);w.writeheader()
        for p in source_files:w.writerow(dict(path=p.relative_to(SOURCE).as_posix(),bytes=p.stat().st_size,sha256=sha(p)))
    shutil.copytree(SOURCE,TARGET,ignore=shutil.ignore_patterns('Debug','Release','*.launch','__pycache__'))
    fw=ROOT/'modelos/cwru/canonico_seed42/firmware'
    for name in ('cwru_feature_extraction.c','cmd_cwru_feat.c'):
        shutil.copy2(fw/name,TARGET/'Src'/name)
    for name in ('cwru_feature_extraction.h','cwru_pipeline_config.h'):
        shutil.copy2(fw/name,TARGET/'Inc'/name)
    # Migración mecánica de identidad/rutas; no cambia el original.
    p=TARGET/'.project'
    text=p.read_text(encoding='utf-8').replace('<name>Prueba 1</name>',f'<name>{NAME}</name>')
    p.write_text(text,encoding='utf-8')
    p=TARGET/'.cproject'
    text=p.read_text(encoding='utf-8').replace('Prueba 1',NAME).replace('Prueba 2',NAME)
    tree=ET.fromstring(text)
    debug=next(c for c in tree.findall('./storageModule/cconfiguration') if '.debug.' in c.attrib['id'])
    debug_chain=debug.find('.//toolChain')
    debug_sources=debug.find('.//sourceEntries')
    keep={'main.c','uart_commands.c','cmd_cwru_feat.c','cwru_feature_extraction.c',
          'stm32f4xx_hal.c','stm32f4xx_hal_cortex.c','stm32f4xx_hal_gpio.c','stm32f4xx_hal_rcc.c',
          'stm32f4xx_hal_rcc_ex.c','stm32f4xx_hal_uart.c','syscalls.c','sysmem.c','system_stm32f4xx.c'}
    exclude='|'.join(sorted(p.name for p in (TARGET/'Src').iterdir() if p.is_file() and p.name not in keep))
    import copy
    for cfg in tree.findall('./storageModule/cconfiguration'):
        chain=cfg.find('.//toolChain')
        for compiler_suffix in ('tool.c.compiler','tool.cpp.compiler'):
            tool=next(e for e in chain.findall('tool') if e.get('superClass','').endswith(compiler_suffix))
            reference=next(e for e in debug_chain.findall('tool') if e.get('superClass','').endswith(compiler_suffix))
            for suffix in ('includepaths','definedsymbols'):
                target_option=next(e for e in tool.findall('option') if e.get('superClass','').endswith('.'+suffix))
                ref=next(e for e in reference.findall('option') if e.get('superClass','').endswith('.'+suffix))
                children=[copy.deepcopy(child) for child in ref]
                for child in list(target_option):target_option.remove(child)
                for child in children:target_option.append(child)
        # Release recibe los mismos agregados CMSIS y exclusiones que Debug.
        sources=cfg.find('.//sourceEntries')
        if cfg is not debug:
            for e in list(sources):sources.remove(e)
            for e in debug_sources:sources.append(copy.deepcopy(e))
        next(e for e in sources if e.get('name')=='Src').set('excluding',exclude)
        # Sólo extracción: no enlazar TFLM ni modelos históricos.
        for tool in chain.findall('tool'):
            if not tool.get('superClass','').endswith(('tool.c.linker','tool.cpp.linker')):continue
            prefix=tool.get('superClass')+'.option.'
            for e in list(tool):
                if e.tag=='option' and e.get('superClass','').endswith('.libraries'):tool.remove(e)
            for suffix,value,value_type in [('libraries',None,'libs'),('librarypaths',None,'libPaths')]:
                if suffix=='libraries':
                    option=ET.SubElement(tool,'option',{'id':prefix+suffix+'.cwru', 'superClass':prefix+suffix,'valueType':value_type,'name':'Libraries (-l)'})
                    ET.SubElement(option,'listOptionValue',{'builtIn':'false','value':'m'})
            # Copia mecánica de las opciones float/nano desde el linker Debug C++.
            ref_link=next(e for e in debug_chain.findall('tool') if e.get('superClass','').endswith('tool.cpp.linker'))
            for ref_opt in list(ref_link.findall('option')):
                sc=ref_opt.get('superClass','')
                if not any(s in sc for s in ('floatprintf','nanolib','systemcalls')):continue
                replacement=sc.replace('tool.cpp.linker','tool.c.linker') if tool.get('superClass','').endswith('tool.c.linker') else sc
                if not any(e.get('superClass')==replacement for e in tool.findall('option')):
                    child=copy.deepcopy(ref_opt);child.set('superClass',replacement);child.set('id',replacement+'.cwru')
                    tool.append(child)
    ET.indent(tree,space='\t')
    p.write_text('<?xml version="1.0" encoding="UTF-8"?>\n<?fileVersion 4.0.0?>\n'+ET.tostring(tree,encoding='unicode')+'\n',encoding='utf-8')
    launch=(SOURCE/'Prueba 2.launch').read_text(encoding='utf-8').replace('Prueba 2',NAME).replace('Prueba 1',NAME)
    (TARGET/(NAME+'.launch')).write_text(launch,encoding='utf-8')
    print(f'Copia creada: {TARGET}; {len(source_files)} archivos originales inventariados')


if __name__=='__main__':main()
