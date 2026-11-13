"""Copia aislada del banco canónico: preparar sin escribir D, build solo por CLI.

No abre COM, programa Flash, entrena o instala archivos en el proyecto origen.
El nuevo banco conserva las 17 configuraciones originales y usa recursos de
ocho modelos ya congelados, con POST de características sin reextraer RAW.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import queue
import re
import shutil
import subprocess
import threading
import time
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[4]
HERE=Path(__file__).resolve().parent
NAME='Prueba_2_CWRU_Extraccion'
SOURCE=Path('D:/STM32IDE')/NAME
IDE=Path('D:/STM32CubeIDE_2.2.0/STM32CubeIDE')
OUT=ROOT/'despliegue_stm32/canonicos_42_180mhz_2026-10-27'
RESOURCES=OUT/'recursos'
PROJECT=OUT/'copia_aislada'/NAME
PREPARATION=OUT/'PREPARACION_PROYECTO.json'
MANIFEST=OUT/'MANIFEST_PREPARACION_PROYECTO_SHA256.json'
BASE='Release_180MHz_TotalV2_SR'
CONFIG='Release_180MHz_Canon42_SR'
PRIVATE='CWRU_Canon8'
BUILD_TIMEOUT=1800
LEGACY_HELPER=ROOT/'codigo/cwru/ampliacion_v2/costo_total/preparar_total.py'
SIMPLE_FILES=('cwru_total.h','cwru_total.c','cwru_total_commands.inc','cwru_total_nn.h',
    'cwru_total_nn.cpp','cwru_total_debug_log.cc')
CATALOG_FILES=('cwru_total_catalog.h','cwru_total_catalog.cpp')
POST_FILES=('cwru_canon_post.h','cwru_canon_post_math.inc','cwru_canon_post_commands.inc')
LAUNCH=NAME+'_Canon42_SR_180MHz_SinRecompilar.launch'


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')


def helper():
    spec=importlib.util.spec_from_file_location('canon42_legacy_xml_helpers',LEGACY_HELPER)
    require(spec is not None and spec.loader is not None,'Helpers XML no disponibles')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def confined(base,relative):
    relative=Path(relative);base=Path(base).resolve()
    require(not relative.is_absolute() and relative.parts and '..' not in relative.parts,'Ruta relativa insegura')
    result=(base/relative).resolve()
    require(result!=base and result.is_relative_to(base),'Ruta fuera del directorio autorizado')
    return result


def copy_exclusive(source,target):
    source,target=Path(source),Path(target)
    require(not target.exists(),'No sobrescribir una copia existente: '+str(target))
    before=sha(source);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    require(sha(source)==sha(target)==before,'Origen cambió durante la copia: '+str(source))
    return before


def skip_path(relative,configuration_names=()):
    """Filtro del preparador original, conservando aquí el TotalV2 histórico."""
    parts=Path(relative).parts
    excluded=set(configuration_names)|{'.git','.metadata','__pycache__','.cache','ipch','build'}
    return any(part in excluded or part.startswith(('Debug','Release')) for part in parts) or (
        Path(relative).suffix.lower() in {'.o','.d','.elf','.map','.su','.cyclo','.list','.log','.bin','.hex','.srec'})


def source_inventory(source,configuration_names):
    source=Path(source).resolve();result={}
    for path in sorted(source.rglob('*')):
        if not path.is_file() or skip_path(path.relative_to(source),configuration_names):continue
        require(path.resolve().is_relative_to(source),'Enlace de fuente fuera del proyecto')
        result[path.relative_to(source).as_posix()]=sha(path)
    require('.cproject' in result and '.project' in result,'Proyecto CDT incompleto')
    return result


def authenticate_resources(resources=RESOURCES):
    resources=Path(resources).resolve();manifest=resources/'MANIFEST_RECURSOS_SHA256.json'
    digest,name=manifest.with_suffix('.sha256').read_text(encoding='ascii').split()
    require(name==manifest.name and sha(manifest)==digest,'Sello del catálogo alterado')
    entries=read(manifest);require(isinstance(entries,dict) and entries,'Manifiesto local vacío')
    for relative,expected in entries.items():
        require(sha(confined(resources,relative))==expected,'Recurso del catálogo alterado: '+relative)
    required={'CATALOGO_TOTAL.json','CATALOGO_CANONICO.json',*CATALOG_FILES}
    require(required<=set(entries),'Faltan recursos del banco canónico')
    catalog=read(resources/'CATALOGO_TOTAL.json')
    require(catalog['count']==len(catalog['representatives'])==8,'Se requieren ocho modelos canónicos')
    rows=catalog['representatives']
    require([r['id'] for r in rows]==list(range(8)) and len({r['representative_id'] for r in rows})==8,
        'IDs canónicos incompletos o duplicados')
    require(re.search(r'^#define\s+CWRUTV_COUNT\s+8U\s*$',(resources/CATALOG_FILES[0]).read_text(encoding='utf-8'),re.M),
        'Cabecera no declara ocho descriptores')
    return catalog,dict(entries),dict(manifest_sha256=digest,manifest_path=str(manifest))


def add_configuration(p,root,settings,old):
    require(len(old)==17 and BASE in old and CONFIG not in old,'Se esperan 17 configuraciones actuales y SR')
    node=copy.deepcopy(old[BASE]);p.unique_ids(node,'.canon42_sr1')
    for element in node.iter():
        for key,value in list(element.attrib.items()):
            if key=='name' and value==BASE:element.set(key,CONFIG)
            elif key=='buildPath' and value.endswith('/'+BASE):element.set(key,value[:-len(BASE)]+CONFIG)
            elif key=='value' and value=='../CWRU_TotalV2':element.set(key,'../'+PRIVATE)
    entries=node.find('.//sourceEntries');require(entries is not None,'Configuración sin sourceEntries')
    replaced=0
    for entry in entries:
        if entry.get('name')=='CWRU_TotalV2':entry.set('name',PRIVATE);replaced+=1
    require(replaced==1 and any(entry.get('name')=='CWRU_TotalSR' for entry in entries),'Fuente privada/SR no definida')
    old_ids={n.get('id') for cfg in old.values() for n in cfg.iter() if n.get('id')}
    require(not old_ids.intersection(n.get('id') for n in node.iter() if n.get('id')),'IDs CDT colisionan')
    settings.append(node)
    return node


def patch_nn(text):
    old='constexpr uint32_t kModelCount = 32U;'
    require(text.count(old)==1,'Conteo NN antiguo ambiguo')
    return text.replace(old,'constexpr uint32_t kModelCount = CWRUTV_COUNT;')


def patch_total(text):
    require('cwru_canon_post_math.inc' not in text,'POST ya incluido')
    return text.rstrip()+'\n\n#include "cwru_canon_post_math.inc"\n'


def patch_commands(text):
    old='"CWRUTOTAL_INFO,1,32,12000,4096,2,'
    require(text.count(old)==1,'INFO anterior ambiguo')
    text=text.replace(old,'"CWRUTOTAL_INFO,2,%u,12000,4096,2,')
    marker='            (unsigned long)HAL_RCC_GetHCLKFreq(),'
    require(text.count(marker)==1,'Argumentos INFO ambiguos')
    text=text.replace(marker,'            (unsigned)CWRUTV_COUNT,(unsigned long)HAL_RCC_GetHCLKFreq(),')
    declaration='void CWRUTOTAL_Command(char *command,char *out,size_t cap)'
    require(text.count(declaration)==1,'Entrada de comandos ambigua')
    text=text.replace(declaration,'#include "cwru_canon_post_commands.inc"\n\n'+declaration)
    guard='    if(command==NULL || out==NULL || cap==0U) return;'
    require(text.count(guard)==1,'Guarda de comando ambigua')
    return text.replace(guard,guard+'\n    if(CWRUCANON_Command(command,out,cap)) return;')


def private_sources(source,resources,templates):
    source,resources,templates=map(Path,(source,resources,templates));result={}
    patches={'cwru_total.c':patch_total,'cwru_total_nn.cpp':patch_nn,'cwru_total_commands.inc':patch_commands}
    for name in SIMPLE_FILES:
        data=(source/'CWRU_TotalV2'/name).read_bytes()
        if name in patches:data=patches[name](data.decode('utf-8-sig')).encode('utf-8')
        result[name]=data
    for name in CATALOG_FILES:result[name]=(resources/name).read_bytes()
    for name in POST_FILES:result[name]=(templates/name).read_bytes()
    require(len(result)==11,'Fuentes privadas incompletas')
    return result


def make_launch(source,node):
    old=Path(source)/(NAME+'_TotalV2_SR_180MHz_SinRecompilar.launch')
    root=ET.parse(old).getroot();attrs={n.get('key'):n for n in root}
    elf=CONFIG+'/'+NAME+'.elf'
    attrs['org.eclipse.cdt.launch.PROGRAM_NAME'].set('value',elf)
    attrs['org.eclipse.cdt.launch.PROJECT_BUILD_CONFIG_ID_ATTR'].set('value',node.get('id'))
    attrs['org.eclipse.cdt.launch.PROJECT_BUILD_CONFIG_AUTO_ATTR'].set('value','false')
    loads=attrs['com.st.stm32cube.ide.mcu.debug.launch.loadList']
    value=json.loads(loads.get('value'))
    for item in value['fItems']:item['fPath']=elf
    loads.set('value',json.dumps(value,separators=(',',':')))
    for key,element in attrs.items():
        if key.endswith('stlink.log_file'):element.set('value',str(PROJECT/CONFIG/'st-link_gdbserver_log.txt'))
    return b'<?xml version="1.0" encoding="UTF-8"?>\n'+ET.tostring(root,encoding='utf-8')+b'\n'


def script_hashes():
    return {p.relative_to(ROOT).as_posix():sha(p) for p in sorted([*HERE.glob('*.py'),LEGACY_HELPER])}


def write_manifest(path,files):
    entries={Path(p).relative_to(ROOT).as_posix():sha(p) for p in sorted(set(files))}
    write(path,entries)
    with Path(path).with_suffix('.sha256').open('x',encoding='ascii') as stream:stream.write(sha(path)+'  '+Path(path).name+'\n')
    return entries


def prepare():
    require(not PROJECT.exists() and not PREPARATION.exists() and not MANIFEST.exists(),'No sobrescribir preparación o copia parcial')
    require(PROJECT.resolve().is_relative_to(OUT.resolve()) and OUT.resolve().is_relative_to(ROOT.resolve()),'Destino fuera del workspace')
    require(not (SOURCE/PRIVATE).exists(),'El origen ya contiene un banco canónico; revisar antes de copiar')
    p=helper();_,_,old=p.configs(SOURCE/'.cproject')
    require(len(old)==17 and BASE in old and CONFIG not in old,'Configuraciones de origen distintas de las auditadas')
    catalog,resource_entries,resource_proof=authenticate_resources(RESOURCES)
    private=private_sources(SOURCE,RESOURCES,HERE)
    before=source_inventory(SOURCE,old)
    elf_hashes={name:sha(SOURCE/name/(NAME+'.elf')) for name in old if (SOURCE/name/(NAME+'.elf')).is_file()}
    PROJECT.mkdir(parents=True,exist_ok=False)
    for relative,digest in before.items():require(copy_exclusive(SOURCE/relative,PROJECT/relative)==digest,'Fuente cambió al copiar')
    root,settings,copied=p.configs(PROJECT/'.cproject')
    legacy={name:hashlib.sha256(ET.tostring(node)).hexdigest() for name,node in copied.items()}
    node=add_configuration(p,root,settings,copied)
    (PROJECT/'.cproject').write_bytes(p.xml_bytes(root))
    fresh=p.configs(PROJECT/'.cproject')[2]
    require(len(fresh)==18 and all(hashlib.sha256(ET.tostring(fresh[name])).hexdigest()==digest for name,digest in legacy.items()),
        'Configuraciones anteriores modificadas')
    target=PROJECT/PRIVATE;target.mkdir()
    for name,data in private.items():
        with (target/name).open('xb') as stream:stream.write(data)
    with (PROJECT/LAUNCH).open('xb') as stream:stream.write(make_launch(SOURCE,node))
    after=source_inventory(SOURCE,old)
    require(before==after and all(sha(SOURCE/name/(NAME+'.elf'))==digest for name,digest in elf_hashes.items()),'Origen D cambió durante preparación')
    snapshot=source_inventory(PROJECT,fresh)
    result=dict(status='CANON42_PROJECT_PREPARED_PENDING_USER_BUILD_AND_ELF_REVIEW',source=str(SOURCE),project=str(PROJECT),
        configuration=CONFIG,previous_configurations=17,total_configurations=18,legacy_config_xml_sha256=legacy,
        model_count=8,catalog_sha256=catalog['catalog_sha256'],resource_manifest=resource_proof,
        resource_sha256=resource_entries,D_source_sha256_before=before,D_source_sha256_after=after,
        D_existing_ELF_sha256=elf_hashes,project_source_sha256=snapshot,source_scripts_sha256=script_hashes(),
        root_post_template_sha256={name:sha(HERE/name) for name in POST_FILES},private_files=sorted(private),
        unchanged_Src_files=True,D_files_modified=False,build_executed=False,serial_COM_opened=False,hardware_flashed=False,
        training_executed=False,raw_reextraction_for_POST=False,POST_route='CWRUTOTAL_POST',build_timeout_seconds=BUILD_TIMEOUT,
        build_attempts_allowed=1,build_runs_in_isolated_workspace=True,physical_validation_pending=True)
    write(PREPARATION,result)
    files=[PREPARATION,*[PROJECT/p for p in snapshot],*[ROOT/p for p in result['source_scripts_sha256']],
        *[HERE/name for name in POST_FILES],RESOURCES/'MANIFEST_RECURSOS_SHA256.json',
        RESOURCES/'MANIFEST_RECURSOS_SHA256.sha256',*[RESOURCES/p for p in resource_entries]]
    write_manifest(MANIFEST,files)
    verify()
    print(json.dumps(dict(status=result['status'],project=str(PROJECT),configuration=CONFIG,
        source_files=len(before),preparation_sha256=sha(MANIFEST),D_files_modified=False,build_executed=False)))
    return result


def verify(after_build=False):
    digest,name=MANIFEST.with_suffix('.sha256').read_text(encoding='ascii').split()
    require(name==MANIFEST.name and sha(MANIFEST)==digest,'Sello de preparación modificado')
    entries=read(MANIFEST)
    metadata=PROJECT/'.settings/language.settings.xml'
    for relative,expected in entries.items():
        path=confined(ROOT,relative)
        if after_build and path==metadata:continue
        require(sha(path)==expected,'Fuente o recurso preparado modificado: '+relative)
    prep=read(PREPARATION);p=helper();_,_,old=p.configs(SOURCE/'.cproject');_,_,new=p.configs(PROJECT/'.cproject')
    require(len(old)==17 and len(new)==18 and CONFIG in new,'Inventario CDT alterado')
    require(source_inventory(SOURCE,old)==prep['D_source_sha256_before'],'Fuentes D cambiaron respecto de la instantánea')
    require(all(sha(SOURCE/name/(NAME+'.elf'))==digest for name,digest in prep['D_existing_ELF_sha256'].items()),'ELF D modificado')
    require(all(hashlib.sha256(ET.tostring(new[name])).hexdigest()==digest for name,digest in prep['legacy_config_xml_sha256'].items()),
        'Configuraciones históricas de la copia cambiaron')
    current=source_inventory(PROJECT,new);expected=dict(prep['project_source_sha256'])
    if after_build:
        current.pop('.settings/language.settings.xml',None);expected.pop('.settings/language.settings.xml',None)
    require(current==expected,'Fuentes nuevas, ausentes o modificadas en la copia preparada')
    require(prep['source_scripts_sha256']==script_hashes(),'Scripts cambiaron tras preparar')
    return prep


def toolchain():
    plugins=IDE/'plugins'
    tools=sorted(plugins.glob('com.st.stm32cube.ide.mcu.externaltools.gnu-tools-for-stm32.*/tools/bin'))
    makes=sorted(plugins.glob('com.st.stm32cube.ide.mcu.externaltools.make.*/tools/bin'))
    require(len(tools)==len(makes)==1,'Toolchain CubeIDE ambiguo o ausente')
    require((IDE/'stm32cubeidec.exe').is_file(),'CubeIDE headless no disponible')
    return tools[0],makes[0]


def run_bounded(arguments,environment,logpath,timeout):
    """Watchdog del árbol hijo; los logs se abren antes de crear el proceso."""
    started=time.monotonic();process=None;timed_out=False;error=None;code=None
    with Path(logpath).open('x',encoding='utf-8',newline='\n') as log:
        try:
            process=subprocess.Popen(arguments,cwd=OUT,env=environment,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                text=True,encoding='utf-8',errors='replace',creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            messages=queue.Queue()
            def read_output():
                for line in process.stdout:messages.put(line)
                messages.put(None)
            threading.Thread(target=read_output,daemon=True).start()
            while True:
                if time.monotonic()-started>=timeout:
                    timed_out=True;error='TIMEOUT';break
                try:line=messages.get(timeout=.25)
                except queue.Empty:continue
                if line is None:break
                log.write(line);log.flush()
                if any(t in line.lower() for t in ('build of','error:','warning:','build finished')):print(line.strip()[:250],flush=True)
            if error is None:code=process.wait(timeout=max(.1,timeout-(time.monotonic()-started)))
        except subprocess.TimeoutExpired:timed_out=True;error='TIMEOUT'
        except KeyboardInterrupt:error='INTERRUPTED'
        except Exception as exc:error=type(exc).__name__+': '+str(exc)
        finally:
            if process is not None and process.poll() is None:
                subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],capture_output=True,
                    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),timeout=30)
                code=process.wait(timeout=30)
            log.write('\nCONTROLADOR: '+str(error or 'PROCESS_EXIT')+'\n')
    return dict(returncode=code,timed_out=timed_out,error=error,seconds=time.monotonic()-started)


def build(timeout=BUILD_TIMEOUT):
    require(type(timeout) is int and 60<=timeout<=BUILD_TIMEOUT,'Timeout fuera del presupuesto 60..1800 s')
    prep=verify();marker=OUT/'INTENTO_BUILD_UNICO.json'
    require(not marker.exists(),'Build ya iniciado; conservar evidencias y no reintentar automáticamente')
    tools,make=toolchain();environment=os.environ.copy()
    environment['PATH']=str(tools)+os.pathsep+str(make)+os.pathsep+environment.get('PATH','')
    arguments=[str(IDE/'stm32cubeidec.exe'),'--launcher.suppressErrors','-nosplash','-application',
        'org.eclipse.cdt.managedbuilder.core.headlessbuild','-data',str(OUT/'workspace_aislado'),'-import',str(PROJECT),
        '-no-indexer','-cleanBuild',NAME+'/'+CONFIG]
    write(marker,dict(attempt=1,command=arguments,timeout_seconds=timeout,preparation_sha256=sha(MANIFEST),
        automatic_retry=False,execution_owner='user_CLI'))
    print('BUILD CANON42: una configuración aislada; sin COM o programación de Flash.',flush=True)
    result=run_bounded(arguments,environment,OUT/'CUBEIDE_CANON42.log',timeout)
    text=(OUT/'CUBEIDE_CANON42.log').read_text(encoding='utf-8')
    counts=re.findall(r'(\d+) errors, (\d+) warnings',text)
    clean=bool(counts) and all(a==b=='0' for a,b in counts) and not re.search(r'\b(error|warning):|undefined reference|multiple definition',text,re.I)
    integrity=False;integrity_error=None
    try:verify(after_build=True);integrity=True
    except Exception as exc:integrity_error=type(exc).__name__+': '+str(exc)
    elf=PROJECT/CONFIG/(NAME+'.elf');mapfile=PROJECT/CONFIG/(NAME+'.map')
    passed=result['returncode']==0 and not result['timed_out'] and result['error'] is None and clean and integrity and elf.is_file() and mapfile.is_file()
    report=dict(**result,status='PASS_BUILD_PENDING_ELF_REVIEW_AND_ARM' if passed else 'FAIL_BUILD_REQUIRES_AUDIT',
        attempt=1,command=arguments,timeout_seconds=timeout,configuration=CONFIG,zero_errors_and_warnings=clean,
        integrity_after_build=integrity,integrity_error=integrity_error,automatic_retry=False,
        elf_path=str(elf),elf_sha256=sha(elf) if elf.is_file() else None,map_path=str(mapfile),
        map_sha256=sha(mapfile) if mapfile.is_file() else None,D_files_modified=False,serial_COM_opened=False,
        hardware_flashed=False,training_executed=False,ELF_machine_and_memory_audit_pending=True)
    try:
        report['D_source_sha256_after_build']=source_inventory(SOURCE,helper().configs(SOURCE/'.cproject')[2])
        report['D_source_unchanged_verified']=report['D_source_sha256_after_build']==prep['D_source_sha256_before']
    except Exception as exc:
        report['D_source_unchanged_verified']=False;report['D_verification_error']=type(exc).__name__+': '+str(exc)
    if not report['D_source_unchanged_verified']:
        report['status']='FAIL_BUILD_REQUIRES_AUDIT'
        passed=False
    write(OUT/'COMPILACION_CANON42.json',report)
    print(json.dumps(report,ensure_ascii=True,indent=2),flush=True)
    return 0 if passed else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    for name in ('prepare','verify','build'):mode.add_argument('--'+name,action='store_true')
    parser.add_argument('--timeout',type=int,default=BUILD_TIMEOUT);args=parser.parse_args()
    require(args.build or args.timeout==BUILD_TIMEOUT,'Timeout solo corresponde a --build')
    if args.prepare:prepare();return 0
    if args.verify:verify();print('CANON42_PREPARED_INTEGRITY_PASS_NOT_ARM_VALIDATED');return 0
    return build(args.timeout)


if __name__=='__main__':raise SystemExit(main())
