"""Preparación aislada CWRUSTK; solo install escribe siete archivos en D.

No abre COM ni programa la placa. Preserva doce configuraciones/ELF previos.
prepare copia; root aplica las guardias Src; build comprueba tres variantes.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
import preparar_pipeline_cwru as prior
from preparar_cache_potencias import configs,sha
from preparar_reloj_180mhz import flash_image

ROOT=Path(__file__).resolve().parents[4]
NAME='Prueba_2_CWRU_Extraccion'
SOURCE=Path('D:/STM32IDE')/NAME
OUT=ROOT/'despliegue_stm32/stack_pipeline_180mhz_2026-10-24'
PRIOR=ROOT/'despliegue_stm32/integracion_pipeline_180mhz_2026-10-23'
PROJECT=OUT/'copia_aislada'/NAME
BACKUP=OUT/'respaldo_antes_stack'
OVERLAY=OUT/'archivos_para_D'
CLIENT=OUT/'Validacion_PC'
TEMPLATES=Path(__file__).with_name('stack_firmware')
IDE=Path('D:/STM32CubeIDE_2.2.0/STM32CubeIDE')
CONFIG='Release_180MHz_Stack'
CONTROL='Release_180MHz_Pipeline'
SELECTED=('Release_180MHz_EntM',CONTROL,CONFIG)
LAUNCH=NAME+'_Stack_180MHz.launch'
TEMPLATE_NAMES=('cwru_stack.h','cwru_stack.c','cwru_stack_commands.inc')
MODIFIED=('.cproject','Src/uart_commands.c','Src/cmd_cwru_feat.c')
ADDED=(LAUNCH,*('CWRU_Stack/'+n for n in TEMPLATE_NAMES))
FILES=(*MODIFIED,*ADDED)
CLIENT_NAMES=('validar_stack.py','validar_pipeline.py','REFERENCIA_PIPELINE.json','ventanas_desarrollo.npz','metadata_ventanas.csv')

def require(ok,message):
    if not ok:raise RuntimeError(message)
def load(name):return json.loads((OUT/name).read_text(encoding='utf-8'))
def save(name,value):
    path=OUT/name;path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8',newline='\n') as stream:json.dump(value,stream,ensure_ascii=False,indent=2);stream.write('\n')
def cp(source,target):
    digest=sha(source);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    require(sha(target)==digest==sha(source),'Copia diferente: '+str(source));return digest
def canonical(node):
    return (node.tag,tuple(sorted(node.attrib.items())),(node.text or '').strip(),tuple(canonical(c) for c in node))
def same_configs(path):
    old,new=configs(BACKUP/'.cproject')[2],configs(path)[2]
    return len(old)==12 and len(new)==13 and CONFIG in new and all(n in new and ET.tostring(c)==ET.tostring(new[n]) for n,c in old.items())
def prior_manifest(name):
    path=PRIOR/name;digest,declared=path.with_suffix('.sha256').read_text(encoding='ascii').split()
    require(declared==name and sha(path)==digest,'Sello anterior distinto: '+name)
    entries=json.loads(path.read_text(encoding='utf-8'))
    for relative,expected in entries.items():
        p=(ROOT/relative).resolve();require(p.is_relative_to(ROOT) and sha(p)==expected,'Entrada anterior distinta: '+relative)
    return entries
def prior_checks():
    entries=prior_manifest('MANIFEST_SHA256.json')
    for label in ('INDICE_0','COMPLETA'):
        entries.update(prior_manifest('MANIFEST_RESULTADOS_FISICOS_PIPELINE_'+label+'_SHA256.json'))
    report=json.loads((PRIOR/'RESULTADOS_FISICOS_PIPELINE_COMPLETA.json').read_text(encoding='utf-8'))
    require(report['status']=='PHYSICAL_PIPELINE_SUITE_AUDITED' and report['completed_executions']==126,'Falta integración física auditada')
    return entries
def client_check():
    manifest=json.loads((CLIENT/'VALIDACION_MANIFEST.json').read_text(encoding='utf-8'))
    require(set(manifest)==set(CLIENT_NAMES),'Manifest cliente diferente')
    for n,d in manifest.items():require(sha(CLIENT/n)==d,'Cliente modificado:'+n)
    require(sha(CLIENT/'validar_stack.py')==sha(Path(__file__).with_name('validar_stack_cwru.py')),'Cliente nuevo difiere del fuente')
    for n in CLIENT_NAMES[1:]:require(sha(CLIENT/n)==sha(PRIOR/'Validacion_PC'/n),'Referencia anterior distinta:'+n)
def origin(prep):
    for row in prep['source_inventory_before']:require(sha(SOURCE/row['path'])==row['sha256'],'D cambió:'+row['path'])
    for n,d in prep['existing_elf_sha256'].items():require(sha(SOURCE/n/(NAME+'.elf'))==d,'ELF anterior cambió:'+n)
    require(not any((SOURCE/n).exists() for n in ADDED),'Destino nuevo ya existe')

def prepare():
    require(not any(p.exists() for p in (PROJECT,BACKUP,CLIENT,OUT/'PREPARACION.json')),'No sobrescribir preparación')
    require(not (SOURCE/'CWRU_Stack').exists(),'Destino Stack ya existe')
    prior_checks()
    installed=json.loads((PRIOR/'APLICACION_EN_D.json').read_text(encoding='utf-8'))
    previous=json.loads((PRIOR/'PREPARACION.json').read_text(encoding='utf-8'))
    reserialized=[]
    for n,d in installed['installed_sha256'].items():
        actual=sha(SOURCE/n)
        if actual!=d:
            require(n.endswith('.launch') and prior.launch_semantics(SOURCE/n)==prior.launch_semantics(PRIOR/'archivos_para_D'/n),'Fuente activa distinta:'+n)
            reserialized.append(dict(path=n,old_sha256=d,current_sha256=actual,semantically_equal=True))
    old=configs(SOURCE/'.cproject')[2]
    require(len(old)==12 and CONTROL in old and CONFIG not in old,'Se esperaban doce configuraciones y Pipeline')
    for n in TEMPLATE_NAMES:require((TEMPLATES/n).is_file(),'Plantilla ausente:'+n)
    require(Path(__file__).with_name('validar_stack_cwru.py').is_file(),'Cliente ausente')
    paths={r['path'] for r in previous['source_inventory_before']}|set(installed['installed_sha256'])
    paths.update(p.name for p in SOURCE.glob('*.launch'))
    snapshot=[]
    for n in sorted(paths):
        d=cp(SOURCE/n,PROJECT/n);require(cp(SOURCE/n,BACKUP/n)==d,'Respaldo distinto');snapshot.append(dict(path=n,sha256=d))
    elfs={n:cp(SOURCE/n/(NAME+'.elf'),BACKUP/n/(NAME+'.elf')) for n in old}
    root,settings,old=configs(PROJECT/'.cproject');node=copy.deepcopy(old[CONTROL])
    mapping={n.get('id'):n.get('id')+'.stack1' for n in node.iter() if n.get('id')}
    for element in node.iter():
        for k,v in list(element.attrib.items()):
            if v in mapping:element.set(k,mapping[v])
            elif k=='targetTool':element.set(k,';'.join(mapping.get(p,p) for p in v.split(';')))
            elif k=='name' and v==CONTROL:element.set(k,CONFIG)
            elif k=='buildPath' and v.endswith('/'+CONTROL):element.set(k,v[:-len(CONTROL)]+CONFIG)
    for tool in node.findall('.//tool'):
        if tool.get('superClass','').endswith(('tool.c.compiler','tool.cpp.compiler')):
            defines=next(o for o in tool.findall('option') if o.get('superClass','').endswith('.definedsymbols'))
            require(not any(v.get('value','').startswith('CWRU_STACK_TEST') for v in defines),'Stack ya definido')
            ET.SubElement(defines,'listOptionValue',builtIn='false',value='CWRU_STACK_TEST=1')
            includes=next(o for o in tool.findall('option') if o.get('superClass','').endswith('.includepaths'))
            ET.SubElement(includes,'listOptionValue',builtIn='false',value='../CWRU_Stack')
    ET.SubElement(node.find('.//sourceEntries'),'entry',flags='VALUE_WORKSPACE_PATH|RESOLVED',kind='sourcePath',name='CWRU_Stack')
    oldids={e.get('id') for c in old.values() for e in c.iter() if e.get('id')}
    require(not oldids&{e.get('id') for e in node.iter() if e.get('id')},'IDs duplicados')
    settings.append(node)
    (PROJECT/'.cproject').write_bytes(b'<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n<?fileVersion 4.0.0?>\n'+ET.tostring(root,encoding='utf-8')+b'\n')
    require(same_configs(PROJECT/'.cproject'),'Se alteró configuración anterior')
    for n in TEMPLATE_NAMES:cp(TEMPLATES/n,PROJECT/'CWRU_Stack'/n)
    launch=ET.parse(SOURCE/(NAME+'_Pipeline_180MHz.launch')).getroot();attrs={n.get('key'):n for n in launch}
    elf=CONFIG+'/'+NAME+'.elf'
    attrs['org.eclipse.cdt.launch.PROGRAM_NAME'].set('value',elf)
    attrs['org.eclipse.cdt.launch.PROJECT_BUILD_CONFIG_ID_ATTR'].set('value',node.get('id'))
    attrs['org.eclipse.cdt.launch.PROJECT_BUILD_CONFIG_AUTO_ATTR'].set('value','false')
    key='com.st.stm32cube.ide.mcu.debug.launch.loadList';loads=json.loads(attrs[key].get('value'))
    for item in loads['fItems']:item['fPath']=elf
    attrs[key].set('value',json.dumps(loads,separators=(',',':')))
    for key,e in attrs.items():
        if key.endswith('stlink.log_file'):e.set('value',str(SOURCE/CONFIG/'st-link_gdbserver_log.txt'))
    (PROJECT/LAUNCH).write_bytes(b'<?xml version="1.0" encoding="UTF-8"?>\n'+ET.tostring(launch,encoding='utf-8')+b'\n')
    cp(Path(__file__).with_name('validar_stack_cwru.py'),CLIENT/'validar_stack.py')
    for n in CLIENT_NAMES[1:]:cp(PRIOR/'Validacion_PC'/n,CLIENT/n)
    with (CLIENT/'VALIDACION_MANIFEST.json').open('x',encoding='utf-8') as stream:json.dump({n:sha(CLIENT/n) for n in CLIENT_NAMES},stream,indent=2)
    prep=dict(status='PREPARED_SRC_GUARDS_PENDING',source=str(SOURCE),configuration=CONFIG,
        source_inventory_before=snapshot,existing_elf_sha256=elfs,accepted_launch_reserializations=reserialized,
        previous_twelve_configurations_preserved=True,modified_files=list(MODIFIED),added_files=list(ADDED),
        hardware_flashed=False,serial_port_opened=False,physical_validation_pending=True)
    origin(prep);client_check();save('PREPARACION.json',prep)
    print('PREPARED: faltan guardias Src en copia aislada',flush=True)

def flags(commands,config):
    def macro(line,n,v):
        values=re.findall(r'(?<!\S)-D'+re.escape(n)+r'(?:=([^\s]+))?(?=\s|$)',line)
        return values==[v] if v is not None else values in ([],['0'])
    expected={'CWRU_TARGET_HCLK_HZ':'180000000U','CWRU_CACHE_SPECTRAL_POWER':'1','CWRU_ENTROPY_MODE':'2',
        'CWRU_INPUT_TRANSFORM':None if config==SELECTED[0] else '1','CWRU_DNN_TEST':None if config==SELECTED[0] else '1',
        'CWRU_NORMALITY_TEST':None if config==SELECTED[0] else '1','CWRU_PIPELINE_TEST':None if config==SELECTED[0] else '1',
        'CWRU_STACK_TEST':'1' if config==CONFIG else None,'CWRU_PROFILE_EXTRACTOR':None}
    return bool(commands) and all(re.findall(r'(?<!\S)-O\S+',line)==['-O3'] and
        all(macro(line,k,v) for k,v in expected.items()) and not any(f in line for f in ('-ffast-math','-Ofast','-funsafe-math-optimizations'))
        and ('-I../CWRU_Stack' in line)==(config==CONFIG) for line in commands)

def inspect(tool,config,prep,attempt):
    folder=PROJECT/config;elf=folder/(NAME+'.elf')
    values=subprocess.check_output([str(tool/'arm-none-eabi-size.exe'),str(elf)],text=True).splitlines()[-1].split()
    text,data,bss=map(int,values[:3]);image=flash_image(elf)
    frames={p.relative_to(folder).as_posix():p.read_text(encoding='utf-8').splitlines() for p in folder.rglob('*.su')}
    result=dict(elf_sha256=sha(elf),flash_load_bytes=text+data,ram_static_with_linker_reservations_bytes=data+bss,
        flags_confirmed=flags(prior.compiler_commands(folder),config),compiler_stack_frames=frames)
    if config!=CONFIG:
        result.update(control_elf_identical=sha(elf)==prep['existing_elf_sha256'][config],
            control_flash_identical=image==flash_image(BACKUP/config/(NAME+'.elf')))
    else:
        symbols=subprocess.check_output([str(tool/'arm-none-eabi-nm.exe'),'-S','-C','--defined-only',str(elf)],text=True)
        (OUT/f'SIMBOLOS_STACK_INTENTO_{attempt}.txt').write_text(symbols,encoding='utf-8')
        expected={'g_window':32768,'g_fft_input':16384,'g_fft_output':16384,'(anonymous namespace)::arena':16384}
        allocations={}
        for n,size in expected.items():
            rows=re.findall(r'(?m)^([0-9a-f]+)\s+([0-9a-f]+)\s+[bBdD]\s+'+re.escape(n)+r'$',symbols)
            require(len(rows)==1 and int(rows[0][1],16)==size,'Buffer duplicado o diferente:'+n)
            allocations[n]=[int(rows[0][0],16),size]
        ranges=sorted((a,a+s) for a,s in allocations.values())
        require(all(a[1]<=b[0] for a,b in zip(ranges,ranges[1:])),'Superposición de buffers')
        large=re.findall(r'(?m)^([0-9a-f]+)\s+([0-9a-f]+)\s+[bBdD]\s+(.+)$',symbols)
        require(len([r for r in large if int(r[1],16)>=16384 and 0x20000000<=int(r[0],16)<0x20020000])==4,'Otro buffer grande inesperado')
        result['buffer_audit']=dict(status='PASS',allocations=allocations,nonoverlapping=True)
        assembly={}
        for symbol in ('CWRUSTK_Measure','stack_probe','stack_pipeline','CWRUPIPE_Execute'):
            assembly[symbol]=subprocess.check_output([str(tool/'arm-none-eabi-objdump.exe'),'-d','--disassemble='+symbol,str(elf)],text=True)
            (OUT/f'DESENSAMBLADO_{symbol}_INTENTO_{attempt}.txt').write_text(assembly[symbol],encoding='utf-8')
        monitor=assembly['CWRUSTK_Measure']
        result['monitor_assembly_prechecks']=dict(callback_blx=len(re.findall(r'\bblx\s+(?:r\d+|fp|sl|ip)\b',monitor))==1,
            msp_capture=len(re.findall(r'\bmrs\s+[^\n]*MSP',monitor,re.I))==2,
            masks_present='cpsid' in monitor and bool(re.search(r'\bmsr\s+PRIMASK',monitor,re.I)),
            probe_128=bool(re.search(r'sub\s+sp,\s*#128',assembly['stack_probe'])))
        result['required_stack_frames_present']=all(n in frames for n in ('CWRU_Stack/cwru_stack.su','Src/cmd_cwru_feat.su','Src/main.su'))
        # Se debe revisar manualmente el ensamblado completo antes de instalar.
        result['manual_disassembly_review_required']=True
    return result

def old_sources(tool):
    includes=('Inc','CWRU_Input','CWRU_DNN','CWRU_Detector','CWRU_Pipeline','Drivers/CMSIS/Include',
        'Drivers/CMSIS/Device/ST/STM32F4xx/Include','Drivers/STM32F4xx_HAL_Driver/Inc',
        'Middlewares/CMSIS-DSP/Include','Middlewares/CMSIS-DSP/PrivateInclude')
    base=[str(tool/'arm-none-eabi-gcc.exe'),'-E','-P','-std=gnu11','-mcpu=cortex-m4','-mfpu=fpv4-sp-d16',
        '-mfloat-abi=hard','-mthumb','-DSTM32F446xx','-DCWRU_STACK_TEST=0','-DCWRU_PROFILE_EXTRACTOR=0',
        '-DCWRU_TARGET_HCLK_HZ=180000000U','-DCWRU_CACHE_SPECTRAL_POWER=1','-DCWRU_ENTROPY_MODE=2',
        *('-I'+str(PROJECT/n) for n in includes)]
    results={}
    for n in MODIFIED[1:]:
        for inp,nn,det,pipe in ((0,0,0,0),(1,0,0,0),(1,1,0,0),(1,1,1,0),(1,1,1,1)):
            args=base+[f'-DCWRU_INPUT_TRANSFORM={inp}',f'-DCWRU_DNN_TEST={nn}',f'-DCWRU_NORMALITY_TEST={det}',f'-DCWRU_PIPELINE_TEST={pipe}']
            before=subprocess.check_output(args+[str(BACKUP/n)]);after=subprocess.check_output(args+[str(PROJECT/n)])
            results[n+f':{inp}{nn}{det}{pipe}']=dict(identical=before==after,before=hashlib.sha256(before).hexdigest(),after=hashlib.sha256(after).hexdigest())
    return results

def build():
    prep=load('PREPARACION.json');origin(prep);client_check()
    require(not (OUT/'COMPILACION_STACK.json').exists(),'Build ya aprobado')
    require('CWRUSTK_Command' in (PROJECT/MODIFIED[1]).read_text(encoding='utf-8'),'Falta guardia UART')
    require('cwru_stack_commands.inc' in (PROJECT/MODIFIED[2]).read_text(encoding='utf-8'),'Falta include receptor')
    for n in TEMPLATE_NAMES:require(sha(PROJECT/'CWRU_Stack'/n)==sha(TEMPLATES/n),'Plantilla diferente:'+n)
    attempts=[int(p.stem.rsplit('_',1)[-1]) for p in OUT.glob('CUBEIDE_STACK_INTENTO_*.log')];attempt=max(attempts,default=0)+1
    require(attempt<=3,'Tres intentos agotados: revisar antes de ampliar')
    for n in FILES:cp(PROJECT/n,OVERLAY/n);cp(PROJECT/n,OUT/'intentos'/str(attempt)/'archivos'/n)
    tools=list((IDE/'plugins').glob('com.st.stm32cube.ide.mcu.externaltools.gnu-tools-for-stm32.*/tools/bin'))
    makes=list((IDE/'plugins').glob('com.st.stm32cube.ide.mcu.externaltools.make.*/tools/bin'))
    require(len(tools)==len(makes)==1,'Toolchain ambigua');tool=tools[0]
    env=dict(os.environ);env['PATH']=str(tool)+os.pathsep+str(makes[0])+os.pathsep+env.get('PATH','');env['PYTHONIOENCODING']='utf-8'
    args=[str(IDE/'stm32cubeidec.exe'),'--launcher.suppressErrors','-nosplash','-application',
        'org.eclipse.cdt.managedbuilder.core.headlessbuild','-data',str(OUT/'workspace_aislado'/str(attempt)),'-import',str(PROJECT),'-no-indexer']
    for n in SELECTED:args.extend(('-cleanBuild',NAME+'/'+n))
    report=dict(attempt=attempt,command=args,configurations={},hardware_flashed=False,serial_port_opened=False)
    with (OUT/f'CUBEIDE_STACK_INTENTO_{attempt}.log').open('x',encoding='utf-8') as log:
        tests=subprocess.run([sys.executable,'-B',str(CLIENT/'validar_stack.py'),'--self-test','--output-root',str(OUT/'_self_test'/str(attempt))],
            capture_output=True,text=True,encoding='utf-8',env=env)
        save(f'PRUEBAS_CLIENTE_INTENTO_{attempt}.json',dict(returncode=tests.returncode,output=tests.stdout+tests.stderr,simulation_only=True))
        require(tests.returncode==0,'Autotest cliente falló; conservar log')
        process=subprocess.Popen(args,cwd=OUT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,encoding='utf-8',errors='replace')
        for line in process.stdout:
            log.write(line);log.flush()
            if any(t in line.lower() for t in ('build of','error:','warning:','build finished')):print(line.strip()[:250],flush=True)
        report['returncode']=process.wait(timeout=30)
    text=(OUT/f'CUBEIDE_STACK_INTENTO_{attempt}.log').read_text(encoding='utf-8')
    summaries=re.findall(r'(\d+) errors, (\d+) warnings',text)
    report['zero_errors_warnings']=len(summaries)>=3 and all(a==b=='0' for a,b in summaries) and not re.search(r'\b(error|warning):|undefined reference',text,re.I)
    for n in SELECTED:
        try:report['configurations'][n]=inspect(tool,n,prep,attempt)
        except Exception as error:report['configurations'][n]=dict(inspection_error=str(error))
        for p in (PROJECT/n).rglob('*'):
            if p.is_file() and (p.suffix in ('.elf','.map','.su','.mk') or p.name in ('makefile','objects.list')):
                cp(p,OUT/'intentos'/str(attempt)/'compilacion'/n/p.relative_to(PROJECT/n))
    report['previous_twelve_configurations_preserved']=same_configs(PROJECT/'.cproject')
    report['other_resources_unchanged']=all(sha(PROJECT/r['path'])==r['sha256'] for r in prep['source_inventory_before']
        if r['path'] not in (*MODIFIED,'.settings/language.settings.xml'))
    report['old_sources_preprocessed']=old_sources(tool)
    report['overlay_sha256']={n:sha(OVERLAY/n) for n in FILES};origin(prep)
    rows=report['configurations'];stack=rows[CONFIG]
    passed=report['returncode']==0 and report['zero_errors_warnings'] and all(r.get('flags_confirmed',False) for r in rows.values())
    passed=passed and all(rows[n].get('control_elf_identical',False) and rows[n].get('control_flash_identical',False) for n in SELECTED[:2])
    passed=passed and report['previous_twelve_configurations_preserved'] and report['other_resources_unchanged'] and all(r['identical'] for r in report['old_sources_preprocessed'].values())
    passed=passed and stack.get('buffer_audit',{}).get('status')=='PASS' and stack.get('required_stack_frames_present',False) and all(stack.get('monitor_assembly_prechecks',{'missing':False}).values())
    report['status']='PASS_PENDING_MANUAL_DISASSEMBLY' if passed else 'FAIL';save(f'COMPILACION_STACK_INTENTO_{attempt}.json',report)
    if passed:save('COMPILACION_STACK.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='configurations'},ensure_ascii=True,indent=2),flush=True)
    if not passed:raise SystemExit(2)

def recheck():
    """Reevaluar el intento1 sin recompilar ni modificar su informe fallido.

    Objdump emplea fp como alias de r11. Solo se corrige su reconocimiento;
    no se cambia el ELF, las comprobaciones físicas o el contrato numérico.
    """
    old=load('COMPILACION_STACK_INTENTO_1.json');prep=load('PREPARACION.json')
    require(old['status']=='FAIL' and not (OUT/'COMPILACION_STACK.json').exists(),'Reevaluación no aplicable')
    origin(prep);client_check()
    for n,d in old['overlay_sha256'].items():require(sha(PROJECT/n)==sha(OVERLAY/n)==d,'Fuente posterior distinta:'+n)
    for n,row in old['configurations'].items():require(sha(PROJECT/n/(NAME+'.elf'))==row['elf_sha256'],'ELF cambió:'+n)
    require(old['zero_errors_warnings'] and old['returncode']==0 and old['other_resources_unchanged']
        and old['previous_twelve_configurations_preserved'],'Otro fallo de preparación')
    require(all(r['identical'] for r in old['old_sources_preprocessed'].values()) and len(old['old_sources_preprocessed'])==10,'Preprocesado diferente')
    for n in SELECTED[:2]:
        row=old['configurations'][n];require(row['control_elf_identical'] and row['control_flash_identical'],'Control distinto')
    require(all(r['flags_confirmed'] for r in old['configurations'].values()),'Flags distintos')
    row=old['configurations'][CONFIG];checks=row['monitor_assembly_prechecks']
    require(checks==dict(callback_blx=False,msp_capture=True,masks_present=True,probe_128=True)
        and row['buffer_audit']['status']=='PASS' and row['required_stack_frames_present'],'Fallo distinto del alias')
    monitor=(OUT/'DESENSAMBLADO_CWRUSTK_Measure_INTENTO_1.txt').read_text(encoding='utf-8')
    calls=re.findall(r'\bblx\s+(r\d+|fp|sl|ip)\b',monitor)
    require(calls==['fp'],'Alias/callback inesperado')
    new=copy.deepcopy(old);new['configurations'][CONFIG]['monitor_assembly_prechecks']['callback_blx']=True
    new.update(status='PASS_PENDING_MANUAL_DISASSEMBLY',recompiled=False,
        reevaluation_reason='fp is the objdump alias of r11; exactly one callback BLX fp',
        original_failed_report_sha256=sha(OUT/'COMPILACION_STACK_INTENTO_1.json'))
    save('REEVALUACION_COMPILACION_INTENTO_1.json',new);save('COMPILACION_STACK.json',new)
    print(json.dumps(dict(status=new['status'],recompiled=False,controls_identical=True)))

def metadata_check():
    n='.settings/language.settings.xml';before=ET.parse(BACKUP/n).getroot();after=ET.parse(SOURCE/n).getroot()
    if sha(BACKUP/n)==sha(SOURCE/n):return True
    require(len(before)==12 and all(c.tag=='configuration' for c in before),'CDT previo ambiguo')
    selected=next(c for c in before if c.get('name')==CONTROL);new=copy.deepcopy(selected)
    new.set('name',CONFIG);new.set('id',configs(SOURCE/'.cproject')[2][CONFIG].get('id'));before.append(new)
    return canonical(before)==canonical(after)
def installed_checks(prep,hashes):
    return dict(previous_twelve_configurations_preserved=same_configs(SOURCE/'.cproject'),
        previous_twelve_elfs_unchanged=all(sha(SOURCE/n/(NAME+'.elf'))==d for n,d in prep['existing_elf_sha256'].items()),
        installed_files_match_overlay=all(sha(SOURCE/n)==d for n,d in hashes.items()),
        other_resources_unchanged_except_exact_cdt_append=all(sha(SOURCE/r['path'])==r['sha256'] or
            (r['path']=='.settings/language.settings.xml' and metadata_check()) for r in prep['source_inventory_before'] if r['path'] not in MODIFIED))
def install():
    prep,built=load('PREPARACION.json'),load('COMPILACION_STACK.json');review=load('REVISION_ENSAMBLADO.json')
    require(built['status']=='PASS_PENDING_MANUAL_DISASSEMBLY' and review['status']=='PASS','Falta build/revisión')
    require(not (OUT/'APLICACION_EN_D.json').exists(),'Instalación ya registrada');origin(prep)
    require(review['elf_sha256']==built['configurations'][CONFIG]['elf_sha256'],'Revisión de otro ELF')
    for n,d in built['overlay_sha256'].items():require(sha(OVERLAY/n)==sha(PROJECT/n)==d,'Overlay cambió:'+n)
    for r in prep['source_inventory_before']:require(sha(BACKUP/r['path'])==r['sha256'],'Respaldo cambió')
    for n in FILES:cp(OVERLAY/n,SOURCE/n)
    checks=installed_checks(prep,built['overlay_sha256'])
    report=dict(status='PASS' if all(checks.values()) else 'FAIL',**checks,installed_sha256=built['overlay_sha256'],
        modified_files=list(MODIFIED),added_files=list(ADDED),hardware_flashed=False,serial_port_opened=False,physical_validation_pending=True)
    save('APLICACION_EN_D.json',report);print(json.dumps(report,indent=2));require(report['status']=='PASS','Instalación falló')
def seal():
    require(not (OUT/'MANIFEST_SHA256.json').exists(),'Sello ya existente')
    prep,installed=load('PREPARACION.json'),load('APLICACION_EN_D.json');require(installed['status']=='PASS','Falta instalación')
    require(all(installed_checks(prep,installed['installed_sha256']).values()),'D cambió después de instalar')
    metadata=SOURCE/'.settings/language.settings.xml';cp(metadata,OUT/'language.settings.postinstalacion.xml')
    save('METADATOS_CDT_POSTINSTALACION.json',dict(status='PASS',exact_append_or_unchanged=True,before_sha256=sha(BACKUP/'.settings/language.settings.xml'),after_sha256=sha(metadata),overwritten_in_D=False))
    client_check();entries=prior_checks();paths={ROOT/n for n in entries}
    paths.update(p for p in OUT.iterdir() if p.is_file())
    for folder in (BACKUP,OVERLAY,CLIENT,OUT/'intentos',OUT/'_self_test',OUT/'_self_test_previo'):
        paths.update(p for p in folder.rglob('*') if p.is_file() and not p.is_relative_to(CLIENT/'resultados'))
    for r in prep['source_inventory_before']:paths.add(PROJECT/r['path'])
    paths.update(PROJECT/n for n in FILES)
    paths.update(TEMPLATES/n for n in TEMPLATE_NAMES)
    paths.update(Path(__file__).with_name(n) for n in ('preparar_stack_cwru.py','validar_stack_cwru.py'))
    for n in SELECTED:
        paths.update(p for p in (PROJECT/n).rglob('*') if p.is_file() and (p.suffix in ('.elf','.map','.su','.mk') or p.name in ('makefile','objects.list')))
    require(all(p.is_file() and p.resolve().is_relative_to(ROOT) for p in paths),'Ruta requerida ausente/fuera')
    save('MANIFEST_SHA256.json',{p.relative_to(ROOT).as_posix():sha(p) for p in sorted(paths)})
    with (OUT/'MANIFEST_SHA256.sha256').open('x',encoding='ascii') as stream:stream.write(sha(OUT/'MANIFEST_SHA256.json')+'  MANIFEST_SHA256.json\n')
    verify()
def verify():
    digest,n=(OUT/'MANIFEST_SHA256.sha256').read_text(encoding='ascii').split();require(n=='MANIFEST_SHA256.json' and sha(OUT/n)==digest,'Sello diferente')
    manifest=load(n)
    for relative,expected in manifest.items():
        p=(ROOT/relative).resolve();require(p.is_relative_to(ROOT) and sha(p)==expected,'Recurso modificado:'+relative)
    print(json.dumps(dict(status='PASS',files_verified=len(manifest),serial_port_opened=False,physical_validation_pending=True)))
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('prepare','build','recheck','install','seal','verify'))
    globals()[parser.parse_args().mode]()
