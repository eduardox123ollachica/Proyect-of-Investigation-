"""Prepara y entrega detector opcional: C aislado, build, install respaldado, sello.

Solo install escribe en D. Ningún modo abre COM o programa la placa.
"""
import argparse
import copy
import json
import os
import re
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from preparar_cache_potencias import configs, sha
from preparar_reloj_180mhz import flash_image

ROOT=Path(__file__).resolve().parents[4]
NAME='Prueba_2_CWRU_Extraccion'
SOURCE=Path('D:/STM32IDE')/NAME
OUT=ROOT/'despliegue_stm32/detector_normalidad_180mhz_2026-10-23'
PRIOR=ROOT/'despliegue_stm32/dnn_int8_180mhz_2026-10-23'
PROJECT=OUT/'copia_aislada'/NAME
BACKUP=OUT/'respaldo_antes_detector'
OVERLAY=OUT/'archivos_para_D'
CLIENT=OUT/'Validacion_PC'
CONFIG='Release_180MHz_Detector'
CONTROL='Release_180MHz_DNN'
LAUNCH=NAME+'_Detector_180MHz.launch'
TEMPLATES=Path(__file__).with_name('detector_firmware')
IDE=Path('D:/STM32CubeIDE_2.2.0/STM32CubeIDE')
NEW=['CWRU_Detector/'+n for n in ('cwru_normality.h','cwru_normality.c','cmd_cwru_normality.c','cwru_normality_params.h','cwru_normality_params.c')]
FILES=['.cproject','Src/uart_commands.c',LAUNCH,*NEW]


def require(condition,message):
    if not condition:raise RuntimeError(message)


def save(name,data):
    with (OUT/name).open('x',encoding='utf-8',newline='\n') as f:
        json.dump(data,f,ensure_ascii=False,indent=2);f.write('\n')


def load(name):return json.loads((OUT/name).read_text(encoding='utf-8'))


def origin(prep):
    for row in prep['source_inventory_before']:
        require(sha(SOURCE/row['path'])==row['sha256'],'D cambió: '+row['path'])
    for cfg,digest in prep['existing_elf_sha256'].items():
        require(sha(SOURCE/cfg/(NAME+'.elf'))==digest,'ELF anterior cambió: '+cfg)


def prepare():
    require(not PROJECT.exists() and not (SOURCE/'CWRU_Detector').exists() and not (SOURCE/LAUNCH).exists(),'Preparación/destino ya existente')
    reference=load('REFERENCIA_DETECTOR.json')
    require(reference['historical_changed_decisions']==0 and len(reference['cases'])==165,'Falta referencia válida')
    for rel,digest in load('ENTRADAS_REFERENCIA_SHA256.json').items():require(sha(ROOT/rel)==digest,'Entrada cambiada: '+rel)
    subprocess.run([sys.executable,'-B',str(Path(__file__).with_name('preparar_dnn_cwru.py')),'verify'],check=True)
    previous=json.loads((PRIOR/'PREPARACION.json').read_text(encoding='utf-8'))
    paths={r['path'] for r in previous['source_inventory_before']}|set(json.loads((PRIOR/'APLICACION_EN_D.json').read_text(encoding='utf-8'))['installed_sha256'])
    paths.update(p.name for p in SOURCE.glob('*.launch'))
    snapshot=[]
    for rel in sorted(paths):
        digest=sha(SOURCE/rel)
        for base in (PROJECT,BACKUP):
            target=base/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(SOURCE/rel,target)
            require(sha(target)==digest,'Copia diferente: '+rel)
        snapshot.append(dict(path=rel,sha256=digest))
    tree,settings,before=configs(PROJECT/'.cproject')
    require(len(before)==10 and CONTROL in before,'Se esperaban diez configuraciones')
    elves={}
    for cfg in before:
        path=SOURCE/cfg/(NAME+'.elf');target=BACKUP/cfg/path.name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,target);elves[cfg]=sha(path);require(sha(target)==elves[cfg],'Respaldo ELF distinto')
    expanded=copy.deepcopy(before[CONTROL])
    mapping={e.get('id'):e.get('id')+'.normality1' for e in expanded.iter() if e.get('id')}
    for e in expanded.iter():
        for key,value in list(e.attrib.items()):
            if value in mapping:e.set(key,mapping[value])
            elif key=='targetTool':e.set(key,';'.join(mapping.get(v,v) for v in value.split(';')))
            elif key=='name' and value==CONTROL:e.set(key,CONFIG)
            elif key=='buildPath' and value.endswith('/'+CONTROL):e.set(key,value[:-len(CONTROL)]+CONFIG)
    for tool in expanded.findall('.//tool'):
        if tool.get('superClass','').endswith(('tool.c.compiler','tool.cpp.compiler')):
            defs=next(o for o in tool.findall('option') if o.get('superClass','').endswith('.definedsymbols'))
            ET.SubElement(defs,'listOptionValue',builtIn='false',value='CWRU_NORMALITY_TEST=1')
            inc=next(o for o in tool.findall('option') if o.get('superClass','').endswith('.includepaths'))
            ET.SubElement(inc,'listOptionValue',builtIn='false',value='../CWRU_Detector')
    ET.SubElement(expanded.find('.//sourceEntries'),'entry',flags='VALUE_WORKSPACE_PATH|RESOLVED',kind='sourcePath',name='CWRU_Detector')
    old_ids={e.get('id') for c in before.values() for e in c.iter() if e.get('id')}
    require(not old_ids&{e.get('id') for e in expanded.iter() if e.get('id')},'IDs duplicados')
    settings.append(expanded)
    (PROJECT/'.cproject').write_bytes(b'<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n<?fileVersion 4.0.0?>\n'+ET.tostring(tree,encoding='utf-8')+b'\n')
    after=configs(PROJECT/'.cproject')[2]
    require(all(ET.tostring(c)==ET.tostring(after[n]) for n,c in before.items()),'Configuración anterior modificada')
    folder=PROJECT/'CWRU_Detector';folder.mkdir()
    for p in TEMPLATES.iterdir():
        if p.suffix in ('.c','.h'):shutil.copy2(p,folder/p.name)
    resources=ROOT/'modelos/cwru/canonico_seed42/firmware'
    shutil.copy2(resources/'cwru_normality_params.cpp',folder/'cwru_normality_params.c')
    shutil.copy2(resources/'cwru_normality_params.h',folder/'cwru_normality_params.h')
    launch=ET.parse(SOURCE/(NAME+'_DNN_180MHz.launch')).getroot();attrs={e.get('key'):e for e in launch};elf=CONFIG+'/'+NAME+'.elf'
    attrs['org.eclipse.cdt.launch.PROGRAM_NAME'].set('value',elf)
    attrs['org.eclipse.cdt.launch.PROJECT_BUILD_CONFIG_ID_ATTR'].set('value',expanded.get('id'))
    attrs['org.eclipse.cdt.launch.PROJECT_BUILD_CONFIG_AUTO_ATTR'].set('value','false')
    key='com.st.stm32cube.ide.mcu.debug.launch.loadList';loads=json.loads(attrs[key].get('value'))
    for item in loads['fItems']:item['fPath']=elf
    attrs[key].set('value',json.dumps(loads,separators=(',',':')))
    for key,e in attrs.items():
        if key.endswith('stlink.log_file'):e.set('value',str(SOURCE/CONFIG/'st-link_gdbserver_log.txt'))
    (PROJECT/LAUNCH).write_bytes(b'<?xml version="1.0" encoding="UTF-8"?>\n'+ET.tostring(launch,encoding='utf-8')+b'\n')
    CLIENT.mkdir()
    for src,name in [(Path(__file__).with_name('validar_detector_cwru.py'),'validar_detector.py'),
        (Path(__file__).with_name('contrato_detector_cwru.py'),'contrato_detector_cwru.py'),(OUT/'REFERENCIA_DETECTOR.json','REFERENCIA_DETECTOR.json')]:
        shutil.copy2(src,CLIENT/name)
    (CLIENT/'VALIDACION_MANIFEST.json').write_text(json.dumps({p.name:sha(p) for p in CLIENT.iterdir() if p.is_file()},indent=2)+'\n',encoding='utf-8')
    save('PREPARACION.json',dict(status='PREPARED_UART_PATCH_PENDING',source_inventory_before=snapshot,existing_elf_sha256=elves,
        previous_ten_configurations_preserved=True,modified_files=FILES[:2],added_files=FILES[2:],
        physical_validation_pending=True,hardware_flashed=False,serial_port_opened=False))
    print('PREPARADO: copia aislada y once configuraciones; falta guardia UART.',flush=True)


def build():
    prep=load('PREPARACION.json');origin(prep)
    require('CWRUND_Command' in (PROJECT/'Src/uart_commands.c').read_text(encoding='utf-8'),'Falta guardia UART')
    require(not (OUT/'COMPILACION_DETECTOR.json').exists(),'Build ya aprobado')
    attempt=len(list(OUT.glob('CUBEIDE_DETECTOR_INTENTO_*.log')))+1
    require(attempt<=3,'Revisar diagnóstico antes de más intentos')
    tests=subprocess.run([sys.executable,'-B',str(CLIENT/'validar_detector.py'),'--self-test'],capture_output=True,text=True,encoding='utf-8',errors='replace')
    save(f'PRUEBAS_CLIENTE_INTENTO_{attempt}.json',dict(status='PASS' if tests.returncode==0 else 'FAIL',output=tests.stdout+tests.stderr,executes_C_ARM=False))
    require(tests.returncode==0,'Autotests fallidos')
    for rel in FILES:
        for base in (OVERLAY,OUT/'intentos'/str(attempt)/'archivos'):
            dest=base/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(PROJECT/rel,dest)
    tool=next((IDE/'plugins').glob('com.st.stm32cube.ide.mcu.externaltools.gnu-tools-for-stm32.*/tools/bin'))
    make=next((IDE/'plugins').glob('com.st.stm32cube.ide.mcu.externaltools.make.*/tools/bin'))
    env=dict(os.environ);env['PATH']=str(tool)+os.pathsep+str(make)+os.pathsep+env.get('PATH','')
    selected=['Release_180MHz_EntM',CONTROL,CONFIG]
    args=[str(IDE/'stm32cubeidec.exe'),'--launcher.suppressErrors','-nosplash','-application','org.eclipse.cdt.managedbuilder.core.headlessbuild',
        '-data',str(OUT/'workspace_aislado'),'-import',str(PROJECT),'-no-indexer']
    for cfg in selected:args.extend(('-build',NAME+'/'+cfg))
    logpath=OUT/f'CUBEIDE_DETECTOR_INTENTO_{attempt}.log';start=time.monotonic()
    with logpath.open('x',encoding='utf-8') as log:
        process=subprocess.Popen(args,cwd=OUT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,encoding='utf-8',errors='replace')
        for line in process.stdout:
            log.write(line);log.flush()
            if any(s in line.lower() for s in ('build of','error:','warning:','undefined reference','build finished')):print(line.strip()[:300],flush=True)
        code=process.wait(timeout=30)
    text=logpath.read_text(encoding='utf-8');summaries=re.findall(r'(\d+) errors, (\d+) warnings',text)
    clean=bool(summaries) and all(int(a)==0 and int(b)==0 for a,b in summaries) and not re.search(r'\b(?:error|warning):|undefined reference',text)
    report=dict(attempt=attempt,returncode=code,elapsed_seconds=time.monotonic()-start,command=args,zero_errors_warnings=clean,configurations={})
    for cfg in selected:
        folder=PROJECT/cfg;elf=folder/(NAME+'.elf')
        if not elf.exists():continue
        size=subprocess.check_output([str(tool/'arm-none-eabi-size.exe'),str(elf)],text=True)
        t,d,b=map(int,size.strip().splitlines()[-1].split()[:3])
        commands=[line for mk in folder.rglob('subdir.mk') for line in mk.read_text(encoding='utf-8').splitlines()
            if '-c ' in line and 'arm-none-eabi-' in line and re.search(r'-std=gnu(?:\+\+|\d)',line)]
        flags=bool(commands) and all(re.findall(r'(?<!\S)-O\S+',line)==['-O3'] and '-DCWRU_TARGET_HCLK_HZ=180000000U' in line
            and '-DCWRU_ENTROPY_MODE=2' in line and ('-DCWRU_NORMALITY_TEST=1' in line)==(cfg==CONFIG)
            and ('-DCWRU_DNN_TEST=1' in line)==(cfg!=selected[0])
            and not any(f in line for f in ('-ffast-math','-Ofast','-funsafe-math-optimizations')) for line in commands)
        image=flash_image(elf);span=max(a+len(data) for a,data in image)-min(a for a,_ in image)
        row=dict(elf_sha256=sha(elf),text=t,data=d,bss=b,flash_load_bytes=t+d,ram_static_with_linker_reservations_bytes=d+b,
            flash_span_bytes=span,expected_download_kib=span/1024,flags_confirmed=flags,
            stack_frames={p.name:p.read_text(encoding='utf-8').splitlines() for p in (folder/'CWRU_Detector').glob('*.su')})
        if cfg!=CONFIG:
            row.update(control_elf_identical=sha(elf)==prep['existing_elf_sha256'][cfg],
                control_flash_identical=image==flash_image(BACKUP/cfg/(NAME+'.elf')))
        else:
            disasm=subprocess.check_output([str(tool/'arm-none-eabi-objdump.exe'),'-d','--disassemble=CWRUND_Compute',str(elf)],text=True)
            (OUT/f'DESENSAMBLADO_DETECTOR_INTENTO_{attempt}.txt').write_text(disasm,encoding='utf-8')
            row['separate_float32_operations_confirmed']=all(s in disasm for s in ('vmul.f32','vadd.f32','vdiv.f32')) and not re.search(r'\b(?:vfma|vfms|vfnma|vfnms|vmla|vmls)\.',disasm)
        report['configurations'][cfg]=row
    previous,current=configs(BACKUP/'.cproject')[2],configs(PROJECT/'.cproject')[2]
    report['previous_ten_configurations_preserved']=all(ET.tostring(c)==ET.tostring(current[n]) for n,c in previous.items())
    allowed={'.cproject','Src/uart_commands.c','.settings/language.settings.xml'}
    report['other_resources_unchanged']=all(sha(PROJECT/r['path'])==r['sha256'] for r in prep['source_inventory_before'] if r['path'] not in allowed)
    includes=['-I'+str(PROJECT/p) for p in ('Inc','CWRU_Input','CWRU_DNN','Drivers/CMSIS/Include','Drivers/CMSIS/Device/ST/STM32F4xx/Include')]
    cpp=[str(tool/'arm-none-eabi-gcc.exe'),'-E','-P','-std=gnu11','-mcpu=cortex-m4','-mfpu=fpv4-sp-d16','-mfloat-abi=hard','-mthumb','-DSTM32F446xx',*includes]
    report['old_dispatch_preprocessed_identical']={f'{a}_{b}':subprocess.check_output([*cpp,f'-DCWRU_INPUT_TRANSFORM={a}',f'-DCWRU_DNN_TEST={b}',str(BACKUP/'Src/uart_commands.c')])==subprocess.check_output([*cpp,f'-DCWRU_INPUT_TRANSFORM={a}',f'-DCWRU_DNN_TEST={b}',str(PROJECT/'Src/uart_commands.c')]) for a,b in ((0,0),(1,0),(1,1))}
    report['overlay_sha256']={r:sha(OVERLAY/r) for r in FILES}
    passed=(code==0 and clean and len(report['configurations'])==3 and all(r['flags_confirmed'] for r in report['configurations'].values())
        and all(report['configurations'][cfg]['control_elf_identical'] and report['configurations'][cfg]['control_flash_identical'] for cfg in selected[:2])
        and report['previous_ten_configurations_preserved'] and report['other_resources_unchanged'] and all(report['old_dispatch_preprocessed_identical'].values())
        and report['configurations'][CONFIG]['separate_float32_operations_confirmed'])
    report.update(status='PASS' if passed else 'FAIL',hardware_flashed=False,physical_validation_pending=True)
    origin(prep);save(f'COMPILACION_DETECTOR_INTENTO_{attempt}.json',report)
    if passed:save('COMPILACION_DETECTOR.json',report)
    print(json.dumps(report,ensure_ascii=True,indent=2),flush=True)
    if not passed:raise SystemExit(2)


def install():
    prep,built=load('PREPARACION.json'),load('COMPILACION_DETECTOR.json')
    require(built['status']=='PASS' and not (SOURCE/'CWRU_Detector').exists() and not (SOURCE/LAUNCH).exists(),'Destino/build no válido')
    require(not (OUT/'APLICACION_EN_D.json').exists(),'Instalación ya registrada');origin(prep)
    for rel,digest in built['overlay_sha256'].items():require(sha(OVERLAY/rel)==digest,'Overlay modificado')
    for rel in FILES:
        target=SOURCE/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(OVERLAY/rel,target)
    before,after=configs(BACKUP/'.cproject')[2],configs(SOURCE/'.cproject')[2]
    same=all(ET.tostring(c)==ET.tostring(after[n]) for n,c in before.items())
    resources=all(sha(SOURCE/r['path'])==r['sha256'] for r in prep['source_inventory_before'] if r['path'] not in FILES[:2])
    elves=all(sha(SOURCE/cfg/(NAME+'.elf'))==digest for cfg,digest in prep['existing_elf_sha256'].items())
    report=dict(status='PASS' if same and resources and elves else 'FAIL',modified_files=FILES[:2],added_files=FILES[2:],
        previous_ten_configurations_preserved=same,previous_ten_elfs_unchanged=elves,other_snapshot_resources_unchanged=resources,
        installed_sha256={r:sha(SOURCE/r) for r in FILES},hardware_flashed=False,serial_port_opened=False,physical_validation_pending=True)
    save('APLICACION_EN_D.json',report);print(json.dumps(report,ensure_ascii=True,indent=2));require(report['status']=='PASS','Instalación no válida')


def seal():
    require(not (OUT/'MANIFEST_SHA256.json').exists(),'Sello ya existente')
    prep,built,installed=load('PREPARACION.json'),load('COMPILACION_DETECTOR.json'),load('APLICACION_EN_D.json')
    require(built['status']==installed['status']=='PASS','Faltan comprobaciones')
    for rel,digest in installed['installed_sha256'].items():require(sha(SOURCE/rel)==digest,'D diferente')
    paths={p for p in OUT.iterdir() if p.is_file()}
    for row in prep['source_inventory_before']:paths.update((PROJECT/row['path'],BACKUP/row['path']))
    paths.update(PROJECT/r for r in FILES);paths.update(OVERLAY/r for r in FILES)
    paths.update(BACKUP/cfg/(NAME+'.elf') for cfg in prep['existing_elf_sha256'])
    for cfg in ('Release_180MHz_EntM',CONTROL,CONFIG):
        folder=PROJECT/cfg;paths.update(folder/r for r in (NAME+'.elf',NAME+'.map','objects.list'));paths.update((folder/'CWRU_Detector').glob('*.su'))
    paths.update(p for p in (OUT/'intentos').rglob('*') if p.is_file())
    paths.update(p for p in CLIENT.iterdir() if p.is_file());paths.update(p for p in TEMPLATES.iterdir() if p.is_file())
    paths.update(Path(__file__).with_name(n) for n in ('preparar_detector_cwru.py','preparar_detector_pc.py','validar_detector_cwru.py','contrato_detector_cwru.py'))
    save('MANIFEST_SHA256.json',{p.relative_to(ROOT).as_posix():sha(p) for p in sorted(paths)})
    with (OUT/'MANIFEST_SHA256.sha256').open('x',encoding='ascii') as f:f.write(sha(OUT/'MANIFEST_SHA256.json')+'  MANIFEST_SHA256.json\n')
    verify()


def verify():
    digest,name=(OUT/'MANIFEST_SHA256.sha256').read_text(encoding='ascii').strip().split()
    require(name=='MANIFEST_SHA256.json' and sha(OUT/name)==digest,'Sello diferente')
    manifest=load(name)
    for rel,digest in manifest.items():
        path=(ROOT/rel).resolve();require(path.is_relative_to(ROOT) and sha(path)==digest,'Recurso diferente: '+rel)
    for rel,digest in load('ENTRADAS_REFERENCIA_SHA256.json').items():require(sha(ROOT/rel)==digest,'Entrada de referencia diferente: '+rel)
    print(json.dumps(dict(status='PASS',files_verified=len(manifest),physical_validation_pending=True,serial_port_opened=False,active_D_rechecked=False)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('prepare','build','install','seal','verify'))
    {'prepare':prepare,'build':build,'install':install,'seal':seal,'verify':verify}[p.parse_args().mode]()
