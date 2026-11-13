"""Prepara una variante aislada CMSIS-NN Single Rounding del banco TotalV2.

Nunca instala en D:, programa la placa, abre COM ni entrena. La unica
implementacion sustituida es el matvec per-channel INT8 de CMSIS-NN.
El archivo .a y todas las configuraciones anteriores permanecen intactos.
El usuario inicia build; prepare exige el diagnostico confirmado y cliente final.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import os
import queue
import re
import shutil
import struct
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[4]
NAME = 'Prueba_2_CWRU_Extraccion'
SOURCE = Path('D:/STM32IDE') / NAME
IDE = Path('D:/STM32CubeIDE_2.2.0/STM32CubeIDE')
OLD = ROOT / 'despliegue_stm32/costo_total_representantes_v2_180mhz_2026-10-25'
OUT = ROOT / 'despliegue_stm32/costo_total_single_rounding_v2_180mhz_2026-10-26'
PROJECT = OUT / 'copia_aislada' / NAME
BACKUP = OUT / 'respaldo_antes_single_rounding'
OVERLAY = OUT / 'archivos_para_D'
CLIENT = OUT / 'Validacion_PC'
CONFIG = 'Release_180MHz_TotalV2_SR'
BASE = 'Release_180MHz_TotalV2'
ENTM = 'Release_180MHz_EntM'
SELECTED = (ENTM, BASE, CONFIG)
FUNC = 'arm_nn_vec_mat_mult_t_per_ch_s8'
C_SOURCE = Path('TFLM/third_party/cmsis_nn/Source/NNSupportFunctions') / (FUNC + '.c')
SR_SOURCE = Path('CWRU_TotalSR') / (FUNC + '.c')
LIBRARY = Path('Lib/libtensorflow-microlite.a')
LIBRARY_SHA = '84cd3fae745ce78530e49f353725f845588ca0298cc35bd01da98403eabbe974'
OLD_ELF_SHA = '85e8d7e2d1848833117fd4704493710cda0058fecedef3b379a80e9560171b1c'
LAUNCH = NAME + '_TotalV2_SR_180MHz_SinRecompilar.launch'
CLIENT_SOURCE = ROOT / 'codigo/cwru/ampliacion_v2/costo_total_single_rounding/validar_total_sr.py'
DIAGNOSIS = OLD / 'diagnostico_divergencia_20261026T042424Z/CONCLUSION_DIAGNOSTICO.json'


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def cp(source, target):
    source, target = Path(source), Path(target)
    digest = sha(source)
    require(not target.exists(), 'No sobrescribir: ' + str(target))
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    require(sha(source) == sha(target) == digest, 'Copia diferente: ' + str(source))
    return digest


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec and spec.loader, 'Modulo no importable')
    loaded = importlib.util.module_from_spec(spec); spec.loader.exec_module(loaded)
    return loaded


def prior():
    return module(ROOT / 'codigo/cwru/ampliacion_v2/costo_total/preparar_total.py', 'total_sr_prior')


def add_configuration(p, root, settings, old):
    require(len(old) == 16 and BASE in old and ENTM in old and CONFIG not in old,
            'Se esperan dieciseis configuraciones previas sin variante SR')
    node = copy.deepcopy(old[BASE]); p.unique_ids(node, '.single_rounding1')
    for element in node.iter():
        for key, value in list(element.attrib.items()):
            if key == 'name' and value == BASE:
                element.set(key, CONFIG)
            elif key == 'buildPath' and value.endswith('/' + BASE):
                element.set(key, value[:-len(BASE)] + CONFIG)
    for tool in node.iter('tool'):
        if tool.get('superClass', '').endswith(('tool.c.compiler', 'tool.cpp.compiler')):
            defines = p.option(tool, '.definedsymbols')
            for key in ('CMSIS_NN_USE_SINGLE_ROUNDING', 'TFLITE_SINGLE_ROUNDING'):
                require(not any(n.get('value', '').split('=')[0] == key for n in defines), 'Macro ya existe: ' + key)
                ET.SubElement(defines, 'listOptionValue', builtIn='false', value=key+'=1')
            includes = p.option(tool, '.includepaths')
            for value in ('../CWRU_TotalSR', '../TFLM/third_party/cmsis_nn/Include'):
                if not any(n.get('value') == value for n in includes):
                    ET.SubElement(includes, 'listOptionValue', builtIn='false', value=value)
    sources = node.find('.//sourceEntries')
    require(sources is not None and not any(n.get('name') == 'CWRU_TotalSR' for n in sources), 'Source entry SR ya existe')
    ET.SubElement(sources, 'entry', flags='VALUE_WORKSPACE_PATH|RESOLVED', kind='sourcePath', name='CWRU_TotalSR')
    old_ids = {n.get('id') for c in old.values() for n in c.iter() if n.get('id')}
    require(not old_ids.intersection(n.get('id') for n in node.iter() if n.get('id')), 'Colision de IDs CDT')
    settings.append(node)
    return node


def same_configs(p, path):
    old, new = p.configs(BACKUP / '.cproject')[2], p.configs(path)[2]
    return len(old) == 16 and len(new) == 17 and CONFIG in new and all(
        name in new and ET.tostring(n) == ET.tostring(new[name]) for name, n in old.items())


def archive_member(data, name):
    require(data[:8] == b'!<arch>\n', 'Archivo GNU ar invalido')
    position = 8; names = b''; matches = []
    while position < len(data):
        header = data[position:position+60]
        require(len(header) == 60 and header[58:60] == b'`\n', 'Cabecera ar invalida')
        length = int(header[48:58].decode('ascii').strip())
        label = header[:16].decode('ascii').strip()
        payload = data[position+60:position+60+length]
        require(len(payload) == length, 'Miembro ar truncado')
        if label == '//': names = payload
        elif re.fullmatch(r'/[0-9]+', label):
            offset = int(label[1:]); end = names.find(b'/\n', offset)
            require(end >= offset, 'Tabla de nombres ar invalida')
            label = names[offset:end].decode('ascii')
        else: label = label.rstrip('/')
        if label == name: matches.append(payload)
        position += 60+length+(length&1)
    require(len(matches) == 1, 'Miembro ar ausente o duplicado: ' + name)
    return matches[0]


def global_definitions(data):
    require(data[:7] == b'\x7fELF\x01\x01\x01' and struct.unpack_from('<H', data, 18)[0] == 40, 'Objeto no ELF32 ARM')
    offset = struct.unpack_from('<I', data, 32)[0]
    width,count = struct.unpack_from('<HH', data, 46)
    require(width == 40, 'Secciones ELF inesperadas')
    sections = [struct.unpack_from('<10I', data, offset+i*width) for i in range(count)]
    names = []
    for section in sections:
        if section[1] != 2: continue
        strings = sections[section[6]]
        table = data[strings[4]:strings[4]+strings[5]]
        require(section[9] == 16, 'Simbolos ELF inesperados')
        for pos in range(section[4], section[4]+section[5], 16):
            name,addr,size,info,other,index = struct.unpack_from('<IIIBBH', data, pos)
            if info>>4 in (1,2) and index != 0:
                end = table.find(b'\0', name)
                require(end >= name, 'String ELF invalida')
                names.append(table[name:end].decode('ascii'))
    return names


def parameter_scope():
    """Comprueba rutas FC per-channel y rangos de shifts de los32 modelos."""
    audited = module(ROOT / 'codigo/cwru/ampliacion_v2/analisis/auditar_elf_total_v2.py', 'total_sr_model_reader')
    cat = read(OLD / 'recursos/CATALOGO_TOTAL.json'); rows = []
    for rep in cat['representatives']:
        model = OLD / 'recursos' / rep['model_path']
        require(sha(model) == rep['model_sha256'], 'Modelo modificado')
        fb = audited.FlatBuffer(model.read_bytes()); start = fb.unpack('<I',0)[0]
        fb.validate(rep)
        graph = fb.tables(start,8)[0]; tensors = fb.tables(graph,4); ops = fb.tables(graph,10)
        shifts = []
        for op in ops[:3]:
            ins, outs = fb.vector(op,6,'<i'), fb.vector(op,8,'<i')
            inp, weights, out = (fb.tensor(tensors[i]) for i in (ins[0],ins[1],outs[0]))
            require(inp['type'] == weights['type'] == out['type'] == 9 and len(out['shape']) == 2,
                    'Ruta FC distinta de tensor INT8 de dos dimensiones')
            require(len(weights['scale_bits']) == weights['shape'][0] > 1 and all(z == 0 for z in weights['zero_points']),
                    'La variante solo sustituye FC per-channel simetrico')
            f32 = lambda b:struct.unpack('<f',struct.pack('<I',int(b,16)))[0]
            sin,sout = f32(inp['scale_bits'][0]), f32(out['scale_bits'][0])
            for scale in weights['scale_bits']:
                effective = sin*f32(scale)/sout
                fraction,shift = math.frexp(effective)
                multiplier = math.floor(fraction*(1<<31)+.5)
                if multiplier == 1<<31: multiplier//=2;shift+=1
                require(multiplier > 0 and -31 <= shift < 30, 'Shift fuera del dominio comun single/double')
                shifts.append(shift)
        require(len(shifts) == 33, 'Se esperan33canales FC por modelo')
        soft_input = fb.tensor(tensors[fb.vector(ops[3],6,'<i')[0]])
        soft_options = fb.pointer(ops[3],12)
        beta = fb.scalar(soft_options,4,'<f',1.0)
        require(math.isfinite(beta) and beta > 0, 'Softmax beta invalido')
        soft_real = min(beta*f32(soft_input['scale_bits'][0])*(1<<26),(1<<31)-1)
        fraction,soft_shift = math.frexp(soft_real)
        soft_multiplier = math.floor(fraction*(1<<31)+.5)
        if soft_multiplier == 1<<31: soft_multiplier//=2;soft_shift+=1
        require(soft_real > 1 and 0 <= soft_shift < 30, 'Softmax shift fuera del dominio comun')
        rows.append(dict(id=rep['id'],model_sha256=rep['model_sha256'],channels=33,
                         minimum_shift=min(shifts),maximum_shift=max(shifts),all_FC_per_channel=True,
                         softmax_quantized_multiplier=soft_multiplier,softmax_multiplier_shift=soft_shift))
    require(len(rows) == 32, 'Se esperan32modelos')
    return dict(models=rows,channels=1056,parameter_generation_unchanged_for_observed_shifts=True,
        all_FC_and_Softmax_multiplier_shifts_below30=True,
        implementation_scope='Three2D FullyConnected ops per model route to per-channel CMSIS matvec; Softmax and all other archive members remain original.',
        whole_TFLM_archive_rebuilt=False)


def origin(prep):
    for row in prep['source_inventory_before']:
        require(sha(SOURCE / row['path']) == row['sha256'], 'D cambio: ' + row['path'])
    for name,digest in prep['previous_ELF_sha256'].items():
        require(sha(SOURCE / name / (NAME+'.elf')) == digest, 'ELF previo D cambio: ' + name)
    require(not (SOURCE / CONFIG).exists() and not (SOURCE / SR_SOURCE.parent).exists()
            and not (SOURCE / LAUNCH).exists(), 'Destino SR ya existe en D')


def diagnosis_check(path):
    manifest=path.parent/'MANIFEST_DIAGNOSTICO_ROUNDING_SHA256.json'
    digest,name=manifest.with_suffix('.sha256').read_text(encoding='ascii').split()
    require(name==manifest.name and sha(manifest)==digest==
        '8be8a8ec59aec4ad4fac58a092e7dc64f1c5a5a3e66339ab430801bf09aca73c',
        'El diagnostico no corresponde al cierre aprobado')
    entries=read(manifest)
    require(len(entries)==15 and path.relative_to(ROOT).as_posix() in entries, 'Cierre diagnostico incompleto')
    for relative,expected in entries.items():
        target=(ROOT/relative).resolve()
        require(target.is_relative_to(ROOT.resolve()) and sha(target)==expected, 'Evidencia diagnostica cambio')
    conclusion=read(path)
    require(conclusion['status']=='CAUSE_ROUNDING_MISMATCH_CONFIRMED_PENDING_ARM_FIX'
        and conclusion['firmware_elf_sha256']==OLD_ELF_SHA
        and all(conclusion.get(key) is True for key in (
            'single_PC_layers_exact','double_rounding_in_linked_CMSIS_FC_verified_by_assembly',
            'double_softmax_matches_ARM','single_softmax_matches_PC')),
        'Causa de redondeo no confirmada')
    return manifest,digest


def prepare(diagnosis):
    require(not any(p.exists() for p in (PROJECT,BACKUP,CLIENT,OUT/'PREPARACION_SR.json')), 'No sobrescribir preparacion SR')
    p = prior(); old = p.configs(SOURCE / '.cproject')[2]
    require(len(old) == 16 and BASE in old and CONFIG not in old, 'Proyecto D inesperado')
    require(CLIENT_SOURCE.is_file() and diagnosis.is_file(), 'Falta cliente SR final o diagnostico confirmado')
    diagnosis_manifest,diagnosis_digest=diagnosis_check(diagnosis)
    require(sha(SOURCE / LIBRARY) == LIBRARY_SHA and sha(SOURCE / BASE / (NAME+'.elf')) == OLD_ELF_SHA,
            'Biblioteca o ELF de origen diferente del diagnosticado')
    archive = archive_member((SOURCE / LIBRARY).read_bytes(), FUNC+'.o')
    definitions = global_definitions(archive)
    require(definitions == [FUNC], 'El miembro sustituido define otros simbolos globales')
    p.resource_check(); scope = parameter_scope()
    skip = set(old) | {CONFIG,'.git','.metadata'}
    files = [f for f in sorted(SOURCE.rglob('*')) if f.is_file() and not any(
        part in skip for part in f.relative_to(SOURCE).parts)]
    inventory = []
    for path in files:
        relative = path.relative_to(SOURCE).as_posix(); digest = cp(path,PROJECT/relative)
        inventory.append(dict(path=relative,sha256=digest))
        if relative in ('.cproject','.settings/language.settings.xml'): cp(path,BACKUP/relative)
    elfs = {name:cp(SOURCE/name/(NAME+'.elf'),BACKUP/name/(NAME+'.elf')) for name in old}
    root,settings,nodes = p.configs(PROJECT / '.cproject')
    added = add_configuration(p,root,settings,nodes)
    (PROJECT / '.cproject').write_bytes(p.xml_bytes(root))
    require(same_configs(p,PROJECT/'.cproject'), 'Una configuracion previa cambio')
    cp(SOURCE/C_SOURCE,PROJECT/SR_SOURCE)
    launch = ET.parse(SOURCE/(NAME+'_TotalV2_180MHz_SinRecompilar.launch')).getroot()
    attrs = {n.get('key'):n for n in launch}
    elf = CONFIG+'/'+NAME+'.elf'
    attrs['org.eclipse.cdt.launch.PROGRAM_NAME'].set('value',elf)
    attrs['org.eclipse.cdt.launch.PROJECT_BUILD_CONFIG_ID_ATTR'].set('value',added.get('id'))
    attrs['org.eclipse.cdt.launch.PROJECT_BUILD_CONFIG_AUTO_ATTR'].set('value','false')
    attrs['org.eclipse.cdt.launch.ATTR_BUILD_BEFORE_LAUNCH_ATTR'].set('value','0')
    key = 'com.st.stm32cube.ide.mcu.debug.launch.loadList'
    loads = json.loads(attrs[key].get('value'))
    for item in loads['fItems']: item.update(fPath=elf,fPerformBuild=False)
    attrs[key].set('value',json.dumps(loads,separators=(',',':')))
    attrs['com.st.stm32cube.ide.mcu.debug.stlink.log_file'].set('value',str(SOURCE/CONFIG/'st-link_gdbserver_log.txt'))
    (PROJECT/LAUNCH).write_bytes(b'<?xml version="1.0" encoding="UTF-8"?>\n'+ET.tostring(launch,encoding='utf-8')+b'\n')
    for relative in (Path('.cproject'),SR_SOURCE,Path(LAUNCH)): cp(PROJECT/relative,OVERLAY/relative)
    for resource in sorted((OLD/'recursos').rglob('*')):
        if resource.is_file(): cp(resource,OUT/'recursos'/resource.relative_to(OLD/'recursos'))
    cp(CLIENT_SOURCE,CLIENT/'validar_total_sr.py')
    cp(diagnosis,OUT/'DIAGNOSTICO_ORIGEN.json')
    cp(diagnosis_manifest,OUT/diagnosis_manifest.name)
    cp(diagnosis_manifest.with_suffix('.sha256'),OUT/diagnosis_manifest.with_suffix('.sha256').name)
    headers = {f.relative_to(SOURCE).as_posix():sha(f) for f in sorted((SOURCE/'TFLM/third_party/cmsis_nn/Include').rglob('*.h'))}
    prep = dict(status='PREPARED_SR_NOT_BUILT_NOT_ARM_VALIDATED',project=str(PROJECT),source=str(SOURCE),
        configuration=CONFIG,old_configurations=16,new_configurations=17,source_inventory_before=inventory,
        previous_ELF_sha256=elfs,original_library_sha256=LIBRARY_SHA,old_total_ELF_sha256=OLD_ELF_SHA,
        replacement_source=dict(original=str(SOURCE/C_SOURCE),original_sha256=sha(SOURCE/C_SOURCE),
            prepared=str(SR_SOURCE),prepared_sha256=sha(PROJECT/SR_SOURCE),byte_exact_original_copy=True,
            original_archive_member_sha256=hashlib.sha256(archive).hexdigest(),archive_member_global_definitions=definitions),
        cmsis_header_sha256=headers,parameter_scope=scope,
        definitions_added=['CMSIS_NN_USE_SINGLE_ROUNDING=1','TFLITE_SINGLE_ROUNDING=1'],
        original_archive_modified=False,whole_precompiled_archive_single_rounding=False,
        overlay_sha256={str(f).replace('\\','/'):sha(OVERLAY/f) for f in (Path('.cproject'),SR_SOURCE,Path(LAUNCH))},
        client_sha256=sha(CLIENT_SOURCE),client_source=str(CLIENT_SOURCE),diagnosis_sha256=sha(diagnosis),diagnosis_source=str(diagnosis),
        diagnosis_manifest_sha256=diagnosis_digest,diagnosis_manifest_source=str(diagnosis_manifest),
        preparer_sha256=sha(__file__),prior_preparer_sha256=sha(ROOT/'codigo/cwru/ampliacion_v2/costo_total/preparar_total.py'),
        model_reader_sha256=sha(ROOT/'codigo/cwru/ampliacion_v2/analisis/auditar_elf_total_v2.py'),
        builds_executed=0,training_fits=0,serial_port_opened=False,hardware_flashed=False,D_files_modified=False,
        selector_costs_enabled=False,gate_requires_both_predefined_cases=True)
    origin(prep);save(OUT/'PREPARACION_SR.json',prep)
    print('PREPARED_SR_NOT_BUILT_NOT_ARM_VALIDATED; D intacto;17configuraciones aisladas.',flush=True)


def verify(prep):
    p = prior(); origin(prep)
    require(same_configs(p,PROJECT/'.cproject'), 'Configuraciones previas diferentes')
    require(sha(__file__) == prep['preparer_sha256'], 'Preparador SR cambio')
    require(sha(ROOT/'codigo/cwru/ampliacion_v2/costo_total/preparar_total.py') == prep['prior_preparer_sha256'], 'Preparador previo cambio')
    require(sha(ROOT/'codigo/cwru/ampliacion_v2/analisis/auditar_elf_total_v2.py') == prep['model_reader_sha256'], 'Lector modelos cambio')
    for relative,digest in prep['overlay_sha256'].items():
        require(sha(OVERLAY/relative) == sha(PROJECT/relative) == digest, 'Overlay cambio: '+relative)
    for row in prep['source_inventory_before']:
        if row['path'] not in ('.cproject','.settings/language.settings.xml'):
            require(sha(PROJECT/row['path']) == row['sha256'], 'Fuente de copia cambio: '+row['path'])
    require(sha(CLIENT/'validar_total_sr.py') == prep['client_sha256'] == sha(CLIENT_SOURCE), 'Cliente SR cambio')
    require(sha(OUT/'DIAGNOSTICO_ORIGEN.json') == prep['diagnosis_sha256'] == sha(Path(prep['diagnosis_source'])), 'Diagnostico cambio')
    manifest,digest=diagnosis_check(Path(prep['diagnosis_source']))
    require(digest==prep['diagnosis_manifest_sha256'] and sha(OUT/manifest.name)==digest, 'Sello diagnostico cambio')
    manifest = OUT/'recursos/MANIFEST_RECURSOS_SHA256.json'
    digest,name = manifest.with_suffix('.sha256').read_text(encoding='ascii').split()
    require(name == manifest.name and digest == sha(manifest) == sha(OLD/'recursos'/name), 'Sello de recursos cambio')
    for relative,digest in read(manifest).items(): require(sha(OUT/'recursos'/relative) == digest, 'Recurso SR diferente')
    return p


def toolchain():
    candidates = [p for p in (IDE/'plugins').glob('com.st.stm32cube.ide.mcu.externaltools.gnu-tools-for-stm32.*/tools/bin')
                  if (p/'arm-none-eabi-gcc.exe').is_file()]
    makes = [p for p in (IDE/'plugins').glob('com.st.stm32cube.ide.mcu.externaltools.make.*/tools/bin') if (p/'make.exe').is_file()]
    require(len(candidates) == len(makes) == 1, 'Toolchain o make ambiguos')
    return candidates[0],makes[0]


def smoke():
    """Sintaxis ARM y macros de la copia; no produce objetos o ELF."""
    prep=read(OUT/'PREPARACION_SR.json');p=verify(prep);tool,_=toolchain()
    config=p.configs(PROJECT/'.cproject')[2][CONFIG]
    compiler=next(n for n in config.iter('tool') if n.get('superClass','').endswith('tool.c.compiler'))
    args=[str(tool/'arm-none-eabi-gcc.exe'),'-mcpu=cortex-m4','-std=gnu11','-O3','-Wall','-Werror',
          '-mfpu=fpv4-sp-d16','-mfloat-abi=hard','-mthumb']
    args+=['-D'+n.get('value') for n in p.option(compiler,'.definedsymbols')]
    args+=['-I'+str((PROJECT/CONFIG/n.get('value')).resolve()) for n in p.option(compiler,'.includepaths')]
    syntax=subprocess.run(args+['-fsyntax-only',str(PROJECT/SR_SOURCE)],capture_output=True,text=True,timeout=60)
    preprocess=subprocess.run(args+['-dM','-E',str(PROJECT/SR_SOURCE)],capture_output=True,text=True,timeout=60)
    cpp_compiler=next(n for n in config.iter('tool') if n.get('superClass','').endswith('tool.cpp.compiler'))
    cpp_args=[str(tool/'arm-none-eabi-g++.exe'),'-mcpu=cortex-m4','-std=gnu++17','-O3','-Wall','-Werror',
        '-fno-exceptions','-fno-rtti','-fno-use-cxa-atexit','-fno-threadsafe-statics',
        '-mfpu=fpv4-sp-d16','-mfloat-abi=hard','-mthumb']
    cpp_args+=['-D'+n.get('value') for n in p.option(cpp_compiler,'.definedsymbols')]
    cpp_args+=['-I'+str((PROJECT/CONFIG/n.get('value')).resolve()) for n in p.option(cpp_compiler,'.includepaths')]
    cpp_source=PROJECT/'CWRU_TotalV2/cwru_total_nn.cpp'
    cpp_syntax=subprocess.run(cpp_args+['-fsyntax-only',str(cpp_source)],capture_output=True,text=True,timeout=60)
    required=('#define CMSIS_NN_USE_SINGLE_ROUNDING 1','#define TFLITE_SINGLE_ROUNDING 1','#define ARM_MATH_DSP 1')
    passed=syntax.returncode==preprocess.returncode==0 and not syntax.stdout and not syntax.stderr
    passed=passed and all(line in preprocess.stdout.splitlines() for line in required)
    passed=passed and cpp_syntax.returncode==0 and not cpp_syntax.stdout and not cpp_syntax.stderr
    report=dict(status='PASS_SR_SYNTAX_MACROS_NOT_BUILT' if passed else 'FAIL',command=args,
        syntax_returncode=syntax.returncode,syntax_output=syntax.stdout+syntax.stderr,
        preprocess_returncode=preprocess.returncode,preprocess_errors=preprocess.stderr,
        cpp_syntax=dict(command=cpp_args,source=str(cpp_source),source_sha256=sha(cpp_source),
            returncode=cpp_syntax.returncode,output=cpp_syntax.stdout+cpp_syntax.stderr),
        required_macro_lines=list(required),macros_confirmed=all(line in preprocess.stdout.splitlines() for line in required),
        source_sha256=sha(PROJECT/SR_SOURCE),D_files_modified=False,builds_executed=0,
        serial_port_opened=False,hardware_flashed=False)
    save(OUT/'SINTAXIS_SR.json',report)
    print(json.dumps(report,ensure_ascii=True,indent=2),flush=True)
    if not passed:raise SystemExit(2)


def link_order(makefile,libraries):
    recipes=[line.strip() for line in makefile.splitlines() if 'arm-none-eabi-g++ -o ' in line]
    declarations=re.findall(r'(?m)^LIBS\s*:?=\s*(.*)$',libraries)
    require(len(recipes)==len(declarations)==1,'Receta o declaracion de bibliotecas ambigua')
    recipe=recipes[0];libs=declarations[0].split()
    objects=re.search(r'@(?:"objects\.list"|objects\.list)',recipe)
    position=recipe.find('$(LIBS)')
    passed=objects is not None and position>objects.end() and libs==['-ltensorflow-microlite','-lm']
    passed=passed and '--whole-archive' not in recipe
    return dict(application_objects_before_prebuilt_archive=bool(passed),link_recipe=recipe,
                libraries_declaration=declarations[0])


def single_definition_report(commands,config):
    result={'compile_definition_values':{}}
    expected=['1'] if config==CONFIG else []
    for key in ('CMSIS_NN_USE_SINGLE_ROUNDING','TFLITE_SINGLE_ROUNDING'):
        actual=[re.findall(r'(?<!\S)-D'+key+r'(?:=([^\s]+))?(?=\s|$)',command) for command in commands]
        result['compile_definition_values'][key]=actual
        result[key+'_matches_expected']=bool(commands) and all(values==expected for values in actual)
    return result


def inspect(p,tool,config,prep,attempt):
    folder = PROJECT/config; elf = folder/(NAME+'.elf')
    numbers = subprocess.check_output([str(tool/'arm-none-eabi-size.exe'),str(elf)],text=True).splitlines()[-1].split()
    text,data,bss = map(int,numbers[:3]); commands = p.compiler_commands(folder)
    expected_config = p.CONFIG if config in (BASE,CONFIG) else p.CONTROL
    result = dict(elf_sha256=sha(elf),flash_bytes=text+data,ram_including_linker_reservations_bytes=data+bss,
                  flags_confirmed=p.flags(commands,expected_config))
    result.update(single_definition_report(commands,config))
    if config != CONFIG:
        result.update(control_ELF_identical=sha(elf) == prep['previous_ELF_sha256'][config],
            control_Flash_identical=p.flash_image(elf) == p.flash_image(BACKUP/config/(NAME+'.elf')))
        return result
    objects = (folder/'objects.list').read_text(encoding='utf-8')
    map_path = folder/(NAME+'.map'); map_text = map_path.read_text(encoding='utf-8')
    app_object = './CWRU_TotalSR/'+FUNC+'.o'
    linkage=link_order((folder/'makefile').read_text(encoding='utf-8'),(folder/'objects.mk').read_text(encoding='utf-8'))
    symbols = subprocess.check_output([str(tool/'arm-none-eabi-nm.exe'),'-S','--defined-only',str(elf)],text=True)
    definitions = re.findall(r'(?m)^([0-9a-f]+)\s+([0-9a-f]+)\s+[tT]\s+'+FUNC+r'$',symbols)
    require(len(definitions) == 1, 'Matvec no definido una sola vez')
    result.update(override=dict(application_object_in_objects_list=app_object in objects,
        application_object_in_map='CWRU_TotalSR/'+FUNC+'.o' in map_text,
        original_archive_member_not_extracted=('libtensorflow-microlite.a('+FUNC+'.o)') not in map_text,
        final_symbol_address=definitions[0][0],final_symbol_bytes=int(definitions[0][1],16),
        archive_unchanged=sha(PROJECT/LIBRARY) == LIBRARY_SHA,**linkage),
        physical_memory_limits_passed=text+data<=512*1024 and data+bss<=128*1024,
        machine_rounding_review_pending=True,model_bytes_and_parameters_review_pending=True,
        runtime_gate_both_cases_pending=True,whole_archive_rebuilt=False)
    (OUT/f'SIMBOLOS_SR_INTENTO_{attempt}.txt').write_text(symbols,encoding='utf-8')
    return result


def run_bounded(args,env,path,timeout):
    """Ejecuta solo el hijo CubeIDE, con cwd nuevo y corte de su arbol."""
    process=subprocess.Popen(args,cwd=OUT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
        text=True,encoding='utf-8',errors='replace')
    messages=queue.Queue()
    def capture():
        for line in process.stdout:messages.put(line)
        messages.put(None)
    threading.Thread(target=capture,daemon=True).start()
    started=time.monotonic();expired=False
    with path.open('x',encoding='utf-8',newline='\n') as log:
        while True:
            if time.monotonic()-started>timeout:
                expired=True
                subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],capture_output=True)
                process.wait(timeout=30)
                log.write('\nTIMEOUT: arbol hijo de este intento terminado.\n');break
            try:line=messages.get(timeout=.5)
            except queue.Empty:continue
            if line is None:break
            log.write(line);log.flush()
            if any(token in line.lower() for token in ('build of','error:','warning:','build finished')):
                print(line.strip()[:250],flush=True)
    return process.wait(timeout=30),expired,time.monotonic()-started


def build(timeout):
    prep = read(OUT/'PREPARACION_SR.json');p = verify(prep)
    require(not (OUT/'COMPILACION_SR.json').exists(), 'Build SR ya aprobado; no repetir')
    attempts = [int(f.stem.rsplit('_',1)[-1]) for f in OUT.glob('CUBEIDE_SR_INTENTO_*.log')]
    attempt = max(attempts,default=0)+1
    require(attempt<=3, 'Tres intentos agotados; revisar antes de ampliar')
    tool,make = toolchain(); env=dict(os.environ)
    env['PATH']=str(tool)+os.pathsep+str(make)+os.pathsep+env.get('PATH','')
    args=[str(IDE/'stm32cubeidec.exe'),'--launcher.suppressErrors','-nosplash','-application',
          'org.eclipse.cdt.managedbuilder.core.headlessbuild','-data',str(OUT/'workspace_aislado'/str(attempt)),
          '-import',str(PROJECT),'-no-indexer']
    for name in SELECTED:args.extend(('-cleanBuild',NAME+'/'+name))
    print(f'INTENTO{attempt}: EntM, TotalV2 y SR; presupuesto{timeout}s; sin placa/COM/entrenamiento.',flush=True)
    log=OUT/f'CUBEIDE_SR_INTENTO_{attempt}.log'
    code,expired,elapsed=run_bounded(args,env,log,timeout)
    txt=log.read_text(encoding='utf-8');counts=re.findall(r'(\d+) errors, (\d+) warnings',txt)
    report=dict(attempt=attempt,command=args,returncode=code,timed_out=expired,seconds=elapsed,
        zero_errors_and_warnings=len(counts)>=3 and all(a==b=='0' for a,b in counts)
          and not re.search(r'\b(error|warning):|undefined reference|multiple definition',txt,re.I),configurations={},
        D_files_modified=False,serial_port_opened=False,hardware_flashed=False,selector_costs_enabled=False)
    for name in SELECTED:
        try:report['configurations'][name]=inspect(p,tool,name,prep,attempt)
        except Exception as error:report['configurations'][name]=dict(inspection_error=str(error))
        for path in (PROJECT/name).rglob('*'):
            if path.is_file() and (path.suffix in ('.elf','.map','.su','.mk','.list') or path.name in ('makefile','objects.list')):
                cp(path,OUT/'intentos'/str(attempt)/'compilacion'/name/path.relative_to(PROJECT/name))
    try:verify(prep);report['integrity_after_build']=True
    except Exception as error:report.update(integrity_after_build=False,integrity_error=str(error))
    rows=report['configurations'];new=rows[CONFIG]
    passed=code==0 and not expired and report['zero_errors_and_warnings'] and report['integrity_after_build']
    passed=passed and all(r.get('flags_confirmed',False) and r.get('CMSIS_NN_USE_SINGLE_ROUNDING_matches_expected',False)
        and r.get('TFLITE_SINGLE_ROUNDING_matches_expected',False) for r in rows.values())
    passed=passed and all(rows[n].get('control_ELF_identical',False) and rows[n].get('control_Flash_identical',False) for n in SELECTED[:2])
    passed=passed and all(new.get('override',{}).get(k,False) for k in (
        'application_object_in_objects_list','application_object_in_map','original_archive_member_not_extracted',
        'archive_unchanged','application_objects_before_prebuilt_archive'))
    passed=passed and new.get('physical_memory_limits_passed',False)
    report['status']='PASS_SR_BUILD_PENDING_ELF_REVIEW_AND_ARM' if passed else 'FAIL'
    save(OUT/f'COMPILACION_SR_INTENTO_{attempt}.json',report)
    if passed:save(OUT/'COMPILACION_SR.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='configurations'},ensure_ascii=True,indent=2),flush=True)
    if not passed:raise SystemExit(2)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',nargs='?',choices=('prepare','verify','smoke','build'))
    modes=parser.add_mutually_exclusive_group()
    for name in ('prepare','verify','smoke','build'):
        modes.add_argument('--'+name,dest='flag_mode',action='store_const',const=name)
    parser.add_argument('--diagnosis-report',type=Path,default=DIAGNOSIS)
    parser.add_argument('--timeout',type=int,default=1800)
    args=parser.parse_args();require(60<=args.timeout<=3600,'Timeout fuera de60..3600')
    require((args.mode is None)!=(args.flag_mode is None),'Especifique un modo posicional o --modo')
    mode=args.mode or args.flag_mode
    if mode=='prepare':prepare(args.diagnosis_report.resolve())
    elif mode=='verify':verify(read(OUT/'PREPARACION_SR.json'));print('PREPARED_SR_INTEGRITY_PASS_NOT_ARM_VALIDATED')
    elif mode=='smoke':smoke()
    else:build(args.timeout)


if __name__=='__main__':
    main()
