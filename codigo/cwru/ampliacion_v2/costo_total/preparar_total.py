"""Copia aislada y compilacion auditable del banco de costo total V2.

Nunca escribe en D:, programa la placa, abre COM, ni ejecuta entrenamientos.
prepare copia fuentes y preserva las quince configuraciones actuales; build
lo inicia el usuario, con un intento por invocacion y limite temporal estricto.
La instalacion en D: requiere una etapa posterior al examen del ELF/mapa.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
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
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'despliegue_stm32/costo_total_representantes_v2_180mhz_2026-10-25'
RESOURCES = OUT / 'recursos'
PROJECT = OUT / 'copia_aislada' / NAME
BACKUP = OUT / 'respaldo_antes_total'
OVERLAY = OUT / 'archivos_para_D'
CLIENT = OUT / 'Validacion_PC'
CONFIG = 'Release_180MHz_TotalV2'
CONTROL = 'Release_180MHz_EntM'
MASKS = 'Release_180MHz_Masks'
SELECTED = (CONTROL, MASKS, CONFIG)
LAUNCH = NAME + '_TotalV2_180MHz.launch'
TEMPLATES = ('cwru_total.h', 'cwru_total.c', 'cwru_total_commands.inc', 'cwru_total_nn.h', 'cwru_total_nn.cpp')
GENERATED = ('cwru_total_catalog.h', 'cwru_total_catalog.cpp')
MODIFIED = ('.cproject', 'Src/uart_commands.c', 'Src/cmd_cwru_feat.c')
ADDED = (LAUNCH, *('CWRU_TotalV2/' + p for p in (*TEMPLATES, *GENERATED, 'cwru_total_debug_log.cc')))


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fnv(data):
    value = 2166136261
    for byte in data:
        value = ((value ^ byte) * 16777619) & 0xffffffff
    return value


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def cp(source, target):
    digest = sha(source)
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    require(not target.exists(), 'No sobrescribir: ' + str(target))
    shutil.copy2(source, target)
    require(sha(source) == sha(target) == digest, 'Copia diferente/origen cambiante: ' + str(source))
    return digest


def configs(path):
    root = ET.parse(path).getroot()
    settings = next(n for n in root if n.get('moduleId') == 'org.eclipse.cdt.core.settings')
    result = {}
    for node in settings:
        cfg = next(n for n in node.iter('configuration') if n.get('name'))
        name = cfg.get('name')
        require(name not in result, 'Configuracion duplicada')
        result[name] = node
    return root, settings, result


def xml_bytes(root):
    return (b'<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n'
            b'<?fileVersion 4.0.0?>\n' + ET.tostring(root, encoding='utf-8') + b'\n')


def unique_ids(node, suffix):
    mapping = {n.get('id'): n.get('id') + suffix for n in node.iter() if n.get('id')}
    for n in node.iter():
        for key, value in list(n.attrib.items()):
            if value in mapping:
                n.set(key, mapping[value])
            elif key == 'targetTool':
                n.set(key, ';'.join(mapping.get(v, v) for v in value.split(';')))


def option(tool, suffix):
    return next(n for n in tool.findall('option') if n.get('superClass', '').endswith(suffix))


def add_configuration(root, settings, old):
    require(len(old) == 15 and CONFIG not in old and {CONTROL, MASKS, 'Release_180MHz_Pipeline'} <= set(old),
            'Se esperan quince configuraciones actuales, EntM/Masks/Pipeline y ningun TotalV2')
    node = copy.deepcopy(old[MASKS])
    unique_ids(node, '.totalv2')
    for element in node.iter():
        for key, value in list(element.attrib.items()):
            if key == 'name' and value == MASKS:
                element.set(key, CONFIG)
            elif key == 'buildPath' and value.endswith('/' + MASKS):
                element.set(key, value[:-len(MASKS)] + CONFIG)
    pipeline = old['Release_180MHz_Pipeline']
    for tool in node.iter('tool'):
        kind = tool.get('superClass', '')
        if kind.endswith(('tool.c.compiler', 'tool.cpp.compiler')):
            defines = option(tool, '.definedsymbols')
            require(not any(v.get('value', '').split('=')[0] == 'CWRU_TOTALV2_TEST' for v in defines), 'Macro ya existe')
            ET.SubElement(defines, 'listOptionValue', builtIn='false', value='CWRU_TOTALV2_TEST=1')
            includes = option(tool, '.includepaths')
            ET.SubElement(includes, 'listOptionValue', builtIn='false', value='../CWRU_TotalV2')
        if kind.endswith('tool.cpp.compiler'):
            donor = next(t for t in pipeline.iter('tool') if t.get('superClass') == kind)
            copied = copy.deepcopy(option(donor, '.otherflags'))
            unique_ids(copied, '.totalv2')
            tool.append(copied)
        if kind.endswith('tool.cpp.linker'):
            libs = option(tool, '.libraries')
            require([n.get('value') for n in libs] == ['m'], 'Bibliotecas Masks inesperadas')
            libs.insert(0, ET.Element('listOptionValue', builtIn='false', value='tensorflow-microlite'))
            donor = next(t for t in pipeline.iter('tool') if t.get('superClass') == kind)
            copied = copy.deepcopy(option(donor, '.directories'))
            unique_ids(copied, '.totalv2')
            tool.append(copied)
    ET.SubElement(node.find('.//sourceEntries'), 'entry', flags='VALUE_WORKSPACE_PATH|RESOLVED',
                  kind='sourcePath', name='CWRU_TotalV2')
    oldids = {n.get('id') for c in old.values() for n in c.iter() if n.get('id')}
    newids = [n.get('id') for n in node.iter() if n.get('id')]
    # CDT repite deliberadamente el ID de cconfiguration en configuration.
    # La identidad se conserva dentro del clon; ningun ID puede colisionar con
    # las quince configuraciones anteriores.
    require(not oldids.intersection(newids), 'IDs nuevos colisionan con configuraciones previas')
    settings.append(node)
    return node


def patch_uart(text):
    guard = ('\n#ifndef CWRU_TOTALV2_TEST\n#define CWRU_TOTALV2_TEST 0\n#endif\n'
             '#if CWRU_TOTALV2_TEST\n#include "cwru_total.h"\n#endif\n')
    require('CWRU_TOTALV2_TEST' not in text, 'UART ya parcheado')
    anchor = '#ifndef CWRU_MASK_BENCH'
    require(text.count(anchor) == 1, 'Guardia Masks ambigua')
    text = text.replace(anchor, guard + '\n' + anchor, 1)
    route = ('#if CWRU_TOTALV2_TEST\n    if (strncmp(comando, "CWRUTOTAL_", 10) == 0)\n'
             '    {\n        CWRUTOTAL_Command(comando, respuesta, capacidad_respuesta);\n        return;\n    }\n#endif\n')
    anchor = '#if CWRU_MASK_BENCH\n    if (strncmp(comando, "CWRUMASK_", 9) == 0)'
    require(text.count(anchor) == 1, 'Ruta UART Masks ambigua')
    return text.replace(anchor, route + anchor, 1)


def patch_receiver(text):
    require('CWRU_TOTALV2_TEST' not in text and text.count('CWRUMASK_Invalidate();') == 3,
            'Receptor previo ambiguo')
    guard = ('#ifndef CWRU_TOTALV2_TEST\n#define CWRU_TOTALV2_TEST 0\n#endif\n'
             '#if CWRU_TOTALV2_TEST\n#include "cwru_total.h"\n#endif\n\n')
    text = text.replace('#ifndef CWRU_MASK_BENCH', guard + '#ifndef CWRU_MASK_BENCH', 1)
    anchor = '#if CWRU_MASK_BENCH\n    CWRUMASK_Invalidate();\n#endif'
    require(text.count(anchor) == 3, 'Tres invalidaciones previas esperadas')
    text = text.replace(anchor, '#if CWRU_TOTALV2_TEST\n    CWRUTOTAL_Invalidate();\n#endif\n' + anchor)
    return text + '\n#if CWRU_TOTALV2_TEST\n#include "cwru_total_commands.inc"\n#endif\n'


def same_configs(path):
    old = configs(BACKUP / '.cproject')[2]
    new = configs(path)[2]
    return len(old) == 15 and len(new) == 16 and CONFIG in new and all(
        name in new and ET.tostring(value) == ET.tostring(new[name]) for name, value in old.items())


def resource_check():
    manifest = RESOURCES / 'MANIFEST_RECURSOS_SHA256.json'
    require(manifest.is_file(), 'Exportar recursos antes de prepare')
    digest, declared = manifest.with_suffix('.sha256').read_text(encoding='ascii').split()
    require(declared == manifest.name and sha(manifest) == digest, 'Sello de recursos diferente')
    entries = load(manifest)
    require(len(entries) == 39, 'Se esperan 39 recursos exportados')
    for name, digest in entries.items():
        p = (RESOURCES / name).resolve()
        require(p.is_relative_to(RESOURCES.resolve()) and sha(p) == digest, 'Recurso diferente: ' + name)
    catalog = load(RESOURCES / 'CATALOGO_TOTAL.json')
    require(catalog['count'] == 32, 'Catalogo distinto de 32')
    rows = catalog['representatives']
    require([r['id'] for r in rows] == list(range(32)), 'IDs representantes diferentes')
    keys = ('id', 'representative_id', 'model_sha256', 'source_fit_signature', 'params_bytes_hex', 'scope')
    canonical = json.dumps([{k: r[k] for k in keys} for r in rows], ensure_ascii=False,
                           sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')
    require((RESOURCES / 'CATALOGO_CANONICO.json').read_bytes() == canonical and
            hashlib.sha256(canonical).hexdigest() == catalog['catalog_sha256'], 'Catalogo canonico diferente')
    pairs = bytearray()
    for row in rows:
        model = (RESOURCES / row['model_path']).resolve()
        params = bytes.fromhex(row['params_bytes_hex'])
        require(model.is_relative_to(RESOURCES.resolve()) and sha(model) == row['model_sha256'] and
                model.stat().st_size == row['model_bytes'] and fnv(model.read_bytes()) == int(row['model_fnv'], 16),
                'Modelo exportado diferente')
        require(len(params) == 297 and fnv(params) == int(row['params_fnv'], 16), 'Parametros exportados diferentes')
        pairs.extend(struct.pack('<II', int(row['model_fnv'], 16), int(row['params_fnv'], 16)))
    require(f'{fnv(pairs):08x}' == catalog['catalog_fnv'], 'CatalogFNV diferente')
    for name, expected in load(RESOURCES / 'FUENTES_SHA256.json').items():
        p = (ROOT / name).resolve()
        require(p.is_relative_to(ROOT) and sha(p) == expected, 'Fuente de recurso cambio: ' + name)
    return entries


def source_inventory(old):
    skip = set(old) | {'.git', '.metadata', 'CWRU_TotalV2', CONFIG}
    return [p for p in sorted(SOURCE.rglob('*')) if p.is_file() and not any(part in skip for part in p.relative_to(SOURCE).parts)]


def origin(prep):
    for row in prep['source_inventory_before']:
        require(sha(SOURCE / row['path']) == row['sha256'], 'Fuente D cambio: ' + row['path'])
    for name, digest in prep['existing_elf_sha256'].items():
        require(sha(SOURCE / name / (NAME + '.elf')) == digest, 'ELF D cambio: ' + name)
    require(not any((SOURCE / name).exists() for name in ADDED), 'Destino TotalV2 ya existe en D')


def prepare():
    require(not any(p.exists() for p in (PROJECT, BACKUP, CLIENT, OUT / 'PREPARACION.json')), 'No sobrescribir preparacion')
    old = configs(SOURCE / '.cproject')[2]
    require(len(old) == 15 and CONFIG not in old and not (SOURCE / 'CWRU_TotalV2').exists(), 'Proyecto origen inesperado')
    resources = resource_check()
    for name in (*TEMPLATES, 'validar_total.py'):
        require((HERE / name).is_file(), 'Falta fuente final: ' + name)
    for name in GENERATED:
        require((RESOURCES / name).is_file(), 'Falta catalogo C: ' + name)
    for name in old:
        require((SOURCE / name / (NAME + '.elf')).is_file(), 'Falta ELF previo: ' + name)
    snapshot = []
    for path in source_inventory(old):
        name = path.relative_to(SOURCE).as_posix()
        digest = cp(path, PROJECT / name)
        if name in (*MODIFIED, '.settings/language.settings.xml'):
            require(cp(path, BACKUP / name) == digest, 'Respaldo diferente')
        snapshot.append(dict(path=name, sha256=digest))
    elfs = {name: cp(SOURCE / name / (NAME + '.elf'), BACKUP / name / (NAME + '.elf')) for name in old}
    root, settings, old = configs(PROJECT / '.cproject')
    node = add_configuration(root, settings, old)
    (PROJECT / '.cproject').write_bytes(xml_bytes(root))
    require(same_configs(PROJECT / '.cproject'), 'Configuraciones anteriores alteradas')
    for name in TEMPLATES:
        cp(HERE / name, PROJECT / 'CWRU_TotalV2' / name)
    for name in GENERATED:
        cp(RESOURCES / name, PROJECT / 'CWRU_TotalV2' / name)
    shim = (SOURCE / 'CWRU_DNN/cwru_debug_log.cc').read_text(encoding='utf-8')
    require(shim.count('#include "cwru_nn.h"') == 1 and shim.count('CWRUNN_LastLog') == 1, 'Shim previo ambiguo')
    shim = shim.replace('#include "cwru_nn.h"', '#include "cwru_total_nn.h"').replace('CWRUNN_LastLog', 'CWRUTNN_LastLog')
    (PROJECT / 'CWRU_TotalV2/cwru_total_debug_log.cc').write_text(shim, encoding='utf-8', newline='\n')
    for name, patch in ((MODIFIED[1], patch_uart), (MODIFIED[2], patch_receiver)):
        path = PROJECT / name
        path.write_text(patch(path.read_text(encoding='utf-8')), encoding='utf-8', newline='\n')
    launch = ET.parse(SOURCE / (NAME + '_Masks_180MHz.launch')).getroot()
    attrs = {n.get('key'): n for n in launch}
    elf = CONFIG + '/' + NAME + '.elf'
    attrs['org.eclipse.cdt.launch.PROGRAM_NAME'].set('value', elf)
    attrs['org.eclipse.cdt.launch.PROJECT_BUILD_CONFIG_ID_ATTR'].set('value', node.get('id'))
    attrs['org.eclipse.cdt.launch.PROJECT_BUILD_CONFIG_AUTO_ATTR'].set('value', 'false')
    key = 'com.st.stm32cube.ide.mcu.debug.launch.loadList'
    loads = json.loads(attrs[key].get('value'))
    for item in loads['fItems']:
        item['fPath'] = elf
    attrs[key].set('value', json.dumps(loads, separators=(',', ':')))
    for key, attr in attrs.items():
        if key.endswith('stlink.log_file'):
            attr.set('value', str(SOURCE / CONFIG / 'st-link_gdbserver_log.txt'))
    (PROJECT / LAUNCH).write_bytes(b'<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(launch, encoding='utf-8') + b'\n')
    cp(HERE / 'validar_total.py', CLIENT / 'validar_total.py')
    for name in (*MODIFIED, *ADDED):
        cp(PROJECT / name, OVERLAY / name)
    prep = dict(status='PREPARED_NOT_BUILT_NOT_ARM_VALIDATED', source=str(SOURCE), project=str(PROJECT),
        configuration=CONFIG, source_inventory_before=snapshot, existing_elf_sha256=elfs,
        previous_fifteen_configurations_preserved=True,
        changed_files=list(MODIFIED), added_files=list(ADDED), resources=resources,
        templates_sha256={name: sha(HERE / name) for name in TEMPLATES},
        client_sha256=sha(CLIENT / 'validar_total.py'),
        overlay_sha256={name: sha(OVERLAY / name) for name in (*MODIFIED, *ADDED)},
        source_scripts_sha256={p.name: sha(p) for p in sorted(HERE.glob('*.py'))},
        hardware_flashed=False, serial_port_opened=False, d_files_modified=False,
        physical_validation_pending=True, selector_costs_enabled=False)
    origin(prep)
    save(OUT / 'PREPARACION.json', prep)
    print('PREPARED: copia aislada con 16 configuraciones; D intacto; compilacion y validacion fisica pendientes', flush=True)


def compiler_commands(folder):
    return [line for makefile in sorted(folder.rglob('subdir.mk'))
            for line in makefile.read_text(encoding='utf-8').splitlines()
            if re.search(r'\barm-none-eabi-g(?:cc|\+\+)(?=\s)', line) and '-c ' in line
            and re.search(r'-std=gnu(?:\+\+|\d)', line)]


def flags(commands, config):
    def macro(line, name, expected):
        values = re.findall(r'(?<!\S)-D' + name + r'(?:=([^\s]+))?(?=\s|$)', line)
        return values == [expected] if expected is not None else values in ([], ['0'])
    expected = dict(CWRU_TARGET_HCLK_HZ='180000000U', CWRU_CACHE_SPECTRAL_POWER='1', CWRU_ENTROPY_MODE='2',
        CWRU_TOTALV2_TEST='1' if config == CONFIG else None,
        CWRU_MASK_BENCH='1' if config in (MASKS, CONFIG) else None,
        CWRU_INPUT_TRANSFORM=None, CWRU_DNN_TEST=None, CWRU_NORMALITY_TEST=None,
        CWRU_PIPELINE_TEST=None, CWRU_STACK_TEST=None, CWRU_COST_PROFILE=None, CWRU_PROFILE_EXTRACTOR=None)
    return bool(commands) and all(re.findall(r'(?<!\S)-O\S+', line) == ['-O3'] and
        all(macro(line, k, v) for k, v in expected.items()) and
        not any(x in line for x in ('-ffast-math', '-Ofast', '-funsafe-math-optimizations')) and
        (config != CONFIG or '-I../CWRU_TotalV2' in line) for line in commands)


def flash_image(path):
    data = Path(path).read_bytes()
    require(data[:6] == b'\x7fELF\x01\x01', 'Se requiere ELF32 LE')
    offset = struct.unpack_from('<I', data, 28)[0]
    step, count = struct.unpack_from('<HH', data, 42)
    result = []
    for i in range(count):
        typ, pos, _, physical, size, _, _, _ = struct.unpack_from('<8I', data, offset + i * step)
        if typ == 1 and size and 0x08000000 <= physical < 0x08080000:
            result.append((physical, data[pos:pos + size]))
    return sorted(result)


def inspect(tool, config, prep, attempt):
    folder = PROJECT / config
    elf = folder / (NAME + '.elf')
    numbers = subprocess.check_output([str(tool / 'arm-none-eabi-size.exe'), str(elf)], text=True).splitlines()[-1].split()
    text, data, bss = map(int, numbers[:3])
    result = dict(elf_sha256=sha(elf), flash_load_bytes=text + data,
        ram_static_with_linker_reservations_bytes=data + bss,
        flags_confirmed=flags(compiler_commands(folder), config))
    if config != CONFIG:
        result.update(control_elf_identical=sha(elf) == prep['existing_elf_sha256'][config],
            control_flash_identical=flash_image(elf) == flash_image(BACKUP / config / (NAME + '.elf')))
        return result
    symbols = subprocess.check_output([str(tool / 'arm-none-eabi-nm.exe'), '-S', '-C', '--defined-only', str(elf)], text=True)
    (OUT / f'SIMBOLOS_TOTAL_INTENTO_{attempt}.txt').write_text(symbols, encoding='utf-8')
    allocations = {}
    for name, size in {'g_window': 32768, 'g_fft_input': 16384, 'g_fft_output': 16384}.items():
        matches = re.findall(r'(?m)^([0-9a-f]+)\s+([0-9a-f]+)\s+[bBdD]\s+' + name + r'$', symbols)
        require(len(matches) == 1 and int(matches[0][1], 16) == size, 'Buffer diferente: ' + name)
        allocations[name] = [int(matches[0][0], 16), size]
    large = [(int(a, 16), int(b, 16), n) for a, b, n in re.findall(
        r'(?m)^([0-9a-f]+)\s+([0-9a-f]+)\s+[bBdD]\s+(.+)$', symbols)
        if int(b, 16) >= 16384 and 0x20000000 <= int(a, 16) < 0x20020000]
    require(len(large) == 4, 'Se esperan ventana, dos FFT y una unica arena de 16 KiB')
    arena = [row for row in large if row[2] not in allocations]
    require(len(arena) == 1 and arena[0][1] == 16384, 'Arena unica inesperada')
    allocations[arena[0][2]] = list(arena[0][:2])
    ranges = sorted((address, address + size) for address, size in allocations.values())
    require(all(a[1] <= b[0] for a, b in zip(ranges, ranges[1:])), 'Buffers superpuestos')
    required = ('CWRUTNN_Select', 'CWRUTNN_Predict', 'CWRUTOTAL_Execute', 'CWRUTOTAL_Command',
                'CWRUMASK_Extract', 'CWRU_FeatureExtraction_Extract')
    result.update(buffer_audit=dict(status='PASS', allocations=allocations, large_buffers=4, one_reused_arena=True),
        required_symbols_present=all(re.search(r'(?m)^.*\s+[tT]\s+' + n + r'$', symbols) for n in required),
        within_physical_memories=(text + data <= 512 * 1024 and data + bss <= 128 * 1024),
        stack_reservation_bytes=4096, heap_reservation_bytes=512,
        arena_runtime_allocation_pending=True, flash_of_32_model_harness_not_final_deployment=True,
        compiler_stack_frames={p.relative_to(folder).as_posix(): p.read_text(encoding='utf-8').splitlines()
                               for p in folder.rglob('*.su')})
    return result


def verify_prepared(prep):
    origin(prep)
    require(same_configs(PROJECT / '.cproject'), 'Config previa alterada')
    resource_check()
    require(sha(CLIENT / 'validar_total.py') == prep['client_sha256'] == sha(HERE / 'validar_total.py'), 'Cliente cambio')
    for name, digest in prep['overlay_sha256'].items():
        require(sha(OVERLAY / name) == digest == sha(PROJECT / name), 'Overlay/proyecto cambio: ' + name)
    for name, digest in prep['source_scripts_sha256'].items():
        require(sha(HERE / name) == digest, 'Script cambio desde prepare: ' + name)
    for row in prep['source_inventory_before']:
        if row['path'] not in (*MODIFIED, '.settings/language.settings.xml'):
            require(sha(PROJECT / row['path']) == row['sha256'], 'Fuente copia cambio: ' + row['path'])


def run_bounded(args, env, logpath, timeout):
    """Lee progreso sin bloquear el plazo; mata solo el arbol hijo al agotarlo."""
    process = subprocess.Popen(args, cwd=OUT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, encoding='utf-8', errors='replace')
    output = queue.Queue()
    def read():
        for line in process.stdout:
            output.put(line)
        output.put(None)
    thread = threading.Thread(target=read, daemon=True)
    thread.start()
    start = time.monotonic()
    timed_out = False
    with logpath.open('x', encoding='utf-8', newline='\n') as log:
        while True:
            if time.monotonic() - start > timeout:
                timed_out = True
                subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'], capture_output=True)
                process.wait(timeout=30)
                log.write('\nTIMEOUT: proceso hijo terminado al alcanzar el limite.\n')
                break
            try:
                line = output.get(timeout=0.5)
            except queue.Empty:
                continue
            if line is None:
                break
            log.write(line)
            log.flush()
            if any(t in line.lower() for t in ('build of', 'error:', 'warning:', 'build finished')):
                print(line.strip()[:250], flush=True)
    return process.wait(timeout=30), timed_out, time.monotonic() - start


def build(timeout):
    prep = load(OUT / 'PREPARACION.json')
    verify_prepared(prep)
    require(not (OUT / 'COMPILACION_TOTAL.json').exists(), 'Build ya aprobado; no repetir')
    attempts = [int(p.stem.rsplit('_', 1)[-1]) for p in OUT.glob('CUBEIDE_TOTAL_INTENTO_*.log')]
    attempt = max(attempts, default=0) + 1
    require(attempt <= 3, 'Tres intentos agotados; revisar fallos antes de ampliar')
    tools = list((IDE / 'plugins').glob('com.st.stm32cube.ide.mcu.externaltools.gnu-tools-for-stm32.*/tools/bin'))
    makes = list((IDE / 'plugins').glob('com.st.stm32cube.ide.mcu.externaltools.make.*/tools/bin'))
    require(len(tools) == len(makes) == 1, 'Toolchain ambigua')
    tool = tools[0]
    env = dict(os.environ)
    env['PATH'] = str(tool) + os.pathsep + str(makes[0]) + os.pathsep + env.get('PATH', '')
    env['PYTHONIOENCODING'] = 'utf-8'
    args = [str(IDE / 'stm32cubeidec.exe'), '--launcher.suppressErrors', '-nosplash', '-application',
        'org.eclipse.cdt.managedbuilder.core.headlessbuild', '-data', str(OUT / 'workspace_aislado' / str(attempt)),
        '-import', str(PROJECT), '-no-indexer']
    for name in SELECTED:
        args.extend(('-cleanBuild', NAME + '/' + name))
    print(f'INTENTO {attempt}: tres builds aislados; limite {timeout}s; sin programar placa ni abrir COM.', flush=True)
    logpath = OUT / f'CUBEIDE_TOTAL_INTENTO_{attempt}.log'
    code, expired, elapsed = run_bounded(args, env, logpath, timeout)
    text = logpath.read_text(encoding='utf-8')
    summaries = re.findall(r'(\d+) errors, (\d+) warnings', text)
    report = dict(attempt=attempt, command=args, returncode=code, timed_out=expired, seconds=elapsed,
        zero_errors_warnings=len(summaries) >= 3 and all(a == b == '0' for a, b in summaries)
            and not re.search(r'\b(error|warning):|undefined reference', text, re.I), configurations={},
        hardware_flashed=False, serial_port_opened=False, d_files_modified=False, selector_costs_enabled=False)
    for name in SELECTED:
        try:
            report['configurations'][name] = inspect(tool, name, prep, attempt)
        except Exception as error:
            report['configurations'][name] = dict(inspection_error=str(error))
        for path in (PROJECT / name).rglob('*'):
            if path.is_file() and (path.suffix in ('.elf', '.map', '.su', '.mk') or path.name in ('makefile', 'objects.list')):
                cp(path, OUT / 'intentos' / str(attempt) / 'compilacion' / name / path.relative_to(PROJECT / name))
    try:
        verify_prepared(prep)
        report['prepared_resources_unchanged'] = True
    except Exception as error:
        report['prepared_resources_unchanged'] = False
        report['integrity_error'] = str(error)
    rows = report['configurations']
    totals = rows[CONFIG]
    passed = code == 0 and not expired and report['zero_errors_warnings'] and report['prepared_resources_unchanged']
    passed = passed and all(row.get('flags_confirmed', False) for row in rows.values())
    # Un ELF puede cambiar por rutas DWARF. La imagen PT_LOAD que se programa
    # debe ser identica; se informa por separado la identidad del ELF completo.
    passed = passed and all(rows[n].get('control_flash_identical', False) for n in SELECTED[:2])
    passed = passed and totals.get('buffer_audit', {}).get('status') == 'PASS' and totals.get('required_symbols_present', False) and totals.get('within_physical_memories', False)
    report['status'] = 'PASS_PENDING_ELF_MANUAL_REVIEW_AND_ARM' if passed else 'FAIL'
    save(OUT / f'COMPILACION_TOTAL_INTENTO_{attempt}.json', report)
    if passed:
        save(OUT / 'COMPILACION_TOTAL.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'configurations'}, ensure_ascii=True, indent=2), flush=True)
    if not passed:
        raise SystemExit(2)


def smoke():
    """Comprobaciones rapidas: sintaxis ARM y preprocesado de controles."""
    prep = load(OUT / 'PREPARACION.json')
    verify_prepared(prep)
    tools = list((IDE / 'plugins').glob('com.st.stm32cube.ide.mcu.externaltools.gnu-tools-for-stm32.*/tools/bin'))
    require(len(tools) == 1, 'Toolchain ambigua')
    tool = tools[0]
    old = configs(PROJECT / '.cproject')[2]
    def compiler(config, cpp=False):
        kind = 'tool.cpp.compiler' if cpp else 'tool.c.compiler'
        node = next(t for t in old[config].iter('tool') if t.get('superClass', '').endswith(kind))
        args = [str(tool / ('arm-none-eabi-g++.exe' if cpp else 'arm-none-eabi-gcc.exe')),
                '-mcpu=cortex-m4', '-mfpu=fpv4-sp-d16', '-mfloat-abi=hard', '-mthumb', '-Wall', '-O3',
                '-std=gnu++17' if cpp else '-std=gnu11']
        if cpp:
            args += ['-fno-exceptions', '-fno-rtti', '-fno-use-cxa-atexit', '-fno-threadsafe-statics']
        args += ['-D' + n.get('value') for n in option(node, '.definedsymbols')]
        args += ['-I' + str((PROJECT / config / n.get('value')).resolve()) for n in option(node, '.includepaths')]
        return args
    report = dict(status='FAIL', simulation_only=True, syntax={}, old_preprocessed={}, serial_port_opened=False,
                  hardware_flashed=False, d_files_modified=False)
    files = ('cwru_total.c', 'cwru_total_nn.cpp', 'cwru_total_catalog.cpp', 'cwru_total_debug_log.cc')
    cases = [('CWRU_TotalV2/' + n, n.endswith(('.cpp', '.cc'))) for n in files]
    cases += [(n, False) for n in MODIFIED[1:]]
    passed = True
    for name, cpp in cases:
        args = compiler(CONFIG, cpp) + ['-fsyntax-only', '-Werror', str(PROJECT / name)]
        result = subprocess.run(args, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=60)
        report['syntax'][name] = dict(returncode=result.returncode, output=result.stdout + result.stderr, command=args)
        passed = passed and result.returncode == 0
    for config in (CONTROL, MASKS):
        for name in MODIFIED[1:]:
            args = compiler(config) + ['-E', '-P']
            before = subprocess.check_output(args + [str(BACKUP / name)], timeout=60)
            after = subprocess.check_output(args + [str(PROJECT / name)], timeout=60)
            report['old_preprocessed'][config + ':' + name] = dict(identical=before == after,
                before_sha256=hashlib.sha256(before).hexdigest(), after_sha256=hashlib.sha256(after).hexdigest())
            passed = passed and before == after
    verify_prepared(prep)
    report['status'] = 'PASS_SYNTAX_AND_CONTROL_PREPROCESSING' if passed else 'FAIL'
    save(OUT / 'SINTAXIS_TOTAL.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'syntax'}, ensure_ascii=True, indent=2), flush=True)
    if not passed:
        raise SystemExit(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'verify', 'smoke', 'build'))
    parser.add_argument('--timeout', type=int, default=1800, help='Limite build, segundos (60..3600)')
    args = parser.parse_args()
    require(60 <= args.timeout <= 3600, 'timeout fuera de 60..3600')
    if args.mode == 'prepare':
        prepare()
    elif args.mode == 'verify':
        verify_prepared(load(OUT / 'PREPARACION.json'))
        print('PREPARED_INTEGRITY_PASS; no es validacion ARM', flush=True)
    elif args.mode == 'smoke':
        smoke()
    else:
        build(args.timeout)


if __name__ == '__main__':
    main()
