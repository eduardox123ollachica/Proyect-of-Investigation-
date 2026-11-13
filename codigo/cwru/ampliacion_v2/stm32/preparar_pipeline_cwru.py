"""Pipeline opcional: copia aislada, tres builds, instalación respaldada y sello.

Sólo el modo install escribe en D:. No abre COM ni programa la placa. prepare
deja pendientes las guardias C de UART y del receptor, que se aplican por separado.
Cada invocación de build constituye un intento; no reintenta automáticamente.
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
import time
import xml.etree.ElementTree as ET
from pathlib import Path

from preparar_cache_potencias import configs, sha
from preparar_reloj_180mhz import flash_image

ROOT = Path(__file__).resolve().parents[4]
NAME = 'Prueba_2_CWRU_Extraccion'
SOURCE = Path('D:/STM32IDE') / NAME
OUT = ROOT / 'despliegue_stm32/integracion_pipeline_180mhz_2026-10-23'
PRIOR = ROOT / 'despliegue_stm32/detector_normalidad_180mhz_2026-10-23'
PROJECT = OUT / 'copia_aislada' / NAME
BACKUP = OUT / 'respaldo_antes_pipeline'
OVERLAY = OUT / 'archivos_para_D'
CLIENT = OUT / 'Validacion_PC'
VECTORS = ROOT / 'despliegue_stm32/mejoras_entropia_180mhz_2026-10-23/Validacion_Mixta'
TEMPLATES = Path(__file__).with_name('pipeline_firmware')
IDE = Path('D:/STM32CubeIDE_2.2.0/STM32CubeIDE')
CONTROL = 'Release_180MHz_Detector'
CONFIG = 'Release_180MHz_Pipeline'
SELECTED = ('Release_180MHz_EntM', CONTROL, CONFIG)
LAUNCH = NAME + '_Pipeline_180MHz.launch'
FLAG = 'CWRU_PIPELINE_TEST=1'
TEMPLATE_NAMES = ('cwru_pipeline.h', 'cwru_pipeline.c', 'cwru_pipeline_commands.inc')
MODIFIED = ('.cproject', 'Src/uart_commands.c', 'Src/cmd_cwru_feat.c')
ADDED = (LAUNCH, *('CWRU_Pipeline/' + name for name in TEMPLATE_NAMES))
FILES = (*MODIFIED, *ADDED)
CLIENT_NAMES = ('validar_pipeline.py', 'REFERENCIA_PIPELINE.json',
                'ventanas_desarrollo.npz', 'metadata_ventanas.csv')
SCRIPT_NAMES = ('preparar_pipeline_cwru.py', 'preparar_referencia_pipeline.py',
                'validar_pipeline_cwru.py', 'preparar_cache_potencias.py',
                'preparar_reloj_180mhz.py')


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def save(name, value):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write('\n')


def load(name):
    return json.loads((OUT / name).read_text(encoding='utf-8-sig'))


def relative_path(base, name):
    path = (base / name).resolve()
    require(path.is_relative_to(base.resolve()) and path != base.resolve(),
            'Ruta fuera del directorio previsto: ' + str(name))
    return path


def copy_checked(source, target):
    digest = sha(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    require(sha(target) == digest and sha(source) == digest,
            'Copia diferente/origen cambiante: ' + str(source))
    return digest


def reference_check():
    reference = load('REFERENCIA_PIPELINE.json')
    require(reference['status'] == 'PC_REFERENCE_READY_NOT_ARM_INTEGRATED'
            and len(reference['cases']) == 42
            and [case['index'] for case in reference['cases']] == list(range(42)),
            'Falta referencia PC fija de las 42 ventanas')
    entries = load('ENTRADAS_REFERENCIA_SHA256.json')
    require(len(entries) == 187, 'Inventario de referencia distinto de 187 entradas')
    for name, digest in entries.items():
        require(sha(relative_path(ROOT, name)) == digest, 'Entrada de referencia diferente: ' + name)
    return entries


def client_check():
    manifest = json.loads((CLIENT / 'VALIDACION_MANIFEST.json').read_text(encoding='utf-8'))
    require(set(manifest) == set(CLIENT_NAMES), 'Manifest PC no contiene exactamente los cuatro recursos')
    for name, digest in manifest.items():
        require(sha(CLIENT / name) == digest, 'Recurso PC diferente: ' + name)
    require(sha(CLIENT / 'REFERENCIA_PIPELINE.json') == sha(OUT / 'REFERENCIA_PIPELINE.json'),
            'Referencia del cliente diferente')
    require(sha(CLIENT / 'validar_pipeline.py') == sha(Path(__file__).with_name('validar_pipeline_cwru.py')),
            'Cliente del paquete diferente del fuente')


def origin(preparation):
    for row in preparation['source_inventory_before']:
        require(sha(relative_path(SOURCE, row['path'])) == row['sha256'], 'D cambió: ' + row['path'])
    for config, digest in preparation['existing_elf_sha256'].items():
        require(sha(SOURCE / config / (NAME + '.elf')) == digest, 'ELF anterior cambió: ' + config)
    require(not any((SOURCE / name).exists() for name in ADDED), 'El destino nuevo ya existe en D')


def previous_configurations_equal(path):
    previous = configs(BACKUP / '.cproject')[2]
    current = configs(path)[2]
    return (len(previous) == 11 and len(current) == 12 and CONFIG in current
            and all(name in current and ET.tostring(node) == ET.tostring(current[name])
                    for name, node in previous.items()))


def launch_semantics(path):
    root = ET.parse(path).getroot()
    keys = [node.get('key') for node in root]
    require(None not in keys and len(set(keys)) == len(keys), 'Lanzamiento con claves ambiguas')
    def canonical(node):
        return (node.tag, tuple(sorted(node.attrib.items())), (node.text or '').strip(),
                tuple(canonical(child) for child in node))
    return (root.tag, tuple(sorted(root.attrib.items())),
            tuple(canonical(node) for node in sorted(root, key=lambda node: node.get('key'))))


def prepare():
    require(not any(path.exists() for path in (PROJECT, BACKUP, CLIENT, OUT / 'PREPARACION.json')),
            'No sobrescribir una preparación anterior')
    require(not any((SOURCE / name).exists() for name in ADDED)
            and not (SOURCE / 'CWRU_Pipeline').exists(), 'Destino nuevo ya existente')
    # Preflight completo antes de crear la copia o Validacion_PC. El cliente debe
    # estar terminado; no se generan ni recalculan referencias durante prepare.
    for name in SCRIPT_NAMES:
        require(Path(__file__).with_name(name).is_file(), 'Falta script terminado: ' + name)
    for name in TEMPLATE_NAMES:
        require((TEMPLATES / name).is_file(), 'Falta plantilla: ' + name)
    for name in ('ventanas_desarrollo.npz', 'metadata_ventanas.csv'):
        require((VECTORS / name).is_file(), 'Falta paquete de ventanas: ' + name)
    entries = reference_check()
    subprocess.run([sys.executable, '-B', str(Path(__file__).with_name('preparar_detector_cwru.py')),
                    'verify'], check=True)
    previous = json.loads((PRIOR / 'PREPARACION.json').read_text(encoding='utf-8'))
    installed = json.loads((PRIOR / 'APLICACION_EN_D.json').read_text(encoding='utf-8'))
    require(installed['status'] == 'PASS', 'Detector previo sin instalación verificada')
    paths = {row['path'] for row in previous['source_inventory_before']} | set(installed['installed_sha256'])
    paths.update(path.name for path in SOURCE.glob('*.launch'))
    require(set(MODIFIED).issubset(paths), 'Inventario previo incompleto')
    reserialized = []
    for name, digest in installed['installed_sha256'].items():
        actual = sha(relative_path(SOURCE, name))
        if actual != digest:
            old_launch = PRIOR / 'archivos_para_D' / name
            require(name.endswith('.launch') and old_launch.is_file()
                    and launch_semantics(SOURCE / name) == launch_semantics(old_launch),
                    'Detector activo diferente: ' + name)
            reserialized.append(dict(path=name, previous_sha256=digest, current_sha256=actual,
                                     attributes_and_values_identical=True, restored_or_overwritten=False))
    before_source = configs(SOURCE / '.cproject')[2]
    require(len(before_source) == 11 and CONTROL in before_source and CONFIG not in before_source,
            'Se esperaban once configuraciones y el control Detector')
    for config in before_source:
        require((SOURCE / config / (NAME + '.elf')).is_file(), 'Falta ELF previo: ' + config)
    snapshot = []
    for name in sorted(paths):
        source = relative_path(SOURCE, name)
        digest = copy_checked(source, PROJECT / name)
        require(copy_checked(source, BACKUP / name) == digest, 'Respaldo distinto: ' + name)
        snapshot.append(dict(path=name, sha256=digest))
    elf_hashes = {config: copy_checked(SOURCE / config / (NAME + '.elf'),
                                     BACKUP / config / (NAME + '.elf')) for config in before_source}
    tree, settings, before = configs(PROJECT / '.cproject')
    expanded = copy.deepcopy(before[CONTROL])
    mapping = {node.get('id'): node.get('id') + '.pipeline1'
               for node in expanded.iter() if node.get('id')}
    for node in expanded.iter():
        for key, value in list(node.attrib.items()):
            if value in mapping:
                node.set(key, mapping[value])
            elif key == 'targetTool':
                node.set(key, ';'.join(mapping.get(part, part) for part in value.split(';')))
            elif key == 'name' and value == CONTROL:
                node.set(key, CONFIG)
            elif key == 'buildPath' and value.endswith('/' + CONTROL):
                node.set(key, value[:-len(CONTROL)] + CONFIG)
    compiler_count = 0
    for tool in expanded.findall('.//tool'):
        if tool.get('superClass', '').endswith(('tool.c.compiler', 'tool.cpp.compiler')):
            compiler_count += 1
            defines = next(option for option in tool.findall('option')
                           if option.get('superClass', '').endswith('.definedsymbols'))
            require(not any(value.get('value', '').startswith('CWRU_PIPELINE_TEST')
                            for value in defines), 'El control ya contiene el flag Pipeline')
            ET.SubElement(defines, 'listOptionValue', builtIn='false', value=FLAG)
            includes = next(option for option in tool.findall('option')
                            if option.get('superClass', '').endswith('.includepaths'))
            ET.SubElement(includes, 'listOptionValue', builtIn='false', value='../CWRU_Pipeline')
    require(compiler_count == 2, 'Se esperaban compiladores C y C++')
    ET.SubElement(expanded.find('.//sourceEntries'), 'entry', flags='VALUE_WORKSPACE_PATH|RESOLVED',
                  kind='sourcePath', name='CWRU_Pipeline')
    old_ids = {node.get('id') for config in before.values() for node in config.iter() if node.get('id')}
    require(not old_ids & {node.get('id') for node in expanded.iter() if node.get('id')},
            'IDs CDT duplicados')
    settings.append(expanded)
    (PROJECT / '.cproject').write_bytes(b'<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n'
                                       b'<?fileVersion 4.0.0?>\n' + ET.tostring(tree, encoding='utf-8') + b'\n')
    require(previous_configurations_equal(PROJECT / '.cproject'), 'Configuración anterior modificada')
    for name in TEMPLATE_NAMES:
        copy_checked(TEMPLATES / name, PROJECT / 'CWRU_Pipeline' / name)
    launch = ET.parse(SOURCE / (NAME + '_Detector_180MHz.launch')).getroot()
    attributes = {node.get('key'): node for node in launch}
    elf = CONFIG + '/' + NAME + '.elf'
    attributes['org.eclipse.cdt.launch.PROGRAM_NAME'].set('value', elf)
    attributes['org.eclipse.cdt.launch.PROJECT_BUILD_CONFIG_ID_ATTR'].set('value', expanded.get('id'))
    attributes['org.eclipse.cdt.launch.PROJECT_BUILD_CONFIG_AUTO_ATTR'].set('value', 'false')
    key = 'com.st.stm32cube.ide.mcu.debug.launch.loadList'
    loads = json.loads(attributes[key].get('value'))
    for item in loads['fItems']:
        item['fPath'] = elf
    attributes[key].set('value', json.dumps(loads, separators=(',', ':')))
    for key, node in attributes.items():
        if key.endswith('stlink.log_file'):
            node.set('value', str(SOURCE / CONFIG / 'st-link_gdbserver_log.txt'))
    (PROJECT / LAUNCH).write_bytes(b'<?xml version="1.0" encoding="UTF-8"?>\n'
                                   + ET.tostring(launch, encoding='utf-8') + b'\n')
    CLIENT.mkdir()
    for source, name in ((Path(__file__).with_name('validar_pipeline_cwru.py'), CLIENT_NAMES[0]),
                         (OUT / CLIENT_NAMES[1], CLIENT_NAMES[1]),
                         (VECTORS / CLIENT_NAMES[2], CLIENT_NAMES[2]),
                         (VECTORS / CLIENT_NAMES[3], CLIENT_NAMES[3])):
        copy_checked(source, CLIENT / name)
    with (CLIENT / 'VALIDACION_MANIFEST.json').open('x', encoding='utf-8', newline='\n') as handle:
        json.dump({name: sha(CLIENT / name) for name in CLIENT_NAMES}, handle, indent=2)
        handle.write('\n')
    preparation = dict(status='PREPARED_UART_AND_RECEIVER_PATCH_PENDING', source=str(SOURCE),
        isolated_project=str(PROJECT), configuration=CONFIG, control=CONTROL,
        source_inventory_before=snapshot, existing_elf_sha256=elf_hashes,
        previous_eleven_configurations_preserved=True, previous_eleven_elfs_backed_up=True,
        modified_files=list(MODIFIED), added_files=list(ADDED), flag=FLAG, optimization='-O3',
        hclk_hz=180000000, reference_source_count=len(entries), reference_cases=42,
        source_patch_pending=True, physical_validation_pending=True, hardware_flashed=False,
        serial_port_opened=False, accepted_ide_launch_reserializations=reserialized)
    origin(preparation)
    client_check()
    save('PREPARACION.json', preparation)
    print('PREPARADO: copia aislada y doce configuraciones; faltan guardias UART/receptor.', flush=True)


def compiler_commands(folder):
    return [line for makefile in sorted(folder.rglob('subdir.mk'))
            for line in makefile.read_text(encoding='utf-8').splitlines()
            if re.search(r'\barm-none-eabi-g(?:cc|\+\+)(?=\s)', line) and '-c ' in line
            and re.search(r'-std=gnu(?:\+\+|\d)', line)]


def flags_confirmed(commands, config):
    def macro(line, name, expected):
        values = re.findall(r'(?<!\S)-D' + re.escape(name) + r'(?:=([^\s]+))?(?=\s|$)', line)
        return values == [expected] if expected is not None else values in ([], ['0'])
    expected = {'CWRU_TARGET_HCLK_HZ': '180000000U', 'CWRU_CACHE_SPECTRAL_POWER': '1',
                'CWRU_ENTROPY_MODE': '2', 'CWRU_INPUT_TRANSFORM': '1' if config != SELECTED[0] else None,
                'CWRU_DNN_TEST': '1' if config != SELECTED[0] else None,
                'CWRU_NORMALITY_TEST': '1' if config != SELECTED[0] else None,
                'CWRU_PIPELINE_TEST': '1' if config == CONFIG else None,
                'CWRU_PROFILE_EXTRACTOR': None}
    return bool(commands) and all(
        re.findall(r'(?<!\S)-O\S+', line) == ['-O3']
        and all(macro(line, name, value) for name, value in expected.items())
        and not any(flag in line for flag in ('-ffast-math', '-Ofast', '-funsafe-math-optimizations'))
        and ('-I../CWRU_Pipeline' in line) == (config == CONFIG) for line in commands)


def buffer_audit(tool, elf, map_path, attempt):
    nm_text = subprocess.check_output([str(tool / 'arm-none-eabi-nm.exe'), '-S', '--size-sort',
                                       '-C', '--defined-only', str(elf)], text=True)
    (OUT / f'SIMBOLOS_PIPELINE_INTENTO_{attempt}.txt').write_text(nm_text, encoding='utf-8')
    symbols = []
    for line in nm_text.splitlines():
        match = re.match(r'^([0-9a-fA-F]+)\s+([0-9a-fA-F]+)\s+([A-Za-z])\s+(.+)$', line.strip())
        if match:
            address, size, kind, name = match.groups()
            symbols.append(dict(address=int(address, 16), size_bytes=int(size, 16), kind=kind, name=name))
    expected = {'g_window': (32768, '.bss.g_window'), 'g_fft_input': (16384, '.bss.g_fft_input'),
                'g_fft_output': (16384, '.bss.g_fft_output'),
                '(anonymous namespace)::arena': (16384, '.bss._ZN12_GLOBAL__N_15arenaE')}
    map_text = map_path.read_text(encoding='utf-8')
    found = {}
    for name, (size, section) in expected.items():
        rows = [row for row in symbols if row['name'] == name and row['kind'] in 'bBdD']
        mapped = [(int(address, 16), int(length, 16)) for address, length in
                  re.findall(r'(?m)^\s*' + re.escape(section) + r'\s+(0x[0-9a-fA-F]+)\s+(0x[0-9a-fA-F]+)', map_text)]
        found[name] = dict(expected_size_bytes=size, nm=rows, map_allocations=mapped,
            confirmed=len(rows) == 1 and rows[0]['size_bytes'] == size
            and mapped == [(rows[0]['address'], size)])
    large = [row for row in symbols if row['kind'] in 'bBdD' and row['size_bytes'] >= 16384
             and 0x20000000 <= row['address'] < 0x20020000]
    ranges = sorted((row['address'], row['address'] + row['size_bytes']) for row in large)
    nonoverlap = all(left[1] <= right[0] for left, right in zip(ranges, ranges[1:]))
    passed = (all(row['confirmed'] for row in found.values()) and len(large) == 4
              and {row['name'] for row in large} == set(expected) and nonoverlap)
    return dict(status='PASS' if passed else 'FAIL', symbols=found, large_static_ram_objects=large,
                buffers_and_arena_nonoverlapping=nonoverlap,
                shared_window_fft_bytes=65536, separately_reserved_tflm_arena_bytes=16384,
                scope='Static symbols/map; no physical stack peak or arena-used measurement')


def inspect_configuration(tool, preparation, config, attempt):
    folder = PROJECT / config
    elf = folder / (NAME + '.elf')
    size_text = subprocess.check_output([str(tool / 'arm-none-eabi-size.exe'), str(elf)], text=True)
    text_size, data_size, bss_size = map(int, size_text.strip().splitlines()[-1].split()[:3])
    image = flash_image(elf)
    require(bool(image), 'ELF sin imagen Flash: ' + config)
    span = max(address + len(data) for address, data in image) - min(address for address, _ in image)
    commands = compiler_commands(folder)
    frames = {path.relative_to(folder).as_posix(): path.read_text(encoding='utf-8').splitlines()
              for path in sorted(folder.rglob('*.su'))}
    row = dict(elf_sha256=sha(elf), text=text_size, data=data_size, bss=bss_size,
        flash_load_bytes=text_size + data_size, ram_static_with_linker_reservations_bytes=data_size + bss_size,
        flash_span_bytes=span, expected_download_kib=span / 1024, flags_confirmed=flags_confirmed(commands, config),
        compiler_command_count=len(commands), compiler_stack_frames=frames,
        stack_interpretation='Individual compiler frames only; not physical peak of the complete call chain')
    if config != CONFIG:
        row.update(control_elf_identical=sha(elf) == preparation['existing_elf_sha256'][config],
                   control_flash_identical=image == flash_image(BACKUP / config / (NAME + '.elf')))
    else:
        row['buffer_audit'] = buffer_audit(tool, elf, folder / (NAME + '.map'), attempt)
        disassembly = subprocess.check_output([str(tool / 'arm-none-eabi-objdump.exe'), '-d',
                                               '--disassemble=CWRUPIPE_Execute', str(elf)], text=True)
        (OUT / f'DESENSAMBLADO_PIPELINE_INTENTO_{attempt}.txt').write_text(disassembly, encoding='utf-8')
        names = ('CWRU_FeatureExtraction_Extract', 'CWRU_InputTransform', 'CWRUNN_PredictQ', 'CWRUND_Compute')
        calls = {name: len(re.findall(r'\bblx?(?:\.\w+)?\s+[0-9a-fA-F]+\s+<' + re.escape(name)
                                     + r'>', disassembly)) for name in names}
        row['observed_execute_direct_callsites'] = calls
        row['one_direct_callsite_each_observed'] = all(count == 1 for count in calls.values())
        row['callsite_interpretation'] = 'Static disassembly evidence only; runtime extraction count is validated separately'
        row['required_stack_frames_present'] = all(name in frames for name in
            ('CWRU_Pipeline/cwru_pipeline.su', 'Src/cmd_cwru_feat.su', 'Src/uart_commands.su',
             'CWRU_DNN/cwru_nn.su', 'CWRU_Detector/cwru_normality.su'))
    return row


def old_sources_check(tool):
    include_names = ('Inc', 'CWRU_Input', 'CWRU_DNN', 'CWRU_Detector', 'Drivers/CMSIS/Include',
                     'Drivers/CMSIS/Device/ST/STM32F4xx/Include', 'Drivers/STM32F4xx_HAL_Driver/Inc',
                     'Middlewares/CMSIS-DSP/Include', 'Middlewares/CMSIS-DSP/PrivateInclude')
    command = [str(tool / 'arm-none-eabi-gcc.exe'), '-E', '-P', '-std=gnu11', '-mcpu=cortex-m4',
               '-mfpu=fpv4-sp-d16', '-mfloat-abi=hard', '-mthumb', '-DSTM32F446xx',
               '-DCWRU_PIPELINE_TEST=0', '-DCWRU_PROFILE_EXTRACTOR=0',
               '-DCWRU_TARGET_HCLK_HZ=180000000U', '-DCWRU_CACHE_SPECTRAL_POWER=1', '-DCWRU_ENTROPY_MODE=2',
               *('-I' + str(PROJECT / name) for name in include_names)]
    comparisons, digests = {}, {}
    for source in MODIFIED[1:]:
        for input_flag, dnn_flag, detector_flag in ((0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 1)):
            args = [*command, f'-DCWRU_INPUT_TRANSFORM={input_flag}', f'-DCWRU_DNN_TEST={dnn_flag}',
                    f'-DCWRU_NORMALITY_TEST={detector_flag}']
            before = subprocess.check_output([*args, str(BACKUP / source)])
            after = subprocess.check_output([*args, str(PROJECT / source)])
            key = f'{source}:{input_flag}_{dnn_flag}_{detector_flag}'
            comparisons[key] = before == after
            digests[key] = dict(before=hashlib.sha256(before).hexdigest(), after=hashlib.sha256(after).hexdigest())
    return comparisons, digests


def snapshot_build(attempt):
    destination = OUT / 'intentos' / str(attempt) / 'compilacion'
    for config in SELECTED:
        folder = PROJECT / config
        for path in sorted(folder.rglob('*')):
            if path.is_file() and (path.suffix in ('.elf', '.map', '.su', '.mk')
                                   or path.name in ('objects.list', 'makefile')):
                copy_checked(path, destination / config / path.relative_to(folder))


def build():
    preparation = load('PREPARACION.json')
    origin(preparation)
    reference_check()
    client_check()
    require(not (OUT / 'COMPILACION_PIPELINE.json').exists(), 'Build ya aprobado')
    uart = (PROJECT / MODIFIED[1]).read_text(encoding='utf-8')
    receiver = (PROJECT / MODIFIED[2]).read_text(encoding='utf-8')
    require('CWRUPIPE_Command' in uart and 'CWRU_PIPELINE_TEST' in uart,
            'Falta guardia UART del pipeline')
    require('cwru_pipeline_commands.inc' in receiver and receiver.count('CWRUPIPE_Invalidate();') >= 3,
            'Falta puente/invalidation del receptor')
    for name in TEMPLATE_NAMES:
        require(sha(PROJECT / 'CWRU_Pipeline' / name) == sha(TEMPLATES / name), 'Plantilla copiada diferente: ' + name)
    existing = [int(path.stem.rsplit('_', 1)[-1]) for path in OUT.glob('CUBEIDE_PIPELINE_INTENTO_*.log')]
    attempt = max(existing, default=0) + 1
    require(attempt <= 3, 'Se agotaron tres intentos; revisar el diagnóstico antes de ampliar el ensayo')
    attempt_folder = OUT / 'intentos' / str(attempt)
    require(not attempt_folder.exists(), 'No sobrescribir evidencias de un intento')
    attempt_folder.mkdir(parents=True)
    for name in FILES:
        copy_checked(PROJECT / name, OVERLAY / name)
        copy_checked(PROJECT / name, attempt_folder / 'archivos' / name)
    tools = list((IDE / 'plugins').glob('com.st.stm32cube.ide.mcu.externaltools.gnu-tools-for-stm32.*/tools/bin'))
    makes = list((IDE / 'plugins').glob('com.st.stm32cube.ide.mcu.externaltools.make.*/tools/bin'))
    require(len(tools) == len(makes) == 1, 'Toolchain/make ausente o ambiguo')
    tool, make = tools[0], makes[0]
    env = dict(os.environ)
    env['PATH'] = str(tool) + os.pathsep + str(make) + os.pathsep + env.get('PATH', '')
    workspace = OUT / 'workspace_aislado' / f'intento_{attempt}'
    require(not workspace.exists(), 'Workspace del intento ya existente')
    args = [str(IDE / 'stm32cubeidec.exe'), '--launcher.suppressErrors', '-nosplash', '-application',
            'org.eclipse.cdt.managedbuilder.core.headlessbuild', '-data', str(workspace),
            '-import', str(PROJECT), '-no-indexer']
    for config in SELECTED:
        args.extend(('-cleanBuild', NAME + '/' + config))
    report = dict(attempt=attempt, command=args, workspace=str(workspace), configurations={},
                  hardware_flashed=False, serial_port_opened=False, physical_validation_pending=True)
    log_path = OUT / f'CUBEIDE_PIPELINE_INTENTO_{attempt}.log'
    start = time.monotonic()
    # Crear el log antes del autotest hace que incluso un fallo PC consuma un
    # intento explícito y deje disponible el diagnóstico sin sobrescritura.
    with log_path.open('x', encoding='utf-8') as log:
        try:
            tests = subprocess.run([sys.executable, '-B', str(CLIENT / 'validar_pipeline.py'), '--self-test',
                                    '--output-root', str(OUT / '_self_test_compilacion' / f'intento_{attempt}')],
                                   capture_output=True, text=True, encoding='utf-8', errors='replace')
            report['client_self_tests_passed'] = tests.returncode == 0
            save(f'PRUEBAS_CLIENTE_INTENTO_{attempt}.json', dict(status='PASS' if tests.returncode == 0 else 'FAIL',
                 returncode=tests.returncode, output=tests.stdout + tests.stderr, executes_C_ARM=False,
                 serial_port_opened=False))
            require(tests.returncode == 0, 'Autotests del cliente fallidos')
            process = subprocess.Popen(args, cwd=OUT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       encoding='utf-8', errors='replace')
            for line in process.stdout:
                log.write(line)
                log.flush()
                if any(token in line.lower() for token in ('build of', 'error:', 'warning:',
                                                           'undefined reference', 'build finished')):
                    print(line.strip()[:300], flush=True)
            report['returncode'] = process.wait(timeout=30)
        except Exception as error:
            report['execution_error'] = f'{type(error).__name__}: {error}'
            report.setdefault('returncode', -1)
    report['elapsed_seconds'] = time.monotonic() - start
    log_text = log_path.read_text(encoding='utf-8')
    summaries = re.findall(r'(\d+) errors, (\d+) warnings', log_text)
    report['build_summaries'] = summaries
    build_headers = re.findall(r'\*{4} Build of configuration (\S+) for project', log_text)
    report['build_configuration_headers'] = build_headers
    report['zero_errors_warnings'] = (len(summaries) >= 3 and set(build_headers) == set(SELECTED)
        and all(int(a) == int(b) == 0 for a, b in summaries)
        and not re.search(r'\b(?:error|warning):|undefined reference', log_text, re.IGNORECASE))
    for config in SELECTED:
        try:
            report['configurations'][config] = inspect_configuration(tool, preparation, config, attempt)
        except Exception as error:
            report['configurations'][config] = dict(inspection_error=f'{type(error).__name__}: {error}')
    report['previous_eleven_configurations_preserved'] = previous_configurations_equal(PROJECT / '.cproject')
    allowed = set(MODIFIED) | {'.settings/language.settings.xml'}
    report['other_resources_unchanged'] = all(sha(PROJECT / row['path']) == row['sha256']
        for row in preparation['source_inventory_before'] if row['path'] not in allowed)
    try:
        identical, digests = old_sources_check(tool)
        report['old_dispatch_preprocessed_identical'] = identical
        report['old_sources_preprocessed_sha256'] = digests
    except Exception as error:
        report['old_dispatch_preprocessed_identical'] = {}
        report['preprocessing_error'] = f'{type(error).__name__}: {error}'
    report['overlay_sha256'] = {name: sha(OVERLAY / name) for name in FILES}
    snapshot_build(attempt)
    origin(preparation)
    controls = [report['configurations'][config] for config in SELECTED[:2]]
    pipeline = report['configurations'][CONFIG]
    passed = (report['returncode'] == 0 and report.get('client_self_tests_passed', False)
        and report['zero_errors_warnings'] and all(row.get('flags_confirmed', False) for row in report['configurations'].values())
        and all(row.get('control_elf_identical', False) and row.get('control_flash_identical', False) for row in controls)
        and report['previous_eleven_configurations_preserved'] and report['other_resources_unchanged']
        and len(report['old_dispatch_preprocessed_identical']) == 8
        and all(report['old_dispatch_preprocessed_identical'].values())
        and pipeline.get('buffer_audit', {}).get('status') == 'PASS'
        and pipeline.get('required_stack_frames_present', False))
    report['status'] = 'PASS' if passed else 'FAIL'
    save(f'COMPILACION_PIPELINE_INTENTO_{attempt}.json', report)
    if passed:
        save('COMPILACION_PIPELINE.json', report)
    print(json.dumps(report, ensure_ascii=True, indent=2), flush=True)
    if not passed:
        raise SystemExit(2)


def cdt_language_settings_audit():
    """Aceptar solo el alta exacta de la nueva configuración en metadatos CDT.

    No modifica D ni ignora cambios de proveedores, macros o valores previos.
    La atribución al IDE es una inferencia; se documenta el cambio observado.
    """
    name = '.settings/language.settings.xml'
    baseline, current = BACKUP / name, SOURCE / name
    old_digest, new_digest = sha(baseline), sha(current)
    if old_digest == new_digest:
        return dict(status='PASS', changed=False, path=name,
                    before_sha256=old_digest, after_sha256=new_digest)
    before, after = ET.parse(baseline).getroot(), ET.parse(current).getroot()
    def canonical(node):
        return (node.tag, tuple(sorted(node.attrib.items())), (node.text or '').strip(),
                (node.tail or '').strip(), tuple(canonical(child) for child in node))
    previous = {node.get('name'): node for node in before}
    require(before.tag == after.tag == 'project' and len(before) == len(previous) == 11,
            'Metadatos CDT anteriores ambiguos')
    require(all(node.tag == 'configuration' for node in before), 'Nodos CDT inesperados')
    require(CONTROL in previous and CONFIG not in previous, 'Metadatos CDT de control inválidos')
    expected = copy.deepcopy(before)
    added = copy.deepcopy(previous[CONTROL])
    added.set('name', CONFIG)
    added.set('id', configs(SOURCE / '.cproject')[2][CONFIG].get('id'))
    expected.append(added)
    accepted = canonical(expected) == canonical(after)
    return dict(status='PASS' if accepted else 'FAIL', changed=True, path=name,
        before_sha256=old_digest, after_sha256=new_digest,
        previous_eleven_entries_semantically_identical=accepted,
        only_expected_pipeline_entry_appended=accepted, added_entry_clones=CONTROL,
        observed_after_installation=True, overwritten_by_preparer=False,
        attribution='Compatible with automatic CDT metadata refresh; actor not independently traced')


def installed_checks(preparation, installed_hashes, allow_cdt_append=False):
    configs_same = previous_configurations_equal(SOURCE / '.cproject')
    accepted_cdt = (allow_cdt_append and cdt_language_settings_audit()['status'] == 'PASS')
    resources_same = all(sha(SOURCE / row['path']) == row['sha256']
        or (accepted_cdt and row['path'] == '.settings/language.settings.xml')
        for row in preparation['source_inventory_before'] if row['path'] not in MODIFIED)
    elfs_same = all(sha(SOURCE / config / (NAME + '.elf')) == digest
                   for config, digest in preparation['existing_elf_sha256'].items())
    hashes_same = all(sha(SOURCE / name) == digest for name, digest in installed_hashes.items())
    return configs_same, resources_same, elfs_same, hashes_same


def install():
    preparation, built = load('PREPARACION.json'), load('COMPILACION_PIPELINE.json')
    require(built['status'] == 'PASS' and not (OUT / 'APLICACION_EN_D.json').exists(),
            'Build/destino no válido o instalación ya registrada')
    origin(preparation)
    require(not (SOURCE / 'CWRU_Pipeline').exists(), 'Directorio Pipeline ya existente')
    require(set(built['overlay_sha256']) == set(FILES), 'Overlay no contiene exactamente siete archivos')
    for name, digest in built['overlay_sha256'].items():
        require(sha(OVERLAY / name) == digest, 'Overlay modificado: ' + name)
        require(sha(PROJECT / name) == digest, 'Proyecto posterior al build modificado: ' + name)
    for row in preparation['source_inventory_before']:
        require(sha(BACKUP / row['path']) == row['sha256'], 'Respaldo original diferente: ' + row['path'])
    for config, digest in preparation['existing_elf_sha256'].items():
        require(sha(BACKUP / config / (NAME + '.elf')) == digest, 'Respaldo ELF diferente: ' + config)
    # Única sección del script que escribe en D:, siete archivos y nada más.
    for name in FILES:
        copy_checked(OVERLAY / name, SOURCE / name)
    same, resources, elfs, hashes = installed_checks(preparation, built['overlay_sha256'])
    report = dict(status='PASS' if same and resources and elfs and hashes else 'FAIL',
        modified_files=list(MODIFIED), added_files=list(ADDED),
        previous_eleven_configurations_preserved=same, previous_eleven_elfs_unchanged=elfs,
        other_snapshot_resources_unchanged=resources, installed_files_match_overlay=hashes,
        installed_sha256={name: sha(SOURCE / name) for name in FILES},
        user_workspace_modified=False, hardware_flashed=False, serial_port_opened=False,
        physical_validation_pending=True)
    save('APLICACION_EN_D.json', report)
    print(json.dumps(report, ensure_ascii=True, indent=2), flush=True)
    require(report['status'] == 'PASS', 'Instalación no válida')


def seal():
    require(not (OUT / 'MANIFEST_SHA256.json').exists() and not (OUT / 'MANIFEST_SHA256.sha256').exists(),
            'Sello ya existente')
    preparation, built, installed = load('PREPARACION.json'), load('COMPILACION_PIPELINE.json'), load('APLICACION_EN_D.json')
    require(built['status'] == installed['status'] == 'PASS', 'Faltan comprobaciones')
    require(all(installed_checks(preparation, installed['installed_sha256'], allow_cdt_append=True)),
            'D diferente después de instalar, fuera del alta exacta de metadatos CDT')
    cdt = cdt_language_settings_audit()
    require(cdt['status'] == 'PASS', 'Metadatos CDT no válidos')
    metadata_snapshot = OUT / 'language.settings.postinstalacion.xml'
    metadata_report = OUT / 'METADATOS_CDT_POSTINSTALACION.json'
    if not metadata_report.exists():
        require(not metadata_snapshot.exists(), 'Snapshot de metadatos sin informe')
        copy_checked(SOURCE / cdt['path'], metadata_snapshot)
        require(sha(metadata_snapshot) == cdt['after_sha256'], 'Metadatos CDT cambiaron durante el cierre')
        save(metadata_report.name, dict(**cdt, snapshot=metadata_snapshot.name,
             seven_installed_files_unchanged=True, previous_eleven_elfs_unchanged=True,
             other_resources_except_this_metadata_unchanged=True))
    else:
        require(load(metadata_report.name)['after_sha256'] == cdt['after_sha256']
                and sha(metadata_snapshot) == cdt['after_sha256'], 'Metadatos cambiaron tras su snapshot')
    entries = reference_check()
    client_check()
    names = ('PREPARACION.json', 'COMPILACION_PIPELINE.json', 'APLICACION_EN_D.json',
             'REFERENCIA_PIPELINE.json', 'ENTRADAS_REFERENCIA_SHA256.json',
             'CONTRATO_PREVIO_PIPELINE.md', 'ESPECIFICACION_UART_PREVIA.md')
    paths = {OUT / name for name in names}
    paths.update(path for path in OUT.iterdir() if path.is_file())
    paths.update(path for path in (OUT / '_self_test_cliente_pipeline').rglob('*') if path.is_file())
    paths.update(path for path in (OUT / '_self_test_compilacion').rglob('*') if path.is_file())
    for pattern in ('CUBEIDE_PIPELINE_INTENTO_*.log', 'COMPILACION_PIPELINE_INTENTO_*.json',
                    'PRUEBAS_CLIENTE_INTENTO_*.json', 'SIMBOLOS_PIPELINE_INTENTO_*.txt',
                    'DESENSAMBLADO_PIPELINE_INTENTO_*.txt'):
        paths.update(OUT.glob(pattern))
    for row in preparation['source_inventory_before']:
        paths.update((PROJECT / row['path'], BACKUP / row['path']))
    paths.update(PROJECT / name for name in FILES)
    paths.update(OVERLAY / name for name in FILES)
    paths.update(BACKUP / config / (NAME + '.elf') for config in preparation['existing_elf_sha256'])
    for config in SELECTED:
        folder = PROJECT / config
        paths.update(folder / name for name in (NAME + '.elf', NAME + '.map', 'objects.list'))
        paths.update(path for path in folder.rglob('*') if path.is_file()
                     and (path.suffix in ('.su', '.mk') or path.name == 'makefile'))
    paths.update(path for path in (OUT / 'intentos').rglob('*') if path.is_file())
    # Sólo los cuatro insumos y su manifest; resultados UART futuros quedan fuera.
    paths.update(CLIENT / name for name in (*CLIENT_NAMES, 'VALIDACION_MANIFEST.json'))
    paths.update(TEMPLATES / name for name in TEMPLATE_NAMES)
    paths.update(Path(__file__).with_name(name) for name in SCRIPT_NAMES)
    paths.update(relative_path(ROOT, name) for name in entries)
    require(all(path.is_file() and path.resolve().is_relative_to(ROOT) for path in paths),
            'Recurso requerido ausente o fuera del proyecto')
    save('MANIFEST_SHA256.json', {path.relative_to(ROOT).as_posix(): sha(path) for path in sorted(paths)})
    with (OUT / 'MANIFEST_SHA256.sha256').open('x', encoding='ascii') as handle:
        handle.write(sha(OUT / 'MANIFEST_SHA256.json') + '  MANIFEST_SHA256.json\n')
    verify()


def verify():
    digest, name = (OUT / 'MANIFEST_SHA256.sha256').read_text(encoding='ascii').strip().split()
    require(name == 'MANIFEST_SHA256.json' and sha(OUT / name) == digest, 'Sello diferente')
    manifest = load(name)
    for relative, expected in manifest.items():
        require(sha(relative_path(ROOT, relative)) == expected, 'Recurso diferente: ' + relative)
    reference_check()
    client_check()
    print(json.dumps(dict(status='PASS', files_verified=len(manifest), reference_sources_verified=187,
        physical_validation_pending=True, serial_port_opened=False, active_D_rechecked=False)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'build', 'install', 'seal', 'verify'))
    modes = dict(prepare=prepare, build=build, install=install, seal=seal, verify=verify)
    modes[parser.parse_args().mode]()
