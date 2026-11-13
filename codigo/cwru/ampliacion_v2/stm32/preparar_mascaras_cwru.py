"""Prepara medicion de kernels por mascara; solo install escribe siete archivos en D.

No programa la placa, abre COM o entrena. Conserva catorce configuraciones/ELF.
prepare crea una copia aislada; root agrega las guardias e invalidaciones Src.
build compara EntM y Cost y compila el harness Masks sin DNN ni detector.
La Flash de diez kernels pertenece al harness, no a un despliegue final.
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
from preparar_cache_potencias import configs, sha
from preparar_reloj_180mhz import flash_image

ROOT = Path(__file__).resolve().parents[4]
NAME = 'Prueba_2_CWRU_Extraccion'
SOURCE = Path('D:/STM32IDE') / NAME
OUT = ROOT / 'despliegue_stm32/costos_mascaras_180mhz_2026-10-24'
PRIOR = ROOT / 'despliegue_stm32/perfil_costos_compartidos_180mhz_2026-10-24'
PROJECT = OUT / 'copia_aislada' / NAME
BACKUP = OUT / 'respaldo_antes_mascaras'
OVERLAY = OUT / 'archivos_para_D'
CLIENT = OUT / 'Validacion_PC'
TEMPLATES = Path(__file__).with_name('mascaras_firmware')
IDE = Path('D:/STM32CubeIDE_2.2.0/STM32CubeIDE')
CONFIG = 'Release_180MHz_Masks'
CONTROL = 'Release_180MHz_EntM'
PROFILE_CONTROL = 'Release_180MHz_Cost'
SELECTED = (CONTROL, PROFILE_CONTROL, CONFIG)
LAUNCH = NAME + '_Masks_180MHz.launch'
TEMPLATE_NAMES = ('cwru_masks.h', 'cwru_masks.c', 'cwru_masks_commands.inc')
MODIFIED = ('.cproject', 'Src/uart_commands.c', 'Src/cmd_cwru_feat.c')
ADDED = (LAUNCH, *('CWRU_Masks/' + n for n in TEMPLATE_NAMES))
FILES = (*MODIFIED, *ADDED)
CLIENT_NAMES = ('validar_mascaras.py', 'validar_pipeline.py', 'REFERENCIA_PIPELINE.json',
                'ventanas_desarrollo.npz', 'metadata_ventanas.csv', 'CATALOGO_MASCARAS.json')
KERNELS = (('F14', 0x3fff), ('T6', 0x0387), ('S8', 0x3c78), ('NF12', 0x37ef),
           ('NM10', 0x3cf9), ('NE12', 0x1fbf), ('DE7', 0x007f), ('FE7', 0x3f80),
           ('FIX_DE13', 0x37ff), ('FIX_FE13', 0x3fef))


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def load(name):
    return json.loads((OUT / name).read_text(encoding='utf-8'))


def save(name, value):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def cp(source, target):
    digest = sha(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    require(sha(target) == digest == sha(source), 'Copia diferente: ' + str(source))
    return digest


def canonical(node):
    return (node.tag, tuple(sorted(node.attrib.items())), (node.text or '').strip(),
            tuple(canonical(child) for child in node))


def same_configs(path):
    old, new = configs(BACKUP / '.cproject')[2], configs(path)[2]
    return (len(old) == 14 and len(new) == 15 and CONFIG in new and
            all(n in new and ET.tostring(c) == ET.tostring(new[n]) for n, c in old.items()))


def prior_manifest(name):
    path = PRIOR / name
    digest, declared = path.with_suffix('.sha256').read_text(encoding='ascii').split()
    require(declared == name and sha(path) == digest, 'Sello anterior diferente: ' + name)
    entries = json.loads(path.read_text(encoding='utf-8'))
    for relative, expected in entries.items():
        p = (ROOT / relative).resolve()
        require(p.is_relative_to(ROOT) and sha(p) == expected, 'Entrada anterior diferente: ' + relative)
    return entries


def prior_checks():
    entries = prior_manifest('MANIFEST_SHA256.json')
    require(len(entries) == 7362, 'El sello de preparacion Cost no tiene 7362 recursos')
    for label in ('INDICE_0', 'COMPLETA'):
        entries.update(prior_manifest('MANIFEST_RESULTADOS_FISICOS_COSTE_' + label + '_SHA256.json'))
    report = json.loads((PRIOR / 'RESULTADOS_FISICOS_COSTE_COMPLETA.json').read_text(encoding='utf-8'))
    require(report['status'] == 'PHYSICAL_COST_SUITE_AUDITED' and report['passed_main_runs'] == 126 and
            report['negative_tests_passed'] == 8, 'Falta perfil compartido completo auditado')
    return entries


def catalog_check(path):
    catalog = json.loads(path.read_text(encoding='utf-8'))
    require(catalog.get('selector_costs_enabled') is False, 'Seleccion por costo no habilitada en esta etapa')
    rows = catalog.get('kernels')
    require(isinstance(rows, list) and len(rows) == len(KERNELS), 'Catalogo de diez kernels requerido')
    for index, ((name, mask), row) in enumerate(zip(KERNELS, rows)):
        require(row['id'] == index and row['name'] == name and row['feature_mask'] == f'{mask:04x}' and
                row['indices'] == [i for i in range(14) if mask & (1 << i)], 'Mascara diferente: ' + name)
    value = 2166136261
    for _, mask in KERNELS:
        for byte in mask.to_bytes(2, 'little'):
            value = ((value ^ byte) * 16777619) & 0xffffffff
    require(catalog.get('catalog_fnv1a') == f'{value:08x}', 'FNV del catalogo diferente')


def client_check():
    manifest = json.loads((CLIENT / 'VALIDACION_MANIFEST.json').read_text(encoding='utf-8'))
    require(set(manifest) == set(CLIENT_NAMES), 'Manifiesto cliente diferente')
    for name, digest in manifest.items():
        require(sha(CLIENT / name) == digest, 'Cliente modificado: ' + name)
    require(sha(CLIENT / 'validar_mascaras.py') == sha(Path(__file__).with_name('validar_mascaras_cwru.py')),
            'Cliente nuevo diferente del fuente')
    for name in CLIENT_NAMES[1:-1]:
        require(sha(CLIENT / name) == sha(PRIOR / 'Validacion_PC' / name), 'Referencia anterior diferente: ' + name)
    require(sha(CLIENT / 'CATALOGO_MASCARAS.json') == sha(OUT / 'CATALOGO_MASCARAS.json'), 'Catalogo diferente')
    catalog_check(CLIENT / 'CATALOGO_MASCARAS.json')


def origin(prep):
    for row in prep['source_inventory_before']:
        require(sha(SOURCE / row['path']) == row['sha256'], 'D cambio: ' + row['path'])
    for name, digest in prep['existing_elf_sha256'].items():
        require(sha(SOURCE / name / (NAME + '.elf')) == digest, 'ELF anterior cambio: ' + name)
    require(not any((SOURCE / name).exists() for name in ADDED), 'Destino nuevo ya existe')


def prepare():
    require(not any(p.exists() for p in (PROJECT, BACKUP, CLIENT, OUT / 'PREPARACION.json')), 'No sobrescribir preparacion')
    require(not (SOURCE / 'CWRU_Masks').exists(), 'Destino Masks ya existe')
    prior_checks()
    installed = json.loads((PRIOR / 'APLICACION_EN_D.json').read_text(encoding='utf-8'))
    previous = json.loads((PRIOR / 'PREPARACION.json').read_text(encoding='utf-8'))
    reserialized = []
    for name, digest in installed['installed_sha256'].items():
        actual = sha(SOURCE / name)
        if actual != digest:
            require(name.endswith('.launch') and prior.launch_semantics(SOURCE / name) ==
                    prior.launch_semantics(PRIOR / 'archivos_para_D' / name), 'Fuente activa diferente: ' + name)
            reserialized.append(dict(path=name, old_sha256=digest, current_sha256=actual, semantically_equal=True))
    old = configs(SOURCE / '.cproject')[2]
    require(len(old) == 14 and CONTROL in old and PROFILE_CONTROL in old and CONFIG not in old,
            'Se esperaban catorce configuraciones, EntM y Cost')
    for name in TEMPLATE_NAMES:
        require((TEMPLATES / name).is_file(), 'Plantilla ausente: ' + name)
    require(Path(__file__).with_name('validar_mascaras_cwru.py').is_file(), 'Cliente ausente')
    require((OUT / 'CATALOGO_MASCARAS.json').is_file(), 'Falta catalogo terminado antes de prepare')
    catalog_check(OUT / 'CATALOGO_MASCARAS.json')
    paths = {row['path'] for row in previous['source_inventory_before']} | set(installed['installed_sha256'])
    paths.update(p.name for p in SOURCE.glob('*.launch'))
    snapshot = []
    for name in sorted(paths):
        digest = cp(SOURCE / name, PROJECT / name)
        require(cp(SOURCE / name, BACKUP / name) == digest, 'Respaldo diferente')
        snapshot.append(dict(path=name, sha256=digest))
    elfs = {name: cp(SOURCE / name / (NAME + '.elf'), BACKUP / name / (NAME + '.elf')) for name in old}
    root, settings, old = configs(PROJECT / '.cproject')
    node = copy.deepcopy(old[CONTROL])
    mapping = {n.get('id'): n.get('id') + '.masks1' for n in node.iter() if n.get('id')}
    for element in node.iter():
        for key, value in list(element.attrib.items()):
            if value in mapping:
                element.set(key, mapping[value])
            elif key == 'targetTool':
                element.set(key, ';'.join(mapping.get(p, p) for p in value.split(';')))
            elif key == 'name' and value == CONTROL:
                element.set(key, CONFIG)
            elif key == 'buildPath' and value.endswith('/' + CONTROL):
                element.set(key, value[:-len(CONTROL)] + CONFIG)
    for tool in node.findall('.//tool'):
        if tool.get('superClass', '').endswith(('tool.c.compiler', 'tool.cpp.compiler')):
            defines = next(o for o in tool.findall('option') if o.get('superClass', '').endswith('.definedsymbols'))
            forbidden = ('CWRU_MASK_BENCH', 'CWRU_PROFILE_EXTRACTOR', 'CWRU_COST_PROFILE', 'CWRU_STACK_TEST',
                         'CWRU_PIPELINE_TEST', 'CWRU_INPUT_TRANSFORM', 'CWRU_DNN_TEST', 'CWRU_NORMALITY_TEST')
            require(not any(v.get('value', '').split('=')[0] in forbidden for v in defines), 'Macro previa ambigua')
            ET.SubElement(defines, 'listOptionValue', builtIn='false', value='CWRU_MASK_BENCH=1')
            includes = next(o for o in tool.findall('option') if o.get('superClass', '').endswith('.includepaths'))
            ET.SubElement(includes, 'listOptionValue', builtIn='false', value='../CWRU_Masks')
    ET.SubElement(node.find('.//sourceEntries'), 'entry', flags='VALUE_WORKSPACE_PATH|RESOLVED',
                  kind='sourcePath', name='CWRU_Masks')
    oldids = {e.get('id') for c in old.values() for e in c.iter() if e.get('id')}
    require(not oldids & {e.get('id') for e in node.iter() if e.get('id')}, 'IDs duplicados')
    settings.append(node)
    (PROJECT / '.cproject').write_bytes(b'<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n'
        b'<?fileVersion 4.0.0?>\n' + ET.tostring(root, encoding='utf-8') + b'\n')
    require(same_configs(PROJECT / '.cproject'), 'Se altero configuracion anterior')
    for name in TEMPLATE_NAMES:
        cp(TEMPLATES / name, PROJECT / 'CWRU_Masks' / name)
    launch = ET.parse(SOURCE / (NAME + '_EntM_180MHz.launch')).getroot()
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
    for key, element in attrs.items():
        if key.endswith('stlink.log_file'):
            element.set('value', str(SOURCE / CONFIG / 'st-link_gdbserver_log.txt'))
    (PROJECT / LAUNCH).write_bytes(b'<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(launch, encoding='utf-8') + b'\n')
    cp(Path(__file__).with_name('validar_mascaras_cwru.py'), CLIENT / 'validar_mascaras.py')
    for name in CLIENT_NAMES[1:-1]:
        cp(PRIOR / 'Validacion_PC' / name, CLIENT / name)
    cp(OUT / 'CATALOGO_MASCARAS.json', CLIENT / 'CATALOGO_MASCARAS.json')
    with (CLIENT / 'VALIDACION_MANIFEST.json').open('x', encoding='utf-8') as stream:
        json.dump({name: sha(CLIENT / name) for name in CLIENT_NAMES}, stream, indent=2)
    prep = dict(status='PREPARED_SRC_GUARDS_AND_INVALIDATIONS_PENDING', source=str(SOURCE), configuration=CONFIG,
        source_inventory_before=snapshot, existing_elf_sha256=elfs, accepted_launch_reserializations=reserialized,
        previous_fourteen_configurations_preserved=True, modified_files=list(MODIFIED), added_files=list(ADDED),
        hardware_flashed=False, serial_port_opened=False, physical_validation_pending=True,
        first_physical_scope='three old EntM BASE plus three new F14 RUN on development index 0',
        training_executed=False, dnn_or_detector_included_in_mask_cost=False,
        flash_of_multikernel_harness_not_final_deployment=True)
    origin(prep)
    client_check()
    save('PREPARACION.json', prep)
    print('PREPARED: faltan guardias Src y tres invalidaciones en copia aislada', flush=True)


def flags(commands, config):
    def macro(line, name, value):
        values = re.findall(r'(?<!\S)-D' + re.escape(name) + r'(?:=([^\s]+))?(?=\s|$)', line)
        return values == [value] if value is not None else values in ([], ['0'])
    integrated = config == PROFILE_CONTROL
    expected = {'CWRU_TARGET_HCLK_HZ': '180000000U', 'CWRU_CACHE_SPECTRAL_POWER': '1', 'CWRU_ENTROPY_MODE': '2',
        'CWRU_INPUT_TRANSFORM': '1' if integrated else None, 'CWRU_DNN_TEST': '1' if integrated else None,
        'CWRU_NORMALITY_TEST': '1' if integrated else None, 'CWRU_PIPELINE_TEST': '1' if integrated else None,
        'CWRU_STACK_TEST': None, 'CWRU_PROFILE_EXTRACTOR': '1' if integrated else None,
        'CWRU_COST_PROFILE': '1' if integrated else None, 'CWRU_MASK_BENCH': '1' if config == CONFIG else None}
    return bool(commands) and all(re.findall(r'(?<!\S)-O\S+', line) == ['-O3'] and
        all(macro(line, k, v) for k, v in expected.items()) and
        not any(f in line for f in ('-ffast-math', '-Ofast', '-funsafe-math-optimizations')) and
        ('-I../CWRU_Masks' in line) == (config == CONFIG) for line in commands)


def inspect(tool, config, prep, attempt):
    folder = PROJECT / config
    elf = folder / (NAME + '.elf')
    values = subprocess.check_output([str(tool / 'arm-none-eabi-size.exe'), str(elf)], text=True).splitlines()[-1].split()
    text, data, bss = map(int, values[:3])
    result = dict(elf_sha256=sha(elf), flash_load_bytes=text + data,
        ram_static_with_linker_reservations_bytes=data + bss,
        flags_confirmed=flags(prior.compiler_commands(folder), config),
        compiler_stack_frames={p.relative_to(folder).as_posix(): p.read_text(encoding='utf-8').splitlines()
                               for p in folder.rglob('*.su')})
    if config != CONFIG:
        result.update(control_elf_identical=sha(elf) == prep['existing_elf_sha256'][config],
            control_flash_identical=flash_image(elf) == flash_image(BACKUP / config / (NAME + '.elf')))
        return result
    symbols = subprocess.check_output([str(tool / 'arm-none-eabi-nm.exe'), '-S', '-C', '--defined-only', str(elf)], text=True)
    (OUT / f'SIMBOLOS_MASCARAS_INTENTO_{attempt}.txt').write_text(symbols, encoding='utf-8')
    expected = {'g_window': 32768, 'g_fft_input': 16384, 'g_fft_output': 16384}
    allocations = {}
    for name, size in expected.items():
        rows = re.findall(r'(?m)^([0-9a-f]+)\s+([0-9a-f]+)\s+[bBdD]\s+' + re.escape(name) + r'$', symbols)
        require(len(rows) == 1 and int(rows[0][1], 16) == size, 'Buffer duplicado o diferente: ' + name)
        allocations[name] = [int(rows[0][0], 16), size]
    ranges = sorted((a, a + size) for a, size in allocations.values())
    require(all(a[1] <= b[0] for a, b in zip(ranges, ranges[1:])), 'Superposicion de buffers')
    large = re.findall(r'(?m)^([0-9a-f]+)\s+([0-9a-f]+)\s+[bBdD]\s+(.+)$', symbols)
    require(len([r for r in large if int(r[1], 16) >= 16384 and 0x20000000 <= int(r[0], 16) < 0x20020000]) == 3,
            'Otro buffer grande inesperado; este harness no reserva arena DNN')
    require(not re.search(r'(?m)^.*\s+[bBdD]\s+(?:\(anonymous namespace\)::arena|g_profile)$', symbols),
            'Arena DNN o perfil inesperados')
    output = re.findall(r'(?m)^([0-9a-f]+)\s+([0-9a-f]+)\s+[bBdD]\s+gmask_output$', symbols)
    require(len(output) == 1 and int(output[0][1], 16) == 56, 'Salida de mascara diferente de 56 B')
    result['buffer_audit'] = dict(status='PASS', allocations=allocations, nonoverlapping=True,
        large_buffers=3, dnn_arena_absent=True, masked_output_bytes=56)
    required = ['CWRUMASK_' + name for name, _ in KERNELS]
    required += ['CWRUMASK_Extract', 'CWRUMASK_GetSpec', 'CWRUMASK_CatalogFNV', 'CWRUMASK_Command',
                 'CWRU_FeatureExtraction_Extract']
    result['required_kernel_symbols_present'] = all(re.search(r'(?m)^.*\s+[tT]\s+' + re.escape(name) + r'$', symbols)
                                                   for name in required)
    assembly = {}
    for name in required:
        assembly[name] = subprocess.check_output([str(tool / 'arm-none-eabi-objdump.exe'), '-d', '--disassemble=' + name, str(elf)], text=True)
        (OUT / f'DESENSAMBLADO_{name}_INTENTO_{attempt}.txt').write_text(assembly[name], encoding='utf-8')
    no_spectral = not re.search(r'<(?:arm_rfft\w*|logf?|__ieee754_log\w*)>', assembly['CWRUMASK_T6'])
    no_entropy = not re.search(r'<(?:logf?|__ieee754_log\w*)>', assembly['CWRUMASK_NE12'])
    result['kernel_assembly_prechecks'] = dict(T6_no_fft_or_log_call=no_spectral, NE12_no_log_call=no_entropy)
    result['manual_kernel_disassembly_review_required'] = True
    result['flash_is_multikernel_harness_not_final_model'] = True
    result['ram_reduction_claimed'] = False
    return result


def old_sources(tool):
    includes = ('Inc', 'CWRU_Input', 'CWRU_DNN', 'CWRU_Detector', 'CWRU_Pipeline', 'CWRU_Cost', 'CWRU_Stack',
        'CWRU_Masks', 'Drivers/CMSIS/Include', 'Drivers/CMSIS/Device/ST/STM32F4xx/Include',
        'Drivers/STM32F4xx_HAL_Driver/Inc', 'Middlewares/CMSIS-DSP/Include', 'Middlewares/CMSIS-DSP/PrivateInclude')
    base = [str(tool / 'arm-none-eabi-gcc.exe'), '-E', '-P', '-std=gnu11', '-mcpu=cortex-m4', '-mfpu=fpv4-sp-d16',
        '-mfloat-abi=hard', '-mthumb', '-DSTM32F446xx', '-DCWRU_MASK_BENCH=0', '-DCWRU_TARGET_HCLK_HZ=180000000U',
        '-DCWRU_CACHE_SPECTRAL_POWER=1', '-DCWRU_ENTROPY_MODE=2', *('-I' + str(PROJECT / n) for n in includes)]
    cases = [(a, b, c, d, 0, 0, 0) for a, b, c, d in ((0, 0, 0, 0), (1, 0, 0, 0),
             (1, 1, 0, 0), (1, 1, 1, 0), (1, 1, 1, 1))]
    cases += [(1, 1, 1, 1, 1, 1, 0), (1, 1, 1, 1, 0, 0, 1)]
    results = {}
    for name in MODIFIED[1:]:
        for inp, nn, det, pipe, profile, cost, stack in cases:
            args = base + [f'-DCWRU_INPUT_TRANSFORM={inp}', f'-DCWRU_DNN_TEST={nn}', f'-DCWRU_NORMALITY_TEST={det}',
                f'-DCWRU_PIPELINE_TEST={pipe}', f'-DCWRU_PROFILE_EXTRACTOR={profile}',
                f'-DCWRU_COST_PROFILE={cost}', f'-DCWRU_STACK_TEST={stack}']
            before = subprocess.check_output(args + [str(BACKUP / name)])
            after = subprocess.check_output(args + [str(PROJECT / name)])
            results[name + f':{inp}{nn}{det}{pipe}{profile}{cost}{stack}'] = dict(identical=before == after,
                before=hashlib.sha256(before).hexdigest(), after=hashlib.sha256(after).hexdigest())
    return results


def build():
    prep = load('PREPARACION.json')
    origin(prep)
    client_check()
    require(not (OUT / 'COMPILACION_MASCARAS.json').exists(), 'Build ya aprobado')
    uart = (PROJECT / MODIFIED[1]).read_text(encoding='utf-8')
    receiver = (PROJECT / MODIFIED[2]).read_text(encoding='utf-8')
    require('CWRUMASK_Command' in uart and 'cwru_masks_commands.inc' in receiver, 'Faltan guardias Src')
    require(receiver.count('CWRUMASK_Invalidate();') == 3, 'Faltan las tres invalidaciones de entrada/FE_RUN')
    for name in TEMPLATE_NAMES:
        require(sha(PROJECT / 'CWRU_Masks' / name) == sha(TEMPLATES / name), 'Plantilla diferente: ' + name)
    attempts = [int(p.stem.rsplit('_', 1)[-1]) for p in OUT.glob('CUBEIDE_MASCARAS_INTENTO_*.log')]
    attempt = max(attempts, default=0) + 1
    require(attempt <= 3, 'Tres intentos agotados: revisar antes de ampliar')
    for name in FILES:
        cp(PROJECT / name, OVERLAY / name)
        cp(PROJECT / name, OUT / 'intentos' / str(attempt) / 'archivos' / name)
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
    report = dict(attempt=attempt, command=args, configurations={}, hardware_flashed=False, serial_port_opened=False)
    with (OUT / f'CUBEIDE_MASCARAS_INTENTO_{attempt}.log').open('x', encoding='utf-8') as log:
        tests = subprocess.run([sys.executable, '-B', str(CLIENT / 'validar_mascaras.py'), '--self-test',
            '--output-root', str(OUT / '_self_test' / str(attempt))], capture_output=True, text=True, encoding='utf-8', env=env)
        save(f'PRUEBAS_CLIENTE_INTENTO_{attempt}.json', dict(returncode=tests.returncode,
            output=tests.stdout + tests.stderr, simulation_only=True))
        require(tests.returncode == 0, 'Autotest cliente fallo; conservar log')
        process = subprocess.Popen(args, cwd=OUT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   encoding='utf-8', errors='replace')
        for line in process.stdout:
            log.write(line)
            log.flush()
            if any(t in line.lower() for t in ('build of', 'error:', 'warning:', 'build finished')):
                print(line.strip()[:250], flush=True)
        report['returncode'] = process.wait(timeout=30)
    text = (OUT / f'CUBEIDE_MASCARAS_INTENTO_{attempt}.log').read_text(encoding='utf-8')
    summaries = re.findall(r'(\d+) errors, (\d+) warnings', text)
    report['zero_errors_warnings'] = (len(summaries) >= 3 and all(a == b == '0' for a, b in summaries)
        and not re.search(r'\b(error|warning):|undefined reference', text, re.I))
    for name in SELECTED:
        try:
            report['configurations'][name] = inspect(tool, name, prep, attempt)
        except Exception as error:
            report['configurations'][name] = dict(inspection_error=str(error))
        for path in (PROJECT / name).rglob('*'):
            if path.is_file() and (path.suffix in ('.elf', '.map', '.su', '.mk') or path.name in ('makefile', 'objects.list')):
                cp(path, OUT / 'intentos' / str(attempt) / 'compilacion' / name / path.relative_to(PROJECT / name))
    report['previous_fourteen_configurations_preserved'] = same_configs(PROJECT / '.cproject')
    report['other_resources_unchanged'] = all(sha(PROJECT / row['path']) == row['sha256']
        for row in prep['source_inventory_before'] if row['path'] not in (*MODIFIED, '.settings/language.settings.xml'))
    report['old_sources_preprocessed'] = old_sources(tool)
    report['overlay_sha256'] = {name: sha(OVERLAY / name) for name in FILES}
    origin(prep)
    rows = report['configurations']
    masks = rows[CONFIG]
    passed = (report['returncode'] == 0 and report['zero_errors_warnings']
        and all(r.get('flags_confirmed', False) for r in rows.values()))
    passed = passed and all(rows[n].get('control_elf_identical', False) and
                           rows[n].get('control_flash_identical', False) for n in SELECTED[:2])
    passed = passed and report['previous_fourteen_configurations_preserved'] and report['other_resources_unchanged']
    passed = passed and all(row['identical'] for row in report['old_sources_preprocessed'].values())
    passed = passed and masks.get('buffer_audit', {}).get('status') == 'PASS' and masks.get('required_kernel_symbols_present', False)
    passed = passed and all(masks.get('kernel_assembly_prechecks', {'missing': False}).values())
    report['status'] = 'PASS_PENDING_MANUAL_KERNEL_REVIEW' if passed else 'FAIL'
    save(f'COMPILACION_MASCARAS_INTENTO_{attempt}.json', report)
    if passed:
        save('COMPILACION_MASCARAS.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'configurations'}, ensure_ascii=True, indent=2), flush=True)
    if not passed:
        raise SystemExit(2)


def metadata_check():
    name = '.settings/language.settings.xml'
    before, after = ET.parse(BACKUP / name).getroot(), ET.parse(SOURCE / name).getroot()
    if sha(BACKUP / name) == sha(SOURCE / name):
        return True
    require(len(before) == 14 and all(c.tag == 'configuration' for c in before), 'CDT previo ambiguo')
    selected = next(c for c in before if c.get('name') == CONTROL)
    new = copy.deepcopy(selected)
    new.set('name', CONFIG)
    new.set('id', configs(SOURCE / '.cproject')[2][CONFIG].get('id'))
    before.append(new)
    return canonical(before) == canonical(after)


def installed_checks(prep, hashes):
    return dict(previous_fourteen_configurations_preserved=same_configs(SOURCE / '.cproject'),
        previous_fourteen_elfs_unchanged=all(sha(SOURCE / n / (NAME + '.elf')) == d for n, d in prep['existing_elf_sha256'].items()),
        installed_files_match_overlay=all(sha(SOURCE / n) == d for n, d in hashes.items()),
        other_resources_unchanged_except_exact_cdt_append=all(sha(SOURCE / r['path']) == r['sha256'] or
            (r['path'] == '.settings/language.settings.xml' and metadata_check())
            for r in prep['source_inventory_before'] if r['path'] not in MODIFIED))


def install():
    prep, built, reviewed = load('PREPARACION.json'), load('COMPILACION_MASCARAS.json'), load('REVISION_KERNELS.json')
    require(built['status'] == 'PASS_PENDING_MANUAL_KERNEL_REVIEW' and reviewed['status'] == 'PASS',
            'Falta build o revision manual aprobada')
    require(reviewed['elf_sha256'] == built['configurations'][CONFIG]['elf_sha256'], 'Revision de otro ELF')
    require(not (OUT / 'APLICACION_EN_D.json').exists(), 'Instalacion ya registrada')
    origin(prep)
    for name, digest in built['overlay_sha256'].items():
        require(sha(OVERLAY / name) == sha(PROJECT / name) == digest, 'Overlay cambio: ' + name)
    for row in prep['source_inventory_before']:
        require(sha(BACKUP / row['path']) == row['sha256'], 'Respaldo cambio')
    for name in FILES:
        cp(OVERLAY / name, SOURCE / name)
    checks = installed_checks(prep, built['overlay_sha256'])
    report = dict(status='PASS' if all(checks.values()) else 'FAIL', **checks,
        installed_sha256=built['overlay_sha256'], modified_files=list(MODIFIED), added_files=list(ADDED),
        hardware_flashed=False, serial_port_opened=False, physical_validation_pending=True,
        dnn_or_detector_included_in_mask_cost=False, flash_is_multikernel_harness_not_final_deployment=True)
    save('APLICACION_EN_D.json', report)
    print(json.dumps(report, indent=2))
    require(report['status'] == 'PASS', 'Instalacion fallo')


def seal():
    require(not (OUT / 'MANIFEST_SHA256.json').exists(), 'Sello ya existente')
    prep, installed = load('PREPARACION.json'), load('APLICACION_EN_D.json')
    require(installed['status'] == 'PASS' and all(installed_checks(prep, installed['installed_sha256']).values()),
            'Instalacion no aprobada o D cambio')
    metadata = SOURCE / '.settings/language.settings.xml'
    cp(metadata, OUT / 'language.settings.postinstalacion.xml')
    save('METADATOS_CDT_POSTINSTALACION.json', dict(status='PASS', exact_append_or_unchanged=True,
        before_sha256=sha(BACKUP / '.settings/language.settings.xml'), after_sha256=sha(metadata), overwritten_in_D=False))
    client_check()
    entries = prior_checks()
    paths = {ROOT / n for n in entries}
    paths.update(p for p in OUT.iterdir() if p.is_file())
    for folder in (BACKUP, OVERLAY, CLIENT, OUT / 'intentos', OUT / '_self_test', OUT / '_self_test_previo'):
        paths.update(p for p in folder.rglob('*') if p.is_file() and not p.is_relative_to(CLIENT / 'resultados'))
    for row in prep['source_inventory_before']:
        paths.add(PROJECT / row['path'])
    paths.update(PROJECT / name for name in FILES)
    paths.update(TEMPLATES / name for name in TEMPLATE_NAMES)
    paths.update(Path(__file__).with_name(name) for name in ('preparar_mascaras_cwru.py', 'validar_mascaras_cwru.py'))
    for name in SELECTED:
        paths.update(p for p in (PROJECT / name).rglob('*') if p.is_file() and
            (p.suffix in ('.elf', '.map', '.su', '.mk') or p.name in ('makefile', 'objects.list')))
    require(all(p.is_file() and p.resolve().is_relative_to(ROOT) for p in paths), 'Ruta requerida ausente o fuera')
    save('MANIFEST_SHA256.json', {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(paths)})
    with (OUT / 'MANIFEST_SHA256.sha256').open('x', encoding='ascii') as stream:
        stream.write(sha(OUT / 'MANIFEST_SHA256.json') + '  MANIFEST_SHA256.json\n')
    verify()


def verify():
    digest, name = (OUT / 'MANIFEST_SHA256.sha256').read_text(encoding='ascii').split()
    require(name == 'MANIFEST_SHA256.json' and sha(OUT / name) == digest, 'Sello diferente')
    manifest = load(name)
    for relative, expected in manifest.items():
        path = (ROOT / relative).resolve()
        require(path.is_relative_to(ROOT) and sha(path) == expected, 'Recurso modificado: ' + relative)
    print(json.dumps(dict(status='PASS', files_verified=len(manifest), serial_port_opened=False,
                         physical_validation_pending=True)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'build', 'install', 'seal', 'verify'))
    globals()[parser.parse_args().mode]()
