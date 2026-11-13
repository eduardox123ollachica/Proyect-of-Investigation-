"""Instala el banco TotalV2 auditado, sin compilar, programar ni abrir COM.

--check es solo lectura. --install requiere ambos dictamenes ELF/maquina,
respalda en el workspace, copia unicamente el overlay y ELF/map verificados,
y activa .cproject al final. No borra ni mueve directorios.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import struct
import time
import traceback
import xml.etree.ElementTree as ET
import zipfile


ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / 'despliegue_stm32/costo_total_representantes_v2_180mhz_2026-10-25'
SOURCE = Path('D:/STM32IDE/Prueba_2_CWRU_Extraccion')
NAME = SOURCE.name
CONFIG = 'Release_180MHz_TotalV2'
PREPARER = ROOT / 'codigo/cwru/ampliacion_v2/costo_total/preparar_total.py'
MANIFEST = OUT / 'MANIFEST_PREPARACION_TOTAL_SHA256.json'
EXPECTED_MANIFEST_SHA = '06d1401e89fb1fc09bf4cfe4b0b022eeeb1639a7c2239d0f5d6c7f37c2ae694c'
BACKUP = OUT / 'respaldo_instalacion_total'
STAGED = OUT / 'archivos_instalacion_total'
REPORT = OUT / 'APLICACION_TOTAL_EN_D.json'
PARTIAL = OUT / 'APLICACION_TOTAL_EN_D_PARCIAL.json'
PLAN = OUT / 'PLAN_INSTALACION_TOTAL.json'
INSTALL_MANIFEST = OUT / 'MANIFEST_INSTALACION_TOTAL_SHA256.json'
SAFE_LAUNCH = NAME + '_TotalV2_180MHz_SinRecompilar.launch'
CDT_JAR = Path('D:/STM32CubeIDE_2.2.0/STM32CubeIDE/plugins/org.eclipse.cdt.debug.core_9.0.300.202512020204.jar')
CDT_CLASS = 'org/eclipse/cdt/debug/core/ICDTLaunchConfigurationConstants.class'
BUILD_KEY = 'org.eclipse.cdt.launch.ATTR_BUILD_BEFORE_LAUNCH_ATTR'
LOAD_KEY = 'com.st.stm32cube.ide.mcu.debug.launch.loadList'


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def confined(base, relative):
    relative = Path(relative)
    require(not relative.is_absolute() and relative.parts and '..' not in relative.parts,
            'Ruta relativa invalida: ' + str(relative))
    target = (base / relative).resolve()
    require(target != base.resolve() and target.is_relative_to(base.resolve()),
            'Ruta fuera del destino autorizado: ' + str(target))
    return target


def canonical(node):
    return node.tag, tuple(sorted(node.attrib.items())), (node.text or '').strip(), tuple(canonical(c) for c in node)


def canonical_cdt(node):
    attrs = dict(node.attrib)
    detector = 'com.st.stm32cube.ide.mcu.toolchain.armnone.setup.CrossBuiltinSpecsDetector'
    if node.tag == 'provider' and attrs.get('class') == attrs.get('id') == detector and 'env-hash' in attrs:
        require(re.fullmatch(r'-?[0-9]+', attrs['env-hash']) is not None, 'env-hash CDT no decimal')
        attrs.pop('env-hash')
    return node.tag, tuple(sorted(attrs.items())), (node.text or '').strip(), tuple(canonical_cdt(c) for c in node)


def metadata_append(before_path, after_path, config_id):
    before = ET.parse(before_path).getroot(); after = ET.parse(after_path).getroot()
    require(before.tag == after.tag == 'project' and before.attrib == after.attrib
            and len(before) == 15 and len(after) == 16, 'Metadatos CDT fuera de una adicion15a16')
    old = {n.get('id'):n for n in before}; new = {n.get('id'):n for n in after}
    require(len(old) == 15 and len(new) == 16 and all(i in new and canonical_cdt(n) == canonical_cdt(new[i])
            for i,n in old.items()), 'Metadatos anteriores cambiaron fuera del cache env-hash del detector')
    extra = [n for i,n in new.items() if i not in old]
    require(len(extra) == 1 and extra[0].get('id') == config_id and extra[0].get('name') == CONFIG,
            'Adicion CDT no corresponde a TotalV2')
    mask = next(n for n in before if n.get('name') == 'Release_180MHz_Masks')
    expected = ET.fromstring(ET.tostring(mask))
    expected.set('name',CONFIG); expected.set('id',config_id)
    require(canonical_cdt(expected) == canonical_cdt(extra[0]), 'Proveedores CDT TotalV2 distintos del clon Masks')


def verify_preparation_seal():
    digest, declared = MANIFEST.with_suffix('.sha256').read_text(encoding='ascii').split()
    require(declared == MANIFEST.name and sha(MANIFEST) == digest == EXPECTED_MANIFEST_SHA,
            'El sello de preparacion no es el revisado')
    entries = read(MANIFEST)
    require(len(entries) == 1968, 'El sello de preparacion debe contener1968recursos')
    exceptions = []
    copy_metadata = OUT/'copia_aislada'/NAME/'.settings/language.settings.xml'
    for name, expected in entries.items():
        target = confined(ROOT, name)
        require(target.is_file(), 'Recurso de preparacion ausente: '+name)
        actual = sha(target)
        if actual != expected:
            require(target == copy_metadata.resolve(), 'Preparacion modificada fuera de metadatos CDT de copia: '+name)
            backup = OUT/'respaldo_antes_total/.settings/language.settings.xml'
            require(sha(backup) == expected, 'Metadatos CDT iniciales no corresponden al sello')
            root = ET.parse(OUT/'archivos_para_D/.cproject').getroot()
            cfg = next(n for n in root.iter('configuration') if n.get('name') == CONFIG)
            metadata_append(backup,target,cfg.get('id'))
            exceptions.append(dict(path=name, sealed_sha256=expected, current_sha256=actual,
                cause='CubeIDE genero la entrada TotalV2 y actualizo solo env-hash de CrossBuiltinSpecsDetector',
                prior_fifteen_semantically_preserved_ignoring_only_detector_env_hash=True,
                original_seal_not_rewritten=True, copied_to_D=False))
    require(len(exceptions) <= 1, 'Excepciones de preparacion inesperadas')
    return exceptions


def load_preparer():
    spec = importlib.util.spec_from_file_location('totalv2_preparer_for_install', PREPARER)
    require(spec and spec.loader, 'Preparador no importable')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def class_constants(data):
    """Lee ConstantValue de los campos de una clase JVM, sin ejecutar Java."""
    require(data[:4] == b'\xca\xfe\xba\xbe', 'Clase JVM invalida')
    pos = 8
    def u2():
        nonlocal pos
        value = struct.unpack_from('>H', data, pos)[0]; pos += 2
        return value
    def u4():
        nonlocal pos
        value = struct.unpack_from('>I', data, pos)[0]; pos += 4
        return value
    count = u2(); cp = [None] * count; i = 1
    while i < count:
        tag = data[pos]; pos += 1
        if tag == 1:
            length = u2(); cp[i] = ('utf8', data[pos:pos+length].decode('utf-8')); pos += length
        elif tag == 3:
            cp[i] = ('integer', struct.unpack_from('>i', data, pos)[0]); pos += 4
        elif tag == 8:
            cp[i] = ('string', u2())
        elif tag in (5, 6):
            pos += 8; i += 1
        elif tag == 4:
            pos += 4
        elif tag in (7, 16, 19, 20):
            pos += 2
        elif tag in (9, 10, 11, 12, 17, 18):
            pos += 4
        elif tag == 15:
            pos += 3
        else:
            raise RuntimeError('Etiqueta JVM no admitida: ' + str(tag))
        i += 1
    def value(index):
        tag, content = cp[index]
        return value(content) if tag == 'string' else content
    pos += 6
    interfaces = u2()
    pos += 2 * interfaces
    fields = {}
    for _ in range(u2()):
        u2(); name = value(u2()); u2()
        for _ in range(u2()):
            attribute = value(u2()); size = u4()
            if attribute == 'ConstantValue':
                require(size == 2, 'ConstantValue JVM invalido')
                fields[name] = value(u2())
            else:
                pos += size
    return fields


def disable_build_provenance():
    with zipfile.ZipFile(CDT_JAR) as jar:
        data = jar.read(CDT_CLASS)
    fields = class_constants(data)
    expected = dict(BUILD_BEFORE_LAUNCH_DISABLED=0, BUILD_BEFORE_LAUNCH_ENABLED=1,
                    BUILD_BEFORE_LAUNCH_USE_WORKSPACE_SETTING=2, ATTR_BUILD_BEFORE_LAUNCH=BUILD_KEY)
    require(all(fields.get(k) == v for k, v in expected.items()), 'Constantes CDT diferentes de las revisadas')
    return dict(jar=str(CDT_JAR), jar_sha256=sha(CDT_JAR), class_entry=CDT_CLASS,
                class_sha256=hashlib.sha256(data).hexdigest(), verified_constants=expected)


def safe_launch(original, config_id):
    root = ET.fromstring(original)
    children = list(root)
    attrs = {n.get('key'): n for n in children}
    require(len(attrs) == len(children), 'Claves launch duplicadas')
    require(attrs[BUILD_KEY].tag == 'intAttribute' and attrs[BUILD_KEY].get('value') == '2', 'Estado build launch inesperado')
    require(attrs['org.eclipse.cdt.launch.PROJECT_BUILD_CONFIG_AUTO_ATTR'].get('value') == 'false'
            and attrs['org.eclipse.cdt.launch.PROJECT_BUILD_CONFIG_ID_ATTR'].get('value') == config_id
            and attrs['org.eclipse.cdt.launch.PROGRAM_NAME'].get('value') == CONFIG+'/'+NAME+'.elf'
            and attrs['org.eclipse.cdt.launch.PROJECT_ATTR'].get('value') == NAME, 'Launch apunta a otro proyecto/ELF')
    before = {k: canonical(n) for k, n in attrs.items()}
    loads = json.loads(attrs[LOAD_KEY].get('value'))
    require(list(loads) == ['fItems'] and len(loads['fItems']) == 1, 'Lista de carga inesperada')
    item = loads['fItems'][0]
    require(item['fPath'] == CONFIG+'/'+NAME+'.elf' and item['fProjectName'] == NAME
            and item['fPerformBuild'] is True and item['fDownload'] is True and item['fLoadSymbols'] is True,
            'Elemento de carga inesperado')
    item['fPerformBuild'] = False
    attrs[LOAD_KEY].set('value', json.dumps(loads, separators=(',', ':')))
    attrs[BUILD_KEY].set('value', '0')
    require(all(canonical(n) == before[k] for k, n in attrs.items() if k not in (BUILD_KEY, LOAD_KEY)),
            'Se altero otra opcion de launch')
    data = b'<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(root, encoding='utf-8') + b'\n'
    return data, dict(source_sha256=hashlib.sha256(original).hexdigest(), derived_sha256=hashlib.sha256(data).hexdigest(),
        changes=[dict(key=BUILD_KEY, before=2, after=0), dict(key=LOAD_KEY+'.fItems[0].fPerformBuild', before=True, after=False)],
        stop_at_main_unchanged=True, other_launch_attributes_unchanged=True)


def common_inputs():
    require(ROOT.resolve() == Path('C:/Users/LENOVO/Desktop/Proyecto Investigación').resolve(), 'Workspace inesperado')
    require(SOURCE.resolve().parent == Path('D:/STM32IDE').resolve() and SOURCE.resolve().name == NAME,
            'Destino D fuera del proyecto previsto')
    require(OUT.resolve().is_relative_to(ROOT.resolve()) and BACKUP.resolve().is_relative_to(OUT.resolve())
            and STAGED.resolve().is_relative_to(OUT.resolve()), 'Destino de respaldo fuera del workspace')
    seal_exceptions = verify_preparation_seal()
    prep = read(OUT/'PREPARACION.json'); built = read(OUT/'COMPILACION_TOTAL.json')
    audit = read(OUT/'AUDITORIA_ELF_TOTAL.json'); machine = read(OUT/'AUDITORIA_MAQUINA_TOTAL.json')
    require(built['status'] == 'PASS_PENDING_ELF_MANUAL_REVIEW_AND_ARM' and built['returncode'] == 0
            and built['zero_errors_warnings'] and not built['timed_out'] and built['prepared_resources_unchanged'],
            'Compilacion no aprobada')
    elf_sha = built['configurations'][CONFIG]['elf_sha256']
    require(audit['status'] == 'PASS_ELF_CATALOG_PARAMETERS_VERIFIED_PENDING_ARM'
            and audit['elf_sha256'] == elf_sha, 'Falta auditoria de bytes ELF aprobada para este binario')
    require(machine['status'] == 'PASS_MACHINE_CODE_REVIEW_PENDING_ARM'
            and machine['elf_sha256'] == elf_sha, 'Falta revision de instrucciones aprobada para este binario')
    require(machine['build_report_sha256'] == sha(OUT/'COMPILACION_TOTAL.json'), 'Revision de instrucciones de otro informe')
    for config, row in built['configurations'].items():
        require(row['flags_confirmed'], 'Banderas de compilacion no confirmadas: ' + config)
        if config != CONFIG:
            require(row['control_elf_identical'] and row['control_flash_identical'], 'Un control de compilacion cambio')
    p = load_preparer()
    require(p.CONFIG == CONFIG and p.SOURCE.resolve() == SOURCE.resolve() and len(prep['existing_elf_sha256']) == 15,
            'Preparador o configuraciones de origen inesperados')
    require(p.same_configs(p.PROJECT/'.cproject'), 'Las quince configuraciones previas no se preservaron')
    p.resource_check()
    for name, expected in prep['overlay_sha256'].items():
        require(sha(confined(p.OVERLAY, name)) == sha(confined(p.PROJECT, name)) == expected, 'Overlay/copia cambio: '+name)
    for name, expected in prep['source_scripts_sha256'].items():
        require(sha(confined(p.HERE, name)) == expected, 'Script de preparacion cambio: '+name)
    require(sha(p.CLIENT/'validar_total.py') == prep['client_sha256'], 'Cliente cambio')
    targets = []
    for name in (*p.ADDED, *p.MODIFIED):
        targets.append(dict(relative=name, source=str(confined(p.OVERLAY,name)), sha256=prep['overlay_sha256'][name],
                            overwrite=(name in p.MODIFIED)))
    new_elf = confined(p.PROJECT, CONFIG+'/'+NAME+'.elf')
    archive = confined(OUT, 'intentos/'+str(built['attempt'])+'/compilacion/'+CONFIG+'/'+NAME+'.elf')
    require(sha(new_elf) == sha(archive) == elf_sha, 'ELF construido/archivado cambio')
    for suffix in ('.elf', '.map'):
        relative = CONFIG+'/'+NAME+suffix
        source = confined(p.PROJECT, relative)
        archived = confined(OUT, 'intentos/'+str(built['attempt'])+'/compilacion/'+relative)
        require(source.is_file() and source.stat().st_size > 0 and sha(source) == sha(archived), 'Artefacto/archivo diferente: '+relative)
        if suffix == '.map':
            require(sha(source) == machine['map_sha256'], 'Mapa diferente del revisado')
        targets.append(dict(relative=relative, source=str(source), sha256=sha(source), overwrite=False))
    provenance = disable_build_provenance()
    config_id = p.configs(p.PROJECT/'.cproject')[2][CONFIG].get('id')
    data, launch_info = safe_launch(confined(p.OVERLAY,p.LAUNCH).read_bytes(), config_id)
    targets.append(dict(relative=SAFE_LAUNCH, source=None, sha256=hashlib.sha256(data).hexdigest(), overwrite=False))
    # .cproject se activa al final, una vez que todas sus fuentes y el ELF existen.
    targets.sort(key=lambda row: (row['relative'] == '.cproject', row['overwrite'], row['relative']))
    require(len(targets) == 15 and len({r['relative'] for r in targets}) == 15, 'Plan de quince archivos inesperado')
    for row in targets:
        confined(SOURCE, row['relative'])
    inputs = {name: sha(OUT/name) for name in ('PREPARACION.json','COMPILACION_TOTAL.json','AUDITORIA_ELF_TOTAL.json',
                                            'AUDITORIA_MAQUINA_TOTAL.json','MANIFEST_PREPARACION_TOTAL_SHA256.json')}
    return p, prep, built, targets, data, dict(launch=launch_info, cdt=provenance, input_sha256=inputs,
        elf_sha256=elf_sha, preparation_seal_exceptions=seal_exceptions)


def metadata_preserved(p, prep):
    name = '.settings/language.settings.xml'
    old_expected = next(r['sha256'] for r in prep['source_inventory_before'] if r['path'] == name)
    actual = confined(SOURCE,name)
    if sha(actual) == old_expected:
        return dict(status='UNCHANGED', sha256=old_expected)
    expected_id = p.configs(SOURCE/'.cproject')[2][CONFIG].get('id')
    metadata_append(p.BACKUP/name,actual,expected_id)
    return dict(status='ONE_CDT_APPEND_WITH_PRIOR_FIFTEEN_PRESERVED_EXCEPT_DETECTOR_ENV_HASH',
                sha256=sha(actual), overwritten_by_installer=False)


def installed_checks(p, prep, targets):
    for row in targets:
        require(sha(confined(SOURCE,row['relative'])) == row['sha256'], 'Archivo instalado diferente: '+row['relative'])
    require(p.same_configs(SOURCE/'.cproject') and b'<?fileVersion 4.0.0?>' in (SOURCE/'.cproject').read_bytes(),
            'Configuraciones/PI CDT cambiaron')
    for name, expected in prep['existing_elf_sha256'].items():
        require(sha(confined(SOURCE,name+'/'+NAME+'.elf')) == expected, 'ELF anterior cambio: '+name)
    for row in prep['source_inventory_before']:
        name = row['path']
        if name not in p.MODIFIED and name != '.settings/language.settings.xml':
            require(sha(confined(SOURCE,name)) == row['sha256'], 'Otra fuente original cambio: '+name)
    return dict(installed_files_verified=len(targets), previous_configurations_preserved=15,
                previous_ELF_hashes_preserved=15, previous_other_source_files_preserved=True,
                metadata=metadata_preserved(p,prep))


def check():
    p,prep,built,targets,data,context = common_inputs()
    require(not PARTIAL.exists(), 'Instalacion parcial registrada; revisar antes de continuar')
    if REPORT.exists():
        report = read(REPORT)
        require(report['status'] == 'PASS_INSTALLED_PENDING_ARM' and report['elf_sha256'] == context['elf_sha256'],
                'Registro de instalacion incompatible')
        digest,declared = INSTALL_MANIFEST.with_suffix('.sha256').read_text(encoding='ascii').split()
        require(declared == INSTALL_MANIFEST.name and sha(INSTALL_MANIFEST) == digest, 'Sello de instalacion diferente')
        for name,expected in read(INSTALL_MANIFEST).items():
            require(sha(confined(ROOT,name)) == expected, 'Evidencia de instalacion modificada: '+name)
        checks = installed_checks(p,prep,targets)
        print(json.dumps(dict(status='ALREADY_INSTALLED_VERIFIED_NO_REWRITE', **checks),ensure_ascii=False))
        return p,prep,built,targets,data,context,True
    require(not PARTIAL.exists() and not PLAN.exists() and not BACKUP.exists() and not STAGED.exists(),
            'Hay preparacion/instalacion parcial; revisar antes de reintentar')
    p.verify_prepared(prep)
    require(not (SOURCE/CONFIG).exists() and not (SOURCE/'CWRU_TotalV2').exists(), 'Destino TotalV2 ya existe sin registro')
    for row in targets:
        target = confined(SOURCE,row['relative'])
        require(target.is_file() if row['overwrite'] else not target.exists(), 'Destino inesperado: '+row['relative'])
    print(json.dumps(dict(status='READY_FOR_INSTALL_NOT_ARM_VALIDATED', targets=len(targets),
                         overwrite=sum(r['overwrite'] for r in targets), added=sum(not r['overwrite'] for r in targets),
                         original_configurations=15, next_configurations=16, elf_sha256=context['elf_sha256'],
                         source_project=str(SOURCE.resolve()), serial_port_opened=False, hardware_flashed=False),ensure_ascii=False))
    return p,prep,built,targets,data,context,False


def install():
    p,prep,built,targets,data,context,already = check()
    if already:
        return
    started = time.monotonic(); touched = []; before = {r['path']:r['sha256'] for r in prep['source_inventory_before']}
    backups = {}
    # Todo respaldo y staging quedan completos antes de la primera escritura D:.
    for name in (*p.MODIFIED,'.settings/language.settings.xml'):
        source = confined(SOURCE,name); target = confined(BACKUP,name)
        target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as stream: stream.write(source.read_bytes())
        require(sha(source) == sha(target) == before[name], 'Respaldo/origen cambio: '+name)
        backups[name] = before[name]
    for row in targets:
        target = confined(STAGED,row['relative']); target.parent.mkdir(parents=True,exist_ok=True)
        payload = data if row['source'] is None else Path(row['source']).read_bytes()
        require(hashlib.sha256(payload).hexdigest() == row['sha256'], 'Entrada cambio antes de staging')
        with target.open('xb') as stream: stream.write(payload)
        require(sha(target) == row['sha256'], 'Staging diferente')
    save_new(PLAN,dict(status='STAGED_BEFORE_D_WRITE', targets=targets, backups=backups, context=context,
                       installer_sha256=sha(__file__), hardware_flashed=False, serial_port_opened=False))
    # Control de origen inmediatamente anterior a las mutaciones.
    p.origin(prep)
    try:
        for row in targets:
            name = row['relative']; target = confined(SOURCE,name); staged = confined(STAGED,name)
            require(sha(staged) == row['sha256'], 'Staging cambio durante instalacion: '+name)
            target.parent.mkdir(parents=True,exist_ok=True)
            if row['overwrite']:
                require(sha(target) == before[name], 'Archivo original cambio antes de reemplazo: '+name)
                temporary = confined(SOURCE,name+'.cwru_totalv2_new')
                require(not temporary.exists(), 'Temporal de instalacion ya existe')
                with temporary.open('xb') as stream: stream.write(staged.read_bytes())
                require(sha(temporary) == row['sha256'], 'Temporal diferente')
                # Ambos caminos absolutos ya fueron confinados al proyecto D.
                os.replace(temporary,target)
            else:
                require(not target.exists(), 'Destino nuevo aparecio durante instalacion: '+name)
                with target.open('xb') as stream: stream.write(staged.read_bytes())
            touched.append(name)
            require(sha(target) == row['sha256'], 'Archivo D diferente tras copia: '+name)
        checks = installed_checks(p,prep,targets)
        report = dict(status='PASS_INSTALLED_PENDING_ARM', elf_sha256=context['elf_sha256'],
            configuration=CONFIG, source_project=str(SOURCE.resolve()), targets=targets, context=context,
            backup_sha256=backups, checks=checks, seconds=time.monotonic()-started,
            installer_sha256=sha(__file__), primary_launch=SAFE_LAUNCH,
            changed_files=list(p.MODIFIED), added_files=[r['relative'] for r in targets if not r['overwrite']],
            hardware_flashed=False, serial_port_opened=False, compiled_here=False,
            physical_validation_pending=True, selector_costs_enabled=False,
            rollback='Restaurar los tres archivos modified desde respaldo_instalacion_total; los archivos nuevos se conservan hasta revision. No hay borrado automatico.')
        entries = {path.relative_to(ROOT).as_posix():sha(path)
                   for folder in (BACKUP,STAGED) for path in sorted(folder.rglob('*')) if path.is_file()}
        entries[PLAN.relative_to(ROOT).as_posix()] = sha(PLAN)
        entries[Path(__file__).resolve().relative_to(ROOT).as_posix()] = sha(__file__)
        report_data = (json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf-8')
        entries[REPORT.relative_to(ROOT).as_posix()] = hashlib.sha256(report_data).hexdigest()
        save_new(INSTALL_MANIFEST,entries)
        with INSTALL_MANIFEST.with_suffix('.sha256').open('x',encoding='ascii') as stream:
            stream.write(sha(INSTALL_MANIFEST)+'  '+INSTALL_MANIFEST.name+'\n')
        # El marcador de exito es el ultimo archivo de evidencia comprometido.
        with REPORT.open('xb') as stream: stream.write(report_data)
        print(json.dumps(dict(status=report['status'], **checks, elf_sha256=report['elf_sha256'],
                             primary_launch=SAFE_LAUNCH, hardware_flashed=False, serial_port_opened=False),ensure_ascii=False))
    except BaseException as error:
        if not PARTIAL.exists():
            save_new(PARTIAL,dict(status='PARTIAL_REQUIRES_REVIEW_NO_AUTOMATIC_RETRY', files_completed=touched,
                error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc(),
                backup_sha256=backups, seconds=time.monotonic()-started,
                hardware_flashed=False, serial_port_opened=False, success_report_exists=REPORT.exists()))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check',action='store_true'); mode.add_argument('--install',action='store_true')
    args = parser.parse_args()
    install() if args.install else check()


if __name__ == '__main__':
    main()
