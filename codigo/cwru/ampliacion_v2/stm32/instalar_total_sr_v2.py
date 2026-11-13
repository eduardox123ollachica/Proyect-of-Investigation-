"""Instala solo la variante SR auditada; --check es estrictamente de lectura.

No compila, programa, entrena ni abre COM. Respaldo y staging se completan
antes de escribir D:, y .cproject se activa al final. No borra archivos.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import time
import traceback
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT/'despliegue_stm32/costo_total_single_rounding_v2_180mhz_2026-10-26'
SOURCE = Path('D:/STM32IDE/Prueba_2_CWRU_Extraccion')
NAME = SOURCE.name
CONFIG = 'Release_180MHz_TotalV2_SR'
BASE = 'Release_180MHz_TotalV2'
PREPARER = Path(__file__).with_name('preparar_total_single_rounding_v2.py')
LEGACY = Path(__file__).with_name('instalar_total_v2.py')
LEGACY_SHA = 'c0ca2ad3d8e0e7cdc7c6242b4795c0d1d598ee443bd5cb9c901b72d027b772b6'
MANIFEST = OUT/'MANIFEST_PREPARACION_SR_SHA256.json'
EXPECTED_MANIFEST_SHA = '75d8773bd38aa37b796f3c8008723eaa5f555e1c046db00714779900bfb2c697'
EXPECTED_ELF_SHA = 'c0c3853d2b0e90bb20fe00be13bcf107d186088406be131b729a605ded44ecc7'
BACKUP = OUT/'respaldo_instalacion_sr'
STAGED = OUT/'archivos_instalacion_sr'
REPORT = OUT/'APLICACION_SR_EN_D.json'
PARTIAL = OUT/'APLICACION_SR_EN_D_PARCIAL.json'
PLAN = OUT/'PLAN_INSTALACION_SR.json'
INSTALL_MANIFEST = OUT/'MANIFEST_INSTALACION_SR_SHA256.json'
SAFE_LAUNCH = NAME+'_TotalV2_SR_180MHz_SinRecompilar.launch'
BUILD_KEY = 'org.eclipse.cdt.launch.ATTR_BUILD_BEFORE_LAUNCH_ATTR'
LOAD_KEY = 'com.st.stm32cube.ide.mcu.debug.launch.loadList'


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


def save_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def confined(base, relative):
    relative = Path(relative)
    require(not relative.is_absolute() and relative.parts and '..' not in relative.parts,
            'Ruta relativa invalida: '+str(relative))
    target = (base/relative).resolve()
    require(target != base.resolve() and target.is_relative_to(base.resolve()),
            'Ruta fuera del destino autorizado: '+str(target))
    return target


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec and spec.loader, 'Modulo no importable')
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def canonical(node, ignore_top=False):
    attrs = dict(node.attrib)
    detector = 'com.st.stm32cube.ide.mcu.toolchain.armnone.setup.CrossBuiltinSpecsDetector'
    if node.tag == 'provider' and attrs.get('class') == attrs.get('id') == detector and 'env-hash' in attrs:
        require(re.fullmatch(r'-?[0-9]+', attrs['env-hash']) is not None, 'env-hash no decimal')
        attrs.pop('env-hash')
    if ignore_top and node.tag == 'configuration':
        attrs.pop('id', None)
        attrs.pop('name', None)
    return node.tag, tuple(sorted(attrs.items())), (node.text or '').strip(), tuple(canonical(n) for n in node)


def metadata_append(before_path, after_path, config_id):
    before = ET.parse(before_path).getroot()
    after = ET.parse(after_path).getroot()
    require(before.tag == after.tag == 'project' and before.attrib == after.attrib
            and len(before) == 16 and len(after) == 17, 'CDT no es una adicion16a17')
    old = {n.get('id'): n for n in before}
    new = {n.get('id'): n for n in after}
    require(len(old) == 16 and len(new) == 17 and all(
        i in new and canonical(n) == canonical(new[i]) for i, n in old.items()),
        'Metadatos anteriores cambiaron fuera del env-hash del detector')
    extra = [n for i, n in new.items() if i not in old]
    require(len(extra) == 1 and extra[0].get('id') == config_id and extra[0].get('name') == CONFIG,
            'Adicion CDT distinta de SR')
    template = next(n for n in before if n.get('name') == BASE)
    require(canonical(template, True) == canonical(extra[0], True), 'Proveedores SR distintos del clon TotalV2')


def verify_preparation_seal():
    digest, declared = MANIFEST.with_suffix('.sha256').read_text(encoding='ascii').split()
    require(declared == MANIFEST.name and sha(MANIFEST) == digest == EXPECTED_MANIFEST_SHA,
            'Sello de preparacion distinto del revisado')
    entries = read(MANIFEST)
    require(len(entries) == 1992, 'Numero de recursos de preparacion inesperado')
    exceptions = []
    metadata = OUT/'copia_aislada'/NAME/'.settings/language.settings.xml'
    for name, expected in entries.items():
        target = confined(ROOT, name)
        actual = sha(target)
        if actual != expected:
            require(target == metadata.resolve(), 'Preparacion modificada fuera de CDT copia: '+name)
            backup = OUT/'respaldo_antes_single_rounding/.settings/language.settings.xml'
            require(sha(backup) == expected, 'Metadatos iniciales distintos del sello')
            root = ET.parse(OUT/'archivos_para_D/.cproject').getroot()
            cfg = next(n for n in root.iter('configuration') if n.get('name') == CONFIG)
            metadata_append(backup, target, cfg.get('id'))
            exceptions.append(dict(path=name, sealed_sha256=expected, current_sha256=actual,
                cause='CubeIDE anadio metadatos SR y actualizo solo env-hash del detector',
                prior_sixteen_semantically_preserved=True, original_seal_not_rewritten=True, copied_to_D=False))
    require(len(exceptions) <= 1, 'Excepciones de preparacion inesperadas')
    return exceptions


def validate_launch(data, config_id):
    root = ET.fromstring(data)
    attrs = {n.get('key'): n for n in root}
    require(len(attrs) == len(root), 'Launch con claves duplicadas')
    require(attrs[BUILD_KEY].tag == 'intAttribute' and attrs[BUILD_KEY].get('value') == '0',
            'Launch permitiria recompilar antes de cargar')
    expected = {
        'org.eclipse.cdt.launch.PROJECT_BUILD_CONFIG_AUTO_ATTR': 'false',
        'org.eclipse.cdt.launch.PROJECT_BUILD_CONFIG_ID_ATTR': config_id,
        'org.eclipse.cdt.launch.PROGRAM_NAME': CONFIG+'/'+NAME+'.elf',
        'org.eclipse.cdt.launch.PROJECT_ATTR': NAME,
    }
    require(all(attrs[k].get('value') == v for k, v in expected.items()), 'Launch apunta a otra configuracion/ELF')
    loads = json.loads(attrs[LOAD_KEY].get('value'))
    require(list(loads) == ['fItems'] and len(loads['fItems']) == 1, 'Carga launch inesperada')
    item = loads['fItems'][0]
    require(item['fPath'] == CONFIG+'/'+NAME+'.elf' and item['fProjectName'] == NAME
            and item['fPerformBuild'] is False and item['fDownload'] is True and item['fLoadSymbols'] is True,
            'Launch carga/recompila otro artefacto')
    return dict(build_before_launch_disabled=True, load_item_build_disabled=True,
                program=expected['org.eclipse.cdt.launch.PROGRAM_NAME'], attributes_unchanged=True)


def common_inputs():
    require(ROOT.resolve() == Path('C:/Users/LENOVO/Desktop/Proyecto Investigación').resolve(), 'Workspace inesperado')
    require(SOURCE.resolve().parent == Path('D:/STM32IDE').resolve() and SOURCE.resolve().name == NAME,
            'Proyecto destino fuera de D:/STM32IDE previsto')
    require(all(f.resolve().is_relative_to(OUT.resolve()) for f in (BACKUP, STAGED, REPORT, PLAN)),
            'Evidencia fuera del workspace')
    seal_exceptions = verify_preparation_seal()
    prep = read(OUT/'PREPARACION_SR.json')
    built = read(OUT/'COMPILACION_SR.json')
    audit = read(OUT/'AUDITORIA_ELF_SR.json')
    machine = read(OUT/'AUDITORIA_MAQUINA_SR.json')
    require(built['status'] == 'PASS_SR_BUILD_PENDING_ELF_REVIEW_AND_ARM' and built['returncode'] == 0
            and built['zero_errors_and_warnings'] and not built['timed_out'] and built['integrity_after_build'],
            'Build no aprobado')
    elf_sha = built['configurations'][CONFIG]['elf_sha256']
    require(elf_sha == EXPECTED_ELF_SHA, 'ELF distinto del revisado')
    require(audit['status'] == 'PASS_SR_ELF_CATALOG_PARAMETERS_VERIFIED_PENDING_ARM'
            and audit['elf_sha256'] == elf_sha and audit['build_report_sha256'] == sha(OUT/'COMPILACION_SR.json'),
            'Falta dictamen ELF para este build')
    require(machine['status'] == 'PASS_SR_MACHINE_ROUNDING_REVIEW_PENDING_ARM'
            and machine['elf_sha256'] == elf_sha and machine['build_report_sha256'] == sha(OUT/'COMPILACION_SR.json'),
            'Falta dictamen maquina para este build')
    for config, row in built['configurations'].items():
        require(row['flags_confirmed'] and row['CMSIS_NN_USE_SINGLE_ROUNDING_matches_expected']
                and row['TFLITE_SINGLE_ROUNDING_matches_expected'], 'Banderas incorrectas: '+config)
        if config != CONFIG:
            require(row['control_ELF_identical'] and row['control_Flash_identical'], 'Control cambio: '+config)
    p = module(PREPARER, 'total_sr_preparer_for_install')
    require(p.CONFIG == CONFIG and p.SOURCE.resolve() == SOURCE.resolve()
            and len(prep['previous_ELF_sha256']) == 16 and len(prep['source_inventory_before']) == 1877,
            'Origen o inventario inesperado')
    require(sha(PREPARER) == prep['preparer_sha256']
            and sha(ROOT/'codigo/cwru/ampliacion_v2/costo_total/preparar_total.py') == prep['prior_preparer_sha256']
            and sha(ROOT/'codigo/cwru/ampliacion_v2/analisis/auditar_elf_total_v2.py') == prep['model_reader_sha256'],
            'Scripts de preparacion cambiaron')
    require(p.same_configs(p.prior(), p.PROJECT/'.cproject'), 'Dieciseis configuraciones previas no preservadas')
    require(sha(p.CLIENT/'validar_total_sr.py') == prep['client_sha256'] == sha(p.CLIENT_SOURCE), 'Cliente SR cambio')
    resource_manifest = OUT/'recursos/MANIFEST_RECURSOS_SHA256.json'
    digest, name = resource_manifest.with_suffix('.sha256').read_text(encoding='ascii').split()
    require(name == resource_manifest.name and sha(resource_manifest) == digest == audit['resource_manifest_sha256'],
            'Sello recursos diferente del auditado')
    for relative, expected in read(resource_manifest).items():
        require(sha(confined(OUT/'recursos', relative)) == expected, 'Recurso SR cambio: '+relative)
    targets = []
    require(set(prep['overlay_sha256']) == {'.cproject', str(p.SR_SOURCE).replace('\\', '/'), p.LAUNCH},
            'Overlay inesperado')
    for relative, expected in prep['overlay_sha256'].items():
        require(sha(confined(p.OVERLAY, relative)) == sha(confined(p.PROJECT, relative)) == expected,
                'Overlay/copia cambio: '+relative)
        targets.append(dict(relative=relative, source=str(confined(p.OVERLAY, relative)),
                            sha256=expected, overwrite=(relative == '.cproject')))
    for suffix in ('.elf', '.map'):
        relative = CONFIG+'/'+NAME+suffix
        source = confined(p.PROJECT, relative)
        archived = confined(OUT, 'intentos/'+str(built['attempt'])+'/compilacion/'+relative)
        require(source.is_file() and source.stat().st_size > 0 and sha(source) == sha(archived),
                'Artefacto/archivo diferente: '+relative)
        require(sha(source) == (elf_sha if suffix == '.elf' else machine['map_sha256']), 'Artefacto no auditado')
        targets.append(dict(relative=relative, source=str(source), sha256=sha(source), overwrite=False))
    targets.sort(key=lambda row: (row['relative'] == '.cproject', row['relative']))
    require(len(targets) == 5 and len({r['relative'] for r in targets}) == 5
            and sum(r['overwrite'] for r in targets) == 1 and targets[-1]['relative'] == '.cproject',
            'Plan distinto de cinco archivos, un reemplazo y cuatro nuevos')
    for row in targets:
        confined(SOURCE, row['relative'])
    config_id = p.prior().configs(p.PROJECT/'.cproject')[2][CONFIG].get('id')
    launch_info = validate_launch((p.OVERLAY/SAFE_LAUNCH).read_bytes(), config_id)
    require(sha(LEGACY) == LEGACY_SHA, 'Helper JVM/CDT previo cambio')
    legacy = module(LEGACY, 'total_sr_legacy_cdt')
    cdt = legacy.disable_build_provenance()
    input_paths = [OUT/name for name in ('PREPARACION_SR.json', 'COMPILACION_SR.json', 'AUDITORIA_ELF_SR.json',
        'AUDITORIA_MAQUINA_SR.json', 'MANIFEST_PREPARACION_SR_SHA256.json')]
    input_paths += [PREPARER, LEGACY, ROOT/'codigo/cwru/ampliacion_v2/analisis/auditar_elf_total_sr_v2.py',
                    ROOT/'codigo/cwru/ampliacion_v2/analisis/auditar_maquina_total_sr_v2.py']
    return p, prep, built, targets, dict(elf_sha256=elf_sha, launch=launch_info, cdt=cdt,
        preparation_seal_exceptions=seal_exceptions,
        input_sha256={path.relative_to(ROOT).as_posix(): sha(path) for path in input_paths})


def metadata_preserved(p, prep):
    relative = '.settings/language.settings.xml'
    expected = next(r['sha256'] for r in prep['source_inventory_before'] if r['path'] == relative)
    actual = confined(SOURCE, relative)
    if sha(actual) == expected:
        return dict(status='UNCHANGED', sha256=expected, overwritten_by_installer=False)
    config_id = p.prior().configs(SOURCE/'.cproject')[2][CONFIG].get('id')
    metadata_append(p.BACKUP/relative, actual, config_id)
    return dict(status='CDT_SR_APPEND_PRIOR_SIXTEEN_PRESERVED_EXCEPT_DETECTOR_ENV_HASH',
                sha256=sha(actual), overwritten_by_installer=False)


def installed_checks(p, prep, targets):
    for row in targets:
        require(sha(confined(SOURCE, row['relative'])) == row['sha256'], 'Archivo instalado diferente: '+row['relative'])
    require(p.same_configs(p.prior(), SOURCE/'.cproject') and b'<?fileVersion 4.0.0?>' in (SOURCE/'.cproject').read_bytes(),
            'Configuraciones/PI CDT cambiaron')
    for name, expected in prep['previous_ELF_sha256'].items():
        require(sha(confined(SOURCE, name+'/'+NAME+'.elf')) == expected, 'ELF anterior cambio: '+name)
    for row in prep['source_inventory_before']:
        if row['path'] not in ('.cproject', '.settings/language.settings.xml'):
            require(sha(confined(SOURCE, row['path'])) == row['sha256'], 'Otra fuente original cambio: '+row['path'])
    return dict(installed_files_verified=5, previous_configurations_preserved=16,
                previous_ELF_hashes_preserved=16, previous_other_source_files_preserved=True,
                metadata=metadata_preserved(p, prep))


def check():
    p, prep, built, targets, context = common_inputs()
    require(not PARTIAL.exists(), 'Instalacion parcial registrada; revisar antes de continuar')
    if REPORT.exists():
        report = read(REPORT)
        require(report['status'] == 'PASS_SR_INSTALLED_PENDING_ARM' and report['elf_sha256'] == context['elf_sha256'],
                'Registro de instalacion incompatible')
        digest, declared = INSTALL_MANIFEST.with_suffix('.sha256').read_text(encoding='ascii').split()
        require(declared == INSTALL_MANIFEST.name and sha(INSTALL_MANIFEST) == digest, 'Sello instalacion diferente')
        for name, expected in read(INSTALL_MANIFEST).items():
            require(sha(confined(ROOT, name)) == expected, 'Evidencia instalacion modificada: '+name)
        checks = installed_checks(p, prep, targets)
        print(json.dumps(dict(status='ALREADY_SR_INSTALLED_VERIFIED_NO_REWRITE', **checks), ensure_ascii=False))
        return p, prep, built, targets, context, True
    require(not any(path.exists() for path in (PLAN, BACKUP, STAGED, INSTALL_MANIFEST)),
            'Hay preparacion/instalacion parcial; revisar antes de reintentar')
    p.verify(prep)
    for row in targets:
        target = confined(SOURCE, row['relative'])
        require(target.is_file() if row['overwrite'] else not target.exists(), 'Destino inesperado: '+row['relative'])
    print(json.dumps(dict(status='READY_FOR_SR_INSTALL_NOT_ARM_VALIDATED', targets=5, overwrite=1, added=4,
        original_configurations=16, next_configurations=17, elf_sha256=context['elf_sha256'],
        source_project=str(SOURCE.resolve()), serial_port_opened=False, hardware_flashed=False), ensure_ascii=False))
    return p, prep, built, targets, context, False


def write_targets(targets, before):
    """Copia el staging validado; .cproject debe ser el ultimo elemento."""
    require(targets[-1]['relative'] == '.cproject', '.cproject debe activarse al final')
    touched = []
    for row in targets:
        relative = row['relative']
        target = confined(SOURCE, relative)
        staged = confined(STAGED, relative)
        require(sha(staged) == row['sha256'], 'Staging cambio antes de copiar: '+relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        if row['overwrite']:
            require(relative == '.cproject' and sha(target) == before[relative], 'Original cambio antes de reemplazo')
            temporary = confined(SOURCE, relative+'.cwru_sr_new')
            require(not temporary.exists(), 'Temporal SR ya existe')
            with temporary.open('xb') as stream:
                stream.write(staged.read_bytes())
            require(sha(temporary) == row['sha256'], 'Temporal SR diferente')
            os.replace(temporary, target)
        else:
            require(not target.exists(), 'Destino nuevo aparecio: '+relative)
            with target.open('xb') as stream:
                stream.write(staged.read_bytes())
        touched.append(relative)
        require(sha(target) == row['sha256'], 'Archivo D diferente tras copia: '+relative)
    return touched


def install():
    p, prep, built, targets, context, already = check()
    if already:
        return
    started = time.monotonic()
    before = {r['path']: r['sha256'] for r in prep['source_inventory_before']}
    backups = {}
    # Ninguna escritura D ocurre antes de terminar respaldos, staging y plan.
    for relative in ('.cproject', '.settings/language.settings.xml'):
        source = confined(SOURCE, relative)
        target = confined(BACKUP, relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write(source.read_bytes())
        require(sha(source) == sha(target) == before[relative], 'Respaldo/origen cambio: '+relative)
        backups[relative] = before[relative]
    for row in targets:
        target = confined(STAGED, row['relative'])
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = Path(row['source']).read_bytes()
        require(hashlib.sha256(payload).hexdigest() == row['sha256'], 'Entrada cambio antes de staging')
        with target.open('xb') as stream:
            stream.write(payload)
        require(sha(target) == row['sha256'], 'Staging diferente')
    save_new(PLAN, dict(status='STAGED_BEFORE_D_WRITE', targets=targets, backups=backups, context=context,
                       installer_sha256=sha(__file__), hardware_flashed=False, serial_port_opened=False))
    p.origin(prep)
    try:
        touched = write_targets(targets, before)
        checks = installed_checks(p, prep, targets)
        report = dict(status='PASS_SR_INSTALLED_PENDING_ARM', elf_sha256=context['elf_sha256'], configuration=CONFIG,
            source_project=str(SOURCE.resolve()), targets=targets, context=context, backup_sha256=backups,
            checks=checks, seconds=time.monotonic()-started, installer_sha256=sha(__file__), primary_launch=SAFE_LAUNCH,
            changed_files=['.cproject'], added_files=[r['relative'] for r in targets if not r['overwrite']],
            hardware_flashed=False, serial_port_opened=False, compiled_here=False,
            physical_validation_pending=True, selector_costs_enabled=False,
            rollback='Restaurar solo .cproject desde respaldo_instalacion_sr. Los cuatro archivos nuevos se conservan hasta revision. Sin borrado automatico.')
        entries = {path.relative_to(ROOT).as_posix(): sha(path)
                   for folder in (BACKUP, STAGED) for path in sorted(folder.rglob('*')) if path.is_file()}
        entries[PLAN.relative_to(ROOT).as_posix()] = sha(PLAN)
        entries[Path(__file__).resolve().relative_to(ROOT).as_posix()] = sha(__file__)
        report_data = (json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode('utf-8')
        entries[REPORT.relative_to(ROOT).as_posix()] = hashlib.sha256(report_data).hexdigest()
        save_new(INSTALL_MANIFEST, entries)
        with INSTALL_MANIFEST.with_suffix('.sha256').open('x', encoding='ascii') as stream:
            stream.write(sha(INSTALL_MANIFEST)+'  '+INSTALL_MANIFEST.name+'\n')
        # Marcador de exito: se compromete despues del manifiesto.
        with REPORT.open('xb') as stream:
            stream.write(report_data)
        print(json.dumps(dict(status=report['status'], **checks, elf_sha256=report['elf_sha256'],
            primary_launch=SAFE_LAUNCH, hardware_flashed=False, serial_port_opened=False), ensure_ascii=False))
    except BaseException as error:
        if not PARTIAL.exists():
            completed = [r['relative'] for r in targets if confined(SOURCE, r['relative']).is_file()
                         and sha(confined(SOURCE, r['relative'])) == r['sha256']]
            save_new(PARTIAL, dict(status='PARTIAL_REQUIRES_REVIEW_NO_AUTOMATIC_RETRY', files_completed=completed,
                error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc(),
                backup_sha256=backups, seconds=time.monotonic()-started, hardware_flashed=False,
                serial_port_opened=False, success_report_exists=REPORT.exists()))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check', action='store_true')
    mode.add_argument('--install', action='store_true')
    args = parser.parse_args()
    install() if args.install else check()


if __name__ == '__main__':
    main()
