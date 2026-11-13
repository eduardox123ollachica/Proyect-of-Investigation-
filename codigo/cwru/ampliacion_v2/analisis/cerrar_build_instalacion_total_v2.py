"""Cierra build e instalacion TotalV2 auditados, sin compilar o usar la placa.

Conserva el sello de preparacion. El nuevo manifiesto registra la unica
mutacion generada por CubeIDE y agrega las evidencias posteriores.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'despliegue_stm32/costo_total_representantes_v2_180mhz_2026-10-25'
METHOD=ROOT/'articulo/ampliacion_cwru/metodologia/calibracion_costo_total_v2_2026-10-25/VALIDACION_BUILD_TOTAL_V2.md'
CLOSURE=OUT/'CIERRE_BUILD_INSTALACION_TOTAL.json'
MANIFEST=OUT/'MANIFEST_BUILD_INSTALACION_TOTAL_SHA256.json'
SEAL=MANIFEST.with_suffix('.sha256')
OLD_MANIFEST=OUT/'MANIFEST_PREPARACION_TOTAL_SHA256.json'
INSTALLER=ROOT/'codigo/cwru/ampliacion_v2/stm32/instalar_total_v2.py'
ELF_SHA='85e8d7e2d1848833117fd4704493710cda0058fecedef3b379a80e9560171b1c'


def require(ok,message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path,value):
    with Path(path).open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n')


def verify_manifest(path):
    digest,name=path.with_suffix('.sha256').read_text(encoding='ascii').split()
    require(name==path.name and digest==sha(path),'Sello invalido: '+str(path))
    entries=read(path)
    require(isinstance(entries,dict) and entries,'Manifiesto vacio')
    for relative,expected in entries.items():
        target=(ROOT/relative).resolve()
        require(target.is_relative_to(ROOT.resolve()) and target.is_file() and sha(target)==expected,
                'Evidencia modificada: '+relative)
    return entries


def load_installer():
    spec=importlib.util.spec_from_file_location('total_installer_closure_readonly',INSTALLER)
    require(spec and spec.loader,'Instalador no importable')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def excluded(path):
    parts=path.relative_to(OUT).parts
    return (parts[0] in ('workspace_aislado','resultados','logs')
            or parts[:2]==('Validacion_PC','resultados')
            or '__pycache__' in parts or path.suffix in ('.pyc','.pyo')
            or path in (CLOSURE,MANIFEST,SEAL))


def main():
    started=time.perf_counter()
    require(not any(p.exists() for p in (CLOSURE,MANIFEST,SEAL)),'No sobrescribir cierre existente')
    require(METHOD.is_file() and (OUT/'PRIMERA_PRUEBA_TOTAL_V2.md').is_file(),'Documentacion final pendiente')
    application=read(OUT/'APLICACION_TOTAL_EN_D.json')
    require(application['status']=='PASS_INSTALLED_PENDING_ARM' and application['elf_sha256']==ELF_SHA,
            'Instalacion no aprobada')
    require(application['hardware_flashed'] is False and application['serial_port_opened'] is False
            and application['compiled_here'] is False and application['selector_costs_enabled'] is False
            and application['physical_validation_pending'] is True,'Instalacion con alcance fisico inesperado')
    require(sha(INSTALLER)==application['installer_sha256'],'Instalador cambio desde la aplicacion')
    verify_manifest(OUT/'MANIFEST_INSTALACION_TOTAL_SHA256.json')
    print('Verificando instalacion real, quince configuraciones/ELF anteriores y fuentes; solo lectura.',flush=True)
    installer=load_installer()
    preparer,prep,build,targets,launch_data,context=installer.common_inputs()
    checks=installer.installed_checks(preparer,prep,targets)
    require(checks['installed_files_verified']==15 and checks['previous_configurations_preserved']==15
            and checks['previous_ELF_hashes_preserved']==15 and checks['previous_other_source_files_preserved'] is True,
            'Comprobacion posterior de instalacion no aprobada')
    require(len(preparer.configs(installer.SOURCE/'.cproject')[2])==16,'Configuraciones instaladas distintas de16')
    launch=context['launch']
    require(launch['stop_at_main_unchanged'] and launch['other_launch_attributes_unchanged']
            and len(launch['changes'])==2,'Launch derivado cambio mas de dos opciones')
    changes={item['key']:(item['before'],item['after']) for item in launch['changes']}
    require(changes=={
        'org.eclipse.cdt.launch.ATTR_BUILD_BEFORE_LAUNCH_ATTR':(2,0),
        'com.st.stm32cube.ide.mcu.debug.launch.loadList.fItems[0].fPerformBuild':(True,False)},
        'Opciones derivadas de launch distintas')
    require(hashlib.sha256(launch_data).hexdigest()==launch['derived_sha256'], 'Bytes launch no corresponden')
    require(build['status']=='PASS_PENDING_ELF_MANUAL_REVIEW_AND_ARM' and build['zero_errors_warnings']
            and build['returncode']==0 and build['timed_out'] is False,'Build no aprobado')
    audit=read(OUT/'AUDITORIA_ELF_TOTAL.json')
    machine=read(OUT/'AUDITORIA_MAQUINA_TOTAL.json')
    require(audit['status']=='PASS_ELF_CATALOG_PARAMETERS_VERIFIED_PENDING_ARM'
            and machine['status']=='PASS_MACHINE_CODE_REVIEW_PENDING_ARM'
            and audit['elf_sha256']==machine['elf_sha256']==ELF_SHA,'Auditorias o ELF distintos')
    require(audit['model_count']==32 and len(audit['models'])==32 and audit['model_payload_bytes']==113888
            and all(r['all_bytes_exact_exported_model'] and r['all_parameters_exact_exported_catalog']
                    for r in audit['models']),'Modelos o parametros no auditados')
    require(machine['primary']['read_count']==2 and machine['primary']['one_specialized_extraction']
            and machine['DNN_predict']['no_DWT_literal_or_read_in_wrapper']
            and machine['DNN_predict']['Invoke_calls']==1 and machine['profile']['included_in_primary_selector_cost'] is False,
            'Envolvente DWT o perfil no auditados')

    # Verificar la cadena antigua completa; no reemplazar su manifiesto.
    expected_old_manifest_sha=application['context']['input_sha256']['MANIFEST_PREPARACION_TOTAL_SHA256.json']
    require(sha(OLD_MANIFEST)==expected_old_manifest_sha,'Manifiesto de preparacion reescrito')
    digest,name=OLD_MANIFEST.with_suffix('.sha256').read_text(encoding='ascii').split()
    require(name==OLD_MANIFEST.name and digest==expected_old_manifest_sha,'Sello anterior no coincide')
    old=read(OLD_MANIFEST)
    require(len(old)==1968,'Cadena anterior distinta de1968 recursos')
    allowed=audit['preparation_resource_mutation']
    differences=[]
    entries={}
    for relative,expected in old.items():
        target=(ROOT/relative).resolve()
        require(target.is_relative_to(ROOT.resolve()) and target.is_file(),'Recurso anterior ausente')
        current=sha(target)
        if current!=expected:
            differences.append(dict(path=relative,before_sha256=expected,after_sha256=current))
        entries[relative]=current
    require(differences==[dict(path=allowed['path'],before_sha256=allowed['before_sha256'],
                              after_sha256=allowed['after_sha256'])], 'Hay mutaciones no autorizadas del sello anterior')
    require(allowed['previous_fifteen_configurations_semantically_equal_ignoring_env_hash']
            and allowed['new_configuration_providers_equal_Masks_except_id_name_env_hash']
            and allowed['previous_manifest_preserved'], 'Excepcion de metadatos no justificada')
    require(context['preparation_seal_exceptions'][0]['path']==allowed['path']
            and context['preparation_seal_exceptions'][0]['current_sha256']==allowed['after_sha256'],
            'Instalacion y auditoria discrepan en excepcion')

    folders=(OUT,ROOT/'codigo/cwru/ampliacion_v2/costo_total')
    added_paths=set()
    for folder in folders:
        for path in folder.rglob('*'):
            if not path.is_file() or '__pycache__' in path.parts or path.suffix in ('.pyc','.pyo'):
                continue
            if path.is_relative_to(OUT) and excluded(path):
                continue
            added_paths.add(path.resolve())
    added_paths.update({Path(__file__).resolve(),INSTALLER,METHOD,
        ROOT/'codigo/cwru/ampliacion_v2/analisis/auditar_elf_total_v2.py',
        ROOT/'codigo/cwru/ampliacion_v2/analisis/cerrar_preparacion_total_v2.py'})
    for path in sorted(added_paths):
        require(path.resolve().is_relative_to(ROOT.resolve()) and path.is_file(),'Entrada externa/no valida')
        relative=path.relative_to(ROOT).as_posix()
        current=sha(path)
        require(relative not in entries or entries[relative]==current,'Entrada cambio durante cierre')
        entries[relative]=current
    report=dict(
        schema_version=1,status='BUILD_AUDITED_INSTALLED_PENDING_PHYSICAL_GATE',
        date_local='2026-10-25',timezone='America/Lima',configuration=preparer.CONFIG,
        elf_sha256=ELF_SHA,original_preparation_manifest_sha256=expected_old_manifest_sha,
        original_preparation_manifest_preserved=True,original_preparation_entries=1968,
        inherited_entries=1968,inherited_entry_changes=differences,
        metadata_exception=allowed,metadata_copied_to_D=False,
        build=dict(attempt=build['attempt'],returncode=0,seconds=build['seconds'],zero_errors_warnings=True,
                   configurations=list(build['configurations']),controls_ELF_and_FLASH_identical=True),
        models=dict(count=32,payload_bytes=113888,all_bytes_SHA256_FNV_and_parameters_verified=True),
        memory=audit['memory'],machine_review=dict(primary_DWT_reads=2,PROFILE_separate=True,
            one_extraction=True,one_input_transform=True,one_DNN_Invoke=True,one_normality_score=True),
        installation=dict(installed_files=15,changed_existing_files=application['changed_files'],
                          added_files=application['added_files'],old_configurations_preserved=15,
                          resulting_configurations=16,old_ELF_hashes_preserved=15,
                          other_original_sources_preserved=True,live_checks=checks,
                          primary_launch=application['primary_launch'],launch_changes=launch['changes'],
                          rollback=application['rollback']),
        new_training_fits_executed=0,new_builds_by_closure=0,new_TFLite_invocations=0,
        serial_port_opened=False,hardware_flashed=False,new_physical_RUN=0,
        selector_costs_enabled=False,physical_validation_pending=True,
        gate=dict(representative_id=0,global_window_id=568,primary_RUN=3,PROFILE=1,BASE=1,
                  negative_rejections=17,complete_campaign_authorized_by_this_closure=False),
        manifest_entries_including_this_closure=len(entries)+1,
        manifest_scope='Root-relative project files only; chained preparation with one documented CubeIDE metadata hash change; build products, archives, audits, installation backup/staging and guides.',
        excluded=['workspace_aislado','Validacion_PC/resultados','resultados','logs','__pycache__','*.pyc','*.pyo'],
        check_seconds_before_writing=time.perf_counter()-started,
        limitations=['Installation is not proof of programmed firmware identity on the physical board.',
                     'AllocateTensors, actual arena use, numerical equivalence and times remain pending physical gate.',
                     'Memory belongs to32-model test harness; final one-model memory must be measured separately.',
                     'Stack reservation and compiler frames are not a runtime maximum stack measurement.'])
    write(CLOSURE,report)
    entries[CLOSURE.relative_to(ROOT).as_posix()]=sha(CLOSURE)
    write(MANIFEST,dict(sorted(entries.items())))
    with SEAL.open('x',encoding='ascii') as stream:
        stream.write(sha(MANIFEST)+'  '+MANIFEST.name+'\n')
    verify_manifest(MANIFEST)
    require(sha(OLD_MANIFEST)==expected_old_manifest_sha,'Sello anterior cambio durante cierre')
    print(json.dumps(dict(status=report['status'],resources=len(entries),manifest_sha256=sha(MANIFEST),
                         closure_sha256=sha(CLOSURE),seconds=time.perf_counter()-started,
                         installed_checks=checks),ensure_ascii=False))


if __name__=='__main__':
    main()
