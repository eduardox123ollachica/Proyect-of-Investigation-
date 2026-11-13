"""Cierra build e instalacion SR con evidencias existentes; no repite ensayos.

Verifica D en lectura mediante el instalador. Conserva el sello previo y
registra la unica regeneracion CDT auditada de la copia de compilacion.
No compila, entrena, invoca TFLite, abre COM ni programa hardware.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT/'despliegue_stm32/costo_total_single_rounding_v2_180mhz_2026-10-26'
METHOD = ROOT/'articulo/ampliacion_cwru/metodologia/calibracion_costo_total_v2_2026-10-25/VALIDACION_BUILD_SR_2026-10-26.md'
INSTALLER = ROOT/'codigo/cwru/ampliacion_v2/stm32/instalar_total_sr_v2.py'
REPORT = OUT/'CIERRE_BUILD_INSTALACION_SR.json'
MANIFEST = OUT/'MANIFEST_BUILD_INSTALACION_SR_SHA256.json'
SEAL = MANIFEST.with_suffix('.sha256')
CONFIG = 'Release_180MHz_TotalV2_SR'
ELF_SHA = 'c0c3853d2b0e90bb20fe00be13bcf107d186088406be131b729a605ded44ecc7'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, 'Modulo no importable')
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def main():
    started = time.perf_counter()
    require(not any(p.exists() for p in (REPORT, MANIFEST, SEAL)), 'No sobrescribir cierre')
    installer = module(INSTALLER, 'sr_installer_closure_readonly')
    _, prep, build, targets, context, installed = installer.check()
    require(installed, 'Falta instalacion aprobada')
    elf = read(OUT/'AUDITORIA_ELF_SR.json')
    machine = read(OUT/'AUDITORIA_MAQUINA_SR.json')
    installed_report = read(OUT/'APLICACION_SR_EN_D.json')
    require(all(row['elf_sha256'] == ELF_SHA for row in (elf, machine, installed_report)),
            'Los dictamenes no corresponden al mismo ELF')
    require(elf['all_model_bytes_and_parameters_exact_to_original_TotalV2']
            and elf['model_count'] == 32 and elf['one_reused_arena']
            and elf['original_archive_unchanged'], 'Modelos/parametros/biblioteca distintos')
    require(len(targets) == 5 and sum(row['overwrite'] for row in targets) == 1,
            'Instalacion fuera del alcance previsto')
    require(installed_report['changed_files'] == ['.cproject']
            and len(installed_report['added_files']) == 4
            and not installed_report['hardware_flashed']
            and not installed_report['serial_port_opened']
            and not installed_report['selector_costs_enabled'], 'Acciones no previstas')
    checks = installed_report['checks']
    require(checks['installed_files_verified'] == 5
            and checks['previous_configurations_preserved'] == 16
            and checks['previous_ELF_hashes_preserved'] == 16
            and checks['previous_other_source_files_preserved'], 'Preservacion incompleta')
    require(len(context['preparation_seal_exceptions']) == 1,
            'Se esperaba solo la regeneracion CDT auditada de la copia')
    mutation = context['preparation_seal_exceptions'][0]
    require(mutation['path'] == elf['preparation_resource_mutation']['path']
            and mutation['current_sha256'] == elf['preparation_resource_mutation']['after_sha256'],
            'Mutacion CDT distinta de la auditada')
    for name in ('auditar_elf_total_sr_v2.py', 'auditar_maquina_total_sr_v2.py'):
        row = elf if name.startswith('auditar_elf') else machine
        key = 'audit_script_sha256' if name.startswith('auditar_elf') else 'script_sha256'
        require(row[key] == sha(Path(__file__).with_name(name)),
                'Auditoria no corresponde a su script actual: '+name)
    files = {
        Path(__file__).resolve(), INSTALLER, METHOD, OUT/'PRUEBA_REGRESION_SR.md',
        OUT/'COMPILACION_SR.json', OUT/'COMPILACION_SR_INTENTO_1.json',
        OUT/'CUBEIDE_SR_INTENTO_1.log', OUT/'SIMBOLOS_SR_INTENTO_1.txt',
        OUT/'AUDITORIA_ELF_SR.json', OUT/'AUDITORIA_MAQUINA_SR.json',
        OUT/'COMPROBACION_PREVIA_INSTALACION_SR.json',
        OUT/'MANIFEST_PREPARACION_SR_SHA256.json', OUT/'MANIFEST_PREPARACION_SR_SHA256.sha256',
        OUT/'MANIFEST_INSTALACION_SR_SHA256.json', OUT/'MANIFEST_INSTALACION_SR_SHA256.sha256',
        OUT/'copia_aislada/Prueba_2_CWRU_Extraccion/.settings/language.settings.xml',
        Path(__file__).with_name('auditar_elf_total_sr_v2.py'),
        Path(__file__).with_name('auditar_maquina_total_sr_v2.py'),
    }
    files.update(ROOT/name for name in read(OUT/'MANIFEST_INSTALACION_SR_SHA256.json'))
    files.update(p for p in (OUT/'intentos/1/compilacion').rglob('*') if p.is_file())
    for name in ('PRUEBAS_INSTALADOR_SR.json',):
        if (OUT/name).is_file():
            tests = read(OUT/name)
            require(tests['status'] == 'PASS_SR_INSTALLER_GUARD_TESTS'
                    and tests['tests_run'] == 5 and tests['failures'] == tests['errors'] == 0
                    and tests['installer_script_sha256'] == sha(INSTALLER)
                    and tests['test_script_sha256'] == sha(INSTALLER.with_name('test_instalar_total_sr_v2.py')),
                    'Pruebas del instalador no aprobadas o de otro codigo')
            files.add(OUT/name)
    test_source = INSTALLER.with_name('test_instalar_total_sr_v2.py')
    if test_source.is_file():
        files.add(test_source)
    for path in files:
        require(path.resolve().is_relative_to(ROOT.resolve()) and path.is_file(),
                'Recurso ausente o externo: '+str(path))
    entries = {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(files)}
    result = dict(
        status='PASS_SR_BUILD_AUDITED_AND_INSTALLED_PENDING_PHYSICAL_GATE',
        date_local='2026-10-26', timezone='America/Lima', configuration=CONFIG,
        elf_sha256=ELF_SHA, build_seconds=build['seconds'], zero_errors_and_warnings=True,
        control_builds_EntM_and_TotalV2_unchanged=True,
        model_count=32, models_and_parameters_unchanged=True,
        source_inventory_before=len(prep['source_inventory_before']),
        old_configurations_and_ELF_preserved=16, current_configurations=17,
        installed_files=5, overwritten_files=['.cproject'], new_files=installed_report['added_files'],
        metadata_in_D_overwritten=False, original_preparation_seal_preserved=True,
        preparation_seal_exception=mutation, memory=elf['memory'],
        causal_diagnosis_preserved=True, rounding_verified_in_ELF=True,
        whole_precompiled_archive_rebuilt=False, whole_precompiled_archive_rounding_changed=False,
        static_audit_status=elf['status'], machine_audit_status=machine['status'],
        installation_status=installed_report['status'],
        physical_gate_plan=dict(cases=[[568, [0]], [232, [22]]], primary_RUN=6,
            PROFILE=2, BASE=2, negative_rejections=17,
            required_status='SR_GATE_PASS_NOT_TOTAL_COST_VALIDATED',
            comparison='exact_INT8_for_the_same_input', max_seconds_argument=600),
        gate_physically_passed=False, full_campaign_passed=False,
        old_partial_timings_imported=False, selector_costs_enabled=False,
        closure_new_training_fits=0, closure_new_TFLite_invocations=0,
        closure_new_builds=0, closure_serial_opened=False, closure_hardware_flashed=False,
        manuscript_modified=False, manifest_entries_including_closure=len(entries)+1,
        seconds_before_write=time.perf_counter()-started,
        limitations=[
            'Static instruction review does not replace physical equivalence tests.',
            'Board full-image identity has not been verified by programmer log or readback here.',
            'Memory is the 32-model benchmark; runtime arena and stack must be checked separately.',
            'Timing and selection require the complete audited campaign on one firmware.',
        ],
    )
    save(REPORT, result)
    entries[REPORT.relative_to(ROOT).as_posix()] = sha(REPORT)
    save(MANIFEST, dict(sorted(entries.items())))
    with SEAL.open('x', encoding='ascii') as stream:
        stream.write(sha(MANIFEST)+'  '+MANIFEST.name+'\n')
    for relative, expected in entries.items():
        require(sha(ROOT/relative) == expected, 'Cambio despues de sellar: '+relative)
    print(json.dumps(dict(status=result['status'], files=len(entries),
        manifest_sha256=sha(MANIFEST), seconds=time.perf_counter()-started), ensure_ascii=True))


if __name__ == '__main__':
    main()
