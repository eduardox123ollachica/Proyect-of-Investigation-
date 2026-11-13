"""Sella preparacion SR usando evidencias existentes; no repite pruebas.

Crea cierre/manifiesto nuevos. Verifica D en solo lectura, fuentes copiadas,
cliente, recursos, diagnostico causal y comprobaciones de sintaxis guardadas.
"""
from __future__ import annotations
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'despliegue_stm32/costo_total_single_rounding_v2_180mhz_2026-10-26'
OLD=ROOT/'despliegue_stm32/costo_total_representantes_v2_180mhz_2026-10-25'
FAILED=OLD/'Validacion_PC/resultados/20261026T042424_323718Z'
PREPARER=ROOT/'codigo/cwru/ampliacion_v2/stm32/preparar_total_single_rounding_v2.py'
PREP_TEST=PREPARER.with_name('test_preparar_total_single_rounding_v2.py')
CLIENT_CODE=ROOT/'codigo/cwru/ampliacion_v2/costo_total_single_rounding'
METHOD=ROOT/'articulo/ampliacion_cwru/metodologia/calibracion_costo_total_v2_2026-10-25/CORRECCION_REDONDEO_2026-10-26.md'
CLOSURE=OUT/'CIERRE_PREPARACION_SR.json'
MANIFEST=OUT/'MANIFEST_PREPARACION_SR_SHA256.json'
SEAL=MANIFEST.with_suffix('.sha256')
EXPECTED_LIBRARY='84cd3fae745ce78530e49f353725f845588ca0298cc35bd01da98403eabbe974'
EXPECTED_MATVEC='316a9b62c9c79a0a390149fa75d704b51ec392bc19751c443dcd73750e809ad6'

def require(ok,message):
    if not ok:raise ValueError(message)

def sha(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):result.update(block)
    return result.hexdigest()

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def write(path,value):
    with Path(path).open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')

def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    require(spec is not None and spec.loader is not None,'Modulo no importable')
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value

def sealed_entries(path,base):
    digest,name=path.with_suffix('.sha256').read_text(encoding='ascii').split()
    require(name==path.name and digest==sha(path),'Sello diferente: '+str(path))
    entries=read(path);require(isinstance(entries,dict) and entries,'Manifiesto vacio')
    resolved_base=base.resolve()
    for relative,expected in entries.items():
        target=(base/relative).resolve()
        require(target.is_relative_to(resolved_base) and target.is_file() and sha(target)==expected,
                'Recurso diferente: '+relative)
    return entries

def test_count(path):
    return sum(isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name.startswith('test_')
               for n in ast.walk(ast.parse(Path(path).read_text(encoding='utf-8-sig'))))

def same_protected_client_routines():
    original=ROOT/'codigo/cwru/ampliacion_v2/costo_total/validar_total.py'
    require(sha(original)=='6c88ad0b09226673de3ddc675336153c41ad5ae5a47e7e412a1194325fe5b6d0','Cliente anterior cambio')
    def definitions(path):
        return {n.name:n for n in ast.parse(path.read_text(encoding='utf-8-sig')).body
                if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    before,after=definitions(original),definitions(CLIENT_CODE/'validar_total_sr.py')
    changed={name for name,node in before.items() if ast.dump(node,include_attributes=False)
             !=ast.dump(after[name],include_attributes=False)}
    require(changed=={'load_resources','schedule','verify_gate','run','main'},'Se modifico otra rutina del cliente')
    protected=('params_bytes','replay','LiteModels','Link','parse_info','validate_model','parse_selected',
        'frames','transmit','baseline','parse_result','get_pages','compare','negatives','statistics')
    require(all(name not in changed for name in protected),'Comparacion o numerica del cliente modificada')
    return list(protected)

def included(path,prep):
    require(path.resolve().is_relative_to(ROOT.resolve()),'Archivo fuera del proyecto')
    if '__pycache__' in path.parts or path.suffix in ('.pyc','.pyo'):return False
    if path in (CLOSURE,MANIFEST,SEAL):return False
    if path.is_relative_to(OUT):
        parts=path.relative_to(OUT).parts
        if parts[0] in ('workspace_aislado','intentos','resultados','logs'):return False
        if parts[:2]==('Validacion_PC','resultados'):return False
        if path.name.startswith(('CUBEIDE_SR_INTENTO_','COMPILACION_SR')):return False
        project=Path(prep['project'])
        if path.is_relative_to(project):
            first=path.relative_to(project).parts[0]
            if first in set(prep['previous_ELF_sha256'])|{prep['configuration'],'.git','.metadata'}:return False
    return True

def main():
    start=time.perf_counter()
    require(not any(p.exists() for p in (CLOSURE,MANIFEST,SEAL)),'No sobrescribir cierre o sello existente')
    require(not list(OUT.glob('COMPILACION_SR*.json')) and not list(OUT.glob('CUBEIDE_SR_INTENTO_*.log')),
            'Este cierre debe anteceder al primer build SR')
    prep=read(OUT/'PREPARACION_SR.json');p=module(PREPARER,'sr_preparer_closure_readonly')
    require(prep['status']=='PREPARED_SR_NOT_BUILT_NOT_ARM_VALIDATED'
        and prep['configuration']=='Release_180MHz_TotalV2_SR'
        and prep['old_configurations']==16 and prep['new_configurations']==17,'Preparacion inesperada')
    require(len(prep['source_inventory_before'])==1877 and len(prep['previous_ELF_sha256'])==16,
            'Inventario o respaldos ELF diferentes')
    require(prep['builds_executed']==prep['training_fits']==0 and all(prep[k] is False for k in
        ('serial_port_opened','hardware_flashed','D_files_modified','selector_costs_enabled')),'Acciones no previstas')
    print('Verificando1877fuentes,16ELF originales/copiados y recursos; no build, TF o COM.',flush=True)
    prior=p.verify(prep)
    require(len(prior.configs(p.SOURCE/'.cproject')[2])==16 and len(prior.configs(p.PROJECT/'.cproject')[2])==17,
            'Configuraciones origen/copia no son16/17')
    replacement=prep['replacement_source']
    require(replacement['byte_exact_original_copy'] and replacement['original_sha256']==replacement['prepared_sha256']==EXPECTED_MATVEC
        and sha(p.PROJECT/replacement['prepared'])==sha(Path(replacement['original']))==EXPECTED_MATVEC,
        'Fuente matvec no es copia exacta')
    require(prep['original_library_sha256']==EXPECTED_LIBRARY and prep['original_archive_modified'] is False
        and prep['whole_precompiled_archive_single_rounding'] is False
        and sha(p.SOURCE/'Lib/libtensorflow-microlite.a')==sha(p.PROJECT/'Lib/libtensorflow-microlite.a')==EXPECTED_LIBRARY,
        'Biblioteca original modificada')
    scope=prep['parameter_scope']
    require(scope['channels']==1056 and len(scope['models'])==32
        and scope['parameter_generation_unchanged_for_observed_shifts']
        and scope['all_FC_and_Softmax_multiplier_shifts_below30'] and not scope['whole_TFLM_archive_rebuilt'],
        'Dominio de parametros no comprobado')
    require(all(row['channels']==33 and row['all_FC_per_channel'] and row['minimum_shift']>=-31
                and row['maximum_shift']<=30 and row['softmax_multiplier_shift']<30 for row in scope['models']),
            'Modelo o shift fuera del alcance previsto')
    syntax=read(OUT/'SINTAXIS_SR.json')
    require(syntax['status']=='PASS_SR_SYNTAX_MACROS_NOT_BUILT' and syntax['syntax_returncode']==0
        and syntax['preprocess_returncode']==0 and syntax['syntax_output']==syntax['preprocess_errors']==''
        and syntax['cpp_syntax']['returncode']==0 and syntax['cpp_syntax']['output']==''
        and syntax['macros_confirmed'] and syntax['source_sha256']==EXPECTED_MATVEC,
        'Sintaxis C/C++ o macros no aprobadas')
    require(set(syntax['required_macro_lines'])=={'#define CMSIS_NN_USE_SINGLE_ROUNDING 1',
        '#define TFLITE_SINGLE_ROUNDING 1','#define ARM_MATH_DSP 1'},'Macros comprobadas diferentes')
    client_tests=read(OUT/'PRUEBAS_CLIENTE_SR.json');prep_tests=read(OUT/'PRUEBAS_PREPARACION_SR_REVISADA.json')
    require(client_tests['status']=='PASS' and client_tests['returncode']==0 and client_tests['tests']==30
        and client_tests['simulation_only'] is True and client_tests['real_TFLite_invocations']==0
        and client_tests['serial_port_opened'] is False and 'Ran 30 tests' in client_tests['output']
        and client_tests['output'].rstrip().endswith('OK'),'Pruebas cliente no aprobadas')
    require(client_tests['client_sha256']==sha(CLIENT_CODE/'validar_total_sr.py')==prep['client_sha256']
        and client_tests['test_sha256']==sha(CLIENT_CODE/'test_validar_total_sr.py')
        and test_count(CLIENT_CODE/'test_validar_total_sr.py')==30,'Fuentes cliente/pruebas no corresponden al informe')
    require(prep_tests['status']=='PASS' and prep_tests['tests']==9 and prep_tests['errors']==prep_tests['failures']==0
        and prep_tests['preparer_sha256']==sha(PREPARER)==prep['preparer_sha256']
        and prep_tests['test_source_sha256']==sha(PREP_TEST) and test_count(PREP_TEST)==9,
        'Pruebas preparador final no corresponden al codigo actual')
    protected=same_protected_client_routines()
    resources=sealed_entries(OUT/'recursos/MANIFEST_RECURSOS_SHA256.json',OUT/'recursos')
    require(len(resources)==39 and resources==read(OLD/'recursos/MANIFEST_RECURSOS_SHA256.json'),
            'Recursos/modelos alterados')
    require(all(sha(OUT/'recursos'/name)==sha(OLD/'recursos'/name) for name in resources),'Copia de recursos diferente')
    diag_manifest=Path(prep['diagnosis_manifest_source']);diagnostic=sealed_entries(diag_manifest,ROOT)
    require(len(diagnostic)==15,'Se esperan15recursos de diagnostico')
    failed=sealed_entries(FAILED/'MANIFEST_EVIDENCIAS_SHA256.json',FAILED)
    require(len(failed)==12 and read(FAILED/'RESULTADO_TOTAL.json')['status']=='FAILED_INCOMPLETE',
            'Captura parcial original no preservada')
    diagnosis=read(OUT/'DIAGNOSTICO_ORIGEN.json')
    require(diagnosis['status']=='CAUSE_ROUNDING_MISMATCH_CONFIRMED_PENDING_ARM_FIX'
        and diagnosis['single_PC_layers_exact'] and diagnosis['double_softmax_matches_ARM']
        and diagnosis['single_softmax_matches_PC'] and not diagnosis['correction_physically_verified']
        and not diagnosis['acceptance_tolerance_relaxed'] and not diagnosis['old_partial_costs_promoted'],
        'El diagnostico no documenta una causa confirmada con correccion pendiente')
    for path in (METHOD,OUT/'LEEME_CORRECCION_SR.md',OUT/'NOTA_CAPTURA_PRUEBAS_CLIENTE_SR.md'):
        require(path.is_file(),'Falta documento final: '+str(path))
    paths={path for path in OUT.rglob('*') if path.is_file() and included(path,prep)}
    paths.update(path for path in CLIENT_CODE.rglob('*') if path.is_file() and included(path,prep))
    paths.update({PREPARER,PREP_TEST,Path(__file__).resolve(),METHOD,diag_manifest,diag_manifest.with_suffix('.sha256')})
    paths.update(ROOT/name for name in diagnostic)
    paths.update(FAILED/name for name in failed)
    paths.update({FAILED/'MANIFEST_EVIDENCIAS_SHA256.json',FAILED/'MANIFEST_EVIDENCIAS_SHA256.sha256'})
    case_manifest=diag_manifest.parent/'MANIFEST_CASO_FALLO_SHA256.json'
    case_entries=sealed_entries(case_manifest,ROOT)
    paths.update(ROOT/name for name in case_entries)
    entries={path.relative_to(ROOT).as_posix():sha(path) for path in sorted(paths)}
    report=dict(status='PREPARED_SR_VERIFIED_PENDING_USER_BUILD',date_local='2026-10-26',timezone='America/Lima',
        configuration=prep['configuration'],source_inventory_verified=1877,original_configurations_verified=16,
        original_ELF_hashes_verified=16,isolated_configurations=17,tests=dict(client=30,preparer=9,total=39,rerun=False),
        syntax=dict(C=True,CPP=True,required_macros=3,rerun=False),resources_identical_to_original=39,
        models_unchanged=32,FC_channel_multipliers_checked=1056,Softmax_multiplier_domains_checked=32,
        whole_archive_rebuilt=False,whole_archive_rounding_changed=False,original_library_sha256=EXPECTED_LIBRARY,
        override_source_sha256=EXPECTED_MATVEC,protected_client_routines_AST_identical=protected,
        diagnosis_status=diagnosis['status'],causal_mechanism_reproduced=True,correction_physically_verified=False,
        old_failed_capture_preserved=True,old_partial_costs_promoted=False,
        physical_gate_plan=dict(cases=[[568,[0]],[232,[22]]],primary_RUN=6,PROFILE=2,BASE=2,negative_rejections=17,
            required_status='SR_GATE_PASS_NOT_TOTAL_COST_VALIDATED',output_comparison='exact_INT8_no_new_tolerance'),
        builds_executed=0,new_TFLite_invocations=0,new_training_fits=0,D_files_modified=False,
        serial_port_opened=False,hardware_flashed=False,selector_costs_enabled=False,
        diagnostic_manifest_sha256=sha(diag_manifest),original_failure_manifest_sha256=sha(FAILED/'MANIFEST_EVIDENCIAS_SHA256.json'),
        manifest_entries_including_closure=len(entries)+1,hashing_seconds_before_write=time.perf_counter()-start,
        exclusions=['mutable ESTADO_ACTUAL_ENTREGA_CWRU.md','future build/workspace/result directories','__pycache__','*.pyc','*.pyo'],
        limitations=['Syntax and configuration checks do not prove linked replacement symbol or runtime equivalence.',
            'User build must be followed by ELF/map/disassembly review and a new control+regression physical gate.',
            'No old physical timing is imported into the future single-firmware total-cost table.'])
    write(CLOSURE,report);entries[CLOSURE.relative_to(ROOT).as_posix()]=sha(CLOSURE)
    write(MANIFEST,dict(sorted(entries.items())))
    with SEAL.open('x',encoding='ascii') as stream:stream.write(sha(MANIFEST)+'  '+MANIFEST.name+'\n')
    for relative,expected in entries.items():require(sha(ROOT/relative)==expected,'Cambio despues del sello: '+relative)
    print(json.dumps(dict(status=report['status'],files=len(entries),manifest_sha256=sha(MANIFEST),
        closure_sha256=sha(CLOSURE),seconds=time.perf_counter()-start),ensure_ascii=True),flush=True)

if __name__=='__main__':main()
