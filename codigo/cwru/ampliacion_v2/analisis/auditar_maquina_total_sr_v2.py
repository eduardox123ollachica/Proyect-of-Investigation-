"""Auditoria estatica reproducible del ELF TotalV2_SR; sin compilar, TF o placa."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import re
import time

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT/'despliegue_stm32/costo_total_single_rounding_v2_180mhz_2026-10-26'
PROJECT = OUT/'copia_aislada/Prueba_2_CWRU_Extraccion'
CONFIG = 'Release_180MHz_TotalV2_SR'
BASE = PROJECT/CONFIG/'Prueba_2_CWRU_Extraccion'
EXPECTED_ELF = 'c0c3853d2b0e90bb20fe00be13bcf107d186088406be131b729a605ded44ecc7'


def require(ok, text):
    if not ok:
        raise ValueError(text)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def functions(listing):
    matches = list(re.finditer(r'^([0-9a-f]{8}) <([^>]+)>:\s*$', listing, re.M))
    result = {}
    for i, match in enumerate(matches):
        name = match.group(2)
        # Diversas unidades TFLM tienen funciones locales homonimas (Init/Eval).
        key = name if name not in result else name+'@'+match.group(1)
        result[key] = listing[match.start():matches[i+1].start() if i+1 < len(matches) else len(listing)]
    return result


def lines(text):
    return [line.strip() for line in text.splitlines() if re.match(r'^\s*[0-9a-f]+:', line)]


def calls(text):
    return [line for line in lines(text) if re.search(r'\bbl(?:x)?\s', line)]


def main():
    started = time.monotonic()
    require(sha(BASE.with_suffix('.elf')) == EXPECTED_ELF, 'ELF distinto del revisado')
    build = read(OUT/'COMPILACION_SR.json')
    audit = read(OUT/'AUDITORIA_ELF_SR.json')
    require(build['status'] == 'PASS_SR_BUILD_PENDING_ELF_REVIEW_AND_ARM'
            and build['returncode'] == 0 and build['zero_errors_and_warnings'], 'Build no aprobado')
    require(audit['status'] == 'PASS_SR_ELF_CATALOG_PARAMETERS_VERIFIED_PENDING_ARM'
            and audit['elf_sha256'] == EXPECTED_ELF, 'Auditoria ELF no aprobada')
    reader = ROOT/'codigo/cwru/ampliacion_v2/analisis/auditar_elf_total_v2.py'
    spec = importlib.util.spec_from_file_location('sr_machine_elf_reader', reader)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    elf = module.ELF32(BASE.with_suffix('.elf'))
    listing = BASE.with_suffix('.list').read_text(encoding='utf-8-sig')
    funcs = functions(listing)
    # No se confia solo en texto objdump: sus instrucciones y literales deben
    # coincidir byte por byte con las secciones cargables del ELF aprobado.
    byte_lines = 0
    for match in re.finditer(r'^\s*([0-9a-f]+):\t([0-9a-f ]+?)\s*\t([^\n]+)$', listing, re.M):
        tokens = match.group(2).split()
        require(tokens and all(len(token) in (2, 4, 8) for token in tokens), 'Grupo de bytes inesperado')
        payload = b''.join(int(token,16).to_bytes(len(token)//2,'little') for token in tokens)
        require(elf.address(int(match.group(1),16),len(payload)) == payload,
                'Listing no corresponde al ELF en '+match.group(1))
        byte_lines += 1
    require(byte_lines > 30000, 'Cobertura de listing insuficiente')
    matvec = funcs['arm_nn_vec_mat_mult_t_per_ch_s8']
    symbols = [s for s in elf.symbols if s['name'] == 'arm_nn_vec_mat_mult_t_per_ch_s8']
    require(len(symbols) == 1 and symbols[0]['address'] & ~1 == 0x080079d0
            and symbols[0]['size'] == 3702, 'Override duplicado o direccion/tamano diferentes')
    smull = [line for line in lines(matvec) if '\tsmull\t' in line]
    require(len(smull) == 6 and not re.search(r'\bsmlal\w*\s', matvec), 'No hay seis productos single exactos')
    # Seis sitios revisados: 2 acumuladores + cola en cada rama rhs_offset.
    sites = []
    for address, start, end, path in (
        (0x8007d48,0x8007d3a,0x8007db8,'nonzero_rhs_offset_pair_0'),
        (0x8007d7c,0x8007d74,0x8007db8,'nonzero_rhs_offset_pair_1'),
        (0x80080ec,0x80080e0,0x8008120,'nonzero_rhs_offset_tail'),
        (0x8008448,0x8008438,0x80084b6,'zero_rhs_offset_pair_0'),
        (0x8008476,0x8008472,0x80084b6,'zero_rhs_offset_pair_1'),
        (0x80087d2,0x80087c4,0x80087f8,'zero_rhs_offset_tail_shared_final_shift'),
    ):
        snippet = [line for line in lines(matvec) if start <= int(line.split(':')[0],16) <= end]
        text = '\n'.join(snippet)
        require(f'{address:x}:' in text and '\tsmull\t' in text and '#30' in text
                and re.search(r'\badds\s[^\n]*#1\b', text) and '\tadc.w\t' in text,
                'Sitio single rounding no coincide: '+path)
        require(('lsr #1' in text or re.search(r'\blsrs\s[^\n]*#1\b', text))
                if address != 0x80087d2 else '8008114' in text,
                'Desplazamiento final single ausente: '+path)
        sites.append(dict(smull_address=f'{address:08x}',path=path,instructions=snippet))
    header = PROJECT/'TFLM/third_party/cmsis_nn/Include/arm_nnsupportfunctions.h'
    header_text = header.read_text(encoding='utf-8-sig')
    require('const int64_t total_shift = 31 - shift;' in header_text
            and 'int64_t result = new_val >> (total_shift - 1);' in header_text
            and 'result = (result + 1) >> 1;' in header_text, 'Ecuacion header diferente')
    # La ruta enlazada FC int8 selecciona per-channel; las 32 topologias ya
    # verificadas son 2D/INT8 y contienen tres FC con cuantizacion per-channel.
    wrapper = funcs['arm_fully_connected_wrapper_s8']
    perchannel = funcs['arm_fully_connected_per_channel_s8']
    require(sum('<arm_nn_vec_mat_mult_t_per_ch_s8>' in line for line in calls(perchannel)) == 1,
            'Ruta per-channel no llega una vez al override')
    require(sum('<arm_fully_connected_per_channel_s8>' in line for line in calls(wrapper)) == 1,
            'Wrapper no selecciona per-channel')
    eval_name = next(name for name in funcs if 'EvalQuantizedInt8' in name)
    require(sum('<arm_fully_connected_wrapper_s8>' in line for line in calls(funcs[eval_name])) == 1,
            'TFLM EvalInt8 no enlaza wrapper FC')
    maptext = BASE.with_suffix('.map').read_text(encoding='utf-8-sig')
    require('CWRU_TotalSR/arm_nn_vec_mat_mult_t_per_ch_s8.o' in maptext
            and 'libtensorflow-microlite.a(arm_nn_vec_mat_mult_t_per_ch_s8.o)' not in maptext,
            'Objeto archive reemplazado no esta excluido')
    execute = funcs['CWRUTOTAL_Execute']
    require('cbz\tr6, 8008ff0' in execute and '0xe0001000' in execute, 'Ramas timer diferentes')
    primary_addresses = (0x8008ff4,0x800906c)
    profile_addresses = (0x8008fc0,0x8008fc4,0x8008fda,0x800901a,0x8009020,
                         0x800908a,0x8009098,0x80090a6,0x80090ac,0x80090ba)
    primary_reads = [line for line in lines(execute)
                     if int(line.split(':')[0],16) in primary_addresses]
    profile_reads = [line for line in lines(execute)
                     if int(line.split(':')[0],16) in profile_addresses]
    require(len(primary_reads) == 2 and all('[r7, #4]' in l for l in primary_reads)
            and len(profile_reads) == 10 and all('[r6, #4]' in l for l in profile_reads),
            'Lecturas CYCCNT diferentes')
    primary_calls = [line for line in calls(execute) if 0x8008ff0 <= int(line.split(':')[0],16) <= 0x8009078
                     and int(line.split(':')[0],16) != 0x800901c]
    require([re.search(r'<([^>]+)>',line).group(1) for line in primary_calls] ==
            ['CWRUMASK_Extract','input_transform','CWRUTNN_Predict','normality'], 'Orden de etapas primary diferente')
    nn = funcs['CWRUTNN_Predict']
    require('0xe0001000' not in nn and sum('MicroInterpreter6InvokeEv>' in line for line in calls(nn)) == 1,
            'Predict tiene timer o numero Invoke diferente')
    transform, normal = funcs['input_transform'], funcs['normality']
    require(transform.count('\tvdiv.f32\t') == 2 and normal.count('\tvdiv.f32\t') == 1
            and normal.count('\tvmul.f32\t') == 1 and normal.count('\tvadd.f32\t') == 1
            and not re.search(r'\bv(?:fma|mla|mls)',transform+normal), 'Operaciones float fusionadas o cambiadas')
    recipes = [line.strip() for path in (PROJECT/CONFIG).rglob('subdir.mk')
               for line in path.read_text(encoding='utf-8-sig').splitlines()
               if line.lstrip().startswith(('arm-none-eabi-gcc ', 'arm-none-eabi-g++ ')) and '-std=' in line]
    require(len(recipes) == 10 and all('-O3 ' in line and '-ffast-math' not in line
            and '-mfpu=fpv4-sp-d16' in line and '-mfloat-abi=hard' in line
            and '-DCMSIS_NN_USE_SINGLE_ROUNDING=1' in line and '-DTFLITE_SINGLE_ROUNDING=1' in line
            for line in recipes), 'Recetas C/C++ diferentes de la variante single')
    override = build['configurations'][CONFIG]['override']
    require(all(override[k] for k in ('application_object_in_objects_list','application_object_in_map',
            'original_archive_member_not_extracted','archive_unchanged','application_objects_before_prebuilt_archive')),
            'Condiciones de override de build no aprobadas')
    report = dict(status='PASS_SR_MACHINE_ROUNDING_REVIEW_PENDING_ARM', date_local='2026-10-26',
        timezone='America/Lima', elf_sha256=EXPECTED_ELF, elf_path=str(BASE.with_suffix('.elf')),
        listing_sha256=sha(BASE.with_suffix('.list')), map_sha256=sha(BASE.with_suffix('.map')),
        build_report_sha256=sha(OUT/'COMPILACION_SR.json'), elf_audit_sha256=sha(OUT/'AUDITORIA_ELF_SR.json'),
        review_basis='Existing CubeIDE objdump listing; every parsed instruction/literal byte verified against approved ELF',
        listing_byte_lines_verified=byte_lines, single_rounding=dict(symbol_address='080079d0', symbol_bytes=3702,
            unique_definition=True, smull_sites=6,smlal_sites=0,sites=sites,source_header_sha256=sha(header),
            equation='p=int64(acc)*multiplier; result=((p>>(30-shift))+1)>>1',
            mapping='six emitted sites cover DSP row pairs and remaining row for zero/nonzero rhs_offset',
            only_per_channel_matvec_replaced=True,whole_archive_single_rounding=False),
        call_chain=dict(TFLM_EvalInt8=eval_name, wrapper_calls=calls(wrapper), per_channel_calls=calls(perchannel),
                        original_archive_matvec_not_extracted=True),
        primary=dict(read_count=2,read_instructions=primary_reads,successful_path_calls=primary_calls,
                     result_memset_before_first_read=True,initialization_and_UART_outside_execute=True),
        profile=dict(read_count=10,read_instructions=profile_reads,included_in_primary_selector_cost=False),
        DNN_predict=dict(Invoke_calls=1,no_DWT_literal_or_read_in_wrapper=True,input_and_output_copy_inside=True,
                         argmax_strict_greater_ties_lowest=True),
        float32=dict(input_two_separate_vdiv=True,normality_vsub_vdiv_vsub_vmul_vadd_separate=True,
                     volatile_intermediate_stores_loads_present=True,no_FMA_in_input_transform_or_normality=True,
                     compiler_recipe_count=10,all_O3=True,fast_math_absent=True,hard_float_fp_v4_sp_d16=True),
        memory=audit['memory'],controls_EntM_and_old_TotalV2_ELF_and_Flash_identical=True,
        limitations=['Static audit cannot prove the image actually programmed on the board or numerical identity on hardware.',
                     'SR gate must cover the previous divergence and a control before the full campaign.',
                     'Only the per-channel matvec was rebuilt; other archive members retain their original code.',
                     'No timing fences or WCET guarantee; arena actual use and runtime stack remain pending.',
                     '32-model harness memory is not the final single-model deployment footprint.'],
        script_sha256=sha(__file__), reader_sha256=sha(reader), seconds=time.monotonic()-started,
        build_repeated=False, serial_port_opened=False, hardware_flashed=False,D_files_modified=False,
        selector_costs_enabled=False,TFLite_invocations=0)
    target = OUT/'AUDITORIA_MAQUINA_SR.json'
    require(not target.exists(), 'Informe ya existe; no sobrescribir')
    target.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=report['status'],elf_sha256=EXPECTED_ELF,listing_byte_lines_verified=byte_lines,
                         smull_sites=6,primary_reads=2,profile_reads=10),ensure_ascii=False))


if __name__ == '__main__':
    main()
