"""Audita el ELF canónico de ocho modelos sin ejecutar inferencias o placa.

Los lectores ELF32/FlatBuffer proceden de la auditoría anterior. Se comprueba
el contenido enlazado, no únicamente el texto de las fuentes generadoras.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import re
import struct
import sys
import time

ROOT=Path(__file__).resolve().parents[4]
HERE=ROOT/'codigo/cwru/ampliacion_v2/canonicos_stm32'
OUT=ROOT/'despliegue_stm32/canonicos_42_180mhz_2026-10-27'
CONFIG='Release_180MHz_Canon42_SR'
PROJECT=OUT/'copia_aislada/Prueba_2_CWRU_Extraccion'
FOLDER=PROJECT/CONFIG
ELF=FOLDER/'Prueba_2_CWRU_Extraccion.elf'
REPORT=OUT/'AUDITORIA_ELF_CANON42.json'
ELF_SHA='c2de62a6e73a7f0f4cc2f4d25f5cbef718bde28eb93e05d0273871337ce84a46'
CATALOG_SHA='e7a0c848ccb293c3dd95c9e150e8e6909f6ae6b82c3ec90e6332faebbc2a6fdd'
KERNEL_REPORT=OUT/'AUDITORIA_INDEPENDIENTE_KERNEL_MEMORIA.json'
METADATA_REPORT=OUT/'AUDITORIA_METADATA_CDT.json'
READER=ROOT/'codigo/cwru/ampliacion_v2/analisis/auditar_elf_total_v2.py'


def require(value,message):
    if not value:raise ValueError(message)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    require(spec is not None and spec.loader is not None,'Módulo no disponible: '+str(path))
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


reader=module('canon_elf32_flatbuffer_reader',READER)


def audit_models(elf,catalog,resources):
    rows=catalog['representatives']
    require(catalog['count']==len(rows)==8 and [r['id'] for r in rows]==list(range(8)),
        'Se requieren exactamente ocho descriptores ordenados')
    require(catalog['catalog_sha256']==CATALOG_SHA,'Catálogo distinto del canónico congelado')
    table=elf.symbol('cwru_total_catalog');stride=328
    require(table['size']==8*stride and table['address']%4==0,'Stride o tamaño de catálogo distinto')
    require(elf.cstring(elf.symbol('cwru_total_catalog_sha256')['address'],64)==catalog['catalog_sha256'],
        'SHA del catálogo enlazado distinto')
    require(f'{struct.unpack("<I",elf.address(elf.symbol("cwru_total_catalog_fnv")["address"],4))[0]:08x}'
        ==catalog['catalog_fnv'],'FNV del catálogo enlazado distinto')
    actual_symbols=[s for s in elf.symbols if 'cwru_c42_' in s['name'] and s['section_index']!=0]
    require(len(actual_symbols)==8,'Modelos enlazados adicionales o ausentes')
    report=[];ranges=[]
    for row in rows:
        rid=row['id'];desc=elf.address(table['address']+rid*stride,stride)
        nameptr,modelptr,length=struct.unpack_from('<III',desc)
        require(elf.cstring(nameptr,100)==row['representative_id'],'ID enlazado distinto')
        require(length==row['model_bytes'] and modelptr%16==0 and
            0x08000000<=modelptr<modelptr+length<=0x08080000,'Modelo/alineación/rango inválido')
        require(list(desc[12:15])==[row['mask_id'],row['dims'],row['det_dims']],
            'Dimensión o máscara enlazada distinta')
        require(desc[41:44]==b'\0\0\0','Padding ABI distinto')
        parameters=bytes([desc[13],desc[14],desc[12]])+desc[15:41]+desc[44:312]
        require(len(parameters)==297 and parameters.hex()==row['params_bytes_hex']
            and reader.fnv(parameters)==row['params_fnv'],'Parámetros enlazados distintos')
        blob=elf.address(modelptr,length);digest=hashlib.sha256(blob).hexdigest()
        require(blob==(Path(resources)/row['model_path']).read_bytes() and digest==row['model_sha256']
            and reader.fnv(blob)==row['model_fnv'],'Modelo enlazado distinto de los pesos congelados')
        mf,pf,shaptr,fitptr=struct.unpack_from('<IIII',desc,312)
        require(f'{mf:08x}'==row['model_fnv'] and f'{pf:08x}'==row['params_fnv'] and
            elf.cstring(shaptr,64)==digest and elf.cstring(fitptr,64)==row['source_fit_signature'],
            'Firmas o fingerprints enlazados distintos')
        syms=[s for s in actual_symbols if 'cwru_c42_'+row['representative_id'] in s['name']]
        require(len(syms)==1 and syms[0]['address']==modelptr and syms[0]['size']==length,
            'Símbolo modelo y descriptor no coinciden')
        sections=[s for s in elf.sections if s['flags']&2 and s['type']!=8
            and s['address']<=modelptr<modelptr+length<=s['address']+s['size']]
        require(len(sections)==1 and not sections[0]['flags']&1,'Modelo no reside en Flash de solo lectura')
        topology=reader.FlatBuffer(blob).validate(row)
        report.append(dict(id=rid,representative_id=row['representative_id'],model_sha256=digest,
            model_fnv=row['model_fnv'],params_fnv=row['params_fnv'],model_bytes=length,
            params_sha256=hashlib.sha256(parameters).hexdigest(),parameter_bytes=297,
            descriptor_address=f'{table["address"]+rid*stride:08x}',model_address=f'{modelptr:08x}',
            all_model_bytes_exact=True,all_parameter_bytes_exact=True,topology=topology))
        ranges.append((modelptr,modelptr+length))
    ranges.sort()
    require(all(a[1]<=b[0] for a,b in zip(ranges,ranges[1:])),'Modelos enlazados superpuestos')
    return report


def audit_memory(elf):
    buffers=[]
    for name,size in [('g_window',32768),('g_fft_input',16384),('g_fft_output',16384)]:
        symbol=elf.symbol(name);require(symbol['size']==size,'Buffer distinto: '+name)
        buffers.append(dict(name=name,address=symbol['address'],bytes=size))
    large=[s for s in elf.symbols if s['size']>=16384 and 0x20000000<=s['address']<0x20020000]
    require(len(large)==4,'Número de buffers SRAM grandes distinto')
    arena=[s for s in large if s['name'] not in ('g_window','g_fft_input','g_fft_output')]
    require(len(arena)==1 and arena[0]['size']==16384 and arena[0]['address']%16==0,'Arena no única/alineada')
    buffers.append(dict(name=arena[0]['name'],address=arena[0]['address'],bytes=16384))
    ranges=sorted((b['address'],b['address']+b['bytes']) for b in buffers)
    require(all(0x20000000<=a<b<=0x20020000 for a,b in ranges) and
        all(a[1]<=b[0] for a,b in zip(ranges,ranges[1:])),'Buffers superpuestos o fuera de SRAM')
    alloc=[s for s in elf.sections if s['flags']&2]
    ram_sections=[s for s in alloc if 0x20000000<=s['address']<0x20020000]
    require(all(s['address']+s['size']<=0x20020000 for s in ram_sections),'Sección fuera de SRAM')
    ram=sum(s['size'] for s in ram_sections)
    ram_span=max(s['address']+s['size'] for s in ram_sections)-0x20000000
    flash_sections=[s for s in alloc if s['type']!=8]
    flash=sum(s['size'] for s in flash_sections)
    segments=[s for s in elf.segments if s['type']==1 and 0x08000000<=s['physical']<0x08080000]
    require(segments and all(s['physical']+s['size']<=0x08080000 for s in segments),'Segmento fuera de Flash')
    segment_bytes=sum(s['size'] for s in segments)
    flash_span=max(s['physical']+s['size'] for s in segments)-0x08000000
    require(flash<=512*1024 and ram<=128*1024 and ram_span<=128*1024,'Capacidad física excedida')
    reservations={name:elf.symbol(name)['address'] for name in ('_Min_Heap_Size','_Min_Stack_Size','_estack','_end')}
    require(reservations['_Min_Heap_Size']==512 and reservations['_Min_Stack_Size']==4096 and
        reservations['_estack']==0x20020000,'Reservas o top de pila diferentes')
    return dict(flash_load_bytes=flash,flash_capacity_bytes=524288,flash_percent=100*flash/524288,
        flash_program_segment_bytes=segment_bytes,flash_program_address_span_bytes=flash_span,
        flash_alignment_padding_bytes=segment_bytes-flash,ram_static_including_linker_reservations_bytes=ram,
        ram_capacity_bytes=131072,ram_percent=100*ram/131072,ram_address_span_bytes=ram_span,
        ram_address_alignment_gap_bytes=ram_span-ram,ram_unreserved_address_bytes=131072-ram_span,
        linker_symbols=reservations,heap_reservation_bytes=512,stack_reservation_bytes=4096,
        buffers=buffers,one_reused_arena=True,arena_capacity_bytes=16384,arena_used_pending_runtime=True,
        stack_high_water_or_WCET_claimed=False,Flash_is_eight_model_validation_harness=True)


def audit():
    started=time.perf_counter()
    if str(HERE) not in sys.path:sys.path.insert(0,str(HERE))
    prep_module=module('canon_audit_preparer',HERE/'preparar_proyecto.py')
    exports=module('canon_audit_exports',HERE/'exportar_catalogo.py')
    banks_module=module('canon_audit_banks',HERE/'preparar_bancos.py')
    prep=prep_module.verify(after_build=True)
    cat=exports.verify_resources(OUT/'recursos');bank=banks_module.verify(OUT/'bancos')
    build=read(OUT/'COMPILACION_CANON42.json')
    require(build['status']=='PASS_BUILD_PENDING_ELF_REVIEW_AND_ARM' and build['returncode']==0
        and build['zero_errors_and_warnings'] and build['integrity_after_build'] and
        build['D_source_unchanged_verified'] and not build['timed_out'] and build['error'] is None,
        'Compilación o integridad no aprobada')
    require(sha(ELF)==build['elf_sha256']==ELF_SHA and
        sha(FOLDER/'Prueba_2_CWRU_Extraccion.map')==build['map_sha256'],'ELF o mapa cambió')
    require(cat['catalog_sha256']==prep['catalog_sha256']==bank['catalog_sha256']==CATALOG_SHA,
        'Identidad de catálogo divergente entre bancos/copia/recursos')
    metadata=read(METADATA_REPORT)
    require(metadata['status']=='PASS_GENERATED_METADATA_ONLY' and
        metadata['original_configurations_semantically_equal']==17 and
        metadata['added_configuration']==CONFIG and metadata['added_providers_equal_SR_except_identity_and_env_hash']
        and metadata['after_sha256']==sha(PROJECT/'.settings/language.settings.xml'),
        'Metadatos regenerados no coinciden con auditoría semántica')
    kernel=read(KERNEL_REPORT)
    require(kernel['status']=='PASS_INDEPENDENT_LINKED_KERNEL_MEMORY_PENDING_ARM' and
        all(kernel['kernel'][k] for k in ('bit_identical_to_validated_SR_ELF',
            'original_archive_member_not_extracted','precompiled_archive_unchanged')) and
        kernel['compile']['all_C_CPP_expected_flags'] and kernel['memory']['physical_limits_passed'],
        'Revisión independiente de kernels/flags/memoria no aprobada')
    elf=reader.ELF32(ELF)
    require(elf.unpack('<H',16)[0]==2 and elf.unpack('<I',36)[0]==0x05000400,
        'Se requiere ELF32 ARM EABI5 hard-float ejecutable')
    vector=elf.address(0x08000000,8);initial_stack,reset=struct.unpack('<II',vector)
    require(initial_stack==0x20020000 and reset==elf.unpack('<I',24)[0] and reset&1,
        'Vector inicial o Reset_Handler no coincide')
    models=audit_models(elf,cat,OUT/'recursos');memory=audit_memory(elf)
    post=elf.symbol('CWRUCANON_PostExecute')
    require(post['info']&15==2 and post['size']>0 and 0x08000000<=post['address']<0x08080000,
        'Ruta POST no enlazada como código ARM')
    require(b'CWRUTOTAL_POST_RESULT,' in elf.data and b'CWRUTOTAL_POST_INFO,1,14,56,0' in elf.data and
        b'CWRUTOTAL_INFO,2,%u,12000,4096,2,' in elf.data,'Protocolo canónico POST/INFO no presente')
    kernel_symbol=elf.symbol('arm_nn_vec_mat_mult_t_per_ch_s8')
    require(kernel_symbol['size']==3702,'Kernel enlazado distinto del SR validado')
    map_text=(FOLDER/'Prueba_2_CWRU_Extraccion.map').read_text(encoding='utf-8')
    require('CWRU_TotalSR/arm_nn_vec_mat_mult_t_per_ch_s8.o' in map_text and
        'libtensorflow-microlite.a(arm_nn_vec_mat_mult_t_per_ch_s8.o)' not in map_text,
        'Override de aplicación no sustituye al miembro del archivo')
    require(sha(ELF)==ELF_SHA and sha(FOLDER/'Prueba_2_CWRU_Extraccion.map')==build['map_sha256'],
        'Artefactos cambiaron durante auditoría')
    return dict(status='PASS_ELF_CANON42_VERIFIED_PENDING_ARM',schema_version=1,
        date_local='2026-10-27',timezone='America/Lima',firmware_elf_path=str(ELF),
        firmware_elf_sha256=ELF_SHA,elf_sha256=ELF_SHA,build_report_sha256=sha(OUT/'COMPILACION_CANON42.json'),
        map_sha256=build['map_sha256'],configuration=CONFIG,build_seconds=build['seconds'],
        zero_errors_and_warnings=True,source_integrity_after_build=True,
        original_D_sources_verified=len(prep['D_source_sha256_before']),
        original_D_ELFs_verified=len(prep['D_existing_ELF_sha256']),
        preparation_manifest_sha256=sha(prep_module.MANIFEST),
        resource_manifest_sha256=sha(OUT/'recursos/MANIFEST_RECURSOS_SHA256.json'),
        bank_manifest_sha256=sha(OUT/'bancos/MANIFEST_BANCOS_SHA256.json'),
        metadata_audit_sha256=sha(METADATA_REPORT),kernel_audit_sha256=sha(KERNEL_REPORT),
        catalog_sha256=cat['catalog_sha256'],catalog_fnv=cat['catalog_fnv'],model_count=8,
        model_payload_bytes=sum(m['model_bytes'] for m in models),models=models,
        descriptor_stride_bytes=328,descriptor_parameter_serialization_bytes=297,
        memory=memory,post_code_symbol=dict(address=post['address'],bytes=post['size']),
        reader_sha256=sha(READER),audit_script_sha256=sha(__file__),seconds=time.perf_counter()-started,
        new_builds_executed=0,classifier_invocations=0,serial_port_opened=False,hardware_flashed=False,
        sources_modified=False,D_files_modified=False,
        limitations=['Static ELF review does not establish the identity of the image programmed on the board.',
            'Physical RAW/POST gates, AllocateTensors arena usage and runtime equivalence remain pending.',
            'Memory totals belong to an eight-model validation harness, not a final single-model firmware.',
            'Linker stack reservation is not a runtime stack maximum; no WCET or timing measured here.',
            'RAW coverage is partial: 56 unique windows, 42 development windows per model.'])


def main():
    require(not REPORT.exists(),'No sobrescribir auditoría existente')
    result=audit()
    with REPORT.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({k:result[k] for k in ('status','firmware_elf_sha256','model_count',
        'model_payload_bytes','memory','seconds')},ensure_ascii=True))


if __name__=='__main__':main()
