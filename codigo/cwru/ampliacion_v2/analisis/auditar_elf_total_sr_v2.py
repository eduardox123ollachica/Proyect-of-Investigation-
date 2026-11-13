"""Audita el ELF SR y los artefactos existentes sin compilar ni usar hardware."""
from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import time
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'despliegue_stm32/costo_total_single_rounding_v2_180mhz_2026-10-26'
OLD=ROOT/'despliegue_stm32/costo_total_representantes_v2_180mhz_2026-10-25'
CONFIG='Release_180MHz_TotalV2_SR'
BASE='Release_180MHz_TotalV2'
PROJECT=OUT/'copia_aislada/Prueba_2_CWRU_Extraccion'
ELF=PROJECT/CONFIG/'Prueba_2_CWRU_Extraccion.elf'
EXPECTED_ELF='c0c3853d2b0e90bb20fe00be13bcf107d186088406be131b729a605ded44ecc7'
REPORT=OUT/'AUDITORIA_ELF_SR.json'

def require(ok,msg):
    if not ok: raise ValueError(msg)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    require(spec and spec.loader,'Modulo no importable')
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result

def canonical(node,ignore_top=False):
    attrs=dict(node.attrib)
    detector='com.st.stm32cube.ide.mcu.toolchain.armnone.setup.CrossBuiltinSpecsDetector'
    if node.tag=='provider' and attrs.get('class')==attrs.get('id')==detector:
        attrs.pop('env-hash',None)
    if ignore_top and node.tag=='configuration':
        attrs.pop('id',None);attrs.pop('name',None)
    return node.tag,tuple(sorted(attrs.items())),(node.text or '').strip(),tuple(canonical(c) for c in node)

def generated_metadata_mutation():
    before_path=OUT/'respaldo_antes_single_rounding/.settings/language.settings.xml'
    after_path=PROJECT/'.settings/language.settings.xml'
    before_xml=ET.parse(before_path).getroot();after_xml=ET.parse(after_path).getroot()
    before={n.get('id'):n for n in before_xml};after={n.get('id'):n for n in after_xml}
    require(before_xml.tag==after_xml.tag=='project' and before_xml.attrib==after_xml.attrib,
            'Cambio de raiz XML no esperado')
    require(len(before)==16 and len(after)==17 and all(k in after and canonical(n)==canonical(after[k])
            for k,n in before.items()),'Metadatos de configuraciones anteriores cambiaron')
    added=[n for k,n in after.items() if k not in before]
    require(len(added)==1 and added[0].get('name')==CONFIG,'Adicion de metadatos no es SR')
    template=next(n for n in before.values() if n.get('name')==BASE)
    require(canonical(template,True)==canonical(added[0],True),'Proveedores SR distintos de TotalV2')
    roots=ET.parse(PROJECT/'.cproject').getroot()
    ids=[n.get('id') for n in roots.iter('cconfiguration')
         if any(c.get('name')==CONFIG for c in n.iter('configuration'))]
    require(ids==[added[0].get('id')],'ID SR de metadatos distinto del build')
    return dict(path=after_path.relative_to(ROOT).as_posix(),before_sha256=sha(before_path),
                after_sha256=sha(after_path),previous_sixteen_configurations_semantically_equal_ignoring_env_hash=True,
                new_configuration_providers_equal_TotalV2_except_id_name_env_hash=True,
                previous_manifest_preserved=True)

def verify_seal(folder,name,root_relative,mutation=None):
    path=folder/name;digest,declared=path.with_suffix('.sha256').read_text(encoding='ascii').split()
    require(declared==name and sha(path)==digest,'Sello invalido: '+name)
    entries=read(path);base=ROOT if root_relative else folder;differences=[]
    for rel,expected in entries.items():
        target=(base/rel).resolve()
        require(target.is_relative_to(base.resolve()) and target.is_file(),'Ruta de manifiesto invalida: '+rel)
        actual=sha(target)
        if actual!=expected:
            require(mutation and rel==mutation['path'] and expected==mutation['before_sha256']
                    and actual==mutation['after_sha256'],'Entrada modificada: '+rel)
            differences.append(rel)
    if mutation:require(differences==[mutation['path']],'Se esperaba solo la regeneracion CDT auditada')
    return entries,digest

def main():
    started=time.perf_counter();require(not REPORT.exists(),'No sobrescribir auditoria')
    old=module(Path(__file__).with_name('auditar_elf_total_v2.py'),'sr_elf_classes')
    preparer=module(ROOT/'codigo/cwru/ampliacion_v2/stm32/preparar_total_single_rounding_v2.py','sr_preparer_audit')
    mutation=generated_metadata_mutation()
    entries,prep_seal=verify_seal(OUT,'MANIFEST_PREPARACION_SR_SHA256.json',True,mutation)
    resource_entries,resource_seal=verify_seal(OUT/'recursos','MANIFEST_RECURSOS_SHA256.json',False)
    require(len(entries)==1992 and len(resource_entries)==39,'Numero de recursos inesperado')
    prep=read(OUT/'PREPARACION_SR.json');preparer.verify(prep)
    build=read(OUT/'COMPILACION_SR.json');rows=build['configurations'];total=rows[CONFIG]
    require(build['status']=='PASS_SR_BUILD_PENDING_ELF_REVIEW_AND_ARM' and build['returncode']==0
            and not build['timed_out'] and build['zero_errors_and_warnings'] and build['integrity_after_build'],
            'Compilacion no aprobada')
    require(all(r['flags_confirmed'] and r['CMSIS_NN_USE_SINGLE_ROUNDING_matches_expected']
                and r['TFLITE_SINGLE_ROUNDING_matches_expected'] for r in rows.values()),'Banderas distintas')
    require(all(rows[n]['control_ELF_identical'] and rows[n]['control_Flash_identical']
                for n in ('Release_180MHz_EntM',BASE)),'Los controles cambiaron')
    archive=OUT/'intentos'/str(build['attempt'])/'compilacion'/CONFIG/ELF.name
    require(sha(ELF)==sha(archive)==total['elf_sha256']==EXPECTED_ELF,'ELF no coincide con build/archivo')
    require(sha(PROJECT/'Lib/libtensorflow-microlite.a')==prep['original_library_sha256']
            ==sha(Path(prep['source'])/'Lib/libtensorflow-microlite.a'),'La biblioteca original cambio')
    cat=read(OUT/'recursos/CATALOGO_TOTAL.json');oldcat=read(OLD/'recursos/CATALOGO_TOTAL.json')
    require(cat==oldcat and len(cat['representatives'])==32,'Catalogo distinto del original')
    elf=old.ELF32(ELF);old_elf=old.ELF32(PROJECT/BASE/ELF.name)
    table=elf.symbol('cwru_total_catalog');oldtable=old_elf.symbol('cwru_total_catalog');stride=328
    require(table['size']==oldtable['size']==32*stride,'Stride/size descriptor inesperado')
    require(elf.cstring(elf.symbol('cwru_total_catalog_sha256')['address'],64)==cat['catalog_sha256'],
            'SHA catalogo no coincide')
    require(f'{struct.unpack("<I",elf.address(elf.symbol("cwru_total_catalog_fnv")["address"],4))[0]:08x}'==cat['catalog_fnv'],
            'FNV catalogo no coincide')
    models=[];ranges=[]
    for rep in cat['representatives']:
        rid=rep['id'];address=table['address']+rid*stride;desc=elf.address(address,stride)
        idptr,modelptr,length=struct.unpack_from('<III',desc)
        require(elf.cstring(idptr,100)==rep['representative_id'] and length==rep['model_bytes']
                and modelptr%16==0 and 0x08000000<=modelptr<modelptr+length<=0x08080000,'Modelo/id/rango inesperado')
        require(list(desc[12:15])==[rep['mask_id'],rep['dims'],rep['det_dims']]
                and desc[41:44]==b'\0\0\0','Dimensiones o padding inesperados')
        params=bytes([desc[13],desc[14],desc[12]])+desc[15:41]+desc[44:312]
        require(len(params)==297 and params.hex()==rep['params_bytes_hex'] and old.fnv(params)==rep['params_fnv'],
                'Parametros distintos')
        od=old_elf.address(oldtable['address']+rid*stride,stride)
        optr,olen=struct.unpack_from('<II',od,4)
        original_params=bytes([od[13],od[14],od[12]])+od[15:41]+od[44:312]
        payload=elf.address(modelptr,length)
        require(payload==(OUT/'recursos'/rep['model_path']).read_bytes()==old_elf.address(optr,olen)
                and params==original_params,'Bytes distintos de la version TotalV2 original')
        require(hashlib.sha256(payload).hexdigest()==rep['model_sha256'] and old.fnv(payload)==rep['model_fnv'],
                'Fingerprint del modelo distinto')
        mf,pf,sp,fp=struct.unpack_from('<IIII',desc,312)
        require(f'{mf:08x}'==rep['model_fnv'] and f'{pf:08x}'==rep['params_fnv']
                and elf.cstring(sp,64)==rep['model_sha256'] and elf.cstring(fp,64)==rep['source_fit_signature'],
                'Fingerprints o firmas distintos')
        syms=[s for s in elf.symbols if f'cwru_tv_model_{rid:02d}' in s['name']]
        require(len(syms)==1 and syms[0]['address']==modelptr and syms[0]['size']==length,'Simbolo de modelo inesperado')
        models.append(dict(id=rid,representative_id=rep['representative_id'],descriptor_address=f'{address:08x}',
            model_address=f'{modelptr:08x}',model_bytes=length,model_sha256=rep['model_sha256'],
            model_fnv=rep['model_fnv'],params_fnv=rep['params_fnv'],serialized_parameter_bytes=297,
            params_sha256=hashlib.sha256(params).hexdigest(),bytes_exact_exported_and_original_TotalV2=True,
            topology=old.FlatBuffer(payload).validate(rep)))
        ranges.append((modelptr,modelptr+length))
    ranges.sort();require(all(a[1]<=b[0] for a,b in zip(ranges,ranges[1:])),'Modelos superpuestos')
    buffers=[]
    for name,size in [('g_window',32768),('g_fft_input',16384),('g_fft_output',16384)]:
        s=elf.symbol(name);require(s['size']==size,'Buffer distinto: '+name)
        buffers.append(dict(name=name,address=s['address'],bytes=size))
    large=[s for s in elf.symbols if s['size']>=16384 and 0x20000000<=s['address']<0x20020000]
    arenas=[s for s in large if s['name'] not in ('g_window','g_fft_input','g_fft_output')]
    require(len(large)==4 and len(arenas)==1 and arenas[0]['size']==16384 and arenas[0]['address']%16==0,
            'Arena unica/alineacion distinta')
    buffers.append(dict(name=arenas[0]['name'],address=arenas[0]['address'],bytes=16384))
    br=sorted((b['address'],b['address']+b['bytes']) for b in buffers)
    require(all(0x20000000<=a<b<=0x20020000 for a,b in br) and all(a[1]<=b[0] for a,b in zip(br,br[1:])),
            'Buffers fuera SRAM o superpuestos')
    allocated=[s for s in elf.sections if s['flags']&2]
    ram_sections=[s for s in allocated if 0x20000000<=s['address']<0x20020000]
    ram=sum(s['size'] for s in ram_sections)
    flash=sum(s['size'] for s in allocated if s['type']!=8)
    fs=[s for s in elf.segments if s['type']==1 and 0x08000000<=s['physical']<0x08080000]
    flash_seg=sum(s['size'] for s in fs);flash_span=max(s['physical']+s['size'] for s in fs)-0x08000000
    ram_span=max(s['address']+s['size'] for s in ram_sections)-0x20000000
    require(flash==total['flash_bytes']==280276 and ram==total['ram_including_linker_reservations_bytes']==88460,
            'Memoria por secciones difiere del build')
    require(flash<=flash_seg<=flash_span<=524288 and ram<=ram_span<=131072,'Rangos de memoria invalidos')
    reservations={n:elf.symbol(n)['address'] for n in ('_Min_Heap_Size','_Min_Stack_Size','_estack','_end')}
    require(reservations['_Min_Heap_Size']==512 and reservations['_Min_Stack_Size']==4096,'Reservas modificadas')
    result=dict(status='PASS_SR_ELF_CATALOG_PARAMETERS_VERIFIED_PENDING_ARM',schema_version=1,
        date_local='2026-10-26',timezone='America/Lima',elf_path=str(ELF),elf_sha256=sha(ELF),
        archive_path=str(archive),archive_identical=True,build_report_sha256=sha(OUT/'COMPILACION_SR.json'),
        build_seconds=build['seconds'],zero_errors_and_warnings=True,
        preparation_resources_verified=len(entries),preparation_manifest_sha256=prep_seal,
        exported_resources_verified=len(resource_entries),resource_manifest_sha256=resource_seal,
        preparation_resource_mutation=mutation,old_D_source_files_verified=len(prep['source_inventory_before']),
        old_D_ELFs_verified=len(prep['previous_ELF_sha256']),old_configurations_preserved=16,isolated_configurations=17,
        control_EntM_and_TotalV2_ELF_and_Flash_identical=True,compiler_flags_verified=True,
        descriptor_stride_bytes=stride,descriptor_abi_padding_bytes=3,descriptor_parameter_serialization_bytes=297,
        catalog_sha256=cat['catalog_sha256'],catalog_fnv=cat['catalog_fnv'],model_count=32,
        model_payload_bytes=sum(m['model_bytes'] for m in models),models=models,
        all_model_bytes_and_parameters_exact_to_original_TotalV2=True,
        buffers=buffers,one_reused_arena=True,original_archive_unchanged=True,
        original_archive_sha256=prep['original_library_sha256'],
        memory=dict(flash_load_bytes=flash,flash_capacity_bytes=524288,flash_percent=100*flash/524288,
            flash_program_segment_bytes=flash_seg,flash_program_address_span_bytes=flash_span,
            flash_alignment_padding_bytes=flash_seg-flash,ram_static_including_linker_reservations_bytes=ram,
            ram_capacity_bytes=131072,ram_percent=100*ram/131072,ram_address_span_bytes=ram_span,
            ram_address_alignment_gap_bytes=ram_span-ram,ram_unreserved_address_bytes=131072-ram_span,
            heap_reservation_bytes=512,stack_reservation_bytes=4096,linker_symbols=reservations,
            runtime_arena_use_pending=True,stack_runtime_measurement_pending=True,
            flash_is_32_model_harness_not_final_one_model_deployment=True),
        sources_modified=False,D_files_modified=False,new_builds_executed=0,TFLite_invocations=0,
        serial_port_opened=False,hardware_flashed=False,selector_costs_enabled=False,
        audit_script_sha256=sha(__file__),reader_source_sha256=sha(Path(__file__).with_name('auditar_elf_total_v2.py')),
        seconds=time.perf_counter()-started,
        limitations=['Static ELF audit does not prove the image programmed on the board.',
            'Machine-code Single Rounding confirmation is a separate audit.',
            'Both physical control and regression cases remain pending.',
            'Arena actual use, stack high-water and total timing remain pending.',
            'No WCET or predictive model improvement is established by this audit.'])
    with REPORT.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({k:result[k] for k in ('status','elf_sha256','model_count','model_payload_bytes','memory','seconds')},ensure_ascii=True))

if __name__=='__main__':main()
