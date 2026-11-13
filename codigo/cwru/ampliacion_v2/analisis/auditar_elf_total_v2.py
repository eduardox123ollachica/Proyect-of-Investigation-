"""Audita bytes ELF32 ARM del banco TotalV2; no compila ni accede a la placa."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT/'despliegue_stm32/costo_total_representantes_v2_180mhz_2026-10-25'
RESOURCES = OUT/'recursos'
PROJECT = OUT/'copia_aislada/Prueba_2_CWRU_Extraccion'
CONFIG = 'Release_180MHz_TotalV2'
ELF = PROJECT/CONFIG/'Prueba_2_CWRU_Extraccion.elf'
REPORT = OUT/'AUDITORIA_ELF_TOTAL.json'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def fnv(data):
    result = 2166136261
    for value in data:
        result = ((result ^ value)*16777619)&0xffffffff
    return f'{result:08x}'


def verify_manifest(folder, filename, relative_to_root, accepted_mutations=None):
    path = folder/filename
    digest, name = path.with_suffix('.sha256').read_text(encoding='ascii').split()
    require(name == path.name and digest == sha(path), 'Sello invalido: '+str(path))
    entries = read(path)
    base = ROOT if relative_to_root else folder
    for relative, expected in entries.items():
        target = (base/relative).resolve()
        require(target.is_relative_to(base.resolve()), 'Ruta fuera del proyecto: '+relative)
        current=sha(target)
        if accepted_mutations and relative in accepted_mutations:
            require(accepted_mutations[relative]==dict(before_sha256=expected,after_sha256=current),
                    'Mutacion generada distinta de la auditada: '+relative)
        else:
            require(current == expected, 'Recurso modificado: '+relative)
    return entries


def language_settings_mutation():
    """CubeIDE regenera env-hash y agrega16a cfg; no reescribir sello previo."""
    backup=OUT/'respaldo_antes_total/.settings/language.settings.xml'
    current=PROJECT/'.settings/language.settings.xml'
    def canonical(n):
        return (n.tag,tuple(sorted((k,v) for k,v in n.attrib.items() if k!='env-hash')),
                (n.text or '').strip(),tuple(canonical(c) for c in n))
    before={n.get('id'):n for n in ET.parse(backup).getroot()}
    after={n.get('id'):n for n in ET.parse(current).getroot()}
    require(len(before)==15 and len(after)==16
            and all(k in after and canonical(n)==canonical(after[k]) for k,n in before.items()),
            'CubeIDE cambio semanticamente language settings anteriores')
    added=[n for k,n in after.items() if k not in before]
    require(len(added)==1 and added[0].get('name')==CONFIG,'Nueva language setting inesperada')
    template=next(n for n in before.values() if n.get('name')=='Release_180MHz_Masks')
    def provider_canonical(n):
        ignored={'env-hash'}|({'id','name'} if n.tag=='configuration' else set())
        return (n.tag,tuple(sorted((k,v) for k,v in n.attrib.items() if k not in ignored)),
                (n.text or '').strip(),tuple(provider_canonical(c) for c in n))
    require(provider_canonical(template)==provider_canonical(added[0]),
            'Providers de TotalV2 distintos de la plantilla Masks')
    croot=ET.parse(PROJECT/'.cproject').getroot()
    ids=[n.get('id') for n in croot.iter('cconfiguration')
         if any(c.get('name')==CONFIG for c in n.iter('configuration'))]
    require(ids==[added[0].get('id')],'ID language settings no corresponde a cproject')
    mutation=dict(before_sha256=sha(backup),after_sha256=sha(current))
    return {current.relative_to(ROOT).as_posix():mutation},dict(
        path=current.relative_to(ROOT).as_posix(),**mutation,
        scope='CubeIDE generated env-hash attributes and added TotalV2 language settings only',
        previous_fifteen_configurations_semantically_equal_ignoring_env_hash=True,
        new_configuration_providers_equal_Masks_except_id_name_env_hash=True,
        previous_manifest_preserved=True)


class ELF32:
    def __init__(self, path):
        self.data = Path(path).read_bytes()
        require(self.data[:7] == b'\x7fELF\x01\x01\x01', 'ELF32 little endian requerido')
        require(self.unpack('<H',18)[0] == 40, 'ELF no es ARM')
        phoff, shoff = self.unpack('<II',28)
        phsize, phcount, shsize, shcount, strindex = self.unpack('<HHHHH',42)
        require(phsize == 32 and shsize == 40, 'Layout ELF inesperado')
        self.segments = []
        for i in range(phcount):
            typ,off,addr,physical,size,memsize,flags,alignment = self.unpack('<8I',phoff+i*phsize)
            self.segments.append(dict(type=typ,offset=off,address=addr,physical=physical,
                                      size=size,memory_size=memsize,flags=flags,alignment=alignment))
        raw = [self.unpack('<10I',shoff+i*shsize) for i in range(shcount)]
        strings = self.range(raw[strindex][4],raw[strindex][5])
        self.sections = []
        for item in raw:
            name,typ,flags,addr,off,size,link,info,align,entry_size = item
            self.sections.append(dict(name=self.string_at(strings,name),type=typ,flags=flags,address=addr,
                                      offset=off,size=size,link=link,info=info,alignment=align,entry_size=entry_size))
        self.symbols = []
        for section in self.sections:
            if section['type'] != 2:
                continue
            require(section['entry_size'] == 16, 'Tabla simbolos no ELF32')
            string_section = self.sections[section['link']]
            strings = self.range(string_section['offset'],string_section['size'])
            for pos in range(section['offset'],section['offset']+section['size'],16):
                name,address,size,info,other,index = self.unpack('<IIIBBH',pos)
                self.symbols.append(dict(name=self.string_at(strings,name),address=address,size=size,
                                         info=info,other=other,section_index=index))

    @staticmethod
    def string_at(data,pos):
        require(0 <= pos < len(data), 'Offset string invalido')
        end = data.find(b'\0',pos)
        require(end >= pos, 'String sin terminador')
        return data[pos:end].decode('utf-8',errors='strict')

    def range(self,pos,length):
        require(0 <= pos <= len(self.data) and 0 <= length <= len(self.data)-pos, 'Lectura fuera ELF')
        return self.data[pos:pos+length]

    def unpack(self,fmt,pos):
        return struct.unpack(fmt,self.range(pos,struct.calcsize(fmt)))

    def address(self,addr,length):
        matches = [s for s in self.sections if s['type'] != 8 and s['flags'] & 2
                   and s['address'] <= addr and addr+length <= s['address']+s['size']]
        require(len(matches)==1, f'Direccion no unica en seccion ELF: {addr:08x}+{length}')
        s = matches[0]
        return self.range(s['offset']+addr-s['address'],length)

    def symbol(self,name):
        matches = [s for s in self.symbols if s['name']==name and s['section_index']!=0]
        require(len(matches)==1,'Simbolo ausente o ambiguo: '+name)
        return matches[0]

    def cstring(self,addr,maximum):
        for i in range(maximum+1):
            if self.address(addr+i,1)==b'\0':
                return self.address(addr,i).decode('ascii')
        raise ValueError('String demasiado largo en ELF')


class FlatBuffer:
    """Lector limitado, con limites, para los campos usados del esquema TFLite3."""
    def __init__(self, data):
        self.data=data
        require(data[4:8]==b'TFL3','Identificador TFLite distinto')

    def unpack(self,fmt,pos):
        size=struct.calcsize(fmt)
        require(0 <= pos and pos+size <= len(self.data),'Offset FlatBuffer fuera de limites')
        return struct.unpack_from(fmt,self.data,pos)

    def field(self,table,vt):
        vtable=table-self.unpack('<i',table)[0]
        length=self.unpack('<H',vtable)[0]
        if vt+2 > length:
            return None
        offset=self.unpack('<H',vtable+vt)[0]
        return table+offset if offset else None

    def scalar(self,table,vt,fmt,default=0):
        field=self.field(table,vt)
        return self.unpack(fmt,field)[0] if field is not None else default

    def pointer(self,table,vt):
        field=self.field(table,vt)
        require(field is not None,'Campo puntero TFLite ausente')
        return field+self.unpack('<I',field)[0]

    def vector(self,table,vt,fmt):
        ptr=self.pointer(table,vt)
        count=self.unpack('<I',ptr)[0]
        require(count < len(self.data),'Vector TFLite excesivo')
        size=struct.calcsize(fmt)
        return [self.unpack(fmt,ptr+4+i*size)[0] for i in range(count)]

    def tables(self,table,vt):
        ptr=self.pointer(table,vt)
        count=self.unpack('<I',ptr)[0]
        require(count < len(self.data)//4,'Vector tablas excesivo')
        return [ptr+4+i*4+self.unpack('<I',ptr+4+i*4)[0] for i in range(count)]

    def tensor(self,table):
        q=self.pointer(table,12)
        scales=self.vector(q,8,'<I')
        zeros=self.vector(q,10,'<q')
        return dict(shape=self.vector(table,4,'<i'),type=self.scalar(table,6,'<b'),
                    scale_bits=[f'{v:08x}' for v in scales],zero_points=zeros)

    def validate(self,rep):
        root=self.unpack('<I',0)[0]
        require(self.scalar(root,4,'<I')==3,'Schema no es3')
        subgraphs=self.tables(root,8)
        require(len(subgraphs)==1,'Subgraphs distintos de1')
        sg=subgraphs[0];tensors=self.tables(sg,4)
        ins,outs=self.vector(sg,6,'<i'),self.vector(sg,8,'<i')
        require(len(ins)==len(outs)==1 and all(0<=i<len(tensors) for i in ins+outs),'I/O invalido')
        inp,out=self.tensor(tensors[ins[0]]),self.tensor(tensors[outs[0]])
        require(inp==dict(shape=[1,rep['dims']],type=9,
                         scale_bits=[rep['input_quantization']['scale_bits']],
                         zero_points=[rep['input_quantization']['zero_point']]),'Tensor entrada distinto')
        require(out==dict(shape=[1,3],type=9,scale_bits=[rep['output_quantization']['scale_bits']],
                         zero_points=[rep['output_quantization']['zero_point']]),'Tensor salida distinto')
        opcodes=[self.scalar(t,10,'<i',self.scalar(t,4,'<b')) for t in self.tables(root,6)]
        operators=self.tables(sg,10)
        indexes=[self.scalar(t,4,'<I') for t in operators]
        require(all(0<=i<len(opcodes) for i in indexes),'Opcode index fuera de rango')
        used=[opcodes[i] for i in indexes]
        require(used==[9,9,9,25],'Se esperan tres FULLY_CONNECTED y SOFTMAX')
        # Tensor dimensions establish the trained dense topology without running inference.
        dense_shapes=[]
        for op in operators[:3]:
            inputs=self.vector(op,6,'<i')
            require(len(inputs)==3 and all(0<=i<len(tensors) for i in inputs),'FullyConnected inputs distintos')
            shape=self.vector(tensors[inputs[1]],4,'<i')
            bias=self.vector(tensors[inputs[2]],4,'<i')
            dense_shapes.append(shape)
            require(bias==[shape[0]],'Bias denso incompatible')
        require(dense_shapes==[[20,rep['dims']],[10,20],[3,10]],'Topologia densa distinta')
        return dict(schema=3,subgraphs=1,input=inp,output=out,operator_builtin_codes=used,
                    dense_weight_shapes=dense_shapes,tensors=len(tensors),executed=False)


def main():
    require(not REPORT.exists(),'No sobrescribir auditoria ELF')
    started=time.perf_counter()
    accepted_mutations,generated_mutation=language_settings_mutation()
    prep_entries=verify_manifest(OUT,'MANIFEST_PREPARACION_TOTAL_SHA256.json',True,accepted_mutations)
    resource_entries=verify_manifest(RESOURCES,'MANIFEST_RECURSOS_SHA256.json',False)
    require(len(prep_entries)==1968 and len(resource_entries)==39,'Numero de recursos sellados distinto')
    # This verifier also confirms untouched D sources and fifteen original ELF.
    spec=importlib.util.spec_from_file_location('preparer_elf_audit',ROOT/'codigo/cwru/ampliacion_v2/costo_total/preparar_total.py')
    preparer=importlib.util.module_from_spec(spec);spec.loader.exec_module(preparer)
    prep=read(OUT/'PREPARACION.json');preparer.verify_prepared(prep)
    build=read(OUT/'COMPILACION_TOTAL.json');cat=read(RESOURCES/'CATALOGO_TOTAL.json')
    require(build['status']=='PASS_PENDING_ELF_MANUAL_REVIEW_AND_ARM' and build['returncode']==0
            and build['zero_errors_warnings'] and build['prepared_resources_unchanged'] and not build['timed_out'],
            'Build no aprobado')
    rows=build['configurations'];total=rows[CONFIG]
    require(all(r['flags_confirmed'] for r in rows.values())
            and all(rows[n]['control_elf_identical'] and rows[n]['control_flash_identical']
                    for n in ('Release_180MHz_EntM','Release_180MHz_Masks')),'Control o flags no iguales')
    require(sha(ELF)==total['elf_sha256']=='85e8d7e2d1848833117fd4704493710cda0058fecedef3b379a80e9560171b1c',
            'ELF distinto del build auditado')
    archive=OUT/'intentos/1/compilacion'/CONFIG/ELF.name
    require(sha(archive)==sha(ELF),'ELF archivado difiere')
    elf=ELF32(ELF)
    table=elf.symbol('cwru_total_catalog')
    stride=328
    require(table['size']==32*stride and table['address']%4==0,'Stride/size del descriptor ARM distinto')
    require(elf.cstring(elf.symbol('cwru_total_catalog_sha256')['address'],64)==cat['catalog_sha256'],
            'SHA canonico en ELF distinto')
    require(f'{struct.unpack("<I",elf.address(elf.symbol("cwru_total_catalog_fnv")["address"],4))[0]:08x}'==cat['catalog_fnv'],
            'FNV de catalogo ELF distinto')
    report_models=[];param_blobs=[];model_ranges=[]
    for rep in cat['representatives']:
        rid=rep['id'];address=table['address']+rid*stride
        desc=elf.address(address,stride)
        idptr,modelptr,length=struct.unpack_from('<III',desc)
        require(elf.cstring(idptr,100)==rep['representative_id'],'ID de descriptor distinto')
        require(length==rep['model_bytes'] and modelptr%16==0
                and 0x08000000<=modelptr<modelptr+length<=0x08080000,'Modelo/alineacion/rango incorrecto')
        require(list(desc[12:15])==[rep['mask_id'],rep['dims'],rep['det_dims']],'Mascara/dimensiones distintas')
        require(desc[41:44]==b'\0\0\0','Padding ABI inesperado')
        params=bytes([desc[13],desc[14],desc[12]])+desc[15:41]+desc[44:312]
        require(len(params)==297 and params.hex()==rep['params_bytes_hex'] and fnv(params)==rep['params_fnv'],
                'Parametros ELF distintos')
        actual_model=elf.address(modelptr,length)
        modelsha=hashlib.sha256(actual_model).hexdigest()
        require(actual_model==(RESOURCES/rep['model_path']).read_bytes()
                and modelsha==rep['model_sha256'] and fnv(actual_model)==rep['model_fnv'],'Bytes de modelo ELF distintos')
        mf,pf,shaptr,fitptr=struct.unpack_from('<IIII',desc,312)
        require(f'{mf:08x}'==rep['model_fnv'] and f'{pf:08x}'==rep['params_fnv']
                and elf.cstring(shaptr,64)==modelsha
                and elf.cstring(fitptr,64)==rep['source_fit_signature'],'Fingerprints o firmas en ELF distintas')
        matches=[s for s in elf.symbols if f'cwru_tv_model_{rid:02d}' in s['name']]
        require(len(matches)==1 and matches[0]['address']==modelptr and matches[0]['size']==length,
                'Simbolo modelo no corresponde al descriptor')
        topology=FlatBuffer(actual_model).validate(rep)
        report_models.append(dict(id=rid,representative_id=rep['representative_id'],descriptor_address=f'{address:08x}',
            model_address=f'{modelptr:08x}',model_bytes=length,model_sha256=modelsha,model_fnv=f'{mf:08x}',
            params_bytes=297,params_fnv=f'{pf:08x}',params_sha256=hashlib.sha256(params).hexdigest(),
            all_bytes_exact_exported_model=True,all_parameters_exact_exported_catalog=True,topology=topology))
        param_blobs.append(params);model_ranges.append((modelptr,modelptr+length))
    model_ranges.sort()
    require(all(a[1]<=b[0] for a,b in zip(model_ranges,model_ranges[1:])),'Modelos se superponen')
    buffers=[]
    for name,size in [('g_window',32768),('g_fft_input',16384),('g_fft_output',16384)]:
        sym=elf.symbol(name)
        require(sym['size']==size,'Buffer distinto: '+name)
        buffers.append(dict(name=name,address=sym['address'],bytes=size))
    large=[s for s in elf.symbols if s['size']>=16384 and 0x20000000<=s['address']<0x20020000]
    require(len(large)==4,'Cantidad de buffers grandes distinta de4')
    arena=[s for s in large if s['name'] not in ('g_window','g_fft_input','g_fft_output')]
    require(len(arena)==1 and arena[0]['size']==16384 and arena[0]['address']%16==0,'Arena unica/alineada incorrecta')
    buffers.append(dict(name=arena[0]['name'],address=arena[0]['address'],bytes=16384))
    ranges=sorted((b['address'],b['address']+b['bytes']) for b in buffers)
    require(all(0x20000000<=a<b<=0x20020000 for a,b in ranges)
            and all(a[1]<=b[0] for a,b in zip(ranges,ranges[1:])),'Buffers fuera SRAM/superpuestos')
    alloc=[s for s in elf.sections if s['flags']&2]
    ram=sum(s['size'] for s in alloc if 0x20000000<=s['address']<0x20020000)
    flash_segments=[s for s in elf.segments if s['type']==1 and 0x08000000<=s['physical']<0x08080000]
    flash_segment_bytes=sum(s['size'] for s in flash_segments)
    flash_bytes=sum(s['size'] for s in alloc if s['type']!=8)
    ram_span=max(s['address']+s['size'] for s in alloc if 0x20000000<=s['address']<0x20020000)-0x20000000
    flash_span=max(s['physical']+s['size'] for s in flash_segments)-0x08000000
    require(ram==total['ram_static_with_linker_reservations_bytes']==88460
            and flash_bytes==total['flash_load_bytes']==280332,'Memoria ELF difiere de size/build')
    require(flash_segment_bytes==flash_span==280352 and ram_span==88464,
            'Span/alineacion de memoria distintos')
    reservations={name:elf.symbol(name)['address'] for name in ('_Min_Heap_Size','_Min_Stack_Size','_estack','_end')}
    require(reservations['_Min_Heap_Size']==512 and reservations['_Min_Stack_Size']==4096,
            'Reservas linker cambiaron')
    result=dict(status='PASS_ELF_CATALOG_PARAMETERS_VERIFIED_PENDING_ARM',schema_version=1,
        date_local='2026-10-25',timezone='America/Lima',elf_path=str(ELF),elf_sha256=sha(ELF),
        archive_path=str(archive),archive_identical=True,build_report_sha256=sha(OUT/'COMPILACION_TOTAL.json'),
        build_seconds=build['seconds'],build_returncode=0,zero_errors_and_warnings=True,
        preparation_resources_verified=len(prep_entries),exported_resources_verified=len(resource_entries),
        preparation_resource_mutation=generated_mutation,
        old_D_source_files_verified=1867,old_D_ELFs_verified=15,control_EntM_and_Masks_ELF_and_FLASH_identical=True,
        compiler_flags_verified=True,descriptor_stride_bytes=stride,descriptor_abi_padding_bytes=3,
        descriptor_parameter_serialization_bytes=297,catalog_sha256=cat['catalog_sha256'],catalog_fnv=cat['catalog_fnv'],
        model_count=32,model_payload_bytes=sum(r['model_bytes'] for r in report_models),
        models=report_models,buffers=buffers,one_reused_arena=True,
        memory=dict(flash_load_bytes=flash_bytes,flash_capacity_bytes=524288,flash_percent=100*flash_bytes/524288,
                    flash_program_segment_bytes=flash_segment_bytes,flash_program_address_span_bytes=flash_span,
                    flash_alignment_padding_bytes=flash_segment_bytes-flash_bytes,
                    ram_static_including_linker_reservations_bytes=ram,ram_capacity_bytes=131072,
                    ram_percent=100*ram/131072,ram_address_span_bytes=ram_span,
                    ram_address_alignment_gap_bytes=ram_span-ram,ram_unreserved_address_bytes=131072-ram_span,
                    heap_reservation_bytes=512,stack_reservation_bytes=4096,
                    linker_symbols=reservations,arena_runtime_used_pending=True,
                    stack_high_water_or_WCET_claimed=False,flash_is_32_model_test_harness_not_final_candidate=True),
        sources_modified=False,D_files_modified=False,new_builds_executed=0,TFLite_invocations=0,
        serial_port_opened=False,hardware_flashed=False,selector_costs_enabled=False,
        audit_script_sha256=sha(__file__),seconds=time.perf_counter()-started,
        limitations=['Static ELF audit does not establish programmed image identity on board.',
                     'Runtime AllocateTensors/arena usage and output equivalence remain pending physical gate.',
                     'Stack reservation and .su static frames do not measure runtime maximum stack.',
                     'Flash belongs to a harness holding32 models, not a final one-model deployment.',
                     'No full-pipeline timing or WCET measured by this audit.'])
    with REPORT.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({k:result[k] for k in ('status','elf_sha256','model_count','model_payload_bytes','memory','seconds')},
                     ensure_ascii=False))


if __name__=='__main__':
    main()
