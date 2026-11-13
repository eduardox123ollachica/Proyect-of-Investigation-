"""Calibración UART SR de los32 pipelines P0; no entrena ni programa la placa.

--gate: control rep0/gid568 y regresión rep22/gid232; seis RUN, dos PROFILE,
dos BASE y17 rechazos. Single rounding es una declaración de configuración;
UART no verifica por sí solo toda la imagen de Flash o esa opción de build.
--full: requiere --gate-evidence SR PASS de este catálogo/ELF; 42 BASE,
2688 RUN principales y896 PROFILE diagnósticos. Todo permanece provisional
hasta auditoría de trazas, firmware y banco. UART, BASE, cambio de modelo y
PROFILE quedan fuera del selector. Replay PC usa las features PUBLICADAS por
ARM; las diferencias frente al caché PC original se informan por separado.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import re
import shutil
import struct
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
HCLK = 180000000
REPETITIONS = 3
ORDER = ('F14','T6','S8','NF12','NM10','NE12','DE7','FE7')
MASKS = (0x3fff,0x0387,0x3c78,0x37ef,0x3cf9,0x1fbf,0x007f,0x3f80)
CANONICAL_KEYS = ('id','representative_id','model_sha256','source_fit_signature','params_bytes_hex','scope')
CONFIGURATION_DECLARED = 'Release_180MHz_TotalV2_SR'
NEW_ROUNDING_MODE_DECLARED = 'single'
UPSTREAM_CLIENT_SHA256 = '6c88ad0b09226673de3ddc675336153c41ad5ae5a47e7e412a1194325fe5b6d0'
UPSTREAM_CATALOG_SHA256 = '1807ea8de795d8c100f2a81f6685b6bd5784dbb986aef7ce4a16fca6837072f0'
UPSTREAM_RESOURCE_MANIFEST_SHA256 = '1bad5ef5cb6aec115ee5ac91edcb77639534d4e2d395a57db49fba11f5b4d5e1'
GATE_CASES = ((568,(0,)),(232,(22,)))
GATE_STATUS = 'SR_GATE_PASS_NOT_TOTAL_COST_VALIDATED'


def gate_plan():
    return [[gid,list(ids)] for gid,ids in GATE_CASES]


def rounding_provenance():
    return dict(configuration_declared=CONFIGURATION_DECLARED,new_rounding_mode=NEW_ROUNDING_MODE_DECLARED,
        rounding_mode_verified_by_UART=False,model_weights_unchanged_from_upstream=True,
        upstream_client_sha256=UPSTREAM_CLIENT_SHA256,upstream_catalog_sha256=UPSTREAM_CATALOG_SHA256,
        upstream_resource_manifest_sha256=UPSTREAM_RESOURCE_MANIFEST_SHA256)



def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def fnv(data):
    state = 2166136261
    for byte in data:
        state = ((state ^ byte) * 16777619) & 0xffffffff
    return f'{state:08x}'


def dec(text, minimum=0, maximum=0xffffffff):
    require(re.fullmatch(r'[0-9]+', text) is not None, 'Entero decimal inválido')
    value = int(text)
    require(minimum <= value <= maximum, 'Entero fuera de rango')
    return value


def signed(text, minimum=-128, maximum=127):
    require(re.fullmatch(r'-?[0-9]+', text) is not None, 'Entero con signo inválido')
    value = int(text)
    require(minimum <= value <= maximum, 'Entero con signo fuera de rango')
    return value


def hext(text, size):
    require(re.fullmatch('[0-9a-fA-F]{'+str(size)+'}', text) is not None, 'Hexadecimal inválido')
    return text.lower()


def bits(values):
    return [f'{int(v):08x}' for v in np.asarray(values,dtype='<f4').view('<u4').ravel()]


def from_bits(values):
    result = np.asarray([int(hext(v,8),16) for v in values],dtype='<u4').view('<f4')
    require(np.isfinite(result).all(), 'Float32 no finito')
    return result


def hex_floats(text, count):
    result = np.frombuffer(bytes.fromhex(hext(text,count*8)),dtype='<f4')
    require(np.isfinite(result).all(), 'Página con float32 no finito')
    return bits(result)


def params_bytes(rep):
    d,nd = rep['dims'],rep['det_dims']
    def pad(values,n,fill):
        a = np.full(n,fill,dtype='<f4'); a[:len(values)] = from_bits(values); return a.tobytes()
    result = struct.pack('<BBB',d,nd,rep['mask_id'])
    result += bytes(rep['feature_indices']+[0]*(14-d))
    result += bytes(rep['detector_indices']+[0]*(12-nd))
    for key,n,fill in [('dnn_mean32_bits',14,0),('dnn_scale32_bits',14,1),
                       ('detector_mean32_bits',12,0),('detector_scale32_bits',12,1),('centroid32_bits',12,0)]:
        result += pad(rep[key],n,fill)
    result += from_bits([rep['threshold_bits'],rep['input_quantization']['scale_bits']]).tobytes()
    return result+struct.pack('<i',rep['input_quantization']['zero_point'])


def contained(folder, relative):
    target = (folder/relative).resolve()
    require(target.is_relative_to(folder.resolve()) and target.is_file(), 'Ruta de recurso inválida')
    return target


def load_resources(folder):
    folder = folder.resolve()
    mp = folder/'MANIFEST_RECURSOS_SHA256.json'
    digest,name = mp.with_suffix('.sha256').read_text(encoding='ascii').split()
    require(name == mp.name and sha(mp) == hext(digest,64) == UPSTREAM_RESOURCE_MANIFEST_SHA256,
            'Sello de recursos distinto del banco P0 original')
    manifest = read(mp)
    require(isinstance(manifest,dict) and len(manifest)>=38, 'Manifiesto incompleto')
    for relative,expected in manifest.items():
        require(sha(contained(folder,relative)) == hext(expected,64), 'Recurso modificado: '+relative)
    require({'CATALOGO_TOTAL.json','CATALOGO_CANONICO.json','BANCO_TOTAL.json','ventanas_banco.npz'} <= manifest.keys(),
            'El manifiesto no cubre catálogo y banco')
    cat,bank = read(folder/'CATALOGO_TOTAL.json'),read(folder/'BANCO_TOTAL.json')
    require(cat['schema_version']==bank['schema_version']==1 and cat['count']==32
            and len(cat['representatives'])==32 and bank['unique_windows']==42
            and bank['repetitions']==3 and bank['total_planned_RUN']==2688, 'Dimensiones de recursos distintas')
    reps=cat['representatives']; canonical=[]; blob=b''
    for i,r in enumerate(reps):
        require(r['id']==i and r['mask_id']==i%8 and r['candidate']==ORDER[i%8]
                and r['outer_load_hp']==i//8, 'Orden de representantes distinto')
        indices=[j for j in range(14) if MASKS[i%8] & (1<<j)]
        require(r['feature_indices']==indices and r['dims']==len(indices)
                and r['detector_indices']==[j for j in indices if j not in (4,11)]
                and r['det_dims']==len(r['detector_indices']), 'Máscara/normalidad distinta')
        model_path=contained(folder,r['model_path'])
        require(r['model_path'] in manifest and sha(model_path)==r['model_sha256']
                and model_path.stat().st_size==r['model_bytes'] and fnv(model_path.read_bytes())==r['model_fnv'],
                'Modelo no autenticado')
        packed=params_bytes(r)
        require(len(packed)==297 and packed.hex()==r['params_bytes_hex'] and fnv(packed)==r['params_fnv'],
                'Parámetros no autenticados')
        require(all(v>0 for key in ('dnn_scale32_bits','detector_scale32_bits') for v in from_bits(r[key]))
                and from_bits([r['input_quantization']['scale_bits']])[0]>0, 'Escala no positiva')
        canonical.append({k:r[k] for k in CANONICAL_KEYS})
        blob+=struct.pack('<II',int(r['model_fnv'],16),int(r['params_fnv'],16))
    canonical_bytes=json.dumps(canonical,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    require((folder/'CATALOGO_CANONICO.json').read_bytes()==canonical_bytes
            and hashlib.sha256(canonical_bytes).hexdigest()==cat['catalog_sha256']
            and fnv(blob)==cat['catalog_fnv'] and cat['catalog_sha256']==UPSTREAM_CATALOG_SHA256,
            'Identidad canónica distinta del banco P0 original')
    require(sha(folder/'ventanas_banco.npz')==bank['windows_npz_sha256'], 'NPZ distinto del banco')
    with np.load(folder/'ventanas_banco.npz',allow_pickle=False) as data:
        windows,ids=data['windows'].copy(),data['global_window_ids'].tolist()
    require(windows.dtype==np.dtype('<f4') and windows.shape==(42,4096,2)
            and np.isfinite(windows).all() and len(set(ids))==42, 'Ventanas inválidas')
    rows=bank['windows']; require(len(rows)==42 and [r['global_id'] for r in rows]==ids,'IDs NPZ distintos')
    for pos,r in enumerate(rows):
        raw=windows[pos].tobytes()
        require(r['npz_index']==pos and r['payload_sha256']==hashlib.sha256(raw).hexdigest()
                and r['checksum_fnv1a']==fnv(raw), 'Payload no corresponde a metadatos')
        require(len(r['features_pc_cache14_bits'])==14 and np.isfinite(from_bits(r['features_pc_cache14_bits'])).all(),
                'Referencia de features incompleta')
    require(len(bank['banks'])==4,'Se necesitan cuatro bancos')
    by_id={r['global_id']:r for r in rows}
    for outer,b in enumerate(bank['banks']):
        require(b['outer_load_hp']==outer and b['representative_ids']==list(range(outer*8,(outer+1)*8))
                and len(b['global_ids'])==len(set(b['global_ids']))==28 and b['repetitions']==3,
                'Banco de outer incompleto')
        for rid in b['representative_ids']:
            require(reps[rid]['scope']['benchmark_global_window_ids']==b['global_ids'], 'Banco cambia por máscara')
            require(set(bank['reference_by_rep_gid'][str(rid)])==set(map(str,b['global_ids'])), 'Referencias PC incompletas')
        for gid in b['global_ids']:
            require(int(by_id[gid]['metadata']['load_hp']) in b['training_loads']
                    and int(by_id[gid]['metadata']['load_hp'])!=outer, 'Uso indebido de carga exterior')
    for r in rows:
        expected=[i for i,rep in enumerate(reps) if r['global_id'] in rep['scope']['benchmark_global_window_ids']]
        require(r['authorized_rep_ids']==expected, 'Autorizaciones por ventana distintas')
    return cat,bank,windows,manifest


def schedule(bank, full):
    """Orden topológico que conserva el orden canónico de cada banco."""
    if not full:
        by_id={row['global_id']:row for row in bank['windows']}
        for gid,ids in GATE_CASES:
            require(gid in by_id, 'Falta ventana de control o regresión SR')
            for rid in ids:
                require(gid in bank['banks'][rid//8]['global_ids']
                        and rid in by_id[gid]['authorized_rep_ids'], 'Caso SR fuera del banco autorizado')
        return [(gid,list(ids)) for gid,ids in GATE_CASES]
    lists=[b['global_ids'] for b in bank['banks']]
    rank={gid:i for i,gid in enumerate(dict.fromkeys(g for row in lists for g in row))}
    edges={g:set() for g in rank}; indegree={g:0 for g in rank}
    for row in lists:
        for a,b in zip(row,row[1:]):
            if b not in edges[a]: edges[a].add(b);indegree[b]+=1
    ordered=[]
    while len(ordered)<len(rank):
        ready=[g for g in rank if indegree[g]==0 and g not in ordered]
        require(ready,'El orden de bancos tiene un ciclo; no reordenar silenciosamente')
        g=min(ready,key=rank.get);ordered.append(g)
        for child in edges[g]:indegree[child]-=1
    rows={r['global_id']:r for r in bank['windows']}
    result=[(g,rows[g]['authorized_rep_ids']) for g in ordered]
    require(len(result)==42 and sum(len(ids) for _,ids in result)==896,'Calendario incompleto')
    return result


def replay(features_bits,rep):
    f=from_bits(features_bits)
    x=f[rep['feature_indices']]
    z=np.divide(np.subtract(x,from_bits(rep['dnn_mean32_bits']),dtype=np.float32),
                from_bits(rep['dnn_scale32_bits']),dtype=np.float32)
    affine=np.add(np.divide(z,from_bits([rep['input_quantization']['scale_bits']])[0],dtype=np.float32),
                  np.float32(rep['input_quantization']['zero_point']),dtype=np.float32)
    q=np.clip(np.rint(affine),-128,127).astype(np.int8)
    low=sum(1<<i for i,v in enumerate(affine) if v<np.float32(-128.5))
    high=sum(1<<i for i,v in enumerate(affine) if v>=np.float32(127.5))
    dz=np.divide(np.subtract(f[rep['detector_indices']],from_bits(rep['detector_mean32_bits']),dtype=np.float32),
                 from_bits(rep['detector_scale32_bits']),dtype=np.float32)
    difference=np.subtract(dz,from_bits(rep['centroid32_bits']),dtype=np.float32)
    terms=np.multiply(difference,difference,dtype=np.float32)
    score=np.float32(0)
    for term in terms:score=np.add(score,term,dtype=np.float32)
    require(np.isfinite(z).all() and np.isfinite(affine).all() and np.isfinite(dz).all()
            and np.isfinite(terms).all() and np.isfinite(score),'Replay produjo un intermedio no finito')
    return dict(standardized_bits=bits(z),affine_bits=bits(affine),quantized=q.astype(int).tolist(),
                saturated_low=low,saturated_high=high,detector_z_bits=bits(dz),detector_terms_bits=bits(terms),
                score_bits=bits([score])[0],rejected=bool(score>from_bits([rep['threshold_bits']])[0]))


class LiteModels:
    def __init__(self,folder,reps):
        import tensorflow as tf
        self.tf=tf;self.folder=folder;self.reps=reps;self.cache={}

    def predict(self,rid,q):
        if rid not in self.cache:
            rep=self.reps[rid]; tf=self.tf
            it=tf.lite.Interpreter(model_content=(self.folder/rep['model_path']).read_bytes(),num_threads=1,
                experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
            it.allocate_tensors(); inp=it.get_input_details()[0];out=it.get_output_details()[0]
            require(list(inp['shape'])==[1,rep['dims']] and list(out['shape'])==[1,3]
                    and inp['dtype']==out['dtype']==np.int8,'I/O TFLite incompatible')
            require(bits([inp['quantization'][0]])==[rep['input_quantization']['scale_bits']]
                    and int(inp['quantization'][1])==rep['input_quantization']['zero_point']
                    and bits([out['quantization'][0]])==[rep['output_quantization']['scale_bits']]
                    and int(out['quantization'][1])==rep['output_quantization']['zero_point'], 'Cuantización Lite distinta')
            require(not any(o['op_name'] in ('DELEGATE','DEQUANTIZE','READ_VARIABLE','VAR_HANDLE')
                            or 'Flex' in o['op_name'] for o in it._get_ops_details()),'Fallback Lite no permitido')
            self.cache[rid]=(it,inp['index'],out['index'])
        it,inp,out=self.cache[rid];it.set_tensor(inp,np.asarray([q],dtype=np.int8));it.invoke()
        result=it.get_tensor(out)[0];require(result.shape==(3,) and result.dtype==np.int8,'Salida Lite inválida')
        return result.astype(int).tolist()


class Trace:
    def __init__(self,path):self.stream=Path(path).open('x',encoding='utf-8',newline='\n');self.count=0
    def append(self,item):
        self.count+=1;self.stream.write(json.dumps(dict(seq=self.count,**item),ensure_ascii=False,allow_nan=False)+'\n');self.stream.flush()
    def close(self):self.stream.close()


class Link:
    """read() acumulado: un timeout parcial nunca se confunde con una línea."""
    def __init__(self,port,trace,timeout=15.,deadline=None):
        self.port=port;self.trace=trace;self.timeout=timeout;self.pending=bytearray();self.deadline=deadline

    def ask(self,command,prefix,allow_error=False):
        require(self.deadline is None or time.monotonic()<self.deadline,'Presupuesto global de ejecución agotado')
        payload=(command+'\r\n').encode('ascii');require(len(payload)<1024,'Trama TX excede1024')
        self.trace.append(dict(direction='TX',command=command,utc=datetime.now(timezone.utc).isoformat()))
        require(self.port.write(payload)==len(payload),'Escritura UART parcial');self.port.flush()
        deadline=time.monotonic()+self.timeout
        if self.deadline is not None:deadline=min(deadline,self.deadline)
        last=[]
        while time.monotonic()<deadline:
            if b'\n' not in self.pending:
                chunk=self.port.read(max(1,min(int(getattr(self.port,'in_waiting',0)),512)))
                self.pending.extend(chunk)
                require(len(self.pending)<4096,'Recepción sin límite de línea')
                if b'\n' not in self.pending:continue
            raw,_,rest=self.pending.partition(b'\n');self.pending=bytearray(rest);raw+=b'\n'
            require(raw.endswith(b'\r\n') and len(raw)<512,'Respuesta truncada o excede512')
            line=raw[:-2].decode('ascii',errors='strict')
            if not line:continue
            self.trace.append(dict(direction='RX',line=line,utc=datetime.now(timezone.utc).isoformat()));last.append(line)
            if '_ERROR,' in line:
                if allow_error:return line
                raise ValueError(line)
            if line.startswith(prefix):return line
        raise TimeoutError(f'No llegó {prefix}; últimas respuestas={last[-5:]}')


def parse_info(line,cat):
    f=line.split(',');require(len(f)==17 and f[0]=='CWRUTOTAL_INFO','INFO incompleto')
    require([dec(v) for v in f[1:8]]==[1,32,12000,4096,2,HCLK,1],'INFO/reloj/DWT incompatible')
    fpscr=int(hext(f[8],8),16)
    require(not fpscr&((3<<22)|(1<<24)),'FPSCR: nearest-even y FTZoff requeridos')
    require(hext(f[9],8)==cat['catalog_fnv'] and hext(f[10],64)==cat['catalog_sha256'],'Catálogo firmware distinto')
    selected=signed(f[11],-1,31);valid=dec(f[12],0,1);last=dec(f[13]);overhead=dec(f[14],1)
    require((not valid or last>0) and f[15:]==['64','0'],'INFO/calibración incompatible')
    return dict(hclk_hz=HCLK,selected_id=selected,result_valid=bool(valid),last_run_id=last,
                fpscr=f[8].lower(),paired_read_cycles_min=overhead,calibration_samples=64,overhead_subtracted=False,
                nominal_hal_clock_not_externally_measured=True,catalog_sha256=f[10],catalog_fnv=f[9])


def validate_model(line,rep):
    f=line.split(',');require(len(f)==13 and f[0]=='CWRUTOTAL_MODEL','MODEL incompleto')
    require([dec(v) for v in f[1:6]]==[rep['id'],rep['mask_id'],rep['dims'],rep['det_dims'],rep['model_bytes']]
            and [hext(v,8) for v in f[6:9]]==[rep['model_fnv'],rep['params_fnv'],rep['input_quantization']['scale_bits']]
            and signed(f[9])==rep['input_quantization']['zero_point'] and hext(f[10],8)==rep['threshold_bits']
            and hext(f[11],64)==rep['model_sha256'] and hext(f[12],64)==rep['source_fit_signature'], 'MODEL distinto del recurso')


def parse_selected(line,rep):
    f=line.split(',');require(len(f)==8 and f[0]=='CWRUTOTAL_SELECTED','SELECT incompleto')
    require(dec(f[1],0,31)==rep['id'] and f[4]=='0' and f[5] in ('0','1')
            and hext(f[6],8)==rep['model_fnv'] and hext(f[7],8)==rep['params_fnv'],'SELECT no autentica modelo/parámetros')
    cycles,arena,reused=dec(f[2]),dec(f[3],1,16384),bool(int(f[5]))
    require(cycles==0 if reused else cycles>0,'SELECT ciclos/reutilización incompatible')
    return dict(rep_id=rep['id'],cycles=cycles,arena_used=arena,reused=reused,excluded_from_primary_cost=True)


def frames(window):
    values=np.asarray(window,dtype='<f4').ravel()
    require(values.shape==(8192,) and np.isfinite(values).all(),'Payload inválido')
    for offset in range(0,len(values),48):
        block=values[offset:offset+48]
        command=f'CWRUFEAT_DATA,{offset},'+','.join(format(float(v),'.9g') for v in block)
        require(len(command)+2<1024 and np.asarray(command.split(',')[2:],dtype='<f4').tobytes()==block.tobytes(),
                'Serialización no conserva float32')
        yield command,offset+len(block)


def transmit(link,window,row):
    require(link.ask('CWRUFEAT_BEGIN','CWRUFEAT_READY,')=='CWRUFEAT_READY,8192','BEGIN incompatible')
    for command,count in frames(window):
        require(link.ask(command,'CWRUFEAT_ACK,')==f'CWRUFEAT_ACK,{count}','ACK incorrecto')
    require(link.ask('CWRUFEAT_CHECK','CWRUFEAT_CHECK,')==f'CWRUFEAT_CHECK,8192,{row["checksum_fnv1a"]}', 'CHECK distinto')


def baseline(link,row):
    f=link.ask('CWRUMASK_BASE,'+row['checksum_fnv1a'],'CWRUMASK_BASE,').split(',')
    require(len(f)==4 and hext(f[1],8)==row['checksum_fnv1a'],'BASE incompleto/de otra ventana')
    cycles=dec(f[2],1);observed=hex_floats(f[3],14);x=from_bits(observed);ref=from_bits(row['features_pc_cache14_bits'])
    require(np.allclose(x,ref,rtol=.001,atol=.00001) and np.array_equal(x[[4,11]],ref[[4,11]]),
            'BASE no cumple tolerancia PC o frecuencia dominante exacta')
    known=row['arm_features14_bits']
    require(known is None or known==observed,'BASE distinta de referencia ARM histórica')
    return dict(global_id=row['global_id'],cycles=cycles,features_bits=observed,
                compared_to_PC_cache_with_tolerance=True,known_ARM_bit_exact=(known is not None),
                new_ARM_window=(known is None),excluded_from_primary_cost=True,
                max_abs_error_PC_cache=float(np.max(np.abs(x.astype(float)-ref.astype(float)))))


def parse_result(line,rep,row,last,mode,arena):
    f=line.split(',');require(len(f)==19 and f[0]=='CWRUTOTAL_RESULT','RESULT incompleto')
    expected=1 if last==0xffffffff else last+1
    require(dec(f[1],1)==expected and dec(f[2],0,31)==rep['id'] and hext(f[3],8)==row['checksum_fnv1a']
            and dec(f[4],0,1)==mode,'RESULT ID/modelo/checksum/modo distinto')
    costs=[dec(v) for v in f[5:11]];total=costs[0]
    require(total>0 and (sum(costs[1:])==total if mode else costs[1:]==[0]*5),'Instrumentación RESULT inválida')
    output=[signed(v) for v in f[12:15]];cls=dec(f[11],0,2)
    require(cls==int(np.argmax(output)) and dec(f[17],1,16384)==arena and dec(f[18])==1,
            'RESULT clase,arena o extracción múltiple incorrectos')
    score=hext(f[15],8);require(from_bits([score])[0]>=0,'Distancia negativa')
    return dict(run_id=expected,rep_id=rep['id'],global_id=row['global_id'],checksum=row['checksum_fnv1a'],
                mode=mode,total_cycles=total,extract_cycles=costs[1],input_cycles=costs[2],nn_cycles=costs[3],
                detector_cycles=costs[4],residual_cycles=costs[5],milliseconds=total/180000.,
                predicted_class=cls,output_int8=output,score_bits=score,rejected=bool(dec(f[16],0,1)),
                arena_used=arena,extractions=1)


def get_pages(link,result,rep):
    pages={}
    for page,length in [('FEATURES',5),('INPUT',9),('DETECTOR',8)]:
        f=link.ask(f'CWRUTOTAL_{page},{result["run_id"]}',f'CWRUTOTAL_{page},').split(',')
        require(len(f)==length and dec(f[1],1)==result['run_id'] and dec(f[2],0,31)==rep['id']
                and hext(f[3],8)==result['checksum'],'Página despareada: '+page)
        pages[page]=f
    a,b,c=pages['FEATURES'],pages['INPUT'],pages['DETECTOR'];d,nd=rep['dims'],rep['det_dims']
    q=np.frombuffer(bytes.fromhex(hext(b[6],d*2)),dtype=np.int8).astype(int).tolist()
    return dict(features_bits=hex_floats(a[4],14),standardized_bits=hex_floats(b[4],d),affine_bits=hex_floats(b[5],d),
                quantized=q,saturated_low=int(hext(b[7],4),16),saturated_high=int(hext(b[8],4),16),
                detector_z_bits=hex_floats(c[4],nd),detector_terms_bits=hex_floats(c[5],nd),
                score_bits=hext(c[6],8),rejected=bool(dec(c[7],0,1)))


def compare(result,pages,base,rep,pc_ref,lite):
    expected=[base['features_bits'][i] if i in rep['feature_indices'] else '00000000' for i in range(14)]
    require(pages['features_bits']==expected,'Kernel no exacto a BASE o slot inactivo no +0')
    reference=replay(pages['features_bits'],rep)
    require(all(pages[k]==v for k,v in reference.items()),'Replay float32 distinto de intermedios ARM')
    require(result['score_bits']==pages['score_bits'] and result['rejected']==pages['rejected'],'Distancia/publicación despareada')
    output=lite.predict(rep['id'],pages['quantized'])
    require(output==result['output_int8'],'Salida INT8 STM32 distinta de PC para LA MISMA entrada')
    # Cached reference remains an independent numerical comparison, not the ARM oracle.
    cached_output=lite.predict(rep['id'],pc_ref['int8_inputs'])
    require(cached_output==pc_ref['int8_outputs'],'Modelo PC no reproduce el banco INT8 sellado')
    return dict(ARM_features_replay_all_intermediates_bit_exact=True,same_INT8_input_output_exact=True,
        selected_features_bit_exact_BASE=True,inactive_features_positive_zero=True,
        cached_PC=dict(feature_bit_differences=sum(a!=b for a,b in zip(base['features_bits'],pc_ref['raw_features14_bits'])),
            quantized_value_differences=sum(a!=b for a,b in zip(pages['quantized'],pc_ref['int8_inputs'])),
            output_value_differences=sum(a!=b for a,b in zip(output,pc_ref['int8_outputs'])),
            class_differs=result['predicted_class']!=int(np.argmax(pc_ref['int8_outputs'])),
            detector_score_bit_differs=result['score_bits']!=pc_ref['detector_score_bits'],
            detector_rejection_differs=result['rejected']!=pc_ref['outside_normality']),
        cached_PC_end_to_end_bit_exact_claimed=False)


def negatives(link,result,rep):
    """17 rechazos, cero RUN válidos extra. Fases y apoyos fuera de coste."""
    rows=[];rid=result['run_id']; stale=rid+1 if rid<0xffffffff else 1
    def error(command,reason,detail=-1):
        actual=link.ask(command,'CWRUTOTAL_ERROR,',allow_error=True)
        require(actual==f'CWRUTOTAL_ERROR,{reason},{detail}','Rechazo inesperado: '+actual)
        rows.append(dict(command=command,response=actual,status='PASS',excluded_from_primary_cost=True))
    original=link.ask(f'CWRUTOTAL_FEATURES,{rid}','CWRUTOTAL_FEATURES,')
    for command,reason in [('CWRUTOTAL_INFO,1','FORMAT'),('CWRUTOTAL_UNKNOWN','UNKNOWN'),
                          ('CWRUTOTAL_MODEL,abc','FORMAT'),('CWRUTOTAL_MODEL,32','REP_ID'),
                          ('CWRUTOTAL_FEATURES,0','FORMAT'),(f'CWRUTOTAL_FEATURES,{stale}','STALE_ID'),
                          (f'CWRUTOTAL_FEATURES,{rid},1','FORMAT')]:
        error(command,reason)
        require(link.ask(f'CWRUTOTAL_FEATURES,{rid}','CWRUTOTAL_FEATURES,')==original,'Consulta inválida destruyó publicación')
    error('CWRUTOTAL_RUN,123','FORMAT');error(f'CWRUTOTAL_FEATURES,{rid}','NO_RESULT')
    wrong=f'{int(result["checksum"],16)^1:08x}';error('CWRUTOTAL_RUN,'+wrong,'CHECKSUM')
    error('CWRUTOTAL_SELECT,32','REP_ID');error('CWRUTOTAL_SELECT,abc','FORMAT')
    support=parse_selected(link.ask(f'CWRUTOTAL_SELECT,{rep["id"]}','CWRUTOTAL_SELECTED,'),rep)
    require(link.ask('CWRUFEAT_BEGIN','CWRUFEAT_READY,')=='CWRUFEAT_READY,8192','BEGIN incompatible')
    error(f'CWRUTOTAL_FEATURES,{rid}','NO_RESULT');error('CWRUTOTAL_RUN,'+result['checksum'],'INCOMPLETE',0)
    require(link.ask('CWRUFEAT_DATA,0,0.25,-0.5,1,-1','CWRUFEAT_ACK,')=='CWRUFEAT_ACK,4','Apoyo DATA falló')
    expected=f'CWRUFEAT_CHECK,4,{fnv(np.asarray([.25,-.5,1,-1],dtype="<f4").tobytes())}'
    for command,reason,detail in [('CWRUFEAT_DATA,3,1','OFFSET',4),('CWRUFEAT_DATA,4,5,nan','BAD_VALUE',1),
                                   ('CWRUFEAT_DATA,4,'+','.join(['0']*49),'FRAME_VALUES',48)]:
        actual=link.ask(command,'CWRUFEAT_ERROR,',allow_error=True)
        require(actual==f'CWRUFEAT_ERROR,{reason},{detail}','Rechazo DATA inesperado')
        require(link.ask('CWRUFEAT_CHECK','CWRUFEAT_CHECK,')==expected,'Trama inválida alteró commit')
        rows.append(dict(command=command,response=actual,status='PASS',excluded_from_primary_cost=True))
    require(len(rows)==17,'Suite negativa incompleta')
    return rows,support


def statistics(rows,reps):
    result=[]
    for rep in reps:
        selected=[r for r in rows if r['mode']==0 and r['rep_id']==rep['id']]
        if not selected:continue
        values=np.asarray([r['total_cycles'] for r in selected],dtype=float)
        result.append(dict(rep_id=rep['id'],candidate=rep['candidate'],outer_load_hp=rep['outer_load_hp'],
            count=len(selected),min_cycles=int(values.min()),max_cycles=int(values.max()),mean_cycles=float(values.mean()),
            mean_ms=float(values.mean()/180000),median_ms=float(np.median(values)/180000),
            p95_ms=float(np.percentile(values,95)/180000),sample_std_ms=float(values.std(ddof=1)/180000),
            PROFILE_included=False,BASE_included=False,initialization_included=False,UART_included=False,
            WCET_claimed=False,selector_enabled=False))
    return result


def verify_gate(folder,cat,firmware_sha,resource_manifest_sha):
    report=read(folder/'RESULTADO_TOTAL.json');mp=folder/'MANIFEST_EVIDENCIAS_SHA256.json'
    digest,name=mp.with_suffix('.sha256').read_text(encoding='ascii').split()
    require(name==mp.name and sha(mp)==digest,'Sello puerta inválido')
    entries=read(mp)
    require({'RESULTADO_TOTAL.json','firmware_declarado.elf','validar_total_ejecutado.py',
             'TRAZA_UART.jsonl','MEDICIONES.jsonl','PLAN_EJECUTADO.json'} <= entries.keys(), 'Sello puerta incompleto')
    for relative,expected in entries.items():
        require(sha(contained(folder,relative))==expected,'Evidencia puerta modificada')
    plan=read(folder/'PLAN_EJECUTADO.json')
    expected=gate_plan()
    require(report['status']==GATE_STATUS and report['mode']=='gate'
            and report['catalog_sha256']==cat['catalog_sha256']==UPSTREAM_CATALOG_SHA256
            and report['firmware_elf_sha256']==firmware_sha and report['completed_primary_RUN']==6
            and report['completed_PROFILE']==2 and report['completed_BASE']==2 and report['negative_rejections']==17
            and report['physical_unique_window_ids']==[568,232] and report['gate_cases']==expected
            and report['model_selection_count']==2
            and report['resource_manifest_sha256']==resource_manifest_sha==UPSTREAM_RESOURCE_MANIFEST_SHA256
            and report['client_sha256']==sha(__file__) and sha(folder/'firmware_declarado.elf')==firmware_sha
            and sha(folder/'validar_total_ejecutado.py')==sha(__file__),
            'Puerta SR no habilita esta campaña/ELF/cliente; se necesitan control y regresión')
    require(plan['mode']=='gate' and plan['schedule']==expected
            and plan['expected_primary_RUN']==6 and plan['expected_PROFILE']==2 and plan['expected_BASE']==2
            and plan['firmware_elf_sha256']==firmware_sha
            and plan['resource_manifest_sha256']==resource_manifest_sha,
            'Plan de puerta SR no corresponde a los dos casos y recursos esperados')
    for key,value in rounding_provenance().items():
        require(report.get(key)==value and plan.get(key)==value,'Procedencia SR de puerta incompatible: '+key)
    statistics_by_rep={r['rep_id']:r for r in report['timing_statistics']}
    require(len(report['timing_statistics'])==len(statistics_by_rep)==2
            and set(statistics_by_rep)=={0,22} and all(r['count']==3 for r in statistics_by_rep.values()),
            'La puerta SR debe tener tres tiempos principales por caso')


def run(args):
    resources=args.resources.resolve();cat,bank,windows,manifest=load_resources(resources)
    firmware=args.firmware_elf.resolve();require(firmware.is_file() and firmware.suffix.lower()=='.elf','ELF requerido')
    header=firmware.read_bytes()[:20]
    require(len(header)==20 and header[:6]==b'\x7fELF\x01\x01' and struct.unpack('<HH',header[16:20])==(2,40),
            'Se requiere un ELF32 little-endian ejecutable para ARM')
    firmware_sha=sha(firmware)
    if args.full:
        require(args.gate_evidence is not None,'--full requiere --gate-evidence de puerta PASS')
        verify_gate(args.gate_evidence.resolve(),cat,firmware_sha,sha(resources/'MANIFEST_RECURSOS_SHA256.json'))
    plan=schedule(bank,args.full); row_by_id={r['global_id']:r for r in bank['windows']}
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');out=args.output/stamp;out.mkdir(parents=True,exist_ok=False)
    shutil.copy2(firmware,out/'firmware_declarado.elf')
    shutil.copy2(__file__,out/'validar_total_ejecutado.py')
    for name in ('CATALOGO_TOTAL.json','BANCO_TOTAL.json','CATALOGO_CANONICO.json','MANIFEST_RECURSOS_SHA256.json','MANIFEST_RECURSOS_SHA256.sha256'):
        shutil.copy2(resources/name,out/name)
    optional={}
    for key,path in [('build_report',args.build_report),('programmer_report',args.programmer_report)]:
        if path:
            path=path.resolve();require(path.is_file(),key+' no existe')
            target=out/(key+path.suffix);shutil.copy2(path,target);optional[key]=dict(path=str(path),sha256=sha(path))
    write(out/'PLAN_EJECUTADO.json',dict(mode='full' if args.full else 'gate',schedule=plan,
        expected_primary_RUN=2688 if args.full else 6,expected_PROFILE=896 if args.full else 2,
        expected_BASE=42 if args.full else 2,**rounding_provenance(),
        resource_manifest_sha256=sha(resources/'MANIFEST_RECURSOS_SHA256.json'),resources=str(resources),
        max_seconds=args.max_seconds,timeout_per_command_seconds=args.timeout,
        python=sys.executable,python_version=sys.version,numpy_version=np.__version__,platform=platform.platform(),
        firmware_elf=str(firmware),firmware_elf_sha256=firmware_sha,attachments=optional,
        gate_evidence=str(args.gate_evidence) if args.gate_evidence else None))
    trace=Trace(out/'TRAZA_UART.jsonl'); events=Trace(out/'MEDICIONES.jsonl')
    report=dict(status='FAILED_INCOMPLETE',mode='full' if args.full else 'gate',catalog_sha256=cat['catalog_sha256'],
        firmware_elf_sha256=firmware_sha,firmware_loaded_verified_by_programmer_report=False,
        gate_cases=gate_plan(),**rounding_provenance(),
        resource_manifest_sha256=sha(resources/'MANIFEST_RECURSOS_SHA256.json'),client_sha256=sha(__file__),
        programmer_report_attached=bool(args.programmer_report),
        firmware_identity_limitation='ELF declarado/copiado; INFO autentica modelos y parámetros, no toda la imagen de Flash. '
            'Un log adjunto requiere auditoría de correspondencia y verificación del programador.',
        completed_primary_RUN=0,completed_PROFILE=0,completed_BASE=0,negative_rejections=0,
        selector_enabled=False,total_cost_validated=False,independent_acquisitions_claimed=False,
        cached_PC_end_to_end_bit_exactness_claimed=False,predictive_outer_evaluation=False,
        primary_timing='outer DWT pair around extraction+scaling/quantization+copy/Invoke/copy/argmax+normality',
        excluded_from_primary_cost=['UART','parser','checksums','SELECT/initialization','BASE','PROFILE'],
        overhead_subtracted=False,nominal_hal_clock_not_externally_measured=True,WCET_claimed=False)
    measured=[];bases=[];selections=[];rejects=[];started=time.monotonic()
    print('Evidencias: '+str(out),flush=True)
    try:
        import serial
        lite=LiteModels(resources,cat['representatives']);report['tensorflow_version']=lite.tf.__version__
        with serial.Serial(args.port,args.baud,timeout=.10,write_timeout=args.timeout) as port:
            link=Link(port,trace,args.timeout,deadline=started+args.max_seconds)
            report['firmware_info']=parse_info(link.ask('CWRUTOTAL_INFO','CWRUTOTAL_INFO,'),cat)
            last=report['firmware_info']['last_run_id']; models_seen=set()
            for gid,rids in plan:
                row=row_by_id[gid];transmit(link,windows[row['npz_index']],row)
                base=baseline(link,row);bases.append(base);events.append(dict(type='BASE',**base));report['completed_BASE']+=1
                for rid in rids:
                    rep=cat['representatives'][rid]
                    if rid not in models_seen:
                        validate_model(link.ask(f'CWRUTOTAL_MODEL,{rid}','CWRUTOTAL_MODEL,'),rep);models_seen.add(rid)
                    selection=parse_selected(link.ask(f'CWRUTOTAL_SELECT,{rid}','CWRUTOTAL_SELECTED,'),rep)
                    selections.append(selection);events.append(dict(type='SELECT',global_id=gid,**selection))
                    repeat_pages=None
                    for repetition,mode in [(1,0),(2,0),(3,0),(1,1)]:
                        command='CWRUTOTAL_PROFILE' if mode else 'CWRUTOTAL_RUN'
                        result=parse_result(link.ask(command+','+row['checksum_fnv1a'],'CWRUTOTAL_RESULT,'),rep,row,last,mode,selection['arena_used'])
                        last=result['run_id'];pages=get_pages(link,result,rep)
                        comparison=compare(result,pages,base,rep,bank['reference_by_rep_gid'][str(rid)][str(gid)],lite)
                        require(repeat_pages is None or repeat_pages==pages,'Repeticiones/PROFILE no conservan intermedios exactos')
                        repeat_pages=pages;result.update(repetition=repetition,status='PASS',pages=pages,comparison=comparison)
                        measured.append(result);events.append(dict(type='PROFILE' if mode else 'RUN',**result))
                        report['completed_PROFILE' if mode else 'completed_primary_RUN']+=1
                    print(f'{rep["representative_id"]}/gid{gid}: PASS; 3 RUN + PROFILE; último principal='
                          f'{measured[-2]["milliseconds"]:.6f} ms',flush=True)
            rejects,support=negatives(link,measured[-1],cat['representatives'][measured[-1]['rep_id']])
            report['negative_rejections']=len(rejects);events.append(dict(type='NEGATIVE_SUPPORT_SELECT',**support))
            for rejection in rejects:events.append(dict(type='NEGATIVE',**rejection))
            final=parse_info(link.ask('CWRUTOTAL_INFO','CWRUTOTAL_INFO,'),cat)
            require(final['last_run_id']==last and not final['result_valid'],'Negativas alteraron contador/publicación')
            report['firmware_info_final']=final
        expected=(2688,896,42) if args.full else (6,2,2)
        require((report['completed_primary_RUN'],report['completed_PROFILE'],report['completed_BASE'])==expected
                and len(rejects)==17,'Campaña incompleta')
        report['status']='FULL_ARM_MEASUREMENTS_READY_FOR_AUDIT' if args.full else GATE_STATUS
    except BaseException as exc:
        report['error']=dict(type=type(exc).__name__,message=str(exc));raise
    finally:
        report['wall_seconds']=time.monotonic()-started;report['timing_statistics']=statistics(measured,cat['representatives'])
        report['physical_unique_window_ids']=[b['global_id'] for b in bases]
        report['model_selection_count']=len(selections);report['valid_RUN_count']=len(measured)
        trace.close();events.close();write(out/'RESULTADO_TOTAL.json',report)
        entries={p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*')) if p.is_file()}
        mp=out/'MANIFEST_EVIDENCIAS_SHA256.json';write(mp,entries)
        with mp.with_suffix('.sha256').open('x',encoding='ascii') as stream:stream.write(sha(mp)+'  '+mp.name+'\n')
        print(report['status']+'; evidencias: '+str(out),flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group();mode.add_argument('--gate',action='store_true');mode.add_argument('--full',action='store_true')
    parser.add_argument('--port');parser.add_argument('--baud',type=int,default=115200)
    parser.add_argument('--timeout',type=float,default=15.)
    parser.add_argument('--max-seconds',type=float,help='Límite global:600s para puerta SR;1800s para full')
    parser.add_argument('--preflight',action='store_true',help='Solo integridad, bancos y replay PC; sin TF/serial')
    default_resources=HERE.parent/'recursos'
    if not default_resources.exists() and len(HERE.parents)>=4:
        default_resources=HERE.parents[3]/'despliegue_stm32/costo_total_single_rounding_v2_180mhz_2026-10-26/recursos'
    parser.add_argument('--resources',type=Path,default=default_resources)
    parser.add_argument('--output',type=Path,default=HERE/'resultados')
    parser.add_argument('--firmware-elf',type=Path)
    parser.add_argument('--build-report',type=Path);parser.add_argument('--programmer-report',type=Path)
    parser.add_argument('--gate-evidence',type=Path)
    args=parser.parse_args();require(args.timeout>0 and args.baud>0,'Timeout/baud deben ser positivos')
    if args.preflight:
        cat,bank,windows,manifest=load_resources(args.resources)
        checks=0
        for rep in cat['representatives']:
            for gid in rep['scope']['benchmark_global_window_ids']:
                ref=bank['reference_by_rep_gid'][str(rep['id'])][str(gid)]
                replayed=replay(ref['raw_features14_bits'],rep)
                require(replayed['standardized_bits']==ref['normalized_features_bits']
                        and replayed['quantized']==ref['int8_inputs']
                        and replayed['score_bits']==ref['detector_score_bits']
                        and replayed['rejected']==ref['outside_normality'],'Referencia PC no reproduce banco sellado')
                checks+=1
        planned=schedule(bank,True)
        print(json.dumps(dict(status='PC_PREFLIGHT_PASS_NOT_ARM_VALIDATED',resources=len(manifest),
            representatives=32,unique_windows=len(planned),PC_replays=checks,planned_primary_RUN=2688,
            planned_PROFILE=896,gate_cases=gate_plan(),**rounding_provenance(),
            serial_opened=False,TFLite_invocations=0,catalog_sha256=cat['catalog_sha256']),ensure_ascii=False))
        return
    require(args.port and args.firmware_elf is not None,'La prueba física requiere --port y --firmware-elf')
    if args.max_seconds is None:args.max_seconds=1800. if args.full else 600.
    require(np.isfinite(args.max_seconds) and args.max_seconds>0,'--max-seconds debe ser finito positivo')
    run(args)


if __name__=='__main__':
    try:main()
    except KeyboardInterrupt:print('INTERRUPTED: evidencia parcial preservada',file=sys.stderr);sys.exit(130)
    except Exception as exc:print('ERROR: '+str(exc),file=sys.stderr);sys.exit(1)
