"""Replay entero de tres FullyConnected; sin TF, UART ni entrenamiento.

Calcula las ramas single/double de CMSIS-NN desde los bytes TFLite. No cambia
el criterio del validador ni presupone equivalencia de Softmax.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import struct
import time

import numpy as np
from auditar_elf_total_v2 import FlatBuffer

ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'despliegue_stm32/costo_total_representantes_v2_180mhz_2026-10-25'
DEST=OUT/'diagnostico_divergencia_20261026T042424Z'

def require(ok,message):
    if not ok: raise ValueError(message)

def quantize_multiplier(value):
    if value==0: return 0,0
    q,shift=math.frexp(value)
    fixed=math.floor(q*(1<<31)+.5)
    if fixed==(1<<31): fixed//=2;shift+=1
    if shift< -31: return 0,0
    require(0<=fixed<1<<31 and shift<=30,'Rango de multiplicador fuera del contrato')
    return fixed,shift

def requantize(value,multiplier,shift,mode):
    if mode=='single':
        total_shift=31-shift
        return (value*multiplier+(1<<(total_shift-1)))>>total_shift
    require(mode=='double','Modo no reconocido')
    shifted=value*(1<<max(shift,0))
    require(-(1<<31)<=shifted<(1<<31),'Overflow int32 en desplazamiento izquierdo')
    high=(shifted*multiplier+(1<<30))>>31
    exponent=max(-shift,0);mask=(1<<exponent)-1
    remainder=high&mask;result=high>>exponent
    threshold=(mask>>1)+(1 if result<0 else 0)
    return result+(1 if remainder>threshold else 0)

def parse_model(model):
    f=FlatBuffer(model);root=f.unpack('<I',0)[0]
    sg=f.tables(root,8)[0];tables=f.tables(sg,4);buffers=f.tables(root,12)
    tensors=[]
    for i,t in enumerate(tables):
        item=f.tensor(t);item['index']=i
        item['scales']=[struct.unpack('<f',struct.pack('<I',int(b,16)))[0] for b in item['scale_bits']]
        index=f.scalar(t,8,'<I');buffer=buffers[index]
        item['constant']=None
        if f.field(buffer,4) is not None:
            raw=bytes(f.vector(buffer,4,'<B'))
            if raw:
                dtype={9:np.dtype('i1'),2:np.dtype('<i4')}[item['type']]
                item['constant']=np.frombuffer(raw,dtype=dtype).astype(np.int64).reshape(item['shape']).tolist()
        tensors.append(item)
    ops=[]
    for op in f.tables(sg,10)[:3]:
        ins=f.vector(op,6,'<i');outs=f.vector(op,8,'<i')
        options=f.pointer(op,12)
        activation=f.scalar(options,4,'<b')
        require(activation in (0,1),'Solo NONE o RELU')
        ops.append(dict(inputs=ins,output=outs[0],activation=activation))
    return tensors,ops

def dense(x,op,tensors,mode):
    a,w,b=[tensors[i] for i in op['inputs']];y=tensors[op['output']]
    weights=w['constant'];bias=b['constant']
    require(weights is not None and bias is not None,'Pesos/bias no constantes')
    require(len(a['scales'])==len(y['scales'])==len(a['zero_points'])==len(y['zero_points'])==1,'Activacion per-tensor requerida')
    require(all(z==0 for z in w['zero_points']),'Pesos no simetricos')
    d=len(x);n=len(weights)
    require(all(len(row)==d for row in weights) and len(bias)==n,'Dimensiones incompatibles')
    require(len(w['scales']) in (1,n),'Escalas per-channel incompatibles')
    out=[];accumulators=[];multipliers=[];shifts=[];before_offset=[]
    lower=max(-128,y['zero_points'][0]) if op['activation']==1 else -128
    for j,row in enumerate(weights):
        value=int(bias[j])+sum((int(v)-a['zero_points'][0])*int(weight) for v,weight in zip(x,row))
        require(-(1<<31)<=value<(1<<31),'Acumulador no int32')
        effective=float(a['scales'][0])*float(w['scales'][j if len(w['scales'])>1 else 0])/float(y['scales'][0])
        multiplier,shift=quantize_multiplier(effective)
        rounded=requantize(value,multiplier,shift,mode)
        out.append(max(lower,min(127,rounded+y['zero_points'][0])))
        accumulators.append(value);multipliers.append(multiplier);shifts.append(shift);before_offset.append(rounded)
    return dict(output_tensor=op['output'],activation=op['activation'],input=x,output=out,
        accumulators=accumulators,multipliers=multipliers,shifts=shifts,
        requantized_before_output_offset=before_offset,output_offset=y['zero_points'][0],
        activation_min=lower,activation_max=127)

def main():
    start=time.monotonic()
    cat=json.loads((OUT/'recursos/CATALOGO_TOTAL.json').read_text(encoding='utf-8'))
    rep=cat['representatives'][22];path=OUT/'recursos'/rep['model_path'];model=path.read_bytes()
    require(hashlib.sha256(model).hexdigest()==rep['model_sha256'],'Modelo modificado')
    tensors,ops=parse_model(model);q=[-45,-15,-38,-36,-69,-39,-42]
    modes={}
    for mode in ('single','double'):
        x=q;layers=[]
        for i,op in enumerate(ops):
            layer=dense(x,op,tensors,mode);layer['layer']=i
            layers.append(layer);x=layer['output']
        modes[mode]=dict(layers=layers,logits=x)
    differences=[]
    for a,b in zip(modes['single']['layers'],modes['double']['layers']):
        for j,(x,y) in enumerate(zip(a['output'],b['output'])):
            if x!=y:differences.append(dict(layer=a['layer'],neuron=j,single=x,double=y,
                single_accumulator=a['accumulators'][j],double_accumulator=b['accumulators'][j],
                multiplier=a['multipliers'][j],shift=a['shifts'][j]))
    comparison_path=DEST/'COMPARACION_KERNELS_PC.json';comparisons={}
    if comparison_path.is_file():
        saved=json.loads(comparison_path.read_text(encoding='utf-8'))
        for variant,record in saved['variants'].items():
            pc=[op['outputs'][0]['values'][0] for op in record['ops'][:3]]
            comparisons[variant]={mode:dict(all_FC_layers_exact=pc==[l['output'] for l in modes[mode]['layers']],
                first_difference=next((dict(layer=i,PC=p,replay=l['output']) for i,(p,l) in enumerate(zip(pc,modes[mode]['layers'])) if p!=l['output']),None))
                for mode in modes}
    report=dict(status='OFFLINE_FC_INTEGER_REPLAY_NOT_SOFTMAX_OR_HARDWARE_EQUIVALENCE',model_sha256=rep['model_sha256'],
        representative_id=rep['representative_id'],global_window_id=232,input_INT8=q,
        rounding_modes=modes,rounding_mode_differences=differences,PC_comparison=comparisons,
        PC_comparison_file=str(comparison_path) if comparison_path.is_file() else None,
        sources=[str(Path('D:/STM32IDE/Prueba_2_CWRU_Extraccion/TFLM/third_party/cmsis_nn/Include/arm_nnsupportfunctions.h')),
            str(Path('D:/STM32IDE/Prueba_2_CWRU_Extraccion/TFLM/tensorflow/lite/kernels/internal/quantization_util.cc'))],
        serial_port_opened=False,TFLite_invocations=0,new_training_fits=0,firmware_modified=False,
        seconds=time.monotonic()-start)
    DEST.mkdir(parents=True,exist_ok=True)
    target=DEST/'REPLAY_FC_ENTERAS.json'
    with target.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(report,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(logits={mode:data['logits'] for mode,data in modes.items()},
        differences=differences,PC_comparison=comparisons,report=str(target)),ensure_ascii=False))

if __name__=='__main__':main()
