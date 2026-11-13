"""Aisla Softmax del modelo existente para comprobar los logits del replay.

Genera un FlatBuffer derivado solo para diagnostico PC en una carpeta nueva.
No modifica el modelo original, las tolerancias ni el firmware.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'despliegue_stm32/costo_total_representantes_v2_180mhz_2026-10-25'
DEST=OUT/'diagnostico_divergencia_20261026T042424Z'

def main():
    started=time.monotonic()
    import numpy as np
    import tensorflow as tf
    import flatbuffers
    from tensorflow.lite.python import schema_py_generated as schema
    cat=json.loads((OUT/'recursos/CATALOGO_TOTAL.json').read_text(encoding='utf-8'))
    rep=cat['representatives'][22]
    original=(OUT/'recursos'/rep['model_path']).read_bytes()
    assert hashlib.sha256(original).hexdigest()==rep['model_sha256']
    model=schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(original,0))
    assert len(model.subgraphs)==1
    sg=model.subgraphs[0]
    assert len(sg.operators)==4
    op=sg.operators[-1]
    assert model.operatorCodes[op.opcodeIndex].builtinCode==25
    assert len(op.inputs)==len(op.outputs)==1
    ix=int(op.inputs[0]);ox=int(op.outputs[0])
    sg.inputs=np.asarray([ix],dtype=np.int32)
    sg.outputs=np.asarray([ox],dtype=np.int32)
    sg.operators=[op]
    model.signatureDefs=[]
    builder=flatbuffers.Builder(4096)
    root=model.Pack(builder)
    builder.Finish(root,file_identifier=b'TFL3')
    isolated=bytes(builder.Output())
    mp=DEST/'softmax_aislado_diagnostico.tflite'
    with mp.open('xb') as f:f.write(isolated)
    it=tf.lite.Interpreter(model_content=isolated,num_threads=1,
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF)
    it.allocate_tensors()
    inp,out=it.get_input_details()[0],it.get_output_details()[0]
    assert list(inp['shape'])==list(out['shape'])==[1,3]
    assert inp['dtype']==out['dtype']==np.int8
    fc=json.loads((DEST/'REPLAY_FC_ENTERAS.json').read_text(encoding='utf-8'))
    # Logits are independently reproduced from original weights and biases.
    logits={'SINGLE':[33,-29,-3],'DOUBLE':[33,-30,-3]}
    results={}
    for mode,x in logits.items():
        it.set_tensor(inp['index'],np.asarray([x],dtype=np.int8));it.invoke()
        y=it.get_tensor(out['index'])[0].astype(int).tolist()
        results[mode]=dict(logits=x,output_INT8=y,predicted_class=int(np.argmax(y)))
    report=dict(status='ISOLATED_SOFTMAX_PC_DIAGNOSTIC_NOT_A_FIRMWARE_FIX',
        original_model_sha256=rep['model_sha256'],derived_model_sha256=hashlib.sha256(isolated).hexdigest(),
        replay_FC_file_sha256=hashlib.sha256((DEST/'REPLAY_FC_ENTERAS.json').read_bytes()).hexdigest(),
        derived_operator_codes=[25],original_logits_tensor_index=ix,original_output_tensor_index=ox,
        input_quantization=[float(inp['quantization'][0]),int(inp['quantization'][1])],
        results=results,original_cached_PC_output=[41,-100,-69],observed_ARM_output=[42,-101,-69],
        single_matches_PC=(results['SINGLE']['output_INT8']==[41,-100,-69]),
        double_matches_ARM=(results['DOUBLE']['output_INT8']==[42,-101,-69]),
        tensorflow_version=tf.__version__,python=sys.executable,TFLite_invocations=2,
        new_training_fits=0,serial_port_opened=False,firmware_modified=False,seconds=time.monotonic()-started)
    path=DEST/'COMPARACION_SOFTMAX_AISLADO.json'
    with path.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(report,ensure_ascii=True),flush=True)

if __name__=='__main__':main()
