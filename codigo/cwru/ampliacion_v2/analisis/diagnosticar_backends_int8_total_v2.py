"""Compara dos kernels PC para la entrada del fallo, sin entrenar ni usar UART.

Preserva activaciones para localizar la primera divergencia entre los kernels
PC. No modifica el validador, su criterio de aceptacion o el firmware.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT/'despliegue_stm32/costo_total_representantes_v2_180mhz_2026-10-25'
DEST = OUT/'diagnostico_divergencia_20261026T042424Z'

def main():
    started = time.monotonic()
    import numpy as np
    import tensorflow as tf
    cat = json.loads((OUT/'recursos/CATALOGO_TOTAL.json').read_text(encoding='utf-8'))
    bank = json.loads((OUT/'recursos/BANCO_TOTAL.json').read_text(encoding='utf-8'))
    rep = cat['representatives'][22]
    ref = bank['reference_by_rep_gid']['22']['232']
    model = (OUT/'recursos'/rep['model_path']).read_bytes()
    assert hashlib.sha256(model).hexdigest() == rep['model_sha256']
    q = np.asarray([ref['int8_inputs']], dtype=np.int8)
    assert q.tolist() == [[-45,-15,-38,-36,-69,-39,-42]]
    variants = {}
    for name,resolver in (
        ('BUILTIN_WITHOUT_DEFAULT_DELEGATES', tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES),
        ('BUILTIN_REF', tf.lite.experimental.OpResolverType.BUILTIN_REF),
    ):
        it = tf.lite.Interpreter(model_content=model, num_threads=1,
            experimental_op_resolver_type=resolver, experimental_preserve_all_tensors=True)
        it.allocate_tensors()
        inp,out = it.get_input_details()[0],it.get_output_details()[0]
        it.set_tensor(inp['index'],q)
        it.invoke()
        ops = []
        for op in it._get_ops_details():
            item = dict(op_name=op['op_name'], index=int(op['index']), outputs=[])
            assert op['op_name'] != 'DELEGATE'
            for ix in op['outputs']:
                a = it.get_tensor(int(ix))
                item['outputs'].append(dict(index=int(ix),shape=list(a.shape),dtype=str(a.dtype),
                    values=a.tolist(),raw_hex=a.tobytes().hex()))
            ops.append(item)
        output = it.get_tensor(out['index'])[0].astype(int).tolist()
        variants[name] = dict(output_INT8=output,predicted_class=int(np.argmax(output)),ops=ops)
    mismatches=[]
    left,right=(variants[k]['ops'] for k in ('BUILTIN_WITHOUT_DEFAULT_DELEGATES','BUILTIN_REF'))
    assert len(left)==len(right)==4
    for a,b in zip(left,right):
        assert a['op_name']==b['op_name']
        for x,y in zip(a['outputs'],b['outputs']):
            if x['raw_hex'] != y['raw_hex']:
                mismatches.append(dict(op_index=a['index'],op_name=a['op_name'],tensor_index=x['index'],
                    optimized=x['values'],reference=y['values']))
    report = dict(status='PC_BACKEND_DIAGNOSTIC_NOT_A_FIRMWARE_FIX',
        model_sha256=rep['model_sha256'],representative_id=rep['representative_id'],global_window_id=232,
        input_INT8=q[0].astype(int).tolist(),cached_PC_output=ref['int8_outputs'],
        observed_ARM_output=[42,-101,-69],tensorflow_version=tf.__version__,numpy_version=np.__version__,
        python=sys.executable,TF_ENABLE_ONEDNN_OPTS=os.environ.get('TF_ENABLE_ONEDNN_OPTS'),
        preserve_all_tensors=True,variants=variants,layer_differences=mismatches,
        new_training_fits=0,TFLite_invocations=2,serial_port_opened=False,firmware_modified=False,
        selector_costs_enabled=False,seconds=time.monotonic()-started)
    DEST.mkdir(parents=True,exist_ok=True)
    path=DEST/'COMPARACION_KERNELS_PC.json'
    with path.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(outputs={k:v['output_INT8'] for k,v in variants.items()},
        layer_differences=mismatches,seconds=report['seconds'],output=str(path)),ensure_ascii=True),flush=True)

if __name__=='__main__':
    main()
