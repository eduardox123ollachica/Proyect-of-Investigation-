"""Cliente CWRUND aislado. --self-test no abre COM ni ejecuta C ARM."""
import argparse
import json
import sys
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from contrato_detector_cwru import bits, floats, raw_hex, fnv, evaluate

HERE = Path(__file__).resolve().parent


def exchange(port,cmd,prefix,timeout,trace,allow_error=False):
    payload=(cmd+'\r\n').encode('ascii')
    trace.append(dict(direction='TX',command=cmd,bytes=len(payload)))
    if port.write(payload)!=len(payload):raise IOError('Escritura incompleta')
    port.flush();deadline=time.monotonic()+timeout;last=[]
    while time.monotonic()<deadline:
        raw=port.readline()
        if not raw:continue
        line=raw.decode('ascii',errors='strict').strip()
        if not line:continue
        trace.append(dict(direction='RX',line=line));last.append(line)
        if line.startswith('CWRUND_ERROR,'):
            if allow_error:return line
            raise RuntimeError(line)
        if line.startswith(prefix):return line
    raise TimeoutError(f'No llegó {prefix}; últimas respuestas={last[-5:]}')


def sha(path):
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bundle():
    manifest = json.loads((HERE/'VALIDACION_MANIFEST.json').read_text(encoding='utf-8'))
    for name,digest in manifest.items():
        if Path(name).name != name or sha(HERE/name) != digest:
            raise ValueError('Recurso modificado: '+name)
    ref = json.loads((HERE/'REFERENCIA_DETECTOR.json').read_text(encoding='utf-8'))
    if len(ref['cases']) != 165 or len(ref['numeric_negative_cases']) != 10 or ref['params']['k'] != 1:
        raise ValueError('Catálogo o K diferente del contrato')
    return ref


def command(case):
    raw = bytes.fromhex(case['input_hex'])
    if raw.hex() != raw_hex(case['input_bits']) or f'{fnv(raw):08x}' != case['checksum_fnv1a']:
        raise ValueError('Entrada/checksum incoherentes')
    return f'CWRUND_{case["mode"]},{fnv(raw):08x},'+raw.hex()


def info(port, ref, timeout, trace):
    fields=exchange(port,'CWRUND_INFO','CWRUND_INFO,',timeout,trace).split(',')
    params=ref['params']
    if len(fields)!=12 or tuple(map(int,fields[1:7]))!=(1,14,12,1,180000000,1):
        raise ValueError('INFO/reloj/DWT incompatibles')
    fpscr=int(fields[7],16)
    if fpscr & ((3<<22)|(1<<24)):
        raise ValueError('FPSCR: se requiere redondeo al par y sin FTZ')
    if fields[8:]!=[params['source_sha256'],params['params_fnv1a'],params['threshold_bits'],bytes(params['indices']).hex()]:
        raise ValueError('Parámetros/índices/umbral diferentes')
    return dict(protocol=1,inputs=14,used_features=12,k=1,hclk_hz=180000000,dwt_enabled=True,
        fpscr=f'{fpscr:08x}',flush_to_zero=False,rounding='nearest_even',
        source_sha256_declared=fields[8],params_fnv1a=fields[9],threshold_bits=fields[10],indices=params['indices'])


def parse(line,case,params):
    f=line.split(',')
    if len(f)!=10 or f[:3]!=['CWRUND_RESULT',case['mode'],case['checksum_fnv1a']]:
        raise ValueError('Trama o checksum incorrectos')
    cycles=int(f[3]); reject=int(f[5]); cluster=int(f[6])
    if cycles<=0 or reject not in (0,1) or cluster!=0 or f[7]!=params['threshold_bits']:
        raise ValueError('Ciclos/decisión/umbral inválidos')
    if len(f[4])!=8 or not np.isfinite(floats([f[4]])[0]):
        raise ValueError('Score inválido')
    count=0 if case['mode']=='CHECK' else 12
    words=[]
    for raw in f[8:]:
        if len(raw)!=count*8:
            raise ValueError('Intermedios incompletos')
        values=np.frombuffer(bytes.fromhex(raw),dtype='<f4')
        if not np.isfinite(values).all():
            raise ValueError('Intermedios no finitos')
        words.append(bits(values))
    return dict(cycles=cycles,z_bits=words[0],term_bits=words[1],score_bits=f[4].lower(),rejected=bool(reject),cluster=cluster)


def numeric(observed):
    return {k:v for k,v in observed.items() if k!='cycles'}


def negatives(ref):
    cmd=command(ref['cases'][0]); p=cmd.split(',')
    wrong=p.copy();wrong[1]=f'{int(p[1],16)^1:08x}'
    invalid=p.copy();invalid[2]='g'+p[2][1:]
    result=[(cmd[:-1],'FORMAT'),(cmd+'00','FORMAT'),(','.join(wrong),'CHECKSUM'),
        (','.join(invalid),'FORMAT'),('CWRUND_BAD','UNKNOWN')]
    result.extend((command(c),c['expected_reason']) for c in ref['numeric_negative_cases'])
    return result


class Tests(unittest.TestCase):
    def test_bundle_reference_and_encoding(self):
        ref=bundle()
        for case in ref['cases']:
            self.assertEqual(evaluate(case['mode'],case['input_bits'],ref['params']),case['expected'])
            self.assertLess(len(command(case)),1024)
    def test_strict_threshold(self):
        cases={c['name']:c for c in bundle()['cases']}
        self.assertFalse(cases['compare_below']['expected']['rejected'])
        self.assertFalse(cases['compare_equal']['expected']['rejected'])
        self.assertTrue(cases['compare_above']['expected']['rejected'])
    def test_numeric_rejections(self):
        ref=bundle()
        for case in ref['numeric_negative_cases']:
            with self.assertRaisesRegex(ValueError,case['expected_reason']):
                evaluate(case['mode'],case['input_bits'],ref['params'])
    def test_parser_and_corruption(self):
        ref=bundle();case=ref['cases'][0];e=case['expected']
        line=f'CWRUND_RESULT,{case["mode"]},{case["checksum_fnv1a"]},100,{e["score_bits"]},{int(e["rejected"])},0,{ref["params"]["threshold_bits"]},{raw_hex(e["z_bits"])},{raw_hex(e["term_bits"])}'
        self.assertEqual(numeric(parse(line,case,ref['params'])),e)
        with self.assertRaises(ValueError):parse(line.replace(',100,',',0,'),case,ref['params'])
        bad=dict(case);bad['input_hex']='00'*56
        with self.assertRaises(ValueError):command(bad)
    def test_negative_catalogue(self):
        self.assertEqual(len(negatives(bundle())),15)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--port');p.add_argument('--self-test',action='store_true')
    g=p.add_mutually_exclusive_group();g.add_argument('--index',type=int);g.add_argument('--all',action='store_true')
    p.add_argument('--repeat',type=int,default=3);p.add_argument('--timeout',type=float,default=5.)
    args=p.parse_args()
    if args.self_test:
        result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
        return 0 if result.wasSuccessful() else 1
    if not args.port or args.repeat!=3 or args.timeout<=0:p.error('Indicar puerto; el contrato usa tres repeticiones')
    ref=bundle();index=0 if args.index is None else args.index
    cases=ref['cases'] if args.all else [c for c in ref['cases'] if c['group']=='development_entm' and c['index']==index]
    if not cases:p.error('Índice fuera de 0..41')
    import serial
    import time
    folder=HERE/'resultados'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    folder.mkdir(parents=True,exist_ok=False)
    trace=[];results=[];rejected=[]
    report=dict(status='ERROR',mode='all' if args.all else 'single',cases=len(cases),repetitions=3,
        extraction_executed=False,dnn_executed=False,training_executed=False,threshold_refitted=False)
    try:
        with serial.Serial(args.port,115200,timeout=.2,write_timeout=args.timeout) as port:
            time.sleep(.2);port.reset_input_buffer()
            report['firmware_info']=info(port,ref,args.timeout,trace)
            print('Firmware:',report['firmware_info'],flush=True)
            for case in cases:
                previous=None
                for repetition in range(1,4):
                    line=exchange(port,command(case),'CWRUND_RESULT,',args.timeout,trace)
                    observed=parse(line,case,ref['params']);current=numeric(observed)
                    stable=previous is None or previous==current
                    passed=current==case['expected'] and stable
                    results.append(dict(name=case['name'],group=case['group'],mode=case['mode'],repetition=repetition,
                        input_hex=case['input_hex'],status='PASS' if passed else 'FAIL',reference=case['expected'],
                        observed=observed,bit_exact=current==case['expected'],repeated_output_stable=stable))
                    if not passed:raise RuntimeError('Discrepancia en '+case['name']+'; conservar evidencia y no relajar criterio')
                    previous=current
                score=float(floats([observed['score_bits']])[0])
                print(f'{case["name"]}: PASS; d2={score:.9g}; fuera_normalidad={observed["rejected"]}; '
                    f'última={observed["cycles"]} ciclos; {observed["cycles"]/180:.4f} us; exacta PC=SI',flush=True)
            if args.all:
                for command_text,reason in negatives(ref):
                    line=exchange(port,command_text,'CWRUND_ERROR,',args.timeout,trace,allow_error=True)
                    fields=line.split(',')
                    if len(fields)!=3 or fields[1]!=reason or int(fields[2])>=0:raise RuntimeError('Rechazo incorrecto: '+line)
                    rejected.append(dict(command=command_text,reason=reason,response=line))
            report['firmware_info_after']=info(port,ref,args.timeout,trace)
            report['status']='PASS'
    except Exception as exc:
        report['error']=str(exc);print('ERROR:',exc,file=sys.stderr)
    finally:
        report.update(results=results,negative_tests=rejected,completed_executions=len(results),
            client_sha256=sha(Path(__file__)),reference_sha256=sha(HERE/'REFERENCIA_DETECTOR.json'),numpy=np.__version__,python=sys.version)
        (folder/'resultados.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        (folder/'uart_trace.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in trace),encoding='utf-8')
        (folder/'MANIFEST_SHA256.json').write_text(json.dumps({f.name:sha(f) for f in folder.iterdir() if f.is_file()},indent=2)+'\n',encoding='utf-8')
        print('Evidencias:',folder,flush=True)
    return 0 if report['status']=='PASS' else 1


if __name__=='__main__':raise SystemExit(main())
