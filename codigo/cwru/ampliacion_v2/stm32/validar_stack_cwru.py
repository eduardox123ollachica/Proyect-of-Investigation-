"""Cliente diagnóstico CWRUSTK: marca de agua escrita, no máximo SP garantizado.

--index 0: cuatro calibraciones y seis RUN (3 por patrón) de una ventana.
--all: 42 ventanas, 252 RUN y nueve rechazos; sin entrenamiento o 3 HP.
--self-test usa transporte ficticio: no ejecuta C/ARM ni mide stack real.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
PATTERNS=('a5a5a5a5','5a5a5a5a')
FIELDS=('pattern','top','low','reference_msp','after_msp','lowest_modified','prefix_bytes',
    'additional_written_bytes','total_observed_bytes','scanned_bytes','guard_intact','callback_status',
    'primask_before','primask_after','basepri_before','basepri_after','faultmask_before','faultmask_after',
    'control_before','control_after','fpccr_before','fpccr_after','ccr_before','ccr_after',
    'ticks_before','ticks_after','paint_masked_cycles','scan_masked_cycles','ipsr_before','ipsr_after',
    'monitor_status','reserved','fpscr_before','fpscr_after')

def require(ok,message):
    if not ok: raise ValueError(message)

def load(folder):
    spec=importlib.util.spec_from_file_location('frozen_pipeline',folder/'validar_pipeline.py')
    pc=importlib.util.module_from_spec(spec);spec.loader.exec_module(pc)
    manifest=json.loads((folder/'VALIDACION_MANIFEST.json').read_text(encoding='utf-8'))
    require(set(manifest)=={'validar_stack.py','validar_pipeline.py','REFERENCIA_PIPELINE.json',
        'ventanas_desarrollo.npz','metadata_ventanas.csv'},'Manifiesto de cinco recursos requerido')
    require(pc.sha(folder/'validar_stack.py')==manifest['validar_stack.py'],'Cliente stack modificado')
    ref,windows,metadata=pc.load_bundle(folder)
    return pc,ref,windows

def exchange(pc,port,command,prefix,trace,timeout=20.,allow_error=False):
    payload=(command+'\r\n').encode('ascii')
    require(len(payload)<1024,'Trama TX demasiado grande')
    trace.append(dict(direction='TX',command=command,bytes=len(payload),utc=datetime.now(timezone.utc).isoformat()))
    require(port.write(payload)==len(payload),'Escritura parcial');port.flush()
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        raw=port.readline()
        if not raw: continue
        line=raw.decode('ascii').rstrip('\r\n')
        if not line: continue
        trace.append(dict(direction='RX',line=line,bytes=len(raw),utc=datetime.now(timezone.utc).isoformat()))
        require(raw.endswith(b'\r\n') and len(raw)<512,'Respuesta truncada o fuera de buffer')
        if '_ERROR,' in line:
            if allow_error: return line
            raise ValueError(line)
        if line.startswith(prefix): return line
    raise TimeoutError('No llegó '+prefix)

def stack_info(pc,port,trace):
    f=exchange(pc,port,'CWRUSTK_INFO','CWRUSTK_INFO,',trace).split(',')
    require(len(f)==15 and f[1:5]==['1','4096','128','34'],'INFO stack incompatible')
    v={name:int(f[i],16) for i,name in ((5,'top'),(6,'low'),(7,'static_end'),(8,'control'),
        (11,'basepri'),(13,'fpccr'),(14,'ccr'))}
    v.update(ipsr=int(f[9]),primask=int(f[10]),faultmask=int(f[12]))
    require(v['top']==0x20020000 and v['low']==0x2001f000 and v['static_end']+512<=v['low'], 'Región stack/heap incompatible')
    require(not (v['control']&3 or v['ipsr'] or v['primask'] or v['basepri'] or v['faultmask']), 'Contexto no privilegiado/MSP o IRQ alteradas')
    require(v['ccr']&(1<<9),'Se requiere alineación de excepciones a8B')
    return v

def parse_stats(text,pattern,mode):
    require(re.fullmatch('[0-9a-fA-F]{272}',text) is not None,'Registro stack truncado')
    v=dict(zip(FIELDS,map(int,np.frombuffer(bytes.fromhex(text),dtype='<u4'))))
    require(v['pattern']==int(pattern,16) and v['top']==0x20020000 and v['low']==0x2001f000,'Patrón/límites distintos')
    ref,first=v['reference_msp'],v['lowest_modified']
    require(ref%8==0 and v['low']+128+256<ref<=v['top'] and v['after_msp']==ref,'MSP/alineación/restauración inválidos')
    require(v['low']<=first<=ref and first%4==0 and v['scanned_bytes']==ref-v['low'],'Región escaneada incoherente')
    require(v['prefix_bytes']==v['top']-ref and v['additional_written_bytes']==ref-first
        and v['total_observed_bytes']==v['top']-first<=3968 and v['guard_intact']==1,'Guarda/huella incoherente o agotada')
    require(all(v[k]==0 for k in ('callback_status','monitor_status','reserved','primask_before','primask_after',
        'basepri_before','basepri_after','faultmask_before','faultmask_after','ipsr_before','ipsr_after')),'Estado/máscaras incorrectos')
    require(not ((v['control_before']|v['control_after'])&3),'Se cambió Thread privilegiado/MSP')
    require(v['ccr_before']==v['ccr_after'] and v['ccr_before']&(1<<9),'CCR cambió')
    require((v['fpccr_before']&0xc0000000)==(v['fpccr_after']&0xc0000000),'Cambió ASPEN/LSPEN')
    require(not ((v['fpscr_before']|v['fpscr_after'])&((3<<22)|(1<<24))),'Cambió RNE/FTZ')
    require(0<v['paint_masked_cycles']<180000 and 0<v['scan_masked_cycles']<180000,'Sección IRQ bloqueada >=1ms o contador inválido')
    v['tick_delta']=(v['ticks_after']-v['ticks_before'])&0xffffffff
    if mode=='PROBE': require(v['additional_written_bytes']>=128,'No detectó probe de128B')
    if mode=='PIPELINE': require(v['tick_delta']>0,'No se observó avance SysTick alrededor del callback')
    return v

def session(pc,port,ref,windows,selected,report,trace,all_cases=False):
    report['pipeline_before']=pc.info(port,ref,20.,trace)
    report['initialization']=pc.initialize(port,ref,20.,trace)
    report['reuse']=pc.initialize(port,ref,20.,trace)
    require(report['reuse']['reused'] and report['reuse']['nn_init_cycles']==0,'INIT no reutilizó')
    report['pipeline_info']=pc.info(port,ref,20.,trace)
    report['stack_info']=stack_info(pc,port,trace)
    last_id=report['pipeline_info']['last_id']
    for pattern in PATTERNS:
        for mode in ('EMPTY','PROBE'):
            line=exchange(pc,port,f'CWRUSTK_CAL,{mode},{pattern}','CWRUSTK_CAL,',trace)
            f=line.split(',');require(len(f)==4 and f[1:3]==[mode,pattern],'CAL incompatible')
            report['calibrations'].append(dict(mode=mode,pattern=pattern,line=line,
                observed=parse_stats(f[3],pattern,mode),status='PASS'))
    for case in selected:
        pc.transmit(port,windows[case['index']],case['checksum_fnv1a'],20.,trace)
        previous=None
        for pattern in PATTERNS:
            for repetition in range(1,4):
                row=dict(index=case['index'],name=case['name'],pattern=pattern,repetition=repetition,status='ERROR')
                report['results'].append(row)
                line=exchange(pc,port,f'CWRUSTK_RUN,{case["checksum_fnv1a"]},{pattern}','CWRUSTK_RESULT,',trace)
                row['stack_line']=line;f=line.split(',')
                require(len(f)==5 and int(f[1])==last_id+1 and f[2:4]==[case['checksum_fnv1a'],pattern],'RUN ID/checksum/patrón incompatible')
                row['stack']=parse_stats(f[4],pattern,'PIPELINE')
                timing=exchange(pc,port,'CWRUSTK_TIMING,'+f[1],'CWRUPIPE_RESULT,',trace)
                row['diagnostic_timing_line']=timing
                result=pc.parse_result(timing,case,ref,last_id)
                pages=pc.get_pages(port,result,20.,trace)
                exact={k:pages[k]==case[k] for k in ('features_bits','input_expected','dnn_expected','detector_expected')}
                original=pc.compare_float64(pages['features_bits'],case)
                stable=previous is None or previous==pages
                row.update(observed=result,pages=pages,bit_exact=exact,original_float64_comparison=original,repeated_numeric_outputs_stable=stable)
                require(all(exact.values()) and original['passed'] and stable,'La instrumentación alteró salidas')
                row['status']='PASS';last_id=result['id'];previous=pages
        rows=[r for r in report['results'] if r['index']==case['index']]
        print(f'{case["name"]}: PASS;6 RUN; huella escrita máxima observada={max(r["stack"]["total_observed_bytes"] for r in rows)} B; guarda intacta',flush=True)
    if all_cases:
        case=selected[-1]
        tests=[('stale_timing',f'CWRUSTK_TIMING,{last_id+1}','STALE_ID',-1),
            ('unknown','CWRUSTK_BAD','UNKNOWN',-1),('info_extra','CWRUSTK_INFO,extra','FORMAT',-1),
            ('cal_mode','CWRUSTK_CAL,OTHER,a5a5a5a5','MODE',-1),
            ('cal_invalidates',f'CWRUSTK_TIMING,{last_id}','NO_RESULT',-1),
            ('cal_pattern','CWRUSTK_CAL,EMPTY,00000000','PATTERN',-1),
            ('run_format','CWRUSTK_RUN,1234567,a5a5a5a5','FORMAT',-1),
            ('wrong_checksum',f'CWRUSTK_RUN,{int(case["checksum_fnv1a"],16)^1:08x},a5a5a5a5','CHECKSUM',-1)]
        for name,cmd,reason,detail in tests:
            line=exchange(pc,port,cmd,'CWRUSTK_ERROR,',trace,allow_error=True)
            require(line==f'CWRUSTK_ERROR,{reason},{detail}','Rechazo incorrecto:'+line)
            report['negative_tests'].append(dict(name=name,command=cmd,response=line,status='PASS'))
        pc.begin(port,20.,trace)
        cmd=f'CWRUSTK_RUN,{case["checksum_fnv1a"]},a5a5a5a5'
        line=exchange(pc,port,cmd,'CWRUSTK_ERROR,',trace,allow_error=True)
        require(line=='CWRUSTK_ERROR,INCOMPLETE,0','RUN incompleto no rechazado')
        report['negative_tests'].append(dict(name='incomplete',command=cmd,response=line,status='PASS'))
    report['pipeline_after']=pc.info(port,ref,20.,trace);report['stack_after']=stack_info(pc,port,trace)
    require(report['pipeline_after']['last_id']==last_id,'Contador final incoherente')
    require(len(report['results'])==len(selected)*6 and len(report['calibrations'])==4,'Suite incompleta')
    report['observed_written_footprint_max_bytes']=max(r['stack']['total_observed_bytes'] for r in report['results'])
    report['status']='PASS'

def fake_transport(pc,ref,windows):
    class Fixture(pc.FakeTransport):
        def stats(self,pattern,mode):
            prefix,additional=2200,{'EMPTY':0,'PROBE':128,'PIPELINE':1008}[mode]
            v=[int(pattern,16),0x20020000,0x2001f000,0x20020000-prefix,0x20020000-prefix,
                0x20020000-prefix-additional,prefix,additional,prefix+additional,4096-prefix,1,0,
                0,0,0,0,0,0,4,4,0xc0000000,0xc0000000,512,512,0,1,2000,4000,0,0,0,0,0,0]
            return np.asarray(v,dtype='<u4').tobytes().hex()
        def write(self,payload):
            cmd=payload.decode('ascii').removesuffix('\r\n');f=cmd.split(',');reply=None
            error=lambda why,detail=-1:f'CWRUSTK_ERROR,{why},{detail}'
            if not cmd.startswith('CWRUSTK_'): return super().write(payload)
            if f[0] in ('CWRUSTK_RUN','CWRUSTK_CAL'): self.result=None;self.stack_summary=None
            if cmd=='CWRUSTK_INFO': reply='CWRUSTK_INFO,1,4096,128,34,20020000,2001f000,20014900,00000004,0,0,00000000,0,c0000000,00000200'
            elif f[0]=='CWRUSTK_INFO': reply=error('FORMAT')
            elif f[0]=='CWRUSTK_CAL':
                if len(f)!=3: reply=error('FORMAT')
                elif f[1] not in ('EMPTY','PROBE'): reply=error('MODE')
                elif f[2] not in PATTERNS: reply=error('PATTERN')
                else: reply=f'CWRUSTK_CAL,{f[1]},{f[2]},'+self.stats(f[2],f[1])
            elif f[0]=='CWRUSTK_RUN':
                if len(f)!=3 or re.fullmatch('[0-9a-fA-F]{8}',f[1]) is None: reply=error('FORMAT')
                elif f[2] not in PATTERNS: reply=error('PATTERN')
                else:
                    super().write(('CWRUPIPE_RUN,'+f[1]+'\r\n').encode('ascii'))
                    line=self.queue.pop().decode('ascii').strip()
                    if line.startswith('CWRUPIPE_ERROR'): reply=line.replace('CWRUPIPE_ERROR','CWRUSTK_ERROR')
                    else:
                        self.stack_summary=line
                        reply=f'CWRUSTK_RESULT,{self.last_id},{f[1]},{f[2]},'+self.stats(f[2],'PIPELINE')
            elif f[0]=='CWRUSTK_TIMING':
                if len(f)!=2 or not f[1].isdigit() or int(f[1])==0: reply=error('FORMAT')
                elif self.result is None: reply=error('NO_RESULT')
                elif int(f[1])!=self.last_id: reply=error('STALE_ID')
                else: reply=self.stack_summary
            else: reply=error('UNKNOWN')
            self.queue.append((reply+'\r\n').encode('ascii'));return len(payload)
    return Fixture(ref,windows)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--port');p.add_argument('--self-test',action='store_true')
    g=p.add_mutually_exclusive_group();g.add_argument('--index',type=int,default=None);g.add_argument('--all',action='store_true')
    p.add_argument('--bundle-dir',type=Path,default=HERE);p.add_argument('--output-root',type=Path);args=p.parse_args()
    if not args.self_test and not args.port: p.error('Se requiere --port')
    if args.index is not None and not 0<=args.index<42:p.error('Índice0..41 requerido')
    folder=(args.output_root or args.bundle_dir/'resultados')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    folder.mkdir(parents=True,exist_ok=False)
    report=dict(status='ERROR',mode='all' if args.all or args.self_test else 'single',results=[],calibrations=[],negative_tests=[],
        simulation_only=args.self_test,firmware_executed=False,serial_port_opened=False,
        training_executed=False,outer_test_loaded=False,physical_stack_peak_guaranteed=False,
        measurement='Observed written watermark plus sampled MSP prefix; not exact minimum SP',
        baseline_subtracted=False,canonical_latency_replaced=False,patterns=list(PATTERNS),repetitions_per_pattern=3,
        utc_started=datetime.now(timezone.utc).isoformat(),python=sys.version,numpy=np.__version__)
    trace=[]
    try:
        pc,ref,windows=load(args.bundle_dir)
        report['reference_sha256']=pc.sha(args.bundle_dir/'REFERENCIA_PIPELINE.json')
        report['bundle_manifest_sha256']=pc.sha(args.bundle_dir/'VALIDACION_MANIFEST.json')
        selected=ref['cases'] if args.all or args.self_test else [ref['cases'][0 if args.index is None else args.index]]
        if args.self_test:
            session(pc,fake_transport(pc,ref,windows),ref,windows,selected,report,trace,all_cases=True)
            tests=[];base=report['results'][0]['stack_line'].split(',')[4]
            for name,index,value in [('guard',10,0),('msp',4,0),('privilege',19,1),('mask',13,1),
                ('huge_mask_interval',26,180000),('fpscr_ftz',33,1<<24),('fpu_mode',21,0),('footprint',8,9999)]:
                v=np.frombuffer(bytes.fromhex(base),dtype='<u4').copy();v[index]=value
                try:parse_stats(v.tobytes().hex(),PATTERNS[0],'PIPELINE')
                except ValueError:tests.append(dict(name=name,status='PASS'))
                else:raise RuntimeError('Corrupción aceptada:'+name)
            report['self_tests']=tests;report['synthetic_watermark_not_physical']=True
            print('SIMULACIÓN: no C ARM, no COM, no stack físico medido.',flush=True)
        else:
            import serial
            with serial.Serial(args.port,115200,timeout=.2,write_timeout=20.) as port:
                report['serial_port_opened']=True;time.sleep(.2);port.reset_input_buffer();report['firmware_executed']=True
                session(pc,port,ref,windows,selected,report,trace,all_cases=args.all)
    except Exception as error:
        report['status']='ERROR';report['error']=f'{type(error).__name__}: {error}';print('ERROR:',report['error'],file=sys.stderr)
    finally:
        report['utc_finished']=datetime.now(timezone.utc).isoformat()
        report['completed_runs']=sum(row.get('status')=='PASS' for row in report['results'])
        report['client_sha256']=__import__('hashlib').sha256(Path(__file__).read_bytes()).hexdigest()
        for name,value in (('resultados.json',report),('uart_trace.jsonl',trace)):
            with (folder/name).open('x',encoding='utf-8',newline='\n') as stream:
                if name.endswith('jsonl'):
                    for row in value:stream.write(json.dumps(row,ensure_ascii=False)+'\n')
                else:json.dump(value,stream,ensure_ascii=False,indent=2);stream.write('\n')
        digests={name:__import__('hashlib').sha256((folder/name).read_bytes()).hexdigest() for name in ('resultados.json','uart_trace.jsonl')}
        with (folder/'MANIFEST_SHA256.json').open('x',encoding='utf-8') as stream:json.dump(digests,stream,indent=2)
        print('Evidencias:',folder,flush=True)
    return 0 if report['status']=='PASS' else 1

if __name__=='__main__':raise SystemExit(main())
