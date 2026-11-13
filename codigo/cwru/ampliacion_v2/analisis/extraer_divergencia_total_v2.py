"""Recupera la primera divergencia TotalV2 desde trazas; sin TF ni hardware.

Conserva la ejecucion fallida y sus sellos. Escribe exclusivamente un caso
derivado y su manifiesto en una carpeta de diagnostico nueva.
"""
from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path
from datetime import datetime
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT/'despliegue_stm32/costo_total_representantes_v2_180mhz_2026-10-25'
RUN = OUT/'Validacion_PC/resultados/20261026T042424_323718Z'
DEST = OUT/'diagnostico_divergencia_20261026T042424Z'
REPORT = DEST/'CASO_FALLO.json'


def require(ok, message):
    if not ok: raise ValueError(message)


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def manifest(path, base):
    digest,name=path.with_suffix('.sha256').read_text(encoding='ascii').split()
    require(name==path.name and sha(path)==digest,'Sello distinto: '+str(path))
    entries=read(path)
    for relative,expected in entries.items():
        target=(base/relative).resolve()
        require(target.is_relative_to(base.resolve()) and target.is_file() and sha(target)==expected,
                'Entrada modificada: '+relative)
    return entries


def load_module(path, name):
    spec=importlib.util.spec_from_file_location(name,path)
    require(spec is not None and spec.loader is not None,'Modulo no importable')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod


def write(path, value):
    with path.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')


class Answers:
    def __init__(self, answers): self.answers=answers
    def ask(self, command, prefix):
        line=self.answers[command]
        require(line.startswith(prefix),'Prefijo de pagina distinto')
        return line


def main():
    require(not REPORT.exists(),'No sobrescribir caso derivado')
    original=manifest(RUN/'MANIFEST_EVIDENCIAS_SHA256.json',RUN)
    exported=manifest(OUT/'recursos/MANIFEST_RECURSOS_SHA256.json',OUT/'recursos')
    result=read(RUN/'RESULTADO_TOTAL.json');plan=read(RUN/'PLAN_EJECUTADO.json')
    client_path=RUN/'validar_total_ejecutado.py'
    require(sha(client_path)==result['client_sha256']==sha(OUT/'Validacion_PC/validar_total.py'),
            'Cliente distinto de la ejecucion')
    v=load_module(client_path,'frozen_failed_total_client')
    independent=load_module(Path(__file__).with_name('auditar_numerica_gate_total_v2.py'),
                            'independent_total_f32_for_failure')
    cat,bank,windows,_=v.load_resources(OUT/'recursos')
    for name in ('CATALOGO_TOTAL.json','CATALOGO_CANONICO.json','BANCO_TOTAL.json'):
        require(sha(RUN/name)==sha(OUT/'recursos'/name),'Referencia de otra exportacion')
    require(result['status']=='FAILED_INCOMPLETE' and result['mode']=='full'
            and result['error']['message']=='Salida INT8 STM32 distinta de PC para LA MISMA entrada'
            and result['completed_primary_RUN']==66 and result['completed_PROFILE']==22
            and result['completed_BASE']==2 and result['negative_rejections']==0,'Otro fallo o conteos')
    require(sha(RUN/'firmware_declarado.elf')==result['firmware_elf_sha256'], 'ELF declarado distinto')
    trace=[json.loads(x) for x in (RUN/'TRAZA_UART.jsonl').read_text(encoding='utf-8').splitlines()]
    saved=[json.loads(x) for x in (RUN/'MEDICIONES.jsonl').read_text(encoding='utf-8').splitlines()]
    require([x['seq'] for x in trace]==list(range(1,len(trace)+1)) and len(trace)%2==0,'Secuencia UART')
    require([x['seq'] for x in saved]==list(range(1,len(saved)+1)),'Secuencia eventos')
    pairs=[];last_date=None
    for tx,rx in zip(trace[::2],trace[1::2]):
        require(tx['direction']=='TX' and rx['direction']=='RX','Emparejamiento TX/RX')
        for entry in (tx,rx):
            stamp=datetime.fromisoformat(entry['utc'])
            require(last_date is None or stamp>=last_date,'Timestamps no monotonicos');last_date=stamp
        pairs.append((tx['command'],rx['line']))
    require(not any('_ERROR,' in answer for _,answer in pairs),'Hubo error UART antes del desacuerdo')
    rows={r['global_id']:r for r in bank['windows']};by_checksum={r['checksum_fnv1a']:r for r in rows.values()}
    saved_runs={r['run_id']:r for r in saved if r['type'] in ('RUN','PROFILE')}
    infos=[];models={};selections=[];bases={};raw_cases=[];publications=[]
    received=[];frame_count=0;row=None;selected=None;last=None;current=None;pages={}
    for command,line in pairs:
        if command=='CWRUTOTAL_INFO':
            info=v.parse_info(line,cat);infos.append(info)
            if last is None:last=info['last_run_id']
        elif command=='CWRUFEAT_BEGIN':
            require(line=='CWRUFEAT_READY,8192','BEGIN invalido');received=[];frame_count=0
        elif command.startswith('CWRUFEAT_DATA,'):
            fields=command.split(',');offset=v.dec(fields[1]);values=np.asarray(fields[2:],dtype='<f4')
            require(offset==len(received) and 1<=len(values)<=48 and np.isfinite(values).all(),'DATA invalida')
            received.extend(values.tolist());frame_count+=1
            require(line==f'CWRUFEAT_ACK,{len(received)}','ACK distinto')
        elif command=='CWRUFEAT_CHECK':
            require(len(received)==8192 and frame_count==171,'RAW incompleto')
            raw=np.asarray(received,dtype='<f4').tobytes();checksum=v.fnv(raw)
            require(line==f'CWRUFEAT_CHECK,8192,{checksum}' and checksum in by_checksum,'CHECK invalido')
            row=by_checksum[checksum]
            require(raw==windows[row['npz_index']].tobytes()
                    and hashlib.sha256(raw).hexdigest()==row['payload_sha256'],'RAW no exacto al banco')
            raw_cases.append(dict(global_id=row['global_id'],frames=frame_count,values=len(received),
                payload_sha256=row['payload_sha256'],checksum_fnv1a=checksum,bit_exact_authorized_bank=True))
        elif command.startswith('CWRUMASK_BASE,'):
            require(row is not None and command.split(',')[1]==row['checksum_fnv1a'],'BASE despareada')
            bases[row['global_id']]=v.baseline(Answers({command:line}),row)
        elif command.startswith('CWRUTOTAL_MODEL,'):
            rid=int(command.split(',')[1]);v.validate_model(line,cat['representatives'][rid]);models[rid]=line
        elif command.startswith('CWRUTOTAL_SELECT,'):
            rid=int(command.split(',')[1]);selection=v.parse_selected(line,cat['representatives'][rid])
            selection['global_id']=row['global_id'];selections.append(selection);selected=rid
        elif command.startswith(('CWRUTOTAL_RUN,','CWRUTOTAL_PROFILE,')):
            require(current is None and row is not None and selected is not None and last is not None,
                    'RUN sin estado completo')
            mode=int(command.startswith('CWRUTOTAL_PROFILE,'));rep=cat['representatives'][selected]
            require(selected in row['authorized_rep_ids'],'Modelo no autorizado para esta ventana')
            current=v.parse_result(line,rep,row,last,mode,selections[-1]['arena_used'])
            current['result_line']=line;last=current['run_id'];pages={}
        elif command.startswith(('CWRUTOTAL_FEATURES,','CWRUTOTAL_INPUT,','CWRUTOTAL_DETECTOR,')):
            require(current is not None and int(command.split(',')[1])==current['run_id'],'Pagina sin RUN')
            pages[command]=line
            if command.startswith('CWRUTOTAL_DETECTOR,'):
                rep=cat['representatives'][current['rep_id']];parsed=v.get_pages(Answers(pages),current,rep)
                base=bases[row['global_id']]
                expected=[base['features_bits'][i] if i in rep['feature_indices'] else '00000000' for i in range(14)]
                require(parsed['features_bits']==expected,'Kernel no exacto a BASE')
                replay=independent.independent_replay(parsed['features_bits'],rep)
                require(all(parsed[k]==value for k,value in replay.items()),'Replay FP32 distinto')
                require(current['score_bits']==parsed['score_bits'] and current['rejected']==parsed['rejected'],
                        'Distancia o decision despareada')
                ref=bank['reference_by_rep_gid'][str(rep['id'])][str(row['global_id'])]
                q_equal=parsed['quantized']==ref['int8_inputs']
                same_output=(current['output_int8']==ref['int8_outputs']) if q_equal else None
                pub=dict(**current,pages=parsed,page_lines=pages.copy(),replay_from_ARM_features_exact=True,
                    quantized_input_exact_to_PC_cache=q_equal,PC_cache=ref,
                    output_exact_to_PC_cache_for_same_input=same_output)
                if current['run_id'] in saved_runs:
                    e=saved_runs[current['run_id']]
                    require(all(current[k]==e[k] for k in current if k!='result_line') and parsed==e['pages'],
                            'Publicacion no reproduce MEDICIONES')
                    require(same_output is not False,'Medicion PASS previa contradice cache con igual q')
                    pub['was_saved_as_PASS']=True
                else:pub['was_saved_as_PASS']=False
                publications.append(pub);current=None;pages={}
        else:raise ValueError('Comando no esperado: '+command)
    require(current is None and len(infos)==1 and len(raw_cases)==2 and len(publications)==89,'Traza truncada')
    require([r['global_id'] for r in raw_cases]==[175,232],'Otros RAW')
    require([p['run_id'] for p in publications]==list(range(5,94)),'IDs RUN distintos')
    preceding=publications[:-1];failed=publications[-1];rid=failed['rep_id'];gid=failed['global_id']
    require(len(preceding)==88 and all(p['was_saved_as_PASS'] for p in preceding)
            and sum(p['mode']==0 for p in preceding)==66 and sum(p['mode']==1 for p in preceding)==22,
            'Conteo previo no coincide')
    require(rid==22 and gid==232 and failed['run_id']==93 and failed['mode']==0
            and not failed['was_saved_as_PASS'] and failed['quantized_input_exact_to_PC_cache']
            and failed['output_exact_to_PC_cache_for_same_input'] is False,'Caso final no es el desacuerdo esperado')
    rep=cat['representatives'][rid];ref=failed['PC_cache'];arm=failed['output_int8'];pc=ref['int8_outputs']
    arm_features=v.from_bits(failed['pages']['features_bits']);pc_features=v.from_bits(ref['raw_features14_bits'])
    active_differences=[dict(index=i,ARM_bits=failed['pages']['features_bits'][i],PC_cache_bits=ref['raw_features14_bits'][i],
        absolute_error=abs(float(arm_features[i])-float(pc_features[i])))
        for i in rep['feature_indices'] if failed['pages']['features_bits'][i]!=ref['raw_features14_bits'][i]]
    report=dict(status='FAILED_CASE_RECOVERED_AND_OFFLINE_AUDITED',date_local='2026-10-26',timezone='America/Lima',
        original_run=str(RUN),original_run_status=result['status'],error=result['error'],
        original_manifest_sha256=sha(RUN/'MANIFEST_EVIDENCIAS_SHA256.json'),original_manifest_resources=len(original),
        resource_manifest_resources=len(exported),script_sha256=sha(__file__),
        source_files_sha256={name:sha(RUN/name) for name in original},trace_rows=len(trace),transactions=len(pairs),
        saved_measurements=len(saved),RAW=raw_cases,preceding_PASS=dict(primary_RUN=66,PROFILE=22,model_window_pairs=22,
            reconstructed_publications=88,all_saved_records_reproduced=True,
            quantized_inputs_exact_to_PC_cache=sum(p['quantized_input_exact_to_PC_cache'] for p in preceding),
            output_exact_to_PC_cache_for_same_input=sum(p['output_exact_to_PC_cache_for_same_input'] is True for p in preceding)),
        failed_publication=failed,representative=rep,global_window_metadata=rows[gid]['metadata'],
        model_received_line=models[rid],selection=selections[-1],
        ARM_output_INT8=arm,PC_cache_output_INT8=pc,output_delta_ARM_minus_PC=[a-b for a,b in zip(arm,pc)],
        ARM_class=failed['predicted_class'],PC_cache_class=int(np.argmax(pc)),same_predicted_class=failed['predicted_class']==int(np.argmax(pc)),
        input_INT8=failed['pages']['quantized'],input_hex_le=bytes(np.asarray(failed['pages']['quantized'],dtype=np.int8)).hex(),
        input_exact_to_original_cache=True,PC_cache_output_is_valid_same_input_oracle=True,
        active_feature_differences_to_PC_cache=active_differences,
        ARM_detector_score=float(v.from_bits([failed['score_bits']])[0]),
        PC_cache_detector_score=float(v.from_bits([ref['detector_score_bits']])[0]),
        detector_threshold=float(v.from_bits([rep['threshold_bits']])[0]),
        detector_decision_same_as_cache=failed['rejected']==ref['outside_normality'],
        extracted_pipeline_stages_before_DNN_output_comparison_pass=True,
        bad_publication_not_in_MEDICIONES_but_preserved_in_UART=True,
        actual_emitted_primary_RUN=67,actual_emitted_PROFILE=22,
        firmware_elf_sha256=result['firmware_elf_sha256'],catalog_sha256=result['catalog_sha256'],
        TFLite_invocations=0,new_physical_RUN=0,serial_port_opened=False,hardware_flashed=False,
        original_files_modified=False,selector_costs_enabled=False,
        limitations=['No backend root cause established by this extraction.',
            'Same class does not satisfy the frozen exact INT8 output contract.',
            'No programmer log/readback proof of the entire programmed image was supplied.',
            'The campaign remains incomplete; the extra emitted RUN was not accepted as a valid timing sample.'])
    DEST.mkdir(parents=True,exist_ok=True);write(REPORT,report)
    entries={REPORT.relative_to(ROOT).as_posix():sha(REPORT),Path(__file__).resolve().relative_to(ROOT).as_posix():sha(__file__)}
    mp=DEST/'MANIFEST_CASO_FALLO_SHA256.json';write(mp,entries)
    with mp.with_suffix('.sha256').open('x',encoding='ascii') as stream:stream.write(sha(mp)+'  '+mp.name+'\n')
    require(manifest(RUN/'MANIFEST_EVIDENCIAS_SHA256.json',RUN)==original,'Original cambio durante auditoria')
    print(json.dumps(dict(status=report['status'],case=str(REPORT),rep_id=rid,global_id=gid,run_id=93,
        input=report['input_INT8'],ARM=arm,PC_cache=pc,delta=report['output_delta_ARM_minus_PC'],
        same_class=report['same_predicted_class'],preceding_PASS=88,report_sha256=sha(REPORT)),ensure_ascii=False))


if __name__=='__main__':main()
