"""Completa el banco canónico reutilizando el gate auditado, sin repetirlo.

No compila, entrena o programa Flash. --check es offline. --full abre UART
únicamente después de autenticar el gate, el mismo ELF y su carga verificada.
Los tiempos POST, perfilado e inicialización no son costo principal RAW.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=next(p for p in HERE.parents if (p/'codigo/cwru/ampliacion_v2').is_dir()
    and (p/'despliegue_stm32').is_dir())
OUT=ROOT/'despliegue_stm32/canonicos_42_180mhz_2026-10-27'
PACKAGE=OUT/'Validacion_Completa_PC'
GATE_FOLDER=OUT/'Validacion_PC/resultados/20261028T050129_905025Z'
GATE_AUDIT=OUT/'AUDITORIA_GATE_CANON42.json'
GATE_ORIGINAL=OUT/'Validacion_PC/validar_canonicos_gate.py'
GATE_NAME='gate_canonicos_congelado.py'
GATE_SHA='24e02e37c2d7f3ac855f674543ed7c60af4bf87a91ea1f16ee087253054dbe9c'
HELPER_NAME='protocolo_total_sr_legacy.py'
OLD_JOINT=OUT/'MANIFEST_CLIENTES_CANON42_SHA256.json'
OLD_JOINT_SHA='b1cbbc5d78ee3268cef2b2e412b9f83a4a139ca5756940ff8ff82dddb467c238'
FULL_MANIFEST=PACKAGE/'MANIFEST_CLIENTE_FULL_SHA256.json'
ATTEMPT=PACKAGE/'INTENTO_VALIDACION_COMPLETA.json'
FULL_STATUS='CANON42_STAGED_FULL_PASS_PENDING_TRACE_AUDIT'
TOTAL_COUNTS=dict(RAW_unique_windows=56,RAW_model_window_pairs=336,RAW_RUN=1008,
    RAW_PROFILE=336,RAW_BASE=56,POST_unique_reserved_windows=1380,POST_model_window_pairs=2760)
IMPORTED_COUNTS=dict(RAW_RUN=24,RAW_PROFILE=8,RAW_BASE=2,POST=24)


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path,value):
    with Path(path).open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')


def gate_module():
    path=HERE/GATE_NAME if (HERE/GATE_NAME).is_file() else GATE_ORIGINAL
    require(sha(path)==GATE_SHA,'Cliente gate distinto del ejecutado y auditado')
    spec=importlib.util.spec_from_file_location('full_canon42_frozen_gate',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def verify_root_manifest(path,expected=None):
    path=Path(path);digest,name=path.with_suffix('.sha256').read_text(encoding='ascii').split()
    require(name==path.name and sha(path)==digest and (expected is None or digest==expected),
        'Sello de procedencia modificado: '+str(path))
    entries=read(path);require(isinstance(entries,dict) and entries,'Sello vacío')
    for relative,value in entries.items():
        target=(ROOT/relative).resolve()
        require(target.is_relative_to(ROOT.resolve()) and target!=ROOT.resolve()
            and target.is_file() and sha(target)==value,'Archivo sellado modificado: '+relative)
    return entries,digest


def load_imports(g,cat,bank,arrays,audit_required=True):
    entries=g.manifest(GATE_FOLDER,'MANIFEST_EVIDENCIAS_SHA256.json')
    report=read(GATE_FOLDER/'RESULTADO_GATE.json')
    require(report['status']==g.GATE_STATUS and report['firmware_elf_sha256']==g.ELF_SHA
        and report['catalog_sha256']==g.CATALOG_SHA and report['client_sha256']==GATE_SHA
        and report['joint_client_manifest_sha256']==OLD_JOINT_SHA,'Gate de otra procedencia')
    audit=None
    if audit_required:
        audit=read(GATE_AUDIT)
        require(audit['status']=='PASS_GATE_CANON42_AUDITED_PENDING_FULL'
            and Path(audit['gate_folder']).resolve()==GATE_FOLDER.resolve()
            and audit['gate_evidence_manifest_sha256']==sha(GATE_FOLDER/'MANIFEST_EVIDENCIAS_SHA256.json')
            and audit['firmware_elf_sha256']==g.ELF_SHA and audit['catalog_sha256']==g.CATALOG_SHA,
            'El gate no tiene una auditoría aprobada de estas evidencias')
    events=[json.loads(line) for line in (GATE_FOLDER/'MEDICIONES.jsonl').read_text(encoding='utf-8').splitlines()]
    bases=[dict(row) for row in events if row['type']=='BASE']
    runs=[dict(row) for row in events if row['type'] in ('RUN','PROFILE')]
    posts=[dict(row) for row in events if row['type']=='POST']
    require(len(bases)==2 and len(runs)==32 and len(posts)==24,'Gate importado incompleto')
    by_base={row['global_id']:row for row in bases}
    require(len(by_base)==2,'BASE duplicada en gate')
    raw_keys={};h=g.helper()
    for row in runs:
        key=(row['global_id'],row['rep_id'],row['mode'],row['repetition'])
        require(key not in raw_keys and row['status']=='PASS','RUN/PROFILE duplicado o fallido')
        raw_keys[key]=row
        require(row['global_id'] in bank['models'][row['rep_id']]['raw_global_window_ids'],
            'Gate RAW fuera de las cargas de desarrollo')
        ref=g.reference(arrays['rawrefs'],bank['models'][row['rep_id']],row['global_id'])
        # El auditor garantiza la salida de red. Se repiten solo cálculos de
        # integridad f32; no se invoca un clasificador al preparar/importar.
        g.compare_intermediates(h,row,row['pages'],cat['representatives'][row['rep_id']],
            h.from_bits(by_base[row['global_id']]['features_bits']))
        require(row['comparison']['same_input_INT8_output_exact'] is True
            and row['comparison']['selected_features_exact_BASE'] is True,'Gate RAW sin equivalencia')
    pairs={(key[0],key[1]) for key in raw_keys}
    require(len(pairs)==8 and all({(mode,rep) for gid,rid,mode,rep in raw_keys if (gid,rid)==pair}
        =={(0,1),(0,2),(0,3),(1,1)} for pair in pairs),'Gate RAW con repeticiones incompletas')
    post_keys=set()
    for row in posts:
        key=(row['rep_id'],row['global_id'])
        require(key not in post_keys and row['status']=='PASS','POST gate duplicado o fallido')
        post_keys.add(key);rep=cat['representatives'][row['rep_id']]
        ref=g.reference(arrays['postrefs'],bank['models'][row['rep_id']],row['global_id'])
        g.compare_post(h,row,row['pages'],rep,ref)
    for row in [*bases,*runs,*posts]:
        row.pop('seq',None);row['measurement_source']='AUDITED_GATE_IMPORT'
        row['imported_from_gate']=str(GATE_FOLDER);row['included_without_reexecution']=True
    return dict(report=report,audit=audit,entries=entries,bases=bases,runs=runs,posts=posts,
        raw_pairs=pairs,post_keys=post_keys,events=events,
        gate_manifest_sha256=sha(GATE_FOLDER/'MANIFEST_EVIDENCIAS_SHA256.json'))


def make_plan(cat,bank,arrays,imports):
    raw=[];all_raw=set();baseline_ids={row['global_id'] for row in imports['bases']}
    for row in bank['RAW_windows']:
        gid=row['global_id'];authorized=row['authorized_model_ids']
        all_raw.update((gid,rid) for rid in authorized)
        remaining=[rid for rid in authorized if (gid,rid) not in imports['raw_pairs']]
        if remaining:raw.append(dict(global_id=gid,model_ids=remaining,
            new_BASE=gid not in baseline_ids,BASE_origin='FULL_NEW_UART' if gid not in baseline_ids else 'AUDITED_GATE_IMPORT'))
    post=[];all_post=set()
    for model in bank['models']:
        rid=model['id'];prefix=model['reference_prefix']
        for i,gid in enumerate(model['post_global_window_ids']):
            key=(rid,gid);require(key not in all_post,'POST fuente duplicado');all_post.add(key)
            if key not in imports['post_keys']:
                post.append(dict(model_id=rid,global_id=gid,label=int(arrays['postrefs'][prefix+'y_true'][i])))
    require(imports['raw_pairs']<=all_raw and imports['post_keys']<=all_post,'Gate no es subconjunto del banco')
    remaining_raw={(row['global_id'],rid) for row in raw for rid in row['model_ids']}
    require(not remaining_raw&imports['raw_pairs'] and remaining_raw|imports['raw_pairs']==all_raw,
        'El plan repite o pierde pares RAW')
    require(len(all_raw)==336 and len(all_post)==2760 and len(raw)==55
        and len(remaining_raw)==328 and len(post)==2736 and sum(r['new_BASE'] for r in raw)==54,
        'Cobertura completa inesperada')
    return dict(mode='full',total_counts=TOTAL_COUNTS,imported_counts=IMPORTED_COUNTS,
        new_counts=dict(RAW_transmissions=55,RAW_model_window_pairs=328,RAW_RUN=984,
            RAW_PROFILE=328,RAW_BASE=54,POST=2736),RAW_schedule=raw,POST_cases=post,
        gate_manifest_sha256=imports['gate_manifest_sha256'],RAW_coverage='Partial development RAW: 42 windows per model; not exhaustive reserved extraction.',
        POST_coverage='All 1380 reserved PC feature vectors in two models per outer load.',
        old_representative_timings_imported=False,current_audited_gate_timings_imported=True,
        already_completed_cases_reexecuted=False,POST_timing_is_full_pipeline=False,
        selection_uses_accuracy_or_confidence=False,automatic_resume=False,automatic_retry=False)


def load_context(args,audit_required=True):
    g=gate_module();verify_root_manifest(OLD_JOINT,OLD_JOINT_SHA)
    g.manifest(OUT/'Validacion_PC','MANIFEST_CLIENTE_GATE_SHA256.json')
    h,cat,bank,arrays=g.load_resources();provenance=g.authenticate_build(args,cat)
    programming=g.authenticate_programming(OLD_JOINT_SHA)
    imports=load_imports(g,cat,bank,arrays,audit_required)
    require(imports['report']['programming_evidence']==programming,'La carga del gate cambió')
    return g,h,cat,bank,arrays,provenance,programming,imports,make_plan(cat,bank,arrays,imports)


def own_manifest():
    entries,digest=verify_root_manifest(FULL_MANIFEST)
    required=[Path(__file__),HERE/GATE_NAME,HERE/HELPER_NAME,PACKAGE/'PLAN_COMPLETO.json',GATE_AUDIT]
    require(all(p.resolve().relative_to(ROOT.resolve()).as_posix() in entries for p in required),
        'Sello FULL incompleto')
    return digest


def prepare(args):
    g,h,cat,bank,arrays,provenance,programming,imports,plan=load_context(args)
    require(not PACKAGE.exists(),'No sobrescribir entrega FULL existente')
    sources=[(Path(__file__),'validar_canonicos_full.py'),(GATE_ORIGINAL,GATE_NAME),
        (OUT/'Validacion_PC'/HELPER_NAME,HELPER_NAME)]
    paths=[Path(__file__),HERE/'test_validar_canonicos_full.py',GATE_ORIGINAL,
        OUT/'Validacion_PC'/HELPER_NAME,GATE_AUDIT,OLD_JOINT,OLD_JOINT.with_suffix('.sha256'),
        Path(args.firmware_elf),Path(args.build_report),Path(args.elf_audit),
        OUT/'recursos/MANIFEST_RECURSOS_SHA256.json',OUT/'bancos/MANIFEST_BANCOS_SHA256.json']
    paths += [p for p in GATE_FOLDER.rglob('*') if p.is_file()]
    paths += [ROOT/relative for relative in programming['files_sha256']]
    # Comprobar todas las fuentes antes de crear la entrega, para que un
    # requisito ausente no deje una carpeta parcial que bloquee --prepare.
    require(all(p.is_file() and p.resolve().is_relative_to(ROOT.resolve()) for p in paths),
        'Fuente FULL ausente o fuera del proyecto')
    PACKAGE.mkdir()
    for source,name in sources:
        with (PACKAGE/name).open('xb') as stream:stream.write(source.read_bytes())
    write(PACKAGE/'PLAN_COMPLETO.json',plan)
    write(PACKAGE/'PROCEDENCIA_FULL.json',dict(status='FULL_CLIENT_READY_PENDING_USER_EXECUTION',
        firmware_elf_sha256=g.ELF_SHA,catalog_sha256=g.CATALOG_SHA,gate_audit_sha256=sha(GATE_AUDIT),
        gate_manifest_sha256=imports['gate_manifest_sha256'],old_client_manifest_sha256=OLD_JOINT_SHA,
        completed_gate_imported_without_reexecution=True,training_fits=0,serial_opened=False,hardware_flashed=False))
    paths += [p for p in PACKAGE.iterdir() if p.is_file()]
    entries={p.resolve().relative_to(ROOT.resolve()).as_posix():sha(p) for p in sorted(set(paths))}
    write(FULL_MANIFEST,entries)
    with FULL_MANIFEST.with_suffix('.sha256').open('x',encoding='ascii') as stream:
        stream.write(sha(FULL_MANIFEST)+'  '+FULL_MANIFEST.name+'\n')
    print(json.dumps(dict(status='FULL_CLIENT_READY_PENDING_USER_EXECUTION',package=str(PACKAGE),
        imported=IMPORTED_COUNTS,new=plan['new_counts'],serial_opened=False,hardware_flashed=False),ensure_ascii=True))


def check(args):
    require(HERE.resolve()==PACKAGE.resolve(),'Comprobar la copia entregada en Validacion_Completa_PC')
    seal=own_manifest();context=load_context(args);plan=context[-1]
    require(plan==read(PACKAGE/'PLAN_COMPLETO.json'),'Plan FULL distinto del preparado')
    result=dict(status='CANON42_FULL_PREFLIGHT_PASS_NO_SERIAL_OPENED',client_manifest_sha256=seal,
        imported=IMPORTED_COUNTS,new=plan['new_counts'],serial_opened=False,hardware_flashed=False)
    print(json.dumps(result,ensure_ascii=True));return result


def tag_import(event):
    result={k:v for k,v in event.items() if k!='seq'}
    result.update(measurement_source='AUDITED_GATE_IMPORT',gate_event_seq=event['seq'],
        imported_from_gate=str(GATE_FOLDER),included_without_reexecution=True)
    return result


def coverage(runs,bases,posts,bank):
    run_keys={(r['global_id'],r['rep_id'],r['mode'],r['repetition']) for r in runs}
    pairs={(gid,rid) for gid,rid,mode,rep in run_keys}
    require(len(runs)==len(run_keys)==1344 and len(pairs)==336,'RAW incompleto o duplicado')
    require(all({(mode,rep) for gid,rid,mode,rep in run_keys if (gid,rid)==pair}
        =={(0,1),(0,2),(0,3),(1,1)} for pair in pairs),'Repeticiones RAW incompletas')
    require(len(bases)==len({r['global_id'] for r in bases})==56,'BASE incompleto o duplicado')
    post_keys={(r['rep_id'],r['global_id']) for r in posts}
    require(len(posts)==len(post_keys)==2760 and len({r['global_id'] for r in posts})==1380,
        'POST incompleto o duplicado')
    require(all(sum(r['rep_id']==rid for r in runs if r['mode']==0)==126 for rid in range(8)),
        'Faltan 126 RUN por modelo')
    expected_pairs={(row['global_id'],rid) for row in bank['RAW_windows'] for rid in row['authorized_model_ids']}
    expected_posts={(row['id'],gid) for row in bank['models'] for gid in row['post_global_window_ids']}
    require(pairs==expected_pairs and post_keys==expected_posts
        and {row['global_id'] for row in bases}=={row['global_id'] for row in bank['RAW_windows']},
        'Cobertura contiene IDs distintos del banco congelado')
    return dict(RAW_RUN=1008,RAW_PROFILE=336,RAW_BASE=56,RAW_pairs=336,POST=2760,POST_unique=1380)


def run(args):
    started=time.monotonic()
    require(args.full and args.port and HERE.resolve()==PACKAGE.resolve(),'Se requiere --full desde la copia y --port')
    require(60<=args.max_seconds<=1800 and 1<=args.timeout<=30,'Presupuesto FULL fuera de límites')
    seal=own_manifest()
    g,h,cat,bank,arrays,provenance,programming,imports,plan=load_context(args)
    require(plan==read(PACKAGE/'PLAN_COMPLETO.json'),'Plan FULL modificado')
    require(not ATTEMPT.exists(),'Ya hubo un intento FULL; conservar evidencias y revisar antes de reintentar')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output=PACKAGE/'resultados'/stamp;output.mkdir(parents=True,exist_ok=False)
    write(ATTEMPT,dict(attempt=1,output=str(output),client_manifest_sha256=seal,
        gate_evidence_manifest_sha256=imports['gate_manifest_sha256'],max_seconds=args.max_seconds,
        automatic_resume=False,automatic_retry=False,execution_owner='user_CLI'))
    for source,name in [(Path(__file__),'validar_canonicos_full_ejecutado.py'),(HERE/GATE_NAME,GATE_NAME),
        (HERE/HELPER_NAME,HELPER_NAME),(GATE_AUDIT,'AUDITORIA_GATE_CANON42.json'),
        (provenance['firmware'],'firmware_declarado.elf'),(provenance['build'],'COMPILACION_CANON42.json'),
        (provenance['audit'],'AUDITORIA_ELF_CANON42.json')]:shutil.copy2(source,output/name)
    shutil.copytree(GATE_FOLDER,output/'gate_importado')
    write(output/'PLAN_EJECUTADO.json',dict(**plan,max_seconds=args.max_seconds,
        client_manifest_sha256=seal,python=sys.version,gate_audit_sha256=sha(GATE_AUDIT)))
    trace,events=h.Trace(output/'TRAZA_UART_NUEVA.jsonl'),h.Trace(output/'MEDICIONES.jsonl')
    for event in imports['events']:events.append(tag_import(event))
    runs,bases,posts=list(imports['runs']),list(imports['bases']),list(imports['posts'])
    base_by_id={row['global_id']:row for row in bases}
    oracle=g.RawOracle(h,cat['representatives'])
    for row in runs:oracle.cache[(row['rep_id'],tuple(row['pages']['quantized']))]=row['output_int8']
    report=dict(status='FAILED_INCOMPLETE',mode='full',firmware_elf_sha256=g.ELF_SHA,
        catalog_sha256=g.CATALOG_SHA,gate_manifest_sha256=imports['gate_manifest_sha256'],
        gate_audit_sha256=sha(GATE_AUDIT),client_manifest_sha256=seal,client_sha256=sha(__file__),
        old_joint_client_manifest_sha256=OLD_JOINT_SHA,programming_evidence=programming,
        imported_counts=IMPORTED_COUNTS,new_completed=dict(RAW_RUN=0,RAW_PROFILE=0,RAW_BASE=0,POST=0,RAW_transmissions=0),
        total_completed=dict(IMPORTED_COUNTS),full_validation_completed=False,
        RAW_exhaustive_reserved_coverage=False,POST_exhaustive_reserved_coverage=False,
        RAW_partial_development_bank=True,POST_timing_is_full_pipeline=False,
        training_fits=0,models_modified=False,hardware_flashed=False,
        selector_enabled=False,current_gate_timings_imported=True,old_representative_timings_imported=False,
        identity_from_programmer_and_UART=True,whole_Flash_readback_performed=False,
        excluded_from_RAW_primary_cost=['UART','parser','checksums','SELECT','BASE','PROFILE','POST'],
        cached_PC_end_to_end_exactness_claimed=False,physical_audit_pending=True)
    print('Evidencias: '+str(output),flush=True)
    try:
        import serial
        with serial.Serial(args.port,args.baud,timeout=.10,write_timeout=args.timeout) as port:
            link=h.Link(port,trace,args.timeout,deadline=started+args.max_seconds)
            info=g.parse_info(h,link.ask('CWRUTOTAL_INFO','CWRUTOTAL_INFO,'),cat)
            require(link.ask('CWRUTOTAL_POST_INFO','CWRUTOTAL_POST_INFO,')=='CWRUTOTAL_POST_INFO,1,14,56,0','POST_INFO incompatible')
            report['firmware_info']=info;last=info['last_run_id']
            for rid in range(8):h.validate_model(link.ask(f'CWRUTOTAL_MODEL,{rid}','CWRUTOTAL_MODEL,'),cat['representatives'][rid])
            by_id={row['global_id']:row for row in bank['RAW_windows']}
            for index,case in enumerate(plan['RAW_schedule'],1):
                gid=case['global_id'];row=dict(by_id[gid],arm_features14_bits=None)
                h.transmit(link,arrays['raw']['windows'][row['npz_index']],row)
                report['new_completed']['RAW_transmissions']+=1
                events.append(dict(type='TRANSMIT',global_id=gid,checksum=row['checksum_fnv1a'],measurement_source='FULL_NEW_UART'))
                if case['new_BASE']:
                    base=h.baseline(link,row);base['measurement_source']='FULL_NEW_UART'
                    bases.append(base);base_by_id[gid]=base;events.append(dict(type='BASE',**base))
                    report['new_completed']['RAW_BASE']+=1;report['total_completed']['RAW_BASE']+=1
                else:
                    base=base_by_id[gid]
                    events.append(dict(type='REUSE_BASE',global_id=gid,features_bits=base['features_bits'],
                        origin='AUDITED_GATE_IMPORT',new_extraction_performed=False,measurement_source='FULL_NEW_UART'))
                for rid in case['model_ids']:
                    rep,model=cat['representatives'][rid],bank['models'][rid]
                    selection=h.parse_selected(link.ask(f'CWRUTOTAL_SELECT,{rid}','CWRUTOTAL_SELECTED,'),rep)
                    events.append(dict(type='SELECT',global_id=gid,measurement_source='FULL_NEW_UART',**selection))
                    ref=g.reference(arrays['rawrefs'],model,gid);previous=None
                    for repetition,mode in [(1,0),(2,0),(3,0),(1,1)]:
                        command='CWRUTOTAL_PROFILE' if mode else 'CWRUTOTAL_RUN'
                        result=h.parse_result(link.ask(command+','+row['checksum_fnv1a'],'CWRUTOTAL_RESULT,'),
                            rep,row,last,mode,selection['arena_used'])
                        last=result['run_id'];pages=h.get_pages(link,result,rep)
                        comparison=g.compare_raw(h,result,pages,base,rep,ref,oracle)
                        require(previous is None or previous==pages,'RUN/PROFILE cambian los intermedios');previous=pages
                        result.update(repetition=repetition,status='PASS',pages=pages,comparison=comparison,
                            measurement_source='FULL_NEW_UART',baseline_origin=case['BASE_origin'])
                        runs.append(result);events.append(dict(type='PROFILE' if mode else 'RUN',**result))
                        key='RAW_PROFILE' if mode else 'RAW_RUN'
                        report['new_completed'][key]+=1;report['total_completed'][key]+=1
                    print(f'RAW {index}/55 {rep["representative_id"]}/gid{gid}: PASS; 3 RUN + PROFILE',flush=True)
            for index,case in enumerate(plan['POST_cases'],1):
                rid,gid=case['model_id'],case['global_id'];rep=cat['representatives'][rid]
                ref=g.reference(arrays['postrefs'],bank['models'][rid],gid)
                selection=h.parse_selected(link.ask(f'CWRUTOTAL_SELECT,{rid}','CWRUTOTAL_SELECTED,'),rep)
                events.append(dict(type='SELECT_POST',global_id=gid,measurement_source='FULL_NEW_UART',**selection))
                command,checksum=g.post_command(h,ref['features14'])
                result=g.parse_post_result(h,link.ask(command,'CWRUTOTAL_POST_RESULT,'),rep,checksum,last,selection['arena_used'])
                last=result['run_id'];pages=h.get_pages(link,result,rep)
                comparison=g.compare_post(h,result,pages,rep,ref)
                result.update(global_id=gid,label=case['label'],pages=pages,comparison=comparison,status='PASS',
                    measurement_source='FULL_NEW_UART')
                posts.append(result);events.append(dict(type='POST',**result))
                report['new_completed']['POST']+=1;report['total_completed']['POST']+=1
                if index%50==0 or index==len(plan['POST_cases']):print(f'POST {index}/2736: PASS',flush=True)
            final=g.parse_info(h,link.ask('CWRUTOTAL_INFO','CWRUTOTAL_INFO,'),cat)
            require(final['last_run_id']==last and final['result_valid'],'Publicación final incompatible')
            report['firmware_info_final']=final
        require(coverage(runs,bases,posts,bank)==dict(RAW_RUN=1008,RAW_PROFILE=336,RAW_BASE=56,RAW_pairs=336,POST=2760,POST_unique=1380),
            'Cobertura incompleta')
        require(report['new_completed']==dict(RAW_RUN=984,RAW_PROFILE=328,RAW_BASE=54,POST=2736,RAW_transmissions=55),'Contadores nuevos incompletos')
        require(own_manifest()==seal and g.authenticate_programming(OLD_JOINT_SHA)==programming,
            'Procedencia cambió durante FULL')
        report.update(status=FULL_STATUS,full_validation_completed=True,POST_exhaustive_reserved_coverage=True,
            physical_audit_pending=True,coverage=coverage(runs,bases,posts,bank))
    except BaseException as exc:
        report['error']=dict(type=type(exc).__name__,message=str(exc));raise
    finally:
        report.update(wall_seconds=time.monotonic()-started,RAW_timing_statistics=h.statistics(runs,cat['representatives']),
            PC_same_input_classifier_invocations_new=oracle.invocations,
            PC_same_input_classifier_invocations_imported=imports['report']['PC_same_input_classifier_invocations'],
            RAW_unique_window_ids=[r['global_id'] for r in bases],RAW_model_ids=sorted({r['rep_id'] for r in runs}),
            POST_model_ids=sorted({r['rep_id'] for r in posts}))
        trace.close();events.close();write(output/'RESULTADO_COMPLETO.json',report);g.seal_evidence(output)
        print(report['status']+'; evidencias: '+str(output),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare',action='store_true');mode.add_argument('--check',action='store_true');mode.add_argument('--full',action='store_true')
    p.add_argument('--port');p.add_argument('--baud',type=int,default=115200);p.add_argument('--timeout',type=float,default=15.)
    p.add_argument('--max-seconds',type=int,default=1800)
    p.add_argument('--firmware-elf',type=Path,default=OUT/'copia_aislada/Prueba_2_CWRU_Extraccion/Release_180MHz_Canon42_SR/Prueba_2_CWRU_Extraccion.elf')
    p.add_argument('--build-report',type=Path,default=OUT/'COMPILACION_CANON42.json')
    p.add_argument('--elf-audit',type=Path,default=OUT/'AUDITORIA_ELF_CANON42.json')
    args=p.parse_args()
    if args.prepare:prepare(args)
    elif args.check:check(args)
    else:run(args)


if __name__=='__main__':
    try:main()
    except KeyboardInterrupt:print('INTERRUPTED',file=sys.stderr);raise SystemExit(130)
    except Exception as exc:print('ERROR: '+str(exc),file=sys.stderr);raise SystemExit(1)
