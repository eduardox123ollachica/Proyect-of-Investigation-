"""Audita el FULL escalonado Canon42 desde trazas, sin UART ni inferencia.

La evidencia original permanece inmutable. La referencia PC guardada basta
para todas las entradas INT8 observadas en esta campaña real.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import auditar_gate_canonicos_v2 as a

ROOT=a.ROOT
OUT=a.OUT
PACKAGE=OUT/'Validacion_Completa_PC'
FOLDER=PACKAGE/'resultados/20261028T162240_469517Z'
REPORT=OUT/'AUDITORIA_FULL_CANON42.json'
EXPECTED_FULL_MANIFEST='5dc27ee495451a612ef668a7000995b40ba8a463c72a69661f6c8bb48015510e'
STATUS='PASS_CANON42_STAGED_FULL_AUDITED'
require,sha,read,canonical_sha=a.require,a.sha,a.read,a.canonical_sha


def load_full():
    path=PACKAGE/'validar_canonicos_full.py'
    spec=importlib.util.spec_from_file_location('audit_full_canon_frozen_client',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def reconstruct(f,g,h,cat,bank,arrays,imports,plan,trace,events):
    """Reproduce comandos y resultados exclusivamente desde texto archivado."""
    link=a.ReplayLink(trace)
    info=g.parse_info(h,link.ask('CWRUTOTAL_INFO','CWRUTOTAL_INFO,'),cat)
    require(link.ask('CWRUTOTAL_POST_INFO','CWRUTOTAL_POST_INFO,')=='CWRUTOTAL_POST_INFO,1,14,56,0',
        'POST_INFO incompatible')
    last=info['last_run_id']
    for rid in range(8):
        h.validate_model(link.ask(f'CWRUTOTAL_MODEL,{rid}','CWRUTOTAL_MODEL,'),cat['representatives'][rid])
    records=[]
    def emit(event):
        records.append(dict(event,seq=len(records)+1))
    for event in imports['events']:emit(f.tag_import(event))
    runs,bases,posts=list(imports['runs']),list(imports['bases']),list(imports['posts'])
    base_by_id={row['global_id']:row for row in bases}
    oracle=a.SavedOnlyOracle()
    selections=[]
    new=dict(RAW_RUN=0,RAW_PROFILE=0,RAW_BASE=0,POST=0,RAW_transmissions=0)
    by_id={row['global_id']:row for row in bank['RAW_windows']}
    for case in plan['RAW_schedule']:
        gid=case['global_id'];row=dict(by_id[gid],arm_features14_bits=None)
        h.transmit(link,arrays['raw']['windows'][row['npz_index']],row)
        new['RAW_transmissions']+=1
        emit(dict(type='TRANSMIT',global_id=gid,checksum=row['checksum_fnv1a'],measurement_source='FULL_NEW_UART'))
        if case['new_BASE']:
            base=h.baseline(link,row);base['measurement_source']='FULL_NEW_UART'
            bases.append(base);base_by_id[gid]=base
            emit(dict(type='BASE',**base));new['RAW_BASE']+=1
        else:
            base=base_by_id[gid]
            emit(dict(type='REUSE_BASE',global_id=gid,features_bits=base['features_bits'],
                origin='AUDITED_GATE_IMPORT',new_extraction_performed=False,measurement_source='FULL_NEW_UART'))
        for rid in case['model_ids']:
            rep,model=cat['representatives'][rid],bank['models'][rid]
            selection=h.parse_selected(link.ask(f'CWRUTOTAL_SELECT,{rid}','CWRUTOTAL_SELECTED,'),rep)
            selections.append(selection)
            emit(dict(type='SELECT',global_id=gid,measurement_source='FULL_NEW_UART',**selection))
            ref=g.reference(arrays['rawrefs'],model,gid);previous=None
            for repetition,mode in [(1,0),(2,0),(3,0),(1,1)]:
                command='CWRUTOTAL_PROFILE' if mode else 'CWRUTOTAL_RUN'
                result=h.parse_result(link.ask(command+','+row['checksum_fnv1a'],'CWRUTOTAL_RESULT,'),
                    rep,row,last,mode,selection['arena_used'])
                last=result['run_id'];pages=h.get_pages(link,result,rep)
                comparison=g.compare_raw(h,result,pages,base,rep,ref,oracle)
                require(previous is None or previous==pages,'RUN/PROFILE cambian intermedios')
                previous=pages
                result.update(repetition=repetition,status='PASS',pages=pages,comparison=comparison,
                    measurement_source='FULL_NEW_UART',baseline_origin=case['BASE_origin'])
                runs.append(result);emit(dict(type='PROFILE' if mode else 'RUN',**result))
                new['RAW_PROFILE' if mode else 'RAW_RUN']+=1
    for case in plan['POST_cases']:
        rid,gid=case['model_id'],case['global_id'];rep=cat['representatives'][rid]
        ref=g.reference(arrays['postrefs'],bank['models'][rid],gid)
        selection=h.parse_selected(link.ask(f'CWRUTOTAL_SELECT,{rid}','CWRUTOTAL_SELECTED,'),rep)
        selections.append(selection)
        emit(dict(type='SELECT_POST',global_id=gid,measurement_source='FULL_NEW_UART',**selection))
        command,checksum=g.post_command(h,ref['features14'])
        result=g.parse_post_result(h,link.ask(command,'CWRUTOTAL_POST_RESULT,'),rep,checksum,last,selection['arena_used'])
        last=result['run_id'];pages=h.get_pages(link,result,rep)
        comparison=g.compare_post(h,result,pages,rep,ref)
        result.update(global_id=gid,label=case['label'],pages=pages,comparison=comparison,status='PASS',
            measurement_source='FULL_NEW_UART')
        posts.append(result);emit(dict(type='POST',**result));new['POST']+=1
    final=g.parse_info(h,link.ask('CWRUTOTAL_INFO','CWRUTOTAL_INFO,'),cat)
    require(final['last_run_id']==last and final['result_valid'],'Publicación final incompatible')
    require(link.index==len(trace),'Intercambios UART no consumidos')
    require(records==events,'MEDICIONES diverge de la reconstrucción UART, banco u origen importado')
    coverage=f.coverage(runs,bases,posts,bank)
    expected=dict(RAW_RUN=984,RAW_PROFILE=328,RAW_BASE=54,POST=2736,RAW_transmissions=55)
    require(new==expected,'Conteos nuevos divergentes')
    return dict(firmware_info=info,firmware_info_final=final,events=records,runs=runs,bases=bases,
        posts=posts,selections=selections,new_completed=new,coverage=coverage,
        RAW_timing_statistics=h.statistics(runs,cat['representatives']),UART_pairs=link.index//2)


def validate_report(f,g,report,rebuilt,plan,seal,programming,imports):
    require(report['status']==f.FULL_STATUS and report['mode']=='full','FULL no terminó con PASS')
    require(report['firmware_elf_sha256']==g.ELF_SHA and report['catalog_sha256']==g.CATALOG_SHA
        and report['gate_manifest_sha256']==imports['gate_manifest_sha256']
        and report['gate_audit_sha256']==sha(f.GATE_AUDIT)
        and report['client_manifest_sha256']==seal
        and report['old_joint_client_manifest_sha256']==f.OLD_JOINT_SHA
        and report['programming_evidence']==programming,'Procedencia FULL divergente')
    require(report['imported_counts']==f.IMPORTED_COUNTS
        and report['new_completed']==rebuilt['new_completed']
        and report['total_completed']==dict(RAW_RUN=1008,RAW_PROFILE=336,RAW_BASE=56,POST=2760)
        and report['coverage']==rebuilt['coverage'],'Conteos reportados divergentes')
    for key in ('firmware_info','firmware_info_final','RAW_timing_statistics'):
        require(report[key]==rebuilt[key],'Resumen reconstruido diferente: '+key)
    require(report['RAW_unique_window_ids']==[r['global_id'] for r in rebuilt['bases']]
        and report['RAW_model_ids']==list(range(8)) and report['POST_model_ids']==list(range(8)),
        'IDs resumidos divergentes')
    require(report['full_validation_completed'] and report['POST_exhaustive_reserved_coverage']
        and report['RAW_partial_development_bank'] and report['physical_audit_pending']
        and report['current_gate_timings_imported'] and report['identity_from_programmer_and_UART'],
        'Alcance FULL no corresponde a esta validación escalonada')
    require(not report['RAW_exhaustive_reserved_coverage'] and not report['POST_timing_is_full_pipeline']
        and not report['cached_PC_end_to_end_exactness_claimed'] and not report['models_modified']
        and not report['hardware_flashed'] and not report['selector_enabled']
        and not report['whole_Flash_readback_performed'] and not report['old_representative_timings_imported'],
        'Afirmación de alcance excesiva o modificación no permitida')
    require(report['training_fits']==report['PC_same_input_classifier_invocations_new']
        ==report['PC_same_input_classifier_invocations_imported']==0,'Invocaciones de entrenamiento/oráculo inesperadas')
    require(report['excluded_from_RAW_primary_cost']==['UART','parser','checksums','SELECT','BASE','PROFILE','POST'],
        'Exclusiones de costo cambiadas')


def audit(folder=FOLDER):
    folder=Path(folder).resolve()
    require(folder==FOLDER.resolve(),'Auditoría fijada a la campaña FULL canónica')
    entries=a.strict_manifest(folder,'MANIFEST_EVIDENCIAS_SHA256.json')
    f=load_full();seal=f.own_manifest()
    require(seal==EXPECTED_FULL_MANIFEST,'Sello FULL distinto del entregado')
    args=SimpleNamespace(firmware_elf=OUT/'copia_aislada/Prueba_2_CWRU_Extraccion/Release_180MHz_Canon42_SR/Prueba_2_CWRU_Extraccion.elf',
        build_report=OUT/'COMPILACION_CANON42.json',elf_audit=OUT/'AUDITORIA_ELF_CANON42.json')
    g,h,cat,bank,arrays,provenance,programming,imports,plan=f.load_context(args)
    # Repetir íntegramente la auditoría del gate; sus evidencias importadas no
    # se consideran nuevas mediciones ni se mezclan sus IDs con los del FULL.
    gate_rebuilt=a.audit()
    require(gate_rebuilt==read(f.GATE_AUDIT),'Auditoría gate guardada distinta de su reconstrucción')
    gate_entries=a.strict_manifest(folder/'gate_importado','MANIFEST_EVIDENCIAS_SHA256.json')
    require(gate_entries==imports['entries'] and sha(folder/'gate_importado/MANIFEST_EVIDENCIAS_SHA256.json')
        ==imports['gate_manifest_sha256'],'Gate importado diferente del gate auditado')
    for name in ['RESULTADO_GATE.json','MEDICIONES.jsonl','TRAZA_UART.jsonl']:
        require(sha(folder/'gate_importado'/name)==sha(f.GATE_FOLDER/name),'Copia gate divergente: '+name)
    report=read(folder/'RESULTADO_COMPLETO.json')
    snapshots={'validar_canonicos_full_ejecutado.py':PACKAGE/'validar_canonicos_full.py',
        f.GATE_NAME:PACKAGE/f.GATE_NAME,f.HELPER_NAME:PACKAGE/f.HELPER_NAME,
        'AUDITORIA_GATE_CANON42.json':f.GATE_AUDIT,'firmware_declarado.elf':provenance['firmware'],
        'COMPILACION_CANON42.json':provenance['build'],'AUDITORIA_ELF_CANON42.json':provenance['audit']}
    for name,source in snapshots.items():require(sha(folder/name)==sha(source),'Snapshot divergente: '+name)
    require(report['client_sha256']==sha(folder/'validar_canonicos_full_ejecutado.py'),
        'Hash del cliente ejecutado divergente')
    executed=read(folder/'PLAN_EJECUTADO.json')
    require(all(executed[key]==value for key,value in plan.items()),'Plan ejecutado divergente')
    require(set(executed)==set(plan)|{'max_seconds','client_manifest_sha256','python','gate_audit_sha256'}
        and executed['client_manifest_sha256']==seal and executed['gate_audit_sha256']==sha(f.GATE_AUDIT),
        'Metadatos del plan divergentes')
    marker=read(f.ATTEMPT)
    require(marker==dict(attempt=1,output=str(folder),client_manifest_sha256=seal,
        gate_evidence_manifest_sha256=imports['gate_manifest_sha256'],max_seconds=1800,
        automatic_resume=False,automatic_retry=False,execution_owner='user_CLI'),'Marcador de intento divergente')
    require(0<report['wall_seconds']<=executed['max_seconds']==1800,'Tiempo fuera del presupuesto')
    trace=a.read_jsonl(folder/'TRAZA_UART_NUEVA.jsonl');events=a.read_jsonl(folder/'MEDICIONES.jsonl')
    rebuilt=reconstruct(f,g,h,cat,bank,arrays,imports,plan,trace,events)
    validate_report(f,g,report,rebuilt,plan,seal,programming,imports)
    require(f.own_manifest()==seal and g.authenticate_programming(f.OLD_JOINT_SHA)==programming,
        'Procedencia alterada durante auditoría')
    a.strict_manifest(folder,'MANIFEST_EVIDENCIAS_SHA256.json')
    unique_pairs=[row for row in rebuilt['runs'] if row['mode']==0 and row['repetition']==1]
    imported_count=len(imports['events'])
    return dict(status=STATUS,date_local='2026-10-28',timezone='America/Lima',full_folder=str(folder),
        full_evidence_manifest_sha256=sha(folder/'MANIFEST_EVIDENCIAS_SHA256.json'),
        full_result_sha256=sha(folder/'RESULTADO_COMPLETO.json'),verified_events_sha256=canonical_sha(events),
        client_manifest_sha256=seal,old_joint_manifest_sha256=f.OLD_JOINT_SHA,
        gate_manifest_sha256=imports['gate_manifest_sha256'],gate_audit_sha256=sha(f.GATE_AUDIT),
        gate_verified_events_sha256=gate_rebuilt['verified_events_sha256'],
        firmware_elf_sha256=g.ELF_SHA,catalog_sha256=g.CATALOG_SHA,bank_manifest_sha256=g.BANK_MANIFEST_SHA,
        programming_evidence=programming,imported_counts=f.IMPORTED_COUNTS,new_completed=rebuilt['new_completed'],
        coverage=rebuilt['coverage'],new_UART_pairs=rebuilt['UART_pairs'],imported_UART_pairs=gate_rebuilt['UART_pairs'],
        event_count=len(events),imported_event_count=imported_count,new_event_count=len(events)-imported_count,
        UTC_monotonic=True,firmware_info=rebuilt['firmware_info'],firmware_info_final=rebuilt['firmware_info_final'],
        RAW_payloads_exact_bank=True,all_same_input_INT8_outputs_exact_saved=True,
        all_intermediates_bit_exact=True,all_selected_features_bit_exact_BASE=True,
        all_POST_saved_external_predictions_exact=True,PROFILE_pages_equal_RUN=True,
        arena_used_by_model={str(rid):sorted({row['arena_used'] for row in rebuilt['selections'] if row['rep_id']==rid})
            for rid in range(8)},RAW_timing_statistics=rebuilt['RAW_timing_statistics'],
        RAW_PC_cache_comparison=[dict(rep_id=row['rep_id'],global_id=row['global_id'],
            **row['comparison']['cached_PC_differences']) for row in unique_pairs],
        RAW_max_abs_feature_error_PC_cache=max(row['max_abs_error_PC_cache'] for row in rebuilt['bases']),
        wall_seconds=report['wall_seconds'],raw_heldout_extraction_exhaustive=False,
        post_reserved_features_exhaustive=True,training_fits=0,classifier_invocations=0,
        serial_port_opened=False,hardware_flashed=False,evidence_files=len(entries),audit_script_sha256=sha(__file__),
        limitations=['RAW is the partial development bank: 56 unique windows, 42 per model; heldout RAW is not exhaustive.',
            'POST covers all 1380 heldout feature vectors in two models per load; it does not execute extraction.',
            'Exactness of INT8 output uses identical inputs; RAW PC cached features can differ numerically.',
            'HAL clock is declared, not externally measured; no WCET, autonomous acquisition or runtime stack maximum established.',
            'Gate and FULL have separate UART counter origins; reused canonical gate measurements retain their provenance.'])


def main():
    import json
    require(not REPORT.exists(),'No sobrescribir auditoría FULL existente')
    result=audit()
    with REPORT.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({key:result[key] for key in ['status','coverage','new_UART_pairs','event_count','wall_seconds']},ensure_ascii=True))


if __name__=='__main__':main()
