"""Audita CWRUND guardado: sin COM, firmware o nuevas inferencias.

Solo --write-report crea un informe nuevo; nunca sobrescribe evidencia previa.
"""
import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from auditar_validacion_o3 import RecordedTransport, sha

ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'despliegue_stm32/detector_normalidad_180mhz_2026-10-23'
CLIENT=OUT/'Validacion_PC'


def require(condition,message):
    if not condition:raise RuntimeError(message)


def timings(rows,hclk):
    cycles=np.asarray([r['observed']['cycles'] for r in rows],dtype=np.float64)
    us=cycles*1e6/hclk
    return dict(executions=len(rows),mean_cycles=float(cycles.mean()),min_cycles=int(cycles.min()),max_cycles=int(cycles.max()),
        mean_us=float(us.mean()),median_us=float(np.median(us)),p95_us=float(np.percentile(us,95,method='linear')),
        min_us=float(us.min()),max_us=float(us.max()),sample_std_us=float(np.std(us,ddof=1)) if len(us)>1 else None)


def audit(run_name):
    require(Path(run_name).name==run_name and run_name not in ('.','..'),'Nombre de ejecución inválido')
    folder=CLIENT/'resultados'/run_name
    manifest=json.loads((folder/'MANIFEST_SHA256.json').read_text(encoding='utf-8'))
    require(set(manifest)=={'resultados.json','uart_trace.jsonl'},'Manifiesto físico inesperado')
    for name,digest in manifest.items():require(sha(folder/name)==digest,'Archivo físico modificado: '+name)
    run=json.loads((folder/'resultados.json').read_text(encoding='utf-8'))
    require(run['status']=='PASS' and run['mode'] in ('single','all'),'Ejecución no aprobada o modo desconocido')
    require(not any(run[k] for k in ('extraction_executed','dnn_executed','training_executed','threshold_refitted')),'Alcance diferente del detector aislado')
    for key,name in (('client_sha256','validar_detector.py'),('reference_sha256','REFERENCIA_DETECTOR.json')):
        require(run[key]==sha(CLIENT/name),'Recurso diferente del ensayo: '+name)
    seal_digest,seal_name=(OUT/'MANIFEST_SHA256.sha256').read_text(encoding='ascii').strip().split()
    require(seal_name=='MANIFEST_SHA256.json' and sha(OUT/seal_name)==seal_digest,'Sello de preparación distinto')
    sealed=json.loads((OUT/seal_name).read_text(encoding='utf-8'))
    for path in (CLIENT/'validar_detector.py',CLIENT/'contrato_detector_cwru.py',CLIENT/'REFERENCIA_DETECTOR.json',OUT/'CONTRATO_PREVIO_DETECTOR.md'):
        require(sha(path)==sealed[path.relative_to(ROOT).as_posix()],'Recurso difiere de preparación: '+path.name)
    sys.path.insert(0,str(CLIENT))
    spec=importlib.util.spec_from_file_location('frozen_detector_client',CLIENT/'validar_detector.py')
    client=importlib.util.module_from_spec(spec);spec.loader.exec_module(client)
    ref=client.bundle();index=None
    if run['mode']=='all':
        cases=ref['cases'];require(len(cases)==165,'Catálogo completo distinto')
    else:
        require(bool(run['results']),'Ejecución vacía')
        case=next((c for c in ref['cases'] if c['name']==run['results'][0]['name']),None)
        require(case is not None and case['group']=='development_entm','Caso individual inesperado')
        cases=[case];index=case['index']
    require(run['cases']==len(cases) and run['repetitions']==3,'Selección o repeticiones diferentes')
    total=len(cases)*3
    require(len(run['results'])==run['completed_executions']==total,'Resultados incompletos')
    records=[json.loads(line) for line in (folder/'uart_trace.jsonl').read_text(encoding='utf-8').splitlines()]
    transport=RecordedTransport(records)
    info=client.info(transport,ref,1.,[])
    require(info==run['firmware_info'],'INFO inicial diferente de la trama')
    ordinal=0;standardized=0;squared=0
    historical={r['name']:r for r in ref['historical_reduction_comparison']}
    historical_checks=[]
    for case in cases:
        expected=client.evaluate(case['mode'],case['input_bits'],ref['params'])
        require(expected==case['expected'],'Referencia no reproduce la aritmética fijada')
        previous=None
        for repetition in range(1,4):
            row=run['results'][ordinal];ordinal+=1
            require(all(row[k]==case[k] for k in ('name','group','mode','input_hex')),'Entrada o modo distintos')
            require(row['repetition']==repetition and row['status']=='PASS' and row['reference']==expected,'Orden, estado o referencia distintos')
            line=client.exchange(transport,client.command(case),'CWRUND_RESULT,',1.,[])
            observed=client.parse(line,case,ref['params']);current=client.numeric(observed)
            require(observed==row['observed'] and current==expected and row['bit_exact'],'Discrepancia de score, decisión o intermedios')
            stable=previous is None or previous==current
            require(stable and row['repeated_output_stable']==stable,'Inestabilidad entre repeticiones')
            standardized+=len(current['z_bits']);squared+=len(current['term_bits'])
            if case['name'] in historical:
                old=historical[case['name']]
                require(old['sequential_bits']==observed['score_bits'],'Score difiere del contraste histórico')
                old_score=client.floats([old['numpy_sum_bits']])[0]
                old_decision=bool(old_score>ref['params']['threshold_value'])
                require(old_decision==observed['rejected'],'Decisión difiere de reducción histórica en desarrollo')
                historical_checks.append(dict(name=case['name'],repetition=repetition,decision_match=True,
                    score_bit_exact=old['numpy_sum_bits']==observed['score_bits']))
            previous=current
    negative_cases=client.negatives(ref) if index is None else []
    require(len(run['negative_tests'])==len(negative_cases),'Rechazos incompletos')
    for case in ref['numeric_negative_cases']:
        try:client.evaluate(case['mode'],case['input_bits'],ref['params'])
        except ValueError as exc:require(str(exc)==case['expected_reason'],'Referencia negativa diferente')
        else:raise RuntimeError('Referencia numérica inválida no rechazada')
    for saved,(command,reason) in zip(run['negative_tests'],negative_cases):
        line=client.exchange(transport,command,'CWRUND_ERROR,',1.,[],allow_error=True)
        f=line.split(',')
        expected_codes={'FORMAT':-6,'CHECKSUM':-7,'UNKNOWN':-8,'NONFINITE_INPUT':-1,
            'NONFINITE_INTERMEDIATE':-2,'NEGATIVE_SCORE':-3}
        require(len(f)==3 and f[1]==reason and int(f[2])==expected_codes[reason],'Rechazo inválido')
        require(saved==dict(command=command,reason=reason,response=line),'Rechazo guardado diferente')
    after=client.info(transport,ref,1.,[])
    require(after==run['firmware_info_after'],'INFO final diferente de la trama')
    require({k:v for k,v in info.items() if k!='fpscr'}=={k:v for k,v in after.items() if k!='fpscr'},'Parámetros o entorno cambiaron')
    require(transport.position==len(records),'Tramas sin asociar a casos')
    boundary={}
    if index is None:
        for name,rejected in (('compare_below',False),('compare_equal',False),('compare_above',True)):
            rows=[r for r in run['results'] if r['name']==name]
            require(len(rows)==3 and all(r['observed']['rejected']==rejected for r in rows),'Comparador estricto incorrecto')
            boundary[name]=dict(score_bits=rows[0]['observed']['score_bits'],rejected=rejected,repetitions=3)
        for name in ('ignored_frequencies_-1','ignored_frequencies_1'):
            a=next(r for r in run['results'] if r['name']=='means')['observed']
            b=next(r for r in run['results'] if r['name']==name)['observed']
            require(client.numeric(a)==client.numeric(b),'Columnas excluidas alteran el resultado')
    groups={}
    for label in sorted({r['group'] for r in run['results']}):
        rows=[r for r in run['results'] if r['group']==label]
        groups[label]=dict(cases=len({r['name'] for r in rows}),exact_executions=len(rows),
            timing_by_mode={mode:timings([r for r in rows if r['mode']==mode],info['hclk_hz']) for mode in sorted({r['mode'] for r in rows})})
    built=json.loads((OUT/'COMPILACION_DETECTOR.json').read_text(encoding='utf-8'))['configurations']['Release_180MHz_Detector']
    local=Path(r'D:\STM32IDE\Prueba_2_CWRU_Extraccion\Release_180MHz_Detector\Prueba_2_CWRU_Extraccion.elf')
    local_hash=sha(local) if local.is_file() else None
    report=dict(status='PHYSICAL_DETECTOR_SUITE_AUDITED' if index is None else 'FIRST_DETECTOR_CASE_AUDITED',
        audited_utc=datetime.now(timezone.utc).isoformat(),run=run_name,mode=run['mode'],development_index=index,
        positive_cases=len(cases),repetitions_per_case=3,passed_executions=total,all_outputs_bit_exact=True,all_repeated_outputs_stable=True,
        standardized_values_bit_exact=standardized,squared_terms_bit_exact=squared,scores_bit_exact=total,decisions_match=total,
        negative_cases_passed=len(negative_cases),threshold_boundary_checks=boundary,groups=groups,
        historical_development_decisions_compared=len(historical_checks),historical_development_decisions_match=len(historical_checks),
        historical_development_score_exact_executions=sum(r['score_bit_exact'] for r in historical_checks),
        firmware_info=info,firmware_info_after=after,raw_records_verified=len(records),recorded_input_output_replay_verified=True,
        results_sha256=sha(folder/'resultados.json'),trace_sha256=sha(folder/'uart_trace.jsonl'),
        run_manifest_sha256=sha(folder/'MANIFEST_SHA256.json'),client_sha256=sha(CLIENT/'validar_detector.py'),
        reference_sha256=sha(CLIENT/'REFERENCIA_DETECTOR.json'),contract_sha256=sha(OUT/'CONTRATO_PREVIO_DETECTOR.md'),
        preparation_manifest_sha256=seal_digest,auditor_sha256=sha(Path(__file__)),
        recorded_transport_source_sha256=sha(Path(__file__).with_name('auditar_validacion_o3.py')),
        audit_python=sys.version,audit_numpy=np.__version__,prepared_elf_sha256=built['elf_sha256'],local_elf_sha256=local_hash,
        local_elf_matches_prepared_build=local_hash==built['elf_sha256'],device_binary_read=False,
        prepared_flash_bytes=built['flash_load_bytes'],prepared_static_ram_with_linker_reservations_bytes=built['ram_static_with_linker_reservations_bytes'],
        serial_port_opened_during_audit=False,hardware_changed_during_audit=False,physical_experiment_repeated=False,
        predictive_metrics_computed=False,full_pipeline_validated=False,
        limitations=[
            'Identity is with the explicitly fixed sequential float32 reference, not all historical NumPy reductions.',
            'Only fixed V1 resources and tested inputs; no independent failure confirmation, IR/OR diagnosis or predictive V2 evaluation.',
            'RUN covers parameter/input validation, scaling, distance, comparison and intermediate writes; excludes parser/checksum/UART, extraction and DNN.',
            'DIST/CHECK are diagnostics; synthetic/repeated inputs are not independent acquisitions or motors.',
            'Nominal HAL HCLK, no instrumentation subtraction; observed maxima are not WCET or integrated real-time proof.',
            'Local ELF identity is not a full device Flash readback. Physical peak stack was not measured.'])
    target=OUT/('RESULTADOS_FISICOS_DETECTOR_COMPLETA.json' if index is None else f'RESULTADOS_FISICOS_DETECTOR_INDICE_{index}.json')
    return report,target


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True);parser.add_argument('--write-report',action='store_true')
    args=parser.parse_args();report,target=audit(args.run)
    if args.write_report:
        with target.open('x',encoding='utf-8',newline='\n') as stream:
            json.dump(report,stream,ensure_ascii=False,indent=2);stream.write('\n')
    print(json.dumps(report,ensure_ascii=True,indent=2))


if __name__=='__main__':main()
