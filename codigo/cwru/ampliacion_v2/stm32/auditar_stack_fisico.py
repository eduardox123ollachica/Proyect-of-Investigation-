"""Audita CWRUSTK existente por reproducción de traza; nunca abre COM.

Sin opciones de escritura solo informa. --write-report crea nuevos informes
y sello físico separado; no modifica cliente, referencias, firmware o sellos.
"""
from __future__ import annotations
import argparse
import contextlib
import importlib.util
import io
import json
import sys
from datetime import datetime,timezone
from pathlib import Path
import preparar_stack_cwru as stage
from auditar_validacion_o3 import RecordedTransport,sha

ROOT,OUT,CLIENT=stage.ROOT,stage.OUT,stage.CLIENT

def require(ok,message):
    if not ok:raise RuntimeError(message)
def bounded(base,name):
    path=(base/name).resolve()
    require(path!=base.resolve() and path.is_relative_to(base.resolve()),'Ruta fuera de ámbito')
    return path
def audit(run_name):
    require(Path(run_name).name==run_name and run_name not in ('.','..'),'Nombre de ejecución inválido')
    folder=bounded(CLIENT/'resultados',run_name)
    with contextlib.redirect_stdout(io.StringIO()):stage.verify()
    manifest=json.loads((folder/'MANIFEST_SHA256.json').read_text(encoding='utf-8'))
    require(set(manifest)=={'resultados.json','uart_trace.jsonl'},'Manifiesto físico diferente')
    for n,d in manifest.items():require(sha(folder/n)==d,'Archivo físico modificado:'+n)
    run=json.loads((folder/'resultados.json').read_text(encoding='utf-8'))
    require(run['status']=='PASS' and not run['simulation_only'] and run['firmware_executed']
        and run['serial_port_opened'],'No es prueba física aprobada')
    require(not any(run[k] for k in ('training_executed','outer_test_loaded','physical_stack_peak_guaranteed',
        'baseline_subtracted','canonical_latency_replaced')),'Alcance distinto del contrato')
    for k,n in (('client_sha256','validar_stack.py'),('reference_sha256','REFERENCIA_PIPELINE.json'),
        ('bundle_manifest_sha256','VALIDACION_MANIFEST.json')):
        require(run[k]==sha(CLIENT/n),'Recurso diferente del usado en placa:'+n)
    spec=importlib.util.spec_from_file_location('sealed_stack_client',CLIENT/'validar_stack.py')
    client=importlib.util.module_from_spec(spec);spec.loader.exec_module(client)
    pc,ref,windows=client.load(CLIENT)
    require(run['mode'] in ('single','all') and run['patterns']==list(client.PATTERNS)
        and run['repetitions_per_pattern']==3,'Modo/patrones/repeticiones diferentes')
    require(bool(run['results']),'Suite vacía')
    all_cases=run['mode']=='all';index=None if all_cases else run['results'][0]['index']
    require(index is None or isinstance(index,int) and 0<=index<42,'Índice inválido')
    selected=ref['cases'] if all_cases else [ref['cases'][index]]
    records=[json.loads(line) for line in (folder/'uart_trace.jsonl').read_text(encoding='utf-8').splitlines()]
    require(len(records)%2==0,'Número de registros impar')
    for i,row in enumerate(records):
        direction='RX' if i%2 else 'TX';require(row['direction']==direction,'Orden de traza inválido')
        text=row['command'] if direction=='TX' else row['line'];length=len((text+'\r\n').encode('ascii'))
        require(length==row['bytes'] and 0<length<(1024 if direction=='TX' else 512),'Longitud UART inválida')
    replay=dict(results=[],calibrations=[],negative_tests=[]);transport=RecordedTransport(records)
    with contextlib.redirect_stdout(io.StringIO()):
        client.session(pc,transport,ref,windows,selected,replay,[],all_cases=all_cases)
    require(transport.position==len(records),'Tramas no asociadas a la suite')
    for k,v in replay.items():require(run[k]==v,'JSON diferente de las respuestas registradas:'+k)
    require(len(run['results'])==run['completed_runs']==6*len(selected),'Número de RUN distinto')
    require(len(run['calibrations'])==4 and len(run['negative_tests'])==(9 if all_cases else 0),'Catálogo incompleto')
    built=json.loads((OUT/'COMPILACION_STACK.json').read_text(encoding='utf-8'))['configurations'][stage.CONFIG]
    local=stage.SOURCE/stage.CONFIG/(stage.NAME+'.elf')
    require(local.is_file() and sha(local)==built['elf_sha256'],'ELF local distinto o ausente')
    rows=run['results'];values=[r['stack']['total_observed_bytes'] for r in rows]
    per_pattern={p:dict(runs=sum(r['pattern']==p for r in rows),
        observed_written_min_bytes=min(r['stack']['total_observed_bytes'] for r in rows if r['pattern']==p),
        observed_written_max_bytes=max(r['stack']['total_observed_bytes'] for r in rows if r['pattern']==p)) for p in client.PATTERNS}
    maximum=max(values);calibrations=[dict(mode=c['mode'],pattern=c['pattern'],
        prefix_bytes=c['observed']['prefix_bytes'],additional_written_bytes=c['observed']['additional_written_bytes'],
        total_observed_bytes=c['observed']['total_observed_bytes']) for c in run['calibrations']]
    report=dict(status='PHYSICAL_STACK_SUITE_AUDITED' if all_cases else 'FIRST_STACK_CASE_AUDITED',
        audited_utc=datetime.now(timezone.utc).isoformat(),run=run_name,mode=run['mode'],development_index=index,
        windows=len(selected),mat_files=len({c['metadata']['file_id'] for c in selected}),
        loads_hp=sorted({int(c['metadata']['load_hp']) for c in selected}),patterns=list(client.PATTERNS),
        repetitions_per_pattern=3,passed_main_runs=len(rows),calibrations=calibrations,
        negative_tests_passed=len(run['negative_tests']),all_intermediates_and_outputs_exact=True,
        original_float64_comparisons_passed=14*len(rows),dominant_bins_exact=2*len(rows),
        guards_intact=True,msp_restored=True,masks_unchanged=True,rne_ftz_preserved=True,
        fpccr_values_observed=sorted({r['stack']['fpccr_before'] for r in rows}|{r['stack']['fpccr_after'] for r in rows}),
        fpscr_values_observed=sorted({r['stack']['fpscr_before'] for r in rows}|{r['stack']['fpscr_after'] for r in rows}),
        fpscr_all_flags_identical=False if len({r['stack']['fpscr_before'] for r in rows}|{r['stack']['fpscr_after'] for r in rows})>1 else True,
        prefix_bytes_observed=sorted({r['stack']['prefix_bytes'] for r in rows}),
        additional_written_bytes_observed=sorted({r['stack']['additional_written_bytes'] for r in rows}),
        observed_written_min_bytes=min(values),observed_written_max_bytes=maximum,per_pattern=per_pattern,
        reserved_stack_bytes=4096,guard_bytes=128,unmodified_bytes_below_deepest_observed_write=4096-maximum,
        unmodified_bytes_are_not_guaranteed_free_stack=True,
        tick_delta_range=[min(r['stack']['tick_delta'] for r in rows),max(r['stack']['tick_delta'] for r in rows)],
        paint_envelope_max_cycles=max(r['stack']['paint_masked_cycles'] for r in rows),
        scan_envelope_max_cycles=max(r['stack']['scan_masked_cycles'] for r in rows),
        paint_envelope_max_us=max(r['stack']['paint_masked_cycles'] for r in rows)/180,
        scan_envelope_max_us=max(r['stack']['scan_masked_cycles'] for r in rows)/180,
        initialization=run['initialization'],reuse=run['reuse'],raw_records_verified=len(records),
        recorded_input_output_replay_verified=True,prepared_elf_sha256=built['elf_sha256'],local_elf_matches_prepared=True,
        results_sha256=sha(folder/'resultados.json'),trace_sha256=sha(folder/'uart_trace.jsonl'),
        run_manifest_sha256=sha(folder/'MANIFEST_SHA256.json'),preparation_manifest_sha256=sha(OUT/'MANIFEST_SHA256.json'),
        client_sha256=sha(CLIENT/'validar_stack.py'),reference_sha256=sha(CLIENT/'REFERENCIA_PIPELINE.json'),
        auditor_sha256=sha(Path(__file__)),recorded_transport_sha256=sha(Path(__file__).with_name('auditar_validacion_o3.py')),
        audit_python=sys.version,serial_port_opened_during_audit=False,physical_experiment_repeated=False,
        firmware_modified_during_audit=False,predictive_metrics_computed=False,physical_minimum_sp_exact=False,
        canonical_latency_replaced=False,
        limitations=['Observed written watermark and sampled prefix, not continuous SP measurement or an absolute bound.',
            'Unused reserved slots, pattern coincidences and lazy FPU stacking can underestimate SP descent.',
            'Guard is a posterior write detector, not an MPU and not universal overflow proof.',
            'Prefix includes this diagnostic calling path; empty calibration is not subtracted.',
            'FPSCR status flags may change; only RNE/FTZ and required control bits are constrained.',
            'Diagnostic cycles do not replace normal integrated latency or establish WCET.',
            'One window is not the 42-window stack suite.' if not all_cases else '42 fixed development windows are not independent motors or predictive V2 evaluation.',
            'Local ELF identity is not full device Flash readback.'])
    return report,('COMPLETA' if all_cases else f'INDICE_{index}'),folder

def write(report,label,folder):
    target=OUT/f'RESULTADOS_FISICOS_STACK_{label}.json';human=OUT/f'RESULTADOS_VALIDACION_STACK_{label}.md'
    manifest=OUT/f'MANIFEST_RESULTADOS_FISICOS_STACK_{label}_SHA256.json';seal=manifest.with_suffix('.sha256')
    require(not any(p.exists() for p in (target,human,manifest,seal)),'No sobrescribir informes físicos')
    with target.open('x',encoding='utf-8',newline='\n') as stream:json.dump(report,stream,ensure_ascii=False,indent=2);stream.write('\n')
    lines=[f'# Auditoría física de stack: {label}','',f"Ejecución `{report['run']}`.",'',
        f"Aprobados {report['passed_main_runs']} RUN, cuatro calibraciones y {report['negative_tests_passed']} rechazos.",
        f"Verificados {report['raw_records_verified']} registros TX/RX mediante reproducción de respuestas guardadas.",'',
        f"Huella escrita observada: mínimo {report['observed_written_min_bytes']} B; máximo {report['observed_written_max_bytes']} B.",
        f"Prefijos observados: {report['prefix_bytes_observed']}; escritos adicionales: {report['additional_written_bytes_observed']} B.",
        'Guarda inferior128B intacta; MSP restaurado; máscaras esperadas, RNE y FTZ preservados.',
        'Todos los intermedios y salidas coinciden exactamente con la referencia integrada.','',
        '| Calibración | Patrón | Prefijo B | Escrito adicional B | Total observado B |','|---|---|---:|---:|---:|']
    for c in report['calibrations']:lines.append(f"| {c['mode']} | {c['pattern']} | {c['prefix_bytes']} | {c['additional_written_bytes']} | {c['total_observed_bytes']} |")
    lines.extend(['','No se resta EMPTY: su camino de llamada tiene un prefijo distinto.',
        f"Envolventes máximas de pintado/escaneo: {report['paint_envelope_max_us']:.6f}/{report['scan_envelope_max_us']:.6f} us nominales.",
        f"Avance de tick alrededor del callback: {report['tick_delta_range']}; corroboración, no prueba aislada de IRQ continuas.",
        'FPSCR cambia algunas banderas de estado; no se afirma identidad completa del registro.','',
        f"Quedaron {report['unmodified_bytes_below_deepest_observed_write']} B por debajo de la escritura más profunda sin modificar.",
        'No son stack libre garantizado ni autorizan reducir la reserva4096B.','',
        '## Límites y siguiente paso',''])
    lines.extend('- '+s for s in report['limitations'])
    lines.extend(['','No se abrió COM, modificó firmware o repitió el ensayo durante esta auditoría.',
        'Revisar este informe antes de la suite completa; no sustituye el perfil de operaciones compartidas o V2.',''])
    with human.open('x',encoding='utf-8',newline='\n') as stream:stream.write('\n'.join(lines))
    paths=[Path(__file__),Path(__file__).with_name('auditar_validacion_o3.py'),target,human,
        OUT/'MANIFEST_SHA256.json',OUT/'MANIFEST_SHA256.sha256',CLIENT/'validar_stack.py',CLIENT/'validar_pipeline.py',
        CLIENT/'REFERENCIA_PIPELINE.json',CLIENT/'VALIDACION_MANIFEST.json',folder/'resultados.json',folder/'uart_trace.jsonl',folder/'MANIFEST_SHA256.json']
    entries={p.relative_to(ROOT).as_posix():sha(p) for p in paths}
    with manifest.open('x',encoding='utf-8',newline='\n') as stream:json.dump(entries,stream,indent=2);stream.write('\n')
    with seal.open('x',encoding='ascii') as stream:stream.write(sha(manifest)+'  '+manifest.name+'\n')
    require(all(sha(bounded(ROOT,n))==d for n,d in entries.items()),'Sello físico diferente al cerrar')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);p.add_argument('--write-report',action='store_true')
    args=p.parse_args();report,label,folder=audit(args.run)
    if args.write_report:write(report,label,folder)
    print(json.dumps(report,ensure_ascii=True,indent=2))
