"""Cierra informes completos CWRUSTK nuevos usando el auditor ya sellado.

No altera ese auditor ni el sello del índice0. Esta redacción sustituye la
frase genérica de su escritor ('antes de suite completa') por un cierre real.
Sin COM, nueva ejecución física, cambios de firmware o entrenamiento.
"""
from __future__ import annotations
import argparse
import json
from collections import Counter
from pathlib import Path
import auditar_stack_fisico as auditor

def close(run_name):
    report,label,folder=auditor.audit(run_name)
    auditor.require(label=='COMPLETA' and report['status']=='PHYSICAL_STACK_SUITE_AUDITED','Solo cierre de suite completa aprobada')
    raw=json.loads((folder/'resultados.json').read_text(encoding='utf-8'))
    ref=json.loads((auditor.CLIENT/'REFERENCIA_PIPELINE.json').read_text(encoding='utf-8'))
    distribution=Counter(r['stack']['total_observed_bytes'] for r in raw['results'])
    report['written_footprint_distribution']={str(k):distribution[k] for k in sorted(distribution)}
    report['deepest_observed_cases']=[dict(index=r['index'],name=r['name'],pattern=r['pattern'],repetition=r['repetition'],
        file_id=ref['cases'][r['index']]['metadata']['file_id'],
        load_hp=ref['cases'][r['index']]['metadata']['load_hp'],
        mechanical_class=ref['cases'][r['index']]['metadata']['class'],
        run_id=r['observed']['id'],lowest_modified=r['stack']['lowest_modified'])
        for r in raw['results'] if r['stack']['total_observed_bytes']==report['observed_written_max_bytes']]
    report['report_writer_sha256']=auditor.sha(Path(__file__))
    report['variation_cause_isolated']=False
    target=auditor.OUT/'RESULTADOS_FISICOS_STACK_COMPLETA.json'
    human=auditor.OUT/'RESULTADOS_VALIDACION_STACK_COMPLETA.md'
    manifest=auditor.OUT/'MANIFEST_RESULTADOS_FISICOS_STACK_COMPLETA_SHA256.json'
    seal=manifest.with_suffix('.sha256')
    auditor.require(not any(p.exists() for p in (target,human,manifest,seal)),'No sobrescribir informes previos')
    with target.open('x',encoding='utf-8',newline='\n') as stream:json.dump(report,stream,ensure_ascii=False,indent=2);stream.write('\n')
    lines=['# Stack: validación física completa','',f'Ejecución: `{run_name}`.','',
        f"42 ventanas de21MAT, cargas0/1/2HP, dos patrones y tres RUN por patrón: {report['passed_main_runs']}/252 aprobados.",
        f"Cuatro calibraciones y {report['negative_tests_passed']}/9 rechazos correctos; {report['raw_records_verified']} registros TX/RX auditados.",
        'Todos los intermedios y salidas coinciden con la referencia integrada; una extracción nueva por RUN.','',
        '## Huella escrita observada','',
        '| Total observado | Ejecuciones |','|---|---:|']
    lines.extend(f'| {k} B | {distribution[k]} |' for k in sorted(distribution))
    lines.extend(['',f"Prefijo observado: {report['prefix_bytes_observed']} B. Escritura adicional: {report['additional_written_bytes_observed']} B.",
        'Ambos patrones registraron un máximo3072B; todas las guardas128B quedaron intactas y MSP se restauró.',
        'El máximo anterior2968B pertenecía solo a la primera ventana; el cierre conserva el mayor de la suite, sin descartar casos.',
        'Los1024B sin escribir bajo la dirección más profunda NO son stack libre garantizado; no se reduce la reserva4096B.',
        'La diferencia104B entre observaciones se registra, pero su causa no quedó aislada por este ensayo.',
        'No atribuirla por sí sola a una función o interrupción concreta.','',
        '| Caso máximo | MAT | Clase | Carga HP | Patrón | Repetición |','|---|---|---|---:|---|---:|'])
    lines.extend(f"| {r['name']} | {r['file_id']} | {r['mechanical_class']} | {r['load_hp']} | {r['pattern']} | {r['repetition']} |"
        for r in report['deepest_observed_cases'])
    lines.extend(['','## Calibración y contexto','',
        'EMPTY:2176B de prefijo,0B adicionales, en ambos patrones. PROBE:2176+128=2304B, en ambos.',
        'No se resta EMPTY porque utiliza otro camino de llamada. El criterio previo era detectar al menos128B.',
        f"Envolventes máximas pintado/escaneo: {report['paint_envelope_max_us']:.6f}/{report['scan_envelope_max_us']:.6f} us nominales (<1ms previo).",
        'El tick avanzó84–85 alrededor del callback. PRIMASK/BASEPRI/FAULTMASK/IPSR y los bits requeridos se conservaron.',
        'FPCCR permanecióc0000018; variaron banderas FPSCR, no RNE/FTZ. No se afirma igualdad completa de FPSCR.',
        'INIT reutilizó la instancia: cero ciclos no es una nueva medición de arranque.',
        'Arena1188B usados/16384B reservados; ELF local idéntico al preparado, sin lectura completa de Flash.','',
        '## Dictamen y continuidad','',
        'El diagnóstico queda aprobado para las42ventanas y los patrones/condiciones ensayados.',
        'No se declara un máximo absoluto de SP: espacios reservados no escritos, coincidencias y lazy stacking limitan el watermark.',
        'Tampoco es una medición de arranque, todas las rutas de firmware o motores independientes.',
        'La guarda es detección posterior, no protección MPU. No se reduce la reserva de memoria.',
        'Los tiempos diagnósticos no sustituyen el total normal84.265429ms ni constituyen WCET.',
        'Siguiente bloque: perfil de operaciones compartidas para el costo de características; después, cierre del protocolo V2.',
        'No se inicia entrenamiento masivo, cambia el manuscrito o repite la suite normal por este cierre.','',
        'La auditoría solo reprodujo tramas guardadas: no abrió COM, repitió hardware o modificó firmware/dataset/modelos.',
        'El auditor sellado del índice0 no se modificó. Este escritor nuevo permite un cierre redactado para la suite completa.',
        'Se mantienen intactos sellos de preparación y primera ventana.',''])
    with human.open('x',encoding='utf-8',newline='\n') as stream:stream.write('\n'.join(lines))
    paths=[Path(__file__),Path(auditor.__file__),Path(auditor.__file__).with_name('auditar_validacion_o3.py'),target,human,
        auditor.OUT/'MANIFEST_SHA256.json',auditor.OUT/'MANIFEST_SHA256.sha256',auditor.CLIENT/'validar_stack.py',
        auditor.CLIENT/'validar_pipeline.py',auditor.CLIENT/'REFERENCIA_PIPELINE.json',auditor.CLIENT/'VALIDACION_MANIFEST.json',
        folder/'resultados.json',folder/'uart_trace.jsonl',folder/'MANIFEST_SHA256.json']
    entries={p.relative_to(auditor.ROOT).as_posix():auditor.sha(p) for p in paths}
    with manifest.open('x',encoding='utf-8',newline='\n') as stream:json.dump(entries,stream,indent=2);stream.write('\n')
    with seal.open('x',encoding='ascii') as stream:stream.write(auditor.sha(manifest)+'  '+manifest.name+'\n')
    auditor.require(all(auditor.sha(auditor.bounded(auditor.ROOT,n))==d for n,d in entries.items()),'Sello físico diferente')
    print(json.dumps(dict(status=report['status'],passed_runs=report['passed_main_runs'],maximum_written_bytes=report['observed_written_max_bytes'],
        distribution=report['written_footprint_distribution'],files_in_physical_seal=len(entries),serial_port_opened=False),ensure_ascii=True))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',required=True);close(parser.parse_args().run)
