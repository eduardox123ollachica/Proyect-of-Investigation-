"""Continuación personal de todos los ajustes pendientes en lotes originales.

--prepare audita lo ya ejecutado; --run es exclusivamente para el usuario.
No cambia el ejecutor sellado ni sus límites, modelos, roles o presupuesto.
"""
from __future__ import annotations

import argparse
import datetime
import importlib.abc
import json
from pathlib import Path
import sys
import traceback

ROOT=Path(__file__).resolve().parents[4]
SOURCE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'codigo/cwru/ampliacion_v2/busqueda_interna'))


class NoParentMLRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in ('tensorflow','keras','ai_edge_litert'):
            raise RuntimeError('El controlador no importa el runtime ML: solo el worker original puede hacerlo')
        return None


sys.meta_path.insert(0,NoParentMLRuntime())
import ejecutar_busqueda_interna as e

CAMPAIGN=e.OUTPUT/'20261027T005636_578225Z'
PREP=ROOT/'articulo/ampliacion_cwru/metodologia/continuacion_restantes_interna_v2_2026-10-26'
RECEIPTS=ROOT/'resultados/cwru/ampliacion_v2/control_continuacion_interna_v2_2026-10-26'
MANIFEST=PREP/'MANIFEST_CONTINUACION_SHA256.json'


def drive(campaign,inspect_fn,execute_fn,emit_fn):
    """Lazo acotado; sus callbacks permiten probarlo sin lanzar entrenamientos."""
    plan,jobs,done,consumed=inspect_fn(campaign)
    limits=dict(plan['limits'])
    e.require(limits == dict(worker_seconds=300,campaign_worker_seconds=7200,
        max_batch_seconds=1800,max_new_jobs=16,maximum_new_fits_total=240,
        default_first_batch_seconds=600), 'Límites distintos de los congelados')
    job_ids=[j['job_id'] for j in jobs]
    e.require(len(job_ids) == len(set(job_ids)) == 240 and len(plan['imported']) == 48
        and done == job_ids[:len(done)] and consumed >= 0, 'Inventario de continuación inválido')
    initial_remaining=len(jobs)-len(done)
    batches=0
    def summary(status,**extra):
        return dict(status=status,completed_new_fits=len(done),remaining_new_fits=len(jobs)-len(done),
            available_logical_cells=48+len(done),batches_launched=batches,
            consumed_worker_seconds=consumed,**extra)
    # Cada lote debe avanzar al menos una celda. No hay reintentos ni lazo infinito.
    for _ in range(initial_remaining+1):
        if len(done) == len(jobs):
            return summary('CAMPAIGN_COMPLETED')
        if limits['campaign_worker_seconds']-consumed < limits['worker_seconds']:
            return summary('CAMPAIGN_BUDGET_BOUNDARY')
        requested=min(limits['max_new_jobs'],len(jobs)-len(done))
        previous=list(done);previous_consumed=consumed
        emit_fn(dict(event='NEXT_BATCH',new_jobs=requested,completed_new_fits=len(done),
            remaining_new_fits=len(jobs)-len(done),budget_remaining_seconds=7200-consumed))
        batches+=1
        code=execute_fn(campaign,max_new_jobs=requested,max_seconds=limits['max_batch_seconds'])
        if code != 0:
            return summary('STOPPED_FOR_AUDIT',child_return_code=code,
                counts_are_last_verified_before_failed_batch=True)
        next_plan,next_jobs,next_done,next_consumed=inspect_fn(campaign)
        e.require(next_plan['limits'] == limits and next_plan['imported'] == plan['imported']
            and next_jobs == jobs and next_done == job_ids[:len(next_done)]
            and next_done[:len(previous)] == previous and next_consumed >= previous_consumed,
            'Cambió el contrato, el orden o el contador durante la continuación')
        e.require(len(next_done) <= len(previous)+requested,'Avance fuera del lote autorizado')
        done,consumed=next_done,next_consumed
        emit_fn(dict(event='BATCH_VERIFIED',**summary('BATCH_VERIFIED')))
        if len(done) == len(previous):
            return summary('CAMPAIGN_BUDGET_BOUNDARY' if 7200-consumed < 300 else 'NO_PROGRESS')
    raise ValueError('Número de lotes fuera de la cota de progreso')


def sources():
    return {p.relative_to(ROOT).as_posix():e.sha(p) for p in sorted(SOURCE.glob('*.py'))}


def closed_snapshot(campaign):
    """Contrastar sellos/cierres; execute() valida numéricamente cada lote.

    Evita repetir toda la reconstrucción numérica en el controlador externo.
    El ejecutor original la hace al reanudar y para cada ajuste nuevo.
    """
    plan=e.validate_campaign(campaign,cheap=True)
    jobs=[j for j in plan['jobs'] if j['job_id'] not in plan['imported']]
    completed=0;consumed=0.
    for batch in sorted((campaign/'lotes').iterdir()):
        e.verify_manifest(batch/'MANIFEST_LOTE_SHA256.json')
        result=e.read(batch/'RESUMEN_LOTE.json')
        batch_plan=e.read(batch/'PLAN_LOTE.json')
        e.require(result['status'] in ('BATCH_COMPLETED','BATCH_BUDGET_BOUNDARY','CAMPAIGN_COMPLETED')
            and result['jobs_started'] == result['jobs_passed'], 'Lote no apto para continuar')
        passed=result['jobs_passed']
        e.require(batch_plan['completed_before'] == [j['job_id'] for j in jobs[:completed]]
            and batch_plan['job_ids'][:passed] == [j['job_id'] for j in jobs[completed:completed+passed]],
            'Lote fuera de la secuencia prevista')
        completed+=passed;consumed+=result['worker_elapsed_seconds']
        e.require(result['completed_new_fits'] == completed
            and result['available_logical_cells'] == 48+completed
            and abs(result['campaign_worker_seconds_consumed']-consumed) < 1e-7,
            'Cierre o presupuesto acumulado inconsistente')
    done=[j['job_id'] for j in jobs[:completed]]
    present={j['job_id'] for j in jobs if (campaign/j['job_id']).exists()}
    e.require(present == set(done),'Ajustes no cerrados o fuera de orden')
    return plan,jobs,done,consumed


def prepare():
    plan=e.preflight(check_environment=True)
    e.verify_manifest(e.PREP/'MANIFEST_PREPARACION_COMPLETA_SHA256.json')
    _,jobs,done,consumed=e.inspect_campaign(CAMPAIGN)
    e.require(len(done) == 17 and not (CAMPAIGN/'EJECUCION_EN_CURSO.lock').exists(),
        'La auditoría de esta entrega requiere 17 ajustes nuevos cerrados')
    cached=closed_snapshot(CAMPAIGN)
    e.require(cached[2:] == (done,consumed),'Las dos comprobaciones del cierre discrepan')
    tests=e.read(PREP/'PRUEBAS_CONTROLADOR.json')
    e.require(tests['status'] == 'PASS' and tests['failures'] == tests['errors'] == 0
        and tests['source_sha256'] == sources() and tests['real_training_fits'] == tests['real_TFLite_invocations'] == 0
        and tests['TensorFlow_imported'] is False,'Faltan pruebas finales del controlador')
    batches=sorted((CAMPAIGN/'lotes').iterdir())
    latest=e.read(batches[-1]/'RESUMEN_LOTE.json')
    e.require(latest['jobs_passed'] == 16 and latest['status'] == 'BATCH_COMPLETED', 'Lote16 incompleto')
    states=[e.read(p) for p in sorted(batches[-1].glob('ESTADO_*.json'))]
    times=[s['worker_elapsed_seconds'] for s in states]
    results=[e.read(CAMPAIGN/j['job_id']/'resultado.json') for j in jobs[:len(done)]]
    mean=sum(times)/len(times);remaining=len(jobs)-len(done)
    audit=dict(status='INNER_BATCH16_AUDITED_READY_FOR_ALL_REMAINING',
        date_local='2026-10-26',timezone='America/Lima',campaign=CAMPAIGN.relative_to(ROOT).as_posix(),
        available_logical_cells=65,reused_cells=48,completed_new_fits=17,new_fits_remaining=remaining,
        latest_batch_passed=16,latest_batch_seconds=latest['worker_elapsed_seconds'],
        latest_process_mean_seconds=mean,latest_process_min_seconds=min(times),latest_process_max_seconds=max(times),
        budget_consumed_seconds=consumed,budget_remaining_seconds=7200-consumed,
        remaining_at_last_mean_minutes=remaining*mean/60,
        remaining_at_last_max_minutes=remaining*max(times)/60,
        initial_cold_import_was_not_repeated_in_batch16=True,
        all_job_metrics_recomputed_from_saved_predictions=True,
        all_checkpoints_scalers_scopes_models_verified=True,
        outer_evaluation_executed=False,new_training_fits=0,new_TFLite_invocations=0,
        TensorFlow_imported='tensorflow' in sys.modules,hardware_connected=False,
        results_summary=[dict(job_id=r['job_id'],best_epoch=r['best_epoch'],epochs_executed=r['epochs_executed'],
            fp32_macro_f1=r['metrics']['FP32']['macro_f1'],int8_macro_f1=r['metrics']['INT8']['macro_f1'],
            class_changes=r['FP32_INT8_class_changes'],validation_clipped_values=r['int8_details']['input_clipped_values'])
            for r in results],
        qualifications=['Internal validation cells; no outer results or final feature selection.',
            'FP32/INT8 differences and validation clipping are recorded; do not recalibrate with validation.',
            'Timing projection excludes repeated shared audits and is not a guarantee.'])
    e.write(PREP/'AUDITORIA_LOTE16.json',audit)
    continuation=dict(status='ALL_REMAINING_INNER_DNN_CONTINUATION_PREPARED',
        campaign=CAMPAIGN.relative_to(ROOT).as_posix(),minimum_closed_new_fits=17,
        initial_pending_new_fits=223,expected_batches_if_16_each=14,
        limits=plan['limits'],prior_preparation_manifest_sha256=e.sha(e.PREP_MANIFEST),
        source_sha256=sources(),new_training_fits_during_preparation=0,
        original_executor_modified=False,automatic_retry=False,user_starts_training=True,
        objective='Continue the same campaign until 288/288 inner DNN cells or frozen stop condition')
    e.write(PREP/'PLAN_CONTINUACION.json',continuation)
    command=f'& "C:\\Users\\LENOVO\\AppData\\Local\\Microsoft\\WindowsApps\\python3.12.exe" -B "{Path(__file__).resolve()}" --run'
    text=f'''# Continuación de todos los ajustes DNN internos pendientes

Auditoría:17 nuevos PASS,48 reutilizados,65/288 celdas. Quedan223 ajustes.
El último lote16 tomó175.265s; media10.954s por ajuste. Proyección de tiempos
contabilizados restantes:41–44min con este lote, sin garantía. Reservar
aproximadamente60–90min de reloj para incluir comprobaciones progresivas;
otras cargas del equipo y otras celdas pueden cambiar el tiempo.

Un solo comando encadena los pendientes sobre la campaña ya existente,
normalmente13 lotes de16 y uno de15. El ejecutor original y sus sellos se
conservan. Cada lote mantiene hasta16 ajustes y1800s; proceso300s; presupuesto
acumulado7200s, del que quedan{(7200-consumed)/60:.2f}min. Verificaciones
compartidas entre lotes pueden añadir tiempo fuera del contador.

```powershell
{command}
```

No necesita placa ni COM. Guardado incremental por ajuste y lote. Se
detiene por fallo, interrupción, falta de progreso o presupuesto agotado;
no reintenta ni amplía límites. Al alcanzar288/288 imprime
`ALL_REMAINING_COMPLETED: 288/288 celdas DNN internas disponibles`.
Las48 importaciones y17 ajustes existentes no vuelven a entrenarse.

El controlador escribe un recibo adicional en
`resultados/cwru/ampliacion_v2/control_continuacion_interna_v2_2026-10-26/`.
Los modelos y lotes originales siguen en la misma campaña. Reejecutar este
comando permite continuar los pendientes completos verificados, salvo
incidencias que exijan auditoría. Conservar la salida final y ambas rutas.
La matriz DNN completa todavía no equivale a cerrar el detector, la búsqueda
clásica, la selección final ni la evaluación exterior. No se edita el manuscrito.
'''
    (PREP/'README_CONTINUACION.md').write_text(text,encoding='utf-8')
    files=[*[p for p in PREP.iterdir() if p.is_file()],*[ROOT/p for p in sources()],e.PREP_MANIFEST,
        e.PREP_MANIFEST.with_suffix('.sha256'),CAMPAIGN/'CONTRATO_CAMPANA.json',e.OUTPUT/'CAMPANA_UNICA.json']
    for batch in batches:
        files += [p for p in batch.iterdir() if p.is_file()]
    e.write_manifest(MANIFEST,extra=files)
    print(json.dumps(dict(status=continuation['status'],available=65,pending=223,
        consumed_seconds=consumed,forecast_process_minutes=audit['remaining_at_last_mean_minutes'],
        manifest_sha256=e.sha(MANIFEST),training_started=False),ensure_ascii=True))


def run():
    e.verify_manifest(MANIFEST)
    contract=e.read(PREP/'PLAN_CONTINUACION.json')
    e.require(contract['source_sha256'] == sources()
        and contract['prior_preparation_manifest_sha256'] == e.sha(e.PREP_MANIFEST),'Preparación modificada')
    plan,jobs,done,consumed=e.inspect_campaign(CAMPAIGN)
    e.require(len(done) >= contract['minimum_closed_new_fits'],'Se perdieron resultados auditados')
    e.require(e.legacy.versions() == plan['tested_versions'],'Usar Python exacto del usuario')
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    receipt=RECEIPTS/stamp;receipt.mkdir(parents=True,exist_ok=False)
    lock=CAMPAIGN/'CONTINUACION_RESTANTES.lock'
    # Si existe otro controlador, no se modifica su lock ni su recibo.
    with lock.open('x',encoding='utf-8') as stream:stream.write(stamp)
    events=receipt/'progreso.jsonl'
    def emit(event):
        with events.open('a',encoding='utf-8') as stream:stream.write(json.dumps(event,allow_nan=False)+'\n')
        print(json.dumps(event,ensure_ascii=True),flush=True)
    try:
        emit(dict(event='START',available_logical_cells=48+len(done),remaining_new_fits=len(jobs)-len(done),
            campaign=str(CAMPAIGN),frozen_budget_remaining_seconds=7200-consumed))
        result=drive(CAMPAIGN,closed_snapshot,e.execute,emit)
    except BaseException as error:
        result=dict(status='STOPPED_FOR_AUDIT',reason=type(error).__name__+': '+str(error),
            traceback=traceback.format_exc(),automatic_retry=False)
    finally:
        lock.unlink()
    result.update(campaign=str(CAMPAIGN),receipt=str(receipt),automatic_retry=False,
        original_preparation_manifest_sha256=e.sha(e.PREP_MANIFEST),continuation_manifest_sha256=e.sha(MANIFEST))
    e.write(receipt/'RESUMEN_CONTINUACION.json',result)
    e.write_manifest(receipt/'MANIFEST_CONTROL_SHA256.json',directory=receipt,
        extra=[Path(__file__).resolve(),MANIFEST,MANIFEST.with_suffix('.sha256')])
    if result['status'] == 'CAMPAIGN_COMPLETED':
        print('ALL_REMAINING_COMPLETED: 288/288 celdas DNN internas disponibles',flush=True)
    else:
        print(result['status']+': continuación detenida; conservar evidencia para revisión',flush=True)
    print('Campana: '+str(CAMPAIGN),flush=True)
    print('Evidencias del controlador: '+str(receipt),flush=True)
    return 0 if result['status'] == 'CAMPAIGN_COMPLETED' else 2


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare',action='store_true');mode.add_argument('--run',action='store_true')
    args=parser.parse_args()
    prepare() if args.prepare else sys.exit(run())
