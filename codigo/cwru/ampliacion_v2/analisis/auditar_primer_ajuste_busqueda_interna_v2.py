"""Audita el primer ajuste real del usuario sin fit ni inferencias nuevas."""
from __future__ import annotations

import datetime
import importlib.abc
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[4]
SOURCE=ROOT/'codigo/cwru/ampliacion_v2/busqueda_interna'
sys.path.insert(0,str(SOURCE))


class ReadOnlyRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in ('tensorflow','keras','ai_edge_litert'):
            raise RuntimeError('Auditoría: runtime ML bloqueado; no entrenar ni inferir')
        return None


sys.meta_path.insert(0,ReadOnlyRuntime())
import ejecutar_busqueda_interna as e


def main():
    campaign=e.OUTPUT/'20261027T005636_578225Z'
    plan=e.preflight(check_environment=True)
    e.verify_manifest(e.PREP/'MANIFEST_PREPARACION_COMPLETA_SHA256.json')
    plan,jobs,done,consumed=e.inspect_campaign(campaign)
    e.require(done == ['inner_outer0_inner1_F14_seed22'], 'No es la puerta de un ajuste prevista')
    e.require(not (campaign/'EJECUCION_EN_CURSO.lock').exists(), 'Hay ejecución activa')
    batches=sorted((campaign/'lotes').iterdir())
    e.require(len(batches) == 1, 'Se esperaba un solo lote')
    batch=batches[0]
    summary=e.read(batch/'RESUMEN_LOTE.json')
    e.require(summary['status'] == 'BATCH_COMPLETED' and summary['jobs_passed'] == summary['jobs_started'] == 1
        and summary['completed_new_fits'] == 1 and summary['available_logical_cells'] == 49
        and summary['campaign_worker_seconds_consumed'] == consumed,
        'El resumen no coincide con los artefactos verificados')
    job=next(j for j in jobs if j['job_id'] == done[0])
    result=e.read(campaign/done[0]/'resultado.json')
    reference=next(v for v in plan['imported'].values() if v['source_job_id'] == 'cost_outer0_inner1_F14_seed11')
    ref=e.read(ROOT/reference['source_directory']/'resultado.json')
    e.require(result['model_sha256'] != ref['model_sha256'], 'El nuevo ajuste no debe ser una copia de la referencia')
    e.require(result['roles']['training_loads'] == [2,3]
        and result['inner_validation_load_hp'] == 1 and result['outer_load_hp'] == 0,
        'Roles de la primera celda distintos')
    e.require(result['elapsed_worker_seconds'] <= plan['limits']['worker_seconds'], 'Proceso fuera del límite')
    remaining=plan['new_fits']-len(done)
    remaining_budget=plan['limits']['campaign_worker_seconds']-consumed
    first_seconds=summary['worker_elapsed_seconds']
    forecast_batch=16*first_seconds/60
    forecast_remaining=remaining*first_seconds/60
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output=ROOT/'articulo/ampliacion_cwru/metodologia/seguimiento_busqueda_interna_v2_2026-10-26'/stamp
    output.mkdir(parents=True,exist_ok=False)
    report=dict(status='FIRST_INNER_DNN_FIT_AUDITED_READY_FOR_USER_BATCH16',
        date_local='2026-10-26',timezone='America/Lima',
        campaign=campaign.relative_to(ROOT).as_posix(),job_id=done[0],
        source_batch_manifest_sha256=e.sha(batch/'MANIFEST_LOTE_SHA256.json'),
        source_job_manifest_sha256=e.sha(campaign/done[0]/'MANIFEST_JOB_SHA256.json'),
        preparation_manifest_sha256=e.sha(e.PREP_MANIFEST),
        original_sources_and_imports_verified=True,model_checkpoints_scaler_quantization_metrics_verified=True,
        result_sha256=e.sha(campaign/done[0]/'resultado.json'),
        imported_cells=48,completed_new_fits=1,available_logical_cells=49,total_logical_cells=288,
        new_fits_remaining=remaining,roles=job['roles'],training_windows=result['training_windows'],
        validation_windows=result['validation_windows'],metrics=result['metrics'],
        best_epoch=result['best_epoch'],epochs_executed=result['epochs_executed'],
        weights_restored=result['weights_restored'],optimizer_updates=result['optimizer_updates'],
        FP32_INT8_class_changes=result['FP32_INT8_class_changes'],
        training_saturated_input_values=result['training_saturated_input_values'],
        validation_saturated_input_values=result['int8_details']['input_clipped_values'],
        worker_reported_seconds=result['elapsed_worker_seconds'],
        controller_execution_and_verification_seconds=first_seconds,
        timings_seconds=result['timings_seconds'],campaign_budget_consumed_seconds=consumed,
        campaign_budget_remaining_seconds=remaining_budget,
        timing_sensitivity=dict(observations=1, following_16_at_same_rate_minutes=forecast_batch,
            remaining_239_at_same_rate_minutes=forecast_remaining,
            remaining_budget_minutes=remaining_budget/60,
            same_rate_would_exceed_remaining_budget=forecast_remaining > remaining_budget/60,
            prediction_is_not_a_guarantee=True,initial_1_to_2_hour_forecast_requires_reassessment=True,
            action='Measure next16, preserve frozen limits; stop rather than raise them automatically'),
        next_batch=dict(new_jobs=16,seconds=1800,worker_seconds=300,
            expected_available_cells_if_all_pass=65,expected_new_fits_remaining_if_all_pass=223),
        new_training_fits_by_auditor=0,new_TFLite_invocations_by_auditor=0,
        TensorFlow_imported='tensorflow' in sys.modules,hardware_connected=False,
        existing_campaign_files_modified=False,manuscript_modified=False,
        limitations=['100 percent is one internal validation cell, not external test performance.',
            'A single startup does not establish a duration distribution for the full campaign.',
            'No independent motors/bearings or predictive selection are established by this gate.'])
    e.write(output/'AUDITORIA_PRIMER_AJUSTE.json',report)
    command=f'& "C:\\Users\\LENOVO\\AppData\\Local\\Microsoft\\WindowsApps\\python3.12.exe" -B "{SOURCE / "ejecutar_busqueda_interna.py"}" --run --campaign "{campaign}" --max-new-jobs 16 --max-seconds 1800'
    text=f'''# Primer ajuste DNN interno auditado

Se verificaron firmas, sellos, fuentes y las 48 importaciones sin nuevos
entrenamientos o inferencias. La celda F14/outer0/inner1/semilla22 aprueba:
812 ventanas de entrenamiento (cargas2 y3),393 de validación (carga1),
carga0 excluida. Exactitud y macro-F1 FP32/INT8:100 %. No son resultados de
la evaluación exterior. Ejecutó50 épocas y restauró la48 (1300 updates).
No hubo cambios de clase FP32–INT8 ni saturaciones de entrada.

La campaña tiene49/288 celdas:48 reutilizadas y una nueva; faltan239.
Proceso informado: {result['elapsed_worker_seconds']:.3f}s; controlador y
comprobación: {first_seconds:.3f}s. Carga de TensorFlow:
{result['timings_seconds']['imports']:.3f}s; entrenamiento:
{result['timings_seconds']['fit_including_checkpoint_IO']:.3f}s.

Este arranque superó las mediciones usadas para prever1–2h. Si ese ritmo se
mantuviera,16 ajustes tomarían {forecast_batch:.2f}min y los239 restantes
{forecast_remaining:.2f}min, por encima de los {remaining_budget/60:.2f}min
de presupuesto acumulado disponibles. Una observación no fija ese ritmo:
el próximo lote permitirá reevaluarlo. No se cambia el presupuesto sellado.

## Siguiente ejecución personal

Ejecutar16 ajustes como máximo sobre la MISMA campaña, sin placa ni COM.
Reservar aproximadamente15–25min incluyendo comprobaciones. Límite de lote:
1800s; proceso:300s. Puede parar antes de16 si no alcanza el presupuesto
restante para otro proceso completo. No hay ampliación ni reintento automático.

```powershell
{command}
```

Si completan los16, el resumen tendrá65/288 celdas disponibles y quedarán223
ajustes nuevos. Conservar la salida y la ruta de evidencias para auditoría.
No repetir el comando de campaña nueva: reanudar con `--campaign`.
'''
    (output/'VALIDACION_Y_SIGUIENTE_LOTE.md').write_text(text,encoding='utf-8')
    e.write_manifest(output/'MANIFEST_AUDITORIA_SHA256.json',extra=[
        output/'AUDITORIA_PRIMER_AJUSTE.json',output/'VALIDACION_Y_SIGUIENTE_LOTE.md',
        Path(__file__).resolve(),e.PLAN_FILE,e.PREP_MANIFEST,e.PREP_MANIFEST.with_suffix('.sha256'),
        *[p for p in (campaign/done[0]).iterdir() if p.is_file()],
        *[p for p in batch.iterdir() if p.is_file()],campaign/'CONTRATO_CAMPANA.json',
        e.OUTPUT/'CAMPANA_UNICA.json'])
    print(json.dumps(dict(status=report['status'],available_logical_cells=49,new_fits_remaining=remaining,
        consumed_seconds=consumed,forecast_next16_minutes=forecast_batch,
        output=str(output),TensorFlow_imported=report['TensorFlow_imported']),ensure_ascii=True))


if __name__ == '__main__':
    main()
