"""Auditar primer reajuste final y documentar continuación, sin reentrenar."""
from pathlib import Path
import datetime
import json
import sys

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'codigo/cwru/ampliacion_v2/reajustes_finales'))
import contrato as c
import ejecutar_reajustes as executor

CAMPAIGN=c.OUTPUT/'20261027T173452_969615Z'
OUT=ROOT/'articulo/ampliacion_cwru/metodologia/cierre_primer_reajuste_final_v2_2026-10-27'


def main():
    io=c.io
    io.require(not OUT.exists(),'Cierre existente; no sobrescribir')
    plan=c.preflight(environment=True)
    io.require(io.read(c.OUTPUT/'CAMPANA_UNICA.json')==dict(run_id=CAMPAIGN.name,
        preparation_sha256=io.sha(c.MANIFEST)),'Campaña distinta de la preparada')
    io.require(not (CAMPAIGN/'ACTIVO.lock').exists() and not (CAMPAIGN/'SOLICITUD_ACTIVA.json').exists(),
        'Campaña activa o incompleta')
    batches=sorted((CAMPAIGN/'lotes').iterdir())
    io.require(len(batches)==1,'Este cierre corresponde únicamente al primer lote')
    for batch in batches:
        io.verify_manifest(batch/'MANIFEST_LOTE_SHA256.json')
        summary=io.read(batch/'RESUMEN_LOTE.json')
        io.require(summary['status']=='BATCH_COMPLETED' and summary['completed_jobs']==1
            and summary['jobs_started']==summary['jobs_passed']==1,'Primer lote incompleto o distinto')
    done,consumed=executor.inspect_campaign(CAMPAIGN,plan)
    io.require(done==['outer0_F14_e45_s11'],'Cierre ajeno al primer reajuste')
    result=io.read(CAMPAIGN/done[0]/'RESULTADO.json')
    io.require(result['epochs_executed']==45 and result['optimizer_updates']==1710
        and result['training_windows']==1205 and result['training_records']==21,'Scope del primer ajuste distinto')
    pending=plan['jobs'][1:]
    counts={family:sum(j['family']==family for j in pending) for family in ('DNN','RF','SVM','detector')}
    io.require(counts==dict(DNN=79,RF=40,SVM=4,detector=7),'Pendientes inesperados')
    command=('& "C:\\Users\\LENOVO\\AppData\\Local\\Microsoft\\WindowsApps\\python3.12.exe" -B '
        '"C:\\Users\\LENOVO\\Desktop\\Proyecto Investigación\\codigo\\cwru\\ampliacion_v2\\reajustes_finales\\ejecutar_reajustes.py" '
        '--run --max-new-jobs 130 --max-seconds 7200')
    OUT.mkdir(parents=True,exist_ok=False)
    closure=dict(status='FIRST_FINAL_REFIT_AUDITED_PASS_READY_FOR_REMAINING_USER_RUN',
        date_local='2026-10-27',timezone='America/Lima',verified_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        campaign=CAMPAIGN.relative_to(ROOT).as_posix(),selection_signature=c.SELECTION_SIGNATURE,
        preparation_manifest_sha256=io.sha(c.MANIFEST),completed_jobs=done,epochs=45,optimizer_updates=1710,
        training_windows=1205,training_MAT=21,training_loads_hp=[1,2,3],heldout_load_hp=0,
        job_execution_verification_seconds=consumed,worker_seconds=result['elapsed_worker_seconds'],
        stage_seconds=result['timings_seconds'],remaining_jobs=len(pending),remaining_counts=counts,
        next_job_id=pending[0]['job_id'],campaign_seconds_remaining=c.LIMITS['campaign_execution_verification_seconds']-consumed,
        continuation_command=command,original_job_repeated=False,new_training_fits=0,new_TFLite_invocations=0,
        training_diagnostics_reconstructed_from_saved_outputs=True,scaler_reconstructed_for_verification=True,
        outer_evaluation_executed=False,outer_values_selected_or_materialized=False,
        TensorFlow_imported='tensorflow' in sys.modules,hardware_connected=False,manuscript_modified=False,
        interpretation='Training diagnostics are technical checks, not heldout accuracy or evidence of generalization. One elapsed process is not a guarantee for all remaining jobs.')
    io.require(closure['TensorFlow_imported'] is False,'Auditoría no requiere TensorFlow')
    io.write(OUT/'CIERRE_PRIMER_REAJUSTE.json',closure)
    (OUT/'CONTINUAR_130_PENDIENTES.ps1').write_text(command+'\n',encoding='utf-8-sig')
    text=f'''# Primer reajuste final auditado y continuación

`outer0_F14_e45_s11` aprobó la reconstrucción de scaler, pesos, historia y
diagnósticos, y la verificación de firmas/manifiestos. Completó 45 épocas,
1710 actualizaciones y calibración INT8 con las 1205 ventanas permitidas de
21 MAT de cargas 1/2/3 HP. La carga 0 HP no se utilizó en este ajuste.

Tiempo de proceso y verificación: **{consumed:.3f} s**; worker **{result['elapsed_worker_seconds']:.3f} s**.
Son tiempos de PC, no latencia STM32. La comprobación no reentrenó la DNN,
ejecutó TFLite ni evaluó la carga reservada. Las métricas perfectas guardadas
son de entrenamiento; todavía no miden generalización.

Restan **130 trabajos: 79 DNN, 40 RF, 4 SVM y 7 detectores**. El siguiente es
`{pending[0]['job_id']}`. El comando reconoce automáticamente esta misma
campaña y conserva los **{consumed:.3f} s** consumidos: quedan
**{closure['campaign_seconds_remaining']:.3f} s** del presupuesto acumulado.
No repite el trabajo aprobado ni modifica sus resultados.

Ejecutar personalmente desde PowerShell:

```powershell
{command}
```

Puede ejecutarse desde cualquier directorio; no abrir otro ejecutor simultáneo.
El presupuesto de lote es 7200 s y permanece el límite acumulado anterior.
Cada proceso conserva su watchdog y se detiene la cola ante un fallo, timeout,
interrupción o datos incompatibles. No borrar locks o resultados parciales;
conservar las evidencias para una revisión técnica. No se amplían márgenes,
semillas o épocas según los resultados. La evaluación exterior sigue bloqueada.

La proyección histórica del paquete completo es de unos 29 minutos; reservar
30–100 minutos mientras varíen los arranques/importaciones. El primer proceso
real fue más rápido que el ensayo sintético frío, pero una observación no
garantiza la duración de toda la cola. Las comprobaciones compartidas y el
cierre pueden añadir tiempo calendario sobre las fronteras descritas.

Al completar se espera:

```text
FINAL_REFITS_COMPLETED_NO_OUTER_EVALUATION: 131/131 trabajos completos.
```

Los modelos, parámetros, solicitudes train-only y logs se seguirán guardando en:
`{CAMPAIGN}`. Este cierre habilita entregar el comando; no ejecuta los pendientes.
'''
    (OUT/'RESULTADO_Y_COMANDO_CONTINUACION.md').write_text(text,encoding='utf-8')
    extra=[Path(__file__),c.MANIFEST,c.MANIFEST.with_suffix('.sha256'),c.PLAN,
        c.OUTPUT/'CAMPANA_UNICA.json',CAMPAIGN/'CONTRATO.json']
    extra += [p for p in CAMPAIGN.rglob('*') if p.is_file()]
    manifest=OUT/'MANIFEST_CIERRE_PRIMER_REAJUSTE_SHA256.json'
    io.write_manifest(manifest,directory=OUT,extra=extra)
    print(json.dumps(dict(status=closure['status'],completed_jobs=1,remaining=counts,
        seconds=consumed,manifest_sha256=io.sha(manifest),output=str(OUT)),ensure_ascii=True))


if __name__=='__main__':main()
