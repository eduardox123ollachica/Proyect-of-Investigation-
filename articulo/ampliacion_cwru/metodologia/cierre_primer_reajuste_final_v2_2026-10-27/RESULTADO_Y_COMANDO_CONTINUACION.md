# Primer reajuste final auditado y continuación

`outer0_F14_e45_s11` aprobó la reconstrucción de scaler, pesos, historia y
diagnósticos, y la verificación de firmas/manifiestos. Completó 45 épocas,
1710 actualizaciones y calibración INT8 con las 1205 ventanas permitidas de
21 MAT de cargas 1/2/3 HP. La carga 0 HP no se utilizó en este ajuste.

Tiempo de proceso y verificación: **12.422 s**; worker **7.950 s**.
Son tiempos de PC, no latencia STM32. La comprobación no reentrenó la DNN,
ejecutó TFLite ni evaluó la carga reservada. Las métricas perfectas guardadas
son de entrenamiento; todavía no miden generalización.

Restan **130 trabajos: 79 DNN, 40 RF, 4 SVM y 7 detectores**. El siguiente es
`outer0_F14_e45_s22`. El comando reconoce automáticamente esta misma
campaña y conserva los **12.422 s** consumidos: quedan
**7187.578 s** del presupuesto acumulado.
No repite el trabajo aprobado ni modifica sus resultados.

Ejecutar personalmente desde PowerShell:

```powershell
& "C:\Users\LENOVO\AppData\Local\Microsoft\WindowsApps\python3.12.exe" -B "C:\Users\LENOVO\Desktop\Proyecto Investigación\codigo\cwru\ampliacion_v2\reajustes_finales\ejecutar_reajustes.py" --run --max-new-jobs 130 --max-seconds 7200
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
`C:\Users\LENOVO\Desktop\Proyecto Investigación\resultados\cwru\ampliacion_v2\reajustes_finales_v2_2026-10-27\20261027T173452_969615Z`. Este cierre habilita entregar el comando; no ejecuta los pendientes.
