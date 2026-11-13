# Primer ajuste DNN interno auditado

Se verificaron firmas, sellos, fuentes y las 48 importaciones sin nuevos
entrenamientos o inferencias. La celda F14/outer0/inner1/semilla22 aprueba:
812 ventanas de entrenamiento (cargas2 y3),393 de validación (carga1),
carga0 excluida. Exactitud y macro-F1 FP32/INT8:100 %. No son resultados de
la evaluación exterior. Ejecutó50 épocas y restauró la48 (1300 updates).
No hubo cambios de clase FP32–INT8 ni saturaciones de entrada.

La campaña tiene49/288 celdas:48 reutilizadas y una nueva; faltan239.
Proceso informado: 46.158s; controlador y
comprobación: 50.578s. Carga de TensorFlow:
41.338s; entrenamiento:
2.397s.

Este arranque superó las mediciones usadas para prever1–2h. Si ese ritmo se
mantuviera,16 ajustes tomarían 13.49min y los239 restantes
201.47min, por encima de los 119.16min
de presupuesto acumulado disponibles. Una observación no fija ese ritmo:
el próximo lote permitirá reevaluarlo. No se cambia el presupuesto sellado.

## Siguiente ejecución personal

Ejecutar16 ajustes como máximo sobre la MISMA campaña, sin placa ni COM.
Reservar aproximadamente15–25min incluyendo comprobaciones. Límite de lote:
1800s; proceso:300s. Puede parar antes de16 si no alcanza el presupuesto
restante para otro proceso completo. No hay ampliación ni reintento automático.

```powershell
& "C:\Users\LENOVO\AppData\Local\Microsoft\WindowsApps\python3.12.exe" -B "C:\Users\LENOVO\Desktop\Proyecto Investigación\codigo\cwru\ampliacion_v2\busqueda_interna\ejecutar_busqueda_interna.py" --run --campaign "C:\Users\LENOVO\Desktop\Proyecto Investigación\resultados\cwru\ampliacion_v2\busqueda_interna_v2_2026-10-26\20261027T005636_578225Z" --max-new-jobs 16 --max-seconds 1800
```

Si completan los16, el resumen tendrá65/288 celdas disponibles y quedarán223
ajustes nuevos. Conservar la salida y la ruta de evidencias para auditoría.
No repetir el comando de campaña nueva: reanudar con `--campaign`.
