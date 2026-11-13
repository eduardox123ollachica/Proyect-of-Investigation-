# Cierre del piloto RF, SVM y detector

Se aprobaron 17/17 procesos en 62.687 s de orquestador.
El usuario ejecutó el piloto; esta auditoría no entrenó o reinfirió. Se conservaron sus
artefactos y fuentes, sin abrir matrices fuente como arrays ni modificar el manuscrito.

| Bloque | Procesos del piloto | Tiempo observado de procesos (s) |
|---|---:|---:|
| RF | 8 | 32.041 |
| SVM | 8 | 27.116 |
| DETECTOR | 1 | 3.295 |

Los 13 minutos eran la suma de límites por proceso, no la duración esperada.
Entrenamiento 1/2 HP: 799 ventanas de 14 MAT; diagnóstico 0 HP: 175 de siete MAT.
Detector: N de MAT 98 para ajuste y N de MAT 99 para P95, 58/58, sin refit.
Las siete representaciones conservaron hashes idénticos en sus tres repeticiones.
NF12 es alias de F14; no se creó una medición duplicada. Estas repeticiones no son semillas.

Se verificaron sellos, roles, versiones, configuraciones, salidas, scalers, parámetros,
scores y tiempos. Se reprodujo la proyección desde los tiempos registrados, sin
volver a ejecutar ajuste, calibración, predicción o hardware.

## Proyección de campaña futura PC

| Bloque | Ajustes máximos antes de reutilizar | Etapas (min) | Con residual de procesos (min) |
|---|---:|---:|---:|
| RF | 328 | 21.818 | 25.506 |
| SVM | 100 | 4.716 | 5.755 |
| detector | 108 | 4.896 | 5.880 |
| Total clásico + detector | — | 31.430 | 37.141 |

Son escenarios por tamaños de datos y tiempos medidos, no cotas ni duración garantizada.
El overhead del detector se copia conservadoramente a cada ajuste antes de caché.
RF/SVM incluyen la llamada diagnóstica adicional de inferencia; no son costos mínimos.
El máximo inicial de 25.1741 ms del ajuste/calibración del detector se conserva.

Los artefactos permiten comprobar consistencia; no hay una traza independiente del sistema
operativo para todos los accesos a archivos. Se guardaron arrays de la primera repetición
del detector y hashes de las otras dos. No se recalculó P95 con fuentes durante el cierre.

Siguiente paso: implementar/verificar el ejecutor definitivo y fijar límites de campaña.
Después, preparar los 32 representantes de costo total STM32. Están incluidos en las
288 DNN internas: no se suman otra vez. Toda ejecución prolongada corresponde al usuario.

Las métricas predictivas, selección y evaluación externa no se ejecutaron en este piloto.
El protocolo completo y el costo total físico STM32 siguen pendientes.
