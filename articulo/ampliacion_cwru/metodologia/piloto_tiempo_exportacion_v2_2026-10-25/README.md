# Corrección de exportación y recuperación del piloto

25 de octubre de 2026. Se mantiene el [alcance del piloto inicial](../piloto_tiempo_2026-10-25/README.md):
F14/T6, semilla 11, entrenamiento 799 de 1/2 HP, diagnóstico 175 de 0 HP,
50 épocas y 1250 actualizaciones, sin cargar arrays de 3 HP ni iniciar búsqueda.

## Incidencia real conservada

El intento `20261025T063025_963041Z` terminó F14 y su conversión FP32, pero falló
al calibrar INT8 con `READ_VARIABLE: variable != nullptr`. Se detuvo la cola:
T6 no comenzó. Se conservan logs, fallo, artefactos y sello del intento.
Su duración de proceso fue 109.193179 s, dominada por la primera importación
(aproximadamente 96 s). No se elimina ese costo ni se interpreta como tiempo
de entrenamiento o latencia STM32.

El primer worker guardaba los pesos al finalizar todas las conversiones. Al
fallar antes, no quedó un checkpoint recuperable. Esta revisión repite una vez
F14 como recuperación técnica y ejecuta T6 por primera vez: dos ajustes nuevos,
tres intentos de entrenamiento contando el fallido. No es una nueva semilla,
selección de hiperparámetros ni repetición de resultados confirmatorios.

## Cambio y fundamento

Se exporta una ConcreteFunction con los valores de las variables convertidos
a constantes antes de producir FP32 e INT8. La implementación está en el
[código oficial de TensorFlow 2.21.0](https://github.com/tensorflow/tensorflow/blob/v2.21.0/tensorflow/python/framework/convert_to_constants.py).
Se trata de una API interna: se registra la versión y su compatibilidad real
deberá revisarse si cambia TensorFlow.

Se comprueba ausencia de operaciones de variables, pesos originales idénticos,
salidas equivalentes en las 175 ventanas de diagnóstico (rtol 1e-6, atol 1e-7)
y ninguna clase distinta al congelar. Esta tolerancia es numérica de exportación,
no el margen predictivo V2. No se modifica arquitectura, normalización, función
de pérdida, optimizador, semillas, particiones o cantidad de épocas.

Además se guardan pesos, historial y tiempos **antes** de convertir. Así un
fallo futuro de exportación conserva el ajuste y permite recuperarlo sin
reentrenar. Los scripts iniciales y sus sellos permanecen intactos; los nuevos
se identifican con `_v2.py`, contrato y sello de preparación independientes.

Cada nuevo proceso conserva el límite de 300 s, secuencia F14/T6 y parada ante
el primer fallo. No hay reintentos automáticos. Las salidas nuevas quedan en
`resultados/cwru/ampliacion_v2/piloto_tiempo_exportacion_v2_2026-10-25/<fecha_UTC>/`.
El contrato registra expresamente la recuperación y sus límites.

La proyección separará las diez etapas del wall time observado. Los 408 ajustes
DNN/816 conversiones máximas no incluyen los pilotos ni esta recuperación;
los 32 representantes ya están incluidos. RF/SVM, detector y mediciones de costo
total STM32 siguen pendientes. No se congela todavía el presupuesto completo.
