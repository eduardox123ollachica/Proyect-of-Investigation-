# Piloto DNN V2: resultados y alcance

Piloto de tiempo y conversion, sin seleccion de representacion ni resultados confirmatorios.

| Mascara | Tiempo operativo PC (s) | Importacion (s) | 50 epocas con diagnostico (s) | FP32 (s) | INT8 (s) | INT8 (B) |
|---|---:|---:|---:|---:|---:|---:|
| F14 | 23.276 | 13.439 | 5.431 | 0.679 | 0.274 | 3904 |
| T6 | 38.901 | 26.988 | 5.532 | 1.001 | 0.188 | 3720 |

Ambos trabajos completaron 799 ventanas de entrenamiento, 175 de diagnostico,
50 epocas, 1250 actualizaciones y calibracion INT8 con las 799 filas de entrenamiento.
Se verificaron historiales, IDs, scaler, checkpoint, pesos, hashes TFLite y salidas registradas.

## Incidencia conservada

El primer intento consumio 109.193 s y fallo en INT8 por READ_VARIABLE.
Se conservo y sello. F14 se repitio una vez por recuperacion tecnica; T6 no se habia iniciado.
En total hubo tres intentos de entrenamiento, dos pilotos completos. El nuevo exportador
congela los pesos sin alterar el bucle de entrenamiento (AST identico) y guarda un checkpoint
antes de convertir. T6 requirio recuperar solo exportacion desde ese checkpoint, con cero
actualizaciones nuevas. Comparando ambas funciones a lote 1 el error fue cero; comparar
lote 175 y lote 1 daba 2.890825e-6, sin cambios de clase. No se relajaron tolerancias.
El tiempo T6 mostrado suma DOS procesos y ambas importaciones/preparaciones; F14 es un
proceso completo. No se comparan como evidencia de que una representacion sea mas rapida.

## Proyeccion de planificacion DNN

| Epocas por ajuste | Rango de perfiles, etapas (h) | Maximo de perfil + gap observado de proceso (h) |
|---:|---:|---:|
| 25 | 2.214--3.779 | 4.183 |
| 50 | 2.594--4.166 | 4.570 |

Los escenarios contemplan 288 ajustes internos y hasta 120 externos (408),
con hasta 816 conversiones antes de reutilizar identicos. Los 32 representantes ya estan
incluidos; pilotos y recuperacion se contabilizan aparte. Son extrapolaciones de dos
perfiles, no intervalos de confianza, WCET, duraciones garantizadas o latencia STM32.
FP32_conversion incluye congelacion y comparacion de 175 inferencias; tratarla como costo
fijo es una aproximacion. El gap agrega arranque, trabajo no atribuido, escritura final y salida.

El intento inicial con importacion fria se conserva separado: dos perfiles posteriores no
garantizan que todos los procesos futuros importen igual de rapido. No se usaron resultados
predictivos para elegir perfiles, mascaras, epocas confirmatorias o margenes.

**Pendiente:** piloto RF/SVM, costo de ajuste/calibracion del detector, presupuesto global y
medicion del pipeline STM32 completo para los representantes. Busqueda masiva y evaluacion
externa no ejecutadas. Manuscrito, particiones y evidencias historicas se conservan.

Evidencias: `resultados/cwru/ampliacion_v2/piloto_tiempo_exportacion_v2_2026-10-25/20261025T063627_547883Z`.

La primera importacion fue aproximadamente 96.285 s (estimada desde eventos).
Si se repitiera en cada ajuste, el escenario de sensibilidad de 50 epocas seria 12.197 h DNN.
Es un supuesto de sensibilidad, no una prediccion ni una garantia; demuestra por que el
presupuesto global todavia no se declara congelado.

## Diagnostico FP32/INT8

Los cambios de clase se conservan como diagnostico de estas instancias de 50 epocas,
sin usarlos para elegir una mascara o modificar el protocolo. No equivalen por si solos
a una perdida de exactitud: se necesitan etiquetas y el protocolo predictivo correspondiente.
- F14: 15 cambios entre TFLite FP32 e INT8 en 175 ventanas.
- T6: 18 cambios entre TFLite FP32 e INT8 en 175 ventanas.
