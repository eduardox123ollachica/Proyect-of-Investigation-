# Fase 4: completar el factorial multisemilla

## Propósito

La ablación original E1--E4 describe un factorial 2x2 entre escalado de las
características y ponderación de clases. Antes de esta fase solo E1 y E4 tenían
diez entrenamientos. Esto permitía comparar dos esquinas del diseño, pero no
separar el efecto del `StandardScaler`, el efecto de `class_weight` y su posible
interacción.

## Cambios necesarios para ejecutar el protocolo

Los scripts de E2 y E3 fijaban internamente la semilla 42. Se añadió el
argumento `--seed` sin cambiar dataset, separación, arquitectura, número de
épocas, tamaño de lote, optimizador, tasa de aprendizaje, regularización ni
cuantización. El orquestador se amplió para seleccionar cualquiera de E1--E4 y
continúa operando en modo de planificación salvo que se indique `--execute`.

## Ejecución realizada

- Variantes nuevas: E2 (con scaler, sin pesos) y E3 (con scaler, con pesos).
- Semillas: 11, 22, 33, 42, 44, 55, 66, 77, 88 y 99.
- Total: 20 entrenamientos nuevos, ejecutados secuencialmente.
- Evidencias por ejecución: modelo Keras, TFLite FP32, TFLite INT8, historial,
  métricas, matriz de confusión, probabilidades y predicciones comparadas.
- E1 y E4 no se volvieron a ejecutar; se conservaron sus veinte resultados
  previos.

Los resultados están en
`resultados/arciniegas/multisemilla/ejecuciones/e2` y
`resultados/arciniegas/multisemilla/ejecuciones/e3`; los registros de consola
están en `resultados/arciniegas/multisemilla/logs`.

## Controles de integridad

La estructura final contiene diez carpetas completas para cada variante E1,
E2, E3 y E4. Ningún registro nuevo contiene `Traceback`, `Exception`, `Failed`
o una línea de error.

Como control de no regresión, la nueva ejecución con semilla 42 se comparó con
la ejecución canónica:

- E2: coincidencia exacta de los CSV de métricas y predicciones, de
  `replica_info.txt` y del modelo INT8 (SHA-256
  `EFADEAFE502A13B7B6B8378F895EE1ADDBBAE9E0ABE8E7008F1A2F82309C83AC`).
- E3: coincidencia exacta de los CSV de métricas y predicciones y de
  `replica_info.txt`. El directorio canónico no conserva el archivo INT8, por
  lo que no fue posible una comparación binaria de ese artefacto.

Esta coincidencia demuestra que parametrizar la semilla no alteró el caso
canónico de semilla 42.

## Alcance de la conclusión

La fase 4 completa la evidencia experimental, pero todavía no autoriza una
conclusión sobre qué factor es superior. La consolidación de macro-F1,
intervalos de confianza, contrastes pareados, efectos marginales e interacción
corresponde a la fase 5. Tampoco se modifica todavía la selección del modelo ni
se inicia CWRU.
