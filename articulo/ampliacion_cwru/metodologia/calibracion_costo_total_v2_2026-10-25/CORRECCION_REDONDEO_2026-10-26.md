# Divergencia y corrección de la ejecución INT8

La campaña física `20261026T042424_323718Z` se detuvo en el representante
22, DE7, ventana 232. Se conservaron 66 RUN y 22 PROFILE aceptados; la
publicación fallida quedó en UART y no se aceptó como muestra de costo.
El RAW y las etapas previas de extracción, escalado y cuantización pasan.
La entrada INT8 coincide, pero STM32 produce `[42,-101,-69]` y PC
`[41,-100,-69]`. La clase coincide únicamente en este caso.

## Evidencia causal

Dos kernels PC de TensorFlow 2.21.0, optimizado y referencia, reproducen
exactamente las 33 salidas de las capas densas al usar redondeo simple.
El binario CMSIS-NN realmente enlazado utiliza doble redondeo, verificado
mediante su ensamblado y la biblioteca precompilada original.

Un replay entero desde los pesos y sesgos localiza la primera diferencia
en la primera neurona: el valor 50.4967897104 se redondea una vez a 50; el
doble redondeo pasa por 6464/128 = 50.5 y produce 51. Con el punto cero
−128, las activaciones resultan −78 y −77. La propagación produce logits
`[33,-29,-3]` y `[33,-30,-3]`. Un Softmax aislado con los parámetros del
modelo reproduce exactamente las salidas PC y STM32 respectivas.

No se capturaron activaciones internas en la placa. El mecanismo se
fundamenta en el binario, el replay entero y la reproducción de ambas
salidas finales; la variante corregida debe comprobarse físicamente.
No se atribuye la divergencia a la semilla ni se relaja la comparación.

La documentación primaria de
[CMSIS-NN](https://github.com/ARM-software/CMSIS-NN/blob/main/README.md)
indica que elegir redondeo simple puede cambiar salidas y que las opciones
que afectan cabeceras deben corresponder con la configuración de ejecución.
Las [ecuaciones publicadas por TensorFlow](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/kernels/internal/common.cc)
se contrastaron con las fuentes instaladas; no se asumió que la versión
actual publicada identificara por sí sola el binario local.

## Variante SR y límites

Se prepara una configuración 17 aislada, preservando las 16 anteriores.
Incorpora la fuente CMSIS-NN original del matvec por canal con redondeo
simple y mantiene intacta la biblioteca precompilada. El alcance es ese
kernel, no toda la biblioteca. Los 32 modelos tienen tres FC INT8 por
canal, con 1056 multiplicadores en el dominio de shifts −12 a −5. Softmax
usa shifts entre 21 y 24. Se comprobaron los dominios para evitar diferencias
en las ramas extremas de preparación de parámetros de la biblioteca.

Los modelos, pesos, datos, particiones, normalización y criterio de salida
exacta se conservan. La prueba previa de la variante incorpora el caso
fallido, además del control original, y no admite la puerta antigua como
autorización de campaña. Los resultados y tiempos de firmwares distintos
no se mezclarán para seleccionar representaciones.

La corrección física, la tabla completa y la selección siguen pendientes.
Este documento registra el desarrollo y no modifica el manuscrito.

Fuentes locales: diagnóstico cerrado en
`despliegue_stm32/costo_total_representantes_v2_180mhz_2026-10-25/diagnostico_divergencia_20261026T042424Z/`
y preparación en
`despliegue_stm32/costo_total_single_rounding_v2_180mhz_2026-10-26/`.
