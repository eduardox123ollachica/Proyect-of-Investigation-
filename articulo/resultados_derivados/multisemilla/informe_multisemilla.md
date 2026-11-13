# Resultados del experimento multisemilla E1-E4

Fecha de cierre: 2026-10-12.

## Diseño

Se entrenaron desde cero E1 y E4 con diez semillas predefinidas: 11, 22, 33,
42, 44, 55, 66, 77, 88 y 99. Para cada ejecución se conservaron los mismos datos,
arquitectura, épocas, batch size, optimizador, learning rate, L2 y procedimiento de
cuantización. Cada modelo fue evaluado en Keras, TFLite FP32 y TFLite INT8 sobre
las mismas 1332 ventanas de prueba.

## Resultados agregados INT8

| Variante | Media accuracy | DE | Mediana | Mínimo | Máximo | Media macro-F1 | DE macro-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| E1 | 98.4459 % | 1.7047 | 99.1742 % | 95.4955 % | 100.0000 % | 98.3644 % | 1.8094 |
| E4 | 99.3994 % | 1.5061 | 99.8498 % | 95.1201 % | 100.0000 % | 99.3690 % | 1.6019 |

La cuantización fue estable: el cambio medio INT8 respecto de Keras fue
+0.0225 puntos porcentuales para E1 y -0.0150 para E4. Todos los modelos INT8
ocuparon 4480 bytes.

## Comparación pareada por semilla

- Diferencia media E4-E1: +0.9535 puntos porcentuales.
- Desviación estándar de las diferencias: 2.5501 puntos porcentuales.
- IC del 95 % de la diferencia media: -0.8707 a +2.7776 puntos porcentuales.
- E4 superó a E1 en 7 semillas, empataron en 2 y E1 superó a E4 en 1.
- Prueba exacta bilateral de los signos, excluyendo empates: p = 0.0703125.

Con diez semillas se observa una tendencia favorable a E4, pero la evidencia no
alcanza el umbral convencional de 0.05 y el intervalo incluye cero. Por ello no debe
afirmarse que la ponderación de clases garantiza una mejora para cualquier
inicialización.

## Ejecución atípica relevante

E4 con semilla 44 obtuvo 95.1201 % en INT8. Clasificó 63 de las 360 ventanas MEDIA
como DETENIDO, pese a alcanzar 99.7054 % de accuracy al final del entrenamiento.
Esto apunta a variabilidad de generalización, no a un fallo de convergencia ni de
cuantización. La ejecución se conserva y no se elimina como outlier.

## Reproducción de los modelos canónicos

La semilla 42 reprodujo exactamente las cifras ya usadas para la validación STM32:

- E1 INT8: 97.8978979 %.
- E4 INT8: 99.9249249 %.

Además, los archivos TFLite INT8 de la semilla 42 coinciden byte a byte con los
modelos canónicos E1 y E4, según su hash SHA-256. Esto confirma la trazabilidad de
los modelos desplegados y del nuevo protocolo.

## Interpretación permitida para el artículo

La formulación respaldada es:

> En diez inicializaciones pareadas, la variante con ponderación de clases presentó
> mayor media y mediana de accuracy INT8 y superó a la línea base en siete semillas;
> no obstante, una ejecución mostró una degradación marcada y la diferencia media
> no fue concluyente al 95 %, evidenciando sensibilidad a la inicialización.

El resultado canónico E4 sigue siendo válido como modelo desplegado, pero no debe
generalizarse como una ventaja determinista del uso de `class_weight`.
