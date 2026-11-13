# Auditoría de rechazos K-means P99

## Alcance

Se auditaron las 12 ventanas de prueba con `score_d2 > 21.850406448`. La auditoría reproduce la distancia a partir del scaler y los centroides, revisa continuidad temporal, resultado DNN, contribuciones por característica y señal cruda con una ventana vecina a cada lado. No asigna una etiqueta de anomalía o normalidad que el corpus no contiene.

## Resultados verificables

- 12 rechazos distribuidos en 7 archivos.
- 10 ventanas temporalmente aisladas y 2 pertenecientes a un único grupo de dos ventanas consecutivas.
- La DNN E4 clasificó correctamente 12/12 ventanas rechazadas.
- La mayor concentración fue 5 ventanas en la captura s10 de VELOCIDAD MEDIA.
- Los scores fueron reproducidos desde las 21 características, el scaler y los centroides con error numérico despreciable.

Características que más veces aportaron la mayor contribución:

| feature | count |
| --- | --- |
| ax_spectral_entropy | 3 |
| ay_kurtosis | 3 |
| az_skewness | 2 |
| ax_kurtosis | 2 |
| ay_skewness | 2 |

## Detalle por ventana

| class | capture | window | score_d2 | consecutive_group_length | dnn_correct | top1_feature | top1_contribution_pct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DETENIDO | s59 | 24 | 68.298 | 1 | True | az_skewness | 29.7 |
| DETENIDO | s55 | 5 | 39.213 | 1 | True | ax_kurtosis | 31.7 |
| VELOCIDAD MEDIA | s10 | 28 | 34.727 | 1 | True | az_skewness | 38.1 |
| VELOCIDAD MEDIA | s10 | 22 | 28.164 | 1 | True | ax_spectral_entropy | 22.2 |
| DETENIDO | s54 | 29 | 27.844 | 1 | True | ay_kurtosis | 55.5 |
| VELOCIDAD MEDIA | s10 | 34 | 24.648 | 2 | True | ax_spectral_entropy | 22.9 |
| VELOCIDAD ALTA | s11 | 24 | 24.024 | 1 | True | ay_kurtosis | 54.7 |
| VELOCIDAD MEDIA | s10 | 26 | 23.294 | 1 | True | ay_skewness | 35.1 |
| VELOCIDAD MEDIA | s11 | 9 | 23.052 | 1 | True | ay_skewness | 26.6 |
| DETENIDO | s58 | 6 | 22.502 | 1 | True | ax_kurtosis | 36.1 |
| VELOCIDAD MEDIA | s10 | 35 | 22.448 | 2 | True | ax_spectral_entropy | 22.7 |
| DETENIDO | s59 | 3 | 22.165 | 1 | True | ay_kurtosis | 34.4 |

## Conclusión sustentable

Los rechazos no corresponden a un fallo sistemático de clasificación: la DNN acertó 12/12, mientras K-means señaló alejamiento local respecto de los centroides. La predominancia de ventanas aisladas y la concentración de varios rechazos en una sola captura muestran una combinación de eventos locales y variación específica de captura. Las contribuciones se originan en características concretas y no en una discrepancia PC--STM32. Sin etiquetas independientes de anomalía, la evidencia permite denominarlas ventanas atípicas o rechazos de dominio conocido, pero no permite decidir si son fallas reales, transitorios normales o falsas alarmas.
