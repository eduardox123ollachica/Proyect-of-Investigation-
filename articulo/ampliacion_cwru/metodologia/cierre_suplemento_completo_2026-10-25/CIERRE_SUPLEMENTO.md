# Cierre del suplemento de desarrollo

## Dictamen

Ejecución y auditoría técnica completas:34DNNnuevas,2referencias importadas y
bloque detector con5ajustes nuevos y1control importado. Sin fallos o reintentos.
Tiempo acumulado: 729.976224s (12.1663min).
Se verificaron sellos, firmas, roles, parámetros train-only, checkpoints,
modelos, cuantización reconstruida, métricas, P95 y decisiones guardadas.
Ninguna variante P1 alcanza la ganancia prefijada; MOM4 tampoco supera sus
guardias. Los resultados negativos se conservan íntegros.

## Clasificador: media igual de tres cargas y tres semillas

| Máscara | F1 INT8 P0 | F1 INT8 P1 | Diferencia en puntos porcentuales | Criterio previo |
|---|---:|---:|---:|---|
| F14 | 0.844429 | 0.830161 | -1.427 | No cumple |
| NF12 | 0.915423 | 0.922665 | +0.724 | No cumple |

F14 pierde rendimiento medio y más de0.01en la carga0HP. NF12 preserva las
cargas, pero su ganancia0.007242es inferior a0.01. En0HP los deltas de NF12
para semillas11/22/33 son−0.113086/−0.030800/+0.207146: la media no representa
una mejora consistente. Se reportan los nueve pares, sin seleccionar semillas.
No se cambia el criterio después de observar que otro secundario resulta favorable.

Por máscara, recortes672→201sobre las mismas evaluaciones: F14 tiene40908
valores y NF1235064. Ventanas afectadas279→180sobre2922evaluaciones por máscara.
Son conteos repetidos por modelos/semillas, no muestras físicas independientes.
F14 cambia71→6clases por cuantización y NF1270→56; reducir recortes o cambios
no garantiza una clasificación mejor. F14 mantiene recallN0=0en ambos modos.

## Detector: compromisos entre aceptación sana y sensibilidad

| Validación HP | Detector | Especificidad N | Recall IR | Recall OR | Guardia>=0.95 |
|---|---|---:|---:|---:|---|
| 0 | CONTROL12 | 0.000000 | 1.000000 | 1.000000 | No |
| 0 | MOM4 | 0.428571 | 1.000000 | 0.833333 | No |
| 1 | CONTROL12 | 1.000000 | 1.000000 | 1.000000 | Sí |
| 1 | MOM4 | 0.982759 | 1.000000 | 0.793103 | No |
| 2 | CONTROL12 | 0.000000 | 1.000000 | 1.000000 | No |
| 2 | MOM4 | 0.741379 | 1.000000 | 0.816092 | No |

MOM4 mejora especificidad media0.333333→0.717570yBA0.666667→0.812329,
pero reduce recallOR1.0→0.814176. Falla la guardia en las tres cargas.
No se adopta automáticamente ni se presenta como confirmación fiable de falla.
CONTROL12también tiene limitaciones fuertes: falla especificidad sana en0y2HP.
No se elevaron umbrales utilizando las normales rechazadas.

## Épocas, tiempo y continuidad

19/36alcanzaron50épocas;17pararon antes. Los12casos de validación0HP
restauran épocas1–3. Estos conteos incluyen las dos referencias importadas.
El bloque detector tardó8.274612s. El segundo lote657.392384s.
La proyección lineal de40min del primer arranque fue conservadora: el total
real de procesos y auditorías del padre fue12min9.976s, sin pausas humanas.
La auditoría actual no se suma como si fuera tiempo de entrenamiento.
Se conserva P0 como referencia; el suplemento no selecciona una máscara final.
No se promueve P1 o MOM4 incumpliendo el diseño fijado. Cualquier variante
posterior exige otra justificación y presupuesto, sin continuar ajustando a
estos mismos resultados como si fuesen una prueba independiente.
El plan principal sigue pendiente de calibración del costo total STM32 y de la
búsqueda completa de máscaras/cargas/semillas. Las32referencias originales
permanecen disponibles. No mezclar costos de firmware y métricas de versiones distintas.
La rama de normalidad debe reportarse con su especificidad y sus limitaciones;
la evidencia actual no respalda interpretarla como verificador fiable de falla.
No se alteraron manuscrito, firmware, datos, particiones o modelos anteriores.

## Archivos y alcance

CIERRE_SUPLEMENTO.json contiene resultados y trazabilidad. CincoCSV permiten
ver pares, cargas, MAT y detector. Este bloque es desarrollo retrospectivo
con outer3excluido de nuevas predicciones; no es evaluación externa final.
No se calculan p-valores o intervalosIID sobre ventanas solapadas.
Comprobación sin nuevos entrenamientos, forwardDNN o invocacionesTFLite.
