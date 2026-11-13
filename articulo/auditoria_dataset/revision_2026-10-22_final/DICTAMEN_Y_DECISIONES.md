# Dictamen y decisiones sobre el dataset Arciniegas

Revisión del 22 de octubre de 2026. Este documento interpreta los resultados de
`resumen_auditoria.json` y los CSV de esta carpeta. El procedimiento reproducible
está en `articulo/analisis/auditar_dataset_origenes.py`. No se modificaron señales,
particiones, matrices canónicas, modelos, firmware ni manuscrito. No se entrenó.

## 1. Dictamen ejecutivo

El corpus es legible y el procesamiento conservado es consistente con sus señales.
La evidencia existente necesita **cambios estructurales para sostener independencia
entre adquisiciones**. Las particiones originales separan JSON, pero no sus familias
nominales de origen. Los 37 JSON de prueba pertenecen a cinco familias que también
aparecen en entrenamiento. No se ha demostrado que las 13 familias sean 13 sesiones
físicamente independientes.

Hay evidencia adicional de fragmentación: cada familia conserva un iat y una firma
declarada comunes, existen sufijos de fragmento y se encontraron segmentos RAW
idénticos entre archivos de varias familias. Las firmas no se autenticaron por no
disponer de la clave HMAC. No deben usarse como certificado criptográfico.

Esto **no demuestra una exactitud falsa ni mide un sesgo de magnitud determinada**.
Las métricas describen correctamente las ventanas evaluadas. Su extrapolación a
otras adquisiciones no queda establecida por la separación actual.

## 2. Resultados verificados

| Comprobación | Resultado | Interpretación |
|---|---|---|
| Inventario original | 239 archivos: 235 JSON, 3 info.labels y README | Corpus local completo respecto de sus metadatos |
| Estructura | 235 señales de 300 × 3 valores finitos | No hay JSON inválidos, NaN, Inf ni ejes completamente constantes |
| Canales | ax, ay, az; interval_ms=10 | 100 Hz declarados, no medidos físicamente |
| Unidades | N/A en los tres canales | No es correcto asignar unidades físicas como hecho verificado |
| Dispositivo | Un nombre declarado y tipo DATA_FORWARDER | No prueba por sí solo cuántos sensores físicos se usaron |
| Metadatos | Raíz y subcarpetas consistentes; etiquetas y categorías coincidentes | No hay conflicto de clase detectado |
| Duplicados completos | Ninguno, tanto por archivo como por señal numérica | No hay JSON completos copiados entre bloques |
| Ventanas | 7128 training + 1332 testing | Totales, etiquetas y posiciones correctos |
| Características | 177660 valores recomputados; todos compatibles con float32 | No se detectó corrupción del preprocesamiento guardado |
| Integridad | 491 entradas comprobadas antes y después; cero cambios | La auditoría conservó los archivos examinados |

La comparación de características usa rtol=1e-6 y atol=1e-5. El mayor error absoluto
es 1.5137e-5 en entrenamiento y 7.5928e-6 en prueba; ambos cumplen la tolerancia
combinada de escala y precisión. La igualdad de valores float64 no se exige a un
archivo guardado en float32.

## 3. El agrupamiento relevante está por encima del archivo

| Clase | JSON | Familias nominales | Training / testing |
|---|---:|---:|---:|
| DETENIDO | 67 | 1 | 52 / 15 |
| MEDIA | 42 | 3 | 32 / 10 |
| ALTA | 126 | 9 | 114 / 12 |

Familias compartidas: `51iirpm7`, `51irp429`, `51is1m87`, `51iq0ccr` y `51irbb9k`.
Las otras ocho familias sólo están en entrenamiento. La prueba representa cinco
familias, no 37 procedencias nominales diferentes; tampoco cinco sesiones físicas
verificadas. La independencia entre familias sigue siendo desconocida.

Con la regla conservadora de mantener cada familia completa en un bloque,
DETENIDO no puede aparecer a la vez en entrenamiento, validación y prueba.
Por tanto, **un reparto aleatorio 60/20/20 de JSON no resuelve la limitación**.

Se encontraron 27 diferencias entre el número de fragmento del nombre exportado
y el de `info.labels`: 13 archivos de MEDIA (`51irp429`) y 14 de ALTA (`51iq0ccr`).
No son conflictos de clase. Sí impiden interpretar automáticamente el sufijo del
archivo como posición temporal original. Incluso la numeración de metadata es
una pista de orden, no un timestamp o desplazamiento verificado.

## 4. Repeticiones de señal: qué existe y qué no

La búsqueda recorrió TODAS las posiciones posibles de secuencias triaxiales de
2, 4, 8, 16 y 32 muestras. No se limitó a ventanas alineadas con hop=8.

Se consolidaron **10 segmentos exactos de al menos 8 muestras entre archivos**:
ocho entre archivos de entrenamiento y dos entre archivos de prueba. Todos
pertenecen a la misma familia en cada pareja. Sus longitudes van de 9 a 70 muestras.
Las diez coincidencias tienen forma de sufijo de un archivo y prefijo del otro.
Esto demuestra reutilización parcial de señales entre JSON, aunque no reconstruye
todo el registro original ni prueba su cronología completa.

Ejemplos con posiciones base cero:

| Familia | Archivos exportados | Coincidencia exacta |
|---|---|---|
| ALTA 51iqc0is | s2 y s3, training | 36 muestras: s2[264:300] = s3[0:36] |
| ALTA 51irbb9k | s6 y s7, training | 44 muestras: s6[256:300] = s7[0:44] |
| MEDIA 51irtdov | s14 y s15, training | 70 muestras: s14[230:300] = s15[0:70] |

En el ventaneo efectivamente utilizado aparecen 9 ventanas repetidas adicionales
en entrenamiento: 7128 filas representan 7119 ventanas RAW distintas. En prueba
las 1332 ventanas RAW son distintas. Las nueve repeticiones están contenidas en
tres parejas de archivos; son un fenómeno pequeño en cantidad, pero demuestran
que archivo distinto no implica señal disjunta.

**Entre training y testing no se encontró ninguna secuencia exacta de cuatro o más
muestras triaxiales.** La secuencia máxima coincidente detectada tiene longitud dos,
compatible con coincidencias breves en datos discretizados; no prueba copia.
En particular, no hay ventanas RAW idénticas de 16 muestras entre ambos conjuntos.
Este resultado no descarta misma adquisición, desfases, transformaciones o
dependencia estadística sin muestras idénticas.

## 5. Correlación y solapamiento

Cada pareja de ventanas vecinas de un JSON comparte 8 de sus 16 muestras. Se
calcularon además autocorrelaciones por eje y correlaciones entre características
de ventanas, sin concatenar artificialmente archivos diferentes.

Ejemplos en prueba, mediana del coeficiente por archivo entre ventanas vecinas:

| Clase | Característica | Mediana r |
|---|---|---:|
| DETENIDO | az_RMS | 0.5060 |
| DETENIDO | ax_energía espectral | 0.4724 |
| MEDIA | ay_energía espectral | 0.3248 |
| ALTA | ay_RMS | 0.4083 |
| ALTA | az_RMS | -0.5781 |

La dependencia no tiene que ser siempre positiva: la periodicidad puede producir
coeficientes negativos. Estos resúmenes son descriptivos y no estiman un tamaño
muestral efectivo independiente.

Se examinaron 2468 parejas training/testing de la misma clase a retardo cero y
retardos de -150 a +150 muestras, siempre con al menos 150 muestras comunes.
Se eliminó la media por eje de cada segmento para que la componente DC no
dominara artificialmente el coeficiente. Para ALTA y la misma familia, la mediana
del máximo entre retardos fue 0.8574; para MEDIA, 0.1362; para DETENIDO, 0.1221.
Los máximos también pueden ser altos entre familias diferentes por periodicidad:
no se usan como prueba de copia o parentesco. Buscar muchos retardos aumenta el
máximo por construcción y no se atribuye significación estadística a estos números.

El bootstrap anterior agrupado por JSON considera el solapamiento interno de cada
archivo, pero puede ignorar dependencia entre archivos con la misma procedencia.
No basta con sustituirlo por un bootstrap por familia: DETENIDO tiene una sola,
lo que impide estimar variabilidad entre orígenes para esa clase.

## 6. Qué ocurre con los resultados E1 y E4 existentes

Se reagruparon las predicciones guardadas sin ejecutar modelos:

- Los 28 errores canónicos de E1 se concentran en la familia MEDIA `51irp429`.
- E4 corrige esos 28 casos; su único error está en ALTA `51irbb9k`.
- PC y STM32 mantienen las mismas clases en las 1332 ventanas de ambas instancias.
- Se conservan los 12 rechazos de K-means y sus etiquetas conocidas.

Estos datos localizan los resultados en pocos orígenes; no permiten afirmar que
la ventaja se repita en capturas nuevas. La identidad funcional PC–STM32 y las
mediciones de recursos siguen siendo evidencia válida de las instancias evaluadas.

## 7. Sensibilidad numérica detectada durante la auditoría

Hubo 11 espectros ventana/eje con máximos casi empatados a tolerancia relativa
1e-12. En una ventana de entrenamiento de DETENIDO (archivo s25, ventana 22,
inicio 176, eje ay), los bins 2 y 6 tienen magnitudes empatadas al redondeo.
La forma escalar del preprocesador escoge 12.5 Hz; una forma vectorizada del
chequeo puede escoger 37.5 Hz. La diferencia aparente de 25 Hz desaparece al
conservar la secuencia aritmética escalar del preprocesamiento original.

No es un error de la señal ni se cambió el extractor. Es una advertencia para la
reimplementación numérica: si se fija una tolerancia de desempate en el futuro,
se versionará el contrato PC/STM32 y se regenerarán los derivados correspondientes.
No debe cambiarse silenciosamente la definición en los experimentos históricos.

## 8. Copias y procedencia

El paquete actual `Eduardo_Andres_Ollachica_Arias_Replication_Evidence` incluye los
239 archivos canónicos y todos coinciden por SHA-256. El paquete histórico
`PAQUETE_VALIDACION_REIMPLEMENTACION_STM32` conserva cuatro archivos auxiliares
idénticos, pero no los 235 JSON RAW. Son ausencias, no archivos alterados.

No se volvió a descargar el suplemento oficial durante esta revisión. El chequeo
de copias locales no demuestra por sí solo identidad actual con un servidor externo
ni confirma el alcance de una licencia. La procedencia publicada y las condiciones
de redistribución se documentan por separado.

## 9. Matriz de problemas y decisiones

| ID | Severidad | Hallazgo | Estado/confianza | Decisión |
|---|---|---|---|---|
| D1 | Crítico para generalización | Toda la prueba comparte familias con training | Confirmado/alta | No llamarla prueba entre adquisiciones independientes |
| D2 | Crítico para nueva partición | DETENIDO tiene una sola familia nominal | Confirmado/alta | No generar un train/val/test disjunto por familia con todas las clases |
| D3 | Mayor | Falta relación verificada entre familia, sesión y motor | Evidencia insuficiente/alta | Solicitar originales, offsets y contexto o adquirir sesiones nuevas |
| D4 | Mayor | Segmentos exactos entre JSON y dependencia por origen | Confirmado/alta | Agrupar por procedencia; reconocer límite del bootstrap por JSON |
| D5 | Mayor | Orden temporal no verificable con nombres solamente | Confirmado/alta | No crear una partición temporal hasta aclarar cronología |
| D6 | Mayor | No hay unidades/calibración/RPM verificables | Evidencia insuficiente/alta | Mantener etiquetas suministradas y declarar límites físicos |
| D7 | Menor | Empates de FFT sensibles al orden numérico | Confirmado/alta | Preservar definición histórica y documentar desempate |

## 10. Siguiente paso defendible

Primero aclarar con autores qué representan los 13 identificadores, obtener los
registros sin fragmentar cuando existan, sus offsets y las sesiones de adquisición.
Esto puede resolver la trazabilidad; no garantiza que aparezcan nuevos grupos.

Si se mantiene un único origen para DETENIDO, hacen falta adquisiciones adicionales
para evaluar las tres clases entre orígenes. Deben recogerse sesiones con todas las
clases, conservarlas completas en un conjunto y reservar la prueba antes de elegir
características, hiperparámetros, umbrales o semillas.

Si no es posible ampliar datos, el corpus permite continuar una evaluación interna
y una contribución de eficiencia embebida con límites explícitos. Una partición
temporal sólo se considerará si se verifica el orden; aun así mediría generalización
dentro del registro, no entre sesiones. No se recomienda repetir una batería amplia
de entrenamientos suponiendo que renombrar o repartir JSON elimina esta limitación.

La interpretación del campo iat sigue la documentación oficial de Edge Impulse:
https://docs.edgeimpulse.com/reference/data-ingestion/data-acquisition-format
Define creación del archivo y depende de la fiabilidad del reloj; no ofrece por sí
solo el desplazamiento de cada fragmento. La capacidad de dividir capturas en la
plataforma no demuestra cómo se produjo cada archivo de este corpus.
