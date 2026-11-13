# Auditoría exhaustiva del dataset Arciniegas

Fecha UTC: 2026-10-22T15:19:49.501611+00:00. Auditoría determinista; no se entrenaron modelos,
no se modificaron datos, particiones, firmware ni manuscrito.

## Dictamen ejecutivo

La evaluación de independencia requiere cambios estructurales. Los archivos son
legibles y sus derivados pueden comprobarse, pero separar únicamente por JSON no
separa los identificadores de origen. Existen 13 familias nominales,
5 compartidas entre entrenamiento y prueba. Los
37 archivos de prueba pertenecen a familias
también presentes en entrenamiento.

Esto confirma procedencia nominal compartida, no demuestra por sí solo que cada
familia sea una sesión física independiente ni cuantifica la inflación de métricas.
Los sufijos sN, el mismo iat y la misma firma declarada por familia son compatibles
con fragmentación; no son firmas autenticadas en esta auditoría. No hay
marcas de tiempo por muestra ni desplazamientos verificados del registro fuente.
El iat es una fecha de creación declarada por el formato; no sustituye esos datos.

## Inventario y grupos

| Clase | JSON | Familias de origen | Training | Testing |
|---|---:|---:|---:|---:|
| DETENIDO | 67 | 1 | 52 | 15 |
| VELOCIDAD MEDIA | 42 | 3 | 32 | 10 |
| VELOCIDAD ALTA | 126 | 9 | 114 | 12 |

Cada JSON contiene 300 ternas finitas, interval_ms=10, canales ax/ay/az y 3 s de
duración nominal (2.99 s entre primera y última muestra). La frecuencia configurada
es 100 Hz; no se verificó físicamente el reloj. Se examinaron los 239 archivos del
corpus (235 JSON, tres info.labels y README) y las matrices/metadata procesadas.

| Clase | ID de origen | Training | Testing | Índices renumerados |
|---|---|---:|---:|---:|
| DETENIDO | `51iirpm7` | 52 | 15 | 0 |
| VELOCIDAD MEDIA | `51irp429` | 8 | 5 | 13 |
| VELOCIDAD MEDIA | `51irtdov` | 15 | 0 | 0 |
| VELOCIDAD MEDIA | `51is1m87` | 9 | 5 | 0 |
| VELOCIDAD ALTA | `51iq0ccr` | 8 | 6 | 14 |
| VELOCIDAD ALTA | `51iq3k4u` | 14 | 0 | 0 |
| VELOCIDAD ALTA | `51iq79ju` | 14 | 0 | 0 |
| VELOCIDAD ALTA | `51iqc0is` | 14 | 0 | 0 |
| VELOCIDAD ALTA | `51iqgbfk` | 15 | 0 | 0 |
| VELOCIDAD ALTA | `51ir0h7i` | 14 | 0 | 0 |
| VELOCIDAD ALTA | `51ir4ph4` | 13 | 0 | 0 |
| VELOCIDAD ALTA | `51ir832t` | 14 | 0 | 0 |
| VELOCIDAD ALTA | `51irbb9k` | 8 | 6 | 0 |

El identificador no equivale automáticamente a motor, montaje, día ni sesión.
Las cantidades de familias son exactas; su interpretación física requiere evidencia.
Con el agrupamiento conservador por origen, DETENIDO sólo tiene un grupo: no puede
aparecer en tres bloques disjuntos por grupo. No es válido prometer una partición
estratificada por origen de tres conjuntos para las tres clases con estos datos.

## Integridad, etiquetas y metadatos

Se contrastaron info.labels raíz, training y testing, existencia de archivos,
categoría, etiqueta, ID y nombre. Incidencias estructurales: 0.
La relación íntegra figura en incidencias.csv. Índices de fragmento distintos
entre nombre de archivo y nombre original en metadata: 27.
Se conservan ambos en inventario_archivos.csv; no se renombró ningún archivo.
Ordenar archivos alfabéticamente o por el sufijo exportado no reconstruye el tiempo.
El índice original de metadata es una pista de orden, tampoco un timestamp verificado.

Unidades declaradas: N/A|N/A|N/A. Tipo de dispositivo:
DATA_FORWARDER. No se infieren unidades físicas, rango,
calibración, orientación, RPM, fallas, filtros del sensor o condiciones del montaje.
Se inventariaron firmas, pero no se verificó HMAC: no se dispone de la clave.

## Duplicación completa y parcial

Grupos de duplicados completos (bytes de archivo o señal numérica):
0. Se buscaron secuencias triaxiales exactas en
TODAS las posiciones, no sólo en los saltos del ventaneo de entrenamiento.

| Longitud de secuencia | Posiciones examinadas | Pares entre archivos | Pares train/test |
|---|---:|---:|---:|
| 2 | 70265 | 375 | 12 |
| 4 | 69795 | 308 | 0 |
| 8 | 68855 | 264 | 0 |
| 16 | 66975 | 190 | 0 |
| 32 | 63215 | 74 | 0 |

Los conteos de la tabla corresponden a parejas de posiciones; varios corresponden
al mismo segmento largo y no deben contarse como adquisiciones independientes.
Se consolidaron 10 segmentos máximos de al menos ocho muestras en
segmentos_exactos_repetidos.csv. Las repeticiones menores también se conservan,
pero pueden ser coincidencias por discretización. Las ventanas RAW únicas que
realmente entran en el modelo son: {'training': {'windows': 7128, 'unique_raw_windows': 7119, 'repeated_excess_windows': 9}, 'testing': {'windows': 1332, 'unique_raw_windows': 1332, 'repeated_excess_windows': 0}}.

No encontrar bloques exactos no descarta registros contiguos, misma sesión,
transformaciones numéricas o similitud estadística. Las coincidencias de una sola
terna, frecuentes con valores discretizados, no se clasifican como fuga de ventanas.

## Ventaneo y derivados

N=16, hop=8: 36 ventanas por JSON, 7128 training y 1332 testing. Dentro de cada
archivo, 35 parejas vecinas comparten ocho muestras. Se cubren 296 de las 300
muestras; las últimas cuatro no entran en ninguna ventana. Ventanas sin solapamiento
(hop=16): 18 por archivo, 3564 training y 666 testing; no se generaron nuevos datasets.
Los tiempos de metadata son locales al JSON, no tiempos absolutos del registro padre.

Se recalcularon las 21 características mediante fórmulas independientes usando
NumPy, incluyendo correcciones de asimetría/curtosis, RFFT, DC y entropía. Se
contrastaron valores, orden, etiquetas, coordenadas, nombres y matrices NPY/CSV.
Resultados detallados: resumen_auditoria.json y verificacion_caracteristicas.csv.
Tolerancia de comparación NPY: rtol=1e-6 y atol=1e-5, por almacenamiento float32.
Un error absoluto aislado no debe interpretarse sin escala y tolerancia.

Se encontraron 11 espectros de ventana/eje con más de un
máximo prácticamente empatado (tolerancia relativa 1e-12). En
1 casos, calcular medias/FFT en
lotes en lugar de hacerlo por ventana/eje cambia el argmax por redondeo. La
comparación final conserva la secuencia escalar del preprocesador original. Esto
es sensibilidad numérica del extractor, no evidencia de corrupción del dataset.
Detalles: empates_numericos_fft.csv. No se alteró el extractor original.

## Dependencia y correlación

Se calcularon autocorrelaciones de cada eje a retardos 1,2,4,8,16,32,64 muestras y
correlaciones de características a retardos 1,2,4 ventanas, separadas por archivo.
Los resúmenes agregan coeficientes por clase/split; no mezclan finales e inicios
de archivos ni ofrecen p-valores bajo independencia ficticia. Correlaciones
indefinidas por señal constante se registran vacías.

También se compararon todas las parejas train/test de la misma clase a retardo cero
y con retardos de -150 a +150 muestras (al menos 150 muestras comunes). Se restó
la media de cada eje del segmento comparado antes de la correlación triaxial. Esto evita que DC/
gravedad domine artificialmente el coeficiente. Una correlación pequeña a retardo
cero no demuestra independencia: un desfase, periodicidad o segmentos distintos
de una misma captura pueden dar valores pequeños. Una correlación alta tampoco
prueba copia. No se usaron estas correlaciones para escoger una nueva partición.

Las fronteras entre fragmentos consecutivos por índice de metadata se inspeccionan
en fronteras_candidatas.csv. Saltos de amplitud y continuidad plausible no prueban
cronología, duración de huecos ni ausencia de solapamiento original.

## Consecuencias para las métricas existentes

Se agruparon las predicciones ya guardadas de E1/E4 por origen, sin inferencia nueva
ni reentrenamiento: resultados_existentes_por_origen.csv. Las exactitudes siguen
siendo descripciones de las ventanas evaluadas. No prueban generalización a
adquisiciones independientes. El bootstrap anterior por JSON considera parte de la
dependencia, pero no dependencia adicional entre JSON con un origen compartido.
McNemar por ventana tampoco resuelve ese agrupamiento. No se calculó un nuevo
p-valor ni un intervalo que pretenda resolver la falta de grupos de DETENIDO.

## Decisiones recomendadas

1. Conservar la partición original como referencia histórica de reimplementación.
2. Solicitar/aclarar registro padre, orden, offsets, duración y sesiones con autores.
3. Para una prueba por adquisiciones nuevas, obtener sesiones adicionales de todas
   las clases, reservarlas antes de seleccionar el modelo y mantener la sesión junta.
4. Si sólo se dispone de este corpus, limitar conclusiones a evaluación interna;
   una separación temporal con exclusión sólo es planteable si se verifica el orden.
5. Mantener el 50% de solapamiento si responde al funcionamiento del sistema, pero
   calcular incertidumbre con la unidad experimental apropiada y suficientes grupos.
6. No iniciar una gran batería de entrenamientos hasta cerrar la viabilidad de grupos.

## Reproducibilidad y límites de cobertura

La auditoría comprobó las copias locales de los dos paquetes cuando existen;
verificacion_copias.csv distingue hashes iguales, ausencias y archivos adicionales.
- `Eduardo_Andres_Ollachica_Arias_Replication_Evidence/03_Datasets/Arciniegas_Raw`: 239 idénticos, 0 ausentes, 0 distintos y 0 adicionales.
- `PAQUETE_VALIDACION_REIMPLEMENTACION_STM32/datos/arciniegas/raw`: 4 idénticos, 235 ausentes, 0 distintos y 0 adicionales.
Una ausencia no es un archivo modificado: el paquete histórico que no contiene
señales RAW no debe confundirse con el paquete de evidencia actual completo.
No se descargó de nuevo el suplemento oficial en esta ejecución: la comparación
local no sustituye una nueva verificación con el servidor del editor.

Las entradas se verificaron por SHA-256 antes y después. Archivos modificados:
0. Comando de reproducción desde la raíz:

```powershell
python articulo/analisis/auditar_dataset_origenes.py --output articulo/auditoria_dataset/otra_ejecucion
```

El directorio de salida debe ser nuevo; sólo se necesita NumPy. Se preservan
manuscrito, modelos, firmware y matrices canónicas. La auditoría no verifica la
realidad física de las etiquetas, autenticidad criptográfica, calibración ni
independencia entre días/motores que el corpus no documenta.
