# Reevaluación del solapamiento CWRU — 22-10-2026

## Resultado y alcance

Quitar el solapamiento evita repetir muestras de salida entre ventanas adyacentes,
pero no demuestra independencia estadística ni crea motores/rodamientos nuevos.
La separación entre archivos/cargas debe seguir realizándose antes del ventaneo.
El 50% actual no produce por sí mismo fuga entre train/test: sus MAT ya son disjuntos.
El cambio se justifica por menor redundancia y claridad de evaluación, no para
reparar una mezcla entre conjuntos que esta auditoría no detectó.

Recuentos exactos desde longitudes válidas a 12 kHz; N=4096, mismo inicio y FIR:

| Solapamiento | Hop | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|---:|
| 50 % | 2048 | 568 | 406 | 406 | 1380 |
| 25 % | 3072 | 382 | 273 | 273 | 928 |
| 0 % | 4096 | 285 | 203 | 203 | 691 |
| 0 % + margen44 | 4140 | 284 | 203 | 203 | 690 |

El perfil 0% se comprobó como selección 0,2,4... de las ventanas existentes de
cada archivo, sin generar arrays nuevos ni entrenar. El registro174 pasa de6 a3
ventanas: quitar redundancia también disminuye ejemplos, sin alterar el número
de adquisiciones. No ajustar N sólo para inflar el recuento sin justificación física.

## Correlación descriptiva, no certificado de independencia

Se reutilizaron las14 características guardadas, verificadas en la auditoría,
y se calculó Pearson de cada característica entre ventanas consecutivas dentro
del mismo MAT. Con 0% el desplazamiento físico se duplica; la comparación cambia
simultáneamente coincidencia de muestras y separación temporal.

Se guardaron todas las estimaciones, pero los resúmenes exigen >=10 pares por
archivo para reducir la inestabilidad de secuencias muy cortas; ese umbral es una
regla descriptiva, no garantía de precisión. Cada archivo pesa igual; no se
concatenaron archivos ni se trataron ventanas como réplicas físicas independientes.
No hay p-valores, selección por accuracy, ni un supuesto tamaño muestral efectivo.
Correlación cero tampoco demostraría independencia o ausencia de parentesco.

Mediana de las14 medianas por característica de |r|: 50% = 0.5267;
0% = 0.2414. Es un resumen exploratorio, no una métrica universal
del dataset: revisar `autocorrelacion_resumen.csv` por característica y los casos
individuales, que pueden subir o bajar por periodicidad/variabilidad finita.

## Matiz del filtro

Con el FIR177 aplicado antes de diezmar por4, ventanas de salida sin solapamiento
todavía pueden depender de173 muestras originales compartidas en sus soportes.
Un margen de44 muestras a12 kHz elimina ese cruce de soporte para este FIR finito.
No elimina la correlación de la maquinaria ni la dependencia de un mismo registro.
Es un cálculo del soporte, no una prueba de ruido blanco ni una recomendación de
usar44 como distancia de decorrelación. Entre MAT distintos no se comparte filtro.
El perfil con margen sólo tiene recuentos teóricos; no se extrajeron sus características.

## Propuesta, aún no adoptada ni ejecutada

Para una referencia principal sencilla, evaluar0% en train/validation/test y
reservar50% como sensibilidad predefinida, conservando los mismos grupos por carga,
semillas y reglas de ajuste. Otra posibilidad es entrenamiento50% y evaluación0%,
pero respondería una pregunta distinta y debe documentarse. No escoger el esquema
según cuál dé mayor métrica de prueba. Con0% hay691 ventanas, no691 ensayos nuevos.

El tiempo de una ventana sigue siendo341.333ms. La cadencia pasa de170.667ms a
341.333ms; esto no duplica el tiempo de inferencia de una ventana. La evaluación
de muestras no solapadas puede coexistir con operación desplegada a50% si se
declara y se evalúa separadamente la cadencia de decisiones.

No recomendaría repartir aleatoriamente ventanas sin solapamiento del mismo MAT
entre train/test para aumentar datos. No resuelve dependencia de fuente y reduciría
el alcance de la prueba a segmentos de registros ya representados en entrenamiento.

## Fundamento y limitaciones externas

Vieira et al., *Towards a more realistic evaluation of machine learning models
for bearing fault diagnosis*, MSSP258 (2026),114640,
https://doi.org/10.1016/j.ymssp.2026.114640 (texto de autor: https://arxiv.org/abs/2509.22267),
contrasta separación por segmentos/condiciones con separación por rodamiento.
Respalda distinguir control temporal de independencia física; no establece
los recuentos ni los valores de correlación locales, calculados aquí.
FIR/decimación: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.upfirdn.html.

## Validación y reproducción

32 comprobaciones aprobadas. Entradas intactas, sin regeneración de datos,
sin entrenamiento ni cambios de manuscrito/protocolo. `entradas_sha256.csv` fija
código y archivos usados; `MANIFEST_SHA256.csv` sella las salidas excepto a sí mismo.
Cada repetición exige un directorio nuevo:

```powershell
python -B codigo/cwru/ampliacion_v2/analisis/evaluar_solapamiento_cwru.py --output articulo/ampliacion_cwru/analisis_solapamiento/repeticion
```

Se requiere NumPy. En esta ejecución se usó el intérprete/runtime y los paquetes
científicos ya instalados registrados en la auditoría principal; no se instaló nada.
La adopción de0% o un margen queda pendiente: este informe no sustituye la decisión
del usuario ni modifica las referencias históricas V1.
