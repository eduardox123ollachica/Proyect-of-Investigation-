# Ensayo de desarrollo: compresión de magnitudes y detector de momentos

Se prepara un suplemento acotado después del diagnóstico de los 32 representantes.
Su objetivo es probar dos hipótesis separadas, con controles y sin sustituir el
protocolo principal antes de conocer los resultados. Es desarrollo retrospectivo;
no se presenta como una confirmación independiente posterior al diagnóstico.

## Clasificador: P0 frente a P1

P0 conserva el preprocesamiento original. P1 comprime RMS, energía espectral y
amplitud dominante de DE/FE (índices 0,3,5,7,10,12) antes de StandardScaler:

`u_j = log1p(x_j / a_j)`

`a_j` es la mediana de los valores estrictamente positivos de la columna de
entrenamiento correspondiente, calculada en float64 después del cast inicial
float32 y exportada a float32. Para una columna de entrenamiento totalmente
nula, `a_j=1`, dejando constancia. Cero es válido; negativos, no finitos,
desbordamientos o escalas inválidas detienen el proceso. No se buscan epsilons,
se toma valor absoluto o se recortan datos para evitar errores.

División y log1p se materializan en float32. Las otras ocho características se
conservan exactamente. Después se ajusta StandardScaler solo con entrenamiento
transformado, y la calibración INT8 utiliza todas esas filas en orden global.
La compresión es monotónica y sublineal, pero no acota el rango. Es una hipótesis
de ingeniería; no garantiza corregir 0 HP o la pérdida INT8.

Se comparan F14/NF12, semillas 11/22/33 y las tres validaciones internas del
desarrollo 0/1/2 HP. No se calculan predicciones nuevas para la carga externa
3 HP ni se abre su caché histórico test. Ventanas y MAT conservan sus roles.

| Validación interna | Cargas de ajuste | Ventanas ajuste / validación |
|---|---|---:|
| 0 HP | 1 y 2 HP | 799 / 175 |
| 1 HP | 0 y 2 HP | 581 / 393 |
| 2 HP | 0 y 1 HP | 568 / 406 |

Son **36 celdas**. Dos P0 (F14/NF12, inner0, seed11) se importan desde los
representantes originales con su firma y sellos intactos. Se comprueban IDs,
orden, máscara, semilla, roles, fuente de entrenamiento y versiones. No se
renombran como ajustes nuevos. Quedan **34 entrenamientos y 68 conversiones**.

Se mantiene la arquitectura, optimizador, pesos de clase, máximo 50 épocas y
ES con paciencia 10 y restauración del mejor checkpoint. El núcleo de
entrenamiento/exportación se importa de la fuente sellada ya verificada.

La comparación será pareada: nueve puntuaciones por máscara (tres cargas por
tres semillas), sin promediar probabilidades. Se reportan F1 macro INT8/FP32,
recall por MAT, diferencias por cuantización, fracción de valores y ventanas
recortados, épocas y tiempos PC. Como criterio de ingeniería para considerar
P1 prometedor se fija ganancia media INT8 de al menos 0.01 y ausencia de una
pérdida media por carga mayor de 0.01. No es una prueba estadística de no
inferioridad ni selecciona automáticamente máscara o preprocesamiento.

## Detector: CONTROL12 frente a MOM4

El bloque secundario usa las características originales, con su propio scaler
normal-only. P1 de la DNN no se aplica al detector. Se comparan:

- CONTROL12: detector V2 original, sin frecuencias dominantes.
- MOM4: asimetría y curtosis de DE/FE, índices 1,2,8,9.

Los momentos estandarizados son invariantes a offset y ganancia positiva de
la señal bajo sus definiciones; la asimetría cambia de signo ante inversión
de polaridad. No se promete invariancia a espectro, ruido o montaje. Entropía
también es invariante a amplitud: se excluye para delimitar un control temporal
sin FFT, no por carecer de esa propiedad. Se pierde información de amplitud
que puede ser útil para detectar fallas, y un MAT normal puede seguir siendo
insuficiente para caracterizar la variación sana.

Se conservan K=1, roles normales fit/calibration disjuntos, P95 y rechazo
estricto sin refit. Son seis casos, sin repetir por semillas DNN: se importa
un CONTROL12 existente y se hacen cinco ajustes nuevos.

Se reportan especificidad N, recalls IR/OR y BA por MAT/carga. Se fija como
guardia descriptiva que N/IR/OR alcancen cada uno 0.95 en cada carga interna.
Un incumplimiento es resultado predictivo negativo, no fallo de ejecución.
No se reajusta un umbral con las normales rechazadas ni se sustituye el
selector principal. MOM4 está compartido con F14/NF12, pero no con todas las
ocho máscaras originales: una adopción futura requiere revisar la unión de
características y medir de nuevo el costo total en STM32.

## Primer comando: un entrenamiento nuevo

Desde cualquier carpeta, en PowerShell:

```powershell
& "C:\Users\LENOVO\AppData\Local\Microsoft\WindowsApps\python3.12.exe" -B "C:\Users\LENOVO\Desktop\Proyecto Investigación\codigo\cwru\ampliacion_v2\suplemento\ejecutar_suplemento.py" --execute --limit 1
```

Esto registra las dos referencias y ejecuta solo
`supp_o3_i0_F14_P1_s11`: F14 con compresión, semilla 11, ajuste 1/2 HP y
validación 0 HP. El bloque de detector todavía no se ejecuta. Se espera:

`BATCH_COMPLETED: 1/34 ajustes nuevos; 3/36 celdas disponibles (2 importadas).`

El contador 3 incluye las dos referencias. Una vez auditado este primer
resultado se entregará el comando con `--resume` y la ruta real para los
33 entrenamientos restantes. El bloque de detector se ejecutará después,
en un proceso separado. Un primer resultado técnicamente válido no se usa
para cambiar el diseño o sus parámetros según su F1.

Salida: `resultados/cwru/ampliacion_v2/suplemento_preprocesamiento_normalidad_2026-10-25/<fecha_UTC>/`.
La consola imprime esa ruta. Hay logs por proceso en `lotes/<fecha_UTC>/` y
`progreso.jsonl`/`historial.jsonl` por modelo, con checkpoint previo a conversión,
medianas P1, scaler, parámetros de cuantización, salidas y métricas.

## Duración, recuperación y comprobaciones

Primer modelo: normalmente decenas de segundos; su worker tiene límite 300 s,
más verificación. Para el bloque completo, reservar orientativamente 15–30 min.
La referencia lineal de la campaña anterior ronda ocho minutos, pero aquí hay
auditorías adicionales y pueden variar los arranques de TensorFlow. Son
proyecciones, no duraciones medidas de este suplemento. Un arranque frío puede
alargarlo sustancialmente. No se cambian épocas para cumplir una estimación.

Topes: 300 s por DNN, 60 s para el bloque de detector, 7200 s acumulados por
campaña. El límite acumulado puede detener la campaña antes de agotar todos
los topes individuales. No incluye pausas humanas, adquisición o flasheo.

No hay reintentos automáticos, force o sobrescritura. Se detiene la cola ante
fallo/timeout y se conservan los parciales. Ctrl+C detiene el hijo activo. Un
fallo después de las 34 DNN requiere recuperación técnica separada del detector;
no volver a entrenar las redes para repetir ese bloque. No se reanuda desde
una época intermedia: los checkpoints de pesos no restauran optimizador/RNG.

La preparación aprobó 44 pruebas en Python 3.12.10 del proyecto, sin ajustes
CWRU ni TensorFlow importado. Cubren roles, lectura selectiva, transformación,
casos límite, importación, firma, preservación y timeouts simulados. Se reutiliza
el núcleo TensorFlow sellado probado en los 32; el primer comando verifica su
integración real con P1 antes del resto. Dos revisiones independientes no
identificaron problemas bloqueantes. Fuentes y documentos se sellan antes de
que el usuario inicie el suplemento.

El manuscrito, firmware, datos, particiones y los 32 modelos se conservan. El
ensayo 50/100 no forma parte de este bloque. No se habilita la evaluación
externa, búsqueda de costo o una conclusión final de publicabilidad.

## Fundamento

La elección de magnitudes procede del diagnóstico local: 93.40% de los recortes
se concentran en RMS/energía/amplitud FE. P1 mantiene las magnitudes DE como
tratamiento simétrico predefinido, sin elegir un subconjunto según futuros
aciertos. Las frecuencias siguen iguales; por eso F14/NF12 son controles útiles.

- [NumPy log1p](https://numpy.org/doc/stable/reference/generated/numpy.log1p.html): definición y cálculo de log(1+x). No se actualizan las versiones instaladas por consultar documentación estable más reciente.
- [SciPy asimetría](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.skew.html) y [curtosis](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kurtosis.html): razones de momentos centrales que fundamentan el control MOM4.
- [LiteRT: cuantización](https://developers.google.com/edge/litert/conversion/tensorflow/quantization/post_training_integer_quant): calibración de rangos con datos representativos y cuantización de pesos/activaciones.
- Diseño exacto y límites: `METODOLOGIA.json`; estado de ejecución: `PLAN_SUPLEMENTO.json`.
