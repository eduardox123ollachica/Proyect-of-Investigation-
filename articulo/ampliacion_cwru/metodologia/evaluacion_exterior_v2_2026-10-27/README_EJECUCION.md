# Evaluación exterior controlada de CWRU V2

Este ejecutor evalúa exclusivamente los modelos y parámetros del cierre
`cierre_reajustes_finales_v2_2026-10-27`. No entrena, ajusta scalers, convierte
modelos, recalibra INT8 o modifica umbrales. No repite la búsqueda interna
ni los 131 reajustes finales.

La separación exterior es por carga: cada modelo se evalúa con todos los
registros de la carga excluida de su ajuste. Las cuatro decisiones y todos
los modelos quedaron fijados antes de esta etapa. El corpus se exploró en
fases anteriores; la evaluación sigue siendo retrospectiva, no una prueba
ciega con otro motor. Las ventanas solapadas y las semillas no son
adquisiciones físicas independientes.

## Ámbitos y operaciones

| Carga reservada (HP) | Ventanas | MAT | DNN únicas | RF | SVM | Detector |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 175 | 7 | 20 | 10 | 1 | 2 |
| 1 | 393 | 7 | 20 | 10 | 1 | 2 |
| 2 | 406 | 7 | 20 | 10 | 1 | 2 |
| 3 | 406 | 7 | 20 | 10 | 1 | 1 |

Se reutilizan los modelos compartidos entre línea base F14, selector por
rendimiento y selector por costo total. Son 55 200 invocaciones Lite en lote
1 (27 600 FP32 y 27 600 INT8), 13 800 predicciones fila RF, 1380 SVM y 2354
scores fila del detector dentro de los workers. Los alias se expanden después
para los resúmenes; no se repite su inferencia ni se crean ensembles.
Las reconstrucciones numéricas durante la verificación se cuentan como
comprobaciones técnicas adicionales, no como experimentos independientes.

Se usan las estadísticas float32 guardadas en cada scaler y se conservan
resta/división, cuantización nearest-even, clipping y desempate de argmax
establecidos. SVM utiliza su predicción nativa. El detector utiliza su scaler,
centroide y umbral guardados, rechaza solo si score > umbral y no sustituye
la clase de la DNN. Los métodos de entrenamiento y conversión están bloqueados
en el proceso evaluador.

Cada worker conserva un snapshot de las filas permitidas, predicciones,
probabilidades o scores, IDs globales, etiquetas, MAT y parámetros de
cuantización. Los modelos/scalers/umbrales y fuentes se autentican antes y
después. El loader materializa únicamente la carga autorizada; el hash/mmap
del almacenamiento histórico compartido no equivale a una traza de accesos
del sistema operativo. No se vuelven a abrir señales MAT.

## Métricas y publicación del resumen

Se guardan matrices de confusión y precision/recall/F1 por clase, carga y
semilla. Por MAT se informa el recall de su clase verdadera; no se presenta
macro-F1 de un MAT monoclase como métrica primaria. El detector se resume
con especificidad normal, recall IR/OR y balanced accuracy, primero por MAT
y luego por carga con pesos iguales.

El resumen supervisado pondera igualmente las cuatro cargas y las diez
semillas cuando corresponden. SVM tiene una evaluación determinista por carga
y el detector no se replica por semilla. Se incluyen diferencias pareadas
entre procedimientos y variación descriptiva, sin intervalos IID sobre
ventanas ni afirmaciones de no inferioridad estadística.

La comparación FP32–INT8 conserva cambios de clase por ID/MAT, clipping por
característica/ventana, valores en extremos y empates. Un valor -128 o 127 no
se equipara automáticamente a clipping. No se cambian escalas, umbrales,
épocas o semillas a partir de estos resultados.

El resumen `RESULTADOS_EXTERIORES.json` se genera solamente cuando las cuatro
cargas y sus verificaciones aprueban. Los archivos intermedios permanecen
como evidencias; no se utilizan para cambiar reglas. Un fallo conserva el
intento completo y bloquea reintentos automáticos. Se requiere revisar cualquier
recuperación técnica antes de volver a ejecutar.

Las banderas false de documentos históricos se conservan. El nuevo plan
habilita únicamente la evaluación exterior operacional, después del cierre
auditado de selección y reajustes; no habilita entrenamiento o redacción.
El respaldo F14 del escenario 3 y la limitación interna de especificidad del
detector permanecen documentados, cualquiera que sea el resultado exterior.

## Ejecución personal

Ejecutar en PowerShell desde cualquier directorio:

```powershell
$evaluacion = "C:\Users\LENOVO\Desktop\Proyecto Investigación\codigo\cwru\ampliacion_v2\evaluacion_exterior\ejecutar_evaluacion.py"

& "C:\Users\LENOVO\AppData\Local\Microsoft\WindowsApps\python3.12.exe" -B $evaluacion --run
```

El comando recorre las cuatro cargas secuencialmente. Orientación provisional:
5–15 minutos, sujeta al arranque TensorFlow y a las verificaciones; no existe
todavía una medición del recorrido completo. Los modelos son pequeños, pero
se cargan e invocan todos los modelos previstos.

Cada worker tiene watchdog de 300 s y la campaña dispone de 1800 s acumulados
para ejecución/verificación por carga. No inicia otro worker sin su timeout
completo restante. El preflight compartido, el cierre/terminación y la
verificación global final pueden añadir tiempo calendario: los 1800 s no
son un límite duro de todo el tiempo transcurrido. No se amplía el presupuesto
automáticamente.

La campaña se guarda en:
`C:\Users\LENOVO\Desktop\Proyecto Investigación\resultados\cwru\ampliacion_v2\evaluacion_exterior_v2_2026-10-27`.
La consola muestra el directorio exacto. No iniciar comandos simultáneos,
borrar locks o repetir un intento iniciado. Si aparece FAIL o timeout,
conservar la salida y la ruta para revisar la causa técnica.

La salida final esperada es:

```text
OUTER_EVALUATION_COMPLETED: 4/4 cargas.
Evidencias: ...
```

Se generarán `RESULTADOS_EXTERIORES.json`, `RESUMEN_CAMPANA.json` y el
manifiesto global, además de snapshots, predicciones y manifiestos por carga.
Estos son resultados de PC. Después quedarán la auditoría de métricas y las
validaciones STM32 de los modelos canónicos; no se presentan los tiempos PC
o los tamaños de archivo TFLite como latencia o Flash del firmware.
