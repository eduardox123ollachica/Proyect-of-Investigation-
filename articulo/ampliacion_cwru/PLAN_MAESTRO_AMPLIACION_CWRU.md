# Plan maestro de ampliación CWRU

Fecha: 22 de octubre de 2026.
Estado: propuesta detallada sustentada en el inventario actual. No es un protocolo
congelado ni autoriza por sí mismo una ejecución masiva. En esta etapa no se
reentrenó, no se alteraron particiones y no se modificó el manuscrito.

## 1. Objetivo y alcance científico

El artículo ampliado estudiará un procedimiento de selección de características
que considera simultáneamente rendimiento predictivo y costo real de procesamiento
en STM32F446RE. La reimplementación de Arciniegas será el antecedente técnico; CWRU
será el banco principal de la ampliación. Incorporar CWRU sin una comparación
metodológica nueva no se considerará por sí solo una contribución suficiente.

Objetivo de trabajo:

> Determinar si seleccionar representaciones considerando operaciones compartidas
> y costo embebido permite reducir el tiempo total del pipeline respecto de la
> representación completa y de una selección guiada sólo por rendimiento.

Se deben responder tres cuestiones mediante experimentos: cuánto se ahorra,
qué variación predictiva se introduce y si el ahorro persiste al incluir el
detector auxiliar. No se anticipará una mejora antes de medirla. El procedimiento
propuesto tiene antecedentes y debe delimitarse frente a ellos; una reducción
trivial o una ganancia sólo debida al compilador no justificarán novedad fuerte.

El alcance principal es evaluación entre cargas de un mismo banco CWRU, con
fallas inducidas y dos posiciones de acelerómetro. No es una prueba en motores
nuevos, una evaluación de degradación temprana ni adquisición autónoma en campo.

## 2. Cómo queda lo que ya existe

| Bloque | Estado constatado | Función en la ampliación | Qué no se hará |
|---|---|---|---|
| Artículo Arciniegas entregado | Conservado en su versión actual | Antecedente y fuente técnica | No reemplazar sus cifras con cifras CWRU |
| Datos/modelos/resultados Arciniegas | Corpus, E1–E4, 40 ejecuciones y evidencia STM32 conservados | Reimplementación y análisis exploratorio histórico | No presentarlos como validación entre adquisiciones independientes |
| Auditoría Arciniegas | Finalizada; 13 familias nominales, una DETENIDO | Justificación de límites y de la ampliación | No intentar resolver independencia repartiendo aleatoriamente JSON |
| CWRU histórico DE a 2 kHz + CNN | Scripts y resultados por carga/diámetro existentes | Antecedente exploratorio y registro de exposición | No mezclar sus métricas con las de DE–FE a 12 kHz |
| CWRU dual a 12 kHz, N=4096 | Datos, metadata y FIR existentes | Referencia de entrada para auditar y reutilizar | No regenerar automáticamente si los hashes y el procedimiento son correctos |
| DNN CWRU 14–20–10–3 | Escalado, diez semillas, FP32/INT8 y prueba disponibles | Referencia CWRU V1 | No elegir nuevas características según su prueba ya observada |
| Detector CWRU normal-only | K=1, 12 características, umbral congelado | Referencia auxiliar V1 | No llamarlo K-means K=32 ni verificación independiente |
| Exportación CWRU a C/C++ | Fuentes, modelo y parámetros disponibles | Punto de partida para la integración | No afirmar que ya existen métricas de STM32 CWRU |
| Firmware histórico en D: | No inspeccionado en esta planificación | Referencia de interfaces y medición | No sobrescribir proyectos históricos |

Los recursos se reutilizan cuando su definición e integridad coinciden con el
nuevo protocolo. Los modelos entrenados no se reutilizan como estimadores de
pliegues nuevos si vieron alguna de sus muestras de evaluación.

### Inventario cuantitativo confirmado

- 28 MAT en `datos/cwru/raw`: 4 normales, 12 IR y 12 OR.
- OR corresponde al subconjunto centrado a las 6:00; se usan diámetros 0.007,
  0.014 y 0.021 pulgadas.
- Manifiesto existente: 14 archivos de cargas 0–1 HP para entrenamiento;
  7 de 2 HP para validación; 7 de 3 HP para prueba.
- Canales DE y FE: posiciones de sensor, no ejes X/Y/Z.
- Ventanas a 12 kHz de 4096 muestras; salto 2048; 50% de solapamiento.
- Cantidades: 568 entrenamiento, 406 validación, 406 prueba; 1380 en total.
- Representación: siete descriptores por posición, 14 en total.
- Referencia: StandardScaler ajustado con entrenamiento, pesos balanceados,
  DNN 14–20–10–3, 543 parámetros y semilla canónica 42.
- Modelo INT8 V1: 4064 bytes; hash verificado en esta revisión.
- Escalador y detector V1: hashes coincidentes con el protocolo conservado.
- Prueba V1 guardada: exactitud INT8 95.5665% y macro-F1 95.5448%.
- El detector V1 separó las 58 ventanas N y las 348 IR/OR de esa prueba; esto
  sigue siendo un resultado de siete registros, con sólo uno normal.
- No se localizaron resultados medidos de validación/latencia CWRU en STM32
  dentro de `resultados/cwru` durante esta inspección.

## 3. Límites que no deben volver a ocultarse

### Exposición previa a la prueba

La carga 3 HP ya fue evaluada en el pipeline V1. Además, los scripts y tablas
históricos de CNN a 2 kHz muestran pruebas por carga y diámetro que incluyen
archivos de esa carga. El documento `protocolo_congelado_pretest.json` respalda
el congelamiento antes de la evaluación de esa instancia V1; no demuestra que
nadie hubiera observado antes esos registros en otros ensayos del proyecto.

El nuevo estudio será una reevaluación retrospectiva con reglas predefinidas.
Una nueva partición o validación anidada mejora la separación entre selección
y evaluación de cada ejecución, pero no borra las decisiones históricas.

### Independencia física

CWRU registra condiciones de un banco y rodamientos con fallas inducidas.
Un archivo, una carga, un diámetro y un rodamiento físico son unidades distintas.
Separar cargas no garantiza separar rodamientos; DE/FE del mismo MAT tampoco
son adquisiciones independientes. El subconjunto normal no debe suponerse
procedente de cuatro rodamientos saludables diferentes.

Si el criterio editorial exige evaluar motores o rodamientos saludables nuevos,
estos 28 archivos no permiten garantizarlo. La conclusión defendible será sobre
el banco, registros y cambios de carga evaluados. Una extensión opcional por
diámetro puede examinar transferencia a severidades no usadas, pero no corrige
automáticamente la falta de diversidad física del estado normal.

## 4. Etapas y condiciones para avanzar

### Etapa 0. Auditoría exhaustiva de CWRU

Entrada: 28 MAT, manifiesto, variables internas, documentación oficial y derivados.

Actividades:

1. Inventariar hashes, procedencia, ID oficial, clase, carga, RPM, diámetro,
   posición OR y ubicación de la falla cuando estén documentados.
2. Enumerar todas las variables MAT, sus formas, longitudes, tipos, finitud y
   posibles duplicados exactos o parciales entre archivos y variables.
3. Verificar explícitamente las claves seleccionadas por el manifiesto. Prestar
   atención a 174 con prefijo X173, las variables adicionales de 175, X215/X217
   en 217 y X098/X099 en 99; no usar automáticamente la primera clave *_time.
4. Confirmar el muestreo aplicable a cada señal seleccionada mediante procedencia
   y coherencia temporal. Igual número de muestras no prueba por sí solo la
   frecuencia; tampoco se confundirá el canal FE de un ensayo DE con la colección
   distinta de fallas en el rodamiento del fan end.
5. Establecer la jerarquía banco–rodamiento/configuración–carga–registro–canal–ventana.
   Marcar IDs físicos desconocidos en vez de inventarlos.
6. Auditar variables y señales duplicadas antes del remuestreo; repetir controles
   necesarios sobre los derivados. Comprobar alineación de DE y FE.
7. Verificar el FIR, bordes descartados, metadata, etiquetas y hashes del dataset
   dual. No regenerar todavía el conjunto canónico.
8. Elaborar un registro de exposición: qué archivos/condiciones se utilizaron en
   entrenamiento, validación o prueba de cada experimento histórico.

Entregables: inventario MAT, registro de procedencia/exposición, mapa de variables,
auditoría de duplicados y dictamen de viabilidad. El procesamiento a 12 kHz queda
como candidato de referencia hasta pasar esta auditoría.

Condición para avanzar: ausencia de ambigüedades críticas en señales seleccionadas,
etiquetas y muestreo. La falta de ID físico se conserva como límite, no se declara
resuelta mediante partición por archivo.

### Etapa 1. Delimitación de la contribución y protocolo V2

Entrada: auditoría y literatura de métodos que consideran costo de características.

Congelar antes de la búsqueda: objetivo, hipótesis, candidatos, métricas, márgenes,
presupuesto, semillas, procedimiento de selección y reglas de parada. Registrar
lo conocido previamente, incluida la necesidad de escalado en CWRU y la inestabilidad
de frecuencia dominante en el detector normal-only V1.

Los objetivos de ingeniería propuestos son una pérdida de macro-F1 no mayor de
1 punto porcentual en selección y un ahorro objetivo de 20% de latencia total.
Son propuestas a justificar y fijar antes de ejecutar; no son estándares ni una
garantía de que se alcanzarán. El margen de selección no equivale a demostrar
no inferioridad estadística en prueba.

Entregable: protocolo V2 y cuadrícula finita de RF/SVM. No se ampliará la búsqueda
porque una evaluación externa del pliegue produzca resultados desfavorables.

Condición para avanzar: diferencia argumentada frente a selección tradicional.
Si la revisión de antecedentes muestra que el procedimiento es trivial o ya está
resuelto en las mismas condiciones, se revisará la contribución antes de entrenar.

### Etapa 2. Evaluación entre cargas con selección interna separada

Propuesta principal: validación anidada por carga con cuatro pliegues externos.

| Pliegue externo | Carga de evaluación | Cargas disponibles para desarrollo |
|---|---|---|
| A | 0 HP | 1, 2, 3 HP |
| B | 1 HP | 0, 2, 3 HP |
| C | 2 HP | 0, 1, 3 HP |
| D | 3 HP | 0, 1, 2 HP |

Dentro de cada pliegue externo se realizarán tres particiones internas: una carga
de validación y dos de entrenamiento. Cada MAT completo y ambos canales permanecen
juntos. El origen de las ventanas se decide antes de preprocesar o ajustar parámetros.

La selección de representación, checkpoints, RF/SVM, calibración del detector y
reglas de desempate utiliza exclusivamente las cargas internas. Tras seleccionar,
el modelo externo se entrena con las tres cargas de desarrollo. Su número de épocas
se fija a partir de una regla de las épocas internas —por ejemplo, mediana de las
mejores épocas, fijada en V2— sin early stopping sobre la carga externa.

Los resultados externos se ejecutan y revelan una vez congelado el procedimiento.
No se usan para decidir la máscara de características del siguiente pliegue; la
regla de selección debe ser la misma y estar definida para todos desde el inicio.

Esto evalúa el procedimiento ante una carga excluida. No es un conjunto público
nunca explorado ni una separación por rodamiento físico. Cuatro pliegues tampoco
equivalen a cuatro bancos independientes.

Entregables: manifiestos por pliegue, validaciones de conjuntos disjuntos por registro
y registros de qué artefactos se ajustan con qué cargas.

### Etapa 3. Contrato de preprocesamiento y representación

Referencia que se mantiene si supera auditoría:

- Seleccionar las claves DE/FE correctas, en ese orden.
- FIR Kaiser de 177 coeficientes float32 y diezmado 48→12 kHz por cuatro.
- Compensar el retardo y retirar las 22 muestras de borde a cada extremo a 12 kHz.
- N=4096, hop=2048 y 50% de solapamiento. No escoger N por maximizar uso de RAM.
- Guardar identificador MAT, carga, posiciones antes/después del remuestreo y ventana.
- Calcular RMS sobre señal original; momentos corregidos y FFT sin media.
- Mantener energía, amplitud, frecuencia y entropía con definiciones numéricas explícitas.

El filtrado no aprende parámetros de las señales de evaluación. No obstante, su
diseño histórico utilizó cargas 0–1: para pliegues donde una de ellas sea externa,
se declara que la especificación es una decisión histórica fija, no una elección
ciega a todo el corpus. No se rediseña el FIR con resultados externos.

Los escaladores, pesos y cuantización representativa se ajustan nuevamente dentro
de cada entrenamiento. No se utiliza el escalador V1 para pliegues V2 distintos.

Se almacenan features crudas con trazabilidad y se reutilizan cuando las definiciones
coinciden. Los productos de V2 no sobrescriben los NPY canónicos actuales.

Condición para avanzar: cálculo PC verificable, matrices finitas, tolerancias numéricas
definidas y tratamiento explícito de señales de baja varianza/empates espectrales.

### Etapa 4. Perfil de costo en STM32 antes de seleccionar

Se medirán bloques necesarios: momentos, centrado, RFFT, magnitudes/energía,
búsqueda de máximo, logaritmos de entropía, escalado, DNN y distancia centroidal.
Se conservará un grafo de dependencias: varias características comparten FFT,
momentos o buffers. El costo de una representación no es la suma ingenua de
costos unitarios y no es proporcional a su número de columnas.

Los benchmarks usan entradas de desarrollo o estímulos deterministas, reloj y
compilación fijados. No se usan métricas predictivas externas. La selección debe
considerar el costo medido o un modelo de costo contrastado para la representación
completa y candidatos finalistas. Registrar incertidumbre de las mediciones.

Primero se compila y prueba la representación completa para comprobar viabilidad
real de RAM y tiempo. El estudio reducido no se ejecuta masivamente si el firmware
base no puede medir de forma coherente esos bloques.

Entregables: perfil por bloques, dependencias y presupuesto de memoria medido.

### Etapa 5. Candidatos y comparadores de selección

Propuesta finita de ocho máscaras, definidas antes de la búsqueda:

| ID | Representación | Dimensión DNN |
|---|---|---:|
| F14 | Siete descriptores en DE y FE | 14 |
| T6 | RMS, asimetría y curtosis en DE y FE | 6 |
| S8 | Cuatro espectrales en DE y FE | 8 |
| NF12 | Todos salvo frecuencia dominante en ambas posiciones | 12 |
| NM10 | Todos salvo asimetría y curtosis en ambas posiciones | 10 |
| NE12 | Todos salvo entropía en ambas posiciones | 12 |
| DE7 | Siete descriptores sólo de DE | 7 |
| FE7 | Siete descriptores sólo de FE | 7 |

El catálogo es una propuesta controlable, no un algoritmo novedoso por sí mismo.
En las alternativas unicanal no se reasignan los dos canales del MAT a conjuntos
distintos; sólo se ignora una posición durante el cálculo.

Tres procedimientos DNN principales:

1. Referencia completa F14.
2. Selección por macro-F1 interno; desempates deterministas fijados.
3. Selección por menor tiempo total entre alternativas que cumplen los márgenes
   predictivos y las restricciones auxiliares fijadas en desarrollo.

Los selectores 2 y 3 usan los mismos candidatos, cargas, semillas, pesos y resultados
internos. Así se mide qué añade considerar costo, evitando una comparación donde
la ventaja provenga de un presupuesto de búsqueda mayor.

La topología será d–20–10–3, con d igual a la dimensión seleccionada. El escalado,
L2, Adam, lotes y máximo de épocas parten de la referencia documentada. No se
exploran simultáneamente arquitecturas, muestreo, N y características.

Comparadores externos: Random Forest y SVM, con F14 y la misma evaluación interna/
externa. Cuadrículas pequeñas predefinidas, no más de ocho configuraciones por
familia en esta propuesta. Sus métricas PC no se presentan como latencia o memoria
STM32 si no se implementan en esa plataforma. CNN a 2 kHz queda histórica; una
CNN nueva comparable sería opcional, fuera del núcleo necesario.

### Etapa 6. Tratamiento explícito del detector auxiliar

Se mantiene una salida de corroboración separada; no sobrescribe la clase DNN.
El ajuste del prototipo y su escalador utiliza sólo señales N de entrenamiento.
La evaluación binaria utiliza N frente a IR/OR etiquetadas de desarrollo/evaluación.
Si se usan esas etiquetas para elegir representación o configuración, se declara:
normal-only describe el ajuste, no todo el proceso de diseño.

Dos condiciones diferentes deben quedar distinguibles:

- Detector de referencia fijo: 12 características, sin frecuencias dominantes.
  Sirve para cuantificar cuánto limita el ahorro del clasificador reducido.
- Representación compartida: para máscara S, el detector recibe las características
  de S excepto frecuencias dominantes, con K=1 como regla predefinida. Se vuelve a
  ajustar escalador, centroide y calibración. Es otro detector, no uno equivalente.

La segunda condición es necesaria para estudiar ahorro del pipeline completo.
Se selecciona con restricciones explícitas sobre corroboración y aceptación normal,
además de macro-F1 DNN. No se acepta un ahorro conseguido degradando silenciosamente
el módulo auxiliar. Si no hay alternativa que cumpla, se conserva la referencia.

La regla candidata de umbral es P95 de puntuaciones N fuera de carga obtenidas en
los pliegues internos. Debe fijarse si se ponderan por igual registros/cargas o
ventanas; se propone igual peso por registro. El prototipo final se reajusta con N
de las cargas de desarrollo sin recalibrar el umbral en la carga externa.
Ese P95 no garantiza exactamente 5% de falsas alarmas en otro registro.

Las métricas del detector incluyen sensibilidad a fallas inducidas, especificidad
normal y exactitud balanceada. Su interpretación se restringe al subconjunto CWRU.
K=1 es un prototipo saludable; no se presenta como innovación en clustering.

### Etapa 7. Entrenamiento, semillas y control de ejecución

Semillas finales: 11, 22, 33, 42, 44, 55, 66, 77, 88 y 99.
Propuesta de semillas internas de selección: 11, 22 y 33, fijadas de antemano.
La semilla 42 mantiene el papel de instancia canónica para hardware, no de mejor
resultado observado. Se emparejan semillas entre procedimientos.

Estimación máxima de planificación para el catálogo completo:

- Selección interna: 4 externos × 3 internos × 8 máscaras × 3 semillas = 288 DNN.
- Evaluación externa: 4 × hasta 3 procedimientos × 10 semillas = hasta 120 DNN.
- Si dos procedimientos seleccionan exactamente la misma configuración y regla
  de entrenamiento, reutilizar el artefacto correspondiente en vez de duplicarlo.
- RF/SVM añaden ajustes cortos independientes del presupuesto DNN.

408 DNN es un límite de este diseño, no una ejecución autorizada en esta fase ni
una promesa de duración. Antes se cronometra un piloto exclusivamente de desarrollo,
se estima el costo total y se fija un presupuesto ejecutable. Reducir semillas o
candidatos se decide antes de consultar resultados externos, no después.

Se usan cachés de features, firmas de configuración, procesos aislados, timeout
basado en el piloto, guardado incremental y reanudación sólo de pendientes. Las
ejecuciones fallidas se registran; nunca se eliminan por rendimiento. Una repetición
por error técnico conserva la incidencia y no reemplaza silenciosamente un resultado.

Se guardan pesos, estructura, historia, IDs, versiones, semillas, métricas, matrices,
probabilidades y tiempos. No se exige serialización Keras completa si el entorno
reproduce la incidencia histórica: pesos NPZ más estructura ya son una vía existente.

### Etapa 8. Cuantización y contrato PC–INT8

Convertir las instancias nuevas seleccionadas, con datos representativos sólo del
entrenamiento correspondiente. Evaluar Keras, TFLite FP32 e INT8 bajo las mismas
particiones. La regla de selección interna debe incluir la evaluación INT8 si la
afirmación principal se refiere al modelo desplegado; no escoger sólo por FP32 y
reparar después utilizando resultados externos INT8.

Registrar escalas, puntos cero, redondeo, saturaciones, operadores, tamaño, cambios
de clase, empates y errores de probabilidad. No esperar igualdad absoluta entre
FP32/INT8: V1 ya muestra una discrepancia en prueba.

Si se cambia el desempate de frecuencia o cuantización, se versiona el contrato y
se regeneran sus derivados; no se reetiqueta la evidencia histórica.

### Etapa 9. Integración y validación STM32

Partir de los fuentes CWRU ya exportados y una copia controlada del proyecto de
integración, conservando los proyectos históricos. Secuencia:

1. Confirmar CMSIS-DSP, toolchain, banderas, HCLK, FPU y enlace.
2. Validar características compiladas contra PC sobre un conjunto de desarrollo
   representativo y determinista; revisar bajas varianzas, empates y extremos.
3. Integrar escalado, INT8 y detector; probar cada bloque y luego la secuencia.
4. Comprobar modelos canónicos de referencia, selector por rendimiento y selector
   por costo. Si las máscaras externas difieren, identificar qué instancias se
   validaron físicamente; no generalizar esa comprobación a todos los modelos.
5. Ejecutar replay completo de la evaluación de la instancia principal con trazabilidad
   por ventana, clase, probabilidades, score y bandera. Las tolerancias se fijan antes.
6. Medir DWT, Flash, RAM estática, arena, buffers y stack. Los buffers estáticos y
   FFT se reutilizan cuando sea correcto; el mapa de enlace determina el ahorro real.

Presupuesto de referencia: ventana dual 32 KiB, dos buffers FFT de 4096 float32
otros 32 KiB, arena inicial 16 KiB, más HAL/estado/stack. Aproximadamente 81 KiB
conocidos no son una medición final. El STM32 dispone de 128 KiB; verificar MAP y
watermark antes de afirmar factibilidad completa. Quitar columnas sin retirar
buffers u operaciones puede no reducir RAM o latencia.

El alcance empieza en una ventana ya remuestreada a 12 kHz. El FIR offline 48→12
no queda incluido en latencia STM32 salvo implementación y medición explícitas.

El transporte UART a 115200 es offline. Una ventana dual float32 ocupa 32768 bytes;
el mínimo binario 8N1 es 2.844 s, frente a un hop de 170.667 ms. ASCII tarda más.
Se reutiliza el protocolo por bloques CWRUFEAT como punto de partida; cualquier
protocolo completo tendrá versión, longitud, orden DE/FE, respuesta y validación
de integridad. Se presupuesta el tiempo de replay antes de una ejecución grande.

No se extrapolan los 6.3086 ms ni los 94200/19312 bytes medidos en Arciniegas a CWRU.

### Etapa 10. Estadística, criterios de éxito y límites

Métrica principal: macro-F1 multiclase; exactitud, precisión/recall por clase y
matrices de confusión como complementos. Reportar por carga y semilla, no sólo
el mejor resultado. Incluir métricas del detector y desacuerdos sin inventar una
exactitud fusionada a posteriori.

Separar variación por semilla de variación por registro/carga. Los 40 pares
carga–semilla no son 40 adquisiciones independientes. No aplicar un t-test sobre
ellos para afirmar generalización física. Las ventanas solapadas tampoco son
ensayos binomiales independientes.

Dar prioridad a diferencias pareadas por carga, rangos y tamaños de efecto. Si se
usan bootstrap por registros o cargas, especificar su carácter condicionado,
el escaso número de grupos, el único registro normal por carga y el solapamiento
de entrenamientos entre pliegues. No interpretar intervalos estrechos de semillas
como certeza ante motores nuevos. No declarar no inferioridad formal si el diseño
y el número de grupos no permiten sostenerla.

Para eficiencia, medir tiempo de características, DNN, detector y total con misma
configuración de compilación y reloj. Reportar media, mediana, P95, máximo y
variación instrumental cuando corresponda. Comparar con 170.667 ms sólo para
cómputo posterior a recepción; no afirmar tiempo real de adquisición por UART.

Una ganancia debe superar la variabilidad de medición y mantener los criterios
predictivos fijados. Si la alternativa completa es la única que los cumple, se
reportará ese resultado. Si la propuesta no supera a selección sin costo o a una
alternativa temporal sencilla, no se presentará como avance metodológico fuerte.

### Etapa 11. Organización del artículo ampliado y material suplementario

Estructura argumental prevista:

1. Problema: costo de extracción/decisión y necesidad de evaluación trazable.
2. Antecedentes: Arciniegas, TinyML y selección considerando recursos.
3. Método: auditoría, evaluación por carga, procedimiento propuesto y STM32.
4. Resultados concretos: rendimiento, comparación de selectores y recursos medidos.
5. Discusión: compromiso costo–rendimiento, límites físicos y exposición retrospectiva.
6. Conclusiones ajustadas a los resultados y disponibilidad de recursos.

Arciniegas permanece como antecedente y caso de reconstrucción. Su factorial completo,
40 semillas, auditoría detallada y tablas extensas pueden ir al suplemento. No se
comparan porcentajes Arciniegas/CWRU como si fueran la misma tarea. E1–E4 conserva
sus nombres; las variantes CWRU usan IDs nuevos para evitar confusión.

El manuscrito actual no se edita hasta cerrar la evidencia principal. La estructura
definitiva se revisa entonces; no se agregan sucesivamente secciones desconectadas.
Las preguntas guían el protocolo, sin necesidad de convertirlas en una sección de
preguntas explícitas dentro del artículo, conforme a la observación del tutor.

Se resolverán las observaciones editoriales pendientes: tablas legibles, unidades,
definiciones, figura de memoria, alcance UART, símbolos y PDF numerado por líneas.

### Etapa 12. Paquete reproducible y revisión final

Entregar datos/referencias de descarga según condiciones verificadas, manifiestos,
configuraciones, código, entorno fijado, modelos, fuentes y resultados por registro.
Separar licencias de código, dependencias y datos. No inventar commits históricos;
conservar sus snapshots/hashes y fijar versiones exactas de nuevas ejecuciones.

Crear matriz observación–acción–evidencia y comprobar cada cifra del artículo.
Hacer una reproducción representativa en entorno limpio, validar integridad y
revisar visualmente el PDF. La publicación de repositorio/DOI se realiza al final
con la autorización y condiciones correspondientes.

## 5. Organización propuesta de productos V2

Se conserva el árbol actual. Nuevos productos, cuando se ejecuten las etapas:

```text
articulo/ampliacion_cwru/
    PLAN_MAESTRO_AMPLIACION_CWRU.md
    protocolo_v2_propuesto.json
    auditoria/
    metodologia/
    tablas_y_figuras/
    respuesta_observaciones/
codigo/cwru/ampliacion_v2/
datos/cwru/particiones/ampliacion_v2/
datos/cwru/procesado/ampliacion_v2/
resultados/cwru/ampliacion_v2/
modelos/cwru/ampliacion_v2/
```

Sólo se crean inicialmente este plan y el esquema propuesto. Las demás carpetas
son destinos previstos, no evidencia ya existente. Nunca se ejecuta un script
histórico cuyo main sobrescriba salidas V1 para producir V2.

## 6. Prioridad y decisiones pendientes

Orden: auditoría CWRU → delimitación/congelamiento → evaluación por cargas →
preprocesamiento verificado → perfil de hardware → búsqueda interna → evaluación
externa del procedimiento → validación/medición STM32 → artículo → paquete.
La cuantización participa en selección y se completa para las instancias finales.

Pendientes antes de congelar: IDs físicos posibles, frecuencia de cada variable,
catálogo definitivo, margen de macro-F1, restricciones del detector, peso de registros
en calibración, regla de épocas finales, cuadrículas RF/SVM y presupuesto tras piloto.
La pausa científica ocurre si la auditoría invalida señales o si no se delimita una
contribución suficientemente distinta. No se declara resuelta la independencia
física por el mero cambio de dataset.

En esta etapa se verificaron documentos y artefactos; no se ejecutaron las 408
redes, no se realizó la auditoría MAT completa ni se programó la placa.

## 7. Evidencias internas utilizadas

- `datos/cwru/particiones/principal_por_carga/README.md` y `manifest_particion.csv`.
- `documentacion/cwru/generacion_dataset_dual_12khz_w4096.md`.
- `documentacion/cwru/protocolo_filtro_antialias_12khz.md`.
- `documentacion/cwru/seleccion_representacion_y_escalado.md`.
- `documentacion/cwru/cuantizacion_int8_features14.md`.
- `documentacion/cwru/kmeans_normal_por_carga.md`.
- `documentacion/cwru/protocolo_congelado_pretest.json`.
- `documentacion/cwru/evaluacion_final_test_cwru.md` y su resumen JSON de resultados.
- `documentacion/cwru/preparacion_despliegue_stm32.md`.
- `documentacion/cwru/integracion_prueba2_extraccion_features.md`.
- `resultados/cwru/generalizacion_carga_diametro.csv` y `diversidad_diametro/`.
- `codigo/cwru/entrenamiento/entrenar_cnn_diversidad_diametro.py`.
- `modelos/cwru/canonico_seed42/firmware/manifest.json`.
- Auditoría Arciniegas en `articulo/auditoria_dataset/revision_2026-10-22_final/`.

Algunas notas históricas dicen que la prueba seguía pendiente: describen su etapa,
no el estado actual. La evidencia posterior de evaluación prevalece para planificar.

## 8. Fundamento externo consultado

1. CWRU, presentación oficial: banco, motor y fallas inducidas.
   https://engineering.case.edu/bearingdatacenter/welcome
2. CWRU, descripción de archivos y canales.
   https://engineering.case.edu/bearingdatacenter/download-data-file
3. Vieira et al., *Towards a more realistic evaluation of machine learning models
   for bearing fault diagnosis*, Mechanical Systems and Signal Processing 258,
   114640 (2026). Se usa para distinguir partición por condición y por rodamiento.
   https://doi.org/10.1016/j.ymssp.2026.114640
   Versión de autores: https://arxiv.org/abs/2509.22267
4. Scikit-learn, validación anidada: separación entre selección y evaluación.
   https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html
5. Tabanelli et al., *Trimming Feature Extraction and Inference for MCU-based Edge
   NILM: a Systematic Approach*, IEEE Transactions on Industrial Informatics (2021).
   Antecedente que impide atribuir novedad general a reducir features/costo en MCU.
   https://doi.org/10.1109/TII.2021.3078186
   Versión de autores: https://arxiv.org/abs/2105.10302

Estas fuentes fundamentan cautelas y antecedentes, no sustituyen los resultados
experimentales que aún debe producir la ampliación.
