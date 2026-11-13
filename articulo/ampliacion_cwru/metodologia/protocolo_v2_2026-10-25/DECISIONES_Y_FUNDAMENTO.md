# Reglas V2 tras cerrar la validación de extracción

25 de octubre de 2026. Nueva versión operativa; conserva propuestas, datos,
modelos, manuscrito y sellos históricos. Reglas definidas para preparar el piloto;
presupuesto y costo total de modelos nuevos pendientes. No es autorización de
la búsqueda masiva ni de la evaluación predictiva externa.

## Objetivo y alcance que se mantienen

El usuario confirmó mantener **costo total del pipeline** como criterio de costo.
La extracción ya está medida y validada, pero DNN, escalado y detector nuevos
siguen sin tiempos completos. No se les asigna cero ni se añade el overhead V1
como constante universal. La selección no podrá ejecutarse con la tabla de
extracción como sustituto silencioso del total.

Se mantienen 28 MAT, 1380 ventanas, 12 kHz, N=4096, salto 2048, 50 % y DE/FE.
La auditoría y las particiones del 22 de octubre ya están terminadas; los campos
pendientes del borrador anterior son históricos. Cuatro cargas excluidas externas
y tres validaciones internas por pliegue conservan juntos MAT, canales y ventanas.
La evaluación es retrospectiva: los registros ya fueron explorados. No demuestra
independencia entre motores, rodamientos físicos nuevos o ensayos de campo.

La selección por costo tiene antecedentes, incluido [Greedy Miser, ICML 2012](https://icml.cc/2012/papers/592.pdf).
La aportación que se estudiará es empírica: costo total medido en ARM, operaciones
compartidas y el efecto de acoplar el detector, con evaluación entre cargas. No
se presenta una lista de ocho máscaras o una búsqueda finita como algoritmo
nuevo. Esta consulta de antecedentes no sustituye una revisión exhaustiva de
novedad antes de reclamar prioridad científica.

## Separación entre elección y evaluación

La regla es la misma para los cuatro pliegues. Scalers, pesos, checkpoints,
calibración INT8, detector y umbral se ajustan dentro de los papeles permitidos.
Los resultados externos se liberarán cuando las cuatro decisiones de selección
estén cerradas y no cambiarán reglas de pliegues posteriores. La separación
interna/externa evita el sesgo de evaluar con el criterio que se optimiza;
véanse [Cawley y Talbot, JMLR 2010](https://jmlr.org/papers/v11/cawley10a.html) y
el [ejemplo oficial de validación anidada](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html).
No restaura un test virgen a este corpus retrospectivo.

## Decisión nueva: calibración del detector con registros separados

Se sustituye expresamente la propuesta de P95 OOF seguido de refit con todos
los N de desarrollo. Un P95 calculado con normales del propio fold evaluado
favorecería su aceptación por construcción; transferir un P95 entre distintos
scalers y centroides también requiere una hipótesis adicional.

Ahora se separan ajuste, calibración y evaluación por MAT. En cada conjunto
permitido de entrenamiento, la **mayor carga numérica disponible** aporta su
registro N para calibración; las restantes cargas aportan N para ajustar scaler
y centroide. Internamente hay un N de ajuste y uno de calibración; externamente,
dos de ajuste y uno de calibración. **El detector no se reajusta después de fijar
el umbral.** La evaluación usa la carga reservada correspondiente.

La regla max es determinista y mantiene para el externo 3 HP los papeles
históricos ajuste N de 0/1 y calibración N de 2. Es una elección de ingeniería,
no una división óptima, balanceada o elegida según resultados nuevos. Reduce
los N disponibles para ajustar respecto del borrador; se acepta ese costo para
mantener el mismo scorer en calibración y evaluación. La separación de ajuste
y umbral coincide con la cautela de la [documentación oficial de umbrales](https://scikit-learn.org/stable/modules/classification_threshold.html).

La DNN sí utiliza todas las cargas de su entrenamiento autorizado. Un N
reservado para calibrar el detector puede formar parte de DNN_train: son módulos
con dependencias distintas y el score del detector no depende de pesos DNN.
Nunca se usa inner_validation u outer_test para esos ajustes o para el P95.

K=1 sigue siendo un único prototipo N. Se calcula el centroide como media
float64 de Z ya transformado en float32 y se exporta float32, sin forzar cero.
Scaler y centroide ponderan ventanas uniformemente. El P95 es el primer score
observado cuya CDF, con igual masa por registro, llega a 19/20; no se interpola.
Cada ventana pesa 1/(R*n_registro). Se exportan sus bits float32 y se rechaza
solo si score > umbral, aceptando empates. No garantiza 5 % de falsas alarmas
en otra carga ni establece cobertura conformal con ventanas correlacionadas.
El ajuste usa N; la comparación de máscaras utiliza etiquetas de desarrollo,
por lo que no se presenta todo el procedimiento como aprendizaje sin etiquetas.

## DNN, precisión e INT8

Se mantienen d–20–10–3, ReLU/softmax, L2=0.001, Adam=0.005, lotes de 32 y
50 épocas máximas. Se conservan lotes manuales tf.function para evitar el
bloqueo histórico del adaptador tf.data. La parada interna usa pérdida de
validación sin pesos, más L2, paciencia 10 y mejora mínima 1e-12; restaura el
mejor checkpoint. Solo entrenamiento utiliza pesos n/(3*n_clase).

Cambio numérico explícito: el scaler se ajusta con entrenamiento y se conservan
sus parámetros float64, pero se exportan float32. Entrenamiento, validación,
calibración e inferencia usan la misma resta/división float32. Esto evita
entrenar con fit_transform y desplegar otra normalización. No convierte los
pesos V1 en modelos V2 ni reescribe sus resultados.

El caché PC auditado se reutiliza sin escalar y se convierte a float32. Su FFT
double/SciPy no pasa a ser bit a bit equivalente a EntM/CMSIS para las 1380 filas
por haber validado 42 ventanas. Cada exportación nueva deberá versionar su
origen numérico y contrastar el pipeline final, distinguiendo error de extracción,
normalización y kernels de inferencia.

Cada ajuste produce TFLite FP32 e INT8. La selección usa macro-F1 INT8, con
calibración representativa de **todas** las ventanas del entrenamiento
correspondiente, en orden de ID. No se usan validación o prueba. Se conservan
operadores INT8, escalas, puntos cero, saturaciones y cambios de clase. La
[documentación oficial de cuantización entera](https://developers.google.com/edge/litert/conversion/tensorflow/quantization/post_training_integer_quant)
explica la función de los datos representativos; la exclusión de evaluación
es la regla metodológica de este proyecto. Los empates argmax eligen la menor
clase y la cuantización mantiene rint al par después del punto cero.

Semillas internas 11/22/33 y externas 11/22/33/42/44/55/66/77/88/99. Se promedian
nueve métricas internas, no probabilidades para crear un ensemble. La máscara
es una por pliegue externo, común a sus diez semillas. La 42 sigue canónica,
no la mejor observada. Las épocas del reajuste externo son la mediana entera
de los nueve mejores checkpoints de esa máscara; no hay parada externa.

## Comparadores justos y márgenes de ingeniería

F14 es referencia. Rendimiento y costo reciben los mismos ocho candidatos,
semillas, ajustes y restricciones del detector. Un ajuste fallido detiene la
cola; no se elimina del promedio. Artefactos idénticos se reutilizan por firma.

El selector de rendimiento maximiza la media de nueve macro-F1 INT8 entre
las máscaras admisibles para el detector. Desempata por mejor media de la peor
carga y después por el orden del catálogo; no usa latencia ni dimensión.
Para evitar que el redondeo binario impida esos desempates, se considera empate
numérico una diferencia de hasta 1e-12 respecto del máximo de cada etapa.
Esta tolerancia no es el margen predictivo de 0.01.

Para el detector, especificidad N y recalls IR/OR se promedian por registro
y luego por carga. BA=(spec_N+(recall_IR+recall_OR)/2)/2. Frente al detector F14
de estos mismos papeles, se admite pérdida media máxima de 0.01 en BA y 0.02
en cada especificidad/recall. Para la DNN se conserva el margen propuesto de
0.01 en macro-F1, ahora respecto del máximo de F14 y el selector de rendimiento.

Son tolerancias de ingeniería para un experimento comparativo, no normas
industriales, valores prescritos por bibliografía ni prueba de no inferioridad.
El límite 0.02 de componentes impide compensar una caída arbitraria de sensibilidad
con especificidad al mantener BA dentro de 0.01. Se publicarán resultados por
carga; estas restricciones medias no garantizan preservación en cada carga.
Un N a 0 HP tiene 28 ventanas y los restantes 58: una sola ventana normal puede
cambiar la especificidad 3.57 o 1.72 puntos, respectivamente. Una referencia
absoluta deficiente no se declara adecuada por conservarla.

El objetivo de costo es al menos 20 % frente a F14, conservado del plan como
objetivo de ingeniería. Para ordenar se usará la envolvente empírica de tiempo
total de las mediciones: L=min observado y U=max observado, no intervalo de
confianza o WCET. Solo se alcanza el objetivo si U_candidato <= 0.8*L_F14.
Se minimiza U; las candidatas cuyo intervalo contiene el mínimo U forman un
empate por incertidumbre y se ordenan por F1, peor carga y catálogo. Si ninguna
cumple restricciones/objetivo se conserva F14 y se informa que no se alcanzó
ahorro. La mejora respecto del selector por rendimiento se mide, no se garantiza.

## Cómo se obtendrá el costo total sin usar la carga externa

Se utilizarán representantes predefinidos: en cada externo, el interno cuya
validación es min(cargas de desarrollo), semilla 11 y ocho máscaras. Son 32
ajustes **incluidos** entre los 288 internos, reutilizables si sus firmas son
idénticas. Se ejecutarán como un bloque de calibración antes de completar el
resto de la cuadrícula y antes de seleccionar por costo; no se escogen por F1.

Cada banco temporal usará exclusivamente las dos cargas de entrenamiento
de su representante: primera y última ventana de sus 14 MAT, 28 ventanas por
tres repeticiones, las mismas para las ocho máscaras. No usará su carga de
validación interna ni la externa. Se medirá extracción de la unión requerida,
escalado/cuantización, DNN INT8, detector y residual integrado. Las tablas
históricas de extracción son evidencia de viabilidad, no sustituyen estas
mediciones completas ni se filtrarán según exactitud externa.

Si un término o su validación falta, el selector de costo no puede ejecutarse.
Las instancias finalmente desplegadas volverán a validarse, especialmente la
canónica 42. La cobertura de hardware se identificará por modelo/pliegue; no se
extenderá una medición a todos los pesos, dimensiones o semillas sin evidencia.

RF F14 tendrá ocho configuraciones: árboles {100,300}, profundidad {None,12},
hoja mínima {1,3}; Gini, sqrt, bootstrap y pesos balanceados. Usará tres semillas
internas y diez externas. SVM RBF F14 tendrá C {0.1,1,10,100} y gamma {scale,0.01},
pesos balanceados, scaler de entrenamiento y probability=False. SVM determinista
no genera diez copias ficticias. Sus tiempos PC no son latencia STM32.

## Piloto y presupuesto antes de la ejecución masiva

Primero se prepara un piloto de tiempo con externo 3 excluido, validación interna
0 y entrenamiento 1/2: 799/175 ventanas. Solo F14/T6 y semilla 11, 50 épocas
cronometradas sin parada temprana, dos conversiones FP32 y dos INT8. Sirve para
cronometraje y compatibilidad de herramientas; sus pesos no son resultados
confirmatorios y sus métricas no eligen máscaras o márgenes. Dos procesos
secuenciales, timeout de 300 s cada uno; 600 s es límite de procesos, no duración
prometida ni presupuesto de todo V2. No carga features o predicciones de 3 HP.

Se separarán importación, scaler, compilación, épocas, validación, conversión,
inferencia e I/O. El máximo interno tiene 812 ventanas y el reajuste externo
mayor 1205: extrapolar por lotes/épocas y sumar conversión, no multiplicar una
cifra histórica parcial. El piloto RF/SVM sigue pendiente para su presupuesto.

| Bloque | Máximo de planificación antes de reutilización |
|---|---:|
| DNN internas | 288 (incluye 32 representantes de costo) |
| DNN externas | 120 |
| Conversiones FP32 + INT8 | 816 |
| Piloto DNN adicional | 2 ajustes y 4 conversiones |
| RF | 288 internos + 40 externos |
| SVM | 96 internos + 4 externos |
| Detector | 96 internos + hasta 12 externos; sin repetir por semilla DNN |

El presupuesto es de tareas, no horas medidas. Se usarán procesos aislados,
CPU con un hilo, guardado incremental y reanudación solo de firmas completas.
No hay force/overwrite o reintento automático; una incidencia se conserva y
detiene la cola hasta revisión. La ejecución masiva y evaluación externa siguen
deshabilitadas. La generación de roles y pruebas deterministas no entrena modelos.

## Entregables y alcance de la comprobación de esta etapa

`protocolo_reglas.json` fija las reglas descritas. `roles_componentes.json`
asigna registros e IDs a los 12 casos internos y cuatro externos.
`calibracion_costo_total_plan.json` enumera los 32 representantes y las 28
ventanas temporales exactas de cada uno, con costos todavía null.
`piloto_plan.json` contiene los dos trabajos de tiempo preparados, sin ejecutar.

`PRUEBAS_REGLAS.json` conserva las 25 pruebas sintéticas aprobadas de roles,
P95, selección, empates y rechazo de datos/costos inválidos. El primer intento
tuvo un fallo por comparación decimal estricta; se preserva en
`PRUEBAS_REGLAS_INTENTO_01.json`. Se corrigió la tolerancia de desempate y se
comprobaron también diferencias reales superiores a esa tolerancia.

`preflight.json` verifica metadatos sellados, reglas ejecutables frente a
márgenes y catálogo declarados, papeles de registros, banco temporal y exclusión
de 3 HP del piloto. El manifiesto SHA-256 vincula estas salidas y sus fuentes.
No se abren matrices NPY/MAT, entrena, convierte modelos o repite hardware.

Esta comprobación no sustituye auditar el futuro entrenador ni autentica una
tabla temporal futura. Antes de aceptar `VALIDATED_TOTAL_COST`, el validador de
mediciones deberá comprobar modelos, parámetros, firmware, trazas y las 28
ventanas del plan. El helper de selección solo consume resultados previamente
validados; una etiqueta escrita manualmente no constituye evidencia de medición.
