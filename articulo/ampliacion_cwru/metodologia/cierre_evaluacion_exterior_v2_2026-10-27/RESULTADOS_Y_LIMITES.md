# Evaluación exterior CWRU V2: resultados auditados y límites

Campaña `20261027T222625_337707Z`: **4/4 cargas PASS**, sin fallos o reintentos.
Firmas, versiones, modelos, scalers, umbrales, snapshots y predicciones fueron
verificados. Los resúmenes se reconstruyeron desde las salidas guardadas;
no se entrenó o volvió a invocar un clasificador. Los scores deterministas
del detector se recalcularon únicamente para comprobar integridad.

Tiempo registrado de procesos y verificación: **55.202 s**. No es tiempo
de inferencia por ventana en STM32 ni incluye todos los pasos compartidos o
esta auditoría posterior. Se evaluaron 1380 ventanas de 28 MAT mediante
80 DNN, 40 RF, 4 SVM y 7 detectores únicos. Los aliases no son nuevos ajustes.

## Resultados supervisados

Medias con igual peso de las cuatro cargas y, cuando corresponde, diez
semillas. Las DNN de la tabla usan INT8; RF/SVM conservan su precisión nativa.

| Procedimiento | Macro-F1 medio (%) | Exactitud media (%) |
|---|---:|---:|
| F14 base | 86.2441 | 90.5200 |
| Selector por rendimiento | 76.5849 | 84.3763 |
| Selector por costo total | 88.0797 | 87.6253 |
| RF | 93.7601 | 96.0833 |
| SVM | 79.2519 | 83.1802 |

La comparación completa FP32/INT8, las métricas por clase, las dispersiones
descriptivas y los modelos canónicos están en los JSON adjuntos. La desviación
entre semillas condiciona el mismo corpus; no estima variación entre motores.
SVM tiene un único ajuste por carga, no diez réplicas artificiales.

| Macro-F1 por carga (%) | 0 HP | 1 HP | 2 HP | 3 HP |
|---|---:|---:|---:|---:|
| F14 base | 46.3199 | 99.9801 | 99.6168 | 99.0594 |
| Selector por rendimiento | 46.3199 | 95.2176 | 99.5977 | 65.2045 |
| Selector por costo total | 58.4442 | 95.2176 | 99.5977 | 99.0594 |
| RF | 76.7606 | 100.0000 | 99.8276 | 98.4522 |
| SVM | 17.9710 | 99.8006 | 99.6168 | 99.6190 |

| Diferencia pareada INT8 | Macro-F1 (pp) | Exactitud (pp) |
|---|---:|---:|
| Selector por rendimiento − F14 base | -9.6591 | -6.1437 |
| Selector por costo total − F14 base | +1.8356 | -2.8947 |
| Selector por costo total − Selector por rendimiento | +11.4948 | +3.2490 |

## Interpretación que puede sostenerse

RF obtuvo el mayor macro-F1 predictivo en este corpus. No corresponde afirmar
superioridad general de la DNN o del selector por costo. Los tamaños joblib o
tiempos PC no representan su viabilidad, Flash o latencia en el microcontrolador.

Costo mejora macro-F1 medio frente a F14, pero pierde exactitud. En 0 HP,
T6 aumenta recall N de 0 a 95 %, mientras recall IR baja de 95 a 12.7419 %;
no mejora todas las clases. En 1 HP FE7 pierde 4.7625 puntos de macro-F1 frente
a F14; el margen de selección interna no garantizaba preservación exterior
por carga. En 2 HP los resultados son cercanos.

La ventaja de costo frente al selector por rendimiento depende fuertemente
del respaldo F14 en 3 HP: NF12 obtiene macro-F1 medio 65.2045 %, frente a
99.0594 % del respaldo. Este último NO alcanzó el objetivo temporal interno;
no se atribuirá ese beneficio al ahorro de cómputo. Se conservó la regla fijada,
sin elegir retrospectivamente máscara o semilla después de ver los resultados.

0 HP es la principal limitación de generalización por carga. Sus medias deben
mostrarse junto a los promedios globales. La semilla canónica 42 permanece
fijada para hardware; no representa necesariamente el centro de las diez
semillas y no se sustituirá por la mejor ejecución exterior.

## Detector y cuantización

Los tres procedimientos tienen la misma especificidad N media por carga:
**50 %**, recall IR/OR **100 %** y balanced accuracy **75 %**. La especificidad
por carga 0/1/2/3 HP es 0/100/100/0 %. Rechaza los 28 N de MAT 97 y los
58 N de MAT 100. El rechazo de 86/202 ventanas N es
**42.5743 %** si se pondera por ventana: no debe confundirse
con el promedio de cuatro cargas. Solo hay cuatro MAT normales. El detector
no es un confirmador fiable de fallas ante cambios de carga con estos datos;
el recall alto de falla debe presentarse junto al rechazo de señales normales.
No se recalibrará el umbral usando esta evaluación.

FP32–INT8 registra **482 cambios en 41/80 modelos**,
entre 27 600 pares modelo-ventana. Hay 2390 valores
recortados de 304290 y 1450
evaluaciones de ventana con clipping. 273 cambios
ocurren en ventanas con clipping y 209 sin él;
1177 ventanas recortadas no cambian de clase.
No se atribuirán todos los cambios causalmente al clipping ni se afirmará
beneficio universal de cuantización. Los denominadores repiten ventanas entre
modelos/semillas y no son adquisiciones independientes.

La mayor pérdida observada es **-31.8924 pp**
en `outer0_T6_e47_s88` y la mayor ganancia
**+31.0318 pp** en
`outer0_T6_e47_s66`. Los cambios son muy variables entre semillas;
una diferencia pequeña en la media no permite declarar cuantización neutra
para todas las ejecuciones o cargas.

## Trabajo siguiente

Mantener este cierre como resultado del protocolo fijado. La evaluación es
retrospectiva sobre un corpus previamente explorado y no prueba independencia
de un motor nuevo; no se calculan intervalos IID o pruebas de no inferioridad
sobre ventanas solapadas. No ajustar decisiones confirmatorias para ocultar
resultados desfavorables. Cualquier modificación futura requeriría delimitarse
como un nuevo estudio exploratorio.

Quedan las validaciones PC–STM32 y mediciones físicas de los ocho modelos
canónicos con semilla 42, utilizando sus parámetros actuales, y la redacción
de estos resultados con las limitaciones indicadas. Los costos medidos durante
la selección pertenecen a representantes internos; todavía no sustituyen las
mediciones de esos pipelines finales. No se modificó el manuscrito.
