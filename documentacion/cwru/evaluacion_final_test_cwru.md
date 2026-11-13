# Evaluación final CWRU en carga 3 HP

## Condición pretest

Antes de cargar las características y etiquetas de prueba se registró `protocolo_congelado_pretest.json`, que fija hashes, arquitectura, semilla, escaladores, modelo INT8, subconjunto de características del detector, `K=1` y umbral `309.9235534667969`. La evaluación verificó esos hashes antes de ejecutarse.

La prueba contiene 406 ventanas de siete archivos a 3 HP: 58 N, 174 IR y 174 OR. Este conjunto no se utilizó para seleccionar representación, escalado, arquitectura, semilla canónica, cuantización, K ni umbral.

## Clasificador DNN

| Modelo | Exactitud | Precisión macro | Recall macro | F1 macro |
|---|---:|---:|---:|---:|
| FP32 | 0.953202 | 0.965065 | 0.940613 | 0.951350 |
| INT8 | 0.955665 | 0.967012 | 0.946360 | 0.955448 |

La matriz de confusión INT8 fue:

| Real / predicha | N | IR | OR |
|---|---:|---:|---:|
| N | 53 | 2 | 3 |
| IR | 0 | 173 | 1 |
| OR | 0 | 12 | 162 |

INT8 y FP32 discreparon en una sola ventana normal. FP32 la asignó a OR, mientras INT8 produjo un empate cuantizado N--OR de 0.449219 y `argmax` seleccionó N. Por tanto, el incremento de una ventana observado en INT8 no debe interpretarse como mejora general de la cuantización.

Tres de 5684 valores de entrada cuantizados saturaron, equivalentes a 0.05278 %. La cuantización conservó el comportamiento global, aunque la equivalencia de clase ya no fue absoluta como en validación.

## Detector normal-only

El detector centroidal utilizó 12 características, un prototipo saludable y el umbral congelado. En prueba obtuvo:

- 58/58 ventanas N aceptadas;
- 174/174 ventanas IR rechazadas como alejadas de normalidad;
- 174/174 ventanas OR rechazadas como alejadas de normalidad;
- sensibilidad binaria de corroboración: 100 %;
- especificidad normal: 100 %;
- exactitud balanceada binaria: 100 %.

Los scores estuvieron claramente separados en este conjunto: N entre 26.1790 y 52.6019; IR entre 16351.3154 y 209993.5625; OR entre 960.0845 y 2506521.0. El umbral fue 309.9236.

Este resultado demuestra separación entre las grabaciones saludables y las fallas inducidas seleccionadas de CWRU a 3 HP. No demuestra por sí solo detección de degradación temprana, fallas desconocidas, variaciones de montaje ni generalización a otro motor.

## Lectura conjunta

La DNN INT8 cometió 18 errores:

- cinco ventanas N se clasificaron como falla, pero el detector las consideró compatibles con normalidad;
- una ventana IR se clasificó como OR;
- doce ventanas OR se clasificaron como IR;
- ninguna ventana IR/OR fue clasificada como N.

El detector corroboró alejamiento de normalidad en las 13 confusiones IR--OR, pero no puede corregir el tipo de falla. En las cinco ventanas N mal clasificadas por la DNN hubo desacuerdo entre ambos módulos. Esta complementariedad es descriptiva: no se definió antes de prueba una regla que permita al detector sobrescribir la clase de la DNN, por lo que no corresponde recalcular post hoc una exactitud fusionada.

## Alcance y bloqueo metodológico

La prueba final ya fue consultada. Las decisiones congeladas no deben modificarse utilizando estos resultados. Cualquier cambio posterior de características, arquitectura, K, umbral o regla de fusión constituirá un experimento exploratorio nuevo y no podrá reemplazar esta evaluación como estimación final independiente.

Los resultados reproducibles se encuentran en `resultados/cwru/modelo_features_14/evaluacion_final_test_load3/`, y el script es `codigo/cwru/entrenamiento/evaluar_test_final_cwru.py`.
