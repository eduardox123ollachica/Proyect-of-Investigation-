# Fase 5: análisis factorial multisemilla y selección del modelo

## Evidencia utilizada

Se consolidaron las 40 ejecuciones INT8 preservadas en
`resultados/arciniegas/multisemilla/ejecuciones`: cuatro variantes E1--E4 por
diez semillas pareadas (11, 22, 33, 42, 44, 55, 66, 77, 88 y 99). No se
repitió ningún entrenamiento durante esta fase.

El procesamiento reproducible está implementado en
`codigo/arciniegas/experimentos_multisemilla/consolidar_multisemilla.py`. Sus salidas
principales son `resultados_por_semilla.csv`,
`contrastes_factoriales_por_semilla.csv` y `resumen_multisemilla.json` dentro
de `articulo/resultados_derivados/multisemilla`.

## Resultados por variante

| Variante | Accuracy INT8, media ± DE (%) | Macro-F1, media ± DE (%) |
|---|---:|---:|
| E1 | 98.4459 ± 1.7047 | 98.3644 ± 1.8094 |
| E2 | 96.0285 ± 2.1511 | 95.4286 ± 2.5146 |
| E3 | 97.3574 ± 2.1360 | 96.9645 ± 2.4926 |
| E4 | 99.3994 ± 1.5061 | 99.3690 ± 1.6019 |

E4 tuvo la mayor media y mediana observadas, pero esto no implica que gane en
todas las inicializaciones ni que su rendimiento se generalice a otro corpus.

## Contrastes pareados

Los contrastes se calcularon dentro de cada semilla. El efecto marginal de un
factor es el promedio de sus dos contrastes condicionales y la interacción se
define como `(E3-E2)-(E4-E1)`.

| Contraste | Media (pp) | IC t 95 % (pp) | Positivos/negativos/empates |
|---|---:|---:|---:|
| Scaler marginal | -2.2297 | [-3.0004, -1.4590] | 0/10/0 |
| Pesos marginal | +1.1411 | [-0.2858, 2.5680] | 7/3/0 |
| Interacción | +0.3754 | [-1.6853, 2.4360] | 4/5/1 |

De forma condicional, E2-E1 promedió -2.4174 pp y E3-E4, -2.0420 pp. E4-E1
promedió +0.9535 pp y E3-E2, +1.3288 pp.

## Interpretación adoptada

- El escalado fue desfavorable en las diez semillas bajo esta configuración;
  el resultado está acotado a estas características, arquitectura,
  hiperparámetros y partición.
- La ponderación mostró una tendencia media favorable, pero su intervalo
  incluyó cero; no puede presentarse como una mejora garantizada.
- La interacción tampoco fue concluyente.
- E4 se conserva como modelo desplegado porque tuvo la mejor media observada,
  la instancia canónica ya fue verificada en STM32 y la ponderación no cambia
  la ruta ni el costo de inferencia. Esta es una elección de ingeniería
  respaldada por la evidencia disponible, no una demostración de optimalidad.

Los intervalos y pruebas de signos son exploratorios: solo hay diez semillas,
se evaluaron varios contrastes y no se aplicó corrección por multiplicidad.
Además, la misma partición de prueba intervino en la comparación y selección de
variantes. Por ello, CWRU sigue siendo la siguiente evaluación externa
preespecificada y no debe mezclarse retrospectivamente con estos resultados.

## Cambios efectuados en el manuscrito

Se actualizaron el resumen, las contribuciones, PI2, el protocolo estadístico,
la tabla multisemilla, los resultados factoriales, la comparación FP32--INT8,
la discusión, las amenazas, las conclusiones y la declaración de
reproducibilidad. También se regeneró la figura FP32--INT8 para incluir las
cuatro variantes. El documento compiló correctamente después de los cambios.
