# Selección congelada antes de los reajustes y evaluación exterior

Firma: `c577339ac0c9450f99d40044850063e7090b19d5e81502eca0378c46e1b2c32f`. Se consumieron 288 celdas DNN INT8, 96 filas del detector,
384 celdas clásicas y 32 costos totales validados, autenticados contra sus cierres.
No se entrenó o ejecutó inferencia, materializó RAW/características ni leyó
predicciones exteriores. Las cuatro decisiones se congelan juntas.

| Carga exterior reservada (HP) | Máscara por rendimiento | Máscara por costo | Ahorro objetivo frente a F14 | RF config | SVM config |
|---|---|---|---|---:|---:|
| 0 | F14 | T6 | Sí | 5 | 5 |
| 1 | FE7 | FE7 | Sí | 5 | 7 |
| 2 | T6 | T6 | Sí | 0 | 4 |
| 3 | NF12 | F14 | No; respaldo F14 | 4 | 5 |

Se reutilizan literalmente select_performance/select_cost/final_epoch_count
del módulo original. Rendimiento usa la media de 9 valores de macro-F1 INT8 internos,
peor carga y orden del catálogo, entre máscaras admisibles para el detector.
Costo conserva guardias predictivas y requiere U(candidata)<=0.8L(F14).
Los intervalos son envolventes observadas, no IC o WCET. Los costos pertenecen
a representantes internos medidos; los pipelines finales aún deben validarse.
RF/SVM eligen media, peor carga y orden del grid con tolerancia 1e-12 por etapa;
RF promedia 3 semillas por carga y SVM no se triplica. No se elige una semilla.

## Excepciones y alcance que deben comunicarse

En el escenario 3 no hay candidata que cumpla simultáneamente margen F1 y
objetivo temporal. La regla previa devuelve F14; selected_satisfies_predictive_margin
es false. Su F1 interno es 84.4429 % frente a 91.5423 % de NF12: pierde 7.0995 puntos,
por lo que NO se declara preservación de F1 ni objetivo cumplido en ese escenario.
El protocolo no se cambia después de observar ese resultado.

Los objetivos temporales frente a F14 se cumplen en 3/4 escenarios internos;
esto no estima una probabilidad de éxito. Solo el escenario 0 añade ahorro
respecto del selector por rendimiento; en los escenarios 1 y 2 ambos eligen la misma máscara y
en el escenario 3 aparece el respaldo. Las guardias relativas no garantizan especificidad
absoluta adecuada: el problema de aceptación de normales del detector permanece.
Ninguna cifra de aquí es rendimiento de evaluación exterior.

## Épocas y trabajo siguiente

| Escenario | F14 base | Selector rendimiento | Selector costo |
|---|---:|---:|---:|
| 0 | 45 | 45 | 47 |
| 1 | 14 | 27 | 27 |
| 2 | 19 | 32 | 32 |
| 3 | 9 | 47 | 9 |

Son medianas enteras de los 9 mejores checkpoints internos por máscara. Se fijan
antes de entrenar con las 3 cargas de desarrollo; no habrá parada temprana con
la carga exterior. Semillas posteriores 11/22/33/42/44/55/66/77/88/99; 42 canónica
para hardware. El plan derivado tiene 80 reajustes DNN únicos
(en lugar de 120 solicitudes si se duplican procedimientos), 40 RF, 4 SVM y
7 ámbitos externos del detector, compartidos
entre semillas cuando corresponde. Este archivo delimita el trabajo pendiente;
el ejecutor y presupuesto se prepararán después. La liberación de predicciones exteriores
continúa bloqueada. No se modificaron manuscrito ni evidencias históricas.
