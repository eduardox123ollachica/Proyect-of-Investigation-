# Cierre de la matriz DNN interna CWRU V2

Estado: `INNER_DNN_MATRIX_COMPLETE_AUDITED_PENDING_DETECTOR_CLASSICAL_AND_SELECTION`. Se completaron288/288 celdas:240 ajustes nuevos
y48 reutilizados por firma (32 representantes y16 P0). Cuatro escenarios
exteriores, tres validaciones internas por escenario, ocho máscaras y semillas
11/22/33. No se entrenó ni ejecutó inferencia nueva para esta auditoría.

Se contrastaron todos los cierres/sellos, checkpoint restaurado, historia de
parada temprana, scaler train-only, cuantización, predicciones y métricas de
los240 nuevos. Las48 referencias mantienen sus firmas y artefactos auditados.
El controlador terminó14 lotes adicionales; son16 lotes en total.
Todos los240 estados nuevos son PASS, sin reintentos. Tiempo contabilizado:
2483.029s (41.38min), frente a7200s disponibles. Excluye las
auditorías compartidas entre lotes y los ajustes reutilizados; no equivale a
duración total de reloj. No se amplió ningún límite.

## Resumen descriptivo de las36 celdas de cada máscara

| Máscara | Macro-F1 INT8 medio (%) | Peor celda (%) |
|---|---:|---:|
| F14 | 84.1438 | 25.1262 |
| T6 | 90.4133 | 52.9809 |
| S8 | 84.4498 | 46.7302 |
| NF12 | 80.3538 | 48.6954 |
| NM10 | 81.5207 | 26.1758 |
| NE12 | 86.3314 | 25.3205 |
| DE7 | 83.1870 | 29.2404 |
| FE7 | 86.0219 | 53.4608 |

La tabla reúne escenarios y semillas dependientes; no es evaluación exterior
ni una selección global. La comparación de cada escenario utilizará sus
nueve resultados internos, las guardias del detector y el costo total ya
validado. Las métricas completas por celda y las medias por escenario están
en los dos JSON de esta carpeta. No se calcularon intervalos IID sobre ventanas.

## Alcance y siguiente etapa

La matriz DNN queda cerrada. El detector conserva36/96 filas lógicas y23/42
ámbitos de parámetros: completar19 ámbitos únicos y60 filas, sin multiplicar
por semillas DNN. Después, completar las búsquedas clásicas, aplicar las
guardias y congelar las decisiones de los cuatro escenarios. Solo entonces
corresponden los reajustes y la evaluación exterior. El piloto temporal RF/SVM
ya cerrado no se repite. No se seleccionaron máscaras ni se editaron el
manuscrito, firmware, dataset o evidencias anteriores.
