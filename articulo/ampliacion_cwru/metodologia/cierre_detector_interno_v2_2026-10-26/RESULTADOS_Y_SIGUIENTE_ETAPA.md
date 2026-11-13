# Detector interno completo y auditado

Estado: `INTERNAL_DETECTOR_COMPLETE_AUDITED_PENDING_CLASSICAL_AND_SELECTION`. Quedan cerradas96/96 filas lógicas,
42/42 ámbitos de parámetros y84 evaluaciones únicas. Se reutilizaron23
parámetros y30 evaluaciones; el usuario ejecutó19ajustes y54evaluaciones
nuevos. Las60 filas lógicas añadidas incluyen alias, no60scores independientes.

El worker terminó en49.094s, dentro de300s,
sin errores o reintentos. Este tiempo no incluye la verificación del padre.
La auditoría contrastó sellos, las73unidades nuevas, el scaler/centroide/P95
de los19ajustes y los scores, decisiones y métricas por MAT de84evaluaciones.
Se preservaron todos los artefactos y firmas originales. No se entrenó una
DNN, ajustó otra vez el detector o utilizó STM32/TensorFlow para este cierre.

## Referencia F14: promedios de las tres cargas internas por escenario

| Escenario exterior reservado (HP) | Especificidad N (%) | Recall IR (%) | Recall OR (%) | BA (%) |
|---|---:|---:|---:|---:|
| 0 | 66.67 | 100.00 | 100.00 | 83.33 |
| 1 | 33.33 | 100.00 | 100.00 | 66.67 |
| 2 | 36.90 | 100.00 | 100.00 | 68.45 |
| 3 | 33.33 | 100.00 | 100.00 | 66.67 |

Estas son métricas internas; la carga exterior indicada no se evaluó en ese
escenario. Los valores por máscara y escenario están en el JSON. Las tasas
se calculan por MAT y se promedian por clase y carga, sin ponderar por número
de ventanas. F14/NF12 comparten parámetros/evaluaciones en los12roles.

Limitación observada: F14 tiene una especificidad N media descriptiva de
42.56% y BA de71.28%,
aunque los recalls IR/OR son100%. En6/12
celdas F14 se rechazan todas las ventanas etiquetadas N. El recall elevado no
debe presentarse sin esta pérdida de aceptación de normales. El JSON incluye
además promedios por carga de validación, sin atribuir una causa a las diferencias.

PASS confirma ejecución válida; no declara que las tasas sean adecuadas para
una aplicación. Las guardias principales comparan con F14 mediante los
márgenes ya fijados: conservar una referencia deficiente no demuestra una
buena detección absoluta. El umbral usa solo N de calibración y no se modifica
después de observar validación. El detector no reemplaza la clase de la DNN,
ni se repite por semilla. No se aplica aquí la guardia absoluta0.95 del
suplemento como si fuera la regla principal.

La DNN interna288/288 y el detector96/96 quedan disponibles. Siguiente etapa:
preparar las búsquedas clásicas RF/SVM y su presupuesto, preservando el piloto
temporal ya cerrado. Después, aplicar las guardias y congelar la selección
antes de los reajustes/evaluación exterior. No se eligieron máscaras ni se
modificaron protocolo, dataset, modelos, firmware o manuscrito.
