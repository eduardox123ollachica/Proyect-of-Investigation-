# Cierre de los 32 representantes V2

Auditoría técnica PASS: 32/32, dos lotes (1+31), sin fallos, timeouts o reintentos.
Tiempo acumulado de procesos y comprobación del padre: 450.844946 s (7.5141 min).
Se comprobaron sellos, firmas, roles e IDs, pesos de clase, ES, checkpoints, modelos,
métricas, P95 y bancos. Sin reentrenamiento ni inferencia DNN adicional.

## Resultados internos descriptivos

Una semilla (11), un inner prefijado por outer. Outer1/2/3 validan la misma carga0.
La tabla no permite seleccionar máscaras ni representa evaluación externa final.

| Outer / validación interna | F14 F1 INT8 | T6 F1 INT8 | NF12 F1 INT8 |
|---|---:|---:|---:|
| 0 / 1 HP | 1.000000 | 1.000000 | 1.000000 |
| 1 / 0 HP | 0.419492 | 0.806942 | 0.949779 |
| 2 / 0 HP | 0.261758 | 0.859886 | 0.843617 |
| 3 / 0 HP | 0.405774 | 0.842019 | 0.985955 |

La pérdida de rendimiento ya aparece en FP32: no puede atribuirse únicamente a INT8.
Por ejemplo, F14 outer2 tiene F1 FP32 0.598474 y F1 INT8 0.261758. No se
ha demostrado todavía una causa única ni corresponde modificar parámetros para mejorar esta tabla.

## Épocas y cuantización

- 26/32 pararon por paciencia antes de50; 6/32 alcanzaron50.
- Los24 que validan0 HP restauraron época1 o2; los6 limitados por50 validan1 HP.
- Subir solo el máximo a100 conservando ES no prolongaría las26 trayectorias ya detenidas.
- Cambios FP32–INT8: 486 en 21 modelos.
- Valores de entrada de validación recortados: 4516 en 25 modelos; ajuste/banco:0.
- Los totales anteriores suman evaluaciones repetidas; no son ventanas físicas únicas.

## Detector de normalidad

IR/OR recall1.0 en los32; al validar0 HP la especificidad N es0 en18/24,
y en los otros6 es0.071429–0.214286. En validación1 HP DE7 tiene0.448276;
las otras siete máscaras1.0. La sensibilidad a fallas no compensa el rechazo
de normales. Parámetros de los alias F14/NF12 idénticos en los cuatro scopes.

## Continuidad y alcance

Se recomienda diagnosticar el comportamiento de0 HP, la reducción FP32–INT8 y
la calibración de normalidad mediante evidencia de desarrollo antes de interpretar
el efecto de aumentar épocas. El ensayo50/100 está propuesto, sin ejecutarse.
No se alteran decisiones, modelos o umbrales automáticamente a partir de estos resultados.
El costo total STM32 sigue pendiente; no se habilitaron búsqueda restante, COST o
evaluación externa. Las32 son subconjunto de288internos y se conservan sus artefactos.
El manuscrito, firmware y recursos sellados anteriores permanecen intactos.
