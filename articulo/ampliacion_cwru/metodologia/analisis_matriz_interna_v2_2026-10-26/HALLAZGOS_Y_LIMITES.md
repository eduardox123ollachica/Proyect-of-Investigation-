# Variabilidad observada en la matriz interna

La matriz está completa y técnicamente auditada. Su rendimiento predictivo
varía con el rol, la máscara y la semilla. PASS confirma integridad de la
ejecución; no establece alto rendimiento. La media global descriptiva INT8
de las288 celdas es84.55% macro-F1, mediana
97.26% y mínimo
25.13%. No es la métrica final del sistema.

| Carga de validación interna (HP) | Macro-F1 FP32 medio (%) | Macro-F1 INT8 medio (%) |
|---|---:|---:|
| 0 | 56.97 | 55.32 |
| 1 | 97.28 | 97.26 |
| 2 | 96.62 | 96.57 |
| 3 | 88.90 | 89.06 |

Cada fila promedia72 celdas relacionadas. La caída al validar con0HP aparece
también en FP32; estos descriptivos no identifican su causa. Debe conservarse
al aplicar las guardias y reportar la evaluación posterior, sin ocultar celdas.

En151/288 celdas cambia alguna clase FP32–INT8. La media del cambio de macro-F1
es-0.3893 puntos porcentuales,
pero hay extremos importantes. `inner_outer3_inner0_NF12_seed22` pasa de
95.22% a60.05%
(-35.17 puntos). Se preservan todas las
predicciones y los cinco extremos de mejora/pérdida en el JSON. Esto tampoco
es una discrepancia PC–STM32: aquí se comparan modelos FP32 e INT8 en PC.

Entrenamiento tiene0 valores recortados; validación tiene13815 valores
de características recortados en180 celdas, sobre943920 evaluaciones
de características repetidas. No son ventanas únicas o independientes. No
se recalibra con validación ni se selecciona una semilla favorable. La métrica
de selección sigue siendo INT8 conforme al protocolo.

El siguiente paso es completar el detector para aplicar sus guardias junto
con estas métricas y los costos medidos. Las búsquedas clásicas y decisiones
de selección preceden a los reajustes y la evaluación exterior. Ningún dato
de esta revisión cambia el protocolo ni el manuscrito.
