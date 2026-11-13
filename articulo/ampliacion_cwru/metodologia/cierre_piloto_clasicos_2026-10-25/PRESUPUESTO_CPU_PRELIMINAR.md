# Presupuesto CPU preliminar consolidado

La duración observada del piloto clásico fue 62.687 s. Las siguientes horas corresponden
a la campaña futura completa antes de reutilizar ajustes: DNN 408, RF 328, SVM 100 y
detector hasta 108, con hasta 816 conversiones DNN. Los pilotos se contabilizan aparte.

| Épocas | Perfil DNN | Solo etapas (h) | Con residual propio de procesos (h) |
|---:|---|---:|---:|
| 25 | F14, proceso completo | 2.737839 | 3.046217 |
| 25 | T6, incluye recuperación | 4.303306 | 4.802407 |
| 50 | F14, proceso completo | 3.117929 | 3.426307 |
| 50 | T6, incluye recuperación | 4.689997 | 5.189098 |

La referencia de planificación limpia a 50 épocas es aproximadamente **3 h 26 min**.
El perfil T6 compuesto suma importación y preparación repetidas por recuperación; el
escenario resultante es conservador y no demuestra que T6 sea más lento.

## Sensibilidad al arranque frío DNN

| Épocas | Si cada importación DNN repite el primer arranque frío (h) |
|---:|---:|
| 25 | 12.435462 |
| 50 | 12.815552 |

Esta sensibilidad cambia únicamente la importación DNN; conserva los clásicos medidos.
No se extrapola un arranque frío desconocido de RF/SVM ni se promete una duración.
Los modelos, semillas y cargas futuros pueden cambiar el tiempo y el trabajo reutilizable.

Los 32 representantes ya están dentro de las DNN internas. Pilotos y recuperación
técnica no se suman como nuevos resultados confirmatorios o adquisiciones independientes.
No se acredita ahorro de caché no medido.

Se excluyen compilación/flasheo y medición física STM32, UART/adquisición y revisión humana.
Tiempo de entrenamiento PC y latencia STM32 son alcances distintos.
Faltan el ejecutor definitivo, límites de campaña y costo total físico de representantes.
El presupuesto es preliminar; el protocolo completo no se declara congelado.

**Ejecución:** corresponde al usuario. Este cierre no inició la campaña prolongada.
