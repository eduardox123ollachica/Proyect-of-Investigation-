# Validación física canónica CWRU: resultados auditados

28 de octubre de 2026, America/Lima.

La validación escalonada acordada quedó completa y auditada para los ocho modelos INT8 de semilla 42. La ejecución nueva duró 1005.016 s (16 min 45.016 s). Se conservaron los modelos, parámetros y decisiones congeladas.

## Cobertura y equivalencia

El total reúne 56 ventanas RAW únicas, 336 pares modelo–ventana, 1008 RUN, 336 PROFILE y 56 BASE. Cada modelo tiene 42 ventanas de desarrollo con tres RUN y un PROFILE por ventana. POST cubre los 2760 pares correspondientes a las 1380 ventanas exteriores, desde sus características PC guardadas.

Del gate se reutilizaron 24 RUN, ocho PROFILE, dos BASE y 24 POST. La ejecución nueva añadió 984 RUN, 328 PROFILE, 54 BASE y 2736 POST; 55 transmisiones RAW. La reconstrucción verifica 28836 pares UART nuevos y 7315 eventos, incluidos 93 importados del gate. El contador inicial informado fue cero, en una sesión distinta de la del gate; el nuevo registro conserva su origen y termina en 4048.

En los 336 pares RAW, la entrada INT8, la salida, la clase y la decisión de rechazo coincidieron con PC. En los 2760 POST, las matrices y métricas coinciden con las salidas INT8 exteriores guardadas de semilla 42. No se invocaron nuevos clasificadores PC para estas comprobaciones. PASS expresa equivalencia funcional; no implica que toda predicción sea correcta frente a su etiqueta.

## Latencia principal medida

Solo se usan los 126 RUN por modelo; UART, parser, checksums, inicialización/SELECT, BASE, PROFILE y POST quedan excluidos. Los percentiles son descriptivos del banco. El reloj informado por HAL es de 180 MHz; no se midió externamente.

| Carga exterior | Modelo | Media (ms) | Mediana (ms) | p95 (ms) | Máximo observado (ms) |
|---|---|---:|---:|---:|---:|
| 0 HP | F14 | 84.0775 | 84.0740 | 84.2057 | 84.2953 |
| 0 HP | T6 | 39.5742 | 39.5727 | 39.6979 | 39.7824 |
| 1 HP | F14 | 84.1021 | 84.0801 | 84.2893 | 84.6697 |
| 1 HP | FE7 | 41.9941 | 41.9760 | 42.1083 | 42.2518 |
| 2 HP | F14 | 84.1035 | 84.0778 | 84.2583 | 84.6693 |
| 2 HP | T6 | 39.6017 | 39.5797 | 39.7451 | 40.1549 |
| 3 HP | F14 | 84.1076 | 84.0850 | 84.2895 | 84.6607 |
| 3 HP | NF12 | 84.5149 | 84.4919 | 84.7033 | 85.0865 |

Las reducciones medias frente a F14 de la misma carga exterior son 52.9313 % para T6 en 0 HP, 50.0677 % para FE7 en 1 HP y 52.9131 % para T6 en 2 HP. NF12 en 3 HP es 0.4842 % más lento que F14. El selector por costo conserva el respaldo F14 en esa última carga; no se atribuye ahorro a ese respaldo.

La extracción ocupa 99.8353–99.9107 % del total instrumentado. El salto de 2048 muestras a 12 kHz equivale a 170.666667 ms. El máximo observado (85.086489 ms) deja 85.580178 ms en ese intervalo teórico. Esta comparación no demuestra adquisición continua ni WCET.

## Diferencias numéricas y memoria

330 de los 336 pares presentan alguna característica PC–ARM distinta en bits; hay 804 diferencias de características y 158 distancias de detector distintas en bits. Los bins dominantes coinciden en las 56 BASE. El máximo error absoluto es 1.0 en energías del orden de millones. Por ejemplo, gid1206 registra 8530657 en PC y 8530656 en ARM: error relativo 1.17224×10⁻⁷. El máximo error relativo no nulo es 1.99461×10⁻⁷ y el mayor error utiliza 0.0199448 % de la tolerancia admitida. Se documenta tolerancia en características y equivalencia posterior; no se afirma identidad bit a bit de toda la extracción.

La arena observada es 1188 B por modelo, con capacidad reservada única de 16 KiB. Los 185388 B de Flash y 88460 B de RAM estática son del banco con ocho modelos y reservas de heap/stack. No describen un firmware productivo individual. No se midió el stack máximo de este firmware.

## Resultados predictivos reproducidos: solo semilla 42

| Carga exterior | Modelo | Aciertos/ventanas | Exactitud (%) | Macro-F1 (%) | Rechazo NORMAL (%) |
|---|---|---:|---:|---:|---:|
| 0 HP | F14 | 102/175 | 58.2857 | 42.1602 | 100.0 |
| 0 HP | T6 | 102/175 | 58.2857 | 57.1848 | 100.0 |
| 1 HP | F14 | 393/393 | 100.0000 | 100.0000 | 0.0 |
| 1 HP | FE7 | 369/393 | 93.8931 | 95.1649 | 0.0 |
| 2 HP | F14 | 404/406 | 99.5074 | 99.6168 | 0.0 |
| 2 HP | T6 | 404/406 | 99.5074 | 99.6168 | 0.0 |
| 3 HP | F14 | 406/406 | 100.0000 | 100.0000 | 100.0 |
| 3 HP | NF12 | 304/406 | 74.8768 | 59.8498 | 100.0 |

Estas ocho instancias fijas no sustituyen el resumen de diez semillas de la evaluación exterior. Las limitaciones predictivas permanecen: generalización débil en 0 HP, deterioro de NF12 en 3 HP y rechazo de todas las ventanas normales en 0 y 3 HP. El detector reproduce el mismo problema de normalidad de PC; no es un confirmador fiable de falla frente al cambio de carga. No se recalibraron umbrales con estos resultados.

## Alcance cerrado y siguiente paso

Quedó cerrada la equivalencia física en el alcance escalonado elegido: RAW parcial de desarrollo y POST exhaustivo desde características exteriores. Se conservaron los reportes y sellos históricos, incluidos sus campos de pendiente; su resolución está en la auditoría posterior AUDITORIA_FULL_CANON42.json.

El siguiente paso es consolidar evaluación exterior, costo físico, comparación con baselines y limitaciones para preparar la redacción de CWRU. El manuscrito permanece sin cambios en este cierre. No se requieren nuevos entrenamientos o repetir esta campaña para trasladar sus resultados a la documentación. Adquisición autónoma, energía, WCET, stack máximo y RAW exhaustivo exterior quedan fuera de lo validado.
