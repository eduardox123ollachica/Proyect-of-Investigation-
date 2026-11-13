# Primera puerta física del banco de costo total V2

Ejecución `20261026T040903_561072Z` (25 de octubre de 2026 en America/Lima).
El cliente terminó con `GATE_PASS_NOT_TOTAL_COST_VALIDATED`. La auditoría
posterior revisó los datos guardados sin repetir mediciones o inferencias.
El manuscrito no se modificó.

Se ensayó únicamente `cost_outer0_inner1_F14_seed11` con la ventana global
568. La reconstrucción de 171 tramas DATA recuperó los 8192 float32 del RAW
autorizado; FNV `9d117352`, payload SHA256
`7ef9a68a721ae282b6e7b55e7f42245e31b6b4087dd073a528026951801b2494`.
Se verificaron 450 filas UART, 225 transacciones y 24 eventos.

## Resultados de la puerta

| Comprobación | Resultado |
|---|---|
| RUN principales | 3/3 correctos, una extracción por recorrido |
| Tiempo principal medio | 84.270196 ms |
| Rango observado | 84.269433–84.270883 ms |
| PROFILE separado | 1/1 correcto; no incluido en la media principal |
| BASE | Exacta a la referencia ARM histórica; dentro de tolerancia PC |
| Intermedios desde características ARM | Exactos al replay float32 independiente |
| Entrada y salida INT8 | Exactas al banco PC para la misma entrada |
| Rechazos del protocolo | 17/17 correctos; sin RUN válidos adicionales |
| Arena utilizada | 1188 B de 16384 B reservados, solo para este modelo |
| Inicialización | 1.376222 ms, excluida del costo principal |

El perfil diagnóstico registró extracción de 84.190683 ms, escalado y
cuantización de 13.361 microsegundos, DNN de 58.678 microsegundos, detector de
7.283 microsegundos y residual de 0.344 microsegundos. Su suma cierra el total
instrumentado de 84.270350 ms. No se resta la calibración DWT de cinco ciclos,
obtenida de 64 parejas de lecturas. La conversión temporal utiliza el HCLK
HAL nominal de 180 MHz, sin medición externa del reloj.

Respecto del caché PC hay una diferencia de una ULP en energía DE: bits ARM
`46c8ed4e`, PC `46c8ed4f`, diferencia absoluta 0.001953125. La tolerancia se
cumple, los catorce INT8 son iguales y la salida es `[127,-128,-128]`, clase 0.
La distancia del detector varía ligeramente (ARM 14.621339798; caché PC
14.621335030) y conserva la decisión bajo el umbral 711.029968262. No hubo
saturaciones ni rechazo de esta ventana. Esto no demuestra identidad bit a
bit desde el caché PC original ni valida sensibilidad a fallas.

## Alcance y continuación

Tres repeticiones de una combinación no representan las 84 observaciones
por representante previstas en la calibración. No se deduce WCET, intervalo
de confianza, robustez del clasificador ni arena suficiente para los otros
31 modelos. Los costos permanecen deshabilitados para el selector.

La campaña siguiente conserva los bancos y modelos: 896 combinaciones,
2688 RUN y 896 PROFILE, con 42 BASE. Se entrega al usuario para su ejecución.
La estimación de 15–25 minutos deriva del tráfico real y los tamaños RAW;
no constituye un tiempo ya medido del lote ni un resultado predictivo.

El ELF declarado y copiado conserva SHA256
`85e8d7e2d1848833117fd4704493710cda0058fecedef3b379a80e9560171b1c`;
los fingerprints y parámetros recibidos coinciden. No se adjuntó log de
verificación del programador ni readback de la Flash completa. Se mantiene
esa limitación al vincular firmware y medición física.

Fuentes: evidencia original en
`despliegue_stm32/costo_total_representantes_v2_180mhz_2026-10-25/Validacion_PC/resultados/20261026T040903_561072Z/`,
auditorías de trazas y numérica, y cierre de la puerta en la carpeta padre.
