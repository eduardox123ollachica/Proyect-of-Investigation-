# Cierre del control inicial y preparación de la validación completa

Fecha: 28 de octubre de 2026, America/Lima.

## Evidencia comprobada

El usuario programó y verificó el ELF canónico, cuyo SHA-256 es
`c2de62a6e73a7f0f4cc2f4d25f5cbef718bde28eb93e05d0273871337ce84a46`,
y ejecutó el gate `20261028T050129_905025Z`. La auditoría posterior reconstruyó
los intercambios UART, el contenido de las ventanas transmitidas, las páginas
de resultados, los parámetros, la procedencia de la carga y los rechazos.
El resultado `PASS_GATE_CANON42_AUDITED_PENDING_FULL` está en
`despliegue_stm32/canonicos_42_180mhz_2026-10-27/AUDITORIA_GATE_CANON42.json`.

La reconstrucción cuenta 621 pares UART, 93 eventos, 24 RUN, ocho PROFILE,
dos BASE, 24 POST y tres rechazos negativos. Se observaron 1188 bytes de
arena en los ocho modelos. No se confundió esa cifra con la RAM estática,
la capacidad de arena o el stack máximo.

En las dos ventanas RAW se exigió tolerancia PC–ARM en las 14 características
y coincidencia exacta de bins dominantes. Se exigió identidad de las
características activas respecto de BASE ARM, repetibilidad RUN–PROFILE y
coincidencia del procesamiento posterior usando la misma entrada. Las salidas
PC guardadas bastaron en este gate: no hubo nuevas invocaciones del
clasificador PC. Algunas características y distancias PC–ARM difieren en bits;
por ello no se afirma identidad bit a bit del pipeline RAW respecto de PC.

Las tres repeticiones de una sola ventana por modelo dieron medias RUN de
84.1056–84.1526 ms para F14, 39.6101–39.6617 ms para T6, 41.9841 ms para
FE7 y 84.5328 ms para NF12. Son comprobaciones preliminares, no una
distribución representativa de latencia ni una medición de WCET.

Tres de los 24 POST coinciden con predicciones PC incorrectas frente a su
etiqueta: modelo 0/gid0 NORMAL→OR, modelo 1/gid28 IR→OR y modelo 7/gid974
NORMAL→OR. El PASS es funcional. No se convierten los 21 aciertos de esta
selección mecánica en una estimación de exactitud general, ni se modifica la
selección de modelos a partir de ellos.

## Decisiones para completar el alcance acordado

1. Se conserva el gate y sus sellos, sin modificarlo ni volver a ejecutar
   sus ocho pares RAW y 24 POST.
2. Se prepara un cliente separado, autenticado con un manifiesto propio.
   La carga física sigue vinculada al sello original; el nuevo sello no
   sustituye la procedencia histórica de esa programación.
3. Se reutilizan 24 RUN, ocho PROFILE, dos BASE y 24 POST, marcados como
   importación del gate canónico auditado. Se añaden 984 RUN, 328 PROFILE,
   54 BASE y 2736 POST. Se excluyen los tiempos históricos de representantes
   de calibración.
4. Se completa el banco RAW de 56 ventanas únicas y 336 pares modelo–ventana:
   42 ventanas de desarrollo por modelo. POST cubre las 1380 ventanas
   exteriores, dos modelos por carga, sin repetir la extracción RAW.
5. El usuario ejecutará el lote largo. La estimación de 17–20 minutos proviene
   del gate; el presupuesto de 30 minutos limita la ejecución. No hay
   reintentos ni reanudaciones automáticas.
6. Las evidencias nuevas y las importadas conservan su origen y sus propios
   IDs de ejecución. El contador UART nuevo se obtiene del firmware; no se
   presupone que la placa conserve el contador final del gate.

La finalización del cliente se etiqueta
`CANON42_STAGED_FULL_PASS_PENDING_TRACE_AUDIT`: las trazas deberán auditarse
antes del cierre físico. No cambia el protocolo de selección, los modelos,
las semillas, las épocas, los umbrales o la evaluación exterior. No se añade
adquisición autónoma, lectura completa de Flash, medición externa del reloj,
energía, stack máximo o validación RAW exhaustiva del conjunto exterior.

El manuscrito permanece sin cambios. Este documento deja la evidencia y las
limitaciones disponibles para la futura consolidación del artículo.

La [guía de ejecución](../../../../despliegue_stm32/canonicos_42_180mhz_2026-10-27/GUIA_VALIDACION_COMPLETA_2026-10-28.md)
incluye el comando, los conteos y las condiciones de interpretación.
