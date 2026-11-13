# Auditoría de la compilación canónica y transición a placa

El usuario ejecutó la compilación de `Release_180MHz_Canon42_SR` en la
copia aislada. Se conservaron los reportes originales de preparación y
compilación; el campo histórico que indicaba auditoría pendiente no se
reescribió. La resolución se registra en una auditoría posterior separada:
`despliegue_stm32/canonicos_42_180mhz_2026-10-27/AUDITORIA_ELF_CANON42.json`.

La auditoría verificó bytes del ELF ARM hard-float, tabla y parámetros de
los ocho modelos, topologías, rangos de memoria, buffers, reservas, código
POST y corrección single-rounding. Una revisión independiente comprobó el
kernel enlazado, el mapa, las opciones de los compiladores y la biblioteca
precompilada. Los metadatos regenerados de CubeIDE se compararon
semánticamente: se conservan 17 configuraciones y se añade la canónica.
Las comprobaciones no ejecutaron clasificadores ni contactaron con la
placa.

Resultado estático: 185 388 bytes de Flash y 88 460 bytes de RAM estática,
incluidas reservas de heap y stack. El firmware contiene ocho modelos de
verificación. No se extrapola esa ocupación al despliegue productivo de un
modelo, ni se equipara capacidad de arena o reserva de pila con uso real.

Se preparó un control físico inicial, previo al lote completo. Comprende
dos RAW con ocho pares modelo–ventana, 24 RUN, ocho PROFILE, dos BASE y
24 POST (primera ventana de cada clase por modelo), además de tres
rechazos de entradas inválidas. La elección es mecánica. El gate no
modifica las reglas, pesos, semilla 42, umbrales ni resultados exteriores.

Se decidió cargar directamente el ELF auditado mediante un ejecutor del
usuario con identificación SWD, verificación y reset. Así se conserva el
proyecto original de D: y se vincula la evidencia UART con un informe de
programación del mismo ELF. La verificación física exige dicho informe
antes de abrir COM; la comprobación local de archivos permanece offline.
Esta relación de procedencia es adicional a autenticar el catálogo por
UART, que por sí solo no demuestra todos los bytes de la imagen en Flash.

La preparación del cliente incluye pruebas sin puerto serial. Su
resultado no se registra como PASS físico. Solo después de que el usuario
cargue el firmware, ejecute el gate y se auditen las trazas podrá
habilitarse el lote completo de la validación escalonada ya elegida.

La [guía de carga y gate](../../../../despliegue_stm32/canonicos_42_180mhz_2026-10-27/GUIA_CARGA_Y_GATE_CANON42.md)
contiene comandos, estados esperados, presupuestos y limitaciones. El
manuscrito permanece sin cambios en esta etapa.
