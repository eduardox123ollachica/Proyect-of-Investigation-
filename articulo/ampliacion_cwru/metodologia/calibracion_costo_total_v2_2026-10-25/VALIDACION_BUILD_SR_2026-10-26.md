# Revisión del firmware Single Rounding y alcance de su validación

Registro del 26 de octubre de 2026 (America/Lima). Se conserva el
[diagnóstico causal](CORRECCION_REDONDEO_2026-10-26.md) y la captura de
la campaña fallida. Este documento complementa ese registro; no modifica
el manuscrito ni presenta como aprobada una prueba física pendiente.

## Compilación y equivalencia de recursos

El usuario ejecutó el build aislado de EntM, TotalV2 y TotalV2_SR. El primer
intento terminó en 100.140 s, sin errores ni advertencias en las tres
configuraciones. Los ELF e imágenes Flash de los dos controles son
idénticos a sus originales. No se entrenó ni se abrió un puerto serie
durante la compilación o la auditoría posterior.

El nuevo ELF de `Release_180MHz_TotalV2_SR` tiene SHA-256
`c0c3853d2b0e90bb20fe00be13bcf107d186088406be131b729a605ded44ecc7`.
La extracción directa del ELF comprobó los bytes de los 32 modelos, sus
escaladores, índices, parámetros de cuantización, centroides y umbrales
contra el catálogo y el firmware TotalV2 anterior. No se alteraron pesos,
semillas, datos, particiones ni referencias para hacer pasar el ensayo.

La ocupación por secciones es 280276 B de Flash (53.4584 %) y 88460 B de
RAM, incluyendo las reservas del enlazador (67.4896 %). El intervalo de
programación Flash ocupa 280288 B, con 12 B de alineación. La reserva de
stack sigue en 4096 B y la de heap en 512 B. Se conserva una sola arena
reutilizable de 16384 B y los buffers de ventana y FFT sin superposición.
Son cifras del banco de 32 modelos, no del despliegue final de un modelo
ni del máximo consumo dinámico medido en ejecución.

## Qué se corrigió

La revisión contrastó 37951 líneas de instrucciones y literales del listado
con los bytes del ELF. El símbolo del matvec por canal es único y procede
del objeto añadido, no del miembro antiguo de la biblioteca. La cadena
FC de TFLM llega al wrapper CMSIS-NN, a la función por canal y al reemplazo.
Sus seis sitios de recuantización, que cubren parejas y fila restante en
ambas ramas de offset, implementan el redondeo simple:

```text
p = int64(acumulador) * multiplicador
salida = ((p >> (30 - shift)) + 1) >> 1
```

La biblioteca precompilada original permanece intacta. Solo se compiló
de nuevo el matvec por canal; no se afirma que todos los objetos de TFLM
se hayan reconstruido con la misma opción. Se mantuvo la comprobación
previa de los dominios de los 1056 multiplicadores FC y de los 32 Softmax.

La medición principal conserva dos lecturas CYCCNT alrededor de extracción,
escalado/cuantización, DNN y detector. PROFILE conserva diez lecturas y
se informa por separado. La inicialización, selección de modelo, UART y
perfil diagnóstico no se incorporan al costo principal del selector.
Esta revisión no proporciona una nueva latencia física ni una garantía WCET.

## Instalación y preservación

Después de aprobar ambos dictámenes, se instaló la configuración SR en D:.
Se añadieron fuente, launch, ELF y mapa; se reemplazó únicamente `.cproject`
al final, después de completar respaldo y staging. Aprobaron las comprobaciones
de los cinco destinos, las 16 configuraciones anteriores, sus 16 ELF y las
demás fuentes originales. `APLICACION_SR_EN_D.json` registra
`PASS_SR_INSTALLED_PENDING_ARM` en la carpeta de la variante SR. Cinco pruebas
con archivos de ejemplo comprobaron confinamiento de rutas, rechazo de
launch incorrectos, preservación de metadatos, orden de activación y detención
ante colisiones. Durante la instalación no se compiló, abrió COM ni programó
la placa.

Se conserva el manifiesto de preparación de 1992 recursos. CubeIDE regeneró
solo los metadatos `language.settings.xml` de la copia aislada: añadió la
entrada SR y cambió el hash de entorno generado del proveedor. Se comprobó
la equivalencia semántica de las 16 entradas previas. Esta excepción se
registra sin reescribir el sello anterior y sin copiar esos metadatos a D:.

## Consecuencias y prueba pendiente

La corrección alinea la implementación aritmética con la referencia PC;
no constituye una mejora de entrenamiento o de rendimiento predictivo.
Se conserva la igualdad INT8 exacta para una misma entrada. Tampoco se
atribuyen al kernel corregido tiempos tomados con el firmware anterior.

La nueva puerta exige control modelo 0/ventana 568 y regresión modelo
22 DE7/ventana 232: seis RUN, dos PROFILE, dos BASE y 17 rechazos. Solo
después de auditarla procederá la campaña completa de 32 representantes,
42 ventanas únicas y 896 combinaciones, con 2688 RUN principales y
896 PROFILE. Hasta entonces no existe una tabla SR de costo total validado
y el selector permanece deshabilitado.

El archivo ELF auditado y los fingerprints por UART no prueban por sí
solos la identidad de toda la Flash de la placa. Se solicita conservar
también el log de descarga/verificación. La equivalencia de las dos ventanas
de la puerta no sustituye la validación sobre el banco completo.

Evidencias locales: `despliegue_stm32/costo_total_single_rounding_v2_180mhz_2026-10-26/`.
Los informes `AUDITORIA_ELF_SR.json`, `AUDITORIA_MAQUINA_SR.json`,
`APLICACION_SR_EN_D.json` y `CIERRE_BUILD_INSTALACION_SR.json` registran
las etapas por separado. El procedimiento de la prueba está en
`PRUEBA_REGRESION_SR.md` de esa carpeta.
