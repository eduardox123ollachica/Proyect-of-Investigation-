# Cierre de identidad de la imagen y calibración de costos SR

Registro del 26 de octubre de 2026 (America/Lima). Complementa la auditoría
de `20261026T144540_396879Z` y conserva sin modificar la tabla provisional,
el readback, su resultado inicialmente fallido y todos sus manifiestos.

## Lectura correcta y causa del fallo del validador

La lectura `20261026T151832_154874Z` terminó correctamente: CubeProgrammer
2.23.0 retornó cero, leyó 524288 B en 4.360 s y confirmó sonda, placa
NUCLEO-F446RE y dispositivo `0x421`. El proceso completo duró 4.765 s.
El comando registrado solicitó lectura SWD/HOTPLUG, sin programación,
borrado ni reset explícito. El texto «Reset mode: Software reset» informa
una configuración; no es evidencia de que se haya enviado un comando reset.

El comparador inicial exigía igualdad de los 280288 B cubiertos por los
segmentos PT_LOAD del ELF. Encontró doce ceros del archivo frente a doce
bytes `0xFF` en Flash, en `0x080001C4–0x080001CF`. La tabla de vectores
termina en `0x080001C4`; `.text` comienza en `0x080001D0`, alineada a 16 B.
Ese hueco no pertenece a ninguna sección ELF. El fallo procedía del alcance
del validador: trataba relleno entre secciones como contenido a programar.

El formato ELF admite espacio inactivo entre secciones para alineación,
con contenido no especificado. La clasificación se sustenta en las
cabeceras, el mapa y el linker del ELF local; no en una suposición de que
todo `0x00` pueda aceptarse como `0xFF`. Véase la
[especificación ELF, secciones](https://gabi.xinuos.com/elf/03-sheader.html).

## Corrección comprobada sin repetir la lectura

La comparación corregida obtiene todas las secciones SHF_ALLOC respaldadas
por archivo y las mapea a su dirección física de carga. Conserva los bytes
de relleno que estén dentro de una sección e incluye la imagen inicial
de `.data` en Flash, aunque su dirección de ejecución esté en RAM. No usa
una excepción por dirección ni tolerancia numérica.

| Sección | Dirección de carga | Bytes | Diferencias |
|---|---|---:|---:|
| `.isr_vector` | `0x08000000` | 452 | 0 |
| `.text` | `0x080001D0` | 104784 | 0 |
| `.rodata` | `0x08019B20` | 174544 | 0 |
| `.ARM` | `0x080444F0` | 8 | 0 |
| `.init_array` | `0x080444F8` | 8 | 0 |
| `.fini_array` | `0x08044500` | 4 | 0 |
| `.data` | `0x08044504` | 476 | 0 |
| **Total** | | **280276** | **0** |

Una conversión independiente del ELF mediante `objcopy -O srec` y
`objcopy -O ihex`, sin relleno explícito de huecos, produjo en ambos casos
los mismos 280276 bytes. Todos los registros tienen checksum válido y
todos los bytes emitidos coinciden con el dump. Ninguno de los dos formatos
codifica los doce bytes del hueco. Las conversiones son diagnósticos
actuales; no se presentan como el archivo histórico utilizado por CubeIDE.
La herramienta y la opción de generación SREC se describen en el
[manual de GNU objcopy](https://sourceware.org/binutils/docs/binutils/objcopy.html).

Cinco pruebas sintéticas comprobaron las guardas: el hueco solo se omite
cuando carece de sección; si se convierte en una sección asignada, sus
doce diferencias deben fallar. Alterar un byte de código, vector o imagen
inicial de datos también falla, y una lectura incompleta se rechaza.
Estas pruebas no conectan la placa. No se relajó la igualdad en código,
constantes, pesos, parámetros ni datos inicializados.

El dump confirma la correspondencia de la imagen programable con el ELF
auditado `c0c3853d2b0e90bb20fe00be13bcf107d186088406be131b729a605ded44ecc7`.
La lectura fue posterior a la última respuesta UART de la campaña por
990.437 s. Se documenta como comprobación posterior, junto con las
identidades de modelos, parámetros, ELF y trazas de la campaña; no prueba
por sí sola la imagen existente en cada instante histórico. Tampoco se
afirma que los bytes no usados de los 512 KiB estén definidos por el ELF.

## Cierre de los costos y continuidad

Con las auditorías numérica y de protocolo aprobadas, y la correspondencia
de imagen comprobada, se crea `COSTOS_TOTALES_SR_VALIDADOS.json`: 32 filas
`VALIDATED_TOTAL_COST`, una por escenario exterior y máscara. Conserva sin
cambios los 84 RUN de cada representante, sus medias y sus envolventes
observadas. La regla sigue siendo `U_candidato <= 0.8 * L_F14`, con mínimos
y máximos observados; no se sustituye por media o P95.

Se cierra la calibración computacional de los representantes predefinidos
con semilla 11. Esto permite utilizar la tabla como insumo de costo cuando
se complete la búsqueda interna. No se selecciona todavía una máscara:
faltan las tres semillas y tres validaciones internas por escenario, las
guardias predictivas y la posterior evaluación exterior. Si coinciden las
firmas de los 32 ajustes reutilizables, restan 256 de los 288 ajustes DNN
internos previstos. Primero debe prepararse el ejecutor reanudable y
congelarse su presupuesto; los procesos prolongados los iniciará el usuario.

El archivo `FAILED_FLASH_READBACK` y el primer verificador quedan como
evidencia histórica del criterio excesivo. La nueva clasificación y el
cierre no los sobrescriben. No hace falta repetir lectura, compilación,
programación o campaña. El manuscrito permanece sin cambios.

Evidencias nuevas en `despliegue_stm32/costo_total_single_rounding_v2_180mhz_2026-10-26/`:
`AUDITORIA_CAPTURA_FLASH_SR.json`, `AUDITORIA_RELLENO_ELF_FLASH_SR.json`,
`AUDITORIA_IMAGEN_PROGRAMABLE_SR.json`,
`PRUEBAS_COMPARACION_SECCIONES_FLASH_SR.json`,
`COSTOS_TOTALES_SR_VALIDADOS.json` y `CIERRE_COSTOS_TOTALES_SR_VALIDADOS.json`.
