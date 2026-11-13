# Compilación auditada y primer control físico

27 de octubre de 2026, America/Lima.

## Resultado confirmado

La compilación del usuario terminó con cero errores y cero advertencias:
`PASS_BUILD_PENDING_ELF_REVIEW_AND_ARM`. El proceso completo de control
registró 61.297 s y CubeIDE informó 10.480 s de build. Son duraciones de
compilación, no tiempos de procesamiento de ventanas.

El ELF fue auditado con estado `PASS_ELF_CANON42_VERIFIED_PENDING_ARM`.
Su SHA-256 es
`c2de62a6e73a7f0f4cc2f4d25f5cbef718bde28eb93e05d0273871337ce84a46`.
Los ocho modelos, sus 28 688 bytes de pesos/estructura INT8, las máscaras,
scalers, centroides, umbrales y parámetros de cuantización coinciden con
los recursos congelados. El kernel single-rounding enlazado coincide byte
por byte con el del firmware SR validado previamente; la biblioteca
precompilada se conserva.

| Recurso del firmware del banco de ocho modelos | Bytes | Capacidad utilizada |
|---|---:|---:|
| Flash, suma de secciones cargadas | 185 388 | 35.36 % de 512 KiB |
| RAM estática con reservas del enlazador | 88 460 | 67.49 % de 128 KiB |
| Reserva de heap, ya incluida en RAM | 512 | — |
| Reserva de stack, ya incluida en RAM | 4096 | — |
| Capacidad de la arena única, ya incluida en RAM | 16 384 | — |

La imagen ocupa un intervalo de 185 408 bytes de Flash y 88 464 bytes de
SRAM al considerar alineación. Estos recursos corresponden al banco con
ocho modelos e instrumentación. No son la ocupación mínima de un firmware
productivo de un solo modelo. La capacidad de arena no equivale a su uso
real; la reserva de stack no equivale a su máximo observado. Ambos límites
se mantienen explícitos.

Los metadatos de CubeIDE conservan semánticamente las 17 configuraciones
anteriores y añaden una; únicamente se exceptuaron los atributos generados
`env-hash`. No se modificaron los sellos previos, el proyecto original en
D:, los modelos, los datos o el manuscrito.

Los directorios `archivo_preparacion_cliente_gate_v1` y
`archivo_preparacion_cliente_gate_v2` conservan versiones de preparación
del cliente, previas a ejecutar la placa. La versión que se debe utilizar
es `Validacion_PC`. Se añadió el enlace obligatorio con el registro de
programación y el sellado de sus evidencias archivadas antes de entregar
esa versión.

## 1. Cargar el ELF auditado

Conecta la Nucleo-F446RE habitual y detén cualquier sesión de depuración
activa. El comando siguiente programa la placa mediante ST-LINK y verifica
la descarga. No importa ni recompila el proyecto en D:.

```powershell
$canon42 = "C:\Users\LENOVO\Desktop\Proyecto Investigación\despliegue_stm32\canonicos_42_180mhz_2026-10-27"
$pythonCwru = "C:\Users\LENOVO\AppData\Local\Microsoft\WindowsApps\python3.12.exe"

& $pythonCwru -B "$canon42\Validacion_PC\cargar_firmware_canon42.py" --program
```

Antes de escribir Flash, el ejecutor comprueba los archivos y el ELF,
identifica por SWD el ST-LINK `0672FF505487525067162526` y el STM32F446,
y exige el dispositivo esperado. Realiza un único intento de carga con
verificación y reset. El presupuesto global del programador es de 120 s;
no se reintenta automáticamente. La comprobación SWD también puede
reiniciar el dispositivo.

El resultado requerido para continuar es
`PASS_CANON42_PROGRAMMED_VERIFIED_RESET_PENDING_UART_GATE`.
Se guardan registros e informe en `programacion_flash/<UTC>/`, además de
`INTENTO_CARGA_CANON42.json`. Si falla, conservar la salida y revisar antes
de otra carga. El ejecutor bloquea un segundo intento automático.

## 2. Ejecutar el gate por COM4

Cuando la carga anterior termine con PASS, cierra cualquier monitor serial
que esté usando COM4 y ejecuta:

```powershell
& $pythonCwru -B "$canon42\Validacion_PC\validar_canonicos_gate.py" --gate --port COM4 --max-seconds 600
```

El gate verifica el informe y los registros de la carga anterior antes de
abrir COM. Los argumentos de firmware, build y auditoría tienen como
valores predeterminados los archivos de esta entrega, cuyos hashes se
comprueban. El límite es de 600 s, no una duración esperada. Este comando
no entrena ni programa la placa.

El control incluye:

- Dos ventanas RAW mecánicas de cargas diferentes y ocho pares
  modelo–ventana autorizados: cubren los ocho modelos y las cuatro máscaras
  canónicas. Se hacen 24 RUN, ocho PROFILE y dos BASE.
- Veinticuatro POST: la primera ventana reservada de cada etiqueta N, IR y
  OR para cada modelo. La selección no utiliza aciertos o confianza.
- Tres rechazos POST: formato, checksum y valor no finito; se comprueba que
  una entrada rechazada no deje visible un resultado anterior.
- Identidad del catálogo y parámetros, reloj declarado por HAL, DWT,
  FPSCR, uso real de arena y equivalencia de los intermedios para la misma
  entrada. Para POST se usan salidas exteriores guardadas; para RAW se
  recurre al intérprete PC solo si el vector INT8 ARM es diferente del
  guardado, conservando un caché por entrada.

Se espera `CANON42_STAGED_GATE_PASS_NOT_FULL_VALIDATED`, con evidencias en
`Validacion_PC/resultados/<UTC>/`. Comparte esa salida o el informe
`RESULTADO_GATE.json` para revisar las trazas antes del lote completo.

## Lo que todavía no quedaría cerrado con este gate

El gate es un control inicial; no ejecuta el banco completo de 56 RAW ni
los 2760 pares POST. El lote completo se habilitará después de auditar sus
evidencias. No demuestra extracción exhaustiva de las 1380 ventanas
exteriores, generalización a otro motor, stack máximo o WCET. El tiempo
POST no es tiempo del pipeline completo, y PROFILE, UART e inicialización
quedan separados del RUN principal.

Para una comprobación local de archivos sin contactar con la placa se
puede sustituir `--program` por `--check`, o usar `--check` en el cliente del
gate. Estos modos no crean resultados físicos ni sustituyen los pasos 1 y 2.
