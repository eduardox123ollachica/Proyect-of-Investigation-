# Trazabilidad embebida recuperada — fase 3

Fecha de auditoría: 2026-10-14.

## Alcance

Esta auditoría se realizó en modo de solo lectura sobre los proyectos STM32
históricos conservados fuera de la carpeta principal. No se recompiló el
firmware, no se volvió a programar la placa y no se generaron nuevas mediciones.
Por tanto, los tiempos y tamaños siguen siendo resultados históricos; la
auditoría únicamente recupera su configuración y los vincula con los artefactos
que aún existen.

## Proyectos inspeccionados

| Función | Ruta histórica | Artefactos conservados |
|---|---|---|
| Firmware E1 | `D:\STM32IDE\Prueba 2` | fuentes, configuración, ELF, MAP y LIST |
| Builds controlados B0–B4 | `D:\STM32IDE\Prueba 2 Recursos` | fuentes, configuración, ELF, MAP y LIST del último B4 |
| Firmware E4 | `D:\STM32IDE\Prueba 3` | fuentes, configuración, ELF, MAP y LIST |

Las tres rutas coinciden con `documentacion/migracion/RUTAS_FIRMWARE_EXTERNAS.md`.
No se encontró un archivo `.ioc`; la configuración reproducible disponible es
la del proyecto Eclipse/STM32CubeIDE (`.cproject`), los `makefile` generados, los
fuentes y los artefactos del enlazador.

## Configuración de compilación verificada

Los `makefile` generados de E1, B4 y E4 identifican **GNU Tools for STM32
14.3.rel1**. Los artefactos evaluados proceden de la configuración **Debug** y
las órdenes de compilación registran:

- optimización `-O0` y símbolos `-g3`;
- destino `-mcpu=cortex-m4`, `-mthumb`, `-mfpu=fpv4-sp-d16` y
  `-mfloat-abi=hard`;
- C en `gnu11` y C++ en `gnu++17` para los builds Debug;
- `-ffunction-sections`, `-fdata-sections` y enlazado con
  `--gc-sections`;
- `--specs=nano.specs`, `--specs=nosys.specs` y `-u _printf_float`;
- definiciones `TF_LITE_STATIC_MEMORY` y `CMSIS_NN` en las unidades C++.

Los archivos `.cproject` de E1 y E4 tienen el mismo SHA-256, y sus
`Debug/Src/subdir.mk` también son idénticos. Esto respalda que la comparación de
recursos E1–E4 utilizó la misma configuración y que el cambio desplegado fue el
modelo, no una opción distinta del compilador.

`Base/CONTEXTO_PROYECTO.md` y `Base/HISTORIAL_COMPLETO.md` registran
STM32CubeIDE 2.2.0. Ese dato no aparece en los archivos generados conservados y,
por ello, se clasifica como **documentado, pero no verificado de forma
independiente**. El dato directamente verificable es la versión 14.3.rel1 del
toolchain incluida en los `makefile`.

## Correspondencia entre modelos y firmware

Se extrajeron en memoria los 4480 bytes declarados en `g_model[]` de cada
`Src/model_data.cc` y se calculó su SHA-256, sin escribir archivos intermedios.

| Variante | SHA-256 del TFLite canónico y de `g_model[]` | Coincidencia |
|---|---|---|
| E1 | `C21EE1E33A22B2A04CC396F5C2F02089D664CE8DA30173A7AB67AD9B974E985D` | exacta, 4480/4480 bytes |
| E4 | `C505A32CC620965A4112596D5548E5C61143AE1CA729BF7CB7D9A1E3B50D5ADD` | exacta, 4480/4480 bytes |

Esta comprobación vincula los modelos de `modelos/arciniegas/*/firmware/` con
los proyectos que produjeron los resultados STM32. También resuelve la
confusión histórica de cuantización: el E4 correcto sin scaler usa escala de
entrada `0.9703094363212585` y punto cero `-125`; `0.079673462/-55`
correspondía al intento descartado que esperaba entradas estandarizadas.

## Medición temporal y comunicación

La fuente `Src/cmd_benchinfo.c` obtiene HCLK mediante
`HAL_RCC_GetHCLKFreq()` y devuelve además si `DWT->CYCCNT` está habilitado. Los
registros `benchall_1332.csv` y `benchall_resumen.txt` conservan HCLK =
16,000,000 Hz para las 1332 ventanas de E1 y E4.

`Src/cmd_benchall.c` lee `DWT->CYCCNT` inmediatamente antes y después de la
extracción de características, la DNN, K-means y el bloque total. El análisis
usa

\[
t_{\mu s}=\frac{\text{ciclos}\,10^6}{16{,}000{,}000}.
\]

El intervalo medido comienza después de recibir y parsear la ventana y termina
antes de transmitir la respuesta. No incluye sensor, adquisición, recepción o
transmisión UART, procesamiento del script de PC ni generación de una alerta.

La configuración conservada en `Src/main.c` es USART2 sobre PA2/PA3, AF7,
115200 baudios, 8 bits, un bit de parada, sin paridad, sin control de flujo y
sobremuestreo 16. UART es el mecanismo de reproducción de ventanas, no parte de
la latencia informada.

## Memoria y stack

`Src/tinyml_inference.cpp` declara una arena estática alineada de 16 KiB. Los
artefactos históricos y la salida de `arm-none-eabi-size` documentada registran:

| Alcance | Flash | RAM estática | Interpretación |
|---|---:|---:|---|
| Firmware completo E1 | 94,200 B | 19,312 B | incluye infraestructura y comandos de validación |
| Firmware completo E4 | 94,200 B | 19,312 B | mismo alcance y configuración que E1 |
| Build controlado B4 | 77,644 B | 19,312 B | pipeline completo dentro del proyecto de atribución de recursos |

Los 77,644 B y 94,200 B no son mediciones rivales: proceden de proyectos con
alcances distintos. En ambos casos, Flash se calculó como `text + data` y RAM
estática como `data + bss`.

El stack se midió en B4 mediante watermark sobre una región de 4096 B con el
sentinela `0xA5A5A5A5`. El pico ajustado fue 1028 B para características, 824 B
para DNN, 156 B para K-means y 1036 B para el pipeline secuencial completo. No
se atribuye ese valor al mismo binario de 94,200 B.

## Bibliotecas

Se verificó que los proyectos conservan los fuentes utilizados de TFLM,
CMSIS-NN y CMSIS-DSP. Los encabezados del árbol principal registran CMSIS-Core
5.6 y STM32F4 HAL 1.8.2. Sin embargo:

- el snapshot de TFLM no contiene metadatos Git, etiqueta ni commit de origen;
- el snapshot de CMSIS-NN no conserva una versión explícita identificable;
- el snapshot de CMSIS-DSP no conserva una etiqueta o archivo de paquete que
  permita asignarle con seguridad una versión upstream.

En consecuencia, el artículo puede declarar las bibliotecas y conservar sus
fuentes exactos, pero no debe inventar números de versión. Para el paquete
reproducible de la fase 6 deberán incorporarse los snapshots utilizados o un
manifiesto íntegro de hashes, además de documentar su procedencia si logra
recuperarse.

## Artefactos y hashes

Los hashes de los modelos, configuraciones y productos de compilación se
conservan en
`articulo/resultados_derivados/trazabilidad_firmware_stm32.csv`. Los proyectos
continúan en rutas externas y no forman todavía parte de un repositorio público;
esa incorporación corresponde a la fase 6 y debe respetar las licencias de las
dependencias.

## Estado de cierre de la fase 3

Quedan respaldados por archivos primarios: toolchain, nivel de optimización,
opciones de CPU/FPU, arena, método DWT, comprobación de HCLK, UART, ELF/MAP,
alcance de 94,200 B, build B4 y correspondencia exacta de los modelos
desplegados. Permanecen como limitaciones declaradas la versión no verificable
de CubeIDE y la ausencia de identificadores upstream de TFLM, CMSIS-NN y
CMSIS-DSP. Ninguna de estas limitaciones exige repetir los benchmarks ya
realizados.
