# Trazabilidad de TensorFlow Lite for Microcontrollers (TFLM) y Recompilación Limpia STM32F446RE

Este documento acredita el origen exacto de `TFLM/` y `Lib/libtensorflow-microlite.a` utilizados en el proyecto canónico STM32F446RE (`despliegue_stm32/canonicos_42_180mhz_2026-10-27/copia_aislada/Prueba_2_CWRU_Extraccion/`, configuración `Release_180MHz_Canon42_SR`), detalla el procedimiento reproducible de compilación limpia desde fuente oficial y documenta la exclusión segura de manuales `.chm` de ST en la publicación abierta (`public_release/`).

---

## 1. Origen Upstream Exacto de `tensorflow/tflite-micro`

La carpeta local de origen `tflite-micro-main/` (extraída sin directorio `.git`, marcas locales `2026-09-28 00:31 UTC`) fue cotejada archivo por archivo mediante hash Git blob SHA-1 contra el repositorio oficial [`tensorflow/tflite-micro`](https://github.com/tensorflow/tflite-micro):

- **Commit upstream verificado:** `aab6e2fdb1b22e424395939462d30ac1bb5d950c`
- **Fecha del commit:** `2026-09-25T22:23:58Z`
- **Mensaje del commit:** `fix(python): remove resource variable printout (#3785)`
- **Resultado del cotejo Git blob SHA-1:** **`1,829 / 1,830`** archivos fuente rastreados en el commit remoto coinciden **byte a byte** (`onlyRemote = 0`, `changed = 1`).
- **Única modificación respecto del commit upstream:** `tensorflow/lite/micro/tools/make/Makefile` (`689` líneas upstream vs `691` líneas locales), únicamente en las líneas `584–587`, para escribir la lista de archivos objeto a un fichero de respuesta `@$(MICROLITE_LIB_PATH).rsp` y evitar el límite de longitud de línea de comandos en Windows al invocar `ar`:

```makefile
$(MICROLITE_LIB_PATH): $(MICROLITE_LIB_OBJS) $(MICROLITE_KERNEL_OBJS) $(MICROLITE_THIRD_PARTY_OBJS) $(MICROLITE_THIRD_PARTY_KERNEL_OBJS) $(MICROLITE_CUSTOM_OP_OBJS)
	$(Q)$(MKDIR_P) $(dir $@)
	$(Q)printf "%s\n" $(MICROLITE_LIB_OBJS) $(MICROLITE_KERNEL_OBJS) \
	$(MICROLITE_THIRD_PARTY_OBJS) $(MICROLITE_THIRD_PARTY_KERNEL_OBJS) \
	$(MICROLITE_CUSTOM_OP_OBJS) > $(MICROLITE_LIB_PATH).rsp
	$(Q)$(AR) $(ARFLAGS) $(MICROLITE_LIB_PATH) @$(MICROLITE_LIB_PATH).rsp
```

---

## 2. Identidad del Árbol `TFLM/` y Dependencias de Terceros Fijadas (`Makefile`)

El árbol exportado `TFLM/` consta de **`644` archivos** que son **idénticos byte a byte (`644 / 644`)** entre:
1. `tflite-micro-main/TFLM/`
2. `despliegue_stm32/canonicos_42_180mhz_2026-10-27/copia_aislada/Prueba_2_CWRU_Extraccion/TFLM/`
3. `PAQUETE_VALIDACION_REIMPLEMENTACION_STM32/firmware_stm32/dependencias_compartidas/TFLM/`

Los `644` archivos provienen íntegramente (`unmapped = 0`) del núcleo TFLM y las descargas fijadas por `tensorflow/lite/micro/tools/make/{download_dependencies.py, third_party_downloads.inc, ext_libs/cmsis_download.sh, ext_libs/cmsis_nn_download.sh}`:

| Componente | Carpeta en `TFLM/` | Archivos en `TFLM/` | Referencia upstream fijada (`Makefile` / scripts Make) | MD5 descarga | Licencia |
| :--- | :--- | ---: | :--- | :--- | :--- |
| **TensorFlow Lite Micro** (núcleo + `LICENSE`) | `tensorflow/`, `signal/`, `LICENSE` | 439 | `tensorflow/tflite-micro` @ `aab6e2fdb1b22e424395939462d30ac1bb5d950c` | — | Apache-2.0 |
| **ARM CMSIS-NN** | `third_party/cmsis_nn/` | 139 | `ARM-software/CMSIS-NN` @ `4ab83cc3cc98fb85ed6dafb55e8ca02f1628dcae` (cabecera `V.19.1.0`) | `c960031e7d10cf31c477fb34ef460106` | Apache-2.0 |
| **ARM CMSIS_6 + Cortex_DFP** | `third_party/cmsis/` | 24 | `CMSIS_6` @ `5782d6f8057906d360f4b95ec08a2354afe5c9b9` + `Cortex_DFP` @ `c2c70a97a20fb355815e2ead3d4a40e35a4a3cdf` | `563e7c6465f63bdc034359e9b536b366` / `3cbb6955b6d093a2fe078ef2341f6b89` | Apache-2.0 |
| **FlatBuffers** | `third_party/flatbuffers/` | 30 | `google/flatbuffers` tag `v25.9.23` (+ `flatbuffers.patch`) | `023eca1e211d64007124420cd6be29c7` | Apache-2.0 |
| **gemmlowp** | `third_party/gemmlowp/` | 5 | `google/gemmlowp` @ `719139ce755a0f31cbf1c37f7f98adcc7fc9f425` | `7e8191b24853d75de2af87622ad293ba` | Apache-2.0 |
| **ruy** | `third_party/ruy/` | 1 | `google/ruy` @ `d37128311b445e758136b8602d1bbd2a755e115d` | `abf7a91eb90d195f016ebe0be885bb6e` | Apache-2.0 |
| **KissFFT** | `third_party/kissfft/` | 6 | `mborgerding/kissfft` tag `v130` (+ `kissfft.patch`) | `438ba1fef5783cc5f5f201395cc477ca` | BSD-3-Clause |
| **eyalroz/printf** | *(no incluido en el árbol exportado)* | 0 | `eyalroz/printf` @ `f8ed5a9bd9fa8384430973465e94aa14c925872d` | `5772534c1d6f718301bca1fefaba28f3` | MIT |
| **Total exportado (`TFLM/`)** | — | **644** | — | — | Apache-2.0 + BSD-3-Clause |

> **Nota sobre Bazel vs Makefile:** En `tensorflow/tflite-micro` @ `aab6e2f...`, las versiones de `gemmlowp` (`719139ce...`) y `ruy` (`d3712831...`) compiladas en este firmware corresponden a `download_dependencies.py` / `third_party_downloads.inc` del sistema `Makefile` (y no a los commits distintos declarados para Bazel en `MODULE.bazel`).

---

## 3. Receta Reproducible de Compilación Limpia de `TFLM/` y `libtensorflow-microlite.a`

### 3.1 Entorno de compilación verificado en Windows

La compilación original y la reconstrucción limpia verificada utilizan:
- **Entorno POSIX en Windows:** **MSYS2 (`C:\msys64\usr\bin`)** con `GNU Make 4.4.1` (`Built for x86_64-pc-cygwin`), `Python 3.12.14` POSIX (`os.sep == '/'`), `GNU findutils 4.11.0`, `bash` y `sh`.
- **Compilador cruzado ARM:** `arm-none-eabi-gcc` / `arm-none-eabi-g++` de **STM32CubeIDE 2.2.0** (`GNU Tools for STM32 14.3.rel1.20251027-0700`, GCC `14.3.1 20250623`), pasado por `TARGET_TOOLCHAIN_ROOT`.
- **Motivo técnico de usar MSYS2 POSIX Python y Make:** El `make` MinGW/BusyBox incluido en CubeIDE y Python nativo de Windows generan separadores `\` en `specialize_files.py` y fallan en reglas `find ... -false` / pipes `$(shell ...)` de los scripts `.sh` de TFLM. Con `C:\msys64\usr\bin` al frente de `PATH` se evitan modificaciones al código fuente.

### 3.2 Paso 1: Regenerar `TFLM/` mediante `create_tflm_tree.py`

Desde la raíz del árbol limpio `tflite-micro` (commit `aab6e2f...` con el parche `.rsp` del `Makefile` y la carpeta `downloads/`):

```bash
python3 tensorflow/lite/micro/tools/project_generation/create_tflm_tree.py \
  --makefile_options="TARGET=cortex_m_generic TARGET_ARCH=cortex-m4+fp OPTIMIZED_KERNEL_DIR=cmsis_nn TOOLCHAIN=gcc TARGET_TOOLCHAIN_ROOT=<RUTA_POSIX_GCC_BIN>/" \
  TFLM_rebuilt
```

- **Resultado verificado en copia aislada (`exit = 0`, `52.98 s`):** `TFLM_rebuilt/` contiene **`644 / 644` archivos idénticos byte a byte** a `Prueba_2_CWRU_Extraccion/TFLM/` (`diff = 0`, `missing = 0`).

### 3.3 Paso 2: Compilar `libtensorflow-microlite.a` desde fuente

```bash
make -j8 -f tensorflow/lite/micro/tools/make/Makefile \
  TARGET=cortex_m_generic \
  TARGET_ARCH=cortex-m4+fp \
  BUILD_TYPE=release \
  OPTIMIZED_KERNEL_DIR=cmsis_nn \
  TOOLCHAIN=gcc \
  TARGET_TOOLCHAIN_ROOT=<RUTA_POSIX_GCC_BIN>/ \
  microlite
```

El producto se genera en `gen/cortex_m_generic_cortex-m4+fp_release_cmsis_nn_gcc/lib/libtensorflow-microlite.a`:

- **Archivo original en proyecto canónico (`Lib/libtensorflow-microlite.a`):** `1,520,740 B`, SHA-256 `84cd3fae745ce78530e49f353725f845588ca0298cc35bd01da98403eabbe974`.
- **Archivo recompilado en copia aislada limpia:** `1,520,740 B`, SHA-256 `e1631e46bee7caa8e1006b4bc6f91bbc95d2c11f91ae89f6e6b053d7877a84c2`.
- **Verificación binaria miembro por miembro del archivador `ar`:**
  - Total de registros en el archivo `ar`: **`318` vs `318`** (`316` objetos `.o` compilados + metadatos de índice de tabla de símbolos y nombres largos).
  - Mismo orden exacto y mismos nombres crudos de miembro.
  - **Cotejo SHA-256 de payloads binarios:** **`318 / 318` payloads idénticos byte a byte (`diffData = 0`)**.
  - Motivo de la diferencia global de SHA-256 del archivo `.a`: El `Makefile` de TFLM define `ARFLAGS := -r` sin bandera determinista (`D`), por lo que `317` encabezados `ar` registran únicamente la marca temporal (`mtime`) del entorno de compilación.

---

## 4. Recompilación Limpia Headless en STM32CubeIDE 2.2.0 (`Prueba_2_CWRU_Extraccion`)

Se ejecutaron **tres compilaciones limpias (`-cleanBuild`) independientes en directorios aislados** con **STM32CubeIDE 2.2.0**:

```powershell
stm32cubeidec.exe --launcher.suppressErrors -nosplash `
  -application org.eclipse.cdt.managedbuilder.core.headlessbuild `
  -data <workspace_aislado> `
  -import <ruta_aislada_Prueba_2_CWRU_Extraccion> `
  -no-indexer `
  -cleanBuild "Prueba_2_CWRU_Extraccion/Release_180MHz_Canon42_SR"
```

| Escenario aislado probado | Entrada `TFLM/` y `libtensorflow-microlite.a` | Manuales `.chm` ST | Salida CDT / tiempo total | `Prueba_2_CWRU_Extraccion.elf` (`254,412 B`) | `Prueba_2_CWRU_Extraccion.map` | ¿Idéntico bit a bit al canónico archivado? |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Proyecto aislado con `.chm`** | Conservados del proyecto aislado | Presentes (8) | `0 errors, 0 warnings` (`15s.984ms` build / `70.41 s` IDE) | `c2de62a6e73a7f0f4cc2f4d25f5cbef718bde28eb93e05d0273871337ce84a46` | `24b4141ae00ab6e1d04da2c51ed8c502d96a19518a5431ca9ad8fc9cd863132d` | **Sí (`True` en ELF y MAP)** |
| **2. Proyecto aislado sin `.chm`** | Conservados del proyecto aislado | Eliminados (0) | `0 errors, 0 warnings` (`10s.359ms` build / `58.24 s` IDE) | `c2de62a6e73a7f0f4cc2f4d25f5cbef718bde28eb93e05d0273871337ce84a46` | `24b4141ae00ab6e1d04da2c51ed8c502d96a19518a5431ca9ad8fc9cd863132d` | **Sí (`True` en ELF y MAP)** |
| **3. Integral *from source*** | `TFLM_rebuilt/` (`644` archivos) + `libtensorflow-microlite.a` recompilada (`e1631e46...`) | Eliminados (0) | `0 errors, 0 warnings` (`14s.155ms` build / `62.33 s` IDE) | `c2de62a6e73a7f0f4cc2f4d25f5cbef718bde28eb93e05d0273871337ce84a46` | `24b4141ae00ab6e1d04da2c51ed8c502d96a19518a5431ca9ad8fc9cd863132d` | **Sí (`True` en ELF y MAP)** |

Las secciones reportadas por `arm-none-eabi-size` en el binario reproducible `Prueba_2_CWRU_Extraccion.elf` son:
`text = 184912`, `data = 476`, `bss = 87984`, `dec = 273372` (`0x42bdc`).

---

## 5. Exclusión Segura de Manuales `.chm` de ST en `public_release/`

Dentro de `despliegue_stm32/canonicos_42_180mhz_2026-10-27/copia_aislada/Prueba_2_CWRU_Extraccion/Drivers/STM32F4xx_HAL_Driver/` existen **8 manuales compilados de ayuda HTML (`.chm`)** de STMicroelectronics que suman **`117,382,070 B` (~`111.94 MiB`)**:

| Archivo `.chm` | Tamaño (bytes) |
| :--- | ---: |
| `STM32F410Rx_User_Manual.chm` | 13,454,020 |
| `STM32F411xE_User_Manual.chm` | 13,029,802 |
| `STM32F412Zx_User_Manual.chm` | 14,463,372 |
| `STM32F417xx_User_Manual.chm` | 15,174,176 |
| `STM32F423xx_User_Manual.chm` | 16,013,172 |
| `STM32F439xx_User_Manual.chm` | 16,513,992 |
| `STM32F446xx_User_Manual.chm` | 15,542,988 |
| `STM32F479xx_User_Manual.chm` | 17,190,548 |
| **Total (8 manuales)** | **117,382,070** |

Verificaciones realizadas:
1. Ninguno de estos 8 archivos `.chm` está referenciado en `.project`, `.cproject` ni `STM32F446RETX_FLASH.ld`.
2. Su eliminación completa no afecta en absoluto la compilación ni el enlazado en STM32CubeIDE 2.2.0 (el ELF y MAP generados siguen siendo idénticos byte a byte al binario canónico validado físicamente en placa).
3. En `PAQUETE_VALIDACION_REIMPLEMENTACION_STM32/firmware_stm32/dependencias_compartidas/` tampoco fueron copiados esos 8 `.chm` (según consta en su `README_FIRMWARE.md`).
4. Por ello, `CONSTRUIR_PUBLIC_RELEASE.ps1` excluye `*.chm` de `public_release/` (ahorrando `111.94 MiB` en el paquete distribuible y evitando redistribuir binarios de documentación propietarios pesados sin valor de compilación), manteniendo íntegros todos los códigos fuente HAL/CMSIS, licencias (`LICENSE.txt`), `TFLM/`, `Middlewares/CMSIS-DSP/`, `Lib/libtensorflow-microlite.a`, `ELF` y `MAP`. Para comprobar integridad en `public_release/`, ejecutar:

```bash
python "Nuevas Observaciones/Reproducibilidad/verificar_firmware.py" --check
```
