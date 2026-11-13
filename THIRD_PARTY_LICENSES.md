# Licencias de Terceros y Atribución de Componentes

El código, los scripts de evaluación y los modelos propios desarrollados por **Eduardo Andres Ollachica Arias** y **Luis Moises Vergara Chaucha** se distribuyen bajo **Apache License 2.0** (véase [LICENSE](LICENSE) y [NOTICE](NOTICE)).

A continuación se documentan los componentes de terceros presentes en los proyectos embebidos STM32, las dependencias de Python requeridas externamente y los corpus públicos externos no redistribuidos.

## 1. Software embebido incluido en el proyecto STM32

Los componentes siguientes figuran en `despliegue_stm32/canonicos_42_180mhz_2026-10-27/copia_aislada/Prueba_2_CWRU_Extraccion/` (y de forma análoga en `Prueba 2/` para Arciniegas):

| Componente | Ruta en proyecto firmware | Versión / detalle verificado | Licencia declarada | Archivo / cabecera verificada |
| :--- | :--- | :--- | :--- | :--- |
| **TensorFlow Lite for Microcontrollers (TFLM)** | `TFLM/` | Commit upstream `tensorflow/tflite-micro` @ `aab6e2fdb1b22e424395939462d30ac1bb5d950c` (`2026-09-25`, 1,829/1,830 blobs idénticos; único parche local: `@$(MICROLITE_LIB_PATH).rsp` en `Makefile` líneas 584–587; 644/644 archivos exportados; `Lib/libtensorflow-microlite.a`, `1,520,740 B`) | **Apache-2.0** | `TFLM/LICENSE` |
| **ARM CMSIS-NN** | `TFLM/third_party/cmsis_nn/` | `ARM-software/CMSIS-NN` @ `4ab83cc3cc98fb85ed6dafb55e8ca02f1628dcae` (`V.19.1.0`, `139` archivos) | **Apache-2.0** | `TFLM/third_party/cmsis_nn/Include/arm_nnfunctions.h`, `TFLM/third_party/cmsis_nn/LICENSE` |
| **FlatBuffers / gemmlowp / ruy / CMSIS_6 + Cortex_DFP (bajo TFLM)** | `TFLM/third_party/{flatbuffers,gemmlowp,ruy,cmsis}/` | `flatbuffers` `v25.9.23` (30 arch.); `gemmlowp` `719139ce...` (5 arch.); `ruy` `d3712831...` (1 arch.); `CMSIS_6` `5782d6f8...` + `Cortex_DFP` `c2c70a97...` (24 arch.) | **Apache-2.0** | `TFLM/LICENSE`, `TFLM/third_party/{cmsis,flatbuffers,gemmlowp}/LICENSE` |
| **KissFFT** | `TFLM/third_party/kissfft/` | `mborgerding/kissfft` tag `v130` (`6` archivos; Copyright (c) 2003-2010 Mark Borgerding) | **BSD-3-Clause** | `TFLM/third_party/kissfft/COPYING` |
| **ARM CMSIS-DSP** | `Middlewares/CMSIS-DSP/` | `V1.10.0` | **Apache-2.0** | `Middlewares/CMSIS-DSP/Include/arm_math.h` |
| **ARM CMSIS Core** | `Drivers/CMSIS/Include/` | `V5.1.2` | **Apache-2.0** | `Drivers/CMSIS/Include/core_cm4.h` |
| **STMicroelectronics CMSIS Device STM32F4xx** | `Drivers/CMSIS/Device/ST/STM32F4xx/` | Serie STM32F4xx | **Apache-2.0** (u opción `Package_license` como paquete ST) | `Drivers/CMSIS/Device/ST/STM32F4xx/LICENSE.txt` |
| **STMicroelectronics STM32F4xx HAL Driver** | `Drivers/STM32F4xx_HAL_Driver/` | `stm32f4xx_hal.c` `V1.8.2` | **BSD-3-Clause** (u opción `Package_license` como paquete ST) | `Drivers/STM32F4xx_HAL_Driver/LICENSE.txt`, `Src/stm32f4xx_hal.c` |

### Notas de compatibilidad y trazabilidad
1. **Combinación Apache-2.0 + BSD-3-Clause:** Cualquier redistribución de código fuente o binario completo del firmware (`*.elf`, `*.bin`) para STM32F446RE debe preservar tanto el texto completo de **Apache-2.0** (`LICENSE`) como los avisos de copyright y no-respaldo de **BSD-3-Clause** correspondientes a `STM32F4xx_HAL_Driver` y `KissFFT`.
2. **Trazabilidad completa de TFLM y recompilación limpia (Frente 2 verificado):** Se acreditó el commit upstream `tensorflow/tflite-micro` @ `aab6e2fdb1b22e424395939462d30ac1bb5d950c` (`1,829/1,830` blobs idénticos; único ajuste local en `tensorflow/lite/micro/tools/make/Makefile` líneas 584–587 para `@$(MICROLITE_LIB_PATH).rsp`) y se ejecutó la recompilación limpia desde fuente (`create_tflm_tree.py`: `644/644` archivos idénticos; `make ... microlite` con MSYS2 GNU Make 4.4.1 / Python 3.12.14 POSIX / GCC `14.3.rel1`: `318/318` payloads idénticos dentro de `libtensorflow-microlite.a`, `1,520,740 B`). La compilación limpia integral (*from source*, sin los 8 manuales `.chm` HAL de ST) en STM32CubeIDE 2.2.0 reproduce byte a byte el binario canónico `Prueba_2_CWRU_Extraccion.elf` (`254,412 B`, SHA-256 `c2de62a6e73a7f0f4cc2f4d25f5cbef718bde28eb93e05d0273871337ce84a46`) y su archivo `.map` (`24b4141a...`). En `public_release/` se excluyen esos 8 manuales `.chm` (`117,382,070 B` / `111.94 MiB`, no referenciados en `.project`/`.cproject`/`.ld` ni en el enlazado), conservando íntegros los archivos fuente HAL/CMSIS y licencias. Véase el informe técnico completo en [TFLM_Y_COMPILACION_STM32.md](documentacion/preparacion_publica/TFLM_Y_COMPILACION_STM32.md).

## 2. Entorno Python (instalación externa, no redistribuido)

El repositorio público no incluye ruedas ni binarios de Python. Las dependencias directas de ejecución listadas en [`requirements.txt`](requirements.txt) (`joblib==1.6.0`, `matplotlib==3.10.8`, `numpy==2.4.4`, `pandas==3.0.2`, `Pillow==12.2.0`, `pyserial==3.5`, `scikit-learn==1.9.1`, `scipy==1.17.1`, `tensorflow==2.21.0`, junto con `flatbuffers==25.12.19` y el resto de sus `49` dependencias transitivas bloqueadas en [`requirements-lock.txt`](requirements-lock.txt)) se instalan externamente mediante `pip` desde PyPI bajo `Python 3.12` (validado en entorno virtual limpio con `Python 3.12.10` en Windows x64 y `pip check` con código de salida `0`), sujetas a sus respectivas licencias de código abierto (`Apache-2.0`, `BSD-3-Clause`, `MIT`, `PSF`, etc.).

## 3. Conjuntos de datos externos (NO redistribuidos en `public_release/`)

| Corpus | Fuente oficial | Estado de licencia acreditada | Política de distribución |
| :--- | :--- | :--- | :--- |
| **Case Western Reserve University (CWRU) Bearing Data Center** | [bearingdatacenter](https://engineering.case.edu/bearingdatacenter/download-data-file) | `terminos_redistribucion_especificos_acreditados: false` (véase `INVENTARIO_DATOS.json`) | Excluido del repositorio público (ni `.mat` crudos ni matrices derivadas `.npy`/`.npz`/`.csv` de características o ventanas). |
| **Arciniegas et al. (2025)** | [Springer DOI 10.1007/s43926-025-00142-4](https://link.springer.com/article/10.1007/s43926-025-00142-4) (`Supplementary Material 1`, `ventilador-prueba-export.zip`) | `licencia_de_datos_separada_acreditada: false` (véase `INVENTARIO_DATOS.json`) | Excluido del repositorio público (ni `.json` crudos ni matrices derivadas `.npy`/`.csv` de características). |

Consulte [DATA_POLICY.md](DATA_POLICY.md) para instrucciones detalladas de descarga y verificación SHA-256 mediante `codigo/comun/verificar_datos_descargados.py`.
