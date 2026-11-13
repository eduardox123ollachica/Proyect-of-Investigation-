# Bitácora Paso a Paso — Preparación de Publicación Abierta

Este registro documenta de forma cronológica e íntegra cada cambio, exclusión, saneamiento y verificación ejecutada para preparar el repositorio público (`public_release/`) y actualizar la documentación legal y metodológica de la raíz del proyecto.

**Regla de alcance vigente:**
- `Revision_2026-11-09/` permanece **intacto** durante los Frentes 1, 2 y 3 hasta el cierre final de sincronización y manifiesto.
- Las pruebas sobre rodamientos o motores físicamente nuevos (Pendiente 2) quedan fuera de alcance por tratarse de trabajo futuro ajeno al artículo actual.

---

## Frente 1 — Licenciamiento, Avisos de Terceros, Autoría, Política de Datos y Saneamiento Público

**Fecha de ejecución:** 2026-11-12  
**Estado:** Implementado y verificado localmente en `public_release/` y raíz del espacio de trabajo.

### Decisiones fijadas antes de ejecutar
1. **Licencia del código y modelos propios:** `Apache-2.0`.
2. **Estrategia de árbol de trabajo:** Construcción de una carpeta independiente limpia `public_release/` + actualización de archivos legales y metadatos en la raíz del proyecto + mantenimiento de esta bitácora en `documentacion/preparacion_publica/BITACORA_PASO_A_PASO.md`, dejando intacta la carpeta `Revision_2026-11-09/` hasta el cierre final.
3. **Política estricta para archivos derivados (`.npy`/`.npz`/`.csv`):** Además de excluir los corpus crudos (`datos/cwru/raw/`, `datos/arciniegas/raw/`), se excluyen o sanean en `public_release/` todas las matrices o tablas con ventanas crudas, características extraídas por ventana o entradas cuantizadas, conservando métricas JSON/CSV consolidadas, modelos `.tflite`/`.keras`/C, predicciones saneadas de evaluación exterior y el script verificador SHA-256.

---

### Paso 1.1 — Auditoría de licencias embebidas y metadatos preexistentes

Se verificaron directamente en los archivos del espacio de trabajo:
- **Ausencia de `.git` y `LICENSE` propios en la raíz**, junto con `CITATION.cff` desactualizado (`v0.1.0`, título anterior y un solo autor) y ausencia del coautor `Vergara Chaucha, Luis Moises` también en la copia `08_Methodological_and_Review_Docs/CITATION.cff`.
- **Autoría canónica comprobada en `articulo_ieee_borrador.tex`:**
  - Título: *"Selección de características guiada por el costo de procesamiento en TinyML: evaluación de vibraciones en ARM Cortex-M4F"*.
  - Autores: `Ollachica Arias Eduardo Andres` (`eollachica@unsa.edu`) y `Vergara Chaucha Luis Moises` (`lvergara@unsa.edu.pe`), Universidad Nacional de San Agustín de Arequipa.
- **Componentes de terceros auditados en `despliegue_stm32/canonicos_42_180mhz_2026-10-27/copia_aislada/Prueba_2_CWRU_Extraccion/`:**
  - `TFLM/LICENSE`: texto íntegro oficial de **Apache License Version 2.0** (`11,358 B`).
  - `TFLM/third_party/kissfft/COPYING`: **BSD-3-Clause** (*Copyright (c) 2003-2010 Mark Borgerding*).
  - `TFLM/third_party/cmsis_nn/Include/arm_nnfunctions.h` (`V.19.1.0`): **Apache-2.0**.
  - `Middlewares/CMSIS-DSP/Include/arm_math.h` (`V1.10.0`): **Apache-2.0**.
  - `Drivers/CMSIS/Include/core_cm4.h` (`V5.1.2`): **Apache-2.0**.
  - `Drivers/CMSIS/Device/ST/STM32F4xx/LICENSE.txt`: **Apache-2.0** (u opción `Package_license`).
  - `Drivers/STM32F4xx_HAL_Driver/LICENSE.txt` y `Src/stm32f4xx_hal.c` (`V1.8.2`): **BSD-3-Clause** (u opción `Package_license`).
  - `Lib/libtensorflow-microlite.a` (`1,520,740 B`, SHA-256 `84cd3fae745ce78530e49f353725f845588ca0298cc35bd01da98403eabbe974`): coincide byte a byte con `tflite-micro-main/gen/cortex_m_generic_cortex-m4+fp_release_cmsis_nn_gcc/lib/libtensorflow-microlite.a` (pendiente de documentar receta/commit exacto y recompilación limpia en el **Frente 2**).

### Paso 1.2 — Creación y actualización de archivos legales, citación y política de datos en raíz

| Archivo | Tamaño (B) | SHA-256 (registrado antes de integrar bitácora final) | Contenido y propósito |
| :--- | ---: | :--- | :--- |
| `LICENSE` | 11,358 | `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30` | Copia exacta de Apache License 2.0 (`TFLM/LICENSE`). |
| `NOTICE` | 1,452 | `544e8c2a16c8fa256dfdc51de072ef313a02eba52be34c1dfc67a05d9fc9dd75` | Aviso de copyright 2026 de ambos autores y atribución de terceros/corpus. |
| `THIRD_PARTY_LICENSES.md` | 4,860 | `bf9dbeb2659a9ecd295fcaa12dc3d957ae548d0aebd7cb13be51f99d4dd3c317` | Tabla verificada componente a componente (TFLM, CMSIS-NN/DSP/Core, KissFFT, ST HAL, Python, CWRU y Arciniegas). |
| `CITATION.cff` | 960 | `27313d1257b3b075670279f9b275008e853c5f9fb2d2ca33f77d78f7bf84725f` | `cff-version: 1.2.0`, `v2.0.0`, `license: Apache-2.0`, ambos autores UNSA; `repository-code` y `doi` comentados hasta publicación en GitHub/Zenodo. |
| `DATA_POLICY.md` | 5,569 | `25ccc5e9561c32f1d74de8b8b666bbbaf0d35bc9734ee77596d92ad4171c01ac` | Política unificada CWRU (28 `.mat`) + Arciniegas (239 archivos), exclusión de derivados y guía de verificación SHA-256. |
| `.gitignore` | 930 | `f11cce864945119488f743e453d333c4a3bf578743b044b9c9583d87a1138883` | Ignora `datos/cwru/raw/`, `datos/arciniegas/raw/`, `*.npy`, `metadata_features*.csv`, `bancos/*.npz`, CSVs con features por ventana, `public_release/`, `.codex_tmp/`, `.venv*/`, `*.zip` y auxiliares LaTeX. |

### Paso 1.3 — Script verificador de integridad de corpus externos

- Se creó [`codigo/comun/verificar_datos_descargados.py`](../../codigo/comun/verificar_datos_descargados.py) (`8,758 B`, SHA-256 `3dbc5e883b234531b71ff34623800bd85e81fa1639893ad1693cb65463dff00f`) usando exclusivamente la biblioteca estándar de Python 3.12 (`argparse`, `hashlib`, `json`, `pathlib`, `sys`).
- Funcionalidades implementadas:
  - Lee `Nuevas Observaciones/Reproducibilidad/INVENTARIO_DATOS.json` y valida existencia, tamaño exacto en bytes y SHA-256 para los **28 `.mat` de CWRU** y los **239 archivos de Arciniegas** (**267 archivos en total**).
  - Soporta `--solo all|cwru|arciniegas`, `--raiz`, `--inventario` y `--reporte-json`.
  - Emite nota técnica automática sobre `CWRU id_mat=174` (`datos/cwru/raw/IR/174_IR_0,014.mat`), cuyas claves internas documentadas son `X173_DE_time` y `X173_FE_time`.
  - Si faltan archivos o hay discrepancias, imprime las URLs oficiales de descarga (CWRU Bearing Data Center y Springer `10.1007/s43926-025-00142-4`) y retorna código de salida `1`.
- **Nota de verificación en sandbox:** Dentro del entorno aislado actual de ejecución no es ejecutable el alias de Python de WindowsApps (`AppExecutionAlias`), por lo que la integridad de los `267/267` archivos locales contra `INVENTARIO_DATOS.json` se comprobó directamente mediante `.NET System.Security.Cryptography.SHA256` dentro de `VERIFICAR_PUBLIC_RELEASE.ps1`, mientras que la ejecución directa del script Python queda prevista para el entorno host Python 3.12 en el **Frente 3**.

### Paso 1.4 — Scripts reproducibles de construcción y verificación de `public_release/`

1. **[`documentacion/preparacion_publica/CONSTRUIR_PUBLIC_RELEASE.ps1`](CONSTRUIR_PUBLIC_RELEASE.ps1):**
   - Limpia y reconstruye desde cero `public_release/`.
   - Copia los documentos legales/metadatos y genera archivos públicos específicos sin dependencias internas (`public_release/README.md`, `public_release/REPRODUCIBILITY.md`, `public_release/.gitignore`, `public_release/datos/README.md`).
   - Excluye de manera determinista:
     - Corpus crudos (`datos/cwru/raw/**`, `datos/arciniegas/raw/**`: `267` archivos).
     - Arreglos procesados y tablas de características (`datos/*/procesado/*.npy`, `metadata_features*.csv`: `22` archivos).
     - Pruebas sintéticas `.npz`/`.npy`/`.mat` (`24` archivos) y auxiliares LaTeX (`26` archivos) en `articulo/`.
     - Bytecode Python `__pycache__/*.pyc` (`22` archivos) en `codigo/`.
     - Checkpoints intermedios `checkpoint_*.npz` (`5,667` archivos) y `.npz` de insumos/entrenamiento (`entrada_evaluacion.npz`, `datos.npz`, `banco_costos_referencia.npz`, `cuantizacion_train.npz`, `diagnosticos_train_only.npz`, `predicciones_validacion.npz`, `calibracion.npz`, etc.: `1,468` archivos) en `resultados/`.
     - CSVs de resultados con vectores de características por ventana (`probabilidades_test.csv`, `scores_test_RAW.csv`, `scores_test_STANDARD.csv`, `top_50_scores_test.csv`, `ventanas_rechazadas_P99.csv`, `muestras_ordenadas_por_margen.csv`, `ventana_*/comparacion_features.csv` y detección dinámica por cabecera CSV: `65` archivos).
     - Bancos embebidos (`despliegue_stm32/**/bancos/*.npz`: `8` archivos), objetos/dependencias GCC (`*.o`, `*.d`, `*.su`: `229` archivos) y campañas STM32 intermedias o duplicadas (`31,810` archivos, incluyendo la copia duplicada `costo_total_single_rounding_v2_180mhz_2026-10-26/copia_aislada/` y la carpeta heredada `Prueba 2/`), conservando íntegramente `canonicos_42_180mhz_2026-10-27` (sin `bancos/`), los JSON de `costo_total_single_rounding_v2_180mhz_2026-10-26` necesarios para `recalcular_evidencia_fase1.py`, y `PAQUETE_VALIDACION_REIMPLEMENTACION_STM32/firmware_stm32`.
   - **Saneamiento interno de contenedores ZIP `.npz`:** Abre los `131` archivos `outer_eval_*/outer*_*_predicciones.npz` (`80` DNN + `40` RF + `4` SVM + `7` detectores) y elimina internamente `normalized_features.npy`, `fp32_inputs.npy`, `int8_inputs.npy` y `model_features.npy` (`284` arreglos retirados: `80 × 3 + 44 × 1 + 7 × 0`), preservando intactos `y_true.npy`, `int8_predictions.npy`, `predictions.npy`, `global_window_ids.npy`, `record_ids.npy`, logits, probabilidades, scores y parámetros de cuantización requeridos por `recalcular_evidencia_fase1.py`.
   - Emite `public_release/PUBLIC_EXCLUSIONS_AND_SANITIZATION.csv` y `public_release/MANIFEST_PUBLIC_SHA256.txt`.

2. **[`documentacion/preparacion_publica/VERIFICAR_PUBLIC_RELEASE.ps1`](VERIFICAR_PUBLIC_RELEASE.ps1):**
   - Comprueba presencia de los 12 archivos obligatorios de publicación limpia.
   - Verifica `0` archivos bajo `datos/{cwru,arciniegas}/raw/`, `0` archivos `.mat`, y `0` archivos `.npy` salvo una lista cerrada explícita de **5 archivos exclusivamente paramétricos** (32 centroides K-Means, 2 vectores de normalización y 177 coeficientes FIR: `centroides_RAW.npy`, `centroides_STANDARD.npy`, `scaler_mean_STANDARD.npy`, `scaler_scale_STANDARD.npy`, `coeficientes_fir_kaiser_177_float32.npy`).
   - Revisa la cabecera de cada `.csv` copiado en `public_release/` confirmando ausencia de columnas de características por ventana (`ax_rms`, `ax_skewness`, `de_rms`, `fe_rms`, `raw_features*`, `normalized_features`, `int8_inputs`, `fp32_inputs`, `model_features`).
   - Abre e inspecciona **todos** los archivos `.npz` de `public_release/` confirmando ausencia de cualquier entrada interna prohibida y presencia de `y_true.npy`, `global_window_ids.npy` y predicciones/scores en los `131` archivos saneados.
   - Recalcula y verifica 1:1 todo `MANIFEST_PUBLIC_SHA256.txt` y valida los `267/267` archivos crudos del espacio de trabajo frente a `INVENTARIO_DATOS.json`.

### Paso 1.5 — Resultados de verificación previa al cierre final del Frente 1

Antes de redactar y re-empaquetar esta bitácora, la corrida previa de verificación reportó (`exit code = 0`, `failures = 0`):
- `manifest_files_verified = 16775` (`532.59 MB` totales; `despliegue_stm32 = 258.76 MB`, `resultados = 100.37 MB`, `PAQUETE_VALIDACION_REIMPLEMENTACION_STM32 = 79.07 MB`, `articulo = 79.02 MB`, `documentacion = 5.97 MB`, `codigo = 3.62 MB`, `datos = 1.43 MB`, `modelos = 0.60 MB`, `Nuevas Observaciones = 0.57 MB`).
- `csv_headers_checked = 512` (`0` fugas).
- `npz_archives_inspected = 771` (`131` de evaluación exterior saneados verificados, `0` entradas prohibidas).
- `workspace_raw_verified = 267 / 267`.

La reconstrucción y verificación posterior tras guardar la bitácora del Frente 1 confirmó (`exit code = 0`, `failures = 0`):
- `public_release/`: `manifest_files_verified = 16776` (`532.60 MB` antes de integrar esta bitácora al árbol base; `csv_headers_checked = 512`, `npz_archives_inspected = 771`, `workspace_raw_verified = 267 / 267`).
- `Revision_2026-11-09/MANIFEST_SHA256.txt`: `22,604` entradas verificadas sin alteraciones (`unregistered_files = 0`, `errors = 0`).

---

## Frente 2 — Trazabilidad Upstream de TFLM, Recompilación Limpia STM32 y Exclusión Segura de `.chm`

**Fecha de ejecución:** 2026-11-13  
**Estado:** Implementado y verificado end-to-end en entorno aislado, espacio de trabajo completo y `public_release/`.

### Paso 2.1 — Identificación exacta del commit upstream de `tensorflow/tflite-micro`

- **Cotejo Git blob SHA-1 contra remoto oficial (`https://github.com/tensorflow/tflite-micro`):**
  - Commit exacto identificado: **`aab6e2fdb1b22e424395939462d30ac1bb5d950c`** (`2026-09-25T22:23:58Z`, *"fix(python): remove resource variable printout (#3785)"*).
  - Coincidencia verificada: **`1,829 / 1,830`** archivos rastreados en el commit remoto son **idénticos byte a byte** a la copia local `tflite-micro-main/` (`onlyRemote = 0`, `changed = 1`).
- **Única modificación local acreditada:** `tensorflow/lite/micro/tools/make/Makefile` (`689` líneas upstream vs `691` locales), exclusivamente en las líneas `584–587`, para pasar los objetos al archivador `ar` mediante fichero de respuesta `@$(MICROLITE_LIB_PATH).rsp` y evitar desbordamiento de línea de comandos en Windows.

### Paso 2.2 — Mapa de dependencias terceras e identidad del árbol `TFLM/` (`644 / 644`)

- El árbol exportado `TFLM/` (`644` archivos) coincide **byte a byte (`644 / 644`)** entre:
  - `tflite-micro-main/TFLM/`
  - `despliegue_stm32/canonicos_42_180mhz_2026-10-27/copia_aislada/Prueba_2_CWRU_Extraccion/TFLM/`
  - `PAQUETE_VALIDACION_REIMPLEMENTACION_STM32/firmware_stm32/dependencias_compartidas/TFLM/`
- Mapa de origen completo (`unmapped = 0`): núcleo TFLM + `LICENSE` (`439`), `ARM-software/CMSIS-NN` @ `4ab83cc3cc98fb85ed6dafb55e8ca02f1628dcae` / `V.19.1.0` (`139`), `ARM-software/CMSIS_6` @ `5782d6f8057906d360f4b95ec08a2354afe5c9b9` + `Cortex_DFP` @ `c2c70a97a20fb355815e2ead3d4a40e35a4a3cdf` (`24`), `google/flatbuffers` `v25.9.23` (`30`), `google/gemmlowp` @ `719139ce755a0f31cbf1c37f7f98adcc7fc9f425` (`5`), `google/ruy` @ `d37128311b445e758136b8602d1bbd2a755e115d` (`1`), `mborgerding/kissfft` `v130` (`6`, BSD-3-Clause); total = **`644` archivos** (las versiones compiladas corresponden a `download_dependencies.py` / `third_party_downloads.inc` / `ext_libs/*.sh`, no a `MODULE.bazel`).

### Paso 2.3 — Reconstrucción limpia aislada de `TFLM/` y `libtensorflow-microlite.a` desde fuente

- **Toolchain y entorno POSIX verificado:** `C:\msys64\usr\bin` (`GNU Make 4.4.1`, `Python 3.12.14` POSIX con `os.sep == '/'`, `GNU findutils 4.11.0`, `bash/sh`) + compilador ARM **STM32CubeIDE 2.2.0** (`GNU Tools for STM32 14.3.rel1.20251027-0700`, GCC `14.3.1 20250623`) vía `TARGET_TOOLCHAIN_ROOT`.
- **Regeneración `create_tflm_tree.py`** (`exit = 0`, `52.98 s`): produjo **`644 / 644`** archivos idénticos byte a byte (`matched = 644`, `diff = 0`, `missing = 0`).
- **Compilación `make -j8 ... microlite`** (`exit = 0`, `704.49 s`):
  - Biblioteca conservada (`Lib/libtensorflow-microlite.a`): `1,520,740 B`, SHA-256 `84cd3fae745ce78530e49f353725f845588ca0298cc35bd01da98403eabbe974`.
  - Biblioteca reconstruida desde fuente: `1,520,740 B`, SHA-256 `e1631e46bee7caa8e1006b4bc6f91bbc95d2c11f91ae89f6e6b053d7877a84c2`.
  - **Cotejo miembro por miembro del archivador `ar`:** `318 / 318` registros (`316` `.o` + índices) en orden y nombre idénticos, con **`318 / 318` payloads binarios idénticos bit a bit (`diffData = 0`)**; los `317` encabezados distintos se deben únicamente a timestamps no deterministas de `ar` (`ARFLAGS := -r`).

### Paso 2.4 — Compilaciones limpias headless en STM32CubeIDE 2.2.0 (`Release_180MHz_Canon42_SR`)

Ejecuciones en directorios temporales aislados (`stm32cubeidec.exe ... -cleanBuild Prueba_2_CWRU_Extraccion/Release_180MHz_Canon42_SR`), preservando intacto el espacio de trabajo histórico:

1. **Con los 8 manuales `.chm`:** `exit = 0` (`70.41 s`; CDT `15s.984ms`, `0 errors, 0 warnings`) → `ELF` `254,412 B` SHA-256 `c2de62a6e73a7f0f4cc2f4d25f5cbef718bde28eb93e05d0273871337ce84a46` y `MAP` SHA-256 `24b4141ae00ab6e1d04da2c51ed8c502d96a19518a5431ca9ad8fc9cd863132d` **idénticos bit a bit al canónico**.
2. **Sin los 8 manuales `.chm`:** `exit = 0` (`58.24 s`; CDT `10s.359ms`, `0 errors, 0 warnings`) → **`ELF` y `MAP` idénticos bit a bit**.
3. **Reconstrucción integral *from source*** (`TFLM_rebuilt/` + `libtensorflow-microlite.a` recompilada `e1631e46...` + 0 `.chm`): `exit = 0` (`62.33 s`; CDT `14s.155ms`, `0 errors, 0 warnings`) → **mismo `ELF` (`c2de62a6...`, `254,412 B`) y mismo `MAP` (`24b4141a...`) bit a bit** (`text=184912`, `data=476`, `bss=87984`, `dec=273372`).

### Paso 2.5 — Documentación, scripts y exclusión segura de manuales `.chm` en `public_release/`

1. Se redactó [`documentacion/preparacion_publica/TFLM_Y_COMPILACION_STM32.md`](TFLM_Y_COMPILACION_STM32.md) con el registro integral de trazabilidad, versiones, receta POSIX MSYS2 + GCC `14.3.rel1`, cotejo `ar` y compilaciones STM32CubeIDE.
2. Se actualizó [`THIRD_PARTY_LICENSES.md`](../../THIRD_PARTY_LICENSES.md) sustituyendo la nota pendiente del Frente 2 por los hashes, commits, `Makefile` diff y referencia al informe técnico.
3. Se actualizó [`Nuevas Observaciones/Reproducibilidad/verificar_firmware.py`](../../Nuevas%20Observaciones/Reproducibilidad/verificar_firmware.py) añadiendo `--sin-chm` y autodetección de `MANIFEST_PUBLIC_SHA256.txt` en modo lectura (`--check`), de modo que verifica `1,889/1,889` en el espacio completo y `1,881` archivos restantes + `.elf` + `.map` en `public_release/` (ambas comprobaciones arrojaron `PASS_IDENTIDAD_LOCAL_FIRMWARE_CANONICO`, código `0`).
4. Se actualizaron [`CONSTRUIR_PUBLIC_RELEASE.ps1`](CONSTRUIR_PUBLIC_RELEASE.ps1) (exclusión y fila de auditoría `firmware_manuales_chm` para los 8 `.chm` de `117,382,070 B` / `111.94 MiB`, ajuste de `.gitignore`, `README.md` y `REPRODUCIBILITY.md` públicos) y [`VERIFICAR_PUBLIC_RELEASE.ps1`](VERIFICAR_PUBLIC_RELEASE.ps1) (prohibición explícita de `.chm`).
5. Tras reconstruir `public_release/`, la corrida de verificación antes de registrar esta bitácora reflejó una reducción de `532.60 MB` a **`413.71 MB`** sin manifiesto (`416.85 MB` totales con `MANIFEST_PUBLIC_SHA256.txt` y metadatos raíz; `despliegue_stm32` bajó de `258.76 MB` a **`143.00 MB`**; `manifest_files_verified = 16769`, `failures = 0`).

Inmediatamente después de guardar esta bitácora se ejecuta la corrida final del Frente 2 (`CONSTRUIR_PUBLIC_RELEASE.ps1`, `VERIFICAR_PUBLIC_RELEASE.ps1`, `verificar_firmware.py --raiz public_release --check` y cotejo de `Revision_2026-11-09/MANIFEST_SHA256.txt`).

---

## Frente 3 — Limpieza operativa, entorno Python bloqueado y preparación final de subida

**Fecha de ejecución:** 2026-11-13  
**Estado:** Implementado y verificado end-to-end en entorno virtual aislado (`Python 3.12.10`), espacio de trabajo completo, `public_release/` y `Revision_2026-11-09/`.

### Paso 3.1 — Entorno virtual limpio y archivo de bloqueo `requirements-lock.txt`

- Se creó un entorno virtual limpio externo (`venv_repro`) con `Python 3.12.10` (`MSC v.1943 64 bit (AMD64)`) e instaló desde cero `requirements.txt` (`joblib==1.6.0`, `matplotlib==3.10.8`, `numpy==2.4.4`, `pandas==3.0.2`, `Pillow==12.2.0`, `pyserial==3.5`, `scikit-learn==1.9.1`, `scipy==1.17.1`, `tensorflow==2.21.0`).
- Todos los módulos del proyecto (`flatbuffers==25.12.19`, `joblib==1.6.0`, `keras==3.15.1`, `matplotlib==3.10.8`, `numpy==2.4.4`, `pandas==3.0.2`, `PIL==12.2.0`, `serial==3.5`, `sklearn==1.9.1`, `scipy==1.17.1`, `tensorflow==2.21.0`) importaron sin errores y `pip check` devolvió código `0` (`No broken requirements found.`).
- Se generó [`requirements-lock.txt`](../../requirements-lock.txt) con los **`49`** paquetes transitivos exactos de `pip freeze` (registrando que `tensorflow==2.21.0` resuelve `keras==3.15.1` frente a `3.15.0` anotado en el entrenamiento histórico de `ENTORNOS_PC.json`).

### Paso 3.2 — Portabilidad y compatibilidad de verificación en `public_release/`

1. [`Nuevas Observaciones/recalcular_evidencia_fase1.py`](../../Nuevas%20Observaciones/recalcular_evidencia_fase1.py): se incorporaron `--salida` (escritura de JSON recalculado en ruta alternativa sin sobrescribir `EVIDENCIA_FASE1_CWRU.json`) y `--saneado` (autodetectado vía `MANIFEST_PUBLIC_SHA256.txt`), validando igualdad exacta de las métricas serializadas, conjunto de `91` insumos y `4` JSON sin sanear, admitiendo cambio de hash ZIP solo en los `87` contenedores `outer_eval_*/*_predicciones.npz` saneados (`80` DNN + `7` detectores).
2. [`Nuevas Observaciones/Reproducibilidad/probar_copia_limpia.py`](../../Nuevas%20Observaciones/Reproducibilidad/probar_copia_limpia.py): limpia automáticamente al finalizar su carpeta temporal `copia_*` bajo `.codex_tmp/reproducibilidad_fase2/` (con bandera opcional `--conservar-copia`).
3. [`generar_diagrama_pipeline_cwru_v2.py`](../../articulo/ampliacion_cwru/redaccion/fase3_metodo_2026-10-28/figuras/generar_diagrama_pipeline_cwru_v2.py) y [`generar_matrices_canonicas.py`](../../articulo/ampliacion_cwru/redaccion/fase4_resultados_2026-10-28/figuras_canonicas/generar_matrices_canonicas.py): eliminadas las rutas literales de usuario (`C:\Users\LENOVO\...`), reemplazándolas por importación estándar y fallback opcional vía `LOCALAPPDATA`.
4. Auditoría de portabilidad en `codigo/`: se diferenciaron los scripts portables basados en `Path(__file__)` de los orquestadores históricos de estación STM32CubeIDE/ST-Link (`D:/STM32IDE`, `D:/STM32CubeIDE_2.2.0`, `STM32_Programmer_CLI.exe`), documentados explícitamente en `REPRODUCIBILITY.md`.
5. Hallazgo sobre `consolidar_articulo_cwru_v2.py --check`: se verificó que los `24` bloques salvo uno coinciden exactamente con los cierres fuente (`source_sha256`, métricas, selecciones y latencias), difiriendo únicamente en `original_manuscript_sha256` por ediciones posteriores del borrador LaTeX (`d1cf3022...` del 28 de octubre frente a `3700f93d...`). No se modificaron cierres ni figuras históricas.

### Paso 3.3 — Saneamiento documental raíz, empaquetado final y guía de publicación

- Se reescribieron los documentos raíz obsoletos [`README.md`](../../README.md) y [`REPRODUCIBILITY.md`](../../REPRODUCIBILITY.md) (eliminando referencias previas a «reimplementación»/«sin licencia» e incorporando el título final del artículo, **Apache-2.0**, `BSD-3-Clause`, `TFLM aab6e2f`, `requirements-lock.txt` y distinción raíz vs `public_release/`).
- Se actualizó [`THIRD_PARTY_LICENSES.md`](../../THIRD_PARTY_LICENSES.md) con el listado exacto de dependencias Python bloqueadas y validadas.
- Se creó el tutorial temporal `GUIA_SUBIDA_GITHUB_ZENODO.md` (eliminado posteriormente en el Frente 5 tras completar la publicación) con instrucciones exactas para subir `public_release/` a GitHub (`< 15 MiB` máx. por archivo individual frente al límite de `100 MiB`), acuñar el DOI `v2.0.0` en Zenodo y actualizar posteriormente `CITATION.cff` y el manuscrito LaTeX en un entorno con `pdflatex` / Overleaf (ausente en este equipo).
- Se actualizaron [`CONSTRUIR_PUBLIC_RELEASE.ps1`](CONSTRUIR_PUBLIC_RELEASE.ps1) (incluyendo `requirements-lock.txt`, `.tmp_*.json` en `.gitignore` público, `README.md` y `REPRODUCIBILITY.md` actualizados) y [`VERIFICAR_PUBLIC_RELEASE.ps1`](VERIFICAR_PUBLIC_RELEASE.ps1) (obligatoriedad de `requirements-lock.txt` y guardia automática `< 100 MiB` por archivo único).

La ejecución final tras registrar esta bitácora regenera y verifica `public_release/`, revalida todas las comprobaciones Python en `venv_repro` sobre ambos árboles y confirma la integridad intacta de `Revision_2026-11-09/`.

---

## Frente 4 — Acotación estricta a código, modelos y reproducibilidad

**Fecha de ejecución:** 2026-11-13  
**Objetivo:** Atender la decisión del usuario de mantener únicamente código, modelos, firmware STM32 y artefactos necesarios para la reproducibilidad pública, excluyendo manuscritos, PDFs y borradores editoriales.

- **Exclusiones reforzadas en [`CONSTRUIR_PUBLIC_RELEASE.ps1`](CONSTRUIR_PUBLIC_RELEASE.ps1):**
  - En `articulo/`: excluidos `articulo/manuscrito/**`, `articulo/ampliacion_cwru/redaccion/**`, `articulo/ampliacion_cwru/literatura/**`, `articulo/figuras/**`, `articulo/fuentes/**`, `articulo/README.md`, `articulo/ampliacion_cwru/README_CONTINUIDAD.md` y cualquier archivo `.pdf`, `.docx`, `.xlsx`, `.tex`, `.bib`, `.pptx`, `.zip`, `.svg`, `.pyc`, `.pyo` o `__pycache__/`; se conservan únicamente scripts Python (`articulo/analisis/`), auditorías del dataset (`articulo/auditoria_dataset/`), tablas numéricas derivadas (`articulo/resultados_derivados/`) y cierres/protocolos técnicos (`articulo/metodologia/`, `articulo/ampliacion_cwru/{metodologia,auditoria,analisis_solapamiento,PLAN_MAESTRO_AMPLIACION_CWRU.md,protocolo_v2_propuesto.json}`).
  - En `documentacion/`: conservados únicamente `documentacion/preparacion_publica/`, `documentacion/cwru/` y `documentacion/consolidacion/matriz_maestra_resultados.csv`; excluidos `documentacion/investigacion/`, `documentacion/tablas_y_reproducibilidad/`, `documentacion/historico/`, `documentacion/migracion/`, `documentacion/continuidad_proyecto/` y los libros `.xlsx` de `consolidacion/`.
  - En `Nuevas Observaciones/`: conservados únicamente `recalcular_evidencia_fase1.py`, `EVIDENCIA_FASE1_CWRU.json` y `Reproducibilidad/**` (sin `.docx`, `PLAN_REVISION_08-11-2026.md` ni `__pycache__/`).
  - Guardias adicionales de copia en `codigo/`, `modelos/`, `resultados/` y `despliegue_stm32/` contra `__pycache__/`, `.py[co]`, `.pdf`, `.docx`, `.xlsx`, `.tex`, `.pptx` y `.zip`, y `.gitignore` público actualizado.
- **Guardia automática en [`VERIFICAR_PUBLIC_RELEASE.ps1`](VERIFICAR_PUBLIC_RELEASE.ps1):** falla si detecta `__pycache__/`, extensiones `.mat`, `.chm`, `.pdf`, `.docx`, `.xlsx`, `.tex`, `.bib`, `.pptx`, `.zip`, `.pyc`, `.pyo`, `.svg`, o cualquiera de las rutas editoriales/internas anteriores dentro de `public_release/`.
- **Documentación sincronizada:** actualizados [`README.md`](../../README.md), [`REPRODUCIBILITY.md`](../../REPRODUCIBILITY.md), la guía temporal de publicación (retirada tras acuñar el DOI) y los documentos generados en `public_release/`.

---

## Frente 5 — Sincronización post-publicación (GitHub, Zenodo DOI, manifiesto y fuentes LaTeX)

**Fecha de ejecución:** 2026-11-13  
**Objetivo:** Sincronizar los metadatos locales, el manifiesto SHA-256 de `public_release/` y las fuentes LaTeX del manuscrito tras la publicación del *release* `v2.0.0` en GitHub y el archivo permanente en Zenodo, y retirar el tutorial temporal de vinculación.

- **Publicación validada en línea:**
  - Repositorio público en GitHub: `https://github.com/eduardox123ollachica/Proyect-of-Investigation-` (rama `main`, licencia `Apache-2.0`, *release* `v2.0.0`).
  - Registro publicado en Zenodo: DOI de versión `10.5281/zenodo.24207867` (`https://doi.org/10.5281/zenodo.24207867`, registro `24207867`, archivo `Proyect-of-Investigation--v2.0.0.zip`) y Concept DOI `10.5281/zenodo.24207866`.
- **Sincronización y limpieza:**
  - Eliminado [`GUIA_SUBIDA_GITHUB_ZENODO.md`](GUIA_SUBIDA_GITHUB_ZENODO.md) y actualizadas sus referencias en [`README.md`](../../README.md), [`REPRODUCIBILITY.md`](../../REPRODUCIBILITY.md) y [`CONSTRUIR_PUBLIC_RELEASE.ps1`](CONSTRUIR_PUBLIC_RELEASE.ps1).
  - Actualizado [`CITATION.cff`](../../CITATION.cff) raíz con `repository-code` y `doi: "10.5281/zenodo.24207867"`.
  - Actualizados [`CONSTRUIR_PUBLIC_RELEASE.ps1`](CONSTRUIR_PUBLIC_RELEASE.ps1) y [`VERIFICAR_PUBLIC_RELEASE.ps1`](VERIFICAR_PUBLIC_RELEASE.ps1) para preservar `.git/` dentro de `public_release/` durante reconstrucciones y excluir `.git/*` del inventario `MANIFEST_PUBLIC_SHA256.txt` y del recuento de verificación.
  - Actualizada la disponibilidad de código, licencias (`Apache-2.0` / `BSD-3-Clause`), commit upstream de TFLM (`aab6e2f`), URL pública y DOI de Zenodo en [`articulo/manuscrito/articulo_ieee_borrador.tex`](../../articulo/manuscrito/articulo_ieee_borrador.tex) y en los anexos suplementarios [`indice_reproducibilidad_cwru.tex`](../../articulo/manuscrito/suplemento/indice_reproducibilidad_cwru.tex) y [`antecedente_y_datos_historicos.tex`](../../articulo/manuscrito/suplemento/antecedente_y_datos_historicos.tex) (fuentes listas para recompilar en Overleaf o entorno con `pdflatex`).
