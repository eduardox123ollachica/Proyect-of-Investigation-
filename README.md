# Selección de características guiada por el costo de procesamiento en TinyML: evaluación de vibraciones en ARM Cortex-M4F

Repositorio público reproducible del proyecto de clasificación de fallas en rodamientos en **STM32F446RE (ARM Cortex-M4F)** con **TensorFlow Lite for Microcontrollers (TFLM)**. Esta distribución pública contiene **únicamente código fuente, configuraciones, modelos exportados, firmware embebido STM32, métricas consolidadas y scripts de reproducibilidad**, excluyendo manuscritos (`.tex`, `.pdf`, `.docx`), borradores editoriales, revisiones bibliográficas, figuras del artículo, respaldos históricos y datos crudos o derivados por ventana de terceros.

**Autores:** Eduardo Andres Ollachica Arias (`eollachica@unsa.edu`) y Luis Moises Vergara Chaucha (`lvergara@unsa.edu.pe`) — Universidad Nacional de San Agustín de Arequipa.

## 1. Licencias

- **Código, scripts, configuración y modelos propios:** [Apache License 2.0](LICENSE) y [NOTICE](NOTICE).
- **Componentes embebidos de terceros:** véase [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md). En particular, `STM32F4xx_HAL_Driver` (V1.8.2) y `KissFFT` v130 se distribuyen bajo **BSD-3-Clause**, mientras que TFLM (`aab6e2f`), CMSIS-Core, CMSIS-DSP, CMSIS-NN, FlatBuffers, gemmlowp, ruy y CMSIS Device STM32F4xx declaran **Apache-2.0**.
- **Conjuntos de datos externos:** **NO redistribuidos** en esta publicación; véase [DATA_POLICY.md](DATA_POLICY.md).

## 2. Estructura del paquete público

- `codigo/`: scripts de procesamiento, entrenamiento, evaluación exterior y verificación (`codigo/comun/verificar_datos_descargados.py`).
- `modelos/`: modelos cuantizados/Keras/C exportados (`.tflite`, `.keras`, `.c`, `.cpp`, `.h`).
- `despliegue_stm32/canonicos_42_180mhz_2026-10-27/`: proyecto firmware canónico CWRU (`copia_aislada/Prueba_2_CWRU_Extraccion/`), tablas/catálogos de validación y trazas en hardware STM32F446RE (sin `bancos/*.npz` ni los 8 manuales `.chm` HAL de ST).
- `despliegue_stm32/costo_total_single_rounding_v2_180mhz_2026-10-26/`: medición consolidada de ciclos/costos totales en placa (`COSTOS_TOTALES_SR_VALIDADOS.json`).
- `PAQUETE_VALIDACION_REIMPLEMENTACION_STM32/firmware_stm32/`: código firmware histórico de contraste sobre Arciniegas (E1/E4/B4).
- `resultados/`: métricas JSON/CSV consolidadas y predicciones saneadas de evaluación exterior (`outer_eval_*/*_predicciones.npz` sin `normalized_features.npy`, `fp32_inputs.npy`, `int8_inputs.npy` ni `model_features.npy`).
- `datos/`: únicamente índices estructurales de partición y ventanas (sin señales `.mat`/`.json`, sin `.npy` y sin `metadata_features*.csv`).
- `articulo/`: únicamente auditorías de corpus, scripts Python de análisis (`articulo/analisis/`), tablas derivadas CSV/JSON y cierres metodológicos/protocolos de `ampliacion_cwru` (sin manuscritos `.tex`/`.pdf`/`.docx`, sin redacción, literatura ni figuras editoriales).
- `Nuevas Observaciones/`: scripts y evidencias numéricas de auditoría reproducible (`recalcular_evidencia_fase1.py`, `EVIDENCIA_FASE1_CWRU.json`, `Reproducibilidad/verificar_firmware.py`, `Reproducibilidad/probar_copia_limpia.py`, `Reproducibilidad/INVENTARIO_DATOS.json`; sin documentos Word ni planes editoriales).
- `documentacion/`: documentación técnica de CWRU (`documentacion/cwru/`), matriz consolidada (`documentacion/consolidacion/matriz_maestra_resultados.csv`), bitácora y guías técnicas ([`TFLM_Y_COMPILACION_STM32.md`](documentacion/preparacion_publica/TFLM_Y_COMPILACION_STM32.md), [`GUIA_SUBIDA_GITHUB_ZENODO.md`](documentacion/preparacion_publica/GUIA_SUBIDA_GITHUB_ZENODO.md), [`BITACORA_PASO_A_PASO.md`](documentacion/preparacion_publica/BITACORA_PASO_A_PASO.md)); sin archivos `.docx`/`.xlsx`, borradores de investigación, ZIPs históricos ni notas internas.

## 3. Verificación rápida (entorno Python 3.12 bloqueado)

1. **Instalar dependencias en entorno virtual limpio** (`requirements-lock.txt` fija los `49` paquetes transitivos verificados con `pip check`):
   ```bash
   python -m venv .venv
   # En Windows: .venv\Scripts\python -m pip install -r requirements-lock.txt
   pip install -r requirements-lock.txt
   ```
2. **Verificar datos externos descargados** (después de obtener CWRU y Arciniegas según [DATA_POLICY.md](DATA_POLICY.md)):
   ```bash
   python codigo/comun/verificar_datos_descargados.py
   ```
3. **Recalcular y contrastar evidencia de Fase 1 a partir de predicciones saneadas y costos congelados**:
   ```bash
   python -B "Nuevas Observaciones/recalcular_evidencia_fase1.py" --check --salida .tmp_recalculo.json
   ```
4. **Verificar identidad del firmware canónico STM32**:
   ```bash
   python -B "Nuevas Observaciones/Reproducibilidad/verificar_firmware.py" --check
   ```
5. **Publicación, DOI y citación académica:** véase [CITATION.cff](CITATION.cff) y [`GUIA_SUBIDA_GITHUB_ZENODO.md`](documentacion/preparacion_publica/GUIA_SUBIDA_GITHUB_ZENODO.md).