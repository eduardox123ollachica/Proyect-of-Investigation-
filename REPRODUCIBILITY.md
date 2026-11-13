# Guía de Reproducibilidad (Publicación Limpia: Código, Modelos y Firmware)

Este documento describe el flujo reproducible sobre `public_release/`, enfocado exclusivamente en código, configuraciones, modelos exportados, firmware STM32 y auditorías numéricas/de entorno, respetando la política de no redistribución de datos de terceros ([DATA_POLICY.md](DATA_POLICY.md)) y omitiendo manuscritos (`.tex`, `.pdf`, `.docx`), hojas `.xlsx` y borradores editoriales.

## Paso 0: Entorno Python bloqueado (`Python 3.12`)

- Validado en `Python 3.12.10` (`Windows x64`, `pip check` con código `0`) con [`requirements.txt`](requirements.txt) (`9` dependencias directas) y [`requirements-lock.txt`](requirements-lock.txt) (`49` dependencias transitivas bloqueadas).
- Nota respecto a `ENTORNOS_PC.json`: el registro histórico anotó `keras==3.15.0`, mientras la resolución actual de `tensorflow==2.21.0` instala `keras==3.15.1` manteniendo exactas `numpy==2.4.4`, `scikit-learn==1.9.1`, `scipy==1.17.1`, `joblib==1.6.0`, `matplotlib==3.10.8`, `pandas==3.0.2`, `Pillow==12.2.0` y `pyserial==3.5`.

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements-lock.txt
.venv\Scripts\python -m pip check
```

## Paso 1: Adquisición y verificación SHA-256 de corpus externos

1. Descargue los 28 registros `.mat` de **CWRU Bearing Data Center** (`datos/cwru/raw/`) y los 239 archivos de **Arciniegas et al. (2025)** (`datos/arciniegas/raw/`) siguiendo [DATA_POLICY.md](DATA_POLICY.md) y `Nuevas Observaciones/Reproducibilidad/README.md`.
2. Valide todos los archivos locales contra los tamaños y SHA-256 auditados en `Nuevas Observaciones/Reproducibilidad/INVENTARIO_DATOS.json`:
   ```bash
   python -B codigo/comun/verificar_datos_descargados.py
   ```

## Paso 2: Recálculo independiente de métricas de evaluación exterior (Fase 1 CWRU)

Los 80 archivos DNN `*_predicciones.npz` incluidos bajo `resultados/cwru/ampliacion_v2/evaluacion_exterior_v2_2026-10-27/20261027T222625_337707Z/outer_eval_*/` conservan `y_true`, `int8_predictions`, `global_window_ids`, `record_ids`, probabilidades, logits y parámetros de cuantización, pero fueron **saneados** eliminando `normalized_features.npy`, `fp32_inputs.npy` e `int8_inputs.npy` (y `model_features.npy` en RF/SVM).

Para recalcular y contrastar todas las métricas de macro-F1, precisión, tasas de error, costo ponderado y aporte de `3 HP`:

```bash
python -B "Nuevas Observaciones/recalcular_evidencia_fase1.py" --check
python -B "Nuevas Observaciones/recalcular_evidencia_fase1.py" --check --salida .tmp_evidencia_recalculada.json
```

> **Cómo valida `--check` en `public_release/`:** al detectar `MANIFEST_PUBLIC_SHA256.txt` (o la bandera `--saneado`), el verificador exige coincidencia exacta en todas las métricas, matrices, restricciones de `3 HP`, conjunto de `91` insumos y los `4` JSON sin sanear, permitiendo variación de contenedor ZIP únicamente en los `87` archivos `outer_eval_*/*_predicciones.npz` saneados.

## Paso 3: Verificación de firmware STM32F446RE

Para verificar la correspondencia entre el proyecto aislado `Prueba_2_CWRU_Extraccion`, el binario canónico `Release_180MHz_Canon42_SR/Prueba_2_CWRU_Extraccion.elf` y las auditorías de compilación y ejecución en hardware:

```bash
python -B "Nuevas Observaciones/Reproducibilidad/verificar_firmware.py" --check
```

`public_release/` excluye los 8 manuales binarios `.chm` de ayuda HAL de STMicroelectronics (`117,382,070 B`), que no intervienen en la compilación ni en el enlazado; el script detecta `MANIFEST_PUBLIC_SHA256.txt` (o `--sin-chm`) y verifica íntegramente los `1,881` archivos restantes, la biblioteca estática, el `.elf` y el `.map`. Para el detalle del commit upstream exacto de `tensorflow/tflite-micro` (`aab6e2fdb1b22e424395939462d30ac1bb5d950c`) y la recompilación limpia desde fuente, consulte [TFLM_Y_COMPILACION_STM32.md](documentacion/preparacion_publica/TFLM_Y_COMPILACION_STM32.md).

## Paso 4: Alcance de portabilidad de scripts

- Los scripts de verificación, preprocesamiento, entrenamiento y análisis (`codigo/`, `Nuevas Observaciones/`, `articulo/analisis/`) resuelven rutas relativas al repositorio vía `Path(__file__)`.
- Los orquestadores históricos de campaña IDE/hardware bajo `codigo/cwru/ampliacion_v2/{stm32,canonicos_stm32,canonicos_validacion,costo_total}/` referencian rutas de banco locales de la estación original (`D:/STM32IDE`, `D:/STM32CubeIDE_2.2.0`, `STM32_Programmer_CLI.exe`, UART físico) y requieren adaptar sus constantes si se ejecutan en otro equipo con placa física.

Para detalles históricos adicionales del protocolo y límites, consulte `Nuevas Observaciones/Reproducibilidad/README.md`, [DATA_POLICY.md](DATA_POLICY.md) y [PUBLIC_EXCLUSIONS_AND_SANITIZATION.csv](PUBLIC_EXCLUSIONS_AND_SANITIZATION.csv).