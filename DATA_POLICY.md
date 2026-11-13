# Política de Datos, Procedencia y Verificación de Integridad

Los autores **no redistribuyen** públicamente los conjuntos de datos de vibración de terceros ni matrices derivadas que permitan reconstruir señales crudas o tablas completas de características por ventana. En el espacio de trabajo local y en la distribución pública (`public_release/`), la separación entre software/métricas y datos de terceros se rige por las reglas siguientes.

## 1. Exclusión estricta en el repositorio público (`public_release/`)

Quedan **excluidos** de `public_release/` (e ignorados en `.gitignore`):

1. **Señales crudas originales:**
   - `datos/cwru/raw/` (28 registros `.mat` del CWRU Bearing Data Center).
   - `datos/arciniegas/raw/` (239 archivos `.json`, `info.labels` y `README.txt` exportados del suplemento de Arciniegas et al.).
2. **Matrices procesadas de ventanas, características o entradas cuantizadas:**
   - Todo `*.npy` bajo `datos/*/procesado/` (`X.npy`, `X_train.npy`, `X_test.npy`, `X_validation.npy`, `X_features_*.npy`, `y*.npy`).
   - Tablas CSV con vectores de características por ventana (`metadata_features*.csv`, `probabilidades_test.csv`, `scores_test_RAW.csv`, `scores_test_STANDARD.csv`, `top_50_scores_test.csv`, `ventanas_rechazadas_P99.csv`, `muestras_ordenadas_por_margen.csv`).
   - Bancos de referencia embebida (`despliegue_stm32/**/bancos/*.npz`, `ventanas_*.npz`) y artefactos intermedios con entradas/características (`entrada_evaluacion.npz`, `datos.npz`, `banco_costos_referencia.npz`, `cuantizacion_train.npz`, `diagnosticos_train_only.npz`, `predicciones_validacion.npz`).
3. **Política de saneamiento para predicciones de evaluación exterior (`evaluacion_exterior_v2_2026-10-27`):**
   - Los 80 archivos `*_predicciones.npz` de redes DNN y los 8 de RF/SVM en `resultados/cwru/ampliacion_v2/evaluacion_exterior_v2_2026-10-27/20261027T222625_337707Z/outer_eval_*/` se incluyen en `public_release/` **saneados**: se conservaron las etiquetas verdaderas (`y_true.npy`), identificadores (`global_window_ids.npy`, `record_ids.npy`), predicciones (`int8_predictions.npy`, `fp32_predictions.npy`, `predictions.npy`), probabilidades/logits/valores y parámetros de cuantización, **eliminando** internamente `normalized_features.npy`, `fp32_inputs.npy`, `int8_inputs.npy` y `model_features.npy`.
   - De este modo, `python "Nuevas Observaciones/recalcular_evidencia_fase1.py"` puede recalcular todas las métricas sin redistribuir características o entradas cuantizadas de las ventanas del corpus. Al variar los bytes del contenedor ZIP tras retirar esas 3 entradas, los SHA-256 archivados para esos `.npz` en `fuentes.archivos` dentro de `EVIDENCIA_FASE1_CWRU.json` corresponden al paquete completo original y no a las copias saneadas, mientras que los valores métricos recalculados son idénticos.

## 2. Corpus externos y obtención desde fuentes oficiales

### A. Case Western Reserve University (CWRU) Bearing Data Center (28 registros `.mat`)

- **Fuente oficial:**
  - Normal Baseline (`Normal_0` a `Normal_3`): `https://engineering.case.edu/bearingdatacenter/normal-baseline-data`
  - 48k Drive-End Bearing Fault Data (`IR007`, `IR014`, `IR021`, `OR007@6`, `OR014@6`, `OR021@6` en cargas `0` a `3 HP`): `https://engineering.case.edu/bearingdatacenter/48k-drive-end-bearing-fault-data`
  - Descarga general: `https://engineering.case.edu/bearingdatacenter/download-data-file`
- **Términos acreditados:** `INVENTARIO_DATOS.json` registra `"terminos_redistribucion_especificos_acreditados": false`. El usuario debe descargar los 28 archivos directamente desde CWRU y ubicarlos bajo `datos/cwru/raw/` siguiendo las rutas declaradas en `Nuevas Observaciones/Reproducibilidad/INVENTARIO_DATOS.json` (`datos/cwru/raw/Normal/`, `IR/`, `OR/`).
- **Nota técnica conocida en CWRU (`id_mat = 174`):** El archivo `datos/cwru/raw/IR/174_IR_0,014.mat` contiene internamente las claves `X173_DE_time` y `X173_FE_time` en lugar de `X174_*`, tal como está documentado en `INVENTARIO_DATOS.json`.

### B. Arciniegas et al. (2025) — Conjunto de vibración tri-axial en ventilador (239 archivos)

- **Fuente editorial:** Artículo en Springer *Engineering Archive / Discover* (`https://link.springer.com/article/10.1007/s43926-025-00142-4`), **Supplementary Material 1** (`43926_2025_142_MOESM1_ESM.zip` / `ventilador-prueba-export.zip`, SHA-256 del ZIP archivado: `c03cf7de1e6ad1de7d5b544844e2a6a42ee66128c3493fded95b41940d54587a`).
- **Términos acreditados:** `INVENTARIO_DATOS.json` registra `"licencia_de_datos_separada_acreditada": false`. El usuario debe obtener el suplemento desde Springer y descomprimir sus 239 archivos dentro de `datos/arciniegas/raw/` respetando `info.labels`, `README.txt`, `training/` y `testing/`.

## 3. Verificación automática de descarga e integridad SHA-256

Una vez descargados ambos corpus (o uno de ellos) en sus carpetas locales, ejecute desde la raíz del proyecto:

```powershell
python codigo/comun/verificar_datos_descargados.py
```

Opciones útiles:
- Verificar solo un corpus: `python codigo/comun/verificar_datos_descargados.py --solo cwru` o `--solo arciniegas`.
- Exportar informe JSON: `python codigo/comun/verificar_datos_descargados.py --reporte-json verificacion_datos.json`.

El script contrasta cada archivo contra tamaño en bytes y SHA-256 esperados en `Nuevas Observaciones/Reproducibilidad/INVENTARIO_DATOS.json` (28 `.mat` de CWRU y 239 archivos de Arciniegas) y devuelve código de salida `0` solo cuando todos los archivos requeridos están presentes e íntegros.
