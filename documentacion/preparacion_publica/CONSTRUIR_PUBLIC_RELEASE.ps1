$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem

$workspaceRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$publicRoot = Join-Path $workspaceRoot "public_release"

Write-Output "Workspace root: $workspaceRoot"
Write-Output "Public release: $publicRoot"

if (Test-Path -LiteralPath $publicRoot) {
    Write-Output "Cleaning previous public_release/ staging directory (preserving .git)..."
    Get-ChildItem -LiteralPath $publicRoot -Force | Where-Object { $_.Name -ne ".git" } | Remove-Item -Recurse -Force
} else {
    [System.IO.Directory]::CreateDirectory($publicRoot) | Out-Null
}

$auditRows = [System.Collections.Generic.List[object]]::new()
$sanitizedCount = 0
$removedNpzEntriesTotal = 0

function Add-AuditRow {
    param([string]$Category, [string]$Action, [string]$PathOrPattern, [int]$Count, [string]$Reason)
    $auditRows.Add([PSCustomObject]@{
        categoria = $Category
        accion = $Action
        ruta_o_patron = $PathOrPattern
        cantidad_archivos = $Count
        motivo = $Reason
    })
}

function Copy-FileSafe {
    param([string]$Src, [string]$Dst)
    $parent = [System.IO.Path]::GetDirectoryName($Dst)
    if (-not [System.IO.Directory]::Exists($parent)) {
        [System.IO.Directory]::CreateDirectory($parent) | Out-Null
    }
    [System.IO.File]::Copy($Src, $Dst, $true)
}

function Sanitize-PredictionNpz {
    param([string]$DstPath)
    $stripNames = [System.Collections.Generic.HashSet[string]]::new(
        [string[]]@("normalized_features.npy", "fp32_inputs.npy", "int8_inputs.npy", "model_features.npy"),
        [System.StringComparer]::OrdinalIgnoreCase
    )
    $removed = 0
    $zip = [System.IO.Compression.ZipFile]::Open($DstPath, [System.IO.Compression.ZipArchiveMode]::Update)
    try {
        $toDelete = @($zip.Entries | Where-Object { $stripNames.Contains($_.FullName) })
        foreach ($entry in $toDelete) {
            $entry.Delete()
            $removed++
        }
    } finally {
        $zip.Dispose()
    }
    return $removed
}

function Header-HasPerWindowFeatures {
    param([string]$CsvPath)
    $reader = [System.IO.StreamReader]::new($CsvPath, [System.Text.Encoding]::UTF8, $true)
    try {
        $firstLine = $reader.ReadLine()
    } finally {
        $reader.Close()
    }
    if ([string]::IsNullOrWhiteSpace($firstLine)) { return $false }
    if ($firstLine -match "(^|[,;])(ax_rms|ax_skewness|de_rms|fe_rms|raw_features\d*|normalized_features|int8_inputs|fp32_inputs|model_features)([,;]|$)") {
        return $true
    }
    return $false
}

# 1. Copy root legal, policy, citation & requirements files
$rootFiles = @(
    "LICENSE",
    "NOTICE",
    "THIRD_PARTY_LICENSES.md",
    "CITATION.cff",
    "DATA_POLICY.md",
    "requirements.txt",
    "requirements-lock.txt",
    "MODEL_HASHES.sha256",
    "GENERAR_MANIFEST.ps1"
)
foreach ($rf in $rootFiles) {
    $src = Join-Path $workspaceRoot $rf
    if (-not (Test-Path -LiteralPath $src)) { throw "Missing required root file: $src" }
    Copy-FileSafe -Src $src -Dst (Join-Path $publicRoot $rf)
}

# 2. Write clean public .gitignore, README.md, REPRODUCIBILITY.md and datos/README.md
$publicGitIgnore = @'
# Corpus crudos externos (no redistribuibles)
datos/arciniegas/raw/
datos/cwru/raw/
datos/**/raw/*.mat
datos/**/raw/*.json

# Arreglos procesados y tablas con características por ventana
datos/*/procesado/*.npy
datos/*/procesado/**/*.npy
datos/*/procesado/metadata_features.csv
datos/*/procesado/**/metadata_features*.csv
**/bancos/*.npz
resultados/**/probabilidades_test.csv
resultados/**/scores_test_*.csv
resultados/**/top_50_scores_test.csv
resultados/**/ventanas_rechazadas_P99.csv
resultados/**/muestras_ordenadas_por_margen.csv

# Python y entornos locales
__pycache__/
*.py[cod]
*$py.class
.venv*/
venv*/
.tmp_*.json

# Manuscritos, documentos ofimáticos y auxiliares LaTeX (excluidos de la publicación)
*.pdf
*.docx
*.xlsx
*.tex
*.bib
*.pptx
articulo/**/*.aux
articulo/**/*.log
articulo/**/*.out
articulo/**/*.toc

# Artefactos intermedios de compilación STM32CubeIDE y manuales CHM
*.o
*.d
*.su
*.chm
'@
[System.IO.File]::WriteAllText((Join-Path $publicRoot ".gitignore"), ($publicGitIgnore -replace "`r`n", "`n"), [System.Text.UTF8Encoding]::new($false))

$publicReadme = @'
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
- `documentacion/`: documentación técnica de CWRU (`documentacion/cwru/`), matriz consolidada (`documentacion/consolidacion/matriz_maestra_resultados.csv`), bitácora y trazabilidad técnica ([`TFLM_Y_COMPILACION_STM32.md`](documentacion/preparacion_publica/TFLM_Y_COMPILACION_STM32.md), [`BITACORA_PASO_A_PASO.md`](documentacion/preparacion_publica/BITACORA_PASO_A_PASO.md)); sin archivos `.docx`/`.xlsx`, borradores de investigación, ZIPs históricos ni notas internas.

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
5. **Publicación, DOI y citación académica:** [CITATION.cff](CITATION.cff) · GitHub (`https://github.com/eduardox123ollachica/Proyect-of-Investigation-`) · Zenodo `v2.0.0` (`https://doi.org/10.5281/zenodo.24207867`) · Concept DOI (`https://doi.org/10.5281/zenodo.24207866`).
'@
[System.IO.File]::WriteAllText((Join-Path $publicRoot "README.md"), ($publicReadme -replace "`r`n", "`n"), [System.Text.UTF8Encoding]::new($false))

$publicRepro = @'
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
'@
[System.IO.File]::WriteAllText((Join-Path $publicRoot "REPRODUCIBILITY.md"), ($publicRepro -replace "`r`n", "`n"), [System.Text.UTF8Encoding]::new($false))

$datosReadme = @'
# Datos externos no redistribuidos

Conforme a `DATA_POLICY.md`, esta carpeta pública **no** contiene los archivos crudos originales ni arreglos derivados de señales/características:
- Coloque los 28 archivos `.mat` oficiales de CWRU bajo `datos/cwru/raw/{Normal,IR,OR}/`.
- Coloque los 239 archivos de exportación del suplemento de Arciniegas et al. (2025) bajo `datos/arciniegas/raw/`.
- Verifique ambos corpus ejecutando desde la raíz: `python codigo/comun/verificar_datos_descargados.py`.
'@
Copy-FileSafe -Src (Join-Path $workspaceRoot "DATA_POLICY.md") -Dst (Join-Path $publicRoot "DATA_POLICY.md")
[System.IO.Directory]::CreateDirectory((Join-Path $publicRoot "datos")) | Out-Null
[System.IO.File]::WriteAllText((Join-Path $publicRoot "datos\README.md"), ($datosReadme -replace "`r`n", "`n"), [System.Text.UTF8Encoding]::new($false))

# 3. Copy codigo/ (excluding __pycache__, *.pyc, office/doc files)
$exCodigo = 0
$srcCodigo = Join-Path $workspaceRoot "codigo"
foreach ($f in [System.IO.Directory]::EnumerateFiles($srcCodigo, "*", [System.IO.SearchOption]::AllDirectories)) {
    $rel = $f.Substring($workspaceRoot.Length + 1).Replace('\', '/')
    $ext = [System.IO.Path]::GetExtension($f).ToLowerInvariant()
    if ($rel -match "(__pycache__|\.py[co]$)" -or $ext -in ".pdf", ".docx", ".xlsx", ".tex", ".pptx", ".zip") { $exCodigo++; continue }
    Copy-FileSafe -Src $f -Dst (Join-Path $publicRoot $rel.Replace('/', '\'))
}
if ($exCodigo -gt 0) { Add-AuditRow "codigo" "excluido" "codigo/**/__pycache__, *.pyc" $exCodigo "Cache bytecode Python" }

# 4. Copy modelos/ (excluding __pycache__, *.pyc, office/doc files)
$exModelos = 0
$srcModelos = Join-Path $workspaceRoot "modelos"
foreach ($f in [System.IO.Directory]::EnumerateFiles($srcModelos, "*", [System.IO.SearchOption]::AllDirectories)) {
    $rel = $f.Substring($workspaceRoot.Length + 1).Replace('\', '/')
    $ext = [System.IO.Path]::GetExtension($f).ToLowerInvariant()
    if ($rel -match "(__pycache__|\.py[co]$)" -or $ext -in ".pdf", ".docx", ".xlsx", ".tex", ".pptx", ".zip") { $exModelos++; continue }
    Copy-FileSafe -Src $f -Dst (Join-Path $publicRoot $rel.Replace('/', '\'))
}
if ($exModelos -gt 0) { Add-AuditRow "modelos" "excluido" "modelos/**/{__pycache__,*.pyc,*.pdf,*.docx,*.zip}" $exModelos "Caches y documentos no requeridos" }

# 5. Copy Nuevas Observaciones/ (strictly reproducibility code/manifests; exclude Word, editorial revision plan, __pycache__)
$exNuevasEd = 0; $exNuevasCache = 0
$srcNuevas = Join-Path $workspaceRoot "Nuevas Observaciones"
foreach ($f in [System.IO.Directory]::EnumerateFiles($srcNuevas, "*", [System.IO.SearchOption]::AllDirectories)) {
    $rel = $f.Substring($workspaceRoot.Length + 1).Replace('\', '/')
    $ext = [System.IO.Path]::GetExtension($f).ToLowerInvariant()
    $name = [System.IO.Path]::GetFileName($f)
    if ($rel -match "(__pycache__|\.py[co]$)" -or $name -like "~`$*" -or $name -like "*.tmp") { $exNuevasCache++; continue }
    if ($ext -in ".docx", ".pdf", ".xlsx", ".tex", ".zip", ".pptx" -or $rel -eq "Nuevas Observaciones/PLAN_REVISION_08-11-2026.md") {
        $exNuevasEd++; continue
    }
    Copy-FileSafe -Src $f -Dst (Join-Path $publicRoot $rel.Replace('/', '\'))
}
if ($exNuevasEd -gt 0) { Add-AuditRow "Nuevas_Observaciones_editorial" "excluido" "Nuevas Observaciones/{*.docx,PLAN_REVISION_08-11-2026.md}" $exNuevasEd "Informes Word de observaciones y plan editorial del manuscrito" }
if ($exNuevasCache -gt 0) { Add-AuditRow "Nuevas_Observaciones_cache" "excluido" "Nuevas Observaciones/**/__pycache__, *.pyc, temporales" $exNuevasCache "Cache bytecode Python y archivos temporales" }

# 6. Copy documentacion/ (strictly technical CWRU, consolidacion/matriz_maestra_resultados.csv, and preparacion_publica/)
$exDocsEditorial = 0; $exDocsInterno = 0; $exDocsMigracion = 0; $exDocsCache = 0
$srcDocs = Join-Path $workspaceRoot "documentacion"
foreach ($f in [System.IO.Directory]::EnumerateFiles($srcDocs, "*", [System.IO.SearchOption]::AllDirectories)) {
    $rel = $f.Substring($workspaceRoot.Length + 1).Replace('\', '/')
    $ext = [System.IO.Path]::GetExtension($f).ToLowerInvariant()
    $name = [System.IO.Path]::GetFileName($f)
    if ($rel -match "(__pycache__|\.py[co]$)" -or $name -like "~`$*" -or $name -like "*.tmp") { $exDocsCache++; continue }
    if ($rel -like "documentacion/investigacion/*" -or $rel -like "documentacion/tablas_y_reproducibilidad/*" -or ($rel -like "documentacion/consolidacion/*" -and $ext -ne ".csv")) {
        $exDocsEditorial++; continue
    }
    if ($rel -like "documentacion/continuidad_proyecto/*" -or $rel -like "documentacion/historico/*") {
        $exDocsInterno++; continue
    }
    if ($rel -like "documentacion/migracion/*") {
        $exDocsMigracion++; continue
    }
    if ($ext -in ".pdf", ".docx", ".xlsx", ".tex", ".bib", ".pptx", ".zip") {
        $exDocsEditorial++; continue
    }
    Copy-FileSafe -Src $f -Dst (Join-Path $publicRoot $rel.Replace('/', '\'))
}
if ($exDocsEditorial -gt 0) { Add-AuditRow "documentacion_editorial" "excluido" "documentacion/{investigacion,tablas_y_reproducibilidad,consolidacion/*.xlsx}" $exDocsEditorial "Documentos Word, borradores de investigacion y libros Excel no requeridos para codigo/reproducibilidad" }
if ($exDocsInterno -gt 0) { Add-AuditRow "documentacion_interna" "excluido" "documentacion/{continuidad_proyecto,historico}" $exDocsInterno "Prompts de continuidad interna y planes historicos descartados" }
if ($exDocsMigracion -gt 0) { Add-AuditRow "documentacion_migracion" "excluido" "documentacion/migracion/{*.zip,*.csv,*.md}" $exDocsMigracion "Respaldo historico ZIP pre-migracion y manifiestos internos de traslado" }
if ($exDocsCache -gt 0) { Add-AuditRow "documentacion_cache" "excluido" "documentacion/**/{__pycache__,*.tmp,~$*}" $exDocsCache "Temporales y cache" }

# 7. Copy articulo/ (strictly Python analysis, dataset audits, derived tables and technical methodology closures; NO manuscript/PDF/DOCX/LaTeX/redaccion/literatura/figuras)
$exArtNpz = 0; $exArtCache = 0; $exArtManuscript = 0; $exArtEditorial = 0; $exArtFigs = 0
$srcArt = Join-Path $workspaceRoot "articulo"
foreach ($f in [System.IO.Directory]::EnumerateFiles($srcArt, "*", [System.IO.SearchOption]::AllDirectories)) {
    $rel = $f.Substring($workspaceRoot.Length + 1).Replace('\', '/')
    $ext = [System.IO.Path]::GetExtension($f).ToLowerInvariant()
    $name = [System.IO.Path]::GetFileName($f)
    if ($rel -match "(__pycache__|\.py[co]$)" -or $name -like "~`$*" -or $name -like "*.tmp") { $exArtCache++; continue }
    if ($ext -in ".aux", ".log", ".toc", ".out", ".gz") { $exArtCache++; continue }
    if ($ext -in ".npz", ".npy", ".mat") { $exArtNpz++; continue }
    if ($rel -like "articulo/manuscrito/*") { $exArtManuscript++; continue }
    if ($rel -like "articulo/ampliacion_cwru/redaccion/*" -or $rel -like "articulo/ampliacion_cwru/literatura/*" -or $rel -like "articulo/fuentes/*" -or $rel -eq "articulo/README.md" -or $rel -eq "articulo/ampliacion_cwru/README_CONTINUIDAD.md") {
        $exArtEditorial++; continue
    }
    if ($rel -like "articulo/figuras/*") { $exArtFigs++; continue }
    if ($ext -in ".pdf", ".docx", ".xlsx", ".tex", ".bib", ".pptx", ".zip", ".svg") {
        $exArtManuscript++; continue
    }
    Copy-FileSafe -Src $f -Dst (Join-Path $publicRoot $rel.Replace('/', '\'))
}
Add-AuditRow "articulo_manuscrito_y_pdf" "excluido" "articulo/{manuscrito/**,**/*.pdf,**/*.tex,**/*.docx,**/*.zip}" $exArtManuscript "Fuentes LaTeX del manuscrito/suplemento, PDFs compilados, DOCX y archivos auxiliares de edicion"
Add-AuditRow "articulo_redaccion_y_literatura" "excluido" "articulo/{ampliacion_cwru/redaccion/**,literatura/**,fuentes/**,README*.md}" $exArtEditorial "Borradores de redaccion editorial, revision bibliografica y notas de continuidad"
Add-AuditRow "articulo_figuras_editoriales" "excluido" "articulo/figuras/{*.png,*.svg}" $exArtFigs "Exportaciones graficas preparadas para el manuscrito (los scripts fuente permanecen en articulo/analisis/)"
Add-AuditRow "articulo_npz_desarrollo" "excluido" "articulo/**/*.{npz,npy,mat}" $exArtNpz "Pruebas sinteticas e insumos binarios de desarrollo en metodologia"
Add-AuditRow "articulo_cache_y_aux" "excluido" "articulo/**/{__pycache__,*.pyc,*.aux,*.log,*.toc,*.out}" $exArtCache "Cache bytecode Python y auxiliares temporales de compilacion LaTeX"

# 8. Copy safe CSV/metadata from datos/ (exclude raw/, *.mat, *.npy, *.npz, metadata_features*.csv)
$exDatosRaw = 0; $exDatosDerived = 0
$srcDatos = Join-Path $workspaceRoot "datos"
foreach ($f in [System.IO.Directory]::EnumerateFiles($srcDatos, "*", [System.IO.SearchOption]::AllDirectories)) {
    $rel = $f.Substring($workspaceRoot.Length + 1).Replace('\', '/')
    $ext = [System.IO.Path]::GetExtension($f).ToLowerInvariant()
    $name = [System.IO.Path]::GetFileName($f)
    if ($rel -like "datos/cwru/raw/*" -or $rel -like "datos/arciniegas/raw/*") {
        $exDatosRaw++; continue
    }
    if ($ext -in ".npy", ".npz", ".mat" -or $name -like "metadata_features*.csv") {
        $exDatosDerived++; continue
    }
    if ($ext -eq ".csv" -and (Header-HasPerWindowFeatures -CsvPath $f)) {
        $exDatosDerived++; continue
    }
    Copy-FileSafe -Src $f -Dst (Join-Path $publicRoot $rel.Replace('/', '\'))
}
Add-AuditRow "datos_raw" "excluido" "datos/{cwru,arciniegas}/raw/**" $exDatosRaw "Senales crudas originales de terceros (28 MAT CWRU + 239 archivos Arciniegas)"
Add-AuditRow "datos_derivados" "excluido" "datos/*/procesado/{*.npy,metadata_features*.csv}" $exDatosDerived "Matrices de ventanas, etiquetas en .npy y tablas de caracteristicas por ventana"

# 9. Copy resultados/ (exclude checkpoint_*.npz, raw/input .npz, feature CSVs; sanitize evaluacion_exterior_v2/*_predicciones.npz)
$exResChk = 0; $exResNpz = 0; $exResCsv = 0
$forbiddenResNpzNames = [System.Collections.Generic.HashSet[string]]::new([string[]]@(
    "banco_costos_referencia.npz", "cuantizacion_train.npz", "datos.npz", "diagnosticos_train_only.npz",
    "entrada_evaluacion.npz", "predicciones_validacion.npz", "diagnostico_entrenamiento.npz",
    "diagnostico_conversion_validacion.npz", "salidas_diagnostico.npz", "calibracion.npz",
    "detector_calibracion.npz", "detector_validacion.npz", "validacion.npz",
    "scores_DE7.npz", "scores_F14.npz", "scores_FE7.npz", "scores_NE12.npz",
    "scores_NM10.npz", "scores_S8.npz", "scores_T6.npz"
), [System.StringComparer]::OrdinalIgnoreCase)

$outerPredPattern = "^resultados/cwru/ampliacion_v2/evaluacion_exterior_v2_2026-10-27/20261027T222625_337707Z/outer_eval_[0-3]/.*_predicciones\.npz$"
$srcRes = Join-Path $workspaceRoot "resultados"
foreach ($f in [System.IO.Directory]::EnumerateFiles($srcRes, "*", [System.IO.SearchOption]::AllDirectories)) {
    $rel = $f.Substring($workspaceRoot.Length + 1).Replace('\', '/')
    $ext = [System.IO.Path]::GetExtension($f).ToLowerInvariant()
    $name = [System.IO.Path]::GetFileName($f)
    if ($rel -match "(__pycache__|\.py[co]$)" -or $ext -in ".pdf", ".docx", ".xlsx", ".tex", ".pptx", ".zip" -or $name -like "~`$*" -or $name -like "*.tmp") { continue }
    if ($name -like "checkpoint_*.npz") { $exResChk++; continue }
    if ($forbiddenResNpzNames.Contains($name)) { $exResNpz++; continue }
    if ($name -eq "comparacion_features.csv" -and $rel -match "/validacion_stm32_10_ventanas/ventana_\d+/") {
        $exResCsv++; continue
    }
    if ($ext -eq ".csv" -and (Header-HasPerWindowFeatures -CsvPath $f)) {
        $exResCsv++; continue
    }
    $dst = Join-Path $publicRoot $rel.Replace('/', '\')
    Copy-FileSafe -Src $f -Dst $dst
    if ($rel -match $outerPredPattern) {
        $sanitizedCount++
        $removed = Sanitize-PredictionNpz -DstPath $dst
        $removedNpzEntriesTotal += $removed
    }
}
Add-AuditRow "resultados_checkpoints" "excluido" "resultados/**/checkpoint_*.npz" $exResChk "Checkpoints intermedios de busqueda interna"
Add-AuditRow "resultados_npz_insumos" "excluido" "resultados/**/{entrada_evaluacion,datos,cuantizacion_train,banco_costos_referencia,diagnosticos_train_only,predicciones_validacion,...}.npz" $exResNpz "Arreglos .npz con caracteristicas, entradas cuantizadas o volcados diagnosticos de entrenamiento"
Add-AuditRow "resultados_csv_features" "excluido" "resultados/**/{probabilidades_test,scores_test_*,top_50_scores_test,ventanas_rechazadas_P99,muestras_ordenadas_por_margen,ventana_*/comparacion_features}.csv" $exResCsv "CSVs que incluyen columnas de caracteristicas extraidas por ventana"
Add-AuditRow "resultados_npz_saneados" "saneado" "resultados/.../outer_eval_*/outer*_*_predicciones.npz" $sanitizedCount "Eliminados normalized_features.npy, fp32_inputs.npy, int8_inputs.npy y model_features.npy ($removedNpzEntriesTotal arreglos retirados); se conservan y_true, predicciones, logits, probabilidades e IDs"

# 10. Copy firmware / STM32 canonical deployment (excluding bancos/, *.o, *.d, *.su, *.chm, intermediate STM32 runs, root Prueba 2)
$exStmBancos = 0; $exStmObj = 0; $exStmChm = 0; $exStmOther = 0
$stmAllowedDirs = @(
    "despliegue_stm32\canonicos_42_180mhz_2026-10-27",
    "despliegue_stm32\costo_total_single_rounding_v2_180mhz_2026-10-26",
    "PAQUETE_VALIDACION_REIMPLEMENTACION_STM32\firmware_stm32"
)
foreach ($d in $stmAllowedDirs) {
    $srcDir = Join-Path $workspaceRoot $d
    foreach ($f in [System.IO.Directory]::EnumerateFiles($srcDir, "*", [System.IO.SearchOption]::AllDirectories)) {
        $rel = $f.Substring($workspaceRoot.Length + 1).Replace('\', '/')
        $ext = [System.IO.Path]::GetExtension($f).ToLowerInvariant()
        if ($rel -match "^despliegue_stm32/costo_total_single_rounding_v2_180mhz_2026-10-26/copia_aislada/") {
            $exStmOther++; continue
        }
        if ($rel -match "(__pycache__|\.py[co]$)" -or $ext -in ".pdf", ".docx", ".xlsx", ".tex", ".pptx", ".zip") { continue }
        if ($rel -match "/bancos/") { $exStmBancos++; continue }
        if ($ext -in ".npz", ".npy", ".mat") { $exStmBancos++; continue }
        if ($ext -in ".o", ".d", ".su") { $exStmObj++; continue }
        if ($ext -eq ".chm") { $exStmChm++; continue }
        Copy-FileSafe -Src $f -Dst (Join-Path $publicRoot $rel.Replace('/', '\'))
    }
}
# Count excluded legacy firmware folders for audit visibility
foreach ($sub in (Get-ChildItem -LiteralPath (Join-Path $workspaceRoot "despliegue_stm32") -Directory)) {
    if ($sub.Name -notin @("canonicos_42_180mhz_2026-10-27", "costo_total_single_rounding_v2_180mhz_2026-10-26")) {
        $exStmOther += ([System.IO.Directory]::GetFiles($sub.FullName, "*", [System.IO.SearchOption]::AllDirectories)).Count
    }
}
$p2Legacy = Join-Path $workspaceRoot "Prueba 2"
if (Test-Path -LiteralPath $p2Legacy) {
    $exStmOther += ([System.IO.Directory]::GetFiles($p2Legacy, "*", [System.IO.SearchOption]::AllDirectories)).Count
}
Add-AuditRow "firmware_bancos" "excluido" "despliegue_stm32/**/bancos/*.npz" $exStmBancos "Bancos de ventanas crudas y vectores de referencia (banco_raw.npz, referencia_*.npz)"
Add-AuditRow "firmware_objetos" "excluido" "despliegue_stm32/**/*.{o,d,su}" $exStmObj "Objetos y dependencias intermedias de compilador GCC"
Add-AuditRow "firmware_manuales_chm" "excluido" "despliegue_stm32/**/Drivers/STM32F4xx_HAL_Driver/*.chm" $exStmChm "Manuales de ayuda HTML compilada CHM de STMicroelectronics (117,382,070 B; no intervienen en compilacion ni enlazado)"
Add-AuditRow "firmware_campanas_intermedias" "excluido" "despliegue_stm32/{cache_potencias,prueba_o3,...} + Prueba 2/" $exStmOther "Instantaneas experimentales intermedias y copia redundante de ST HAL; reemplazado por canonicos_42_180mhz y PAQUETE_VALIDACION/firmware_stm32"

# 11. Export PUBLIC_EXCLUSIONS_AND_SANITIZATION.csv
$csvAuditPath = Join-Path $publicRoot "PUBLIC_EXCLUSIONS_AND_SANITIZATION.csv"
$auditRows | Export-Csv -LiteralPath $csvAuditPath -NoTypeInformation -Encoding UTF8

# 12. Generate MANIFEST_PUBLIC_SHA256.txt sorted deterministically
$manifestPath = Join-Path $publicRoot "MANIFEST_PUBLIC_SHA256.txt"
$sha = [System.Security.Cryptography.SHA256]::Create()
$lines = [System.Collections.Generic.List[string]]::new()
$allPublicFiles = [System.IO.Directory]::EnumerateFiles($publicRoot, "*", [System.IO.SearchOption]::AllDirectories) |
    ForEach-Object { $_.Substring($publicRoot.Length + 1).Replace('\', '/') } |
    Where-Object { $_ -ne "MANIFEST_PUBLIC_SHA256.txt" -and $_ -notlike ".git/*" } |
    Sort-Object

$totalPublicBytes = 0L
foreach ($rel in $allPublicFiles) {
    $full = Join-Path $publicRoot $rel.Replace('/', '\')
    $totalPublicBytes += ([System.IO.FileInfo]::new($full)).Length
    $fs = [System.IO.File]::OpenRead($full)
    try {
        $hashBytes = $sha.ComputeHash($fs)
    } finally {
        $fs.Dispose()
    }
    $hex = ([System.BitConverter]::ToString($hashBytes)).Replace("-", "").ToLowerInvariant()
    $lines.Add("$hex  $rel")
}
$sha.Dispose()
[System.IO.File]::WriteAllLines($manifestPath, $lines, [System.Text.UTF8Encoding]::new($false))

Write-Output ("BUILD COMPLETE: files={0} | size_MB={1:N2} | sanitized_npz={2} | removed_npz_arrays={3}" -f $lines.Count, ($totalPublicBytes / 1MB), $sanitizedCount, $removedNpzEntriesTotal)
