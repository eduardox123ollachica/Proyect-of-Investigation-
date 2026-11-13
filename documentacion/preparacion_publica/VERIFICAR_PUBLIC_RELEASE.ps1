$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem

$workspaceRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$publicRoot = Join-Path $workspaceRoot "public_release"

if (-not (Test-Path -LiteralPath $publicRoot)) {
    throw "Missing public_release directory at $publicRoot"
}

$failures = [System.Collections.Generic.List[string]]::new()

# 1. Required root legal/policy/metadata files
$requiredFiles = @(
    "LICENSE",
    "NOTICE",
    "THIRD_PARTY_LICENSES.md",
    "CITATION.cff",
    "DATA_POLICY.md",
    "README.md",
    "REPRODUCIBILITY.md",
    ".gitignore",
    "requirements.txt",
    "requirements-lock.txt",
    "codigo/comun/verificar_datos_descargados.py",
    "PUBLIC_EXCLUSIONS_AND_SANITIZATION.csv",
    "MANIFEST_PUBLIC_SHA256.txt"
)
foreach ($rf in $requiredFiles) {
    $p = Join-Path $publicRoot $rf.Replace('/', '\')
    if (-not (Test-Path -LiteralPath $p -PathType Leaf)) {
        $failures.Add("Missing required file: $rf")
    }
}

# 2. Check no raw or derived corpus files anywhere
$forbiddenFiles = [System.Collections.Generic.List[string]]::new()
$npzTotal = 0
$sanitizedNpzChecked = 0
$csvChecked = 0
$manifestCount = 0
$maxFileBytes = 0L
$maxFileRel = ""

$forbiddenNpzArrayPattern = "(^|/)(windows|raw|raw_features.*|raw_fit|raw_calibration|normalized_features|fp32_inputs|int8_inputs|model_features|features.*|X.*)\.npy$"
$allowedParamNpy = [System.Collections.Generic.HashSet[string]]::new([string[]]@(
    "resultados/arciniegas/componentes_compartidos/kmeans/centroides_RAW.npy",
    "resultados/arciniegas/componentes_compartidos/kmeans/centroides_STANDARD.npy",
    "resultados/arciniegas/componentes_compartidos/kmeans/scaler_mean_STANDARD.npy",
    "resultados/arciniegas/componentes_compartidos/kmeans/scaler_scale_STANDARD.npy",
    "resultados/cwru/preprocesamiento_12khz/filtro_antialias/coeficientes_fir_kaiser_177_float32.npy"
), [System.StringComparer]::Ordinal)

foreach ($full in [System.IO.Directory]::EnumerateFiles($publicRoot, "*", [System.IO.SearchOption]::AllDirectories)) {
    $rel = $full.Substring($publicRoot.Length + 1).Replace('\', '/')
    $ext = [System.IO.Path]::GetExtension($full).ToLowerInvariant()
    $name = [System.IO.Path]::GetFileName($full)
    $fileBytes = ([System.IO.FileInfo]::new($full)).Length
    if ($fileBytes -gt $maxFileBytes) {
        $maxFileBytes = $fileBytes
        $maxFileRel = $rel
    }
    if ($fileBytes -ge 104857600L) {
        $forbiddenFiles.Add("EXCEEDS_GITHUB_100MIB_LIMIT ($fileBytes B): $rel")
    }

    if ($rel -like "datos/cwru/raw/*" -or $rel -like "datos/arciniegas/raw/*") {
        $forbiddenFiles.Add("RAW_DIR: $rel")
    }
    if ($rel -match "(^|/)__pycache__(/|$)") {
        $forbiddenFiles.Add("FORBIDDEN_PYCACHE: $rel")
    }
    if ($ext -in ".mat", ".chm", ".pdf", ".docx", ".xlsx", ".tex", ".bib", ".pptx", ".zip", ".pyc", ".pyo", ".svg") {
        $forbiddenFiles.Add("FORBIDDEN_EXT($ext): $rel")
    }
    if (
        $rel -like "articulo/manuscrito/*" -or
        $rel -like "articulo/ampliacion_cwru/redaccion/*" -or
        $rel -like "articulo/ampliacion_cwru/literatura/*" -or
        $rel -like "articulo/figuras/*" -or
        $rel -like "articulo/fuentes/*" -or
        $rel -eq "articulo/README.md" -or
        $rel -eq "articulo/ampliacion_cwru/README_CONTINUIDAD.md" -or
        $rel -like "documentacion/investigacion/*" -or
        $rel -like "documentacion/tablas_y_reproducibilidad/*" -or
        $rel -like "documentacion/historico/*" -or
        $rel -like "documentacion/migracion/*" -or
        $rel -like "documentacion/continuidad_proyecto/*" -or
        ($rel -like "documentacion/consolidacion/*" -and $ext -ne ".csv") -or
        $rel -eq "Nuevas Observaciones/PLAN_REVISION_08-11-2026.md"
    ) {
        $forbiddenFiles.Add("FORBIDDEN_MANUSCRIPT_OR_EDITORIAL_SCOPE: $rel")
    }
    if ($ext -eq ".npy" -and -not $allowedParamNpy.Contains($rel)) {
        $forbiddenFiles.Add("FORBIDDEN_EXT($ext): $rel")
    }
    if ($name -like "metadata_features*.csv") {
        $forbiddenFiles.Add("FORBIDDEN_FEATURE_CSV: $rel")
    }
    if ($name -eq "comparacion_features.csv" -and $rel -match "/ventana_\d+/") {
        $forbiddenFiles.Add("FORBIDDEN_WINDOW_FEATURES_CSV: $rel")
    }
    if ($ext -eq ".csv") {
        $csvChecked++
        $reader = [System.IO.StreamReader]::new($full, [System.Text.Encoding]::UTF8, $true)
        try { $hdr = $reader.ReadLine() } finally { $reader.Close() }
        if ($hdr -and ($hdr -match "(^|[,;])(ax_rms|ax_skewness|de_rms|fe_rms|raw_features\d*|normalized_features|int8_inputs|fp32_inputs|model_features)([,;]|$)")) {
            $forbiddenFiles.Add("FORBIDDEN_CSV_HEADER($rel): $hdr")
        }
    }
    if ($ext -eq ".npz") {
        $npzTotal++
        $zip = [System.IO.Compression.ZipFile]::OpenRead($full)
        try {
            $entryNames = @($zip.Entries | ForEach-Object { $_.FullName })
            foreach ($en in $entryNames) {
                if ($en -match $forbiddenNpzArrayPattern) {
                    $forbiddenFiles.Add("FORBIDDEN_NPZ_ARRAY: $rel :: $en")
                }
            }
            if ($rel -match "/evaluacion_exterior_v2_2026-10-27/.*/outer_eval_\d+/.*_predicciones\.npz$") {
                $sanitizedNpzChecked++
                if (-not ($entryNames -contains "y_true.npy") -or -not ($entryNames -contains "global_window_ids.npy")) {
                    $forbiddenFiles.Add("BROKEN_PREDICTION_NPZ_MISSING_ID_OR_Y: $rel")
                }
                $hasPredOrScore = ($entryNames -contains "int8_predictions.npy") -or ($entryNames -contains "predictions.npy") -or ($entryNames -contains "scores.npy")
                if (-not $hasPredOrScore) {
                    $forbiddenFiles.Add("BROKEN_PREDICTION_NPZ_MISSING_PRED_OR_SCORE: $rel")
                }
            }
        } finally {
            $zip.Dispose()
        }
    }
}

foreach ($ff in $forbiddenFiles) { $failures.Add($ff) }

# 3. Verify MANIFEST_PUBLIC_SHA256.txt matches actual tree 1:1
$manifestPath = Join-Path $publicRoot "MANIFEST_PUBLIC_SHA256.txt"
if (Test-Path -LiteralPath $manifestPath) {
    $manifestLines = Get-Content -LiteralPath $manifestPath -Encoding UTF8
    $manifestMap = @{}
    $sha = [System.Security.Cryptography.SHA256]::Create()
    try {
        foreach ($line in $manifestLines) {
            if ([string]::IsNullOrWhiteSpace($line)) { continue }
            $parts = $line -split "  ", 2
            if ($parts.Count -ne 2) { $failures.Add("Malformed manifest line: $line"); continue }
            $expectedHash = $parts[0].ToLowerInvariant()
            $rel = $parts[1]
            $manifestMap[$rel] = $expectedHash
            $target = Join-Path $publicRoot $rel.Replace('/', '\')
            if (-not (Test-Path -LiteralPath $target -PathType Leaf)) {
                $failures.Add("Manifest missing file: $rel")
                continue
            }
            $fs = [System.IO.File]::OpenRead($target)
            try { $actualHash = ([System.BitConverter]::ToString($sha.ComputeHash($fs))).Replace("-", "").ToLowerInvariant() } finally { $fs.Dispose() }
            if ($actualHash -ne $expectedHash) {
                $failures.Add("Manifest hash mismatch: $rel")
            }
            $manifestCount++
        }
    } finally {
        $sha.Dispose()
    }
    $actualSet = [System.IO.Directory]::EnumerateFiles($publicRoot, "*", [System.IO.SearchOption]::AllDirectories) |
        ForEach-Object { $_.Substring($publicRoot.Length + 1).Replace('\', '/') } |
        Where-Object { $_ -ne "MANIFEST_PUBLIC_SHA256.txt" }
    foreach ($rel in $actualSet) {
        if (-not $manifestMap.ContainsKey($rel)) {
            $failures.Add("Unlisted file outside manifest: $rel")
        }
    }
}

# 4. Also verify workspace INVENTARIO_DATOS.json matches raw files 267/267 on disk (validating dataset inventory consistency)
$invPath = Join-Path $workspaceRoot "Nuevas Observaciones\Reproducibilidad\INVENTARIO_DATOS.json"
$inv = Get-Content -LiteralPath $invPath -Raw -Encoding UTF8 | ConvertFrom-Json
$rawOk = 0; $rawTotal = 0
$sha2 = [System.Security.Cryptography.SHA256]::Create()
try {
    foreach ($item in $inv.CWRU.archivos_mat) {
        $rawTotal++
        $f = Join-Path $workspaceRoot $item.ruta_local.Replace('/', '\')
        if (Test-Path -LiteralPath $f -PathType Leaf) {
            $len = ([System.IO.FileInfo]::new($f)).Length
            $fs = [System.IO.File]::OpenRead($f)
            try { $h = ([System.BitConverter]::ToString($sha2.ComputeHash($fs))).Replace("-", "").ToLowerInvariant() } finally { $fs.Dispose() }
            if ($len -eq $item.bytes -and $h -eq $item.sha256.ToLowerInvariant()) { $rawOk++ } else { $failures.Add("Raw CWRU mismatch: $($item.ruta_local)") }
        } else { $failures.Add("Raw CWRU missing: $($item.ruta_local)") }
    }
    foreach ($item in $inv.Arciniegas.archivos_exportacion) {
        $rawTotal++
        $rel = "$($inv.Arciniegas.raiz_local)/$($item.ruta_relativa_exportacion)"
        $f = Join-Path $workspaceRoot $rel.Replace('/', '\')
        if (Test-Path -LiteralPath $f -PathType Leaf) {
            $len = ([System.IO.FileInfo]::new($f)).Length
            $fs = [System.IO.File]::OpenRead($f)
            try { $h = ([System.BitConverter]::ToString($sha2.ComputeHash($fs))).Replace("-", "").ToLowerInvariant() } finally { $fs.Dispose() }
            if ($len -eq $item.bytes -and $h -eq $item.sha256.ToLowerInvariant()) { $rawOk++ } else { $failures.Add("Raw Arciniegas mismatch: $rel") }
        } else { $failures.Add("Raw Arciniegas missing: $rel") }
    }
} finally {
    $sha2.Dispose()
}

Write-Output ("VERIFY SUMMARY:")
Write-Output ("  manifest_files_verified = $manifestCount")
Write-Output ("  csv_headers_checked     = $csvChecked")
Write-Output ("  npz_archives_inspected  = $npzTotal (outer_eval sanitized checked = $sanitizedNpzChecked)")
Write-Output ("  workspace_raw_verified  = $rawOk / $rawTotal")
Write-Output ("  max_single_file_MB      = {0:N2} ({1})" -f ($maxFileBytes / 1MB), $maxFileRel)
Write-Output ("  failures                = $($failures.Count)")
Write-Output "  -- Top-level public_release folder sizes --"
$totalPubBytes = 0L
Get-ChildItem -LiteralPath $publicRoot -Directory | Sort-Object Name | ForEach-Object {
    $files = [System.IO.Directory]::GetFiles($_.FullName, "*", [System.IO.SearchOption]::AllDirectories)
    $b = 0L
    foreach ($x in $files) { $b += ([System.IO.FileInfo]::new($x)).Length }
    $totalPubBytes += $b
    Write-Output ("     {0,-42} files={1,5}  MB={2,7:N2}" -f $_.Name, $files.Count, ($b / 1MB))
}
Get-ChildItem -LiteralPath $publicRoot -File | ForEach-Object { $totalPubBytes += $_.Length }
Write-Output ("     TOTAL public_release MB = {0:N2}" -f ($totalPubBytes / 1MB))
if ($failures.Count -gt 0) {
    foreach ($fail in ($failures | Select-Object -First 40)) {
        Write-Output ("  [FAIL] $fail")
    }
    exit 1
}
Write-Output "ALL PUBLIC RELEASE SECURITY, LICENSING, SANITIZATION AND HASH CHECKS PASSED."
