$raiz = Split-Path -Parent $MyInvocation.MyCommand.Path
$manifiesto = Join-Path $raiz "MANIFEST_SHA256.txt"
$lineas = [System.Collections.Generic.List[string]]::new()

$archivos = Get-ChildItem -LiteralPath $raiz -Recurse -File |
    Where-Object { $_.FullName -ne $manifiesto } |
    Sort-Object FullName

foreach ($archivo in $archivos) {
    $hash = (Get-FileHash -LiteralPath $archivo.FullName -Algorithm SHA256).Hash.ToUpperInvariant()
    $relativa = $archivo.FullName.Substring($raiz.Length + 1).Replace([char]92, [char]47)
    $lineas.Add("$hash  $relativa")
}

Set-Content -LiteralPath $manifiesto -Value $lineas -Encoding utf8
Write-Host "Manifiesto generado para $($archivos.Count) archivos."
