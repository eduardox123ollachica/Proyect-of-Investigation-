param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("E1", "E4", "B4")]
    [string]$Proyecto
)

$mapa = @{
    "E1" = "E1_Prueba_2"
    "E4" = "E4_Prueba_3"
    "B4" = "B4_Prueba_2_Recursos"
}

$raiz = Split-Path -Parent $MyInvocation.MyCommand.Path
$origen = Join-Path $raiz "dependencias_compartidas"
$destino = Join-Path $raiz $mapa[$Proyecto]

foreach ($nombre in @("Drivers", "Lib", "Middlewares", "TFLM")) {
    $entrada = Join-Path $origen $nombre
    $salida = Join-Path $destino $nombre

    if (Test-Path -LiteralPath $salida) {
        Write-Host "Se conserva $salida porque ya existe."
        continue
    }

    Copy-Item -LiteralPath $entrada -Destination $salida -Recurse
    Write-Host "Copiado: $nombre"
}

Write-Host "Proyecto preparado en: $destino"

