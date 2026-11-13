# Reproducibilidad para la revisión del artículo

Esta carpeta reúne la evidencia de la fase 2 para las observaciones mayores 2 y 6 y para la declaración de datos del informe del 8 de noviembre. La revisión académica puede cotejar las copias y los resultados locales. La distribución pública sigue pendiente de licencia, permisos y una versión archivada con URL y DOI propios.

## Prueba desde una copia limpia

[PRUEBA_COPIA_LIMPIA.json](PRUEBA_COPIA_LIMPIA.json) registra una ejecución en un directorio temporal nuevo. Se copiaron solo el [script de análisis](../recalcular_evidencia_fase1.py) y los 91 insumos congelados del inventario, con sus rutas relativas y huellas. Son cuatro JSON y 87 NPZ. El proceso volvió a contrastar las 120 celdas INT8 y regeneró [el resultado de la fase 1](../EVIDENCIA_FASE1_CWRU.json) byte a byte. Ese archivo permite cotejar la Tabla 8 y las Tablas S21–S23. El resultado esperado se usó para comprobar la salida después de la ejecución.

Para repetir la prueba desde la raíz del proyecto o de una copia restaurada con las mismas rutas, usar Windows x64, Python 3.12.14 y NumPy 2.3.5. Este fue el entorno ejecutado. El script crea una nueva carpeta temporal, comprueba las huellas y comunica el resultado. Puede ejecutarse con el Python del entorno que tenga esa versión de NumPy.

~~~powershell
python -I -B -X utf8 "Nuevas Observaciones/Reproducibilidad/probar_copia_limpia.py"
~~~

El recálculo parte de predicciones y mediciones guardadas. La comprobación en copia limpia es de los archivos de trabajo. La instalación de las dependencias, el preprocesamiento desde los datos crudos, el entrenamiento, una reconstrucción de firmware y la repetición física son alcances separados.

## Entornos de PC

[ENTORNOS_PC.json](ENTORNOS_PC.json) agrupa las versiones declaradas en los 131 reajustes finales y las cuatro evaluaciones exteriores, con las huellas de sus registros. Distingue los reajustes DNN, RF/SVM y detector. En todos se registró Python 3.12.10 y NumPy 2.4.4. En DNN se registró TensorFlow 2.21.0 y Keras 3.15.0. La evaluación exterior registra, además, scikit-learn 1.9.1, SciPy 1.17.1 y joblib 1.6.0. Los requisitos históricos de la raíz sirven como referencia para el proyecto. Estas versiones directas no sustituyen un bloqueo de dependencias transitivas por plataforma.

La generación editorial de figuras se documentó con Python 3.12.14, NumPy 2.3.5 y Matplotlib 3.10.8. El recálculo de esta revisión utiliza Python y NumPy y parte de los archivos NPZ y JSON.

## Firmware de la validación canónica

[FIRMWARE_VERIFICADO.json](FIRMWARE_VERIFICADO.json) separa TFLM, CMSIS-NN, CMSIS-DSP, HAL y demás insumos del build, y coteja la biblioteca estática, el ELF y el MAP. La configuración es `Release_180MHz_Canon42_SR`, registrada con STM32CubeIDE 2.2.0 y GNU Tools for STM32 14.3.rel1. Los 1889 archivos registrados de la copia aislada coinciden con el manifiesto de preparación y la auditoría CDT. En la copia académica previa, las ocho ausencias respecto de ese inventario son manuales HAL en formato CHM. El [informe de compilación](../../despliegue_stm32/canonicos_42_180mhz_2026-10-27/COMPILACION_CANON42.json) y la [auditoría de metadatos](../../despliegue_stm32/canonicos_42_180mhz_2026-10-27/AUDITORIA_METADATA_CDT.json) explican las dos diferencias de configuración respecto del proyecto original. Los avisos o revisiones de cabecera identifican archivos concretos. Para atribuir un commit upstream se necesitaría verificar la identidad de la copia completa con ese commit.

~~~powershell
python -B -X utf8 "Nuevas Observaciones/Reproducibilidad/verificar_firmware.py" --check
~~~

Para **identificar el binario utilizado**, cotejar la huella del ELF con la auditoría de la validación física y conservar el MAP y el reporte de compilación asociados. Para **reconstruir el proyecto**, usar la copia aislada, la configuración indicada, los archivos de aplicación, la biblioteca TFLM precompilada y el toolchain registrado. En esa comprobación se debe generar y registrar el ELF/MAP de la reconstrucción por separado. La identidad de los binarios guardados y la reconstrucción integral son comprobaciones distintas.

## Acceso e integridad de los datos

[INVENTARIO_DATOS.json](INVENTARIO_DATOS.json) registra las rutas, tamaños y SHA-256 de ambos corpus. En CWRU incluye los identificadores y las claves exactas DE y FE.

- **CWRU.** Obtener los cuatro MAT normales desde la página de referencia y las 24 fallas de accionamiento a 48 kHz desde la tabla oficial. Seguir las etiquetas de descarga del inventario para IR y para OR en la columna centrada a las 6:00, con diámetros de 0.007, 0.014 y 0.021 pulgadas y cargas de 0 a 3 HP. Comprobar el SHA-256 y el tamaño antes de colocar cada archivo en la ruta indicada. Para el MAT 174, seguir las claves internas `X173_DE_time` y `X173_FE_time` registradas. Los 28 archivos conservados coinciden con el inventario auditado. La verificación actual corresponde a las copias locales. El acceso por descarga está documentado y no se acreditó un permiso específico de redistribución. [Datos normales](https://engineering.case.edu/bearingdatacenter/normal-baseline-data) · [Fallas a 48 kHz](https://engineering.case.edu/bearingdatacenter/48k-drive-end-bearing-fault-data)
- **Arciniegas.** Obtener el ZIP desde el Material suplementario 1 de la página del artículo. Verificar el SHA-256 del ZIP y, tras la extracción, los tamaños, rutas y huellas de los 239 archivos antes de ubicarlos en `datos/arciniegas/raw/`. La comprobación local cotejó el ZIP editorial archivado con la copia utilizada. Coincidieron los 198 JSON de entrenamiento, 37 de prueba, tres archivos de etiquetas y el README. Springer indica CC BY-NC-ND 4.0 en la declaración de derechos del artículo. Los términos aplicables a la distribución separada del corpus siguen sin acreditarse. [Artículo y ZIP editorial](https://link.springer.com/article/10.1007/s43926-025-00142-4)

En PowerShell, la huella de cada descarga se obtiene con `Get-FileHash -LiteralPath <archivo> -Algorithm SHA256`. Una diferencia se registra antes de incorporar la descarga al entorno de reproducción. Una vez colocados los archivos en las rutas del inventario, se pueden comprobar en conjunto desde la raíz:

~~~powershell
$inv = Get-Content -Raw -Encoding UTF8 -LiteralPath 'Nuevas Observaciones/Reproducibilidad/INVENTARIO_DATOS.json' | ConvertFrom-Json
$checks = @($inv.CWRU.archivos_mat)
$checks += @($inv.Arciniegas.archivos_exportacion | ForEach-Object {
    [pscustomobject]@{
        ruta_local = Join-Path $inv.Arciniegas.raiz_local $_.ruta_relativa_exportacion
        bytes = $_.bytes
        sha256 = $_.sha256
    }
})
foreach ($row in $checks) {
    $file = Get-Item -ErrorAction Stop -LiteralPath $row.ruta_local
    if ($file.Length -ne $row.bytes -or (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash -ne $row.sha256) {
        throw "Diferencia de integridad: $($row.ruta_local)"
    }
}
"$($checks.Count) archivos cotejados"
~~~

## Contenido previsto para una versión pública

| Componente | Preparación local | Condición para incluirlo |
| --- | --- | --- |
| Código propio, scripts y configuración | Rutas de código y de recálculo identificadas. El recálculo se ejecutó desde una copia limpia. | Acordar la licencia con quienes tengan los derechos y publicar la configuración exacta. |
| Fuentes de terceros, ELF/MAP y biblioteca estática | Copias y huellas identificadas. Hay avisos Apache-2.0 y BSD de terceros. Los avisos HAL y CMSIS Device remiten a la licencia del paquete ST y declaran condiciones subsidiarias. | Revisar los términos aplicables a cada componente y al binario enlazado. Adjuntar avisos y licencias que correspondan. |
| Datos originales | Inventario, enlaces oficiales e instrucciones de obtención y verificación. | La versión pública puede ofrecer las instrucciones y las huellas. La inclusión de los MAT o JSON requiere aclarar las condiciones aplicables a cada corpus. |
| Resultados derivados, modelos y bancos | Salidas, modelos y bancos identificados. Algunos NPZ de predicciones contienen características o entradas cuantizadas además de clases predichas. | Revisar por separado los derechos de distribución de cada derivado. Revisar registros, rutas locales y metadatos antes de exportar el inventario público. |
| Versión y citación | El borrador conserva ambos autores. El archivo de citación de la etapa histórica identifica otro artefacto. | Preparar los metadatos de esta versión con la atribución y licencia acordadas. Verificar las descargas, los hashes y la URL y el DOI de la versión archivada antes de citarlos en el artículo. [Versionado en Zenodo](https://help.zenodo.org/docs/deposit/manage-versions/) |
