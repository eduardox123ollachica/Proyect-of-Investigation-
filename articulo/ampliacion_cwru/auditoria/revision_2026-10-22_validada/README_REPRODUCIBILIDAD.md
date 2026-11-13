# Reproducibilidad de esta auditoría

Desde la raíz del proyecto, con Python 3.12, NumPy, SciPy y Matplotlib:

```powershell
python -B codigo/cwru/ampliacion_v2/auditoria/test_auditoria_cwru.py
python -B codigo/cwru/ampliacion_v2/auditoria/verificar_auditoria_cwru.py --directory articulo/ampliacion_cwru/auditoria/revision_2026-10-22_validada
python -B codigo/cwru/ampliacion_v2/auditoria/auditar_dataset_cwru.py --output articulo/ampliacion_cwru/auditoria/revision_repetida
```

La repetición exige un nombre nuevo: no sobrescribe una auditoría anterior.
El verificador comprueba hashes de entradas y salidas y ejecuta las pruebas
sintéticas, pero no repite el barrido completo de señales. El tercer comando sí.
No requiere TensorFlow, conexión al STM32 ni reentrenamiento.

La ejecución original usó Python del runtime de Codex y los paquetes científicos
Python312 ya instalados por el usuario, sin instalar dependencias. Sus versiones
exactas y el ejecutable están registrados en entorno.json. Si python no encuentra
SciPy, usar ese intérprete científico o incorporar su directorio site-packages
antes de importar bibliotecas, como se hizo en esta ejecución.

Tolerancias: ventanas rtol=1e-6/atol=1e-7; características rtol=1e-9/atol=1e-8.
El FIR se recalcula con convolución directa, no llamando al generador canónico.
Las características usan momentos con corrección de sesgo y FFT por ventana,
no llaman al extractor canónico. El barrido exacto revisa todos los inicios de
bloques de 16 muestras en DE, FE y par DE/FE; todo candidato se confirma valor
por valor. No demuestra ausencia de parentesco, copias escaladas parciales,
similitud aproximada ni independencia física. Las copias afines se comparan
sólo a registro completo e igual longitud, incluidas columnas auxiliares.

MANIFEST_SHA256.csv cubre todas las salidas excepto a sí mismo. Los códigos y
entradas están cubiertos por entradas_sha256.csv. Una edición invalida el sello;
no regenerarlo para ocultar una discrepancia. Las figuras son diagnósticas y
no se insertaron en el manuscrito. Ningún archivo canónico fue modificado.
