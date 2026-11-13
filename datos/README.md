# Datos externos no redistribuidos

Conforme a `DATA_POLICY.md`, esta carpeta pública **no** contiene los archivos crudos originales ni arreglos derivados de señales/características:
- Coloque los 28 archivos `.mat` oficiales de CWRU bajo `datos/cwru/raw/{Normal,IR,OR}/`.
- Coloque los 239 archivos de exportación del suplemento de Arciniegas et al. (2025) bajo `datos/arciniegas/raw/`.
- Verifique ambos corpus ejecutando desde la raíz: `python codigo/comun/verificar_datos_descargados.py`.