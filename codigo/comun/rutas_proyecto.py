"""Rutas canonicas del proyecto.

Este modulo no crea carpetas ni modifica archivos. Centraliza la estructura
para evitar que los scripts dependan del directorio desde el que se ejecutan.
"""

from pathlib import Path


RAIZ_PROYECTO = Path(__file__).resolve().parents[2]

DOCUMENTACION = RAIZ_PROYECTO / "documentacion"

DATOS = RAIZ_PROYECTO / "datos"
DATOS_ARCINIEGAS_RAW = DATOS / "arciniegas" / "raw"
DATOS_ARCINIEGAS_PROCESADO = DATOS / "arciniegas" / "procesado"
DATOS_CWRU_RAW = DATOS / "cwru" / "raw"
DATOS_CWRU_PROCESADO = DATOS / "cwru" / "procesado"

MODELOS = RAIZ_PROYECTO / "modelos"
MODELOS_ARCINIEGAS = MODELOS / "arciniegas"
MODELOS_CWRU = MODELOS / "cwru"

RESULTADOS = RAIZ_PROYECTO / "resultados"
RESULTADOS_ARCINIEGAS = RESULTADOS / "arciniegas"
RESULTADOS_CWRU = RESULTADOS / "cwru"
RESULTADOS_E1 = RESULTADOS_ARCINIEGAS / "e1"
RESULTADOS_E2 = RESULTADOS_ARCINIEGAS / "e2"
RESULTADOS_E3 = RESULTADOS_ARCINIEGAS / "e3"
RESULTADOS_E4 = RESULTADOS_ARCINIEGAS / "e4"
RESULTADOS_COMPARTIDOS = RESULTADOS_ARCINIEGAS / "componentes_compartidos"
RESULTADOS_AUDITORIA = RESULTADOS_COMPARTIDOS / "auditoria"
RESULTADOS_KMEANS = RESULTADOS_COMPARTIDOS / "kmeans"

