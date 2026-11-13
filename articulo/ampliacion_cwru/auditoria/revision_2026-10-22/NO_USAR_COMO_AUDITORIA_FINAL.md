# Ejecución de desarrollo incompleta

Esta ejecución se detuvo al comparar como cadenas los tiempos de dos CSV.
Las diferencias eran redondeos de representación decimal (orden de 1e-15 s),
no desplazamientos reales de ventanas. No se modificó ningún dato fuente.

Se corrigió únicamente el verificador para comparar los campos temporales como
números con tolerancia explícita 1e-12, manteniendo claves e índices exactos.
Se añadió una prueba negativa que rechaza desplazamientos reales y claves falsas.

Los resultados completos, sellados y citables se encuentran en la carpeta hermana
`revision_2026-10-22_validada`. Este directorio conserva sólo el rastro de la
ejecución incompleta, sin dictamen final ni sello de integridad.
