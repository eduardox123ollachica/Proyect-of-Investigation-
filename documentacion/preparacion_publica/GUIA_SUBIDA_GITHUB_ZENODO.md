# Guía de Publicación en GitHub y Archivo en Zenodo (`v2.0.0`)

Esta guía describe exactamente qué subir, cómo enlazar **GitHub → Zenodo** para acuñar el DOI permanente, y qué archivos actualizar después de obtener la URL pública y el DOI.

## 1. ¿Qué carpeta se sube?

Suba **únicamente el contenido de `public_release/`** (no la raíz privada del espacio de trabajo).

`public_release/` ya fue construida y verificada **estrictamente con código fuente, modelos, firmware STM32 y artefactos de reproducibilidad**: excluye manuscritos (`.tex`, `.pdf`, `.docx`), hojas `.xlsx`, borradores editoriales, revisión bibliográfica, figuras editoriales, respaldo ZIP histórico, notas de continuidad interna, señales crudas de terceros (`datos/*/raw/`), matrices derivadas por ventana/características (`.npy`, `.mat`, `metadata_features*.csv`), `__pycache__/`, y manuales `.chm` binarios de ST (`111.94 MiB` ahorrados). Todos sus archivos individuales cumplen holgadamente el límite de GitHub (`100 MiB`) y el límite de Zenodo (`50 GB` por registro).

## 2. Lista previa ya verificada en Frente 1, Frente 2 y Frente 3

- `LICENSE` (**Apache-2.0**), `NOTICE` y `THIRD_PARTY_LICENSES.md` (**Apache-2.0** + **BSD-3-Clause** de `STM32F4xx_HAL_Driver` V1.8.2 y `KissFFT` v130).
- `DATA_POLICY.md` y `codigo/comun/verificar_datos_descargados.py` (`267/267` corpus externos verificables localmente por SHA-256).
- `CITATION.cff` (`v2.0.0`, ambos autores y filiación UNSA; listos los campos `# repository-code` y `# doi` para descomentarse tras publicar).
- `requirements.txt` y `requirements-lock.txt` (`49` paquetes validados en entorno virtual limpio con `Python 3.12.10` y `pip check`).
- Trazabilidad y recompilación limpia verificada de **TFLM** (`tensorflow/tflite-micro` @ `aab6e2fdb1b22e424395939462d30ac1bb5d950c`) y binario canónico **STM32F446RE** (`ELF` `c2de62a6...`).
- `MANIFEST_PUBLIC_SHA256.txt` y `PUBLIC_EXCLUSIONS_AND_SANITIZATION.csv` con `0` fallos en `VERIFICAR_PUBLIC_RELEASE.ps1`.

## 3. Paso A — Subir `public_release/` a un repositorio nuevo en GitHub

1. Cree un repositorio público vacío en GitHub (sin `README`, `.gitignore` ni `LICENSE` automáticos), por ejemplo:
   `https://github.com/<USUARIO_O_ORG>/<REPO_PUBLICO>`
2. Desde una terminal dentro de `public_release/`:

```powershell
cd "C:\Users\LENOVO\Desktop\Proyecto Investigación\public_release"
git init -b main
git add .
git status
git commit -m "v2.0.0: publicacion reproducible de seleccion guiada por costo en TinyML (STM32F446RE)"
git remote add origin https://github.com/<USUARIO_O_ORG>/<REPO_PUBLICO>.git
git push -u origin main
```

> **Nota:** antes de crear el *release* o la etiqueta `v2.0.0`, active primero el gancho de Zenodo en el paso siguiente para que Zenodo capture la instantánea al instante.

## 4. Paso B — Enlazar GitHub con Zenodo y emitir el Release `v2.0.0`

1. Inicie sesión en [Zenodo](https://zenodo.org/) (o [Zenodo Sandbox](https://sandbox.zenodo.org/) si desea una prueba previa) utilizando su cuenta de GitHub.
2. En **GitHub → Zenodo** (`Account → GitHub`), active el interruptor (**ON**) para `<USUARIO_O_ORG>/<REPO_PUBLICO>` **antes** de crear el *release* en GitHub.
3. En GitHub, cree el *Release* correspondiente a la etiqueta `v2.0.0`:
   - **Tag:** `v2.0.0` (sobre `main`).
   - **Title:** `v2.0.0 — Feature Selection Guided by Processing Cost in TinyML (STM32F446RE)`
   - **Description:** enlace a `REPRODUCIBILITY.md`, `DATA_POLICY.md` y `THIRD_PARTY_LICENSES.md`.
4. Espere unos minutos hasta que Zenodo muestre el depósito verde y genere el **DOI de versión** (y opcionalmente el *Concept DOI* que apunta siempre a la última versión). Zenodo lee automáticamente `CITATION.cff` para metadatos de autores, título y licencia Apache-2.0.

## 5. Paso C — Actualización posterior a la obtención de URL y DOI (fuera de este entorno sin `pdflatex`)

Una vez emitidos la URL pública del repositorio y el DOI de Zenodo:

1. **Actualizar [`CITATION.cff`](../../CITATION.cff)** (descomentar líneas `25–26`):
   - `repository-code: "https://github.com/<USUARIO_O_ORG>/<REPO_PUBLICO>"`
   - `doi: "10.5281/zenodo.<ID_ZENODO>"`
2. **Actualizar el manuscrito LaTeX únicamente en el espacio privado / Overleaf** (el repositorio público omite intencionalmente todos los `.tex` y `.pdf`):
   - [`articulo/manuscrito/articulo_ieee_borrador.tex`](../../articulo/manuscrito/articulo_ieee_borrador.tex) (sección `\section*{Datos y reproducibilidad}`, líneas `610–616`): reemplazar la frase *"Quedan pendientes la URL pública, el DOI de la versión archivada, la licencia del código propio..."* citando `Apache-2.0`, la URL de GitHub, el DOI de Zenodo y la política de corpus en `DATA_POLICY.md`.
   - [`articulo/manuscrito/suplemento/indice_reproducibilidad_cwru.tex`](../../articulo/manuscrito/suplemento/indice_reproducibilidad_cwru.tex) (líneas `44–50`): actualizar el cierre sobre `requirements-lock.txt`, el commit TFLM `aab6e2f`, la licencia y el DOI archivado.
3. **Recompilar los PDF** del artículo principal y del material suplementario en un equipo con `pdflatex` / `latexmk` o Overleaf (en la máquina actual no hay distribución TeX instalada).
4. **Si actualiza `Revision_2026-11-09/`** con el PDF final y `CITATION.cff`/`DATA_POLICY.md` con DOI, ejecute luego:
   - `Revision_2026-11-09` → regeneración de `MANIFEST_SHA256.txt` y `VALIDATE_INTEGRITY.ps1` para mantener la integridad de sus archivos archivados.
5. *(Opcional en GitHub)* reconstruir `public_release/` con `CONSTRUIR_PUBLIC_RELEASE.ps1`, verificar con `VERIFICAR_PUBLIC_RELEASE.ps1` y publicar un parche `v2.0.1` en GitHub/Zenodo incorporando únicamente el [`CITATION.cff`](../../CITATION.cff) actualizado con la URL y el DOI (manteniendo fuera del repositorio público las fuentes `.tex` y los `.pdf`).
