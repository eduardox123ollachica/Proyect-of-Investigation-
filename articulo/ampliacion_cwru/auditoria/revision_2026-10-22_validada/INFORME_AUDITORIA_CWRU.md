# Auditoría CWRU — 22 de octubre de 2026

## Dictamen ejecutivo

El subconjunto local es utilizable para una evaluación retrospectiva, controlada
y por carga, pero no está libre de problemas de adquisición ni demuestra
generalización a motores/rodamientos nuevos. El pipeline V1 seleccionó los canales
correctos: se verificaron todas sus ventanas y características desde los MAT.
No se entrenó ningún modelo, no se alteraron datos, particiones, resultados V1 ni
el manuscrito. Estado: **AUDITADO_CON_ADVERTENCIAS**.

## Inventario y balance

28 MAT, 213,775,376 bytes, 90 variables numéricas.
Tres clases N/IR/OR, cuatro cargas 0–3 HP, diámetros 0.007/0.014/0.021 pulgadas,
OR centrado a las 6. No es toda la base CWRU: no incluye clase bola ni todos los
diámetros/posiciones/colecciones. DE y FE son posiciones, no ejes X/Y/Z.

| Subconjunto histórico | Clase | MAT | Ventanas | Duración original (s, Fs documentada) |
|---|---|---:|---:|---:|
| train | N | 2 | 86 | 15.1634 |
| train | IR | 6 | 223 | 39.6925 |
| train | OR | 6 | 259 | 45.7382 |
| validation | N | 1 | 58 | 10.1055 |
| validation | IR | 3 | 174 | 30.5219 |
| validation | OR | 3 | 174 | 30.4494 |
| test | N | 1 | 58 | 10.1176 |
| test | IR | 3 | 174 | 30.4131 |
| test | OR | 3 | 174 | 30.5340 |

Fs fuente documentada 48 kHz, remuestreo a 12 kHz, N=4096, hop=2048,
341.333 ms/ventana y 170.667 ms entre decisiones. 1380 ventanas no equivalen a
1380 ensayos independientes. El archivo 174 tiene
1.3289 s y 6 ventanas;
es mucho más corto que la mayoría: registrar el desbalance, no duplicarlo.

## Hallazgos confirmados

1. **Canales redundantes:** 201, 213 y 238 (carga 0, train) tienen FE ≈
   0.984848484848485 × DE. Residuo completo máximo inferior a 1e-15. Se verificó
   con todas las muestras, no sólo Pearson. No afirmar diversidad de dos sensores
   para estos registros. Ver `equivalencias_afines.csv` y `evidencia_canales_213.png`.
2. **Variables auxiliares peligrosas:** 99 incluye copias X098; 217 incluye X215.
   175 contiene una matriz X217 y otra señal que son equivalentes, tras escala,
   a señales del test 217. Existen 7 pares
   afines cruzados con al menos una variable auxiliar. No constituyen fuga en V1:
   sus claves seleccionadas excluyen estos extras, confirmado por el recálculo.
3. **Identificador interno:** 174 usa X173; está declarado en el manifiesto.
   RPM interna y RPM tabulada aproximada se conservan por separado; no son un
   error de clase por diferir unos pocos RPM. 98/99 no contienen RPM seleccionada.
4. **Búsqueda de copias seleccionadas:** barrido exhaustivo en todos los inicios de
   bloques exactos de 16 muestras, por DE, FE y par. Consultar los tres CSV y
   `cobertura_busqueda_duplicados.json`: no extender ausencia de coincidencias a
   una afirmación de independencia. Comparación afín de registros completos:
   0 pares cruzados seleccionados.
5. **Integridad y reconstrucción:** 1380 ventanas, 19320
   características, 1352 uniones solapadas verificadas.
   Error absoluto máximo de ventanas 0; de
   características 7.1054274e-15. Se aplica tolerancia
   absoluta y relativa, no igualdad decimal arbitraria.
6. **Filtro:** 177 taps float32, ripple 0.015460 dB a 0–5 kHz,
   máximo en rechazo >=6 kHz -60.020103 dB. Compensación de 22
   muestras a 12 kHz y retirada de 22 muestras de cada extremo reconstruidas.
   Es respuesta numérica del filtro, no una medición de antialiasing analógico.

## Advertencias externas, no correcciones silenciosas

Smith y Randall (2015), DOI 10.1016/j.ymssp.2015.04.021, Tabla 3, documentan
problemas de adquisición para registros de esta selección; la redundancia afín
se confirmó localmente. Ruido eléctrico reportado en 177 y recorte reportado en
214/215 DE permanecen advertencias bibliográficas: no se estableció aquí un umbral
instrumental de saturación ni se borraron muestras. Ver `fuentes_y_alcance.md`,
`advertencias_bibliograficas.csv` y `fuentes.json`.

## Límites y decisiones pendientes

- `CW05` (WARN): ['213', '201', '238']. Evidencia: `inventario_archivos.csv; equivalencias_afines.csv`.
- `CW08` (WARN): 7. Evidencia: `equivalencias_afines.csv`.
- `CW16` (LIMITATION): 7 configuraciones nominales; identidad física no verificable; configuraciones compartidas entre cargas. Evidencia: `grupos_y_limitaciones.csv`.
- `CW17` (LIMITATION): El conjunto 3HP ya tiene resultados y fue inspeccionado; nuevo split no borra la exposición. Evidencia: `exposicion_historica.json`.
- `CW18` (LIMITATION): Fs documentada=48000 Hz; no hay reloj/timestamps/fs embebidos que permitan medición independiente. Evidencia: `inventario_variables_mat.csv; fuentes_y_alcance.md`.
- `CW19` (LIMITATION): No se descargaron ni compararon bytes oficiales en esta ejecución; procedencia por IDs y claves, no autenticación remota. Evidencia: `entradas_sha256.csv; fuentes_y_alcance.md`.
- `CW20` (WARN): Ruido eléctrico reportado en 177; recorte de DE reportado en 214 y 215. Evidencia: `advertencias_bibliograficas.csv`.

Las siete configuraciones nominales clase–diámetro son grupos descriptivos,
**no siete rodamientos físicos demostrados**. La separación por archivos/carga
evita mezclar ventanas del mismo MAT, pero no prueba independencia de montaje,
defecto o rodamiento entre cargas. Tampoco hay un test nuevo: 3 HP ya fue evaluado.
Autocorrelaciones son descriptivas; no se calculan p-valores tratando ventanas
solapadas como réplicas independientes.

## Siguiente paso recomendado, antes de reentrenar

1. Congelar la selección explícita de claves y estos hashes; conservar V1 intacta.
2. Resolver el alcance de canales: mantener DE/FE con limitación declarada y
   comparación de sensibilidad DE-solo, o establecer DE-solo como referencia
   principal y DE/FE como ampliación. No fabricar un tercer eje, ni eliminar
   automáticamente los registros redundantes por favorecer métricas.
3. Predefinir tratamiento de advertencias 177/214/215: resultado principal con
   datos intactos y, si se justifica, sensibilidad secundaria de calidad definida
   antes del entrenamiento. Sin corregir según aciertos/errores de test.
4. Fijar evaluación externa por carga y selección interna separada; declarar
   retrospectividad, usar métricas por archivo/carga y no prometer motores nuevos.
5. Sólo después habilitar los experimentos V2 del plan maestro. Esta auditoría
   no autoriza por sí misma nuevas exclusiones, cambios de etiquetas o filtros.

## Cómo comprobarlo después

Leer `registro_pruebas.csv` (IDs estables), `resumen_auditoria.json` y
`README_REPRODUCIBILIDAD.md`. Las 18 pruebas
PASS no anulan las advertencias/limitaciones. Los nueve tests sintéticos incluyen
casos negativos: clave auxiliar, canales invertidos, inicio incorrecto y archivo
alterado. `MANIFEST_SHA256.csv` sella las salidas; `entradas_sha256.csv` identifica
los datos y códigos, con comprobación antes/después sin cambios.
