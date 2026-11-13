# Particiones CWRU V2 por carga

Estado: preparación validada, sin entrenamiento ni métricas nuevas.
Se conserva el 50% de solapamiento, N=4096 y 12 kHz. No se movieron ni copiaron
MAT, NPY o modelos. No se modificaron los documentos/sellos de la auditoría.

| Carga de prueba | MAT desarrollo | Ventanas desarrollo | MAT prueba | Ventanas prueba |
|---|---:|---:|---:|---:|
| 0 HP | 21 | 1205 | 7 | 175 |
| 1 HP | 21 | 987 | 7 | 393 |
| 2 HP | 21 | 974 | 7 | 406 |
| 3 HP | 21 | 974 | 7 | 406 |

Cada conjunto de desarrollo se divide tres veces: dos cargas para entrenamiento
y una para validación. La carga externa queda excluida en los tres casos. Tras
seleccionar sólo dentro de desarrollo, se reajustaría con sus tres cargas; no
se ha hecho ese ajuste. Hay 4 pliegues externos y 12 internos.

## Cómo usar las tablas

- `manifiesto_registros_externo.csv`: filtrar outer_load_hp. Papeles development/outer_test.
- `manifiesto_registros_interno.csv`: filtrar outer_load_hp e inner_validation_load_hp.
  Papeles inner_train/inner_validation/outer_excluded.
- `indice_ventanas_fuente.csv`: identificador global de ventana y ruta/fila NPY
  original. Contiene 1380 ventanas únicas; ambos canales permanecen juntos.
- `indices_ventanas_externo.csv` / `indices_ventanas_interno.csv`: asignan esos
  identificadores a un papel por pliegue, no crean observaciones nuevas.
- `alcance_ajustes_por_pliegue.json`: allowlist de MAT para cada futuro fit.
- `contrato_datos_y_particiones.json`: snapshot exacto de las reglas operativas.

El orden global concatena las ubicaciones históricas train, validation, test.
**Esos nombres históricos no determinan el papel V2**: sólo indican dónde leer.
Las características no están escaladas. Cada nuevo fit requiere un scaler nuevo,
pesos de clase y calibración INT8 usando únicamente su allowlist. El detector
requiere, además, restringir su ajuste a N. No reutilizar modelos o scaler V1
como estimadores de pliegues que contienen datos vistos por ellos.

La guarda `guard_fit_scope` debe integrarse antes de los futuros ajustes. Estas
pruebas validan el contrato y sus índices; no certifican entrenamientos aún no
implementados ni pueden impedir una lectura externa que ignore deliberadamente
la guarda. La carga externa no determina early stopping, umbral, máscaras ni
hiperparámetros, y no cambia la regla de selección de posteriores pliegues.

## Límites

No hay separación verificada por rodamiento/motor. Las cargas son condiciones de
un banco; no réplicas físicas independientes. La evaluación es retrospectiva:
el corpus ya fue explorado. Las 1380 ventanas no son 1380 adquisiciones. Los
cuatro pliegues externos y las semillas tampoco son bancos independientes.
Se conserva la advertencia de diseño histórico del FIR con cargas 0/1.

No se excluyeron 201/213/238 por canales equivalentes ni 177/214/215 por
advertencias de adquisición. Cualquier sensibilidad adicional debe fijarse
antes de entrenar y sin selección según resultados externos.

## Reproducción

Desde la raíz, con Python y NumPy:

```powershell
python -B codigo/cwru/ampliacion_v2/particiones/test_particiones.py
python -B codigo/cwru/ampliacion_v2/particiones/verificar_particiones.py --directory datos/cwru/particiones/ampliacion_v2/por_carga_2026-10-22
python -B codigo/cwru/ampliacion_v2/particiones/generar_particiones.py --output datos/cwru/particiones/ampliacion_v2/repeticion
```

Una repetición exige directorio nuevo. `entradas_sha256.csv` identifica entradas
y código; `MANIFEST_SHA256.csv` sella todas las salidas salvo a sí mismo. No
regenerar un sello para ocultar discrepancias. El protocolo experimental completo
sigue pendiente de costo STM32, contribución, reglas de selección y presupuesto.
