# Búsqueda interna clásica completa y auditada

Estado: `CLASSICAL_INNER_MATRIX_COMPLETE_AUDITED_READY_FOR_SELECTION_PREPARATION`. Se cerraron384/384 ajustes,288RF y96SVM,
sin fallos o reintentos. Tiempo contabilizado: 2535.830s
(42.26min), dentro de3600s. Incluye ejecución y comprobación por
ajuste; no es todo el tiempo calendario. El piloto temporal no se repitió.

La auditoría autenticó y cargó los384modelos propios, reprodujo predicciones,
diagnósticos y métricas, contrastó roles/IDs y reconstruyó los96scalers SVM.
Los métodos fit de RF/SVC se bloquearon durante la revisión. RF conserva datos
sin escalar y SVC su votación nativa. No hubo nuevos entrenamientos, acceso a
datos exteriores por escenario, TensorFlow, hardware o cambios al manuscrito.

## Descriptivos de todas las configuraciones, sin selección

| Familia | Celdas | Macro-F1 medio (%) | Mediana (%) | Peor celda (%) |
|---|---:|---:|---:|---:|
| RF | 288 | 91.40 | 99.23 | 52.32 |
| SVM | 96 | 78.93 | 97.75 | 16.38 |

Estos promedios no son el rendimiento de modelos finales seleccionados.

| Carga de validación interna (HP) | Macro-F1 RF medio (%) | Macro-F1 SVM medio (%) |
|---|---:|---:|
| 0 | 67.49 | 26.64 |
| 1 | 99.91 | 95.35 |
| 2 | 99.25 | 97.47 |
| 3 | 98.97 | 96.24 |

La caída en0HP reaparece en ambas familias. En SVM, las24/24celdas de
validación0HP tienen recall N cero; en RF ocurre en18/72. No se atribuye aquí
una causa ni se cambia el grid para mejorar estas cifras. Las tres semillas
RF y los peores casos se conservan en el JSON; no se elige la semilla favorable.

Hay144hashes de modelos RF y48SVM: corresponden a192ámbitos de entrenamiento
compartidos entre roles outer/inner intercambiados. Se ejecutaron los384ajustes
por rol previstos; los modelos de cada par son iguales, sus validaciones y
firmas de ejecución son distintas. No son384adquisiciones independientes.

La DNN interna288/288, detector96/96 y comparadores384/384 quedan completos.
Siguiente etapa: preparar y auditar la selección con las reglas ya fijadas,
costos totales validados y guardias relativas del detector. Congelar decisiones
de los cuatro escenarios antes de reajustar y liberar resultados exteriores.
La baja especificidad absoluta del detector se conserva como limitación.
No se seleccionaron aquí configuraciones o máscaras ni se usaron predicciones
exteriores para modificar reglas.
