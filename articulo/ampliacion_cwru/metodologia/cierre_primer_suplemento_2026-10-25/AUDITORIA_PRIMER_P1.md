# Auditoría del primer P1

PASS técnico: firma, roles, medianas train-only, scaler, checkpoint restaurado,
cuantización, métricas, modelos y sellos verificados. Dos referencias importadas
autenticadas con sus firmas originales. Sin entrenar o inferir durante auditoría.

F14, semilla11, ajuste1/2HP (799ventanas), validación0HP (175).
Ejecutó 11épocas y restauró la mejor (1).

| Métrica | P0 original | P1 compresión |
|---|---:|---:|
| F1 macro FP32 | 0.580998 | 0.310271 |
| F1 macro INT8 | 0.405774 | 0.261758 |
| Valores de entrada recortados | 221.000000 | 64.000000 |
| Cambios FP32–INT8 | 32.000000 | 5.000000 |

La compresión reduce recortes pero empeora el F1 en este primer par. Las28normales
siguen clasificadas OR. Menos recorte no demuestra mejor clasificación. Este dato
no justifica cambiar fórmula o detener selectivamente el diseño fijado por desempeño.
Faltan las demás semillas, máscara y cargas antes de concluir sobre P1.

Proceso y auditoría del padre: 72.583840s; entrenamiento: 1.502196s.
El intervalo importaciones/datos ocupó 65.362499s; no es tiempo aislado de importación TensorFlow.
Si los33siguientes repitieran este ritmo, sumarían 39.92min más el bloque detector.
Es una proyección lineal de un caso, no duración garantizada. La orientación inicial
15–30min puede quedarse corta si se repite este arranque. Se conserva el límite
global7200s y300s por worker; las pausas humanas no se cuentan.

Continuar la misma campaña con --resume y --limit33. No repetir el primer P1.
Al completar las34DNNnuevas se ejecuta el bloque detector (1importado+5fits).
Éxito técnico y cumplimiento predictivo se reportarán por separado.
No se cambian protocolos principales, datos, firmware o manuscrito.
