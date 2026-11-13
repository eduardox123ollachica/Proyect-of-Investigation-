# Piloto acotado de tiempo V2

25 de octubre de 2026. Esta etapa ejecuta exclusivamente los dos trabajos
preparados en el protocolo V2: F14/T6, semilla 11, 799 ventanas de 1/2 HP para
entrenar y 175 de 0 HP para diagnóstico. No abre matrices de 3 HP ni modifica
particiones, manuscrito, modelos o sellos históricos.

## Qué se hace y por qué

Cada trabajo corre en un proceso CPU separado, con hilos fijados a uno y límite
de 300 segundos. Se realizan 50 épocas completas, sin parada temprana ni
selección de checkpoint: 25 lotes por época y exactamente 1250 actualizaciones.
El piloto mide el caso completo; sus pesos no se reutilizarán como evidencia
confirmatoria ni sus resultados elegirán máscaras o márgenes.

Se conserva d–20–10–3, L2 0.001, Adam 0.005, lotes de 32, pesos balanceados y
semilla 11. El scaler se ajusta solo al entrenamiento; entrenamiento,
validación y calibración INT8 comparten resta/división explícitas float32 con
parámetros exportados float32. Las etiquetas se leen de metadatos sellados.

Se verifica el manifiesto del protocolo y los hashes de los dos cachés antes de
usarlos. Solo se abren `X_features_train.npy` y `X_features_validation.npy`.
Los registros de ajuste y validación, sus IDs y sus matrices deben coincidir
exactamente con el plan. Se conservan sus procedencias y hashes.

El tracing de `tf.function` usa `get_concrete_function()` sin ejecutar un lote.
Se comprueban pesos idénticos e iteraciones del optimizador 0 antes y después;
véase la [guía oficial de TensorFlow](https://www.tensorflow.org/guide/function).
La compilación y cualquier trabajo diferido al primer lote quedan identificados
por sus etapas: el primer lote se incluye en actualizaciones, no se oculta como
warmup. Las salidas `.numpy()` sincronizan los bloques cronometrados.

Se separan importaciones, datos/escalado, preparación/tracing, actualizaciones,
diagnóstico de entrenamiento, validación por época, conversión FP32, conversión
INT8, inferencia de comprobación e I/O. El wall time del proceso se registra por
separado, porque también incluye arranque, escrituras finales y salida.

INT8 calibra exactamente las 799 filas de entrenamiento, en orden global. Se
exigen entrada/salida int8, tensores enteros y ausencia de fallback float/Flex;
los bias int32 son válidos. La inspección desactiva los delegates por defecto y
usa un hilo. La compatibilidad PC no demuestra equivalencia ARM por sí sola.

## Autorización y conservación

El campo `pilot_execution_enabled=false` del protocolo previo se conserva como
estado histórico de preparación. `EJECUCION_PILOTO_ACOTADO.json` establece esta
etapa nueva, tras el «sigamos» del usuario, limitada a dos ajustes y cuatro
conversiones. No habilita la búsqueda masiva ni la evaluación externa.

`--prepare` crea el contrato y sello nuevos sin sobrescribir. El ejecutor es
de solo lectura por defecto; `--execute` inicia el piloto. Un archivo de ejecución
única impide repetirlo accidentalmente. El primer fallo o timeout detiene la
cola; se conservan logs y artefactos y no hay reintentos automáticos.

Las salidas se guardan en
`resultados/cwru/ampliacion_v2/piloto_tiempo_2026-10-25/<fecha_UTC>/`.
Cada trabajo guarda progreso, historial, scaler, pesos, dos TFLite y diagnóstico
de conversión. El resumen y el manifiesto SHA-256 vinculan resultados y fuentes.

## Cómo se interpretará el tiempo

Los 600 segundos son el límite combinado de workers, no una duración estimada.
Dos perfiles dan una proyección de planificación; no un intervalo estadístico,
WCET o garantía temporal. Se presentarán escenarios de 25 y 50 épocas y se
escalarán por separado lotes (26/38 máximos internos/externos), filas de
calibración y validación, conservando costos fijos por ajuste.

El máximo DNN sigue siendo 288 ajustes internos y 120 externos, con hasta 816
conversiones antes de reutilizar artefactos idénticos. Los 32 representantes
temporales ya están dentro de los 288. El piloto añade dos ajustes y cuatro
conversiones. Los tiempos RF/SVM, detector y trabajo físico STM32 siguen
pendientes: esta proyección no congela el presupuesto completo.

Las [operaciones deterministas de TensorFlow](https://www.tensorflow.org/api_docs/python/tf/config/experimental/enable_op_determinism)
ayudan a controlar la ejecución con versiones y entorno fijados; no hacen
determinista la duración ni garantizan los mismos resultados entre versiones.
