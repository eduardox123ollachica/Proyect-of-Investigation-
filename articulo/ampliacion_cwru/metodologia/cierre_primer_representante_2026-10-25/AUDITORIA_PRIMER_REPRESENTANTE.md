# Primer representante V2 auditado

Resultado PASS. Se verificaron sellos, firma, IDs/etiquetas, pesos de clase,
historia ES, restauración de la época 48, modelos, salidas guardadas,
métricas, P95, detector y banco. No se reentrenó ni se ejecutó inferencia.

F14, semilla 11: ajuste 812 ventanas de 14 MAT (2/3 HP), validación
393 ventanas de 7 MAT (1 HP); carga externa 0 HP excluida. Cincuenta
épocas, 1300 actualizaciones, mejor época 48 restaurada. No se activó
parada temprana antes del límite de 50 épocas.

FP32 e INT8: exactitud y macro-F1 1.0000 en esta validación interna,
matriz diagonal 58/161/174 (N/IR/OR), cero cambios de clase y cero
valores de entrada recortados en ajuste, validación o banco. Detector:
especificidad N y recalls IR/OR 1.0000 en estos registros. Estos valores
no representan el resultado final multisemilla o externo.

Proceso con comprobación del padre: 14.577924 s; worker hasta resultado:
12.232391 s. El entrenamiento con guardado ocupó 2.800226 s. La
extrapolación lineal de los 31 es 451.915656 s (7.53 min), sensible
a importaciones, otras máscaras y parada temprana: no es una ETA garantizada.
Mantener la reserva orientativa de 15–30 min y el límite global de
7200 s acumulados. No se midió costo total STM32.

Se puede continuar la misma campaña con --resume y --limit 31. Los
fuentes sellados y el primer resultado se conservan. Véase el JSON
para la ruta y el comando completos; la ejecución la inicia el usuario.
