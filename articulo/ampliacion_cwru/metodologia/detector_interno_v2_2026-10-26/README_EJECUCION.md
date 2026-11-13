# Completar el detector interno CWRU V2

Preparación sin ajustes nuevos. La matriz tiene96filas lógicas;36 se conservan
y60 faltan. F14/NF12 comparten índices y parámetros: son84evaluaciones únicas,
30 conservadas y54 nuevas. Los parámetros son42ámbitos únicos,23 conservados
y19nuevos. No se repiten por semilla. Las firmas originales se conservan;
cuando un detector histórico no tenía firma individual se citan sus hashes y
su manifiesto de cobertura, sin inventar una firma de ejecución.

Se usa el núcleo original K=1: scaler/centroide con normales de la carga menor
de cada par, P95 con normales de la mayor, sin reajuste posterior. La validación
usa otra carga completa N/IR/OR. Umbral observado float32, CDF con igual masa
por MAT y rechazo estricto score>umbral. No utiliza el scaler ni errores DNN.

```powershell
& "C:\Users\LENOVO\AppData\Local\Microsoft\WindowsApps\python3.12.exe" -B "C:\Users\LENOVO\Desktop\Proyecto Investigación\codigo\cwru\ampliacion_v2\detector_interno\ejecutar_detector.py" --run
```

Un worker secuencial, CPU e hilos limitados; máximo300s de proceso, sin
reintentos automáticos. Reserva orientativa3–10min incluyendo preflight y
comprobaciones. El bloque previo de5ajustes+6evaluaciones duró8.275s; escalar
por número de operaciones daría54.9s,
pero mezcla tareas distintas y no garantiza duración. La preparación y la
verificación del padre pueden añadir tiempo fuera del límite del worker.

Guarda parámetros, IDs, calibración, scores, decisiones y métricas por MAT
incrementalmente. Un fallo o interrupción conserva el intento y bloquea repetir
con este comando; requiere auditoría de recuperación. No eliminar locks o marcador
único. Al terminar imprime `DETECTOR_INTERNAL_COMPLETED: 96/96 filas; 42/42 parametros`.
No necesita placa, COM, modelos nuevos DNN o cambios al manuscrito.

Este bloque no selecciona máscaras ni ejecuta las guardias del selector o
evaluación exterior. El mínimo absoluto0.95 del suplemento es descriptivo;
no reemplaza las guardias relativas a F14 del protocolo principal. Las tasas
deficientes se conservan como resultados, no se corrigen cambiando umbrales.
