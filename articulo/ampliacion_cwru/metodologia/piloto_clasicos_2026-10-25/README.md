# Piloto de tiempos RF, SVM y detector — preparado para el usuario

25 de octubre de 2026. Se prepara esta etapa para completar el presupuesto PC
del protocolo V2. El usuario inicia personalmente la ejecución prolongada.
Preparar o comprobar el ejecutor no inicia ningún modelo con CWRU.

## Alcance exacto

Se conserva la cuadrícula del protocolo sellado, usando 799 ventanas de 1/2 HP
para entrenamiento y 175 de 0 HP para comprobar salidas y medir inferencia.
Las matrices de 3 HP no se abren. El trabajo DNN F14 previo se reutiliza solo
como manifiesto de filas; no se vuelve a entrenar una DNN ni se importa TensorFlow.

| Bloque | Ajustes del piloto | Configuración | Límite por proceso |
|---|---:|---|---:|
| RF | 8 | {100,300} árboles × profundidad {None,12} × hoja mínima {1,3}; semilla 11 | 60 s |
| SVM | 8 | RBF; C {0.1,1,10,100} × gamma {scale,0.01}; sin probabilidades | 30 s |
| Detector | 7 representaciones × 3 repeticiones | K=1; ajuste normal 1 HP, calibración normal 2 HP | 60 s para el bloque completo |

Son 17 procesos secuenciales. La suma de límites es **780 s = 13 minutos**,
más preparación, cierre/I/O y salida de procesos. Es un límite operativo, no
una duración medida o prometida. El primer fallo o timeout detiene la cola.
Ctrl+C durante un trabajo también detiene al hijo y conserva el estado.
No hay reintentos automáticos, force, sobrescritura o reanudación silenciosa.

RF usa F14 sin escalar y `n_jobs=1`; la [documentación de RF](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
describe el paralelismo por árboles y la semilla. SVM usa un scaler ajustado
solo a entrenamiento, con parámetros exportados y resta/división float32
explícitas. La [documentación de StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
define estadísticas de entrenamiento y escala unitaria para varianza cero.

No se calculan F1, exactitud o métricas de selección. Se conservan formas,
finitud, salidas, versiones, configuraciones, tiempos y procedencias. Los
pilotos no eligen máscaras o hiperparámetros y sus modelos no se reutilizan
como resultados confirmatorios. Los objetos RF/SVM ajustados no se exportan.

La etapa de inferencia mide `predict` y, además, `predict_proba` para RF o
`decision_function` para SVM como comprobación diagnóstica. Se conserva ese
trabajo adicional en la proyección; no representa el tiempo mínimo de una
llamada aislada a `predict`.

## Detector y eliminación de duplicados

Cada detector ajusta scaler/centroide solo con las 58 ventanas N de 1 HP y
calibra el P95 con las 58 N de 2 HP, sin reajustarse después. Su aritmética
materializa resta/división/cuadrado float32 y suma secuencial. P95 usa el
helper sellado, igual masa por registro y comparación estricta score > umbral.
Las 175 ventanas de 0 HP no intervienen en ajuste o umbral.

F14 sin columnas 4/11 y NF12 producen los mismos 12 índices. Por ello hay
ocho máscaras del clasificador y **siete representaciones únicas del detector**.
NF12 se registra como alias de F14, sin inventarle una medición adicional.
Se realizan 21 ajustes/calibraciones: tres por representación. Las repeticiones
comprueban tiempo y estabilidad binaria; no son semillas o adquisiciones nuevas.

## Proyección después de ejecutar

La salida proyectará por separado los máximos de trabajo fijados: RF 328,
SVM 100 y detector 96 internos + hasta 12 externos, antes de reutilización.
El piloto añade RF 8, SVM 8 y 21 llamadas al detector, fuera de esos conteos.
Los 32 representantes de costo total siguen incluidos en las DNN internas;
esta etapa no los entrena ni mide todavía el costo total STM32.

RF utiliza un factor n·log(n) y SVM un factor cuadrático por filas solo como
sensibilidades de ingeniería. La [documentación de SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
advierte que el ajuste escala al menos cuadráticamente; el factor escogido
no constituye una cota superior. La inferencia SVM contempla también crecimiento
proporcional de trabajo por vectores de soporte. Cambiar carga o semilla puede
alterar estos tiempos; el piloto no prueba un máximo temporal universal.

Se suman importación, datos/scaler, ajuste, inferencia e I/O; el residual del
proceso queda separado. El bloque agrupado de detector copia conservadoramente
su overhead a cada ajuste proyectado, sin dividirlo por siete/21 ni multiplicarlo
por semillas DNN. No se atribuye ahorro de caché todavía no medido. No se
declaran intervalos estadísticos, WCET, latencia STM32 o presupuesto global congelado.

## Comando para ejecutar una vez

En PowerShell, desde cualquier carpeta:

```powershell
& "C:\Users\LENOVO\AppData\Local\Microsoft\WindowsApps\python3.12.exe" -B "C:\Users\LENOVO\Desktop\Proyecto Investigación\codigo\cwru\ampliacion_v2\piloto_clasicos\ejecutar_piloto_clasicos.py" --execute
```

El entorno debe coincidir con el registrado por las pruebas. Un archivo de
ejecución única evita repetir accidentalmente el piloto. La ejecución de solo
lectura se obtiene con el mismo comando sin `--execute`.

La consola mostrará `[1/17]` hasta `[17/17]`, tiempos de cada proceso y al final
`PILOTO COMPLETADO: 17/17 procesos aprobados.` o `PILOTO DETENIDO`.
La ruta de evidencias se imprime siempre al cerrar normalmente el intento:

`resultados/cwru/ampliacion_v2/piloto_clasicos_2026-10-25/<fecha_UTC>/`

Allí quedarán `RESUMEN_PILOTO.json`, estados, logs, progreso por trabajo,
artefactos de diagnóstico y un manifiesto SHA-256. Cuando todos aprueben,
también se generarán `RESULTADOS_PARA_PRESUPUESTO.json` y
`PROYECCION_PRESUPUESTO_CLASICOS.json`. Los errores y tiempos de intentos
fallidos se conservan; no se rellenan tiempos ausentes con cero.

El manuscrito, firmware, datos, particiones y sellos históricos se mantienen.
