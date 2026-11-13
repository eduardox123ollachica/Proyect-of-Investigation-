# Reajustes finales completos y auditados

Campaña `20261027T173452_969615Z`: **131/131 trabajos PASS**, sin fallos, reintentos
o pendientes. Se verificaron las cuatro decisiones, las fuentes, las versiones,
los payloads de datos, las firmas y los manifiestos de trabajo/paso/lote.

| Familia | Trabajos finalizados | Alcance |
|---|---:|---|
| DNN | 80 | Ocho ámbitos × diez semillas; 2400 épocas y 80 840 actualizaciones |
| RF | 40 | Cuatro configuraciones seleccionadas × diez semillas |
| SVM | 4 | Una configuración seleccionada por carga reservada |
| Detector | 7 | Normales de dos cargas para ajuste y de otra para calibración |

Tiempo acumulado de ejecución/verificación: **1132.315 s (18 min 52.315 s)**.
La suma de los relojes de ambos lotes es **1150.063 s**;
ninguno incluye el intervalo entre comandos del usuario. Los tiempos son de PC,
no latencia STM32. La auditoría posterior se mide aparte.

El inventario guarda las rutas y huellas de todos los modelos, scalers y
parámetros. El mapa asocia de antemano los tres procedimientos DNN, RF, SVM y
detector con las cuatro cargas exteriores, sin volver a elegir máscaras,
semillas, épocas o umbrales. Conserva los ocho modelos DNN canónicos con
semilla 42 y los alias entre procedimientos.

## Qué se comprobó

Los reajustes DNN usan exactamente las épocas congeladas, los pesos finales y
el orden de calibración de sus ventanas de desarrollo. Se reconstruyeron
scalers, pesos e historias, y las métricas se recalcularon desde salidas de
entrenamiento guardadas, sin volver a invocar TFLite. RF/SVM se recargaron y
reprodujeron sus predicciones nativas de entrenamiento. Se reconstruyeron
scaler, centroide y P95 del detector y se contrastó la calibración. Estas
comprobaciones no constituyen nuevos experimentos independientes.

Los diagnósticos del archivo adjunto son exclusivamente de entrenamiento.
La cercanía FP32–INT8 y las métricas altas aquí no prueban generalización.
Los contenedores INT8 de T6 y FE7 son ligeramente mayores que los FP32;
no se afirmará ahorro universal de archivo ni se equipararán bytes TFLite
con Flash del firmware. Los avisos de deprecación se conservan en los logs;
no se cambian las bibliotecas fijadas al cerrar esta campaña.

## Siguiente etapa

Preparar y comprobar un ejecutor separado para una evaluación exterior
controlada de todos los modelos y procedimientos ya fijados. Debe consumir
solo el inventario sellado, generar predicciones por ID de ventana y métricas
por clase, carga, semilla y MAT, conservar diagnósticos FP32–INT8 y del detector,
y presentar resultados completos sin seleccionar semillas favorables o
cambiar decisiones con los resultados. No se crean nuevas particiones.

Esta auditoría **no ejecutó esa evaluación**. Las cargas reservadas lo están
respecto de cada reajuste; el corpus ya se exploró anteriormente y el estudio
continúa siendo retrospectivo, sin independencia de un nuevo motor. Las
semillas y ventanas solapadas no son adquisiciones físicas independientes.

El respaldo F14 del escenario 3 sigue registrado como objetivo no alcanzado:
no cumple margen predictivo interno ni ahorro temporal. La baja especificidad
interna del detector permanece como limitación. Se evaluarán sin modificar
umbrales ni convertir el detector en sustituto de la clase DNN.

No se modificaron modelos, datos, protocolo, firmware o manuscrito. Después
de la evaluación exterior quedará validar en STM32 los pipelines canónicos
finales y consolidar la redacción con sus límites.
