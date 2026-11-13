# Selección de representación y escalado para CWRU

## Decisión metodológica

Para conservar la lógica del pipeline reproducido de Arciniegas, cada ventana CWRU se representa mediante los mismos siete descriptores por señal: RMS, asimetría, curtosis de Fisher, energía espectral, frecuencia dominante, amplitud dominante y entropía espectral.

En CWRU se emplean dos posiciones de acelerómetro, no tres ejes ortogonales:

- DE: acelerómetro ubicado en el extremo de accionamiento (*drive end*).
- FE: acelerómetro ubicado en el extremo del ventilador (*fan end*).

La concatenación `[DE (7), FE (7)]` produce 14 características por ventana. El clasificador mantiene las capas ocultas del experimento anterior y adapta solamente la entrada: arquitectura 14--20--10--3.

## Protocolo de selección

Se compararon dos variantes con la partición ya fijada y sin cargar el conjunto de prueba:

1. `raw`: las 14 características sin escalado.
2. `standard`: `StandardScaler` ajustado exclusivamente con entrenamiento y aplicado después a validación.

Ambas variantes utilizaron semilla 42, pesos de clase balanceados, regularización L2 de 0.001, Adam con tasa de aprendizaje 0.005, lotes de 32, hasta 50 épocas y parada temprana con paciencia 10. La selección se definió por F1 macro de validación y, en caso de empate, por exactitud.

## Resultado de validación

| Variante | Épocas | Mejor época | Pérdida de validación | Exactitud | Precisión macro | Recall macro | F1 macro |
|---|---:|---:|---:|---:|---:|---:|---:|
| Sin escalado | 40 | 30 | 2.462097 | 0.852217 | 0.914530 | 0.885057 | 0.881536 |
| StandardScaler | 50 | 48 | 0.047331 | 0.995074 | 0.996212 | 0.996169 | 0.996168 |

La matriz de confusión de validación de la variante seleccionada fue:

| Clase real | N | IR | OR |
|---|---:|---:|---:|
| N | 58 | 0 | 0 |
| IR | 0 | 174 | 0 |
| OR | 0 | 2 | 172 |

Por tanto, se selecciona `standard` para la continuación del pipeline. El escalador guardado fue ajustado solo con entrenamiento.

## Comprobación multisemilla

La comparación pareada se repitió desde cero con las semillas predefinidas 11, 22, 33, 42, 44, 55, 66, 77, 88 y 99. Se mantuvieron constantes la partición, las características, la arquitectura, los pesos de clase y los hiperparámetros. Cada semilla se ejecutó en un proceso aislado, con un límite de 300 segundos y guardado incremental. Las diez ejecuciones finalizaron sin fallos ni tiempos excedidos.

| Variante | Exactitud media ± DE | IC 95 % de la media | F1 macro medio ± DE | IC 95 % de la media |
|---|---:|---:|---:|---:|
| Sin escalado | 0.668966 ± 0.145407 | [0.564948, 0.772983] | 0.661908 ± 0.209421 | [0.512097, 0.811718] |
| StandardScaler | 0.995074 ± 0.000000 | [0.995074, 0.995074] | 0.996168 ± 0.000000 | [0.996168, 0.996168] |

STANDARD superó a RAW en las diez comparaciones pareadas. La diferencia STANDARD--RAW fue de 0.326108 en exactitud, con IC t del 95 % [0.222091, 0.430126], y de 0.334261 en F1 macro, con IC t del 95 % [0.184450, 0.484071]. La prueba exacta bilateral de signos produjo $p=0.001953$ en ambas métricas. Dado el número reducido de semillas y el uso de una sola partición de validación, estos contrastes se interpretan como evidencia interna del protocolo y no como generalización a otros conjuntos de datos.

Las diez redes STANDARD produjeron exactamente las mismas decisiones discretas sobre validación: 404 aciertos y las mismas dos ventanas OR clasificadas como IR. RAW, en cambio, varió entre 0.428571 y 0.852217 de exactitud y entre 0.200000 y 0.881536 de F1 macro. Esta estabilidad respalda congelar el uso de `StandardScaler` ajustado solo con entrenamiento.

## Alcance de la conclusión

Esta es una decisión de diseño basada en validación, no el resultado final del modelo. El conjunto de prueba permaneció sin cargar durante la comparación y deberá utilizarse una sola vez después de congelar las restantes decisiones del pipeline. La repetición multisemilla evalúa la sensibilidad a la inicialización dentro de una partición fija; no sustituye una evaluación externa ni elimina la dependencia entre ventanas procedentes de un mismo archivo.

## Incidencia de ejecución y trazabilidad

Una ejecución inicial completó las métricas RAW, pero quedó bloqueada al serializar el modelo completo en formato Keras. Ese archivo parcial se conservó como `modelo.keras.incompleto_no_usar` para evitar confundirlo con un artefacto válido. Se sustituyó esa operación por el guardado explícito de los pesos en NPZ y su estructura en JSON. La ejecución corregida terminó ambas variantes y produjo los archivos de selección final.

Los resultados reproducibles se encuentran en `resultados/cwru/modelo_features_14/seleccion_escalado_seed42/`.
