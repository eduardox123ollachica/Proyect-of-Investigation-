# Cuantización INT8 del clasificador CWRU

## Alcance

Se congeló como instancia canónica la red STANDARD con semilla 42 y arquitectura 14--20--10--3. La elección de la semilla conserva la convención predefinida del proyecto; no se seleccionó después de observar la prueba. La conversión y todas las comprobaciones de esta fase utilizaron entrenamiento y validación. El conjunto de prueba no fue cargado.

## Conversión

- El `StandardScaler` fue ajustado exclusivamente con las 568 ventanas de entrenamiento.
- Las 568 representaciones estandarizadas de entrenamiento se utilizaron como conjunto representativo.
- Se restringió el conversor a operaciones `TFLITE_BUILTINS_INT8`.
- La entrada y la salida del modelo cuantizado son `int8`.
- La ruta directa `from_keras_model` resultó incompatible con la combinación local de TensorFlow/Keras. Se exportó en su lugar una función TensorFlow concreta con entrada fija `[1,14]`. Esta adaptación no modifica arquitectura, pesos ni inferencia.

La red contiene 543 parámetros: 300 en la primera capa, 210 en la segunda y 33 en la salida.

## Parámetros de interfaz INT8

- Forma de entrada: `[1,14]`.
- Escala de entrada: `0.033691830933094025`.
- Punto cero de entrada: `-9`.
- Forma de salida: `[1,3]`.
- Escala de salida: `0.00390625`.
- Punto cero de salida: `-128`.
- Operadores: tres `FULLY_CONNECTED` y `SOFTMAX`.

Para cada característica original, el firmware deberá aplicar primero el escalado de entrenamiento y después la cuantización de entrada:

```text
z_i = (x_i - mean_i) / scale_i
q_i = clip(round(z_i / 0.033691830933094025 - 9), -128, 127)
```

La salida se recupera como `p_i=(q_i+128)/256`.

## Resultado en validación

| Implementación | Exactitud | Precisión macro | Recall macro | F1 macro |
|---|---:|---:|---:|---:|
| Keras FP32 | 0.995074 | 0.996212 | 0.996169 | 0.996168 |
| TFLite FP32 | 0.995074 | 0.996212 | 0.996169 | 0.996168 |
| TFLite INT8 | 0.995074 | 0.996212 | 0.996169 | 0.996168 |

No hubo cambios de clase entre Keras FP32, TFLite FP32 e INT8 en las 406 ventanas. La diferencia absoluta media de las salidas TFLite FP32--INT8 fue 0.00132875 y la máxima 0.00818628. Estas cifras demuestran equivalencia de decisión en validación, no identidad numérica ni equivalencia todavía en STM32.

## Artefactos congelados

- TFLite FP32: 4480 bytes; SHA-256 `bccd4de81af78fa2d86d495cf065f0ba5e2692f68c6af899790e8bd313fea78a`.
- TFLite INT8: 4064 bytes; SHA-256 `cabc6e967b91691287c0cd8b899958d6784f0712b9bc956f68565273c9652596`.
- Script reproducible: `codigo/cwru/entrenamiento/cuantizar_validar_int8_features14.py`.
- Resultados: `resultados/cwru/modelo_features_14/canonico_seed42_int8/`.

La evaluación de prueba permanece pendiente hasta congelar el resto del pipeline, incluido el detector K-means que mantiene la correspondencia metodológica con el trabajo base.
