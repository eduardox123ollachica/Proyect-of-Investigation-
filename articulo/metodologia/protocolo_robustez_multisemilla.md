# Protocolo de robustez multisemilla

## Objetivo

Completar el factorial E1--E4 para estimar si los efectos observados del
escalado y de la ponderación de clases son estables frente a la aleatoriedad
del entrenamiento y no dependen de una única inicialización.

## Diseño mínimo

- Variantes: E1 (sin scaler, sin pesos), E2 (con scaler, sin pesos), E3
  (con scaler, con pesos) y E4 (sin scaler, con pesos).
- Semillas predefinidas: 11, 22, 33, 42, 44, 55, 66, 77, 88 y 99.
- Mantener sin cambios dataset, separación train/test, arquitectura, épocas,
  batch size, optimizador, learning rate, L2 y procedimiento de cuantización.
- Entrenar desde cero una red por semilla y variante.
- Convertir cada red resultante a INT8 con el mismo conjunto representativo.
- Evaluar Keras e INT8 sobre las mismas 1332 ventanas de prueba.

## Resultados que deben conservarse

- Accuracy, macro-F1 y métricas por clase de Keras e INT8.
- Matriz de confusión por ejecución.
- Diferencia INT8-Keras.
- Tamaño del modelo INT8.
- Semilla, versiones de bibliotecas y parámetros completos.

## Resumen estadístico

Reportar por variante media, desviación estándar, mínimo, máximo e intervalo de
confianza de las diez ejecuciones. Mantener el emparejamiento por semilla para
calcular los contrastes E2--E1, E3--E4, E4--E1 y E3--E2, los efectos marginales
del scaler y de class_weight, y la interacción como diferencia de diferencias.

## Despliegue

No es necesario cargar los diez modelos en STM32. La equivalencia PC-STM32 ya fue
demostrada para los modelos definitivos; solo se repetirá el despliegue si una nueva
ejecución presenta operadores, cuantización o tamaño diferentes.
