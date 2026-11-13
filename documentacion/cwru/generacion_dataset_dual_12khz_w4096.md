# Generación del dataset CWRU dual a 12 kHz con ventanas de 4096

## Estado

Dataset generado y validado estructuralmente. No se aplicó normalización y no
se entrenó ningún modelo durante esta fase.

## Configuración

- Canales: `DE`, `FE`, en ese orden.
- Frecuencia original: 48 kHz.
- Frecuencia resultante: 12 kHz.
- Filtro: FIR Kaiser de 177 coeficientes `float32` previamente validado.
- Longitud de ventana: 4096 muestras = 341.333 ms.
- Solapamiento: 50 %.
- Salto: 2048 muestras = 170.667 ms.
- Resolución de una FFT de 4096 puntos: 2.9296875 Hz.
- Representación guardada: `(ventanas, muestras, canales)` en `float32`.
- Normalización: ninguna.

Antes de segmentar se excluyeron 22 muestras de cada extremo de las señales a
12 kHz, correspondientes al margen inválido introducido por el FIR.

## Partición y ventanas obtenidas

| Subconjunto | N | IR | OR | Total | Archivos |
|---|---:|---:|---:|---:|---:|
| Entrenamiento | 86 | 223 | 259 | 568 | 14 |
| Validación | 58 | 174 | 174 | 406 | 7 |
| Prueba | 58 | 174 | 174 | 406 | 7 |
| **Total** | **202** | **571** | **607** | **1380** | **28** |

La diferencia de cantidades se debe al número de archivos por clase y a la
duración desigual de algunas capturas, no a una selección aleatoria de
ventanas. Las ventanas de un archivo pertenecen siempre al mismo subconjunto.

## Validaciones estructurales

- 28 archivos únicos y ningún archivo compartido entre subconjuntos.
- Formas obtenidas: entrenamiento `(568, 4096, 2)`, validación
  `(406, 4096, 2)` y prueba `(406, 4096, 2)`.
- Correspondencia exacta entre cada matriz `X`, sus etiquetas `y` y su metadata.
- Todas las muestras son finitas.
- Se comprobaron 1352 pares consecutivos dentro de archivo: la segunda mitad
  de cada ventana coincide exactamente con la primera mitad de la siguiente,
  confirmando el 50 % de solapamiento.
- La prueba solo se procesó y verificó estructuralmente; no se utilizó para
  elegir normalización, modelo ni hiperparámetros.

## Archivos generados

Los productos están en `datos/cwru/procesado/dual_12khz_w4096/`:

- `X_train.npy`, `y_train.npy`, `metadata_train.csv`.
- `X_validation.npy`, `y_validation.npy`, `metadata_validation.csv`.
- `X_test.npy`, `y_test.npy`, `metadata_test.csv`.
- `metadata_todas_las_ventanas.csv`.
- `resumen_por_clase.csv`.
- `configuracion.json`.
- `HASHES.sha256`.

El script reproducible es
`codigo/cwru/preprocesamiento/generar_dataset_dual_12khz_w4096.py`.

## Decisiones resueltas posteriormente

- Representación: 14 características, siete para DE y siete para FE.
- Normalización: `StandardScaler` ajustado exclusivamente con entrenamiento.
- Desbalance: pesos de clase balanceados durante el entrenamiento.
- Arquitectura: DNN 14--20--10--3, con 543 parámetros.
- Instancia canónica: semilla predefinida 42.
- Cuantización: TFLite entero INT8 con entrada y salida `int8`; no produjo cambios de clase respecto a FP32 en validación.

La justificación multisemilla se conserva en `seleccion_representacion_y_escalado.md` y la conversión en `cuantizacion_int8_features14.md`.

