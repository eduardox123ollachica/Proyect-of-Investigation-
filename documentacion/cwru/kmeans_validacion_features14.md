# Validación del detector K-means en CWRU

> **Estado:** ensayo inicial conservado para trazabilidad, pero descartado como detector final. Este diseño aprendió las tres clases como dominio conocido y no corresponde al objetivo posterior de corroborar alejamiento respecto a normalidad. El rediseño normal-only se documenta en `kmeans_normal_por_carga.md`.

## Protocolo reproducido

Para conservar la correspondencia con el pipeline anterior se ajustó un detector K-means independiente del clasificador:

- entrada: las mismas 14 características DE--FE;
- `StandardScaler` propio ajustado solo con las 568 ventanas de entrenamiento;
- `K=32`, `random_state=42`, `n_init=20` y algoritmo Lloyd;
- puntuación: distancia euclídea cuadrática al centroide más cercano;
- umbral: percentil 99 de las puntuaciones de entrenamiento calculadas con los artefactos `float32` exportables;
- rechazo: `score > threshold`.

El conjunto de prueba no fue cargado. El umbral obtenido fue `1.277197003364563`.

## Resultado

En entrenamiento, 6/568 ventanas superaron estrictamente el percentil interpolado, equivalentes a 1.0563 %. En validación, aun cuando todas las ventanas pertenecen a clases conocidas, fueron rechazadas 184/406, equivalentes a 45.3202 %.

| Clase conocida | Rechazadas | Total | Fracción |
|---|---:|---:|---:|
| N | 4 | 58 | 6.90 % |
| IR | 117 | 174 | 67.24 % |
| OR | 63 | 174 | 36.21 % |

La mediana del score aumentó de 0.165193 en entrenamiento a 1.197249 en validación. El P99 de validación fue 9.131555, muy superior al umbral de entrenamiento. El clúster asignado mediante los parámetros exportados en `float32` coincidió con scikit-learn en las 568 ventanas de entrenamiento y las 406 de validación, por lo que el desplazamiento no procede de la conversión numérica.

## Diagnóstico

La frecuencia dominante FE aportó 59.74 % de la suma de scores entre las ventanas rechazadas. Le siguieron amplitud dominante DE (11.42 %), entropía espectral DE (7.02 %), asimetría FE (4.82 %) y energía espectral DE (4.29 %).

El rechazo también se concentró por archivo y diámetro de falla:

- IR 0.014 pulgadas: 58/58;
- IR 0.021 pulgadas: 55/58;
- OR 0.007 pulgadas: 41/58;
- OR 0.021 pulgadas: 20/58;
- IR 0.007 pulgadas y N: 4/58 cada uno;
- OR 0.014 pulgadas: 2/58.

La partición separa cargas: entrenamiento usa 0--1 HP y validación 2 HP. Por tanto, el resultado muestra que el detector global calibrado con entrenamiento es sensible al cambio de condiciones y a combinaciones concretas de clase/diámetro. No demuestra que las 184 ventanas contengan anomalías físicas.

## Interpretación y decisión pendiente

Estas 184 observaciones deben denominarse *ventanas conocidas rechazadas* o *observaciones estadísticamente alejadas*, no falsas alarmas ni fallas detectadas: CWRU no aporta una etiqueta adicional de anomalía desconocida para este experimento.

Recalibrar el umbral con validación es metodológicamente posible antes de abrir prueba, pero cambiaría la regla reproducida y elevaría el umbral aproximadamente desde 1.277 hasta el entorno del P99 de validación, 9.132. Esto podría ocultar el desplazamiento y reducir la sensibilidad ante entradas realmente desconocidas. La alternativa científicamente más conservadora es mantener el P99 de entrenamiento, reportar que K-means no fue estable frente al cambio de carga y evaluar en prueba la regla preespecificada sin reajuste.

Los artefactos están en `resultados/cwru/modelo_features_14/kmeans_k32_seed42/`. El archivo `kmeans_k32_modelo_float32.npz` contiene media, escala, centroides 32x14 y umbral; su SHA-256 es `e2e9c27e5f8c16c0acffe82e98c42ffa3ed8c09b61025a43913679429dac4397`.
