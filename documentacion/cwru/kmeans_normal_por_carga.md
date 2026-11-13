# Detector centroidal de alejamiento respecto a normalidad

## Objetivo

El módulo auxiliar no clasifica IR u OR. Aprende exclusivamente el funcionamiento saludable `N` y produce evidencia complementaria de alejamiento respecto a esa normalidad. La DNN conserva la responsabilidad de clasificar `N`, `IR` u `OR`.

La salida se interpreta de esta manera:

| DNN | Detector normal-only | Interpretación |
|---|---|---|
| N | dentro del umbral | normalidad coherente |
| IR/OR | fuera del umbral | falla clasificada y alejamiento de normalidad corroborado |
| IR/OR | dentro del umbral | falla clasificada sin corroboración geométrica |
| N | fuera del umbral | ventana sospechosa o fuera del dominio; no confirma una falla |

El detector no constituye verificación independiente porque comparte señales y características con la DNN.

## Desarrollo agrupado por carga

Se utilizaron exclusivamente las cargas de desarrollo 0, 1 y 2 HP. La carga 3 HP y todos los archivos de prueba permanecieron sin cargar.

Para cada candidato `K={1,2,4,8}` se realizaron tres iteraciones:

1. se excluyó por completo una carga;
2. el escalador y los centroides se ajustaron solo con ventanas `N` de las otras dos cargas;
3. se puntuaron todas las ventanas de la carga excluida;
4. se reunieron las puntuaciones normales fuera de carga y el umbral se fijó en su P95.

La selección maximizó exactitud balanceada binaria `N` frente a `IR+OR`; los desempates favorecieron la menor complejidad.

## Exclusión de las frecuencias dominantes

El primer ensayo empleó las 14 características y no fue válido: obtuvo solo 3.37 % de sensibilidad de corroboración. En las grabaciones normales, la frecuencia dominante DE fue exactamente 4145.51 Hz a 0 HP y 360.35 Hz a 1--2 HP, sin variación dentro de cada una de esas cargas. FE mostró un patrón semejante. Al ajustar un escalador normal-only con dos cargas, una característica constante o casi constante generó distancias de aproximadamente 28.7 millones en la carga excluida y dominó artificialmente el umbral.

Por ello, el detector auxiliar excluye `de_dominant_frequency` y `fe_dominant_frequency`. La DNN continúa utilizando las 14 características. El detector utiliza las 12 posiciones `[0,1,2,3,5,6,7,8,9,10,12,13]` del vector de la DNN.

## Selección de K

Los cuatro candidatos alcanzaron los mismos resultados fuera de carga:

- corroboración de IR/OR: 830/830, 100 %;
- normales aceptadas: 136/144, 94.44 %;
- normales rechazadas: 8/144, 5.56 %;
- exactitud balanceada: 97.22 %;
- sensibilidad mínima entre cargas: 100 %.

La especificidad normal varió por carga: 71.43 % en 0 HP y 100 % en 1 y 2 HP. Los ocho rechazos normales pertenecieron a la carga 0 HP. Como ningún K mayor mejoró el resultado, se seleccionó `K=1`. En este caso, K-means se reduce a un detector de distancia cuadrática a un único prototipo saludable estandarizado; por claridad se denomina *detector centroidal de normalidad*.

El umbral congelado es `309.9235534667969`, calculado como P95 de las puntuaciones normales fuera de carga. Entre esas predicciones fuera de carga, el score normal máximo fue 389.7834 y el score de falla mínimo fue 403.0774. Esta separación se obtuvo en desarrollo y no garantiza que persista en prueba.

## Modelo final de desarrollo

Después de seleccionar el protocolo, el escalador y el centroide final se ajustaron con las 144 ventanas normales de 0, 1 y 2 HP. El umbral no se recalculó con puntuaciones de ajuste: se conservó el P95 fuera de carga. De forma descriptiva, el modelo reajustado aceptó las 144 ventanas normales de desarrollo y rechazó las 830 IR/OR, pero estas cifras no sustituyen la evaluación final porque el centroide ya había visto las normales de desarrollo y K fue seleccionado observando IR/OR de desarrollo.

## Limitaciones

- Las 144 ventanas normales proceden de solo tres archivos y están solapadas; el tamaño efectivo es mucho menor que 144 observaciones independientes.
- El resultado 100 % sobre IR/OR refleja fallas inducidas del banco CWRU y no demuestra sensibilidad equivalente ante degradación temprana real.
- El umbral controla alejamiento estadístico, no confirma una causa física.
- La exclusión de frecuencia dominante fue una decisión de desarrollo tomada antes de abrir prueba y debe declararse de forma transparente.

## Artefactos

- Script: `codigo/cwru/entrenamiento/seleccionar_kmeans_normal_por_carga.py`.
- Resultados: `resultados/cwru/modelo_features_14/kmeans_normal_lolo_sin_frecuencia_dominante/`.
- Modelo final: `kmeans_normal_final_float32.npz`, con media y escala de 12 elementos, centroide 1x12 y umbral.
- SHA-256: `5aa9fd59fae5ed3c42fa3d8b51c4ef6b1123894fcd6c1145c9040cc9ae7d22ca`.
- El ensayo fallido de 14 características se preserva en `kmeans_normal_lolo_all14_descartado/`.

Tras congelar el protocolo, el conjunto de prueba de 3 HP se evaluó una sola vez. El detector aceptó las 58 ventanas N y rechazó las 348 ventanas IR/OR. El resultado completo y sus límites se documentan en `evaluacion_final_test_cwru.md`; no se reajustó ningún parámetro después de observarlo.
