# Protocolo del filtro antialias CWRU 48 kHz → 12 kHz

## Estado

Filtro diseñado y validado el 14 de octubre de 2026. Esta fase congela la
especificación y los coeficientes, pero todavía no genera el dataset
remuestreado ni define el ventaneo.

## Alcance de la decisión

La inspección espectral utilizó exclusivamente los 14 archivos de
entrenamiento definidos en
`datos/cwru/particiones/principal_por_carga/manifest_particion.csv`. No se
consultaron los archivos de validación ni de prueba para seleccionar el filtro.

Se analizaron las dos señales simultáneas de cada captura:

- `DE`: acelerómetro del lado de accionamiento.
- `FE`: acelerómetro del lado del ventilador.

## Evidencia espectral

La energía media medida en entrenamiento mostró que la banda de 3–5 kHz
contiene 33.20 % en DE y 39.77 % en FE. La banda 5–6 kHz contiene solo 0.27 %
en DE, pero 9.13 % en FE; dentro de FE, 7.00 % se concentra entre 5 y
5.25 kHz. Por ello, un corte sustancialmente inferior a 5 kHz descartaría
información potencialmente útil de FE.

También se observó energía por encima del nuevo Nyquist de 6 kHz. Aunque la
media entre 6 y 12 kHz fue 1.65 % para DE y 3.52 % para FE, algunos archivos
alcanzaron aproximadamente 14–16 %. El diezmado directo sin filtro produciría
aliasing relevante en esos casos.

## Especificación congelada

| Parámetro | Valor |
|---|---:|
| Frecuencia de entrada | 48 000 Hz |
| Frecuencia de salida | 12 000 Hz |
| Factor de diezmado | 4 |
| Tipo | FIR de fase lineal tipo I |
| Ventana | Kaiser |
| Coeficientes | 177 |
| Beta Kaiser | 5.653260 |
| Fin de banda pasante | 5 000 Hz |
| Corte nominal | 5 500 Hz |
| Inicio de banda rechazada | 6 000 Hz |
| Rechazo objetivo | 60 dB |
| Precisión canónica de coeficientes | `float32` |
| Retardo de grupo | 88 muestras a 48 kHz = 1.833333 ms |
| Retardo tras diezmar | 22 muestras a 12 kHz |

La respuesta calculada con los coeficientes `float32` presentó una ondulación
de 0.015462 dB entre 0 y 5 kHz y un máximo de −60.019917 dB desde 6 kHz.

## Validaciones realizadas

1. Respuesta frecuencial densa con 262 144 puntos.
2. Tonos sintéticos dentro de la banda pasante, transición y banda rechazada.
3. Verificación de las frecuencias alias tras el diezmado.
4. Coincidencia de la ganancia temporal con la respuesta teórica, con tolerancia
   máxima permitida de 0.1 dB.
5. Filtrado y diezmado de DE y FE de los 14 archivos de entrenamiento.
6. Igualdad de longitud entre ambos canales antes y después del proceso.
7. Comprobación de longitud esperada y ausencia de valores no finitos.

Todos los controles fueron satisfactorios. Los tonos de 6.1, 7 y 9 kHz se
atenuaron aproximadamente 63.38, 69.76 y 79.79 dB, respectivamente.

## Tratamiento de bordes

La implementación de referencia combina FIR causal y diezmado, y después
compensa el retardo entero. Debido a la extensión implícita por ceros, las
primeras y últimas 22 muestras de la señal a 12 kHz se consideran margen de
borde no válido. El ventaneo deberá excluirlas.

Esta regla evita que los transitorios artificiales del filtro participen en el
entrenamiento, validación o prueba.

## Trazabilidad

- Script: `codigo/cwru/preprocesamiento/validar_filtro_antialias_12khz.py`.
- Especificación verificable:
  `resultados/cwru/preprocesamiento_12khz/filtro_antialias/especificacion_filtro.json`.
- Coeficientes:
  `resultados/cwru/preprocesamiento_12khz/filtro_antialias/coeficientes_fir_kaiser_177.csv`.
- Figura:
  `resultados/cwru/preprocesamiento_12khz/filtro_antialias/respuesta_filtro_antialias.png`.
- SHA-256 de los 177 coeficientes `float32`:
  `f4e6d23b976a6f448674496182853f706e88b604fed47a3fa2b01218047befbb`.

## Trabajo todavía no realizado

- No se han escrito señales CWRU remuestreadas.
- No se ha elegido la duración ni el solapamiento de las ventanas.
- No se ha definido todavía si el modelo recibirá DE y FE como dos canales,
  ramas separadas o características fusionadas.
- No se ha entrenado ni evaluado ningún modelo con este procesamiento.
- No se ha implementado todavía el FIR en STM32.

