# Preparación del despliegue CWRU en STM32F446RE

## Paquete exportado

Los artefactos congelados se exportaron a `modelos/cwru/canonico_seed42/firmware/` sin modificar los proyectos STM32 históricos externos. El paquete contiene:

- `cwru_model_data.h/.cpp`: modelo TFLite INT8 de 4064 bytes;
- `cwru_dnn_scaler.h/.cpp`: media y escala de las 14 características, además de los parámetros de cuantización;
- `cwru_normality_params.h/.cpp`: índices de las 12 características, escalador normal-only, centroide 1x12 y umbral;
- `cwru_pipeline_config.h`: 12 kHz, ventana 4096, salto 2048 y orden DE--FE;
- `cwru_feature_extraction.h/.c`: implementación C independiente de las 14 características mediante RFFT de 4096 puntos de CMSIS-DSP;
- `manifest.json`: tamaños, hashes de fuentes generados y hashes de los artefactos originales.

La reconstrucción byte a byte desde `cwru_model_data.cpp` coincidió con el TFLite congelado. Las 14 medias y escalas DNN reconstruidas desde C++ coincidieron exactamente en `float32` con el NPZ. Todos los valores verificados fueron finitos. No existe actualmente un compilador C++ o `arm-none-eabi-g++` disponible en el `PATH`; por ello, la compilación real queda pendiente para la integración en STM32CubeIDE y no se afirma todavía que estos fuentes hayan sido compilados para Cortex-M4F.

La extracción C conserva el orden `[DE: 7, FE: 7]`, calcula RMS sobre la señal original y resta la media únicamente antes de la FFT. La asimetría y la curtosis utilizan las correcciones `bias=False` de SciPy; la energía y la entropía incluyen los 2049 bins de la RFFT, mientras la búsqueda del máximo ignora DC. La interfaz exige dos buffers externos de 4096 `float32_t`, reutilizados secuencialmente entre DE y FE, por lo que no introduce arreglos grandes en el stack. Una emulación de las mismas operaciones y redondeos `float32` sobre 12 ventanas de prueba conservó los 24 bins dominantes y presentó error relativo máximo de aproximadamente `6.38e-8` frente a las características congeladas. Esta comprobación valida las fórmulas, pero no sustituye la futura comparación con la salida compilada de CMSIS-DSP en STM32.

## Presupuesto preliminar de memoria

El STM32F446RE dispone de 128 KiB de SRAM. Para procesamiento con 50 % de solapamiento, una implementación conservadora requiere aproximadamente:

| Recurso | Estimación |
|---|---:|
| Ventana DE+FE, `2 x 4096 float32` | 32 KiB |
| Entrada/salida o memoria auxiliar FFT de 4096 | hasta 32 KiB |
| Arena TFLM inicial conservadora | 16 KiB |
| Características, escaladores y estado auxiliar | <1 KiB |
| Total de trabajo conocido | aproximadamente 81 KiB |

Esto deja alrededor de 47 KiB para `.data`, `.bss`, HAL, stack y otras estructuras. La cifra demuestra factibilidad preliminar, no suficiencia garantizada. Debe confirmarse con el archivo MAP, `size`, watermark de stack y medición del tensor arena en el firmware final.

La memoria FFT puede reducirse mediante reutilización secuencial de buffers: calcular primero los estadísticos temporales, preservar las mitades necesarias para el solapamiento y procesar DE y FE de forma consecutiva con un único buffer de salida. No se deben reservar simultáneamente espectros, magnitudes y potencias completos si pueden acumularse en una sola pasada.

## Presupuesto temporal y transporte

- Duración de una ventana: 4096/12000 = 341.333 ms.
- Periodo entre ventanas con 50 % de solapamiento: 2048/12000 = 170.667 ms.
- Carga binaria de una ventana dual `float32`: 32768 bytes.
- Tiempo mínimo por UART a 115200, 8N1: aproximadamente 2.844 s por ventana.

Por tanto, USART a 115200 no puede reproducir estas ventanas en tiempo real: la transmisión binaria es aproximadamente 16.7 veces más lenta que el periodo entre ventanas, sin contar protocolo ni procesamiento. UART sigue siendo válido para validación offline ventana por ventana, pero no como ruta de adquisición en tiempo real.

Una aplicación autónoma necesitaría dos canales de vibración simultáneos ubicados en DE y FE, DMA y una interfaz/sensor capaz de sostener 12 kHz por canal. DE y FE son posiciones de sensor, no ejes ortogonales de un único acelerómetro. Si la adquisición física se realiza directamente a 12 kHz, debe existir protección antialias coherente con la banda útil; el FIR offline 48--12 kHz no se ejecuta automáticamente en esa ruta.

## Integración pendiente

1. integrar `cwru_feature_extraction.c` y su cabecera en un proyecto STM32 nuevo o copia controlada;
2. compilar y validar numéricamente PC--C/CMSIS-DSP sobre un subconjunto antes de ejecutar inferencia;
3. integrar TFLM y el detector normal-only;
4. medir Flash, RAM, arena y stack con el firmware completo;
5. reproducir por UART un subconjunto y luego las 406 ventanas de prueba para verificar equivalencia PC--STM32;
6. medir latencia por bloques con DWT y compararla con el periodo de 170.667 ms.
