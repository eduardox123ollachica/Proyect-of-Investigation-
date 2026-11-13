#ifndef CWRU_FEATURE_EXTRACTION_H_
#define CWRU_FEATURE_EXTRACTION_H_

#include <stdint.h>

#include "arm_math.h"
#include "cwru_pipeline_config.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Extracción de las 14 características congeladas para CWRU.
 *
 * Entrada:
 *   window[n][0] -> acelerómetro Drive End (DE)
 *   window[n][1] -> acelerómetro Fan End (FE)
 *
 * DE y FE son posiciones de medida; no representan ejes ortogonales X/Y/Z.
 * La ventana debe contener 4096 muestras por señal a 12 kHz.
 */

#define CWRU_FE_RFFT_BINS ((CWRU_WINDOW_SAMPLES / 2) + 1)
#define CWRU_FE_SCRATCH_FLOATS CWRU_WINDOW_SAMPLES
#define CWRU_FE_SCRATCH_BYTES \
    (2U * CWRU_FE_SCRATCH_FLOATS * (uint32_t)sizeof(float32_t))

typedef enum
{
    CWRU_FE_RMS = 0,
    CWRU_FE_SKEWNESS,
    CWRU_FE_KURTOSIS,
    CWRU_FE_SPECTRAL_ENERGY,
    CWRU_FE_DOMINANT_FREQUENCY,
    CWRU_FE_DOMINANT_AMPLITUDE,
    CWRU_FE_SPECTRAL_ENTROPY
} CWRU_FeatureIndex;

typedef enum
{
    CWRU_FE_OK = 0,
    CWRU_FE_ERROR_NULL_POINTER = -1,
    CWRU_FE_ERROR_NOT_INITIALIZED = -2,
    CWRU_FE_ERROR_FFT_INIT = -3
} CWRU_FeatureStatus;

typedef struct
{
    arm_rfft_fast_instance_f32 rfft;
    uint8_t initialized;
} CWRU_FeatureExtractor;

/* Inicializar una sola vez antes de procesar ventanas. */
int CWRU_FeatureExtraction_Init(CWRU_FeatureExtractor *extractor);

/*
 * Extrae [DE: 7 características, FE: 7 características].
 *
 * scratch_input y scratch_output deben apuntar cada uno a 4096 float32_t.
 * Se reutilizan entre DE y FE y su contenido se destruye. Deben reservarse de
 * forma estática/global; no se recomienda colocarlos en el stack.
 */
int CWRU_FeatureExtraction_Extract(
    CWRU_FeatureExtractor *extractor,
    const float32_t window[CWRU_WINDOW_SAMPLES][CWRU_NUM_CHANNELS],
    float32_t features[CWRU_TOTAL_FEATURES],
    float32_t scratch_input[CWRU_FE_SCRATCH_FLOATS],
    float32_t scratch_output[CWRU_FE_SCRATCH_FLOATS]
);

#ifdef __cplusplus
}
#endif

#endif /* CWRU_FEATURE_EXTRACTION_H_ */
