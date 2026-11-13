#ifndef CWRU_EXTRACTION_PROFILE_H_
#define CWRU_EXTRACTION_PROFILE_H_

#include <stdint.h>
#include "cwru_pipeline_config.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Version 1: orden compartido con el cliente de diagnostico. */
typedef enum
{
    CWRU_PROFILE_RMS_MEAN = 0,
    CWRU_PROFILE_MOMENTS_CENTERING,
    CWRU_PROFILE_MOMENT_CORRECTIONS,
    CWRU_PROFILE_FFT,
    CWRU_PROFILE_ENERGY_DOMINANT,
    CWRU_PROFILE_ENTROPY,
    CWRU_PROFILE_OUTPUT,
    CWRU_PROFILE_BLOCKS
} CWRU_ProfileBlock;

typedef struct
{
    uint32_t cycles[CWRU_NUM_CHANNELS][CWRU_PROFILE_BLOCKS];
} CWRU_ExtractionProfile;

const CWRU_ExtractionProfile *CWRU_FeatureExtraction_GetProfile(void);
/* Minimo de 64 pares de lectura; orientativo y NO sustraido de los bloques. */
uint32_t CWRU_FeatureExtraction_ProfileReadOverhead(void);

#ifdef __cplusplus
}
#endif
#endif
