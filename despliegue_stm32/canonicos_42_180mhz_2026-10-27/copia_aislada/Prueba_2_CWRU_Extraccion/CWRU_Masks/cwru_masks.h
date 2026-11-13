#ifndef CWRU_MASKS_H_
#define CWRU_MASKS_H_

#include <stddef.h>
#include <stdint.h>
#include "cwru_feature_extraction.h"

#define CWRUMASK_COUNT 10U
#define CWRUMASK_ERROR_ID (-10)

#ifdef __cplusplus
extern "C" {
#endif

typedef struct
{
    const char *name;
    uint16_t feature_mask;
} CWRUMASK_Spec;

/* Orden: F14,T6,S8,NF12,NM10,NE12,DE7,FE7,FIX_DE13,FIX_FE13. */
const CWRUMASK_Spec *CWRUMASK_GetSpec(uint32_t id);
/* FNV-1a de las diez mascaras uint16, cada una serializada little-endian. */
uint32_t CWRUMASK_CatalogFNV(void);

/* Receptor opcional implementado por el include del banco UART. */
void CWRUMASK_Command(char *command, char *response, size_t capacity);
void CWRUMASK_Invalidate(void);

/* Escribe SOLO los slots seleccionados, en el orden original DE7+FE7.
 * El llamador limpia los catorce slots fuera del intervalo DWT. Los buffers
 * scratch y la instancia RFFT son los mismos que en el extractor original.
 * No hay otra ventana, FFT, arena o perfil reservado por este modulo. */
int CWRUMASK_Extract(
    uint32_t id,
    CWRU_FeatureExtractor *extractor,
    const float32_t window[CWRU_WINDOW_SAMPLES][CWRU_NUM_CHANNELS],
    float32_t features[CWRU_TOTAL_FEATURES],
    float32_t scratch_input[CWRU_FE_SCRATCH_FLOATS],
    float32_t scratch_output[CWRU_FE_SCRATCH_FLOATS]
);

#ifdef __cplusplus
}
#endif
#endif
