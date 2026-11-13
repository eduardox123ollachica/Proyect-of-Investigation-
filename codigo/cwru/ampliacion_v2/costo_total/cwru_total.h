#ifndef CWRU_TOTAL_V2_H_
#define CWRU_TOTAL_V2_H_
#include <stdint.h>
#include <stddef.h>
#include "cwru_masks.h"
#include "cwru_total_catalog.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float features[14];
    float standardized[14],affine[14];
    int8_t quantized[14],output[3];
    float detector_z[12],detector_terms[12],score;
    uint16_t saturated_low,saturated_high;
    uint32_t total_cycles,extract_cycles,input_cycles,nn_cycles,detector_cycles,residual_cycles;
    uint32_t extractions,failed_stage;
    int predicted_class,rejected;
} CWRUTOTAL_Result;

uint32_t CWRUTOTAL_ParamsFNV(const CWRUTV_Descriptor *descriptor);
int CWRUTOTAL_ValidateDescriptor(uint32_t id);
uint32_t CWRUTOTAL_ReadOverhead(void);
int CWRUTOTAL_Execute(uint32_t id,unsigned profile,CWRU_FeatureExtractor *extractor,
    const float32_t window[CWRU_WINDOW_SAMPLES][CWRU_NUM_CHANNELS],
    float32_t scratch_input[CWRU_FE_SCRATCH_FLOATS],float32_t scratch_output[CWRU_FE_SCRATCH_FLOATS],
    CWRUTOTAL_Result *result);
void CWRUTOTAL_Command(char *command,char *response,size_t capacity);
void CWRUTOTAL_Invalidate(void);
#ifdef __cplusplus
}
#endif
#endif
