#ifndef CWRU_INTEGRATED_PIPELINE_H_
#define CWRU_INTEGRATED_PIPELINE_H_
#include <stddef.h>
#include <stdint.h>
#include "cwru_feature_extraction.h"
#include "cwru_input_transform.h"
#include "cwru_nn.h"
#include "cwru_normality.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef struct {
    float features[14];
    CWRU_InputResult input;
    int8_t output[3];
    int predicted_class;
    CWRUND_Result detector;
    uint32_t total_cycles,extract_cycles,input_cycles,nn_call_cycles,nn_invoke_cycles,detector_cycles;
    uint32_t extractions;
    unsigned failed_stage;
} CWRUPIPE_Result;
int CWRUPIPE_Execute(CWRU_FeatureExtractor *extractor,
    const float32_t window[CWRU_WINDOW_SAMPLES][CWRU_NUM_CHANNELS],
    float32_t scratch_input[CWRU_FE_SCRATCH_FLOATS],float32_t scratch_output[CWRU_FE_SCRATCH_FLOATS],
    CWRUPIPE_Result *result);
void CWRUPIPE_Invalidate(void);
void CWRUPIPE_Command(char *command,char *response,size_t capacity);
#ifdef __cplusplus
}
#endif
#endif
