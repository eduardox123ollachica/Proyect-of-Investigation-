#include "cwru_pipeline.h"
#include "stm32f4xx_hal.h"
#include <string.h>

_Static_assert(CWRU_WINDOW_SAMPLES==4096 && CWRU_NUM_CHANNELS==2,"Contrato de ventana incompatible");
_Static_assert(CWRU_TOTAL_FEATURES==14 && CWRU_INPUT_FEATURES==14 && CWRUNN_INPUTS==14 && CWRUNN_OUTPUTS==3,
    "Contrato de tensores incompatible");

int CWRUPIPE_Execute(CWRU_FeatureExtractor *extractor,
    const float32_t window[CWRU_WINDOW_SAMPLES][CWRU_NUM_CHANNELS],
    float32_t scratch_input[CWRU_FE_SCRATCH_FLOATS],float32_t scratch_output[CWRU_FE_SCRATCH_FLOATS],
    CWRUPIPE_Result *result)
{
    if (extractor==NULL || window==NULL || scratch_input==NULL || scratch_output==NULL || result==NULL) return -1;
    /* Limpiar diagnóstico fuera de la medición por ventana; no almacenar ventanas aquí. */
    memset(result,0,sizeof(*result));
    uint32_t total_start=DWT->CYCCNT;
    uint32_t start=DWT->CYCCNT;
    ++result->extractions;
    int status=CWRU_FeatureExtraction_Extract(extractor,window,result->features,scratch_input,scratch_output);
    result->extract_cycles=DWT->CYCCNT-start;
    if (status!=0) { result->failed_stage=1U;return status; }
    start=DWT->CYCCNT;
    status=CWRU_InputTransform(result->features,&result->input);
    result->input_cycles=DWT->CYCCNT-start;
    if (status!=0) { result->failed_stage=2U;return status; }
    start=DWT->CYCCNT;
    status=CWRUNN_PredictQ(result->input.quantized,result->output,&result->predicted_class,&result->nn_invoke_cycles);
    result->nn_call_cycles=DWT->CYCCNT-start;
    if (status!=0) { result->failed_stage=3U;return status; }
    start=DWT->CYCCNT;
    /* El detector recibe las características originales, no Z ni INT8 de la DNN. */
    status=CWRUND_Compute(result->features,CWRUND_RUN,&result->detector);
    result->detector_cycles=DWT->CYCCNT-start;
    if (status!=0) { result->failed_stage=4U;return status; }
    result->total_cycles=DWT->CYCCNT-total_start;
    return 0;
}
