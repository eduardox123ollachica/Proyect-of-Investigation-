#include "cwru_nn.h"
#include "cwru_model_data.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "stm32f4xx_hal.h"
#include <cstring>

namespace {
alignas(16) uint8_t arena[CWRUNN_ARENA_CAPACITY];
tflite::MicroMutableOpResolver<2> resolver;
tflite::MicroInterpreter *interpreter = nullptr;
TfLiteTensor *input = nullptr;
TfLiteTensor *output = nullptr;
int state = -100;
bool attempted = false;
size_t used = 0U;
uint32_t bits(float x) { uint32_t value; std::memcpy(&value, &x, 4); return value; }
}

extern "C" int CWRUNN_Init(uint32_t *cycles, int *reused)
{
    if (cycles == nullptr || reused == nullptr) return -1;
    *cycles = 0U;
    *reused = attempted ? 1 : 0;
    if (attempted) return state;
    if (!(DWT->CTRL & DWT_CTRL_CYCCNTENA_Msk)) return -2;
    attempted = true;
    uint32_t start = DWT->CYCCNT;
    const tflite::Model *model = tflite::GetModel(g_cwru_model);
    if (model == nullptr || model->version() != TFLITE_SCHEMA_VERSION || g_cwru_model_len != 4064U) state = -3;
    else if (resolver.AddFullyConnected() != kTfLiteOk || resolver.AddSoftmax() != kTfLiteOk) state = -4;
    else {
        static tflite::MicroInterpreter instance(model, resolver, arena, sizeof(arena));
        interpreter = &instance;
        if (interpreter->AllocateTensors() != kTfLiteOk) state = -5;
        else {
            input = interpreter->input(0);
            output = interpreter->output(0);
            used = interpreter->arena_used_bytes(); /* Incluir wrappers públicos IO. */
            if (interpreter->inputs_size() != 1U || interpreter->outputs_size() != 1U || input == nullptr || output == nullptr) state = -6;
            else if (input->type != kTfLiteInt8 || output->type != kTfLiteInt8 ||
                     input->bytes != 14U || output->bytes != 3U ||
                     input->dims == nullptr || output->dims == nullptr ||
                     input->dims->size != 2 || output->dims->size != 2 ||
                     input->dims->data[0] != 1 || input->dims->data[1] != 14 ||
                     output->dims->data[0] != 1 || output->dims->data[1] != 3) state = -7;
            else if (bits(input->params.scale) != 0x3d0a0072U || input->params.zero_point != -9 ||
                     bits(output->params.scale) != 0x3b800000U || output->params.zero_point != -128) state = -8;
            else if (used == 0U || used > sizeof(arena)) state = -9;
            else state = 0;
        }
    }
    *cycles = DWT->CYCCNT - start;
    return state;
}

extern "C" int CWRUNN_Status(void) { return state; }
extern "C" size_t CWRUNN_ArenaUsed(void) { return used; }
extern "C" uint32_t CWRUNN_ModelFNV(void)
{
    uint32_t hash = 2166136261U;
    for (unsigned i = 0; i < g_cwru_model_len; ++i) hash = (hash ^ g_cwru_model[i]) * 16777619U;
    return hash;
}

extern "C" int CWRUNN_PredictQ(const int8_t values[CWRUNN_INPUTS], int8_t result[CWRUNN_OUTPUTS],
                               int *predicted_class, uint32_t *invoke_cycles)
{
    if (state != 0 || interpreter == nullptr) return -10;
    if (values == nullptr || result == nullptr || predicted_class == nullptr || invoke_cycles == nullptr) return -11;
    if (!(DWT->CTRL & DWT_CTRL_CYCCNTENA_Msk)) return -12;
    std::memcpy(input->data.int8, values, CWRUNN_INPUTS);
    uint32_t start = DWT->CYCCNT;
    TfLiteStatus status = interpreter->Invoke();
    *invoke_cycles = DWT->CYCCNT - start;
    if (status != kTfLiteOk) return -13;
    std::memcpy(result, output->data.int8, CWRUNN_OUTPUTS);
    int best = 0;
    for (unsigned j = 1U; j < CWRUNN_OUTPUTS; ++j)
        if (result[j] > result[best]) best = (int)j;
    *predicted_class = best; /* Primer índice en empate, como numpy.argmax. */
    return 0;
}
