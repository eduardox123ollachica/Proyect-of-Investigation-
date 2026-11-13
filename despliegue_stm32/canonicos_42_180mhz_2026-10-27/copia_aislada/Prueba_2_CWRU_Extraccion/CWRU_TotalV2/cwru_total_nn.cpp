#include "cwru_total_nn.h"
#include "cwru_total_catalog.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "stm32f4xx_hal.h"

#include <cmath>
#include <cstring>
#include <new>

namespace {

static_assert(sizeof(float) == sizeof(uint32_t), "Quantization requires float32");
constexpr uint32_t kModelCount = 32U;
constexpr unsigned kMaximumInputs = 14U;
constexpr unsigned kMaximumModelBytes = 512U * 1024U;

alignas(16) uint8_t arena[CWRUTNN_ARENA_CAPACITY];
alignas(tflite::MicroInterpreter)
    unsigned char interpreter_storage[sizeof(tflite::MicroInterpreter)];
tflite::MicroMutableOpResolver<2> resolver;
tflite::MicroInterpreter *interpreter = nullptr;
TfLiteTensor *input_tensor = nullptr;
TfLiteTensor *output_tensor = nullptr;
const CWRUTV_Descriptor *active_descriptor = nullptr;
uint32_t selected_id = CWRUTNN_NO_MODEL;
size_t arena_used = 0U;
int state = -100;
bool resolver_attempted = false;
bool resolver_ready = false;

uint32_t float_bits(float value)
{
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

uint32_t model_fnv(const CWRUTV_Descriptor &descriptor)
{
    uint32_t hash = 2166136261U;
    for (unsigned int i = 0U; i < descriptor.model_size; ++i)
        hash = (hash ^ descriptor.model[i]) * 16777619U;
    return hash;
}

void invalidate(int error)
{
    /* FreeSubgraphs must run while its allocator and tensors still exist. */
    if (interpreter != nullptr) {
        interpreter->~MicroInterpreter();
        interpreter = nullptr;
    }
    input_tensor = nullptr;
    output_tensor = nullptr;
    active_descriptor = nullptr;
    selected_id = CWRUTNN_NO_MODEL;
    arena_used = 0U;
    state = error;
    std::memset(arena, 0, sizeof(arena));
}

int selection_error(int error, uint32_t start, uint32_t *cycles)
{
    invalidate(error);
    *cycles = DWT->CYCCNT - start;
    return error;
}

bool tensor_shape(const TfLiteTensor *tensor, int columns)
{
    return tensor != nullptr && tensor->dims != nullptr &&
        tensor->dims->size == 2 && tensor->dims->data[0] == 1 &&
        tensor->dims->data[1] == columns;
}

}  // namespace

extern "C" int CWRUTNN_Select(uint32_t id, uint32_t *cycles, int *reused)
{
    if (cycles == nullptr || reused == nullptr) {
        if (cycles != nullptr) *cycles = 0U;
        if (reused != nullptr) *reused = 0;
        invalidate(-1);
        return -1;
    }
    *cycles = 0U;
    *reused = 0;
    if (!(DWT->CTRL & DWT_CTRL_CYCCNTENA_Msk)) {
        invalidate(-2);
        return -2;
    }
    if (state == 0 && interpreter != nullptr && active_descriptor != nullptr &&
        selected_id == id) {
        *reused = 1;
        return 0;
    }

    const uint32_t start = DWT->CYCCNT;
    /* An unsuccessful change never leaves the old model usable. */
    invalidate(-100);
    if (id >= kModelCount) return selection_error(-3, start, cycles);
    const CWRUTV_Descriptor &descriptor = cwru_total_catalog[id];
    if (descriptor.model == nullptr || descriptor.model_size < 8U ||
        descriptor.model_size > kMaximumModelBytes || descriptor.dims == 0U ||
        descriptor.dims > kMaximumInputs ||
        !std::isfinite(descriptor.qin_scale) || descriptor.qin_scale <= 0.0f ||
        descriptor.qin_zero < -128 || descriptor.qin_zero > 127)
        return selection_error(-4, start, cycles);

    /* Authenticate the bytes before dereferencing FlatBuffer fields. */
    if (model_fnv(descriptor) != descriptor.model_fnv)
        return selection_error(-5, start, cycles);
    if (!tflite::ModelBufferHasIdentifier(descriptor.model))
        return selection_error(-6, start, cycles);
    const tflite::Model *model = tflite::GetModel(descriptor.model);
    if (model == nullptr || model->version() != TFLITE_SCHEMA_VERSION)
        return selection_error(-6, start, cycles);

    if (!resolver_attempted) {
        resolver_attempted = true;
        /* Register once; re-adding operators for every selection is invalid. */
        const TfLiteStatus fc = resolver.AddFullyConnected();
        const TfLiteStatus softmax = resolver.AddSoftmax();
        resolver_ready = fc == kTfLiteOk && softmax == kTfLiteOk;
    }
    if (!resolver_ready) return selection_error(-7, start, cycles);

    interpreter = new (static_cast<void *>(interpreter_storage))
        tflite::MicroInterpreter(model, resolver, arena, sizeof(arena));
    if (interpreter->AllocateTensors() != kTfLiteOk)
        return selection_error(-8, start, cycles);
    if (interpreter->inputs_size() != 1U || interpreter->outputs_size() != 1U)
        return selection_error(-9, start, cycles);

    /* Public IO wrappers may allocate persistent arena memory; obtain them
     * once per selection, then measure arena usage after those allocations.
     */
    input_tensor = interpreter->input(0);
    output_tensor = interpreter->output(0);
    arena_used = interpreter->arena_used_bytes();
    if (!tensor_shape(input_tensor, descriptor.dims) ||
        !tensor_shape(output_tensor, CWRUTNN_OUTPUTS) ||
        input_tensor->type != kTfLiteInt8 || output_tensor->type != kTfLiteInt8 ||
        input_tensor->bytes != static_cast<size_t>(descriptor.dims) ||
        output_tensor->bytes != CWRUTNN_OUTPUTS ||
        input_tensor->data.int8 == nullptr || output_tensor->data.int8 == nullptr)
        return selection_error(-9, start, cycles);
    if (float_bits(input_tensor->params.scale) != float_bits(descriptor.qin_scale) ||
        input_tensor->params.zero_point != descriptor.qin_zero)
        return selection_error(-10, start, cycles);
    if (float_bits(output_tensor->params.scale) != 0x3b800000U ||
        output_tensor->params.zero_point != -128)
        return selection_error(-11, start, cycles);
    if (arena_used == 0U || arena_used > sizeof(arena))
        return selection_error(-12, start, cycles);

    active_descriptor = &descriptor;
    selected_id = id;
    state = 0;
    *cycles = DWT->CYCCNT - start;
    return 0;
}

extern "C" int CWRUTNN_Predict(const int8_t *input, uint8_t count,
    int8_t output[CWRUTNN_OUTPUTS], int *predicted_class)
{
    if (state != 0 || interpreter == nullptr || active_descriptor == nullptr ||
        input_tensor == nullptr || output_tensor == nullptr)
        return -20;
    if (input == nullptr || output == nullptr || predicted_class == nullptr ||
        count != active_descriptor->dims)
        return -21;

    std::memcpy(input_tensor->data.int8, input, count);
    if (interpreter->Invoke() != kTfLiteOk) {
        invalidate(-22);
        return -22;
    }
    std::memcpy(output, output_tensor->data.int8, CWRUTNN_OUTPUTS);
    int best = 0;
    for (unsigned i = 1U; i < CWRUTNN_OUTPUTS; ++i)
        if (output[i] > output[best]) best = static_cast<int>(i);
    *predicted_class = best;
    return 0;
}

extern "C" int CWRUTNN_Status(void) { return state; }
extern "C" size_t CWRUTNN_ArenaUsed(void) { return arena_used; }
extern "C" uint32_t CWRUTNN_SelectedId(void) { return selected_id; }
extern "C" uint32_t CWRUTNN_ActualModelFNV(void)
{
    return state == 0 && active_descriptor != nullptr ?
        model_fnv(*active_descriptor) : 0U;
}
