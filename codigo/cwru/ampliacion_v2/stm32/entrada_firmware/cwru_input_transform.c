#include "cwru_input_transform.h"
#include "cwru_dnn_scaler.h"
#include <float.h>
#include <math.h>
#include <string.h>

_Static_assert(sizeof(float) == 4 && FLT_RADIX == 2 && FLT_MANT_DIG == 24,
               "Se requiere float32 binario");
_Static_assert(CWRU_DNN_NUM_FEATURES == CWRU_INPUT_FEATURES,
               "Dimensiones incompatibles");

static int round_one(float value, int8_t *code, uint16_t bit, CWRU_InputResult *out)
{
    if (!isfinite(value)) return CWRU_INPUT_INTERMEDIATE;
    /* Los límites son asimétricos por ties-to-even: -128.5 -> -128;
     * 127.5 -> 128 antes de saturar. Se evita convertir floats enormes a int.
     */
    if (value >= 127.5f) {
        *code = 127;
        out->saturated_high |= bit;
    } else if (value < -128.5f) {
        *code = -128;
        out->saturated_low |= bit;
    } else {
        float lower_f = floorf(value);
        int lower = (int)lower_f; /* Siempre entre -129 y 127 en esta rama. */
        float fraction = value - lower_f;
        if (fraction > 0.5f || (fraction == 0.5f && lower % 2 != 0)) ++lower;
        *code = (int8_t)lower;
    }
    return CWRU_INPUT_OK;
}

static int execute(const float *input, CWRU_InputResult *out, unsigned mode)
{
    if (input == NULL || out == NULL) return CWRU_INPUT_NULL;
    out->saturated_low = 0U;
    out->saturated_high = 0U;
    for (unsigned j = 0U; j < CWRU_INPUT_FEATURES; ++j) {
        float value = input[j];
        if (!isfinite(value)) return CWRU_INPUT_NONFINITE;
        float standardized = value;
        if (mode == 0U) {
            float centered = value - cwru_dnn_scaler_mean[j];
            standardized = centered / cwru_dnn_scaler_scale[j];
        }
        if (!isfinite(standardized)) return CWRU_INPUT_INTERMEDIATE;
        out->standardized[j] = standardized;
        float affine = standardized;
        if (mode != 2U) {
            float divided = standardized / cwru_dnn_input_quant_scale;
            affine = divided + (float)cwru_dnn_input_quant_zero_point;
        }
        out->affine[j] = affine;
        int status = round_one(affine, &out->quantized[j], (uint16_t)(1U << j), out);
        if (status != CWRU_INPUT_OK) return status;
    }
    return CWRU_INPUT_OK;
}

int CWRU_InputTransform(const float input[CWRU_INPUT_FEATURES], CWRU_InputResult *out)
{
    return execute(input, out, 0U);
}

int CWRU_InputQuantize(const float standardized[CWRU_INPUT_FEATURES], CWRU_InputResult *out)
{
    return execute(standardized, out, 1U);
}

int CWRU_InputRound(const float affine[CWRU_INPUT_FEATURES], CWRU_InputResult *out)
{
    return execute(affine, out, 2U);
}
