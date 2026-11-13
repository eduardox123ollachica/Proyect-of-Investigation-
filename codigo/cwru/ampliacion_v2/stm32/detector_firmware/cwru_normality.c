#include "cwru_normality.h"
#include "cwru_normality_params.h"
#include <math.h>
#include <string.h>

static int parameters_valid(void)
{
    if (CWRU_NORMALITY_NUM_CLUSTERS != 1 || CWRU_NORMALITY_NUM_FEATURES != 12 ||
        !isfinite(cwru_normality_threshold_d2) || cwru_normality_threshold_d2 < 0.0f) return 0;
    for (unsigned i = 0; i < 12U; ++i) {
        if (cwru_normality_feature_indices[i] >= 14U ||
            !isfinite(cwru_normality_scaler_mean[i]) ||
            !isfinite(cwru_normality_scaler_scale[i]) || cwru_normality_scaler_scale[i] <= 0.0f ||
            !isfinite(cwru_normality_centroids[0][i])) return 0;
    }
    return 1;
}

int CWRUND_Compute(const float *values, unsigned mode, CWRUND_Result *result)
{
    if (values == NULL || result == NULL || mode > CWRUND_CHECK) return -6;
    if (!parameters_valid()) return -4;
    unsigned count = mode == CWRUND_RUN ? 14U : mode == CWRUND_DIST ? 12U : 1U;
    for (unsigned i = 0; i < count; ++i) if (!isfinite(values[i])) return -1;
    if (mode == CWRUND_CHECK) {
        if (values[0] < 0.0f) return -3;
        result->score = values[0];
    } else {
        float total = 0.0f;
        for (unsigned i = 0; i < 12U; ++i) {
            if (mode == CWRUND_RUN) {
                volatile float centered = values[cwru_normality_feature_indices[i]] - cwru_normality_scaler_mean[i];
                result->z[i] = centered / cwru_normality_scaler_scale[i];
            } else result->z[i] = values[i];
            if (!isfinite(result->z[i])) return -2;
            /* Barreras de redondeo float32; evitar FMA sin cambiar flags globales. */
            volatile float difference = result->z[i] - cwru_normality_centroids[0][i];
            volatile float square = difference * difference;
            result->terms[i] = square;
            volatile float next = total + square;
            total = next;
            if (!isfinite(difference) || !isfinite(square) || !isfinite(total)) return -2;
        }
        result->score = total;
    }
    result->rejected = result->score > cwru_normality_threshold_d2;
    return 0;
}

static uint32_t hash_bytes(uint32_t hash, const void *pointer, size_t size)
{
    const unsigned char *p = (const unsigned char *)pointer;
    for (size_t i = 0; i < size; ++i) hash = (hash ^ p[i]) * 16777619U;
    return hash;
}

uint32_t CWRUND_ParamsFNV(void)
{
    uint32_t hash = hash_bytes(2166136261U, cwru_normality_feature_indices, 12U);
    hash = hash_bytes(hash, cwru_normality_scaler_mean, 12U * sizeof(float));
    hash = hash_bytes(hash, cwru_normality_scaler_scale, 12U * sizeof(float));
    hash = hash_bytes(hash, cwru_normality_centroids, 12U * sizeof(float));
    return hash_bytes(hash, &cwru_normality_threshold_d2, sizeof(float));
}
