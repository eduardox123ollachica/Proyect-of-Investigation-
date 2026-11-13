#include "cwru_masks.h"

#include <math.h>
#include <stddef.h>
#include <string.h>

#if !CWRU_MASK_BENCH
#error "Este modulo pertenece unicamente al ensayo CWRU_MASK_BENCH."
#endif
#if CWRU_ENTROPY_MODE != 2 || CWRU_CACHE_SPECTRAL_POWER != 1
#error "Las mascaras se contrastan con el extractor EntM/cache."
#endif
#if CWRU_PROFILE_EXTRACTOR || CWRU_PIPELINE_TEST || CWRU_STACK_TEST
#error "Las mascaras se cronometran sin perfil, DNN o monitor de stack."
#endif
#if !defined(__GNUC__)
#error "La especializacion explicita de este ensayo requiere GNU C."
#endif

_Static_assert(CWRU_WINDOW_SAMPLES == 4096 && CWRU_NUM_CHANNELS == 2,
    "Contrato de senales incompatible.");
_Static_assert(CWRU_NUM_FEATURES_PER_CHANNEL == 7 && CWRU_TOTAL_FEATURES == 14,
    "Orden de caracteristicas incompatible.");
_Static_assert(sizeof(double) == 2U * sizeof(float32_t),
    "La cache requiere double de ocho bytes y float32 de cuatro.");
_Static_assert((CWRU_FE_RFFT_BINS - 1U) * sizeof(double) <=
    CWRU_FE_SCRATCH_FLOATS * sizeof(float32_t),
    "Las potencias deben caber en el scratch existente.");

#define MASK_INLINE inline __attribute__((always_inline))
#define MASK_WRAPPER __attribute__((noinline))
#define MASK_FREQUENCY_RESOLUTION_HZ \
    ((float32_t)CWRU_SAMPLE_RATE_HZ / (float32_t)CWRU_WINDOW_SAMPLES)

static const CWRUMASK_Spec catalog[CWRUMASK_COUNT] = {
    {"F14", 0x3fffU}, {"T6", 0x0387U}, {"S8", 0x3c78U},
    {"NF12", 0x37efU}, {"NM10", 0x3cf9U}, {"NE12", 0x1fbfU},
    {"DE7", 0x007fU}, {"FE7", 0x3f80U},
    {"FIX_DE13", 0x37ffU}, {"FIX_FE13", 0x3fefU}
};

const CWRUMASK_Spec *CWRUMASK_GetSpec(uint32_t id)
{
    return id < CWRUMASK_COUNT ? &catalog[id] : NULL;
}

uint32_t CWRUMASK_CatalogFNV(void)
{
    uint32_t hash = 2166136261U;
    for (uint32_t i = 0U; i < CWRUMASK_COUNT; ++i)
    {
        hash = (hash ^ (uint32_t)(catalog[i].feature_mask & 0xffU)) * 16777619U;
        hash = (hash ^ (uint32_t)(catalog[i].feature_mask >> 8U)) * 16777619U;
    }
    return hash;
}

static MASK_INLINE float32_t mask_finite_or_zero(double value)
{
    if (!isfinite(value)) return 0.0f;
    /* Se conserva la convencion historica: finitud double antes del cast. */
    return (float32_t)value;
}

static MASK_INLINE double mask_power_at_bin(const float32_t *fft_output, uint32_t bin)
{
    double real;
    double imag;
    if (bin == 0U)
    {
        real = (double)fft_output[0];
        imag = 0.0;
    }
    else if (bin == (CWRU_WINDOW_SAMPLES / 2U))
    {
        real = (double)fft_output[1];
        imag = 0.0;
    }
    else
    {
        real = (double)fft_output[2U * bin];
        imag = (double)fft_output[(2U * bin) + 1U];
    }
    return (real * real) + (imag * imag);
}

static MASK_INLINE void mask_cache_store(float32_t *workspace, uint32_t bin, double power)
{
    memcpy((unsigned char *)workspace + (bin - 1U) * sizeof(power), &power, sizeof(power));
}

static MASK_INLINE double mask_cache_load(const float32_t *workspace, uint32_t bin)
{
    double power;
    memcpy(&power, (const unsigned char *)workspace + (bin - 1U) * sizeof(power), sizeof(power));
    return power;
}

static MASK_INLINE double mask_entropy_log_mixed(double probability)
{
    const float probability_f = (float)probability;
    if (!isnormal(probability_f) || probability_f >= 1.0f)
        return log(probability);
    return (double)logf(probability_f);
}

/* Cada llamada de los wrappers tiene canal y mascara constantes. always_inline
 * elimina los calculos ausentes sin evaluar la mascara en los bucles de datos.
 * Las sumas retenidas mantienen orden, tipos y formulas del extractor EntM. */
static MASK_INLINE void mask_channel(
    const float32_t window[CWRU_WINDOW_SAMPLES][CWRU_NUM_CHANNELS],
    uint32_t channel,
    uint32_t selected,
    float32_t output[CWRU_NUM_FEATURES_PER_CHANNEL],
    CWRU_FeatureExtractor *extractor,
    float32_t scratch_input[CWRU_FE_SCRATCH_FLOATS],
    float32_t scratch_output[CWRU_FE_SCRATCH_FLOATS])
{
    if (selected == 0U) return;
    const int need_rms = (selected & (1U << CWRU_FE_RMS)) != 0U;
    const int need_skew = (selected & (1U << CWRU_FE_SKEWNESS)) != 0U;
    const int need_kurt = (selected & (1U << CWRU_FE_KURTOSIS)) != 0U;
    const int need_energy = (selected & (1U << CWRU_FE_SPECTRAL_ENERGY)) != 0U;
    const int need_freq = (selected & (1U << CWRU_FE_DOMINANT_FREQUENCY)) != 0U;
    const int need_amp = (selected & (1U << CWRU_FE_DOMINANT_AMPLITUDE)) != 0U;
    const int need_entropy = (selected & (1U << CWRU_FE_SPECTRAL_ENTROPY)) != 0U;
    const int need_moments = need_skew || need_kurt;
    const int need_spectral = need_energy || need_freq || need_amp || need_entropy;
    const int need_center = need_moments || need_spectral;
    const double sample_count = (double)CWRU_WINDOW_SAMPLES;
    double sum = 0.0;
    double sum_square = 0.0;
    double mean = 0.0;
    double m2 = 0.0;
    double m3 = 0.0;
    double m4 = 0.0;
    double skewness = 0.0;
    double kurtosis = 0.0;
    uint32_t index;

    for (index = 0U; index < CWRU_WINDOW_SAMPLES; ++index)
    {
        const double value = (double)window[index][channel];
        if (need_center) sum += value;
        if (need_rms) sum_square += value * value;
    }
    if (need_center) mean = sum / sample_count;

    if (need_center)
    {
        for (index = 0U; index < CWRU_WINDOW_SAMPLES; ++index)
        {
            const double difference = (double)window[index][channel] - mean;
            if (need_moments)
            {
                const double difference_squared = difference * difference;
                m2 += difference_squared;
                if (need_skew) m3 += difference_squared * difference;
                if (need_kurt) m4 += difference_squared * difference_squared;
            }
            if (need_spectral) scratch_input[index] = (float32_t)difference;
        }
    }

    if (need_moments)
    {
        m2 /= sample_count;
        if (need_skew) m3 /= sample_count;
        if (need_kurt) m4 /= sample_count;
        if (m2 > 0.0)
        {
            if (need_skew)
            {
                const double standardized_third = m3 / (m2 * sqrt(m2));
                const double skew_correction = sqrt(sample_count * (sample_count - 1.0)) /
                    (sample_count - 2.0);
                skewness = skew_correction * standardized_third;
            }
            if (need_kurt)
            {
                const double b2 = m4 / (m2 * m2);
                kurtosis = (((sample_count * sample_count) - 1.0) * b2 -
                    3.0 * (sample_count - 1.0) * (sample_count - 1.0)) /
                    ((sample_count - 2.0) * (sample_count - 3.0));
            }
        }
    }

    if (need_spectral)
    {
        arm_rfft_fast_f32(&extractor->rfft, scratch_input, scratch_output, 0U);
        double spectral_energy = 0.0;
        double entropy = 0.0;
        double dominant_power = 0.0;
        uint32_t dominant_index = 1U;
        uint32_t non_zero_bins = 0U;
        double cached_dc_power = 0.0;
        if (need_freq || need_amp) dominant_power = mask_power_at_bin(scratch_output, 1U);

        for (index = 0U; index < CWRU_FE_RFFT_BINS; ++index)
        {
            const double power = mask_power_at_bin(scratch_output, index);
            if (need_energy || need_entropy) spectral_energy += power;
            if (need_entropy)
            {
                if (index == 0U) cached_dc_power = power;
                else mask_cache_store(scratch_input, index, power);
            }
            /* BIN1 inicial y comparacion estricta, tambien con potencia cero. */
            if ((need_freq || need_amp) && index > 1U && power > dominant_power)
            {
                dominant_power = power;
                dominant_index = index;
            }
        }

        if (need_entropy && spectral_energy > 0.0)
        {
            for (index = 0U; index < CWRU_FE_RFFT_BINS; ++index)
            {
                const double power = index == 0U ? cached_dc_power : mask_cache_load(scratch_input, index);
                if (power > 0.0)
                {
                    const double probability = power / spectral_energy;
                    entropy -= probability * mask_entropy_log_mixed(probability);
                    ++non_zero_bins;
                }
            }
            if (non_zero_bins > 1U) entropy /= log((double)CWRU_FE_RFFT_BINS);
            else entropy = 0.0;
        }

        /* Las escrituras seleccionadas se hacen en el mismo orden del control. */
        if (need_rms) output[CWRU_FE_RMS] = mask_finite_or_zero(sqrt(sum_square / sample_count));
        if (need_skew) output[CWRU_FE_SKEWNESS] = mask_finite_or_zero(skewness);
        if (need_kurt) output[CWRU_FE_KURTOSIS] = mask_finite_or_zero(kurtosis);
        if (need_energy) output[CWRU_FE_SPECTRAL_ENERGY] = mask_finite_or_zero(spectral_energy);
        if (need_freq) output[CWRU_FE_DOMINANT_FREQUENCY] = mask_finite_or_zero(
            (double)dominant_index * (double)MASK_FREQUENCY_RESOLUTION_HZ);
        if (need_amp) output[CWRU_FE_DOMINANT_AMPLITUDE] = mask_finite_or_zero(sqrt(dominant_power));
        if (need_entropy) output[CWRU_FE_SPECTRAL_ENTROPY] = mask_finite_or_zero(entropy);
    }
    else
    {
        if (need_rms) output[CWRU_FE_RMS] = mask_finite_or_zero(sqrt(sum_square / sample_count));
        if (need_skew) output[CWRU_FE_SKEWNESS] = mask_finite_or_zero(skewness);
        if (need_kurt) output[CWRU_FE_KURTOSIS] = mask_finite_or_zero(kurtosis);
    }
}

/* noinline mantiene simbolos auditables por mascara. La especializacion se
 * realiza dentro de cada wrapper, no mediante una mascara variable por dato. */
#define DEFINE_MASK_WRAPPER(name, de, fe) \
    MASK_WRAPPER void CWRUMASK_##name( \
        CWRU_FeatureExtractor *extractor, \
        const float32_t window[CWRU_WINDOW_SAMPLES][CWRU_NUM_CHANNELS], \
        float32_t features[CWRU_TOTAL_FEATURES], \
        float32_t scratch_input[CWRU_FE_SCRATCH_FLOATS], \
        float32_t scratch_output[CWRU_FE_SCRATCH_FLOATS]) \
    { \
        mask_channel(window, 0U, (de), features, extractor, scratch_input, scratch_output); \
        mask_channel(window, 1U, (fe), features + CWRU_NUM_FEATURES_PER_CHANNEL, \
            extractor, scratch_input, scratch_output); \
    }

DEFINE_MASK_WRAPPER(F14, 0x7fU, 0x7fU)
DEFINE_MASK_WRAPPER(T6, 0x07U, 0x07U)
DEFINE_MASK_WRAPPER(S8, 0x78U, 0x78U)
DEFINE_MASK_WRAPPER(NF12, 0x6fU, 0x6fU)
DEFINE_MASK_WRAPPER(NM10, 0x79U, 0x79U)
DEFINE_MASK_WRAPPER(NE12, 0x3fU, 0x3fU)
DEFINE_MASK_WRAPPER(DE7, 0x7fU, 0x00U)
DEFINE_MASK_WRAPPER(FE7, 0x00U, 0x7fU)
DEFINE_MASK_WRAPPER(FIX_DE13, 0x7fU, 0x6fU)
DEFINE_MASK_WRAPPER(FIX_FE13, 0x6fU, 0x7fU)

#undef DEFINE_MASK_WRAPPER

int CWRUMASK_Extract(
    uint32_t id,
    CWRU_FeatureExtractor *extractor,
    const float32_t window[CWRU_WINDOW_SAMPLES][CWRU_NUM_CHANNELS],
    float32_t features[CWRU_TOTAL_FEATURES],
    float32_t scratch_input[CWRU_FE_SCRATCH_FLOATS],
    float32_t scratch_output[CWRU_FE_SCRATCH_FLOATS])
{
    if (extractor == NULL || window == NULL || features == NULL ||
        scratch_input == NULL || scratch_output == NULL) return CWRU_FE_ERROR_NULL_POINTER;
    if (extractor->initialized == 0U) return CWRU_FE_ERROR_NOT_INITIALIZED;
    switch (id)
    {
    case 0U: CWRUMASK_F14(extractor, window, features, scratch_input, scratch_output); break;
    case 1U: CWRUMASK_T6(extractor, window, features, scratch_input, scratch_output); break;
    case 2U: CWRUMASK_S8(extractor, window, features, scratch_input, scratch_output); break;
    case 3U: CWRUMASK_NF12(extractor, window, features, scratch_input, scratch_output); break;
    case 4U: CWRUMASK_NM10(extractor, window, features, scratch_input, scratch_output); break;
    case 5U: CWRUMASK_NE12(extractor, window, features, scratch_input, scratch_output); break;
    case 6U: CWRUMASK_DE7(extractor, window, features, scratch_input, scratch_output); break;
    case 7U: CWRUMASK_FE7(extractor, window, features, scratch_input, scratch_output); break;
    case 8U: CWRUMASK_FIX_DE13(extractor, window, features, scratch_input, scratch_output); break;
    case 9U: CWRUMASK_FIX_FE13(extractor, window, features, scratch_input, scratch_output); break;
    default: return CWRUMASK_ERROR_ID;
    }
    return CWRU_FE_OK;
}
