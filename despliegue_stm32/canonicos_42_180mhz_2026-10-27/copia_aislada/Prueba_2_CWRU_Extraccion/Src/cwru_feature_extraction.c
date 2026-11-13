#include "cwru_feature_extraction.h"

#include <math.h>
#include <stddef.h>

/* 0: control; 1: normalizacion natural double; 2: lo anterior mas logf. */
#ifndef CWRU_ENTROPY_MODE
#define CWRU_ENTROPY_MODE 0
#endif

#if (CWRU_ENTROPY_MODE < 0) || (CWRU_ENTROPY_MODE > 2)
#error "CWRU_ENTROPY_MODE debe ser 0, 1 o 2."
#endif

#if CWRU_ENTROPY_MODE == 2
static double entropy_log_mixed(double probability)
{
    const float probability_f = (float)probability;

    /* Evitar logf(0) por subdesbordamiento, subnormales/FTZ y redondeo a 1.
     * Tambien preserva el comportamiento double en entradas no finitas. */
    if (!isnormal(probability_f) || (probability_f >= 1.0f))
    {
        return log(probability);
    }

    return (double)logf(probability_f);
}
#endif

#ifndef CWRU_CACHE_SPECTRAL_POWER
#define CWRU_CACHE_SPECTRAL_POWER 0
#endif

#if CWRU_CACHE_SPECTRAL_POWER
#include <string.h>

/* DC queda en un double local; los otros N/2 doubles ocupan N floats. */
_Static_assert(sizeof(double) == 2U * sizeof(float32_t),
    "La reutilizacion requiere double de 8 bytes y float32 de 4 bytes.");
_Static_assert((CWRU_FE_RFFT_BINS - 1U) * sizeof(double) <=
    CWRU_FE_SCRATCH_FLOATS * sizeof(float32_t),
    "Las potencias deben caber en el buffer de entrada FFT existente.");

/* Copia de representacion: no convertir potencias a float ni usar double*. */
static void cache_power_store(float32_t *workspace, uint32_t bin, double power)
{
    memcpy((unsigned char *)workspace + (bin - 1U) * sizeof(power),
        &power, sizeof(power));
}

static double cache_power_load(const float32_t *workspace, uint32_t bin)
{
    double power;
    memcpy(&power,
        (const unsigned char *)workspace + (bin - 1U) * sizeof(power),
        sizeof(power));
    return power;
}
#endif

#if CWRU_WINDOW_SAMPLES != 4096
#error "La RFFT de este modulo esta fijada a 4096 muestras."
#endif

#if CWRU_NUM_CHANNELS != 2
#error "Este modulo espera exactamente las senales DE y FE."
#endif

#if CWRU_NUM_FEATURES_PER_CHANNEL != 7
#error "El orden congelado contiene siete caracteristicas por senal."
#endif

#define CWRU_FE_FREQUENCY_RESOLUTION_HZ \
    ((float32_t)CWRU_SAMPLE_RATE_HZ / (float32_t)CWRU_WINDOW_SAMPLES)

#if CWRU_PROFILE_EXTRACTOR
#include "stm32f4xx.h"

static CWRU_ExtractionProfile g_profile;

static uint32_t profile_clock_read(void)
{
    /* Barreras CMSIS: serializacion hardware y clobber de memoria GCC. */
    __DSB();
    __ISB();
    return DWT->CYCCNT;
}

static void profile_reset(void)
{
    for (uint32_t channel = 0U; channel < CWRU_NUM_CHANNELS; ++channel)
        for (uint32_t block = 0U; block < CWRU_PROFILE_BLOCKS; ++block)
            g_profile.cycles[channel][block] = 0U;
}

const CWRU_ExtractionProfile *CWRU_FeatureExtraction_GetProfile(void)
{
    return &g_profile;
}

uint32_t CWRU_FeatureExtraction_ProfileReadOverhead(void)
{
    uint32_t best = UINT32_MAX;
    for (uint32_t i = 0U; i < 64U; ++i)
    {
        uint32_t start = profile_clock_read();
        uint32_t elapsed = profile_clock_read() - start;
        if (elapsed < best) best = elapsed;
    }
    return best;
}

/* El registro y una nueva lectura quedan fuera del intervalo del bloque. */
#define CWRU_PROFILE_MARK(block) do { \
    uint32_t profile_end = profile_clock_read(); \
    g_profile.cycles[channel][(block)] = profile_end - profile_start; \
    profile_start = profile_clock_read(); \
} while (0)
#endif

/* Convierte resultados no finitos igual que el script Python del proyecto. */
static float32_t finite_or_zero(double value)
{
    if (!isfinite(value))
    {
        return 0.0f;
    }

    return (float32_t)value;
}

/*
 * arm_rfft_fast_f32 empaqueta N/2+1 bins dentro de N floats:
 *   output[0]       = Re(X[0])
 *   output[1]       = Re(X[N/2])
 *   output[2*k]     = Re(X[k]), 1 <= k < N/2
 *   output[2*k + 1] = Im(X[k]), 1 <= k < N/2
 */
static double rfft_power_at_bin(const float32_t *fft_output, uint32_t bin)
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

static void extract_channel_features(
    const float32_t window[CWRU_WINDOW_SAMPLES][CWRU_NUM_CHANNELS],
    uint32_t channel,
    float32_t output[CWRU_NUM_FEATURES_PER_CHANNEL],
    CWRU_FeatureExtractor *extractor,
    float32_t scratch_input[CWRU_FE_SCRATCH_FLOATS],
    float32_t scratch_output[CWRU_FE_SCRATCH_FLOATS]
)
{
    const double sample_count = (double)CWRU_WINDOW_SAMPLES;
    double sum = 0.0;
    double sum_square = 0.0;
    double mean;
    double m2 = 0.0;
    double m3 = 0.0;
    double m4 = 0.0;
    double skewness = 0.0;
    double kurtosis = 0.0;
    uint32_t index;

#if CWRU_PROFILE_EXTRACTOR
    uint32_t profile_start = profile_clock_read();
#endif

    /* RMS sobre la señal original y media usada solamente para centrar. */
    for (index = 0U; index < CWRU_WINDOW_SAMPLES; ++index)
    {
        const double value = (double)window[index][channel];
        sum += value;
        sum_square += value * value;
    }

    mean = sum / sample_count;

#if CWRU_PROFILE_EXTRACTOR
    CWRU_PROFILE_MARK(CWRU_PROFILE_RMS_MEAN);
#endif

    /* Momentos centrales con la misma convención de SciPy (bias=False). */
    for (index = 0U; index < CWRU_WINDOW_SAMPLES; ++index)
    {
        const double difference =
            (double)window[index][channel] - mean;
        const double difference_squared = difference * difference;

        m2 += difference_squared;
        m3 += difference_squared * difference;
        m4 += difference_squared * difference_squared;

        /* Solo la ruta espectral recibe la señal sin componente DC. */
        scratch_input[index] = (float32_t)difference;
    }

#if CWRU_PROFILE_EXTRACTOR
    CWRU_PROFILE_MARK(CWRU_PROFILE_MOMENTS_CENTERING);
#endif

    m2 /= sample_count;
    m3 /= sample_count;
    m4 /= sample_count;

    if (m2 > 0.0)
    {
        const double standardized_third = m3 / (m2 * sqrt(m2));
        const double skew_correction =
            sqrt(sample_count * (sample_count - 1.0)) /
            (sample_count - 2.0);
        const double b2 = m4 / (m2 * m2);

        skewness = skew_correction * standardized_third;
        kurtosis =
            (((sample_count * sample_count) - 1.0) * b2 -
             3.0 * (sample_count - 1.0) * (sample_count - 1.0)) /
            ((sample_count - 2.0) * (sample_count - 3.0));
    }

#if CWRU_PROFILE_EXTRACTOR
    CWRU_PROFILE_MARK(CWRU_PROFILE_MOMENT_CORRECTIONS);
#endif

    /* FFT directa sin normalización, igual que numpy.fft.rfft. */
    arm_rfft_fast_f32(
        &extractor->rfft,
        scratch_input,
        scratch_output,
        0U
    );

#if CWRU_PROFILE_EXTRACTOR
    CWRU_PROFILE_MARK(CWRU_PROFILE_FFT);
#endif

    {
        double spectral_energy = 0.0;
        double entropy = 0.0;
        double dominant_power = rfft_power_at_bin(scratch_output, 1U);
        uint32_t dominant_index = 1U;
        uint32_t non_zero_bins = 0U;
#if CWRU_CACHE_SPECTRAL_POWER
        double cached_dc_power = 0.0;
#endif

        /* Incluye DC y Nyquist en energía y entropía. */
        for (index = 0U; index < CWRU_FE_RFFT_BINS; ++index)
        {
            const double power = rfft_power_at_bin(scratch_output, index);
            spectral_energy += power;

#if CWRU_CACHE_SPECTRAL_POWER
            /* La RFFT termino; scratch_input ya no contiene datos necesarios.
             * Se reutilizan sus bytes sin modificar scratch_output ni window.
             * memcpy evita aliasing de tipos y no exige alineacion double. */
            if (index == 0U)
            {
                cached_dc_power = power;
            }
            else
            {
                cache_power_store(scratch_input, index, power);
            }
#endif

            /* '>' conserva el primer bin en empates, como numpy.argmax. */
            if ((index > 1U) && (power > dominant_power))
            {
                dominant_power = power;
                dominant_index = index;
            }
        }

#if CWRU_PROFILE_EXTRACTOR
        CWRU_PROFILE_MARK(CWRU_PROFILE_ENERGY_DOMINANT);
#endif

        if (spectral_energy > 0.0)
        {
            for (index = 0U; index < CWRU_FE_RFFT_BINS; ++index)
            {
#if CWRU_CACHE_SPECTRAL_POWER
                const double power = (index == 0U) ? cached_dc_power :
                    cache_power_load(scratch_input, index);
#else
                const double power =
                    rfft_power_at_bin(scratch_output, index);
#endif

                if (power > 0.0)
                {
                    const double probability = power / spectral_energy;
#if CWRU_ENTROPY_MODE == 1
                    entropy -= probability * log(probability);
#elif CWRU_ENTROPY_MODE == 2
                    entropy -= probability * entropy_log_mixed(probability);
#else
                    entropy -= probability * (log(probability) / log(2.0));
#endif
                    ++non_zero_bins;
                }
            }

            if (non_zero_bins > 1U)
            {
#if CWRU_ENTROPY_MODE != 0
                /* Cambio de base cancelado: -sum(p*ln(p))/ln(numero_bins).
                 * Misma definicion matematica; el redondeo puede diferir. */
                entropy /= log((double)CWRU_FE_RFFT_BINS);
#else
                entropy /=
                    (log((double)CWRU_FE_RFFT_BINS) / log(2.0));
#endif
            }
            else
            {
                entropy = 0.0;
            }
        }

#if CWRU_PROFILE_EXTRACTOR
        CWRU_PROFILE_MARK(CWRU_PROFILE_ENTROPY);
#endif

        output[CWRU_FE_RMS] =
            finite_or_zero(sqrt(sum_square / sample_count));
        output[CWRU_FE_SKEWNESS] = finite_or_zero(skewness);
        output[CWRU_FE_KURTOSIS] = finite_or_zero(kurtosis);
        output[CWRU_FE_SPECTRAL_ENERGY] =
            finite_or_zero(spectral_energy);
        output[CWRU_FE_DOMINANT_FREQUENCY] =
            finite_or_zero(
                (double)dominant_index *
                (double)CWRU_FE_FREQUENCY_RESOLUTION_HZ
            );
        output[CWRU_FE_DOMINANT_AMPLITUDE] =
            finite_or_zero(sqrt(dominant_power));
        output[CWRU_FE_SPECTRAL_ENTROPY] = finite_or_zero(entropy);

#if CWRU_PROFILE_EXTRACTOR
        CWRU_PROFILE_MARK(CWRU_PROFILE_OUTPUT);
#endif
    }
}

int CWRU_FeatureExtraction_Init(CWRU_FeatureExtractor *extractor)
{
    arm_status status;

    if (extractor == NULL)
    {
        return CWRU_FE_ERROR_NULL_POINTER;
    }

    extractor->initialized = 0U;

    /* La función específica evita enlazar tablas FFT de tamaños no usados. */
    status = arm_rfft_fast_init_4096_f32(&extractor->rfft);
    if (status != ARM_MATH_SUCCESS)
    {
        return CWRU_FE_ERROR_FFT_INIT;
    }

    extractor->initialized = 1U;
    return CWRU_FE_OK;
}

int CWRU_FeatureExtraction_Extract(
    CWRU_FeatureExtractor *extractor,
    const float32_t window[CWRU_WINDOW_SAMPLES][CWRU_NUM_CHANNELS],
    float32_t features[CWRU_TOTAL_FEATURES],
    float32_t scratch_input[CWRU_FE_SCRATCH_FLOATS],
    float32_t scratch_output[CWRU_FE_SCRATCH_FLOATS]
)
{
    uint32_t channel;

    if ((extractor == NULL) ||
        (window == NULL) ||
        (features == NULL) ||
        (scratch_input == NULL) ||
        (scratch_output == NULL))
    {
        return CWRU_FE_ERROR_NULL_POINTER;
    }

    if (extractor->initialized == 0U)
    {
        return CWRU_FE_ERROR_NOT_INITIALIZED;
    }

#if CWRU_PROFILE_EXTRACTOR
    profile_reset();
#endif

    for (channel = 0U; channel < CWRU_NUM_CHANNELS; ++channel)
    {
        extract_channel_features(
            window,
            channel,
            &features[channel * CWRU_NUM_FEATURES_PER_CHANNEL],
            extractor,
            scratch_input,
            scratch_output
        );
    }

    return CWRU_FE_OK;
}
