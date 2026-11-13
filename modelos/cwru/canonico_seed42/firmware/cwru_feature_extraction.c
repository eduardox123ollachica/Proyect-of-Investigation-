#include "cwru_feature_extraction.h"

#include <math.h>
#include <stddef.h>

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

    /* RMS sobre la señal original y media usada solamente para centrar. */
    for (index = 0U; index < CWRU_WINDOW_SAMPLES; ++index)
    {
        const double value = (double)window[index][channel];
        sum += value;
        sum_square += value * value;
    }

    mean = sum / sample_count;

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

    /* FFT directa sin normalización, igual que numpy.fft.rfft. */
    arm_rfft_fast_f32(
        &extractor->rfft,
        scratch_input,
        scratch_output,
        0U
    );

    {
        double spectral_energy = 0.0;
        double entropy = 0.0;
        double dominant_power = rfft_power_at_bin(scratch_output, 1U);
        uint32_t dominant_index = 1U;
        uint32_t non_zero_bins = 0U;

        /* Incluye DC y Nyquist en energía y entropía. */
        for (index = 0U; index < CWRU_FE_RFFT_BINS; ++index)
        {
            const double power = rfft_power_at_bin(scratch_output, index);
            spectral_energy += power;

            /* '>' conserva el primer bin en empates, como numpy.argmax. */
            if ((index > 1U) && (power > dominant_power))
            {
                dominant_power = power;
                dominant_index = index;
            }
        }

        if (spectral_energy > 0.0)
        {
            for (index = 0U; index < CWRU_FE_RFFT_BINS; ++index)
            {
                const double power =
                    rfft_power_at_bin(scratch_output, index);

                if (power > 0.0)
                {
                    const double probability = power / spectral_energy;
                    entropy -= probability * (log(probability) / log(2.0));
                    ++non_zero_bins;
                }
            }

            if (non_zero_bins > 1U)
            {
                entropy /=
                    (log((double)CWRU_FE_RFFT_BINS) / log(2.0));
            }
            else
            {
                entropy = 0.0;
            }
        }

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
