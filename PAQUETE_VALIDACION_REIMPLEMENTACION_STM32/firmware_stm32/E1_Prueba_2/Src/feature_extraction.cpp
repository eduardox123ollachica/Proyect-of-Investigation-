#include "feature_extraction.h"

#include "arm_math.h"
#include "arm_const_structs.h"

#include <math.h>
#include <stddef.h>


/* ============================================================
 * CONSTANTES INTERNAS
 * ============================================================ */

namespace
{

/*
 * Resolución espectral:
 *
 * Fs / N = 100 / 16 = 6.25 Hz
 */
constexpr float FREQUENCY_RESOLUTION =
    FE_SAMPLE_RATE_HZ / FE_WINDOW_SIZE;


/*
 * La implementación CMSIS-DSP utilizada aquí depende
 * explícitamente de una CFFT de 16 puntos.
 */
static_assert(
    FE_WINDOW_SIZE == 16,
    "CMSIS-DSP CFFT configurada para 16 muestras"
);

static_assert(
    FE_RFFT_BINS == 9,
    "Para N=16 deben existir 9 bins equivalentes a np.fft.rfft"
);


/* ============================================================
 * PROTECCIÓN NUMÉRICA
 * ============================================================ */

static float finite_or_zero(float value)
{
    if (!isfinite(value))
    {
        return 0.0f;
    }

    return value;
}


/* ============================================================
 * EXTRACCIÓN DE LAS 7 CARACTERÍSTICAS DE UN EJE
 * ============================================================ */

static void extract_axis_features(
    const float signal[FE_WINDOW_SIZE],
    float output[FE_FEATURES_PER_AXIS]
)
{
    /* ========================================================
     * 1. MEDIA
     * ======================================================== */

    float mean = 0.0f;

    for (int i = 0; i < FE_WINDOW_SIZE; i++)
    {
        mean += signal[i];
    }

    mean /= (float)FE_WINDOW_SIZE;


    /* ========================================================
     * 2. RMS
     *
     * RMS = sqrt(
     *          (1/N) sum(x[i]^2)
     *       )
     *
     * IMPORTANTE:
     * se calcula sobre la señal ORIGINAL.
     * ======================================================== */

    float sum_square = 0.0f;

    for (int i = 0; i < FE_WINDOW_SIZE; i++)
    {
        sum_square +=
            signal[i] * signal[i];
    }

    float rms =
        sqrtf(
            sum_square
            / (float)FE_WINDOW_SIZE
        );


    /* ========================================================
     * 3. MOMENTOS CENTRALES
     *
     * m2 = (1/N) sum((x - mean)^2)
     * m3 = (1/N) sum((x - mean)^3)
     * m4 = (1/N) sum((x - mean)^4)
     * ======================================================== */

    float m2 = 0.0f;
    float m3 = 0.0f;
    float m4 = 0.0f;

    for (int i = 0; i < FE_WINDOW_SIZE; i++)
    {
        float d =
            signal[i] - mean;

        float d2 =
            d * d;

        m2 += d2;

        m3 +=
            d2 * d;

        m4 +=
            d2 * d2;
    }

    m2 /= (float)FE_WINDOW_SIZE;
    m3 /= (float)FE_WINDOW_SIZE;
    m4 /= (float)FE_WINDOW_SIZE;


    /* ========================================================
     * 4. SKEWNESS
     *
     * scipy.stats.skew(
     *     signal,
     *     bias=False
     * )
     *
     * g1 = m3 / m2^(3/2)
     *
     * skew =
     *
     * sqrt(N(N-1))
     * ------------- * g1
     *      N-2
     * ======================================================== */

    float skewness = 0.0f;

    /*
     * Si m2 = 0 significa que todas las muestras son
     * prácticamente iguales.
     *
     * SciPy produciría un valor no finito y posteriormente
     * el script Python lo convierte a 0.
     */
    if (m2 > 0.0f)
    {
        float denominator =
            m2 * sqrtf(m2);

        float g1 =
            m3 / denominator;

        const float N =
            (float)FE_WINDOW_SIZE;

        float correction =
            sqrtf(
                N * (N - 1.0f)
            )
            /
            (N - 2.0f);

        skewness =
            correction * g1;
    }

    skewness =
        finite_or_zero(
            skewness
        );


    /* ========================================================
     * 5. KURTOSIS FISHER
     *
     * scipy.stats.kurtosis(
     *     signal,
     *     fisher=True,
     *     bias=False
     * )
     *
     * b2 = m4 / m2^2
     *
     *              (N²-1)b2 - 3(N-1)²
     * kurtosis = ------------------------
     *                 (N-2)(N-3)
     * ======================================================== */

    float kurtosis = 0.0f;

    if (m2 > 0.0f)
    {
        const float N =
            (float)FE_WINDOW_SIZE;

        float b2 =
            m4
            /
            (m2 * m2);

        float numerator =
            (
                (N * N - 1.0f)
                * b2
            )
            -
            (
                3.0f
                * (N - 1.0f)
                * (N - 1.0f)
            );

        float denominator =
            (N - 2.0f)
            *
            (N - 3.0f);

        kurtosis =
            numerator
            /
            denominator;
    }

    kurtosis =
        finite_or_zero(
            kurtosis
        );


    /* ========================================================
     * 6. ELIMINAR COMPONENTE DC PARA LA FFT
     *
     * x_ac[i] = x[i] - mean
     *
     * La eliminación de la media se realiza ÚNICAMENTE
     * para las características espectrales.
     * ======================================================== */

    float x_ac[FE_WINDOW_SIZE];

    for (int i = 0; i < FE_WINDOW_SIZE; i++)
    {
        x_ac[i] =
            signal[i] - mean;
    }


    /* ========================================================
     * 7. FFT CON CMSIS-DSP
     *
     * La implementación anterior calculaba directamente:
     *
     * X[k] =
     * sum(
     *     x[n] * exp(-j*2*pi*k*n/N)
     * )
     *
     * Ahora se utiliza la CFFT optimizada de CMSIS-DSP:
     *
     * arm_cfft_f32()
     *
     * Para representar una señal real mediante una CFFT:
     *
     * entrada =
     *
     * [x0, 0,
     *  x1, 0,
     *  ...
     *  x15, 0]
     *
     * La salida queda:
     *
     * [Re(X0), Im(X0),
     *  Re(X1), Im(X1),
     *  ...
     *  Re(X15), Im(X15)]
     *
     * Como la señal de entrada es real, para reproducir
     * np.fft.rfft solamente se utilizan:
     *
     * k = 0 ... 8
     *
     * La FFT directa de CMSIS-DSP no aplica escalamiento,
     * manteniendo la misma convención de np.fft.rfft.
     * ======================================================== */

    float32_t fft_buffer[
        2 * FE_WINDOW_SIZE
    ] = {0.0f};


    /*
     * Convertir señal real centrada a representación
     * compleja intercalada.
     */
    for (int n = 0; n < FE_WINDOW_SIZE; n++)
    {
        fft_buffer[2 * n] =
            x_ac[n];

        fft_buffer[2 * n + 1] =
            0.0f;
    }


    /*
     * FFT compleja de 16 puntos.
     *
     * ifftFlag       = 0 -> FFT directa
     * bitReverseFlag = 1 -> salida en orden natural
     */
    arm_cfft_f32(
        &arm_cfft_sR_f32_len16,
        fft_buffer,
        0,
        1
    );


    /*
     * CMSIS-DSP calcula las magnitudes de los
     * 16 bins complejos.
     */
    float32_t magnitude_full[
        FE_WINDOW_SIZE
    ];

    arm_cmplx_mag_f32(
        fft_buffer,
        magnitude_full,
        FE_WINDOW_SIZE
    );


    /*
     * Solo conservamos los 9 bins equivalentes
     * a np.fft.rfft para N=16.
     */
    float magnitude[FE_RFFT_BINS];
    float power[FE_RFFT_BINS];

    for (int k = 0; k < FE_RFFT_BINS; k++)
    {
        magnitude[k] =
            magnitude_full[k];

        power[k] =
            magnitude[k]
            *
            magnitude[k];
    }


    /* ========================================================
     * 8. ENERGÍA ESPECTRAL
     *
     * spectral_energy =
     *
     * sum(
     *     magnitude[k]^2
     * )
     *
     * Incluye los 9 bins de la RFFT.
     * ======================================================== */

    float spectral_energy =
        0.0f;

    for (int k = 0; k < FE_RFFT_BINS; k++)
    {
        spectral_energy +=
            power[k];
    }


    /* ========================================================
     * 9. FRECUENCIA DOMINANTE
     *
     * Se ignora explícitamente k=0.
     *
     * np.argmax(magnitude[1:]) + 1
     *
     * IMPORTANTE:
     *
     * Se utiliza '>' y NO '>='.
     *
     * De esa manera, en caso de empate se conserva el primer
     * bin, igual que np.argmax().
     * ======================================================== */

    int dominant_index = 1;

    float dominant_magnitude =
        magnitude[1];

    for (int k = 2; k < FE_RFFT_BINS; k++)
    {
        if (
            magnitude[k]
            >
            dominant_magnitude
        )
        {
            dominant_magnitude =
                magnitude[k];

            dominant_index =
                k;
        }
    }

    float dominant_frequency =
        (float)dominant_index
        *
        FREQUENCY_RESOLUTION;

    float dominant_amplitude =
        magnitude[dominant_index];


    /* ========================================================
     * 10. ENTROPÍA ESPECTRAL
     *
     * p[k] = power[k] / total_power
     *
     * H =
     *
     * -sum(
     *     p[k] * log2(p[k])
     * )
     *
     * Hnorm =
     *
     * H / log2(9)
     * ======================================================== */

    float spectral_entropy =
        0.0f;

    float total_power =
        spectral_energy;

    if (total_power > 0.0f)
    {
        float entropy =
            0.0f;

        int non_zero_bins =
            0;

        for (int k = 0; k < FE_RFFT_BINS; k++)
        {
            if (power[k] > 0.0f)
            {
                float p =
                    power[k]
                    /
                    total_power;

                entropy -=
                    p
                    *
                    log2f(p);

                non_zero_bins++;
            }
        }

        /*
         * El Python retorna 0 cuando queda 0 o 1
         * componente espectral no nula.
         */
        if (non_zero_bins > 1)
        {
            float max_entropy =
                log2f(
                    (float)FE_RFFT_BINS
                );

            if (max_entropy > 0.0f)
            {
                spectral_entropy =
                    entropy
                    /
                    max_entropy;
            }
        }
    }

    spectral_entropy =
        finite_or_zero(
            spectral_entropy
        );


    /* ========================================================
     * 11. GUARDAR LAS 7 FEATURES DEL EJE
     * ======================================================== */

    output[FE_RMS] =
        finite_or_zero(rms);

    output[FE_SKEWNESS] =
        skewness;

    output[FE_KURTOSIS] =
        kurtosis;

    output[FE_SPECTRAL_ENERGY] =
        finite_or_zero(
            spectral_energy
        );

    output[FE_DOMINANT_FREQUENCY] =
        finite_or_zero(
            dominant_frequency
        );

    output[FE_DOMINANT_AMPLITUDE] =
        finite_or_zero(
            dominant_amplitude
        );

    output[FE_SPECTRAL_ENTROPY] =
        spectral_entropy;
}

} /* namespace */


/* ============================================================
 * FUNCIÓN PÚBLICA
 * ============================================================ */

extern "C" int FeatureExtraction_Extract(
    const float window[FE_WINDOW_SIZE][FE_AXIS_COUNT],
    float features[FE_TOTAL_FEATURES]
)
{
    if (
        window == nullptr
        ||
        features == nullptr
    )
    {
        return -1;
    }


    /* ========================================================
     * Procesar:
     *
     * axis = 0 -> ax
     * axis = 1 -> ay
     * axis = 2 -> az
     * ======================================================== */

    for (int axis = 0; axis < FE_AXIS_COUNT; axis++)
    {
        float signal[
            FE_WINDOW_SIZE
        ];


        /* Extraer las 16 muestras del eje */
        for (
            int n = 0;
            n < FE_WINDOW_SIZE;
            n++
        )
        {
            signal[n] =
                window[n][axis];
        }


        /*
         * Posición inicial dentro del vector
         * de 21 características.
         *
         * ax -> 0
         * ay -> 7
         * az -> 14
         */
        int offset =
            axis
            *
            FE_FEATURES_PER_AXIS;


        extract_axis_features(
            signal,
            &features[offset]
        );
    }


    return 0;
}
