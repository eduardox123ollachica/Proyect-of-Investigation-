#ifndef FEATURE_EXTRACTION_H
#define FEATURE_EXTRACTION_H

#ifdef __cplusplus
extern "C" {
#endif


/* ============================================================
 * CONFIGURACIÓN DEL PREPROCESAMIENTO
 * ============================================================ */

/* Número de muestras por ventana */
#define FE_WINDOW_SIZE       16

/* Número de ejes: ax, ay, az */
#define FE_AXIS_COUNT         3

/* Características calculadas por cada eje */
#define FE_FEATURES_PER_AXIS  7

/* Total: 7 características × 3 ejes */
#define FE_TOTAL_FEATURES    21

/* Frecuencia de muestreo */
#define FE_SAMPLE_RATE_HZ   100.0f

/* Número de bins de una RFFT de N=16:
 *
 * N/2 + 1 = 9
 */
#define FE_RFFT_BINS          9


/* ============================================================
 * ÍNDICES DE LAS CARACTERÍSTICAS
 *
 * El orden coincide exactamente con el utilizado en Python.
 * ============================================================ */

enum
{
    FE_RMS = 0,
    FE_SKEWNESS,
    FE_KURTOSIS,
    FE_SPECTRAL_ENERGY,
    FE_DOMINANT_FREQUENCY,
    FE_DOMINANT_AMPLITUDE,
    FE_SPECTRAL_ENTROPY
};


/* ============================================================
 * EXTRACCIÓN DE CARACTERÍSTICAS
 * ============================================================
 *
 * window[n][0] -> ax
 * window[n][1] -> ay
 * window[n][2] -> az
 *
 * La entrada tiene:
 *
 *      16 muestras × 3 ejes
 *
 * La salida contiene:
 *
 *      21 características
 *
 * Orden:
 *
 *  0  ax RMS
 *  1  ax Skewness
 *  2  ax Kurtosis
 *  3  ax Spectral Energy
 *  4  ax Dominant Frequency
 *  5  ax Dominant Amplitude
 *  6  ax Spectral Entropy
 *
 *  7  ay RMS
 *  ...
 * 13  ay Spectral Entropy
 *
 * 14  az RMS
 *  ...
 * 20  az Spectral Entropy
 *
 * Retorna:
 *
 *  0  -> cálculo correcto
 * -1  -> puntero inválido
 *
 * ============================================================ */

int FeatureExtraction_Extract(
    const float window[FE_WINDOW_SIZE][FE_AXIS_COUNT],
    float features[FE_TOTAL_FEATURES]
);


#ifdef __cplusplus
}
#endif

#endif
