#include "cwru_normality_params.h"

const uint8_t cwru_normality_feature_indices[CWRU_NORMALITY_NUM_FEATURES] = {
    0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 12, 13,
};

const float cwru_normality_scaler_mean[CWRU_NORMALITY_NUM_FEATURES] = {
    6.165274233e-02f,
    2.287803590e-02f,
    -6.472888589e-02f,
    3.079793359e+04f,
    7.536759949e+01f,
    5.833395123e-01f,
    7.025116682e-02f,
    3.726961091e-02f,
    -9.480717778e-02f,
    3.310170312e+04f,
    7.688312531e+01f,
    5.682620406e-01f,
};

const float cwru_normality_scaler_scale[CWRU_NORMALITY_NUM_FEATURES] = {
    5.297460593e-03f,
    3.519118577e-02f,
    1.063688025e-01f,
    5.708224121e+03f,
    1.179195881e+01f,
    2.764130943e-02f,
    5.994489882e-03f,
    4.057900608e-02f,
    1.321938783e-01f,
    7.550308594e+03f,
    9.995051384e+00f,
    3.008041158e-02f,
};

const float cwru_normality_centroids[CWRU_NORMALITY_NUM_CLUSTERS][CWRU_NORMALITY_NUM_FEATURES] = {
    {
        -2.624259992e-07f,
        -9.934107759e-09f,
        -1.036872490e-07f,
        8.775128180e-08f,
        -1.399053531e-07f,
        1.821253051e-07f,
        4.967053613e-08f,
        6.622738358e-09f,
        -1.473559337e-07f,
        2.400742716e-08f,
        -1.092751774e-07f,
        -1.019901674e-06f,
    },
};

const float cwru_normality_threshold_d2 = 3.099235535e+02f;
