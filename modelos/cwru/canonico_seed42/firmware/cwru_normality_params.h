#ifndef CWRU_NORMALITY_PARAMS_H_
#define CWRU_NORMALITY_PARAMS_H_

#include <stdint.h>

#define CWRU_NORMALITY_NUM_CLUSTERS 1
#define CWRU_NORMALITY_NUM_FEATURES 12

extern const uint8_t cwru_normality_feature_indices[CWRU_NORMALITY_NUM_FEATURES];
extern const float cwru_normality_scaler_mean[CWRU_NORMALITY_NUM_FEATURES];
extern const float cwru_normality_scaler_scale[CWRU_NORMALITY_NUM_FEATURES];
extern const float cwru_normality_centroids[CWRU_NORMALITY_NUM_CLUSTERS][CWRU_NORMALITY_NUM_FEATURES];
extern const float cwru_normality_threshold_d2;

#endif  // CWRU_NORMALITY_PARAMS_H_
