#ifndef KMEANS_PARAMS_H
#define KMEANS_PARAMS_H

#ifdef __cplusplus
extern "C" {
#endif

#define KMEANS_NUM_CLUSTERS 32
#define KMEANS_NUM_FEATURES 21

extern const float kmeans_scaler_mean[KMEANS_NUM_FEATURES];
extern const float kmeans_scaler_scale[KMEANS_NUM_FEATURES];

extern const float kmeans_centroids[
    KMEANS_NUM_CLUSTERS
][KMEANS_NUM_FEATURES];

extern const float kmeans_threshold_d2;

#ifdef __cplusplus
}
#endif

#endif
