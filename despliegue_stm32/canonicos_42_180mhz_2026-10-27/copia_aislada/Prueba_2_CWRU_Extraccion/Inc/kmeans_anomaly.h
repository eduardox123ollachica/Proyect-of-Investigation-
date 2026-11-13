#ifndef KMEANS_ANOMALY_H
#define KMEANS_ANOMALY_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct
{
    int cluster;
    float score_d2;
    int anomaly;

} KMeansResult;

int KMeans_Compute(
    const float features[21],
    KMeansResult *result
);

#ifdef __cplusplus
}
#endif

#endif
