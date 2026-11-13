#include "kmeans_anomaly.h"
#include "kmeans_params.h"

#include <float.h>
#include <math.h>

int KMeans_Compute(
    const float features[21],
    KMeansResult *result
)
{
    if (features == nullptr || result == nullptr)
    {
        return -1;
    }

    float z[KMEANS_NUM_FEATURES];

    // StandardScaler
    for (int j = 0; j < KMEANS_NUM_FEATURES; ++j)
    {
        const float scale = kmeans_scaler_scale[j];

        if (scale == 0.0f)
        {
            return -2;
        }

        z[j] =
            (
                features[j]
                -
                kmeans_scaler_mean[j]
            )
            /
            scale;

        if (!isfinite(z[j]))
        {
            return -3;
        }
    }

    float mejor_d2 = FLT_MAX;
    int mejor_cluster = -1;

    // Distancia a los 32 centroides
    for (int k = 0; k < KMEANS_NUM_CLUSTERS; ++k)
    {
        float suma = 0.0f;

        for (int j = 0; j < KMEANS_NUM_FEATURES; ++j)
        {
            const float d =
                z[j]
                -
                kmeans_centroids[k][j];

            suma += d * d;
        }

        if (suma < mejor_d2)
        {
            mejor_d2 = suma;
            mejor_cluster = k;
        }
    }

    result->cluster = mejor_cluster;
    result->score_d2 = mejor_d2;

    result->anomaly =
        (mejor_d2 > kmeans_threshold_d2)
        ? 1
        : 0;

    return 0;
}
