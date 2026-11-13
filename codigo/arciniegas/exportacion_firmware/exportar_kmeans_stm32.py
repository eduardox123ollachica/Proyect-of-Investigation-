from pathlib import Path
import sys
import numpy as np

CODIGO_DIR = Path(__file__).resolve().parents[2]
if str(CODIGO_DIR) not in sys.path:
    sys.path.insert(0, str(CODIGO_DIR))

from comun.rutas_proyecto import DATOS_ARCINIEGAS_PROCESADO, RESULTADOS_KMEANS

BASE = RESULTADOS_KMEANS

OUT = BASE / "STM32_EXPORT"
OUT.mkdir(parents=True, exist_ok=True)


mean = np.load(
    BASE / "scaler_mean_STANDARD.npy"
)

scale = np.load(
    BASE / "scaler_scale_STANDARD.npy"
)

centroids = np.load(
    BASE / "centroides_STANDARD.npy"
)


THRESHOLD = 21.850406448


# ============================================================
# VALIDACIONES
# ============================================================

assert mean.shape == (21,)
assert scale.shape == (21,)
assert centroids.shape == (32, 21)

if np.any(scale == 0):
    raise ValueError("Existe scale = 0")


# ============================================================
# CONVERSIÓN A FLOAT32
# ============================================================

mean32 = mean.astype(np.float32)
scale32 = scale.astype(np.float32)
centroids32 = centroids.astype(np.float32)
threshold32 = np.float32(THRESHOLD)


# ============================================================
# HEADER
# ============================================================

header = """\
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
"""

(OUT / "kmeans_params.h").write_text(
    header,
    encoding="utf-8"
)


# ============================================================
# UTILIDADES PARA C
# ============================================================

def float_c(x):
    return f"{float(x):.9e}f"


# ============================================================
# SOURCE
# ============================================================

lines = []

lines.append('#include "kmeans_params.h"')
lines.append("")

lines.append(
    "const float kmeans_scaler_mean"
    "[KMEANS_NUM_FEATURES] = {"
)

for x in mean32:
    lines.append(f"    {float_c(x)},")

lines.append("};")
lines.append("")


lines.append(
    "const float kmeans_scaler_scale"
    "[KMEANS_NUM_FEATURES] = {"
)

for x in scale32:
    lines.append(f"    {float_c(x)},")

lines.append("};")
lines.append("")


lines.append(
    "const float kmeans_centroids"
    "[KMEANS_NUM_CLUSTERS]"
    "[KMEANS_NUM_FEATURES] = {"
)

for k in range(32):

    lines.append("    {")

    for j in range(21):
        lines.append(
            f"        {float_c(centroids32[k, j])},"
        )

    lines.append("    },")

lines.append("};")
lines.append("")


lines.append(
    "const float kmeans_threshold_d2 = "
    f"{float_c(threshold32)};"
)


(OUT / "kmeans_params.cpp").write_text(
    "\n".join(lines),
    encoding="utf-8"
)


# ============================================================
# COMPROBAR FLOAT32 CONTRA REFERENCIA
# ============================================================

X_test = np.load(
    DATOS_ARCINIEGAS_PROCESADO / "X_test.npy"
).astype(np.float64)


def scores_reference(X):

    Z = (
        X - mean
    ) / scale

    scores = []
    clusters = []

    for x in Z:

        d2 = np.sum(
            (centroids - x) ** 2,
            axis=1
        )

        k = int(
            np.argmin(d2)
        )

        clusters.append(k)
        scores.append(d2[k])

    return (
        np.asarray(scores),
        np.asarray(clusters),
    )


def scores_float32(X):

    X32 = X.astype(
        np.float32
    )

    Z = (
        X32 - mean32
    ) / scale32

    scores = np.empty(
        len(X32),
        dtype=np.float32
    )

    clusters = np.empty(
        len(X32),
        dtype=np.int32
    )

    for i, x in enumerate(Z):

        mejor = np.float32(
            np.inf
        )

        mejor_k = -1

        for k in range(32):

            suma = np.float32(
                0.0
            )

            for j in range(21):

                d = np.float32(
                    x[j]
                    -
                    centroids32[k, j]
                )

                suma = np.float32(
                    suma
                    +
                    np.float32(
                        d * d
                    )
                )

            if suma < mejor:

                mejor = suma
                mejor_k = k

        scores[i] = mejor
        clusters[i] = mejor_k

    return (
        scores,
        clusters,
    )


score_ref, cluster_ref = (
    scores_reference(
        X_test
    )
)

score_f32, cluster_f32 = (
    scores_float32(
        X_test
    )
)


flags_ref = (
    score_ref
    >
    THRESHOLD
)

flags_f32 = (
    score_f32
    >
    threshold32
)


print("=" * 70)
print("EXPORTACIÓN K-MEANS STM32")
print("=" * 70)

print(
    "Mean shape       :",
    mean32.shape
)

print(
    "Scale shape      :",
    scale32.shape
)

print(
    "Centroids shape  :",
    centroids32.shape
)

print(
    "Threshold float32:",
    threshold32
)

print()

print(
    "Clusters diferentes:",
    np.sum(
        cluster_ref
        !=
        cluster_f32
    )
)

print(
    "Flags diferentes:",
    np.sum(
        flags_ref
        !=
        flags_f32
    )
)

print(
    "MAE score:",
    np.mean(
        np.abs(
            score_ref
            -
            score_f32
        )
    )
)

print(
    "Max error score:",
    np.max(
        np.abs(
            score_ref
            -
            score_f32
        )
    )
)

print()

print(
    "Archivos generados en:"
)

print(OUT)
