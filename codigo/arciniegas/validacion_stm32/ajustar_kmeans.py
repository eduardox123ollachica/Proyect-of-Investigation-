from pathlib import Path
import sys

CODIGO_DIR = Path(__file__).resolve().parents[2]
if str(CODIGO_DIR) not in sys.path:
    sys.path.insert(0, str(CODIGO_DIR))

from comun.rutas_proyecto import DATOS_ARCINIEGAS_PROCESADO, RESULTADOS_KMEANS

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

BASE_PREPROCESADO = DATOS_ARCINIEGAS_PROCESADO
OUTPUT_DIR = RESULTADOS_KMEANS

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# ARCHIVOS DE ENTRADA
# ============================================================

X_TRAIN_PATH = (
    BASE_PREPROCESADO
    / "X_train.npy"
)

Y_TRAIN_PATH = (
    BASE_PREPROCESADO
    / "y_train.npy"
)

X_TEST_PATH = (
    BASE_PREPROCESADO
    / "X_test.npy"
)

Y_TEST_PATH = (
    BASE_PREPROCESADO
    / "y_test.npy"
)

METADATA_PATH = (
    BASE_PREPROCESADO
    / "metadata_features.csv"
)


# ============================================================
# PARÁMETROS K-MEANS
# ============================================================

N_CLUSTERS = 32

RANDOM_STATE = 42

N_INIT = 20


# ============================================================
# CLASES
# ============================================================

CLASES = {
    0: "DETENIDO",
    1: "VELOCIDAD MEDIA",
    2: "VELOCIDAD ALTA",
}


# ============================================================
# CARGAR DATOS
# ============================================================

def cargar_datos():

    X_train = np.load(
        X_TRAIN_PATH,
        allow_pickle=False
    ).astype(
        np.float64
    )

    y_train = np.load(
        Y_TRAIN_PATH,
        allow_pickle=False
    ).reshape(-1)


    X_test = np.load(
        X_TEST_PATH,
        allow_pickle=False
    ).astype(
        np.float64
    )

    y_test = np.load(
        Y_TEST_PATH,
        allow_pickle=False
    ).reshape(-1)


    metadata = pd.read_csv(
        METADATA_PATH
    )


    metadata_test = (
        metadata[
            metadata["split"]
            ==
            "testing"
        ]
        .reset_index(
            drop=True
        )
    )


    return (
        X_train,
        y_train,
        X_test,
        y_test,
        metadata_test,
    )


# ============================================================
# DISTANCIA AL CENTROIDE MÁS CERCANO
#
# IMPORTANTE:
#
# Se utiliza distancia euclídea AL CUADRADO:
#
# d²(x,c) = sum((x_j - c_j)²)
#
# No necesitamos sqrt porque:
#
# - preserva exactamente el orden de las distancias
# - genera el mismo centroide más cercano
# - permite aplicar el mismo criterio de umbral
# - será más barato posteriormente en STM32
# ============================================================

def calcular_scores(
    X,
    centroides
):

    scores = np.empty(
        len(X),
        dtype=np.float64
    )

    cluster_asignado = np.empty(
        len(X),
        dtype=np.int32
    )


    for i in range(
        len(X)
    ):

        diferencias = (
            centroides
            -
            X[i]
        )


        distancias2 = np.sum(
            diferencias
            *
            diferencias,
            axis=1
        )


        cluster = int(
            np.argmin(
                distancias2
            )
        )


        scores[i] = (
            distancias2[
                cluster
            ]
        )


        cluster_asignado[i] = (
            cluster
        )


    return (
        scores,
        cluster_asignado
    )


# ============================================================
# UMBRALES CANDIDATOS
#
# NO significan todavía que alguno sea el umbral final.
#
# Todos se calculan exclusivamente con TRAINING.
# ============================================================

def calcular_umbrales(
    scores_train
):

    media = float(
        np.mean(
            scores_train
        )
    )


    std = float(
        np.std(
            scores_train,
            ddof=1
        )
    )


    return {

        "P95":
            float(
                np.percentile(
                    scores_train,
                    95
                )
            ),

        "P99":
            float(
                np.percentile(
                    scores_train,
                    99
                )
            ),

        "MEDIA_MAS_3STD":
            media
            +
            3.0 * std,

        "MAX_TRAIN":
            float(
                np.max(
                    scores_train
                )
            ),
    }


# ============================================================
# ESTADÍSTICAS
# ============================================================

def estadisticas_scores(
    scores
):

    return {

        "cantidad":
            len(scores),

        "media":
            float(
                np.mean(scores)
            ),

        "std":
            float(
                np.std(
                    scores,
                    ddof=1
                )
            ),

        "min":
            float(
                np.min(scores)
            ),

        "p50":
            float(
                np.percentile(
                    scores,
                    50
                )
            ),

        "p90":
            float(
                np.percentile(
                    scores,
                    90
                )
            ),

        "p95":
            float(
                np.percentile(
                    scores,
                    95
                )
            ),

        "p99":
            float(
                np.percentile(
                    scores,
                    99
                )
            ),

        "max":
            float(
                np.max(scores)
            ),
    }


# ============================================================
# ANALIZAR UNA VARIANTE
# ============================================================

def analizar_variante(
    nombre,
    X_train,
    X_test,
    y_train,
    y_test,
    metadata_test,
    usar_scaler
):

    print()

    print("=" * 80)

    print(
        f"K-MEANS - {nombre}"
    )

    print("=" * 80)


    # ========================================================
    # ESCALADO
    # ========================================================

    if usar_scaler:

        scaler = (
            StandardScaler()
        )


        X_train_proc = (
            scaler.fit_transform(
                X_train
            )
        )


        X_test_proc = (
            scaler.transform(
                X_test
            )
        )


        np.save(
            OUTPUT_DIR
            /
            f"scaler_mean_{nombre}.npy",
            scaler.mean_
        )


        np.save(
            OUTPUT_DIR
            /
            f"scaler_scale_{nombre}.npy",
            scaler.scale_
        )


    else:

        scaler = None

        X_train_proc = (
            X_train.copy()
        )

        X_test_proc = (
            X_test.copy()
        )


    # ========================================================
    # ENTRENAR K-MEANS
    # ========================================================

    modelo = KMeans(
        n_clusters=N_CLUSTERS,
        random_state=RANDOM_STATE,
        n_init=N_INIT
    )


    modelo.fit(
        X_train_proc
    )


    centroides = (
        modelo.cluster_centers_
        .astype(
            np.float64
        )
    )


    print(
        f"Centroides       : "
        f"{centroides.shape}"
    )


    print(
        f"Inertia          : "
        f"{modelo.inertia_:.6f}"
    )


    # ========================================================
    # GUARDAR CENTROIDES
    # ========================================================

    np.save(
        OUTPUT_DIR
        /
        f"centroides_{nombre}.npy",
        centroides
    )


    pd.DataFrame(
        centroides
    ).to_csv(
        OUTPUT_DIR
        /
        f"centroides_{nombre}.csv",
        index=False
    )


    # ========================================================
    # SCORES
    # ========================================================

    (
        scores_train,
        clusters_train,
    ) = calcular_scores(
        X_train_proc,
        centroides
    )


    (
        scores_test,
        clusters_test,
    ) = calcular_scores(
        X_test_proc,
        centroides
    )


    # ========================================================
    # UMBRALES CANDIDATOS
    # ========================================================

    umbrales = calcular_umbrales(
        scores_train
    )


    # ========================================================
    # RESULTADOS TEST
    # ========================================================

    df_test = (
        metadata_test.copy()
    )


    df_test[
        "label_real"
    ] = y_test


    df_test[
        "nombre_real"
    ] = [
        CLASES[
            int(y)
        ]
        for y in y_test
    ]


    df_test[
        "cluster_kmeans"
    ] = clusters_test


    df_test[
        "score_d2"
    ] = scores_test


    # ========================================================
    # APLICAR UMBRALES SOLO PARA ESTUDIO
    # ========================================================

    for nombre_umbral, valor in (
        umbrales.items()
    ):

        df_test[
            f"anomalia_{nombre_umbral}"
        ] = (
            scores_test
            >
            valor
        )


    df_test.to_csv(
        OUTPUT_DIR
        /
        f"scores_test_{nombre}.csv",
        index=False
    )


    # ========================================================
    # TRAIN SCORES
    # ========================================================

    df_train = pd.DataFrame(
        {
            "label":
                y_train,

            "nombre":
                [
                    CLASES[
                        int(y)
                    ]
                    for y in y_train
                ],

            "cluster_kmeans":
                clusters_train,

            "score_d2":
                scores_train,
        }
    )


    df_train.to_csv(
        OUTPUT_DIR
        /
        f"scores_train_{nombre}.csv",
        index=False
    )


    # ========================================================
    # OCUPACIÓN DE CLUSTERS
    # ========================================================

    ocupacion_train = (
        pd.Series(
            clusters_train
        )
        .value_counts()
        .reindex(
            range(
                N_CLUSTERS
            ),
            fill_value=0
        )
        .sort_index()
    )


    ocupacion_test = (
        pd.Series(
            clusters_test
        )
        .value_counts()
        .reindex(
            range(
                N_CLUSTERS
            ),
            fill_value=0
        )
        .sort_index()
    )


    df_ocupacion = pd.DataFrame(
        {
            "cluster":
                range(
                    N_CLUSTERS
                ),

            "train":
                ocupacion_train.values,

            "test":
                ocupacion_test.values,
        }
    )


    df_ocupacion.to_csv(
        OUTPUT_DIR
        /
        f"ocupacion_clusters_{nombre}.csv",
        index=False
    )


    # ========================================================
    # ESTADÍSTICAS GLOBALES
    # ========================================================

    stats_train = (
        estadisticas_scores(
            scores_train
        )
    )


    stats_test = (
        estadisticas_scores(
            scores_test
        )
    )


    # ========================================================
    # ESTADÍSTICAS POR CLASE EN TEST
    #
    # Esto es únicamente DESCRIPTIVO.
    #
    # Las etiquetas NO se utilizaron para entrenar K-means.
    # ========================================================

    estadisticas_clase = []


    for label, nombre_clase in (
        CLASES.items()
    ):

        mascara = (
            y_test
            ==
            label
        )


        scores_clase = (
            scores_test[
                mascara
            ]
        )


        stats = (
            estadisticas_scores(
                scores_clase
            )
        )


        stats[
            "clase"
        ] = label


        stats[
            "nombre"
        ] = nombre_clase


        estadisticas_clase.append(
            stats
        )


    pd.DataFrame(
        estadisticas_clase
    ).to_csv(
        OUTPUT_DIR
        /
        f"estadisticas_por_clase_{nombre}.csv",
        index=False
    )


    # ========================================================
    # FALSE ANOMALY RATE
    #
    # Debido a que test contiene las tres condiciones
    # CONOCIDAS, cualquier detección como "anomalía"
    # se interpreta provisionalmente como rechazo de una
    # muestra perteneciente al dominio conocido.
    # ========================================================

    resultados_umbral = []


    for nombre_umbral, valor in (
        umbrales.items()
    ):

        anomalas_test = (
            scores_test
            >
            valor
        )


        cantidad = int(
            np.sum(
                anomalas_test
            )
        )


        porcentaje = (
            cantidad
            /
            len(scores_test)
            *
            100.0
        )


        fila = {

            "metodo":
                nombre_umbral,

            "umbral_d2":
                valor,

            "test_rechazadas":
                cantidad,

            "test_total":
                len(scores_test),

            "false_anomaly_rate_pct":
                porcentaje,
        }


        for label, nombre_clase in (
            CLASES.items()
        ):

            mascara = (
                y_test
                ==
                label
            )


            cantidad_clase = int(
                np.sum(
                    anomalas_test[
                        mascara
                    ]
                )
            )


            total_clase = int(
                np.sum(
                    mascara
                )
            )


            fila[
                f"rechazadas_{nombre_clase}"
            ] = cantidad_clase


            fila[
                f"total_{nombre_clase}"
            ] = total_clase


        resultados_umbral.append(
            fila
        )


    df_umbrales = pd.DataFrame(
        resultados_umbral
    )


    df_umbrales.to_csv(
        OUTPUT_DIR
        /
        f"umbrales_{nombre}.csv",
        index=False
    )


    # ========================================================
    # RESUMEN TXT
    # ========================================================

    resumen = f"""
================================================================================
K-MEANS - {nombre}
================================================================================

Naturaleza:
Detector no supervisado de alejamiento respecto al dominio conocido.

Clusters:
{N_CLUSTERS}

Features:
21

StandardScaler:
{"SI" if usar_scaler else "NO"}

Random state:
{RANDOM_STATE}

n_init:
{N_INIT}

IMPORTANTE:

K=32 corresponde al valor reportado por el artículo.

random_state, n_init, el uso/no uso de StandardScaler,
la definición explícita del score y los criterios de umbral
son decisiones de implementación de esta réplica.


================================================================================
DATOS
================================================================================

X_train:
{X_train.shape}

X_test:
{X_test.shape}

Las etiquetas NO fueron utilizadas durante model.fit().


================================================================================
TRAIN - SCORE d²
================================================================================

{stats_train}


================================================================================
TEST - SCORE d²
================================================================================

{stats_test}


================================================================================
UMBRALES CANDIDATOS
================================================================================

{df_umbrales.to_string(index=False)}


================================================================================
INTERPRETACIÓN
================================================================================

score_d2 = distancia euclídea cuadrática al centroide
K-means más cercano.

Un score mayor indica que la ventana se encuentra más alejada
de los patrones representados por X_train.

En esta etapa NO se interpreta todavía un score elevado como
una anomalía real confirmada, debido a que no existe ground
truth explícito de anomalías en el dataset disponible.
"""


    (
        OUTPUT_DIR
        /
        f"resumen_{nombre}.txt"
    ).write_text(
        resumen,
        encoding="utf-8"
    )


    print()

    print(
        "Scores TRAIN:"
    )

    print(
        stats_train
    )


    print()

    print(
        "Scores TEST:"
    )

    print(
        stats_test
    )


    print()

    print(
        "Umbrales candidatos:"
    )

    print(
        df_umbrales.to_string(
            index=False
        )
    )


    return {

        "nombre":
            nombre,

        "modelo":
            modelo,

        "scaler":
            scaler,

        "centroides":
            centroides,

        "scores_train":
            scores_train,

        "scores_test":
            scores_test,

        "umbrales":
            umbrales,

        "resultados_umbral":
            df_umbrales,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)

    print(
        "EXPERIMENTO 5"
    )

    print(
        "K-MEANS PARA DETECCIÓN DE ALEJAMIENTO / ANOMALÍAS"
    )

    print("=" * 80)


    (
        X_train,
        y_train,
        X_test,
        y_test,
        metadata_test,
    ) = cargar_datos()


    print()

    print(
        f"X_train : {X_train.shape}"
    )

    print(
        f"y_train : {y_train.shape}"
    )

    print(
        f"X_test  : {X_test.shape}"
    )

    print(
        f"y_test  : {y_test.shape}"
    )


    if X_train.shape[1] != 21:

        raise ValueError(
            "Se esperaban 21 características."
        )


    if X_test.shape[1] != 21:

        raise ValueError(
            "Se esperaban 21 características."
        )


    if len(metadata_test) != len(
        X_test
    ):

        raise ValueError(
            "metadata testing y X_test "
            "no tienen la misma cantidad."
        )


    # ========================================================
    # VARIANTE 1
    #
    # Features originales.
    # ========================================================

    resultado_raw = (
        analizar_variante(
            nombre="RAW",
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            metadata_test=metadata_test,
            usar_scaler=False
        )
    )


    # ========================================================
    # VARIANTE 2
    #
    # StandardScaler ajustado SOLO con training.
    # ========================================================

    resultado_scaled = (
        analizar_variante(
            nombre="STANDARD",
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            metadata_test=metadata_test,
            usar_scaler=True
        )
    )


    # ========================================================
    # COMPARACIÓN
    # ========================================================

    comparacion = []


    for resultado in [
        resultado_raw,
        resultado_scaled,
    ]:

        for _, fila in (
            resultado[
                "resultados_umbral"
            ]
            .iterrows()
        ):

            comparacion.append(
                {
                    "variante":
                        resultado[
                            "nombre"
                        ],

                    "metodo_umbral":
                        fila[
                            "metodo"
                        ],

                    "umbral_d2":
                        fila[
                            "umbral_d2"
                        ],

                    "false_anomaly_rate_pct":
                        fila[
                            "false_anomaly_rate_pct"
                        ],

                    "test_rechazadas":
                        fila[
                            "test_rechazadas"
                        ],

                    "test_total":
                        fila[
                            "test_total"
                        ],
                }
            )


    df_comparacion = pd.DataFrame(
        comparacion
    )


    df_comparacion.to_csv(
        OUTPUT_DIR
        /
        "comparacion_raw_vs_standard.csv",
        index=False
    )


    print()

    print("=" * 80)

    print(
        "COMPARACIÓN RAW vs STANDARD"
    )

    print("=" * 80)

    print(
        df_comparacion.to_string(
            index=False
        )
    )


    print()

    print(
        "Resultados guardados en:"
    )

    print(
        OUTPUT_DIR
    )


# ============================================================
# EJECUTAR
# ============================================================

if __name__ == "__main__":

    main()
