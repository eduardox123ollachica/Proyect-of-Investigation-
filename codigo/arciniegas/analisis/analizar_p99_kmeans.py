from pathlib import Path
import sys

CODIGO_DIR = Path(__file__).resolve().parents[2]
if str(CODIGO_DIR) not in sys.path:
    sys.path.insert(0, str(CODIGO_DIR))

from comun.rutas_proyecto import RESULTADOS_KMEANS

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURACIÓN
# ============================================================

KMEANS_DIR = RESULTADOS_KMEANS


SCORES_PATH = (
    KMEANS_DIR
    / "scores_test_STANDARD.csv"
)

UMBRALES_PATH = (
    KMEANS_DIR
    / "umbrales_STANDARD.csv"
)


OUTPUT_DIR = (
    KMEANS_DIR
    / "Analisis_P99"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CARGAR DATOS
# ============================================================

df = pd.read_csv(
    SCORES_PATH
)

df_umbrales = pd.read_csv(
    UMBRALES_PATH
)


# ============================================================
# OBTENER P99 DIRECTAMENTE DEL EXPERIMENTO
# ============================================================

fila_p99 = df_umbrales[
    df_umbrales["metodo"] == "P99"
]

if len(fila_p99) != 1:
    raise ValueError(
        "No se encontró exactamente una fila P99."
    )


T_P99 = float(
    fila_p99.iloc[0]["umbral_d2"]
)


print("=" * 80)
print("ANÁLISIS DE VENTANAS SOBRE P99")
print("=" * 80)

print(
    f"\nUmbral P99 = {T_P99:.9f}"
)


# ============================================================
# FILTRAR VENTANAS RECHAZADAS
# ============================================================

rechazadas = df[
    df["score_d2"] > T_P99
].copy()


rechazadas = rechazadas.sort_values(
    "score_d2",
    ascending=False
).reset_index(
    drop=True
)


print(
    f"\nVentanas rechazadas: "
    f"{len(rechazadas)} / {len(df)}"
)

print(
    f"Porcentaje: "
    f"{len(rechazadas) / len(df) * 100:.6f} %"
)


# ============================================================
# MAGNITUD DEL EXCESO SOBRE EL UMBRAL
# ============================================================

rechazadas[
    "exceso_absoluto"
] = (
    rechazadas["score_d2"]
    -
    T_P99
)


rechazadas[
    "exceso_porcentual"
] = (
    (
        rechazadas["score_d2"]
        /
        T_P99
    )
    -
    1.0
) * 100.0


rechazadas[
    "razon_score_umbral"
] = (
    rechazadas["score_d2"]
    /
    T_P99
)


# ============================================================
# MOSTRAR COLUMNAS ÚTILES
# ============================================================

columnas_deseadas = [
    "file",
    "nombre_real",
    "label_real",
    "window",
    "start_sample",
    "end_sample_exclusive",
    "start_time_s",
    "end_time_s",
    "cluster_kmeans",
    "score_d2",
    "exceso_absoluto",
    "exceso_porcentual",
    "razon_score_umbral",
]

columnas_existentes = [
    col
    for col in columnas_deseadas
    if col in rechazadas.columns
]


print()
print("=" * 80)
print("LAS 12 VENTANAS RECHAZADAS")
print("=" * 80)

print(
    rechazadas[
        columnas_existentes
    ].to_string(
        index=False
    )
)


# ============================================================
# AGRUPACIÓN POR CLASE
# ============================================================

print()
print("=" * 80)
print("DISTRIBUCIÓN POR CLASE")
print("=" * 80)


if "nombre_real" in rechazadas.columns:

    por_clase = (
        rechazadas
        .groupby(
            "nombre_real"
        )
        .agg(
            ventanas_rechazadas=(
                "score_d2",
                "size"
            ),
            score_medio=(
                "score_d2",
                "mean"
            ),
            score_maximo=(
                "score_d2",
                "max"
            ),
        )
        .reset_index()
    )

    print(
        por_clase.to_string(
            index=False
        )
    )

else:

    por_clase = None


# ============================================================
# AGRUPACIÓN POR ARCHIVO
# ============================================================

print()
print("=" * 80)
print("CONCENTRACIÓN POR ARCHIVO")
print("=" * 80)


if "file" in rechazadas.columns:

    agregaciones = {
        "ventanas_rechazadas":
            (
                "score_d2",
                "size"
            ),

        "score_medio":
            (
                "score_d2",
                "mean"
            ),

        "score_maximo":
            (
                "score_d2",
                "max"
            ),

        "score_minimo":
            (
                "score_d2",
                "min"
            ),
    }


    if "nombre_real" in rechazadas.columns:

        agregaciones[
            "clase"
        ] = (
            "nombre_real",
            "first"
        )


    por_archivo = (
        rechazadas
        .groupby(
            "file"
        )
        .agg(
            **agregaciones
        )
        .sort_values(
            "ventanas_rechazadas",
            ascending=False
        )
        .reset_index()
    )


    print(
        por_archivo.to_string(
            index=False
        )
    )

else:

    por_archivo = None


# ============================================================
# PORCENTAJE RECHAZADO DENTRO DE CADA ARCHIVO
#
# Esto es importante:
#
# 2 ventanas rechazadas de 2 totales
# no significa lo mismo que
# 2 ventanas rechazadas de 36.
# ============================================================

if "file" in df.columns:

    total_por_archivo = (
        df
        .groupby(
            "file"
        )
        .size()
        .rename(
            "ventanas_totales"
        )
        .reset_index()
    )


    rechazadas_por_archivo = (
        rechazadas
        .groupby(
            "file"
        )
        .size()
        .rename(
            "ventanas_rechazadas"
        )
        .reset_index()
    )


    proporcion_archivos = (
        total_por_archivo
        .merge(
            rechazadas_por_archivo,
            on="file",
            how="left"
        )
    )


    proporcion_archivos[
        "ventanas_rechazadas"
    ] = (
        proporcion_archivos[
            "ventanas_rechazadas"
        ]
        .fillna(0)
        .astype(int)
    )


    proporcion_archivos[
        "porcentaje_rechazado"
    ] = (
        proporcion_archivos[
            "ventanas_rechazadas"
        ]
        /
        proporcion_archivos[
            "ventanas_totales"
        ]
        *
        100.0
    )


    proporcion_archivos = (
        proporcion_archivos[
            proporcion_archivos[
                "ventanas_rechazadas"
            ] > 0
        ]
        .sort_values(
            [
                "ventanas_rechazadas",
                "porcentaje_rechazado"
            ],
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )


    print()
    print("=" * 80)
    print("PROPORCIÓN RECHAZADA POR ARCHIVO")
    print("=" * 80)

    print(
        proporcion_archivos.to_string(
            index=False
        )
    )

else:

    proporcion_archivos = None


# ============================================================
# DETECTAR VENTANAS CONSECUTIVAS
#
# Como el hop es 8 y las ventanas tienen 50 % de overlap,
# window n y window n+1 son temporalmente adyacentes.
# ============================================================

grupos_consecutivos = []


if (
    "file" in rechazadas.columns
    and
    "window" in rechazadas.columns
):

    for nombre_archivo, grupo in (
        rechazadas
        .groupby(
            "file"
        )
    ):

        grupo = (
            grupo
            .sort_values(
                "window"
            )
            .copy()
        )


        ventanas = (
            grupo["window"]
            .astype(int)
            .to_numpy()
        )


        if len(ventanas) == 0:
            continue


        inicio = ventanas[0]
        anterior = ventanas[0]


        for actual in ventanas[1:]:

            if actual == anterior + 1:

                anterior = actual

            else:

                grupos_consecutivos.append(
                    {
                        "file":
                            nombre_archivo,

                        "window_inicio":
                            inicio,

                        "window_fin":
                            anterior,

                        "longitud":
                            anterior
                            -
                            inicio
                            +
                            1,
                    }
                )


                inicio = actual
                anterior = actual


        grupos_consecutivos.append(
            {
                "file":
                    nombre_archivo,

                "window_inicio":
                    inicio,

                "window_fin":
                    anterior,

                "longitud":
                    anterior
                    -
                    inicio
                    +
                    1,
            }
        )


    df_consecutivas = pd.DataFrame(
        grupos_consecutivos
    )


    df_consecutivas = (
        df_consecutivas
        .sort_values(
            "longitud",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )


    print()
    print("=" * 80)
    print("GRUPOS DE VENTANAS CONSECUTIVAS")
    print("=" * 80)

    print(
        df_consecutivas.to_string(
            index=False
        )
    )


else:

    df_consecutivas = None


# ============================================================
# DISTANCIA DE TODAS LAS VENTANAS AL UMBRAL
#
# Para ver las ventanas inmediatamente por debajo de P99.
# ============================================================

df_ordenado = (
    df
    .sort_values(
        "score_d2",
        ascending=False
    )
    .reset_index(
        drop=True
    )
)


print()
print("=" * 80)
print("20 SCORES MÁS ALTOS DE TEST")
print("=" * 80)


columnas_top = [
    "file",
    "nombre_real",
    "window",
    "cluster_kmeans",
    "score_d2",
]

columnas_top = [
    c
    for c in columnas_top
    if c in df_ordenado.columns
]


print(
    df_ordenado[
        columnas_top
    ]
    .head(20)
    .to_string(
        index=False
    )
)


# ============================================================
# GUARDAR RESULTADOS
# ============================================================

rechazadas.to_csv(
    OUTPUT_DIR
    / "ventanas_rechazadas_P99.csv",
    index=False
)


if por_clase is not None:

    por_clase.to_csv(
        OUTPUT_DIR
        / "resumen_por_clase.csv",
        index=False
    )


if por_archivo is not None:

    por_archivo.to_csv(
        OUTPUT_DIR
        / "resumen_por_archivo.csv",
        index=False
    )


if proporcion_archivos is not None:

    proporcion_archivos.to_csv(
        OUTPUT_DIR
        / "proporcion_rechazada_por_archivo.csv",
        index=False
    )


if df_consecutivas is not None:

    df_consecutivas.to_csv(
        OUTPUT_DIR
        / "grupos_ventanas_consecutivas.csv",
        index=False
    )


df_ordenado.head(
    50
).to_csv(
    OUTPUT_DIR
    / "top_50_scores_test.csv",
    index=False
)


# ============================================================
# RESUMEN AUTOMÁTICO
# ============================================================

lineas = []

lineas.append(
    "ANALISIS P99 - KMEANS STANDARD"
)

lineas.append(
    "=" * 60
)

lineas.append(
    f"Umbral P99: {T_P99:.9f}"
)

lineas.append(
    f"Ventanas test: {len(df)}"
)

lineas.append(
    f"Ventanas rechazadas: {len(rechazadas)}"
)

lineas.append(
    "False anomaly rate sobre dominio conocido: "
    f"{len(rechazadas) / len(df) * 100:.6f} %"
)


if "file" in rechazadas.columns:

    archivos_afectados = (
        rechazadas["file"]
        .nunique()
    )

    lineas.append(
        f"Archivos afectados: "
        f"{archivos_afectados}"
    )


if len(rechazadas) > 0:

    lineas.append(
        f"Score mínimo rechazado: "
        f"{rechazadas['score_d2'].min():.9f}"
    )

    lineas.append(
        f"Score máximo rechazado: "
        f"{rechazadas['score_d2'].max():.9f}"
    )

    lineas.append(
        f"Score medio rechazado: "
        f"{rechazadas['score_d2'].mean():.9f}"
    )


(
    OUTPUT_DIR
    / "resumen_analisis_P99.txt"
).write_text(
    "\n".join(
        lineas
    ),
    encoding="utf-8"
)


print()
print("=" * 80)
print("ANÁLISIS COMPLETADO")
print("=" * 80)

print(
    f"\nResultados guardados en:\n"
    f"{OUTPUT_DIR}"
)
