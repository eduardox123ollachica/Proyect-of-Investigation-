from pathlib import Path
import sys

CODIGO_DIR = Path(__file__).resolve().parents[2]
if str(CODIGO_DIR) not in sys.path:
    sys.path.insert(0, str(CODIGO_DIR))

from comun.rutas_proyecto import RESULTADOS_E2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURACION
# ============================================================

VERSION = "sin_class_weight"


# Carpeta del experimento E2 analizado originalmente por este script.
SCRIPT_DIR = RESULTADOS_E2 / "entrenamiento"


# CSV de probabilidades
CSV_PATH = SCRIPT_DIR / "probabilidades_test.csv"


# Carpeta de salida
OUTPUT_DIR = (
    SCRIPT_DIR
    / f"analisis_probabilidades_{VERSION}"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# COLUMNAS REALES DEL CSV
# ============================================================

TRUE_CLASS_COL = "class"

PROBABILITY_COLUMNS = [
    "P_DETENIDO",
    "P_MEDIA",
    "P_ALTA",
]


# Mapeo probabilidad -> nombre de clase
PROB_TO_CLASS = {
    "P_DETENIDO": "DETENIDO",
    "P_MEDIA": "VELOCIDAD MEDIA",
    "P_ALTA": "VELOCIDAD ALTA",
}


# Mapeo clase real -> columna de probabilidad
CLASS_TO_PROB = {
    "DETENIDO": "P_DETENIDO",
    "VELOCIDAD MEDIA": "P_MEDIA",
    "VELOCIDAD ALTA": "P_ALTA",
}


# ============================================================
# CARGAR CSV
# ============================================================

print("\n")
print("=" * 80)
print("ANALISIS DE PROBABILIDADES SOFTMAX")
print("=" * 80)

print(
    f"Version analizada: {VERSION}"
)

print(
    f"CSV: {CSV_PATH.resolve()}"
)

print(
    f"Salida: {OUTPUT_DIR.resolve()}"
)


if not CSV_PATH.exists():

    raise FileNotFoundError(
        f"\nNo existe el archivo:\n"
        f"{CSV_PATH.resolve()}"
    )


df = pd.read_csv(
    CSV_PATH
)


print("\nColumnas encontradas:")

for col in df.columns:

    print(
        f"  - {col}"
    )


# ============================================================
# VALIDAR COLUMNAS
# ============================================================

required_columns = [
    TRUE_CLASS_COL,
    *PROBABILITY_COLUMNS,
]


missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]


if missing_columns:

    raise ValueError(
        f"Faltan columnas en el CSV: "
        f"{missing_columns}"
    )


# ============================================================
# NORMALIZAR CLASE REAL
# ============================================================

df[TRUE_CLASS_COL] = (
    df[TRUE_CLASS_COL]
    .astype(str)
    .str.strip()
    .str.upper()
)


print("\nClases reales encontradas:")

print(
    df[TRUE_CLASS_COL]
    .value_counts()
)


unknown_classes = (
    set(
        df[TRUE_CLASS_COL]
        .unique()
    )
    -
    set(
        CLASS_TO_PROB.keys()
    )
)


if unknown_classes:

    raise ValueError(
        f"Existen clases reales no reconocidas: "
        f"{unknown_classes}"
    )


# ============================================================
# CONVERTIR PROBABILIDADES A NUMERICO
# ============================================================

for col in PROBABILITY_COLUMNS:

    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )


if df[PROBABILITY_COLUMNS].isna().any().any():

    raise ValueError(
        "Existen valores no numericos o NaN "
        "en las columnas de probabilidades."
    )


# ============================================================
# RECONSTRUIR PREDICCION DESDE SOFTMAX
# ============================================================

# La clase predicha es la que tenga la mayor probabilidad

df["prediccion_softmax"] = (
    df[
        PROBABILITY_COLUMNS
    ]
    .idxmax(
        axis=1
    )
    .map(
        PROB_TO_CLASS
    )
)


print("\nClases predichas desde Softmax:")

print(
    df[
        "prediccion_softmax"
    ]
    .value_counts()
)


# ============================================================
# PREDICCION CORRECTA
# ============================================================

df["correct"] = (
    df[TRUE_CLASS_COL]
    ==
    df["prediccion_softmax"]
)


# ============================================================
# PROBABILIDAD DE LA CLASE VERDADERA
# ============================================================

def get_true_probability(row):

    true_class = row[
        TRUE_CLASS_COL
    ]

    probability_column = (
        CLASS_TO_PROB[
            true_class
        ]
    )

    return row[
        probability_column
    ]


df["P_true"] = df.apply(
    get_true_probability,
    axis=1
)


# ============================================================
# MATRIZ DE PROBABILIDADES
# ============================================================

prob_matrix = df[
    PROBABILITY_COLUMNS
].to_numpy(
    dtype=float
)


# ============================================================
# PROBABILIDAD MAXIMA Y SEGUNDA MAYOR
# ============================================================

sorted_probs = np.sort(
    prob_matrix,
    axis=1
)


df["P_second"] = (
    sorted_probs[:, -2]
)


df["P_max"] = (
    sorted_probs[:, -1]
)


# ============================================================
# MARGEN DE DECISION
# ============================================================

# Si la prediccion es correcta:
# margin sera positivo.
#
# Si la prediccion es incorrecta:
# margin sera negativo.

df["margin"] = (
    df["P_true"]
    -
    df["P_second"]
)


# ============================================================
# RESUMEN GLOBAL
# ============================================================

accuracy = (
    df["correct"]
    .mean()
)


mean_p_true = (
    df["P_true"]
    .mean()
)


mean_confidence = (
    df["P_max"]
    .mean()
)


mean_margin = (
    df["margin"]
    .mean()
)


correct_conf_90 = (
    (
        df["correct"]
        &
        (
            df["P_max"]
            >= 0.90
        )
    )
    .mean()
)


correct_conf_95 = (
    (
        df["correct"]
        &
        (
            df["P_max"]
            >= 0.95
        )
    )
    .mean()
)


num_errors = (
    (~df["correct"])
    .sum()
)


print("\n")
print("=" * 80)
print("RESUMEN GLOBAL")
print("=" * 80)


print(
    f"Numero de ventanas: "
    f"{len(df)}"
)


print(
    f"Accuracy: "
    f"{accuracy * 100:.2f}%"
)


print(
    f"P verdadera promedio: "
    f"{mean_p_true:.4f}"
)


print(
    f"Confianza maxima promedio: "
    f"{mean_confidence:.4f}"
)


print(
    f"Margen promedio: "
    f"{mean_margin:.4f}"
)


print(
    f"Correctas con confianza >= 0.90: "
    f"{correct_conf_90 * 100:.2f}%"
)


print(
    f"Correctas con confianza >= 0.95: "
    f"{correct_conf_95 * 100:.2f}%"
)


print(
    f"Predicciones incorrectas: "
    f"{num_errors}"
)


# ============================================================
# GUARDAR RESUMEN GLOBAL
# ============================================================

global_summary = pd.DataFrame(
    [
        {
            "version":
                VERSION,

            "samples":
                len(df),

            "accuracy":
                accuracy,

            "mean_P_true":
                mean_p_true,

            "mean_confidence":
                mean_confidence,

            "mean_margin":
                mean_margin,

            "correct_conf_gt_090":
                correct_conf_90,

            "correct_conf_gt_095":
                correct_conf_95,

            "num_errors":
                num_errors,
        }
    ]
)


global_summary.to_csv(
    OUTPUT_DIR
    / "resumen_global.csv",
    index=False
)


# ============================================================
# RESUMEN POR CLASE
# ============================================================

summary_rows = []


for class_name in CLASS_TO_PROB.keys():

    subset = df[
        df[TRUE_CLASS_COL]
        ==
        class_name
    ].copy()


    if len(subset) == 0:
        continue


    summary_rows.append(
        {
            "version":
                VERSION,

            "class":
                class_name,

            "samples":
                len(subset),

            "accuracy":
                subset[
                    "correct"
                ].mean(),

            "mean_P_true":
                subset[
                    "P_true"
                ].mean(),

            "median_P_true":
                subset[
                    "P_true"
                ].median(),

            "mean_margin":
                subset[
                    "margin"
                ].mean(),

            "median_margin":
                subset[
                    "margin"
                ].median(),

            "mean_confidence":
                subset[
                    "P_max"
                ].mean(),

            "correct_conf_gt_090":
                (
                    subset["correct"]
                    &
                    (
                        subset["P_max"]
                        >=
                        0.90
                    )
                ).mean(),

            "correct_conf_gt_095":
                (
                    subset["correct"]
                    &
                    (
                        subset["P_max"]
                        >=
                        0.95
                    )
                ).mean(),

            "errors":
                (
                    ~subset[
                        "correct"
                    ]
                ).sum(),
        }
    )


summary_df = pd.DataFrame(
    summary_rows
)


print("\n")
print("=" * 80)
print("RESUMEN POR CLASE")
print("=" * 80)


print(
    summary_df
    .round(4)
    .to_string(
        index=False
    )
)


summary_df.to_csv(
    OUTPUT_DIR
    / "resumen_probabilidades_por_clase.csv",
    index=False
)


# ============================================================
# PROBABILIDADES MEDIAS POR CLASE REAL
# ============================================================

mean_prob_rows = []


for class_name in CLASS_TO_PROB.keys():

    subset = df[
        df[TRUE_CLASS_COL]
        ==
        class_name
    ]


    if len(subset) == 0:
        continue


    mean_prob_rows.append(
        {
            "version":
                VERSION,

            "true_class":
                class_name,

            "samples":
                len(subset),

            "P_DETENIDO_mean":
                subset[
                    "P_DETENIDO"
                ].mean(),

            "P_MEDIA_mean":
                subset[
                    "P_MEDIA"
                ].mean(),

            "P_ALTA_mean":
                subset[
                    "P_ALTA"
                ].mean(),
        }
    )


mean_probs_df = pd.DataFrame(
    mean_prob_rows
)


print("\n")
print("=" * 80)
print("PROBABILIDADES MEDIAS POR CLASE REAL")
print("=" * 80)


print(
    mean_probs_df
    .round(4)
    .to_string(
        index=False
    )
)


mean_probs_df.to_csv(
    OUTPUT_DIR
    / "probabilidades_medias_por_clase_real.csv",
    index=False
)


# ============================================================
# ANALISIS ESPECIFICO DE VELOCIDAD MEDIA
# ============================================================

media_df = df[
    df[TRUE_CLASS_COL]
    ==
    "VELOCIDAD MEDIA"
].copy()


if len(media_df) > 0:

    print("\n")
    print("=" * 80)
    print("ANALISIS VELOCIDAD MEDIA")
    print("=" * 80)


    print(
        f"Muestras MEDIA: "
        f"{len(media_df)}"
    )


    print(
        f"Correctas: "
        f"{media_df['correct'].sum()}"
    )


    print(
        f"Incorrectas: "
        f"{(~media_df['correct']).sum()}"
    )


    print(
        f"P_MEDIA promedio: "
        f"{media_df['P_MEDIA'].mean():.4f}"
    )


    print(
        f"P_ALTA promedio: "
        f"{media_df['P_ALTA'].mean():.4f}"
    )


    print(
        f"P_DETENIDO promedio: "
        f"{media_df['P_DETENIDO'].mean():.4f}"
    )


    print(
        f"P verdadera promedio: "
        f"{media_df['P_true'].mean():.4f}"
    )


    print(
        f"Margen promedio: "
        f"{media_df['margin'].mean():.4f}"
    )


# ============================================================
# MEDIA CORRECTAS VS INCORRECTAS
# ============================================================

comparison_rows = []


if len(media_df) > 0:

    media_correct = media_df[
        media_df["correct"]
    ]


    media_wrong = media_df[
        ~media_df["correct"]
    ]


    if len(media_correct) > 0:

        comparison_rows.append(
            {
                "version":
                    VERSION,

                "group":
                    "MEDIA_correctas",

                "samples":
                    len(media_correct),

                "P_DETENIDO":
                    media_correct[
                        "P_DETENIDO"
                    ].mean(),

                "P_MEDIA":
                    media_correct[
                        "P_MEDIA"
                    ].mean(),

                "P_ALTA":
                    media_correct[
                        "P_ALTA"
                    ].mean(),

                "P_true":
                    media_correct[
                        "P_true"
                    ].mean(),

                "P_max":
                    media_correct[
                        "P_max"
                    ].mean(),

                "margin":
                    media_correct[
                        "margin"
                    ].mean(),
            }
        )


    if len(media_wrong) > 0:

        comparison_rows.append(
            {
                "version":
                    VERSION,

                "group":
                    "MEDIA_incorrectas",

                "samples":
                    len(media_wrong),

                "P_DETENIDO":
                    media_wrong[
                        "P_DETENIDO"
                    ].mean(),

                "P_MEDIA":
                    media_wrong[
                        "P_MEDIA"
                    ].mean(),

                "P_ALTA":
                    media_wrong[
                        "P_ALTA"
                    ].mean(),

                "P_true":
                    media_wrong[
                        "P_true"
                    ].mean(),

                "P_max":
                    media_wrong[
                        "P_max"
                    ].mean(),

                "margin":
                    media_wrong[
                        "margin"
                    ].mean(),
            }
        )


comparison_df = pd.DataFrame(
    comparison_rows
)


if len(comparison_df) > 0:

    print("\n")
    print("=" * 80)
    print(
        "VELOCIDAD MEDIA: "
        "CORRECTAS VS INCORRECTAS"
    )
    print("=" * 80)


    print(
        comparison_df
        .round(4)
        .to_string(
            index=False
        )
    )


    comparison_df.to_csv(
        OUTPUT_DIR
        / "media_correctas_vs_incorrectas.csv",
        index=False
    )


# ============================================================
# MATRIZ DE CONFUSION
# ============================================================

CLASS_ORDER = [
    "DETENIDO",
    "VELOCIDAD MEDIA",
    "VELOCIDAD ALTA",
]


confusion_matrix = pd.crosstab(
    df[TRUE_CLASS_COL],
    df["prediccion_softmax"],
    rownames=[
        "REAL"
    ],
    colnames=[
        "PREDICCION"
    ]
)


confusion_matrix = (
    confusion_matrix
    .reindex(
        index=CLASS_ORDER,
        columns=CLASS_ORDER,
        fill_value=0
    )
)


print("\n")
print("=" * 80)
print("MATRIZ DE CONFUSION")
print("=" * 80)


print(
    confusion_matrix
)


confusion_matrix.to_csv(
    OUTPUT_DIR
    / "matriz_confusion_softmax.csv"
)


# ============================================================
# GUARDAR PREDICCIONES INCORRECTAS
# ============================================================

errors_df = df[
    ~df["correct"]
].copy()


errors_df.to_csv(
    OUTPUT_DIR
    / "predicciones_incorrectas.csv",
    index=False
)


# ============================================================
# ORDENAR TODAS LAS MUESTRAS POR MARGEN
# ============================================================

uncertain = (
    df
    .sort_values(
        by="margin",
        ascending=True
    )
    .copy()
)


uncertain.to_csv(
    OUTPUT_DIR
    / "muestras_ordenadas_por_margen.csv",
    index=False
)


# ============================================================
# TOP 30 CON MENOR MARGEN
# ============================================================

cols_to_show = [
    "file",
    "window",
    TRUE_CLASS_COL,
    "prediccion_softmax",
    "P_DETENIDO",
    "P_MEDIA",
    "P_ALTA",
    "P_true",
    "P_max",
    "P_second",
    "margin",
    "correct",
]


cols_to_show = [
    col
    for col in cols_to_show
    if col in df.columns
]


top30 = (
    uncertain[
        cols_to_show
    ]
    .head(30)
)


print("\n")
print("=" * 80)
print("30 VENTANAS CON MENOR MARGEN")
print("=" * 80)


print(
    top30
    .round(4)
    .to_string(
        index=False
    )
)


top30.to_csv(
    OUTPUT_DIR
    / "top30_muestras_menor_margen.csv",
    index=False
)


# ============================================================
# HISTOGRAMA P_MEDIA PARA MEDIA REAL
# ============================================================

if len(media_df) > 0:

    plt.figure(
        figsize=(8, 5)
    )


    plt.hist(
        media_df[
            "P_MEDIA"
        ],
        bins=20
    )


    plt.xlabel(
        "P(VELOCIDAD MEDIA)"
    )


    plt.ylabel(
        "Numero de ventanas"
    )


    plt.title(
        "P(MEDIA) para muestras reales MEDIA\n"
        f"{VERSION}"
    )


    plt.tight_layout()


    plt.savefig(
        OUTPUT_DIR
        / "histograma_P_MEDIA_reales_MEDIA.png",
        dpi=300
    )


    plt.close()


# ============================================================
# HISTOGRAMA MARGEN PARA MEDIA
# ============================================================

if len(media_df) > 0:

    plt.figure(
        figsize=(8, 5)
    )


    plt.hist(
        media_df[
            "margin"
        ],
        bins=20
    )


    plt.axvline(
        x=0,
        linestyle="--"
    )


    plt.xlabel(
        "Margen = P(clase verdadera) - segunda probabilidad"
    )


    plt.ylabel(
        "Numero de ventanas"
    )


    plt.title(
        "Margen de decision - VELOCIDAD MEDIA\n"
        f"{VERSION}"
    )


    plt.tight_layout()


    plt.savefig(
        OUTPUT_DIR
        / "histograma_margen_MEDIA.png",
        dpi=300
    )


    plt.close()


# ============================================================
# SCATTER MEDIA VS ALTA
# ============================================================

if len(media_df) > 0:

    plt.figure(
        figsize=(7, 6)
    )


    plt.scatter(
        media_df[
            "P_MEDIA"
        ],
        media_df[
            "P_ALTA"
        ],
        alpha=0.6
    )


    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )


    plt.xlim(
        0,
        1
    )


    plt.ylim(
        0,
        1
    )


    plt.xlabel(
        "P(VELOCIDAD MEDIA)"
    )


    plt.ylabel(
        "P(VELOCIDAD ALTA)"
    )


    plt.title(
        "MEDIA real: P_MEDIA vs P_ALTA\n"
        f"{VERSION}"
    )


    plt.tight_layout()


    plt.savefig(
        OUTPUT_DIR
        / "scatter_MEDIA_vs_ALTA.png",
        dpi=300
    )


    plt.close()


# ============================================================
# GUARDAR CSV COMPLETO CON ANALISIS
# ============================================================

df.to_csv(
    OUTPUT_DIR
    / "probabilidades_con_analisis.csv",
    index=False
)


# ============================================================
# GUARDAR RESUMEN TXT
# ============================================================

TXT_PATH = (
    OUTPUT_DIR
    / "resumen_analisis.txt"
)


with open(
    TXT_PATH,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "ANALISIS DE PROBABILIDADES SOFTMAX\n"
    )

    f.write(
        "=" * 70
        + "\n\n"
    )


    f.write(
        f"Version: {VERSION}\n"
    )


    f.write(
        f"CSV analizado: "
        f"{CSV_PATH.resolve()}\n\n"
    )


    f.write(
        f"Numero de ventanas: "
        f"{len(df)}\n"
    )


    f.write(
        f"Accuracy: "
        f"{accuracy * 100:.2f}%\n"
    )


    f.write(
        f"P verdadera promedio: "
        f"{mean_p_true:.4f}\n"
    )


    f.write(
        f"Confianza maxima promedio: "
        f"{mean_confidence:.4f}\n"
    )


    f.write(
        f"Margen promedio: "
        f"{mean_margin:.4f}\n"
    )


    f.write(
        f"Correctas con confianza >= 0.90: "
        f"{correct_conf_90 * 100:.2f}%\n"
    )


    f.write(
        f"Correctas con confianza >= 0.95: "
        f"{correct_conf_95 * 100:.2f}%\n"
    )


    f.write(
        f"Predicciones incorrectas: "
        f"{num_errors}\n"
    )


    if len(media_df) > 0:

        f.write(
            "\nVELOCIDAD MEDIA\n"
        )

        f.write(
            "-" * 70
            + "\n"
        )


        f.write(
            f"Muestras: "
            f"{len(media_df)}\n"
        )


        f.write(
            f"Correctas: "
            f"{media_df['correct'].sum()}\n"
        )


        f.write(
            f"Incorrectas: "
            f"{(~media_df['correct']).sum()}\n"
        )


        f.write(
            f"P_MEDIA promedio: "
            f"{media_df['P_MEDIA'].mean():.4f}\n"
        )


        f.write(
            f"P_ALTA promedio: "
            f"{media_df['P_ALTA'].mean():.4f}\n"
        )


        f.write(
            f"P_DETENIDO promedio: "
            f"{media_df['P_DETENIDO'].mean():.4f}\n"
        )


        f.write(
            f"Margen promedio: "
            f"{media_df['margin'].mean():.4f}\n"
        )


# ============================================================
# FIN
# ============================================================

print("\n")
print("=" * 80)
print("ANALISIS FINALIZADO")
print("=" * 80)


print(
    f"Version: {VERSION}"
)


print(
    f"\nResultados guardados en:\n"
    f"{OUTPUT_DIR.resolve()}"
)
