from pathlib import Path
import sys

CODIGO_DIR = Path(__file__).resolve().parents[2]
if str(CODIGO_DIR) not in sys.path:
    sys.path.insert(0, str(CODIGO_DIR))

from comun.rutas_proyecto import RESULTADOS_E1

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support
)


# ============================================================
# CONFIGURACIÓN
# ============================================================

# CAMBIA SOLO ESTA RUTA PARA ANALIZAR OTRO EXPERIMENTO

EXPERIMENTO_DIR = RESULTADOS_E1 / "validacion_stm32"


# ============================================================
# ARCHIVO FINAL DEL EXPERIMENTO
# ============================================================

CSV_PATH = (
    EXPERIMENTO_DIR
    / "resultados_1332_ventanas.csv"
)


# ============================================================
# CLASES
# ============================================================

CLASES = {
    0: "DET",
    1: "MEDIA",
    2: "ALTA"
}

LABELS = [0, 1, 2]

NOMBRES_CLASES = [
    CLASES[i]
    for i in LABELS
]


# ============================================================
# VERIFICACIONES
# ============================================================

if not EXPERIMENTO_DIR.exists():
    raise FileNotFoundError(
        f"No existe la carpeta:\n{EXPERIMENTO_DIR}"
    )

if not CSV_PATH.exists():
    raise FileNotFoundError(
        "No se encontró el archivo final de 1332 ventanas:\n"
        f"{CSV_PATH}"
    )


# ============================================================
# CARPETA DE ANÁLISIS
# ============================================================

ANALISIS_DIR = (
    EXPERIMENTO_DIR
    / "Analisis_final_1332"
)

ANALISIS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


print("=" * 80)
print("ANÁLISIS FINAL STM32 - 1332 VENTANAS")
print("=" * 80)

print("\nExperimento:")
print(EXPERIMENTO_DIR)

print("\nArchivo:")
print(CSV_PATH)

print("\nResultados se guardarán en:")
print(ANALISIS_DIR)


# ============================================================
# CARGAR CSV
# ============================================================

df = pd.read_csv(
    CSV_PATH,
    sep=None,
    engine="python"
)


print("\nNúmero de ventanas cargadas:")
print(len(df))


if len(df) != 1332:
    print(
        "\nADVERTENCIA:"
        f" se esperaban 1332 ventanas y se encontraron {len(df)}."
    )


# ============================================================
# VERIFICAR COLUMNAS
# ============================================================

columnas_necesarias = [
    "clase_real_id",
    "clase_pc_id",
    "clase_stm32_id"
]

for columna in columnas_necesarias:

    if columna not in df.columns:
        raise ValueError(
            f"Falta la columna requerida: {columna}"
        )


# ============================================================
# VARIABLES
# ============================================================

y_real = (
    df["clase_real_id"]
    .astype(int)
    .to_numpy()
)

y_pc = (
    df["clase_pc_id"]
    .astype(int)
    .to_numpy()
)

y_stm32 = (
    df["clase_stm32_id"]
    .astype(int)
    .to_numpy()
)


# ============================================================
# MÉTRICAS GENERALES
# ============================================================

accuracy_pc = accuracy_score(
    y_real,
    y_pc
)

accuracy_stm32 = accuracy_score(
    y_real,
    y_stm32
)

coincidencia_pc_stm32 = np.mean(
    y_pc == y_stm32
)

correctas_pc = int(
    np.sum(
        y_pc == y_real
    )
)

incorrectas_pc = int(
    np.sum(
        y_pc != y_real
    )
)

correctas_stm32 = int(
    np.sum(
        y_stm32 == y_real
    )
)

incorrectas_stm32 = int(
    np.sum(
        y_stm32 != y_real
    )
)


# ============================================================
# MÉTRICAS GLOBALES STM32
# ============================================================

precision_macro, recall_macro, f1_macro, _ = (
    precision_recall_fscore_support(
        y_real,
        y_stm32,
        labels=LABELS,
        average="macro",
        zero_division=0
    )
)

precision_weighted, recall_weighted, f1_weighted, _ = (
    precision_recall_fscore_support(
        y_real,
        y_stm32,
        labels=LABELS,
        average="weighted",
        zero_division=0
    )
)


# ============================================================
# MÉTRICAS POR CLASE
# ============================================================

precision_clase, recall_clase, f1_clase, soporte_clase = (
    precision_recall_fscore_support(
        y_real,
        y_stm32,
        labels=LABELS,
        zero_division=0
    )
)


metricas_por_clase = pd.DataFrame(
    {
        "clase_id": LABELS,
        "clase_nombre": NOMBRES_CLASES,
        "precision": precision_clase,
        "recall": recall_clase,
        "f1_score": f1_clase,
        "soporte": soporte_clase
    }
)


metricas_por_clase.to_csv(
    ANALISIS_DIR
    / "metricas_por_clase.csv",
    index=False,
    sep=";",
    encoding="utf-8-sig"
)


# ============================================================
# MATRIZ DE CONFUSIÓN STM32
# ============================================================

mc_stm32 = confusion_matrix(
    y_real,
    y_stm32,
    labels=LABELS
)


df_mc_stm32 = pd.DataFrame(
    mc_stm32,
    index=[
        f"Real_{nombre}"
        for nombre in NOMBRES_CLASES
    ],
    columns=[
        f"Pred_{nombre}"
        for nombre in NOMBRES_CLASES
    ]
)


df_mc_stm32.to_csv(
    ANALISIS_DIR
    / "matriz_confusion_stm32.csv",
    sep=";",
    encoding="utf-8-sig"
)


# ============================================================
# IMAGEN MATRIZ DE CONFUSIÓN STM32
# ============================================================

fig, ax = plt.subplots(
    figsize=(7, 6)
)

im = ax.imshow(
    mc_stm32
)

ax.set_xticks(
    np.arange(len(NOMBRES_CLASES))
)

ax.set_yticks(
    np.arange(len(NOMBRES_CLASES))
)

ax.set_xticklabels(
    NOMBRES_CLASES
)

ax.set_yticklabels(
    NOMBRES_CLASES
)

ax.set_xlabel(
    "Clase predicha"
)

ax.set_ylabel(
    "Clase real"
)

ax.set_title(
    "Matriz de confusión - STM32"
)


for i in range(
    mc_stm32.shape[0]
):

    for j in range(
        mc_stm32.shape[1]
    ):

        ax.text(
            j,
            i,
            str(mc_stm32[i, j]),
            ha="center",
            va="center"
        )


fig.colorbar(
    im,
    ax=ax
)

fig.tight_layout()

fig.savefig(
    ANALISIS_DIR
    / "matriz_confusion_stm32.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# MATRIZ DE CONFUSIÓN PC
# ============================================================

mc_pc = confusion_matrix(
    y_real,
    y_pc,
    labels=LABELS
)


df_mc_pc = pd.DataFrame(
    mc_pc,
    index=[
        f"Real_{nombre}"
        for nombre in NOMBRES_CLASES
    ],
    columns=[
        f"Pred_{nombre}"
        for nombre in NOMBRES_CLASES
    ]
)


df_mc_pc.to_csv(
    ANALISIS_DIR
    / "matriz_confusion_pc.csv",
    sep=";",
    encoding="utf-8-sig"
)


# ============================================================
# ERRORES DE CLASIFICACIÓN
# ============================================================

errores = df[
    df["clase_real_id"]
    != df["clase_stm32_id"]
].copy()


errores.to_csv(
    ANALISIS_DIR
    / "errores_clasificacion.csv",
    index=False,
    sep=";",
    encoding="utf-8-sig"
)


# ============================================================
# TIPOS DE ERROR
# ============================================================

tipos_error = []

for real in LABELS:

    for pred in LABELS:

        if real == pred:
            continue

        cantidad = int(
            np.sum(
                (y_real == real)
                &
                (y_stm32 == pred)
            )
        )

        tipos_error.append(
            {
                "clase_real_id": real,
                "clase_real": CLASES[real],
                "clase_predicha_id": pred,
                "clase_predicha": CLASES[pred],
                "cantidad": cantidad
            }
        )


df_tipos_error = pd.DataFrame(
    tipos_error
)


df_tipos_error.to_csv(
    ANALISIS_DIR
    / "tipos_error.csv",
    index=False,
    sep=";",
    encoding="utf-8-sig"
)


# ============================================================
# GRÁFICA DE TIPOS DE ERROR
# ============================================================

errores_grafica = (
    df_tipos_error[
        df_tipos_error["cantidad"] > 0
    ]
    .copy()
)


if len(errores_grafica) > 0:

    etiquetas = (
        errores_grafica["clase_real"]
        + " → "
        + errores_grafica["clase_predicha"]
    )

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.bar(
        etiquetas,
        errores_grafica["cantidad"]
    )

    ax.set_title(
        "Errores de clasificación por tipo"
    )

    ax.set_xlabel(
        "Clase real → Clase predicha"
    )

    ax.set_ylabel(
        "Número de ventanas"
    )

    ax.tick_params(
        axis="x",
        rotation=45
    )

    fig.tight_layout()

    fig.savefig(
        ANALISIS_DIR
        / "errores_por_tipo.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# GRÁFICA PRECISION / RECALL / F1
# ============================================================

x_pos = np.arange(
    len(NOMBRES_CLASES)
)

ancho = 0.25


fig, ax = plt.subplots(
    figsize=(9, 6)
)


ax.bar(
    x_pos - ancho,
    precision_clase,
    width=ancho,
    label="Precision"
)

ax.bar(
    x_pos,
    recall_clase,
    width=ancho,
    label="Recall"
)

ax.bar(
    x_pos + ancho,
    f1_clase,
    width=ancho,
    label="F1-score"
)


ax.set_xticks(
    x_pos
)

ax.set_xticklabels(
    NOMBRES_CLASES
)

ax.set_ylim(
    0,
    1.05
)

ax.set_ylabel(
    "Valor"
)

ax.set_title(
    "Métricas por clase - STM32"
)

ax.legend()

fig.tight_layout()

fig.savefig(
    ANALISIS_DIR
    / "metricas_por_clase.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# DIFERENCIA NUMÉRICA PC vs STM32
# ============================================================

columnas_error = [
    "error_abs_det",
    "error_abs_media",
    "error_abs_alta"
]


error_medio_prob = None
error_max_prob = None


if all(
    columna in df.columns
    for columna in columnas_error
):

    matriz_errores_prob = (
        df[columnas_error]
        .to_numpy()
    )

    error_medio_prob = float(
        np.mean(
            matriz_errores_prob
        )
    )

    error_max_prob = float(
        np.max(
            matriz_errores_prob
        )
    )


# ============================================================
# MÉTRICAS GENERALES CSV
# ============================================================

metricas_generales = pd.DataFrame(
    [
        {
            "ventanas_totales":
                len(df),

            "clasificaciones_correctas_pc":
                correctas_pc,

            "clasificaciones_incorrectas_pc":
                incorrectas_pc,

            "accuracy_pc":
                accuracy_pc,

            "clasificaciones_correctas_stm32":
                correctas_stm32,

            "clasificaciones_incorrectas_stm32":
                incorrectas_stm32,

            "accuracy_stm32":
                accuracy_stm32,

            "coincidencia_pc_stm32":
                coincidencia_pc_stm32,

            "precision_macro_stm32":
                precision_macro,

            "recall_macro_stm32":
                recall_macro,

            "f1_macro_stm32":
                f1_macro,

            "precision_weighted_stm32":
                precision_weighted,

            "recall_weighted_stm32":
                recall_weighted,

            "f1_weighted_stm32":
                f1_weighted,

            "error_medio_prob_pc_stm32":
                error_medio_prob,

            "error_max_prob_pc_stm32":
                error_max_prob
        }
    ]
)


metricas_generales.to_csv(
    ANALISIS_DIR
    / "metricas_generales.csv",
    index=False,
    sep=";",
    encoding="utf-8-sig"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

reporte = classification_report(
    y_real,
    y_stm32,
    labels=LABELS,
    target_names=NOMBRES_CLASES,
    digits=6,
    zero_division=0
)


# ============================================================
# RESUMEN TXT
# ============================================================

RESUMEN_PATH = (
    ANALISIS_DIR
    / "resumen_resultados.txt"
)


with open(
    RESUMEN_PATH,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "=" * 75
        + "\n"
    )

    f.write(
        "ANÁLISIS FINAL STM32 - 1332 VENTANAS\n"
    )

    f.write(
        "=" * 75
        + "\n\n"
    )

    f.write(
        f"Archivo analizado: {CSV_PATH.name}\n"
    )

    f.write(
        f"Número total de ventanas: {len(df)}\n\n"
    )


    f.write(
        "RESULTADOS GENERALES\n"
    )

    f.write(
        "-" * 75
        + "\n"
    )

    f.write(
        f"Accuracy PC: "
        f"{accuracy_pc * 100:.4f}%\n"
    )

    f.write(
        f"Accuracy STM32: "
        f"{accuracy_stm32 * 100:.4f}%\n"
    )

    f.write(
        f"Coincidencia PC vs STM32: "
        f"{coincidencia_pc_stm32 * 100:.4f}%\n"
    )

    f.write(
        f"Clasificaciones correctas STM32: "
        f"{correctas_stm32}/{len(df)}\n"
    )

    f.write(
        f"Clasificaciones incorrectas STM32: "
        f"{incorrectas_stm32}/{len(df)}\n\n"
    )


    f.write(
        "MÉTRICAS GLOBALES STM32\n"
    )

    f.write(
        "-" * 75
        + "\n"
    )

    f.write(
        f"Precision macro: {precision_macro:.6f}\n"
    )

    f.write(
        f"Recall macro: {recall_macro:.6f}\n"
    )

    f.write(
        f"F1-score macro: {f1_macro:.6f}\n"
    )

    f.write(
        f"Precision weighted: {precision_weighted:.6f}\n"
    )

    f.write(
        f"Recall weighted: {recall_weighted:.6f}\n"
    )

    f.write(
        f"F1-score weighted: {f1_weighted:.6f}\n\n"
    )


    if error_medio_prob is not None:

        f.write(
            "COMPARACIÓN NUMÉRICA PC vs STM32\n"
        )

        f.write(
            "-" * 75
            + "\n"
        )

        f.write(
            f"Error absoluto medio de probabilidades: "
            f"{error_medio_prob:.10f}\n"
        )

        f.write(
            f"Error absoluto máximo de probabilidades: "
            f"{error_max_prob:.10f}\n\n"
        )


    f.write(
        "MÉTRICAS POR CLASE\n"
    )

    f.write(
        "-" * 75
        + "\n\n"
    )

    f.write(
        reporte
    )


    f.write(
        "\n\nMATRIZ DE CONFUSIÓN STM32\n"
    )

    f.write(
        "-" * 75
        + "\n"
    )

    f.write(
        str(df_mc_stm32)
    )


    f.write(
        "\n\nTIPOS DE ERROR\n"
    )

    f.write(
        "-" * 75
        + "\n"
    )


    for _, fila in df_tipos_error.iterrows():

        if fila["cantidad"] > 0:

            f.write(
                f"{fila['clase_real']} -> "
                f"{fila['clase_predicha']}: "
                f"{fila['cantidad']}\n"
            )


# ============================================================
# TERMINAL
# ============================================================

print("\n")
print("=" * 80)
print("RESULTADOS FINALES")
print("=" * 80)

print(
    f"Ventanas analizadas : {len(df)}"
)

print(
    f"Accuracy PC         : {accuracy_pc * 100:.2f}%"
)

print(
    f"Accuracy STM32      : {accuracy_stm32 * 100:.2f}%"
)

print(
    f"Coincidencia PC-STM32: {coincidencia_pc_stm32 * 100:.2f}%"
)

print(
    f"Correctas STM32     : {correctas_stm32}"
)

print(
    f"Incorrectas STM32   : {incorrectas_stm32}"
)

print(
    f"Precision macro     : {precision_macro:.4f}"
)

print(
    f"Recall macro        : {recall_macro:.4f}"
)

print(
    f"F1 macro            : {f1_macro:.4f}"
)

print("\nMatriz de confusión STM32:")
print(df_mc_stm32)

print("\nArchivos guardados en:")
print(ANALISIS_DIR)
