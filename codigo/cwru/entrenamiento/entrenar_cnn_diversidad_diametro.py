"""
==============================================================
EFECTO DE LA DIVERSIDAD DE DIAMETROS EN LA GENERALIZACION
CWRU - CNN TinyML
==============================================================

TEST FIJO:
    100_N_0,000.mat
    217_IR_0,021.mat
    241_OR_0,021.mat

MODELO A:
    Train faults = 0.007"

MODELO B:
    Train faults = 0.014"

MODELO C:
    Train faults = 0.007" + 0.014"

Objetivo:
    Comprobar si aumentar la variedad de diametros durante
    entrenamiento mejora la generalizacion hacia 0.021".

Adicionalmente:
    Se guardan las probabilidades Softmax:
        P(N)
        P(IR)
        P(OR)

==============================================================
"""

from pathlib import Path
import random
import sys

CODIGO_DIR = Path(__file__).resolve().parents[2]
if str(CODIGO_DIR) not in sys.path:
    sys.path.insert(0, str(CODIGO_DIR))

from comun.rutas_proyecto import DATOS_CWRU_PROCESADO, RESULTADOS_CWRU

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix
)

from sklearn.utils.class_weight import compute_class_weight


# ============================================================
# CONFIGURACION
# ============================================================

DATA_DIR = DATOS_CWRU_PROCESADO

X_PATH = DATA_DIR / "X.npy"
Y_PATH = DATA_DIR / "y.npy"
METADATA_PATH = DATA_DIR / "metadata.csv"

OUTPUT_DIR = RESULTADOS_CWRU / "diversidad_diametro"
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

SEED = 42

EPOCHS = 60
BATCH_SIZE = 16
LEARNING_RATE = 1e-3

USE_CLASS_WEIGHTS = True

CLASS_NAMES = [
    "Normal",
    "Inner Race",
    "Outer Race"
]

LABEL_TO_CLASS = {
    0: "Normal",
    1: "Inner Race",
    2: "Outer Race"
}


# ============================================================
# TEST FIJO
# ============================================================

TEST_FILES = [
    "100_N_0,000.mat",
    "217_IR_0,021.mat",
    "241_OR_0,021.mat"
]


# ============================================================
# SEMILLA
# ============================================================

def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


set_seed(SEED)


# ============================================================
# CARGAR DATA
# ============================================================

X = np.load(X_PATH)
y = np.load(Y_PATH)

metadata = pd.read_csv(
    METADATA_PATH
)


# ============================================================
# MODELO
# ============================================================

def build_model():

    model = tf.keras.Sequential([

        tf.keras.layers.Input(
            shape=(1024, 1)
        ),

        tf.keras.layers.Conv1D(
            16,
            7,
            padding="same",
            activation="relu"
        ),

        tf.keras.layers.MaxPooling1D(
            2
        ),

        tf.keras.layers.Conv1D(
            32,
            5,
            padding="same",
            activation="relu"
        ),

        tf.keras.layers.MaxPooling1D(
            2
        ),

        tf.keras.layers.Conv1D(
            32,
            3,
            padding="same",
            activation="relu"
        ),

        tf.keras.layers.GlobalAveragePooling1D(),

        tf.keras.layers.Dense(
            3,
            activation="softmax"
        )
    ])

    model.compile(

        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),

        loss="sparse_categorical_crossentropy",

        metrics=["accuracy"]
    )

    return model


# ============================================================
# TEST FIJO
# ============================================================

def get_test_mask():

    return metadata["file"].isin(
        TEST_FILES
    )


# ============================================================
# NORMAL PARA TRAIN
#
# Como el test usa Normal 3 HP,
# dejamos Normal 0,1 HP para entrenamiento
# y Normal 2 HP para validacion.
# ============================================================

def normal_train_mask():

    return (
        (metadata["class"] == "N")
        &
        metadata["load_hp"].isin([0, 1])
    )


def normal_val_mask():

    return (
        (metadata["class"] == "N")
        &
        (metadata["load_hp"] == 2)
    )


# ============================================================
# MODELO A
#
# Train:
#     Fault = 0.007
#     HP = 0,1
#
# Val:
#     Fault = 0.007
#     HP = 2
#
# Test:
#     0.021 / HP=3 fijo
# ============================================================

def model_A_split():

    fault = metadata["class"].isin(
        ["IR", "OR"]
    )

    train_fault = (
        fault
        &
        np.isclose(
            metadata["diameter_in"],
            0.007
        )
        &
        metadata["load_hp"].isin(
            [0, 1]
        )
    )

    val_fault = (
        fault
        &
        np.isclose(
            metadata["diameter_in"],
            0.007
        )
        &
        (metadata["load_hp"] == 2)
    )

    train_mask = (
        train_fault
        |
        normal_train_mask()
    )

    val_mask = (
        val_fault
        |
        normal_val_mask()
    )

    test_mask = get_test_mask()

    return (
        train_mask,
        val_mask,
        test_mask,
        "Modelo_A_007"
    )


# ============================================================
# MODELO B
#
# Train:
#     Fault = 0.014
#     HP = 0,1
#
# Val:
#     Fault = 0.014
#     HP = 2
#
# Test:
#     0.021 / HP=3 fijo
# ============================================================

def model_B_split():

    fault = metadata["class"].isin(
        ["IR", "OR"]
    )

    train_fault = (
        fault
        &
        np.isclose(
            metadata["diameter_in"],
            0.014
        )
        &
        metadata["load_hp"].isin(
            [0, 1]
        )
    )

    val_fault = (
        fault
        &
        np.isclose(
            metadata["diameter_in"],
            0.014
        )
        &
        (metadata["load_hp"] == 2)
    )

    train_mask = (
        train_fault
        |
        normal_train_mask()
    )

    val_mask = (
        val_fault
        |
        normal_val_mask()
    )

    test_mask = get_test_mask()

    return (
        train_mask,
        val_mask,
        test_mask,
        "Modelo_B_014"
    )


# ============================================================
# MODELO C
#
# Train:
#     Fault = 0.007 + 0.014
#     HP = 0,1
#
# Val:
#     Fault = 0.007 + 0.014
#     HP = 2
#
# Test:
#     0.021 / HP=3 fijo
# ============================================================

def model_C_split():

    fault = metadata["class"].isin(
        ["IR", "OR"]
    )

    diameter_condition = (
        np.isclose(
            metadata["diameter_in"],
            0.007
        )
        |
        np.isclose(
            metadata["diameter_in"],
            0.014
        )
    )

    train_fault = (
        fault
        &
        diameter_condition
        &
        metadata["load_hp"].isin(
            [0, 1]
        )
    )

    val_fault = (
        fault
        &
        diameter_condition
        &
        (metadata["load_hp"] == 2)
    )

    train_mask = (
        train_fault
        |
        normal_train_mask()
    )

    val_mask = (
        val_fault
        |
        normal_val_mask()
    )

    test_mask = get_test_mask()

    return (
        train_mask,
        val_mask,
        test_mask,
        "Modelo_C_007_014"
    )


# ============================================================
# CLASS WEIGHTS
# ============================================================

def get_class_weights(
    y_train
):

    classes = np.unique(
        y_train
    )

    weights = compute_class_weight(

        class_weight="balanced",

        classes=classes,

        y=y_train
    )

    return {
        int(c): float(w)

        for c, w
        in zip(classes, weights)
    }


# ============================================================
# MOSTRAR DISTRIBUCION
# ============================================================

def show_distribution(
    name,
    mask
):

    subset = metadata.loc[
        mask
    ]

    print(
        f"\n{name}"
    )

    print("-" * 70)

    print(
        "Ventanas:",
        len(subset)
    )

    print("\nClases:")

    print(
        subset["class"]
        .value_counts()
        .sort_index()
    )

    print("\nDiametros:")

    print(
        subset.groupby(
            [
                "class",
                "diameter_in"
            ]
        )
        .size()
    )

    print("\nHP:")

    print(
        subset.groupby(
            [
                "load_hp",
                "rpm"
            ]
        )
        .size()
    )

    print("\nArchivos:")

    print(
        subset["file"]
        .drop_duplicates()
        .tolist()
    )


# ============================================================
# MATRIZ CONFUSION
# ============================================================

def plot_confusion_matrix(
    cm,
    name
):

    plt.figure(
        figsize=(6, 5)
    )

    plt.imshow(cm)

    plt.xticks(
        range(3),
        CLASS_NAMES
    )

    plt.yticks(
        range(3),
        CLASS_NAMES
    )

    plt.xlabel(
        "Prediccion"
    )

    plt.ylabel(
        "Clase real"
    )

    plt.title(
        name
    )

    for i in range(3):

        for j in range(3):

            plt.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center"
            )

    plt.colorbar()

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR /
        f"{name}_confusion_matrix.png",
        dpi=300
    )

    plt.show()


# ============================================================
# CURVAS
# ============================================================

def plot_history(
    history,
    name
):

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        history.history["accuracy"],
        label="Train"
    )

    plt.plot(
        history.history["val_accuracy"],
        label="Validation"
    )

    plt.xlabel(
        "Epoca"
    )

    plt.ylabel(
        "Accuracy"
    )

    plt.title(
        name
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR /
        f"{name}_history.png",
        dpi=300
    )

    plt.show()


# ============================================================
# RESUMEN DE PROBABILIDADES POR ARCHIVO
# ============================================================

def probability_summary(
    prediction_df
):

    summary = (
        prediction_df
        .groupby(
            [
                "file",
                "class"
            ]
        )
        .agg(

            windows=(
                "window",
                "count"
            ),

            accuracy=(
                "correct",
                "mean"
            ),

            mean_P_N=(
                "P_N",
                "mean"
            ),

            mean_P_IR=(
                "P_IR",
                "mean"
            ),

            mean_P_OR=(
                "P_OR",
                "mean"
            ),

            mean_confidence=(
                "confidence",
                "mean"
            )

        )
        .reset_index()
    )

    return summary


# ============================================================
# EJECUTAR MODELO
# ============================================================

def run_experiment(
    split_function
):

    tf.keras.backend.clear_session()

    set_seed(SEED)

    (
        train_mask,
        val_mask,
        test_mask,
        name
    ) = split_function()


    print("\n")
    print("=" * 80)
    print(name)
    print("=" * 80)


    # ========================================================
    # CHECK LEAKAGE
    # ========================================================

    train_files = set(
        metadata.loc[
            train_mask,
            "file"
        ]
    )

    val_files = set(
        metadata.loc[
            val_mask,
            "file"
        ]
    )

    test_files = set(
        metadata.loc[
            test_mask,
            "file"
        ]
    )


    assert train_files.isdisjoint(
        val_files
    )

    assert train_files.isdisjoint(
        test_files
    )

    assert val_files.isdisjoint(
        test_files
    )


    # ========================================================
    # DISTRIBUCION
    # ========================================================

    show_distribution(
        "TRAIN",
        train_mask
    )

    show_distribution(
        "VALIDATION",
        val_mask
    )

    show_distribution(
        "TEST",
        test_mask
    )


    # ========================================================
    # DATOS
    # ========================================================

    X_train = X[
        train_mask
    ]

    y_train = y[
        train_mask
    ]

    X_val = X[
        val_mask
    ]

    y_val = y[
        val_mask
    ]

    X_test = X[
        test_mask
    ]

    y_test = y[
        test_mask
    ]


    X_train = X_train[
        ...,
        np.newaxis
    ]

    X_val = X_val[
        ...,
        np.newaxis
    ]

    X_test = X_test[
        ...,
        np.newaxis
    ]


    # ========================================================
    # PESOS
    # ========================================================

    if USE_CLASS_WEIGHTS:

        class_weights = (
            get_class_weights(
                y_train
            )
        )

    else:

        class_weights = None


    # ========================================================
    # MODELO
    # ========================================================

    model = build_model()


    early_stopping = (
        tf.keras.callbacks.EarlyStopping(

            monitor="val_loss",

            patience=10,

            restore_best_weights=True,

            verbose=1
        )
    )


    # ========================================================
    # TRAIN
    # ========================================================

    history = model.fit(

        X_train,
        y_train,

        validation_data=(
            X_val,
            y_val
        ),

        epochs=EPOCHS,

        batch_size=BATCH_SIZE,

        class_weight=class_weights,

        shuffle=True,

        callbacks=[
            early_stopping
        ],

        verbose=1
    )


    # ========================================================
    # SOFTMAX
    # ========================================================

    probabilities = model.predict(
        X_test,
        verbose=0
    )


    # probabilities[:,0] = P(N)
    # probabilities[:,1] = P(IR)
    # probabilities[:,2] = P(OR)


    y_pred = np.argmax(
        probabilities,
        axis=1
    )


    confidence = np.max(
        probabilities,
        axis=1
    )


    # ========================================================
    # METRICAS
    # ========================================================

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    macro_precision = precision_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    macro_recall = recall_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    macro_f1 = f1_score(
        y_test,
        y_pred,
        average="macro"
    )


    print("\n")
    print("=" * 80)
    print("TEST")
    print("=" * 80)

    print(
        f"Accuracy        : {accuracy:.4f}"
    )

    print(
        f"Macro Precision : {macro_precision:.4f}"
    )

    print(
        f"Macro Recall    : {macro_recall:.4f}"
    )

    print(
        f"Macro F1        : {macro_f1:.4f}"
    )


    print("\n")

    print(
        classification_report(

            y_test,
            y_pred,

            labels=[
                0,
                1,
                2
            ],

            target_names=
                CLASS_NAMES,

            digits=4,

            zero_division=0
        )
    )


    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    cm = confusion_matrix(

        y_test,
        y_pred,

        labels=[
            0,
            1,
            2
        ]
    )

    print(
        "\nConfusion Matrix:"
    )

    print(cm)


    # ========================================================
    # DATAFRAME DE PROBABILIDADES
    # ========================================================

    test_metadata = (
        metadata.loc[
            test_mask
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )


    test_metadata[
        "true_label"
    ] = y_test

    test_metadata[
        "true_class"
    ] = [
        LABEL_TO_CLASS[int(v)]
        for v in y_test
    ]


    test_metadata[
        "P_N"
    ] = probabilities[:, 0]

    test_metadata[
        "P_IR"
    ] = probabilities[:, 1]

    test_metadata[
        "P_OR"
    ] = probabilities[:, 2]


    test_metadata[
        "predicted_label"
    ] = y_pred

    test_metadata[
        "predicted_class"
    ] = [
        LABEL_TO_CLASS[int(v)]
        for v in y_pred
    ]


    test_metadata[
        "confidence"
    ] = confidence


    test_metadata[
        "correct"
    ] = (
        y_test == y_pred
    )


    # ========================================================
    # GUARDAR PROBABILIDADES POR VENTANA
    # ========================================================

    probability_file = (
        OUTPUT_DIR /
        f"{name}_probabilidades.csv"
    )


    columns_to_save = [

        "file",
        "window",
        "diameter_in",
        "load_hp",
        "rpm",

        "true_label",
        "true_class",

        "P_N",
        "P_IR",
        "P_OR",

        "predicted_label",
        "predicted_class",

        "confidence",
        "correct"
    ]


    test_metadata[
        columns_to_save
    ].to_csv(

        probability_file,

        index=False
    )


    # ========================================================
    # MOSTRAR PROBABILIDADES
    # ========================================================

    print("\n")
    print("=" * 80)
    print("PROBABILIDADES SOFTMAX POR VENTANA")
    print("=" * 80)


    print(

        test_metadata[
            [
                "file",
                "window",

                "true_class",

                "P_N",
                "P_IR",
                "P_OR",

                "predicted_class",
                "confidence",
                "correct"
            ]
        ]
        .round(4)
        .to_string(
            index=False
        )
    )


    # ========================================================
    # PROMEDIOS POR ARCHIVO
    # ========================================================

    prob_summary = (
        probability_summary(
            test_metadata
        )
    )


    print("\n")
    print("=" * 80)
    print("PROMEDIO DE PROBABILIDADES POR ARCHIVO")
    print("=" * 80)


    print(

        prob_summary
        .round(4)
        .to_string(
            index=False
        )
    )


    prob_summary.to_csv(

        OUTPUT_DIR /
        f"{name}_probabilidades_promedio.csv",

        index=False
    )


    # ========================================================
    # GRAFICAS
    # ========================================================

    plot_confusion_matrix(
        cm,
        name
    )

    plot_history(
        history,
        name
    )


    # ========================================================
    # GUARDAR MODELO
    # ========================================================

    model.save(
        OUTPUT_DIR /
        f"{name}.keras"
    )


    # ========================================================
    # F1 POR CLASE
    # ========================================================

    f1_per_class = f1_score(

        y_test,
        y_pred,

        labels=[
            0,
            1,
            2
        ],

        average=None,

        zero_division=0
    )


    return {

        "model":
            name,

        "train_samples":
            len(y_train),

        "val_samples":
            len(y_val),

        "test_samples":
            len(y_test),

        "accuracy":
            accuracy,

        "macro_precision":
            macro_precision,

        "macro_recall":
            macro_recall,

        "macro_f1":
            macro_f1,

        "f1_normal":
            f1_per_class[0],

        "f1_IR":
            f1_per_class[1],

        "f1_OR":
            f1_per_class[2],

        "mean_confidence":
            confidence.mean()
    }


# ============================================================
# MAIN
# ============================================================

def main():

    experiments = [

        model_A_split,
        model_B_split,
        model_C_split

    ]


    results = []


    for experiment in experiments:

        result = run_experiment(
            experiment
        )

        results.append(
            result
        )


    # ========================================================
    # TABLA FINAL
    # ========================================================

    results_df = pd.DataFrame(
        results
    )


    print("\n")
    print("=" * 100)
    print("COMPARACION FINAL")
    print("=" * 100)


    print(

        results_df
        .round(4)
        .to_string(
            index=False
        )
    )


    results_df.to_csv(

        OUTPUT_DIR /
        "comparacion_diversidad_diametro.csv",

        index=False
    )


    print(
        "\nResultados guardados en:"
    )

    print(
        OUTPUT_DIR.resolve()
    )


# ============================================================
# EJECUCION
# ============================================================

if __name__ == "__main__":

    main()
