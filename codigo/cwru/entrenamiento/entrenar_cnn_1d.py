"""
==============================================================
EXPERIMENTOS DE GENERALIZACIÓN - CWRU TinyML
==============================================================

Dataset:
    X.npy
    y.npy
    metadata.csv

Modelo:
    CNN 1D pequeña (5923 parámetros aprox.)

Experimentos:

EXP 1:
    Generalización de carga / RPM

EXP 2:
    Diámetro pequeño -> diámetro grande
    0.007 -> 0.021

EXP 3:
    Diámetro grande -> diámetro pequeño
    0.021 -> 0.007

EXP 4:
    Generalización combinada
    diámetro + carga/RPM no vistos

==============================================================
"""

from pathlib import Path
import random
import sys

CODIGO_DIR = Path(__file__).resolve().parents[2]
if str(CODIGO_DIR) not in sys.path:
    sys.path.insert(0, str(CODIGO_DIR))

from comun.rutas_proyecto import DATOS_CWRU_PROCESADO, MODELOS_CWRU, RESULTADOS_CWRU

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
# CONFIGURACIÓN GENERAL
# ============================================================

DATA_DIR = DATOS_CWRU_PROCESADO
MODEL_OUTPUT_DIR = MODELOS_CWRU
RESULTS_OUTPUT_DIR = RESULTADOS_CWRU
MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

X_PATH = DATA_DIR / "X.npy"
Y_PATH = DATA_DIR / "y.npy"
METADATA_PATH = DATA_DIR / "metadata.csv"

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


# ============================================================
# REPRODUCIBILIDAD
# ============================================================

def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


set_seed(SEED)


# ============================================================
# CARGAR DATASET
# ============================================================

X = np.load(X_PATH)
y = np.load(Y_PATH)

metadata = pd.read_csv(
    METADATA_PATH
)


print("=" * 80)
print("DATASET CWRU")
print("=" * 80)

print("X:", X.shape)
print("y:", y.shape)

print("\nClases:")
print(
    metadata["class"].value_counts()
)

print("\nDiámetros:")
print(
    metadata.groupby(
        ["class", "diameter_in"]
    ).size()
)

print("\nCarga:")
print(
    metadata.groupby(
        ["load_hp", "rpm"]
    ).size()
)


# ============================================================
# FUNCIÓN PARA CONSTRUIR MODELO
# ============================================================

def build_model():

    model = tf.keras.Sequential([

        tf.keras.layers.Input(
            shape=(1024, 1)
        ),

        tf.keras.layers.Conv1D(
            filters=16,
            kernel_size=7,
            padding="same",
            activation="relu"
        ),

        tf.keras.layers.MaxPooling1D(
            pool_size=2
        ),

        tf.keras.layers.Conv1D(
            filters=32,
            kernel_size=5,
            padding="same",
            activation="relu"
        ),

        tf.keras.layers.MaxPooling1D(
            pool_size=2
        ),

        tf.keras.layers.Conv1D(
            filters=32,
            kernel_size=3,
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
# NORMAL
#
# Normal no posee diámetro de falla.
# Por eso sus registros se distribuyen según HP.
# ============================================================

def normal_mask(hp_values):

    return (
        (metadata["class"] == "N")
        &
        (metadata["load_hp"].isin(hp_values))
    )


# ============================================================
# EXPERIMENTO 1
#
# GENERALIZACIÓN ENTRE CARGAS
#
# Train : HP 0,1
# Val   : HP 2
# Test  : HP 3
#
# Todos los diámetros IR/OR disponibles.
# ============================================================

def experiment_1():

    train_mask = (
        metadata["load_hp"].isin([0, 1])
    )

    val_mask = (
        metadata["load_hp"] == 2
    )

    test_mask = (
        metadata["load_hp"] == 3
    )

    return (
        train_mask,
        val_mask,
        test_mask,
        "EXP1_Load_Generalization"
    )


# ============================================================
# EXPERIMENTO 2
#
# DIÁMETRO PEQUEÑO -> GRANDE
#
# Fault:
# Train : 0.007
# Val   : 0.014
# Test  : 0.021
#
# Normal:
# Train : HP 0,1
# Val   : HP 2
# Test  : HP 3
# ============================================================

def experiment_2():

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
    )

    val_fault = (
        fault
        &
        np.isclose(
            metadata["diameter_in"],
            0.014
        )
    )

    test_fault = (
        fault
        &
        np.isclose(
            metadata["diameter_in"],
            0.021
        )
    )

    train_mask = (
        train_fault
        |
        normal_mask([0, 1])
    )

    val_mask = (
        val_fault
        |
        normal_mask([2])
    )

    test_mask = (
        test_fault
        |
        normal_mask([3])
    )

    return (
        train_mask,
        val_mask,
        test_mask,
        "EXP2_007_to_021"
    )


# ============================================================
# EXPERIMENTO 3
#
# DIÁMETRO GRANDE -> PEQUEÑO
#
# Fault:
# Train : 0.021
# Val   : 0.014
# Test  : 0.007
#
# Normal:
# Train : HP 1,2
# Val   : HP 3
# Test  : HP 0
# ============================================================

def experiment_3():

    fault = metadata["class"].isin(
        ["IR", "OR"]
    )

    train_fault = (
        fault
        &
        np.isclose(
            metadata["diameter_in"],
            0.021
        )
    )

    val_fault = (
        fault
        &
        np.isclose(
            metadata["diameter_in"],
            0.014
        )
    )

    test_fault = (
        fault
        &
        np.isclose(
            metadata["diameter_in"],
            0.007
        )
    )

    train_mask = (
        train_fault
        |
        normal_mask([1, 2])
    )

    val_mask = (
        val_fault
        |
        normal_mask([3])
    )

    test_mask = (
        test_fault
        |
        normal_mask([0])
    )

    return (
        train_mask,
        val_mask,
        test_mask,
        "EXP3_021_to_007"
    )


# ============================================================
# EXPERIMENTO 4
#
# GENERALIZACIÓN COMBINADA
#
# Train:
#     Fault diameter = 0.007 / 0.014
#     HP = 0 / 1
#
# Validation:
#     Fault diameter = 0.007 / 0.014
#     HP = 2
#
# Test:
#     Fault diameter = 0.021
#     HP = 3
#
# Normal sigue la misma distribución HP.
# ============================================================

def experiment_4():

    fault = metadata["class"].isin(
        ["IR", "OR"]
    )

    train_fault = (

        fault

        &

        metadata["diameter_in"].isin(
            [0.007, 0.014]
        )

        &

        metadata["load_hp"].isin(
            [0, 1]
        )
    )

    val_fault = (

        fault

        &

        metadata["diameter_in"].isin(
            [0.007, 0.014]
        )

        &

        (metadata["load_hp"] == 2)
    )

    test_fault = (

        fault

        &

        np.isclose(
            metadata["diameter_in"],
            0.021
        )

        &

        (metadata["load_hp"] == 3)
    )

    train_mask = (
        train_fault
        |
        normal_mask([0, 1])
    )

    val_mask = (
        val_fault
        |
        normal_mask([2])
    )

    test_mask = (
        test_fault
        |
        normal_mask([3])
    )

    return (
        train_mask,
        val_mask,
        test_mask,
        "EXP4_Diameter_Load_Generalization"
    )


# ============================================================
# MOSTRAR DISTRIBUCIÓN
# ============================================================

def show_distribution(
    name,
    mask
):

    subset = metadata.loc[mask]

    print(
        f"\n{name}"
    )

    print("-" * 60)

    print(
        f"Ventanas: {len(subset)}"
    )

    print("\nClases:")

    print(
        subset["class"]
        .value_counts()
        .sort_index()
    )

    print("\nDiámetros:")

    print(
        subset.groupby(
            ["class", "diameter_in"]
        )
        .size()
    )

    print("\nHP / RPM:")

    print(
        subset.groupby(
            ["load_hp", "rpm"]
        )
        .size()
    )

    print("\nArchivos:")

    print(
        subset["file"]
        .drop_duplicates()
        .to_list()
    )


# ============================================================
# CLASS WEIGHTS
# ============================================================

def get_class_weights(y_train):

    classes = np.unique(
        y_train
    )

    weights = compute_class_weight(

        class_weight="balanced",

        classes=classes,

        y=y_train
    )

    return {
        int(cls): float(weight)

        for cls, weight
        in zip(classes, weights)
    }


# ============================================================
# MATRIZ DE CONFUSIÓN
# ============================================================

def plot_confusion_matrix(
    cm,
    experiment_name
):

    plt.figure(
        figsize=(6, 5)
    )

    plt.imshow(cm)

    plt.title(
        experiment_name
    )

    plt.xlabel(
        "Predicción"
    )

    plt.ylabel(
        "Real"
    )

    plt.xticks(
        range(3),
        CLASS_NAMES
    )

    plt.yticks(
        range(3),
        CLASS_NAMES
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

    plt.show()


# ============================================================
# CURVAS DE ENTRENAMIENTO
# ============================================================

def plot_history(
    history,
    experiment_name
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
        "Época"
    )

    plt.ylabel(
        "Accuracy"
    )

    plt.title(
        f"{experiment_name} - Accuracy"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    plt.show()


# ============================================================
# EJECUTAR UN EXPERIMENTO
# ============================================================

def run_experiment(
    experiment_function
):

    # Limpiar modelo anterior
    tf.keras.backend.clear_session()

    set_seed(SEED)

    (
        train_mask,
        val_mask,
        test_mask,
        experiment_name
    ) = experiment_function()


    print("\n")
    print("=" * 80)
    print(experiment_name)
    print("=" * 80)


    # ========================================================
    # COMPROBAR QUE NO HAYA SOLAPAMIENTO
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
    ), "Leakage TRAIN / VALIDATION"

    assert train_files.isdisjoint(
        test_files
    ), "Leakage TRAIN / TEST"

    assert val_files.isdisjoint(
        test_files
    ), "Leakage VALIDATION / TEST"


    # ========================================================
    # MOSTRAR DISTRIBUCIÓN
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
    # EXTRAER DATOS
    # ========================================================

    X_train = X[train_mask]
    y_train = y[train_mask]

    X_val = X[val_mask]
    y_val = y[val_mask]

    X_test = X[test_mask]
    y_test = y[test_mask]


    # ========================================================
    # CANAL CNN
    # ========================================================

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


    print("\nShapes:")

    print(
        "Train:",
        X_train.shape
    )

    print(
        "Validation:",
        X_val.shape
    )

    print(
        "Test:",
        X_test.shape
    )


    # ========================================================
    # CLASS WEIGHTS
    # ========================================================

    if USE_CLASS_WEIGHTS:

        class_weights = (
            get_class_weights(
                y_train
            )
        )

        print(
            "\nClass weights:",
            class_weights
        )

    else:

        class_weights = None


    # ========================================================
    # CREAR MODELO
    # ========================================================

    model = build_model()


    # ========================================================
    # EARLY STOPPING
    # ========================================================

    early_stopping = (
        tf.keras.callbacks.EarlyStopping(

            monitor="val_loss",

            patience=10,

            restore_best_weights=True,

            verbose=1
        )
    )


    # ========================================================
    # ENTRENAMIENTO
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
    # PREDICCIÓN
    # ========================================================

    probabilities = model.predict(
        X_test,
        verbose=0
    )

    y_pred = np.argmax(
        probabilities,
        axis=1
    )


    # ========================================================
    # MÉTRICAS
    # ========================================================

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    macro_f1 = f1_score(
        y_test,
        y_pred,
        average="macro"
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


    print("\n")
    print("=" * 80)
    print("RESULTADOS TEST")
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


    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    print(
        "\nClassification Report:"
    )

    print(
        classification_report(

            y_test,
            y_pred,

            labels=[0, 1, 2],

            target_names=CLASS_NAMES,

            digits=4,

            zero_division=0
        )
    )


    # ========================================================
    # MATRIZ CONFUSIÓN
    # ========================================================

    cm = confusion_matrix(

        y_test,
        y_pred,

        labels=[0, 1, 2]
    )


    print(
        "\nConfusion Matrix:"
    )

    print(cm)


    # ========================================================
    # RESULTADOS POR ARCHIVO
    # ========================================================

    test_metadata = (
        metadata.loc[test_mask]
        .copy()
        .reset_index(drop=True)
    )

    test_metadata[
        "prediction"
    ] = y_pred

    test_metadata[
        "confidence"
    ] = np.max(
        probabilities,
        axis=1
    )

    test_metadata[
        "correct"
    ] = (
        y_test == y_pred
    )


    print("\nResultados por archivo:")
    print("-" * 80)


    file_results = []

    for file_name, group in (
        test_metadata.groupby("file")
    ):

        file_accuracy = (
            group["correct"].mean()
        )

        confidence = (
            group["confidence"].mean()
        )

        file_results.append({

            "file":
                file_name,

            "class":
                group["class"].iloc[0],

            "diameter":
                group["diameter_in"].iloc[0],

            "hp":
                group["load_hp"].iloc[0],

            "rpm":
                group["rpm"].iloc[0],

            "windows":
                len(group),

            "accuracy":
                file_accuracy,

            "confidence":
                confidence
        })


    file_results_df = pd.DataFrame(
        file_results
    )


    print(
        file_results_df.to_string(
            index=False
        )
    )


    # ========================================================
    # GRÁFICAS
    # ========================================================

    plot_confusion_matrix(
        cm,
        experiment_name
    )

    plot_history(
        history,
        experiment_name
    )


    # ========================================================
    # GUARDAR MODELO
    # ========================================================

    model_name = MODEL_OUTPUT_DIR / f"{experiment_name.lower()}.keras"

    model.save(
        model_name
    )


    # ========================================================
    # RESULTADO PARA TABLA FINAL
    # ========================================================

    return {

        "experiment":
            experiment_name,

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
            macro_f1
    }


# ============================================================
# SELECCIONAR EXPERIMENTOS
# ============================================================

EXPERIMENTS = [

    experiment_1,
    experiment_2,
    experiment_3,
    experiment_4

]


# ============================================================
# MAIN
# ============================================================

def main():

    results = []

    for experiment in EXPERIMENTS:

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
    print("RESUMEN FINAL DE EXPERIMENTOS")
    print("=" * 100)

    print(
        results_df.to_string(
            index=False
        )
    )


    # ========================================================
    # GUARDAR RESULTADOS
    # ========================================================

    results_df.to_csv(

        RESULTS_OUTPUT_DIR / "generalizacion_carga_diametro.csv",

        index=False
    )


    print(
        "\nResultados guardados en:"
    )

    print(
        RESULTS_OUTPUT_DIR / "generalizacion_carga_diametro.csv"
    )


# ============================================================
# EJECUCIÓN
# ============================================================

if __name__ == "__main__":

    main()
