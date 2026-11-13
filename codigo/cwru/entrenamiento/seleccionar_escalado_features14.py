"""Compara RAW y StandardScaler para la DNN CWRU 14-20-10-3.

La selección utiliza entrenamiento y validación. El conjunto de prueba no se
carga en este script para impedir su consulta accidental.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

CODIGO_DIR = Path(__file__).resolve().parents[2]
if str(CODIGO_DIR) not in sys.path:
    sys.path.insert(0, str(CODIGO_DIR))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow.keras import layers, regularizers

from comun.rutas_proyecto import DATOS_CWRU_PROCESADO, RESULTADOS_CWRU


FEATURES_DIR = DATOS_CWRU_PROCESADO / "dual_12khz_w4096" / "features_14"
OUTPUT_DIR = (
    RESULTADOS_CWRU
    / "modelo_features_14"
    / "seleccion_escalado_seed42"
)

SEED = 42
MAX_EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.005
L2_FACTOR = 0.001
EARLY_STOPPING_PATIENCE = 10
CLASS_NAMES = ("N", "IR", "OR")


def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()


def build_model() -> keras.Model:
    penalty = regularizers.l2(L2_FACTOR)
    model = keras.Sequential(
        [
            layers.Input(shape=(14,), name="entrada_14_features_de_fe"),
            layers.Dense(
                20,
                activation="relu",
                kernel_regularizer=penalty,
                name="oculta_20",
            ),
            layers.Dense(
                10,
                activation="relu",
                kernel_regularizer=penalty,
                name="oculta_10",
            ),
            layers.Dense(3, activation="softmax", name="salida_3_clases"),
        ],
        name="cwru_features_de_fe_14_20_10_3",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def calculate_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict:
    predictions = np.argmax(probabilities, axis=1)
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "macro_precision": float(
            precision_score(y_true, predictions, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_true, predictions, average="macro", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(y_true, predictions, average="macro", zero_division=0)
        ),
        "predictions": predictions,
    }


def save_curves(history: pd.DataFrame, output_path: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), constrained_layout=True)
    axes[0].plot(history["epoch"], history["loss"], label="Entrenamiento")
    axes[0].plot(history["epoch"], history["val_loss"], label="Validación")
    axes[0].set(xlabel="Época", ylabel="Pérdida", title="Pérdida")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(history["epoch"], history["accuracy"], label="Entrenamiento")
    axes[1].plot(history["epoch"], history["val_accuracy"], label="Validación")
    axes[1].set(xlabel="Época", ylabel="Exactitud", title="Exactitud")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.suptitle(title)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def train_variant(
    variant: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    class_weights: dict[int, float],
    seed: int,
    output_dir: Path,
) -> dict:
    set_reproducibility(seed)
    variant_dir = output_dir / variant
    variant_dir.mkdir(parents=True, exist_ok=True)
    model = build_model()

    # Lotes explícitos: evita el adaptador tf.data, que puede bloquearse en
    # determinadas combinaciones de TensorFlow/Windows. No cambia el modelo,
    # el optimizador ni la definición de la función objetivo.
    random_generator = np.random.default_rng(seed)
    sample_weights = np.asarray(
        [class_weights[int(label)] for label in y_train], dtype=np.float32
    )
    best_val_loss = np.inf
    best_weights = model.get_weights()
    epochs_without_improvement = 0
    history_rows = []

    @tf.function(reduce_retracing=True)
    def training_step(batch_x, batch_y, batch_weights):
        with tf.GradientTape() as tape:
            probabilities = model(batch_x, training=True)
            per_example = keras.losses.sparse_categorical_crossentropy(
                batch_y, probabilities
            )
            weighted_loss = tf.reduce_mean(per_example * batch_weights)
            regularization_loss = tf.add_n(model.losses)
            total_loss = weighted_loss + regularization_loss
        gradients = tape.gradient(total_loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return total_loss

    for epoch_index in range(MAX_EPOCHS):
        shuffled = random_generator.permutation(len(x_train))
        for start in range(0, len(shuffled), BATCH_SIZE):
            indices = shuffled[start : start + BATCH_SIZE]
            training_step(
                tf.convert_to_tensor(x_train[indices]),
                tf.convert_to_tensor(y_train[indices]),
                tf.convert_to_tensor(sample_weights[indices]),
            )

        train_probabilities = model(x_train, training=False).numpy()
        validation_probabilities = model(x_validation, training=False).numpy()
        train_per_example = keras.losses.sparse_categorical_crossentropy(
            y_train, train_probabilities
        ).numpy()
        validation_per_example = keras.losses.sparse_categorical_crossentropy(
            y_validation, validation_probabilities
        ).numpy()
        regularization = float(sum(float(value.numpy()) for value in model.losses))
        train_loss = float(
            np.mean(train_per_example * sample_weights) + regularization
        )
        validation_loss = float(np.mean(validation_per_example) + regularization)
        train_accuracy = float(
            np.mean(np.argmax(train_probabilities, axis=1) == y_train)
        )
        validation_accuracy = float(
            np.mean(np.argmax(validation_probabilities, axis=1) == y_validation)
        )
        history_rows.append(
            {
                "epoch": epoch_index + 1,
                "loss": train_loss,
                "accuracy": train_accuracy,
                "val_loss": validation_loss,
                "val_accuracy": validation_accuracy,
            }
        )
        if not np.isfinite(
            [train_loss, validation_loss, train_accuracy, validation_accuracy]
        ).all():
            raise RuntimeError(f"{variant}: el entrenamiento produjo NaN o Inf.")

        if validation_loss < best_val_loss - 1e-12:
            best_val_loss = validation_loss
            best_weights = model.get_weights()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                break

    model.set_weights(best_weights)
    history = pd.DataFrame(history_rows)
    if not np.isfinite(history.select_dtypes(include=[np.number])).all().all():
        raise RuntimeError(f"{variant}: el entrenamiento produjo valores no finitos.")

    probabilities = model(x_validation, training=False).numpy()
    metrics = calculate_metrics(y_validation, probabilities)
    predictions = metrics.pop("predictions")
    matrix = confusion_matrix(y_validation, predictions, labels=[0, 1, 2])
    report = classification_report(
        y_validation,
        predictions,
        labels=[0, 1, 2],
        target_names=CLASS_NAMES,
        zero_division=0,
        output_dict=True,
    )

    best_epoch = int(history.loc[history["val_loss"].idxmin(), "epoch"])
    result = {
        "variant": variant,
        "seed": seed,
        "epochs_executed": len(history),
        "best_epoch_by_val_loss": best_epoch,
        "best_val_loss": float(history["val_loss"].min()),
        **metrics,
    }

    history.to_csv(variant_dir / "historial.csv", index=False)
    pd.DataFrame(matrix, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
        variant_dir / "matriz_confusion_validacion.csv"
    )
    pd.DataFrame(report).transpose().to_csv(
        variant_dir / "reporte_clasificacion_validacion.csv"
    )
    prediction_table = pd.DataFrame(
        {
            "y_true": y_validation,
            "y_pred": predictions,
            "p_N": probabilities[:, 0],
            "p_IR": probabilities[:, 1],
            "p_OR": probabilities[:, 2],
        }
    )
    prediction_table.to_csv(variant_dir / "predicciones_validacion.csv", index=False)
    with (variant_dir / "metricas_validacion.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(result, file, indent=2, ensure_ascii=False)
        file.write("\n")
    save_curves(history, variant_dir / "curvas.png", variant.upper())
    weights = model.get_weights()
    np.savez_compressed(
        variant_dir / "pesos_modelo.npz",
        **{f"weight_{index:02d}": value for index, value in enumerate(weights)},
    )
    with (variant_dir / "estructura_pesos.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(
            {
                "architecture": [14, 20, 10, 3],
                "weight_order": [f"weight_{index:02d}" for index in range(len(weights))],
                "weight_shapes": [list(value.shape) for value in weights],
                "weight_dtypes": [str(value.dtype) for value in weights],
            },
            file,
            indent=2,
            ensure_ascii=False,
        )
        file.write("\n")
    return result


def main(seed: int = SEED, output_dir: Path = OUTPUT_DIR) -> None:
    started_at = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    x_train_raw = np.load(FEATURES_DIR / "X_features_train.npy").astype(np.float32)
    y_train = np.load(FEATURES_DIR / "y_train.npy").astype(np.int64)
    x_validation_raw = np.load(
        FEATURES_DIR / "X_features_validation.npy"
    ).astype(np.float32)
    y_validation = np.load(FEATURES_DIR / "y_validation.npy").astype(np.int64)

    if x_train_raw.shape != (568, 14) or x_validation_raw.shape != (406, 14):
        raise ValueError("Las matrices de características no tienen la forma esperada.")

    classes = np.asarray([0, 1, 2], dtype=np.int64)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train,
    )
    class_weights = {int(label): float(weight) for label, weight in zip(classes, weights)}

    scaler = StandardScaler()
    x_train_standard = scaler.fit_transform(x_train_raw).astype(np.float32)
    x_validation_standard = scaler.transform(x_validation_raw).astype(np.float32)
    np.savez(
        output_dir / "standard_scaler_train_only.npz",
        mean=scaler.mean_.astype(np.float64),
        scale=scaler.scale_.astype(np.float64),
        var=scaler.var_.astype(np.float64),
    )

    results = []
    for variant, x_train, x_validation in (
        ("raw", x_train_raw, x_validation_raw),
        ("standard", x_train_standard, x_validation_standard),
    ):
        variant_started_at = time.perf_counter()
        print(f"Semilla {seed}: iniciando {variant}...", flush=True)
        result = train_variant(
            variant,
            x_train,
            y_train,
            x_validation,
            y_validation,
            class_weights,
            seed,
            output_dir,
        )
        result["duration_seconds"] = time.perf_counter() - variant_started_at
        results.append(result)
        print(
            f"Semilla {seed}: {variant} finalizado en "
            f"{result['duration_seconds']:.2f} s.",
            flush=True,
        )

    ranking = sorted(
        results,
        key=lambda item: (item["macro_f1"], item["accuracy"]),
        reverse=True,
    )
    selected = ranking[0]["variant"]
    summary = {
        "status": "selected_on_validation_test_not_loaded",
        "selection_metric_primary": "macro_f1",
        "selection_metric_tiebreaker": "accuracy",
        "selected_variant": selected,
        "class_weights": class_weights,
        "architecture": [14, 20, 10, 3],
        "l2_factor": L2_FACTOR,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "seed": seed,
        "duration_seconds": time.perf_counter() - started_at,
        "results": results,
    }
    with (output_dir / "seleccion.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)
        file.write("\n")
    pd.DataFrame(results).to_csv(output_dir / "comparacion_validacion.csv", index=False)

    print(pd.DataFrame(results).to_string(index=False))
    print(f"Variante seleccionada por validación: {selected}")
    print("El conjunto de prueba no fue cargado.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compara RAW y StandardScaler sin cargar el conjunto de prueba."
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output-dir", type=Path)
    arguments = parser.parse_args()
    selected_output_dir = arguments.output_dir or (
        RESULTADOS_CWRU
        / "modelo_features_14"
        / f"seleccion_escalado_seed{arguments.seed}"
    )
    main(seed=arguments.seed, output_dir=selected_output_dir)
