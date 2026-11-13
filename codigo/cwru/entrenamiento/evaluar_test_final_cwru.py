"""Ejecuta una única evaluación final del protocolo CWRU congelado."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

CODIGO_DIR = Path(__file__).resolve().parents[2]
if str(CODIGO_DIR) not in sys.path:
    sys.path.insert(0, str(CODIGO_DIR))

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from comun.rutas_proyecto import DATOS_CWRU_PROCESADO, RESULTADOS_CWRU
from cwru.entrenamiento.seleccionar_escalado_features14 import build_model


CLASS_NAMES = ("N", "IR", "OR")
FEATURES_DIR = DATOS_CWRU_PROCESADO / "dual_12khz_w4096" / "features_14"
DATASET_DIR = DATOS_CWRU_PROCESADO / "dual_12khz_w4096"
DNN_DIR = RESULTADOS_CWRU / "modelo_features_14" / "canonico_seed42_int8"
WEIGHTS_DIR = (
    RESULTADOS_CWRU
    / "modelo_features_14"
    / "multisemilla_escalado"
    / "seed_42"
    / "standard"
)
KMEANS_DIR = (
    RESULTADOS_CWRU
    / "modelo_features_14"
    / "kmeans_normal_lolo_sin_frecuencia_dominante"
)
OUTPUT_DIR = RESULTADOS_CWRU / "modelo_features_14" / "evaluacion_final_test_load3"
EXPECTED_HASHES = {
    DNN_DIR / "modelo_cwru_de_fe_seed42_int8.tflite": "cabc6e967b91691287c0cd8b899958d6784f0712b9bc956f68565273c9652596",
    DNN_DIR / "standard_scaler_train_only.npz": "e2b1cf9675853f94778f42d992a08b428e49a54fe9d0e2cc604f5a42a35fb08c",
    KMEANS_DIR / "kmeans_normal_final_float32.npz": "5aa9fd59fae5ed3c42fa3d8b51c4ef6b1123894fcd6c1145c9040cc9ae7d22ca",
    FEATURES_DIR / "X_features_test.npy": "3ad9da211d986483d0d2c55c88fe600d9d942b21e0e66ef501852494cdc99853",
    FEATURES_DIR / "y_test.npy": "e1088b720732f0058f0825ce17dea45f1b71a3a3b6808636f4cffb1061a0b8f5",
    DATASET_DIR / "metadata_test.csv": "e433e561eb46d4eb74f85d3806f39dd0708c7a6bae816584c34036d89f5858c7",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_frozen_artifacts() -> None:
    mismatches = []
    for path, expected in EXPECTED_HASHES.items():
        observed = sha256(path)
        if observed != expected:
            mismatches.append({"path": str(path), "expected": expected, "observed": observed})
    if mismatches:
        raise RuntimeError(f"Artefactos distintos del protocolo congelado: {mismatches}")


def classification_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_precision": float(
            precision_score(labels, predictions, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(labels, predictions, average="macro", zero_division=0)
        ),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
    }


def run_int8(model_content: bytes, inputs: np.ndarray):
    interpreter = tf.lite.Interpreter(model_content=model_content)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    input_scale, input_zero_point = input_detail["quantization"]
    output_scale, output_zero_point = output_detail["quantization"]
    probabilities = []
    saturated_low = 0
    saturated_high = 0
    for sample in inputs:
        unbounded = np.rint(sample[np.newaxis, :] / input_scale + input_zero_point)
        saturated_low += int(np.sum(unbounded < -128))
        saturated_high += int(np.sum(unbounded > 127))
        quantized = np.clip(unbounded, -128, 127).astype(np.int8)
        interpreter.set_tensor(input_detail["index"], quantized)
        interpreter.invoke()
        output = interpreter.get_tensor(output_detail["index"])
        dequantized = (output.astype(np.float32) - output_zero_point) * output_scale
        probabilities.append(dequantized[0])
    return np.asarray(probabilities), {
        "input_scale": float(input_scale),
        "input_zero_point": int(input_zero_point),
        "output_scale": float(output_scale),
        "output_zero_point": int(output_zero_point),
        "input_values_total": int(inputs.size),
        "input_saturated_low": saturated_low,
        "input_saturated_high": saturated_high,
        "input_saturation_fraction": float(
            (saturated_low + saturated_high) / inputs.size
        ),
    }


def main() -> None:
    assert_frozen_artifacts()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    features = np.load(FEATURES_DIR / "X_features_test.npy").astype(np.float32)
    labels = np.load(FEATURES_DIR / "y_test.npy").astype(np.int64)
    metadata = pd.read_csv(DATASET_DIR / "metadata_test.csv")
    if features.shape != (406, 14) or len(labels) != 406 or len(metadata) != 406:
        raise ValueError("El conjunto de prueba no tiene la forma congelada.")
    if set(metadata["load_hp"].unique()) != {3}:
        raise ValueError("La prueba final debe corresponder exclusivamente a 3 HP.")
    if not np.array_equal(labels, metadata["label"].to_numpy()):
        raise ValueError("Las etiquetas de prueba no coinciden con la metadata.")

    scaler = np.load(DNN_DIR / "standard_scaler_train_only.npz")
    standardized = (
        (features - scaler["mean"].astype(np.float32))
        / scaler["scale"].astype(np.float32)
    ).astype(np.float32)
    model = build_model()
    stored_weights = np.load(WEIGHTS_DIR / "pesos_modelo.npz")
    model.set_weights([stored_weights[key] for key in sorted(stored_weights.files)])
    fp32_probabilities = model(standardized, training=False).numpy()
    fp32_predictions = np.argmax(fp32_probabilities, axis=1)

    int8_probabilities, quantization = run_int8(
        (DNN_DIR / "modelo_cwru_de_fe_seed42_int8.tflite").read_bytes(), standardized
    )
    int8_predictions = np.argmax(int8_probabilities, axis=1)

    detector = np.load(KMEANS_DIR / "kmeans_normal_final_float32.npz")
    feature_indices = np.asarray([0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 12, 13])
    detector_features = features[:, feature_indices]
    detector_standardized = (
        (detector_features - detector["scaler_mean"]) / detector["scaler_scale"]
    ).astype(np.float32)
    differences = (
        detector_standardized[:, np.newaxis, :] - detector["centroids"][np.newaxis, :, :]
    )
    distances = np.sum(differences * differences, axis=2, dtype=np.float32)
    detector_clusters = np.argmin(distances, axis=1)
    detector_scores = distances[np.arange(len(features)), detector_clusters]
    detector_threshold = float(detector["threshold"])
    rejected = detector_scores > detector_threshold

    is_fault = labels != 0
    is_normal = ~is_fault
    detector_binary = {
        "fault_windows_corroborated": int(np.sum(rejected & is_fault)),
        "fault_windows_not_corroborated": int(np.sum(~rejected & is_fault)),
        "normal_windows_rejected": int(np.sum(rejected & is_normal)),
        "normal_windows_accepted": int(np.sum(~rejected & is_normal)),
        "fault_corroboration_sensitivity": float(np.mean(rejected[is_fault])),
        "normal_specificity": float(np.mean(~rejected[is_normal])),
    }
    detector_binary["balanced_accuracy"] = float(
        (detector_binary["fault_corroboration_sensitivity"] + detector_binary["normal_specificity"])
        / 2.0
    )

    per_class_detector = {}
    for index, name in enumerate(CLASS_NAMES):
        mask = labels == index
        per_class_detector[name] = {
            "windows": int(np.sum(mask)),
            "rejected": int(np.sum(rejected[mask])),
            "rejection_fraction": float(np.mean(rejected[mask])),
            "score_median": float(np.median(detector_scores[mask])),
            "score_minimum": float(np.min(detector_scores[mask])),
            "score_maximum": float(np.max(detector_scores[mask])),
        }

    prediction_table = metadata.copy()
    prediction_table["y_pred_fp32"] = fp32_predictions
    prediction_table["y_pred_int8"] = int8_predictions
    prediction_table["dnn_int8_correct"] = (int8_predictions == labels).astype(np.uint8)
    prediction_table["normality_score"] = detector_scores
    prediction_table["normality_rejected"] = rejected.astype(np.uint8)
    prediction_table["normality_cluster"] = detector_clusters
    for index, name in enumerate(CLASS_NAMES):
        prediction_table[f"p_int8_{name}"] = int8_probabilities[:, index]
    prediction_table.to_csv(OUTPUT_DIR / "predicciones_finales_test.csv", index=False)

    fp32_matrix = confusion_matrix(labels, fp32_predictions, labels=[0, 1, 2])
    int8_matrix = confusion_matrix(labels, int8_predictions, labels=[0, 1, 2])
    pd.DataFrame(fp32_matrix, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
        OUTPUT_DIR / "matriz_confusion_dnn_fp32_test.csv"
    )
    pd.DataFrame(int8_matrix, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
        OUTPUT_DIR / "matriz_confusion_dnn_int8_test.csv"
    )

    summary = {
        "status": "final_test_evaluated_once_with_frozen_protocol",
        "test_load_hp": 3,
        "test_windows": 406,
        "dnn_fp32_metrics": classification_metrics(labels, fp32_predictions),
        "dnn_int8_metrics": classification_metrics(labels, int8_predictions),
        "fp32_int8_prediction_disagreements": int(
            np.sum(fp32_predictions != int8_predictions)
        ),
        "quantization": quantization,
        "normality_detector": {
            "k": int(detector["selected_k"]),
            "features": 12,
            "threshold": detector_threshold,
            "binary_metrics": detector_binary,
            "per_class": per_class_detector,
        },
        "joint_descriptive_counts": {
            "dnn_correct_and_detector_rejected": int(
                np.sum((int8_predictions == labels) & rejected)
            ),
            "dnn_correct_and_detector_accepted": int(
                np.sum((int8_predictions == labels) & ~rejected)
            ),
            "dnn_incorrect_and_detector_rejected": int(
                np.sum((int8_predictions != labels) & rejected)
            ),
            "dnn_incorrect_and_detector_accepted": int(
                np.sum((int8_predictions != labels) & ~rejected)
            ),
        },
        "frozen_artifact_hashes_verified": True,
        "post_test_rule": (
            "No frozen decision may be changed in response to this result; later "
            "variants must be labeled exploratory."
        ),
    }
    with (OUTPUT_DIR / "resumen_evaluacion_final_test.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)
        file.write("\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
