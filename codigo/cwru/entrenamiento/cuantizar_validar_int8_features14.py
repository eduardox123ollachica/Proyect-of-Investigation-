"""Convierte el modelo CWRU canónico a TFLite FP32 e INT8.

La calibración usa exclusivamente entrenamiento y la comparación FP32--INT8
usa exclusivamente validación. Este script no carga archivos de prueba.
"""

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


SEED = 42
CLASS_NAMES = ("N", "IR", "OR")
FEATURES_DIR = DATOS_CWRU_PROCESADO / "dual_12khz_w4096" / "features_14"
SEED_DIR = (
    RESULTADOS_CWRU
    / "modelo_features_14"
    / "multisemilla_escalado"
    / f"seed_{SEED}"
)
OUTPUT_DIR = RESULTADOS_CWRU / "modelo_features_14" / "canonico_seed42_int8"


def metrics(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[dict, np.ndarray]:
    predictions = np.argmax(probabilities, axis=1)
    return (
        {
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
        },
        predictions,
    )


def evaluate_tflite(model_content: bytes, inputs: np.ndarray) -> tuple[np.ndarray, dict]:
    interpreter = tf.lite.Interpreter(model_content=model_content)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    input_scale, input_zero_point = input_detail["quantization"]
    output_scale, output_zero_point = output_detail["quantization"]

    probabilities = []
    for sample in inputs:
        batch = sample[np.newaxis, :].astype(np.float32)
        if input_detail["dtype"] == np.int8:
            batch = np.clip(
                np.rint(batch / input_scale + input_zero_point), -128, 127
            ).astype(np.int8)
        interpreter.set_tensor(input_detail["index"], batch)
        interpreter.invoke()
        output = interpreter.get_tensor(output_detail["index"])
        if output_detail["dtype"] == np.int8:
            output = (output.astype(np.float32) - output_zero_point) * output_scale
        probabilities.append(output[0])

    details = {
        "input_dtype": np.dtype(input_detail["dtype"]).name,
        "input_shape": input_detail["shape"].astype(int).tolist(),
        "input_scale": float(input_scale),
        "input_zero_point": int(input_zero_point),
        "output_dtype": np.dtype(output_detail["dtype"]).name,
        "output_shape": output_detail["shape"].astype(int).tolist(),
        "output_scale": float(output_scale),
        "output_zero_point": int(output_zero_point),
        "operators": [item["op_name"] for item in interpreter._get_ops_details()],
    }
    return np.asarray(probabilities, dtype=np.float32), details


def file_record(path: Path) -> dict:
    content = path.read_bytes()
    return {
        "path": str(path.relative_to(OUTPUT_DIR)),
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    x_train_raw = np.load(FEATURES_DIR / "X_features_train.npy").astype(np.float32)
    x_validation_raw = np.load(FEATURES_DIR / "X_features_validation.npy").astype(
        np.float32
    )
    y_validation = np.load(FEATURES_DIR / "y_validation.npy").astype(np.int64)
    if x_train_raw.shape != (568, 14) or x_validation_raw.shape != (406, 14):
        raise ValueError("Las matrices de entrenamiento o validación son inesperadas.")

    scaler_data = np.load(SEED_DIR / "standard_scaler_train_only.npz")
    scaler_mean = scaler_data["mean"].astype(np.float32)
    scaler_scale = scaler_data["scale"].astype(np.float32)
    x_train = ((x_train_raw - scaler_mean) / scaler_scale).astype(np.float32)
    x_validation = ((x_validation_raw - scaler_mean) / scaler_scale).astype(np.float32)

    model = build_model()
    stored_weights = np.load(SEED_DIR / "standard" / "pesos_modelo.npz")
    weights = [stored_weights[key] for key in sorted(stored_weights.files)]
    model.set_weights(weights)

    keras_probabilities = model(x_validation, training=False).numpy()

    @tf.function(input_signature=[tf.TensorSpec(shape=[1, 14], dtype=tf.float32)])
    def inference_function(inputs):
        return model(inputs, training=False)

    concrete_function = inference_function.get_concrete_function()
    converter_fp32 = tf.lite.TFLiteConverter.from_concrete_functions(
        [concrete_function], model
    )
    tflite_fp32 = converter_fp32.convert()
    fp32_path = OUTPUT_DIR / "modelo_cwru_de_fe_seed42_fp32.tflite"
    fp32_path.write_bytes(tflite_fp32)

    def representative_dataset():
        for sample in x_train:
            yield [sample[np.newaxis, :].astype(np.float32)]

    converter_int8 = tf.lite.TFLiteConverter.from_concrete_functions(
        [concrete_function], model
    )
    converter_int8.optimizations = [tf.lite.Optimize.DEFAULT]
    converter_int8.representative_dataset = representative_dataset
    converter_int8.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter_int8.inference_input_type = tf.int8
    converter_int8.inference_output_type = tf.int8
    tflite_int8 = converter_int8.convert()
    int8_path = OUTPUT_DIR / "modelo_cwru_de_fe_seed42_int8.tflite"
    int8_path.write_bytes(tflite_int8)

    fp32_probabilities, fp32_details = evaluate_tflite(tflite_fp32, x_validation)
    int8_probabilities, int8_details = evaluate_tflite(tflite_int8, x_validation)
    if int8_details["input_dtype"] != "int8" or int8_details["output_dtype"] != "int8":
        raise RuntimeError("El modelo cuantizado no tiene entrada y salida INT8.")

    rows = []
    predictions_by_model = {}
    for name, probabilities in (
        ("keras_fp32", keras_probabilities),
        ("tflite_fp32", fp32_probabilities),
        ("tflite_int8", int8_probabilities),
    ):
        result, predictions = metrics(y_validation, probabilities)
        result["model"] = name
        rows.append(result)
        predictions_by_model[name] = predictions
        pd.DataFrame(
            confusion_matrix(y_validation, predictions, labels=[0, 1, 2]),
            index=CLASS_NAMES,
            columns=CLASS_NAMES,
        ).to_csv(OUTPUT_DIR / f"matriz_confusion_{name}_validacion.csv")

    prediction_table = pd.DataFrame({"y_true": y_validation})
    for name, predictions in predictions_by_model.items():
        prediction_table[f"y_pred_{name}"] = predictions
    prediction_table.to_csv(OUTPUT_DIR / "comparacion_predicciones_validacion.csv", index=False)

    keras_vs_fp32 = int(
        np.sum(predictions_by_model["keras_fp32"] != predictions_by_model["tflite_fp32"])
    )
    keras_vs_int8 = int(
        np.sum(predictions_by_model["keras_fp32"] != predictions_by_model["tflite_int8"])
    )
    fp32_vs_int8 = int(
        np.sum(predictions_by_model["tflite_fp32"] != predictions_by_model["tflite_int8"])
    )
    probability_difference = np.abs(fp32_probabilities - int8_probabilities)

    summary = {
        "status": "validated_on_validation_test_not_loaded",
        "seed": SEED,
        "architecture": [14, 20, 10, 3],
        "standard_scaler_fit_scope": "training_only",
        "representative_dataset_scope": "all_568_training_samples_only",
        "validation_samples": int(len(y_validation)),
        "metrics": rows,
        "prediction_disagreements": {
            "keras_fp32_vs_tflite_fp32": keras_vs_fp32,
            "keras_fp32_vs_tflite_int8": keras_vs_int8,
            "tflite_fp32_vs_tflite_int8": fp32_vs_int8,
        },
        "tflite_fp32_vs_int8_probability_absolute_difference": {
            "mean": float(np.mean(probability_difference)),
            "maximum": float(np.max(probability_difference)),
        },
        "fp32_tensor_details": fp32_details,
        "int8_tensor_details": int8_details,
        "files": {
            "fp32": file_record(fp32_path),
            "int8": file_record(int8_path),
        },
    }
    with (OUTPUT_DIR / "resumen_validacion_fp32_int8.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)
        file.write("\n")
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "metricas_validacion_fp32_int8.csv", index=False)
    np.savez(
        OUTPUT_DIR / "standard_scaler_train_only.npz",
        mean=scaler_mean,
        scale=scaler_scale,
        var=scaler_data["var"].astype(np.float32),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
