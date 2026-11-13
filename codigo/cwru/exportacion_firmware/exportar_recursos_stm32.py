"""Exporta los artefactos CWRU congelados como fuentes C/C++ para STM32."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

CODIGO_DIR = Path(__file__).resolve().parents[2]
if str(CODIGO_DIR) not in sys.path:
    sys.path.insert(0, str(CODIGO_DIR))

import numpy as np

from comun.rutas_proyecto import MODELOS_CWRU, RESULTADOS_CWRU


DNN_DIR = RESULTADOS_CWRU / "modelo_features_14" / "canonico_seed42_int8"
KMEANS_DIR = (
    RESULTADOS_CWRU
    / "modelo_features_14"
    / "kmeans_normal_lolo_sin_frecuencia_dominante"
)
OUTPUT_DIR = MODELOS_CWRU / "canonico_seed42" / "firmware"
EXPECTED_MODEL_HASH = "cabc6e967b91691287c0cd8b899958d6784f0712b9bc956f68565273c9652596"
EXPECTED_DNN_SCALER_HASH = "e2b1cf9675853f94778f42d992a08b428e49a54fe9d0e2cc604f5a42a35fb08c"
EXPECTED_KMEANS_HASH = "5aa9fd59fae5ed3c42fa3d8b51c4ef6b1123894fcd6c1145c9040cc9ae7d22ca"
FEATURE_INDICES = np.asarray([0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 12, 13], dtype=np.uint8)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def float_c(value: float) -> str:
    return f"{float(np.float32(value)):.9e}f"


def write_model(model_bytes: bytes) -> None:
    header = """\
#ifndef CWRU_MODEL_DATA_H_
#define CWRU_MODEL_DATA_H_

extern const unsigned char g_cwru_model[];
extern const unsigned int g_cwru_model_len;

#endif  // CWRU_MODEL_DATA_H_
"""
    lines = ['#include "cwru_model_data.h"', "", "alignas(16) const unsigned char g_cwru_model[] = {"]
    for start in range(0, len(model_bytes), 12):
        block = model_bytes[start : start + 12]
        suffix = "," if start + 12 < len(model_bytes) else ""
        lines.append("    " + ", ".join(f"0x{value:02x}" for value in block) + suffix)
    lines.extend(["};", "", f"const unsigned int g_cwru_model_len = {len(model_bytes)};", ""])
    (OUTPUT_DIR / "cwru_model_data.h").write_text(header, encoding="utf-8")
    (OUTPUT_DIR / "cwru_model_data.cpp").write_text("\n".join(lines), encoding="utf-8")


def write_dnn_scaler(mean: np.ndarray, scale: np.ndarray) -> None:
    header = """\
#ifndef CWRU_DNN_SCALER_H_
#define CWRU_DNN_SCALER_H_

#define CWRU_DNN_NUM_FEATURES 14

extern const float cwru_dnn_scaler_mean[CWRU_DNN_NUM_FEATURES];
extern const float cwru_dnn_scaler_scale[CWRU_DNN_NUM_FEATURES];
extern const float cwru_dnn_input_quant_scale;
extern const int cwru_dnn_input_quant_zero_point;
extern const float cwru_dnn_output_quant_scale;
extern const int cwru_dnn_output_quant_zero_point;

#endif  // CWRU_DNN_SCALER_H_
"""
    lines = ['#include "cwru_dnn_scaler.h"', ""]
    lines.append("const float cwru_dnn_scaler_mean[CWRU_DNN_NUM_FEATURES] = {")
    lines.extend(f"    {float_c(value)}," for value in mean)
    lines.extend(["};", ""])
    lines.append("const float cwru_dnn_scaler_scale[CWRU_DNN_NUM_FEATURES] = {")
    lines.extend(f"    {float_c(value)}," for value in scale)
    lines.extend(
        [
            "};",
            "",
            f"const float cwru_dnn_input_quant_scale = {float_c(0.033691830933094025)};",
            "const int cwru_dnn_input_quant_zero_point = -9;",
            f"const float cwru_dnn_output_quant_scale = {float_c(0.00390625)};",
            "const int cwru_dnn_output_quant_zero_point = -128;",
            "",
        ]
    )
    (OUTPUT_DIR / "cwru_dnn_scaler.h").write_text(header, encoding="utf-8")
    (OUTPUT_DIR / "cwru_dnn_scaler.cpp").write_text("\n".join(lines), encoding="utf-8")


def write_normality_detector(data: np.lib.npyio.NpzFile) -> None:
    mean = data["scaler_mean"].astype(np.float32)
    scale = data["scaler_scale"].astype(np.float32)
    centroids = data["centroids"].astype(np.float32)
    threshold = np.float32(data["threshold"])
    selected_k = int(data["selected_k"])
    if mean.shape != (12,) or scale.shape != (12,) or centroids.shape != (1, 12):
        raise ValueError("Dimensiones inesperadas en el detector normal-only.")
    if selected_k != 1 or np.any(scale <= 0) or not np.isfinite(centroids).all():
        raise ValueError("Parámetros inválidos en el detector normal-only.")

    header = """\
#ifndef CWRU_NORMALITY_PARAMS_H_
#define CWRU_NORMALITY_PARAMS_H_

#include <stdint.h>

#define CWRU_NORMALITY_NUM_CLUSTERS 1
#define CWRU_NORMALITY_NUM_FEATURES 12

extern const uint8_t cwru_normality_feature_indices[CWRU_NORMALITY_NUM_FEATURES];
extern const float cwru_normality_scaler_mean[CWRU_NORMALITY_NUM_FEATURES];
extern const float cwru_normality_scaler_scale[CWRU_NORMALITY_NUM_FEATURES];
extern const float cwru_normality_centroids[CWRU_NORMALITY_NUM_CLUSTERS][CWRU_NORMALITY_NUM_FEATURES];
extern const float cwru_normality_threshold_d2;

#endif  // CWRU_NORMALITY_PARAMS_H_
"""
    lines = ['#include "cwru_normality_params.h"', ""]
    lines.append("const uint8_t cwru_normality_feature_indices[CWRU_NORMALITY_NUM_FEATURES] = {")
    lines.append("    " + ", ".join(str(int(value)) for value in FEATURE_INDICES) + ",")
    lines.extend(["};", ""])
    lines.append("const float cwru_normality_scaler_mean[CWRU_NORMALITY_NUM_FEATURES] = {")
    lines.extend(f"    {float_c(value)}," for value in mean)
    lines.extend(["};", ""])
    lines.append("const float cwru_normality_scaler_scale[CWRU_NORMALITY_NUM_FEATURES] = {")
    lines.extend(f"    {float_c(value)}," for value in scale)
    lines.extend(["};", ""])
    lines.append(
        "const float cwru_normality_centroids[CWRU_NORMALITY_NUM_CLUSTERS]"
        "[CWRU_NORMALITY_NUM_FEATURES] = {"
    )
    lines.append("    {")
    lines.extend(f"        {float_c(value)}," for value in centroids[0])
    lines.extend(["    },", "};", ""])
    lines.append(f"const float cwru_normality_threshold_d2 = {float_c(threshold)};")
    lines.append("")
    (OUTPUT_DIR / "cwru_normality_params.h").write_text(header, encoding="utf-8")
    (OUTPUT_DIR / "cwru_normality_params.cpp").write_text("\n".join(lines), encoding="utf-8")


def write_pipeline_config() -> None:
    header = """\
#ifndef CWRU_PIPELINE_CONFIG_H_
#define CWRU_PIPELINE_CONFIG_H_

#define CWRU_SAMPLE_RATE_HZ 12000
#define CWRU_WINDOW_SAMPLES 4096
#define CWRU_HOP_SAMPLES 2048
#define CWRU_NUM_CHANNELS 2
#define CWRU_NUM_FEATURES_PER_CHANNEL 7
#define CWRU_TOTAL_FEATURES 14

// Orden de entrada: DE, FE. No representan ejes ortogonales X/Y/Z.
// La ruta embebida recibe o adquiere señales ya muestreadas a 12 kHz.

#endif  // CWRU_PIPELINE_CONFIG_H_
"""
    (OUTPUT_DIR / "cwru_pipeline_config.h").write_text(header, encoding="utf-8")


def main() -> None:
    model_path = DNN_DIR / "modelo_cwru_de_fe_seed42_int8.tflite"
    scaler_path = DNN_DIR / "standard_scaler_train_only.npz"
    detector_path = KMEANS_DIR / "kmeans_normal_final_float32.npz"
    expected = {
        model_path: EXPECTED_MODEL_HASH,
        scaler_path: EXPECTED_DNN_SCALER_HASH,
        detector_path: EXPECTED_KMEANS_HASH,
    }
    for path, expected_hash in expected.items():
        if sha256(path) != expected_hash:
            raise RuntimeError(f"Hash congelado no coincide: {path}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_bytes = model_path.read_bytes()
    dnn_scaler = np.load(scaler_path)
    detector = np.load(detector_path)
    write_model(model_bytes)
    write_dnn_scaler(
        dnn_scaler["mean"].astype(np.float32),
        dnn_scaler["scale"].astype(np.float32),
    )
    write_normality_detector(detector)
    write_pipeline_config()

    generated = sorted(path for path in OUTPUT_DIR.iterdir() if path.is_file())
    manifest = {
        "status": "firmware_resources_exported_from_frozen_artifacts",
        "target": "STM32F446RE_ARM_Cortex_M4F",
        "model_bytes": len(model_bytes),
        "window_samples": 4096,
        "hop_samples": 2048,
        "sample_rate_hz": 12000,
        "channels": ["DE", "FE"],
        "generated_files": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in generated
            if path.name != "manifest.json"
        },
        "source_hashes": {
            "tflite_int8": EXPECTED_MODEL_HASH,
            "dnn_scaler": EXPECTED_DNN_SCALER_HASH,
            "normality_detector": EXPECTED_KMEANS_HASH,
        },
    }
    (OUTPUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
