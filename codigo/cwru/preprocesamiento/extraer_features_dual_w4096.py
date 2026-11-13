"""Extrae 14 características del dataset CWRU dual DE-FE.

Se conservan las siete definiciones numéricas del pipeline de Arciniegas y se
aplican por separado a las dos posiciones de acelerómetro. Este paso no escala
las características ni entrena modelos.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

CODIGO_DIR = Path(__file__).resolve().parents[2]
if str(CODIGO_DIR) not in sys.path:
    sys.path.insert(0, str(CODIGO_DIR))

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

from comun.rutas_proyecto import DATOS_CWRU_PROCESADO


INPUT_DIR = DATOS_CWRU_PROCESADO / "dual_12khz_w4096"
OUTPUT_DIR = INPUT_DIR / "features_14"
FS_HZ = 12_000
FFT_SIZE = 4096
CHANNEL_NAMES = ("de", "fe")
BASE_FEATURE_NAMES = (
    "rms",
    "skewness",
    "kurtosis",
    "spectral_energy",
    "dominant_frequency",
    "dominant_amplitude",
    "spectral_entropy",
)
FEATURE_NAMES = tuple(
    f"{channel}_{feature}"
    for channel in CHANNEL_NAMES
    for feature in BASE_FEATURE_NAMES
)


def spectral_entropy(power: np.ndarray) -> float:
    power = np.asarray(power, dtype=np.float64)
    total = float(np.sum(power))
    if total <= 0:
        return 0.0
    probabilities = power / total
    probabilities = probabilities[probabilities > 0]
    if len(probabilities) <= 1:
        return 0.0
    maximum_entropy = np.log2(len(power))
    if maximum_entropy <= 0:
        return 0.0
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return float(entropy / maximum_entropy)


def extract_channel_features(signal: np.ndarray) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float64)
    if signal.shape != (FFT_SIZE,):
        raise ValueError(f"Ventana inválida: {signal.shape}.")

    rms_value = float(np.sqrt(np.mean(signal**2)))
    skew_value = float(skew(signal, bias=False))
    kurtosis_value = float(kurtosis(signal, fisher=True, bias=False))
    if not np.isfinite(skew_value):
        skew_value = 0.0
    if not np.isfinite(kurtosis_value):
        kurtosis_value = 0.0

    ac_signal = signal - np.mean(signal)
    spectrum = np.fft.rfft(ac_signal, n=FFT_SIZE)
    magnitude = np.abs(spectrum)
    power = magnitude**2
    frequencies = np.fft.rfftfreq(FFT_SIZE, d=1.0 / FS_HZ)
    spectral_energy_value = float(np.sum(power))
    dominant_index = int(np.argmax(magnitude[1:]) + 1)

    return np.asarray(
        [
            rms_value,
            skew_value,
            kurtosis_value,
            spectral_energy_value,
            float(frequencies[dominant_index]),
            float(magnitude[dominant_index]),
            spectral_entropy(power),
        ],
        dtype=np.float64,
    )


def extract_dataset_features(windows: np.ndarray) -> np.ndarray:
    if windows.ndim != 3 or windows.shape[1:] != (FFT_SIZE, 2):
        raise ValueError(f"Matriz de ventanas inválida: {windows.shape}.")

    result = np.empty((len(windows), len(FEATURE_NAMES)), dtype=np.float64)
    for index, window in enumerate(windows):
        result[index, :7] = extract_channel_features(window[:, 0])
        result[index, 7:] = extract_channel_features(window[:, 1])
    return result


def validate_features(features: np.ndarray, split: str) -> None:
    if features.shape[1] != 14:
        raise ValueError(f"{split}: se esperaban 14 características.")
    if not np.isfinite(features).all():
        raise ValueError(f"{split}: hay características no finitas.")

    for offset in (0, 7):
        frequencies = features[:, offset + 4]
        entropy = features[:, offset + 6]
        bin_numbers = frequencies / (FS_HZ / FFT_SIZE)
        if np.any(frequencies <= 0) or np.any(frequencies > FS_HZ / 2):
            raise ValueError(f"{split}: frecuencia dominante fuera de rango.")
        if not np.allclose(bin_numbers, np.rint(bin_numbers), atol=1e-10):
            raise ValueError(f"{split}: frecuencia dominante fuera de bins FFT.")
        if np.any(entropy < 0) or np.any(entropy > 1):
            raise ValueError(f"{split}: entropía espectral fuera de [0, 1].")


def main() -> None:
    with (INPUT_DIR / "configuracion.json").open(encoding="utf-8") as file:
        source_configuration = json.load(file)
    if source_configuration["window_size_samples"] != FFT_SIZE:
        raise ValueError("La longitud de ventana fuente no coincide con la FFT.")
    if source_configuration["output_sampling_rate_hz"] != FS_HZ:
        raise ValueError("La frecuencia fuente no coincide con 12 kHz.")
    if source_configuration["channel_order"] != ["DE", "FE"]:
        raise ValueError("El orden de canales fuente debe ser DE, FE.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for split in ("train", "validation", "test"):
        windows = np.load(INPUT_DIR / f"X_{split}.npy", mmap_mode="r")
        labels = np.load(INPUT_DIR / f"y_{split}.npy")
        metadata = pd.read_csv(INPUT_DIR / f"metadata_{split}.csv")
        if len(windows) != len(labels) or len(windows) != len(metadata):
            raise ValueError(f"{split}: ventanas, etiquetas y metadata no coinciden.")

        features = extract_dataset_features(windows)
        validate_features(features, split)
        np.save(OUTPUT_DIR / f"X_features_{split}.npy", features)
        np.save(OUTPUT_DIR / f"y_{split}.npy", labels)

        enriched_metadata = metadata.copy()
        for feature_index, feature_name in enumerate(FEATURE_NAMES):
            enriched_metadata[feature_name] = features[:, feature_index]
        enriched_metadata.to_csv(
            OUTPUT_DIR / f"metadata_features_{split}.csv",
            index=False,
        )

        for feature_index, feature_name in enumerate(FEATURE_NAMES):
            values = features[:, feature_index]
            summary_rows.append(
                {
                    "split": split,
                    "feature": feature_name,
                    "minimum": float(values.min()),
                    "mean": float(values.mean()),
                    "std_population": float(values.std(ddof=0)),
                    "median": float(np.median(values)),
                    "maximum": float(values.max()),
                }
            )
        print(f"{split:10s}: {features.shape}")

    pd.DataFrame(summary_rows).to_csv(
        OUTPUT_DIR / "resumen_features_por_split.csv",
        index=False,
    )
    with (OUTPUT_DIR / "feature_names.txt").open("w", encoding="utf-8") as file:
        for index, feature_name in enumerate(FEATURE_NAMES):
            file.write(f"{index:02d}: {feature_name}\n")

    configuration = {
        "status": "features_generated_not_scaled_not_trained",
        "source_directory": str(INPUT_DIR),
        "sampling_rate_hz": FS_HZ,
        "fft_size": FFT_SIZE,
        "fft_bin_spacing_hz": FS_HZ / FFT_SIZE,
        "channel_order": ["DE", "FE"],
        "feature_order": list(FEATURE_NAMES),
        "feature_count": len(FEATURE_NAMES),
        "rms_input": "original_window",
        "fft_input": "window_minus_its_mean",
        "dominant_frequency": "maximum_non_dc_rfft_bin",
        "kurtosis": "fisher_bias_false",
        "skewness": "bias_false",
        "spectral_energy": "sum_squared_rfft_magnitudes",
        "spectral_entropy": "normalized_base2_over_all_rfft_bins",
        "scaling": "none",
    }
    with (OUTPUT_DIR / "configuracion_features.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(configuration, file, indent=2, ensure_ascii=False)
        file.write("\n")

    print(f"Características generadas en: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
