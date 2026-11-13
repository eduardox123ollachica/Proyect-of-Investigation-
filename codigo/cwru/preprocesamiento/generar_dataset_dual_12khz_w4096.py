"""Genera ventanas CWRU DE-FE a 12 kHz con N=4096 y 50 % de solapamiento.

La partición se lee del manifiesto congelado. El filtrado usa exactamente los
coeficientes float32 validados. Las ventanas se guardan sin normalización para
no mezclar esta decisión con el remuestreo y la segmentación.
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
from scipy.io import loadmat
from scipy.signal import upfirdn

from comun.rutas_proyecto import (
    DATOS_CWRU_PROCESADO,
    DATOS_CWRU_RAW,
    RESULTADOS_CWRU,
)


RAIZ_PROYECTO = CODIGO_DIR.parent
PARTITION_MANIFEST = (
    RAIZ_PROYECTO
    / "datos"
    / "cwru"
    / "particiones"
    / "principal_por_carga"
    / "manifest_particion.csv"
)
FILTER_RESULTS_DIR = (
    RESULTADOS_CWRU / "preprocesamiento_12khz" / "filtro_antialias"
)
FILTER_SPEC_PATH = FILTER_RESULTS_DIR / "especificacion_filtro.json"
FILTER_COEFFICIENTS_PATH = (
    FILTER_RESULTS_DIR / "coeficientes_fir_kaiser_177_float32.npy"
)
OUTPUT_DIR = DATOS_CWRU_PROCESADO / "dual_12khz_w4096"

FS_INPUT_HZ = 48_000
FS_OUTPUT_HZ = 12_000
DECIMATION_FACTOR = 4
WINDOW_SIZE = 4096
OVERLAP = 0.50
HOP_SIZE = 2048
CHANNEL_ORDER = ("DE", "FE")
EXPECTED_SPLIT_COUNTS = {
    "train": 568,
    "validation": 406,
    "test": 406,
}


def load_filter() -> tuple[np.ndarray, dict]:
    with FILTER_SPEC_PATH.open(encoding="utf-8") as file:
        specification = json.load(file)

    taps = np.load(FILTER_COEFFICIENTS_PATH)
    if taps.dtype != np.float32:
        raise TypeError(f"Los coeficientes deben ser float32 y son {taps.dtype}.")

    checks = {
        "input_sampling_rate_hz": FS_INPUT_HZ,
        "output_sampling_rate_hz": FS_OUTPUT_HZ,
        "decimation_factor": DECIMATION_FACTOR,
        "num_taps": len(taps),
    }
    for key, expected in checks.items():
        if specification[key] != expected:
            raise ValueError(
                f"Especificación incompatible en {key}: "
                f"{specification[key]} frente a {expected}."
            )

    actual_hash = hashlib.sha256(taps.tobytes()).hexdigest()
    expected_hash = specification["sha256_float32_coefficients"]
    if actual_hash != expected_hash:
        raise ValueError("El hash de los coeficientes FIR no coincide.")
    return taps, specification


def locate_raw_file(filename: str) -> Path:
    matches = list(DATOS_CWRU_RAW.rglob(filename))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Se esperaba un archivo {filename} y se encontraron {len(matches)}."
        )
    return matches[0]


def load_channel(mat_data: dict, key: str, expected_samples: int) -> np.ndarray:
    if key not in mat_data:
        raise KeyError(f"No existe {key} en el archivo MAT.")
    signal = np.asarray(mat_data[key], dtype=np.float64).squeeze()
    if signal.ndim != 1:
        raise ValueError(f"{key} no es un vector 1D.")
    if len(signal) != expected_samples:
        raise ValueError(
            f"{key} contiene {len(signal)} muestras; se esperaban "
            f"{expected_samples}."
        )
    if not np.isfinite(signal).all():
        raise ValueError(f"{key} contiene valores no finitos.")
    return signal


def decimate_and_trim(
    signal: np.ndarray,
    taps: np.ndarray,
    invalid_edge_samples: int,
) -> np.ndarray:
    group_delay_input = (len(taps) - 1) // 2
    if group_delay_input % DECIMATION_FACTOR:
        raise ValueError("El retardo FIR no es divisible por el factor 4.")
    group_delay_output = group_delay_input // DECIMATION_FACTOR
    if group_delay_output != invalid_edge_samples:
        raise ValueError("El margen de borde no coincide con el retardo FIR.")

    filtered = upfirdn(taps, signal, up=1, down=DECIMATION_FACTOR)
    expected_length = int(np.ceil(len(signal) / DECIMATION_FACTOR))
    aligned = filtered[
        group_delay_output : group_delay_output + expected_length
    ]
    if len(aligned) != expected_length:
        raise RuntimeError("Longitud inesperada después del diezmado.")
    if len(aligned) <= 2 * invalid_edge_samples:
        raise RuntimeError("La señal es demasiado corta para retirar los bordes.")

    valid = aligned[invalid_edge_samples:-invalid_edge_samples]
    return np.asarray(valid, dtype=np.float32)


def create_windows(two_channel_signal: np.ndarray) -> np.ndarray:
    if two_channel_signal.ndim != 2 or two_channel_signal.shape[1] != 2:
        raise ValueError("La señal debe tener forma (muestras, 2 canales).")
    if len(two_channel_signal) < WINDOW_SIZE:
        return np.empty((0, WINDOW_SIZE, 2), dtype=np.float32)

    count = (len(two_channel_signal) - WINDOW_SIZE) // HOP_SIZE + 1
    windows = np.empty((count, WINDOW_SIZE, 2), dtype=np.float32)
    for index in range(count):
        start = index * HOP_SIZE
        windows[index] = two_channel_signal[start : start + WINDOW_SIZE]
    return windows


def process_file(
    row: dict,
    taps: np.ndarray,
    invalid_edge_samples: int,
) -> tuple[np.ndarray, list[dict]]:
    raw_path = locate_raw_file(row["file"])
    mat_data = loadmat(raw_path)
    expected_samples = int(row["samples_per_channel"])

    de = load_channel(mat_data, row["de_key"], expected_samples)
    fe = load_channel(mat_data, row["fe_key"], expected_samples)
    if len(de) != len(fe):
        raise ValueError(f"DE y FE no coinciden en {row['file']}.")

    de_12k = decimate_and_trim(de, taps, invalid_edge_samples)
    fe_12k = decimate_and_trim(fe, taps, invalid_edge_samples)
    if len(de_12k) != len(fe_12k):
        raise RuntimeError(f"DE y FE difieren después del FIR en {row['file']}.")

    signal = np.column_stack((de_12k, fe_12k)).astype(np.float32, copy=False)
    windows = create_windows(signal)

    metadata_rows = []
    for window_index in range(len(windows)):
        start_valid = window_index * HOP_SIZE
        end_valid = start_valid + WINDOW_SIZE
        start_aligned = start_valid + invalid_edge_samples
        end_aligned = end_valid + invalid_edge_samples
        metadata_rows.append(
            {
                "split": row["split"],
                "file": row["file"],
                "file_id": int(row["file_id"]),
                "class": row["class"],
                "label": int(row["label"]),
                "diameter_in": float(row["diameter_in"]),
                "load_hp": int(row["load_hp"]),
                "rpm": int(row["rpm"]),
                "window_index_in_file": window_index,
                "start_sample_12khz_valid": start_valid,
                "end_sample_12khz_valid_exclusive": end_valid,
                "start_sample_12khz_aligned": start_aligned,
                "end_sample_12khz_aligned_exclusive": end_aligned,
                "start_time_s": start_aligned / FS_OUTPUT_HZ,
                "end_time_s": end_aligned / FS_OUTPUT_HZ,
                "de_key": row["de_key"],
                "fe_key": row["fe_key"],
            }
        )
    return windows, metadata_rows


def save_split(
    split: str,
    windows: list[np.ndarray],
    metadata_rows: list[dict],
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    x = np.concatenate(windows, axis=0)
    metadata = pd.DataFrame(metadata_rows).reset_index(drop=True)
    y = metadata["label"].to_numpy(dtype=np.int64)

    expected = EXPECTED_SPLIT_COUNTS[split]
    if x.shape != (expected, WINDOW_SIZE, 2):
        raise ValueError(
            f"{split}: forma {x.shape}; se esperaba "
            f"({expected}, {WINDOW_SIZE}, 2)."
        )
    if len(y) != expected or len(metadata) != expected:
        raise ValueError(f"{split}: X, y y metadata no coinciden.")
    if not np.isfinite(x).all():
        raise ValueError(f"{split}: X contiene valores no finitos.")

    np.save(OUTPUT_DIR / f"X_{split}.npy", x)
    np.save(OUTPUT_DIR / f"y_{split}.npy", y)
    metadata.to_csv(OUTPUT_DIR / f"metadata_{split}.csv", index=False)
    return x, y, metadata


def main() -> None:
    taps, filter_specification = load_filter()
    invalid_edge_samples = int(
        filter_specification["invalid_edge_samples_each_side_at_12khz"]
    )
    manifest = pd.read_csv(PARTITION_MANIFEST)
    if len(manifest) != 28 or manifest["file"].nunique() != 28:
        raise ValueError("El manifiesto debe contener 28 archivos únicos.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    windows_by_split = {split: [] for split in EXPECTED_SPLIT_COUNTS}
    metadata_by_split = {split: [] for split in EXPECTED_SPLIT_COUNTS}

    for row in manifest.to_dict(orient="records"):
        split = row["split"]
        if split not in windows_by_split:
            raise ValueError(f"Subconjunto desconocido: {split}.")
        windows, metadata_rows = process_file(row, taps, invalid_edge_samples)
        windows_by_split[split].append(windows)
        metadata_by_split[split].extend(metadata_rows)

    summary_rows = []
    all_metadata = []
    for split in ("train", "validation", "test"):
        x, y, metadata = save_split(
            split,
            windows_by_split[split],
            metadata_by_split[split],
        )
        all_metadata.append(metadata)
        for class_name, class_rows in metadata.groupby("class"):
            summary_rows.append(
                {
                    "split": split,
                    "class": class_name,
                    "windows": len(class_rows),
                    "files": class_rows["file"].nunique(),
                }
            )
        print(f"{split:10s}: X={x.shape}, y={y.shape}")

    combined_metadata = pd.concat(all_metadata, ignore_index=True)
    file_split_counts = combined_metadata.groupby("file")["split"].nunique()
    if int(file_split_counts.max()) != 1:
        raise RuntimeError("Un archivo aparece en más de un subconjunto.")

    combined_metadata.to_csv(OUTPUT_DIR / "metadata_todas_las_ventanas.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(OUTPUT_DIR / "resumen_por_clase.csv", index=False)

    configuration = {
        "status": "generated_not_normalized_not_trained",
        "source_dataset": "CWRU 48 kHz drive-end bearing fault collection",
        "partition_manifest": str(PARTITION_MANIFEST.relative_to(RAIZ_PROYECTO)),
        "input_sampling_rate_hz": FS_INPUT_HZ,
        "output_sampling_rate_hz": FS_OUTPUT_HZ,
        "decimation_factor": DECIMATION_FACTOR,
        "channel_order": list(CHANNEL_ORDER),
        "window_size_samples": WINDOW_SIZE,
        "window_duration_ms": 1000 * WINDOW_SIZE / FS_OUTPUT_HZ,
        "overlap_fraction": OVERLAP,
        "hop_size_samples": HOP_SIZE,
        "hop_duration_ms": 1000 * HOP_SIZE / FS_OUTPUT_HZ,
        "fft_bin_spacing_hz_if_nfft_equals_window": FS_OUTPUT_HZ / WINDOW_SIZE,
        "normalization": "none",
        "filter_coefficients_sha256": filter_specification[
            "sha256_float32_coefficients"
        ],
        "filter_num_taps": len(taps),
        "invalid_edge_samples_each_side_at_12khz": invalid_edge_samples,
        "split_window_counts": EXPECTED_SPLIT_COUNTS,
        "array_layout": "windows, samples, channels",
        "array_dtype": "float32",
        "labels": {"N": 0, "IR": 1, "OR": 2},
    }
    with (OUTPUT_DIR / "configuracion.json").open("w", encoding="utf-8") as file:
        json.dump(configuration, file, indent=2, ensure_ascii=False)
        file.write("\n")

    print(f"Dataset generado en: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
