"""Diseña y valida el filtro antialias para CWRU 48 kHz -> 12 kHz.

La selección utiliza únicamente los archivos asignados a entrenamiento. Este
script no modifica los .mat ni genera todavía ventanas para el modelo.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

CODIGO_DIR = Path(__file__).resolve().parents[2]
if str(CODIGO_DIR) not in sys.path:
    sys.path.insert(0, str(CODIGO_DIR))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.io import loadmat
from scipy.signal import firwin, freqz, kaiserord, upfirdn, welch

from comun.rutas_proyecto import DATOS_CWRU_RAW, RESULTADOS_CWRU


RAIZ_PROYECTO = CODIGO_DIR.parent
MANIFEST_PATH = (
    RAIZ_PROYECTO
    / "datos"
    / "cwru"
    / "particiones"
    / "principal_por_carga"
    / "manifest_particion.csv"
)
OUTPUT_DIR = RESULTADOS_CWRU / "preprocesamiento_12khz" / "filtro_antialias"

FS_IN_HZ = 48_000
FS_OUT_HZ = 12_000
DECIMATION_FACTOR = FS_IN_HZ // FS_OUT_HZ

PASSBAND_EDGE_HZ = 5_000
STOPBAND_EDGE_HZ = 6_000
NOMINAL_CUTOFF_HZ = (PASSBAND_EDGE_HZ + STOPBAND_EDGE_HZ) / 2
TARGET_STOPBAND_DB = 60.0

TRAINING_BANDS_HZ = (
    (0, 1_000),
    (1_000, 3_000),
    (3_000, 5_000),
    (5_000, 5_250),
    (5_250, 5_500),
    (5_500, 5_750),
    (5_750, 6_000),
    (6_000, 7_000),
    (7_000, 9_000),
    (9_000, 12_000),
    (12_000, 24_000),
)

RESPONSE_FREQUENCIES_HZ = (
    0,
    1_000,
    3_000,
    4_500,
    5_000,
    5_250,
    5_500,
    5_750,
    6_000,
    6_500,
    7_000,
    9_000,
    12_000,
    24_000,
)

TONE_FREQUENCIES_HZ = (
    1_000,
    3_000,
    5_000,
    5_250,
    5_500,
    5_750,
    5_900,
    6_100,
    7_000,
    9_000,
)


def design_filter() -> tuple[np.ndarray, np.ndarray, float]:
    """Devuelve coeficientes float64/float32 y beta Kaiser."""

    transition_width = STOPBAND_EDGE_HZ - PASSBAND_EDGE_HZ
    normalized_width = transition_width / (FS_IN_HZ / 2)
    num_taps, beta = kaiserord(TARGET_STOPBAND_DB, normalized_width)

    # Longitud impar: fase lineal tipo I y retardo entero.
    if num_taps % 2 == 0:
        num_taps += 1

    taps64 = firwin(
        numtaps=num_taps,
        cutoff=NOMINAL_CUTOFF_HZ,
        window=("kaiser", beta),
        pass_zero="lowpass",
        scale=True,
        fs=FS_IN_HZ,
    )
    taps32 = taps64.astype(np.float32)
    return taps64, taps32, float(beta)


def calculate_response(taps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    frequencies, response = freqz(taps, worN=262_144, fs=FS_IN_HZ)
    response_db = 20 * np.log10(np.maximum(np.abs(response), 1e-15))
    return frequencies, response_db


def validate_response(
    frequencies: np.ndarray,
    response_db: np.ndarray,
) -> dict[str, float]:
    passband = response_db[frequencies <= PASSBAND_EDGE_HZ]
    stopband = response_db[frequencies >= STOPBAND_EDGE_HZ]

    metrics = {
        "passband_min_db": float(passband.min()),
        "passband_max_db": float(passband.max()),
        "passband_ripple_db": float(passband.max() - passband.min()),
        "stopband_max_db": float(stopband.max()),
    }

    if metrics["stopband_max_db"] > -TARGET_STOPBAND_DB:
        raise RuntimeError(
            "El filtro no cumple la atenuación de banda rechazada: "
            f"{metrics['stopband_max_db']:.6f} dB."
        )
    return metrics


def response_table(taps: np.ndarray) -> pd.DataFrame:
    requested = np.asarray(RESPONSE_FREQUENCIES_HZ, dtype=np.float64)
    angular = 2 * np.pi * requested / FS_IN_HZ
    _, response = freqz(taps, worN=angular)
    attenuation = 20 * np.log10(np.maximum(np.abs(response), 1e-15))
    return pd.DataFrame(
        {
            "frequency_hz": requested.astype(int),
            "magnitude_db": attenuation,
        }
    )


def load_training_manifest() -> pd.DataFrame:
    manifest = pd.read_csv(MANIFEST_PATH)
    training = manifest.loc[manifest["split"] == "train"].copy()
    if len(training) != 14:
        raise ValueError(
            f"Se esperaban 14 archivos de entrenamiento y hay {len(training)}."
        )
    return training


def locate_raw_file(filename: str) -> Path:
    matches = list(DATOS_CWRU_RAW.rglob(filename))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Se esperaba una coincidencia para {filename} y se hallaron "
            f"{len(matches)}."
        )
    return matches[0]


def load_channel(mat_data: dict, key: str, expected_samples: int) -> np.ndarray:
    if key not in mat_data:
        raise KeyError(f"No existe la variable {key}.")
    signal = np.asarray(mat_data[key], dtype=np.float64).squeeze()
    if signal.ndim != 1:
        raise ValueError(f"{key} no es un vector 1D.")
    if len(signal) != expected_samples:
        raise ValueError(
            f"{key}: {len(signal)} muestras; se esperaban {expected_samples}."
        )
    if not np.isfinite(signal).all():
        raise ValueError(f"{key} contiene valores no finitos.")
    return signal


def band_energy_percentages(signal: np.ndarray) -> dict[tuple[int, int], float]:
    centered = signal - signal.mean()
    frequencies, psd = welch(
        centered,
        fs=FS_IN_HZ,
        nperseg=min(8192, len(centered)),
        detrend=False,
        scaling="density",
    )
    total_energy = np.trapezoid(psd, frequencies)
    if total_energy <= 0:
        raise ValueError("La energía espectral total debe ser positiva.")

    result = {}
    for lower, upper in TRAINING_BANDS_HZ:
        include_upper = upper == FS_IN_HZ // 2
        mask = (frequencies >= lower) & (
            frequencies <= upper if include_upper else frequencies < upper
        )
        result[(lower, upper)] = float(
            100 * np.trapezoid(psd[mask], frequencies[mask]) / total_energy
        )
    return result


def decimate_aligned(signal: np.ndarray, taps: np.ndarray) -> np.ndarray:
    """FIR causal + diezmado, compensando el retardo entero del FIR.

    Los extremos todavía incluyen respuesta a extensión implícita por ceros.
    Se registran como margen no válido y no deben usarse al ventanear.
    """

    group_delay_input = (len(taps) - 1) // 2
    if group_delay_input % DECIMATION_FACTOR != 0:
        raise ValueError("El retardo debe ser divisible por el factor de diezmado.")

    group_delay_output = group_delay_input // DECIMATION_FACTOR
    filtered = upfirdn(taps, signal, up=1, down=DECIMATION_FACTOR)
    expected_length = int(np.ceil(len(signal) / DECIMATION_FACTOR))
    aligned = filtered[
        group_delay_output : group_delay_output + expected_length
    ]
    if len(aligned) != expected_length:
        raise RuntimeError("La salida alineada no tiene la longitud esperada.")
    return np.asarray(aligned, dtype=np.float32)


def validate_synthetic_tones(taps: np.ndarray) -> pd.DataFrame:
    """Comprueba en el dominio temporal la ganancia y el alias tras diezmar."""

    duration_seconds = 2.0
    sample_count = int(duration_seconds * FS_IN_HZ)
    time = np.arange(sample_count, dtype=np.float64) / FS_IN_HZ
    edge_margin = ((len(taps) - 1) // 2) // DECIMATION_FACTOR
    rows = []

    for frequency in TONE_FREQUENCIES_HZ:
        signal = np.cos(2 * np.pi * frequency * time + 0.37)
        output = decimate_aligned(signal, taps).astype(np.float64)
        valid_output = output[edge_margin:-edge_margin]
        measured_gain = np.sqrt(np.mean(valid_output**2)) / np.sqrt(
            np.mean(signal**2)
        )
        measured_db = 20 * np.log10(max(measured_gain, 1e-15))

        angular = np.asarray([2 * np.pi * frequency / FS_IN_HZ])
        _, response = freqz(taps, worN=angular)
        theoretical_db = 20 * np.log10(max(abs(response[0]), 1e-15))

        alias_frequency = abs(
            ((frequency + FS_OUT_HZ / 2) % FS_OUT_HZ) - FS_OUT_HZ / 2
        )
        rows.append(
            {
                "input_frequency_hz": frequency,
                "output_alias_frequency_hz": alias_frequency,
                "theoretical_gain_db": theoretical_db,
                "measured_gain_after_decimation_db": measured_db,
                "absolute_error_db": abs(measured_db - theoretical_db),
            }
        )

    result = pd.DataFrame(rows)
    if float(result["absolute_error_db"].max()) > 0.1:
        raise RuntimeError(
            "La validación temporal por tonos difiere más de 0.1 dB de la "
            "respuesta teórica."
        )
    return result


def analyze_training(
    manifest: pd.DataFrame,
    taps: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    energy_rows: list[dict] = []
    file_rows: list[dict] = []
    edge_margin_output = ((len(taps) - 1) // 2) // DECIMATION_FACTOR

    for row in manifest.to_dict(orient="records"):
        raw_path = locate_raw_file(row["file"])
        mat_data = loadmat(raw_path)
        expected_samples = int(row["samples_per_channel"])

        de = load_channel(mat_data, row["de_key"], expected_samples)
        fe = load_channel(mat_data, row["fe_key"], expected_samples)
        if len(de) != len(fe):
            raise ValueError(f"DE y FE no están sincronizados en {row['file']}.")

        outputs = {}
        for channel_name, signal in (("DE", de), ("FE", fe)):
            energies = band_energy_percentages(signal)
            for (lower, upper), percentage in energies.items():
                energy_rows.append(
                    {
                        "file": row["file"],
                        "class": row["class"],
                        "load_hp": int(row["load_hp"]),
                        "channel": channel_name,
                        "band_low_hz": lower,
                        "band_high_hz": upper,
                        "energy_percent": percentage,
                    }
                )
            outputs[channel_name] = decimate_aligned(signal, taps)

        if len(outputs["DE"]) != len(outputs["FE"]):
            raise RuntimeError(
                f"DE y FE difieren tras el diezmado en {row['file']}."
            )
        if not np.isfinite(outputs["DE"]).all() or not np.isfinite(
            outputs["FE"]
        ).all():
            raise RuntimeError(f"Salida no finita en {row['file']}.")

        file_rows.append(
            {
                "file": row["file"],
                "input_samples_per_channel": len(de),
                "output_samples_per_channel": len(outputs["DE"]),
                "expected_output_samples": int(
                    np.ceil(len(de) / DECIMATION_FACTOR)
                ),
                "invalid_edge_samples_each_side_at_12khz": edge_margin_output,
                "channels_synchronized": True,
                "finite_output": True,
            }
        )

    return pd.DataFrame(energy_rows), pd.DataFrame(file_rows)


def summarize_energy(energy: pd.DataFrame) -> pd.DataFrame:
    return (
        energy.groupby(["channel", "band_low_hz", "band_high_hz"])[
            "energy_percent"
        ]
        .agg(["mean", "median", "max"])
        .reset_index()
        .rename(
            columns={
                "mean": "mean_percent",
                "median": "median_percent",
                "max": "max_percent",
            }
        )
    )


def save_coefficients(taps64: np.ndarray, taps32: np.ndarray) -> None:
    coefficients = pd.DataFrame(
        {
            "index": np.arange(len(taps64)),
            "coefficient_float64": taps64,
            "coefficient_float32": taps32,
            "absolute_quantization_error": np.abs(taps64 - taps32),
        }
    )
    coefficients.to_csv(OUTPUT_DIR / "coeficientes_fir_kaiser_177.csv", index=False)
    np.save(OUTPUT_DIR / "coeficientes_fir_kaiser_177_float32.npy", taps32)


def save_response_figure(frequencies: np.ndarray, response_db: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)

    axes[0].plot(frequencies / 1_000, response_db, color="#1F5A94", linewidth=1.1)
    axes[0].axvline(6, color="#B33A3A", linestyle="--", linewidth=1)
    axes[0].set(xlim=(0, 24), ylim=(-120, 5), xlabel="Frecuencia (kHz)", ylabel="Magnitud (dB)")
    axes[0].grid(alpha=0.25)
    axes[0].set_title("Respuesta completa")

    axes[1].plot(frequencies / 1_000, response_db, color="#1F5A94", linewidth=1.2)
    axes[1].axvspan(0, 5, color="#4C9F70", alpha=0.12, label="Banda pasante")
    axes[1].axvspan(5, 6, color="#E6A93D", alpha=0.16, label="Transición")
    axes[1].axvspan(6, 6.5, color="#B33A3A", alpha=0.10, label="Banda rechazada")
    axes[1].axhline(-60, color="#B33A3A", linestyle=":", linewidth=1)
    axes[1].set(xlim=(4.5, 6.5), ylim=(-100, 3), xlabel="Frecuencia (kHz)")
    axes[1].grid(alpha=0.25)
    axes[1].set_title("Región de transición")
    axes[1].legend(fontsize=8, loc="lower left")

    fig.suptitle("Filtro antialias FIR Kaiser: 48 kHz → 12 kHz")
    fig.savefig(OUTPUT_DIR / "respuesta_filtro_antialias.png", dpi=220)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    taps64, taps32, beta = design_filter()
    frequencies, response_db = calculate_response(taps32)
    response_metrics = validate_response(frequencies, response_db)
    detailed_response = response_table(taps32)
    tone_validation = validate_synthetic_tones(taps32)
    manifest = load_training_manifest()
    energy, file_validation = analyze_training(manifest, taps32)
    energy_summary = summarize_energy(energy)

    coefficient_hash = hashlib.sha256(taps32.tobytes()).hexdigest()
    group_delay_input = (len(taps32) - 1) // 2
    group_delay_output = group_delay_input // DECIMATION_FACTOR

    specification = {
        "status": "validated_on_training_only",
        "input_sampling_rate_hz": FS_IN_HZ,
        "output_sampling_rate_hz": FS_OUT_HZ,
        "decimation_factor": DECIMATION_FACTOR,
        "filter_type": "linear_phase_fir_kaiser_type_i",
        "coefficient_dtype_for_pipeline": "float32",
        "num_taps": len(taps32),
        "kaiser_beta": beta,
        "passband_edge_hz": PASSBAND_EDGE_HZ,
        "nominal_cutoff_hz": NOMINAL_CUTOFF_HZ,
        "stopband_edge_hz": STOPBAND_EDGE_HZ,
        "target_stopband_attenuation_db": TARGET_STOPBAND_DB,
        "group_delay_input_samples": group_delay_input,
        "group_delay_output_samples": group_delay_output,
        "group_delay_ms": 1000 * group_delay_input / FS_IN_HZ,
        "invalid_edge_samples_each_side_at_12khz": group_delay_output,
        "response_float32": response_metrics,
        "training_files_used_for_signal_analysis": len(manifest),
        "validation_files_used": 0,
        "test_files_used": 0,
        "sha256_float32_coefficients": coefficient_hash,
        "scipy_version": scipy.__version__,
    }

    save_coefficients(taps64, taps32)
    detailed_response.to_csv(OUTPUT_DIR / "respuesta_frecuencias_control.csv", index=False)
    tone_validation.to_csv(OUTPUT_DIR / "validacion_tonos_sinteticos.csv", index=False)
    energy.to_csv(OUTPUT_DIR / "energia_espectral_training_por_archivo.csv", index=False)
    energy_summary.to_csv(OUTPUT_DIR / "energia_espectral_training_resumen.csv", index=False)
    file_validation.to_csv(OUTPUT_DIR / "validacion_diezmado_training.csv", index=False)
    with (OUTPUT_DIR / "especificacion_filtro.json").open("w", encoding="utf-8") as file:
        json.dump(specification, file, indent=2, ensure_ascii=False)
        file.write("\n")
    save_response_figure(frequencies, response_db)

    print("Filtro antialias validado.")
    print(f"Coeficientes              : {len(taps32)}")
    print(f"Beta Kaiser               : {beta:.6f}")
    print(f"Ondulación 0-5 kHz        : {response_metrics['passband_ripple_db']:.6f} dB")
    print(f"Máximo desde 6 kHz        : {response_metrics['stopband_max_db']:.6f} dB")
    print(f"Retardo                    : {group_delay_input} muestras ({specification['group_delay_ms']:.6f} ms)")
    print(f"Archivos training validados: {len(file_validation)}")
    print(f"SHA-256 coeficientes f32  : {coefficient_hash}")
    print(f"Resultados                : {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
