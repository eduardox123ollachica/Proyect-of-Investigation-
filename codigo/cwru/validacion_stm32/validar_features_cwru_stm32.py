"""Valida las 14 características CWRU entre PC y STM32 por UART.

Transmite una ventana dual de 4096 muestras mediante tramas ASCII pequeñas,
recibe las características calculadas con CMSIS-DSP y las compara con el
artefacto Python congelado. La transmisión es offline y no representa una ruta
de adquisición en tiempo real.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "datos" / "cwru" / "procesado" / "dual_12khz_w4096"
FEATURE_DIR = DATA_DIR / "features_14"
RESULTS_DIR = PROJECT_ROOT / "resultados" / "cwru" / "validacion_stm32_features"

WINDOW_PATH = DATA_DIR / "X_test.npy"
FEATURE_PATH = FEATURE_DIR / "X_features_test.npy"
LABEL_PATH = DATA_DIR / "y_test.npy"

FEATURE_NAMES = (
    "de_rms",
    "de_skewness",
    "de_kurtosis",
    "de_spectral_energy",
    "de_dominant_frequency",
    "de_dominant_amplitude",
    "de_spectral_entropy",
    "fe_rms",
    "fe_skewness",
    "fe_kurtosis",
    "fe_spectral_energy",
    "fe_dominant_frequency",
    "fe_dominant_amplitude",
    "fe_spectral_entropy",
)
DOMINANT_FREQUENCY_INDICES = (4, 11)
EXPECTED_WINDOW_SHAPE = (4096, 2)
EXPECTED_VALUE_COUNT = 8192


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="COM4", help="Puerto serie, por ejemplo COM4.")
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--index", type=int, default=0, help="Índice de X_test.npy.")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=32,
        help="Floats por trama; 32 mantiene la línea bajo 1024 bytes.",
    )
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--atol", type=float, default=1e-5)
    return parser.parse_args()


def read_expected_line(serial_port, prefix: str, timeout: float) -> str:
    deadline = time.monotonic() + timeout
    observed: list[str] = []

    while time.monotonic() < deadline:
        raw_line = serial_port.readline()
        if not raw_line:
            continue
        line = raw_line.decode("ascii", errors="replace").strip()
        if not line:
            continue
        observed.append(line)
        if line.startswith("CWRUFEAT_ERROR,"):
            raise RuntimeError(f"STM32 respondió: {line}")
        if line.startswith(prefix):
            return line

    raise TimeoutError(
        f"No se recibió una línea con prefijo {prefix!r}. "
        f"Últimas líneas: {observed[-5:]}"
    )


def send_command(serial_port, command: str, prefix: str, timeout: float) -> str:
    serial_port.write((command + "\r\n").encode("ascii"))
    serial_port.flush()
    return read_expected_line(serial_port, prefix, timeout)


def load_window(index: int) -> tuple[np.ndarray, np.ndarray, int]:
    for path in (WINDOW_PATH, FEATURE_PATH, LABEL_PATH):
        if not path.exists():
            raise FileNotFoundError(path)

    windows = np.load(WINDOW_PATH, mmap_mode="r")
    expected_features = np.load(FEATURE_PATH, mmap_mode="r")
    labels = np.load(LABEL_PATH, mmap_mode="r")

    if not (0 <= index < len(windows)):
        raise IndexError(f"Índice {index} fuera de 0..{len(windows) - 1}.")
    if windows[index].shape != EXPECTED_WINDOW_SHAPE:
        raise ValueError(f"Forma inesperada: {windows[index].shape}.")
    if expected_features[index].shape != (14,):
        raise ValueError(f"Vector de referencia inesperado: {expected_features[index].shape}.")

    return (
        np.asarray(windows[index], dtype=np.float32),
        np.asarray(expected_features[index], dtype=np.float64),
        int(labels[index]),
    )


def transmit_window(
    serial_port,
    window: np.ndarray,
    chunk_size: int,
    timeout: float,
) -> tuple[int, np.ndarray]:
    flattened = window.reshape(-1)
    if len(flattened) != EXPECTED_VALUE_COUNT:
        raise ValueError(f"Se esperaban {EXPECTED_VALUE_COUNT} valores.")

    ready = send_command(serial_port, "CWRUFEAT_BEGIN", "CWRUFEAT_READY,", timeout)
    expected_by_board = int(ready.split(",")[1])
    if expected_by_board != EXPECTED_VALUE_COUNT:
        raise RuntimeError(f"El STM32 espera {expected_by_board} valores.")

    for offset in range(0, EXPECTED_VALUE_COUNT, chunk_size):
        block = flattened[offset : offset + chunk_size]
        command = "CWRUFEAT_DATA," + str(offset) + "," + ",".join(
            format(float(value), ".9g") for value in block
        )
        acknowledgement = send_command(
            serial_port,
            command,
            "CWRUFEAT_ACK,",
            timeout,
        )
        received = int(acknowledgement.split(",")[1])
        expected_received = offset + len(block)
        if received != expected_received:
            raise RuntimeError(
                f"ACK inconsistente: STM32={received}, PC={expected_received}."
            )

        if received % 1024 == 0 or received == EXPECTED_VALUE_COUNT:
            print(f"  transmitidos {received}/{EXPECTED_VALUE_COUNT} valores")

    result = send_command(
        serial_port,
        "CWRUFEAT_RUN",
        "CWRUFEAT_RESULT,",
        timeout,
    )
    fields = result.split(",")
    if len(fields) != 16:
        raise RuntimeError(
            f"Respuesta incompleta: se esperaban ciclos + 14 características, "
            f"se recibió: {result}"
        )

    cycles = int(fields[1])
    features = np.asarray([float(value) for value in fields[2:]], dtype=np.float64)
    return cycles, features


def save_results(
    index: int,
    label: int,
    cycles: int,
    expected: np.ndarray,
    observed: np.ndarray,
    rtol: float,
    atol: float,
) -> tuple[Path, bool]:
    output_dir = RESULTS_DIR / f"ventana_{index:04d}"
    output_dir.mkdir(parents=True, exist_ok=True)

    absolute_error = np.abs(observed - expected)
    relative_error = absolute_error / np.maximum(np.abs(expected), 1e-30)
    close = np.isclose(observed, expected, rtol=rtol, atol=atol)
    frequencies_match = bool(
        np.all(observed[list(DOMINANT_FREQUENCY_INDICES)] == expected[list(DOMINANT_FREQUENCY_INDICES)])
    )
    passed = bool(np.all(close) and frequencies_match)

    with (output_dir / "comparacion_features.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(
            (
                "feature_index",
                "feature",
                "pc",
                "stm32",
                "error_absoluto",
                "error_relativo",
                "isclose",
            )
        )
        for feature_index, feature_name in enumerate(FEATURE_NAMES):
            writer.writerow(
                (
                    feature_index,
                    feature_name,
                    expected[feature_index],
                    observed[feature_index],
                    absolute_error[feature_index],
                    relative_error[feature_index],
                    bool(close[feature_index]),
                )
            )

    summary = {
        "status": "pass" if passed else "fail",
        "test_window_index": index,
        "label": label,
        "cycles_feature_extraction": cycles,
        "rtol": rtol,
        "atol": atol,
        "maximum_absolute_error": float(absolute_error.max()),
        "maximum_relative_error": float(relative_error.max()),
        "dominant_frequencies_exact": frequencies_match,
        "features_within_tolerance": int(np.sum(close)),
        "feature_count": len(FEATURE_NAMES),
    }
    with (output_dir / "resumen.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)
        file.write("\n")

    return output_dir, passed


def main() -> int:
    args = parse_arguments()
    if not (1 <= args.chunk_size <= 48):
        raise ValueError("--chunk-size debe estar entre 1 y 48 para el buffer actual.")

    try:
        import serial
    except ImportError as error:
        raise RuntimeError("Falta pyserial. Instálelo con: pip install pyserial") from error

    window, expected, label = load_window(args.index)
    print(f"Ventana test: {args.index}; clase: {label}; forma: {window.shape}")
    print(f"Abriendo {args.port} a {args.baudrate} baudios...")

    with serial.Serial(
        args.port,
        args.baudrate,
        timeout=0.25,
        write_timeout=args.timeout,
    ) as serial_port:
        time.sleep(1.0)
        serial_port.reset_input_buffer()
        cycles, observed = transmit_window(
            serial_port,
            window,
            args.chunk_size,
            args.timeout,
        )

    output_dir, passed = save_results(
        args.index,
        label,
        cycles,
        expected,
        observed,
        args.rtol,
        args.atol,
    )

    absolute_error = np.abs(observed - expected)
    relative_error = absolute_error / np.maximum(np.abs(expected), 1e-30)
    print(f"Ciclos extracción STM32: {cycles}")
    print(f"Error absoluto máximo: {absolute_error.max():.12g}")
    print(f"Error relativo máximo: {relative_error.max():.12g}")
    print(
        "Frecuencias dominantes exactas:",
        np.all(
            observed[list(DOMINANT_FREQUENCY_INDICES)]
            == expected[list(DOMINANT_FREQUENCY_INDICES)]
        ),
    )
    print("Resultado:", "PASS" if passed else "FAIL")
    print("Archivos:", output_dir)
    return 0 if passed else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, TimeoutError, ValueError, IndexError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error
