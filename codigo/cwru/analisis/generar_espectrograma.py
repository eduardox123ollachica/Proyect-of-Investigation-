"""
ESPECTROGRAMAS CWRU A 2 kHz

Archivos:
    Normal       -> 97.mat
    Inner Race   -> 109.mat
    Outer Race   -> 135.mat

Preprocesamiento:
    48 kHz
      ↓
    LPF 800 Hz
      ↓
    Decimación ×24
      ↓
    2 kHz

Después:
    STFT / Spectrogram
"""

from pathlib import Path
import sys

CODIGO_DIR = Path(__file__).resolve().parents[2]
if str(CODIGO_DIR) not in sys.path:
    sys.path.insert(0, str(CODIGO_DIR))

from comun.rutas_proyecto import DATOS_CWRU_RAW

import numpy as np
import matplotlib.pyplot as plt

from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt, spectrogram


# ============================================================
# CONFIGURACIÓN
# ============================================================

DATASET_DIR = DATOS_CWRU_RAW

FS_ORIGINAL = 48_000
FS_TARGET = 2_000

DECIMATION_FACTOR = 24

CUTOFF_HZ = 800
FILTER_ORDER = 6

# Archivos representativos
FILES = {
    "Normal": 97,
    "Inner Race": 109,
    "Outer Race": 135
}

# Duración que queremos visualizar
# None = utilizar todo el registro
DURATION_TO_PLOT = 5.0


# ============================================================
# BUSCAR ARCHIVO
# ============================================================

def find_file(file_number):

    filename = f"{file_number}.mat"

    for folder in [
        DATASET_DIR / "Normal",
        DATASET_DIR / "normal",
        DATASET_DIR / "IR",
        DATASET_DIR / "OR"
    ]:

        filepath = folder / filename

        if filepath.exists():
            return filepath

    raise FileNotFoundError(
        f"No se encontró {filename}"
    )


# ============================================================
# CARGAR SEÑAL
# ============================================================

def load_cwru(file_number):

    filepath = find_file(file_number)

    data = loadmat(filepath)

    expected_key = f"X{file_number:03d}_DE_time"

    if expected_key not in data:

        available = [
            key for key in data.keys()
            if "_DE_time" in key
        ]

        raise KeyError(
            f"No se encontró {expected_key} en "
            f"{filepath.name}\n"
            f"Disponibles: {available}"
        )

    signal = np.asarray(
        data[expected_key],
        dtype=np.float64
    ).squeeze()

    return signal, filepath


# ============================================================
# FILTRO ANTI-ALIAS
# ============================================================

def lowpass_filter(signal):

    sos = butter(
        N=FILTER_ORDER,
        Wn=CUTOFF_HZ,
        btype="lowpass",
        fs=FS_ORIGINAL,
        output="sos"
    )

    filtered = sosfiltfilt(
        sos,
        signal
    )

    return filtered


# ============================================================
# DIEZMADO
# ============================================================

def downsample(signal):

    return signal[::DECIMATION_FACTOR]


# ============================================================
# PREPROCESAMIENTO COMPLETO
# ============================================================

def preprocess(signal):

    # 48 kHz -> filtro antialias
    filtered = lowpass_filter(signal)

    # 48 kHz -> 2 kHz
    signal_2k = downsample(filtered)

    return signal_2k


# ============================================================
# ESPECTROGRAMA
# ============================================================

def plot_spectrogram(signal, title):

    # --------------------------------------------------------
    # Limitar duración
    # --------------------------------------------------------

    if DURATION_TO_PLOT is not None:

        max_samples = int(
            DURATION_TO_PLOT * FS_TARGET
        )

        signal = signal[:max_samples]

    # --------------------------------------------------------
    # Parámetros STFT
    # --------------------------------------------------------

    # 256 muestras a 2 kHz
    #
    # Resolución temporal:
    #   256 / 2000 = 128 ms
    #
    # Resolución frecuencial:
    #   2000 / 256 = 7.8125 Hz

    nperseg = 1024
    noverlap = 512

    frequencies, times, Sxx = spectrogram(
        signal,
        fs=FS_TARGET,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=False,
        scaling="density",
        mode="magnitude"
    )

    # --------------------------------------------------------
    # Convertir a dB
    # --------------------------------------------------------

    Sxx_dB = 20 * np.log10(
        Sxx + 1e-12
    )

    # --------------------------------------------------------
    # Graficar
    # --------------------------------------------------------

    plt.figure(
        figsize=(12, 6)
    )

    plt.pcolormesh(
        times,
        frequencies,
        Sxx_dB,
        shading="gouraud"
    )

    plt.xlabel(
        "Tiempo [s]"
    )

    plt.ylabel(
        "Frecuencia [Hz]"
    )

    plt.title(
        title
    )

    # Nyquist = 1000 Hz
    plt.ylim(
        0,
        FS_TARGET / 2
    )

    plt.colorbar(
        label="Magnitud [dB]"
    )

    plt.grid(
        alpha=0.2
    )

    plt.tight_layout()

    plt.show()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("ESPECTROGRAMAS CWRU")
    print("=" * 70)

    print(
        f"Frecuencia original : {FS_ORIGINAL} Hz"
    )

    print(
        f"Frecuencia final    : {FS_TARGET} Hz"
    )

    print(
        f"Filtro LPF          : {CUTOFF_HZ} Hz"
    )

    print(
        f"Diezmado            : ×{DECIMATION_FACTOR}"
    )

    print("=" * 70)

    for class_name, file_number in FILES.items():

        print(
            f"\nProcesando {class_name}: "
            f"{file_number}.mat"
        )

        signal, filepath = load_cwru(
            file_number
        )

        print(
            f"Archivo: {filepath}"
        )

        print(
            f"Muestras originales: "
            f"{len(signal):,}"
        )

        # ----------------------------------------------------
        # Preprocesamiento
        # ----------------------------------------------------

        signal_2k = preprocess(
            signal
        )

        print(
            f"Muestras a 2 kHz: "
            f"{len(signal_2k):,}"
        )

        print(
            f"Duración: "
            f"{len(signal_2k) / FS_TARGET:.2f} s"
        )

        # ----------------------------------------------------
        # Espectrograma
        # ----------------------------------------------------

        plot_spectrogram(
            signal_2k,
            f"{class_name} - {file_number}.mat - "
            f"Espectrograma a 2 kHz"
        )


if __name__ == "__main__":
    main()
