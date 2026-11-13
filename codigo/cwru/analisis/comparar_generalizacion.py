"""
ANÁLISIS DEL DATASET CWRU PARA TINYML

Objetivos:
1. Visualizar ventanas de 1024 muestras a 2 kHz.
2. Analizar su contenido frecuencial mediante FFT.
3. Comparar señal original de 48 kHz con:
       - señal filtrada
       - señal diezmada a 2 kHz
4. Verificar el efecto del filtro anti-alias.

Dataset:
    Normal / IR / OR

Preprocesamiento:
    48 kHz -> LPF 800 Hz -> decimación x24 -> 2 kHz
    Ventana = 1024
"""

from pathlib import Path
import sys

CODIGO_DIR = Path(__file__).resolve().parents[2]
if str(CODIGO_DIR) not in sys.path:
    sys.path.insert(0, str(CODIGO_DIR))

from comun.rutas_proyecto import DATOS_CWRU_PROCESADO, DATOS_CWRU_RAW

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt


# ============================================================
# CONFIGURACIÓN
# ============================================================

PROCESSED_DIR = DATOS_CWRU_PROCESADO
DATASET_DIR = DATOS_CWRU_RAW

FS_ORIGINAL = 48_000
FS_TARGET = 2_000

DECIMATION_FACTOR = 24

WINDOW_SIZE = 1024
OVERLAP = 0.50
HOP_SIZE = int(WINDOW_SIZE * (1 - OVERLAP))

CUTOFF_HZ = 800
FILTER_ORDER = 6

# Archivo usado para estudiar detalladamente
# Puedes cambiarlo por 98, 99, 100, 109, etc.
ANALYSIS_FILE = 97

# Ventana que queremos analizar dentro del archivo
WINDOW_TO_ANALYZE = 5


# Etiquetas
LABEL_NAMES = {
    0: "Normal",
    1: "Inner Race",
    2: "Outer Race"
}


# ============================================================
# FUNCIONES
# ============================================================

def load_processed_dataset():
    """
    Carga X, y y metadata.
    """

    X = np.load(PROCESSED_DIR / "X.npy")
    y = np.load(PROCESSED_DIR / "y.npy")

    metadata_path = PROCESSED_DIR / "metadata.csv"

    if metadata_path.exists():
        metadata = pd.read_csv(metadata_path)
    else:
        metadata = None

    return X, y, metadata


def find_signal_key(mat_data, filepath):
    """
    Busca exactamente Xxxx_DE_time según el número del archivo.
    """

    file_number = Path(filepath).stem

    expected_key = f"X{int(file_number):03d}_DE_time"

    if expected_key not in mat_data:
        available = [
            key for key in mat_data.keys()
            if "_DE_time" in key
        ]

        raise KeyError(
            f"No se encontró {expected_key} en {filepath}.\n"
            f"Disponibles: {available}"
        )

    return expected_key


def load_original_signal(file_number):
    """
    Busca el archivo .mat dentro de las carpetas del dataset.
    """

    possible_dirs = [
        DATASET_DIR / "Normal",
        DATASET_DIR / "normal",
        DATASET_DIR / "IR",
        DATASET_DIR / "OR"
    ]

    filename = f"{file_number}.mat"

    filepath = None

    for directory in possible_dirs:

        candidate = directory / filename

        if candidate.exists():
            filepath = candidate
            break

    if filepath is None:

        raise FileNotFoundError(
            f"No se encontró {filename} dentro de dataset/"
        )

    mat_data = loadmat(filepath)

    key = find_signal_key(
        mat_data,
        filepath
    )

    signal = np.asarray(
        mat_data[key],
        dtype=np.float64
    ).squeeze()

    return signal, filepath, key


def apply_filter(signal):
    """
    Aplica el mismo filtro utilizado durante el
    preprocesamiento.
    """

    sos = butter(
        FILTER_ORDER,
        CUTOFF_HZ,
        btype="lowpass",
        fs=FS_ORIGINAL,
        output="sos"
    )

    return sosfiltfilt(
        sos,
        signal
    )


def decimate(signal):
    """
    Diezmado x24:
        48 kHz -> 2 kHz
    """

    return signal[::DECIMATION_FACTOR]


def fft_signal(signal, fs):
    """
    FFT de un solo lado.
    """

    n = len(signal)

    # Quitar componente DC para visualizar mejor
    signal = signal - np.mean(signal)

    spectrum = np.fft.rfft(signal)

    frequencies = np.fft.rfftfreq(
        n,
        d=1 / fs
    )

    magnitude = np.abs(spectrum) / n

    return frequencies, magnitude


def find_processed_window(
    metadata,
    file_number,
    window_number
):
    """
    Busca una ventana concreta dentro de X.npy.
    """

    if metadata is None:
        raise RuntimeError(
            "No se encontró metadata.csv"
        )

    filename = f"{file_number}.mat"

    mask = (
        (metadata["file"] == filename) &
        (metadata["window"] == window_number)
    )

    indexes = np.where(mask)[0]

    if len(indexes) == 0:

        raise ValueError(
            f"No se encontró la ventana "
            f"{window_number} de {filename}"
        )

    return indexes[0]


# ============================================================
# ANÁLISIS 1
# VENTANA REPRESENTATIVA POR CLASE
# ============================================================

def plot_one_window_per_class(X, y, metadata):
    """
    Muestra una ventana representativa de cada clase.
    """

    print("\n")
    print("=" * 70)
    print("ANÁLISIS DE VENTANAS A 2 kHz")
    print("=" * 70)

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(12, 9),
        sharex=True
    )

    time = np.arange(WINDOW_SIZE) / FS_TARGET

    for label in [0, 1, 2]:

        indices = np.where(y == label)[0]

        if len(indices) == 0:
            continue

        # Seleccionamos una ventana central de la clase
        index = indices[len(indices) // 2]

        signal = X[index]

        axes[label].plot(
            time,
            signal
        )

        axes[label].set_title(
            f"{LABEL_NAMES[label]} "
            f"| índice {index}"
        )

        axes[label].set_ylabel(
            "Amplitud normalizada"
        )

        axes[label].grid(
            True,
            alpha=0.3
        )

    axes[-1].set_xlabel(
        "Tiempo [s]"
    )

    fig.suptitle(
        "Ventanas de 1024 muestras a 2 kHz",
        fontsize=14
    )

    plt.tight_layout()
    plt.show()


# ============================================================
# ANÁLISIS 2
# FFT POR CLASE
# ============================================================

def plot_fft_one_window_per_class(X, y):
    """
    FFT de una ventana representativa de cada clase.
    """

    print("\n")
    print("=" * 70)
    print("ANÁLISIS ESPECTRAL")
    print("=" * 70)

    plt.figure(
        figsize=(12, 6)
    )

    for label in [0, 1, 2]:

        indices = np.where(y == label)[0]

        if len(indices) == 0:
            continue

        index = indices[len(indices) // 2]

        signal = X[index]

        frequencies, magnitude = fft_signal(
            signal,
            FS_TARGET
        )

        plt.plot(
            frequencies,
            magnitude,
            label=LABEL_NAMES[label]
        )

    plt.xlim(
        0,
        FS_TARGET / 2
    )

    plt.xlabel(
        "Frecuencia [Hz]"
    )

    plt.ylabel(
        "Magnitud"
    )

    plt.title(
        "FFT de una ventana representativa por clase"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()
    plt.show()


# ============================================================
# ANÁLISIS 3
# 48 kHz -> FILTRO -> 2 kHz
# ============================================================

def analyze_downsampling(file_number):
    """
    Compara la señal original y procesada.
    """

    print("\n")
    print("=" * 70)
    print("ANÁLISIS DEL DIEZMADO")
    print("=" * 70)

    signal_original, filepath, key = load_original_signal(
        file_number
    )

    print(f"Archivo       : {filepath}")
    print(f"Variable      : {key}")
    print(
        f"Muestras orig.: {len(signal_original):,}"
    )

    # --------------------------------------------------------
    # Filtrado
    # --------------------------------------------------------

    signal_filtered = apply_filter(
        signal_original
    )

    # --------------------------------------------------------
    # Diezmado
    # --------------------------------------------------------

    signal_2k = decimate(
        signal_filtered
    )

    print(
        f"Muestras 2 kHz: {len(signal_2k):,}"
    )

    # ========================================================
    # TIEMPO
    # ========================================================

    # Mostramos solamente los primeros 0.1 s
    duration = 0.1

    n_original = int(
        duration * FS_ORIGINAL
    )

    n_2k = int(
        duration * FS_TARGET
    )

    t_original = (
        np.arange(n_original)
        / FS_ORIGINAL
    )

    t_2k = (
        np.arange(n_2k)
        / FS_TARGET
    )

    plt.figure(
        figsize=(12, 8)
    )

    plt.plot(
        t_original,
        signal_original[:n_original],
        label="48 kHz original",
        alpha=0.6
    )

    plt.plot(
        t_original,
        signal_filtered[:n_original],
        label="Después LPF 800 Hz"
    )

    plt.plot(
        t_2k,
        signal_2k[:n_2k],
        ".-",
        label="Después de diezmado 2 kHz"
    )

    plt.xlabel(
        "Tiempo [s]"
    )

    plt.ylabel(
        "Amplitud"
    )

    plt.title(
        f"Comparación temporal - archivo {file_number}.mat"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    plt.show()

    # ========================================================
    # ESPECTRO
    # ========================================================

    # Para que la comparación sea más útil,
    # usamos una parte de la señal original.

    N = min(
        len(signal_original),
        48000
    )

    original_segment = (
        signal_original[:N]
    )

    filtered_segment = (
        signal_filtered[:N]
    )

    # Segmento equivalente después del diezmado
    reduced_segment = signal_2k[:N // 24]

    f_original, mag_original = fft_signal(
        original_segment,
        FS_ORIGINAL
    )

    f_filtered, mag_filtered = fft_signal(
        filtered_segment,
        FS_ORIGINAL
    )

    f_2k, mag_2k = fft_signal(
        reduced_segment,
        FS_TARGET
    )

    # --------------------------------------------------------
    # Espectro original
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12, 9)
    )

    axes[0].plot(
        f_original,
        mag_original
    )

    axes[0].set_xlim(
        0,
        6000
    )

    axes[0].set_title(
        "Espectro original - 48 kHz"
    )

    axes[0].set_xlabel(
        "Frecuencia [Hz]"
    )

    axes[0].set_ylabel(
        "Magnitud"
    )

    axes[0].grid(
        True,
        alpha=0.3
    )

    # --------------------------------------------------------
    # Espectro después de LPF
    # --------------------------------------------------------

    axes[1].plot(
        f_filtered,
        mag_filtered
    )

    axes[1].set_xlim(
        0,
        6000
    )

    axes[1].set_title(
        "Espectro después del LPF de 800 Hz"
    )

    axes[1].set_xlabel(
        "Frecuencia [Hz]"
    )

    axes[1].set_ylabel(
        "Magnitud"
    )

    axes[1].grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------
    # Espectro final a 2 kHz
    # --------------------------------------------------------

    plt.figure(
        figsize=(12, 6)
    )

    plt.plot(
        f_2k,
        mag_2k
    )

    plt.xlim(
        0,
        1000
    )

    plt.xlabel(
        "Frecuencia [Hz]"
    )

    plt.ylabel(
        "Magnitud"
    )

    plt.title(
        "Espectro final después del diezmado - 2 kHz"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    plt.show()


# ============================================================
# ANÁLISIS 4
# INFORMACIÓN ESTADÍSTICA
# ============================================================

def dataset_statistics(X, y, metadata):
    """
    Estadísticas simples del dataset.
    """

    print("\n")
    print("=" * 70)
    print("ESTADÍSTICAS DEL DATASET")
    print("=" * 70)

    print(
        f"Número total de ventanas: {len(X)}"
    )

    print(
        f"Tamaño de cada ventana   : {X.shape[1]}"
    )

    print(
        f"Frecuencia de muestreo   : {FS_TARGET} Hz"
    )

    print(
        f"Duración por ventana     : "
        f"{WINDOW_SIZE / FS_TARGET:.3f} s"
    )

    print(
        f"Hop                      : {HOP_SIZE}"
    )

    print()

    for label in [0, 1, 2]:

        indices = np.where(
            y == label
        )[0]

        subset = X[indices]

        rms = np.sqrt(
            np.mean(subset ** 2)
        )

        std = np.std(
            subset
        )

        print(
            f"{LABEL_NAMES[label]:12s}: "
            f"{len(indices):4d} ventanas | "
            f"RMS promedio = {rms:.4f} | "
            f"STD = {std:.4f}"
        )

    if metadata is not None:

        print("\nVentanas por archivo:")

        counts = (
            metadata
            .groupby(["file", "class"])
            .size()
            .reset_index(name="windows")
        )

        print(
            counts.to_string(
                index=False
            )
        )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Cargar dataset procesado
    # --------------------------------------------------------

    X, y, metadata = load_processed_dataset()

    print(
        f"X shape: {X.shape}"
    )

    print(
        f"y shape: {y.shape}"
    )

    # --------------------------------------------------------
    # Estadísticas
    # --------------------------------------------------------

    dataset_statistics(
        X,
        y,
        metadata
    )

    # --------------------------------------------------------
    # Ventanas de las tres clases
    # --------------------------------------------------------

    plot_one_window_per_class(
        X,
        y,
        metadata
    )

    # --------------------------------------------------------
    # FFT
    # --------------------------------------------------------

    plot_fft_one_window_per_class(
        X,
        y
    )

    # --------------------------------------------------------
    # Análisis detallado de 48 kHz -> 2 kHz
    # --------------------------------------------------------

    analyze_downsampling(
        ANALYSIS_FILE
    )


if __name__ == "__main__":
    main()
