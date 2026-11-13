"""
PREPROCESAMIENTO CWRU 48 kHz PARA TINYML

Convención de nombres:
    <ID>_<CLASE>_<DIAMETRO>.mat

Ejemplos:
    97_N_0,000.mat
    109_IR_0,007.mat
    174_IR_0,014.mat
    213_IR_0,021.mat
    135_OR_0,007.mat

Clases:
    N  = Normal
    IR = Inner Race
    OR = Outer Race

Pipeline:
    48 kHz
      ↓
    LPF Butterworth 800 Hz
      ↓
    Decimación x24
      ↓
    2 kHz
      ↓
    Ventanas de 1024 muestras
      ↓
    50 % overlap
      ↓
    Z-score por ventana

Salida:
    processed/
        X.npy
        y.npy
        metadata.csv
        config.txt
"""

from pathlib import Path
import re
import sys

CODIGO_DIR = Path(__file__).resolve().parents[2]
if str(CODIGO_DIR) not in sys.path:
    sys.path.insert(0, str(CODIGO_DIR))

from comun.rutas_proyecto import DATOS_CWRU_PROCESADO, DATOS_CWRU_RAW

import numpy as np
import pandas as pd

from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt


# ============================================================
# CONFIGURACIÓN
# ============================================================

INPUT_DIR = DATOS_CWRU_RAW
OUTPUT_DIR = DATOS_CWRU_PROCESADO

FS_ORIGINAL = 48_000
FS_TARGET = 2_000

DECIMATION_FACTOR = FS_ORIGINAL // FS_TARGET

WINDOW_SIZE = 1024
OVERLAP = 0.50
HOP_SIZE = int(WINDOW_SIZE * (1 - OVERLAP))

CUTOFF_HZ = 800
FILTER_ORDER = 6


# ============================================================
# CLASES
# ============================================================

CLASS_LABELS = {
    "N": 0,
    "IR": 1,
    "OR": 2,
}


# ============================================================
# HP Y RPM SEGÚN TABLA CWRU 48 kHz
# ============================================================

LOAD_TO_RPM = {
    0: 1797,
    1: 1772,
    2: 1750,
    3: 1730,
}


# ============================================================
# MAPEO FILE_ID -> CARGA HP
#
# Completa esta tabla con TODOS los IDs que descargues.
# Los primeros registros conocidos se dejan incluidos.
# ============================================================

FILE_ID_TO_HP = {

    # -------------------------
    # NORMAL
    # -------------------------
    97: 0,
    98: 1,
    99: 2,
    100: 3,

    # -------------------------
    # IR 0.007
    # -------------------------
    109: 0,
    110: 1,
    111: 2,
    112: 3,

    # -------------------------
    # IR 0.014 
    # -------------------------
    174: 0,
    175: 1,
    176: 2,
    177: 3,

    # -------------------------
    # IR 0.021 
    # -------------------------
    213: 0,
    214: 1,
    215: 2,
    217: 3,

    # -------------------------
    # OR 0.007 Centered @6:00
    # -------------------------
    135: 0,
    136: 1,
    137: 2,
    138: 3,

    # -------------------------
    # OR 0.014 Centered @6:00
    # -------------------------
    201: 0,
    202: 1,
    203: 2,
    204: 3,

    # -------------------------
    # OR 0.007 Centered @6:00
    # -------------------------
    238: 0,
    239: 1,
    240: 2,
    241: 3,
}


# ============================================================
# PARSER DEL NOMBRE
# ============================================================

def parse_filename(filepath):
    """
    Espera nombres como:

        109_IR_0,007.mat
        135_OR_0,007.mat
        97_N_0,000.mat

    Retorna:
        file_id
        class_name
        label
        diameter_in
    """

    stem = filepath.stem

    pattern = r"^(\d+)_(N|IR|OR)_([0-9]+,[0-9]+)$"

    match = re.match(
        pattern,
        stem,
        flags=re.IGNORECASE
    )

    if not match:
        raise ValueError(
            f"\nNombre inválido: {filepath.name}\n"
            f"Formato esperado:\n"
            f"<ID>_<N|IR|OR>_<DIAMETRO>.mat\n"
            f"Ejemplo: 109_IR_0,007.mat"
        )

    file_id = int(
        match.group(1)
    )

    class_name = (
        match.group(2)
        .upper()
    )

    diameter_text = (
        match.group(3)
        .replace(",", ".")
    )

    diameter_in = float(
        diameter_text
    )

    label = CLASS_LABELS[
        class_name
    ]

    return (
        file_id,
        class_name,
        label,
        diameter_in
    )


# ============================================================
# HP Y RPM
# ============================================================

def get_operating_conditions(file_id):
    """
    Obtiene HP y RPM usando el ID del archivo.

    RPM se deriva de la tabla CWRU:
        0 HP -> 1797 rpm
        1 HP -> 1772 rpm
        2 HP -> 1750 rpm
        3 HP -> 1730 rpm
    """

    if file_id not in FILE_ID_TO_HP:

        raise KeyError(
            f"\nEl archivo ID {file_id} no está registrado "
            f"en FILE_ID_TO_HP.\n"
            f"Agrega su carga HP antes de continuar."
        )

    load_hp = FILE_ID_TO_HP[
        file_id
    ]

    rpm = LOAD_TO_RPM[
        load_hp
    ]

    return load_hp, rpm


# ============================================================
# BUSCAR VARIABLE DE_time
# ============================================================

def find_signal_key(mat_data, file_id):
    """
    Busca la señal Drive End.

    Primero intenta encontrar:
        X<file_id>_DE_time

    Si no existe, busca todas las variables *_DE_time.

    Si existe exactamente una, la utiliza y genera
    una advertencia indicando que el ID interno de CWRU
    no coincide con el nombre del archivo.
    """

    expected_key = f"X{file_id:03d}_DE_time"

    # Caso ideal
    if expected_key in mat_data:
        return expected_key

    # Buscar variables Drive End disponibles
    available = [
        key for key in mat_data.keys()
        if "_DE_time" in key
    ]

    # Si solo existe una señal DE, utilizarla
    if len(available) == 1:

        signal_key = available[0]

        print(
            f"  [ADVERTENCIA CWRU]\n"
            f"  Archivo ID        : {file_id}\n"
            f"  Variable esperada : {expected_key}\n"
            f"  Variable encontrada: {signal_key}\n"
            f"  Se utilizará {signal_key}."
        )

        return signal_key

    # Si no hay ninguna
    if len(available) == 0:
        raise KeyError(
            f"\nNo se encontró ninguna variable *_DE_time "
            f"en el archivo ID {file_id}."
        )

    # Si hay varias, no podemos elegir automáticamente
    raise KeyError(
        f"\nNo se encontró {expected_key}.\n"
        f"Hay varias señales DE disponibles:\n"
        f"{available}\n"
        f"No se puede determinar automáticamente cuál usar."
    )


# ============================================================
# CARGA DE SEÑAL
# ============================================================

def load_signal(filepath, file_id):

    mat_data = loadmat(
        filepath
    )

    signal_key = find_signal_key(
        mat_data,
        file_id
    )

    signal = np.asarray(
        mat_data[signal_key],
        dtype=np.float64
    ).squeeze()

    if signal.ndim != 1:
        raise ValueError(
            f"La señal {signal_key} "
            f"no es un vector 1D."
        )

    return signal, signal_key


# ============================================================
# FILTRO ANTI-ALIAS
# ============================================================

def anti_alias_filter(signal):

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

def downsample_signal(signal):

    return signal[
        ::DECIMATION_FACTOR
    ]


# ============================================================
# VENTANAS
# ============================================================

def create_windows(signal):

    if len(signal) < WINDOW_SIZE:

        return np.empty(
            (0, WINDOW_SIZE),
            dtype=np.float32
        )

    num_windows = (
        (len(signal) - WINDOW_SIZE)
        // HOP_SIZE
    ) + 1

    windows = []

    for i in range(
        num_windows
    ):

        start = (
            i * HOP_SIZE
        )

        end = (
            start + WINDOW_SIZE
        )

        window = signal[
            start:end
        ]

        windows.append(
            window
        )

    return np.asarray(
        windows,
        dtype=np.float32
    )


# ============================================================
# NORMALIZACIÓN
# ============================================================

def normalize_windows(windows):
    """
    Z-score por ventana:

        x_norm = (x - mean) / std
    """

    mean = np.mean(
        windows,
        axis=1,
        keepdims=True
    )

    std = np.std(
        windows,
        axis=1,
        keepdims=True
    )

    std = np.maximum(
        std,
        1e-8
    )

    normalized = (
        windows - mean
    ) / std

    return normalized.astype(
        np.float32
    )


# ============================================================
# PROCESAMIENTO PRINCIPAL
# ============================================================

def process_dataset():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    all_windows = []
    all_labels = []
    metadata = []

    print("=" * 80)
    print("PREPROCESAMIENTO CWRU 48 kHz - TinyML")
    print("=" * 80)

    print(
        f"Frecuencia original : {FS_ORIGINAL} Hz"
    )

    print(
        f"Frecuencia objetivo : {FS_TARGET} Hz"
    )

    print(
        f"Decimación          : x{DECIMATION_FACTOR}"
    )

    print(
        f"LPF                 : {CUTOFF_HZ} Hz"
    )

    print(
        f"Ventana             : {WINDOW_SIZE}"
    )

    print(
        f"Overlap             : {OVERLAP * 100:.0f}%"
    )

    print(
        f"Hop                 : {HOP_SIZE}"
    )

    print("=" * 80)


    # ========================================================
    # Buscar archivos en las tres carpetas
    # ========================================================

    possible_dirs = [
        INPUT_DIR / "Normal",
        INPUT_DIR / "normal",
        INPUT_DIR / "IR",
        INPUT_DIR / "OR"
    ]


    files = []

    for directory in possible_dirs:

        if directory.exists():

            files.extend(
                directory.glob("*.mat")
            )


    files = sorted(
        set(files),
        key=lambda p: p.name
    )


    if not files:

        raise RuntimeError(
            "No se encontraron archivos .mat."
        )


    print(
        f"\nArchivos encontrados: {len(files)}"
    )


    # ========================================================
    # PROCESAR ARCHIVOS
    # ========================================================

    for filepath in files:

        print("\n" + "-" * 80)

        print(
            f"Procesando: {filepath.name}"
        )

        # ----------------------------------------------------
        # Nombre
        # ----------------------------------------------------

        (
            file_id,
            class_name,
            label,
            diameter_in
        ) = parse_filename(
            filepath
        )


        # ----------------------------------------------------
        # HP y RPM
        # ----------------------------------------------------

        load_hp, rpm = (
            get_operating_conditions(
                file_id
            )
        )


        # ----------------------------------------------------
        # Posición Outer Race
        # ----------------------------------------------------

        if class_name == "OR":

            or_location = "Centered"
            or_position = "6:00"

        else:

            or_location = ""
            or_position = ""


        print(
            f"  ID              : {file_id}"
        )

        print(
            f"  Clase           : {class_name}"
        )

        print(
            f"  Label           : {label}"
        )

        print(
            f"  Diámetro        : {diameter_in:.3f}\""
        )

        print(
            f"  Carga           : {load_hp} HP"
        )

        print(
            f"  RPM             : {rpm}"
        )

        if class_name == "OR":

            print(
                f"  OR location     : "
                f"{or_location} @{or_position}"
            )


        # ----------------------------------------------------
        # Cargar señal
        # ----------------------------------------------------

        signal, signal_key = (
            load_signal(
                filepath,
                file_id
            )
        )


        print(
            f"  Variable        : {signal_key}"
        )

        print(
            f"  Muestras 48 kHz : {len(signal):,}"
        )


        # ----------------------------------------------------
        # Filtrado
        # ----------------------------------------------------

        filtered = (
            anti_alias_filter(
                signal
            )
        )


        # ----------------------------------------------------
        # Diezmado
        # ----------------------------------------------------

        signal_2k = (
            downsample_signal(
                filtered
            )
        )


        print(
            f"  Muestras 2 kHz  : {len(signal_2k):,}"
        )


        # ----------------------------------------------------
        # Ventanas
        # ----------------------------------------------------

        windows = (
            create_windows(
                signal_2k
            )
        )


        print(
            f"  Ventanas        : {len(windows)}"
        )


        # ----------------------------------------------------
        # Normalización
        # ----------------------------------------------------

        windows = (
            normalize_windows(
                windows
            )
        )


        # ----------------------------------------------------
        # Guardar
        # ----------------------------------------------------

        labels = np.full(
            len(windows),
            label,
            dtype=np.int64
        )


        all_windows.append(
            windows
        )

        all_labels.append(
            labels
        )


        # ----------------------------------------------------
        # Metadata por ventana
        # ----------------------------------------------------

        for window_id in range(
            len(windows)
        ):

            start_sample_2k = (
                window_id
                * HOP_SIZE
            )

            start_time_s = (
                start_sample_2k
                / FS_TARGET
            )

            metadata.append({

                "file":
                    filepath.name,

                "file_id":
                    file_id,

                "signal_key":
                    signal_key,

                "class":
                    class_name,

                "label":
                    label,

                "diameter_in":
                    diameter_in,

                "load_hp":
                    load_hp,

                "rpm":
                    rpm,

                "or_location":
                    or_location,

                "or_position":
                    or_position,

                "window":
                    window_id,

                "start_sample_2k":
                    start_sample_2k,

                "start_time_s":
                    start_time_s,

                "fs_original":
                    FS_ORIGINAL,

                "fs_target":
                    FS_TARGET,

                "decimation_factor":
                    DECIMATION_FACTOR,

                "cutoff_hz":
                    CUTOFF_HZ,

                "filter_order":
                    FILTER_ORDER,

                "window_size":
                    WINDOW_SIZE,

                "overlap":
                    OVERLAP,

                "hop_size":
                    HOP_SIZE,

                "normalization":
                    "zscore_per_window"
            })


    # ========================================================
    # UNIR DATASET
    # ========================================================

    X = np.concatenate(
        all_windows,
        axis=0
    )

    y = np.concatenate(
        all_labels,
        axis=0
    )

    metadata_df = pd.DataFrame(
        metadata
    )


    # ========================================================
    # GUARDAR
    # ========================================================

    np.save(
        OUTPUT_DIR / "X.npy",
        X
    )

    np.save(
        OUTPUT_DIR / "y.npy",
        y
    )

    metadata_df.to_csv(
        OUTPUT_DIR / "metadata.csv",
        index=False
    )


    # ========================================================
    # CONFIG
    # ========================================================

    with open(
        OUTPUT_DIR / "config.txt",
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "CWRU TinyML preprocessing\n"
        )

        f.write(
            f"Original sampling rate: "
            f"{FS_ORIGINAL} Hz\n"
        )

        f.write(
            f"Target sampling rate: "
            f"{FS_TARGET} Hz\n"
        )

        f.write(
            f"Decimation factor: "
            f"{DECIMATION_FACTOR}\n"
        )

        f.write(
            f"LPF cutoff: "
            f"{CUTOFF_HZ} Hz\n"
        )

        f.write(
            f"Filter order: "
            f"{FILTER_ORDER}\n"
        )

        f.write(
            f"Window size: "
            f"{WINDOW_SIZE}\n"
        )

        f.write(
            f"Overlap: "
            f"{OVERLAP}\n"
        )

        f.write(
            f"Hop size: "
            f"{HOP_SIZE}\n"
        )

        f.write(
            "Normalization: "
            "Z-score per window\n"
        )

        f.write(
            "Outer Race position: "
            "Centered @6:00\n"
        )


    # ========================================================
    # RESUMEN
    # ========================================================

    print("\n")
    print("=" * 80)
    print("RESULTADO FINAL")
    print("=" * 80)


    print(
        f"X shape: {X.shape}"
    )

    print(
        f"y shape: {y.shape}"
    )


    print("\nVentanas por clase:")

    class_summary = (
        metadata_df
        .groupby(
            ["class"]
        )
        .size()
        .reset_index(
            name="windows"
        )
    )

    print(
        class_summary.to_string(
            index=False
        )
    )


    print(
        "\nVentanas por diámetro y clase:"
    )

    diameter_summary = (
        metadata_df
        .groupby(
            [
                "class",
                "diameter_in"
            ]
        )
        .size()
        .reset_index(
            name="windows"
        )
    )

    print(
        diameter_summary.to_string(
            index=False
        )
    )


    print(
        "\nVentanas por carga:"
    )

    load_summary = (
        metadata_df
        .groupby(
            [
                "load_hp",
                "rpm"
            ]
        )
        .size()
        .reset_index(
            name="windows"
        )
    )

    print(
        load_summary.to_string(
            index=False
        )
    )


    print(
        "\nDetalle por archivo:"
    )

    file_summary = (
        metadata_df
        .groupby(
            [
                "file",
                "class",
                "diameter_in",
                "load_hp",
                "rpm"
            ]
        )
        .size()
        .reset_index(
            name="windows"
        )
    )

    print(
        file_summary.to_string(
            index=False
        )
    )


    print(
        f"\nDataset guardado en:\n"
        f"{OUTPUT_DIR.resolve()}"
    )

    print("=" * 80)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    process_dataset()
