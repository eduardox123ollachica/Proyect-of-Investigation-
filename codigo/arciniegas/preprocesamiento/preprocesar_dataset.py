from pathlib import Path
import json
import math
import sys
from collections import Counter

CODIGO_DIR = Path(__file__).resolve().parents[2]
if str(CODIGO_DIR) not in sys.path:
    sys.path.insert(0, str(CODIGO_DIR))

from comun.rutas_proyecto import DATOS_ARCINIEGAS_PROCESADO, DATOS_ARCINIEGAS_RAW

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis


# ============================================================
# CONFIGURACION
# ============================================================

# Carpeta original
BASE_DIR = DATOS_ARCINIEGAS_RAW

TRAIN_DIR = BASE_DIR / "training"
TEST_DIR = BASE_DIR / "testing"

# Carpeta de salida separada de los datos originales.
OUTPUT_DIR = DATOS_ARCINIEGAS_PROCESADO

# Parametros del articulo
EXPECTED_FS = 100.0
FFT_SIZE = 16
OVERLAP = 0.50
HOP_SIZE = int(FFT_SIZE * (1 - OVERLAP))  # 8 muestras

# No se aplica normalizacion aqui.
# Se guardan features crudas para poder decidir despues
# si se replica Edge Impulse tal cual o se estandariza
# usando exclusivamente estadisticas de training.
NORMALIZE_FEATURES = False

CLASS_TO_LABEL = {
    "DETENIDO": 0,
    "VELOCIDAD MEDIA": 1,
    "VELOCIDAD ALTA": 2,
}

LABEL_TO_CLASS = {
    0: "DETENIDO",
    1: "VELOCIDAD MEDIA",
    2: "VELOCIDAD ALTA",
}


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def infer_label_from_filename(filename: str) -> str:
    name = filename.upper()

    if "DETENIDO" in name:
        return "DETENIDO"
    elif "VELOCIDAD MEDIA" in name:
        return "VELOCIDAD MEDIA"
    elif "VELOCIDAD ALTA" in name:
        return "VELOCIDAD ALTA"

    raise ValueError(f"No se pudo inferir la clase del archivo: {filename}")


def spectral_entropy(power: np.ndarray) -> float:
    """
    Entropia espectral normalizada en [0, 1].
    """
    power = np.asarray(power, dtype=np.float64)

    total = np.sum(power)

    if total <= 0:
        return 0.0

    p = power / total
    p = p[p > 0]

    if len(p) <= 1:
        return 0.0

    entropy = -np.sum(p * np.log2(p))
    max_entropy = np.log2(len(power))

    if max_entropy <= 0:
        return 0.0

    return float(entropy / max_entropy)


def extract_axis_features(signal: np.ndarray, fs: float) -> dict:
    """
    Extrae features temporales/espectrales de una ventana de 16 muestras.

    Features elegidas a partir de las variables mencionadas en el articulo:
    - RMS
    - skewness
    - kurtosis
    - spectral energy
    - dominant frequency
    - dominant amplitude
    - spectral entropy

    Para la FFT se elimina la media de la ventana antes de calcular
    el espectro, reduciendo la componente DC/gravedad.
    """

    signal = np.asarray(signal, dtype=np.float64)

    # -----------------------------
    # Tiempo
    # -----------------------------
    rms_value = float(np.sqrt(np.mean(signal ** 2)))

    skew_value = float(skew(signal, bias=False))
    kurt_value = float(kurtosis(signal, fisher=True, bias=False))

    if not np.isfinite(skew_value):
        skew_value = 0.0

    if not np.isfinite(kurt_value):
        kurt_value = 0.0

    # -----------------------------
    # Frecuencia
    # -----------------------------
    # Remover DC de la ventana
    x_ac = signal - np.mean(signal)

    spectrum = np.fft.rfft(x_ac, n=FFT_SIZE)
    magnitude = np.abs(spectrum)
    power = magnitude ** 2

    freqs = np.fft.rfftfreq(
        FFT_SIZE,
        d=1.0 / fs
    )

    spectral_energy_value = float(np.sum(power))

    # Ignorar bin DC al buscar frecuencia dominante
    if len(magnitude) > 1:
        idx = int(np.argmax(magnitude[1:]) + 1)
    else:
        idx = 0

    dominant_frequency = float(freqs[idx])
    dominant_amplitude = float(magnitude[idx])

    spectral_entropy_value = spectral_entropy(power)

    return {
        "rms": rms_value,
        "skewness": skew_value,
        "kurtosis": kurt_value,
        "spectral_energy": spectral_energy_value,
        "dominant_frequency": dominant_frequency,
        "dominant_amplitude": dominant_amplitude,
        "spectral_entropy": spectral_entropy_value,
    }


def build_feature_names():
    features = [
        "rms",
        "skewness",
        "kurtosis",
        "spectral_energy",
        "dominant_frequency",
        "dominant_amplitude",
        "spectral_entropy",
    ]

    names = []

    for axis in ["ax", "ay", "az"]:
        for feature in features:
            names.append(f"{axis}_{feature}")

    return names


FEATURE_NAMES = build_feature_names()


# ============================================================
# LECTURA DE JSON
# ============================================================

def load_json_file(filepath: Path):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "payload" not in data:
        raise ValueError("No existe 'payload'")

    payload = data["payload"]

    interval_ms = payload.get("interval_ms")

    if interval_ms is None or interval_ms <= 0:
        raise ValueError("interval_ms invalido")

    fs = 1000.0 / float(interval_ms)

    if not math.isclose(fs, EXPECTED_FS, rel_tol=1e-9):
        raise ValueError(
            f"Frecuencia inesperada: {fs} Hz "
            f"(se esperaba {EXPECTED_FS} Hz)"
        )

    sensors = [
        sensor.get("name", "")
        for sensor in payload.get("sensors", [])
    ]

    expected_sensors = ["ax", "ay", "az"]

    if sensors != expected_sensors:
        raise ValueError(
            f"Sensores encontrados: {sensors}. "
            f"Se esperaban {expected_sensors}"
        )

    values = np.asarray(
        payload.get("values", []),
        dtype=np.float64
    )

    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError(
            f"Forma invalida de values: {values.shape}"
        )

    if np.isnan(values).any() or np.isinf(values).any():
        raise ValueError(
            "El archivo contiene NaN o Inf"
        )

    return values, fs


# ============================================================
# SEGMENTACION Y EXTRACCION DE FEATURES
# ============================================================

def process_capture(values, fs, source_file, split, class_name):
    """
    Convierte una captura de 300x3 en ventanas de 16 muestras
    con 50% de overlap.

    Con 300 muestras:
        floor((300 - 16) / 8) + 1 = 36 ventanas
    """

    rows = []
    X_rows = []
    y_rows = []

    n_samples = values.shape[0]

    window_index = 0

    for start in range(
        0,
        n_samples - FFT_SIZE + 1,
        HOP_SIZE
    ):
        end = start + FFT_SIZE

        window = values[start:end, :]

        feature_vector = []

        per_axis = {}

        for axis_index, axis_name in enumerate(
            ["ax", "ay", "az"]
        ):
            features = extract_axis_features(
                window[:, axis_index],
                fs
            )

            per_axis[axis_name] = features

            feature_vector.extend([
                features["rms"],
                features["skewness"],
                features["kurtosis"],
                features["spectral_energy"],
                features["dominant_frequency"],
                features["dominant_amplitude"],
                features["spectral_entropy"],
            ])

        label = CLASS_TO_LABEL[class_name]

        X_rows.append(feature_vector)
        y_rows.append(label)

        metadata = {
            "split": split,
            "file": source_file,
            "class": class_name,
            "label": label,
            "window": window_index,
            "start_sample": start,
            "end_sample_exclusive": end,
            "start_time_s": start / fs,
            "end_time_s": end / fs,
            "fs_hz": fs,
            "fft_size": FFT_SIZE,
            "overlap": OVERLAP,
            "hop_size": HOP_SIZE,
        }

        # Agregar features tambien al CSV para trazabilidad
        for axis_name in ["ax", "ay", "az"]:
            for feature_name, value in per_axis[axis_name].items():
                metadata[f"{axis_name}_{feature_name}"] = value

        rows.append(metadata)

        window_index += 1

    return (
        np.asarray(X_rows, dtype=np.float32),
        np.asarray(y_rows, dtype=np.int64),
        rows,
    )


# ============================================================
# PROCESAR SPLIT
# ============================================================

def process_split(folder: Path, split: str):
    json_files = sorted(folder.glob("*.json"))

    if len(json_files) == 0:
        raise FileNotFoundError(
            f"No se encontraron JSON en: {folder.resolve()}"
        )

    X_all = []
    y_all = []
    metadata_all = []

    capture_counter = Counter()
    window_counter = Counter()

    errors = []

    print("\n" + "=" * 80)
    print(f"PROCESANDO {split.upper()}")
    print("=" * 80)

    for idx, filepath in enumerate(json_files, start=1):
        try:
            class_name = infer_label_from_filename(
                filepath.name
            )

            values, fs = load_json_file(
                filepath
            )

            X_file, y_file, meta_file = process_capture(
                values=values,
                fs=fs,
                source_file=filepath.name,
                split=split,
                class_name=class_name,
            )

            X_all.append(X_file)
            y_all.append(y_file)
            metadata_all.extend(meta_file)

            capture_counter[class_name] += 1
            window_counter[class_name] += len(y_file)

            print(
                f"[{idx:03d}/{len(json_files):03d}] "
                f"{class_name:17s} | "
                f"{filepath.name} | "
                f"{len(y_file)} ventanas"
            )

        except Exception as e:
            errors.append({
                "split": split,
                "file": filepath.name,
                "error": str(e),
            })

            print(
                f"[ERROR] {filepath.name}: {e}"
            )

    if not X_all:
        raise RuntimeError(
            f"No se pudo procesar ningun archivo de {split}"
        )

    X_split = np.concatenate(
        X_all,
        axis=0
    )

    y_split = np.concatenate(
        y_all,
        axis=0
    )

    return (
        X_split,
        y_split,
        metadata_all,
        capture_counter,
        window_counter,
        errors,
    )


# ============================================================
# TXT DE DOCUMENTACION
# ============================================================

def write_info_txt(
    X_train,
    y_train,
    X_test,
    y_test,
    train_capture_counter,
    test_capture_counter,
    train_window_counter,
    test_window_counter,
    errors,
):
    info_path = OUTPUT_DIR / "preprocesamiento_info.txt"

    total_captures_train = sum(train_capture_counter.values())
    total_captures_test = sum(test_capture_counter.values())

    with open(info_path, "w", encoding="utf-8") as f:

        f.write(
            "PREPROCESAMIENTO DATASET ARCINIEGAS\n"
        )
        f.write("=" * 70 + "\n\n")

        f.write("1. OBJETIVO\n")
        f.write("-" * 70 + "\n")
        f.write(
            "Preparar el dataset utilizado en el articulo "
            "'IoT device for detecting abnormal vibrations in motors using TinyML' "
            "para una replica inicial en Python.\n\n"
        )

        f.write("2. ESTRUCTURA DE ENTRADA\n")
        f.write("-" * 70 + "\n")
        f.write(
            f"Carpeta original: {BASE_DIR}\n"
        )
        f.write(
            f"Training: {TRAIN_DIR}\n"
        )
        f.write(
            f"Testing: {TEST_DIR}\n\n"
        )

        f.write("3. CLASES\n")
        f.write("-" * 70 + "\n")

        for class_name, label in CLASS_TO_LABEL.items():
            f.write(
                f"{label}: {class_name}\n"
            )

        f.write("\n")

        f.write("4. DATOS CRUDOS\n")
        f.write("-" * 70 + "\n")
        f.write(
            "Cada JSON contiene aceleracion triaxial: ax, ay, az.\n"
        )
        f.write(
            f"Frecuencia de muestreo esperada: {EXPECTED_FS:.1f} Hz.\n"
        )
        f.write(
            "interval_ms esperado: 10 ms.\n"
        )
        f.write(
            "Las capturas auditadas tienen 300 muestras, equivalentes a 3 s.\n\n"
        )

        f.write("5. SEGMENTACION\n")
        f.write("-" * 70 + "\n")
        f.write(
            f"Tamano de ventana / FFT: {FFT_SIZE} muestras.\n"
        )
        f.write(
            f"Overlap: {OVERLAP*100:.0f}%.\n"
        )
        f.write(
            f"Hop: {HOP_SIZE} muestras.\n"
        )
        f.write(
            "Para un archivo de 300 muestras se generan 36 ventanas.\n"
        )
        f.write(
            "No se mezclan archivos entre training y testing: "
            "cada ventana conserva el split de su JSON de origen.\n\n"
        )

        f.write("6. FEATURES EXTRAIDAS\n")
        f.write("-" * 70 + "\n")
        f.write(
            "Se extraen 7 caracteristicas por cada eje (X, Y y Z), "
            "para un total de 21 features por ventana:\n"
        )
        f.write(
            "- RMS\n"
            "- Skewness\n"
            "- Kurtosis\n"
            "- Energia espectral\n"
            "- Frecuencia dominante\n"
            "- Amplitud dominante\n"
            "- Entropia espectral\n\n"
        )

        f.write(
            "Antes de la FFT se resta la media de cada ventana para "
            "reducir la componente DC, especialmente la gravedad en az. "
            "El RMS se calcula sobre la senal original de la ventana.\n\n"
        )

        f.write("7. FFT\n")
        f.write("-" * 70 + "\n")
        f.write(
            "Se utiliza np.fft.rfft con N=16 y fs=100 Hz.\n"
        )
        f.write(
            "La resolucion espectral es fs/N = 6.25 Hz por bin.\n"
        )
        f.write(
            "La frecuencia dominante se obtiene ignorando el bin DC.\n\n"
        )

        f.write("8. NORMALIZACION\n")
        f.write("-" * 70 + "\n")
        f.write(
            "En esta etapa NO se normalizan/estandarizan las features. "
            "Se guardan valores crudos para mantener trazabilidad. "
            "Si posteriormente se usa StandardScaler, debe ajustarse "
            "solo con training y aplicarse despues a testing.\n\n"
        )

        f.write("9. RESULTADOS DEL PREPROCESAMIENTO\n")
        f.write("-" * 70 + "\n")

        f.write(
            f"Capturas training: {total_captures_train}\n"
        )

        for cls in CLASS_TO_LABEL:
            f.write(
                f"  {cls}: "
                f"{train_capture_counter.get(cls, 0)} archivos, "
                f"{train_window_counter.get(cls, 0)} ventanas\n"
            )

        f.write(
            f"\nCapturas testing: {total_captures_test}\n"
        )

        for cls in CLASS_TO_LABEL:
            f.write(
                f"  {cls}: "
                f"{test_capture_counter.get(cls, 0)} archivos, "
                f"{test_window_counter.get(cls, 0)} ventanas\n"
            )

        f.write("\n")

        f.write(
            f"X_train shape: {X_train.shape}\n"
        )
        f.write(
            f"y_train shape: {y_train.shape}\n"
        )
        f.write(
            f"X_test shape: {X_test.shape}\n"
        )
        f.write(
            f"y_test shape: {y_test.shape}\n\n"
        )

        f.write("10. ARCHIVOS GENERADOS\n")
        f.write("-" * 70 + "\n")
        f.write(
            "X_train.npy\n"
            "y_train.npy\n"
            "X_test.npy\n"
            "y_test.npy\n"
            "metadata_features.csv\n"
            "feature_names.txt\n"
            "preprocesamiento_info.txt\n"
        )

        if errors:
            f.write("\n11. ERRORES / ARCHIVOS OMITIDOS\n")
            f.write("-" * 70 + "\n")

            for item in errors:
                f.write(
                    f"{item['split']} | "
                    f"{item['file']} | "
                    f"{item['error']}\n"
                )

        f.write("\n")
        f.write("NOTA METODOLOGICA\n")
        f.write("-" * 70 + "\n")
        f.write(
            "El articulo no especifica con suficiente detalle el vector exacto "
            "de features que Edge Impulse entrego a la DNN y presenta una "
            "inconsistencia al describir una entrada de 3 neuronas mientras "
            "tambien afirma utilizar features espectrales. Por ello, este "
            "preprocesamiento constituye una replica reproducible inspirada "
            "directamente en las caracteristicas mencionadas en el articulo, "
            "no una reconstruccion exacta del pipeline interno de Edge Impulse.\n"
        )

    return info_path


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 80)
    print("PREPROCESAMIENTO DATASET ARCINIEGAS")
    print("=" * 80)

    print(
        f"Dataset: {BASE_DIR.resolve()}"
    )

    print(
        f"Salida:  {OUTPUT_DIR.resolve()}"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------
    (
        X_train,
        y_train,
        train_metadata,
        train_capture_counter,
        train_window_counter,
        train_errors,
    ) = process_split(
        TRAIN_DIR,
        "training"
    )

    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------
    (
        X_test,
        y_test,
        test_metadata,
        test_capture_counter,
        test_window_counter,
        test_errors,
    ) = process_split(
        TEST_DIR,
        "testing"
    )

    # --------------------------------------------------------
    # VALIDACIONES
    # --------------------------------------------------------
    assert X_train.shape[1] == len(FEATURE_NAMES)
    assert X_test.shape[1] == len(FEATURE_NAMES)

    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)

    # --------------------------------------------------------
    # GUARDAR NPY
    # --------------------------------------------------------
    np.save(
        OUTPUT_DIR / "X_train.npy",
        X_train
    )

    np.save(
        OUTPUT_DIR / "y_train.npy",
        y_train
    )

    np.save(
        OUTPUT_DIR / "X_test.npy",
        X_test
    )

    np.save(
        OUTPUT_DIR / "y_test.npy",
        y_test
    )

    # --------------------------------------------------------
    # METADATA
    # --------------------------------------------------------
    metadata_df = pd.DataFrame(
        train_metadata + test_metadata
    )

    metadata_df.to_csv(
        OUTPUT_DIR / "metadata_features.csv",
        index=False
    )

    # --------------------------------------------------------
    # FEATURE NAMES
    # --------------------------------------------------------
    with open(
        OUTPUT_DIR / "feature_names.txt",
        "w",
        encoding="utf-8"
    ) as f:
        for i, feature in enumerate(
            FEATURE_NAMES
        ):
            f.write(
                f"{i:02d}: {feature}\n"
            )

    # --------------------------------------------------------
    # ERRORES
    # --------------------------------------------------------
    all_errors = (
        train_errors
        +
        test_errors
    )

    if all_errors:
        pd.DataFrame(
            all_errors
        ).to_csv(
            OUTPUT_DIR / "preprocessing_errors.csv",
            index=False
        )

    # --------------------------------------------------------
    # TXT DOCUMENTACION
    # --------------------------------------------------------
    info_path = write_info_txt(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        train_capture_counter=train_capture_counter,
        test_capture_counter=test_capture_counter,
        train_window_counter=train_window_counter,
        test_window_counter=test_window_counter,
        errors=all_errors,
    )

    # --------------------------------------------------------
    # RESUMEN FINAL
    # --------------------------------------------------------
    print("\n" + "=" * 80)
    print("RESUMEN FINAL")
    print("=" * 80)

    print(
        f"X_train: {X_train.shape}"
    )

    print(
        f"y_train: {y_train.shape}"
    )

    print(
        f"X_test : {X_test.shape}"
    )

    print(
        f"y_test : {y_test.shape}"
    )

    print("\nVentanas training por clase:")

    for cls in CLASS_TO_LABEL:
        print(
            f"  {cls:17s}: "
            f"{train_window_counter.get(cls, 0)}"
        )

    print("\nVentanas testing por clase:")

    for cls in CLASS_TO_LABEL:
        print(
            f"  {cls:17s}: "
            f"{test_window_counter.get(cls, 0)}"
        )

    print(
        f"\nMetadata: "
        f"{(OUTPUT_DIR / 'metadata_features.csv').resolve()}"
    )

    print(
        f"Descripcion: "
        f"{info_path.resolve()}"
    )

    print(
        f"\nArchivos guardados en:\n"
        f"{OUTPUT_DIR.resolve()}"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()
