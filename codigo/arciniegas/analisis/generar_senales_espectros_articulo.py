"""Genera la figura de senales y espectros representativos del articulo.

La seleccion usa exclusivamente el split de entrenamiento. Para evitar una
eleccion visual subjetiva, cada ejemplo es la ventana mas cercana al centro
robusto de su clase en el espacio de las 21 caracteristicas. La distancia se
calcula despues de centrar y escalar cada caracteristica mediante la mediana y
el IQR globales del entrenamiento. Este escalado solo sirve para seleccionar
la ventana que se muestra; no modifica E1, E4 ni ningun modelo entrenado.
"""

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


CODIGO_DIR = Path(__file__).resolve().parents[2]
if str(CODIGO_DIR) not in sys.path:
    sys.path.insert(0, str(CODIGO_DIR))

from comun.rutas_proyecto import (  # noqa: E402
    DATOS_ARCINIEGAS_PROCESADO,
    DATOS_ARCINIEGAS_RAW,
    RAIZ_PROYECTO,
)
from arciniegas.preprocesamiento.preprocesar_dataset import (  # noqa: E402
    CLASS_TO_LABEL,
    EXPECTED_FS,
    FEATURE_NAMES,
    FFT_SIZE,
    load_json_file,
)


TRAIN_DIR = DATOS_ARCINIEGAS_RAW / "training"
FIGURES_DIR = RAIZ_PROYECTO / "articulo" / "figuras"
DERIVED_RESULTS_DIR = RAIZ_PROYECTO / "articulo" / "resultados_derivados"
PNG_PATH = FIGURES_DIR / "senales_espectros_representativos.png"
SVG_PATH = FIGURES_DIR / "senales_espectros_representativos.svg"
SELECTION_PATH = DERIVED_RESULTS_DIR / "seleccion_ventanas_representativas.csv"

CLASS_ORDER = ["DETENIDO", "VELOCIDAD MEDIA", "VELOCIDAD ALTA"]
AXIS_NAMES = ["ax", "ay", "az"]
AXIS_LABELS = ["X", "Y", "Z"]
AXIS_COLORS = ["#0072B2", "#D55E00", "#009E73"]


def load_training_table():
    """Carga las features y verifica su correspondencia con el metadata."""
    x_train = np.load(DATOS_ARCINIEGAS_PROCESADO / "X_train.npy")
    metadata = pd.read_csv(
        DATOS_ARCINIEGAS_PROCESADO / "metadata_features.csv"
    )
    metadata = metadata.loc[metadata["split"] == "training"].reset_index(drop=True)

    if x_train.shape != (len(metadata), len(FEATURE_NAMES)):
        raise ValueError(
            "X_train y metadata de training no tienen dimensiones compatibles: "
            f"{x_train.shape} frente a ({len(metadata)}, {len(FEATURE_NAMES)})."
        )

    metadata_features = metadata[FEATURE_NAMES].to_numpy(dtype=np.float64)
    if not np.allclose(x_train, metadata_features, rtol=1e-6, atol=1e-6):
        raise ValueError(
            "El orden o los valores de X_train no coinciden con metadata_features.csv."
        )

    return x_train.astype(np.float64), metadata


def select_representative_windows(x_train, metadata):
    """Selecciona deterministamente una ventana central por clase."""
    median = np.median(x_train, axis=0)
    q25, q75 = np.percentile(x_train, [25, 75], axis=0)
    iqr = q75 - q25
    iqr[iqr == 0.0] = 1.0
    x_robust = (x_train - median) / iqr

    selections = []
    for class_name in CLASS_ORDER:
        class_indices = np.flatnonzero(metadata["class"].to_numpy() == class_name)
        class_vectors = x_robust[class_indices]
        robust_center = np.median(class_vectors, axis=0)
        distances = np.linalg.norm(class_vectors - robust_center, axis=1)
        local_index = int(np.argmin(distances))
        global_index = int(class_indices[local_index])
        row = metadata.iloc[global_index]

        selections.append(
            {
                "split": "training",
                "class": class_name,
                "label": int(row["label"]),
                "file": str(row["file"]),
                "window": int(row["window"]),
                "start_sample": int(row["start_sample"]),
                "end_sample_exclusive": int(row["end_sample_exclusive"]),
                "robust_center_distance": float(distances[local_index]),
                "selection_criterion": (
                    "minimum Euclidean distance to class median after global "
                    "training median/IQR scaling"
                ),
            }
        )

    return pd.DataFrame(selections)


def load_training_windows(metadata):
    """Reconstruye las ventanas del training desde los JSON originales."""
    cache = {}
    windows = {class_name: [] for class_name in CLASS_ORDER}

    window_rows = metadata[
        ["file", "class", "start_sample", "end_sample_exclusive"]
    ].itertuples(index=False, name=None)

    for filename, class_name, start_sample, end_sample in window_rows:
        filename = str(filename)
        if filename not in cache:
            values, fs = load_json_file(TRAIN_DIR / filename)
            if not np.isclose(fs, EXPECTED_FS):
                raise ValueError(f"Frecuencia inesperada en {filename}: {fs}")
            cache[filename] = values

        start = int(start_sample)
        end = int(end_sample)
        window = cache[filename][start:end, :]
        if window.shape != (FFT_SIZE, 3):
            raise ValueError(f"Ventana invalida en {filename}: {window.shape}")
        windows[str(class_name)].append(window)

    return {
        class_name: np.asarray(class_windows, dtype=np.float64)
        for class_name, class_windows in windows.items()
    }


def representative_window(selection, windows_by_class, metadata):
    """Recupera la ventana ya seleccionada manteniendo el orden del metadata."""
    mask = (
        (metadata["class"] == selection["class"])
        & (metadata["file"] == selection["file"])
        & (metadata["window"] == selection["window"])
    )
    matches = np.flatnonzero(mask.to_numpy())
    if len(matches) != 1:
        raise ValueError(f"La seleccion no identifica una ventana unica: {selection}")

    class_rows = metadata.index[metadata["class"] == selection["class"]].to_numpy()
    class_position = int(np.flatnonzero(class_rows == matches[0])[0])
    return windows_by_class[selection["class"]][class_position]


def create_figure(selections, metadata, windows_by_class):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7.5,
        }
    )

    fig, axes = plt.subplots(
        3,
        2,
        figsize=(7.16, 5.25),
        sharex="col",
        constrained_layout=True,
    )
    time_ms = np.arange(FFT_SIZE) * 1000.0 / EXPECTED_FS
    frequencies = np.fft.rfftfreq(FFT_SIZE, d=1.0 / EXPECTED_FS)[1:]

    centered_representatives = []
    spectral_summaries = {}
    for class_name in CLASS_ORDER:
        class_windows = windows_by_class[class_name]
        centered = class_windows - class_windows.mean(axis=1, keepdims=True)
        magnitude = np.abs(np.fft.rfft(centered, n=FFT_SIZE, axis=1))[:, 1:, :]
        spectral_summaries[class_name] = {
            "q25": np.percentile(magnitude, 25, axis=0),
            "median": np.median(magnitude, axis=0),
            "q75": np.percentile(magnitude, 75, axis=0),
        }

        selection = selections.loc[selections["class"] == class_name].iloc[0]
        representative = representative_window(selection, windows_by_class, metadata)
        centered_representatives.append(
            representative - representative.mean(axis=0, keepdims=True)
        )

    time_limit = max(np.max(np.abs(item)) for item in centered_representatives) * 1.08
    spectrum_limit = max(
        np.max(summary["q75"]) for summary in spectral_summaries.values()
    ) * 1.08

    for row_index, class_name in enumerate(CLASS_ORDER):
        ax_time, ax_spectrum = axes[row_index]
        centered = centered_representatives[row_index]
        summary = spectral_summaries[class_name]

        for axis_index, (axis_label, color) in enumerate(
            zip(AXIS_LABELS, AXIS_COLORS)
        ):
            ax_time.plot(
                time_ms,
                centered[:, axis_index],
                marker="o",
                markersize=2.2,
                linewidth=1.0,
                color=color,
                label=axis_label,
            )
            ax_spectrum.plot(
                frequencies,
                summary["median"][:, axis_index],
                marker="o",
                markersize=2.2,
                linewidth=1.0,
                color=color,
            )
            ax_spectrum.fill_between(
                frequencies,
                summary["q25"][:, axis_index],
                summary["q75"][:, axis_index],
                color=color,
                alpha=0.13,
                linewidth=0,
            )

        for axis in (ax_time, ax_spectrum):
            axis.grid(True, color="#D9D9D9", linewidth=0.45)
            axis.set_axisbelow(True)

        ax_time.axhline(0.0, color="#555555", linewidth=0.55)
        ax_time.set_ylim(-time_limit, time_limit)
        ax_spectrum.set_ylim(0.0, spectrum_limit)
        ax_time.set_ylabel(class_name.replace("VELOCIDAD ", "VEL. ") + "\nSeñal centrada")
        ax_spectrum.set_ylabel("Magnitud FFT")

    axes[0, 0].set_title("Ventana representativa de entrenamiento")
    axes[0, 1].set_title("Distribución espectral de la clase")
    axes[-1, 0].set_xlabel("Tiempo (ms)")
    axes[-1, 1].set_xlabel("Frecuencia (Hz), sin componente DC")
    axes[-1, 0].set_xlim(time_ms[0], time_ms[-1])
    axes[-1, 1].set_xlim(frequencies[0], frequencies[-1])

    legend_handles = [
        Line2D([0], [0], color=color, marker="o", markersize=3, linewidth=1.1, label=label)
        for label, color in zip(AXIS_LABELS, AXIS_COLORS)
    ]
    fig.legend(
        handles=legend_handles,
        loc="outside upper center",
        ncol=3,
        frameon=False,
        title="Eje del acelerómetro",
    )

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG_PATH, dpi=600, bbox_inches="tight")
    fig.savefig(SVG_PATH, bbox_inches="tight")
    plt.close(fig)


def main():
    x_train, metadata = load_training_table()
    selections = select_representative_windows(x_train, metadata)
    windows_by_class = load_training_windows(metadata)

    DERIVED_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    selections.to_csv(SELECTION_PATH, index=False)
    create_figure(selections, metadata, windows_by_class)

    print("Ventanas representativas:")
    print(selections[["class", "file", "window", "robust_center_distance"]].to_string(index=False))
    print(f"\nPNG: {PNG_PATH}")
    print(f"SVG: {SVG_PATH}")
    print(f"CSV: {SELECTION_PATH}")


if __name__ == "__main__":
    main()
