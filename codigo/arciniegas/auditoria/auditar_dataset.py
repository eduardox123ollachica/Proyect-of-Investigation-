from pathlib import Path
import json
import math
import sys

CODIGO_DIR = Path(__file__).resolve().parents[2]
if str(CODIGO_DIR) not in sys.path:
    sys.path.insert(0, str(CODIGO_DIR))

from comun.rutas_proyecto import DATOS_ARCINIEGAS_RAW, RESULTADOS_AUDITORIA

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURACION
# ============================================================

BASE_DIR = DATOS_ARCINIEGAS_RAW

TRAIN_DIR = BASE_DIR / "training"
TEST_DIR = BASE_DIR / "testing"

OUTPUT_CSV = RESULTADOS_AUDITORIA / "arciniegas_dataset_audit.csv"

EXPECTED_FS = 100
EXPECTED_INTERVAL_MS = 10
EXPECTED_SAMPLES = 300


# ============================================================
# UTILIDADES
# ============================================================

def infer_label_from_filename(filename):
    """
    Obtiene la clase a partir del nombre del archivo.

    Clases esperadas:
        DETENIDO
        VELOCIDAD MEDIA
        VELOCIDAD ALTA
    """

    name = filename.upper()

    if "DETENIDO" in name:
        return "DETENIDO"

    elif "VELOCIDAD MEDIA" in name:
        return "VELOCIDAD MEDIA"

    elif "VELOCIDAD ALTA" in name:
        return "VELOCIDAD ALTA"

    else:
        return "DESCONOCIDA"


def rms(x):
    """
    Root Mean Square
    """
    x = np.asarray(x, dtype=np.float64)

    return np.sqrt(
        np.mean(
            np.square(x)
        )
    )


# ============================================================
# LEER UN JSON
# ============================================================

def analyze_json(filepath, split_name):

    result = {
        "split": split_name,
        "file": filepath.name,
        "label": infer_label_from_filename(filepath.name),

        "valid_json": False,
        "valid_structure": False,

        "interval_ms": np.nan,
        "fs_hz": np.nan,
        "n_samples": np.nan,
        "duration_s": np.nan,

        "ax_mean": np.nan,
        "ay_mean": np.nan,
        "az_mean": np.nan,

        "ax_std": np.nan,
        "ay_std": np.nan,
        "az_std": np.nan,

        "ax_rms": np.nan,
        "ay_rms": np.nan,
        "az_rms": np.nan,

        "ax_min": np.nan,
        "ay_min": np.nan,
        "az_min": np.nan,

        "ax_max": np.nan,
        "ay_max": np.nan,
        "az_max": np.nan,

        "has_nan": False,

        "expected_interval": False,
        "expected_fs": False,
        "expected_samples": False,

        "warning": ""
    }

    warnings = []

    # ========================================================
    # CARGAR JSON
    # ========================================================

    try:

        with open(
            filepath,
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(f)

        result["valid_json"] = True

    except Exception as e:

        result["warning"] = (
            f"Error leyendo JSON: {e}"
        )

        return result


    # ========================================================
    # BUSCAR PAYLOAD
    # ========================================================

    if "payload" not in data:

        result["warning"] = (
            "No existe campo payload"
        )

        return result


    payload = data["payload"]


    # ========================================================
    # INTERVALO DE MUESTREO
    # ========================================================

    interval_ms = payload.get(
        "interval_ms",
        None
    )

    if interval_ms is None:

        warnings.append(
            "Falta interval_ms"
        )

    else:

        result["interval_ms"] = interval_ms

        if interval_ms > 0:

            fs_hz = (
                1000.0
                / interval_ms
            )

            result["fs_hz"] = fs_hz

            result["expected_interval"] = (
                interval_ms
                == EXPECTED_INTERVAL_MS
            )

            result["expected_fs"] = (
                math.isclose(
                    fs_hz,
                    EXPECTED_FS,
                    rel_tol=1e-9
                )
            )


    # ========================================================
    # SENSORES
    # ========================================================

    sensors = payload.get(
        "sensors",
        []
    )

    sensor_names = [
        s.get("name", "")
        for s in sensors
    ]

    expected_sensors = [
        "ax",
        "ay",
        "az"
    ]

    if sensor_names != expected_sensors:

        warnings.append(
            f"Sensores encontrados: {sensor_names}"
        )


    # ========================================================
    # VALORES
    # ========================================================

    values = payload.get(
        "values",
        None
    )

    if values is None:

        warnings.append(
            "Falta values"
        )

        result["warning"] = "; ".join(
            warnings
        )

        return result


    try:

        values = np.asarray(
            values,
            dtype=np.float64
        )

    except Exception as e:

        warnings.append(
            f"Error convirtiendo values: {e}"
        )

        result["warning"] = "; ".join(
            warnings
        )

        return result


    # ========================================================
    # VALIDAR DIMENSIONES
    # ========================================================

    if values.ndim != 2:

        warnings.append(
            f"values tiene ndim={values.ndim}"
        )

        result["warning"] = "; ".join(
            warnings
        )

        return result


    if values.shape[1] != 3:

        warnings.append(
            f"Se esperaban 3 ejes, encontrados {values.shape[1]}"
        )

        result["warning"] = "; ".join(
            warnings
        )

        return result


    result["valid_structure"] = True


    # ========================================================
    # NUMERO DE MUESTRAS
    # ========================================================

    n_samples = values.shape[0]

    result["n_samples"] = n_samples

    result["expected_samples"] = (
        n_samples
        == EXPECTED_SAMPLES
    )


    # ========================================================
    # DURACION
    # ========================================================

    if not np.isnan(
        result["fs_hz"]
    ):

        result["duration_s"] = (
            n_samples
            / result["fs_hz"]
        )


    # ========================================================
    # EXTRAER EJES
    # ========================================================

    ax = values[:, 0]
    ay = values[:, 1]
    az = values[:, 2]


    # ========================================================
    # NAN / INF
    # ========================================================

    has_invalid = (
        np.isnan(values).any()
        or
        np.isinf(values).any()
    )

    result["has_nan"] = has_invalid

    if has_invalid:

        warnings.append(
            "Existen NaN o Inf"
        )


    # ========================================================
    # ESTADISTICAS
    # ========================================================

    result["ax_mean"] = np.mean(ax)
    result["ay_mean"] = np.mean(ay)
    result["az_mean"] = np.mean(az)

    result["ax_std"] = np.std(ax)
    result["ay_std"] = np.std(ay)
    result["az_std"] = np.std(az)

    result["ax_rms"] = rms(ax)
    result["ay_rms"] = rms(ay)
    result["az_rms"] = rms(az)

    result["ax_min"] = np.min(ax)
    result["ay_min"] = np.min(ay)
    result["az_min"] = np.min(az)

    result["ax_max"] = np.max(ax)
    result["ay_max"] = np.max(ay)
    result["az_max"] = np.max(az)


    # ========================================================
    # ADVERTENCIAS
    # ========================================================

    if not result["expected_interval"]:

        warnings.append(
            f"interval_ms != {EXPECTED_INTERVAL_MS}"
        )


    if not result["expected_fs"]:

        warnings.append(
            f"fs != {EXPECTED_FS} Hz"
        )


    if not result["expected_samples"]:

        warnings.append(
            f"n_samples != {EXPECTED_SAMPLES}"
        )


    if result["label"] == "DESCONOCIDA":

        warnings.append(
            "Clase no reconocida"
        )


    result["warning"] = "; ".join(
        warnings
    )


    return result


# ============================================================
# PROCESAR UNA CARPETA
# ============================================================

def process_folder(
    folder,
    split_name
):

    rows = []

    if not folder.exists():

        raise FileNotFoundError(
            f"No existe la carpeta: {folder}"
        )


    json_files = sorted(
        folder.glob("*.json")
    )


    print("\n")
    print("=" * 80)
    print(split_name.upper())
    print("=" * 80)

    print(
        f"Archivos JSON encontrados: {len(json_files)}"
    )


    for filepath in json_files:

        row = analyze_json(
            filepath,
            split_name
        )

        rows.append(row)


        status = (
            "OK"
            if row["warning"] == ""
            else "REVISION"
        )

        print(
            f"{status:8s} | "
            f"{filepath.name}"
        )


    return rows


# ============================================================
# RESUMEN
# ============================================================

def print_summary(df):

    print("\n")
    print("=" * 80)
    print("RESUMEN GENERAL")
    print("=" * 80)


    # --------------------------------------------------------
    # ARCHIVOS POR SPLIT Y CLASE
    # --------------------------------------------------------

    print("\nArchivos por split y clase:")

    summary = (
        df
        .groupby(
            [
                "split",
                "label"
            ]
        )
        .size()
        .reset_index(
            name="files"
        )
    )

    print(
        summary.to_string(
            index=False
        )
    )


    # --------------------------------------------------------
    # MUESTRAS
    # --------------------------------------------------------

    print(
        "\nNumero de muestras:"
    )

    sample_summary = (
        df
        .groupby(
            [
                "split",
                "label"
            ]
        )["n_samples"]
        .agg(
            [
                "count",
                "min",
                "max",
                "mean"
            ]
        )
        .reset_index()
    )

    print(
        sample_summary.to_string(
            index=False
        )
    )


    # --------------------------------------------------------
    # FRECUENCIA
    # --------------------------------------------------------

    print(
        "\nFrecuencias de muestreo encontradas:"
    )

    fs_summary = (
        df
        .groupby(
            [
                "split",
                "fs_hz"
            ]
        )
        .size()
        .reset_index(
            name="files"
        )
    )

    print(
        fs_summary.to_string(
            index=False
        )
    )


    # --------------------------------------------------------
    # ESTADISTICAS POR CLASE
    # --------------------------------------------------------

    print(
        "\nPromedio de desviacion estandar por clase:"
    )

    std_summary = (
        df
        .groupby(
            [
                "split",
                "label"
            ]
        )[
            [
                "ax_std",
                "ay_std",
                "az_std"
            ]
        ]
        .mean()
        .reset_index()
    )

    print(
        std_summary
        .round(4)
        .to_string(
            index=False
        )
    )


    print(
        "\nPromedio RMS por clase:"
    )

    rms_summary = (
        df
        .groupby(
            [
                "split",
                "label"
            ]
        )[
            [
                "ax_rms",
                "ay_rms",
                "az_rms"
            ]
        ]
        .mean()
        .reset_index()
    )

    print(
        rms_summary
        .round(4)
        .to_string(
            index=False
        )
    )


    # --------------------------------------------------------
    # ARCHIVOS CON ADVERTENCIAS
    # --------------------------------------------------------

    problematic = df[
        df["warning"] != ""
    ]


    print("\n")

    if len(problematic) == 0:

        print(
            "No se encontraron archivos problematicos."
        )

    else:

        print(
            f"Archivos para revision: "
            f"{len(problematic)}"
        )

        print(

            problematic[
                [
                    "split",
                    "file",
                    "label",
                    "fs_hz",
                    "n_samples",
                    "warning"
                ]
            ]
            .to_string(
                index=False
            )
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("AUDITORIA DATASET ARCINIEGAS")
    print("=" * 80)

    print(
        f"Ruta base: {BASE_DIR.resolve()}"
    )


    # ========================================================
    # TRAINING
    # ========================================================

    training_rows = process_folder(

        TRAIN_DIR,

        "training"
    )


    # ========================================================
    # TESTING
    # ========================================================

    testing_rows = process_folder(

        TEST_DIR,

        "testing"
    )


    # ========================================================
    # DATAFRAME FINAL
    # ========================================================

    all_rows = (
        training_rows
        +
        testing_rows
    )

    df = pd.DataFrame(
        all_rows
    )


    # ========================================================
    # GUARDAR CSV
    # ========================================================

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(

        OUTPUT_CSV,

        index=False
    )


    # ========================================================
    # MOSTRAR RESUMEN
    # ========================================================

    print_summary(df)


    print("\n")
    print("=" * 80)

    print(
        "Auditoria guardada en:"
    )

    print(
        OUTPUT_CSV.resolve()
    )

    print("=" * 80)


# ============================================================
# EJECUCION
# ============================================================

if __name__ == "__main__":

    main()
