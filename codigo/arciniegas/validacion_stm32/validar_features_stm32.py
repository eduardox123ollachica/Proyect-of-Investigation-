import json
import time
from pathlib import Path
import sys

CODIGO_DIR = Path(__file__).resolve().parents[2]
if str(CODIGO_DIR) not in sys.path:
    sys.path.insert(0, str(CODIGO_DIR))

from comun.rutas_proyecto import DATOS_ARCINIEGAS_PROCESADO, DATOS_ARCINIEGAS_RAW, RESULTADOS_COMPARTIDOS

import numpy as np
import pandas as pd
import serial


# ============================================================
# CONFIGURACIÓN
# ============================================================

PUERTO = "COM4"
BAUDRATE = 115200

PREPROCESADO_DIR = DATOS_ARCINIEGAS_PROCESADO
DATASET_DIR = DATOS_ARCINIEGAS_RAW

TESTING_DIR = (
    DATASET_DIR
    / "testing"
)


# ============================================================
# ARCHIVOS
# ============================================================

X_TEST_PATH = (
    PREPROCESADO_DIR
    / "X_test.npy"
)

Y_TEST_PATH = (
    PREPROCESADO_DIR
    / "y_test.npy"
)

METADATA_PATH = (
    PREPROCESADO_DIR
    / "metadata_features.csv"
)


# ============================================================
# 10 VENTANAS REPRESENTATIVAS
# ============================================================

INDICES = [
    0,
    180,
    360,
    539,

    540,
    720,
    899,

    900,
    1115,
    1331
]


# ============================================================
# CLASES
# ============================================================

CLASES = {
    0: "DETENIDO",
    1: "VELOCIDAD MEDIA",
    2: "VELOCIDAD ALTA"
}


# ============================================================
# NOMBRES DE FEATURES
# ============================================================

FEATURE_NAMES = [

    "ax_rms",
    "ax_skewness",
    "ax_kurtosis",
    "ax_spectral_energy",
    "ax_dominant_frequency",
    "ax_dominant_amplitude",
    "ax_spectral_entropy",

    "ay_rms",
    "ay_skewness",
    "ay_kurtosis",
    "ay_spectral_energy",
    "ay_dominant_frequency",
    "ay_dominant_amplitude",
    "ay_spectral_entropy",

    "az_rms",
    "az_skewness",
    "az_kurtosis",
    "az_spectral_energy",
    "az_dominant_frequency",
    "az_dominant_amplitude",
    "az_spectral_entropy",
]


# Índices de frecuencia dominante dentro
# del vector de 21 features.

DOMINANT_FREQUENCY_INDICES = [
    4,      # ax
    11,     # ay
    18      # az
]


# ============================================================
# CARPETA DE SALIDA
# ============================================================

SALIDA_DIR = RESULTADOS_COMPARTIDOS / "features" / "validacion_stm32_10_ventanas"

SALIDA_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# VERIFICAR ARCHIVOS
# ============================================================

for ruta in [
    X_TEST_PATH,
    Y_TEST_PATH,
    METADATA_PATH
]:

    if not ruta.exists():

        raise FileNotFoundError(
            f"No se encontró:\n{ruta}"
        )


if not TESTING_DIR.exists():

    raise FileNotFoundError(
        f"No existe:\n{TESTING_DIR}"
    )


# ============================================================
# CARGAR DATOS
# ============================================================

X_test = np.load(
    X_TEST_PATH
)

y_test = np.load(
    Y_TEST_PATH
)

metadata = pd.read_csv(
    METADATA_PATH
)


# ============================================================
# METADATA DE TESTING
# ============================================================

metadata_test = (
    metadata[
        metadata["split"] == "testing"
    ]
    .reset_index(drop=True)
)


if len(metadata_test) != len(X_test):

    raise RuntimeError(
        "metadata testing y X_test "
        "no tienen la misma cantidad de ventanas."
    )


# ============================================================
# COMPROBAR ÍNDICES
# ============================================================

for indice in INDICES:

    if indice < 0 or indice >= len(X_test):

        raise IndexError(
            f"Índice fuera de rango: {indice}"
        )


# ============================================================
# MOSTRAR DISTRIBUCIÓN
# ============================================================

print("=" * 90)
print("VALIDACIÓN DEL PREPROCESAMIENTO PC vs STM32")
print("=" * 90)

print("\nVentanas seleccionadas:")

for indice in INDICES:

    clase = int(
        y_test[indice]
    )

    print(
        f"{indice:4d} -> "
        f"{clase} "
        f"{CLASES[clase]}"
    )


# ============================================================
# UART
# ============================================================

ser = serial.Serial(
    PUERTO,
    BAUDRATE,
    timeout=5
)

time.sleep(1)

ser.reset_input_buffer()

print("\nConectado a", PUERTO)


# ============================================================
# ESTRUCTURAS PARA RESULTADOS
# ============================================================

resultados_ventanas = []


errores_abs_por_feature = {
    nombre: []
    for nombre in FEATURE_NAMES
}


errores_rel_por_feature = {
    nombre: []
    for nombre in FEATURE_NAMES
}


# Contadores para las frecuencias dominantes

frecuencias_dominantes_totales = 0
frecuencias_dominantes_iguales = 0


# ============================================================
# PROCESAR LAS 10 VENTANAS
# ============================================================

for numero, INDICE in enumerate(
    INDICES,
    start=1
):

    print("\n")
    print("=" * 90)

    print(
        f"VENTANA {numero}/{len(INDICES)} "
        f"- ÍNDICE {INDICE}"
    )

    print("=" * 90)


    # ========================================================
    # METADATA
    # ========================================================

    fila_meta = (
        metadata_test.iloc[
            INDICE
        ]
    )


    archivo_json = str(
        fila_meta["file"]
    )

    start_sample = int(
        fila_meta["start_sample"]
    )

    end_sample = int(
        fila_meta[
            "end_sample_exclusive"
        ]
    )

    ventana_numero = int(
        fila_meta["window"]
    )

    y_real = int(
        y_test[INDICE]
    )


    JSON_PATH = (
        TESTING_DIR
        / archivo_json
    )


    if not JSON_PATH.exists():

        raise FileNotFoundError(
            f"No se encontró:\n{JSON_PATH}"
        )


    # ========================================================
    # LEER JSON
    # ========================================================

    with open(
        JSON_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)


    values = np.asarray(
        data["payload"]["values"],
        dtype=np.float64
    )


    # ========================================================
    # EXTRAER VENTANA 16×3
    # ========================================================

    window = values[
        start_sample:end_sample,
        :
    ]


    if window.shape != (16, 3):

        raise RuntimeError(
            f"Ventana {INDICE} "
            f"tiene forma {window.shape}"
        )


    # ========================================================
    # FEATURES ORIGINALES DEL PC
    # ========================================================

    features_pc = (
        X_test[INDICE]
        .astype(np.float64)
    )


    # ========================================================
    # CONSTRUIR RAWFEAT
    # ========================================================

    raw_values = (
        window.flatten()
    )


    comando = (
        "RAWFEAT,"
        +
        ",".join(
            f"{valor:.9f}"
            for valor in raw_values
        )
    )


    # ========================================================
    # ENVIAR STM32
    # ========================================================

    ser.reset_input_buffer()

    ser.write(
        (
            comando
            + "\r\n"
        ).encode()
    )


    respuesta = (
        ser.readline()
        .decode(
            errors="ignore"
        )
        .strip()
    )


    # ========================================================
    # VALIDAR RESPUESTA
    # ========================================================

    if not respuesta.startswith(
        "FEATURES,"
    ):

        print(
            "ERROR STM32:"
        )

        print(respuesta)

        continue


    partes = (
        respuesta.split(",")
    )


    if len(partes) != 22:

        print(
            "ERROR: se esperaban "
            "21 features."
        )

        print(
            "Recibidas:",
            len(partes) - 1
        )

        continue


    # ========================================================
    # FEATURES STM32
    # ========================================================

    features_stm32 = np.asarray(
        [
            float(valor)
            for valor
            in partes[1:]
        ],
        dtype=np.float64
    )


    # ========================================================
    # ERROR ABSOLUTO
    # ========================================================

    error_abs = np.abs(
        features_pc
        -
        features_stm32
    )


    # ========================================================
    # ERROR RELATIVO
    # ========================================================

    EPS = 1e-12


    error_rel = (
        error_abs
        /
        np.maximum(
            np.abs(
                features_pc
            ),
            EPS
        )
    )


    error_rel_pct = (
        error_rel
        *
        100.0
    )


    # ========================================================
    # GUARDAR ERROR POR FEATURE
    # ========================================================

    for i, nombre in enumerate(
        FEATURE_NAMES
    ):

        errores_abs_por_feature[
            nombre
        ].append(
            float(
                error_abs[i]
            )
        )


        errores_rel_por_feature[
            nombre
        ].append(
            float(
                error_rel_pct[i]
            )
        )


    # ========================================================
    # COMPARAR FRECUENCIA DOMINANTE
    # ========================================================

    coincidencias_frecuencia_ventana = 0


    for indice_feature in (
        DOMINANT_FREQUENCY_INDICES
    ):

        frecuencias_dominantes_totales += 1


        if (
            features_pc[indice_feature]
            ==
            features_stm32[indice_feature]
        ):

            frecuencias_dominantes_iguales += 1

            coincidencias_frecuencia_ventana += 1


    # ========================================================
    # MAYOR ERROR ABSOLUTO
    # ========================================================

    indice_max = int(
        np.argmax(
            error_abs
        )
    )


    error_medio = float(
        np.mean(
            error_abs
        )
    )


    error_maximo = float(
        np.max(
            error_abs
        )
    )


    error_rel_medio = float(
        np.mean(
            error_rel_pct
        )
    )


    error_rel_maximo = float(
        np.max(
            error_rel_pct
        )
    )


    # ========================================================
    # TERMINAL
    # ========================================================

    print(
        "Clase:",
        y_real,
        CLASES[y_real]
    )

    print(
        "Archivo:",
        archivo_json
    )

    print(
        "Ventana archivo:",
        ventana_numero
    )

    print()

    print(
        "Error absoluto medio :",
        f"{error_medio:.12f}"
    )

    print(
        "Error absoluto máximo:",
        f"{error_maximo:.12f}"
    )

    print(
        "Error relativo medio :",
        f"{error_rel_medio:.8f}%"
    )

    print(
        "Error relativo máximo:",
        f"{error_rel_maximo:.8f}%"
    )

    print(
        "Mayor error absoluto :",
        FEATURE_NAMES[
            indice_max
        ]
    )

    print(
        "Frecuencias dominantes iguales:",
        f"{coincidencias_frecuencia_ventana}/3"
    )


    # ========================================================
    # CARPETA VENTANA
    # ========================================================

    VENTANA_DIR = (
        SALIDA_DIR
        / f"ventana_{INDICE:04d}"
    )

    VENTANA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    # ========================================================
    # CSV DE LA VENTANA
    # ========================================================

    comparacion = pd.DataFrame(
        {
            "feature_num":
                np.arange(
                    1,
                    22
                ),

            "feature":
                FEATURE_NAMES,

            "valor_pc":
                features_pc,

            "valor_stm32":
                features_stm32,

            "error_absoluto":
                error_abs,

            "error_relativo_pct":
                error_rel_pct
        }
    )


    comparacion.to_csv(
        VENTANA_DIR
        / "comparacion_features.csv",
        index=False,
        sep=";",
        encoding="utf-8-sig"
    )


    # ========================================================
    # RESUMEN DE LA VENTANA
    # ========================================================

    resultados_ventanas.append(
        {
            "indice_ventana":
                INDICE,

            "clase_id":
                y_real,

            "clase_nombre":
                CLASES[y_real],

            "archivo_json":
                archivo_json,

            "ventana_archivo":
                ventana_numero,

            "start_sample":
                start_sample,

            "end_sample":
                end_sample,

            "error_abs_medio":
                error_medio,

            "error_abs_maximo":
                error_maximo,

            "error_rel_medio_pct":
                error_rel_medio,

            "error_rel_maximo_pct":
                error_rel_maximo,

            "feature_mayor_error":
                FEATURE_NAMES[
                    indice_max
                ],

            "frecuencias_dominantes_iguales":
                coincidencias_frecuencia_ventana
        }
    )


# ============================================================
# CERRAR UART
# ============================================================

ser.close()


# ============================================================
# CSV RESUMEN DE LAS 10 VENTANAS
# ============================================================

df_resumen = pd.DataFrame(
    resultados_ventanas
)


df_resumen.to_csv(
    SALIDA_DIR
    / "resumen_10_ventanas.csv",
    index=False,
    sep=";",
    encoding="utf-8-sig"
)


# ============================================================
# RESUMEN DETALLADO POR FEATURE
# ============================================================

resumen_features = []


for nombre in FEATURE_NAMES:

    errores_abs = np.asarray(
        errores_abs_por_feature[
            nombre
        ],
        dtype=np.float64
    )


    errores_rel = np.asarray(
        errores_rel_por_feature[
            nombre
        ],
        dtype=np.float64
    )


    resumen_features.append(
        {
            "feature":
                nombre,

            "error_abs_medio":
                float(
                    np.mean(
                        errores_abs
                    )
                ),

            "error_abs_maximo":
                float(
                    np.max(
                        errores_abs
                    )
                ),

            "error_abs_minimo":
                float(
                    np.min(
                        errores_abs
                    )
                ),

            "error_rel_medio_pct":
                float(
                    np.mean(
                        errores_rel
                    )
                ),

            "error_rel_maximo_pct":
                float(
                    np.max(
                        errores_rel
                    )
                )
        }
    )


df_features = pd.DataFrame(
    resumen_features
)


df_features.to_csv(
    SALIDA_DIR
    / "resumen_error_por_feature.csv",
    index=False,
    sep=";",
    encoding="utf-8-sig"
)


# ============================================================
# MÉTRICAS GLOBALES ABSOLUTAS
# ============================================================

todos_errores_abs = []


for valores in (
    errores_abs_por_feature.values()
):

    todos_errores_abs.extend(
        valores
    )


todos_errores_abs = np.asarray(
    todos_errores_abs,
    dtype=np.float64
)


error_abs_global_medio = float(
    np.mean(
        todos_errores_abs
    )
)


error_abs_global_maximo = float(
    np.max(
        todos_errores_abs
    )
)


# ============================================================
# MÉTRICAS GLOBALES RELATIVAS
# ============================================================

todos_errores_rel = []


for valores in (
    errores_rel_por_feature.values()
):

    todos_errores_rel.extend(
        valores
    )


todos_errores_rel = np.asarray(
    todos_errores_rel,
    dtype=np.float64
)


error_rel_global_medio = float(
    np.mean(
        todos_errores_rel
    )
)


error_rel_global_maximo = float(
    np.max(
        todos_errores_rel
    )
)


# ============================================================
# FEATURES CON MAYORES ERRORES
# ============================================================

fila_max_abs = (
    df_features.loc[
        df_features[
            "error_abs_maximo"
        ].idxmax()
    ]
)


fila_max_rel = (
    df_features.loc[
        df_features[
            "error_rel_maximo_pct"
        ].idxmax()
    ]
)


# ============================================================
# TXT FINAL
# ============================================================

RESUMEN_TXT = (
    SALIDA_DIR
    / "resumen_general.txt"
)


with open(
    RESUMEN_TXT,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "VALIDACIÓN DEL PREPROCESAMIENTO "
        "PC vs STM32\n"
    )

    f.write(
        "=" * 80
        + "\n\n"
    )


    f.write(
        f"Ventanas analizadas: "
        f"{len(resultados_ventanas)}\n"
    )

    f.write(
        "Features por ventana: 21\n"
    )

    f.write(
        f"Comparaciones totales: "
        f"{len(todos_errores_abs)}\n\n"
    )


    # ========================================================
    # DISTRIBUCIÓN
    # ========================================================

    f.write(
        "DISTRIBUCIÓN DE CLASES\n"
    )

    f.write(
        "-" * 80
        + "\n"
    )


    for clase_id in [
        0,
        1,
        2
    ]:

        cantidad = int(
            np.sum(
                y_test[
                    INDICES
                ]
                ==
                clase_id
            )
        )

        f.write(
            f"{CLASES[clase_id]}: "
            f"{cantidad}\n"
        )


    # ========================================================
    # RESULTADOS GLOBALES
    # ========================================================

    f.write(
        "\nRESULTADOS GLOBALES\n"
    )

    f.write(
        "-" * 80
        + "\n"
    )

    f.write(
        "Error absoluto medio global: "
        f"{error_abs_global_medio:.12f}\n"
    )

    f.write(
        "Error absoluto máximo global: "
        f"{error_abs_global_maximo:.12f}\n"
    )

    f.write(
        "Error relativo medio global: "
        f"{error_rel_global_medio:.8f}%\n"
    )

    f.write(
        "Error relativo máximo global: "
        f"{error_rel_global_maximo:.8f}%\n\n"
    )


    # ========================================================
    # FRECUENCIAS
    # ========================================================

    f.write(
        "VALIDACIÓN DE FRECUENCIA DOMINANTE\n"
    )

    f.write(
        "-" * 80
        + "\n"
    )

    f.write(
        "Comparaciones realizadas: "
        f"{frecuencias_dominantes_totales}\n"
    )

    f.write(
        "Coincidencias exactas: "
        f"{frecuencias_dominantes_iguales}\n"
    )

    f.write(
        "Diferencias: "
        f"{frecuencias_dominantes_totales - frecuencias_dominantes_iguales}\n"
    )

    f.write(
        "Porcentaje de coincidencia: "
        f"{100.0 * frecuencias_dominantes_iguales / frecuencias_dominantes_totales:.2f}%\n\n"
    )


    # ========================================================
    # MAYORES ERRORES
    # ========================================================

    f.write(
        "MAYOR ERROR ABSOLUTO\n"
    )

    f.write(
        "-" * 80
        + "\n"
    )

    f.write(
        f"Feature: "
        f"{fila_max_abs['feature']}\n"
    )

    f.write(
        f"Error máximo: "
        f"{fila_max_abs['error_abs_maximo']:.12f}\n\n"
    )


    f.write(
        "MAYOR ERROR RELATIVO\n"
    )

    f.write(
        "-" * 80
        + "\n"
    )

    f.write(
        f"Feature: "
        f"{fila_max_rel['feature']}\n"
    )

    f.write(
        f"Error máximo: "
        f"{fila_max_rel['error_rel_maximo_pct']:.8f}%\n\n"
    )


    # ========================================================
    # TABLA POR FEATURE
    # ========================================================

    f.write(
        "ERROR POR FEATURE\n"
    )

    f.write(
        "-" * 80
        + "\n"
    )

    f.write(
        f"{'Feature':28s}"
        f"{'Abs medio':>16s}"
        f"{'Abs máximo':>16s}"
        f"{'Rel medio %':>16s}"
        f"{'Rel máximo %':>16s}\n"
    )

    f.write(
        "-" * 92
        + "\n"
    )


    for _, fila in (
        df_features.iterrows()
    ):

        f.write(
            f"{fila['feature']:28s}"
            f"{fila['error_abs_medio']:16.10f}"
            f"{fila['error_abs_maximo']:16.10f}"
            f"{fila['error_rel_medio_pct']:16.8f}"
            f"{fila['error_rel_maximo_pct']:16.8f}\n"
        )


# ============================================================
# TERMINAL
# ============================================================

print("\n")
print("=" * 90)
print("RESUMEN FINAL DE LAS 10 VENTANAS")
print("=" * 90)

print(
    "Ventanas procesadas:",
    len(resultados_ventanas)
)

print(
    "Comparaciones realizadas:",
    len(todos_errores_abs)
)

print()

print(
    "Error absoluto medio global :",
    f"{error_abs_global_medio:.12f}"
)

print(
    "Error absoluto máximo global:",
    f"{error_abs_global_maximo:.12f}"
)

print(
    "Error relativo medio global :",
    f"{error_rel_global_medio:.8f}%"
)

print(
    "Error relativo máximo global:",
    f"{error_rel_global_maximo:.8f}%"
)

print()

print(
    "Feature mayor error absoluto:",
    fila_max_abs["feature"]
)

print(
    "Feature mayor error relativo:",
    fila_max_rel["feature"]
)

print()

print(
    "Frecuencias dominantes:"
)

print(
    f"{frecuencias_dominantes_iguales}"
    f"/"
    f"{frecuencias_dominantes_totales}"
    " coincidencias exactas"
)

print()

print(
    "Resultados guardados en:"
)

print(
    SALIDA_DIR
)
