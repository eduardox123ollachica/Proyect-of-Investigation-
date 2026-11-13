from pathlib import Path
import json
import time
import sys

CODIGO_DIR = Path(__file__).resolve().parents[2]
if str(CODIGO_DIR) not in sys.path:
    sys.path.insert(0, str(CODIGO_DIR))

from comun.rutas_proyecto import DATOS_ARCINIEGAS_PROCESADO, DATOS_ARCINIEGAS_RAW, RESULTADOS_KMEANS

import numpy as np
import pandas as pd
import serial


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

DATASET_TEST_DIR = DATOS_ARCINIEGAS_RAW / "testing"
PREPROCESADO_DIR = DATOS_ARCINIEGAS_PROCESADO
KMEANS_DIR = RESULTADOS_KMEANS

OUTPUT_DIR = (
    KMEANS_DIR
    / "Validacion_STM32"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# UART
# ============================================================

PUERTO = "COM4"

BAUDRATE = 115200

TIMEOUT_S = 3.0

PAUSA_ENTRE_COMANDOS_S = 0.02


# ============================================================
# MODO DE EJECUCIÓN
#
# "UNA"
# "VARIAS"
# "TODAS"
# ============================================================

MODO = "TODAS"


# ============================================================
# SELECCIÓN DE VENTANAS
#
# índice = índice dentro del dataframe TEST:
#
# 0 ... 1331
#
# Ejemplos:
#
# UNA:
# INDICES_TEST = [0]
#
# VARIAS:
# INDICES_TEST = [0, 100, 500, 1000]
#
# TODAS:
# esta lista se ignora.
# ============================================================

INDICES_TEST = [
    0
]


# ============================================================
# OPCIÓN ALTERNATIVA
#
# Si quieres seleccionar por:
#
# archivo + número de ventana
#
# activa:
#
# USAR_ARCHIVO_Y_WINDOW = True
#
# Ejemplo:
#
# FILE_OBJETIVO =
# "DETENIDO.json....s59.json"
#
# WINDOW_OBJETIVO = 24
# ============================================================

USAR_ARCHIVO_Y_WINDOW = False

FILE_OBJETIVO = (
    "DETENIDO.json.51iirpm7."
    "ingestion-6f998cf944-t6sxf.s59.json"
)

WINDOW_OBJETIVO = 24


# ============================================================
# ARCHIVOS
# ============================================================

METADATA_PATH = (
    PREPROCESADO_DIR
    / "metadata_features.csv"
)

SCORES_PC_PATH = (
    KMEANS_DIR
    / "scores_test_STANDARD.csv"
)


# ============================================================
# CONSTANTES
# ============================================================

WINDOW_SIZE = 16

AXIS_COUNT = 3

TOTAL_RAW_VALUES = (
    WINDOW_SIZE
    *
    AXIS_COUNT
)


# ============================================================
# CARGAR JSON
# ============================================================

def cargar_json_raw(
    filepath: Path
):
    with open(
        filepath,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)


    if "payload" not in data:
        raise ValueError(
            f"No existe payload en {filepath.name}"
        )


    payload = data[
        "payload"
    ]


    values = np.asarray(
        payload.get(
            "values",
            []
        ),
        dtype=np.float64
    )


    if (
        values.ndim != 2
        or
        values.shape[1] != 3
    ):
        raise ValueError(
            f"Forma inválida en "
            f"{filepath.name}: "
            f"{values.shape}"
        )


    return values


# ============================================================
# RECONSTRUIR UNA VENTANA
# ============================================================

def obtener_ventana(
    fila_metadata
):
    nombre_archivo = str(
        fila_metadata[
            "file"
        ]
    )


    archivo_json = (
        DATASET_TEST_DIR
        /
        nombre_archivo
    )


    if not archivo_json.exists():
        raise FileNotFoundError(
            f"No existe:\n"
            f"{archivo_json}"
        )


    values = cargar_json_raw(
        archivo_json
    )


    start = int(
        fila_metadata[
            "start_sample"
        ]
    )


    end = int(
        fila_metadata[
            "end_sample_exclusive"
        ]
    )


    window = values[
        start:end,
        :
    ]


    if window.shape != (
        WINDOW_SIZE,
        AXIS_COUNT
    ):
        raise ValueError(
            f"Ventana inválida: "
            f"{window.shape}"
        )


    return window


# ============================================================
# CREAR COMANDO UART
# ============================================================

def construir_comando(
    window
):
    valores = (
        window
        .reshape(-1)
    )


    if len(valores) != TOTAL_RAW_VALUES:
        raise ValueError(
            "La ventana no tiene 48 valores."
        )


    texto_valores = ",".join(
        f"{float(x):.9g}"
        for x in valores
    )


    return (
        "KMEANSRAW,"
        +
        texto_valores
        +
        "\r\n"
    )


# ============================================================
# LEER RESPUESTA UART
# ============================================================

def enviar_y_recibir(
    ser,
    comando
):
    ser.reset_input_buffer()


    ser.write(
        comando.encode(
            "ascii"
        )
    )


    ser.flush()


    inicio = time.time()


    while (
        time.time()
        -
        inicio
        <
        TIMEOUT_S
    ):
        linea = (
            ser.readline()
            .decode(
                "ascii",
                errors="replace"
            )
            .strip()
        )


        if not linea:
            continue


        if linea.startswith(
            "KMEANSRAW,"
        ):
            return linea


        if linea.startswith(
            "KMEANSRAW_ERROR"
        ):
            raise RuntimeError(
                f"STM32 respondió: {linea}"
            )


    raise TimeoutError(
        "No se recibió respuesta KMEANSRAW."
    )


# ============================================================
# PARSEAR RESPUESTA
#
# KMEANSRAW,cluster,score_d2,anomaly
# ============================================================

def parsear_respuesta(
    linea
):
    partes = linea.split(
        ","
    )


    if (
        len(partes) != 4
        or
        partes[0] != "KMEANSRAW"
    ):
        raise ValueError(
            f"Respuesta inválida: {linea}"
        )


    cluster = int(
        partes[1]
    )


    score = float(
        partes[2]
    )


    anomaly = int(
        partes[3]
    )


    return (
        cluster,
        score,
        anomaly
    )


# ============================================================
# PREPARAR DATAFRAME DE TEST
# ============================================================

def cargar_test():
    metadata = pd.read_csv(
        METADATA_PATH
    )


    metadata_test = (
        metadata[
            metadata[
                "split"
            ]
            ==
            "testing"
        ]
        .reset_index(
            drop=True
        )
    )


    scores_pc = pd.read_csv(
        SCORES_PC_PATH
    )


    if (
        len(metadata_test)
        !=
        len(scores_pc)
    ):
        raise ValueError(
            "metadata TEST y scores PC "
            "no tienen la misma longitud."
        )


    df = (
        metadata_test.copy()
    )


    df[
        "cluster_pc"
    ] = scores_pc[
        "cluster_kmeans"
    ].to_numpy()


    df[
        "score_pc"
    ] = scores_pc[
        "score_d2"
    ].to_numpy()


    if (
        "anomalia_P99"
        in
        scores_pc.columns
    ):
        df[
            "anomaly_pc"
        ] = (
            scores_pc[
                "anomalia_P99"
            ]
            .astype(int)
            .to_numpy()
        )

    else:
        raise ValueError(
            "No existe columna "
            "'anomalia_P99' "
            "en scores_test_STANDARD.csv"
        )


    return df


# ============================================================
# SELECCIONAR VENTANAS
# ============================================================

def seleccionar_filas(
    df
):
    if (
        USAR_ARCHIVO_Y_WINDOW
    ):
        seleccion = df[
            (
                df["file"]
                ==
                FILE_OBJETIVO
            )
            &
            (
                df["window"]
                ==
                WINDOW_OBJETIVO
            )
        ]


        if len(seleccion) != 1:
            raise ValueError(
                "La selección por file/window "
                "no produjo exactamente una fila."
            )


        return seleccion


    if MODO == "UNA":

        if len(INDICES_TEST) != 1:
            raise ValueError(
                "MODO UNA requiere "
                "exactamente un índice."
            )


        return df.iloc[
            INDICES_TEST
        ]


    elif MODO == "VARIAS":

        return df.iloc[
            INDICES_TEST
        ]


    elif MODO == "TODAS":

        return df.copy()


    else:

        raise ValueError(
            f"MODO inválido: {MODO}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 80
    )

    print(
        "VALIDACIÓN K-MEANS PC vs STM32"
    )

    print(
        "=" * 80
    )


    df = cargar_test()


    seleccion = (
        seleccionar_filas(
            df
        )
        .copy()
    )


    print(
        f"\nVentanas seleccionadas: "
        f"{len(seleccion)}"
    )


    print(
        f"Puerto: {PUERTO}"
    )


    resultados = []


    with serial.Serial(
        port=PUERTO,
        baudrate=BAUDRATE,
        timeout=0.5
    ) as ser:

        time.sleep(
            1.0
        )


        ser.reset_input_buffer()


        for contador, (
            indice,
            fila
        ) in enumerate(
            seleccion.iterrows(),
            start=1
        ):

            window = obtener_ventana(
                fila
            )


            comando = construir_comando(
                window
            )


            respuesta = enviar_y_recibir(
                ser,
                comando
            )


            (
                cluster_stm32,
                score_stm32,
                anomaly_stm32
            ) = parsear_respuesta(
                respuesta
            )


            cluster_pc = int(
                fila[
                    "cluster_pc"
                ]
            )


            score_pc = float(
                fila[
                    "score_pc"
                ]
            )


            anomaly_pc = int(
                fila[
                    "anomaly_pc"
                ]
            )


            error_abs_score = abs(
                score_pc
                -
                score_stm32
            )


            cluster_igual = int(
                cluster_pc
                ==
                cluster_stm32
            )


            anomaly_igual = int(
                anomaly_pc
                ==
                anomaly_stm32
            )


            resultados.append(
                {
                    "indice_test":
                        int(indice),

                    "file":
                        fila["file"],

                    "clase":
                        fila["class"],

                    "window":
                        int(
                            fila[
                                "window"
                            ]
                        ),

                    "start_sample":
                        int(
                            fila[
                                "start_sample"
                            ]
                        ),

                    "cluster_pc":
                        cluster_pc,

                    "cluster_stm32":
                        cluster_stm32,

                    "cluster_igual":
                        cluster_igual,

                    "score_pc":
                        score_pc,

                    "score_stm32":
                        score_stm32,

                    "error_abs_score":
                        error_abs_score,

                    "anomaly_pc":
                        anomaly_pc,

                    "anomaly_stm32":
                        anomaly_stm32,

                    "anomaly_igual":
                        anomaly_igual,
                }
            )


            print()

            print(
                f"[{contador}/{len(seleccion)}] "
                f"índice test = {indice}"
            )


            print(
                f"Archivo : "
                f"{fila['file']}"
            )


            print(
                f"Clase   : "
                f"{fila['class']}"
            )


            print(
                f"Window  : "
                f"{int(fila['window'])}"
            )


            print(
                f"PC      : "
                f"cluster={cluster_pc}, "
                f"score={score_pc:.9f}, "
                f"anomaly={anomaly_pc}"
            )


            print(
                f"STM32   : "
                f"cluster={cluster_stm32}, "
                f"score={score_stm32:.9f}, "
                f"anomaly={anomaly_stm32}"
            )


            print(
                f"|error score| = "
                f"{error_abs_score:.9e}"
            )


            time.sleep(
                PAUSA_ENTRE_COMANDOS_S
            )


    # ========================================================
    # DATAFRAME RESULTADOS
    # ========================================================

    df_resultados = pd.DataFrame(
        resultados
    )


    # ========================================================
    # RESUMEN
    # ========================================================

    total = len(
        df_resultados
    )


    clusters_iguales = int(
        df_resultados[
            "cluster_igual"
        ].sum()
    )


    anomaly_iguales = int(
        df_resultados[
            "anomaly_igual"
        ].sum()
    )


    mae_score = float(
        df_resultados[
            "error_abs_score"
        ].mean()
    )


    max_error_score = float(
        df_resultados[
            "error_abs_score"
        ].max()
    )


    anomalías_pc = int(
        df_resultados[
            "anomaly_pc"
        ].sum()
    )


    anomalías_stm32 = int(
        df_resultados[
            "anomaly_stm32"
        ].sum()
    )


    print()

    print(
        "=" * 80
    )

    print(
        "RESUMEN"
    )

    print(
        "=" * 80
    )


    print(
        f"Ventanas procesadas       : "
        f"{total}"
    )


    print(
        f"Clusters iguales          : "
        f"{clusters_iguales}/{total}"
    )


    print(
        f"Flags anomaly iguales     : "
        f"{anomaly_iguales}/{total}"
    )


    print(
        f"Anomalías PC              : "
        f"{anomalías_pc}"
    )


    print(
        f"Anomalías STM32           : "
        f"{anomalías_stm32}"
    )


    print(
        f"MAE score                 : "
        f"{mae_score:.9e}"
    )


    print(
        f"Máximo error score        : "
        f"{max_error_score:.9e}"
    )


    # ========================================================
    # GUARDAR
    # ========================================================

    output_csv = (
        OUTPUT_DIR
        /
        f"validacion_kmeans_{MODO.lower()}.csv"
    )


    df_resultados.to_csv(
        output_csv,
        index=False
    )


    resumen_txt = (
        OUTPUT_DIR
        /
        f"resumen_kmeans_{MODO.lower()}.txt"
    )


    texto = f"""
VALIDACION K-MEANS PC vs STM32
========================================

Modo:
{MODO}

Ventanas procesadas:
{total}

Clusters iguales:
{clusters_iguales}/{total}

Flags anomaly iguales:
{anomaly_iguales}/{total}

Anomalias PC:
{anomalías_pc}

Anomalias STM32:
{anomalías_stm32}

MAE score:
{mae_score:.9e}

Max error score:
{max_error_score:.9e}
"""


    resumen_txt.write_text(
        texto,
        encoding="utf-8"
    )


    print()

    print(
        f"CSV guardado en:\n"
        f"{output_csv}"
    )


    print()

    print(
        f"Resumen guardado en:\n"
        f"{resumen_txt}"
    )


if __name__ == "__main__":
    main()
