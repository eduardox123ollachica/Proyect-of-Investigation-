from pathlib import Path
import json
import time
import sys

CODIGO_DIR = Path(__file__).resolve().parents[2]
if str(CODIGO_DIR) not in sys.path:
    sys.path.insert(0, str(CODIGO_DIR))

from comun.rutas_proyecto import DATOS_ARCINIEGAS_PROCESADO, DATOS_ARCINIEGAS_RAW, MODELOS_ARCINIEGAS, RESULTADOS_E4, RESULTADOS_KMEANS

import numpy as np
import pandas as pd
import serial
import tensorflow as tf


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

DATASET_TEST_DIR = DATOS_ARCINIEGAS_RAW / "testing"
PREPROCESADO_DIR = DATOS_ARCINIEGAS_PROCESADO
KMEANS_DIR = RESULTADOS_KMEANS

MODELO_PATH = MODELOS_ARCINIEGAS / "e4" / "firmware" / "modelo_arciniegas_e4_int8.tflite"

OUTPUT_DIR = RESULTADOS_E4 / "validacion_rawall"

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
# MODO
#
# "UNA"
# "VARIAS"
# "TODAS"
# ============================================================

MODO = "TODAS"


# ============================================================
# ÍNDICES TEST
#
# UNA:
# INDICES_TEST = [0]
#
# VARIAS:
# INDICES_TEST = [0, 240, 500]
#
# TODAS:
# la lista se ignora
# ============================================================

INDICES_TEST = [
    0
]


# ============================================================
# OPCIONAL:
# seleccionar por archivo + window
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

X_TEST_PATH = (
    PREPROCESADO_DIR
    / "X_test.npy"
)

Y_TEST_PATH = (
    PREPROCESADO_DIR
    / "y_test.npy"
)

SCALER_MEAN_PATH = (
    KMEANS_DIR
    / "scaler_mean_STANDARD.npy"
)

SCALER_SCALE_PATH = (
    KMEANS_DIR
    / "scaler_scale_STANDARD.npy"
)

CENTROIDES_PATH = (
    KMEANS_DIR
    / "centroides_STANDARD.npy"
)


# ============================================================
# CONSTANTES
# ============================================================

WINDOW_SIZE = 16
AXIS_COUNT = 3

KMEANS_THRESHOLD = np.float32(
    21.850407
)


# ============================================================
# CARGAR JSON RAW
# ============================================================

def cargar_json_raw(filepath: Path):

    with open(
        filepath,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)


    payload = data["payload"]


    values = np.asarray(
        payload["values"],
        dtype=np.float64
    )


    if values.shape[1] != 3:
        raise ValueError(
            f"Forma inesperada: {values.shape}"
        )


    return values


# ============================================================
# RECONSTRUIR VENTANA RAW
# ============================================================

def obtener_ventana(fila):

    archivo = (
        DATASET_TEST_DIR
        /
        str(fila["file"])
    )


    values = cargar_json_raw(
        archivo
    )


    inicio = int(
        fila["start_sample"]
    )


    fin = int(
        fila["end_sample_exclusive"]
    )


    window = values[
        inicio:fin,
        :
    ]


    if window.shape != (
        WINDOW_SIZE,
        AXIS_COUNT
    ):
        raise ValueError(
            f"Ventana inválida: {window.shape}"
        )


    return window


# ============================================================
# TFLITE
# ============================================================

def cargar_modelo_tflite():

    modelo_bytes = (
        MODELO_PATH
        .read_bytes()
    )


    interpreter = tf.lite.Interpreter(
        model_content=modelo_bytes
    )


    interpreter.allocate_tensors()


    input_details = (
        interpreter
        .get_input_details()[0]
    )


    output_details = (
        interpreter
        .get_output_details()[0]
    )


    return (
        interpreter,
        input_details,
        output_details
    )


# ============================================================
# INFERENCIA INT8 PC
# ============================================================

def inferencia_pc(
    interpreter,
    input_details,
    output_details,
    features
):

    input_scale, input_zero_point = (
        input_details["quantization"]
    )


    output_scale, output_zero_point = (
        output_details["quantization"]
    )


    x = np.asarray(
        features,
        dtype=np.float32
    )


    q = np.round(
        x / input_scale
        +
        input_zero_point
    )


    q = np.clip(
        q,
        -128,
        127
    ).astype(
        np.int8
    )


    q = q.reshape(
        1,
        -1
    )


    interpreter.set_tensor(
        input_details["index"],
        q
    )


    interpreter.invoke()


    output_q = interpreter.get_tensor(
        output_details["index"]
    )[0]


    probs = (
        (
            output_q.astype(
                np.float32
            )
            -
            output_zero_point
        )
        *
        output_scale
    )


    clase = int(
        np.argmax(
            probs
        )
    )


    return (
        clase,
        probs
    )


# ============================================================
# K-MEANS PC FLOAT32
# ============================================================

def kmeans_pc(
    features,
    mean,
    scale,
    centroids
):

    x = np.asarray(
        features,
        dtype=np.float32
    )


    z = (
        x
        -
        mean
    ) / scale


    mejor_score = np.float32(
        np.inf
    )

    mejor_cluster = -1


    for k in range(
        centroids.shape[0]
    ):

        suma = np.float32(
            0.0
        )


        for j in range(
            centroids.shape[1]
        ):

            d = np.float32(
                z[j]
                -
                centroids[k, j]
            )


            suma = np.float32(
                suma
                +
                d * d
            )


        if suma < mejor_score:

            mejor_score = suma
            mejor_cluster = k


    anomaly = int(
        mejor_score
        >
        KMEANS_THRESHOLD
    )


    return (
        mejor_cluster,
        float(mejor_score),
        anomaly
    )


# ============================================================
# CONSTRUIR RAWALL
# ============================================================

def construir_comando_rawall(
    window
):

    valores = (
        window
        .reshape(-1)
    )


    texto = ",".join(
        f"{float(x):.9g}"
        for x in valores
    )


    return (
        "RAWALL,"
        +
        texto
        +
        "\r\n"
    )


# ============================================================
# UART
# ============================================================

def enviar_rawall(
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
            "RAWALL,"
        ):
            return linea


        if linea.startswith(
            "RAWALL_ERROR"
        ):
            raise RuntimeError(
                linea
            )


    raise TimeoutError(
        "No se recibió RAWALL."
    )


# ============================================================
# PARSEAR RESPUESTA
#
# RAWALL,
# clase,
# p0,p1,p2,
# cluster,
# score,
# anomaly
# ============================================================

def parsear_rawall(
    linea
):

    partes = linea.split(
        ","
    )


    if (
        len(partes) != 8
        or
        partes[0] != "RAWALL"
    ):
        raise ValueError(
            f"Respuesta inválida:\n{linea}"
        )


    clase = int(
        partes[1]
    )


    probs = np.asarray(
        [
            float(partes[2]),
            float(partes[3]),
            float(partes[4]),
        ],
        dtype=np.float64
    )


    cluster = int(
        partes[5]
    )


    score = float(
        partes[6]
    )


    anomaly = int(
        partes[7]
    )


    return (
        clase,
        probs,
        cluster,
        score,
        anomaly
    )


# ============================================================
# CARGAR DATA
# ============================================================

def cargar_datos():

    metadata = pd.read_csv(
        METADATA_PATH
    )


    metadata_test = (
        metadata[
            metadata["split"]
            ==
            "testing"
        ]
        .reset_index(
            drop=True
        )
    )


    X_test = np.load(
        X_TEST_PATH
    )


    y_test = np.load(
        Y_TEST_PATH
    )


    if len(metadata_test) != len(X_test):
        raise ValueError(
            "metadata y X_test no coinciden."
        )


    return (
        metadata_test,
        X_test,
        y_test
    )


# ============================================================
# SELECCIÓN
# ============================================================

def seleccionar_indices(
    metadata
):

    if USAR_ARCHIVO_Y_WINDOW:

        mask = (
            (
                metadata["file"]
                ==
                FILE_OBJETIVO
            )
            &
            (
                metadata["window"]
                ==
                WINDOW_OBJETIVO
            )
        )


        indices = np.where(
            mask.to_numpy()
        )[0]


        if len(indices) != 1:
            raise ValueError(
                "No se encontró exactamente "
                "una ventana."
            )


        return indices


    if MODO == "UNA":

        if len(INDICES_TEST) != 1:
            raise ValueError(
                "MODO UNA requiere "
                "un índice."
            )


        return np.asarray(
            INDICES_TEST,
            dtype=int
        )


    elif MODO == "VARIAS":

        return np.asarray(
            INDICES_TEST,
            dtype=int
        )


    elif MODO == "TODAS":

        return np.arange(
            len(metadata)
        )


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
        "VALIDACIÓN RAWALL PC vs STM32"
    )

    print(
        "=" * 80
    )


    (
        metadata,
        X_test,
        y_test
    ) = cargar_datos()


    mean = np.load(
        SCALER_MEAN_PATH
    ).astype(
        np.float32
    )


    scale = np.load(
        SCALER_SCALE_PATH
    ).astype(
        np.float32
    )


    centroids = np.load(
        CENTROIDES_PATH
    ).astype(
        np.float32
    )


    (
        interpreter,
        input_details,
        output_details
    ) = cargar_modelo_tflite()


    indices = seleccionar_indices(
        metadata
    )


    print(
        f"\nVentanas seleccionadas: "
        f"{len(indices)}"
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


        for n, indice in enumerate(
            indices,
            start=1
        ):

            fila = metadata.iloc[
                indice
            ]


            features_pc = np.asarray(
                X_test[indice],
                dtype=np.float32
            )


            # =================================================
            # DNN PC
            # =================================================

            (
                clase_pc,
                probs_pc
            ) = inferencia_pc(
                interpreter,
                input_details,
                output_details,
                features_pc
            )


            # =================================================
            # KMEANS PC
            # =================================================

            (
                cluster_pc,
                score_pc,
                anomaly_pc
            ) = kmeans_pc(
                features_pc,
                mean,
                scale,
                centroids
            )


            # =================================================
            # RAW WINDOW
            # =================================================

            window = obtener_ventana(
                fila
            )


            comando = (
                construir_comando_rawall(
                    window
                )
            )


            respuesta = enviar_rawall(
                ser,
                comando
            )


            (
                clase_stm32,
                probs_stm32,
                cluster_stm32,
                score_stm32,
                anomaly_stm32
            ) = parsear_rawall(
                respuesta
            )


            # =================================================
            # COMPARACIONES
            # =================================================

            clase_igual = int(
                clase_pc
                ==
                clase_stm32
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


            error_probs = np.abs(
                probs_pc
                -
                probs_stm32
            )


            mae_probs = float(
                np.mean(
                    error_probs
                )
            )


            max_error_probs = float(
                np.max(
                    error_probs
                )
            )


            error_score = abs(
                score_pc
                -
                score_stm32
            )


            resultados.append(
                {
                    "indice_test":
                        int(indice),

                    "file":
                        fila["file"],

                    "class":
                        fila["class"],

                    "label_real":
                        int(y_test[indice]),

                    "window":
                        int(fila["window"]),

                    "clase_pc":
                        clase_pc,

                    "clase_stm32":
                        clase_stm32,

                    "clase_igual":
                        clase_igual,

                    "p0_pc":
                        float(probs_pc[0]),

                    "p1_pc":
                        float(probs_pc[1]),

                    "p2_pc":
                        float(probs_pc[2]),

                    "p0_stm32":
                        float(probs_stm32[0]),

                    "p1_stm32":
                        float(probs_stm32[1]),

                    "p2_stm32":
                        float(probs_stm32[2]),

                    "mae_probs":
                        mae_probs,

                    "max_error_probs":
                        max_error_probs,

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
                        error_score,

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
                f"[{n}/{len(indices)}] "
                f"índice={indice}"
            )


            print(
                f"Archivo : "
                f"{fila['file']}"
            )


            print(
                f"Clase real : "
                f"{int(y_test[indice])}"
            )


            print(
                f"DNN PC     : "
                f"clase={clase_pc}, "
                f"p={probs_pc}"
            )


            print(
                f"DNN STM32  : "
                f"clase={clase_stm32}, "
                f"p={probs_stm32}"
            )


            print(
                f"KMeans PC  : "
                f"cluster={cluster_pc}, "
                f"score={score_pc:.9f}, "
                f"anomaly={anomaly_pc}"
            )


            print(
                f"KMeans STM : "
                f"cluster={cluster_stm32}, "
                f"score={score_stm32:.9f}, "
                f"anomaly={anomaly_stm32}"
            )


            time.sleep(
                PAUSA_ENTRE_COMANDOS_S
            )


    # ========================================================
    # RESULTADOS
    # ========================================================

    df = pd.DataFrame(
        resultados
    )


    total = len(
        df
    )


    clases_iguales = int(
        df[
            "clase_igual"
        ].sum()
    )


    clusters_iguales = int(
        df[
            "cluster_igual"
        ].sum()
    )


    flags_iguales = int(
        df[
            "anomaly_igual"
        ].sum()
    )


    mae_prob_global = float(
        df[
            "mae_probs"
        ].mean()
    )


    max_prob_global = float(
        df[
            "max_error_probs"
        ].max()
    )


    mae_score = float(
        df[
            "error_abs_score"
        ].mean()
    )


    max_score = float(
        df[
            "error_abs_score"
        ].max()
    )


    accuracy_pc = float(
        np.mean(
            df["clase_pc"].to_numpy()
            ==
            df["label_real"].to_numpy()
        )
    )


    accuracy_stm32 = float(
        np.mean(
            df["clase_stm32"].to_numpy()
            ==
            df["label_real"].to_numpy()
        )
    )


    anomalies_pc = int(
        df[
            "anomaly_pc"
        ].sum()
    )


    anomalies_stm32 = int(
        df[
            "anomaly_stm32"
        ].sum()
    )


    print()

    print(
        "=" * 80
    )

    print(
        "RESUMEN RAWALL"
    )

    print(
        "=" * 80
    )


    print(
        f"Ventanas procesadas        : "
        f"{total}"
    )


    print(
        f"Clases DNN iguales         : "
        f"{clases_iguales}/{total}"
    )


    print(
        f"Accuracy DNN PC            : "
        f"{accuracy_pc * 100:.6f}%"
    )


    print(
        f"Accuracy DNN STM32         : "
        f"{accuracy_stm32 * 100:.6f}%"
    )


    print(
        f"MAE probabilidades         : "
        f"{mae_prob_global:.12f}"
    )


    print(
        f"Max error probabilidades   : "
        f"{max_prob_global:.12f}"
    )


    print(
        f"Clusters KMeans iguales    : "
        f"{clusters_iguales}/{total}"
    )


    print(
        f"Flags anomaly iguales      : "
        f"{flags_iguales}/{total}"
    )


    print(
        f"Anomalías PC               : "
        f"{anomalies_pc}"
    )


    print(
        f"Anomalías STM32            : "
        f"{anomalies_stm32}"
    )


    print(
        f"MAE score KMeans           : "
        f"{mae_score:.12e}"
    )


    print(
        f"Max error score KMeans     : "
        f"{max_score:.12e}"
    )


    # ========================================================
    # GUARDAR
    # ========================================================

    csv_path = (
        OUTPUT_DIR
        /
        f"rawall_{MODO.lower()}.csv"
    )


    df.to_csv(
        csv_path,
        index=False
    )


    resumen_path = (
        OUTPUT_DIR
        /
        f"rawall_{MODO.lower()}_resumen.txt"
    )


    resumen = f"""
VALIDACION RAWALL PC vs STM32
========================================

Modo:
{MODO}

Ventanas:
{total}

DNN
----------------------------------------
Clases iguales:
{clases_iguales}/{total}

Accuracy PC:
{accuracy_pc * 100:.6f} %

Accuracy STM32:
{accuracy_stm32 * 100:.6f} %

MAE probabilidades:
{mae_prob_global:.12f}

Max error probabilidades:
{max_prob_global:.12f}

KMEANS
----------------------------------------
Clusters iguales:
{clusters_iguales}/{total}

Flags anomaly iguales:
{flags_iguales}/{total}

Anomalias PC:
{anomalies_pc}

Anomalias STM32:
{anomalies_stm32}

MAE score:
{mae_score:.12e}

Max error score:
{max_score:.12e}
"""


    resumen_path.write_text(
        resumen,
        encoding="utf-8"
    )


    print()

    print(
        f"CSV:\n{csv_path}"
    )


    print()

    print(
        f"Resumen:\n{resumen_path}"
    )


if __name__ == "__main__":
    main()  
