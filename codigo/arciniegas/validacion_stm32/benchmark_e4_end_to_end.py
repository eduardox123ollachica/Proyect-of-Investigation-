from pathlib import Path
import json
import time
import sys

CODIGO_DIR = Path(__file__).resolve().parents[2]
if str(CODIGO_DIR) not in sys.path:
    sys.path.insert(0, str(CODIGO_DIR))

from comun.rutas_proyecto import DATOS_ARCINIEGAS_PROCESADO, DATOS_ARCINIEGAS_RAW, RESULTADOS_E4

import numpy as np
import pandas as pd
import serial


# ============================================================
# CONFIGURACIÓN
# ============================================================

DATASET_TEST_DIR = DATOS_ARCINIEGAS_RAW / "testing"
PREPROCESADO_DIR = DATOS_ARCINIEGAS_PROCESADO
OUTPUT_DIR = RESULTADOS_E4 / "benchmark_end_to_end"

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
# ARCHIVOS
# ============================================================

METADATA_PATH = (
    PREPROCESADO_DIR
    / "metadata_features.csv"
)

Y_TEST_PATH = (
    PREPROCESADO_DIR
    / "y_test.npy"
)


# ============================================================
# CONSTANTES
# ============================================================

WINDOW_SIZE = 16
AXIS_COUNT = 3

FS_HZ = 100.0
HOP_SIZE = 8

HOP_TIME_S = (
    HOP_SIZE / FS_HZ
)

HOP_TIME_US = (
    HOP_TIME_S * 1e6
)


# ============================================================
# CARGAR JSON
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
# RECONSTRUIR VENTANA
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
            f"Ventana inválida: "
            f"{window.shape}"
        )

    return window


# ============================================================
# CONSTRUIR COMANDO
# ============================================================

def construir_benchall(window):

    valores = (
        window
        .reshape(-1)
    )

    texto = ",".join(
        f"{float(x):.9g}"
        for x in valores
    )

    return (
        "BENCHALL,"
        +
        texto
        +
        "\r\n"
    )


# ============================================================
# UART
# ============================================================

def enviar_benchall(
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
            "BENCHALL,"
        ):
            return linea

        if linea.startswith(
            "BENCHALL_ERROR"
        ):
            raise RuntimeError(
                linea
            )

    raise TimeoutError(
        "No se recibió respuesta BENCHALL."
    )


# ============================================================
# PARSEAR
#
# BENCHALL,
# clase,
# anomaly,
# ciclos_features,
# ciclos_dnn,
# ciclos_kmeans,
# ciclos_total
# ============================================================

def parsear_benchall(
    linea
):

    partes = linea.split(
        ","
    )

    if (
        len(partes) != 7
        or
        partes[0] != "BENCHALL"
    ):
        raise ValueError(
            f"Respuesta inválida:\n"
            f"{linea}"
        )

    clase = int(
        partes[1]
    )

    anomaly = int(
        partes[2]
    )

    ciclos_features = int(
        partes[3]
    )

    ciclos_dnn = int(
        partes[4]
    )

    ciclos_kmeans = int(
        partes[5]
    )

    ciclos_total = int(
        partes[6]
    )

    return (
        clase,
        anomaly,
        ciclos_features,
        ciclos_dnn,
        ciclos_kmeans,
        ciclos_total
    )


# ============================================================
# BENCHINFO
# ============================================================

def obtener_hclk(
    ser
):

    ser.reset_input_buffer()

    ser.write(
        b"BENCHINFO\r\n"
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
            "BENCHINFO,"
        ):

            partes = linea.split(
                ","
            )

            if len(partes) != 3:
                raise ValueError(
                    f"BENCHINFO inválido: "
                    f"{linea}"
                )

            hclk_hz = int(
                partes[1]
            )

            dwt_enabled = int(
                partes[2]
            )

            return (
                hclk_hz,
                dwt_enabled
            )

    raise TimeoutError(
        "No se recibió BENCHINFO."
    )


# ============================================================
# ESTADÍSTICOS
# ============================================================

def estadisticos(
    datos
):

    datos = np.asarray(
        datos,
        dtype=np.float64
    )

    return {
        "mean":
            float(np.mean(datos)),

        "median":
            float(np.median(datos)),

        "min":
            float(np.min(datos)),

        "max":
            float(np.max(datos)),

        "std":
            float(np.std(datos)),

        "p95":
            float(
                np.percentile(
                    datos,
                    95
                )
            ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 80
    )

    print(
        "BENCHMARK FINAL STM32 - BENCHALL"
    )

    print(
        "=" * 80
    )


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

    y_test = np.load(
        Y_TEST_PATH
    )


    if (
        len(metadata_test)
        !=
        len(y_test)
    ):
        raise ValueError(
            "metadata TEST y y_test "
            "no coinciden."
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


        (
            hclk_hz,
            dwt_enabled
        ) = obtener_hclk(
            ser
        )


        print(
            f"\nHCLK       : "
            f"{hclk_hz} Hz"
        )


        print(
            f"DWT enabled: "
            f"{dwt_enabled}"
        )


        if dwt_enabled != 1:
            raise RuntimeError(
                "DWT no está habilitado."
            )


        for indice, fila in (
            metadata_test
            .iterrows()
        ):

            window = obtener_ventana(
                fila
            )


            comando = construir_benchall(
                window
            )


            respuesta = enviar_benchall(
                ser,
                comando
            )


            (
                clase,
                anomaly,
                ciclos_features,
                ciclos_dnn,
                ciclos_kmeans,
                ciclos_total
            ) = parsear_benchall(
                respuesta
            )


            factor_us = (
                1e6
                /
                hclk_hz
            )


            us_features = (
                ciclos_features
                *
                factor_us
            )


            us_dnn = (
                ciclos_dnn
                *
                factor_us
            )


            us_kmeans = (
                ciclos_kmeans
                *
                factor_us
            )


            us_total = (
                ciclos_total
                *
                factor_us
            )


            resultados.append(
                {
                    "indice_test":
                        indice,

                    "file":
                        fila["file"],

                    "class":
                        fila["class"],

                    "label_real":
                        int(
                            y_test[indice]
                        ),

                    "window":
                        int(
                            fila["window"]
                        ),

                    "clase_stm32":
                        clase,

                    "anomaly_stm32":
                        anomaly,

                    "cycles_features":
                        ciclos_features,

                    "cycles_dnn":
                        ciclos_dnn,

                    "cycles_kmeans":
                        ciclos_kmeans,

                    "cycles_total":
                        ciclos_total,

                    "us_features":
                        us_features,

                    "us_dnn":
                        us_dnn,

                    "us_kmeans":
                        us_kmeans,

                    "us_total":
                        us_total,
                }
            )


            if (
                indice % 100
                ==
                0
            ):
                print(
                    f"{indice + 1}/"
                    f"{len(metadata_test)}"
                )


            time.sleep(
                PAUSA_ENTRE_COMANDOS_S
            )


    # ========================================================
    # DATAFRAME
    # ========================================================

    df = pd.DataFrame(
        resultados
    )


    # ========================================================
    # ACCURACY FUNCIONAL
    # ========================================================

    accuracy = float(
        np.mean(
            df["clase_stm32"].to_numpy()
            ==
            df["label_real"].to_numpy()
        )
    )


    anomalies = int(
        df[
            "anomaly_stm32"
        ].sum()
    )


    # ========================================================
    # ESTADÍSTICOS
    # ========================================================

    stats_features = estadisticos(
        df["us_features"]
    )

    stats_dnn = estadisticos(
        df["us_dnn"]
    )

    stats_kmeans = estadisticos(
        df["us_kmeans"]
    )

    stats_total = estadisticos(
        df["us_total"]
    )


    # ========================================================
    # PORCENTAJES DEL PIPELINE
    # ========================================================

    porcentaje_features = (
        stats_features["mean"]
        /
        stats_total["mean"]
        *
        100.0
    )


    porcentaje_dnn = (
        stats_dnn["mean"]
        /
        stats_total["mean"]
        *
        100.0
    )


    porcentaje_kmeans = (
        stats_kmeans["mean"]
        /
        stats_total["mean"]
        *
        100.0
    )


    # ========================================================
    # OCUPACIÓN RESPECTO AL HOP
    # ========================================================

    ocupacion_media = (
        stats_total["mean"]
        /
        HOP_TIME_US
        *
        100.0
    )


    ocupacion_max = (
        stats_total["max"]
        /
        HOP_TIME_US
        *
        100.0
    )


    ocupacion_p95 = (
        stats_total["p95"]
        /
        HOP_TIME_US
        *
        100.0
    )


    # ========================================================
    # IMPRIMIR
    # ========================================================

    print()

    print(
        "=" * 80
    )

    print(
        "RESUMEN BENCHALL"
    )

    print(
        "=" * 80
    )


    print(
        f"HCLK                         : "
        f"{hclk_hz} Hz"
    )


    print(
        f"Ventanas                     : "
        f"{len(df)}"
    )


    print(
        f"Accuracy STM32               : "
        f"{accuracy * 100:.6f}%"
    )


    print(
        f"Anomalías STM32              : "
        f"{anomalies}"
    )


    print()


    def imprimir_stats(
        nombre,
        s
    ):

        print(
            f"{nombre}"
        )

        print(
            f"  mean   : "
            f"{s['mean']:.6f} us"
        )

        print(
            f"  median : "
            f"{s['median']:.6f} us"
        )

        print(
            f"  min    : "
            f"{s['min']:.6f} us"
        )

        print(
            f"  max    : "
            f"{s['max']:.6f} us"
        )

        print(
            f"  std    : "
            f"{s['std']:.6f} us"
        )

        print(
            f"  p95    : "
            f"{s['p95']:.6f} us"
        )

        print()


    imprimir_stats(
        "FEATURE EXTRACTION",
        stats_features
    )


    imprimir_stats(
        "DNN",
        stats_dnn
    )


    imprimir_stats(
        "KMEANS",
        stats_kmeans
    )


    imprimir_stats(
        "PIPELINE TOTAL",
        stats_total
    )


    print(
        "DISTRIBUCIÓN MEDIA DEL TIEMPO"
    )


    print(
        f"  Features : "
        f"{porcentaje_features:.3f}%"
    )


    print(
        f"  DNN      : "
        f"{porcentaje_dnn:.3f}%"
    )


    print(
        f"  KMeans   : "
        f"{porcentaje_kmeans:.3f}%"
    )


    print()


    print(
        f"Hop temporal                 : "
        f"{HOP_TIME_US / 1000:.3f} ms"
    )


    print(
        f"Ocupación media              : "
        f"{ocupacion_media:.3f}%"
    )


    print(
        f"Ocupación P95                : "
        f"{ocupacion_p95:.3f}%"
    )


    print(
        f"Ocupación máxima             : "
        f"{ocupacion_max:.3f}%"
    )


    # ========================================================
    # GUARDAR CSV
    # ========================================================

    csv_path = (
        OUTPUT_DIR
        /
        "benchall_1332.csv"
    )


    df.to_csv(
        csv_path,
        index=False
    )


    # ========================================================
    # GUARDAR RESUMEN
    # ========================================================

    resumen_path = (
        OUTPUT_DIR
        /
        "benchall_resumen.txt"
    )


    with open(
        resumen_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "BENCHMARK FINAL STM32 - BENCHALL\n"
        )

        f.write(
            "=" * 70
            +
            "\n\n"
        )


        f.write(
            f"HCLK: "
            f"{hclk_hz} Hz\n"
        )


        f.write(
            f"Ventanas: "
            f"{len(df)}\n"
        )


        f.write(
            f"Accuracy STM32: "
            f"{accuracy * 100:.6f}%\n"
        )


        f.write(
            f"Anomalias STM32: "
            f"{anomalies}\n\n"
        )


        for nombre, s in [
            (
                "FEATURE EXTRACTION",
                stats_features
            ),
            (
                "DNN",
                stats_dnn
            ),
            (
                "KMEANS",
                stats_kmeans
            ),
            (
                "PIPELINE TOTAL",
                stats_total
            ),
        ]:

            f.write(
                nombre
                +
                "\n"
            )

            for clave in [
                "mean",
                "median",
                "min",
                "max",
                "std",
                "p95",
            ]:

                f.write(
                    f"{clave}: "
                    f"{s[clave]:.6f} us\n"
                )

            f.write(
                "\n"
            )


        f.write(
            "DISTRIBUCION MEDIA\n"
        )


        f.write(
            f"Features: "
            f"{porcentaje_features:.3f}%\n"
        )


        f.write(
            f"DNN: "
            f"{porcentaje_dnn:.3f}%\n"
        )


        f.write(
            f"KMeans: "
            f"{porcentaje_kmeans:.3f}%\n\n"
        )


        f.write(
            f"Hop: "
            f"{HOP_TIME_US / 1000:.3f} ms\n"
        )


        f.write(
            f"Ocupacion media: "
            f"{ocupacion_media:.3f}%\n"
        )


        f.write(
            f"Ocupacion P95: "
            f"{ocupacion_p95:.3f}%\n"
        )


        f.write(
            f"Ocupacion maxima: "
            f"{ocupacion_max:.3f}%\n"
        )


    print()

    print(
        f"CSV guardado:\n"
        f"{csv_path}"
    )


    print()

    print(
        f"Resumen guardado:\n"
        f"{resumen_path}"
    )


if __name__ == "__main__":
    main()
