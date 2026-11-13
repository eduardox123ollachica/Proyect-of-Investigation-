from pathlib import Path
import json
import time
import sys

CODIGO_DIR = Path(__file__).resolve().parents[2]
if str(CODIGO_DIR) not in sys.path:
    sys.path.insert(0, str(CODIGO_DIR))

from comun.rutas_proyecto import DATOS_ARCINIEGAS_PROCESADO, DATOS_ARCINIEGAS_RAW, RESULTADOS_E1

import numpy as np
import pandas as pd
import serial


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

BASE_PREPROCESADO = DATOS_ARCINIEGAS_PROCESADO
BASE_DATASET_TEST = DATOS_ARCINIEGAS_RAW / "testing"
OUTPUT_DIR = RESULTADOS_E1 / "benchmark_inferencia"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# ARCHIVOS
# ============================================================

METADATA_PATH = (
    BASE_PREPROCESADO
    / "metadata_features.csv"
)

Y_TEST_PATH = (
    BASE_PREPROCESADO
    / "y_test.npy"
)


# ============================================================
# UART
# ============================================================

SERIAL_PORT = "COM4"

BAUDRATE = 115200

TIMEOUT_S = 2.0

REINTENTOS = 3

PAUSA_ENTRE_COMANDOS_S = 0.01


# ============================================================
# PIPELINE
# ============================================================

WINDOW_SIZE = 16

AXIS_COUNT = 3

RAW_VALUES = (
    WINDOW_SIZE
    *
    AXIS_COUNT
)


CLASES = {
    0: "DETENIDO",
    1: "VELOCIDAD MEDIA",
    2: "VELOCIDAD ALTA",
}


# ============================================================
# SALIDAS
# ============================================================

CSV_RESULTADOS = (
    OUTPUT_DIR
    / "resultados_1332_benchmark.csv"
)

CSV_TEMPORAL = (
    OUTPUT_DIR
    / "resultados_benchmark_temporal.csv"
)

TXT_RESUMEN = (
    OUTPUT_DIR
    / "resumen_benchmark.txt"
)


# ============================================================
# BUSCAR JSON
# ============================================================

def buscar_json(nombre_archivo):

    ruta_directa = (
        BASE_DATASET_TEST
        / nombre_archivo
    )

    if ruta_directa.exists():
        return ruta_directa


    encontrados = list(
        BASE_DATASET_TEST.rglob(
            nombre_archivo
        )
    )

    if len(encontrados) == 0:

        raise FileNotFoundError(
            f"No se encontró:\n{nombre_archivo}"
        )


    return encontrados[0]


# ============================================================
# CACHE DE JSON
# ============================================================

json_cache = {}


def cargar_values(nombre_archivo):

    if nombre_archivo in json_cache:

        return json_cache[
            nombre_archivo
        ]


    ruta = buscar_json(
        nombre_archivo
    )


    with open(
        ruta,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)


    values = np.asarray(
        data["payload"]["values"],
        dtype=np.float32
    )


    if (
        values.ndim != 2
        or
        values.shape[1] < 3
    ):

        raise ValueError(
            f"Formato inesperado:\n{ruta}"
        )


    values = values[:, :3]


    json_cache[
        nombre_archivo
    ] = values


    return values


# ============================================================
# CONSTRUIR BENCHRAW
# ============================================================

def construir_benchraw(window):

    if window.shape != (
        WINDOW_SIZE,
        AXIS_COUNT
    ):

        raise ValueError(
            f"Shape incorrecto: {window.shape}"
        )


    valores = []


    for n in range(
        WINDOW_SIZE
    ):

        for axis in range(
            AXIS_COUNT
        ):

            valores.append(
                f"{float(window[n, axis]):.9f}"
            )


    if len(valores) != RAW_VALUES:

        raise RuntimeError(
            f"Se generaron {len(valores)} valores "
            f"en lugar de {RAW_VALUES}."
        )


    return (
        "BENCHRAW,"
        +
        ",".join(valores)
    )


# ============================================================
# ENVIAR COMANDO SIMPLE
# ============================================================

def enviar_linea(
    ser,
    comando
):

    ser.reset_input_buffer()


    ser.write(
        (
            comando
            +
            "\r\n"
        ).encode(
            "ascii"
        )
    )

    ser.flush()


    respuesta = (
        ser.readline()
        .decode(
            "ascii",
            errors="replace"
        )
        .strip()
    )


    return respuesta


def obtener_benchinfo(
    ser
):

    # ========================================================
    # LIMPIAR MENSAJES DE ARRANQUE
    # ========================================================

    time.sleep(
        0.5
    )

    while ser.in_waiting > 0:

        linea = (
            ser.readline()
            .decode(
                "ascii",
                errors="replace"
            )
            .strip()
        )

        if linea:

            print(
                f"[STM32 inicio] {linea}"
            )


    # ========================================================
    # ENVIAR BENCHINFO
    # ========================================================

    ser.write(
        b"BENCHINFO\r\n"
    )

    ser.flush()


    # ========================================================
    # BUSCAR LA RESPUESTA CORRECTA
    #
    # Ignoramos:
    #
    # TINYML,0
    # STM32 listo para recibir datos
    #
    # u otras líneas residuales.
    # ========================================================

    tiempo_limite = (
        time.perf_counter()
        +
        3.0
    )


    ultima_respuesta = ""


    while (
        time.perf_counter()
        <
        tiempo_limite
    ):

        respuesta = (
            ser.readline()
            .decode(
                "ascii",
                errors="replace"
            )
            .strip()
        )


        if not respuesta:

            continue


        ultima_respuesta = (
            respuesta
        )


        print(
            f"[STM32] {respuesta}"
        )


        if not respuesta.startswith(
            "BENCHINFO,"
        ):

            continue


        partes = (
            respuesta.split(",")
        )


        if len(partes) != 3:

            raise RuntimeError(
                "Formato BENCHINFO inválido:\n"
                f"{respuesta}"
            )


        try:

            hclk_hz = int(
                partes[1]
            )

            dwt_enabled = int(
                partes[2]
            )

        except ValueError:

            raise RuntimeError(
                "Valores BENCHINFO inválidos:\n"
                f"{respuesta}"
            )


        if hclk_hz <= 0:

            raise RuntimeError(
                f"HCLK inválido: {hclk_hz}"
            )


        if dwt_enabled != 1:

            raise RuntimeError(
                "DWT->CYCCNT no está habilitado.\n"
                f"Respuesta: {respuesta}"
            )


        return (
            hclk_hz,
            dwt_enabled,
            respuesta
        )


    raise RuntimeError(
        "No se recibió BENCHINFO válido.\n"
        f"Última línea recibida: "
        f"{ultima_respuesta}"
    )   

# ============================================================
# ENVIAR BENCHRAW
# ============================================================

def enviar_benchraw(
    ser,
    comando
):

    ultima_respuesta = ""


    for intento in range(
        1,
        REINTENTOS + 1
    ):

        respuesta = enviar_linea(
            ser,
            comando
        )


        ultima_respuesta = respuesta


        if respuesta.startswith(
            "BENCHRAW,"
        ):

            partes = (
                respuesta.split(",")
            )


            if len(partes) != 5:

                time.sleep(
                    0.05
                )

                continue


            try:

                clase = int(
                    partes[1]
                )

                ciclos_features = int(
                    partes[2]
                )

                ciclos_inferencia = int(
                    partes[3]
                )

                ciclos_total = int(
                    partes[4]
                )


            except ValueError:

                time.sleep(
                    0.05
                )

                continue


            return (
                clase,
                ciclos_features,
                ciclos_inferencia,
                ciclos_total,
                respuesta,
                intento,
            )


        time.sleep(
            0.05
        )


    raise RuntimeError(
        "No se obtuvo BENCHRAW válido.\n"
        f"Última respuesta:\n{ultima_respuesta}"
    )


# ============================================================
# ESTADÍSTICAS
# ============================================================

def calcular_estadisticas(
    datos
):

    datos = np.asarray(
        datos,
        dtype=np.float64
    )


    return {
        "media":
            float(
                np.mean(datos)
            ),

        "mediana":
            float(
                np.median(datos)
            ),

        "minimo":
            float(
                np.min(datos)
            ),

        "maximo":
            float(
                np.max(datos)
            ),

        "desviacion":
            float(
                np.std(
                    datos,
                    ddof=1
                )
            ),

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

    print("=" * 80)

    print(
        "EXPERIMENTO 4 - BENCHMARK STM32"
    )

    print(
        "PIPELINE END-TO-END CON DWT->CYCCNT"
    )

    print("=" * 80)


    # ========================================================
    # METADATA
    # ========================================================

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


    y_test = (
        np.load(
            Y_TEST_PATH
        )
        .reshape(-1)
    )


    if (
        len(metadata_test)
        !=
        len(y_test)
    ):

        raise ValueError(
            "metadata testing y y_test "
            "no coinciden."
        )


    total_ventanas = (
        len(metadata_test)
    )


    print()

    print(
        f"Ventanas testing : "
        f"{total_ventanas}"
    )


    # ========================================================
    # UART
    # ========================================================

    resultados = []


    with serial.Serial(
        port=SERIAL_PORT,
        baudrate=BAUDRATE,
        timeout=TIMEOUT_S
    ) as ser:


        time.sleep(
            0.5
        )

        ser.reset_output_buffer()

        (
            hclk_hz,
            dwt_enabled,
            respuesta_info,
        ) = obtener_benchinfo(
            ser
        )   


        print()

        print("=" * 80)

        print("CONFIGURACIÓN STM32")

        print("=" * 80)


        print(
            f"HCLK              : "
            f"{hclk_hz} Hz"
        )


        print(
            f"HCLK              : "
            f"{hclk_hz / 1e6:.3f} MHz"
        )


        print(
            f"DWT habilitado    : "
            f"{dwt_enabled}"
        )


        print(
            f"Respuesta         : "
            f"{respuesta_info}"
        )


        print()

        print(
            "Iniciando benchmark..."
        )


        # ====================================================
        # 1332 VENTANAS
        # ====================================================

        for index in range(
            total_ventanas
        ):


            fila = (
                metadata_test
                .iloc[index]
            )


            archivo = str(
                fila["file"]
            )


            clase_nombre = str(
                fila["class"]
            )


            label_metadata = int(
                fila["label"]
            )


            label_test = int(
                y_test[index]
            )


            if (
                label_metadata
                !=
                label_test
            ):

                raise ValueError(
                    f"Etiqueta inconsistente "
                    f"en índice {index}."
                )


            window_index = int(
                fila["window"]
            )


            start = int(
                fila["start_sample"]
            )


            end = int(
                fila[
                    "end_sample_exclusive"
                ]
            )


            # ================================================
            # RECONSTRUIR VENTANA
            # ================================================

            values = cargar_values(
                archivo
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
                    f"Ventana {index} "
                    f"shape={window.shape}"
                )


            # ================================================
            # BENCHRAW
            # ================================================

            comando = (
                construir_benchraw(
                    window
                )
            )


            (
                clase_stm32,
                ciclos_features,
                ciclos_inferencia,
                ciclos_total,
                respuesta_raw,
                intentos,
            ) = enviar_benchraw(
                ser,
                comando
            )


            # ================================================
            # CONVERTIR CICLOS A TIEMPO
            # ================================================

            us_features = (
                ciclos_features
                *
                1_000_000.0
                /
                hclk_hz
            )


            us_inferencia = (
                ciclos_inferencia
                *
                1_000_000.0
                /
                hclk_hz
            )


            us_total = (
                ciclos_total
                *
                1_000_000.0
                /
                hclk_hz
            )


            correcto = (
                clase_stm32
                ==
                label_test
            )


            # ================================================
            # GUARDAR
            # ================================================

            resultado = {

                "index":
                    index,

                "file":
                    archivo,

                "window":
                    window_index,

                "start_sample":
                    start,

                "end_sample_exclusive":
                    end,

                "clase_real":
                    label_test,

                "nombre_real":
                    clase_nombre,

                "clase_stm32":
                    clase_stm32,

                "nombre_stm32":
                    CLASES.get(
                        clase_stm32,
                        "DESCONOCIDA"
                    ),

                "correcto":
                    correcto,

                "hclk_hz":
                    hclk_hz,

                "ciclos_features":
                    ciclos_features,

                "ciclos_inferencia":
                    ciclos_inferencia,

                "ciclos_total":
                    ciclos_total,

                "us_features":
                    us_features,

                "us_inferencia":
                    us_inferencia,

                "us_total":
                    us_total,

                "uart_intentos":
                    intentos,

                "comando_longitud":
                    len(comando),

                "respuesta_raw":
                    respuesta_raw,
            }


            resultados.append(
                resultado
            )


            procesadas = (
                index + 1
            )


            # ================================================
            # PROGRESO
            # ================================================

            if (
                procesadas == 1
                or
                procesadas % 25 == 0
                or
                procesadas == total_ventanas
            ):

                df_temp = pd.DataFrame(
                    resultados
                )


                accuracy_actual = (
                    df_temp[
                        "correcto"
                    ]
                    .mean()
                    *
                    100
                )


                media_total = (
                    df_temp[
                        "us_total"
                    ]
                    .mean()
                )


                print(
                    f"[{procesadas:4d}/"
                    f"{total_ventanas}] "
                    f"ACC="
                    f"{accuracy_actual:7.3f}% | "
                    f"Pipeline medio="
                    f"{media_total:10.3f} us"
                )


            # ================================================
            # BACKUP CADA 50
            # ================================================

            if (
                procesadas
                %
                50
                ==
                0
            ):

                pd.DataFrame(
                    resultados
                ).to_csv(
                    CSV_TEMPORAL,
                    index=False
                )


            time.sleep(
                PAUSA_ENTRE_COMANDOS_S
            )


    # ========================================================
    # DATAFRAME FINAL
    # ========================================================

    df = pd.DataFrame(
        resultados
    )


    df.to_csv(
        CSV_RESULTADOS,
        index=False
    )


    # ========================================================
    # ACCURACY
    # ========================================================

    correctas = int(
        df[
            "correcto"
        ].sum()
    )


    accuracy = (
        correctas
        /
        len(df)
        *
        100
    )


    # ========================================================
    # ESTADÍSTICAS TIEMPO
    # ========================================================

    stats_features = (
        calcular_estadisticas(
            df[
                "us_features"
            ]
        )
    )


    stats_inferencia = (
        calcular_estadisticas(
            df[
                "us_inferencia"
            ]
        )
    )


    stats_total = (
        calcular_estadisticas(
            df[
                "us_total"
            ]
        )
    )


    # ========================================================
    # ESTADÍSTICAS CICLOS
    # ========================================================

    stats_cycles_features = (
        calcular_estadisticas(
            df[
                "ciclos_features"
            ]
        )
    )


    stats_cycles_inferencia = (
        calcular_estadisticas(
            df[
                "ciclos_inferencia"
            ]
        )
    )


    stats_cycles_total = (
        calcular_estadisticas(
            df[
                "ciclos_total"
            ]
        )
    )


    # ========================================================
    # RESUMEN
    # ========================================================

    resumen = f"""
================================================================================
EXPERIMENTO 4 - BENCHMARK STM32
================================================================================

Plataforma:
STM32 Nucleo-F446RE

Ventanas analizadas:
{len(df)}

Frecuencia HCLK:
{hclk_hz} Hz
{hclk_hz / 1e6:.6f} MHz

DWT CYCCNT:
Habilitado

Pipeline medido:

ventana cruda 16x3
-> FeatureExtraction_Extract()
-> CMSIS-DSP FFT
-> 21 features
-> cuantización INT8
-> TFLM / CMSIS-NN
-> clasificación


================================================================================
VERIFICACIÓN FUNCIONAL
================================================================================

Clasificaciones correctas:
{correctas}/{len(df)}

Accuracy:
{accuracy:.4f} %


================================================================================
PREPROCESAMIENTO / FEATURE EXTRACTION
================================================================================

Ciclos medios:
{stats_cycles_features["media"]:.3f}

Ciclos mínimos:
{stats_cycles_features["minimo"]:.0f}

Ciclos máximos:
{stats_cycles_features["maximo"]:.0f}

Tiempo medio:
{stats_features["media"]:.6f} us

Mediana:
{stats_features["mediana"]:.6f} us

Mínimo:
{stats_features["minimo"]:.6f} us

Máximo:
{stats_features["maximo"]:.6f} us

Desviación estándar:
{stats_features["desviacion"]:.6f} us

Percentil 95:
{stats_features["p95"]:.6f} us


================================================================================
INFERENCIA TINYML
================================================================================

Ciclos medios:
{stats_cycles_inferencia["media"]:.3f}

Ciclos mínimos:
{stats_cycles_inferencia["minimo"]:.0f}

Ciclos máximos:
{stats_cycles_inferencia["maximo"]:.0f}

Tiempo medio:
{stats_inferencia["media"]:.6f} us

Mediana:
{stats_inferencia["mediana"]:.6f} us

Mínimo:
{stats_inferencia["minimo"]:.6f} us

Máximo:
{stats_inferencia["maximo"]:.6f} us

Desviación estándar:
{stats_inferencia["desviacion"]:.6f} us

Percentil 95:
{stats_inferencia["p95"]:.6f} us


================================================================================
PIPELINE END-TO-END STM32
================================================================================

Ciclos medios:
{stats_cycles_total["media"]:.3f}

Ciclos mínimos:
{stats_cycles_total["minimo"]:.0f}

Ciclos máximos:
{stats_cycles_total["maximo"]:.0f}

Tiempo medio:
{stats_total["media"]:.6f} us

Mediana:
{stats_total["mediana"]:.6f} us

Mínimo:
{stats_total["minimo"]:.6f} us

Máximo:
{stats_total["maximo"]:.6f} us

Desviación estándar:
{stats_total["desviacion"]:.6f} us

Percentil 95:
{stats_total["p95"]:.6f} us


================================================================================
ACLARACIÓN
================================================================================

Los tiempos anteriores fueron obtenidos internamente mediante DWT->CYCCNT.

No incluyen:

- transmisión UART desde el PC,
- recepción UART,
- reconstrucción de la ventana en Python,
- escritura del CSV,
- pausas del script.

Por tanto, representan el tiempo computacional interno del
preprocesamiento y la inferencia en el microcontrolador.
"""


    TXT_RESUMEN.write_text(
        resumen,
        encoding="utf-8"
    )


    # ========================================================
    # ELIMINAR BACKUP
    # ========================================================

    if CSV_TEMPORAL.exists():

        CSV_TEMPORAL.unlink()


    # ========================================================
    # TERMINAL
    # ========================================================

    print()

    print(resumen)

    print()

    print("=" * 80)

    print("ARCHIVOS GENERADOS")

    print("=" * 80)

    print(
        CSV_RESULTADOS
    )

    print(
        TXT_RESUMEN
    )


# ============================================================
# EJECUTAR
# ============================================================

if __name__ == "__main__":

    main()
