from pathlib import Path
import json
import time
import sys

CODIGO_DIR = Path(__file__).resolve().parents[2]
if str(CODIGO_DIR) not in sys.path:
    sys.path.insert(0, str(CODIGO_DIR))

from comun.rutas_proyecto import DATOS_ARCINIEGAS_PROCESADO, DATOS_ARCINIEGAS_RAW, MODELOS_ARCINIEGAS, RESULTADOS_E1

import numpy as np
import pandas as pd
import serial
import tensorflow as tf


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

BASE_PREPROCESADO = DATOS_ARCINIEGAS_PROCESADO
BASE_DATASET_TEST = DATOS_ARCINIEGAS_RAW / "testing"
OUTPUT_DIR = RESULTADOS_E1 / "pipeline_end_to_end"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# ARCHIVOS DEL PREPROCESAMIENTO
# ============================================================

METADATA_PATH = (
    BASE_PREPROCESADO
    / "metadata_features.csv"
)

X_TEST_PATH = (
    BASE_PREPROCESADO
    / "X_test.npy"
)

Y_TEST_PATH = (
    BASE_PREPROCESADO
    / "y_test.npy"
)


# ============================================================
# MODELO TFLITE INT8
# ============================================================

MODELO_PATH = MODELOS_ARCINIEGAS / "e1" / "firmware" / "modelo_arciniegas_e1_int8.tflite"


# ============================================================
# UART
# ============================================================

SERIAL_PORT = "COM4"

BAUDRATE = 115200

TIMEOUT_S = 2.0

REINTENTOS = 3

PAUSA_ENTRE_COMANDOS_S = 0.01


# ============================================================
# CONFIGURACIÓN DEL PIPELINE
# ============================================================

WINDOW_SIZE = 16

AXIS_COUNT = 3

RAW_VALUES = (
    WINDOW_SIZE
    *
    AXIS_COUNT
)

NUM_CLASSES = 3


CLASES = {
    0: "DETENIDO",
    1: "VELOCIDAD MEDIA",
    2: "VELOCIDAD ALTA",
}


# ============================================================
# ARCHIVOS DE SALIDA
# ============================================================

CSV_RESULTADOS = (
    OUTPUT_DIR
    / "resultados_1332_rawpred.csv"
)

CSV_TEMPORAL = (
    OUTPUT_DIR
    / "resultados_temporales.csv"
)

CSV_DISCREPANCIAS = (
    OUTPUT_DIR
    / "discrepancias_pc_stm32.csv"
)

CSV_CONFUSION_STM32 = (
    OUTPUT_DIR
    / "matriz_confusion_stm32.csv"
)

CSV_CONFUSION_PC = (
    OUTPUT_DIR
    / "matriz_confusion_pc.csv"
)

TXT_RESUMEN = (
    OUTPUT_DIR
    / "resumen_experimento3.txt"
)


# ============================================================
# BUSCAR JSON ORIGINAL
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
# CARGAR JSON CON CACHE
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
            f"Formato inesperado en:\n{ruta}"
        )

    values = values[:, :3]

    json_cache[
        nombre_archivo
    ] = values

    return values


# ============================================================
# CONSTRUIR COMANDO RAWPRED
# ============================================================

def construir_rawpred(window):

    if window.shape != (
        WINDOW_SIZE,
        AXIS_COUNT
    ):
        raise ValueError(
            "La ventana debe tener shape "
            f"({WINDOW_SIZE}, {AXIS_COUNT}), "
            f"pero tiene {window.shape}"
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
            f"Se generaron {len(valores)} "
            f"valores y deberían ser {RAW_VALUES}."
        )


    comando = (
        "RAWPRED,"
        +
        ",".join(valores)
    )

    return comando


# ============================================================
# ENVIAR RAWPRED
# ============================================================

def enviar_rawpred(
    ser,
    comando
):

    ultima_respuesta = ""

    for intento in range(
        1,
        REINTENTOS + 1
    ):

        ser.reset_input_buffer()

        trama = (
            comando
            +
            "\r\n"
        )

        ser.write(
            trama.encode(
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


        ultima_respuesta = (
            respuesta
        )


        if respuesta.startswith(
            "RAWPRED,"
        ):

            partes = (
                respuesta.split(",")
            )

            if len(partes) != 5:
                continue


            try:

                clase = int(
                    partes[1]
                )

                probs = np.array(
                    [
                        float(partes[2]),
                        float(partes[3]),
                        float(partes[4]),
                    ],
                    dtype=np.float64
                )

            except ValueError:
                continue


            return (
                clase,
                probs,
                respuesta,
                intento,
            )


        time.sleep(
            0.05
        )


    raise RuntimeError(
        "No se obtuvo una respuesta RAWPRED válida.\n"
        f"Última respuesta: {ultima_respuesta}"
    )


# ============================================================
# PREPARAR MODELO TFLITE PC
# ============================================================

def preparar_interprete(
    modelo_path
):

    # ========================================================
    # LEER MODELO COMO BYTES
    #
    # Esto evita problemas de TensorFlow con rutas de Windows
    # que contienen espacios o caracteres especiales.
    # ========================================================

    if not modelo_path.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo:\n{modelo_path}"
        )

    modelo_bytes = (
        modelo_path.read_bytes()
    )


    print()

    print("=" * 80)
    print("MODELO INT8 PC")
    print("=" * 80)

    print(
        f"Modelo        : {modelo_path}"
    )

    print(
        f"Tamaño        : "
        f"{len(modelo_bytes)} bytes"
    )


    # ========================================================
    # CREAR INTERPRETER DESDE MEMORIA
    # ========================================================

    interpreter = (
        tf.lite.Interpreter(
            model_content=modelo_bytes
        )
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


    input_scale, input_zp = (
        input_details[
            "quantization"
        ]
    )

    output_scale, output_zp = (
        output_details[
            "quantization"
        ]
    )


    print(
        f"Input dtype   : "
        f"{input_details['dtype']}"
    )

    print(
        f"Input scale   : "
        f"{input_scale}"
    )

    print(
        f"Input zp      : "
        f"{input_zp}"
    )

    print(
        f"Output dtype  : "
        f"{output_details['dtype']}"
    )

    print(
        f"Output scale  : "
        f"{output_scale}"
    )

    print(
        f"Output zp     : "
        f"{output_zp}"
    )


    return (
        interpreter,
        input_details,
        output_details,
    )

# ============================================================
# INFERENCIA INT8 EN PC
# ============================================================

def predecir_pc(
    interpreter,
    input_details,
    output_details,
    features
):

    input_scale, input_zp = (
        input_details[
            "quantization"
        ]
    )


    output_scale, output_zp = (
        output_details[
            "quantization"
        ]
    )


    # --------------------------------------------------------
    # Cuantización:
    #
    # q = round(x / scale) + zero_point
    # --------------------------------------------------------

    q = np.round(
        features
        /
        input_scale
        +
        input_zp
    )


    q = np.clip(
        q,
        -128,
        127
    ).astype(
        np.int8
    )


    entrada = (
        q.reshape(
            input_details[
                "shape"
            ]
        )
    )


    interpreter.set_tensor(
        input_details[
            "index"
        ],
        entrada
    )


    interpreter.invoke()


    salida_q = (
        interpreter
        .get_tensor(
            output_details[
                "index"
            ]
        )
        .reshape(-1)
    )


    probs = (
        salida_q.astype(
            np.float64
        )
        -
        output_zp
    ) * output_scale


    clase = int(
        np.argmax(
            probs
        )
    )


    return (
        clase,
        probs,
        q.reshape(-1),
        salida_q.copy(),
    )


# ============================================================
# MATRIZ DE CONFUSIÓN
# ============================================================

def matriz_confusion(
    reales,
    predichas
):

    matriz = np.zeros(
        (
            NUM_CLASSES,
            NUM_CLASSES
        ),
        dtype=int
    )


    for real, pred in zip(
        reales,
        predichas
    ):

        matriz[
            int(real),
            int(pred)
        ] += 1


    nombres = [
        CLASES[i]
        for i in range(
            NUM_CLASSES
        )
    ]


    return pd.DataFrame(
        matriz,
        index=[
            f"REAL_{x}"
            for x in nombres
        ],
        columns=[
            f"PRED_{x}"
            for x in nombres
        ]
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("EXPERIMENTO 3")
    print("PIPELINE END-TO-END PC vs STM32")
    print("=" * 80)


    # ========================================================
    # CARGAR METADATA
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


    # ========================================================
    # CARGAR TEST
    # ========================================================

    X_test = np.load(
        X_TEST_PATH
    )

    y_test = np.load(
        Y_TEST_PATH
    ).reshape(-1)


    print()

    print(
        f"Ventanas metadata testing : "
        f"{len(metadata_test)}"
    )

    print(
        f"X_test                    : "
        f"{X_test.shape}"
    )

    print(
        f"y_test                    : "
        f"{y_test.shape}"
    )


    # ========================================================
    # VERIFICACIONES
    # ========================================================

    if len(
        metadata_test
    ) != len(
        X_test
    ):

        raise ValueError(
            "metadata testing y X_test "
            "no tienen la misma cantidad."
        )


    if len(
        X_test
    ) != len(
        y_test
    ):

        raise ValueError(
            "X_test y y_test no tienen "
            "la misma cantidad."
        )


    if X_test.shape[1] != 21:

        raise ValueError(
            "X_test debe contener "
            "21 características."
        )


    total_ventanas = len(
        metadata_test
    )


    # ========================================================
    # MODELO
    # ========================================================

    modelo_path = MODELO_PATH

    if not modelo_path.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo:\n{modelo_path}"
        )


    (
        interpreter,
        input_details,
        output_details,
    ) = preparar_interprete(
        modelo_path
    )


    # ========================================================
    # ABRIR UART
    # ========================================================

    print()

    print("=" * 80)
    print("STM32")
    print("=" * 80)

    print(
        f"Puerto    : {SERIAL_PORT}"
    )

    print(
        f"Baudrate  : {BAUDRATE}"
    )

    print(
        f"Ventanas  : {total_ventanas}"
    )

    print()

    print(
        "Iniciando experimento..."
    )


    resultados = []


    inicio_experimento = (
        time.perf_counter()
    )


    with serial.Serial(
        port=SERIAL_PORT,
        baudrate=BAUDRATE,
        timeout=TIMEOUT_S
    ) as ser:


        time.sleep(
            0.3
        )


        ser.reset_input_buffer()
        ser.reset_output_buffer()


        # ====================================================
        # LOOP SOBRE TODAS LAS VENTANAS
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


            # ------------------------------------------------
            # Verificar etiqueta
            # ------------------------------------------------

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
                    f"en índice {index}: "
                    f"metadata={label_metadata}, "
                    f"y_test={label_test}"
                )


            # ------------------------------------------------
            # Reconstruir ventana cruda
            # ------------------------------------------------

            values = (
                cargar_values(
                    archivo
                )
            )


            window = (
                values[
                    start:end,
                    :
                ]
            )


            if window.shape != (
                WINDOW_SIZE,
                AXIS_COUNT
            ):

                raise ValueError(
                    f"Ventana {index} "
                    f"tiene shape "
                    f"{window.shape}"
                )


            # ------------------------------------------------
            # PC INT8
            # ------------------------------------------------

            (
                clase_pc,
                probs_pc,
                entrada_q_pc,
                salida_q_pc,
            ) = predecir_pc(
                interpreter,
                input_details,
                output_details,
                X_test[index]
            )


            # ------------------------------------------------
            # STM32 RAWPRED
            # ------------------------------------------------

            comando = (
                construir_rawpred(
                    window
                )
            )


            (
                clase_stm32,
                probs_stm32,
                respuesta_raw,
                intentos,
            ) = enviar_rawpred(
                ser,
                comando
            )


            # ------------------------------------------------
            # MÉTRICAS DE ESTA VENTANA
            # ------------------------------------------------

            coincide = (
                clase_pc
                ==
                clase_stm32
            )


            pc_correcto = (
                clase_pc
                ==
                label_test
            )


            stm32_correcto = (
                clase_stm32
                ==
                label_test
            )


            error_probs = (
                np.abs(
                    probs_pc
                    -
                    probs_stm32
                )
            )


            # ------------------------------------------------
            # FILA BASE
            # ------------------------------------------------

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

                "clase_pc":
                    clase_pc,

                "nombre_pc":
                    CLASES.get(
                        clase_pc,
                        "DESCONOCIDA"
                    ),

                "clase_stm32":
                    clase_stm32,

                "nombre_stm32":
                    CLASES.get(
                        clase_stm32,
                        "DESCONOCIDA"
                    ),

                "coincide_pc_stm32":
                    coincide,

                "pc_correcto":
                    pc_correcto,

                "stm32_correcto":
                    stm32_correcto,

                "pc_prob_det":
                    probs_pc[0],

                "pc_prob_media":
                    probs_pc[1],

                "pc_prob_alta":
                    probs_pc[2],

                "stm32_prob_det":
                    probs_stm32[0],

                "stm32_prob_media":
                    probs_stm32[1],

                "stm32_prob_alta":
                    probs_stm32[2],

                "error_prob_det":
                    error_probs[0],

                "error_prob_media":
                    error_probs[1],

                "error_prob_alta":
                    error_probs[2],

                "error_prob_medio":
                    float(
                        np.mean(
                            error_probs
                        )
                    ),

                "error_prob_max":
                    float(
                        np.max(
                            error_probs
                        )
                    ),

                "pc_output_int8_0":
                    int(
                        salida_q_pc[0]
                    ),

                "pc_output_int8_1":
                    int(
                        salida_q_pc[1]
                    ),

                "pc_output_int8_2":
                    int(
                        salida_q_pc[2]
                    ),

                "uart_intentos":
                    intentos,

                "comando_longitud":
                    len(comando),

                "respuesta_raw":
                    respuesta_raw,
            }


            # ------------------------------------------------
            # GUARDAR 21 FEATURES PC
            # ------------------------------------------------

            for i in range(21):

                resultado[
                    f"feature_pc_{i:02d}"
                ] = float(
                    X_test[
                        index,
                        i
                    ]
                )


                resultado[
                    f"feature_pc_q_{i:02d}"
                ] = int(
                    entrada_q_pc[i]
                )


            # ------------------------------------------------
            # GUARDAR 48 VALORES CRUDOS
            # ------------------------------------------------

            for n in range(
                WINDOW_SIZE
            ):

                resultado[
                    f"raw_ax_{n:02d}"
                ] = float(
                    window[
                        n,
                        0
                    ]
                )

                resultado[
                    f"raw_ay_{n:02d}"
                ] = float(
                    window[
                        n,
                        1
                    ]
                )

                resultado[
                    f"raw_az_{n:02d}"
                ] = float(
                    window[
                        n,
                        2
                    ]
                )


            resultados.append(
                resultado
            )


            # ------------------------------------------------
            # PROGRESO
            # ------------------------------------------------

            procesadas = (
                index + 1
            )


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


                coincidencia_actual = (
                    df_temp[
                        "coincide_pc_stm32"
                    ]
                    .mean()
                    *
                    100
                )


                accuracy_stm_actual = (
                    df_temp[
                        "stm32_correcto"
                    ]
                    .mean()
                    *
                    100
                )


                print(
                    f"[{procesadas:4d}/"
                    f"{total_ventanas}] "
                    f"PC↔STM32="
                    f"{coincidencia_actual:7.3f}% | "
                    f"ACC STM32="
                    f"{accuracy_stm_actual:7.3f}%"
                )


            # ------------------------------------------------
            # BACKUP CADA 50
            # ------------------------------------------------

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
    # FIN
    # ========================================================

    fin_experimento = (
        time.perf_counter()
    )


    duracion = (
        fin_experimento
        -
        inicio_experimento
    )


    # ========================================================
    # DATAFRAME FINAL
    # ========================================================

    df_resultados = pd.DataFrame(
        resultados
    )


    df_resultados.to_csv(
        CSV_RESULTADOS,
        index=False
    )


    # ========================================================
    # MÉTRICAS GLOBALES
    # ========================================================

    total = len(
        df_resultados
    )


    coincidencias = int(
        df_resultados[
            "coincide_pc_stm32"
        ]
        .sum()
    )


    correctas_pc = int(
        df_resultados[
            "pc_correcto"
        ]
        .sum()
    )


    correctas_stm32 = int(
        df_resultados[
            "stm32_correcto"
        ]
        .sum()
    )


    acuerdo_pct = (
        coincidencias
        /
        total
        *
        100
    )


    accuracy_pc_pct = (
        correctas_pc
        /
        total
        *
        100
    )


    accuracy_stm32_pct = (
        correctas_stm32
        /
        total
        *
        100
    )


    error_prob_medio = float(
        df_resultados[
            [
                "error_prob_det",
                "error_prob_media",
                "error_prob_alta",
            ]
        ]
        .to_numpy()
        .mean()
    )


    error_prob_max = float(
        df_resultados[
            [
                "error_prob_det",
                "error_prob_media",
                "error_prob_alta",
            ]
        ]
        .to_numpy()
        .max()
    )


    # ========================================================
    # DISCREPANCIAS
    # ========================================================

    discrepancias = (
        df_resultados[
            ~df_resultados[
                "coincide_pc_stm32"
            ]
        ]
        .copy()
    )


    discrepancias.to_csv(
        CSV_DISCREPANCIAS,
        index=False
    )


    # ========================================================
    # MATRICES DE CONFUSIÓN
    # ========================================================

    confusion_stm32 = (
        matriz_confusion(
            df_resultados[
                "clase_real"
            ],
            df_resultados[
                "clase_stm32"
            ]
        )
    )


    confusion_pc = (
        matriz_confusion(
            df_resultados[
                "clase_real"
            ],
            df_resultados[
                "clase_pc"
            ]
        )
    )


    confusion_stm32.to_csv(
        CSV_CONFUSION_STM32
    )


    confusion_pc.to_csv(
        CSV_CONFUSION_PC
    )


    # ========================================================
    # RESUMEN TXT
    # ========================================================

    resumen = f"""
================================================================================
EXPERIMENTO 3 - PIPELINE END-TO-END PC vs STM32
================================================================================

Configuración:

Modelo:
{modelo_path}

Puerto:
{SERIAL_PORT}

Baudrate:
{BAUDRATE}

Ventanas procesadas:
{total}

Pipeline STM32:
ventana cruda 16x3
-> FeatureExtraction_Extract()
-> CMSIS-DSP FFT
-> 21 features
-> cuantización INT8
-> TFLM / CMSIS-NN
-> clase y probabilidades


================================================================================
RESULTADOS
================================================================================

Coincidencia PC vs STM32:
{coincidencias}/{total}
{acuerdo_pct:.4f} %

Accuracy INT8 PC:
{correctas_pc}/{total}
{accuracy_pc_pct:.4f} %

Accuracy STM32 end-to-end:
{correctas_stm32}/{total}
{accuracy_stm32_pct:.4f} %

Predicciones diferentes PC vs STM32:
{total - coincidencias}

Error absoluto medio de probabilidades:
{error_prob_medio:.12f}

Error absoluto máximo de probabilidades:
{error_prob_max:.12f}


================================================================================
TIEMPO DEL EXPERIMENTO
================================================================================

Tiempo total:
{duracion:.3f} s

Tiempo medio PC+UART por ventana:
{duracion / total:.6f} s

NOTA:
Este tiempo incluye procesamiento en PC, comunicación UART,
esperas y ejecución STM32. No representa la latencia interna
del microcontrolador. La latencia embebida se medirá
posteriormente mediante DWT->CYCCNT.


================================================================================
MATRIZ DE CONFUSIÓN STM32
================================================================================

{confusion_stm32.to_string()}


================================================================================
MATRIZ DE CONFUSIÓN PC
================================================================================

{confusion_pc.to_string()}
"""


    TXT_RESUMEN.write_text(
        resumen,
        encoding="utf-8"
    )


    # ========================================================
    # ELIMINAR TEMPORAL
    # ========================================================

    if CSV_TEMPORAL.exists():

        CSV_TEMPORAL.unlink()


    # ========================================================
    # MOSTRAR RESULTADOS
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

    print(
        CSV_CONFUSION_STM32
    )

    print(
        CSV_CONFUSION_PC
    )

    print(
        CSV_DISCREPANCIAS
    )


# ============================================================
# EJECUTAR
# ============================================================

if __name__ == "__main__":

    main()
