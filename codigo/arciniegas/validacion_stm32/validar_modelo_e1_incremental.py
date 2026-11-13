import os

# ============================================================
# REDUCIR MENSAJES INFORMATIVOS DE TENSORFLOW
# ============================================================

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import csv
import time
from pathlib import Path
import sys

CODIGO_DIR = Path(__file__).resolve().parents[2]
if str(CODIGO_DIR) not in sys.path:
    sys.path.insert(0, str(CODIGO_DIR))

from comun.rutas_proyecto import DATOS_ARCINIEGAS_PROCESADO, MODELOS_ARCINIEGAS, RESULTADOS_E1

import numpy as np
import serial
import tensorflow as tf


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

PUERTO = "COM4"
BAUDRATE = 115200

# ------------------------------------------------------------
# Carpeta principal del proyecto
# ------------------------------------------------------------

# ------------------------------------------------------------
# Carpeta donde se encuentran X_test.npy y y_test.npy
#
# CAMBIA ESTA RUTA POR LA QUE YA ESTÁS USANDO ACTUALMENTE
# ------------------------------------------------------------

DATA_DIR = DATOS_ARCINIEGAS_PROCESADO


# ------------------------------------------------------------
# Modelo INT8 correspondiente al experimento actual
# ------------------------------------------------------------

MODEL_PATH = MODELOS_ARCINIEGAS / "e1" / "firmware" / "modelo_arciniegas_e1_int8.tflite"


# ============================================================
# IDENTIFICACIÓN DEL EXPERIMENTO
# ============================================================

NOMBRE_EXPERIMENTO = (
    "experimento_sin_standardscaler_sin_class_weight"
)

DESCRIPCION_EXPERIMENTO = """
EXPERIMENTO STM32 - MODELO SIN STANDARDSCALER Y SIN CLASS_WEIGHT

Modelo:
Red neuronal DNN cuantizada INT8.

Arquitectura:
21 entradas
Dense 20 - ReLU
Dense 10 - ReLU
Dense 3 - Softmax

Preprocesamiento:
Sin StandardScaler.

Balanceo durante entrenamiento:
Sin class_weight.

Plataforma de despliegue:
STM32 Nucleo-F446RE.

Framework embebido:
TensorFlow Lite Micro.

Optimización:
INT8.
CMSIS-NN.

Objetivo:
Comparar las predicciones del modelo TFLite ejecutado en PC con
el mismo modelo desplegado mediante TensorFlow Lite Micro en
STM32.

Las características son enviadas desde el PC a la STM32 mediante
USART2 a 115200 baudios.

Clases:
0 = DET
1 = MEDIA
2 = ALTA
"""


# ============================================================
# VENTANAS A PROBAR
# ============================================================
#
# Empezaremos progresivamente.
#
# Puedes cambiar esto posteriormente por:
#
# [0]
# [0, 1, 2, 3, 4]
# list(range(10))
# list(range(50))
# list(range(100))
#
# Finalmente:
# list(range(1332))
#
# ============================================================

INDICES = list(range(1332))



# ============================================================
# CLASES
# ============================================================

CLASES = {
    0: "DET",
    1: "MEDIA",
    2: "ALTA"
}


# ============================================================
# CREAR ESTRUCTURA DE CARPETAS
# ============================================================

EXPERIMENTO_DIR = RESULTADOS_E1 / "validacion_stm32"

EXPERIMENTO_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# ARCHIVOS DEL EXPERIMENTO
# ============================================================

DESCRIPCION_PATH = (
    EXPERIMENTO_DIR
    / "descripcion_experimento.txt"
)

CSV_PATH = (
    EXPERIMENTO_DIR
    / f"resultados_{len(INDICES)}_ventanas.csv"
)


# ============================================================
# CREAR DESCRIPCIÓN DEL EXPERIMENTO
# ============================================================

with open(
    DESCRIPCION_PATH,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        DESCRIPCION_EXPERIMENTO.strip()
    )


print("=" * 75)
print("EXPERIMENTO STM32")
print("=" * 75)

print("\nCarpeta:")
print(EXPERIMENTO_DIR)

print("\nCSV:")
print(CSV_PATH)


# ============================================================
# VERIFICAR ARCHIVOS
# ============================================================

X_PATH = DATA_DIR / "X_test.npy"
Y_PATH = DATA_DIR / "y_test.npy"

if not X_PATH.exists():
    raise FileNotFoundError(
        f"No se encontró:\n{X_PATH}"
    )

if not Y_PATH.exists():
    raise FileNotFoundError(
        f"No se encontró:\n{Y_PATH}"
    )

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"No se encontró:\n{MODEL_PATH}"
    )


# ============================================================
# CARGAR DATASET
# ============================================================

X_test = np.load(X_PATH)
y_test = np.load(Y_PATH)

print("\nX_test:", X_test.shape)
print("y_test:", y_test.shape)


# ============================================================
# CARGAR MODELO TFLITE
# ============================================================
#
# Se cargan los bytes manualmente para evitar problemas
# de rutas Unicode en Windows.
# ============================================================

with open(MODEL_PATH, "rb") as f:
    modelo_tflite = f.read()

print("\nTamaño modelo TFLite:",
      len(modelo_tflite),
      "bytes")


interpreter = tf.lite.Interpreter(
    model_content=modelo_tflite
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


# ============================================================
# CUANTIZACIÓN
# ============================================================

input_scale, input_zero_point = (
    input_details["quantization"]
)

output_scale, output_zero_point = (
    output_details["quantization"]
)


print("\nCuantización entrada:")
print("Scale     :", input_scale)
print("Zero point:", input_zero_point)

print("\nCuantización salida:")
print("Scale     :", output_scale)
print("Zero point:", output_zero_point)


# ============================================================
# CABECERA CSV
# ============================================================

COLUMNAS = [

    "indice_ventana",

    "clase_real_id",
    "clase_real_nombre",

    "clase_pc_id",
    "clase_pc_nombre",

    "clase_stm32_id",
    "clase_stm32_nombre",

    "coincidencia_pc_stm32",
    "stm32_clasificacion_correcta",

    "prob_pc_det",
    "prob_pc_media",
    "prob_pc_alta",

    "prob_stm32_det",
    "prob_stm32_media",
    "prob_stm32_alta",

    "error_abs_det",
    "error_abs_media",
    "error_abs_alta",

    "error_max_probabilidad",

    "pc_output_int8_det",
    "pc_output_int8_media",
    "pc_output_int8_alta",

    "input_scale",
    "input_zero_point",

    "output_scale",
    "output_zero_point",

    "longitud_comando_uart",

    "respuesta_stm32_raw"
]


# ============================================================
# AÑADIR LAS 21 FEATURES ORIGINALES
# ============================================================

for i in range(1, 22):
    COLUMNAS.append(
        f"feature_{i:02d}"
    )


# ============================================================
# AÑADIR LAS 21 FEATURES CUANTIZADAS
# ============================================================

for i in range(1, 22):
    COLUMNAS.append(
        f"feature_int8_{i:02d}"
    )


# ============================================================
# CONECTAR CON STM32
# ============================================================

ser = serial.Serial(
    PUERTO,
    BAUDRATE,
    timeout=3
)

time.sleep(1)

ser.reset_input_buffer()

print("\nConectado a", PUERTO)


# ============================================================
# LISTA PARA GUARDAR RESULTADOS
# ============================================================

resultados = []


# ============================================================
# PROCESAR CADA VENTANA
# ============================================================

for numero_prueba, indice in enumerate(
    INDICES,
    start=1
):

    print("\n")
    print("=" * 75)

    print(
        f"VENTANA {numero_prueba}/{len(INDICES)}"
    )

    print(
        "Índice:",
        indice
    )

    print("=" * 75)


    # ========================================================
    # VERIFICAR ÍNDICE
    # ========================================================

    if indice < 0 or indice >= len(X_test):

        print(
            "Índice fuera de rango."
        )

        continue


    # ========================================================
    # OBTENER VENTANA
    # ========================================================

    x = (
        X_test[indice]
        .astype(np.float32)
    )

    y_real = int(
        y_test[indice]
    )


    # ========================================================
    # CUANTIZAR EN PC
    # ========================================================

    x_quant = np.round(
        x / input_scale
    ) + input_zero_point

    x_quant = np.clip(
        x_quant,
        -128,
        127
    ).astype(np.int8)


    # ========================================================
    # INFERENCIA PC
    # ========================================================

    interpreter.set_tensor(
        input_details["index"],
        x_quant.reshape(1, 21)
    )

    interpreter.invoke()


    output_quant = (
        interpreter.get_tensor(
            output_details["index"]
        )[0]
    )


    probs_pc = (
        output_quant.astype(np.int32)
        - output_zero_point
    ) * output_scale


    clase_pc = int(
        np.argmax(probs_pc)
    )


    # ========================================================
    # CREAR COMANDO UART
    # ========================================================

    comando = (
        "PRED,"
        + ",".join(
            f"{valor:.9f}"
            for valor in x
        )
    )


    # ========================================================
    # ENVIAR A STM32
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
    # VERIFICAR RESPUESTA
    # ========================================================

    if not respuesta.startswith(
        "PRED,"
    ):

        print(
            "ERROR STM32:"
        )

        print(respuesta)

        continue


    partes = respuesta.split(",")


    if len(partes) != 5:

        print(
            "Respuesta inesperada:"
        )

        print(respuesta)

        continue


    # ========================================================
    # RESULTADO STM32
    # ========================================================

    clase_stm32 = int(
        partes[1]
    )

    probs_stm32 = np.array(
        [
            float(partes[2]),
            float(partes[3]),
            float(partes[4])
        ],
        dtype=np.float64
    )


    # ========================================================
    # COMPARACIONES
    # ========================================================

    coincide_pc_stm32 = (
        clase_pc
        == clase_stm32
    )


    stm32_correcto = (
        clase_stm32
        == y_real
    )


    errores = np.abs(
        probs_pc
        - probs_stm32
    )


    error_max = float(
        np.max(errores)
    )


    # ========================================================
    # MOSTRAR EN TERMINAL
    # ========================================================

    print(
        "Clase real :",
        y_real,
        CLASES[y_real]
    )

    print(
        "Clase PC   :",
        clase_pc,
        CLASES[clase_pc]
    )

    print(
        "Clase STM32:",
        clase_stm32,
        CLASES[clase_stm32]
    )


    print(
        "\nPC:"
    )

    print(
        f"DET={probs_pc[0]:.6f}, "
        f"MEDIA={probs_pc[1]:.6f}, "
        f"ALTA={probs_pc[2]:.6f}"
    )


    print(
        "\nSTM32:"
    )

    print(
        f"DET={probs_stm32[0]:.6f}, "
        f"MEDIA={probs_stm32[1]:.6f}, "
        f"ALTA={probs_stm32[2]:.6f}"
    )


    print(
        "\nPC vs STM32:",
        "COINCIDEN"
        if coincide_pc_stm32
        else "NO COINCIDEN"
    )


    print(
        "STM32 vs real:",
        "CORRECTA"
        if stm32_correcto
        else "INCORRECTA"
    )


    print(
        "Error máximo:",
        f"{error_max:.9f}"
    )


    # ========================================================
    # CREAR FILA CSV
    # ========================================================

    fila = {

        "indice_ventana":
            indice,

        "clase_real_id":
            y_real,

        "clase_real_nombre":
            CLASES[y_real],

        "clase_pc_id":
            clase_pc,

        "clase_pc_nombre":
            CLASES[clase_pc],

        "clase_stm32_id":
            clase_stm32,

        "clase_stm32_nombre":
            CLASES[clase_stm32],

        "coincidencia_pc_stm32":
            int(coincide_pc_stm32),

        "stm32_clasificacion_correcta":
            int(stm32_correcto),

        "prob_pc_det":
            float(probs_pc[0]),

        "prob_pc_media":
            float(probs_pc[1]),

        "prob_pc_alta":
            float(probs_pc[2]),

        "prob_stm32_det":
            float(probs_stm32[0]),

        "prob_stm32_media":
            float(probs_stm32[1]),

        "prob_stm32_alta":
            float(probs_stm32[2]),

        "error_abs_det":
            float(errores[0]),

        "error_abs_media":
            float(errores[1]),

        "error_abs_alta":
            float(errores[2]),

        "error_max_probabilidad":
            error_max,

        "pc_output_int8_det":
            int(output_quant[0]),

        "pc_output_int8_media":
            int(output_quant[1]),

        "pc_output_int8_alta":
            int(output_quant[2]),

        "input_scale":
            float(input_scale),

        "input_zero_point":
            int(input_zero_point),

        "output_scale":
            float(output_scale),

        "output_zero_point":
            int(output_zero_point),

        "longitud_comando_uart":
            len(comando),

        "respuesta_stm32_raw":
            respuesta
    }


    # ========================================================
    # GUARDAR 21 FEATURES FLOAT
    # ========================================================

    for i in range(21):

        fila[
            f"feature_{i+1:02d}"
        ] = float(x[i])


    # ========================================================
    # GUARDAR 21 FEATURES INT8
    # ========================================================

    for i in range(21):

        fila[
            f"feature_int8_{i+1:02d}"
        ] = int(x_quant[i])


    resultados.append(fila)


# ============================================================
# CERRAR UART
# ============================================================

ser.close()


# ============================================================
# GUARDAR CSV
# ============================================================

with open(
    CSV_PATH,
    "w",
    newline="",
    encoding="utf-8-sig"
) as archivo_csv:

    writer = csv.DictWriter(
        archivo_csv,
        fieldnames=COLUMNAS,
        delimiter=";"
    )

    writer.writeheader()

    writer.writerows(
        resultados
    )


# ============================================================
# RESUMEN
# ============================================================

print("\n")
print("=" * 75)
print("RESUMEN DEL EXPERIMENTO")
print("=" * 75)

print(
    "Ventanas solicitadas:",
    len(INDICES)
)

print(
    "Ventanas procesadas:",
    len(resultados)
)


if len(resultados) > 0:

    coincidencias = sum(
        r[
            "coincidencia_pc_stm32"
        ]
        for r in resultados
    )


    correctas = sum(
        r[
            "stm32_clasificacion_correcta"
        ]
        for r in resultados
    )


    print(
        "Coincidencias PC-STM32:",
        coincidencias,
        "/",
        len(resultados)
    )


    print(
        "Clasificaciones STM32 correctas:",
        correctas,
        "/",
        len(resultados)
    )


    print(
        "Coincidencia PC-STM32:",
        f"{100 * coincidencias / len(resultados):.2f}%"
    )


    print(
        "Accuracy STM32:",
        f"{100 * correctas / len(resultados):.2f}%"
    )


print("\nCSV guardado en:")
print(CSV_PATH)

print("\nDescripción guardada en:")
print(DESCRIPCION_PATH)

print("\nExperimento terminado.")
