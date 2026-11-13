import numpy as np
import serial
import time
import tensorflow as tf
from pathlib import Path
import sys

CODIGO_DIR = Path(__file__).resolve().parents[2]
if str(CODIGO_DIR) not in sys.path:
    sys.path.insert(0, str(CODIGO_DIR))

from comun.rutas_proyecto import DATOS_ARCINIEGAS_PROCESADO, MODELOS_ARCINIEGAS


# ============================================================
# CONFIGURACIÓN
# ============================================================

PUERTO = "COM4"
BAUDRATE = 115200

# Carpeta donde tu preprocesamiento guardó:
# X_test.npy
# y_test.npy
DATA_DIR = DATOS_ARCINIEGAS_PROCESADO

MODEL_PATH = MODELOS_ARCINIEGAS / "e1" / "firmware" / "modelo_arciniegas_e1_int8.tflite"

print("Modelo buscado en:")
print(MODEL_PATH)

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"No se encontró el modelo TFLite en:\n{MODEL_PATH}"
    )
# Primera ventana del conjunto de test
INDICE = 0


# ============================================================
# NOMBRES DE CLASE
# ============================================================

CLASES = {
    0: "DET",
    1: "MEDIA",
    2: "ALTA"
}


# ============================================================
# CARGAR TEST
# ============================================================

X_test = np.load(DATA_DIR / "X_test.npy")
y_test = np.load(DATA_DIR / "y_test.npy")

print("=" * 70)
print("VALIDACIÓN DE UNA VENTANA REAL")
print("=" * 70)

print("\nForma X_test:", X_test.shape)
print("Forma y_test:", y_test.shape)

x = X_test[INDICE].astype(np.float32)
y_real = int(y_test[INDICE])

print("\nVentana seleccionada:", INDICE)
print("Clase real:", y_real, CLASES.get(y_real, "DESCONOCIDA"))

print("\n21 características:")
for i, valor in enumerate(x):
    print(f"  f{i+1:02d} = {valor:.9f}")


# ============================================================
# CARGAR MODELO TFLITE EN PC
# ============================================================

# Leer el modelo con Python.
# Esto evita problemas de TensorFlow Lite con rutas
# que contienen caracteres como "ó" en "Investigación".
with open(MODEL_PATH, "rb") as f:
    modelo_tflite = f.read()

print("Tamaño del modelo:", len(modelo_tflite), "bytes")

interpreter = tf.lite.Interpreter(
    model_content=modelo_tflite
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

print("\n" + "=" * 70)
print("MODELO TFLITE EN PC")
print("=" * 70)

print("Input dtype :", input_details["dtype"])
print("Input shape :", input_details["shape"])

print("Output dtype:", output_details["dtype"])
print("Output shape:", output_details["shape"])


# ============================================================
# PARÁMETROS DE CUANTIZACIÓN
# ============================================================

input_scale, input_zero_point = input_details["quantization"]
output_scale, output_zero_point = output_details["quantization"]

print("\nCuantización entrada:")
print("  scale      =", input_scale)
print("  zero_point =", input_zero_point)

print("\nCuantización salida:")
print("  scale      =", output_scale)
print("  zero_point =", output_zero_point)


# ============================================================
# CUANTIZAR ENTRADA PARA TFLITE PC
# ============================================================

x_quant = np.round(
    x / input_scale
) + input_zero_point

x_quant = np.clip(
    x_quant,
    -128,
    127
).astype(np.int8)

x_quant = x_quant.reshape(1, 21)

print("\nEntrada cuantizada:")
print(x_quant)


# ============================================================
# INFERENCIA TFLITE EN PC
# ============================================================

interpreter.set_tensor(
    input_details["index"],
    x_quant
)

interpreter.invoke()

output_quant = interpreter.get_tensor(
    output_details["index"]
)[0]

probs_pc = (
    output_quant.astype(np.int32)
    - output_zero_point
) * output_scale

clase_pc = int(np.argmax(probs_pc))

print("\n" + "=" * 70)
print("RESULTADO PC")
print("=" * 70)

print("Salida INT8 :", output_quant)

print(
    f"Probabilidades: "
    f"DET={probs_pc[0]:.6f}, "
    f"MEDIA={probs_pc[1]:.6f}, "
    f"ALTA={probs_pc[2]:.6f}"
)

print(
    "Clase PC:",
    clase_pc,
    CLASES.get(clase_pc, "DESCONOCIDA")
)


# ============================================================
# CONSTRUIR COMANDO PARA STM32
# ============================================================

comando = "PRED," + ",".join(
    f"{valor:.9f}"
    for valor in x
)

print("\n" + "=" * 70)
print("ENVÍO A STM32")
print("=" * 70)

print("Número de características:", len(x))
print("Longitud comando:", len(comando), "caracteres")


# ============================================================
# UART
# ============================================================

ser = serial.Serial(
    PUERTO,
    BAUDRATE,
    timeout=3
)

time.sleep(1)

ser.reset_input_buffer()

ser.write(
    (comando + "\r\n").encode()
)

respuesta = (
    ser.readline()
    .decode(errors="ignore")
    .strip()
)

ser.close()

print("\nRespuesta STM32:")
print(respuesta)


# ============================================================
# INTERPRETAR RESPUESTA
# ============================================================

if not respuesta.startswith("PRED,"):
    print("\nERROR:")
    print("La STM32 no devolvió una respuesta PRED válida.")
    raise SystemExit


partes = respuesta.split(",")

if len(partes) != 5:
    print("\nERROR:")
    print("Formato inesperado:", respuesta)
    raise SystemExit


clase_stm32 = int(partes[1])

probs_stm32 = np.array(
    [
        float(partes[2]),
        float(partes[3]),
        float(partes[4])
    ]
)


# ============================================================
# RESULTADO STM32
# ============================================================

print("\n" + "=" * 70)
print("RESULTADO STM32")
print("=" * 70)

print(
    f"Probabilidades: "
    f"DET={probs_stm32[0]:.6f}, "
    f"MEDIA={probs_stm32[1]:.6f}, "
    f"ALTA={probs_stm32[2]:.6f}"
)

print(
    "Clase STM32:",
    clase_stm32,
    CLASES.get(clase_stm32, "DESCONOCIDA")
)


# ============================================================
# COMPARACIÓN
# ============================================================

print("\n" + "=" * 70)
print("COMPARACIÓN FINAL")
print("=" * 70)

print(
    "Clase real :",
    y_real,
    CLASES.get(y_real, "DESCONOCIDA")
)

print(
    "Clase PC   :",
    clase_pc,
    CLASES.get(clase_pc, "DESCONOCIDA")
)

print(
    "Clase STM32:",
    clase_stm32,
    CLASES.get(clase_stm32, "DESCONOCIDA")
)

print()

if clase_pc == clase_stm32:
    print("PC vs STM32: COINCIDEN")
else:
    print("PC vs STM32: NO COINCIDEN")

if clase_stm32 == y_real:
    print("STM32 vs real: CLASIFICACIÓN CORRECTA")
else:
    print("STM32 vs real: CLASIFICACIÓN INCORRECTA")

print("\nDiferencia absoluta de probabilidades:")
print(
    np.abs(
        probs_pc - probs_stm32
    )
)
