import os
from pathlib import Path
import sys
import tensorflow as tf

CODIGO_DIR = Path(__file__).resolve().parents[2]
if str(CODIGO_DIR) not in sys.path:
    sys.path.insert(0, str(CODIGO_DIR))

from comun.rutas_proyecto import MODELOS_ARCINIEGAS

ruta_modelo = MODELOS_ARCINIEGAS / "e1" / "firmware" / "modelo_arciniegas_e1_int8.tflite"

print("Ruta:", ruta_modelo)
print("Existe:", os.path.exists(ruta_modelo))
print("Tamaño:", os.path.getsize(ruta_modelo), "bytes")

with open(ruta_modelo, "rb") as f:
    modelo_bytes = f.read()

print("Bytes leídos:", len(modelo_bytes))

interpreter = tf.lite.Interpreter(
    model_content=modelo_bytes
)

interpreter.allocate_tensors()

print("\n=== OPERADORES ===")
for i, op in enumerate(interpreter._get_ops_details()):
    print(i, op["op_name"])

print("\n=== ENTRADA ===")
for x in interpreter.get_input_details():
    print("shape:", x["shape"])
    print("dtype:", x["dtype"])
    print("quantization:", x["quantization"])

print("\n=== SALIDA ===")
for x in interpreter.get_output_details():
    print("shape:", x["shape"])
    print("dtype:", x["dtype"])
    print("quantization:", x["quantization"])
