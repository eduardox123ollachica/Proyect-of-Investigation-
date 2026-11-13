from pathlib import Path
import sys

CODIGO_DIR = Path(__file__).resolve().parents[2]
if str(CODIGO_DIR) not in sys.path:
    sys.path.insert(0, str(CODIGO_DIR))

from comun.rutas_proyecto import MODELOS_ARCINIEGAS

FIRMWARE_DIR = MODELOS_ARCINIEGAS / "e4" / "firmware"
MODELO = FIRMWARE_DIR / "modelo_arciniegas_e4_int8.tflite"
SALIDA_H = FIRMWARE_DIR / "model_data.h"
SALIDA_CC = FIRMWARE_DIR / "model_data.cc"

datos = MODELO.read_bytes()

with SALIDA_H.open("w", encoding="utf-8") as f:
    f.write("#ifndef MODEL_DATA_H\n")
    f.write("#define MODEL_DATA_H\n\n")
    f.write("extern const unsigned char g_model[];\n")
    f.write("extern const unsigned int g_model_len;\n\n")
    f.write("#endif\n")

with SALIDA_CC.open("w", encoding="utf-8") as f:
    f.write('#include "model_data.h"\n\n')
    f.write("alignas(16) const unsigned char g_model[] = {\n")

    for i, byte in enumerate(datos):
        if i % 12 == 0:
            f.write("    ")

        f.write(f"0x{byte:02x}")

        if i != len(datos) - 1:
            f.write(", ")

        if i % 12 == 11:
            f.write("\n")

    if len(datos) % 12 != 0:
        f.write("\n")

    f.write("};\n\n")
    f.write(
        f"const unsigned int g_model_len = {len(datos)};\n"
    )

print("Modelo convertido correctamente.")
print("Tamaño:", len(datos), "bytes")
print("Generados:")
print(SALIDA_H.resolve())
print(SALIDA_CC.resolve())
