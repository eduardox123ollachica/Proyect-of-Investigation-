import serial
import time

PORT = "COM4"
BAUDRATE = 115200

ser = serial.Serial(
    PORT,
    BAUDRATE,
    timeout=3
)

time.sleep(1)

ser.reset_input_buffer()

print("Conectado a", PORT)

# ==========================================================
# VECTOR DE PRUEBA
#
# Deben ser exactamente 21 características.
#
# Por ahora son valores de ejemplo.
# Después reemplazaremos esto con una ventana real
# de tu conjunto de test.
# ==========================================================

features = [
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,

    0.80,
    0.90,
    1.00,
    1.10,
    1.20,
    1.30,
    1.40,

    1.50,
    1.60,
    1.70,
    1.80,
    1.90,
    2.00,
    2.10
]

if len(features) != 21:
    raise ValueError(
        "El vector debe tener exactamente 21 características"
    )

# Construir comando UART
comando = "PRED," + ",".join(
    f"{x:.9f}"
    for x in features
)

print("Enviando:")
print(comando)

ser.write(
    (comando + "\r\n").encode()
)

respuesta = (
    ser.readline()
    .decode(errors="ignore")
    .strip()
)

print()
print("Respuesta STM32:")
print(respuesta)

ser.close()