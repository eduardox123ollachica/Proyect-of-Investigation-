import serial
import time

ser = serial.Serial(
    "COM4",
    115200,
    timeout=2
)

time.sleep(1)

# Limpia cualquier mensaje viejo que haya enviado
# el STM32 al arrancar.
ser.reset_input_buffer()

print("Conectado a COM4")

comandos = [
    "MODELO",
    "TINYML",
    "INFO",
    "QUANT"
]

for comando in comandos:

    print("Enviando:", comando)

    ser.write(
        (comando + "\r\n").encode()
    )

    respuesta = (
        ser.readline()
        .decode(errors="ignore")
        .strip()
    )

    print("Respuesta STM32:", respuesta)

ser.close()