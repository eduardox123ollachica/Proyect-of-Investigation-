from pathlib import Path
import json
import time
import sys

CODIGO_DIR = Path(__file__).resolve().parents[2]
if str(CODIGO_DIR) not in sys.path:
    sys.path.insert(0, str(CODIGO_DIR))

from comun.rutas_proyecto import DATOS_ARCINIEGAS_PROCESADO, DATOS_ARCINIEGAS_RAW

import pandas as pd
import serial


# ============================================================
# CONFIGURACIÓN
# ============================================================

BASE_PREPROCESADO = DATOS_ARCINIEGAS_PROCESADO
BASE_DATASET = DATOS_ARCINIEGAS_RAW / "testing"

INDEX = 0
WINDOW_SIZE = 16

SERIAL_PORT = "COM4"
BAUDRATE = 115200

TIMEOUT_S = 2.0


# ============================================================
# CARGAR METADATA
# ============================================================

metadata_path = (
    BASE_PREPROCESADO
    / "metadata_features.csv"
)

df = pd.read_csv(
    metadata_path
)


# ============================================================
# FILTRAR SOLO TESTING
# ============================================================

df = df[
    df["split"] == "testing"
].reset_index(drop=True)


if INDEX < 0 or INDEX >= len(df):
    raise IndexError(
        f"INDEX={INDEX} fuera de rango. "
        f"El conjunto testing tiene {len(df)} ventanas."
    )


# ============================================================
# SELECCIONAR VENTANA
# ============================================================

fila = df.iloc[INDEX]

archivo = fila["file"]
clase = fila["class"]
label = int(fila["label"])
window_index = int(fila["window"])

start = int(
    fila["start_sample"]
)

end = int(
    fila["end_sample_exclusive"]
)


print("=" * 80)
print("VENTANA SELECCIONADA")
print("=" * 80)

print(
    f"Índice test    : {INDEX}"
)

print(
    f"Archivo        : {archivo}"
)

print(
    f"Clase real     : {clase}"
)

print(
    f"Label real     : {label}"
)

print(
    f"Window         : {window_index}"
)

print(
    f"Inicio         : {start}"
)

print(
    f"Fin exclusivo  : {end}"
)

print(
    f"Muestras       : {end - start}"
)


# ============================================================
# CARGAR JSON ORIGINAL
# ============================================================

json_path = (
    BASE_DATASET
    / archivo
)

if not json_path.exists():
    raise FileNotFoundError(
        f"No se encontró:\n{json_path}"
    )


with open(
    json_path,
    "r",
    encoding="utf-8"
) as f:

    data = json.load(f)


# ============================================================
# EXTRAER VENTANA CRUDA
# ============================================================

values = (
    data["payload"]["values"]
)

window = values[
    start:end
]


if len(window) != WINDOW_SIZE:
    raise ValueError(
        f"La ventana tiene "
        f"{len(window)} muestras. "
        f"Se esperaban {WINDOW_SIZE}."
    )


# ============================================================
# CONSTRUIR COMANDO RAWPRED
# ============================================================

valores = []

for muestra in window:

    if len(muestra) < 3:
        raise ValueError(
            "Una muestra tiene menos "
            "de 3 ejes."
        )

    ax = float(
        muestra[0]
    )

    ay = float(
        muestra[1]
    )

    az = float(
        muestra[2]
    )

    valores.extend(
        [
            f"{ax:.9f}",
            f"{ay:.9f}",
            f"{az:.9f}",
        ]
    )


if len(valores) != 48:
    raise ValueError(
        f"Se generaron {len(valores)} "
        "valores en lugar de 48."
    )


comando = (
    "RAWPRED,"
    +
    ",".join(valores)
)


# ============================================================
# MOSTRAR COMANDO
# ============================================================

print()

print("=" * 80)
print("COMANDO RAWPRED")
print("=" * 80)

print(comando)

print()

print(
    f"Cantidad valores : "
    f"{len(valores)}"
)

print(
    f"Longitud comando : "
    f"{len(comando)} caracteres"
)


# ============================================================
# ABRIR PUERTO SERIAL
# ============================================================

print()

print("=" * 80)
print("COMUNICACIÓN STM32")
print("=" * 80)

print(
    f"Abriendo "
    f"{SERIAL_PORT} "
    f"a {BAUDRATE} baudios..."
)


try:

    with serial.Serial(
        port=SERIAL_PORT,
        baudrate=BAUDRATE,
        timeout=TIMEOUT_S
    ) as ser:

        # ----------------------------------------------------
        # Dar un pequeño tiempo al puerto
        # ----------------------------------------------------

        time.sleep(
            0.2
        )


        # ----------------------------------------------------
        # Limpiar buffers
        # ----------------------------------------------------

        ser.reset_input_buffer()
        ser.reset_output_buffer()


        # ----------------------------------------------------
        # Enviar comando
        #
        # IMPORTANTE:
        # añadimos \r\n porque el firmware normalmente
        # interpreta el final de línea como fin del comando.
        # ----------------------------------------------------

        trama = (
            comando
            +
            "\r\n"
        )


        print(
            "Enviando RAWPRED..."
        )


        ser.write(
            trama.encode(
                "ascii"
            )
        )

        ser.flush()


        # ----------------------------------------------------
        # RECIBIR RESPUESTA
        # ----------------------------------------------------

        respuesta = (
            ser.readline()
            .decode(
                "ascii",
                errors="replace"
            )
            .strip()
        )


        if not respuesta:

            print(
                "ERROR: "
                "No se recibió respuesta "
                "de la STM32."
            )

        else:

            print()

            print(
                "Respuesta STM32:"
            )

            print(
                respuesta
            )


            # ================================================
            # INTERPRETAR RAWPRED
            # ================================================

            if respuesta.startswith(
                "RAWPRED,"
            ):

                partes = (
                    respuesta.split(",")
                )


                if len(partes) == 5:

                    clase_stm32 = int(
                        partes[1]
                    )

                    prob0 = float(
                        partes[2]
                    )

                    prob1 = float(
                        partes[3]
                    )

                    prob2 = float(
                        partes[4]
                    )


                    print()

                    print("=" * 80)
                    print("RESULTADO")
                    print("=" * 80)

                    print(
                        f"Clase real STM32 : "
                        f"{label} ({clase})"
                    )

                    print(
                        f"Clase predicha   : "
                        f"{clase_stm32}"
                    )

                    print(
                        f"P(DETENIDO)      : "
                        f"{prob0:.6f}"
                    )

                    print(
                        f"P(MEDIA)         : "
                        f"{prob1:.6f}"
                    )

                    print(
                        f"P(ALTA)          : "
                        f"{prob2:.6f}"
                    )


                    correcto = (
                        clase_stm32
                        ==
                        label
                    )


                    print()

                    print(
                        "Clasificación     : "
                        +
                        (
                            "CORRECTA"
                            if correcto
                            else
                            "INCORRECTA"
                        )
                    )


                else:

                    print(
                        "Respuesta RAWPRED "
                        "con formato inesperado."
                    )


            elif respuesta.startswith(
                "RAWPRED_ERROR"
            ):

                print(
                    "La STM32 reportó "
                    "un error RAWPRED."
                )


            else:

                print(
                    "Respuesta inesperada "
                    "del firmware."
                )


except serial.SerialException as e:

    print()

    print(
        "ERROR SERIAL:"
    )

    print(
        e
    )

    print()

    print(
        "Verifica que:"
    )

    print(
        "- COM4 sea el puerto correcto"
    )

    print(
        "- ningún terminal serial "
        "tenga abierto COM4"
    )

    print(
        "- la STM32 esté conectada"
    )
