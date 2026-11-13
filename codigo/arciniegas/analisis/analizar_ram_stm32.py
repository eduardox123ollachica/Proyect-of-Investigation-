import serial
import time
import numpy as np


# ============================================================
# CONFIGURACIÓN
# ============================================================

PUERTO = "COM4"
BAUDRATE = 115200
TIMEOUT = 5

# Elegir uno:
#
# "FEATURES"
# "DNN"
# "KMEANS"
# "ALL"
#
MODO = "ALL"

# Si es True, ignora MODO y ejecuta los 4.
EJECUTAR_TODOS = True


# ============================================================
# VENTANA RAW
#
# Debe contener:
#
# 16 muestras x 3 ejes
#
# [
#   [ax0, ay0, az0],
#   [ax1, ay1, az1],
#   ...
#   [ax15, ay15, az15]
# ]
#
# Sustituye esta ventana por una ventana REAL de tu dataset.
# ============================================================

window = np.array([
    [ 0.12, -0.03,  0.98],
    [ 0.15, -0.02,  1.01],
    [ 0.11, -0.04,  0.99],
    [ 0.13, -0.01,  1.00],
    [ 0.14, -0.03,  0.97],
    [ 0.12, -0.02,  0.98],
    [ 0.16, -0.01,  1.02],
    [ 0.13, -0.04,  0.99],
    [ 0.11, -0.03,  1.00],
    [ 0.14, -0.02,  0.98],
    [ 0.12, -0.01,  1.01],
    [ 0.13, -0.03,  0.99],
    [ 0.15, -0.02,  1.00],
    [ 0.11, -0.04,  0.98],
    [ 0.14, -0.01,  1.02],
    [ 0.12, -0.03,  0.99],
], dtype=np.float32)


# ============================================================
# VALIDAR VENTANA
# ============================================================

if window.shape != (16, 3):
    raise ValueError(
        f"La ventana debe tener forma (16,3), "
        f"pero tiene {window.shape}"
    )


# ============================================================
# CONSTRUIR COMANDO
# ============================================================

def construir_comando(modo, window):
    modo = modo.upper()

    modos_validos = [
        "FEATURES",
        "DNN",
        "KMEANS",
        "ALL"
    ]

    if modo not in modos_validos:
        raise ValueError(
            f"Modo inválido: {modo}"
        )

    # Orden:
    #
    # ax0, ay0, az0,
    # ax1, ay1, az1,
    # ...
    valores = window.flatten()

    texto_valores = ",".join(
        f"{float(v):.9f}"
        for v in valores
    )

    comando = (
        f"STACKBENCH,{modo},"
        f"{texto_valores}\n"
    )

    return comando


# ============================================================
# ENVIAR COMANDO
# ============================================================

def ejecutar_stackbench(ser, modo, window):

    comando = construir_comando(
        modo,
        window
    )

    print("\n" + "=" * 70)
    print(f"MODO: {modo}")
    print("=" * 70)

    print("Enviando comando...")

    # Limpiar buffers
    ser.reset_input_buffer()
    ser.reset_output_buffer()

    # Enviar
    ser.write(
        comando.encode("ascii")
    )

    ser.flush()

    # Leer respuesta
    respuesta = ser.readline().decode(
        "ascii",
        errors="replace"
    ).strip()

    print("STM32:")
    print(respuesta)

    return respuesta


# ============================================================
# INTERPRETAR RESPUESTA
# ============================================================

def interpretar_respuesta(respuesta):

    partes = respuesta.split(",")

    if len(partes) != 6:
        print(
            "Respuesta con formato inesperado."
        )
        return None

    if partes[0] != "STACKBENCH":
        print(
            "La STM32 devolvió un error "
            "o una respuesta diferente."
        )
        return None

    try:

        resultado = {
            "modo": partes[1],
            "raw_bytes": int(partes[2]),
            "baseline_bytes": int(partes[3]),
            "ajustado_bytes": int(partes[4]),
            "status": int(partes[5]),
        }

    except ValueError:
        print(
            "No se pudieron convertir "
            "los valores de la respuesta."
        )
        return None

    return resultado


# ============================================================
# MAIN
# ============================================================

print(f"Abriendo {PUERTO}...")

with serial.Serial(
    PUERTO,
    BAUDRATE,
    timeout=TIMEOUT
) as ser:

    # Dar tiempo por si la placa reinicia al abrir COM
    time.sleep(2)

    resultados = []


    # ========================================================
    # EJECUTAR LOS CUATRO
    # ========================================================

    if EJECUTAR_TODOS:

        modos = [
            "FEATURES",
            "DNN",
            "KMEANS",
            "ALL"
        ]

    else:

        modos = [
            MODO
        ]


    for modo in modos:

        respuesta = ejecutar_stackbench(
            ser,
            modo,
            window
        )

        resultado = interpretar_respuesta(
            respuesta
        )

        if resultado is not None:
            resultados.append(
                resultado
            )

        time.sleep(0.2)


# ============================================================
# RESUMEN
# ============================================================

print("\n")
print("=" * 70)
print("RESUMEN STACKBENCH")
print("=" * 70)

if not resultados:
    print("No se obtuvieron resultados válidos.")

else:

    print(
        f"{'Modo':<12}"
        f"{'Raw [B]':>12}"
        f"{'Base [B]':>12}"
        f"{'Ajustado [B]':>16}"
        f"{'Status':>10}"
    )

    print("-" * 62)

    for r in resultados:

        print(
            f"{r['modo']:<12}"
            f"{r['raw_bytes']:>12}"
            f"{r['baseline_bytes']:>12}"
            f"{r['ajustado_bytes']:>16}"
            f"{r['status']:>10}"
        )