import json
from pathlib import Path
import sys

CODIGO_DIR = Path(__file__).resolve().parents[2]
if str(CODIGO_DIR) not in sys.path:
    sys.path.insert(0, str(CODIGO_DIR))

from comun.rutas_proyecto import DATOS_ARCINIEGAS_RAW

import numpy as np
try:
    import matplotlib.pyplot as plt
    from scipy.signal import butter, sosfiltfilt, freqz
except ModuleNotFoundError:
    # El modo publicable predeterminado no requiere estas dependencias.
    # Se conservan para reproducir el ensayo interactivo historico mediante
    # el argumento --legacy-interactive en un entorno que incluya SciPy.
    plt = None
    butter = None
    sosfiltfilt = None
    freqz = None

from numpy.fft import (
    rfft,
    rfftfreq,
)


# ============================================================
# CONFIGURACION
# ============================================================

JSON_PATH = (
    DATOS_ARCINIEGAS_RAW
    / "training"
    / "VELOCIDAD ALTA.json.51ir832t.ingestion-6f998cf944-t6sxf.s9.json"
)

# Frecuencia de corte indicada en el paper
FC = 1.0

# Orden del Butterworth
FILTER_ORDER = 4


# ============================================================
# CARGAR JSON
# ============================================================

def cargar_json(path: Path):

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        data = json.load(f)

    payload = data["payload"]

    interval_ms = float(
        payload["interval_ms"]
    )

    fs = (
        1000.0
        /
        interval_ms
    )

    values = np.asarray(
        payload["values"],
        dtype=np.float64,
    )

    if (
        values.ndim != 2
        or
        values.shape[1] != 3
    ):

        raise ValueError(
            "Se esperaba una matriz "
            "[N, 3] correspondiente a "
            "ax, ay y az."
        )

    ax = values[:, 0]
    ay = values[:, 1]
    az = values[:, 2]

    return (
        ax,
        ay,
        az,
        fs,
    )


# ============================================================
# FILTRO PASABAJOS
# ============================================================

def crear_lowpass(
    fs,
    fc=1.0,
    order=4,
):

    nyquist = (
        fs
        /
        2.0
    )

    frecuencia_normalizada = (
        fc
        /
        nyquist
    )

    sos = butter(
        order,
        frecuencia_normalizada,
        btype="lowpass",
        output="sos",
    )

    return sos


def aplicar_filtro(
    x,
    sos,
):

    return sosfiltfilt(
        sos,
        x,
    )


# ============================================================
# RMS
# ============================================================

def rms(x):

    return np.sqrt(
        np.mean(
            x ** 2
        )
    )


# ============================================================
# ENERGIA
# ============================================================

def energia(x):

    return np.sum(
        x ** 2
    )


# ============================================================
# ESPECTRO
# ============================================================

def calcular_espectro(
    x,
    fs,
):
    """
    Para observar vibracion se elimina la media
    antes de calcular FFT.

    Esto NO modifica la señal utilizada por el filtro;
    solo facilita la visualizacion espectral.
    """

    x_centered = (
        x
        -
        np.mean(x)
    )

    X = rfft(
        x_centered
    )

    frecuencias = rfftfreq(
        len(x),
        d=1.0 / fs,
    )

    magnitud = (
        2.0
        /
        len(x)
        *
        np.abs(X)
    )

    return (
        frecuencias,
        magnitud,
    )


# ============================================================
# IMPRIMIR ESTADISTICAS
# ============================================================

def mostrar_estadisticas(
    nombre,
    original,
    filtrada,
):

    rms_original = rms(
        original
    )

    rms_filtrada = rms(
        filtrada
    )

    energia_original = energia(
        original
    )

    energia_filtrada = energia(
        filtrada
    )

    reduccion_rms = (
        1
        -
        rms_filtrada
        /
        rms_original
    ) * 100

    reduccion_energia = (
        1
        -
        energia_filtrada
        /
        energia_original
    ) * 100

    print(
        f"\nEJE {nombre}"
    )

    print(
        "-" * 50
    )

    print(
        f"RMS original       : "
        f"{rms_original:.8f}"
    )

    print(
        f"RMS filtrado       : "
        f"{rms_filtrada:.8f}"
    )

    print(
        f"Reduccion RMS      : "
        f"{reduccion_rms:.2f} %"
    )

    print(
        f"Energia original   : "
        f"{energia_original:.8f}"
    )

    print(
        f"Energia filtrada   : "
        f"{energia_filtrada:.8f}"
    )

    print(
        f"Reduccion energia  : "
        f"{reduccion_energia:.2f} %"
    )


# ============================================================
# GRAFICA TEMPORAL
# ============================================================

def graficar_temporal(
    tiempo,
    original,
    filtrada,
    eje,
):

    plt.figure(
        figsize=(11, 5)
    )

    plt.plot(
        tiempo,
        original,
        label="Original",
    )

    plt.plot(
        tiempo,
        filtrada,
        label="LPF 1 Hz",
    )

    plt.title(
        f"{eje}: señal original vs filtro pasabajos 1 Hz"
    )

    plt.xlabel(
        "Tiempo [s]"
    )

    plt.ylabel(
        "Aceleracion"
    )

    plt.grid(
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    plt.show()


# ============================================================
# GRAFICA SOLO COMPONENTE AC
# ============================================================

def graficar_ac(
    tiempo,
    original,
    filtrada,
    eje,
):
    """
    Restar la media aqui sirve solo para visualizar
    la componente variable de la señal.
    """

    original_ac = (
        original
        -
        np.mean(original)
    )

    filtrada_ac = (
        filtrada
        -
        np.mean(filtrada)
    )

    plt.figure(
        figsize=(11, 5)
    )

    plt.plot(
        tiempo,
        original_ac,
        label="Original sin DC",
    )

    plt.plot(
        tiempo,
        filtrada_ac,
        label="LPF 1 Hz sin DC",
    )

    plt.title(
        f"{eje}: componente variable antes y despues del LPF"
    )

    plt.xlabel(
        "Tiempo [s]"
    )

    plt.ylabel(
        "Amplitud respecto a la media"
    )

    plt.grid(
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    plt.show()


# ============================================================
# GRAFICA ESPECTRAL
# ============================================================

def graficar_espectro(
    original,
    filtrada,
    fs,
    eje,
):

    (
        freq_original,
        mag_original,
    ) = calcular_espectro(
        original,
        fs,
    )

    (
        freq_filtrada,
        mag_filtrada,
    ) = calcular_espectro(
        filtrada,
        fs,
    )

    plt.figure(
        figsize=(11, 5)
    )

    plt.plot(
        freq_original,
        mag_original,
        label="Original",
    )

    plt.plot(
        freq_filtrada,
        mag_filtrada,
        label="LPF 1 Hz",
    )

    plt.axvline(
        FC,
        linestyle="--",
        label="fc = 1 Hz",
    )

    plt.title(
        f"{eje}: espectro antes y despues del filtro"
    )

    plt.xlabel(
        "Frecuencia [Hz]"
    )

    plt.ylabel(
        "Magnitud"
    )

    plt.xlim(
        0,
        fs / 2
    )

    plt.grid(
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    plt.show()


# ============================================================
# RESPUESTA EN FRECUENCIA DEL FILTRO
# ============================================================

def graficar_respuesta_filtro(
    sos,
    fs,
):

    # Butterworth SOS -> respuesta global
    # Se evalua multiplicando las respuestas
    # de cada seccion.

    frecuencias = np.linspace(
        0,
        fs / 2,
        5000,
    )

    w = (
        frecuencias
        /
        (fs / 2)
        *
        np.pi
    )

    respuesta_total = np.ones(
        len(w),
        dtype=np.complex128,
    )

    for section in sos:

        b = section[:3]
        a = section[3:]

        _, h = freqz(
            b,
            a,
            worN=w,
        )

        respuesta_total *= h

    magnitud_db = (
        20
        *
        np.log10(
            np.maximum(
                np.abs(
                    respuesta_total
                ),
                1e-12,
            )
        )
    )

    plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        frecuencias,
        magnitud_db,
    )

    plt.axvline(
        FC,
        linestyle="--",
        label="fc = 1 Hz",
    )

    plt.axhline(
        -3,
        linestyle="--",
        label="-3 dB",
    )

    plt.title(
        "Respuesta en frecuencia del filtro Butterworth"
    )

    plt.xlabel(
        "Frecuencia [Hz]"
    )

    plt.ylabel(
        "Ganancia [dB]"
    )

    plt.xlim(
        0,
        fs / 2
    )

    plt.ylim(
        -100,
        5
    )

    plt.grid(
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    plt.show()


# ============================================================
# COMPARAR BINS DE LA FFT DE 16 MUESTRAS
# ============================================================

def mostrar_bins_fft(
    fs,
):

    bins = rfftfreq(
        16,
        d=1.0 / fs,
    )

    print("\n")
    print("=" * 70)
    print(
        "BINS DE FRECUENCIA PARA FFT = 16"
    )
    print("=" * 70)

    print(
        bins
    )

    print(
        "\nResolucion espectral:"
    )

    print(
        f"Δf = Fs/N = "
        f"{fs:.2f}/16 = "
        f"{fs / 16:.4f} Hz"
    )

    print(
        "\nPrimer bin distinto de DC:"
    )

    print(
        f"{bins[1]:.4f} Hz"
    )

    print(
        "\nFrecuencia de corte del filtro:"
    )

    print(
        f"{FC:.4f} Hz"
    )

    if bins[1] > FC:

        print(
            "\nATENCION:"
        )

        print(
            "El primer bin no-DC de la FFT "
            "esta por encima de la frecuencia "
            "de corte de 1 Hz."
        )


# ============================================================
# MAIN
# ============================================================

def main():

    # Por defecto se generan figuras y tablas persistentes para el articulo.
    # El flujo interactivo historico permanece disponible de forma explicita.
    if "--legacy-interactive" not in sys.argv:
        from generar_evidencia_filtro_1hz import generar_evidencia

        generar_evidencia(JSON_PATH)
        return

    if plt is None or butter is None or sosfiltfilt is None or freqz is None:
        raise RuntimeError(
            "El modo --legacy-interactive requiere matplotlib y scipy. "
            "Ejecute sin ese argumento para generar la evidencia publicable."
        )

    (
        ax,
        ay,
        az,
        fs,
    ) = cargar_json(
        JSON_PATH
    )

    numero_muestras = len(
        ax
    )

    duracion = (
        numero_muestras
        /
        fs
    )

    tiempo = (
        np.arange(
            numero_muestras
        )
        /
        fs
    )

    print("\n")
    print("=" * 70)
    print(
        "ANALISIS DEL EFECTO DEL FILTRO"
    )
    print("=" * 70)

    print(
        f"Archivo:"
    )

    print(
        JSON_PATH
    )

    print(
        f"\nMuestras: "
        f"{numero_muestras}"
    )

    print(
        f"Fs: "
        f"{fs:.2f} Hz"
    )

    print(
        f"Duracion: "
        f"{duracion:.2f} s"
    )

    print(
        f"Filtro: Butterworth "
        f"pasabajos"
    )

    print(
        f"Orden: "
        f"{FILTER_ORDER}"
    )

    print(
        f"fc: "
        f"{FC:.2f} Hz"
    )

    # ========================================================
    # CREAR FILTRO
    # ========================================================

    sos = crear_lowpass(
        fs=fs,
        fc=FC,
        order=FILTER_ORDER,
    )

    # ========================================================
    # FILTRAR
    # ========================================================

    ax_f = aplicar_filtro(
        ax,
        sos,
    )

    ay_f = aplicar_filtro(
        ay,
        sos,
    )

    az_f = aplicar_filtro(
        az,
        sos,
    )

    # ========================================================
    # ESTADISTICAS
    # ========================================================

    mostrar_estadisticas(
        "AX",
        ax,
        ax_f,
    )

    mostrar_estadisticas(
        "AY",
        ay,
        ay_f,
    )

    mostrar_estadisticas(
        "AZ",
        az,
        az_f,
    )

    # ========================================================
    # BINS FFT
    # ========================================================

    mostrar_bins_fft(
        fs
    )

    # ========================================================
    # RESPUESTA DEL FILTRO
    # ========================================================

    graficar_respuesta_filtro(
        sos,
        fs,
    )

    # ========================================================
    # EJE X
    # ========================================================

    graficar_temporal(
        tiempo,
        ax,
        ax_f,
        "AX",
    )

    graficar_ac(
        tiempo,
        ax,
        ax_f,
        "AX",
    )

    graficar_espectro(
        ax,
        ax_f,
        fs,
        "AX",
    )

    # ========================================================
    # EJE Y
    # ========================================================

    graficar_temporal(
        tiempo,
        ay,
        ay_f,
        "AY",
    )

    graficar_ac(
        tiempo,
        ay,
        ay_f,
        "AY",
    )

    graficar_espectro(
        ay,
        ay_f,
        fs,
        "AY",
    )

    # ========================================================
    # EJE Z
    # ========================================================

    graficar_temporal(
        tiempo,
        az,
        az_f,
        "AZ",
    )

    graficar_ac(
        tiempo,
        az,
        az_f,
        "AZ",
    )

    graficar_espectro(
        az,
        az_f,
        fs,
        "AZ",
    )


# ============================================================
# EJECUCION
# ============================================================

if __name__ == "__main__":
    main()
