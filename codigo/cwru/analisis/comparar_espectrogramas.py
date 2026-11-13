# ============================================================
# ESPECTROGRAMAS DATASET PROCESADO CWRU TinyML
#
# Usa:
# processed/X.npy
# processed/y.npy
# processed/metadata.csv
#
# El dataset ya tiene:
# - filtro anti alias
# - decimación 24
# - Fs=2000 Hz
# - ventanas 1024
# - Z-score
#
# ============================================================


from pathlib import Path
import sys

CODIGO_DIR = Path(__file__).resolve().parents[2]
if str(CODIGO_DIR) not in sys.path:
    sys.path.insert(0, str(CODIGO_DIR))

from comun.rutas_proyecto import DATOS_CWRU_PROCESADO

import numpy as np
import pandas as pd

import scipy.signal as signal

import matplotlib.pyplot as plt



# ============================================================
# CONFIGURACIÓN
# ============================================================


PROCESSED_DIR = DATOS_CWRU_PROCESADO


X_FILE = PROCESSED_DIR / "X.npy"

Y_FILE = PROCESSED_DIR / "y.npy"

META_FILE = PROCESSED_DIR / "metadata.csv"



FS_TARGET = 2000


WINDOW_SIZE = 1024


OVERLAP = 0.5


N_WINDOWS = 20
# cantidad de ventanas consecutivas a unir



CLASS_NAMES = {

    0:"Normal",

    1:"Inner Race",

    2:"Outer Race"

}



# ============================================================
# CARGAR DATASET
# ============================================================


def load_dataset():


    X = np.load(
        X_FILE
    )


    y = np.load(
        Y_FILE
    )


    metadata = pd.read_csv(
        META_FILE
    )


    print("==============================")

    print("Dataset cargado")

    print("X:",X.shape)

    print("y:",y.shape)

    print("==============================")


    return X,y,metadata




# ============================================================
# CREAR SEÑAL LARGA
# ============================================================


def build_long_signal(X,y,label):


    indexes = np.where(

        y == label

    )[0]



    if len(indexes)<N_WINDOWS:

        raise ValueError(

            "No existen suficientes ventanas"

        )



    # tomar primeras ventanas consecutivas

    selected = indexes[:N_WINDOWS]



    signal_long = np.concatenate(

        X[selected]

    )



    return signal_long, selected




# ============================================================
# ESPECTROGRAMA
# ============================================================


def compute_spectrogram(x):


    f,t,S = signal.spectrogram(


        x,


        fs=FS_TARGET,


        window="hann",


        nperseg=WINDOW_SIZE,


        noverlap=int(

            WINDOW_SIZE*OVERLAP

        ),


        scaling="density"

    )



    # normalización visual

    S_db = 10*np.log10(

        S / np.max(S)

        + 1e-12

    )


    return f,t,S_db




# ============================================================
# GRAFICAR
# ============================================================


def plot_results(results):


    fig,axes = plt.subplots(

        3,

        1,

        figsize=(12,10),

        sharex=True

    )



    for ax,(name,data) in zip(

        axes,

        results.items()

    ):


        f,t,S_db = data



        img=ax.pcolormesh(

            t,

            f,

            S_db,

            shading="auto"

        )



        ax.set_title(

            name +

            " - Espectrograma 2 kHz"

        )



        ax.set_ylabel(

            "Frecuencia [Hz]"

        )



        ax.set_ylim(

            0,

            1000

        )



        fig.colorbar(

            img,

            ax=ax,

            label="dB"

        )



    axes[-1].set_xlabel(

        "Tiempo [s]"

    )



    plt.tight_layout()

    plt.show()




# ============================================================
# MAIN
# ============================================================


if __name__=="__main__":


    X,y,metadata = load_dataset()



    results={}



    for label,name in CLASS_NAMES.items():


        print("\nProcesando:",name)



        x_long,indexes = build_long_signal(

            X,

            y,

            label

        )


        print(

            "Ventanas usadas:",

            len(indexes)

        )


        print(

            "Duración:",

            len(x_long)/FS_TARGET,

            "segundos"

        )



        f,t,S_db = compute_spectrogram(

            x_long

        )



        results[name]=(

            f,

            t,

            S_db

        )



    plot_results(results)
