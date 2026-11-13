from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Iterable

CODIGO_DIR = Path(__file__).resolve().parents[2]
if str(CODIGO_DIR) not in sys.path:
    sys.path.insert(0, str(CODIGO_DIR))

from comun.rutas_proyecto import DATOS_ARCINIEGAS_PROCESADO, RESULTADOS_E2

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import joblib
import matplotlib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler,
)

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import (
    layers,
    regularizers,
)


# ============================================================
# CONFIGURACION GENERAL
# ============================================================

SEMILLA = 42

NOMBRES_CLASE = [
    "DETENIDO",
    "MEDIA",
    "ALTA",
]

ARCHIVOS_REQUERIDOS = (
    "X_train.npy",
    "y_train.npy",
    "X_test.npy",
    "y_test.npy",
    "metadata_features.csv",
)


# ============================================================
# ARGUMENTOS
# ============================================================

def argumentos() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Replica metodologica DNN "
            "21->20->10->3 de Arciniegas "
            "con analisis TinyML FP32 e INT8."
        )
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATOS_ARCINIEGAS_PROCESADO,
        help=(
            "Carpeta que contiene los "
            ".npy y metadata_features.csv."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTADOS_E2 / "entrenamiento",
        help=(
            "Carpeta donde se guardaran "
            "todos los resultados."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=SEMILLA,
        help="Semilla reproducible para Python, NumPy y TensorFlow.",
    )

    return parser.parse_args()


# ============================================================
# SEMILLAS
# ============================================================

def fijar_semillas(
    semilla: int = SEMILLA,
) -> None:

    os.environ[
        "PYTHONHASHSEED"
    ] = str(semilla)

    random.seed(
        semilla
    )

    np.random.seed(
        semilla
    )

    tf.keras.utils.set_random_seed(
        semilla
    )


# ============================================================
# VALIDACION DE ARCHIVOS
# ============================================================

def validar_archivos(
    data_dir: Path,
) -> None:

    faltantes = [
        nombre
        for nombre in ARCHIVOS_REQUERIDOS
        if not (
            data_dir / nombre
        ).is_file()
    ]

    if faltantes:

        raise FileNotFoundError(
            f"Faltan archivos en "
            f"'{data_dir.resolve()}': "
            f"{', '.join(faltantes)}"
        )


# ============================================================
# CARGA DE DATOS
# ============================================================

def cargar_datos(
    data_dir: Path,
) -> tuple[np.ndarray, ...]:

    validar_archivos(
        data_dir
    )

    X_train = np.load(
        data_dir / "X_train.npy",
        allow_pickle=False,
    )

    y_train = np.load(
        data_dir / "y_train.npy",
        allow_pickle=False,
    )

    X_test = np.load(
        data_dir / "X_test.npy",
        allow_pickle=False,
    )

    y_test = np.load(
        data_dir / "y_test.npy",
        allow_pickle=False,
    )

    X_train = np.asarray(
        X_train,
        dtype=np.float32,
    )

    X_test = np.asarray(
        X_test,
        dtype=np.float32,
    )

    y_train = np.asarray(
        y_train
    ).reshape(-1)

    y_test = np.asarray(
        y_test
    ).reshape(-1)

    if (
        X_train.ndim != 2
        or
        X_test.ndim != 2
    ):

        raise ValueError(
            "X_train y X_test deben ser "
            "matrices 2D "
            "[muestras, caracteristicas]."
        )

    if (
        X_train.shape[1] != 21
        or
        X_test.shape[1] != 21
    ):

        raise ValueError(
            "La arquitectura exige "
            "21 caracteristicas; se recibieron "
            f"{X_train.shape[1]} en train y "
            f"{X_test.shape[1]} en test."
        )

    if (
        len(X_train) != len(y_train)
        or
        len(X_test) != len(y_test)
    ):

        raise ValueError(
            "El numero de muestras de "
            "X e y no coincide."
        )

    if (
        not np.isfinite(
            X_train
        ).all()
        or
        not np.isfinite(
            X_test
        ).all()
    ):

        raise ValueError(
            "X_train o X_test contiene "
            "NaN o valores infinitos."
        )

    return (
        X_train,
        y_train,
        X_test,
        y_test,
    )


# ============================================================
# CODIFICACION DE ETIQUETAS
# ============================================================

def codificar_etiquetas(
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[str],
    LabelEncoder,
]:

    encoder = LabelEncoder()

    y_train_cod = (
        encoder.fit_transform(
            y_train
        )
    )

    try:

        y_test_cod = (
            encoder.transform(
                y_test
            )
        )

    except ValueError as exc:

        raise ValueError(
            "y_test contiene clases "
            "que no aparecen en y_train."
        ) from exc

    if len(
        encoder.classes_
    ) != 3:

        raise ValueError(
            "Se esperaban exactamente "
            "3 clases y se encontraron "
            f"{len(encoder.classes_)}: "
            f"{encoder.classes_.tolist()}"
        )

    if np.array_equal(
        encoder.classes_,
        np.array([0, 1, 2]),
    ):

        nombres = (
            NOMBRES_CLASE.copy()
        )

    else:

        nombres = [
            str(valor)
            for valor
            in encoder.classes_
        ]

        print(
            "Aviso: las etiquetas "
            "originales no son [0, 1, 2]."
        )

        print(
            "Las probabilidades siguen "
            "el orden de LabelEncoder:"
        )

        print(
            nombres
        )

    return (
        y_train_cod,
        y_test_cod,
        nombres,
        encoder,
    )


# ============================================================
# METADATA
# ============================================================

def cargar_metadata(
    data_dir: Path,
    n_train: int,
    n_test: int,
) -> pd.DataFrame:

    metadata = pd.read_csv(
        data_dir
        /
        "metadata_features.csv"
    )

    columnas_split = [
        c
        for c in metadata.columns
        if c.lower()
        in {
            "split",
            "set",
            "conjunto",
        }
    ]

    if columnas_split:

        serie = (
            metadata[
                columnas_split[0]
            ]
            .astype(str)
            .str.lower()
            .str.strip()
        )

        mascara = serie.isin(
            {
                "test",
                "testing",
                "prueba",
            }
        )

        if int(
            mascara.sum()
        ) == n_test:

            return (
                metadata
                .loc[mascara]
                .reset_index(
                    drop=True
                )
            )

    if len(
        metadata
    ) == n_test:

        return (
            metadata
            .reset_index(
                drop=True
            )
        )

    if len(
        metadata
    ) == (
        n_train
        +
        n_test
    ):

        return (
            metadata
            .iloc[n_train:]
            .reset_index(
                drop=True
            )
        )

    raise ValueError(
        "No se pudo alinear "
        "metadata_features.csv con test: "
        f"metadata={len(metadata)}, "
        f"train={n_train}, "
        f"test={n_test}."
    )


# ============================================================
# MODELO DNN
# ============================================================

def construir_modelo() -> keras.Model:

    penalizacion = (
        regularizers.l2(
            0.001
        )
    )

    modelo = keras.Sequential(
        [
            layers.Input(
                shape=(21,),
                name=(
                    "entrada_21_features"
                ),
            ),

            layers.Dense(
                20,
                activation="relu",
                kernel_regularizer=(
                    penalizacion
                ),
                name="oculta_20",
            ),

            layers.Dense(
                10,
                activation="relu",
                kernel_regularizer=(
                    penalizacion
                ),
                name="oculta_10",
            ),

            layers.Dense(
                3,
                activation="softmax",
                name=(
                    "salida_3_clases"
                ),
            ),
        ],

        name=(
            "replica_arciniegas_"
            "21_20_10_3"
        ),
    )

    modelo.compile(
        optimizer=(
            keras.optimizers.Adam(
                learning_rate=0.005
            )
        ),

        loss=(
            "sparse_categorical_"
            "crossentropy"
        ),

        metrics=[
            "accuracy"
        ],
    )

    return modelo


# ============================================================
# CURVAS
# ============================================================

def guardar_curvas(
    historial: pd.DataFrame,
    output_dir: Path,
) -> None:

    epocas = historial[
        "epoch"
    ]

    for (
        metrica,
        etiqueta,
        archivo,
    ) in (
        (
            "loss",
            "Perdida",
            "curva_loss.png",
        ),
        (
            "accuracy",
            "Accuracy",
            "curva_accuracy.png",
        ),
    ):

        fig, ax = plt.subplots(
            figsize=(8, 5)
        )

        ax.plot(
            epocas,
            historial[metrica],
            label="Entrenamiento",
            linewidth=2,
        )

        clave_val = (
            f"val_{metrica}"
        )

        if clave_val in historial:

            ax.plot(
                epocas,
                historial[
                    clave_val
                ],
                label="Validacion",
                linewidth=2,
            )

        ax.set(
            title=(
                f"Curva de {etiqueta}"
            ),
            xlabel="Epoca",
            ylabel=etiqueta,
        )

        ax.grid(
            alpha=0.25
        )

        ax.legend()

        fig.tight_layout()

        fig.savefig(
            output_dir
            /
            archivo,
            dpi=200,
        )

        plt.close(
            fig
        )


# ============================================================
# MATRIZ DE CONFUSION
# ============================================================

def guardar_matriz_confusion(
    matriz: np.ndarray,
    nombres: Iterable[str],
    output_dir: Path,
) -> None:

    nombres = list(
        nombres
    )

    pd.DataFrame(
        matriz,

        index=[
            f"REAL_{n}"
            for n in nombres
        ],

        columns=[
            f"PRED_{n}"
            for n in nombres
        ],

    ).to_csv(
        output_dir
        /
        "matriz_confusion.csv",

        index_label=(
            "clase_real"
        ),
    )

    fig, ax = plt.subplots(
        figsize=(7, 6)
    )

    imagen = ax.imshow(
        matriz,
        interpolation="nearest",
        cmap="Blues",
    )

    fig.colorbar(
        imagen,
        ax=ax,
        fraction=0.046,
        pad=0.04,
    )

    ax.set_xticks(
        np.arange(
            len(nombres)
        ),
        labels=nombres,
    )

    ax.set_yticks(
        np.arange(
            len(nombres)
        ),
        labels=nombres,
    )

    umbral = (
        matriz.max() / 2
        if matriz.size
        else 0
    )

    for fila in range(
        matriz.shape[0]
    ):

        for columna in range(
            matriz.shape[1]
        ):

            ax.text(
                columna,
                fila,

                str(
                    matriz[
                        fila,
                        columna,
                    ]
                ),

                ha="center",
                va="center",

                color=(
                    "white"
                    if
                    matriz[
                        fila,
                        columna,
                    ]
                    >
                    umbral
                    else
                    "black"
                ),
            )

    ax.set(
        title=(
            "Matriz de confusion"
        ),
        xlabel=(
            "Clase predicha"
        ),
        ylabel=(
            "Clase real"
        ),
    )

    fig.tight_layout()

    fig.savefig(
        output_dir
        /
        "matriz_confusion.png",
        dpi=200,
    )

    plt.close(
        fig
    )


# ============================================================
# METRICAS
# ============================================================

def calcular_metricas(
    y_real: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:

    (
        p_macro,
        r_macro,
        f_macro,
        _,
    ) = (
        precision_recall_fscore_support(
            y_real,
            y_pred,
            average="macro",
            zero_division=0,
        )
    )

    (
        p_weighted,
        r_weighted,
        f_weighted,
        _,
    ) = (
        precision_recall_fscore_support(
            y_real,
            y_pred,
            average="weighted",
            zero_division=0,
        )
    )

    return {
        "accuracy":
            accuracy_score(
                y_real,
                y_pred,
            ),

        "macro_precision":
            p_macro,

        "macro_recall":
            r_macro,

        "macro_f1":
            f_macro,

        "weighted_precision":
            p_weighted,

        "weighted_recall":
            r_weighted,

        "weighted_f1":
            f_weighted,
    }


# ============================================================
# INFORME DE LA REPLICA
# ============================================================

def guardar_info(
    output_dir: Path,
    X_train: np.ndarray,
    X_test: np.ndarray,
    clases_originales: np.ndarray,
    nombres: list[str],
    metricas: dict[str, float],
) -> None:

    publicadas = {
        "accuracy": 0.9650,
        "precision": 0.9619,
        "recall": 0.9711,
        "f1": 0.9664,
    }

    texto = f"""
REPLICA METODOLOGICA DEL ARTICULO DE ARCINIEGAS
================================================

Naturaleza del experimento
--------------------------
Esta ejecucion es una replica metodologica
y no una reproduccion exacta.

Arquitectura
------------
Entrada: 21 caracteristicas
Oculta 1: 20 neuronas, ReLU, L2=0.001
Oculta 2: 10 neuronas, ReLU, L2=0.001
Salida: 3 neuronas, Softmax
Topologia: 21 -> 20 -> 10 -> 3

Hiperparametros
---------------
Optimizador: Adam
Learning rate: 0.005
Batch size: 32
Epocas: 50
Funcion de perdida:
sparse categorical crossentropy
class_weight: no utilizado
Semilla: {SEMILLA}

Datos y metodologia
-------------------
Shape X_train original:
{X_train.shape}

Shape X_test original:
{X_test.shape}

Clases originales:
{clases_originales.tolist()}

Nombres mostrados:
{nombres}

StandardScaler fue ajustado
exclusivamente con X_train.

El conjunto test no se uso
durante model.fit.

Metricas de esta replica
------------------------
Accuracy:
{metricas['accuracy'] * 100:.2f} %

Macro precision:
{metricas['macro_precision'] * 100:.2f} %

Macro recall:
{metricas['macro_recall'] * 100:.2f} %

Macro F1:
{metricas['macro_f1'] * 100:.2f} %

Weighted precision:
{metricas['weighted_precision'] * 100:.2f} %

Weighted recall:
{metricas['weighted_recall'] * 100:.2f} %

Weighted F1:
{metricas['weighted_f1'] * 100:.2f} %

Metricas publicadas
-------------------
Accuracy:
{publicadas['accuracy'] * 100:.2f} %

Precision:
{publicadas['precision'] * 100:.2f} %

Recall:
{publicadas['recall'] * 100:.2f} %

F1:
{publicadas['f1'] * 100:.2f} %

Diferencia replica - articulo
-----------------------------
Accuracy:
{(metricas['accuracy'] - publicadas['accuracy']) * 100:+.2f} pp

Precision:
{(metricas['macro_precision'] - publicadas['precision']) * 100:+.2f} pp

Recall:
{(metricas['macro_recall'] - publicadas['recall']) * 100:+.2f} pp

F1:
{(metricas['macro_f1'] - publicadas['f1']) * 100:+.2f} pp
"""

    (
        output_dir
        /
        "replica_info.txt"
    ).write_text(
        texto,
        encoding="utf-8",
    )


# ============================================================
# FUNCIONES TINYML
# ============================================================

def tamano_archivo_bytes(
    ruta: Path,
) -> int:

    return ruta.stat().st_size


def bytes_a_kib(
    numero_bytes: int | float,
) -> float:

    return (
        float(numero_bytes)
        /
        1024.0
    )


def estimar_memoria_tensores_tflite(
    interpreter: tf.lite.Interpreter,
) -> int:
    """
    Estimacion orientativa de memoria asociada
    a los tensores conocidos por TFLite.

    NO debe interpretarse como RAM real del MCU
    ni como Tensor Arena real de TFLite Micro.
    """

    total_bytes = 0

    tensor_details = (
        interpreter
        .get_tensor_details()
    )

    for detalle in tensor_details:

        shape = detalle[
            "shape"
        ]

        if shape is None:
            continue

        numero_elementos = int(
            np.prod(
                shape
            )
        )

        bytes_elemento = (
            np.dtype(
                detalle[
                    "dtype"
                ]
            )
            .itemsize
        )

        total_bytes += (
            numero_elementos
            *
            bytes_elemento
        )

    return int(
        total_bytes
    )


# ============================================================
# EVALUAR MODELO TFLITE
# ============================================================

def evaluar_modelo_tflite(
    ruta_modelo: Path,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[
    float,
    int,
    np.ndarray,
]:
    """
    Evalua automaticamente modelos:
    - TFLite FP32
    - TFLite INT8

    Retorna:
    - accuracy
    - memoria estimada de tensores
    - predicciones
    """

    # ========================================================
    # VERIFICAR QUE EL MODELO EXISTA
    # ========================================================

    if not ruta_modelo.exists():

        raise FileNotFoundError(
            f"No se encontro el modelo TFLite:\n"
            f"{ruta_modelo}"
        )

    print(
        f"\nCargando modelo TFLite:\n"
        f"{ruta_modelo}"
    )

    print(
        f"Tamaño del archivo: "
        f"{ruta_modelo.stat().st_size} bytes"
    )

    # ========================================================
    # CARGAR MODELO COMO BYTES
    # ========================================================
    #
    # Se usa model_content en lugar de model_path
    # para evitar problemas con rutas Unicode en Windows,
    # por ejemplo carpetas como "Proyecto Investigación".
    # ========================================================

    contenido_modelo = (
        ruta_modelo.read_bytes()
    )

    interpreter = tf.lite.Interpreter(
        model_content=contenido_modelo
    )

    interpreter.allocate_tensors()

    # ========================================================
    # INFORMACION DE ENTRADA Y SALIDA
    # ========================================================

    input_details = (
        interpreter
        .get_input_details()[0]
    )

    output_details = (
        interpreter
        .get_output_details()[0]
    )

    input_index = (
        input_details[
            "index"
        ]
    )

    output_index = (
        output_details[
            "index"
        ]
    )

    input_dtype = (
        input_details[
            "dtype"
        ]
    )

    output_dtype = (
        output_details[
            "dtype"
        ]
    )

    (
        input_scale,
        input_zero_point,
    ) = (
        input_details[
            "quantization"
        ]
    )

    (
        output_scale,
        output_zero_point,
    ) = (
        output_details[
            "quantization"
        ]
    )

    print(
        f"Tipo de entrada : "
        f"{input_dtype}"
    )

    print(
        f"Tipo de salida  : "
        f"{output_dtype}"
    )

    if input_dtype == np.int8:

        print(
            f"Input scale     : "
            f"{input_scale}"
        )

        print(
            f"Input zero point: "
            f"{input_zero_point}"
        )

    # ========================================================
    # INFERENCIA
    # ========================================================

    predicciones = []

    for muestra in X_test:

        muestra = (
            muestra
            .reshape(
                1,
                -1
            )
            .astype(
                np.float32
            )
        )

        # ====================================================
        # MODELO INT8
        # ====================================================

        if input_dtype == np.int8:

            if input_scale == 0:

                raise ValueError(
                    "El modelo INT8 tiene "
                    "input_scale = 0."
                )

            entrada = np.round(
                muestra
                /
                input_scale
                +
                input_zero_point
            )

            entrada = np.clip(
                entrada,
                -128,
                127
            )

            entrada = (
                entrada
                .astype(
                    np.int8
                )
            )

        # ====================================================
        # MODELO FP32
        # ====================================================

        else:

            entrada = (
                muestra
                .astype(
                    input_dtype
                )
            )

        # ====================================================
        # EJECUTAR INFERENCIA
        # ====================================================

        interpreter.set_tensor(
            input_index,
            entrada
        )

        interpreter.invoke()

        salida = (
            interpreter
            .get_tensor(
                output_index
            )[0]
        )

        # ====================================================
        # SALIDA INT8 -> FLOAT
        # ====================================================

        if output_dtype == np.int8:

            salida = (
                (
                    salida.astype(
                        np.float32
                    )
                    -
                    output_zero_point
                )
                *
                output_scale
            )

        prediccion = int(
            np.argmax(
                salida
            )
        )

        predicciones.append(
            prediccion
        )

    # ========================================================
    # ARRAY DE PREDICCIONES
    # ========================================================

    predicciones = np.asarray(
        predicciones,
        dtype=int
    )

    # ========================================================
    # ACCURACY
    # ========================================================

    accuracy = accuracy_score(
        y_test,
        predicciones
    )

    # ========================================================
    # MEMORIA ESTIMADA
    # ========================================================

    memoria_tensores = (
        estimar_memoria_tensores_tflite(
            interpreter
        )
    )

    return (
        float(
            accuracy
        ),
        memoria_tensores,
        predicciones
    )

# ============================================================
# ANALISIS TINYML COMPLETO
# ============================================================

def analizar_recursos_tinyml(
    modelo: keras.Model,
    X_train_std: np.ndarray,
    X_test_std: np.ndarray,
    y_test_cod: np.ndarray,
    output_dir: Path,
    accuracy_keras: float,
) -> pd.DataFrame:

    print("\n")
    print("=" * 80)
    print(
        "ANALISIS DE RECURSOS TINYML"
    )
    print("=" * 80)

    # ========================================================
    # PARAMETROS
    # ========================================================

    parametros = (
        modelo.count_params()
    )

    bytes_parametros_fp32 = (
        parametros
        *
        4
    )

    kib_parametros_fp32 = (
        bytes_parametros_fp32
        /
        1024
    )

    bytes_parametros_int8 = (
        parametros
    )

    kib_parametros_int8 = (
        bytes_parametros_int8
        /
        1024
    )

    print("\nMODELO")

    print(
        f"Parametros totales       : "
        f"{parametros}"
    )

    print(
        f"Parametros FP32          : "
        f"{bytes_parametros_fp32} bytes "
        f"({kib_parametros_fp32:.4f} KiB)"
    )

    print(
        f"Parametros INT8 teoricos : "
        f"{bytes_parametros_int8} bytes "
        f"({kib_parametros_int8:.4f} KiB)"
    )

    # ========================================================
    # KERAS
    # ========================================================

    keras_path = (
        output_dir
        /
        "modelo_arciniegas.keras"
    )

    if not keras_path.exists():

        modelo.save(
            keras_path
        )

    keras_bytes = (
        tamano_archivo_bytes(
            keras_path
        )
    )

    keras_kib = (
        bytes_a_kib(
            keras_bytes
        )
    )

    # ========================================================
    # CONVERSION TFLITE FP32
    # ========================================================

    print("\n")
    print(
        "CONVIRTIENDO A TFLITE FP32..."
    )

    converter_fp32 = (
        tf.lite.TFLiteConverter
        .from_keras_model(
            modelo
        )
    )

    modelo_tflite_fp32 = (
        converter_fp32
        .convert()
    )

    fp32_path = (
        output_dir
        /
        "modelo_arciniegas_fp32.tflite"
    )

    fp32_path.write_bytes(
        modelo_tflite_fp32
    )

    fp32_bytes = (
        tamano_archivo_bytes(
            fp32_path
        )
    )

    fp32_kib = (
        bytes_a_kib(
            fp32_bytes
        )
    )

    # ========================================================
    # REPRESENTATIVE DATASET
    # ========================================================

    def representative_dataset():

        numero_muestras = min(
            500,
            len(
                X_train_std
            ),
        )

        indices = np.linspace(
            0,
            len(
                X_train_std
            ) - 1,
            numero_muestras,
            dtype=int,
        )

        for indice in indices:

            muestra = (
                X_train_std[
                    indice:
                    indice + 1
                ]
                .astype(
                    np.float32
                )
            )

            yield [
                muestra
            ]

    # ========================================================
    # CONVERSION INT8
    # ========================================================

    print(
        "CONVIRTIENDO A TFLITE INT8..."
    )

    converter_int8 = (
        tf.lite.TFLiteConverter
        .from_keras_model(
            modelo
        )
    )

    converter_int8.optimizations = [
        tf.lite.Optimize.DEFAULT
    ]

    converter_int8.representative_dataset = (
        representative_dataset
    )

    converter_int8.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8
    ]

    converter_int8.inference_input_type = (
        tf.int8
    )

    converter_int8.inference_output_type = (
        tf.int8
    )

    modelo_tflite_int8 = (
        converter_int8
        .convert()
    )

    int8_path = (
        output_dir
        /
        "modelo_arciniegas_int8.tflite"
    )

    int8_path.write_bytes(
        modelo_tflite_int8
    )

    int8_bytes = (
        tamano_archivo_bytes(
            int8_path
        )
    )

    int8_kib = (
        bytes_a_kib(
            int8_bytes
        )
    )

    # ========================================================
    # EVALUACION TFLITE FP32
    # ========================================================

    (
        accuracy_fp32,
        memoria_fp32_bytes,
        pred_fp32,
    ) = evaluar_modelo_tflite(
        fp32_path,
        X_test_std,
        y_test_cod,
    )

    # ========================================================
    # EVALUACION TFLITE INT8
    # ========================================================

    (
        accuracy_int8,
        memoria_int8_bytes,
        pred_int8,
    ) = evaluar_modelo_tflite(
        int8_path,
        X_test_std,
        y_test_cod,
    )

    memoria_fp32_kib = (
        bytes_a_kib(
            memoria_fp32_bytes
        )
    )

    memoria_int8_kib = (
        bytes_a_kib(
            memoria_int8_bytes
        )
    )

    # ========================================================
    # CAMBIO EN ACCURACY
    # ========================================================

    delta_fp32_pp = (
        accuracy_fp32
        -
        accuracy_keras
    ) * 100

    delta_int8_pp = (
        accuracy_int8
        -
        accuracy_keras
    ) * 100

    # ========================================================
    # REDUCCION DE TAMAÑO
    # ========================================================

    reduccion_int8_pct = (
        1
        -
        (
            int8_bytes
            /
            fp32_bytes
        )
    ) * 100

    # ========================================================
    # CAMBIO DE PREDICCIONES
    # ========================================================

    cambios_fp32_int8 = int(
        np.sum(
            pred_fp32
            !=
            pred_int8
        )
    )

    # ========================================================
    # MOSTRAR RESULTADOS
    # ========================================================

    print("\n")
    print("=" * 80)
    print(
        "RESULTADOS TINYML"
    )
    print("=" * 80)

    print("\nTAMAÑO DE ARCHIVOS")

    print(
        f"Keras FP32       : "
        f"{keras_bytes} bytes "
        f"({keras_kib:.2f} KiB)"
    )

    print(
        f"TFLite FP32      : "
        f"{fp32_bytes} bytes "
        f"({fp32_kib:.2f} KiB)"
    )

    print(
        f"TFLite INT8      : "
        f"{int8_bytes} bytes "
        f"({int8_kib:.2f} KiB)"
    )

    print(
        f"Reduccion FP32 -> INT8: "
        f"{reduccion_int8_pct:.2f}%"
    )

    print("\nACCURACY")

    print(
        f"Keras FP32       : "
        f"{accuracy_keras * 100:.4f}%"
    )

    print(
        f"TFLite FP32      : "
        f"{accuracy_fp32 * 100:.4f}%"
    )

    print(
        f"TFLite INT8      : "
        f"{accuracy_int8 * 100:.4f}%"
    )

    print(
        f"Delta FP32       : "
        f"{delta_fp32_pp:+.4f} pp"
    )

    print(
        f"Delta INT8       : "
        f"{delta_int8_pp:+.4f} pp"
    )

    print(
        f"Predicciones distintas "
        f"FP32 vs INT8: "
        f"{cambios_fp32_int8}"
    )

    print("\nMEMORIA ESTIMADA DE TENSORES")

    print(
        f"TFLite FP32      : "
        f"{memoria_fp32_bytes} bytes "
        f"({memoria_fp32_kib:.2f} KiB)"
    )

    print(
        f"TFLite INT8      : "
        f"{memoria_int8_bytes} bytes "
        f"({memoria_int8_kib:.2f} KiB)"
    )

    print(
        "\nIMPORTANTE:"
    )

    print(
        "La memoria anterior es una "
        "estimacion obtenida a partir "
        "de tensores TFLite."
    )

    print(
        "NO corresponde todavia a la RAM "
        "real / Tensor Arena del MCU."
    )

    # ========================================================
    # CSV PRINCIPAL
    # ========================================================

    resultados = pd.DataFrame(
        [
            {
                "modelo":
                    "Keras_FP32",

                "parametros":
                    parametros,

                "bytes_parametros_teoricos":
                    bytes_parametros_fp32,

                "archivo_bytes":
                    keras_bytes,

                "archivo_KiB":
                    keras_kib,

                "accuracy":
                    accuracy_keras,

                "accuracy_percent":
                    accuracy_keras
                    *
                    100,

                "delta_accuracy_pp_vs_keras":
                    0.0,

                "memoria_tensores_estimada_bytes":
                    np.nan,

                "memoria_tensores_estimada_KiB":
                    np.nan,
            },

            {
                "modelo":
                    "TFLite_FP32",

                "parametros":
                    parametros,

                "bytes_parametros_teoricos":
                    bytes_parametros_fp32,

                "archivo_bytes":
                    fp32_bytes,

                "archivo_KiB":
                    fp32_kib,

                "accuracy":
                    accuracy_fp32,

                "accuracy_percent":
                    accuracy_fp32
                    *
                    100,

                "delta_accuracy_pp_vs_keras":
                    delta_fp32_pp,

                "memoria_tensores_estimada_bytes":
                    memoria_fp32_bytes,

                "memoria_tensores_estimada_KiB":
                    memoria_fp32_kib,
            },

            {
                "modelo":
                    "TFLite_INT8",

                "parametros":
                    parametros,

                "bytes_parametros_teoricos":
                    bytes_parametros_int8,

                "archivo_bytes":
                    int8_bytes,

                "archivo_KiB":
                    int8_kib,

                "accuracy":
                    accuracy_int8,

                "accuracy_percent":
                    accuracy_int8
                    *
                    100,

                "delta_accuracy_pp_vs_keras":
                    delta_int8_pp,

                "memoria_tensores_estimada_bytes":
                    memoria_int8_bytes,

                "memoria_tensores_estimada_KiB":
                    memoria_int8_kib,
            },
        ]
    )

    resultados.to_csv(
        output_dir
        /
        "metricas_tinyml_fp32_int8.csv",
        index=False,
    )

    # ========================================================
    # CSV COMPARACION FP32 VS INT8
    # ========================================================

    comparacion_predicciones = pd.DataFrame(
        {
            "y_real":
                y_test_cod,

            "pred_tflite_fp32":
                pred_fp32,

            "pred_tflite_int8":
                pred_int8,

            "fp32_correcto":
                (
                    pred_fp32
                    ==
                    y_test_cod
                ),

            "int8_correcto":
                (
                    pred_int8
                    ==
                    y_test_cod
                ),

            "cambio_prediccion":
                (
                    pred_fp32
                    !=
                    pred_int8
                ),
        }
    )

    comparacion_predicciones.to_csv(
        output_dir
        /
        "comparacion_predicciones_fp32_int8.csv",
        index=False,
    )

    # ========================================================
    # RESUMEN TXT
    # ========================================================

    texto = f"""
ANALISIS DE RECURSOS TINYML
==============================================

ARQUITECTURA
------------
21 -> 20 -> 10 -> 3

Parametros totales:
{parametros}

MEMORIA TEORICA DE PARAMETROS
-----------------------------
FP32:
{bytes_parametros_fp32} bytes
{kib_parametros_fp32:.4f} KiB

INT8 aproximado:
{bytes_parametros_int8} bytes
{kib_parametros_int8:.4f} KiB

TAMAÑO DE ARCHIVOS
------------------
Keras FP32:
{keras_bytes} bytes
{keras_kib:.4f} KiB

TFLite FP32:
{fp32_bytes} bytes
{fp32_kib:.4f} KiB

TFLite INT8:
{int8_bytes} bytes
{int8_kib:.4f} KiB

Reduccion TFLite FP32 -> INT8:
{reduccion_int8_pct:.2f} %

ACCURACY
--------
Keras FP32:
{accuracy_keras * 100:.4f} %

TFLite FP32:
{accuracy_fp32 * 100:.4f} %

TFLite INT8:
{accuracy_int8 * 100:.4f} %

Cambio TFLite FP32 vs Keras:
{delta_fp32_pp:+.4f} pp

Cambio TFLite INT8 vs Keras:
{delta_int8_pp:+.4f} pp

Predicciones distintas entre
TFLite FP32 y TFLite INT8:
{cambios_fp32_int8}

MEMORIA ESTIMADA DE TENSORES
----------------------------
TFLite FP32:
{memoria_fp32_bytes} bytes
{memoria_fp32_kib:.4f} KiB

TFLite INT8:
{memoria_int8_bytes} bytes
{memoria_int8_kib:.4f} KiB

IMPORTANTE
----------
La memoria calculada aqui es una estimacion
basada en los tensores reportados por el
interprete TFLite de Python.

NO representa todavia la RAM real requerida
por TensorFlow Lite Micro.

La RAM real debera medirse posteriormente
mediante el Tensor Arena durante el
despliegue en el microcontrolador.

El tamaño del archivo .keras tampoco debe
interpretarse como memoria Flash requerida
por el dispositivo.

La comparacion relevante para despliegue es:

TFLite FP32
vs
TFLite INT8.
"""

    (
        output_dir
        /
        "resumen_tinyml_fp32_int8.txt"
    ).write_text(
        texto,
        encoding="utf-8",
    )

    return resultados


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    global SEMILLA

    args = argumentos()

    SEMILLA = args.seed

    data_dir = (
        args.data_dir
        .resolve()
    )

    output_dir = (
        args.output_dir
        .resolve()
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    fijar_semillas(SEMILLA)

    # ========================================================
    # CARGAR DATOS
    # ========================================================

    (
        X_train,
        y_train,
        X_test,
        y_test,
    ) = cargar_datos(
        data_dir
    )

    (
        y_train_cod,
        y_test_cod,
        nombres,
        encoder,
    ) = codificar_etiquetas(
        y_train,
        y_test,
    )

    metadata_test = (
        cargar_metadata(
            data_dir,
            len(X_train),
            len(X_test),
        )
    )

    # ========================================================
    # VALIDAR COLUMNAS RESERVADAS
    # ========================================================

    columnas_resultado = {
        "P_DETENIDO",
        "P_MEDIA",
        "P_ALTA",
        "predicción",
        "confianza",
        "correcto",
    }

    coincidencias = (
        columnas_resultado
        .intersection(
            metadata_test.columns
        )
    )

    if coincidencias:

        raise ValueError(
            "metadata_features.csv "
            "ya contiene columnas "
            "reservadas: "
            +
            ", ".join(
                sorted(
                    coincidencias
                )
            )
        )

    # ========================================================
    # STANDARD SCALER
    # ========================================================

    scaler = (
        StandardScaler()
    )

    X_train_std = (
        scaler
        .fit_transform(
            X_train
        )
        .astype(
            np.float32
        )
    )

    X_test_std = (
        scaler
        .transform(
            X_test
        )
        .astype(
            np.float32
        )
    )

    joblib.dump(
        scaler,
        output_dir
        /
        "scaler.pkl",
    )

    # ========================================================
    # CONSTRUIR MODELO
    # ========================================================

    modelo = (
        construir_modelo()
    )

    print("\n")
    print("=" * 80)
    print(
        "ARQUITECTURA DEL MODELO"
    )
    print("=" * 80)

    modelo.summary()

    # ========================================================
    # ENTRENAMIENTO
    # ========================================================

    historial_obj = (
        modelo.fit(
            X_train_std,
            y_train_cod,

            epochs=50,

            batch_size=32,

            verbose=2,

            shuffle=True,
        )
    )

    # IMPORTANTE:
    # SIN CLASS_WEIGHT

    # ========================================================
    # GUARDAR MODELO KERAS
    # ========================================================

    modelo.save(
        output_dir
        /
        "modelo_arciniegas.keras"
    )

    # ========================================================
    # HISTORIAL
    # ========================================================

    historial = pd.DataFrame(
        historial_obj.history
    )

    historial.insert(
        0,
        "epoch",
        np.arange(
            1,
            len(historial) + 1,
        ),
    )

    historial.to_csv(
        output_dir
        /
        "historial_entrenamiento.csv",
        index=False,
    )

    guardar_curvas(
        historial,
        output_dir,
    )

    # ========================================================
    # PREDICCIONES KERAS FP32
    # ========================================================

    probabilidades = (
        modelo.predict(
            X_test_std,
            batch_size=32,
            verbose=0,
        )
    )

    y_pred = np.argmax(
        probabilidades,
        axis=1,
    )

    predicciones = (
        encoder
        .inverse_transform(
            y_pred
        )
    )

    reales = (
        encoder
        .inverse_transform(
            y_test_cod
        )
    )

    # ========================================================
    # GUARDAR PROBABILIDADES
    # ========================================================

    salida_prob = (
        metadata_test
        .copy()
    )

    salida_prob[
        "P_DETENIDO"
    ] = (
        probabilidades[:, 0]
    )

    salida_prob[
        "P_MEDIA"
    ] = (
        probabilidades[:, 1]
    )

    salida_prob[
        "P_ALTA"
    ] = (
        probabilidades[:, 2]
    )

    salida_prob[
        "predicción"
    ] = predicciones

    salida_prob[
        "confianza"
    ] = (
        probabilidades
        .max(
            axis=1
        )
    )

    salida_prob[
        "correcto"
    ] = (
        predicciones
        ==
        reales
    )

    salida_prob.to_csv(
        output_dir
        /
        "probabilidades_test.csv",
        index=False,
    )

    # ========================================================
    # MATRIZ DE CONFUSION
    # ========================================================

    matriz = confusion_matrix(
        y_test_cod,
        y_pred,
        labels=np.arange(
            3
        ),
    )

    guardar_matriz_confusion(
        matriz,
        nombres,
        output_dir,
    )

    # ========================================================
    # METRICAS DEL MODELO KERAS
    # ========================================================

    metricas = (
        calcular_metricas(
            y_test_cod,
            y_pred,
        )
    )

    pd.DataFrame(
        [
            metricas
        ]
    ).to_csv(
        output_dir
        /
        "metricas.csv",
        index=False,
    )

    # ========================================================
    # ANALISIS TINYML FP32 / INT8
    # ========================================================

    resultados_tinyml = (
        analizar_recursos_tinyml(
            modelo=modelo,

            X_train_std=(
                X_train_std
            ),

            X_test_std=(
                X_test_std
            ),

            y_test_cod=(
                y_test_cod
            ),

            output_dir=(
                output_dir
            ),

            accuracy_keras=(
                metricas[
                    "accuracy"
                ]
            ),
        )
    )

    # ========================================================
    # GUARDAR INFORMACION GENERAL
    # ========================================================

    guardar_info(
        output_dir,
        X_train,
        X_test,
        encoder.classes_,
        nombres,
        metricas,
    )

    # ========================================================
    # RESULTADOS FINALES
    # ========================================================

    print("\n")
    print("=" * 80)
    print(
        "EJECUCION COMPLETADA"
    )
    print("=" * 80)

    print(
        f"\nResultados guardados en:\n"
        f"{output_dir}"
    )

    print("\nMETRICAS KERAS")

    print(
        pd.Series(
            metricas
        )
        .to_string(
            float_format=(
                lambda x:
                f"{x:.6f}"
            )
        )
    )

    print("\n")
    print(
        "COMPARACION TINYML"
    )

    print(
        resultados_tinyml[
            [
                "modelo",
                "archivo_KiB",
                "accuracy_percent",
                "delta_accuracy_pp_vs_keras",
                "memoria_tensores_estimada_KiB",
            ]
        ]
        .round(4)
        .to_string(
            index=False
        )
    )


# ============================================================
# EJECUCION
# ============================================================

if __name__ == "__main__":

    main()
