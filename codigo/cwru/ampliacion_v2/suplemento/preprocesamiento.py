"""Compresion de magnitudes para el suplemento de desarrollo CWRU V2.

P0 conserva las catorce entradas float32. P1 aplica log1p(x / a) solo a RMS,
energia espectral y amplitud dominante de DE/FE. Para cada columna, ``a`` es
la mediana float64 de los valores estrictamente positivos de las filas de
entrenamiento suministradas; una columna sin positivos usa a=1. La escala
se exporta float32 y se materializan division y log1p float32 separadamente.

Esta formula es una decision de ingenieria fijada para un ensayo acotado.
No ajusta lambda Box-Cox, no busca hiperparametros y no usa etiquetas. La
transformacion es monotona y sublineal, pero no acota las entradas. El llamador
debe verificar los roles train/validacion; este modulo no conoce IDs ni MAT.
StandardScaler, cuantizacion y cualquier entrenamiento pertenecen al worker.
"""
from __future__ import annotations

import numpy as np

INDICES = [0, 3, 5, 7, 10, 12]
MODES = ('P0', 'P1')


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _matrix(raw):
    _require(isinstance(raw, np.ndarray) and raw.dtype == np.float32
             and raw.ndim == 2 and raw.shape[0] > 0 and raw.shape[1] == 14,
             'Se requiere una matriz no vacia n por 14 con dtype float32')
    _require(np.isfinite(raw).all(), 'La matriz contiene valores no finitos')
    return raw


def _mode(value):
    _require(isinstance(value, str) and value in MODES, 'Modo debe ser P0 o P1')
    return value


def _parameters(params):
    _require(isinstance(params, dict), 'Parametros deben ser un diccionario')
    mode = _mode(params.get('mode'))
    indices = params.get('indices')
    _require(isinstance(indices, list) and len(indices) == len(INDICES)
             and all(type(index) is int for index in indices) and indices == INDICES,
             'Indices de magnitudes distintos o desordenados')
    arrays = []
    for name, dtype in (('scales64', np.float64), ('scales32', np.float32)):
        value = params.get(name)
        _require(isinstance(value, np.ndarray) and value.shape == (len(INDICES),)
                 and value.dtype == dtype and np.isfinite(value).all() and (value > 0).all(),
                 name + ' debe ser vector finito positivo con dtype y forma exactos')
        arrays.append(value)
    scales64, scales32 = arrays
    with np.errstate(over='ignore', invalid='ignore', under='ignore'):
        exported = scales64.astype(np.float32)
    _require(np.isfinite(exported).all() and (exported > 0).all()
             and np.array_equal(exported, scales32), 'Escalas float32 distintas de su exportacion float64')
    if mode == 'P0':
        _require(np.array_equal(scales64, np.ones(len(INDICES), dtype=np.float64))
                 and np.array_equal(scales32, np.ones(len(INDICES), dtype=np.float32)),
                 'P0 requiere escalas de identidad iguales a uno')
    return mode, scales32


def fit_params(raw_train14_f32, mode):
    """Retorna mode, indices, scales64 y scales32 sin alterar la entrada.

    Se aceptan exclusivamente las filas train ya autorizadas por el llamador.
    P0 devuelve escalas de uno. En P1, cero es valido y no participa en la
    mediana positiva; magnitudes negativas detienen el ajuste de parametros.
    No se aplica un epsilon, valor absoluto, clipping o busqueda de escalas.
    """
    raw = _matrix(raw_train14_f32)
    mode = _mode(mode)
    scales64 = np.ones(len(INDICES), dtype=np.float64)
    if mode == 'P1':
        magnitudes = raw[:, INDICES]
        _require((magnitudes >= 0).all(), 'P1 requiere magnitudes no negativas')
        for column in range(len(INDICES)):
            positive = magnitudes[:, column][magnitudes[:, column] > 0]
            if positive.size:
                # Cast antes de la mediana para no sumar extremos en float32.
                scales64[column] = np.median(positive.astype(np.float64))
    with np.errstate(over='ignore', invalid='ignore', under='ignore'):
        scales32 = scales64.astype(np.float32)
    params = dict(mode=mode, indices=INDICES.copy(), scales64=scales64, scales32=scales32)
    _parameters(params)
    return params


def apply(raw14f32, params):
    """Transforma una copia n por 14 float32; las otras ocho columnas son exactas.

    P0 es identidad incluso ante magnitudes negativas. P1 exige magnitudes no
    negativas y rechaza todo intermedio no finito. Cero y subnormales finitos
    siguen la aritmetica float32, sin reparaciones ni nuevos ajustes.
    """
    raw = _matrix(raw14f32)
    mode, scales32 = _parameters(params)
    result = raw.copy()
    if mode == 'P0':
        return result
    magnitudes = raw[:, INDICES]
    _require((magnitudes >= 0).all(), 'P1 requiere magnitudes no negativas')
    with np.errstate(over='ignore', invalid='ignore', divide='ignore', under='ignore'):
        divided = np.divide(magnitudes, scales32, dtype=np.float32)
    _require(np.isfinite(divided).all(), 'Division de P1 produjo un intermedio no finito')
    with np.errstate(over='ignore', invalid='ignore', divide='ignore', under='ignore'):
        compressed = np.log1p(divided, dtype=np.float32)
    _require(np.isfinite(compressed).all(), 'log1p de P1 produjo un intermedio no finito')
    result[:, INDICES] = compressed
    return result
