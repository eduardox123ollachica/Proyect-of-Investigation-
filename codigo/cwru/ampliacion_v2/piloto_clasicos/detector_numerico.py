"""Detector K=1 V2: ajuste normal y aritmetica float32 explicita.

Este modulo no lee archivos o etiquetas y no certifica que los registros sean
normales o disjuntos. El llamador debe comprobar esos papeles antes de invocarlo.
El scaler y el centroide se ajustan solo con ``x_fit_f32``; ``x_cal_f32`` fija
exclusivamente el P95. No existe reajuste posterior a esa calibracion.
"""
from __future__ import annotations

from numbers import Integral
from pathlib import Path
import sys

import numpy as np

PROTOCOL_CODE = Path(__file__).resolve().parents[1] / 'protocolo'
if str(PROTOCOL_CODE) not in sys.path:
    sys.path.insert(0, str(PROTOCOL_CODE))
from reglas_v2 import weighted_p95


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _matrix(value, name):
    _require(isinstance(value, np.ndarray) and value.ndim == 2
             and value.shape[0] > 0 and value.shape[1] == 14,
             name + ' debe ser una matriz no vacia de catorce columnas')
    _require(value.dtype == np.float32, name + ' debe tener dtype float32')
    _require(np.isfinite(value).all(), name + ' contiene valores no finitos')
    return value


def _indices(values):
    _require(isinstance(values, (list, tuple, np.ndarray)), 'Indices deben ser una secuencia ordenada')
    if isinstance(values, np.ndarray):
        _require(values.ndim == 1, 'Indices deben ser un vector')
    result = list(values)
    _require(result and all(isinstance(v, Integral) and not isinstance(v, (bool, np.bool_))
                           for v in result), 'Indices deben ser enteros no vacios')
    result = [int(v) for v in result]
    _require(len(set(result)) == len(result) and all(0 <= v < 14 and v not in (4, 11) for v in result),
             'Indices duplicados, fuera de rango o frecuencias dominantes incluidas')
    return result


def _vector(params, name, size):
    _require(name in params and isinstance(params[name], np.ndarray), 'Parametro ausente o no vectorial: ' + name)
    value = params[name]
    _require(value.shape == (size,) and value.dtype == np.float32 and np.isfinite(value).all(),
             'Parametro no float32, no finito o con forma distinta: ' + name)
    return value


def _transform(selected, mean32, scale32):
    with np.errstate(over='ignore', invalid='ignore', divide='ignore', under='ignore'):
        centered = np.subtract(selected, mean32, dtype=np.float32)
        transformed = np.divide(centered, scale32, dtype=np.float32)
    _require(np.isfinite(centered).all() and np.isfinite(transformed).all(),
             'Normalizacion produjo un intermedio no finito')
    return transformed


def detector_score(x_f32, params):
    """Distancias cuadradas float32; suma secuencial en el orden de indices.

    Las operaciones resta, division, resta al centroide y multiplicacion se
    materializan por separado. No se usa dot, sum, einsum o fusion FMA.
    El comparador del detector es ``scores > params['threshold32']``; el empate
    pertenece a normalidad. Esta funcion calcula scores, no decide etiquetas.
    """
    x_f32 = _matrix(x_f32, 'x_f32')
    _require(isinstance(params, dict) and 'indices' in params, 'Parametros incompletos')
    indices = _indices(params['indices'])
    mean32 = _vector(params, 'mean32', len(indices))
    scale32 = _vector(params, 'scale32', len(indices))
    centroid32 = _vector(params, 'centroid32', len(indices))
    _require((scale32 > 0).all(), 'Escalas deben ser positivas')
    if 'threshold32' in params:
        threshold = params['threshold32']
        _require(isinstance(threshold, np.float32) and np.isfinite(threshold) and threshold >= 0,
                 'Umbral debe ser escalar float32 finito y no negativo')
    z = _transform(x_f32[:, indices], mean32, scale32)
    with np.errstate(over='ignore', invalid='ignore', under='ignore'):
        differences = np.subtract(z, centroid32, dtype=np.float32)
        terms = np.multiply(differences, differences, dtype=np.float32)
    _require(np.isfinite(differences).all() and np.isfinite(terms).all(),
             'Distancia produjo un intermedio no finito')
    total = np.zeros(len(x_f32), dtype=np.float32)
    for column in range(len(indices)):
        with np.errstate(over='ignore', invalid='ignore', under='ignore'):
            total = np.add(total, terms[:, column], dtype=np.float32)
        _require(np.isfinite(total).all(), 'Suma de distancia produjo un intermedio no finito')
    return total


def fit_normality(x_fit_f32, x_cal_f32, cal_record_ids, indices):
    """Ajusta scaler/centroide con fit y P95 con calibracion, sin refit.

    Se requieren matrices float32 completas de 14 columnas. Se conserva el
    orden de ``indices`` y se excluyen siempre 4/11. El ajuste de StandardScaler
    usa ventanas uniformes; la calibracion P95 da la misma masa a cada registro.
    El llamador conserva los IDs y prueba disjuncion; aqui no se reciben clases.
    StandardScaler se importa al ajustar para permitir uso del scorer sin sklearn.
    """
    x_fit_f32 = _matrix(x_fit_f32, 'x_fit_f32')
    x_cal_f32 = _matrix(x_cal_f32, 'x_cal_f32')
    indices = _indices(indices)
    _require(not isinstance(cal_record_ids, (str, bytes)), 'IDs de calibracion deben ser una secuencia por fila')
    try:
        records = list(cal_record_ids)
    except TypeError as error:
        raise ValueError('IDs de calibracion deben ser una secuencia') from error
    _require(len(records) == len(x_cal_f32), 'IDs de calibracion no coinciden con las filas')
    _require(all(isinstance(v, (str, Integral)) and not isinstance(v, (bool, np.bool_))
                 and str(v) and str(v).strip() == str(v) for v in records),
             'IDs de calibracion invalidos')

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(x_fit_f32[:, indices])
    mean64 = np.asarray(scaler.mean_, dtype=np.float64).copy()
    scale64 = np.asarray(scaler.scale_, dtype=np.float64).copy()
    var64 = np.asarray(scaler.var_, dtype=np.float64).copy()
    _require(all(a.shape == (len(indices),) and np.isfinite(a).all() for a in (mean64, scale64, var64))
             and (scale64 > 0).all() and (var64 >= 0).all(), 'Scaler produjo parametros invalidos')
    with np.errstate(over='ignore', invalid='ignore', under='ignore'):
        mean32, scale32 = mean64.astype(np.float32), scale64.astype(np.float32)
    _require(np.isfinite(mean32).all() and np.isfinite(scale32).all() and (scale32 > 0).all(),
             'Exportacion float32 del scaler produjo parametros invalidos')
    z_fit = _transform(x_fit_f32[:, indices], mean32, scale32)
    centroid32 = np.mean(z_fit, axis=0, dtype=np.float64).astype(np.float32)
    _require(np.isfinite(centroid32).all(), 'Centroide no finito')
    params = dict(indices=indices, mean64=mean64, scale64=scale64, var64=var64,
                  mean32=mean32, scale32=scale32, centroid32=centroid32,
                  k=1, fit_rows=len(x_fit_f32), calibration_rows=len(x_cal_f32),
                  fit_window_weighting='uniform_windows', calibration_record_weighting='equal_record_mass',
                  refit_after_calibration=False)
    calibration_scores = detector_score(x_cal_f32, params)
    params['threshold32'] = weighted_p95(calibration_scores, records)
    return params
