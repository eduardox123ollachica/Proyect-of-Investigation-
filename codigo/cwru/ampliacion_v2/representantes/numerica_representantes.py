"""Operaciones numericas puras para los representantes CWRU V2.

No lee archivos, ajusta modelos ni importa TensorFlow/sklearn. El llamador
verifica los roles y la procedencia de las filas antes de usar estas funciones.
Las transformaciones materializan cada operacion float32 por separado.
"""
from __future__ import annotations

from math import fsum, isfinite
from numbers import Integral, Real

import numpy as np


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value, name):
    _require(isinstance(value, Integral) and not isinstance(value, (bool, np.bool_)),
             name + ' debe ser entero')
    return int(value)


def _finite_scalar(value, name):
    _require(isinstance(value, Real) and not isinstance(value, (bool, np.bool_)),
             name + ' debe ser real')
    value = float(value)
    _require(isfinite(value), name + ' debe ser finito')
    return value


def _labels(values, name):
    array = np.asarray(values)
    _require(array.ndim == 1 and array.size > 0 and array.dtype.kind in 'iu',
             name + ' debe ser vector entero no vacio')
    _require(np.logical_and(array >= 0, array <= 2).all(),
             name + ' debe contener solo etiquetas 0, 1 y 2')
    return array.astype(np.int64, copy=False)


def _f32(values, name):
    source = np.asarray(values)
    _require(source.dtype.kind in 'fiu' and np.isfinite(source).all(),
             name + ' debe contener reales finitos')
    with np.errstate(over='ignore', invalid='ignore', under='ignore'):
        array = source.astype(np.float32, copy=False)
    _require(np.isfinite(array).all(), name + ' no es representable como float32 finito')
    return array


class EarlyStopping:
    """ES interno sobre perdida de validacion sin ponderar, incluido L2.

    ``observe`` devuelve ``(improved, should_stop)``. Las epocas empiezan en 1
    y son consecutivas. Una mejora requiere ``best_loss - val_loss > min_delta``;
    empates y mejoras insuficientes consumen paciencia. El llamador conserva y
    restaura los pesos de ``best_epoch``: esta clase no conoce los pesos.
    """

    def __init__(self, patience=10, min_delta=1e-12):
        self.patience = _integer(patience, 'patience')
        self.min_delta = _finite_scalar(min_delta, 'min_delta')
        _require(self.patience > 0 and self.min_delta >= 0,
                 'Paciencia debe ser positiva y min_delta no negativo')
        self.best_epoch = None
        self.best_loss = float('inf')
        self.wait = 0
        self.last_epoch = 0
        self.stopped = False

    def observe(self, epoch, val_loss):
        epoch = _integer(epoch, 'epoch')
        loss = _finite_scalar(val_loss, 'val_loss')
        _require(not self.stopped, 'No observar mas epocas despues de detenerse')
        _require(epoch == self.last_epoch + 1, 'Epocas deben empezar en 1 y ser consecutivas')
        improved = self.best_epoch is None or self.best_loss - loss > self.min_delta
        self.last_epoch = epoch
        if improved:
            self.best_loss, self.best_epoch, self.wait = loss, epoch, 0
        else:
            self.wait += 1
        self.stopped = self.wait >= self.patience
        return bool(improved), bool(self.stopped)


def balanced_weights(y):
    """Devuelve counts int64, pesos por clase float64 y pesos por fila float32.

    Se exigen las tres clases. No se inventan pesos para una clase ausente.
    """
    labels = _labels(y, 'y')
    counts = np.bincount(labels, minlength=3).astype(np.int64, copy=False)
    _require((counts > 0).all(), 'Entrenamiento debe contener las tres clases')
    weights = len(labels) / (3. * counts.astype(np.float64))
    samples = weights[labels].astype(np.float32)
    _require(np.isfinite(weights).all() and np.isfinite(samples).all(), 'Pesos no finitos')
    return counts, weights, samples


def normalize_f32(x, mean, scale):
    """Cast float32, resta y division separadas; no modifica las entradas.

    ``x`` es n por d; mean y scale son vectores d, sin broadcasting ambiguo.
    Las estadisticas se ajustan fuera de esta funcion exclusivamente en train.
    """
    matrix, means, scales = _f32(x, 'x'), _f32(mean, 'mean'), _f32(scale, 'scale')
    _require(matrix.ndim == 2 and all(size > 0 for size in matrix.shape),
             'x debe ser matriz no vacia')
    _require(means.shape == scales.shape == (matrix.shape[1],),
             'mean y scale deben tener exactamente una entrada por columna')
    _require((scales > 0).all(), 'Escalas deben ser positivas')
    with np.errstate(over='ignore', invalid='ignore', divide='ignore', under='ignore'):
        centered = np.subtract(matrix, means, dtype=np.float32)
        normalized = np.divide(centered, scales, dtype=np.float32)
    _require(np.isfinite(centered).all() and np.isfinite(normalized).all(),
             'Normalizacion produjo un intermedio no finito')
    return normalized


def quantize_int8(x, scale, zero):
    """Afin float32, nearest-even y clip; retorna (int8, valores recortados).

    El contador incluye valores fuera de [-128,127] DESPUES de redondear.
    Una entrada o un intermedio no finito detiene la operacion.
    """
    matrix = _f32(x, 'x')
    _require(matrix.ndim >= 1 and matrix.size > 0, 'x debe ser arreglo no vacio')
    scale = _finite_scalar(scale, 'scale')
    zero = _integer(zero, 'zero')
    _require(scale > 0 and -128 <= zero <= 127, 'Escala o zero_point INT8 invalido')
    scale32 = _f32(scale, 'scale')
    _require(scale32.item() > 0, 'Escala se anulo al convertir a float32')
    with np.errstate(over='ignore', invalid='ignore', divide='ignore', under='ignore'):
        divided = np.divide(matrix, scale32, dtype=np.float32)
        affine = np.add(divided, np.float32(zero), dtype=np.float32)
        rounded = np.rint(affine)
    _require(np.isfinite(divided).all() and np.isfinite(affine).all()
             and np.isfinite(rounded).all(), 'Cuantizacion produjo un intermedio no finito')
    clipped = int(np.count_nonzero((rounded < -128) | (rounded > 127)))
    return np.clip(rounded, -128, 127).astype(np.int8), clipped


def integer_argmax(probabilities):
    """Predice desde n por 3 valores finitos; empate al menor indice."""
    values = np.asarray(probabilities)
    _require(values.ndim == 2 and values.shape[0] > 0 and values.shape[1] == 3
             and values.dtype.kind in 'fiu' and np.isfinite(values).all(),
             'Se requiere matriz real finita n por tres')
    return np.argmax(values, axis=1).astype(np.int64, copy=False)


def metrics3(y, pred):
    """Metricas sobre etiquetas [0,1,2], incluso con clases ausentes.

    Las filas de confusion_matrix son clases reales y las columnas predichas.
    Precision/recall/F1 indefinidos valen cero. Retorna tipos JSON nativos.
    """
    truth, predicted = _labels(y, 'y'), _labels(pred, 'pred')
    _require(truth.shape == predicted.shape, 'Etiquetas y predicciones deben tener igual longitud')
    confusion = np.bincount(3 * truth + predicted, minlength=9).reshape(3, 3)
    by_class = []
    for label in range(3):
        correct = int(confusion[label, label])
        support, called = int(confusion[label].sum()), int(confusion[:, label].sum())
        precision = correct / called if called else 0.
        recall = correct / support if support else 0.
        f1 = 2. * correct / (support + called) if support + called else 0.
        by_class.append(dict(label=label, precision=precision, recall=recall, f1=f1, support=support))
    return dict(accuracy=float(np.trace(confusion) / len(truth)),
                macro_f1=fsum(row['f1'] for row in by_class) / 3.,
                confusion_matrix=confusion.tolist(), per_class=by_class)


def detector_metrics(y, rejected, record_ids):
    """Tasas por MAT y medias iguales por MAT dentro de N, IR y OR.

    Cada MAT debe corresponder a una sola clase y deben existir las tres clases.
    N especificidad es aceptacion; IR/OR recall son rechazo. Esta funcion no
    mezcla cargas: el llamador la usa por carga de evaluacion antes de agregar.
    El orden de filas y de IDs no altera el resultado.
    """
    truth = _labels(y, 'y')
    decisions = np.asarray(rejected)
    _require(decisions.shape == truth.shape and decisions.dtype == np.bool_,
             'rejected debe ser vector booleano de igual longitud que y')
    _require(not isinstance(record_ids, (str, bytes)), 'record_ids debe contener un ID por fila')
    try:
        raw_ids = list(record_ids)
    except TypeError as error:
        raise ValueError('record_ids debe ser secuencia') from error
    _require(len(raw_ids) == len(truth), 'Cantidad de IDs distinta de las filas')
    identifiers = []
    for identifier in raw_ids:
        _require(isinstance(identifier, (str, Integral)) and not isinstance(identifier, (bool, np.bool_))
                 and str(identifier) and str(identifier).strip() == str(identifier), 'ID MAT invalido')
        identifiers.append(str(identifier))
    groups = {}
    for offset, identifier in enumerate(identifiers):
        groups.setdefault(identifier, []).append(offset)
    per_record, rates = [], {label: [] for label in range(3)}
    for identifier in sorted(groups):
        offsets = groups[identifier]
        classes = set(int(truth[offset]) for offset in offsets)
        _require(len(classes) == 1, 'Un MAT no puede tener etiquetas distintas')
        label = next(iter(classes))
        rejects = int(np.count_nonzero(decisions[offsets]))
        success = (len(offsets) - rejects if label == 0 else rejects) / len(offsets)
        rates[label].append(success)
        per_record.append(dict(record_id=identifier, label=label, windows=len(offsets),
                               rejected_windows=rejects, rate=success))
    _require(all(rates.values()), 'Evaluacion del detector debe contener N, IR y OR')
    normal, ir, outer = [fsum(rates[label]) / len(rates[label]) for label in range(3)]
    sensitivity = (ir + outer) / 2.
    balanced = (normal + sensitivity) / 2.
    return dict(normal_spec=normal, ir_recall=ir, or_recall=outer,
                fault_sensitivity=sensitivity, balanced_accuracy=balanced, BA=balanced,
                per_record=per_record)
