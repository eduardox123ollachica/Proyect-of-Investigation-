"""Aritmética portable y explícita del detector V1; no entrenamiento ni E/S."""
import numpy as np

INDICES = [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 12, 13]


def bits(values):
    return [f"{int(v):08x}" for v in np.asarray(values, dtype='<f4').view('<u4').ravel()]


def floats(words):
    return np.asarray([int(v, 16) for v in words], dtype='<u4').view('<f4')


def raw_hex(words):
    return floats(words).astype('<f4', copy=False).tobytes().hex()


def fnv(raw):
    result = 2166136261
    for byte in raw:
        result = ((result ^ byte) * 16777619) & 0xffffffff
    return result


def evaluate(mode, words, params):
    x = floats(words)
    if mode not in ('RUN', 'DIST', 'CHECK') or len(x) != {'RUN': 14, 'DIST': 12, 'CHECK': 1}[mode]:
        raise ValueError('FORMAT')
    if not np.isfinite(x).all():
        raise ValueError('NONFINITE_INPUT')
    threshold = floats([params['threshold_bits']])[0]
    if mode == 'CHECK':
        if x[0] < 0:
            raise ValueError('NEGATIVE_SCORE')
        return dict(z_bits=[], term_bits=[], score_bits=words[0].lower(),
                    rejected=bool(x[0] > threshold), cluster=0)
    with np.errstate(over='ignore', invalid='ignore', divide='ignore', under='ignore'):
        if mode == 'RUN':
            centered = np.subtract(x[INDICES], floats(params['mean_bits']), dtype=np.float32)
            z = np.divide(centered, floats(params['scale_bits']), dtype=np.float32)
            if not np.isfinite(centered).all() or not np.isfinite(z).all():
                raise ValueError('NONFINITE_INTERMEDIATE')
        else:
            z = x.copy()
        differences = np.subtract(z, floats(params['centroid_bits']), dtype=np.float32)
        terms = np.multiply(differences, differences, dtype=np.float32)
        if not np.isfinite(differences).all() or not np.isfinite(terms).all():
            raise ValueError('NONFINITE_INTERMEDIATE')
        total = np.float32(0)
        for term in terms:
            total = np.add(total, term, dtype=np.float32)
            if not np.isfinite(total):
                raise ValueError('NONFINITE_INTERMEDIATE')
    return dict(z_bits=bits(z), term_bits=bits(terms), score_bits=bits([total])[0],
                rejected=bool(total > threshold), cluster=0)
