"""Primitivas auditables de CWRU. Sin entrenamiento ni escritura de originales."""

from __future__ import annotations

import hashlib
import itertools
from collections import defaultdict
from pathlib import Path

import numpy as np


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def numeric_hash(array: np.ndarray) -> str:
    a = np.array(array, dtype="<f8", order="C", copy=True)
    a[a == 0] = 0.0
    return hashlib.sha256(a.tobytes()).hexdigest()


def choose_channel(data: dict, key: str, length: int) -> np.ndarray:
    if key not in data:
        raise ValueError(f"No existe la variable explícita {key}")
    a = np.asarray(data[key], dtype=np.float64).squeeze()
    if a.ndim != 1 or len(a) != length or not np.isfinite(a).all():
        raise ValueError(f"Variable incompatible: {key}, shape={a.shape}")
    return a


def independent_decimation(signal: np.ndarray, taps: np.ndarray) -> np.ndarray:
    """Convolución directa, distinta de scipy.signal.upfirdn del generador V1."""
    signal = np.asarray(signal, dtype=np.float64)
    taps = np.asarray(taps, dtype=np.float64)
    if len(taps) != 177:
        raise ValueError("Se esperaba el FIR canónico de 177 coeficientes")
    causal = np.convolve(signal, taps, mode="full")[::4]
    length = (len(signal) + 3) // 4
    aligned = causal[22:22 + length]
    return np.asarray(aligned[22:-22], dtype=np.float32)


def independent_features(window: np.ndarray, fs: float = 12000) -> np.ndarray:
    w = np.asarray(window, dtype=np.float64)
    if w.shape != (4096, 2) or not np.isfinite(w).all():
        raise ValueError("Ventana esperada: 4096 muestras, dos canales finitos")
    result = []
    n = len(w)
    for j in range(2):
        s = w[:, j]
        ac = s - s.mean()
        m2, m3, m4 = np.mean(ac**2), np.mean(ac**3), np.mean(ac**4)
        if m2 <= (np.finfo(float).eps * abs(s.mean()))**2:
            skew, kurt = 0.0, 0.0
        else:
            skew = np.sqrt(n*(n-1))/(n-2) * m3 / m2**1.5
            kurt = ((n*n-1)*m4/m2**2 - 3*(n-1)**2)/((n-2)*(n-3))
        magnitude = np.abs(np.fft.rfft(ac))
        power = magnitude**2
        energy = power.sum()
        k = int(np.argmax(magnitude[1:]) + 1)
        positive = power[power > 0]
        p = positive / energy if energy > 0 else np.empty(0)
        entropy = float(-np.sum(p*np.log2(p))/np.log2(len(power))) if len(p) > 1 else 0.0
        result.extend([np.sqrt(np.mean(s*s)), skew, kurt, energy,
                       k*fs/n, magnitude[k], entropy])
    return np.asarray(result, dtype=np.float64)


def centered_corr(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    a, b = a-a.mean(), b-b.mean()
    denom = float(np.sqrt(np.sum(a*a)*np.sum(b*b)))
    return float(np.sum(a*b)/denom) if denom else float("nan")


def affine_relation(x: np.ndarray, y: np.ndarray) -> dict:
    """y ≈ scale*x + intercept; comprobar residual, no sólo correlación."""
    x, y = np.asarray(x, dtype=float).ravel(), np.asarray(y, dtype=float).ravel()
    if len(x) != len(y):
        raise ValueError("La comparación afín exige igual longitud")
    xm, ym = x.mean(), y.mean()
    xc, yc = x-xm, y-ym
    vx, vy = np.sum(xc*xc), np.sum(yc*yc)
    if vx <= 0 or vy <= 0:
        return {"affine_match": False, "correlation": None, "scale": None,
                "intercept": None, "max_abs_residual": None}
    scale = float(np.sum(xc*yc)/vx)
    intercept = float(ym-scale*xm)
    residual = float(np.max(np.abs(y-(scale*x+intercept))))
    return {"affine_match": residual <= 1e-10*max(1.0,float(np.max(np.abs(y)))),
            "correlation": float(np.sum(xc*yc)/np.sqrt(vx*vy)),
            "scale": scale, "intercept": intercept, "max_abs_residual": residual}


def block_fingerprints(a: np.ndarray, length: int = 16) -> np.ndarray:
    """Huella uint64 ordenada; toda coincidencia se valida después con array_equal.

    Se examinan TODAS las posiciones. Puede recibir señal de uno o dos canales.
    No se considera prueba criptográfica y no se acepta una colisión como copia.
    """
    a = np.asarray(a, dtype="<f8")
    if a.ndim == 1:
        a = a[:, None]
    if len(a) < length:
        return np.empty(0, dtype=np.uint64)
    words = np.array(a, dtype="<f8", order="C", copy=True)
    words[words == 0] = 0.0
    words = words.view(np.uint64)
    size = len(words)-length+1
    h = np.zeros(size, dtype=np.uint64)
    mask = (1 << 64)-1
    for j in range(length):
        for channel in range(a.shape[1]):
            seed = np.uint64((0x9E3779B97F4A7C15*(j*a.shape[1]+channel+1)) & mask)
            shift = (j*11+channel*23) % 63 + 1
            v = words[j:j+size,channel] * seed
            h ^= (v << np.uint64(shift)) | (v >> np.uint64(64-shift))
    return h


def scan_exact_blocks(records: list[dict], signals: list[np.ndarray], length: int = 16):
    """Retorna segmentos máximos consolidados entre registros diferentes.

    No guarda las señales. Si aparecen grupos patológicos con más de 10000
    ocurrencias cruzadas potenciales, informa cobertura incompleta y no oculta
    la omisión ni afirma ausencia de coincidencias.
    """
    hashes = [block_fingerprints(a,length) for a in signals]
    starts_checked = sum(len(h) for h in hashes)
    merged = np.concatenate(hashes)
    owner = np.concatenate([np.full(len(h),i,dtype=np.int16) for i,h in enumerate(hashes)])
    positions = np.concatenate([np.arange(len(h),dtype=np.int32) for h in hashes])
    del hashes
    order = np.argsort(merged,kind="quicksort")
    ordered = merged[order]
    repeated = np.flatnonzero(ordered[1:] == ordered[:-1])
    runs = []
    for k in repeated:
        if runs and k == runs[-1][1]:
            runs[-1][1] = int(k)+1
        else:
            runs.append([int(k),int(k)+1])
    hits = defaultdict(list)
    hash_collisions, skipped, verified_pairs = 0, 0, 0
    for lo,hi in runs:
        entries = order[lo:hi+1]
        owners = owner[entries]
        unique,counts = np.unique(owners,return_counts=True)
        if len(unique) < 2:
            continue
        possible = (len(entries)**2-int(np.sum(counts.astype(np.int64)**2)))//2
        if possible > 10000:
            skipped += 1
            continue
        for aa,bb in itertools.combinations(entries,2):
            i,j,p,q = int(owner[aa]),int(owner[bb]),int(positions[aa]),int(positions[bb])
            if i == j:
                continue
            if i > j:
                i,j,p,q = j,i,q,p
            if np.array_equal(signals[i][p:p+length],signals[j][q:q+length]):
                hits[(i,j,q-p)].append(p)
                verified_pairs += 1
            else:
                hash_collisions += 1
    segments = []
    for (i,j,offset),pp in sorted(hits.items()):
        groups = []
        for p in sorted(set(pp)):
            if groups and p == groups[-1][-1]+1:
                groups[-1].append(p)
            else:
                groups.append([p])
        for group in groups:
            p,q,n = group[0],group[0]+offset,len(group)+length-1
            assert np.array_equal(signals[i][p:p+n],signals[j][q:q+n])
            segments.append({"file_a":records[i]["file"],"file_b":records[j]["file"],
                             "split_a":records[i]["split"],"split_b":records[j]["split"],
                             "start_a":p,"start_b":q,"length_samples":n,
                             "cross_split":records[i]["split"]!=records[j]["split"]})
    return segments,{"block_length":length,"starts_checked":starts_checked,
                     "verified_matching_position_pairs":verified_pairs,
                     "segments":len(segments),"fingerprint_collisions_rejected":hash_collisions,
                     "pathological_groups_skipped":skipped,"coverage_complete":skipped==0}


def assert_windows(expected: np.ndarray, stored: np.ndarray, start: int, n: int = 4096):
    candidate = expected[start:start+n]
    if candidate.shape != stored.shape or not np.allclose(candidate,stored,rtol=1e-6,atol=1e-7):
        raise ValueError("Ventana no coincide con señales originales seleccionadas")


def assert_metadata(base: dict, enriched: dict):
    """Tolera sólo redondeo decimal de campos float; claves/índices son exactos."""
    for key, value in base.items():
        if key in ("start_time_s", "end_time_s", "diameter_in"):
            if abs(float(value)-float(enriched[key])) > 1e-12:
                raise ValueError(f"Metadata numérica diferente: {key}")
        elif value != enriched[key]:
            raise ValueError(f"Metadata diferente: {key}")


def verify_manifest(directory: Path, filename: str = "MANIFEST_SHA256.csv") -> list[dict]:
    import csv
    with (directory/filename).open(encoding="utf-8-sig",newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        target = (directory/row["path"]).resolve()
        if not target.is_relative_to(directory.resolve()) or not target.is_file():
            raise ValueError(f"Archivo ausente o ruta fuera del paquete: {row['path']}")
        if file_hash(target) != row["sha256"]:
            raise ValueError(f"Hash de salida distinto: {row['path']}")
    return rows
