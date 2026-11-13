"""Auditoría determinista: lee originales, derivados y evidencia; no entrena.

Dependencia: NumPy. No importa el preprocesador ni ejecuta su main.
Los únicos archivos escritos pertenecen al directorio de salida de esta auditoría.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import platform
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
LABELS = {"DETENIDO": 0, "VELOCIDAD MEDIA": 1, "VELOCIDAD ALTA": 2}
AXES = ("ax", "ay", "az")
DESCRIPTORS = ("rms", "skewness", "kurtosis", "spectral_energy",
               "dominant_frequency", "dominant_amplitude", "spectral_entropy")
FEATURES = [f"{a}_{d}" for a in AXES for d in DESCRIPTORS]
FILE_PATTERN = re.compile(r"^(.*?)\.json\.([^.]+)\.ingestion-(.*?)\.s(\d+)\.json$")
NAME_PATTERN = re.compile(r"^(.*?)\.([^.]+)\.s(\d+)$")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array_digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(array, dtype="<f8").tobytes()).hexdigest()


def csv_read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def csv_write(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    fields = fields or list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(rows)


def finite(value):
    if isinstance(value, dict):
        return {str(k): finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite(v) for v in value]
    if isinstance(value, np.ndarray):
        return finite(value.tolist())
    if isinstance(value, np.generic):
        return finite(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.ravel().astype(float), b.ravel().astype(float)
    a, b = a - a.mean(), b - b.mean()
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / norm) if norm > 0 else float("nan")


def independently_extract(values: np.ndarray) -> np.ndarray:
    # Shape: ventanas, eje, tiempo. Fórmulas independientes, equivalentes a SciPy
    # skew(bias=False), kurtosis(fisher=True, bias=False) para N=16.
    w = np.lib.stride_tricks.sliding_window_view(values, 16, axis=0)[::8]
    centered = w - w.mean(axis=-1, keepdims=True)
    m2 = np.mean(centered ** 2, axis=-1)
    m3 = np.mean(centered ** 3, axis=-1)
    m4 = np.mean(centered ** 4, axis=-1)
    valid = m2 > (np.finfo(float).eps * np.abs(w.mean(axis=-1))) ** 2
    skew = np.zeros(m2.shape)
    kurt = np.zeros(m2.shape)
    skew[valid] = math.sqrt(16 * 15) / 14 * m3[valid] / m2[valid] ** 1.5
    kurt[valid] = (255 * m4[valid] / m2[valid] ** 2 - 675) / 182
    # La forma escalar conserva la secuencia aritmética del preprocesador.
    # Una reducción vectorizada cambia el redondeo y puede desempatar otro bin.
    mag = np.empty((*m2.shape, 9))
    for wi in range(len(w)):
        for j in range(3):
            s = w[wi, j]
            mag[wi, j] = np.abs(np.fft.rfft(s - s.mean(), n=16))
    power = mag ** 2
    total = power.sum(axis=-1)
    idx = np.argmax(mag[..., 1:], axis=-1) + 1
    probabilities = np.divide(power, total[..., None], out=np.zeros_like(power),
                              where=total[..., None] > 0)
    logs = np.zeros_like(probabilities)
    np.log2(probabilities, out=logs, where=probabilities > 0)
    entropy = -(probabilities * logs).sum(axis=-1) / np.log2(9)
    features = np.stack([
        np.sqrt(np.mean(w ** 2, axis=-1)), skew, kurt, total,
        idx * 6.25, np.take_along_axis(mag, idx[..., None], axis=-1)[..., 0], entropy,
    ], axis=-1)
    return features.reshape(len(w), 21)


def metadata_audit(raw: Path, issues: list[dict]):
    root_entries = json.loads((raw / "info.labels").read_text(encoding="utf-8"))["files"]
    root_map = {e["path"]: e for e in root_entries}
    lookup = {}
    for split in ("training", "testing"):
        entries = json.loads((raw / split / "info.labels").read_text(encoding="utf-8"))["files"]
        for e in entries:
            key = f"{split}/{e['path']}"
            if key in lookup:
                issues.append({"type": "duplicate_metadata_path", "path": key})
            lookup[key] = e
            if root_map.get(key) != {**e, "path": key}:
                issues.append({"type": "root_split_metadata_disagreement", "path": key})
            if e.get("category") != split:
                issues.append({"type": "metadata_category_mismatch", "path": key})
    if len(root_map) != len(root_entries):
        issues.append({"type": "duplicate_root_metadata_path"})
    actual = {p.relative_to(raw).as_posix() for p in raw.rglob("*.json")}
    for key in sorted(actual.symmetric_difference(lookup)):
        issues.append({"type": "metadata_file_set_mismatch", "path": key})
    return lookup, {"root_entries": len(root_entries), "split_entries": len(lookup)}


def load_records(raw: Path, lookup: dict, issues: list[dict]):
    records, signals = [], {}
    for split in ("training", "testing"):
        for path in sorted((raw / split).glob("*.json")):
            rel = path.relative_to(raw).as_posix()
            data = json.loads(path.read_text(encoding="utf-8"))
            payload = data["payload"]
            values = np.asarray(payload["values"], dtype=float)
            match = FILE_PATTERN.fullmatch(path.name)
            if not match:
                raise ValueError(f"Nombre no reconocido: {path.name}")
            label, origin, ingestion, file_index = match.groups()
            entry = lookup[rel]
            name_match = NAME_PATTERN.fullmatch(entry["name"])
            if not name_match:
                raise ValueError(f"Nombre metadata no reconocido: {entry['name']}")
            name_label, name_origin, original_index = name_match.groups()
            if (label, origin) != (name_label, name_origin) or entry["label"]["label"] != label:
                issues.append({"type": "label_or_origin_conflict", "path": rel})
            good = (values.shape == (300, 3) and np.isfinite(values).all()
                    and payload.get("interval_ms") == 10
                    and [s.get("name") for s in payload["sensors"]] == list(AXES))
            if not good:
                raise ValueError(f"Estructura inesperada o valores no finitos: {rel}")
            iat = data.get("protected", {}).get("iat")
            row = {
                "path": rel, "split": split, "file": path.name, "class": label,
                "label": LABELS[label], "origin_id": origin, "ingestion": ingestion,
                "metadata_name": entry["name"], "file_fragment_index": int(file_index),
                "metadata_fragment_index": int(original_index),
                "renumbered_fragment": int(file_index) != int(original_index),
                "iat": iat, "iat_utc": datetime.fromtimestamp(iat, timezone.utc).isoformat() if iat else "",
                "signature_algorithm": data.get("protected", {}).get("alg"),
                "signature_sha256": hashlib.sha256(str(data.get("signature")).encode()).hexdigest(),
                "device_name": payload.get("device_name"), "device_type": payload.get("device_type"),
                "sensor_units": "|".join(s.get("units", "") for s in payload["sensors"]),
                "n_samples": len(values), "interval_ms": payload["interval_ms"],
                "duration_nominal_s": len(values) * payload["interval_ms"] / 1000,
                "sample_span_s": (len(values) - 1) * payload["interval_ms"] / 1000,
                "raw_sha256": digest(path), "signal_sha256": array_digest(values),
                "unique_triaxial_samples": len(np.unique(values, axis=0)),
                "consecutive_equal_triaxial_pairs": int(np.all(np.diff(values, axis=0) == 0, axis=1).sum()),
                "windows_n16_h8": 36, "windows_n16_h16": 18,
            }
            for j, axis in enumerate(AXES):
                s = values[:, j]
                row.update({f"{axis}_mean": float(s.mean()), f"{axis}_std": float(s.std()),
                            f"{axis}_min": float(s.min()), f"{axis}_max": float(s.max()),
                            f"{axis}_unique_values": len(np.unique(s)),
                            f"{axis}_constant": bool(np.ptp(s) == 0),
                            f"{axis}_on_0p01_grid_fraction": float(np.mean(np.isclose(s * 100, np.rint(s * 100), atol=1e-7, rtol=0)))})
            records.append(row)
            signals[rel] = values
    return records, signals


def origins_audit(records: list[dict]):
    groups = defaultdict(list)
    for r in records:
        groups[r["origin_id"]].append(r)
    rows = []
    for origin, rr in sorted(groups.items(), key=lambda item: (item[1][0]["label"], item[0])):
        source_indexes = [r["metadata_fragment_index"] for r in rr]
        unique = set(source_indexes)
        rows.append({
            "origin_id": origin, "class": rr[0]["class"], "files": len(rr),
            "training_files": sum(r["split"] == "training" for r in rr),
            "testing_files": sum(r["split"] == "testing" for r in rr),
            "shared_between_splits": len({r["split"] for r in rr}) > 1,
            "metadata_index_min": min(unique), "metadata_index_max": max(unique),
            "missing_metadata_indexes": "|".join(map(str, sorted(set(range(1, max(unique) + 1)) - unique))),
            "duplicate_metadata_indexes": "|".join(str(k) for k, n in Counter(source_indexes).items() if n > 1),
            "renumbered_files": sum(r["renumbered_fragment"] for r in rr),
            "distinct_iat": len({r["iat"] for r in rr}), "iat_utc": rr[0]["iat_utc"],
            "distinct_device_names": len({r["device_name"] for r in rr}),
            "distinct_signatures": len({r["signature_sha256"] for r in rr}),
            "duration_nominal_s": sum(r["duration_nominal_s"] for r in rr),
            "metadata_order_split_sequence": " ".join(
                f"{r['metadata_fragment_index']}:{'TR' if r['split']=='training' else 'TE'}"
                for r in sorted(rr, key=lambda r: r["metadata_fragment_index"])),
        })
    return rows, groups


def duplicate_audit(records: list[dict], signals: dict):
    full = []
    for key in ("raw_sha256", "signal_sha256"):
        buckets = defaultdict(list)
        for r in records:
            buckets[r[key]].append(r)
        for hh, rr in buckets.items():
            if len(rr) > 1:
                full.append({"criterion": key, "hash": hh, "files": "|".join(r["path"] for r in rr),
                             "cross_split": len({r["split"] for r in rr}) > 1})
    block_rows, block_summary = [], []
    for length in (2, 4, 8, 16, 32):
        buckets = defaultdict(list)
        for r in records:
            a = signals[r["path"]]
            for pos in range(len(a) - length + 1):
                buckets[array_digest(a[pos:pos + length])].append((r, pos))
        hits, cross = 0, 0
        for hh, entries in buckets.items():
            if len({r["path"] for r, _ in entries}) < 2:
                continue
            for (a, p), (b, q) in itertools.combinations(entries, 2):
                if a["path"] == b["path"]:
                    continue
                if not np.array_equal(signals[a["path"]][p:p+length], signals[b["path"]][q:q+length]):
                    raise AssertionError("Colisión de hash")
                cross_split = a["split"] != b["split"]
                block_rows.append({"length_samples": length, "file_a": a["path"], "start_a": p,
                                   "file_b": b["path"], "start_b": q, "cross_split": cross_split,
                                   "same_origin": a["origin_id"] == b["origin_id"], "sha256": hh})
                hits += 1
                cross += int(cross_split)
        block_summary.append({"length_samples": length, "starts_checked": sum(len(a)-length+1 for a in signals.values()),
                              "cross_file_matching_pairs": hits, "cross_split_matching_pairs": cross})
    return full, block_rows, block_summary


def repeated_segments(blocks: list[dict], records: list[dict], signals: dict):
    lookup = {r['path']: r for r in records}
    grouped = defaultdict(list)
    for b in blocks:
        if b['length_samples'] == 2:
            grouped[(b['file_a'], b['file_b'], b['start_b'] - b['start_a'])].append(b['start_a'])
    rows = []
    for (pa, pb, offset), starts in sorted(grouped.items()):
        ordered = sorted(set(starts))
        runs = []
        for start in ordered:
            if runs and start == runs[-1][-1] + 1:
                runs[-1].append(start)
            else:
                runs.append([start])
        for run in runs:
            sa, sb, length = run[0], run[0] + offset, len(run) + 1
            assert np.array_equal(signals[pa][sa:sa+length], signals[pb][sb:sb+length])
            is_boundary = (sa == 0 and sb+length == len(signals[pb])) or (sb == 0 and sa+length == len(signals[pa]))
            aligned = sum(p % 8 == 0 and (p+offset) % 8 == 0 and p+16 <= sa+length
                          and p <= 280 and p+offset <= 280 for p in range(sa, sa+length))
            rows.append({'file_a': pa, 'file_b': pb, 'start_a': sa, 'start_b': sb,
                         'length_samples': length, 'cross_split': lookup[pa]['split'] != lookup[pb]['split'],
                         'same_origin': lookup[pa]['origin_id'] == lookup[pb]['origin_id'],
                         'origin_a': lookup[pa]['origin_id'], 'origin_b': lookup[pb]['origin_id'],
                         'class_a': lookup[pa]['class'], 'class_b': lookup[pb]['class'],
                         'suffix_prefix_match': is_boundary, 'aligned_n16_h8_window_pairs': aligned})
    return rows


def spectral_ties(records: list[dict], signals: dict):
    rows = []
    for record in records:
        a = signals[record['path']]
        w = np.lib.stride_tricks.sliding_window_view(a, 16, axis=0)[::8]
        batched_mag = np.abs(np.fft.rfft(w - w.mean(axis=-1, keepdims=True), axis=-1))
        for wi in range(len(w)):
            for j, axis in enumerate(AXES):
                s = w[wi,j]
                m = np.abs(np.fft.rfft(s-s.mean()))
                mx = m[1:].max()
                tied = np.flatnonzero(np.abs(m[1:]-mx) <= 1e-12 * max(mx, 1e-12)) + 1
                if len(tied) > 1:
                    scalar = int(np.argmax(m[1:])+1)
                    batched = int(np.argmax(batched_mag[wi,j,1:])+1)
                    rows.append({'path': record['path'], 'window': wi, 'start_sample': wi*8,
                                 'axis': axis, 'nearly_tied_bins_rtol1e12': '|'.join(map(str,tied)),
                                 'scalar_bin': scalar, 'vectorized_bin': batched,
                                 'scalar_vectorized_disagree': scalar != batched,
                                 'max_magnitude': float(mx),
                                 'top_two_magnitude_gap': float(mx-np.sort(m[1:])[-2])})
    return rows


def boundaries_audit(groups: dict, signals: dict):
    rows = []
    for origin, rr in groups.items():
        ordered = sorted(rr, key=lambda r: r["metadata_fragment_index"])
        for a, b in zip(ordered, ordered[1:]):
            x, y = signals[a["path"]], signals[b["path"]]
            internal = np.concatenate([np.linalg.norm(np.diff(x, axis=0), axis=1),
                                       np.linalg.norm(np.diff(y, axis=0), axis=1)])
            jump = float(np.linalg.norm(y[0] - x[-1]))
            overlap = 0
            for length in range(min(len(x), len(y)), 0, -1):
                if np.array_equal(x[-length:], y[:length]):
                    overlap = length
                    break
            rows.append({"origin_id": origin, "class": a["class"],
                         "file_a": a["path"], "file_b": b["path"],
                         "metadata_index_a": a["metadata_fragment_index"],
                         "metadata_index_b": b["metadata_fragment_index"],
                         "consecutive_metadata_indexes": b["metadata_fragment_index"] == a["metadata_fragment_index"]+1,
                         "cross_split_boundary": a["split"] != b["split"],
                         "euclidean_boundary_jump": jump,
                         "boundary_jump_internal_percentile": float(np.mean(internal <= jump) * 100),
                         "exact_suffix_prefix_overlap_samples": overlap,
                         "chronology_verified": False})
    return rows


def correlations_audit(records: list[dict], signals: dict, features_by_file: dict):
    raw_buckets, feat_buckets = defaultdict(list), defaultdict(list)
    raw_file_rows, feat_file_rows = [], []
    for r in records:
        a = signals[r["path"]]
        for j, axis in enumerate(AXES):
            for lag in (1, 2, 4, 8, 16, 32, 64):
                corr = pearson(a[:-lag, j], a[lag:, j])
                raw_buckets[(r["split"], r["class"], axis, lag)].append(corr)
                raw_file_rows.append({"path": r["path"], "origin_id": r["origin_id"],
                                      "axis": axis, "lag_samples": lag, "pearson_r": corr})
        f = features_by_file[r["path"]]
        for j, name in enumerate(FEATURES):
            for lag in (1, 2, 4):
                corr = pearson(f[:-lag, j], f[lag:, j])
                feat_buckets[(r["split"], r["class"], name, lag)].append(corr)
                feat_file_rows.append({"path": r["path"], "origin_id": r["origin_id"],
                                       "feature": name, "lag_windows": lag, "pearson_r": corr})
    def aggregate(buckets, keys):
        rows = []
        for key, values in sorted(buckets.items()):
            valid = np.array([v for v in values if np.isfinite(v)])
            rows.append({**dict(zip(keys, key)), "files": len(values), "defined_correlations": len(valid),
                         "median_r": float(np.median(valid)) if len(valid) else None,
                         "mean_r": float(np.mean(valid)) if len(valid) else None,
                         "p10_r": float(np.percentile(valid, 10)) if len(valid) else None,
                         "p90_r": float(np.percentile(valid, 90)) if len(valid) else None})
        return rows
    return (aggregate(raw_buckets, ["split", "class", "axis", "lag_samples"]),
            aggregate(feat_buckets, ["split", "class", "feature", "lag_windows"]),
            raw_file_rows, feat_file_rows)


def cross_split_correlations(records: list[dict], signals: dict):
    # Sólo comparaciones dentro de la misma clase. Se elimina la media por eje
    # para que la gravedad/DC no domine una correlación aparentemente alta.
    rows = []
    for test in (r for r in records if r["split"] == "testing"):
        y = signals[test["path"]]
        yc = y - y.mean(axis=0)
        for train in (r for r in records if r["split"] == "training" and r["class"] == test["class"]):
            x = signals[train["path"]]
            xc = x - x.mean(axis=0)
            norm = float(np.linalg.norm(xc) * np.linalg.norm(yc))
            corr = float(np.sum(xc * yc) / norm) if norm else float("nan")
            lags = np.arange(-150, 151)
            start_x, start_y = np.maximum(lags, 0), np.maximum(-lags, 0)
            n = len(x)-np.abs(lags)
            px = np.vstack([np.zeros(3), np.cumsum(x, axis=0)])
            py = np.vstack([np.zeros(3), np.cumsum(y, axis=0)])
            px2 = np.vstack([np.zeros(3), np.cumsum(x*x, axis=0)])
            py2 = np.vstack([np.zeros(3), np.cumsum(y*y, axis=0)])
            sx, sy = px[start_x+n]-px[start_x], py[start_y+n]-py[start_y]
            vx = ((px2[start_x+n]-px2[start_x])-sx*sx/n[:,None]).sum(axis=1)
            vy = ((py2[start_y+n]-py2[start_y])-sy*sy/n[:,None]).sum(axis=1)
            dot = sum(np.correlate(x[:,j],y[:,j],mode='full')[lags+len(x)-1] for j in range(3))
            covariance = dot-(sx*sy).sum(axis=1)/n
            denom = np.sqrt(np.maximum(vx,0)*np.maximum(vy,0))
            corrs = np.divide(covariance,denom,out=np.full(len(n),np.nan),where=denom>0)
            best = int(np.nanargmax(corrs))
            rows.append({"class": test["class"], "test_file": test["path"], "train_file": train["path"],
                         "test_origin": test["origin_id"], "train_origin": train["origin_id"],
                         "same_origin": train["origin_id"] == test["origin_id"],
                         "zero_lag_centered_triaxial_r": corr,
                         "max_centered_r_lags_pm150": float(corrs[best]),
                         "best_lag_samples_x_minus_y": int(lags[best]),
                         "best_lag_overlap_samples": int(n[best]),
                         "raw_rmse": float(np.sqrt(np.mean((x-y)**2))),
                         "centered_rmse": float(np.sqrt(np.mean((xc-yc)**2)))})
    return rows


def distributions(records: list[dict], signals: dict, features_by_file: dict):
    raw_rows, feature_rows = [], []
    for split in ('training','testing'):
        for label in LABELS:
            rr = [r for r in records if r['split']==split and r['class']==label]
            a = np.vstack([signals[r['path']] for r in rr])
            f = np.vstack([features_by_file[r['path']] for r in rr])
            for j,axis in enumerate(AXES):
                s = a[:,j]
                raw_rows.append({'split':split,'class':label,'axis':axis,'samples':len(s),
                                 'mean':float(s.mean()),'std':float(s.std()),'min':float(s.min()),
                                 'max':float(s.max()),'p01':float(np.percentile(s,1)),
                                 'p99':float(np.percentile(s,99)),'distinct_values':len(np.unique(s))})
            for j,name in enumerate(FEATURES):
                s = f[:,j]
                feature_rows.append({'split':split,'class':label,'feature':name,'windows':len(s),
                                     'mean':float(s.mean()),'std':float(s.std()),'min':float(s.min()),
                                     'max':float(s.max()),'p01':float(np.percentile(s,1)),
                                     'p99':float(np.percentile(s,99))})
    return raw_rows, feature_rows


def processed_audit(root: Path, records: list[dict], features_by_file: dict, issues: list[dict]):
    processed = root / "datos/arciniegas/procesado"
    metadata = csv_read(processed / "metadata_features.csv")
    rows_by_split = {s: [r for r in metadata if r["split"] == s] for s in ("training", "testing")}
    source_lookup = {r["path"]: r for r in records}
    seen = set()
    error_fields = Counter()
    feature_rows, summary = [], {}
    metadata_feature_max = np.zeros(21)
    for split, short in (("training", "train"), ("testing", "test")):
        rr = [r for r in records if r["split"] == split]
        expected_x = np.vstack([features_by_file[r["path"]] for r in rr])
        expected_y = np.concatenate([np.full(36, r["label"], dtype=np.int64) for r in rr])
        expected_keys = [(r["path"], i) for r in rr for i in range(36)]
        x = np.load(processed / f"X_{short}.npy", allow_pickle=False)
        y = np.load(processed / f"y_{short}.npy", allow_pickle=False)
        assert x.shape == expected_x.shape and y.shape == expected_y.shape
        same_labels = np.array_equal(y, expected_y)
        allclose = bool(np.allclose(x, expected_x, rtol=1e-6, atol=1e-5))
        if not same_labels or not allclose:
            issues.append({"type": "processed_array_mismatch", "split": split})
        meta_rr = rows_by_split[split]
        assert len(meta_rr) == len(expected_keys)
        actual_keys = []
        for i, row in enumerate(meta_rr):
            path = f"{split}/{row['file']}"
            wi = int(row["window"])
            actual_keys.append((path, wi))
            if (path, wi) in seen:
                error_fields["duplicate_window_metadata"] += 1
            seen.add((path, wi))
            record = source_lookup.get(path)
            if record is None or wi not in range(36):
                raise AssertionError(f"Ventana desconocida: {path}/{wi}")
            expected_fields = {"label": record["label"], "start_sample": wi*8,
                               "end_sample_exclusive": wi*8+16, "fs_hz": 100,
                               "fft_size": 16, "overlap": 0.5, "hop_size": 8,
                               "start_time_s": wi*0.08, "end_time_s": (wi*8+16)/100}
            for field, value in expected_fields.items():
                if not math.isclose(float(row[field]), value, abs_tol=1e-10):
                    error_fields[field] += 1
            if row["class"] != record["class"]:
                error_fields["class"] += 1
            mx = np.array([float(row[name]) for name in FEATURES])
            metadata_feature_max = np.maximum(metadata_feature_max, np.abs(mx-features_by_file[path][wi]))
            if not np.array_equal(mx.astype(x.dtype), x[i]):
                error_fields["csv_features_vs_npy"] += 1
        order_matches = actual_keys == expected_keys
        if not order_matches:
            issues.append({"type": "processed_order_mismatch", "split": split})
        for j, feature in enumerate(FEATURES):
            delta = np.abs(x[:, j].astype(float) - expected_x[:, j])
            feature_rows.append({"split": split, "feature": feature,
                                 "values": len(delta), "max_abs_error_npy": float(delta.max()),
                                 "mean_abs_error_npy": float(delta.mean()),
                                 "independent_formula_matches_rtol1e6_atol1e5": bool(np.allclose(x[:,j], expected_x[:,j], rtol=1e-6, atol=1e-5))})
        summary[split] = {"x_shape": list(x.shape), "x_dtype": str(x.dtype), "finite": bool(np.isfinite(x).all()),
                          "y_shape": list(y.shape), "label_counts": dict(Counter(map(int,y))),
                          "labels_match_raw": same_labels, "order_matches_raw": order_matches,
                          "features_independently_recomputed_match": allclose,
                          "max_abs_error_npy": float(np.max(np.abs(x.astype(float)-expected_x)))}
    for row in feature_rows:
        row["max_abs_error_metadata_csv"] = float(metadata_feature_max[FEATURES.index(row["feature"])])
    saved_names = [(line.split(": ",1)[1]) for line in (processed/"feature_names.txt").read_text().splitlines()]
    summary["feature_order_matches"] = saved_names == FEATURES
    summary["metadata_errors"] = dict(error_fields)
    summary["metadata_windows"] = len(metadata)
    return summary, feature_rows


def predictions_audit(root: Path, records: list[dict]):
    lookup = {r["file"]: r for r in records if r["split"] == "testing"}
    rows = []
    for exp in ("e1", "e4"):
        path = root / f"resultados/arciniegas/{exp}/validacion_rawall/rawall_todas.csv"
        if not path.exists():
            continue
        buckets = defaultdict(list)
        for pred in csv_read(path):
            record = lookup[pred["file"]]
            buckets[record["origin_id"]].append(pred)
        for origin, preds in sorted(buckets.items()):
            errors = sum(int(p["label_real"]) != int(p["clase_stm32"]) for p in preds)
            rows.append({"experiment": exp.upper(), "origin_id": origin,
                         "class": preds[0]["class"], "files": len({p['file'] for p in preds}),
                         "windows": len(preds), "stm32_errors": errors,
                         "accuracy_percent": (1-errors/len(preds))*100,
                         "rejections": sum(int(p["anomaly_stm32"]) for p in preds),
                         "pc_stm32_class_disagreements": sum(int(p["clase_pc"]) != int(p["clase_stm32"]) for p in preds)})
    return rows


def copy_audit(root: Path, raw: Path, raw_paths: list[Path]):
    copies = [root / "Eduardo_Andres_Ollachica_Arias_Replication_Evidence/03_Datasets/Arciniegas_Raw",
              root / "PAQUETE_VALIDACION_REIMPLEMENTACION_STM32/datos/arciniegas/raw"]
    rows = []
    rels = {p.relative_to(raw).as_posix() for p in raw_paths}
    for copy in copies:
        if not copy.exists():
            continue
        for rel in sorted(rels):
            other = copy / rel
            rows.append({"copy_root": copy.relative_to(root).as_posix(), "path": rel,
                         "exists": other.is_file(),
                         "matches_canonical_sha256": other.is_file() and digest(raw/rel) == digest(other),
                         "status": 'missing' if not other.is_file() else ('identical' if digest(raw/rel)==digest(other) else 'different_bytes')})
        extra = {p.relative_to(copy).as_posix() for p in copy.rglob("*") if p.is_file()} - rels
        for rel in sorted(extra):
            rows.append({"copy_root": copy.relative_to(root).as_posix(), "path": rel,
                         "exists": True, "matches_canonical_sha256": "extra_file", "status":'extra_file'})
    return rows


def write_report(output: Path, summary: dict, origins: list[dict], issues: list[dict]):
    classes = summary["class_summary"]
    table = "\n".join(f"| {r['class']} | {r['files']} | {r['origins']} | {r['train_files']} | {r['test_files']} |" for r in classes)
    origin_table = "\n".join(f"| {r['class']} | `{r['origin_id']}` | {r['training_files']} | {r['testing_files']} | {r['renumbered_files']} |" for r in origins)
    blocks = "\n".join(f"| {r['length_samples']} | {r['starts_checked']} | {r['cross_file_matching_pairs']} | {r['cross_split_matching_pairs']} |" for r in summary["exact_blocks"])
    repeated_long = summary['repeated_segments_ge8']
    copy_notes = '\n'.join(f"- `{c['copy']}`: {c['identical']} idénticos, {c['missing']} ausentes, {c['different_bytes']} distintos y {c['extra_files']} adicionales." for c in summary['copy_summary'])
    text = f"""# Auditoría exhaustiva del dataset Arciniegas

Fecha UTC: {summary['created_utc']}. Auditoría determinista; no se entrenaron modelos,
no se modificaron datos, particiones, firmware ni manuscrito.

## Dictamen ejecutivo

La evaluación de independencia requiere cambios estructurales. Los archivos son
legibles y sus derivados pueden comprobarse, pero separar únicamente por JSON no
separa los identificadores de origen. Existen {len(origins)} familias nominales,
{summary['shared_origins']} compartidas entre entrenamiento y prueba. Los
{summary['test_files_shared_origin']} archivos de prueba pertenecen a familias
también presentes en entrenamiento.

Esto confirma procedencia nominal compartida, no demuestra por sí solo que cada
familia sea una sesión física independiente ni cuantifica la inflación de métricas.
Los sufijos sN, el mismo iat y la misma firma declarada por familia son compatibles
con fragmentación; no son firmas autenticadas en esta auditoría. No hay
marcas de tiempo por muestra ni desplazamientos verificados del registro fuente.
El iat es una fecha de creación declarada por el formato; no sustituye esos datos.

## Inventario y grupos

| Clase | JSON | Familias de origen | Training | Testing |
|---|---:|---:|---:|---:|
{table}

Cada JSON contiene 300 ternas finitas, interval_ms=10, canales ax/ay/az y 3 s de
duración nominal (2.99 s entre primera y última muestra). La frecuencia configurada
es 100 Hz; no se verificó físicamente el reloj. Se examinaron los 239 archivos del
corpus (235 JSON, tres info.labels y README) y las matrices/metadata procesadas.

| Clase | ID de origen | Training | Testing | Índices renumerados |
|---|---|---:|---:|---:|
{origin_table}

El identificador no equivale automáticamente a motor, montaje, día ni sesión.
Las cantidades de familias son exactas; su interpretación física requiere evidencia.
Con el agrupamiento conservador por origen, DETENIDO sólo tiene un grupo: no puede
aparecer en tres bloques disjuntos por grupo. No es válido prometer una partición
estratificada por origen de tres conjuntos para las tres clases con estos datos.

## Integridad, etiquetas y metadatos

Se contrastaron info.labels raíz, training y testing, existencia de archivos,
categoría, etiqueta, ID y nombre. Incidencias estructurales: {len(issues)}.
La relación íntegra figura en incidencias.csv. Índices de fragmento distintos
entre nombre de archivo y nombre original en metadata: {summary['renumbered_files']}.
Se conservan ambos en inventario_archivos.csv; no se renombró ningún archivo.
Ordenar archivos alfabéticamente o por el sufijo exportado no reconstruye el tiempo.
El índice original de metadata es una pista de orden, tampoco un timestamp verificado.

Unidades declaradas: {', '.join(summary['sensor_units'])}. Tipo de dispositivo:
{', '.join(summary['device_types'])}. No se infieren unidades físicas, rango,
calibración, orientación, RPM, fallas, filtros del sensor o condiciones del montaje.
Se inventariaron firmas, pero no se verificó HMAC: no se dispone de la clave.

## Duplicación completa y parcial

Grupos de duplicados completos (bytes de archivo o señal numérica):
{summary['full_duplicate_groups']}. Se buscaron secuencias triaxiales exactas en
TODAS las posiciones, no sólo en los saltos del ventaneo de entrenamiento.

| Longitud de secuencia | Posiciones examinadas | Pares entre archivos | Pares train/test |
|---|---:|---:|---:|
{blocks}

Los conteos de la tabla corresponden a parejas de posiciones; varios corresponden
al mismo segmento largo y no deben contarse como adquisiciones independientes.
Se consolidaron {repeated_long} segmentos máximos de al menos ocho muestras en
segmentos_exactos_repetidos.csv. Las repeticiones menores también se conservan,
pero pueden ser coincidencias por discretización. Las ventanas RAW únicas que
realmente entran en el modelo son: {summary['aligned_window_uniqueness']}.

No encontrar bloques exactos no descarta registros contiguos, misma sesión,
transformaciones numéricas o similitud estadística. Las coincidencias de una sola
terna, frecuentes con valores discretizados, no se clasifican como fuga de ventanas.

## Ventaneo y derivados

N=16, hop=8: 36 ventanas por JSON, 7128 training y 1332 testing. Dentro de cada
archivo, 35 parejas vecinas comparten ocho muestras. Se cubren 296 de las 300
muestras; las últimas cuatro no entran en ninguna ventana. Ventanas sin solapamiento
(hop=16): 18 por archivo, 3564 training y 666 testing; no se generaron nuevos datasets.
Los tiempos de metadata son locales al JSON, no tiempos absolutos del registro padre.

Se recalcularon las 21 características mediante fórmulas independientes usando
NumPy, incluyendo correcciones de asimetría/curtosis, RFFT, DC y entropía. Se
contrastaron valores, orden, etiquetas, coordenadas, nombres y matrices NPY/CSV.
Resultados detallados: resumen_auditoria.json y verificacion_caracteristicas.csv.
Tolerancia de comparación NPY: rtol=1e-6 y atol=1e-5, por almacenamiento float32.
Un error absoluto aislado no debe interpretarse sin escala y tolerancia.

Se encontraron {summary['near_tied_spectra']} espectros de ventana/eje con más de un
máximo prácticamente empatado (tolerancia relativa 1e-12). En
{summary['scalar_vectorized_argmax_disagreements']} casos, calcular medias/FFT en
lotes en lugar de hacerlo por ventana/eje cambia el argmax por redondeo. La
comparación final conserva la secuencia escalar del preprocesador original. Esto
es sensibilidad numérica del extractor, no evidencia de corrupción del dataset.
Detalles: empates_numericos_fft.csv. No se alteró el extractor original.

## Dependencia y correlación

Se calcularon autocorrelaciones de cada eje a retardos 1,2,4,8,16,32,64 muestras y
correlaciones de características a retardos 1,2,4 ventanas, separadas por archivo.
Los resúmenes agregan coeficientes por clase/split; no mezclan finales e inicios
de archivos ni ofrecen p-valores bajo independencia ficticia. Correlaciones
indefinidas por señal constante se registran vacías.

También se compararon todas las parejas train/test de la misma clase a retardo cero
y con retardos de -150 a +150 muestras (al menos 150 muestras comunes). Se restó
la media de cada eje del segmento comparado antes de la correlación triaxial. Esto evita que DC/
gravedad domine artificialmente el coeficiente. Una correlación pequeña a retardo
cero no demuestra independencia: un desfase, periodicidad o segmentos distintos
de una misma captura pueden dar valores pequeños. Una correlación alta tampoco
prueba copia. No se usaron estas correlaciones para escoger una nueva partición.

Las fronteras entre fragmentos consecutivos por índice de metadata se inspeccionan
en fronteras_candidatas.csv. Saltos de amplitud y continuidad plausible no prueban
cronología, duración de huecos ni ausencia de solapamiento original.

## Consecuencias para las métricas existentes

Se agruparon las predicciones ya guardadas de E1/E4 por origen, sin inferencia nueva
ni reentrenamiento: resultados_existentes_por_origen.csv. Las exactitudes siguen
siendo descripciones de las ventanas evaluadas. No prueban generalización a
adquisiciones independientes. El bootstrap anterior por JSON considera parte de la
dependencia, pero no dependencia adicional entre JSON con un origen compartido.
McNemar por ventana tampoco resuelve ese agrupamiento. No se calculó un nuevo
p-valor ni un intervalo que pretenda resolver la falta de grupos de DETENIDO.

## Decisiones recomendadas

1. Conservar la partición original como referencia histórica de reimplementación.
2. Solicitar/aclarar registro padre, orden, offsets, duración y sesiones con autores.
3. Para una prueba por adquisiciones nuevas, obtener sesiones adicionales de todas
   las clases, reservarlas antes de seleccionar el modelo y mantener la sesión junta.
4. Si sólo se dispone de este corpus, limitar conclusiones a evaluación interna;
   una separación temporal con exclusión sólo es planteable si se verifica el orden.
5. Mantener el 50% de solapamiento si responde al funcionamiento del sistema, pero
   calcular incertidumbre con la unidad experimental apropiada y suficientes grupos.
6. No iniciar una gran batería de entrenamientos hasta cerrar la viabilidad de grupos.

## Reproducibilidad y límites de cobertura

La auditoría comprobó las copias locales de los dos paquetes cuando existen;
verificacion_copias.csv distingue hashes iguales, ausencias y archivos adicionales.
{copy_notes}
Una ausencia no es un archivo modificado: el paquete histórico que no contiene
señales RAW no debe confundirse con el paquete de evidencia actual completo.
No se descargó de nuevo el suplemento oficial en esta ejecución: la comparación
local no sustituye una nueva verificación con el servidor del editor.

Las entradas se verificaron por SHA-256 antes y después. Archivos modificados:
{summary['changed_inputs']}. Comando de reproducción desde la raíz:

```powershell
python articulo/analisis/auditar_dataset_origenes.py --output articulo/auditoria_dataset/otra_ejecucion
```

El directorio de salida debe ser nuevo; sólo se necesita NumPy. Se preservan
manuscrito, modelos, firmware y matrices canónicas. La auditoría no verifica la
realidad física de las etiquetas, autenticidad criptográfica, calibración ni
independencia entre días/motores que el corpus no documenta.
"""
    (output / "INFORME_AUDITORIA.md").write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    allowed = (ROOT / "articulo/auditoria_dataset").resolve()
    if not output.is_relative_to(allowed) or output == allowed:
        raise ValueError("La salida debe estar dentro de articulo/auditoria_dataset/<ejecución>")
    if output.exists():
        raise FileExistsError("No se sobrescribe una auditoría existente")
    raw = ROOT / "datos/arciniegas/raw"
    raw_paths = sorted(p for p in raw.rglob("*") if p.is_file())
    processed_paths = sorted(p for p in (ROOT/"datos/arciniegas/procesado").iterdir() if p.is_file())
    prediction_paths = [ROOT/f"resultados/arciniegas/{e}/validacion_rawall/rawall_todas.csv" for e in ("e1","e4")]
    copy_roots = [ROOT/'Eduardo_Andres_Ollachica_Arias_Replication_Evidence/03_Datasets/Arciniegas_Raw',
                  ROOT/'PAQUETE_VALIDACION_REIMPLEMENTACION_STM32/datos/arciniegas/raw']
    copy_paths = [p for c in copy_roots if c.exists() for p in sorted(c.rglob('*')) if p.is_file()]
    input_paths = raw_paths + processed_paths + [p for p in prediction_paths if p.exists()] + copy_paths
    before = {p: digest(p) for p in input_paths}
    issues = []
    lookup, metadata_summary = metadata_audit(raw, issues)
    records, signals = load_records(raw, lookup, issues)
    origins, groups = origins_audit(records)
    print(f"Leídos {len(records)} JSON, {len(origins)} familias", flush=True)
    full, blocks, block_summary = duplicate_audit(records, signals)
    segments = repeated_segments(blocks, records, signals)
    print("Terminada búsqueda exhaustiva de bloques exactos de 2/4/8/16/32 muestras", flush=True)
    boundaries = boundaries_audit(groups, signals)
    features_by_file = {r["path"]: independently_extract(signals[r["path"]]) for r in records}
    ties = spectral_ties(records, signals)
    processed, feature_checks = processed_audit(ROOT, records, features_by_file, issues)
    rawcorr, featcorr, raw_file_corr, feat_file_corr = correlations_audit(records, signals, features_by_file)
    crosscorr = cross_split_correlations(records, signals)
    raw_distribution, feature_distribution = distributions(records, signals, features_by_file)
    pred = predictions_audit(ROOT, records)
    copies = copy_audit(ROOT, raw, raw_paths)
    print("Terminadas verificaciones de matrices, dependencia y copias", flush=True)
    changed = [p.relative_to(ROOT).as_posix() for p in input_paths if digest(p) != before[p]]
    if changed:
        raise RuntimeError(f"Cambios detectados en entradas: {changed}")
    class_summary = [{"class": label, "files": sum(r['class']==label for r in records),
                      "origins": sum(r['class']==label for r in origins),
                      "train_files": sum(r['class']==label and r['split']=='training' for r in records),
                      "test_files": sum(r['class']==label and r['split']=='testing' for r in records)} for label in LABELS]
    shared = {r['origin_id'] for r in origins if r['shared_between_splits']}
    aligned_unique = {}
    for split in ('training','testing'):
        rr = [r for r in records if r['split']==split]
        window_hashes = [array_digest(signals[r['path']][i:i+16]) for r in rr for i in range(0,285,8)]
        aligned_unique[split] = {'windows':len(window_hashes),'unique_raw_windows':len(set(window_hashes)),
                                 'repeated_excess_windows':len(window_hashes)-len(set(window_hashes))}
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
        "numpy": np.__version__, "script_sha256": digest(Path(__file__)),
        "raw_files": len(raw_paths), "json_files": len(records), "origins": len(origins),
        "class_summary": class_summary, "metadata": metadata_summary, "issues": issues,
        "shared_origins": len(shared),
        "test_files_shared_origin": sum(r['split']=='testing' and r['origin_id'] in shared for r in records),
        "renumbered_files": sum(r['renumbered_fragment'] for r in records),
        "sensor_units": sorted({r['sensor_units'] for r in records}),
        "device_types": sorted({r['device_type'] for r in records}),
        "device_names_count": len({r['device_name'] for r in records}),
        "iat_count": len({r['iat'] for r in records}),
        "constant_axis_files": sum(any(r[f'{axis}_constant'] for axis in AXES) for r in records),
        "full_duplicate_groups": len(full), "exact_blocks": block_summary,
        "repeated_segments_ge8": sum(r['length_samples']>=8 for r in segments),
        "repeated_segments_ge8_training": sum(r['length_samples']>=8 and r['file_a'].startswith('training/') and r['file_b'].startswith('training/') for r in segments),
        "repeated_segments_ge8_testing": sum(r['length_samples']>=8 and r['file_a'].startswith('testing/') and r['file_b'].startswith('testing/') for r in segments),
        "max_cross_split_exact_segment_samples": max((r['length_samples'] for r in segments if r['cross_split']), default=0),
        "aligned_window_uniqueness": aligned_unique,
        "near_tied_spectra":len(ties),
        "scalar_vectorized_argmax_disagreements":sum(r['scalar_vectorized_disagree'] for r in ties),
        "candidate_boundaries": len(boundaries),
        "cross_split_candidate_boundaries": sum(r['cross_split_boundary'] for r in boundaries),
        "processed": processed, "same_class_cross_split_pairs": len(crosscorr),
        "rawall_by_origin": pred, "changed_inputs": len(changed),
        "inputs_hashed": len(input_paths),
        "copy_summary": [{"copy": c, "files_checked": sum(r['copy_root']==c for r in copies),
                          "identical":sum(r['copy_root']==c and r['status']=='identical' for r in copies),
                          "missing":sum(r['copy_root']==c and r['status']=='missing' for r in copies),
                          "different_bytes":sum(r['copy_root']==c and r['status']=='different_bytes' for r in copies),
                          "extra_files":sum(r['copy_root']==c and r['status']=='extra_file' for r in copies)}
                         for c in sorted({r['copy_root'] for r in copies})],
    }
    output.mkdir(parents=True)
    outputs = {
        "inventario_archivos.csv": records, "familias_origen.csv": origins,
        "segmentos_exactos_repetidos.csv":segments,
        "empates_numericos_fft.csv":ties,
        "distribucion_senales.csv":raw_distribution,
        "distribucion_caracteristicas.csv":feature_distribution,
        "fronteras_candidatas.csv": boundaries, "verificacion_caracteristicas.csv": feature_checks,
        "autocorrelacion_senal_resumen.csv": rawcorr, "correlacion_ventanas_resumen.csv": featcorr,
        "autocorrelacion_senal_por_archivo.csv": raw_file_corr,
        "correlacion_ventanas_por_archivo.csv": feat_file_corr,
        "correlacion_train_test_misma_clase.csv": crosscorr,
        "resultados_existentes_por_origen.csv": pred, "verificacion_copias.csv": copies,
    }
    for name, rows in outputs.items():
        csv_write(output / name, finite(rows))
    csv_write(output/"incidencias.csv", issues, ["type", "path", "split"])
    csv_write(output/"duplicados_completos.csv", full, ["criterion", "hash", "files", "cross_split"])
    csv_write(output/"bloques_exactos_repetidos.csv", blocks,
              ["length_samples", "file_a", "start_a", "file_b", "start_b", "cross_split", "same_origin", "sha256"])
    csv_write(output/"integridad_entradas.csv", [{"path": p.relative_to(ROOT).as_posix(), "sha256_before": before[p],
                                               "sha256_after": digest(p), "unchanged": True} for p in input_paths])
    (output/"resumen_auditoria.json").write_text(json.dumps(finite(summary), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    write_report(output, summary, origins, issues)
    manifest = [{"path": p.name, "sha256": digest(p)} for p in sorted(output.iterdir()) if p.is_file()]
    csv_write(output/"manifest_salidas_sha256.csv", manifest)
    print(json.dumps(finite(summary), indent=2, ensure_ascii=False, allow_nan=False), flush=True)
    print(f"INFORME: {output/'INFORME_AUDITORIA.md'}", flush=True)


if __name__ == "__main__":
    main()
