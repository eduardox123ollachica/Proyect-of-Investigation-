"""Ajusta y valida K-means CWRU sin consultar el conjunto de prueba."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

CODIGO_DIR = Path(__file__).resolve().parents[2]
if str(CODIGO_DIR) not in sys.path:
    sys.path.insert(0, str(CODIGO_DIR))

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from comun.rutas_proyecto import DATOS_CWRU_PROCESADO, RESULTADOS_CWRU


SEED = 42
N_CLUSTERS = 32
N_INIT = 20
THRESHOLD_PERCENTILE = 99.0
CLASS_NAMES = ("N", "IR", "OR")
FEATURES_DIR = DATOS_CWRU_PROCESADO / "dual_12khz_w4096" / "features_14"
OUTPUT_DIR = RESULTADOS_CWRU / "modelo_features_14" / "kmeans_k32_seed42"


def embedded_scores(
    features: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    centroids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    standardized = ((features.astype(np.float32) - mean) / scale).astype(np.float32)
    difference = standardized[:, np.newaxis, :] - centroids[np.newaxis, :, :]
    squared_distances = np.sum(
        difference * difference, axis=2, dtype=np.float32
    ).astype(np.float32)
    clusters = np.argmin(squared_distances, axis=1).astype(np.int32)
    scores = squared_distances[np.arange(len(features)), clusters]
    return scores.astype(np.float32), clusters


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    x_train = np.load(FEATURES_DIR / "X_features_train.npy").astype(np.float32)
    y_train = np.load(FEATURES_DIR / "y_train.npy").astype(np.int64)
    x_validation = np.load(FEATURES_DIR / "X_features_validation.npy").astype(
        np.float32
    )
    y_validation = np.load(FEATURES_DIR / "y_validation.npy").astype(np.int64)
    if x_train.shape != (568, 14) or x_validation.shape != (406, 14):
        raise ValueError("Las matrices de características no tienen la forma esperada.")

    scaler = StandardScaler()
    train_standard = scaler.fit_transform(x_train)
    validation_standard = scaler.transform(x_validation)
    kmeans = KMeans(
        n_clusters=N_CLUSTERS,
        random_state=SEED,
        n_init=N_INIT,
        algorithm="lloyd",
    )
    kmeans.fit(train_standard)

    mean = scaler.mean_.astype(np.float32)
    scale = scaler.scale_.astype(np.float32)
    centroids = kmeans.cluster_centers_.astype(np.float32)
    train_scores, train_clusters = embedded_scores(x_train, mean, scale, centroids)
    validation_scores, validation_clusters = embedded_scores(
        x_validation, mean, scale, centroids
    )
    threshold = np.float32(np.percentile(train_scores, THRESHOLD_PERCENTILE))
    train_rejected = train_scores > threshold
    validation_rejected = validation_scores > threshold

    sklearn_train_clusters = kmeans.predict(train_standard)
    sklearn_validation_clusters = kmeans.predict(validation_standard)
    cluster_agreement = {
        "training": int(np.sum(train_clusters == sklearn_train_clusters)),
        "training_total": int(len(train_clusters)),
        "validation": int(np.sum(validation_clusters == sklearn_validation_clusters)),
        "validation_total": int(len(validation_clusters)),
    }

    per_class = {}
    for class_index, class_name in enumerate(CLASS_NAMES):
        mask = y_validation == class_index
        per_class[class_name] = {
            "windows": int(np.sum(mask)),
            "rejected_known_domain_windows": int(np.sum(validation_rejected[mask])),
            "rejection_fraction": float(np.mean(validation_rejected[mask])),
            "score_median": float(np.median(validation_scores[mask])),
            "score_p99": float(np.percentile(validation_scores[mask], 99)),
        }

    model_path = OUTPUT_DIR / "kmeans_k32_modelo_float32.npz"
    np.savez(
        model_path,
        scaler_mean=mean,
        scaler_scale=scale,
        centroids=centroids,
        threshold=np.asarray(threshold, dtype=np.float32),
    )
    pd.DataFrame(
        {
            "y_true": y_train,
            "cluster": train_clusters,
            "score_squared_distance": train_scores,
            "rejected": train_rejected.astype(np.uint8),
        }
    ).to_csv(OUTPUT_DIR / "scores_entrenamiento.csv", index=False)
    pd.DataFrame(
        {
            "y_true": y_validation,
            "cluster": validation_clusters,
            "score_squared_distance": validation_scores,
            "rejected_known_domain": validation_rejected.astype(np.uint8),
        }
    ).to_csv(OUTPUT_DIR / "scores_validacion.csv", index=False)

    summary = {
        "status": "fitted_on_training_validated_on_validation_test_not_loaded",
        "features": 14,
        "n_clusters": N_CLUSTERS,
        "random_state": SEED,
        "n_init": N_INIT,
        "algorithm": "lloyd",
        "scaler_fit_scope": "training_only",
        "threshold_definition": "percentile_99_of_float32_training_min_squared_distance",
        "threshold": float(threshold),
        "training": {
            "windows": int(len(x_train)),
            "rejected_by_strict_greater_than": int(np.sum(train_rejected)),
            "rejection_fraction": float(np.mean(train_rejected)),
            "score_median": float(np.median(train_scores)),
            "score_p99": float(np.percentile(train_scores, 99)),
            "score_maximum": float(np.max(train_scores)),
        },
        "validation_known_domain": {
            "windows": int(len(x_validation)),
            "rejected_windows": int(np.sum(validation_rejected)),
            "rejection_fraction": float(np.mean(validation_rejected)),
            "score_median": float(np.median(validation_scores)),
            "score_p99": float(np.percentile(validation_scores, 99)),
            "score_maximum": float(np.max(validation_scores)),
            "per_class": per_class,
        },
        "float32_export_vs_sklearn_cluster_agreement": cluster_agreement,
        "artifact": {
            "path": model_path.name,
            "bytes": model_path.stat().st_size,
            "sha256": sha256(model_path),
            "centroid_shape": list(centroids.shape),
        },
        "interpretation_limit": (
            "Validation contains known classes but no independent anomaly labels; "
            "rejections are statistically atypical known-domain windows, not proven "
            "false alarms or detected faults."
        ),
    }
    with (OUTPUT_DIR / "resumen_kmeans_validacion.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)
        file.write("\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
