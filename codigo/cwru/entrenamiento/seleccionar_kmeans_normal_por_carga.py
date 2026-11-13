"""Selecciona K-means normal-only mediante validación cruzada por carga.

Desarrollo usa exclusivamente las cargas 0, 1 y 2 HP. La carga 3 HP y todos
los archivos de prueba permanecen sin cargar.
"""

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
N_INIT = 20
CANDIDATE_K = (1, 2, 4, 8)
NORMAL_LABEL = 0
THRESHOLD_PERCENTILE = 95.0
FEATURE_INDICES = (0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 12, 13)
FEATURE_NAMES = (
    "de_rms",
    "de_skewness",
    "de_kurtosis",
    "de_spectral_energy",
    "de_dominant_amplitude",
    "de_spectral_entropy",
    "fe_rms",
    "fe_skewness",
    "fe_kurtosis",
    "fe_spectral_energy",
    "fe_dominant_amplitude",
    "fe_spectral_entropy",
)
FEATURES_DIR = DATOS_CWRU_PROCESADO / "dual_12khz_w4096" / "features_14"
DATASET_DIR = DATOS_CWRU_PROCESADO / "dual_12khz_w4096"
OUTPUT_DIR = (
    RESULTADOS_CWRU
    / "modelo_features_14"
    / "kmeans_normal_lolo_sin_frecuencia_dominante"
)


def fit_normal_model(normal_features: np.ndarray, n_clusters: int):
    scaler = StandardScaler()
    standardized = scaler.fit_transform(normal_features)
    model = KMeans(
        n_clusters=n_clusters,
        random_state=SEED,
        n_init=N_INIT,
        algorithm="lloyd",
    )
    model.fit(standardized)
    return (
        scaler.mean_.astype(np.float32),
        scaler.scale_.astype(np.float32),
        model.cluster_centers_.astype(np.float32),
    )


def scores_float32(
    features: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    centroids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    standardized = ((features.astype(np.float32) - mean) / scale).astype(np.float32)
    differences = standardized[:, np.newaxis, :] - centroids[np.newaxis, :, :]
    squared_distances = np.sum(
        differences * differences, axis=2, dtype=np.float32
    ).astype(np.float32)
    clusters = np.argmin(squared_distances, axis=1).astype(np.int32)
    scores = squared_distances[np.arange(len(features)), clusters]
    return scores.astype(np.float32), clusters


def binary_metrics(labels: np.ndarray, rejected: np.ndarray) -> dict:
    is_fault = labels != NORMAL_LABEL
    is_normal = ~is_fault
    true_positive = int(np.sum(rejected & is_fault))
    false_negative = int(np.sum(~rejected & is_fault))
    false_positive = int(np.sum(rejected & is_normal))
    true_negative = int(np.sum(~rejected & is_normal))
    sensitivity = true_positive / (true_positive + false_negative)
    specificity = true_negative / (true_negative + false_positive)
    return {
        "true_positive_fault_corroborated": true_positive,
        "false_negative_fault_not_corroborated": false_negative,
        "false_positive_normal_rejected": false_positive,
        "true_negative_normal_accepted": true_negative,
        "fault_corroboration_sensitivity": float(sensitivity),
        "normal_specificity": float(specificity),
        "balanced_accuracy": float((sensitivity + specificity) / 2.0),
    }


def load_development_data() -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    blocks = []
    labels = []
    metadata = []
    for split in ("train", "validation"):
        features = np.load(FEATURES_DIR / f"X_features_{split}.npy").astype(np.float32)
        split_labels = np.load(FEATURES_DIR / f"y_{split}.npy").astype(np.int64)
        split_metadata = pd.read_csv(DATASET_DIR / f"metadata_{split}.csv")
        if len(features) != len(split_labels) or len(features) != len(split_metadata):
            raise ValueError(f"Desalineación en {split}.")
        if not np.array_equal(split_labels, split_metadata["label"].to_numpy()):
            raise ValueError(f"Etiquetas y metadata no coinciden en {split}.")
        blocks.append(features)
        labels.append(split_labels)
        metadata.append(split_metadata)
    combined_metadata = pd.concat(metadata, ignore_index=True)
    if set(combined_metadata["load_hp"].unique()) != {0, 1, 2}:
        raise ValueError("Desarrollo debe contener exclusivamente cargas 0, 1 y 2 HP.")
    full_features = np.concatenate(blocks)
    return full_features[:, FEATURE_INDICES], np.concatenate(labels), combined_metadata


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    features, labels, metadata = load_development_data()
    loads = metadata["load_hp"].to_numpy(dtype=int)
    row_ids = np.arange(len(features), dtype=int)
    all_oof_rows = []
    candidate_summaries = []

    for n_clusters in CANDIDATE_K:
        candidate_rows = []
        for held_load in (0, 1, 2):
            normal_fit = (labels == NORMAL_LABEL) & (loads != held_load)
            held_out = loads == held_load
            mean, scale, centroids = fit_normal_model(features[normal_fit], n_clusters)
            held_scores, held_clusters = scores_float32(
                features[held_out], mean, scale, centroids
            )
            held_indices = row_ids[held_out]
            for local_index, global_index in enumerate(held_indices):
                candidate_rows.append(
                    {
                        "k": n_clusters,
                        "held_load_hp": held_load,
                        "global_development_index": int(global_index),
                        "file": metadata.iloc[global_index]["file"],
                        "class": metadata.iloc[global_index]["class"],
                        "label": int(labels[global_index]),
                        "score_squared_distance": float(held_scores[local_index]),
                        "cluster": int(held_clusters[local_index]),
                    }
                )

        candidate_table = pd.DataFrame(candidate_rows).sort_values(
            "global_development_index"
        )
        if len(candidate_table) != len(features):
            raise RuntimeError("La validación fuera de carga no cubrió todo desarrollo.")
        normal_scores = candidate_table.loc[
            candidate_table["label"] == NORMAL_LABEL, "score_squared_distance"
        ].to_numpy(dtype=np.float32)
        threshold = np.float32(np.percentile(normal_scores, THRESHOLD_PERCENTILE))
        rejected = candidate_table["score_squared_distance"].to_numpy() > threshold
        candidate_table["rejected_as_non_normal"] = rejected.astype(np.uint8)
        aggregate = binary_metrics(candidate_table["label"].to_numpy(), rejected)
        by_load = {}
        for load in (0, 1, 2):
            mask = candidate_table["held_load_hp"].to_numpy() == load
            by_load[str(load)] = binary_metrics(
                candidate_table.loc[mask, "label"].to_numpy(), rejected[mask]
            )
        candidate_summaries.append(
            {
                "k": n_clusters,
                "threshold_oof_normal_p95": float(threshold),
                **aggregate,
                "minimum_fault_sensitivity_across_loads": float(
                    min(
                        item["fault_corroboration_sensitivity"]
                        for item in by_load.values()
                    )
                ),
                "minimum_normal_specificity_across_loads": float(
                    min(item["normal_specificity"] for item in by_load.values())
                ),
                "by_held_load": by_load,
            }
        )
        all_oof_rows.append(candidate_table)

    summary_table = pd.DataFrame(
        [
            {key: value for key, value in item.items() if key != "by_held_load"}
            for item in candidate_summaries
        ]
    )
    summary_table.to_csv(OUTPUT_DIR / "comparacion_k_validacion_por_carga.csv", index=False)
    pd.concat(all_oof_rows, ignore_index=True).to_csv(
        OUTPUT_DIR / "scores_fuera_de_carga.csv", index=False
    )

    selected = sorted(
        candidate_summaries,
        key=lambda item: (
            item["balanced_accuracy"],
            item["minimum_fault_sensitivity_across_loads"],
            -item["k"],
        ),
        reverse=True,
    )[0]
    selected_k = int(selected["k"])
    selected_threshold = np.float32(selected["threshold_oof_normal_p95"])

    development_normal = labels == NORMAL_LABEL
    final_mean, final_scale, final_centroids = fit_normal_model(
        features[development_normal], selected_k
    )
    final_scores, final_clusters = scores_float32(
        features, final_mean, final_scale, final_centroids
    )
    final_rejected = final_scores > selected_threshold
    final_development_metrics = binary_metrics(labels, final_rejected)

    model_path = OUTPUT_DIR / "kmeans_normal_final_float32.npz"
    np.savez(
        model_path,
        scaler_mean=final_mean,
        scaler_scale=final_scale,
        centroids=final_centroids,
        threshold=np.asarray(selected_threshold, dtype=np.float32),
        selected_k=np.asarray(selected_k, dtype=np.int32),
    )
    final_table = metadata[
        ["split", "file", "class", "label", "diameter_in", "load_hp", "window_index_in_file"]
    ].copy()
    final_table["cluster"] = final_clusters
    final_table["score_squared_distance"] = final_scores
    final_table["rejected_as_non_normal"] = final_rejected.astype(np.uint8)
    final_table.to_csv(OUTPUT_DIR / "scores_modelo_final_desarrollo.csv", index=False)

    output = {
        "status": "selected_with_load_grouped_development_test_not_loaded",
        "development_loads_hp": [0, 1, 2],
        "test_load_hp": 3,
        "test_loaded": False,
        "normal_training_definition": "class_N_only",
        "input_feature_indices_from_dnn_vector": list(FEATURE_INDICES),
        "input_feature_names": list(FEATURE_NAMES),
        "excluded_features": ["de_dominant_frequency", "fe_dominant_frequency"],
        "exclusion_reason": (
            "Within-load zero variance and a discontinuous healthy-load shift made "
            "normal-only standardization singular and dominated the distance."
        ),
        "candidate_k": list(CANDIDATE_K),
        "selection_rule": (
            "maximum pooled leave-one-load-out balanced accuracy; then maximum "
            "minimum fault sensitivity across loads; then smaller K"
        ),
        "threshold_rule": "P95_of_out_of_fold_normal_scores_for_each_candidate_K",
        "candidate_results": candidate_summaries,
        "selected_k": selected_k,
        "selected_threshold": float(selected_threshold),
        "selected_oof_metrics": {
            key: value
            for key, value in selected.items()
            if key not in {"k", "threshold_oof_normal_p95", "by_held_load"}
        },
        "selected_oof_metrics_by_load": selected["by_held_load"],
        "final_model_fit_scope": "all_normal_windows_from_loads_0_1_2",
        "final_model_development_descriptive_metrics": final_development_metrics,
        "interpretation": (
            "Rejection is corroborative evidence of departure from learned healthy "
            "operation; it does not identify IR/OR or independently confirm a fault."
        ),
        "artifact": {
            "path": model_path.name,
            "bytes": model_path.stat().st_size,
            "sha256": sha256(model_path),
            "centroid_shape": list(final_centroids.shape),
        },
    }
    with (OUTPUT_DIR / "seleccion_kmeans_normal.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(output, file, indent=2, ensure_ascii=False)
        file.write("\n")
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
