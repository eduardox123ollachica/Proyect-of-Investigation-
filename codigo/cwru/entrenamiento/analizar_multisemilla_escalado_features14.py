"""Consolida la comparación multisemilla RAW--StandardScaler en validación."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = (
    PROJECT_ROOT
    / "resultados"
    / "cwru"
    / "modelo_features_14"
    / "multisemilla_escalado"
)
SEEDS = (11, 22, 33, 42, 44, 55, 66, 77, 88, 99)
METRICS = ("accuracy", "macro_f1")


def descriptive(values: np.ndarray) -> dict:
    mean = float(np.mean(values))
    standard_deviation = float(np.std(values, ddof=1))
    half_width = float(stats.t.ppf(0.975, len(values) - 1) * stats.sem(values))
    return {
        "mean": mean,
        "sample_standard_deviation": standard_deviation,
        "median": float(np.median(values)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "ci95_t": [mean - half_width, mean + half_width],
    }


def main() -> None:
    table = pd.read_csv(RESULTS_DIR / "resultados_validacion_por_semilla.csv")
    expected_rows = len(SEEDS) * 2
    if len(table) != expected_rows:
        raise RuntimeError(f"Se esperaban {expected_rows} filas y se encontraron {len(table)}.")

    summary: dict[str, object] = {
        "scope": "training_and_validation_only",
        "test_loaded": False,
        "seeds": list(SEEDS),
        "n_seeds": len(SEEDS),
        "variants": {},
        "paired_differences_standard_minus_raw": {},
    }
    for variant, group in table.groupby("variant"):
        summary["variants"][variant] = {
            metric: descriptive(group[metric].to_numpy(dtype=float))
            for metric in METRICS
        }

    pivot = table.pivot(index="seed", columns="variant", values=list(METRICS))
    for metric in METRICS:
        differences = (
            pivot[metric]["standard"] - pivot[metric]["raw"]
        ).to_numpy(dtype=float)
        paired = descriptive(differences)
        wins = int(np.sum(differences > 0))
        losses = int(np.sum(differences < 0))
        ties = int(np.sum(differences == 0))
        paired.update(
            {
                "standard_wins": wins,
                "ties": ties,
                "raw_wins": losses,
                "exact_two_sided_sign_test_p": float(
                    stats.binomtest(wins, wins + losses, 0.5).pvalue
                ),
            }
        )
        summary["paired_differences_standard_minus_raw"][metric] = paired

    weight_checks = []
    prediction_hashes: dict[str, list[str]] = {"raw": [], "standard": []}
    for seed in SEEDS:
        for variant in ("raw", "standard"):
            variant_dir = RESULTS_DIR / f"seed_{seed}" / variant
            weights = np.load(variant_dir / "pesos_modelo.npz")
            shapes = [weights[key].shape for key in sorted(weights.files)]
            valid = shapes == [(14, 20), (20,), (20, 10), (10,), (10, 3), (3,)]
            valid = valid and all(np.isfinite(weights[key]).all() for key in weights.files)
            weight_checks.append(valid)

            predictions = pd.read_csv(variant_dir / "predicciones_validacion.csv")[
                "y_pred"
            ].to_numpy()
            prediction_hashes[variant].append(
                hashlib.sha256(predictions.tobytes()).hexdigest()
            )

    summary["artifact_checks"] = {
        "valid_weight_sets": int(sum(weight_checks)),
        "expected_weight_sets": len(weight_checks),
        "all_weight_sets_valid": bool(all(weight_checks)),
        "identical_discrete_predictions_across_seeds": {
            variant: len(set(hashes)) == 1
            for variant, hashes in prediction_hashes.items()
        },
    }

    output_path = RESULTS_DIR / "resumen_multisemilla.json"
    output_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
