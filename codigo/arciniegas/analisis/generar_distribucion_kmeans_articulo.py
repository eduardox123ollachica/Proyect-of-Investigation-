"""Genera la figura y el resumen del detector K-means para el articulo.

Utiliza los scores STANDARD ya consolidados del experimento compartido. No
reentrena K-means ni recalcula el umbral: verifica que el P99 registrado sea
coherente con training y representa su comportamiento en datos conocidos.
"""

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CODIGO_DIR = Path(__file__).resolve().parents[2]
if str(CODIGO_DIR) not in sys.path:
    sys.path.insert(0, str(CODIGO_DIR))

from comun.rutas_proyecto import RAIZ_PROYECTO, RESULTADOS_KMEANS  # noqa: E402


TRAIN_PATH = RESULTADOS_KMEANS / "scores_train_STANDARD.csv"
TEST_PATH = RESULTADOS_KMEANS / "scores_test_STANDARD.csv"
THRESHOLDS_PATH = RESULTADOS_KMEANS / "umbrales_STANDARD.csv"

FIGURES_DIR = RAIZ_PROYECTO / "articulo" / "figuras"
DERIVED_RESULTS_DIR = RAIZ_PROYECTO / "articulo" / "resultados_derivados"
PNG_PATH = FIGURES_DIR / "distribucion_scores_kmeans_p99.png"
SVG_PATH = FIGURES_DIR / "distribucion_scores_kmeans_p99.svg"
SUMMARY_PATH = DERIVED_RESULTS_DIR / "resumen_distribucion_kmeans_p99.csv"

CLASS_ORDER = ["DETENIDO", "VELOCIDAD MEDIA", "VELOCIDAD ALTA"]
SHORT_CLASS_NAMES = ["DETENIDO", "MEDIA", "ALTA"]
CLASS_COLORS = ["#4C78A8", "#F58518", "#54A24B"]
RNG_SEED = 20261012


def load_evidence():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    thresholds = pd.read_csv(THRESHOLDS_PATH)

    required_train = {"nombre", "score_d2"}
    required_test = {"nombre_real", "score_d2"}
    if not required_train.issubset(train.columns):
        raise ValueError(f"Columnas ausentes en training: {required_train - set(train.columns)}")
    if not required_test.issubset(test.columns):
        raise ValueError(f"Columnas ausentes en test: {required_test - set(test.columns)}")

    p99_rows = thresholds.loc[thresholds["metodo"] == "P99", "umbral_d2"]
    if len(p99_rows) != 1:
        raise ValueError("umbrales_STANDARD.csv no contiene un unico umbral P99")
    threshold = float(p99_rows.iloc[0])

    calculated_p99 = float(np.percentile(train["score_d2"], 99))
    if not np.isclose(threshold, calculated_p99, rtol=1e-10, atol=1e-10):
        raise ValueError(
            f"El P99 registrado ({threshold}) no coincide con training ({calculated_p99})"
        )

    if (train["score_d2"] <= 0).any() or (test["score_d2"] <= 0).any():
        raise ValueError("La escala logaritmica requiere scores estrictamente positivos")

    return train, test, threshold


def empirical_cdf(values):
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    cumulative = np.arange(1, len(ordered) + 1, dtype=np.float64) / len(ordered)
    return ordered, cumulative


def summarize_group(split, class_name, values, threshold):
    values = np.asarray(values, dtype=np.float64)
    rejected = int(np.sum(values > threshold))
    return {
        "split": split,
        "class": class_name,
        "windows": len(values),
        "rejected_p99": rejected,
        "rejected_pct": 100.0 * rejected / len(values),
        "score_median": float(np.median(values)),
        "score_p95": float(np.percentile(values, 95)),
        "score_p99": float(np.percentile(values, 99)),
        "score_max": float(np.max(values)),
        "threshold_p99_training": threshold,
    }


def build_summary(train, test, threshold):
    rows = [
        summarize_group("training", "TODAS", train["score_d2"], threshold),
        summarize_group("testing", "TODAS", test["score_d2"], threshold),
    ]
    for class_name in CLASS_ORDER:
        train_values = train.loc[train["nombre"] == class_name, "score_d2"]
        test_values = test.loc[test["nombre_real"] == class_name, "score_d2"]
        rows.append(summarize_group("training", class_name, train_values, threshold))
        rows.append(summarize_group("testing", class_name, test_values, threshold))
    return pd.DataFrame(rows)


def create_figure(train, test, threshold):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7.5,
        }
    )

    fig, (ax_cdf, ax_class) = plt.subplots(
        2,
        1,
        figsize=(3.45, 5.25),
        constrained_layout=True,
    )

    train_x, train_y = empirical_cdf(train["score_d2"])
    test_x, test_y = empirical_cdf(test["score_d2"])
    ax_cdf.plot(train_x, train_y, color="#4C78A8", linewidth=1.4, label="Entrenamiento")
    ax_cdf.plot(test_x, test_y, color="#E45756", linewidth=1.4, label="Prueba")
    ax_cdf.axvline(
        threshold,
        color="#222222",
        linestyle="--",
        linewidth=1.1,
        label=f"P99 training = {threshold:.2f}",
    )
    ax_cdf.set_xscale("log")
    ax_cdf.set_ylim(0.0, 1.015)
    ax_cdf.set_xlabel(r"Distancia al centroide más cercano, $d^2$")
    ax_cdf.set_ylabel("Distribución acumulada empírica")
    ax_cdf.set_title("(a) Distribución global")
    ax_cdf.legend(loc="lower right", frameon=True, framealpha=0.9)

    rng = np.random.default_rng(RNG_SEED)
    class_values = []
    rejection_labels = []
    for position, (class_name, color) in enumerate(
        zip(CLASS_ORDER, CLASS_COLORS), start=1
    ):
        values = test.loc[test["nombre_real"] == class_name, "score_d2"].to_numpy()
        class_values.append(values)
        jitter = rng.uniform(-0.18, 0.18, size=len(values))
        rejected = values > threshold
        ax_class.scatter(
            position + jitter[~rejected],
            values[~rejected],
            s=5,
            color=color,
            alpha=0.22,
            linewidths=0,
            rasterized=True,
        )
        ax_class.scatter(
            position + jitter[rejected],
            values[rejected],
            s=20,
            marker="x",
            color="#B00020",
            linewidths=1.0,
            zorder=4,
        )
        rejection_labels.append(f"{int(rejected.sum())}/{len(values)}")

    box = ax_class.boxplot(
        class_values,
        positions=np.arange(1, 4),
        widths=0.42,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#111111", "linewidth": 1.2},
        whiskerprops={"color": "#555555", "linewidth": 0.8},
        capprops={"color": "#555555", "linewidth": 0.8},
        boxprops={"color": "#555555", "linewidth": 0.8},
    )
    for patch, color in zip(box["boxes"], CLASS_COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.22)

    ax_class.axhline(
        threshold,
        color="#222222",
        linestyle="--",
        linewidth=1.1,
        label="Umbral P99",
    )
    ax_class.set_yscale("log")
    ax_class.set_xticks(np.arange(1, 4), SHORT_CLASS_NAMES)
    ax_class.set_xlabel("Clase real del conjunto de prueba")
    ax_class.set_ylabel(r"Distancia al centroide más cercano, $d^2$")
    ax_class.set_title("(b) Scores y rechazos por clase")

    top_y = max(float(test["score_d2"].max()), threshold) * 1.5
    ax_class.set_ylim(float(test["score_d2"].min()) * 0.75, top_y)
    for position, label in enumerate(rejection_labels, start=1):
        ax_class.text(
            position,
            0.97,
            label,
            transform=ax_class.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=7.5,
            color="#B00020",
        )

    for axis in (ax_cdf, ax_class):
        axis.grid(True, which="both", color="#D9D9D9", linewidth=0.45)
        axis.set_axisbelow(True)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG_PATH, dpi=600, bbox_inches="tight")
    fig.savefig(SVG_PATH, bbox_inches="tight")
    plt.close(fig)


def main():
    train, test, threshold = load_evidence()
    summary = build_summary(train, test, threshold)

    DERIVED_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_PATH, index=False)
    create_figure(train, test, threshold)

    print(summary.to_string(index=False))
    print(f"\nPNG: {PNG_PATH}")
    print(f"SVG: {SVG_PATH}")
    print(f"CSV: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
