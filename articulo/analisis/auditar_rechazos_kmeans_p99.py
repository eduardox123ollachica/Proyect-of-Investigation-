"""Audita las 12 ventanas rechazadas por el umbral P99 de K-means.

El analisis no asigna etiquetas de anomalia. Verifica continuidad temporal,
resultado de la DNN, contribuciones al score y forma de la senal cruda para
sustentar una conclusion limitada a lo observable.
"""

from pathlib import Path
import json
import re

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "datos" / "arciniegas" / "procesado"
RAW_TEST = ROOT / "datos" / "arciniegas" / "raw" / "testing"
KMEANS = ROOT / "resultados" / "arciniegas" / "componentes_compartidos" / "kmeans"
REJECTED_PATH = KMEANS / "Analisis_P99" / "ventanas_rechazadas_P99.csv"
ALL_SCORES_PATH = KMEANS / "scores_test_STANDARD.csv"
CENTROIDS_PATH = KMEANS / "centroides_STANDARD.npy"
SCALER_MEAN_PATH = KMEANS / "scaler_mean_STANDARD.npy"
SCALER_SCALE_PATH = KMEANS / "scaler_scale_STANDARD.npy"
RAWALL_E4_PATH = ROOT / "resultados" / "arciniegas" / "e4" / "validacion_rawall" / "rawall_todas.csv"

DERIVED = ROOT / "articulo" / "resultados_derivados"
FIGURES = ROOT / "articulo" / "figuras"
AUDIT_CSV = DERIVED / "auditoria_rechazos_kmeans_p99.csv"
REPORT_MD = DERIVED / "informe_auditoria_rechazos_kmeans_p99.md"
PNG_PATH = FIGURES / "auditoria_ventanas_rechazadas_kmeans.png"
SVG_PATH = FIGURES / "auditoria_ventanas_rechazadas_kmeans.svg"

THRESHOLD = 21.85040644767507
HOP = 8
AXIS_COLORS = ["#0072B2", "#D55E00", "#009E73"]
AXIS_LABELS = ["X", "Y", "Z"]
LABEL_TO_CLASS = {0: "DETENIDO", 1: "VELOCIDAD MEDIA", 2: "VELOCIDAD ALTA"}


def load_feature_names():
    names = []
    for line in (PROCESSED / "feature_names.txt").read_text(encoding="utf-8").splitlines():
        if ":" in line:
            names.append(line.split(":", 1)[1].strip())
    if len(names) != 21:
        raise ValueError(f"Se esperaban 21 features y se encontraron {len(names)}")
    return names


def load_raw_file(filename, cache):
    if filename not in cache:
        data = json.loads((RAW_TEST / filename).read_text(encoding="utf-8"))
        cache[filename] = np.asarray(data["payload"]["values"], dtype=np.float64)
    return cache[filename]


def consecutive_group_length(rejected_windows, current):
    values = set(int(value) for value in rejected_windows)
    start = current
    end = current
    while start - 1 in values:
        start -= 1
    while end + 1 in values:
        end += 1
    return end - start + 1


def capture_id(filename):
    match = re.search(r"\.s(\d+)\.json$", filename)
    return f"s{match.group(1)}" if match else filename[-12:]


def audit():
    rejected = pd.read_csv(REJECTED_PATH).sort_values("score_d2", ascending=False).reset_index(drop=True)
    all_scores = pd.read_csv(ALL_SCORES_PATH)
    rawall = pd.read_csv(RAWALL_E4_PATH)
    centroids = np.load(CENTROIDS_PATH).astype(np.float64)
    scaler_mean = np.load(SCALER_MEAN_PATH).astype(np.float64)
    scaler_scale = np.load(SCALER_SCALE_PATH).astype(np.float64)
    feature_names = load_feature_names()

    if len(rejected) != 12:
        raise ValueError(f"La auditoria esperaba 12 rechazos y encontro {len(rejected)}")

    dnn_lookup = rawall.set_index(["file", "window"])
    score_lookup = all_scores.set_index(["file", "window"])["score_d2"]
    rejected_by_file = rejected.groupby("file")["window"].apply(list).to_dict()
    file_counts = rejected["file"].value_counts().to_dict()
    raw_cache = {}
    audit_rows = []

    for rank, row in rejected.iterrows():
        filename = str(row["file"])
        window_index = int(row["window"])
        feature_vector = row[feature_names].to_numpy(dtype=np.float64)
        standardized = (feature_vector - scaler_mean) / scaler_scale
        cluster = int(row["cluster_kmeans"])
        contributions = (standardized - centroids[cluster]) ** 2
        reconstructed_score = float(contributions.sum())
        source_score = float(row["score_d2"])
        if not np.isclose(reconstructed_score, source_score, rtol=1e-6, atol=1e-5):
            raise ValueError(
                f"Score no reproducible para {filename}, ventana {window_index}: "
                f"{reconstructed_score} != {source_score}"
            )

        order = np.argsort(contributions)[::-1]
        total = contributions.sum()
        dnn = dnn_lookup.loc[(filename, window_index)]
        pred_label = int(dnn["clase_stm32"])
        true_label = int(row["label_real"])

        values = load_raw_file(filename, raw_cache)
        start = int(row["start_sample"])
        end = int(row["end_sample_exclusive"])
        window_values = values[start:end]
        centered = window_values - window_values.mean(axis=0, keepdims=True)
        peak_to_peak = np.ptp(window_values, axis=0)
        max_abs_centered = np.max(np.abs(centered), axis=0)

        key_prev = (filename, window_index - 1)
        key_next = (filename, window_index + 1)
        prev_score = float(score_lookup.loc[key_prev]) if key_prev in score_lookup.index else np.nan
        next_score = float(score_lookup.loc[key_next]) if key_next in score_lookup.index else np.nan
        group_length = consecutive_group_length(rejected_by_file[filename], window_index)

        audit_row = {
            "rank_score": rank + 1,
            "class": str(row["class"]),
            "capture": capture_id(filename),
            "file": filename,
            "window": window_index,
            "start_time_s": float(row["start_time_s"]),
            "end_time_s": float(row["end_time_s"]),
            "score_d2": source_score,
            "score_threshold_ratio": source_score / THRESHOLD,
            "cluster": cluster,
            "rejected_windows_in_file": int(file_counts[filename]),
            "consecutive_group_length": group_length,
            "temporally_isolated": group_length == 1,
            "previous_window_score": prev_score,
            "next_window_score": next_score,
            "dnn_true_class": LABEL_TO_CLASS[true_label],
            "dnn_predicted_class": LABEL_TO_CLASS[pred_label],
            "dnn_correct": pred_label == true_label,
            "top1_feature": feature_names[int(order[0])],
            "top1_contribution_pct": 100.0 * contributions[order[0]] / total,
            "top2_feature": feature_names[int(order[1])],
            "top2_contribution_pct": 100.0 * contributions[order[1]] / total,
            "top3_feature": feature_names[int(order[2])],
            "top3_contribution_pct": 100.0 * contributions[order[2]] / total,
            "top3_combined_pct": 100.0 * contributions[order[:3]].sum() / total,
            "ax_peak_to_peak": float(peak_to_peak[0]),
            "ay_peak_to_peak": float(peak_to_peak[1]),
            "az_peak_to_peak": float(peak_to_peak[2]),
            "ax_max_abs_centered": float(max_abs_centered[0]),
            "ay_max_abs_centered": float(max_abs_centered[1]),
            "az_max_abs_centered": float(max_abs_centered[2]),
            "score_reconstructed": reconstructed_score,
            "score_reconstruction_error": abs(reconstructed_score - source_score),
        }
        audit_rows.append(audit_row)

    return pd.DataFrame(audit_rows), rejected, raw_cache


def create_context_figure(audit_df, rejected, raw_cache):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7,
            "axes.titlesize": 7.5,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
        }
    )
    fig, axes = plt.subplots(4, 3, figsize=(7.16, 7.9), constrained_layout=True)
    axes = axes.ravel()

    rejected_index = rejected.set_index(["file", "window"])
    for axis, audit_row in zip(axes, audit_df.itertuples(index=False)):
        source = rejected_index.loc[(audit_row.file, audit_row.window)]
        values = raw_cache[audit_row.file]
        start = int(source["start_sample"])
        end = int(source["end_sample_exclusive"])
        context_start = max(0, start - HOP)
        context_end = min(len(values), end + HOP)
        context = values[context_start:context_end]
        centered = context - context.mean(axis=0, keepdims=True)
        time = np.arange(context_start, context_end) / 100.0

        axis.axvspan(start / 100.0, (end - 1) / 100.0, color="#FFD54F", alpha=0.25)
        for axis_index, (label, color) in enumerate(zip(AXIS_LABELS, AXIS_COLORS)):
            axis.plot(time, centered[:, axis_index], color=color, linewidth=0.9, label=label)
        axis.axhline(0.0, color="#666666", linewidth=0.4)
        axis.grid(True, color="#E0E0E0", linewidth=0.35)
        short_class = str(source["class"]).replace("VELOCIDAD ", "VEL. ")
        axis.set_title(
            f"{short_class}, {audit_row.capture}, w{audit_row.window} | "
            f"$d^2$={audit_row.score_d2:.2f}"
        )
        axis.set_xlabel("Tiempo (s)")
        axis.set_ylabel("Señal centrada")

    handles = [
        Line2D([0], [0], color=color, linewidth=1.1, label=label)
        for label, color in zip(AXIS_LABELS, AXIS_COLORS)
    ]
    handles.append(
        plt.Rectangle((0, 0), 1, 1, color="#FFD54F", alpha=0.25, label="Ventana rechazada")
    )
    fig.legend(handles=handles, loc="outside upper center", ncol=4, frameon=False)

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG_PATH, dpi=500, bbox_inches="tight")
    fig.savefig(SVG_PATH, bbox_inches="tight")
    plt.close(fig)


def write_report(audit_df):
    def dataframe_to_markdown(frame):
        headers = [str(column) for column in frame.columns]
        rows = [[str(value) for value in row] for row in frame.itertuples(index=False, name=None)]
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        lines.extend("| " + " | ".join(row) + " |" for row in rows)
        return "\n".join(lines)

    total = len(audit_df)
    isolated = int(audit_df["temporally_isolated"].sum())
    consecutive = total - isolated
    affected_files = audit_df["file"].nunique()
    dnn_correct = int(audit_df["dnn_correct"].sum())
    dominant_files = (
        audit_df.groupby(["class", "capture"], as_index=False)
        .size()
        .sort_values("size", ascending=False)
    )
    top_features = (
        audit_df["top1_feature"].value_counts().rename_axis("feature").reset_index(name="count")
    )

    table_columns = [
        "class",
        "capture",
        "window",
        "score_d2",
        "consecutive_group_length",
        "dnn_correct",
        "top1_feature",
        "top1_contribution_pct",
    ]
    table = audit_df[table_columns].copy()
    table["score_d2"] = table["score_d2"].map(lambda value: f"{value:.3f}")
    table["top1_contribution_pct"] = table["top1_contribution_pct"].map(
        lambda value: f"{value:.1f}"
    )

    lines = [
        "# Auditoría de rechazos K-means P99",
        "",
        "## Alcance",
        "",
        "Se auditaron las 12 ventanas de prueba con `score_d2 > 21.850406448`. "
        "La auditoría reproduce la distancia a partir del scaler y los centroides, "
        "revisa continuidad temporal, resultado DNN, contribuciones por característica "
        "y señal cruda con una ventana vecina a cada lado. No asigna una etiqueta de "
        "anomalía o normalidad que el corpus no contiene.",
        "",
        "## Resultados verificables",
        "",
        f"- {total} rechazos distribuidos en {affected_files} archivos.",
        f"- {isolated} ventanas temporalmente aisladas y {consecutive} pertenecientes "
        "a un único grupo de dos ventanas consecutivas.",
        f"- La DNN E4 clasificó correctamente {dnn_correct}/{total} ventanas rechazadas.",
        f"- La mayor concentración fue {int(dominant_files.iloc[0]['size'])} ventanas "
        f"en la captura {dominant_files.iloc[0]['capture']} de "
        f"{dominant_files.iloc[0]['class']}.",
        "- Los scores fueron reproducidos desde las 21 características, el scaler y "
        "los centroides con error numérico despreciable.",
        "",
        "Características que más veces aportaron la mayor contribución:",
        "",
        dataframe_to_markdown(top_features),
        "",
        "## Detalle por ventana",
        "",
        dataframe_to_markdown(table),
        "",
        "## Conclusión sustentable",
        "",
        "Los rechazos no corresponden a un fallo sistemático de clasificación: la DNN "
        f"acertó {dnn_correct}/{total}, mientras K-means señaló alejamiento local respecto "
        "de los centroides. La predominancia de ventanas aisladas y la concentración de "
        "varios rechazos en una sola captura muestran una combinación de eventos locales "
        "y variación específica de captura. Las contribuciones se originan en características "
        "concretas y no en una discrepancia PC--STM32. Sin etiquetas independientes de "
        "anomalía, la evidencia permite denominarlas ventanas atípicas o rechazos de dominio "
        "conocido, pero no permite decidir si son fallas reales, transitorios normales o falsas "
        "alarmas.",
        "",
    ]
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def main():
    audit_df, rejected, raw_cache = audit()
    DERIVED.mkdir(parents=True, exist_ok=True)
    audit_df.to_csv(AUDIT_CSV, index=False)
    create_context_figure(audit_df, rejected, raw_cache)
    write_report(audit_df)

    print(audit_df[[
        "class", "capture", "window", "score_d2", "consecutive_group_length",
        "dnn_correct", "top1_feature", "top1_contribution_pct"
    ]].to_string(index=False))
    print(f"\nCSV: {AUDIT_CSV}")
    print(f"Informe: {REPORT_MD}")
    print(f"Figura: {PNG_PATH}")


if __name__ == "__main__":
    main()
