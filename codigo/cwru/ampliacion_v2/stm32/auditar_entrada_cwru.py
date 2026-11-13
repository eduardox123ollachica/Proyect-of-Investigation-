"""Audita escalado/INT8 desde salidas físicas existentes; no abre COM ni entrena."""
from __future__ import annotations
import argparse
import csv
import json
import sys
from pathlib import Path
import numpy as np
from contrato_entrada_cwru import (ROOT, RESOURCES, MODEL, SCALER, FEATURES,
                                 sha, bits, parameters, transform, quantize_affine_manual)

STAGE = ROOT / "despliegue_stm32/mejoras_entropia_180mhz_2026-10-23"
OUT = ROOT / "despliegue_stm32/escalado_cuantizacion_180mhz_2026-10-23"
RUNS = {
    "cache": ROOT / "despliegue_stm32/cache_potencias_180mhz_2026-10-23/Validacion_PC/resultados/20261023T211112_480689Z",
    "entd": STAGE / "Validacion_Double/resultados/20261023T220644_147974Z",
    "entm": STAGE / "Validacion_Mixta/resultados/20261023T221815_802310Z",
}


def save(path, data):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def audit(output: Path):
    if output.exists():
        raise FileExistsError("No sobrescribir una auditoría; elegir otro directorio")
    params = parameters()
    manifest = json.loads((RESOURCES / "manifest.json").read_text(encoding="utf-8"))
    if sha(MODEL) != manifest["source_hashes"]["tflite_int8"] or sha(SCALER) != manifest["source_hashes"]["dnn_scaler"]:
        raise ValueError("Artefactos históricos distintos")
    inputs = [MODEL, SCALER, RESOURCES / "manifest.json", RESOURCES / "cwru_dnn_scaler.cpp",
              RESOURCES / "cwru_dnn_scaler.h", Path(__file__), Path(__file__).with_name("contrato_entrada_cwru.py")]
    reports = {"cache": ROOT / "despliegue_stm32/cache_potencias_180mhz_2026-10-23/RESULTADOS_FISICOS_CACHE_42_VENTANAS.json",
               "entd": STAGE / "RESULTADOS_FISICOS_DOUBLE_42_VENTANAS.json",
               "entm": STAGE / "RESULTADOS_FISICOS_MIXTA_42_VENTANAS.json"}
    arrays, metadata = {}, {}
    npz_path = STAGE / "Validacion_Mixta/ventanas_desarrollo.npz"
    if sha(npz_path) != "ff09491a486716b7176986cc504138a726e003389cb93b657565a3c422c234cb":
        raise ValueError("NPZ de desarrollo diferente")
    with np.load(npz_path, allow_pickle=False) as bundle:
        reference_npz = bundle["features"].astype(np.float32)
    inputs.append(npz_path)
    for label, folder in RUNS.items():
        report = json.loads(reports[label].read_text(encoding="utf-8"))
        source = json.loads((folder / "resultados.json").read_text(encoding="utf-8"))
        if sha(folder / "resultados.json") != report["trace"]["results_sha256"] or sha(folder / "uart_trace.jsonl") != report["trace"]["uart_trace_sha256"]:
            raise ValueError("Traza física distinta: " + label)
        if [row["index"] for row in source] != list(range(42)) or any(int(row["load_hp"]) == 3 for row in source):
            raise ValueError("Desarrollo incompleto o incluye 3 HP")
        observed, reference = [], []
        trace = [json.loads(line) for line in (folder / "uart_trace.jsonl").read_text(encoding="utf-8").splitlines()]
        responses = [row["line"].split(",") for row in trace if row["direction"] == "RX" and row["line"].startswith("CWRUFEAT_RESULT,")]
        if len(responses) != 42:
            raise ValueError("Respuestas de extracción incompletas")
        inputs.extend([reports[label], folder / "resultados.json", folder / "uart_trace.jsonl"])
        for index in range(42):
            path = folder / f"ventana_{index:03d}.csv"
            with path.open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            # El cliente histórico tiene un espacio accidental en el nombre FE frecuencia.
            names = [row["feature"].replace(" ", "") for row in rows]
            if names != list(FEATURES) or not all(row["pass"] == "True" for row in rows):
                raise ValueError("CSV no aprobado o con orden diferente")
            observed.append([float(row["stm32_float32"]) for row in rows])
            if int(responses[index][1]) != source[index]["cycles"] or bits(observed[-1]) != bits(np.asarray(responses[index][2:], dtype=np.float32)):
                raise ValueError("CSV diferente de la respuesta física")
            reference.append([float(row["pc_float64"]) for row in rows])
            inputs.append(path)
        arrays[label] = np.asarray(observed, dtype=np.float32)
        current_reference = np.asarray(reference, dtype=np.float64)
        if "pc" in arrays and not np.array_equal(arrays["pc"], current_reference.astype(np.float32)):
            raise ValueError("Referencias PC distintas")
        arrays["pc"] = current_reference.astype(np.float32)
        if not np.array_equal(arrays["pc"], reference_npz):
            raise ValueError("Referencia PC diferente del NPZ")
        metadata[label] = source
    for label in ("entd", "entm"):
        for a, b in zip(metadata["cache"], metadata[label]):
            if any(a[key] != b[key] for key in ("index", "file_id", "class", "load_hp", "window_payload_sha256")):
                raise ValueError("Ventanas no pareadas")
    before_hashes = {str(path): sha(path) for path in inputs}
    transformed = {label: [transform(row, params) for row in array] for label, array in arrays.items()}
    comparisons = {}
    for label, base in (("entd", "cache"), ("entm", "cache"), ("cache", "pc"), ("entm", "pc")):
        z = np.stack([row["standardized"] for row in transformed[label]])
        bz = np.stack([row["standardized"] for row in transformed[base]])
        q = np.stack([row["quantized"] for row in transformed[label]])
        bq = np.stack([row["quantized"] for row in transformed[base]])
        comparisons[label + "_vs_" + base] = dict(
            feature_bits_equal=int((arrays[label].view("u4") == arrays[base].view("u4")).sum()),
            standardized_bits_equal=int((z.view("u4") == bz.view("u4")).sum()),
            quantized_values_equal=int((q == bq).sum()), changed_windows=int(np.any(q != bq, axis=1).sum()),
            changed_codes=[dict(index=int(i), feature=FEATURES[j], base=int(bq[i, j]), candidate=int(q[i, j]))
                           for i, j in np.argwhere(q != bq)],
            max_standardized_absolute_difference=float(np.abs(z.astype(float) - bz.astype(float)).max()))

    # Todos los empates del rango útil, sus vecinos float32 y casos de saturación.
    ties = np.arange(-140, 141, dtype=np.float32) + np.float32(.5)
    neighbors = np.concatenate([ties, np.nextafter(ties, -np.inf, dtype=np.float32),
                                np.nextafter(ties, np.inf, dtype=np.float32),
                                np.array([-1e30, 1e30, -0., 0.], dtype=np.float32)])
    rng = np.random.default_rng(20261023)
    tests = np.concatenate([neighbors, rng.uniform(-1000, 1000, 100000).astype(np.float32)])
    np.testing.assert_array_equal(quantize_affine_manual(tests), np.clip(np.rint(tests), -128, 127).astype(np.int8))
    output.mkdir(parents=True)
    detailed = []
    for label, values in transformed.items():
        for i, row in enumerate(values):
            for j, name in enumerate(FEATURES):
                detailed.append(dict(source=label, index=i, file_id=metadata["entm"][i]["file_id"],
                    load_hp=metadata["entm"][i]["load_hp"], feature=name,
                    raw_float32=float(arrays[label][i, j]), raw_bits=bits([arrays[label][i, j]])[0],
                    standardized=float(row["standardized"][j]), standardized_bits=bits([row["standardized"][j]])[0],
                    affine=float(row["affine"][j]), affine_bits=bits([row["affine"][j]])[0],
                    int8=int(row["quantized"][j]), saturated_low=bool(row["saturated_low"][j]),
                    saturated_high=bool(row["saturated_high"][j])))
    with (output / "comparacion_por_caracteristica.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(detailed[0]))
        writer.writeheader(); writer.writerows(detailed)
    cases = [dict(name=f"{label}_{i:03d}", mode="RUN", group=label, index=i, input_bits=bits(values[i]))
             for label, values in arrays.items() if label != "entd" for i in range(42)]
    # QUANT recibe z, no x. Estos casos no sufren reconstrucción artificial del escalado.
    z_ties = np.concatenate([np.arange(-140, 141, dtype=np.float32) + .5, np.array([-0., 0.], dtype=np.float32)])
    for offset in range(0, len(z_ties), 14):
        desired = np.resize(z_ties[offset:offset + 14], 14).astype(np.float32)
        z = np.multiply(np.subtract(desired, np.float32(params["zero_point"]), dtype=np.float32), params["quant_scale"], dtype=np.float32)
        for kind, values in (("center", z), ("below", np.nextafter(z, -np.inf, dtype=np.float32)),
                             ("above", np.nextafter(z, np.inf, dtype=np.float32))):
            cases.append(dict(name=f"quant_boundary_{offset}_{kind}", mode="QUANT", group="synthetic", input_bits=bits(values)))
    cases.append(dict(name="zero_from_mean", mode="RUN", group="synthetic", input_bits=bits(params["mean"])))
    cases.append(dict(name="finite_large_saturation", mode="QUANT", group="synthetic", input_bits=bits(np.resize(np.array([-1e20, 1e20], dtype=np.float32), 14))))
    save(output / "casos_validacion.json", cases)
    sat = {label: dict(low=sum(int(r["saturated_low"].sum()) for r in rows),
                       high=sum(int(r["saturated_high"].sum()) for r in rows)) for label, rows in transformed.items()}
    special = [row for row in detailed if row["index"] == 30 and row["feature"] == "de_spectral_entropy"]
    report = dict(status="HOST_CONTRACT_AUDITED_NOT_ARM_VALIDATED", windows=42, features_per_source=588,
        historical_model_only_not_V2=True, model_metadata=params["metadata"],
        quant_scale=float(params["quant_scale"]), quant_scale_bits=bits([params["quant_scale"]])[0],
        zero_point=params["zero_point"], mean_bits=bits(params["mean"]), scale_bits=bits(params["scale"]),
        comparisons=comparisons, saturation=sat, one_ulp_case=special,
        manual_rounding_comparisons=len(tests), serial_test_cases_prepared=len(cases),
        firmware_executed=False, serial_port_opened=False, model_inference_executed=False,
        extraction_experiments_repeated=False, training_executed=False,
        limitations=["Host float32 contract, not execution of C on ARM.",
                     "No classifier outputs, timing or physical independence are established here.",
                     "Quantization result applies to the historical scaler/model and tested vectors only."],
        python=sys.version, numpy=np.__version__)
    save(output / "RESULTADOS_PC.json", report)
    for path in inputs:
        if sha(path) != before_hashes[str(path)]:
            raise RuntimeError("Entrada modificada durante auditoría")
    save(output / "ENTRADAS_SHA256.json", {str(Path(path).relative_to(ROOT)): digest for path, digest in before_hashes.items()})
    save(output / "MANIFEST_SHA256.json", {p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()})
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT / "auditoria_pc")
    audit(parser.parse_args().output.resolve())
