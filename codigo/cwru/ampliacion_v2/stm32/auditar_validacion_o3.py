"""Audita trazas fisicas ya guardadas; no conecta hardware ni repite ensayos.

Reproduce las tramas registradas contra el parser original y los vectores NPZ,
compara caracteristicas y tiempos pareados -Os/-O3 y genera un informe nuevo.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import math
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "despliegue_stm32/prueba_o3_180mhz_2026-10-23"
CLIENT = OUT / "Validacion_PC/validar_extraccion.py"
CONTROL = ROOT / "despliegue_stm32/cwru_extraccion_prueba2/Validacion_PC/resultados/20261023T184141_575962Z"
EXPECTED_CLIENT_SHA = "14b5503f9e4b6bf09c8ba6cd53ee6e9b7bc2ee053e0caaf6e7ad95acc632c0ab"
EXPECTED_NPZ_SHA = "ff09491a486716b7176986cc504138a726e003389cb93b657565a3c422c234cb"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RecordedTransport:
    """Devuelve solo respuestas existentes; no inventa datos ni mide tiempos."""
    def __init__(self, records):
        self.records = records
        self.position = 0

    def write(self, payload):
        row = self.records[self.position]
        assert row["direction"] == "TX", "Orden TX/RX incoherente"
        assert payload == (row["command"] + "\r\n").encode("ascii"), "Entrada distinta del NPZ"
        assert len(payload) == row["bytes"], "Longitud TX incoherente"
        self.position += 1
        return len(payload)

    def flush(self):
        pass

    def readline(self):
        row = self.records[self.position]
        assert row["direction"] == "RX", "Falta respuesta RX"
        self.position += 1
        return (row["line"] + "\r\n").encode("ascii")


def replay(folder, windows, reference, client):
    rows = json.loads((folder / "resultados.json").read_text(encoding="utf-8"))
    records = [json.loads(line) for line in (folder / "uart_trace.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 42 and [row["index"] for row in rows] == list(range(42))
    transport = RecordedTransport(records)
    info = client.get_info(transport, 1, [])
    assert info["hclk_hz"] == 180000000 and info["dwt_enabled"]
    observed, comparisons, frequencies = [], [], []
    for row in rows:
        i = row["index"]
        assert row["hclk_hz"] == info["hclk_hz"]
        assert sha_bytes(windows[i]) == row["window_payload_sha256"]
        cycles, features, checksum = client.transmit(transport, windows[i], 32, 1, [])
        assert cycles == row["cycles"] and f"{checksum:08x}" == row["checksum_fnv1a"]
        assert math.isclose(cycles * 1000 / info["hclk_hz"], row["extraction_ms"], rel_tol=0, abs_tol=1e-10)
        assert row["rtol"] == 0.001 and row["atol"] == 0.00001
        close, passed, bins = client.compare(reference[i], features, 0.001, 0.00001)
        assert row["status"] == ("PASS" if passed else "FAIL") and row["exact_dominant_bins"] == bins
        assert math.isclose(float(np.abs(features - reference[i]).max()), row["max_absolute_error"], rel_tol=0, abs_tol=1e-10)
        observed.append(features)
        comparisons.append(close)
        frequencies.append(bins)
    assert transport.position == len(records), "Quedaron tramas sin asociar a ventanas"
    return rows, np.asarray(observed), np.asarray(comparisons), np.asarray(frequencies), dict(
        run=folder.name, results_sha256=sha(folder / "resultados.json"),
        uart_trace_sha256=sha(folder / "uart_trace.jsonl"), raw_records_checked=len(records),
        complete_input_output_replay_verified=True, hclk_hz=info["hclk_hz"])


def sha_bytes(window):
    return hashlib.sha256(np.asarray(window, dtype="<f4").tobytes(order="C")).hexdigest()


def timings(values):
    return dict(mean_ms=float(np.mean(values)), median_ms=float(np.median(values)),
        min_ms=float(np.min(values)), max_ms=float(np.max(values)),
        sample_std_ms=float(np.std(values, ddof=1)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="20261023T202133_701975Z")
    args = parser.parse_args()
    folder = OUT / "Validacion_PC/resultados" / args.run
    target = OUT / "RESULTADOS_FISICOS_O3_42_VENTANAS.json"
    if target.exists():
        raise FileExistsError("No sobrescribir informe fisico anterior")
    assert folder.is_dir() and folder.parent == OUT / "Validacion_PC/resultados"
    assert sha(CLIENT) == EXPECTED_CLIENT_SHA
    bundle = CLIENT.with_name("ventanas_desarrollo.npz")
    assert sha(bundle) == EXPECTED_NPZ_SHA
    spec = importlib.util.spec_from_file_location("original_uart_client", CLIENT)
    client = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(client)
    with np.load(bundle) as data:
        windows, reference = data["windows"], data["features"]
    assert windows.shape == (42, 4096, 2) and reference.shape == (42, 14)
    rows, features, close, bins, trace = replay(folder, windows, reference, client)
    base_rows, base_features, base_close, base_bins, base_trace = replay(CONTROL, windows, reference, client)
    assert base_close.all() and base_bins.all()
    t = np.asarray([r["extraction_ms"] for r in rows])
    bt = np.asarray([r["extraction_ms"] for r in base_rows])
    for r, b in zip(rows, base_rows):
        assert all(r[key] == b[key] for key in ("index", "file_id", "class", "load_hp", "window_payload_sha256"))
    bits = np.asarray(features, dtype="<f4").view("<u4") == np.asarray(base_features, dtype="<f4").view("<u4")
    hop = 2048 / 12000 * 1000
    by_class = {}
    for label in ("N", "IR", "OR"):
        indices = [i for i, r in enumerate(rows) if r["class"] == label]
        by_class[label] = dict(windows=len(indices), timing_o3=timings(t[indices]),
            timing_os=timings(bt[indices]), windows_exceeding_hop=int((t[indices] > hop).sum()))
    local_elf = Path(r"D:\STM32IDE\Prueba_2_CWRU_Extraccion\Release_180MHz_O3\Prueba_2_CWRU_Extraccion.elf")
    report = dict(date="2026-10-23", status="PHYSICAL_42_WINDOW_RUN_AUDITED",
        optimization="-O3", profiling_enabled=False, trace=trace, paired_control_trace=base_trace,
        windows=42, mat_files=len({r["file_id"] for r in rows}),
        loads_hp=sorted({int(r["load_hp"]) for r in rows}), reserved_load_3hp_used=any(r["load_hp"] == "3" for r in rows),
        validation=dict(passed_windows=int(close.all(axis=1).sum()), passed_feature_comparisons=int(close.sum()),
            passed_dominant_bins=int(bins.sum()), float32_features_bit_identical_to_os=int(bits.sum()),
            windows_all_float32_outputs_bit_identical_to_os=int(bits.all(axis=1).sum()),
            rtol=0.001, atol=0.00001, max_relative_error_against_pc=float(
                (np.abs(features-reference)/np.maximum(np.abs(reference), 1e-30)).max()),
            original_client_sha256=EXPECTED_CLIENT_SHA, npz_sha256=EXPECTED_NPZ_SHA),
        timing_o3=timings(t), timing_os=timings(bt), paired_timing=dict(
            mean_reduction_ms=float((bt-t).mean()), reduction_of_mean_percent=float((bt.mean()-t.mean())/bt.mean()*100),
            faster_windows=int((t < bt).sum()), slowest_reduction_ms=float((bt-t).min()),
            largest_reduction_ms=float((bt-t).max())),
        hop_budget=dict(milliseconds=hop, exceeding_windows=int((t>hop).sum()),
            at_or_below_windows=int((t<=hop).sum()), mean_excess_ms=float(t.mean()-hop),
            maximum_excess_ms=float(t.max()-hop)), by_class=by_class,
        local_elf_sha256=sha(local_elf),
        local_elf_matches_prepared_build=sha(local_elf) == "51064b6b6e11aec10cac894fc27d5128cf248bd9ea7396c8baaaa44f0bf43585",
        binary_fingerprint_reported_by_uart=False, firmware_changed_during_audit=False,
        limitations=[
            "One extraction per development window per variant; no confidence interval or worst-case bound is claimed.",
            "Windows are not independent motors or bearings, and 3 HP is reserved and unused.",
            "HAL reports nominal HCLK; no external clock calibration or device binary readback was performed.",
            "Times include extraction only, not UART, acquisition, classifier or K-means.",
            "Entropy profiling was measured on -Os, not -O3; its precise -O3 block share was not measured."
        ])
    assert report["validation"]["passed_windows"] == 42 and report["validation"]["passed_feature_comparisons"] == 588
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
