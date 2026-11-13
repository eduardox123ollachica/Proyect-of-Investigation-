"""Audita trazas existentes EntD/EntM; no conecta hardware ni repite ensayos."""
from pathlib import Path
import argparse
import importlib.util
import json
import math
import numpy as np
import auditar_validacion_o3 as audit

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "despliegue_stm32/mejoras_entropia_180mhz_2026-10-23"
CONTROL = ROOT / "despliegue_stm32/cache_potencias_180mhz_2026-10-23/Validacion_PC/resultados/20261023T211112_480689Z"


def read_run(folder, windows, reference, client):
    rows = json.loads((folder / "resultados.json").read_text(encoding="utf-8"))
    indices = [r["index"] for r in rows]
    assert indices == [0] or indices == list(range(42)), "Ejecucion incompleta/inesperada"
    records = [json.loads(line) for line in (folder / "uart_trace.jsonl").read_text(encoding="utf-8").splitlines()]
    transport = audit.RecordedTransport(records)
    info = client.get_info(transport, 1, [])
    assert info["hclk_hz"] == 180000000
    features, checks, bins = [], [], []
    for row in rows:
        i = row["index"]
        assert audit.sha_bytes(windows[i]) == row["window_payload_sha256"]
        cycles, observed, checksum = client.transmit(transport, windows[i], 32, 1, [])
        assert cycles == row["cycles"] and f"{checksum:08x}" == row["checksum_fnv1a"]
        assert row["hclk_hz"] == info["hclk_hz"]
        assert math.isclose(cycles*1000/info["hclk_hz"], row["extraction_ms"], rel_tol=0, abs_tol=1e-10)
        assert row["rtol"] == .001 and row["atol"] == .00001
        close, passed, freq = client.compare(reference[i], observed, .001, .00001)
        assert row["status"] == ("PASS" if passed else "FAIL") and row["exact_dominant_bins"] == freq
        assert math.isclose(float(np.abs(observed-reference[i]).max()), row["max_absolute_error"], rel_tol=0, abs_tol=1e-10)
        features.append(observed); checks.append(close); bins.append(freq)
    assert transport.position == len(records)
    trace = dict(run=folder.name, results_sha256=audit.sha(folder / "resultados.json"),
        uart_trace_sha256=audit.sha(folder / "uart_trace.jsonl"), raw_records_verified=len(records),
        full_input_output_replay_verified=True, hclk_hz=info["hclk_hz"])
    return rows, np.asarray(features), np.asarray(checks), np.asarray(bins), trace


def timing(values):
    return dict(mean_ms=float(values.mean()), median_ms=float(np.median(values)),
        min_ms=float(values.min()), max_ms=float(values.max()),
        sample_std_ms=float(values.std(ddof=1)) if len(values)>1 else None)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", choices=("double", "mixta"), required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--double-run", help="Control EntD opcional al analizar EntM")
    args = p.parse_args()
    label, cfg = ("Double", "Release_180MHz_EntD") if args.variant == "double" else ("Mixta", "Release_180MHz_EntM")
    assert Path(args.run).name == args.run
    folder = OUT / ("Validacion_"+label) / "resultados" / args.run
    client_path = folder.parents[1] / "validar_extraccion.py"
    assert audit.sha(client_path) == audit.EXPECTED_CLIENT_SHA
    bundle_path = client_path.with_name("ventanas_desarrollo.npz")
    assert audit.sha(bundle_path) == audit.EXPECTED_NPZ_SHA
    spec = importlib.util.spec_from_file_location("original_client", client_path)
    client = importlib.util.module_from_spec(spec); spec.loader.exec_module(client)
    with np.load(bundle_path) as bundle:
        windows, reference = bundle["windows"], bundle["features"]
    rows, features, close, bins, trace = read_run(folder, windows, reference, client)
    base_rows, base_features, base_close, base_bins, base_trace = read_run(CONTROL, windows, reference, client)
    assert base_close.all() and base_bins.all()
    indices = [r["index"] for r in rows]
    base_features = base_features[indices]
    for row, i in zip(rows, indices):
        assert all(row[k] == base_rows[i][k] for k in ("window_payload_sha256", "file_id", "class", "load_hp"))
    bits = features.astype("<f4").view("<u4") == base_features.astype("<f4").view("<u4")
    t = np.asarray([r["extraction_ms"] for r in rows])
    bt = np.asarray([base_rows[i]["extraction_ms"] for i in indices])
    entropy_columns = [6, 13]
    other_columns = [i for i in range(14) if i not in entropy_columns]
    hop = 2048/12000*1000
    n = len(rows)
    target = OUT / ("RESULTADOS_FISICOS_"+label.upper()+ ("_VENTANA_0.json" if n == 1 else "_42_VENTANAS.json"))
    if target.exists():
        raise FileExistsError("No sobrescribir evidencia")
    built = json.loads((OUT / "COMPILACION_ENTROPIA.json").read_text(encoding="utf-8"))
    local_elf = Path(r"D:\STM32IDE") / "Prueba_2_CWRU_Extraccion" / cfg / "Prueba_2_CWRU_Extraccion.elf"
    report = dict(date="2026-10-23", status="FIRST_WINDOW_AUDITED" if n == 1 else "PHYSICAL_42_WINDOWS_AUDITED",
        variant=label, configuration=cfg, windows=n, indices=indices, mat_files=len({r["file_id"] for r in rows}),
        loads_hp=sorted({int(r["load_hp"]) for r in rows}), reserved_load_3hp_used=any(int(r["load_hp"])==3 for r in rows),
        trace=trace, paired_cache_control_trace=base_trace,
        validation=dict(passed_windows=int(close.all(axis=1).sum()), passed_features=int(close.sum()),
            correct_dominant_bins=int(bins.sum()), bit_identical_features_to_cache=int(bits.sum()),
            bit_identical_nonentropy_features_to_cache=int(bits[:,other_columns].sum()),
            bit_identical_entropy_features_to_cache=int(bits[:,entropy_columns].sum()),
            max_entropy_absolute_difference_to_cache=float(np.abs(features[:,entropy_columns]-base_features[:,entropy_columns]).max()),
            max_relative_error_against_pc=float((np.abs(features-reference[indices])/np.maximum(np.abs(reference[indices]),1e-30)).max()),
            rtol=.001, atol=.00001),
        timing=timing(t), paired_cache_timing=timing(bt), reduction_ms=float((bt-t).mean()),
        reduction_percent=float((bt.mean()-t.mean())/bt.mean()*100), faster_windows=int((t<bt).sum()),
        hop_budget=dict(milliseconds=hop, at_or_below_windows=int((t<=hop).sum()),
            minimum_observed_margin_ms=float(hop-t.max()), mean_observed_margin_ms=float(hop-t.mean())),
        local_elf_sha256=audit.sha(local_elf),
        local_elf_matches_prepared_build=audit.sha(local_elf)==built["configurations"][cfg]["elf_sha256"],
        binary_fingerprint_reported_by_uart=False, firmware_changed_during_audit=False,
        limitations=["Only the tested development windows, not all inputs or independent motors.",
            "One extraction per window/variant is not a worst-case bound or confidence interval.",
            "HAL reports nominal clock; UART does not fingerprint the device binary.",
            "Times cover extraction only, not UART, acquisition, resampling, classifier or detector.",
            "Feature tolerance or bit identity does not replace downstream quantization/prediction validation."])
    if args.double_run:
        assert args.variant == "mixta" and Path(args.double_run).name == args.double_run
        drows, dfeatures, dclose, dbins, dtrace = read_run(OUT / "Validacion_Double/resultados" / args.double_run, windows, reference, client)
        lookup = {r["index"]: i for i,r in enumerate(drows)}
        assert all(i in lookup for i in indices) and dclose.all() and dbins.all()
        dt = np.asarray([drows[lookup[i]]["extraction_ms"] for i in indices])
        report["paired_double_comparison"] = dict(trace=dtrace, timing=timing(dt),
            mean_reduction_ms=float((dt-t).mean()), reduction_percent=float((dt.mean()-t.mean())/dt.mean()*100))
    assert report["validation"]["passed_windows"] == n and report["validation"]["passed_features"] == 14*n
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
