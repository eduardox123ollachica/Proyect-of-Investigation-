"""Audita resultados fisicos guardados de cache frente al control O3.

Reutiliza el auditor de trazas existente sin modificarlo. No abre el puerto
serie, no repite ensayos ni altera firmware, vectores o controles anteriores.
"""
from pathlib import Path
import argparse
import importlib.util
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "despliegue_stm32/cache_potencias_180mhz_2026-10-23"
CONTROL = ROOT / "despliegue_stm32/prueba_o3_180mhz_2026-10-23/Validacion_PC/resultados/20261023T202133_701975Z"


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="20261023T211112_480689Z")
    args = parser.parse_args()
    folder = OUT / "Validacion_PC/resultados" / args.run
    target = OUT / "RESULTADOS_FISICOS_CACHE_42_VENTANAS.json"
    if target.exists():
        raise FileExistsError("No sobrescribir evidencia existente")
    assert folder.is_dir() and folder.parent == OUT / "Validacion_PC/resultados"
    auditor = import_file("trace_auditor", Path(__file__).with_name("auditar_validacion_o3.py"))
    client_path = OUT / "Validacion_PC/validar_extraccion.py"
    bundle_path = OUT / "Validacion_PC/ventanas_desarrollo.npz"
    assert auditor.sha(client_path) == auditor.EXPECTED_CLIENT_SHA
    assert auditor.sha(bundle_path) == auditor.EXPECTED_NPZ_SHA
    client = import_file("original_client", client_path)
    with np.load(bundle_path) as bundle:
        windows, reference = bundle["windows"], bundle["features"]
    rows, observed, close, bins, trace = auditor.replay(folder, windows, reference, client)
    base_rows, base_observed, base_close, base_bins, base_trace = auditor.replay(CONTROL, windows, reference, client)
    assert base_close.all() and base_bins.all()
    for r, b in zip(rows, base_rows):
        assert all(r[key] == b[key] for key in ("index", "file_id", "class", "load_hp", "window_payload_sha256"))
    bits = np.asarray(observed, dtype="<f4").view("<u4") == np.asarray(base_observed, dtype="<f4").view("<u4")
    t = np.asarray([r["extraction_ms"] for r in rows])
    bt = np.asarray([r["extraction_ms"] for r in base_rows])
    hop = 2048 / 12000 * 1000
    by_class = {}
    for label in ("N", "IR", "OR"):
        ix = [i for i, row in enumerate(rows) if row["class"] == label]
        by_class[label] = dict(windows=len(ix), timing_cache=auditor.timings(t[ix]),
            timing_o3_control=auditor.timings(bt[ix]),
            minimum_observed_hop_margin_ms=float(hop-t[ix].max()))
    local_elf = Path(r"D:\STM32IDE\Prueba_2_CWRU_Extraccion\Release_180MHz_O3_Cache\Prueba_2_CWRU_Extraccion.elf")
    report = dict(date="2026-10-23", status="PHYSICAL_42_WINDOW_CACHE_RUN_AUDITED",
        optimization="-O3", cache_enabled=True, profiling_enabled=False,
        trace=trace, paired_o3_control_trace=base_trace, windows=len(rows),
        mat_files=len({r["file_id"] for r in rows}), loads_hp=sorted({int(r["load_hp"]) for r in rows}),
        reserved_load_3hp_used=any(int(r["load_hp"]) == 3 for r in rows),
        validation=dict(passed_windows=int(close.all(axis=1).sum()), passed_feature_comparisons=int(close.sum()),
            passed_dominant_bins=int(bins.sum()), float32_features_bit_identical_to_o3=int(bits.sum()),
            windows_all_float32_outputs_bit_identical_to_o3=int(bits.all(axis=1).sum()),
            rtol=0.001, atol=0.00001,
            max_relative_error_against_pc=float((np.abs(observed-reference)/np.maximum(np.abs(reference), 1e-30)).max()),
            original_client_sha256=auditor.EXPECTED_CLIENT_SHA, npz_sha256=auditor.EXPECTED_NPZ_SHA),
        timing_cache=auditor.timings(t), timing_o3_control=auditor.timings(bt),
        paired_timing=dict(mean_reduction_ms=float((bt-t).mean()),
            reduction_of_mean_percent=float((bt.mean()-t.mean())/bt.mean()*100),
            faster_windows=int((t<bt).sum()), minimum_reduction_ms=float((bt-t).min()),
            maximum_reduction_ms=float((bt-t).max())),
        hop_budget=dict(milliseconds=hop, at_or_below_windows=int((t<=hop).sum()),
            exceeding_windows=int((t>hop).sum()), mean_observed_margin_ms=float(hop-t.mean()),
            minimum_observed_margin_ms=float(hop-t.max()), maximum_observed_margin_ms=float(hop-t.min())),
        by_class=by_class, local_elf_sha256=auditor.sha(local_elf),
        local_elf_matches_prepared_build=auditor.sha(local_elf) == "754ede39f1a7f5dcf9c8c94f5b06cca6c2ea7ed6fcd488998055136798e0b40e",
        binary_fingerprint_reported_by_uart=False, firmware_changed_during_audit=False,
        limitations=[
            "One extraction per development window per variant; these observations do not establish a worst-case timing bound.",
            "The windows come from 21 MAT files, not independent motors or bearings. Reserved 3 HP is not used.",
            "HAL reports nominal HCLK; no external clock calibration or device binary readback was performed.",
            "Only extraction is timed, excluding UART transport, acquisition, classifier, K-means and scheduling overhead.",
            "Observed hop margin is not a certified budget for the remaining stages or proof of end-to-end real-time operation.",
            "Bit identity is established on the 42 tested windows, not all possible inputs."
        ])
    assert report["validation"]["passed_windows"] == 42 and report["validation"]["passed_feature_comparisons"] == 588
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
