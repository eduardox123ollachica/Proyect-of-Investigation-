"""Audita tramas CWRUNN existentes, sin COM ni nuevas inferencias.

Por defecto solo informa; --write-report crea un informe nuevo sin sobrescribir.
No cambia referencias, tolerancias, firmware, manifiestos o resultados físicos.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from auditar_validacion_o3 import RecordedTransport, sha

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "despliegue_stm32/dnn_int8_180mhz_2026-10-23"
CLIENT = OUT / "Validacion_PC"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def timing(rows, hclk):
    result = {"executions": len(rows)}
    for key in ("invoke_cycles", "call_cycles"):
        cycles = np.asarray([r["observed"][key] for r in rows], dtype=np.float64)
        us = cycles * 1e6 / hclk
        result[key.removesuffix("_cycles")] = dict(
            mean_cycles=float(cycles.mean()), min_cycles=int(cycles.min()), max_cycles=int(cycles.max()),
            mean_us=float(us.mean()), median_us=float(np.median(us)),
            p95_us=float(np.percentile(us, 95, method="linear")), min_us=float(us.min()),
            max_us=float(us.max()), sample_std_us=float(np.std(us, ddof=1)) if len(us) > 1 else None)
    return result


def audit(run_name):
    require(Path(run_name).name == run_name and run_name not in (".", ".."), "Nombre de ejecución inválido")
    folder = CLIENT / "resultados" / run_name
    manifest = json.loads((folder / "MANIFEST_SHA256.json").read_text(encoding="utf-8"))
    require(set(manifest) == {"resultados.json", "uart_trace.jsonl"}, "Manifiesto físico inesperado")
    for name, digest in manifest.items():
        require(sha(folder / name) == digest, "Archivo físico modificado: " + name)
    run = json.loads((folder / "resultados.json").read_text(encoding="utf-8"))
    require(run["status"] == "PASS", "La ejecución no pasó: conservar incidencia")
    require(run["mode"] in ("single", "all"), "Modo desconocido")
    require(not any(run[k] for k in ("extraction_executed", "scaling_executed", "detector_executed", "training_executed")),
            "Alcance distinto de la DNN aislada")
    for field, name in (("client_sha256", "validar_dnn.py"), ("reference_sha256", "REFERENCIA_DNN.json")):
        require(run[field] == sha(CLIENT / name), "Recurso diferente del usado en placa: " + name)
    sealed = json.loads((OUT / "MANIFEST_SHA256.json").read_text(encoding="utf-8"))
    digest, name = (OUT / "MANIFEST_SHA256.sha256").read_text(encoding="ascii").strip().split()
    require(name == "MANIFEST_SHA256.json" and sha(OUT / name) == digest, "Sello de entrega diferente")
    for p in (CLIENT / "validar_dnn.py", CLIENT / "REFERENCIA_DNN.json", OUT / "CONTRATO_PREVIO_DNN.md"):
        require(sha(p) == sealed[p.relative_to(ROOT).as_posix()], "Recurso difiere del contrato/entrega previa")
    spec = importlib.util.spec_from_file_location("frozen_dnn_client", CLIENT / "validar_dnn.py")
    client = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(client)
    ref = client.bundle()
    require(ref["policy"] == dict(max_output_code_absolute_difference=1, probability_equivalent=1/256,
        class_must_match_primary=True, repeated_outputs_must_be_exact=True,
        primary_must_not_change_after_observing_hardware=True), "Política diferente del contrato")
    index = None
    if run["mode"] == "all":
        cases = ref["cases"]
        require(len(cases) == 81 and sum(c["group"] == "development" for c in cases) == 42, "Suite diferente")
    else:
        require(bool(run["results"]), "Ejecución sin resultados")
        case = next((c for c in ref["cases"] if c["name"] == run["results"][0]["name"]), None)
        require(case is not None and case["group"] == "development", "Caso individual inesperado")
        index = case["window_index"]
        cases = [case]
    repetitions = run["repetitions"]
    require(repetitions == 3 and run["cases"] == len(cases), "Número de casos o repeticiones distinto")
    total = len(cases) * repetitions
    require(len(run["results"]) == run["completed_executions"] == total, "Resultados incompletos")
    records = [json.loads(line) for line in (folder / "uart_trace.jsonl").read_text(encoding="utf-8").splitlines()]
    transport = RecordedTransport(records)
    before = client.info(transport, ref, 1., [])
    initialization = client.initialize(transport, 1., [])
    info = client.info(transport, ref, 1., [])
    require(before == run["info_before_init"] and initialization == run["initialization"]
            and info == run["firmware_info"], "INFO/INIT difieren de las tramas")
    require(info["initialization_status"] == 0 and info["arena_used"] == initialization["arena_used"],
            "Estado inicial o arena incoherentes")
    require(before["initialization_status"] == (0 if initialization["reused"] else -100),
            "Estado previo incompatible con reutilización")
    require(before["arena_used"] == (info["arena_used"] if initialization["reused"] else 0),
            "Arena previa incoherente")
    differences, output_deltas = [], []
    row_index = 0
    for case in cases:
        previous = None
        for repetition in range(1, repetitions + 1):
            row = run["results"][row_index]
            row_index += 1
            require(all(row[k] == case[k] for k in ("name", "group", "input_hex")), "Entrada diferente de la referencia")
            require(row["repetition"] == repetition and row["status"] == "PASS", "Orden o estado incorrectos")
            require(row["primary_reference"] == case["primary"] and row["secondary_reference"] == case["secondary_reference"],
                    "Referencia guardada diferente")
            line = client.exchange(transport, client.command(case), "CWRUNN_RESULT,", 1., [])
            observed = client.parse_result(line, case)
            comparison = client.compare(observed, case)
            stable = previous is None or previous == observed["output_int8"]
            require(observed == row["observed"] and comparison == row["comparison"] and comparison["passed"],
                    "Discrepancia al auditar la respuesta")
            require(stable and row["repeated_output_stable"] == stable, "Inestabilidad entre repeticiones")
            require(observed["arena_used"] == info["arena_used"], "La arena reportada varió")
            delta = np.abs(np.asarray(observed["output_int8"]) - np.asarray(case["primary"]["output_int8"]))
            output_deltas.extend(delta.tolist())
            if np.any(delta):
                differences.append(dict(name=case["name"], group=case["group"], repetition=repetition,
                    input_hex=case["input_hex"], expected_primary_int8=case["primary"]["output_int8"],
                    expected_secondary_int8=case["secondary_reference"]["output_int8"], observed_int8=observed["output_int8"],
                    absolute_differences=delta.tolist(), predicted_class=observed["predicted_class"],
                    class_match=comparison["class_match"], max_probability_difference=float(delta.max()/256)))
            previous = observed["output_int8"]
    negatives = client.negatives(cases[0]) if run["mode"] == "all" else []
    require(len(run["negative_tests"]) == len(negatives), "Pruebas negativas incompletas")
    for saved, (command, reason) in zip(run["negative_tests"], negatives):
        line = client.exchange(transport, command, "CWRUNN_ERROR,", 1., [], allow_error=True)
        fields = line.split(",")
        require(len(fields) == 4 and fields[1] == reason and int(fields[2]) < 0, "Rechazo incorrecto")
        bytes.fromhex(fields[3])
        require(saved == dict(command=command, reason=reason, response=line), "Trama negativa diferente")
    require(transport.position == len(records), "Tramas sin asociar a casos")
    summaries = {}
    for label in ("development", "synthetic", "all"):
        rows = [r for r in run["results"] if label == "all" or r["group"] == label]
        if not rows:
            continue
        nonexact_names = {r["name"] for r in rows if not r["comparison"]["output_bit_exact"]}
        summaries[label] = dict(cases=len({r["name"] for r in rows}), compatible_executions=len(rows),
            exact_output_cases=len({r["name"] for r in rows}) - len(nonexact_names),
            exact_output_executions=sum(r["comparison"]["output_bit_exact"] for r in rows),
            matching_class_executions=sum(r["comparison"]["class_match"] for r in rows),
            nonexact_case_names=sorted(nonexact_names), timing=timing(rows, info["hclk_hz"]))
    built = json.loads((OUT / "COMPILACION_DNN.json").read_text(encoding="utf-8"))
    build = built["configurations"]["Release_180MHz_DNN"]
    local = Path(r"D:\STM32IDE\Prueba_2_CWRU_Extraccion\Release_180MHz_DNN\Prueba_2_CWRU_Extraccion.elf")
    local_hash = sha(local) if local.is_file() else None
    report = dict(status="PHYSICAL_DNN_SUITE_AUDITED" if index is None else "FIRST_DNN_CASE_AUDITED",
        compatibility_contract_passed=True, all_output_codes_bit_exact=not bool(differences),
        audited_utc=datetime.now(timezone.utc).isoformat(), run=run_name, mode=run["mode"], development_index=index,
        positive_cases=len(cases), repetitions_per_case=repetitions, passed_executions=total,
        output_codes_compared=len(output_deltas), output_codes_bit_exact=sum(d == 0 for d in output_deltas),
        maximum_code_difference=max(output_deltas), negative_cases_passed=len(negatives),
        all_repeated_outputs_stable=True, summaries=summaries, nonexact_outputs=differences,
        initialization=initialization, firmware_info=info, raw_records_verified=len(records),
        recorded_input_output_replay_verified=True, results_sha256=sha(folder / "resultados.json"),
        uart_trace_sha256=sha(folder / "uart_trace.jsonl"), run_manifest_sha256=sha(folder / "MANIFEST_SHA256.json"),
        client_sha256=sha(CLIENT / "validar_dnn.py"), reference_sha256=sha(CLIENT / "REFERENCIA_DNN.json"),
        contract_sha256=sha(OUT / "CONTRATO_PREVIO_DNN.md"), auditor_sha256=sha(Path(__file__)),
        preparation_manifest_sha256=digest, prepared_elf_sha256=build["elf_sha256"], local_elf_sha256=local_hash,
        local_elf_matches_prepared_build=local_hash == build["elf_sha256"], device_binary_read=False,
        prepared_flash_bytes=build["flash_load_bytes"],
        prepared_static_ram_with_linker_reservations_bytes=build["ram_static_with_linker_reservations_bytes"],
        serial_port_opened_during_audit=False, hardware_changed_during_audit=False, physical_experiment_repeated=False,
        predictive_metrics_computed=False, full_pipeline_validated=False,
        limitations=[
            "Only the fixed historical V1 model and tested INT8 inputs; not predictive validation or future V2 fits.",
            "A compatible one-code discrepancy is not bit-exact equality; the internal operator causing it was not isolated.",
            "Reported arena usage is not a proved minimum capacity or total RAM. Reserved capacity remains 16384 bytes.",
            "Invoke/call exclude initialization, extraction, scaling, detector, parser and UART; instrumentation is not subtracted.",
            "HAL reports nominal HCLK. Repetitions are not independent acquisitions and do not establish WCET.",
            "A matching local ELF and declared model hash do not constitute a complete readback of the flashed device.",
            "If initialization is reused, zero cycles do not represent a new initialization time."])
    return report, OUT / ("RESULTADOS_FISICOS_DNN_COMPLETA.json" if index is None else f"RESULTADOS_FISICOS_DNN_INDICE_{index}.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()
    report, target = audit(args.run)
    if args.write_report:
        with target.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
