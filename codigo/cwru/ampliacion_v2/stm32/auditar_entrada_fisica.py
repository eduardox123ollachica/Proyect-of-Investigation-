"""Audita respuestas CWRUIN existentes sin abrir COM ni repetir cálculo en placa."""
from __future__ import annotations
import argparse
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from auditar_validacion_o3 import RecordedTransport, sha

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "despliegue_stm32/escalado_cuantizacion_180mhz_2026-10-23"
CLIENT = OUT / "Validacion_PC"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    require(Path(args.run).name == args.run, "Nombre de ejecución inválido")
    folder = CLIENT / "resultados" / args.run
    run = json.loads((folder / "resultados.json").read_text(encoding="utf-8"))
    mode = run["mode"]
    require(mode in ("all", "single_development_index"), "Modo de ejecución inesperado")
    index = None
    if mode == "single_development_index":
        require(bool(run["results"]), "Ejecución vacía")
        name = run["results"][0]["name"]
        require(name.startswith("entm_") and name[5:].isdigit(), "Caso individual inesperado")
        index = int(name[5:])
    target = OUT / ("RESULTADOS_FISICOS_ENTRADA_COMPLETA.json" if index is None
                    else f"RESULTADOS_FISICOS_ENTRADA_INDICE_{index}.json")
    require(not target.exists(), "No sobrescribir informe físico anterior")
    manifest = json.loads((folder / "MANIFEST_SHA256.json").read_text(encoding="utf-8"))
    require(set(manifest) == {"resultados.json", "uart_trace.jsonl"}, "Manifiesto de ejecución inesperado")
    for name, digest in manifest.items():
        require(sha(folder / name) == digest, "Archivo físico modificado: " + name)
    require(run["status"] == "PASS", "Ejecución no aprobada; conservar para diagnóstico")
    require(not run["physical_extraction_repeated"] and not run["classifier_executed"] and not run["training_executed"],
            "Alcance diferente del bloque de entrada")
    for field, name in (("client_sha256", "validar_entrada.py"), ("cases_sha256", "casos_validacion.json"),
                        ("parameters_sha256", "PARAMETROS_PC.json")):
        require(run[field] == sha(CLIENT / name), "Recurso del ensayo diferente: " + name)
    spec = importlib.util.spec_from_file_location("frozen_input_client", CLIENT / "validar_entrada.py")
    client = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(client)
    params, cases = client.bundle()
    selected = cases if index is None else [c for c in cases if c.get("group") == "entm" and c.get("index") == index]
    repetitions = run["repetitions"]
    require(1 <= repetitions <= 100 and run["selected_cases"] == len(selected) and bool(selected), "Selección inválida")
    total = len(selected) * repetitions
    require(len(run["results"]) == run["completed_executions"] == total, "Resultados incompletos")
    records = [json.loads(line) for line in (folder / "uart_trace.jsonl").read_text(encoding="utf-8").splitlines()]
    transport = RecordedTransport(records)
    info = client.info(transport, params, 1., [])
    require(info == run["firmware_info"], "INFO guardado diferente de la trama")
    results, row_index = [], 0
    for case in selected:
        expected = client.expected(case, params)
        for repetition in range(1, repetitions + 1):
            row = run["results"][row_index]; row_index += 1
            require(all(row[k] == case[k] for k in ("name", "group", "mode", "input_bits")), "Entrada diferente")
            require(row["repetition"] == repetition and row["status"] == "PASS" and row["reference"] == expected,
                    "Referencia, repetición o estado incorrectos")
            line = client.exchange(transport, client.command(case), "CWRUIN_RESULT,", 1., [])
            observed = client.parse_result(line, case)
            require(observed == row["observed"] and client.compare(observed, expected), "Respuesta física diferente")
            results.append(row)
    negatives = client.negatives(params) if index is None else []
    require(len(run["negative_tests"]) == len(negatives), "Pruebas negativas incompletas")
    for saved, (command, reason) in zip(run["negative_tests"], negatives):
        line = client.exchange(transport, command, "CWRUIN_ERROR,", 1., [], allow_error=True)
        require(line.split(",")[1] == reason and saved == dict(command=command, expected_reason=reason, response=line),
                "Rechazo de trama inválida incorrecto")
    require(transport.position == len(records), "Tramas sin asociar a casos")
    timing = {}
    for label in client.MODES:
        values = np.asarray([r["observed"]["cycles"] for r in results if r["mode"] == label], dtype=float)
        if not len(values):
            continue
        source_timing = run["timing_by_mode_not_full_pipeline"][label]
        require(source_timing == dict(count=len(values), mean_cycles=float(values.mean()),
                                     min_cycles=int(values.min()), max_cycles=int(values.max())), "Resumen temporal inconsistente")
        us = values * 1e6 / info["hclk_hz"]
        timing[label] = dict(executions=len(values), mean_cycles=float(values.mean()),
            minimum_cycles=int(values.min()), maximum_cycles=int(values.max()),
            mean_us=float(us.mean()), median_us=float(np.median(us)), min_us=float(us.min()), max_us=float(us.max()))
    built = json.loads((OUT / "COMPILACION_ENTRADA.json").read_text(encoding="utf-8"))
    expected_elf = built["configurations"]["Release_180MHz_Input"]["elf_sha256"]
    local_elf = Path(r"D:\STM32IDE\Prueba_2_CWRU_Extraccion\Release_180MHz_Input\Prueba_2_CWRU_Extraccion.elf")
    local_hash = sha(local_elf) if local_elf.is_file() else None
    report = dict(status="PHYSICAL_INPUT_SUITE_AUDITED" if index is None else "FIRST_INPUT_CASE_AUDITED",
        audited_utc=datetime.now(timezone.utc).isoformat(), run=args.run, mode=mode, development_index=index,
        positive_cases=len(selected), repetitions_per_case=repetitions, passed_executions=total,
        standardized_values_bit_exact=14*total, affine_values_bit_exact=14*total, int8_codes_equal=14*total,
        negative_cases_passed=len(negatives), firmware_info=info, timing_by_mode=timing,
        input_output_records_verified=len(records), full_recorded_input_output_replay_verified=True,
        results_sha256=sha(folder / "resultados.json"), trace_sha256=sha(folder / "uart_trace.jsonl"),
        client_sha256=sha(CLIENT / "validar_entrada.py"), auditor_sha256=sha(Path(__file__)),
        expected_elf_sha256=expected_elf, local_elf_sha256=local_hash,
        local_elf_matches_prepared_build=local_hash == expected_elf, device_binary_fingerprint_available=False,
        serial_port_opened_during_audit=False, hardware_changed_during_audit=False, physical_experiment_repeated=False,
        classifier_validated=False, full_pipeline_validated=False,
        limitations=["Only the tested inputs and historical scaler/quantization, not all possible features or V2 fits.",
                     "DWT timing covers the input numerical function only, before UART formatting; overhead is not subtracted.",
                     "Repeated inputs are not independent windows, motors or acquisitions; no worst-case bound is established.",
                     "HAL reports nominal HCLK; source parameter hash is not a fingerprint of the complete flashed binary.",
                     "Standalone block timing cannot establish integrated pipeline timing or classifier outputs."])
    with target.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2); stream.write("\n")
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
