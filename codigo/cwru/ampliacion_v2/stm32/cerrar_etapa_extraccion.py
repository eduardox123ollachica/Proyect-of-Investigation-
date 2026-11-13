"""Sella referencias locales al candidato EntM sin modificar firmware ni evidencia.

Sin argumentos crea un cierre nuevo; --verify comprueba el cierre existente.
No conecta COM, no compila, no entrena y no escribe en D.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
STAGE = ROOT / "despliegue_stm32/mejoras_entropia_180mhz_2026-10-23"
NAME = "Prueba_2_CWRU_Extraccion"
PROJECT = STAGE / "copia_aislada" / NAME
ACTIVE = Path(r"D:\STM32IDE") / NAME
OUT = ROOT / "despliegue_stm32/cierre_extraccion_2026-10-23"
MANIFEST = OUT / "CIERRE.json"
SEAL = OUT / "CIERRE.sha256"
CONFIG = "Release_180MHz_EntM"
ELF_SHA = "fe3372711fbeddfdf08601cea8b501f5b3e38e01a1b7abdaea33e0eae6767df1"
RUN = "20261023T221815_802310Z"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def stored_path(relative: str) -> Path:
    path = (ROOT / relative).resolve()
    require(path.is_relative_to(ROOT), "Referencia fuera del proyecto")
    return path


def xml_content(path: Path, ignore_env_hash: bool = False) -> tuple:
    def content(element: ET.Element) -> tuple:
        attributes = {key: value for key, value in element.attrib.items()
                      if not (ignore_env_hash and key == "env-hash")}
        return (element.tag, sorted(attributes.items()), (element.text or "").strip(),
                [content(child) for child in element])
    return content(ET.parse(path).getroot())


def verify() -> None:
    expected, filename = SEAL.read_text(encoding="ascii").strip().split()
    require(filename == MANIFEST.name, "Nombre inesperado en el sello")
    require(sha(MANIFEST) == expected, "El manifiesto no coincide con su sello")
    data = load(MANIFEST)
    for row in data["referenced_files"]:
        path = stored_path(row["path"])
        require(path.is_file(), "Falta archivo: " + row["path"])
        require(path.stat().st_size == row["bytes"], "Tamaño distinto: " + row["path"])
        require(sha(path) == row["sha256"], "Hash distinto: " + row["path"])
    print(json.dumps({"status": "PASS", "referenced_files_verified": len(data["referenced_files"]),
                      "serial_port_opened": False, "experiments_repeated": False,
                      "active_D_rechecked": False}, indent=2))


def create() -> None:
    require(not MANIFEST.exists() and not SEAL.exists(), "El cierre ya existe; usar --verify")
    prep = load(STAGE / "PREPARACION.json")
    build = load(STAGE / "COMPILACION_ENTROPIA.json")
    applied = load(STAGE / "APLICACION_EN_D.json")
    physical_path = STAGE / "RESULTADOS_FISICOS_MIXTA_42_VENTANAS.json"
    physical = load(physical_path)
    require(build["status"] == applied["status"] == "PASS", "Preparación no aprobada")
    require(physical["configuration"] == CONFIG and physical["trace"]["run"] == RUN,
            "Ejecución física diferente")
    require(physical["validation"]["passed_windows"] == physical["windows"] == 42,
            "Validación física incompleta")
    require(physical["validation"]["passed_features"] == 588 and
            physical["validation"]["correct_dominant_bins"] == 84 and
            physical["validation"]["bit_identical_features_to_cache"] == 587,
            "Resultado numérico inesperado")
    require(build["configurations"][CONFIG]["elf_sha256"] == ELF_SHA ==
            physical["local_elf_sha256"], "Identificación ELF diferente")

    paths: set[Path] = set()
    expected_source = {row["path"]: row["sha256"] for row in prep["source_inventory_before"]}
    expected_source.update(build["overlay_sha256"])
    require(applied["installed_sha256"] == build["overlay_sha256"], "Instalación distinta")
    exact_matches = 0
    ide_differences = []
    for relative, expected in sorted(expected_source.items()):
        preserved, active = PROJECT / relative, ACTIVE / relative
        preserved_sha, active_sha = sha(preserved), sha(active)
        if relative == ".settings/language.settings.xml":
            # CubeIDE generó los proveedores de las dos configuraciones nuevas.
            # La única diferencia C/D permitida es la huella del entorno indexador.
            require(xml_content(preserved, True) == xml_content(active, True),
                    "Proveedores de lenguaje distintos, además de env-hash")
            ide_differences.append(dict(path=relative, original_sha256=expected,
                                        preserved_sha256=preserved_sha, active_sha256=active_sha,
                                        reason="generated_language_providers; C/D differ only in env-hash"))
        else:
            require(preserved_sha == expected, "Copia conservada distinta: " + relative)
            if active_sha != expected:
                require(relative in build["overlay_sha256"] and relative.endswith(".launch"),
                        "D cambió desde la preparación: " + relative)
                require(xml_content(preserved) == xml_content(active),
                        "El lanzamiento cambió atributos: " + relative)
                ide_differences.append(dict(path=relative, preserved_sha256=preserved_sha,
                                            active_sha256=active_sha,
                                            reason="XML serialization only; all attributes identical"))
            else:
                exact_matches += 1
        paths.add(preserved)

    build_dir = PROJECT / CONFIG
    for relative in [NAME + ".elf", NAME + ".map", "objects.list", "Src/cwru_feature_extraction.su"]:
        preserved, active = build_dir / relative, ACTIVE / CONFIG / relative
        require(sha(preserved) == sha(active), "Build D diferente: " + relative)
        paths.add(preserved)
    require(sha(build_dir / (NAME + ".elf")) == ELF_SHA, "ELF conservado diferente")
    units = (build_dir / "objects.list").read_text(encoding="utf-8").splitlines()
    require(len(units) == 18 and not any("tinyml_inference" in unit for unit in units),
            "Unidades del extractor inesperadas")
    old_adapter = PROJECT / "Src/tinyml_inference.cpp"
    require("input->bytes != 21" in old_adapter.read_text(encoding="utf-8"),
            "El adaptador histórico requiere nueva revisión")

    run_dir = STAGE / "Validacion_Mixta/resultados" / RUN
    require(sha(run_dir / "resultados.json") == physical["trace"]["results_sha256"],
            "Resultados físicos modificados")
    require(sha(run_dir / "uart_trace.jsonl") == physical["trace"]["uart_trace_sha256"],
            "Traza física modificada")
    require(len(list(run_dir.glob("ventana_*.csv"))) == 42, "CSV de ventanas incompletos")
    paths.update(path for path in run_dir.iterdir() if path.is_file())
    paths.update(path for path in (STAGE / "Validacion_Mixta").iterdir() if path.is_file())
    paths.update(STAGE / name for name in [
        "PREPARACION.json", "COMPILACION_ENTROPIA.json", "APLICACION_EN_D.json",
        "PRUEBAS_NUMERICAS_HOST.json", "CUBEIDE_ENTROPIA.log",
        "RESULTADOS_FISICOS_MIXTA_42_VENTANAS.json", "RESULTADOS_VALIDACION_MIXTA.md",
        "RESULTADOS_FISICOS_DOUBLE_42_VENTANAS.json", "RESULTADOS_VALIDACION_DOUBLE.md"])
    paths.update([Path(__file__).resolve(), OUT / "LEEME_CIERRE.md",
                  ROOT / "despliegue_stm32/RUTA_VALIDACION_EMBEBIDA_CWRU.md"])

    # Recursos V1: inventario de disponibilidad, no aceptación científica V2.
    resources = ROOT / "modelos/cwru/canonico_seed42/firmware"
    historical = load(resources / "manifest.json")
    for relative, expected in historical["generated_files"].items():
        path = resources / relative
        require(sha(path) == expected["sha256"], "Recurso histórico modificado: " + relative)
        paths.add(path)
    paths.add(resources / "manifest.json")
    model = ROOT / "resultados/cwru/modelo_features_14/canonico_seed42_int8/modelo_cwru_de_fe_seed42_int8.tflite"
    require(sha(model) == historical["source_hashes"]["tflite_int8"], "Modelo histórico distinto")
    paths.add(model)

    inventory = [{"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size,
                  "sha256": sha(path)} for path in sorted(paths)]
    report = {
        "status": "EXTRACTION_CANDIDATE_CLOSED_NOT_FULL_PIPELINE",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": CONFIG, "elf_sha256": ELF_SHA,
        "source_resources_checked": len(expected_source),
        "source_resources_bit_identical_to_prepared_copy_and_active_D_at_closure": exact_matches,
        "ide_metadata_differences_inspected_not_overwritten": ide_differences,
        "active_project_at_closure": str(ACTIVE),
        "active_D_is_not_a_permanent_immutable_reference": True,
        "build_units": len(units), "old_adapter_input_bytes": 21,
        "cwru_required_input_bytes": 14, "old_adapter_linked": False,
        "physical_run": RUN, "validation": physical["validation"],
        "timing_extraction_only": physical["timing"],
        "hop_budget_observed_only": physical["hop_budget"],
        "build_memory_and_stack_scope": build["configurations"][CONFIG],
        "limitations": physical["limitations"],
        "next_block": "scale_and_quantization_contract_and_validation",
        "historical_model_use": "hardware_compatibility_and_cost_pilot_only_not_V2_metrics",
        "firmware_modified": False, "serial_port_opened": False,
        "experiments_repeated": False, "scientific_V2_protocol_frozen": False,
        "referenced_files": inventory,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    with SEAL.open("x", encoding="ascii", newline="\n") as stream:
        stream.write(f"{sha(MANIFEST)}  {MANIFEST.name}\n")
    verify()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    verify() if args.verify else create()
