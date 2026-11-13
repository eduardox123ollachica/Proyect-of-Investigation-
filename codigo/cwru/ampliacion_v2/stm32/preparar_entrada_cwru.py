"""Prepara/compila/instala una configuración opcional de entrada CWRU.

prepare sólo escribe en C. build usa CubeIDE aislado (requiere permiso).
install escribe ocho archivos concretos en D, con respaldo (requiere permiso).
Ningún modo abre COM ni programa la placa.
"""
from __future__ import annotations
import argparse
import copy
import json
import os
import re
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from preparar_cache_potencias import configs, sha
from preparar_reloj_180mhz import flash_image

ROOT = Path(__file__).resolve().parents[4]
NAME = "Prueba_2_CWRU_Extraccion"
SOURCE = Path(r"D:\STM32IDE") / NAME
OUT = ROOT / "despliegue_stm32/escalado_cuantizacion_180mhz_2026-10-23"
PRIOR = ROOT / "despliegue_stm32/mejoras_entropia_180mhz_2026-10-23"
PROJECT = OUT / "copia_aislada" / NAME
BACKUP = OUT / "respaldo_antes_entrada"
OVERLAY = OUT / "archivos_para_D"
CLIENT = OUT / "Validacion_PC"
CONTROL = "Release_180MHz_EntM"
CONFIG = "Release_180MHz_Input"
LAUNCH = NAME + "_Input_180MHz.launch"
IDE = Path(r"D:\STM32CubeIDE_2.2.0\STM32CubeIDE")
FIRMWARE = Path(__file__).with_name("entrada_firmware")
FILES = [".cproject", "Src/uart_commands.c", LAUNCH, *["CWRU_Input/" + name for name in (
    "cwru_input_transform.c", "cwru_input_transform.h", "cmd_cwru_input.c", "cwru_dnn_scaler.c", "cwru_dnn_scaler.h")]]


def save(name, data):
    with (OUT / name).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2); stream.write("\n")


def load(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def check_origin(prep):
    for row in prep["source_inventory_before"]:
        if sha(SOURCE / row["path"]) != row["sha256"]:
            raise RuntimeError("D cambió desde la preparación: " + row["path"])
    for cfg, digest in prep["existing_elf_sha256"].items():
        if sha(SOURCE / cfg / (NAME + ".elf")) != digest:
            raise RuntimeError("ELF anterior cambió: " + cfg)


def prepare():
    if PROJECT.exists() or (SOURCE / LAUNCH).exists() or (SOURCE / "CWRU_Input").exists():
        raise FileExistsError("Preparación/destino ya existente")
    pc = load("auditoria_pc/RESULTADOS_PC.json")
    if pc["comparisons"]["entm_vs_cache"]["quantized_values_equal"] != 588:
        raise RuntimeError("Diferencias INT8 requieren revisión antes de preparar")
    closure = ROOT / "codigo/cwru/ampliacion_v2/stm32/cerrar_etapa_extraccion.py"
    subprocess.run([sys.executable, "-B", str(closure), "--verify"], check=True)
    source_list = json.loads((PRIOR / "PREPARACION.json").read_text(encoding="utf-8"))["source_inventory_before"]
    paths = {r["path"] for r in source_list} | {p.name for p in SOURCE.glob("*.launch")}
    snapshot = []
    for rel in sorted(paths):
        digest = sha(SOURCE / rel)
        for target_root in (PROJECT, BACKUP):
            target = target_root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(SOURCE / rel, target)
            if sha(target) != digest:
                raise RuntimeError("Copia diferente: " + rel)
        snapshot.append(dict(path=rel, sha256=digest))
    tree, settings, before = configs(PROJECT / ".cproject")
    if len(before) != 8 or CONTROL not in before:
        raise RuntimeError("Se esperaban las ocho configuraciones anteriores")
    elves = {}
    for cfg in before:
        path = SOURCE / cfg / (NAME + ".elf")
        target = BACKUP / cfg / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        elves[cfg] = sha(path)
        if sha(target) != elves[cfg]:
            raise RuntimeError("Respaldo ELF incorrecto")
    expanded = copy.deepcopy(before[CONTROL])
    mapping = {e.get("id"): e.get("id") + ".input1" for e in expanded.iter() if e.get("id")}
    for element in expanded.iter():
        for key, value in list(element.attrib.items()):
            if value in mapping: element.set(key, mapping[value])
            elif key == "name" and value == CONTROL: element.set(key, CONFIG)
            elif key == "buildPath" and value.endswith("/" + CONTROL): element.set(key, value[:-len(CONTROL)] + CONFIG)
    for tool in expanded.findall(".//tool"):
        if tool.get("superClass", "").endswith(("tool.c.compiler", "tool.cpp.compiler")):
            definitions = next(o for o in tool.findall("option") if o.get("superClass", "").endswith(".definedsymbols"))
            ET.SubElement(definitions, "listOptionValue", builtIn="false", value="CWRU_INPUT_TRANSFORM=1")
            includes = next(o for o in tool.findall("option") if o.get("superClass", "").endswith(".includepaths"))
            ET.SubElement(includes, "listOptionValue", builtIn="false", value="../CWRU_Input")
    entries = expanded.find(".//sourceEntries")
    ET.SubElement(entries, "entry", flags="VALUE_WORKSPACE_PATH|RESOLVED", kind="sourcePath", name="CWRU_Input")
    old_ids = {e.get("id") for c in before.values() for e in c.iter() if e.get("id")}
    if old_ids & {e.get("id") for e in expanded.iter() if e.get("id")}:
        raise RuntimeError("IDs CDT duplicados")
    settings.append(expanded)
    (PROJECT / ".cproject").write_bytes(b'<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n'
        b'<?fileVersion 4.0.0?>\n' + ET.tostring(tree, encoding="utf-8") + b"\n")
    after = configs(PROJECT / ".cproject")[2]
    if not all(ET.tostring(c) == ET.tostring(after[n]) for n, c in before.items()):
        raise RuntimeError("Cambió una configuración previa")
    folder = PROJECT / "CWRU_Input"; folder.mkdir()
    for path in FIRMWARE.iterdir():
        if path.suffix in (".c", ".h"):
            shutil.copy2(path, folder / path.name)
    resources = ROOT / "modelos/cwru/canonico_seed42/firmware"
    shutil.copy2(resources / "cwru_dnn_scaler.cpp", folder / "cwru_dnn_scaler.c")
    shutil.copy2(resources / "cwru_dnn_scaler.h", folder / "cwru_dnn_scaler.h")
    launch = ET.parse(SOURCE / (NAME + "_EntM_180MHz.launch")).getroot()
    attrs = {e.get("key"): e for e in launch}
    elf = CONFIG + "/" + NAME + ".elf"
    attrs["org.eclipse.cdt.launch.PROGRAM_NAME"].set("value", elf)
    attrs["org.eclipse.cdt.launch.PROJECT_BUILD_CONFIG_ID_ATTR"].set("value", expanded.get("id"))
    attrs["org.eclipse.cdt.launch.PROJECT_BUILD_CONFIG_AUTO_ATTR"].set("value", "false")
    key = "com.st.stm32cube.ide.mcu.debug.launch.loadList"
    loads = json.loads(attrs[key].get("value"))
    for item in loads["fItems"]: item["fPath"] = elf
    attrs[key].set("value", json.dumps(loads, separators=(",", ":")))
    for key, element in attrs.items():
        if key.endswith("stlink.log_file"):
            element.set("value", str(SOURCE / CONFIG / "st-link_gdbserver_log.txt"))
    (PROJECT / LAUNCH).write_bytes(b'<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(launch, encoding="utf-8") + b"\n")
    CLIENT.mkdir()
    shutil.copy2(Path(__file__).with_name("validar_entrada_cwru.py"), CLIENT / "validar_entrada.py")
    shutil.copy2(OUT / "auditoria_pc/RESULTADOS_PC.json", CLIENT / "PARAMETROS_PC.json")
    shutil.copy2(OUT / "auditoria_pc/casos_validacion.json", CLIENT / "casos_validacion.json")
    (CLIENT / "VALIDACION_MANIFEST.json").write_text(json.dumps({p.name: sha(p) for p in CLIENT.iterdir() if p.is_file()}, indent=2) + "\n", encoding="utf-8")
    save("PREPARACION.json", dict(status="PREPARED_DISPATCH_PATCH_PENDING", source_inventory_before=snapshot,
        existing_elf_sha256=elves, configuration=CONFIG, control=CONTROL, previous_eight_configurations_preserved=True,
        added_resources=FILES[2:], modified_resources=FILES[:2], hardware_flashed=False,
        serial_port_opened=False, model_inference_included=False, historical_model_parameters_only=True))
    print("PREPARADO: copia aislada, nueve configuraciones y cliente; aplicar guardia UART antes de build.", flush=True)


def build():
    prep = load("PREPARACION.json"); check_origin(prep)
    if (OUT / "workspace_aislado").exists() or (OUT / "COMPILACION_ENTRADA.json").exists():
        raise FileExistsError("No repetir/sobrescribir build")
    dispatcher = (PROJECT / "Src/uart_commands.c").read_text(encoding="utf-8")
    if "CWRU_INPUT_TRANSFORM" not in dispatcher or "CWRU_InputCommand" not in dispatcher:
        raise RuntimeError("Falta parche aislado UART")
    test = subprocess.run([sys.executable, "-B", str(CLIENT / "validar_entrada.py"), "--self-test"],
                          text=True, capture_output=True, encoding="utf-8", errors="replace")
    save("PRUEBAS_CLIENTE.json", dict(status="PASS" if test.returncode == 0 else "FAIL",
        output=test.stdout + test.stderr, client_sha256=sha(CLIENT / "validar_entrada.py"),
        executes_ARM_C=False, serial_port_opened=False))
    if test.returncode: raise RuntimeError("Pruebas del cliente fallaron")
    for rel in FILES:
        target = OVERLAY / rel; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PROJECT / rel, target)
    tool = next((IDE / "plugins").glob("com.st.stm32cube.ide.mcu.externaltools.gnu-tools-for-stm32.*/tools/bin"))
    make = next((IDE / "plugins").glob("com.st.stm32cube.ide.mcu.externaltools.make.*/tools/bin"))
    env = dict(os.environ); env["PATH"] = str(tool) + os.pathsep + str(make) + os.pathsep + env.get("PATH", "")
    args = [str(IDE / "stm32cubeidec.exe"), "--launcher.suppressErrors", "-nosplash", "-application",
        "org.eclipse.cdt.managedbuilder.core.headlessbuild", "-data", str(OUT / "workspace_aislado"),
        "-import", str(PROJECT), "-no-indexer", "-build", NAME + "/" + CONTROL, "-build", NAME + "/" + CONFIG]
    start = time.monotonic()
    with (OUT / "CUBEIDE_ENTRADA.log").open("x", encoding="utf-8") as log:
        process = subprocess.Popen(args, cwd=OUT, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, encoding="utf-8", errors="replace")
        for line in process.stdout:
            log.write(line); log.flush()
            if any(s in line.lower() for s in ("build of", "error:", "warning:", "build finished")):
                print(line.strip()[:300], flush=True)
        code = process.wait(timeout=30)
    old_elf, control_elf = BACKUP / CONTROL / (NAME + ".elf"), PROJECT / CONTROL / (NAME + ".elf")
    text = (OUT / "CUBEIDE_ENTRADA.log").read_text(encoding="utf-8")
    report = dict(returncode=code, elapsed_seconds=time.monotonic() - start, command=args,
                  zero_errors_warnings_two=text.count("0 errors, 0 warnings") == 2,
                  control_elf_identical=sha(old_elf) == sha(control_elf),
                  control_flash_identical=flash_image(old_elf) == flash_image(control_elf), configurations={})
    for cfg in (CONTROL, CONFIG):
        folder = PROJECT / cfg; elf = folder / (NAME + ".elf")
        size = subprocess.check_output([str(tool / "arm-none-eabi-size.exe"), str(elf)], text=True)
        t, d, b = map(int, size.strip().splitlines()[-1].split()[:3])
        commands = [line for mk in folder.rglob("subdir.mk") for line in mk.read_text(encoding="utf-8").splitlines()
                    if "arm-none-eabi-gcc" in line and "-std=gnu11" in line]
        flags = bool(commands) and all(re.findall(r"(?<!\S)-O\S+", line) == ["-O3"]
            and "-DCWRU_TARGET_HCLK_HZ=180000000U" in line and "-DCWRU_ENTROPY_MODE=2" in line
            and "-DCWRU_CACHE_SPECTRAL_POWER=1" in line
            and ("-DCWRU_INPUT_TRANSFORM=1" in line) == (cfg == CONFIG)
            and not any(f in line for f in ("-ffast-math", "-Ofast", "-funsafe-math-optimizations")) for line in commands)
        image = flash_image(elf); span = max(a + len(data) for a, data in image) - min(a for a, _ in image)
        units = (folder / "objects.list").read_text(encoding="utf-8").splitlines()
        report["configurations"][cfg] = dict(elf_sha256=sha(elf), text=t, data=d, bss=b,
            flash_load_bytes=t+d, ram_static_with_linker_reservations_bytes=d+b, flags_confirmed=flags,
            flash_span_bytes=span, expected_download_kib=span / 1024, compilation_units=len(units),
            input_units=sum("CWRU_Input" in line for line in units),
            new_module_stack_frames={p.name: p.read_text(encoding="utf-8").splitlines() for p in (folder / "CWRU_Input").glob("*.su")})
    includes = ["-I" + str(PROJECT / p) for p in ("Inc", "Drivers/CMSIS/Include", "Drivers/CMSIS/Device/ST/STM32F4xx/Include")]
    command = [str(tool / "arm-none-eabi-gcc.exe"), "-E", "-P", "-std=gnu11", "-mcpu=cortex-m4",
               "-mfpu=fpv4-sp-d16", "-mfloat-abi=hard", "-mthumb", "-DSTM32F446xx", *includes]
    before = subprocess.check_output([*command, str(BACKUP / "Src/uart_commands.c")])
    after = subprocess.check_output([*command, str(PROJECT / "Src/uart_commands.c")])
    report["control_uart_preprocessed_identical"] = before == after
    before_cfg, after_cfg = configs(BACKUP / ".cproject")[2], configs(PROJECT / ".cproject")[2]
    report["previous_eight_configurations_preserved"] = all(ET.tostring(c) == ET.tostring(after_cfg[n]) for n, c in before_cfg.items())
    allowed = {".cproject", "Src/uart_commands.c", ".settings/language.settings.xml"}
    report["all_other_copied_resources_unchanged"] = all(sha(PROJECT / r["path"]) == r["sha256"] for r in prep["source_inventory_before"] if r["path"] not in allowed)
    report["overlay_sha256"] = {rel: sha(OVERLAY / rel) for rel in FILES}
    passed = (code == 0 and report["zero_errors_warnings_two"] and report["control_elf_identical"]
        and report["control_flash_identical"] and report["control_uart_preprocessed_identical"]
        and report["previous_eight_configurations_preserved"] and report["all_other_copied_resources_unchanged"]
        and all(r["flags_confirmed"] for r in report["configurations"].values())
        and report["configurations"][CONTROL]["input_units"] == 0 and report["configurations"][CONFIG]["input_units"] == 3)
    report.update(status="PASS" if passed else "FAIL", hardware_flashed=False, physical_validation_pending=True,
                  serial_port_opened=False, training_executed=False)
    check_origin(prep); save("COMPILACION_ENTRADA.json", report)
    print(json.dumps(report, ensure_ascii=True, indent=2), flush=True)
    if not passed: raise SystemExit(2)


def install():
    prep, built = load("PREPARACION.json"), load("COMPILACION_ENTRADA.json")
    if (OUT / "APLICACION_EN_D.json").exists() or (SOURCE / "CWRU_Input").exists() or (SOURCE / LAUNCH).exists():
        raise FileExistsError("Destino ya instalado")
    if built["status"] != "PASS": raise RuntimeError("Build no aprobado")
    check_origin(prep)
    for rel, expected in built["overlay_sha256"].items():
        if sha(OVERLAY / rel) != expected: raise RuntimeError("Overlay modificado: " + rel)
    for rel in FILES:
        target = SOURCE / rel; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(OVERLAY / rel, target)
    previous, current = configs(BACKUP / ".cproject")[2], configs(SOURCE / ".cproject")[2]
    same = all(ET.tostring(c) == ET.tostring(current[n]) for n, c in previous.items())
    others = all(sha(SOURCE / r["path"]) == r["sha256"] for r in prep["source_inventory_before"] if r["path"] not in FILES[:2])
    elfs = all(sha(SOURCE / cfg / (NAME + ".elf")) == digest for cfg, digest in prep["existing_elf_sha256"].items())
    report = dict(status="PASS" if same and others and elfs else "FAIL", modified_files=FILES[:2], added_files=FILES[2:],
        previous_eight_configurations_preserved=same, previous_eight_elfs_unchanged=elfs,
        all_other_snapshot_resources_unchanged=others, installed_sha256={r: sha(SOURCE / r) for r in FILES},
        hardware_flashed=False, physical_validation_pending=True, user_workspace_modified=False)
    save("APLICACION_EN_D.json", report); print(json.dumps(report, ensure_ascii=True, indent=2), flush=True)
    if report["status"] != "PASS": raise SystemExit(2)


def seal():
    if (OUT / "MANIFEST_SHA256.json").exists() or (OUT / "MANIFEST_SHA256.sha256").exists():
        raise FileExistsError("No sobrescribir sello")
    prep, built, installed = load("PREPARACION.json"), load("COMPILACION_ENTRADA.json"), load("APLICACION_EN_D.json")
    if built["status"] != "PASS" or installed["status"] != "PASS":
        raise RuntimeError("Preparación no aprobada")
    for rel, expected in installed["installed_sha256"].items():
        if sha(SOURCE / rel) != expected: raise RuntimeError("Destino D diferente: " + rel)
    for cfg, expected in prep["existing_elf_sha256"].items():
        if sha(SOURCE / cfg / (NAME + ".elf")) != expected: raise RuntimeError("Build anterior distinto: " + cfg)
    paths = set()
    for row in prep["source_inventory_before"]:
        paths.update((PROJECT / row["path"], BACKUP / row["path"]))
    paths.update(BACKUP / cfg / (NAME + ".elf") for cfg in prep["existing_elf_sha256"])
    paths.update(OVERLAY / rel for rel in FILES)
    paths.update(PROJECT / rel for rel in FILES)
    for cfg in (CONTROL, CONFIG):
        folder = PROJECT / cfg
        paths.update(folder / rel for rel in (NAME + ".elf", NAME + ".map", "objects.list", "Src/cwru_feature_extraction.su"))
        paths.update((folder / "CWRU_Input").glob("*.su"))
    paths.update(p for folder in (CLIENT, OUT / "auditoria_pc") for p in folder.iterdir() if p.is_file())
    paths.update(p for p in OUT.iterdir() if p.is_file())
    paths.update((Path(__file__).resolve(), Path(__file__).with_name("validar_entrada_cwru.py")))
    paths.update(p for p in FIRMWARE.iterdir() if p.is_file())
    save("MANIFEST_SHA256.json", {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(paths)})
    with (OUT / "MANIFEST_SHA256.sha256").open("x", encoding="ascii", newline="\n") as stream:
        stream.write(sha(OUT / "MANIFEST_SHA256.json") + "  MANIFEST_SHA256.json\n")
    verify()


def verify():
    expected, name = (OUT / "MANIFEST_SHA256.sha256").read_text(encoding="ascii").strip().split()
    if name != "MANIFEST_SHA256.json" or sha(OUT / name) != expected:
        raise RuntimeError("Manifiesto de entrega modificado")
    manifest = load(name)
    for relative, expected in manifest.items():
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT) or sha(path) != expected:
            raise RuntimeError("Recurso diferente: " + relative)
    audit_inputs = load("auditoria_pc/ENTRADAS_SHA256.json")
    for relative, expected in audit_inputs.items():
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT) or sha(path) != expected:
            raise RuntimeError("Entrada de auditoría PC modificada: " + relative)
    print(json.dumps(dict(status="PASS", local_delivery_files_verified=len(manifest),
        original_audit_inputs_verified=len(audit_inputs), serial_port_opened=False,
        physical_validation_pending=True, active_D_rechecked=False), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "build", "install", "seal", "verify"))
    {"prepare": prepare, "build": build, "install": install, "seal": seal, "verify": verify}[parser.parse_args().mode]()
