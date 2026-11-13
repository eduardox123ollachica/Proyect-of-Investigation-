"""Prepara y verifica Release en una copia aislada, sin programar el STM32.

No cambia fuentes, reloj, .cproject ni workspace del usuario en D:.
La instalacion del nuevo .launch se realiza aparte, con aprobacion.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[4]
NAME = "Prueba_2_CWRU_Extraccion"
SOURCE = Path(r"D:\STM32IDE") / NAME
TEMPLATE = ROOT / "despliegue_stm32/correccion_cubeide_2026-10-22/prueba_ide" / NAME
OUT = ROOT / "despliegue_stm32/prueba_release_16mhz_2026-10-23"
PROJECT = OUT / "copia_aislada" / NAME
IDE = Path(r"D:\STM32CubeIDE_2.2.0\STM32CubeIDE")
LAUNCH = NAME + "_Release_16MHz.launch"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, data):
    (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def prepare():
    if PROJECT.exists() or (OUT / "PREPARACION.json").exists():
        raise FileExistsError("La preparacion ya existe; no se sobrescribe")
    raw = (SOURCE / ".cproject").read_bytes()
    if b"<?fileVersion 4.0.0?>" not in raw:
        raise RuntimeError("Falta el marcador requerido por CDT")
    tree = ET.fromstring(raw)
    configs = {c.find("storageModule").get("name"): c for c in tree.findall("./storageModule/cconfiguration")}
    release = configs["Release"]
    compiler = next(t for t in release.findall(".//tool") if t.get("superClass", "").endswith("tool.c.compiler"))
    optimization = next(o for o in compiler.findall("option") if o.get("superClass", "").endswith("optimization.level"))
    if not optimization.get("value", "").endswith(".os"):
        raise RuntimeError("Se esperaba Release -Os, sin modificar configuracion")
    if "SystemClock_Config(" in (SOURCE / "Src/main.c").read_text(encoding="utf-8-sig"):
        raise RuntimeError("El main cambio; revisar el reloj antes de continuar")

    # El inventario de la copia ya compilada selecciona recursos, no sus bytes:
    # todos los bytes de fuentes/configuracion se toman del proyecto actual D:.
    inventory = []
    for existing in sorted(TEMPLATE.rglob("*")):
        if not existing.is_file():
            continue
        rel = existing.relative_to(TEMPLATE)
        if rel.parts[0] in ("Debug", "Release") or existing.suffix == ".launch":
            continue
        source = SOURCE / rel
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = PROJECT / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        before = digest(source)
        shutil.copy2(source, destination)
        if before != digest(destination) or before != digest(source):
            raise RuntimeError("Copia no identica o fuente modificada durante preparacion")
        inventory.append(dict(path=rel.as_posix(), sha256=before))

    # Nueva configuracion independiente: conserva intacta la de Debug.
    launch = (SOURCE / (NAME + ".launch")).read_text(encoding="utf-8")
    launch = launch.replace("Debug/" + NAME + ".elf", "Release/" + NAME + ".elf")
    launch = launch.replace("\\Debug\\st-link_gdbserver_log.txt", "\\Release\\st-link_gdbserver_log.txt")
    launch = launch.replace(configs["Debug"].get("id"), release.get("id"))
    launch, count = re.subn(
        r'(<booleanAttribute key="org.eclipse.cdt.launch.PROJECT_BUILD_CONFIG_AUTO_ATTR" value=")true("/>)',
        r'\g<1>false\2', launch)
    if count != 1:
        raise RuntimeError("No se pudo fijar la configuracion de compilacion Release")
    parsed = ET.fromstring(launch)
    attrs = {e.get("key"): e.get("value") for e in parsed}
    if attrs["org.eclipse.cdt.launch.PROGRAM_NAME"] != "Release/" + NAME + ".elf":
        raise RuntimeError("El .launch no apunta al ELF Release")
    loads = json.loads(attrs["com.st.stm32cube.ide.mcu.debug.launch.loadList"])["fItems"]
    if not all(i["fPath"] == "Release/" + NAME + ".elf" for i in loads):
        raise RuntimeError("La lista de carga conserva un binario distinto")
    (OUT / LAUNCH).write_text(launch, encoding="utf-8")
    save("PREPARACION.json", dict(
        source=str(SOURCE), isolated_project=str(PROJECT), optimization="-Os",
        intended_hclk_hz=16000000, hardware_clock_validation_pending=True,
        changes_to_source_project=[], firmware_flashed=False,
        release_configuration_id=release.get("id"), copied_files=inventory,
        launch_name=LAUNCH, launch_sha256=digest(OUT / LAUNCH),
        debug_baseline_results=str(ROOT / "despliegue_stm32/cwru_extraccion_prueba2/Validacion_PC/resultados/20261023T161442_499065Z/resultados.json"),
        note="No cambio de formulas, precision, ventanas, reloj ni entrenamiento. Solo Release -Os."))
    print(f"PREPARADO: {len(inventory)} recursos identicos; .launch Release independiente.", flush=True)


def build():
    preparation = json.loads((OUT / "PREPARACION.json").read_text(encoding="utf-8"))
    for row in preparation["copied_files"]:
        if digest(SOURCE / row["path"]) != row["sha256"] or digest(PROJECT / row["path"]) != row["sha256"]:
            raise RuntimeError("Cambio de fuentes desde la preparacion: " + row["path"])
    workspace = OUT / "workspace_aislado"
    if workspace.exists() or (OUT / "COMPILACION_RELEASE.json").exists():
        raise FileExistsError("La compilacion ya existe; no se sobrescribe")
    argv = [str(IDE / "stm32cubeidec.exe"), "--launcher.suppressErrors", "-nosplash",
            "-application", "org.eclipse.cdt.managedbuilder.core.headlessbuild",
            "-data", str(workspace), "-import", str(PROJECT), "-no-indexer", "-build", NAME + "/Release"]
    tool = next((IDE / "plugins").glob("com.st.stm32cube.ide.mcu.externaltools.gnu-tools-for-stm32.*/tools/bin"))
    make = next((IDE / "plugins").glob("com.st.stm32cube.ide.mcu.externaltools.make.*/tools/bin"))
    env = dict(os.environ)
    env["PATH"] = str(tool) + os.pathsep + str(make) + os.pathsep + env.get("PATH", "")
    start = time.monotonic()
    with (OUT / "CUBEIDE_RELEASE.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(argv, env=env, cwd=OUT, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, encoding="utf-8", errors="replace")
        for line in process.stdout:
            log.write(line)
            log.flush()
            if any(s in line.lower() for s in ("build of", "error", "warning", "finished", "text", "data", "bss")):
                print(line.strip()[:350], flush=True)
        code = process.wait(timeout=30)
    elf = PROJECT / "Release" / (NAME + ".elf")
    mk = (PROJECT / "Release/Src/subdir.mk").read_text(encoding="utf-8") if elf.is_file() else ""
    logtext = (OUT / "CUBEIDE_RELEASE.log").read_text(encoding="utf-8")
    same_sources = all(digest(SOURCE / r["path"]) == r["sha256"] for r in preparation["copied_files"])
    report = dict(returncode=code, elapsed_seconds=time.monotonic()-start, command=argv,
                  status="COMPILED_NOT_FLASHED" if code == 0 and elf.is_file() else "BUILD_FAILED",
                  user_sources_unchanged=same_sources, user_workspace_modified=False, hardware_flashed=False,
                  optimization_os_confirmed="-Os " in mk, no_fast_math="-ffast-math" not in mk,
                  zero_errors_warnings="0 errors, 0 warnings" in logtext,
                  elf=str(elf), elf_sha256=digest(elf) if elf.is_file() else None)
    if elf.is_file():
        size = subprocess.check_output([str(tool / "arm-none-eabi-size.exe"), str(elf)], text=True)
        text_bytes, data_bytes, bss_bytes = map(int, size.strip().splitlines()[-1].split()[:3])
        report.update(text_bytes=text_bytes, data_bytes=data_bytes, bss_bytes=bss_bytes,
                      flash_load_bytes=text_bytes+data_bytes,
                      ram_static_with_linker_reservations_bytes=data_bytes+bss_bytes)
    save("COMPILACION_RELEASE.json", report)
    print(json.dumps(report, ensure_ascii=True, indent=2), flush=True)
    if code or not elf.is_file() or not same_sources or not report["optimization_os_confirmed"] or not report["zero_errors_warnings"]:
        raise SystemExit(2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "build"))
    args = parser.parse_args()
    prepare() if args.mode == "prepare" else build()
