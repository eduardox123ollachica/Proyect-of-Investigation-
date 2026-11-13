"""Ensayo de reloj: copia aislada, Release de control y Release_180MHz.

Nunca escribe en D:, ni conecta/programa la placa. Conserva respaldo byte a byte
de main/.cproject y genera un overlay para instalacion separada con aprobacion.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import os
import shutil
import struct
import subprocess
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[4]
NAME = "Prueba_2_CWRU_Extraccion"
CONFIG = "Release_180MHz"
SOURCE = Path(r"D:\STM32IDE") / NAME
PREVIOUS = ROOT / "despliegue_stm32/prueba_release_16mhz_2026-10-23"
OUT = ROOT / "despliegue_stm32/prueba_reloj_180mhz_2026-10-23"
PROJECT = OUT / "copia_aislada" / NAME
OVERLAY = OUT / "archivos_para_D"
IDE = Path(r"D:\STM32CubeIDE_2.2.0\STM32CubeIDE")
LAUNCH = NAME + "_Release_180MHz.launch"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def flash_image(path):
    data = path.read_bytes()
    assert data[:6] == b"\x7fELF\x01\x01", "Se requiere ELF32 little endian"
    offset = struct.unpack_from("<I", data, 28)[0]
    step, count = struct.unpack_from("<HH", data, 42)
    image = []
    for i in range(count):
        typ, pos, _, physical, size, _, _, _ = struct.unpack_from("<8I", data, offset+i*step)
        if typ == 1 and size and 0x08000000 <= physical < 0x08080000:
            image.append((physical, data[pos:pos+size]))
    return sorted(image)


def prepare():
    if PROJECT.exists() or (OUT / "PREPARACION.json").exists():
        raise FileExistsError("No se sobrescribe una preparacion anterior")
    previous = json.loads((PREVIOUS / "PREPARACION.json").read_text(encoding="utf-8"))
    for row in previous["copied_files"]:
        if sha(SOURCE / row["path"]) != row["sha256"]:
            raise RuntimeError("El origen cambio; revisar antes de aplicar: " + row["path"])
    for row in previous["copied_files"]:
        rel = Path(row["path"])
        destination = PROJECT / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SOURCE / rel, destination)
    for rel in (".cproject", "Src/main.c", "Release/"+NAME+".elf"):
        target = OUT / "respaldo_16mhz" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SOURCE / rel, target)

    # El cambio C de main se aplica despues con apply_patch, en la copia y
    # el overlay. Este preparador solo copia recursos y genera metadatos CDT.
    shutil.copy2(Path(__file__).with_name("cwru_clock_180mhz.inc"), PROJECT / "Inc/cwru_clock_180mhz.inc")

    cproject = PROJECT / ".cproject"
    original = ET.fromstring(cproject.read_bytes())
    settings = original.find("storageModule[@moduleId='org.eclipse.cdt.core.settings']")
    release = next(c for c in settings.findall("cconfiguration") if c.find("storageModule").get("name") == "Release")
    expanded = copy.deepcopy(release)
    ids = {e.get("id"): e.get("id")+".cwru180" for e in expanded.iter() if e.get("id")}
    for element in expanded.iter():
        for key, value in list(element.attrib.items()):
            if value in ids:
                element.set(key, ids[value])
            elif key == "name" and value == "Release":
                element.set(key, CONFIG)
            elif key == "buildPath" and value.endswith("/Release"):
                element.set(key, value[:-7]+CONFIG)
    for tool in expanded.findall(".//tool"):
        if tool.get("superClass", "").endswith(("tool.c.compiler", "tool.cpp.compiler")):
            defines = next(o for o in tool.findall("option") if o.get("superClass", "").endswith(".definedsymbols"))
            ET.SubElement(defines, "listOptionValue", builtIn="false", value="CWRU_TARGET_HCLK_HZ=180000000U")

    driver_rel = Path("Drivers/STM32F4xx_HAL_Driver/Src")
    driver_name = "stm32f4xx_hal_pwr_ex.c"
    driver = SOURCE / driver_rel / driver_name
    destination = PROJECT / driver_rel / driver_name
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(driver, destination)
    excluded = sorted(p.relative_to(SOURCE / driver_rel).as_posix()
                      for p in (SOURCE / driver_rel).rglob("*")
                      if p.is_file() and p.suffix.lower() in (".c", ".cc", ".cpp", ".s") and p != driver)
    ET.SubElement(expanded.find(".//sourceEntries"), "entry", excluding="|".join(excluded),
                  flags="VALUE_WORKSPACE_PATH|RESOLVED", kind="sourcePath", name=driver_rel.as_posix())
    settings.append(expanded)
    cproject.write_bytes(b'<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n<?fileVersion 4.0.0?>\n' + ET.tostring(original, encoding="utf-8") + b"\n")

    launch = ET.fromstring((SOURCE / (NAME+"_Release_16MHz.launch")).read_bytes())
    attrs = {e.get("key"): e for e in launch}
    attrs["org.eclipse.cdt.launch.PROGRAM_NAME"].set("value", CONFIG+"/"+NAME+".elf")
    attrs["org.eclipse.cdt.launch.PROJECT_BUILD_CONFIG_ID_ATTR"].set("value", expanded.get("id"))
    attrs["org.eclipse.cdt.launch.PROJECT_BUILD_CONFIG_AUTO_ATTR"].set("value", "false")
    loadkey = "com.st.stm32cube.ide.mcu.debug.launch.loadList"
    loads = json.loads(attrs[loadkey].get("value"))
    for item in loads["fItems"]:
        item["fPath"] = CONFIG+"/"+NAME+".elf"
    attrs[loadkey].set("value", json.dumps(loads, separators=(",", ":")))
    for key, element in attrs.items():
        if "swv_" in key and "hclk" in key:
            element.set("value", "180000000")
        if key.endswith("stlink.log_file"):
            element.set("value", str(SOURCE / CONFIG / "st-link_gdbserver_log.txt"))
    # Mantener la parada en main: el usuario reanuda con F8.
    (OUT / LAUNCH).write_bytes(b'<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(launch, encoding="utf-8") + b"\n")

    overlay_files = [".cproject", "Src/main.c", "Inc/cwru_clock_180mhz.inc"]
    for rel in overlay_files:
        target = OVERLAY / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PROJECT / rel, target)
    shutil.copy2(OUT / LAUNCH, OVERLAY / LAUNCH)
    inventory = previous["copied_files"] + [dict(path=(driver_rel/driver_name).as_posix(), sha256=sha(driver))]
    save("PREPARACION.json", dict(source=str(SOURCE), isolated_project=str(PROJECT),
        configuration=CONFIG, intended_hclk_hz=180000000, oscillator="HSI nominal 16 MHz",
        pll=dict(M=16,N=360,P=2,Q=8,R=6), hclk_div=1, apb1_div=4, apb2_div=2,
        voltage_scale=1, overdrive=True, flash_wait_states=5, optimization="-Os",
        source_inventory_before=inventory, overlay_files=overlay_files+[LAUNCH],
        new_configuration_id=expanded.get("id"), hardware_flashed=False,
        original_debug_release_configurations_preserved=True,
        note="Solo arranque de reloj; extractor, protocolo, UART 115200, datos y precision se conservan."))
    print("PREPARADO: reloj opcional 180 MHz; Release original conservado; D: intacto.", flush=True)


def build():
    preparation = json.loads((OUT / "PREPARACION.json").read_text(encoding="utf-8"))
    main = (PROJECT / "Src/main.c").read_text(encoding="utf-8-sig")
    if main.count("CWRU_SystemClock180_Config();") != 1 or "#define CWRU_TARGET_HCLK_HZ 16000000U" not in main:
        raise RuntimeError("Falta aplicar el parche condicional de main.c")
    if (PROJECT / "Src/main.c").read_bytes() != (OVERLAY / "Src/main.c").read_bytes():
        raise RuntimeError("El overlay main.c no coincide con la copia a compilar")
    for row in preparation["source_inventory_before"]:
        if sha(SOURCE / row["path"]) != row["sha256"]:
            raise RuntimeError("El origen cambio antes de compilar: " + row["path"])
    workspace = OUT / "workspace_aislado"
    if workspace.exists():
        raise FileExistsError("No se sobrescribe una compilacion anterior")
    args = [str(IDE/"stm32cubeidec.exe"), "--launcher.suppressErrors", "-nosplash",
            "-application", "org.eclipse.cdt.managedbuilder.core.headlessbuild", "-data", str(workspace),
            "-import", str(PROJECT), "-no-indexer", "-build", NAME+"/Release", "-build", NAME+"/"+CONFIG]
    tool = next((IDE/"plugins").glob("com.st.stm32cube.ide.mcu.externaltools.gnu-tools-for-stm32.*/tools/bin"))
    make = next((IDE/"plugins").glob("com.st.stm32cube.ide.mcu.externaltools.make.*/tools/bin"))
    env = dict(os.environ)
    env["PATH"] = str(tool)+os.pathsep+str(make)+os.pathsep+env.get("PATH", "")
    start = time.monotonic()
    with (OUT/"CUBEIDE_RELOJ.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(args, cwd=OUT, env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, encoding="utf-8", errors="replace")
        for line in process.stdout:
            log.write(line)
            log.flush()
            if any(s in line.lower() for s in ("build of", "error:", "warning:", "build finished", "text\t")):
                print(line.strip()[:350], flush=True)
        code = process.wait(timeout=30)
    reference = OUT/"respaldo_16mhz/Release"/(NAME+".elf")
    control = PROJECT/"Release"/(NAME+".elf")
    fast = PROJECT/CONFIG/(NAME+".elf")
    log = (OUT/"CUBEIDE_RELOJ.log").read_text(encoding="utf-8")
    images_equal = control.is_file() and flash_image(reference) == flash_image(control)
    unchanged = all(sha(SOURCE/r["path"]) == r["sha256"] for r in preparation["source_inventory_before"])
    report = dict(returncode=code, elapsed_seconds=time.monotonic()-start, command=args,
        original_sources_unchanged=unchanged, control_16mhz_flash_bytes_identical=images_equal,
        zero_errors_warnings_both=log.count("0 errors, 0 warnings") == 2,
        hardware_flashed=False, hardware_validation_pending=True, configurations={})
    for cfg, elf in (("Release",control),(CONFIG,fast)):
        if not elf.is_file():
            continue
        size = subprocess.check_output([str(tool/"arm-none-eabi-size.exe"),str(elf)],text=True)
        text_bytes,data_bytes,bss_bytes = map(int,size.strip().splitlines()[-1].split()[:3])
        report["configurations"][cfg] = dict(elf=str(elf),elf_sha256=sha(elf),text_bytes=text_bytes,
            data_bytes=data_bytes,bss_bytes=bss_bytes,flash_load_bytes=text_bytes+data_bytes,
            ram_static_with_linker_reservations_bytes=data_bytes+bss_bytes)
    save("COMPILACION_RELOJ.json",report)
    print(json.dumps(report,ensure_ascii=True,indent=2),flush=True)
    if code or not images_equal or not unchanged or not report["zero_errors_warnings_both"] or len(report["configurations"]) != 2:
        raise SystemExit(2)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=("prepare","build"))
    args=parser.parse_args()
    prepare() if args.mode == "prepare" else build()
