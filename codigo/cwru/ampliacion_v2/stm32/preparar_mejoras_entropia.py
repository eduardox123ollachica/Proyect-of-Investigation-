"""Prepara dos variantes opcionales; no conecta ni programa el STM32.

EntD simplifica algebraicamente manteniendo double. EntM agrega logf con
acumulacion double y fallback double para probabilidades extremas.
"""
from pathlib import Path
import argparse
import copy
import json
import os
import re
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET
from preparar_cache_potencias import sha, configs
from preparar_reloj_180mhz import flash_image

ROOT = Path(__file__).resolve().parents[4]
NAME = "Prueba_2_CWRU_Extraccion"
SOURCE = Path(r"D:\STM32IDE") / NAME
PRIOR = ROOT / "despliegue_stm32/cache_potencias_180mhz_2026-10-23"
OUT = ROOT / "despliegue_stm32/mejoras_entropia_180mhz_2026-10-23"
PROJECT = OUT / "copia_aislada" / NAME
BACKUP = OUT / "respaldo_antes_entropia"
OVERLAY = OUT / "archivos_para_D"
IDE = Path(r"D:\STM32CubeIDE_2.2.0\STM32CubeIDE")
CONTROL = "Release_180MHz_O3_Cache"
VARIANTS = {"Release_180MHz_EntD": (1, "Double"), "Release_180MHz_EntM": (2, "Mixta")}
LAUNCHES = [NAME+"_EntD_180MHz.launch", NAME+"_EntM_180MHz.launch"]
FILES = [".cproject", "Src/cwru_feature_extraction.c", *LAUNCHES]


def save(name, obj):
    path = OUT / name
    if path.exists():
        raise FileExistsError("No sobrescribir evidencia: "+str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")


def load(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def check_origin(prep):
    for row in prep["source_inventory_before"]:
        if sha(SOURCE / row["path"]) != row["sha256"]:
            raise RuntimeError("Cambio en origen: "+row["path"])


def prepare():
    if PROJECT.exists() or any((SOURCE / p).exists() for p in LAUNCHES):
        raise FileExistsError("Preparacion/destino existente")
    prior = json.loads((PRIOR / "PREPARACION.json").read_text(encoding="utf-8"))
    installed = json.loads((PRIOR / "APLICACION_EN_D.json").read_text(encoding="utf-8"))
    if sha(SOURCE / "Src/cwru_feature_extraction.c") != installed["installed_sha256"]["Src/cwru_feature_extraction.c"]:
        raise RuntimeError("Extractor diferente al control fisico validado")
    validation = json.loads((PRIOR / "RESULTADOS_FISICOS_CACHE_42_VENTANAS.json").read_text(encoding="utf-8"))
    if validation["validation"]["passed_windows"] != 42:
        raise RuntimeError("Control fisico incompleto")
    paths = {r["path"] for r in prior["source_inventory_before"]} | {p.name for p in SOURCE.glob("*.launch")}
    snapshot = []
    for rel in sorted(paths):
        digest = sha(SOURCE / rel)
        for base in (PROJECT, BACKUP):
            target = base / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(SOURCE / rel, target)
            if sha(target) != digest or sha(SOURCE / rel) != digest:
                raise RuntimeError("Copia no identica: "+rel)
        snapshot.append(dict(path=rel, sha256=digest))
    tree, settings, before = configs(PROJECT / ".cproject")
    if len(before) != 6 or CONTROL not in before:
        raise RuntimeError("Configuraciones inesperadas")
    elves = {}
    for cfg in before:
        source = SOURCE / cfg / (NAME+".elf")
        target = BACKUP / cfg / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        elves[cfg] = sha(source)
        if sha(target) != elves[cfg]:
            raise RuntimeError("Respaldo ELF distinto")
    old_ids = {e.get("id") for c in before.values() for e in c.iter() if e.get("id")}
    for (cfg, (mode, label)), launch_name in zip(VARIANTS.items(), LAUNCHES):
        expanded = copy.deepcopy(before[CONTROL])
        mapping = {e.get("id"): e.get("id")+".entropy"+str(mode) for e in expanded.iter() if e.get("id")}
        for e in expanded.iter():
            for key, value in list(e.attrib.items()):
                if value in mapping:
                    e.set(key, mapping[value])
                elif key == "name" and value == CONTROL:
                    e.set(key, cfg)
                elif key == "buildPath" and value.endswith("/"+CONTROL):
                    e.set(key, value[:-len(CONTROL)]+cfg)
        for tool in expanded.findall(".//tool"):
            if tool.get("superClass", "").endswith(("tool.c.compiler", "tool.cpp.compiler")):
                definitions = next(o for o in tool.findall("option") if o.get("superClass", "").endswith(".definedsymbols"))
                ET.SubElement(definitions, "listOptionValue", builtIn="false", value="CWRU_ENTROPY_MODE="+str(mode))
        new_ids = {e.get("id") for e in expanded.iter() if e.get("id")}
        if old_ids & new_ids:
            raise RuntimeError("Colision de IDs CDT")
        old_ids |= new_ids
        settings.append(expanded)
        launch = ET.fromstring((SOURCE / (NAME+"_Cache_180MHz.launch")).read_bytes())
        attrs = {e.get("key"): e for e in launch}
        elf = cfg+"/"+NAME+".elf"
        attrs["org.eclipse.cdt.launch.PROGRAM_NAME"].set("value", elf)
        attrs["org.eclipse.cdt.launch.PROJECT_BUILD_CONFIG_ID_ATTR"].set("value", expanded.get("id"))
        attrs["org.eclipse.cdt.launch.PROJECT_BUILD_CONFIG_AUTO_ATTR"].set("value", "false")
        key = "com.st.stm32cube.ide.mcu.debug.launch.loadList"
        loads = json.loads(attrs[key].get("value"))
        for item in loads["fItems"]:
            item["fPath"] = elf
        attrs[key].set("value", json.dumps(loads, separators=(",", ":")))
        for key, e in attrs.items():
            if key.endswith("stlink.log_file"):
                e.set("value", str(SOURCE / cfg / "st-link_gdbserver_log.txt"))
        (PROJECT / launch_name).write_bytes(b'<?xml version="1.0" encoding="UTF-8"?>\n'+ET.tostring(launch, encoding="utf-8")+b"\n")
        destination = OUT / ("Validacion_"+label)
        destination.mkdir(parents=True)
        for rel in ("validar_extraccion.py", "ventanas_desarrollo.npz", "metadata_ventanas.csv", "PROCEDENCIA_VECTORES.json"):
            shutil.copy2(PRIOR / "Validacion_PC" / rel, destination / rel)
            if sha(destination / rel) != sha(PRIOR / "Validacion_PC" / rel):
                raise RuntimeError("Cliente/vector diferente")
    (PROJECT / ".cproject").write_bytes(b'<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n'
        b'<?fileVersion 4.0.0?>\n'+ET.tostring(tree, encoding="utf-8")+b"\n")
    after = configs(PROJECT / ".cproject")[2]
    if not all(ET.tostring(c) == ET.tostring(after[n]) for n, c in before.items()):
        raise RuntimeError("Configuracion anterior modificada")
    save("PREPARACION.json", dict(source_inventory_before=snapshot, existing_elf_sha256=elves,
        variants={k: dict(mode=v[0], client_folder="Validacion_"+v[1]) for k,v in VARIANTS.items()},
        control=CONTROL, source=str(SOURCE), previous_six_configurations_preserved=True,
        hclk_hz=180000000, optimization="-O3", profiling_enabled=False, fast_math_enabled=False,
        hardware_flashed=False, physical_validation_pending=True, source_patch_pending=True,
        numeric_change="EntD: misma entropia normalizada en base natural. EntM: EntD mas logf, probabilidad/acumulacion double y fallback para extremos."))
    print(f"PREPARADO: {len(snapshot)} recursos; dos configuraciones; aplicar parche C.", flush=True)


def build():
    prep = load("PREPARACION.json")
    check_origin(prep)
    tests = load("PRUEBAS_NUMERICAS_HOST.json")
    if tests["status"] != "PASS" or tests["source_c_sha256"] != sha(PROJECT / "Src/cwru_feature_extraction.c"):
        raise RuntimeError("Faltan pruebas del parche actual")
    if (OUT / "workspace_aislado").exists():
        raise FileExistsError("No sobrescribir build")
    for rel in FILES:
        dest = OVERLAY / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PROJECT / rel, dest)
    tool = next((IDE / "plugins").glob("com.st.stm32cube.ide.mcu.externaltools.gnu-tools-for-stm32.*/tools/bin"))
    make = next((IDE / "plugins").glob("com.st.stm32cube.ide.mcu.externaltools.make.*/tools/bin"))
    env = dict(os.environ)
    env["PATH"] = str(tool)+os.pathsep+str(make)+os.pathsep+env.get("PATH", "")
    selected = ["Release_180MHz_O3", CONTROL, *VARIANTS]
    args = [str(IDE / "stm32cubeidec.exe"), "--launcher.suppressErrors", "-nosplash", "-application",
        "org.eclipse.cdt.managedbuilder.core.headlessbuild", "-data", str(OUT / "workspace_aislado"),
        "-import", str(PROJECT), "-no-indexer"]
    for cfg in selected:
        args.extend(("-build", NAME+"/"+cfg))
    start = time.monotonic()
    with (OUT / "CUBEIDE_ENTROPIA.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(args, cwd=OUT, env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, encoding="utf-8", errors="replace")
        for line in process.stdout:
            log.write(line); log.flush()
            if any(s in line.lower() for s in ("build of", "error:", "warning:", "build finished")):
                print(line.strip()[:350], flush=True)
        code = process.wait(timeout=30)
    controls = {cfg: flash_image(BACKUP / cfg / (NAME+".elf")) == flash_image(PROJECT / cfg / (NAME+".elf")) for cfg in selected[:2]}
    logtext = (OUT / "CUBEIDE_ENTROPIA.log").read_text(encoding="utf-8")
    report = dict(returncode=code, elapsed_seconds=time.monotonic()-start, command=args,
        zero_errors_warnings_four=logtext.count("0 errors, 0 warnings") == 4,
        control_flash_images_identical=controls, configurations={}, hardware_flashed=False,
        physical_validation_pending=True)
    for cfg in selected:
        elf = PROJECT / cfg / (NAME+".elf")
        size = subprocess.check_output([str(tool / "arm-none-eabi-size.exe"), str(elf)], text=True)
        t, d, b = map(int, size.strip().splitlines()[-1].split()[:3])
        commands = [line for mk in (PROJECT / cfg).rglob("subdir.mk")
            for line in mk.read_text(encoding="utf-8").splitlines() if "arm-none-eabi-gcc" in line and "-std=gnu11" in line]
        expected_mode = VARIANTS[cfg][0] if cfg in VARIANTS else 0
        flags_ok = bool(commands) and all(re.findall(r"(?<!\S)-O\S+", line) == ["-O3"]
            and "-DCWRU_TARGET_HCLK_HZ=180000000U" in line
            and ("-DCWRU_CACHE_SPECTRAL_POWER=1" in line) == (cfg != selected[0])
            and re.findall(r"-DCWRU_ENTROPY_MODE=(\d+)", line) == ([str(expected_mode)] if expected_mode else [])
            and not any(f in line for f in ("CWRU_PROFILE_EXTRACTOR=1", "-ffast-math", "-Ofast", "-funsafe-math-optimizations")) for line in commands)
        image = flash_image(elf)
        span = max(a+len(data) for a, data in image)-min(a for a, _ in image)
        symbols = subprocess.check_output([str(tool / "arm-none-eabi-nm.exe"), str(elf)], text=True)
        report["configurations"][cfg] = dict(elf_sha256=sha(elf), text=t, data=d, bss=b,
            flash_load_bytes=t+d, ram_static_with_linker_reservations_bytes=d+b,
            flash_span_bytes=span, expected_download_kib=span/1024, flags_confirmed=flags_ok,
            logf_symbol_linked=bool(re.search(r"\b[Tt] logf$", symbols, re.M)),
            extractor_compiler_stack_usage=(PROJECT / cfg / "Src/cwru_feature_extraction.su").read_text(encoding="utf-8").splitlines())
    includes = ["-I"+str(PROJECT / p) for p in ("Inc", "Drivers/CMSIS/Include", "Drivers/CMSIS/Device/ST/STM32F4xx/Include",
        "Middlewares/CMSIS-DSP/Include", "Middlewares/CMSIS-DSP/PrivateInclude")]
    cpp = [str(tool / "arm-none-eabi-gcc.exe"), "-E", "-P", "-std=gnu11", "-mcpu=cortex-m4",
        "-mfpu=fpv4-sp-d16", "-mfloat-abi=hard", "-mthumb", "-DSTM32F446xx", "-DCWRU_ENTROPY_MODE=0", *includes]
    isolation = {}
    for cache in (0, 1):
        flags = [*cpp, "-DCWRU_CACHE_SPECTRAL_POWER="+str(cache)]
        before = subprocess.check_output([*flags, str(BACKUP / "Src/cwru_feature_extraction.c")])
        after = subprocess.check_output([*flags, str(PROJECT / "Src/cwru_feature_extraction.c")])
        isolation[str(cache)] = before == after
    report["mode0_preprocessed_source_identical_with_cache_disabled_and_enabled"] = isolation
    allowed = {".cproject", "Src/cwru_feature_extraction.c", ".settings/language.settings.xml"}
    report["all_other_copied_resources_unchanged"] = all(sha(PROJECT / r["path"]) == r["sha256"] for r in prep["source_inventory_before"] if r["path"] not in allowed)
    report["overlay_sha256"] = {r: sha(OVERLAY / r) for r in FILES}
    check_origin(prep)
    passed = (code == 0 and all(controls.values()) and all(isolation.values()) and report["zero_errors_warnings_four"]
        and report["all_other_copied_resources_unchanged"] and all(v["flags_confirmed"] for v in report["configurations"].values())
        and report["configurations"]["Release_180MHz_EntM"]["logf_symbol_linked"])
    report["status"] = "PASS" if passed else "FAIL"
    save("COMPILACION_ENTROPIA.json", report)
    print(json.dumps(report, ensure_ascii=True, indent=2), flush=True)
    if not passed:
        raise SystemExit(2)


def install():
    prep, built = load("PREPARACION.json"), load("COMPILACION_ENTROPIA.json")
    if (OUT / "APLICACION_EN_D.json").exists() or any((SOURCE / p).exists() for p in LAUNCHES):
        raise FileExistsError("Destino ya instalado")
    check_origin(prep)
    if built["status"] != "PASS":
        raise RuntimeError("Compilacion no validada")
    for rel, digest in built["overlay_sha256"].items():
        if sha(OVERLAY / rel) != digest:
            raise RuntimeError("Overlay modificado")
    for rel in FILES:
        shutil.copy2(OVERLAY / rel, SOURCE / rel)
    before, after = configs(BACKUP / ".cproject")[2], configs(SOURCE / ".cproject")[2]
    old_same = all(ET.tostring(c) == ET.tostring(after[n]) for n, c in before.items())
    allowed = {".cproject", "Src/cwru_feature_extraction.c"}
    others_same = all(sha(SOURCE / r["path"]) == r["sha256"] for r in prep["source_inventory_before"] if r["path"] not in allowed)
    elfs_same = all(sha(SOURCE / cfg / (NAME+".elf")) == digest for cfg, digest in prep["existing_elf_sha256"].items())
    report = dict(status="PASS" if old_same and others_same and elfs_same else "FAIL",
        modified_files=sorted(allowed), added_files=LAUNCHES,
        previous_six_configurations_preserved=old_same, previous_six_elfs_unchanged=elfs_same,
        all_other_snapshot_resources_unchanged=others_same, installed_sha256={r: sha(SOURCE / r) for r in FILES},
        hardware_flashed=False, physical_validation_pending=True, user_workspace_modified=False)
    save("APLICACION_EN_D.json", report)
    print(json.dumps(report, ensure_ascii=True, indent=2), flush=True)
    if report["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "build", "install"))
    {"prepare": prepare, "build": build, "install": install}[parser.parse_args().mode]()
