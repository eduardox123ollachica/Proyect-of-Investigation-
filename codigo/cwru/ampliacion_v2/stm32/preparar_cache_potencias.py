"""Variante opcional: reutiliza potencias double sin reservar otro buffer.

No modifica D: salvo en install (con permiso). Nunca abre el puerto serie ni
programa la placa. prepare copia; el parche C se aplica luego con apply_patch.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[4]
NAME = "Prueba_2_CWRU_Extraccion"
CONTROL = "Release_180MHz_O3"
CONFIG = "Release_180MHz_O3_Cache"
SOURCE = Path(r"D:\STM32IDE") / NAME
PREVIOUS = ROOT / "despliegue_stm32/prueba_o3_180mhz_2026-10-23"
OUT = ROOT / "despliegue_stm32/cache_potencias_180mhz_2026-10-23"
PROJECT = OUT / "copia_aislada" / NAME
BACKUP = OUT / "respaldo_antes_cache"
OVERLAY = OUT / "archivos_para_D"
IDE = Path(r"D:\STM32CubeIDE_2.2.0\STM32CubeIDE")
LAUNCH = NAME + "_Cache_180MHz.launch"
FILES = (".cproject", "Src/cwru_feature_extraction.c", LAUNCH)
FLAG = "CWRU_CACHE_SPECTRAL_POWER=1"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, obj):
    path = OUT / name
    if path.exists():
        raise FileExistsError("No sobrescribir evidencia: " + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")


def load(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def configs(path):
    tree = ET.fromstring(path.read_bytes())
    settings = tree.find("storageModule[@moduleId='org.eclipse.cdt.core.settings']")
    return tree, settings, {c.find("storageModule").get("name"): c for c in settings.findall("cconfiguration")}


def check_source(prep):
    for row in prep["source_inventory_before"]:
        if sha(SOURCE / row["path"]) != row["sha256"]:
            raise RuntimeError("Origen cambio desde el respaldo: " + row["path"])


def prepare():
    if PROJECT.exists() or (SOURCE / LAUNCH).exists():
        raise FileExistsError("Preparacion/destino ya existe")
    prior = json.loads((PREVIOUS / "PREPARACION.json").read_text(encoding="utf-8"))
    validation = json.loads((PREVIOUS / "RESULTADOS_FISICOS_O3_42_VENTANAS.json").read_text(encoding="utf-8"))
    if validation["validation"]["passed_windows"] != 42:
        raise RuntimeError("Falta control -O3 validado")
    critical = next(r for r in prior["source_inventory_before"] if r["path"] == "Src/cwru_feature_extraction.c")
    if sha(SOURCE / critical["path"]) != critical["sha256"]:
        raise RuntimeError("El extractor cambio desde la validacion -O3")
    paths = {r["path"] for r in prior["source_inventory_before"]}
    paths.update(p.name for p in SOURCE.glob("*.launch"))
    snapshot = []
    for rel in sorted(paths):
        digest = sha(SOURCE / rel)
        for base in (PROJECT, BACKUP):
            destination = base / rel
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(SOURCE / rel, destination)
            if sha(destination) != digest or sha(SOURCE / rel) != digest:
                raise RuntimeError("Copia no identica: " + rel)
        snapshot.append(dict(path=rel, sha256=digest))
    tree, settings, before = configs(PROJECT / ".cproject")
    if len(before) != 5 or CONTROL not in before or CONFIG in before:
        raise RuntimeError("Configuraciones inesperadas")
    elf_hashes = {}
    for cfg in before:
        source = SOURCE / cfg / (NAME + ".elf")
        target = BACKUP / cfg / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        elf_hashes[cfg] = sha(source)
        if sha(target) != elf_hashes[cfg]:
            raise RuntimeError("Respaldo ELF no identico")
    expanded = copy.deepcopy(before[CONTROL])
    mapping = {e.get("id"): e.get("id")+".cachepower" for e in expanded.iter() if e.get("id")}
    for element in expanded.iter():
        for key, value in list(element.attrib.items()):
            if value in mapping:
                element.set(key, mapping[value])
            elif key == "name" and value == CONTROL:
                element.set(key, CONFIG)
            elif key == "buildPath" and value.endswith("/"+CONTROL):
                element.set(key, value[:-len(CONTROL)]+CONFIG)
    for tool in expanded.findall(".//tool"):
        if tool.get("superClass", "").endswith(("tool.c.compiler", "tool.cpp.compiler")):
            optimization = next(o for o in tool.findall("option") if o.get("superClass", "").endswith("optimization.level"))
            if not optimization.get("value", "").endswith(".o3"):
                raise RuntimeError("Control no usa -O3")
            defines = next(o for o in tool.findall("option") if o.get("superClass", "").endswith(".definedsymbols"))
            ET.SubElement(defines, "listOptionValue", builtIn="false", value=FLAG)
    old_ids = {e.get("id") for c in before.values() for e in c.iter() if e.get("id")}
    if old_ids & {e.get("id") for e in expanded.iter() if e.get("id")}:
        raise RuntimeError("Colision de IDs CDT")
    settings.append(expanded)
    (PROJECT / ".cproject").write_bytes(b'<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n'
        b'<?fileVersion 4.0.0?>\n'+ET.tostring(tree, encoding="utf-8")+b"\n")
    after = configs(PROJECT / ".cproject")[2]
    if not all(ET.tostring(c) == ET.tostring(after[n]) for n, c in before.items()):
        raise RuntimeError("Se altero una configuracion anterior")
    launch = ET.fromstring((SOURCE / (NAME+"_O3_180MHz.launch")).read_bytes())
    attrs = {e.get("key"): e for e in launch}
    elf_rel = CONFIG+"/"+NAME+".elf"
    attrs["org.eclipse.cdt.launch.PROGRAM_NAME"].set("value", elf_rel)
    attrs["org.eclipse.cdt.launch.PROJECT_BUILD_CONFIG_ID_ATTR"].set("value", expanded.get("id"))
    attrs["org.eclipse.cdt.launch.PROJECT_BUILD_CONFIG_AUTO_ATTR"].set("value", "false")
    key = "com.st.stm32cube.ide.mcu.debug.launch.loadList"
    loads = json.loads(attrs[key].get("value"))
    for row in loads["fItems"]:
        row["fPath"] = elf_rel
    attrs[key].set("value", json.dumps(loads, separators=(",", ":")))
    for key, element in attrs.items():
        if key.endswith("stlink.log_file"):
            element.set("value", str(SOURCE / CONFIG / "st-link_gdbserver_log.txt"))
    (PROJECT / LAUNCH).write_bytes(b'<?xml version="1.0" encoding="UTF-8"?>\n'+ET.tostring(launch, encoding="utf-8")+b"\n")
    destination = OUT / "Validacion_PC"
    destination.mkdir(parents=True)
    for rel in ("validar_extraccion.py", "ventanas_desarrollo.npz", "metadata_ventanas.csv", "PROCEDENCIA_VECTORES.json"):
        shutil.copy2(PREVIOUS / "Validacion_PC" / rel, destination / rel)
        if sha(destination / rel) != sha(PREVIOUS / "Validacion_PC" / rel):
            raise RuntimeError("Cliente/vector no identico")
    save("PREPARACION.json", dict(source=str(SOURCE), config=CONFIG, control=CONTROL,
        source_inventory_before=snapshot, existing_elf_sha256=elf_hashes,
        previous_five_configurations_preserved=True, flag=FLAG, optimization="-O3",
        hclk_hz=180000000, profiling_enabled=False, fast_math_enabled=False,
        source_patch_pending=True, hardware_flashed=False, physical_validation_pending=True,
        numeric_change="Cache de potencias double por copia de bytes; formulas y orden de sumas conservados.",
        canonical_o3_run="20261023T202133_701975Z"))
    print(f"PREPARADO: {len(snapshot)} recursos respaldados. Aplicar parche C antes de build.", flush=True)


def build():
    prep = load("PREPARACION.json")
    check_source(prep)
    if "cache_power_store(" not in (PROJECT / "Src/cwru_feature_extraction.c").read_text(encoding="utf-8"):
        raise RuntimeError("Falta parche C")
    tested = load("PRUEBAS_CACHE_SOFTWARE.json")
    if tested["status"] != "PASS" or tested["source_c_sha256"] != sha(PROJECT / "Src/cwru_feature_extraction.c"):
        raise RuntimeError("Pruebas logicas pendientes")
    if (OUT / "workspace_aislado").exists():
        raise FileExistsError("No sobrescribir compilacion")
    for rel in FILES:
        target = OVERLAY / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PROJECT / rel, target)
    tool = next((IDE / "plugins").glob("com.st.stm32cube.ide.mcu.externaltools.gnu-tools-for-stm32.*/tools/bin"))
    make = next((IDE / "plugins").glob("com.st.stm32cube.ide.mcu.externaltools.make.*/tools/bin"))
    env = dict(os.environ)
    env["PATH"] = str(tool)+os.pathsep+str(make)+os.pathsep+env.get("PATH", "")
    configs_to_build = ("Release_180MHz", CONTROL, CONFIG)
    args = [str(IDE / "stm32cubeidec.exe"), "--launcher.suppressErrors", "-nosplash", "-application",
        "org.eclipse.cdt.managedbuilder.core.headlessbuild", "-data", str(OUT / "workspace_aislado"),
        "-import", str(PROJECT), "-no-indexer"]
    for cfg in configs_to_build:
        args.extend(("-build", NAME+"/"+cfg))
    start = time.monotonic()
    with (OUT / "CUBEIDE_CACHE.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(args, cwd=OUT, env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, encoding="utf-8", errors="replace")
        for line in process.stdout:
            log.write(line); log.flush()
            if any(s in line.lower() for s in ("build of", "error:", "warning:", "build finished")):
                print(line.strip()[:350], flush=True)
        code = process.wait(timeout=30)
    spec = importlib.util.spec_from_file_location("clock_helper", Path(__file__).with_name("preparar_reloj_180mhz.py"))
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    controls = {cfg: helper.flash_image(BACKUP / cfg / (NAME+".elf")) == helper.flash_image(PROJECT / cfg / (NAME+".elf"))
                for cfg in configs_to_build[:2]}
    logtext = (OUT / "CUBEIDE_CACHE.log").read_text(encoding="utf-8")
    report = dict(returncode=code, elapsed_seconds=time.monotonic()-start,
        zero_errors_warnings_three=logtext.count("0 errors, 0 warnings") == 3,
        control_flash_images_identical=controls, configurations={}, hardware_flashed=False,
        physical_validation_pending=True, command=args)
    for cfg in configs_to_build:
        elf = PROJECT / cfg / (NAME+".elf")
        size = subprocess.check_output([str(tool / "arm-none-eabi-size.exe"), str(elf)], text=True)
        t, d, b = map(int, size.strip().splitlines()[-1].split()[:3])
        commands = [line for mk in (PROJECT / cfg).rglob("subdir.mk")
            for line in mk.read_text(encoding="utf-8").splitlines() if "arm-none-eabi-gcc" in line and "-std=gnu11" in line]
        expected_opt = "-Os" if cfg == "Release_180MHz" else "-O3"
        flags_ok = bool(commands) and all(re.findall(r"(?<!\S)-O\S+", c) == [expected_opt]
            and (FLAG in c) == (cfg == CONFIG) and "-DCWRU_TARGET_HCLK_HZ=180000000U" in c
            and "CWRU_PROFILE_EXTRACTOR=1" not in c
            and not any(f in c for f in ("-ffast-math", "-Ofast", "-funsafe-math-optimizations")) for c in commands)
        image = helper.flash_image(elf)
        span = max(a+len(data) for a, data in image)-min(a for a, _ in image)
        su = (PROJECT / cfg / "Src/cwru_feature_extraction.su").read_text(encoding="utf-8").splitlines()
        report["configurations"][cfg] = dict(elf_sha256=sha(elf), text=t, data=d, bss=b,
            flash_load_bytes=t+d, ram_static_with_linker_reservations_bytes=d+b,
            flash_span_bytes=span, expected_download_kib=span/1024, flags_confirmed=flags_ok,
            extractor_compiler_stack_usage=su)
    include_flags = ["-I"+str(PROJECT / p) for p in ("Inc", "Drivers/CMSIS/Include",
        "Drivers/CMSIS/Device/ST/STM32F4xx/Include", "Middlewares/CMSIS-DSP/Include",
        "Middlewares/CMSIS-DSP/PrivateInclude")]
    cpp = [str(tool / "arm-none-eabi-gcc.exe"), "-E", "-P", "-std=gnu11", "-mcpu=cortex-m4",
        "-mfpu=fpv4-sp-d16", "-mfloat-abi=hard", "-mthumb", "-DSTM32F446xx", *include_flags]
    originals = subprocess.check_output([*cpp, str(BACKUP / "Src/cwru_feature_extraction.c")])
    disabled = subprocess.check_output([*cpp, str(PROJECT / "Src/cwru_feature_extraction.c")])
    report["disabled_cache_preprocessed_source_identical"] = originals == disabled
    report["preprocessed_disabled_sha256"] = hashlib.sha256(disabled).hexdigest()
    allowed = {".cproject", "Src/cwru_feature_extraction.c", ".settings/language.settings.xml"}
    report["all_other_copied_resources_unchanged"] = all(sha(PROJECT / r["path"]) == r["sha256"]
        for r in prep["source_inventory_before"] if r["path"] not in allowed)
    report["overlay_sha256"] = {r: sha(OVERLAY / r) for r in FILES}
    check_source(prep)
    passed = (code == 0 and all(controls.values()) and report["zero_errors_warnings_three"]
        and report["disabled_cache_preprocessed_source_identical"] and report["all_other_copied_resources_unchanged"]
        and all(v["flags_confirmed"] for v in report["configurations"].values()))
    report["status"] = "PASS" if passed else "FAIL"
    save("COMPILACION_CACHE.json", report)
    print(json.dumps(report, ensure_ascii=True, indent=2), flush=True)
    if not passed:
        raise SystemExit(2)


def install():
    prep, built = load("PREPARACION.json"), load("COMPILACION_CACHE.json")
    if (OUT / "APLICACION_EN_D.json").exists() or (SOURCE / LAUNCH).exists():
        raise FileExistsError("Destino ya instalado/ambiguo")
    check_source(prep)
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
    untouched = all(sha(SOURCE / r["path"]) == r["sha256"] for r in prep["source_inventory_before"] if r["path"] not in allowed)
    elfs_same = all(sha(SOURCE / cfg / (NAME+".elf")) == digest for cfg, digest in prep["existing_elf_sha256"].items())
    report = dict(status="PASS" if old_same and untouched and elfs_same else "FAIL",
        modified_files=sorted(allowed), added_files=[LAUNCH], installed_sha256={r: sha(SOURCE / r) for r in FILES},
        previous_five_configurations_preserved=old_same, other_snapshot_resources_unchanged=untouched,
        previous_five_elfs_unchanged=elfs_same, user_workspace_modified=False,
        hardware_flashed=False, physical_validation_pending=True)
    save("APLICACION_EN_D.json", report)
    print(json.dumps(report, ensure_ascii=True, indent=2), flush=True)
    if report["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "build", "install"))
    {"prepare": prepare, "build": build, "install": install}[parser.parse_args().mode]()
