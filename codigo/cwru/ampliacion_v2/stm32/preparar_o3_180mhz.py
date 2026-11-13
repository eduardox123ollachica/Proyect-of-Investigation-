"""Comparacion controlada -Os/-O3, sin cambios C ni programacion de hardware.

prepare copia los recursos actuales de D: y genera una configuracion adicional.
build usa CubeIDE en un workspace aislado. install requiere permiso para D: y
solo copia .cproject y un nuevo .launch, tras validar compilacion y respaldos.
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
CONFIG = "Release_180MHz_O3"
CONTROL = "Release_180MHz"
SOURCE = Path(r"D:\STM32IDE") / NAME
PREVIOUS = ROOT / "despliegue_stm32/perfilado_extraccion_180mhz_2026-10-23"
OUT = ROOT / "despliegue_stm32/prueba_o3_180mhz_2026-10-23"
PROJECT = OUT / "copia_aislada" / NAME
BACKUP = OUT / "respaldo_antes_o3"
OVERLAY = OUT / "archivos_para_D"
IDE = Path(r"D:\STM32CubeIDE_2.2.0\STM32CubeIDE")
LAUNCH = NAME + "_O3_180MHz.launch"
FILES = (".cproject", LAUNCH)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, value):
    path = OUT / name
    if path.exists():
        raise FileExistsError("No sobrescribir evidencias: " + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def configurations(path):
    tree = ET.fromstring(path.read_bytes())
    settings = tree.find("storageModule[@moduleId='org.eclipse.cdt.core.settings']")
    return tree, settings, {c.find("storageModule").get("name"): c
                            for c in settings.findall("cconfiguration")}


def check_originals(prep):
    for row in prep["source_inventory_before"]:
        if sha(SOURCE / row["path"]) != row["sha256"]:
            raise RuntimeError("Origen cambio desde el respaldo: " + row["path"])


def prepare():
    if PROJECT.exists() or (OUT / "PREPARACION.json").exists():
        raise FileExistsError("La preparacion ya existe")
    if (SOURCE / LAUNCH).exists():
        raise FileExistsError("Ya existe el .launch de destino")
    previous = json.loads((PREVIOUS / "PREPARACION.json").read_text(encoding="utf-8"))
    paths = {r["path"] for r in previous["source_inventory_before"]}
    paths.add("Inc/cwru_extraction_profile.h")
    paths.update(p.name for p in SOURCE.glob("*.launch"))
    snapshot = []
    for rel in sorted(paths):
        source = SOURCE / rel
        digest = sha(source)
        for base in (PROJECT, BACKUP):
            target = base / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            if sha(target) != digest or sha(source) != digest:
                raise RuntimeError("Copia no identica: " + rel)
        snapshot.append(dict(path=rel, sha256=digest))
    elfs = {}
    for cfg in ("Debug", "Release", CONTROL, "Release_180MHz_Perfil"):
        source = SOURCE / cfg / (NAME + ".elf")
        target = BACKUP / cfg / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        digest = sha(source)
        shutil.copy2(source, target)
        if sha(target) != digest:
            raise RuntimeError("Respaldo ELF no identico")
        elfs[cfg] = digest

    tree, settings, configs = configurations(PROJECT / ".cproject")
    if len(configs) != 4 or CONTROL not in configs or CONFIG in configs:
        raise RuntimeError("Configuraciones inesperadas; revisar antes de continuar")
    clone = copy.deepcopy(configs[CONTROL])
    mapping = {e.get("id"): e.get("id") + ".o3" for e in clone.iter() if e.get("id")}
    for element in clone.iter():
        for key, value in list(element.attrib.items()):
            if value in mapping:
                element.set(key, mapping[value])
            elif key == "name" and value == CONTROL:
                element.set(key, CONFIG)
            elif key == "buildPath" and value.endswith("/" + CONTROL):
                element.set(key, value[:-len(CONTROL)] + CONFIG)
    changes = 0
    for option in clone.findall(".//option"):
        if option.get("superClass", "").endswith("optimization.level"):
            value = option.get("value", "")
            if not value.endswith(".os"):
                raise RuntimeError("Control no usa -Os")
            option.set("value", value[:-2] + "o3")
            changes += 1
    if changes != 2:
        raise RuntimeError("Se requieren niveles C/C++ explicitos")
    defines = [e.get("value", "") for e in clone.findall(".//listOptionValue")]
    if "CWRU_PROFILE_EXTRACTOR=1" in defines or "CWRU_TARGET_HCLK_HZ=180000000U" not in defines:
        raise RuntimeError("Reloj o instrumentacion incorrectos")
    settings.append(clone)
    cproject = PROJECT / ".cproject"
    cproject.write_bytes(b'<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n'
                        b'<?fileVersion 4.0.0?>\n' + ET.tostring(tree, encoding="utf-8") + b"\n")
    _, _, after = configurations(cproject)
    if not all(ET.tostring(c) == ET.tostring(after[n]) for n, c in configs.items()):
        raise RuntimeError("Cambio una configuracion anterior")
    # CDT repite legitimamente el ID de una configuracion en sus tres nodos.
    # Se exige que la nueva configuracion no reutilice IDs de las anteriores.
    old_ids = {e.get("id") for c in configs.values() for e in c.iter() if e.get("id")}
    new_ids = {e.get("id") for e in clone.iter() if e.get("id")}
    if old_ids & new_ids:
        raise RuntimeError("La configuracion nueva reutiliza un ID anterior")

    launch = ET.fromstring((SOURCE / (NAME + "_Release_180MHz.launch")).read_bytes())
    attrs = {e.get("key"): e for e in launch}
    elf_rel = CONFIG + "/" + NAME + ".elf"
    attrs["org.eclipse.cdt.launch.PROGRAM_NAME"].set("value", elf_rel)
    attrs["org.eclipse.cdt.launch.PROJECT_BUILD_CONFIG_ID_ATTR"].set("value", clone.get("id"))
    attrs["org.eclipse.cdt.launch.PROJECT_BUILD_CONFIG_AUTO_ATTR"].set("value", "false")
    key = "com.st.stm32cube.ide.mcu.debug.launch.loadList"
    loads = json.loads(attrs[key].get("value"))
    for item in loads["fItems"]:
        item["fPath"] = elf_rel
    attrs[key].set("value", json.dumps(loads, separators=(",", ":")))
    for key, attr in attrs.items():
        if key.endswith("stlink.log_file"):
            attr.set("value", str(SOURCE / CONFIG / "st-link_gdbserver_log.txt"))
    (PROJECT / LAUNCH).write_bytes(b'<?xml version="1.0" encoding="UTF-8"?>\n'
                                  + ET.tostring(launch, encoding="utf-8") + b"\n")
    OVERLAY.mkdir(parents=True)
    for rel in FILES:
        shutil.copy2(PROJECT / rel, OVERLAY / rel)
    save("PREPARACION.json", dict(source=str(SOURCE), config=CONFIG, control=CONTROL,
        new_configuration_id=clone.get("id"), source_inventory_before=snapshot,
        existing_elf_sha256=elfs, overlay_sha256={r: sha(OVERLAY / r) for r in FILES},
        intended_hclk_hz=180000000, optimization="-O3", control_optimization="-Os",
        profiling_enabled=False, fast_math_enabled=False, formulas_and_source_types_changed=False,
        previous_four_configurations_preserved=True, hardware_flashed=False,
        canonical_control_run="20261023T184141_575962Z"))
    print(f"PREPARADO: {len(snapshot)} recursos respaldados; solo nueva configuracion y launch.", flush=True)


def build():
    prep = load("PREPARACION.json")
    check_originals(prep)
    workspace = OUT / "workspace_aislado"
    if workspace.exists() or (OUT / "COMPILACION_O3.json").exists():
        raise FileExistsError("No sobrescribir compilacion")
    tool = next((IDE / "plugins").glob("com.st.stm32cube.ide.mcu.externaltools.gnu-tools-for-stm32.*/tools/bin"))
    make = next((IDE / "plugins").glob("com.st.stm32cube.ide.mcu.externaltools.make.*/tools/bin"))
    env = dict(os.environ)
    env["PATH"] = str(tool) + os.pathsep + str(make) + os.pathsep + env.get("PATH", "")
    args = [str(IDE / "stm32cubeidec.exe"), "--launcher.suppressErrors", "-nosplash",
            "-application", "org.eclipse.cdt.managedbuilder.core.headlessbuild", "-data", str(workspace),
            "-import", str(PROJECT), "-no-indexer", "-build", NAME + "/" + CONTROL,
            "-build", NAME + "/" + CONFIG]
    start = time.monotonic()
    with (OUT / "CUBEIDE_O3.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(args, cwd=OUT, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, encoding="utf-8", errors="replace")
        for line in process.stdout:
            log.write(line)
            log.flush()
            if any(s in line.lower() for s in ("build of", "error:", "warning:", "build finished")):
                print(line.strip()[:350], flush=True)
        code = process.wait(timeout=30)
    spec = importlib.util.spec_from_file_location("clock_helper", Path(__file__).with_name("preparar_reloj_180mhz.py"))
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    logtext = (OUT / "CUBEIDE_O3.log").read_text(encoding="utf-8")
    controls_equal = helper.flash_image(BACKUP / CONTROL / (NAME + ".elf")) == helper.flash_image(PROJECT / CONTROL / (NAME + ".elf"))
    report = dict(returncode=code, elapsed_seconds=time.monotonic()-start, command=args,
        zero_errors_warnings_both=logtext.count("0 errors, 0 warnings") == 2,
        control_flash_identical=controls_equal, configurations={}, hardware_flashed=False,
        hardware_validation_pending=True)
    for cfg, opt in ((CONTROL, "-Os"), (CONFIG, "-O3")):
        elf = PROJECT / cfg / (NAME + ".elf")
        size = subprocess.check_output([str(tool / "arm-none-eabi-size.exe"), str(elf)], text=True)
        t, d, b = map(int, size.strip().splitlines()[-1].split()[:3])
        commands = []
        for mk in (PROJECT / cfg).rglob("subdir.mk"):
            commands.extend(line for line in mk.read_text(encoding="utf-8").splitlines()
                            if "arm-none-eabi-gcc" in line and "-std=gnu11" in line)
        forbidden = ("-Ofast", "-ffast-math", "-funsafe-math-optimizations", "-fassociative-math",
                     "-freciprocal-math", "-ffinite-math-only", "-fsingle-precision-constant")
        flags_ok = bool(commands) and all(re.findall(r"(?<!\S)-O\S+", line) == [opt]
            and "-DCWRU_TARGET_HCLK_HZ=180000000U" in line
            and "CWRU_PROFILE_EXTRACTOR=1" not in line
            and not any(flag in line for flag in forbidden) for line in commands)
        sources_same = all(sha(PROJECT / row["path"]) == row["sha256"]
                           for row in prep["source_inventory_before"]
                           if row["path"] not in (".cproject", ".settings/language.settings.xml"))
        image = helper.flash_image(elf)
        flash_span = max(address + len(data) for address, data in image) - min(a for a, _ in image)
        report["configurations"][cfg] = dict(elf_sha256=sha(elf), text=t, data=d, bss=b,
            flash_load_bytes=t+d, ram_static_with_linker_reservations_bytes=d+b,
            flash_span_bytes=flash_span, expected_download_kib=flash_span/1024,
            compiler_command_templates=len(commands), flags_confirmed=flags_ok,
            copied_sources_unchanged=sources_same)
    symbols = subprocess.check_output([str(tool / "arm-none-eabi-nm.exe"),
                                       str(PROJECT / CONFIG / (NAME + ".elf"))], text=True)
    report["no_profile_symbols_in_o3_elf"] = not any(s in symbols for s in
        ("g_profile", "profile_clock_read", "CWRU_FeatureExtraction_GetProfile"))
    check_originals(prep)
    save("COMPILACION_O3.json", report)
    print(json.dumps(report, ensure_ascii=True, indent=2), flush=True)
    if (code or not controls_equal or not report["zero_errors_warnings_both"]
            or not report["no_profile_symbols_in_o3_elf"]
            or not all(v["flags_confirmed"] and v["copied_sources_unchanged"]
                       for v in report["configurations"].values())):
        raise SystemExit(2)


def verify():
    """Audita una compilacion ya hecha, sin repetirla ni alterar su registro."""
    prep = load("PREPARACION.json")
    report = load("COMPILACION_O3.json")
    check_originals(prep)
    mismatches = [row["path"] for row in prep["source_inventory_before"]
                  if row["path"] != ".cproject" and sha(PROJECT / row["path"]) != row["sha256"]]
    allowed = {".settings/language.settings.xml"}
    sources_same = not (set(mismatches) - allowed)
    for values in report["configurations"].values():
        values["copied_sources_unchanged"] = sources_same
    units = []
    for line in (PROJECT / CONFIG / "objects.list").read_text(encoding="utf-8").splitlines():
        rel = line.strip().strip('"').removeprefix("./")
        if not rel.endswith(".o"):
            raise RuntimeError("Objeto inesperado: " + rel)
        candidates = [PROJECT / (rel[:-2] + suffix) for suffix in (".c", ".s", ".S", ".cpp")]
        matches = list({os.path.normcase(str(p)): p for p in candidates if p.is_file()}.values())
        if len(matches) != 1:
            raise RuntimeError("Fuente de objeto ambigua: " + rel)
        path = matches[0].relative_to(PROJECT).as_posix()
        if sha(SOURCE / path) != sha(PROJECT / path):
            raise RuntimeError("Fuente compilada distinta del origen: " + path)
        units.append(dict(path=path, sha256=sha(PROJECT / path)))
    report.update(isolated_generated_metadata_changed=mismatches,
        metadata_change_not_installed_in_D=True, original_build_report_preserved=True,
        note="CubeIDE regenera language.settings.xml en la copia. No es una fuente ni se instala en D:. No se repitio la compilacion.",
        source_units=units, direct_compilation_units=len(units))
    passed = (report["returncode"] == 0 and sources_same and len(units) == 18
        and report["control_flash_identical"] and report["zero_errors_warnings_both"]
        and report["no_profile_symbols_in_o3_elf"]
        and all(v["flags_confirmed"] for v in report["configurations"].values()))
    report["status"] = "PASS" if passed else "FAIL"
    save("VERIFICACION_COMPILACION_O3.json", report)
    print(json.dumps(dict(status=report["status"], direct_compilation_units=len(units),
        generated_metadata_changed=mismatches, actual_sources_unchanged=sources_same), indent=2), flush=True)
    if not passed:
        raise SystemExit(2)


def install():
    prep = load("PREPARACION.json")
    compiled = load("VERIFICACION_COMPILACION_O3.json")
    if (OUT / "APLICACION_EN_D.json").exists() or (SOURCE / LAUNCH).exists():
        raise FileExistsError("Instalacion previa o destino ambiguo")
    check_originals(prep)
    if not (compiled["returncode"] == 0 and compiled["control_flash_identical"]
            and compiled["zero_errors_warnings_both"] and compiled["no_profile_symbols_in_o3_elf"]
            and all(v["flags_confirmed"] and v["copied_sources_unchanged"]
                    for v in compiled["configurations"].values())):
        raise RuntimeError("Compilacion no validada")
    for rel, digest in prep["overlay_sha256"].items():
        if sha(OVERLAY / rel) != digest:
            raise RuntimeError("Overlay modificado: " + rel)
    for rel in FILES:
        shutil.copy2(OVERLAY / rel, SOURCE / rel)
    original = configurations(BACKUP / ".cproject")[2]
    current = configurations(SOURCE / ".cproject")[2]
    previous_same = all(ET.tostring(c) == ET.tostring(current[n]) for n, c in original.items())
    unrelated_same = all(sha(SOURCE / row["path"]) == row["sha256"]
                         for row in prep["source_inventory_before"] if row["path"] != ".cproject")
    elfs_same = all(sha(SOURCE / cfg / (NAME + ".elf")) == digest
                    for cfg, digest in prep["existing_elf_sha256"].items())
    report = dict(status="PASS" if previous_same and unrelated_same and elfs_same else "FAIL",
        installed_sha256={r: sha(SOURCE / r) for r in FILES},
        overlay_identical=all(sha(SOURCE / r) == prep["overlay_sha256"][r] for r in FILES),
        previous_four_configurations_unchanged=previous_same, all_other_snapshot_files_unchanged=unrelated_same,
        existing_four_elfs_unchanged=elfs_same, modified_files=[".cproject"], added_files=[LAUNCH],
        source_c_or_headers_modified=False, user_workspace_modified=False, hardware_flashed=False,
        next_action="Usuario compila y carga explicitamente Release_180MHz_O3, reanuda y valida ventana 0.")
    save("APLICACION_EN_D.json", report)
    print(json.dumps(report, ensure_ascii=True, indent=2), flush=True)
    if report["status"] != "PASS" or not report["overlay_identical"]:
        raise SystemExit(2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "build", "verify", "install"))
    {"prepare": prepare, "build": build, "verify": verify, "install": install}[parser.parse_args().mode]()
