"""Prepara perfilado opcional. Solo escribe en una nueva copia del proyecto.

Las tres modificaciones C/header se aplican con apply_patch tras prepare.
build verifica controles sin instrumentacion y genera el overlay para D:.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[4]
NAME = "Prueba_2_CWRU_Extraccion"
CONFIG = "Release_180MHz_Perfil"
SOURCE = Path(r"D:\STM32IDE") / NAME
PREVIOUS = ROOT / "despliegue_stm32/prueba_reloj_180mhz_2026-10-23"
OUT = ROOT / "despliegue_stm32/perfilado_extraccion_180mhz_2026-10-23"
PROJECT = OUT / "copia_aislada" / NAME
OVERLAY = OUT / "archivos_para_D"
IDE = Path(r"D:\STM32CubeIDE_2.2.0\STM32CubeIDE")
LAUNCH = NAME + "_Perfil_180MHz.launch"
FILES = [".cproject", "Src/cwru_feature_extraction.c", "Inc/cwru_feature_extraction.h",
         "Src/cmd_cwru_feat.c", "Inc/cwru_extraction_profile.h", LAUNCH]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def save(name, obj):
    (OUT/name).write_text(json.dumps(obj, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")


def prepare():
    if PROJECT.exists() or (OUT/"PREPARACION.json").exists():
        raise FileExistsError("No sobrescribir una preparacion anterior")
    inventory = json.loads((PREVIOUS/"PREPARACION.json").read_text(encoding="utf-8"))["source_inventory_before"]
    paths = sorted({r["path"] for r in inventory} | {"Inc/cwru_clock_180mhz.inc"})
    snapshot = []
    for rel in paths:
        source = SOURCE/rel
        destination = PROJECT/rel
        backup = OUT/"respaldo_antes_perfilado"/rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        backup.parent.mkdir(parents=True, exist_ok=True)
        before = sha(source)
        shutil.copy2(source, destination)
        shutil.copy2(source, backup)
        if sha(source) != before or sha(destination) != before or sha(backup) != before:
            raise RuntimeError("Origen cambio durante copia: "+rel)
        snapshot.append(dict(path=rel, sha256=before))
    for cfg in ("Release", "Release_180MHz"):
        dest = OUT/"respaldo_antes_perfilado"/cfg/(NAME+".elf")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SOURCE/cfg/(NAME+".elf"), dest)
    shutil.copy2(Path(__file__).with_name("cwru_extraction_profile.h"), PROJECT/"Inc/cwru_extraction_profile.h")

    tree = ET.fromstring((PROJECT/".cproject").read_bytes())
    settings = tree.find("storageModule[@moduleId='org.eclipse.cdt.core.settings']")
    original = next(c for c in settings.findall("cconfiguration") if c.find("storageModule").get("name")=="Release_180MHz")
    clone = copy.deepcopy(original)
    mapping = {e.get("id"):e.get("id")+".profile" for e in clone.iter() if e.get("id")}
    for e in clone.iter():
        for key, value in list(e.attrib.items()):
            if value in mapping:
                e.set(key, mapping[value])
            elif key=="name" and value=="Release_180MHz":
                e.set(key, CONFIG)
            elif key=="buildPath" and value.endswith("/Release_180MHz"):
                e.set(key, value[:-len("Release_180MHz")]+CONFIG)
    for tool in clone.findall(".//tool"):
        if tool.get("superClass", "").endswith(("tool.c.compiler", "tool.cpp.compiler")):
            defs=next(o for o in tool.findall("option") if o.get("superClass", "").endswith(".definedsymbols"))
            ET.SubElement(defs, "listOptionValue", builtIn="false", value="CWRU_PROFILE_EXTRACTOR=1")
    settings.append(clone)
    (PROJECT/".cproject").write_bytes(b'<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n<?fileVersion 4.0.0?>\n'+ET.tostring(tree, encoding="utf-8")+b"\n")

    launch = ET.fromstring((SOURCE/(NAME+"_Release_180MHz.launch")).read_bytes())
    attrs = {e.get("key"):e for e in launch}
    attrs["org.eclipse.cdt.launch.PROGRAM_NAME"].set("value", CONFIG+"/"+NAME+".elf")
    attrs["org.eclipse.cdt.launch.PROJECT_BUILD_CONFIG_ID_ATTR"].set("value", clone.get("id"))
    attrs["org.eclipse.cdt.launch.PROJECT_BUILD_CONFIG_AUTO_ATTR"].set("value", "false")
    key = "com.st.stm32cube.ide.mcu.debug.launch.loadList"
    loads = json.loads(attrs[key].get("value"))
    for item in loads["fItems"]:
        item["fPath"] = CONFIG+"/"+NAME+".elf"
    attrs[key].set("value", json.dumps(loads,separators=(",", ":")))
    for key,e in attrs.items():
        if key.endswith("stlink.log_file"):
            e.set("value", str(SOURCE/CONFIG/"st-link_gdbserver_log.txt"))
    (OUT/LAUNCH).write_bytes(b'<?xml version="1.0" encoding="UTF-8"?>\n'+ET.tostring(launch, encoding="utf-8")+b"\n")
    save("PREPARACION.json", dict(source=str(SOURCE), source_inventory_before=snapshot,
        overlay_files=FILES, config=CONFIG, config_id=clone.get("id"),
        intended_hclk_hz=180000000, optimization="-Os", blocks_per_channel=7,
        instrumentation_optional=True, hardware_flashed=False,
        canonical_baseline_run="20261023T184141_575962Z"))
    print("PREPARADO: copia aislada y configuracion opcional; aplicar parche C antes de build.", flush=True)


def build():
    prep=json.loads((OUT/"PREPARACION.json").read_text(encoding="utf-8"))
    if "CWRU_PROFILE_MARK(CWRU_PROFILE_ENTROPY)" not in (PROJECT/"Src/cwru_feature_extraction.c").read_text(encoding="utf-8-sig"):
        raise RuntimeError("Falta parche de instrumentacion")
    if not all(sha(SOURCE/r["path"])==r["sha256"] for r in prep["source_inventory_before"]):
        raise RuntimeError("Origen cambio desde la copia")
    for rel in FILES:
        dest=OVERLAY/rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(OUT/LAUNCH if rel==LAUNCH else PROJECT/rel, dest)
    workspace=OUT/"workspace_aislado"
    if workspace.exists():
        raise FileExistsError("No sobrescribir compilacion anterior")
    configs=("Release", "Release_180MHz", CONFIG)
    args=[str(IDE/"stm32cubeidec.exe"),"--launcher.suppressErrors","-nosplash",
          "-application","org.eclipse.cdt.managedbuilder.core.headlessbuild","-data",str(workspace),
          "-import",str(PROJECT),"-no-indexer"]
    for cfg in configs:
        args.extend(["-build",NAME+"/"+cfg])
    tool=next((IDE/"plugins").glob("com.st.stm32cube.ide.mcu.externaltools.gnu-tools-for-stm32.*/tools/bin"))
    make=next((IDE/"plugins").glob("com.st.stm32cube.ide.mcu.externaltools.make.*/tools/bin"))
    env=dict(os.environ)
    env["PATH"]=str(tool)+os.pathsep+str(make)+os.pathsep+env.get("PATH", "")
    start=time.monotonic()
    with (OUT/"CUBEIDE_PERFILADO.log").open("w",encoding="utf-8") as log:
        process=subprocess.Popen(args,cwd=OUT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,encoding="utf-8",errors="replace")
        for line in process.stdout:
            log.write(line);log.flush()
            if any(s in line.lower() for s in ("build of","error:","warning:","build finished")):
                print(line.strip()[:350],flush=True)
        code=process.wait(timeout=30)
    spec=importlib.util.spec_from_file_location("clock_helper",Path(__file__).with_name("preparar_reloj_180mhz.py"))
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    logtext=(OUT/"CUBEIDE_PERFILADO.log").read_text(encoding="utf-8")
    controls={cfg:(PROJECT/cfg/(NAME+".elf")).is_file() and
        helper.flash_image(OUT/"respaldo_antes_perfilado"/cfg/(NAME+".elf"))==helper.flash_image(PROJECT/cfg/(NAME+".elf"))
        for cfg in configs[:2]}
    report=dict(returncode=code,elapsed_seconds=time.monotonic()-start,command=args,
        zero_errors_warnings_three=logtext.count("0 errors, 0 warnings")==3,
        controls_flash_identical=controls,hardware_flashed=False,physical_profile_validation_pending=True,
        configurations={})
    for cfg in configs:
        elf=PROJECT/cfg/(NAME+".elf")
        if elf.is_file():
            size=subprocess.check_output([str(tool/"arm-none-eabi-size.exe"),str(elf)],text=True)
            t,d,b=map(int,size.strip().splitlines()[-1].split()[:3])
            report["configurations"][cfg]=dict(elf=str(elf),elf_sha256=sha(elf),text=t,data=d,bss=b,
                flash_load_bytes=t+d,ram_static_with_linker_reservations_bytes=d+b)
    save("COMPILACION_PERFILADO.json",report)
    print(json.dumps(report,ensure_ascii=True,indent=2),flush=True)
    if code or not all(controls.values()) or not report["zero_errors_warnings_three"] or len(report["configurations"])!=3:
        raise SystemExit(2)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=("prepare","build"))
    mode=parser.parse_args().mode
    prepare() if mode=="prepare" else build()
