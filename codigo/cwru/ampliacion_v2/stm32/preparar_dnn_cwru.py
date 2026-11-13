"""Configuración DNN opcional: copia aislada, build, instalación respaldada y sello.

prepare/seal/verify sólo escriben en C. build ejecuta CubeIDE aislado;
install modifica nueve archivos concretos de D con autorización. Nunca abre COM.
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
from preparar_cache_potencias import sha, configs
from preparar_reloj_180mhz import flash_image

ROOT=Path(__file__).resolve().parents[4]
NAME="Prueba_2_CWRU_Extraccion"
SOURCE=Path(r"D:\STM32IDE")/NAME
OUT=ROOT/"despliegue_stm32/dnn_int8_180mhz_2026-10-23"
PRIOR=ROOT/"despliegue_stm32/escalado_cuantizacion_180mhz_2026-10-23"
PROJECT=OUT/"copia_aislada"/NAME
BACKUP=OUT/"respaldo_antes_dnn"
OVERLAY=OUT/"archivos_para_D"
CLIENT=OUT/"Validacion_PC"
CONTROL="Release_180MHz_Input"
CONFIG="Release_180MHz_DNN"
LAUNCH=NAME+"_DNN_180MHz.launch"
IDE=Path(r"D:\STM32CubeIDE_2.2.0\STM32CubeIDE")
TEMPLATES=Path(__file__).with_name("dnn_firmware")
NEW=["CWRU_DNN/"+n for n in ("cwru_nn.h","cwru_nn.cpp","cwru_debug_log.cc","cmd_cwru_nn.c","cwru_model_data.cpp","cwru_model_data.h")]
FILES=[".cproject","Src/uart_commands.c",LAUNCH,*NEW]


def require(condition,message):
    if not condition: raise RuntimeError(message)
def save(name,data):
    with (OUT/name).open("x",encoding="utf-8",newline="\n") as f:
        json.dump(data,f,ensure_ascii=False,indent=2);f.write("\n")
def load(name): return json.loads((OUT/name).read_text(encoding="utf-8"))
def origin(prep):
    for row in prep["source_inventory_before"]:
        require(sha(SOURCE/row["path"])==row["sha256"],"D cambió: "+row["path"])
    for cfg,digest in prep["existing_elf_sha256"].items():
        require(sha(SOURCE/cfg/(NAME+".elf"))==digest,"ELF previo cambió: "+cfg)


def prepare():
    require(not PROJECT.exists() and not (SOURCE/"CWRU_DNN").exists() and not (SOURCE/LAUNCH).exists(),"Preparación ya existente")
    ref=load("REFERENCIA_DNN.json")
    require(ref["status"]=="PC_REFERENCES_READY_NOT_ARM_VALIDATED" and len(ref["cases"])==81,"Falta referencia PC")
    for name,digest in load("REFERENCIA_MANIFEST_SHA256.json").items(): require(sha(OUT/name)==digest,"Referencia modificada")
    subprocess.run([sys.executable,"-B",str(Path(__file__).with_name("preparar_entrada_cwru.py")),"verify"],check=True)
    old=json.loads((PRIOR/"PREPARACION.json").read_text(encoding="utf-8"))
    paths={r["path"] for r in old["source_inventory_before"]}|{p.name for p in SOURCE.glob("*.launch")}
    for folder in ("CWRU_Input","TFLM","Lib"):
        paths.update(p.relative_to(SOURCE).as_posix() for p in (SOURCE/folder).rglob("*") if p.is_file())
    snapshot=[]
    for rel in sorted(paths):
        digest=sha(SOURCE/rel)
        for root in (PROJECT,BACKUP):
            target=root/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(SOURCE/rel,target)
            require(sha(target)==digest,"Copia distinta: "+rel)
        snapshot.append(dict(path=rel,sha256=digest))
    tree,settings,before=configs(PROJECT/".cproject")
    require(len(before)==9 and CONTROL in before,"Se esperaban nueve configuraciones previas")
    elves={}
    for cfg in before:
        elf=SOURCE/cfg/(NAME+".elf");target=BACKUP/cfg/elf.name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(elf,target);elves[cfg]=sha(elf);require(sha(target)==elves[cfg],"Respaldo ELF distinto")
    expanded=copy.deepcopy(before[CONTROL]);mapping={e.get("id"):e.get("id")+".nn1" for e in expanded.iter() if e.get("id")}
    for e in expanded.iter():
        for key,value in list(e.attrib.items()):
            if value in mapping:e.set(key,mapping[value])
            elif key=="targetTool":e.set(key,";".join(mapping.get(v,v) for v in value.split(";")))
            elif key=="name" and value==CONTROL:e.set(key,CONFIG)
            elif key=="buildPath" and value.endswith("/"+CONTROL):e.set(key,value[:-len(CONTROL)]+CONFIG)
    for tool in expanded.findall(".//tool"):
        sc=tool.get("superClass","")
        if sc.endswith(("tool.c.compiler","tool.cpp.compiler")):
            defs=next(o for o in tool.findall("option") if o.get("superClass","").endswith(".definedsymbols"))
            ET.SubElement(defs,"listOptionValue",builtIn="false",value="CWRU_DNN_TEST=1")
            inc=next(o for o in tool.findall("option") if o.get("superClass","").endswith(".includepaths"))
            ET.SubElement(inc,"listOptionValue",builtIn="false",value="../CWRU_DNN")
        if sc.endswith("tool.cpp.compiler"):
            ET.SubElement(tool,"option",id=tool.get("id")+".nn_misc",superClass=sc+".option.otherflags",
                          valueType="string",value="-std=gnu++17 -fno-threadsafe-statics")
        if sc.endswith("tool.cpp.linker"):
            libs=next(o for o in tool.findall("option") if o.get("superClass","").endswith(".libraries"))
            for child in list(libs):libs.remove(child)
            for name in ("tensorflow-microlite","m"):ET.SubElement(libs,"listOptionValue",builtIn="false",value=name)
            libpaths=ET.SubElement(tool,"option",id=tool.get("id")+".nn_libpaths",superClass=sc+".option.directories",valueType="libPaths")
            ET.SubElement(libpaths,"listOptionValue",builtIn="false",value="../Lib")
    ET.SubElement(expanded.find(".//sourceEntries"),"entry",flags="VALUE_WORKSPACE_PATH|RESOLVED",kind="sourcePath",name="CWRU_DNN")
    old_ids={e.get("id") for c in before.values() for e in c.iter() if e.get("id")}
    require(not old_ids&{e.get("id") for e in expanded.iter() if e.get("id")},"Colisión de IDs")
    settings.append(expanded)
    (PROJECT/".cproject").write_bytes(b'<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n<?fileVersion 4.0.0?>\n'+ET.tostring(tree,encoding="utf-8")+b"\n")
    after=configs(PROJECT/".cproject")[2]
    require(all(ET.tostring(c)==ET.tostring(after[n]) for n,c in before.items()),"Configuración previa modificada")
    folder=PROJECT/"CWRU_DNN";folder.mkdir()
    for p in TEMPLATES.iterdir():
        if p.suffix in (".c",".h",".cpp",".cc"):shutil.copy2(p,folder/p.name)
    resources=ROOT/"modelos/cwru/canonico_seed42/firmware"
    for name in ("cwru_model_data.cpp","cwru_model_data.h"):shutil.copy2(resources/name,folder/name)
    launch=ET.parse(SOURCE/(NAME+"_Input_180MHz.launch")).getroot();attrs={e.get("key"):e for e in launch};elf=CONFIG+"/"+NAME+".elf"
    attrs["org.eclipse.cdt.launch.PROGRAM_NAME"].set("value",elf)
    attrs["org.eclipse.cdt.launch.PROJECT_BUILD_CONFIG_ID_ATTR"].set("value",expanded.get("id"))
    attrs["org.eclipse.cdt.launch.PROJECT_BUILD_CONFIG_AUTO_ATTR"].set("value","false")
    key="com.st.stm32cube.ide.mcu.debug.launch.loadList";loads=json.loads(attrs[key].get("value"))
    for item in loads["fItems"]:item["fPath"]=elf
    attrs[key].set("value",json.dumps(loads,separators=(",",":")))
    for key,e in attrs.items():
        if key.endswith("stlink.log_file"):e.set("value",str(SOURCE/CONFIG/"st-link_gdbserver_log.txt"))
    (PROJECT/LAUNCH).write_bytes(b'<?xml version="1.0" encoding="UTF-8"?>\n'+ET.tostring(launch,encoding="utf-8")+b"\n")
    CLIENT.mkdir();shutil.copy2(Path(__file__).with_name("validar_dnn_cwru.py"),CLIENT/"validar_dnn.py")
    shutil.copy2(OUT/"REFERENCIA_DNN.json",CLIENT/"REFERENCIA_DNN.json")
    (CLIENT/"VALIDACION_MANIFEST.json").write_text(json.dumps({p.name:sha(p) for p in CLIENT.iterdir() if p.is_file()},indent=2)+"\n",encoding="utf-8")
    save("PREPARACION.json",dict(status="PREPARED_DISPATCH_PATCH_PENDING",source_inventory_before=snapshot,
        existing_elf_sha256=elves,configuration=CONFIG,control=CONTROL,previous_nine_configurations_preserved=True,
        modified_files=FILES[:2],added_files=FILES[2:],hardware_flashed=False,serial_port_opened=False,
        tensorflow_library_sha256=sha(PROJECT/"Lib/libtensorflow-microlite.a"),physical_validation_pending=True))
    print("PREPARADO: diez configuraciones, snapshots TFLM/Lib y cliente; falta guardia UART.",flush=True)


def build():
    prep=load("PREPARACION.json");origin(prep)
    require("CWRUNN_Command" in (PROJECT/"Src/uart_commands.c").read_text(encoding="utf-8"),"Falta parche UART")
    require(not (OUT/"COMPILACION_DNN.json").exists(),"Build ya aprobado; no repetir")
    # Cuarto intento acotado tras diagnóstico documentado, no bucle de reintentos.
    attempt=len(list(OUT.glob("CUBEIDE_DNN_INTENTO_*.log")))+1;require(attempt<=4,"Revisar diagnóstico antes de más intentos")
    tests=subprocess.run([sys.executable,"-B",str(CLIENT/"validar_dnn.py"),"--self-test"],capture_output=True,text=True,encoding="utf-8",errors="replace")
    save(f"PRUEBAS_CLIENTE_INTENTO_{attempt}.json",dict(status="PASS" if tests.returncode==0 else "FAIL",output=tests.stdout+tests.stderr,executes_ARM=False))
    require(tests.returncode==0,"Autotests fallidos")
    for rel in FILES:
        for root in (OVERLAY,OUT/"intentos"/str(attempt)/"archivos"):
            target=root/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(PROJECT/rel,target)
    tool=next((IDE/"plugins").glob("com.st.stm32cube.ide.mcu.externaltools.gnu-tools-for-stm32.*/tools/bin"))
    make=next((IDE/"plugins").glob("com.st.stm32cube.ide.mcu.externaltools.make.*/tools/bin"))
    env=dict(os.environ);env["PATH"]=str(tool)+os.pathsep+str(make)+os.pathsep+env.get("PATH","")
    selected=["Release_180MHz_EntM",CONTROL,CONFIG]
    args=[str(IDE/"stm32cubeidec.exe"),"--launcher.suppressErrors","-nosplash","-application","org.eclipse.cdt.managedbuilder.core.headlessbuild",
        "-data",str(OUT/"workspace_aislado"),"-import",str(PROJECT),"-no-indexer"]
    for cfg in selected:args.extend(("-build",NAME+"/"+cfg))
    start=time.monotonic();logpath=OUT/f"CUBEIDE_DNN_INTENTO_{attempt}.log"
    with logpath.open("x",encoding="utf-8") as log:
        proc=subprocess.Popen(args,cwd=OUT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,encoding="utf-8",errors="replace")
        for line in proc.stdout:
            log.write(line);log.flush()
            if any(s in line.lower() for s in ("build of","error:","warning:","undefined reference","build finished")):print(line.strip()[:300],flush=True)
        code=proc.wait(timeout=30)
    text=logpath.read_text(encoding="utf-8");summaries=re.findall(r"(\d+) errors, (\d+) warnings",text)
    clean=bool(summaries) and all(int(a)==0 and int(b)==0 for a,b in summaries) and not re.search(r"\b(?:error|warning):|undefined reference",text)
    report=dict(attempt=attempt,returncode=code,elapsed_seconds=time.monotonic()-start,command=args,zero_errors_warnings=clean,configurations={})
    for cfg in selected:
        folder=PROJECT/cfg;elf=folder/(NAME+".elf")
        if not elf.exists():continue
        size=subprocess.check_output([str(tool/"arm-none-eabi-size.exe"),str(elf)],text=True)
        t,d,b=map(int,size.strip().splitlines()[-1].split()[:3])
        commands=[line for mk in folder.rglob("subdir.mk") for line in mk.read_text(encoding="utf-8").splitlines()
                  if "-c " in line and "arm-none-eabi-" in line and re.search(r"-std=gnu(?:\+\+|\d)",line)]
        flags=bool(commands) and all(re.findall(r"(?<!\S)-O\S+",line)==["-O3"] and "-DCWRU_TARGET_HCLK_HZ=180000000U" in line
            and "-DCWRU_ENTROPY_MODE=2" in line and ("-DCWRU_DNN_TEST=1" in line)==(cfg==CONFIG)
            and not any(f in line for f in ("-ffast-math","-Ofast","-funsafe-math-optimizations")) for line in commands)
        if cfg==CONFIG:
            cpp_commands=[line for line in commands if "arm-none-eabi-g++" in line]
            flags=flags and bool(cpp_commands) and all("-fno-rtti" in line and "-fno-exceptions" in line
                and "-fno-threadsafe-statics" in line and re.findall(r"-std=(\S+)",line)[-1]=="gnu++17" for line in cpp_commands)
        image=flash_image(elf);span=max(a+len(data) for a,data in image)-min(a for a,_ in image)
        units=(folder/"objects.list").read_text(encoding="utf-8").splitlines()
        row=dict(elf_sha256=sha(elf),text=t,data=d,bss=b,flash_load_bytes=t+d,ram_static_with_linker_reservations_bytes=d+b,
            flash_span_bytes=span,expected_download_kib=span/1024,flags_confirmed=flags,compilation_units=len(units),
            dnn_units=sum("CWRU_DNN" in u for u in units),stack_frames={p.name:p.read_text(encoding="utf-8").splitlines() for p in (folder/"CWRU_DNN").glob("*.su")})
        if cfg!=CONFIG:row["control_elf_identical"]=sha(elf)==prep["existing_elf_sha256"][cfg];row["control_flash_identical"]=flash_image(elf)==flash_image(BACKUP/cfg/(NAME+".elf"))
        else:
            symbols=subprocess.check_output([str(tool/"arm-none-eabi-nm.exe"),"-C",str(elf)],text=True)
            (OUT/f"SIMBOLOS_DNN_INTENTO_{attempt}.txt").write_text(symbols,encoding="utf-8")
            row["cmsis_nn_fully_connected_symbol_linked"]=bool(re.search(r"\b[Tt] arm_fully_connected_s8$",symbols,re.M))
            row["cmsis_nn_softmax_symbol_linked"]=bool(re.search(r"\b[Tt] arm_softmax_s8$",symbols,re.M))
            row["tensorflow_archive_in_map"]="libtensorflow-microlite.a" in (folder/(NAME+".map")).read_text(encoding="utf-8")
            row["tensorflow_search_path_confirmed"]=any(
                "arm-none-eabi-g++ -o " in line and re.search(r'-L\s*"?\.\./Lib"?(?:\s|$)',line)
                for line in text.splitlines())
        report["configurations"][cfg]=row
    previous,current=configs(BACKUP/".cproject")[2],configs(PROJECT/".cproject")[2]
    report["previous_nine_configurations_preserved"]=all(ET.tostring(c)==ET.tostring(current[n]) for n,c in previous.items())
    allowed={".cproject","Src/uart_commands.c",".settings/language.settings.xml"}
    report["other_resources_unchanged"]=all(sha(PROJECT/r["path"])==r["sha256"] for r in prep["source_inventory_before"] if r["path"] not in allowed)
    includes=["-I"+str(PROJECT/p) for p in ("Inc","CWRU_Input","Drivers/CMSIS/Include","Drivers/CMSIS/Device/ST/STM32F4xx/Include")]
    cpp=[str(tool/"arm-none-eabi-gcc.exe"),"-E","-P","-std=gnu11","-mcpu=cortex-m4","-mfpu=fpv4-sp-d16","-mfloat-abi=hard","-mthumb","-DSTM32F446xx",*includes]
    report["uart_old_preprocessed_identical"]={str(flag):subprocess.check_output([*cpp,f"-DCWRU_INPUT_TRANSFORM={flag}",str(BACKUP/"Src/uart_commands.c")])==subprocess.check_output([*cpp,f"-DCWRU_INPUT_TRANSFORM={flag}",str(PROJECT/"Src/uart_commands.c")]) for flag in (0,1)}
    report["overlay_sha256"]={r:sha(OVERLAY/r) for r in FILES}
    passed=(code==0 and clean and len(report["configurations"])==3 and all(row["flags_confirmed"] for row in report["configurations"].values())
        and all(report["configurations"].get(cfg,{}).get("control_elf_identical",False) and report["configurations"][cfg]["control_flash_identical"] for cfg in selected[:2])
        and report["previous_nine_configurations_preserved"] and report["other_resources_unchanged"] and all(report["uart_old_preprocessed_identical"].values())
        and report["configurations"].get(CONFIG,{}).get("tensorflow_archive_in_map",False)
        and report["configurations"].get(CONFIG,{}).get("tensorflow_search_path_confirmed",False))
    report.update(status="PASS" if passed else "FAIL",hardware_flashed=False,physical_validation_pending=True)
    origin(prep);save(f"COMPILACION_DNN_INTENTO_{attempt}.json",report)
    if passed:save("COMPILACION_DNN.json",report)
    print(json.dumps(report,ensure_ascii=True,indent=2),flush=True)
    if not passed:raise SystemExit(2)


def install():
    prep,built=load("PREPARACION.json"),load("COMPILACION_DNN.json")
    require(not (SOURCE/"CWRU_DNN").exists() and not (SOURCE/LAUNCH).exists() and not (OUT/"APLICACION_EN_D.json").exists(),"Destino ya instalado")
    require(built["status"]=="PASS","Build no aprobado");origin(prep)
    for rel,digest in built["overlay_sha256"].items():require(sha(OVERLAY/rel)==digest,"Overlay modificado")
    for rel in FILES:
        target=SOURCE/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(OVERLAY/rel,target)
    before,after=configs(BACKUP/".cproject")[2],configs(SOURCE/".cproject")[2]
    same=all(ET.tostring(c)==ET.tostring(after[n]) for n,c in before.items())
    resources=all(sha(SOURCE/r["path"])==r["sha256"] for r in prep["source_inventory_before"] if r["path"] not in FILES[:2])
    elves=all(sha(SOURCE/cfg/(NAME+".elf"))==digest for cfg,digest in prep["existing_elf_sha256"].items())
    report=dict(status="PASS" if same and resources and elves else "FAIL",modified_files=FILES[:2],added_files=FILES[2:],
        previous_nine_configurations_preserved=same,previous_nine_elfs_unchanged=elves,other_snapshot_resources_unchanged=resources,
        installed_sha256={r:sha(SOURCE/r) for r in FILES},hardware_flashed=False,physical_validation_pending=True,user_workspace_modified=False)
    save("APLICACION_EN_D.json",report);print(json.dumps(report,ensure_ascii=True,indent=2),flush=True);require(report["status"]=="PASS","Instalación no válida")


def seal():
    require(not (OUT/"MANIFEST_SHA256.json").exists(),"No sobrescribir sello")
    prep,built,installed=load("PREPARACION.json"),load("COMPILACION_DNN.json"),load("APLICACION_EN_D.json")
    require(built["status"]==installed["status"]=="PASS","Faltan comprobaciones")
    for rel,digest in installed["installed_sha256"].items():require(sha(SOURCE/rel)==digest,"D diferente")
    paths=set(p for p in OUT.iterdir() if p.is_file())
    for row in prep["source_inventory_before"]:paths.update((PROJECT/row["path"],BACKUP/row["path"]))
    paths.update(PROJECT/r for r in FILES);paths.update(OVERLAY/r for r in FILES)
    paths.update(BACKUP/cfg/(NAME+".elf") for cfg in prep["existing_elf_sha256"])
    for cfg in ("Release_180MHz_EntM",CONTROL,CONFIG):
        folder=PROJECT/cfg;paths.update(folder/r for r in (NAME+".elf",NAME+".map","objects.list"));paths.update((folder/"CWRU_DNN").glob("*.su"))
    paths.update(p for p in CLIENT.iterdir() if p.is_file());paths.update(p for p in TEMPLATES.iterdir() if p.is_file())
    paths.update(p for p in (OUT/"intentos").rglob("*") if p.is_file())
    paths.update((Path(__file__).resolve(),Path(__file__).with_name("validar_dnn_cwru.py"),Path(__file__).with_name("preparar_referencia_dnn.py")))
    save("MANIFEST_SHA256.json",{p.relative_to(ROOT).as_posix():sha(p) for p in sorted(paths)})
    with (OUT/"MANIFEST_SHA256.sha256").open("x",encoding="ascii") as f:f.write(sha(OUT/"MANIFEST_SHA256.json")+"  MANIFEST_SHA256.json\n")
    verify()


def verify():
    digest,name=(OUT/"MANIFEST_SHA256.sha256").read_text(encoding="ascii").strip().split()
    require(name=="MANIFEST_SHA256.json" and sha(OUT/name)==digest,"Sello distinto")
    manifest=load(name)
    for rel,expected in manifest.items():
        p=(ROOT/rel).resolve();require(p.is_relative_to(ROOT) and sha(p)==expected,"Recurso distinto: "+rel)
    for rel,expected in load("REFERENCIA_ENTRADAS_SHA256.json").items():require(sha(ROOT/rel)==expected,"Entrada de referencia distinta: "+rel)
    print(json.dumps(dict(status="PASS",files_verified=len(manifest),physical_validation_pending=True,serial_port_opened=False,active_D_rechecked=False),indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("mode",choices=("prepare","build","install","seal","verify"))
    {"prepare":prepare,"build":build,"install":install,"seal":seal,"verify":verify}[parser.parse_args().mode]()
