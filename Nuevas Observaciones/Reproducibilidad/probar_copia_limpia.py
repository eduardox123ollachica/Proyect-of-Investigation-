"""Reproduce la evidencia de la fase 1 en una copia mínima del proyecto.

Ejecutar con Python 3.12 y NumPy 2.3.5 disponibles para el intérprete.
Con --registrar se actualiza PRUEBA_COPIA_LIMPIA.json tras la verificación.
La copia y el resultado recién generado quedan bajo .codex_tmp.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
from tempfile import mkdtemp
from time import perf_counter


REVIEW = "Nuevas Observaciones"
SCRIPT = f"{REVIEW}/recalcular_evidencia_fase1.py"
EVIDENCE = f"{REVIEW}/EVIDENCIA_FASE1_CWRU.json"
ROOT = Path(__file__).resolve().parents[2]
REPORT = Path(__file__).with_name("PRUEBA_COPIA_LIMPIA.json")

# El auditor se instala después de cargar el intérprete y sus dependencias.
# A partir de ese punto se restringen las aperturas auditadas por Python.
AUDITED_RECALCULATION = """
import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PureWindowsPath
import pkgutil
import platform
import runpy
from statistics import mean, stdev
import sys

import numpy as np

# Cargar los lectores y módulos diferidos que se usan durante el recálculo.
sample = io.BytesIO()
np.savez(sample, prueba=np.array([0]))
sample.seek(0)
with np.load(sample, allow_pickle=False) as npz:
    values = npz["prueba"]
np.unique(values)
np.isin(values, (0, 1, 2))

runtime = {
    "sistema": platform.system(),
    "arquitectura": platform.machine(),
    "python": platform.python_version(),
    "numpy": np.__version__,
    "python_aislado": bool(sys.flags.isolated),
    "sin_bytecode": bool(sys.flags.dont_write_bytecode),
}

stage = Path.cwd().resolve()
script = sys.argv[1]
target = sys.argv[2]
read_files = set()
written_files = set()
write_flags = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND

def inside_stage(path):
    if isinstance(path, int):
        raise PermissionError("El recálculo debe abrir rutas de la copia")
    resolved = Path(os.fsdecode(path)).resolve()
    if not resolved.is_relative_to(stage):
        raise PermissionError(f"Acceso fuera de la copia limpia: {resolved}")
    return resolved.relative_to(stage).as_posix()

def audit(event, args):
    if event == "open":
        path, mode, flags = args
        relative = inside_stage(path)
        if flags & write_flags:
            if relative != target:
                raise PermissionError(f"Salida no autorizada en la copia: {relative}")
            written_files.add(relative)
        else:
            read_files.add(relative)
    elif event in {"os.chdir", "os.listdir", "os.scandir"} and args[0] is not None:
        inside_stage(args[0])
    elif event in {"subprocess.Popen", "os.system"} or event.startswith("socket."):
        raise PermissionError(f"Operación ajena al recálculo: {event}")

sys.addaudithook(audit)
try:
    (stage.parent / "sonda_de_lectura_externa").read_bytes()
except PermissionError as error:
    if not str(error).startswith("Acceso fuera de la copia limpia:"):
        raise
else:
    raise RuntimeError("La auditoría permitió una apertura externa")

sys.argv = [script]
runpy.run_path(script, run_name="__main__")
print(json.dumps({
    **runtime,
    "lecturas_auditadas": sorted(read_files),
    "escrituras_auditadas": sorted(written_files),
    "sonda_externa_bloqueada": True,
}, sort_keys=True))
"""


def verify(condition, message):
    if not condition:
        raise ValueError(message)


def relative_input(name):
    """Validar rutas portables antes de copiarlas a un árbol nuevo."""
    path = PurePosixPath(name)
    verify(path.parts and not path.is_absolute() and ".." not in path.parts
           and "\\" not in name and ":" not in name and path.as_posix() == name,
           f"Ruta de insumo no relativa o no normalizada: {name!r}")
    return path


def copy_inputs(stage, expected_hashes):
    inventory = []
    for name, expected_hash in sorted(expected_hashes.items()):
        relative = relative_input(name)
        source = (ROOT / relative).resolve(strict=True)
        verify(source.is_file() and source.is_relative_to(ROOT),
               f"El insumo no es un archivo interno: {name}")
        payload = source.read_bytes()
        actual_hash = sha256(payload).hexdigest()
        verify(actual_hash == expected_hash, f"SHA-256 distinto para {name}")
        target = stage / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        inventory.append({"ruta": name, "bytes": len(payload), "sha256": actual_hash})
    return inventory


def reproduce():
    reference_bytes = (ROOT / EVIDENCE).read_bytes()
    reference = json.loads(reference_bytes)
    input_hashes = reference["insumos_sha256"]
    counts = Counter(relative_input(name).suffix for name in input_hashes)
    verify(counts == {".json": 4, ".npz": 87}, "Inventario inesperado de fase 1")
    verify(EVIDENCE not in input_hashes and SCRIPT not in input_hashes,
           "El resultado de referencia debe quedar fuera de la copia")

    temp_parent = ROOT / ".codex_tmp" / "reproducibilidad_fase2"
    temp_parent.mkdir(parents=True, exist_ok=True)
    verify(temp_parent.resolve().is_relative_to(ROOT),
           "La carpeta temporal debe permanecer dentro del proyecto")
    stage = Path(mkdtemp(prefix="copia_", dir=temp_parent))
    inventory = copy_inputs(stage, input_hashes)
    script_bytes = (ROOT / SCRIPT).read_bytes()
    script_copy = stage / SCRIPT
    script_copy.parent.mkdir()
    script_copy.write_bytes(script_bytes)
    output = stage / EVIDENCE
    verify(not output.exists(), "La evidencia debe generarse desde cero")

    command = [sys.executable, "-I", "-B", "-X", "utf8", "-c",
               AUDITED_RECALCULATION, SCRIPT, EVIDENCE]
    start = perf_counter()
    process = subprocess.run(command, cwd=stage, capture_output=True,
                             text=True, encoding="utf-8", timeout=120)
    seconds = perf_counter() - start
    if process.returncode:
        raise RuntimeError(f"Falló el recálculo en {stage}: {process.stderr.strip()}")
    stdout = process.stdout.splitlines()
    runtime = json.loads(stdout[-1])
    reads = runtime.pop("lecturas_auditadas")
    writes = runtime.pop("escrituras_auditadas")
    verify(set(reads) == set(input_hashes) | {SCRIPT},
           "El conjunto de lecturas no coincide con el inventario y el script")
    verify(writes == [EVIDENCE], "El recálculo debe generar solo la evidencia esperada")
    generated = output.read_bytes()
    verify(generated == reference_bytes, "La evidencia regenerada difiere en bytes")
    verify({path.relative_to(stage).as_posix() for path in stage.rglob("*") if path.is_file()}
           == set(input_hashes) | {SCRIPT, EVIDENCE}, "Archivos inesperados en la copia")

    return {
        "estado": "RECALCULO_EN_COPIA_LIMPIA_VERIFICADO",
        "ejecutado_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "alcance": (
            "Recálculo de matrices, métricas, comparación por carga, restricciones de 3 HP "
            "y diagnóstico del detector desde predicciones y mediciones congeladas."
        ),
        "carpeta_temporal_relativa": stage.relative_to(ROOT).as_posix(),
        "script": {"ruta": SCRIPT, "sha256": sha256(script_bytes).hexdigest()},
        "entorno": runtime,
        "ejecucion": {
            "segundos": round(seconds, 3),
            "codigo_salida": process.returncode,
            "salida_recálculo": stdout[:-1],
            "auditoria": (
                "Tras inicializar Python y NumPy, se restringieron las aperturas auditadas "
                "por Python al árbol temporal. La sonda de lectura externa fue denegada."
            ),
            "insumos_leidos": len(reads) - 1,
            "script_leido": SCRIPT in reads,
            "archivos_generados": writes,
            "evidencia_esperada_presente_antes_del_recalculo": False,
        },
        "insumos": {
            "cantidad": len(inventory),
            "formatos": {suffix[1:]: count for suffix, count in sorted(counts.items())},
            "bytes_totales": sum(row["bytes"] for row in inventory),
            "inventario": inventory,
        },
        "resultado": {
            "ruta": EVIDENCE,
            "celdas_INT8_contrastadas": reference["celdas_INT8_contrastadas_con_predicciones_NPZ"],
            "sha256_referencia": sha256(reference_bytes).hexdigest(),
            "sha256_regenerado": sha256(generated).hexdigest(),
            "coincidencia_byte_a_byte": True,
        },
        "limites": [
            "Se reutilizó un entorno Python y NumPy ya instalado. La limpieza corresponde al árbol de trabajo.",
            "El ensayo comienza en predicciones y registros congelados. No incluyó MAT, "
            "señales crudas, preprocesamiento, entrenamiento ni nueva inferencia.",
            "La compilación del firmware, la repetición de mediciones STM32 y la publicación "
            "de los artefactos quedan fuera de este ensayo.",
            "La auditoría se limita a los eventos de apertura observables por Python.",
        ],
    }


def cleanup_stage(stage_rel_path):
    import shutil
    stage_dir = (ROOT / stage_rel_path).resolve()
    if stage_dir.is_dir():
        shutil.rmtree(stage_dir)
    for parent_name in ("reproducibilidad_fase2", ".codex_tmp"):
        parent_dir = (ROOT / parent_name) if parent_name == ".codex_tmp" else (ROOT / ".codex_tmp" / parent_name)
        if parent_dir.is_dir():
            try:
                parent_dir.rmdir()
            except OSError:
                pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registrar", action="store_true",
                        help="Guardar el recibo de la verificación junto a este script")
    parser.add_argument("--conservar-copia", action="store_true",
                        help="Conservar el árbol temporal de copia limpia tras finalizar")
    args = parser.parse_args()
    result = reproduce()
    if not args.conservar_copia:
        cleanup_stage(result["carpeta_temporal_relativa"])
    if args.registrar:
        REPORT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                          encoding="utf-8", newline="\n")
    print(f"{result['estado']}: {result['insumos']['cantidad']} insumos, "
          f"{result['resultado']['celdas_INT8_contrastadas']} celdas INT8, "
          f"SHA-256 {result['resultado']['sha256_regenerado']}")


if __name__ == "__main__":
    main()
