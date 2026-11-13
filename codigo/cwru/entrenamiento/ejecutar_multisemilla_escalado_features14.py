"""Orquesta la comparación RAW--StandardScaler con aislamiento por semilla.

Cada semilla se ejecuta en un proceso independiente y dispone de un tiempo
máximo. Los resultados completados se consolidan después de cada proceso. El
trabajador utilizado no carga el conjunto de prueba.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_SEEDS = (11, 22, 33, 42, 44, 55, 66, 77, 88, 99)
SCRIPT_DIR = Path(__file__).resolve().parent
WORKER = SCRIPT_DIR / "seleccionar_escalado_features14.py"
PROJECT_ROOT = SCRIPT_DIR.parents[2]
OUTPUT_ROOT = (
    PROJECT_ROOT
    / "resultados"
    / "cwru"
    / "modelo_features_14"
    / "multisemilla_escalado"
)


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
        file.write("\n")
    temporary.replace(path)


def valid_result(path: Path, seed: int) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    variants = {item.get("variant") for item in payload.get("results", [])}
    return (
        payload.get("status") == "selected_on_validation_test_not_loaded"
        and payload.get("seed") == seed
        and variants == {"raw", "standard"}
    )


def consolidate(seeds: list[int], statuses: list[dict]) -> None:
    rows = []
    for seed in seeds:
        result_path = OUTPUT_ROOT / f"seed_{seed}" / "seleccion.json"
        if not valid_result(result_path, seed):
            continue
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        by_variant = {item["variant"]: item for item in payload["results"]}
        for variant in ("raw", "standard"):
            rows.append(by_variant[variant])

    csv_path = OUTPUT_ROOT / "resultados_validacion_por_semilla.csv"
    if rows:
        fields = list(rows[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    atomic_json(
        OUTPUT_ROOT / "estado_ejecucion.json",
        {
            "protocol": "paired_raw_vs_standard_validation_only",
            "test_loaded": False,
            "requested_seeds": seeds,
            "completed_seeds": sorted({int(row["seed"]) for row in rows}),
            "statuses": statuses,
        },
    )


def run_seed(seed: int, timeout_seconds: int) -> dict:
    seed_dir = OUTPUT_ROOT / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(WORKER),
        "--seed",
        str(seed),
        "--output-dir",
        str(seed_dir),
    ]
    environment = os.environ.copy()
    inherited_paths = [entry for entry in sys.path if entry]
    if environment.get("PYTHONPATH"):
        inherited_paths.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(inherited_paths)

    started_at = time.perf_counter()
    print(f"Semilla {seed}: proceso aislado iniciado.", flush=True)
    try:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        elapsed = time.perf_counter() - started_at
        print(f"Semilla {seed}: excedió {timeout_seconds} s y fue detenida.", flush=True)
        return {
            "seed": seed,
            "status": "timeout",
            "duration_seconds": elapsed,
            "stdout": error.stdout or "",
            "stderr": error.stderr or "",
        }

    elapsed = time.perf_counter() - started_at
    if completed.stdout:
        print(completed.stdout.rstrip(), flush=True)
    if completed.stderr:
        print(completed.stderr.rstrip(), file=sys.stderr, flush=True)
    status = "completed" if completed.returncode == 0 else "failed"
    print(f"Semilla {seed}: {status} en {elapsed:.2f} s.", flush=True)
    return {
        "seed": seed,
        "status": status,
        "return_code": completed.returncode,
        "duration_seconds": elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    statuses = []
    for seed in arguments.seeds:
        result_path = OUTPUT_ROOT / f"seed_{seed}" / "seleccion.json"
        if not arguments.force and valid_result(result_path, seed):
            print(f"Semilla {seed}: resultado completo reutilizado.", flush=True)
            statuses.append({"seed": seed, "status": "reused"})
        else:
            statuses.append(run_seed(seed, arguments.timeout_seconds))
        consolidate(arguments.seeds, statuses)

    failed = [item for item in statuses if item["status"] in {"failed", "timeout"}]
    print(
        f"Final: {len(statuses) - len(failed)}/{len(statuses)} semillas disponibles; "
        f"fallos={len(failed)}.",
        flush=True,
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
