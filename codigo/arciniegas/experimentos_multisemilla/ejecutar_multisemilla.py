"""Orquesta los entrenamientos multisemilla E1--E4 sin sobrescribir resultados.

Por defecto solo muestra los comandos. Use --execute para iniciar entrenamientos.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
TRAINING_DIR = PROJECT_ROOT / "codigo/arciniegas/entrenamiento"
RESULTS_ROOT = PROJECT_ROOT / "resultados/arciniegas/multisemilla/ejecuciones"
LOGS_ROOT = PROJECT_ROOT / "resultados/arciniegas/multisemilla/logs"
DEFAULT_SEEDS = (11, 22, 33, 42, 44, 55, 66, 77, 88, 99)
SCRIPTS = {
    "E1": TRAINING_DIR / "entrenar_e1_sin_scaler_sin_class_weight.py",
    "E2": TRAINING_DIR / "entrenar_e2_con_scaler_sin_class_weight.py",
    "E3": TRAINING_DIR / "entrenar_e3_con_scaler_con_class_weight.py",
    "E4": TRAINING_DIR / "entrenar_e4_sin_scaler_con_class_weight.py",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="Ejecuta los entrenamientos; sin esta opción solo muestra el plan.")
    parser.add_argument(
        "--experiments",
        choices=tuple(SCRIPTS),
        nargs="+",
        default=list(SCRIPTS),
        help="Variantes que se planificarán o ejecutarán.",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--python", type=Path, default=Path(sys.executable), help="Intérprete con TensorFlow, NumPy, pandas, sklearn y matplotlib.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    commands: list[tuple[list[str], Path, str, int]] = []
    for experiment in args.experiments:
        script = SCRIPTS[experiment]
        for seed in args.seeds:
            output_dir = RESULTS_ROOT / experiment.lower() / f"seed_{seed}"
            if output_dir.exists() and any(output_dir.iterdir()):
                required = (
                    output_dir / "metricas_tinyml_fp32_int8.csv",
                    output_dir / "replica_info.txt",
                )
                if all(path.is_file() for path in required):
                    print(f"SKIP_COMPLETED: {experiment} seed {seed}", flush=True)
                    continue
                raise FileExistsError(f"La salida está incompleta y no se sobrescribirá: {output_dir}")
            commands.append(
                ([
                    str(args.python),
                    str(script),
                    "--seed",
                    str(seed),
                    "--output-dir",
                    str(output_dir),
                ], output_dir, experiment, seed)
            )

    for command, output_dir, experiment, seed in commands:
        print(" ".join(f'"{part}"' if " " in part else part for part in command), flush=True)
        if args.execute:
            LOGS_ROOT.mkdir(parents=True, exist_ok=True)
            log_path = LOGS_ROOT / f"{experiment.lower()}_seed_{seed}.log"
            print(f"RUNNING: {experiment} seed {seed} -> {output_dir}", flush=True)
            with log_path.open("w", encoding="utf-8") as log:
                subprocess.run(
                    command,
                    cwd=PROJECT_ROOT,
                    check=True,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            print(f"COMPLETED: {experiment} seed {seed}; log={log_path}", flush=True)

    if not args.execute:
        print("PLAN_ONLY: use --execute con un intérprete que tenga las dependencias de entrenamiento.")


if __name__ == "__main__":
    main()
