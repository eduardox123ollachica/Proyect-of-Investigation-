#!/usr/bin/env python3
"""Verifica integridad SHA-256 y tamaños de los corpus externos CWRU y Arciniegas.

Utiliza exclusivamente la biblioteca estándar de Python 3.12 contra el inventario
auditado en `Nuevas Observaciones/Reproducibilidad/INVENTARIO_DATOS.json`.
No descarga archivos ni redistribuye señales.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT_DEFAULT = Path(__file__).resolve().parents[2]
INVENTORY_DEFAULT = (
    ROOT_DEFAULT / "Nuevas Observaciones" / "Reproducibilidad" / "INVENTARIO_DATOS.json"
)
CHUNK_BYTES = 1024 * 1024


def sha256_and_size(path: Path) -> tuple[str, int]:
    """Calcula SHA-256 hex (minúsculas) y tamaño real en bytes de un archivo."""
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_BYTES):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def build_specs(inventory: dict[str, Any], corpus: str) -> list[dict[str, Any]]:
    """Normaliza entradas de CWRU y/o Arciniegas en una lista uniforme."""
    specs: list[dict[str, Any]] = []
    if corpus in {"all", "cwru"}:
        cwru = inventory["CWRU"]
        for item in cwru["archivos_mat"]:
            specs.append(
                {
                    "corpus": "CWRU",
                    "rel_path": Path(item["ruta_local"]).as_posix(),
                    "expected_bytes": int(item["bytes"]),
                    "expected_sha256": str(item["sha256"]).lower(),
                    "official_label": item.get("etiqueta_en_tabla_oficial"),
                    "source_url": item.get("pagina_oficial", cwru.get("fuente_general")),
                    "mat_keys": item.get("claves_mat_DE_FE"),
                    "id_mat": item.get("id_mat"),
                }
            )
    if corpus in {"all", "arciniegas"}:
        arc = inventory["Arciniegas"]
        base = Path(arc["raiz_local"])
        for item in arc["archivos_exportacion"]:
            specs.append(
                {
                    "corpus": "Arciniegas",
                    "rel_path": (base / item["ruta_relativa_exportacion"]).as_posix(),
                    "expected_bytes": int(item["bytes"]),
                    "expected_sha256": str(item["sha256"]).lower(),
                    "official_label": item["ruta_relativa_exportacion"],
                    "source_url": arc.get("fuente_editorial"),
                    "resource": arc.get("recurso"),
                    "mat_keys": None,
                    "id_mat": None,
                }
            )
    return specs


def verify_specs(root: Path, specs: list[dict[str, Any]]) -> dict[str, Any]:
    """Verifica existencia, tamaño y SHA-256 para cada archivo esperado."""
    ok_items: list[dict[str, Any]] = []
    missing_items: list[dict[str, Any]] = []
    mismatch_items: list[dict[str, Any]] = []
    notes: list[str] = []

    for spec in specs:
        target = root / spec["rel_path"]
        if not target.is_file():
            missing_items.append(spec)
            continue

        actual_sha256, actual_bytes = sha256_and_size(target)
        bytes_match = actual_bytes == spec["expected_bytes"]
        sha_match = actual_sha256 == spec["expected_sha256"]
        if bytes_match and sha_match:
            record = {
                "corpus": spec["corpus"],
                "rel_path": spec["rel_path"],
                "bytes": actual_bytes,
                "sha256": actual_sha256,
            }
            if spec["id_mat"] == 174 and spec["mat_keys"]:
                keys_txt = ", ".join(spec["mat_keys"])
                notes.append(
                    f"CWRU id_mat=174 ({spec['rel_path']}) usa claves internas {keys_txt}."
                )
            ok_items.append(record)
        else:
            mismatch_items.append(
                {
                    **spec,
                    "actual_bytes": actual_bytes,
                    "actual_sha256": actual_sha256,
                }
            )

    by_corpus: dict[str, dict[str, int]] = {}
    for c_name in sorted({s["corpus"] for s in specs}):
        by_corpus[c_name] = {
            "esperados": sum(1 for s in specs if s["corpus"] == c_name),
            "verificados_ok": sum(1 for s in ok_items if s["corpus"] == c_name),
            "faltantes": sum(1 for s in missing_items if s["corpus"] == c_name),
            "con_discrepancia": sum(1 for s in mismatch_items if s["corpus"] == c_name),
        }

    return {
        "fecha_verificacion_utc": datetime.now(timezone.utc).isoformat(),
        "raiz": str(root),
        "total_esperados": len(specs),
        "total_ok": len(ok_items),
        "total_faltantes": len(missing_items),
        "total_discrepancias": len(mismatch_items),
        "todo_integro": len(missing_items) == 0 and len(mismatch_items) == 0,
        "resumen_por_corpus": by_corpus,
        "notas_tecnicas": notes,
        "faltantes": missing_items,
        "discrepancias": mismatch_items,
    }


def print_summary(result: dict[str, Any], inventory: dict[str, Any]) -> None:
    """Imprime un resumen legible de verificación e instrucciones si falta algo."""
    print(f"Raiz verificada : {result['raiz']}")
    print(
        f"Resultado global: {result['total_ok']}/{result['total_esperados']} archivos integros "
        f"(faltantes={result['total_faltantes']}, discrepancias={result['total_discrepancias']})"
    )
    for corpus_name, stats in result["resumen_por_corpus"].items():
        print(
            f"  - {corpus_name}: {stats['verificados_ok']}/{stats['esperados']} OK "
            f"(faltantes={stats['faltantes']}, discrepancias={stats['con_discrepancia']})"
        )

    for note in result["notas_tecnicas"]:
        print(f"Nota tecnica: {note}")

    if result["faltantes"]:
        print("\nArchivos faltantes (primeros 15):")
        for item in result["faltantes"][:15]:
            print(
                f"  [FALTA] {item['rel_path']} "
                f"(etiqueta={item['official_label']}, origen={item['source_url']})"
            )
        print("\nConsulte DATA_POLICY.md para descargar y ubicar los archivos desde:")
        print(f"  - CWRU      : {inventory['CWRU']['fuente_general']}")
        print(
            f"  - Arciniegas: {inventory['Arciniegas']['fuente_editorial']} "
            f"({inventory['Arciniegas']['recurso']})"
        )

    if result["discrepancias"]:
        print("\nArchivos con tamanio o SHA-256 distinto (primeros 15):")
        for item in result["discrepancias"][:15]:
            print(
                f"  [HASH/BYTES] {item['rel_path']} "
                f"esperado=({item['expected_bytes']} B, {item['expected_sha256']}) "
                f"observado=({item['actual_bytes']} B, {item['actual_sha256']})"
            )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verifica la integridad SHA-256 de los corpus CWRU y Arciniegas."
    )
    parser.add_argument(
        "--raiz",
        type=Path,
        default=ROOT_DEFAULT,
        help="Raiz del proyecto donde residen datos/cwru/raw y datos/arciniegas/raw.",
    )
    parser.add_argument(
        "--inventario",
        type=Path,
        default=INVENTORY_DEFAULT,
        help="Ruta al archivo INVENTARIO_DATOS.json.",
    )
    parser.add_argument(
        "--solo",
        choices=("all", "cwru", "arciniegas"),
        default="all",
        help="Permite validar solo un corpus o ambos (por defecto: all).",
    )
    parser.add_argument(
        "--reporte-json",
        type=Path,
        default=None,
        help="Ruta opcional para guardar el informe completo en JSON.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    inv_path = args.inventario.resolve()
    if not inv_path.is_file():
        print(f"Error: no se encontro el inventario en {inv_path}", file=sys.stderr)
        return 2

    inventory = json.loads(inv_path.read_text(encoding="utf-8-sig"))
    specs = build_specs(inventory, args.solo)
    result = verify_specs(args.raiz.resolve(), specs)
    print_summary(result, inventory)

    if args.reporte_json is not None:
        args.reporte_json.parent.mkdir(parents=True, exist_ok=True)
        args.reporte_json.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"Informe JSON escrito en: {args.reporte_json}")

    return 0 if result["todo_integro"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
