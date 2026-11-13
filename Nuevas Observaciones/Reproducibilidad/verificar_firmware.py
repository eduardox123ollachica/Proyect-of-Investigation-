"""Contrasta la copia canónica CWRU con los registros históricos de compilación."""

import argparse
from collections import Counter
from hashlib import file_digest
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import sys
from xml.etree import ElementTree


CAMPAIGN = "canonicos_42_180mhz_2026-10-27"
PROJECT = "Prueba_2_CWRU_Extraccion"
CONFIGURATION = "Release_180MHz_Canon42_SR"
CAMPAIGN_PATH = Path("despliegue_stm32") / CAMPAIGN
PROJECT_PATH = CAMPAIGN_PATH / "copia_aislada" / PROJECT
REVIEW_PATH = Path("Revisar como articulo nuevo/04_Embedded_Deployment/Project_Files")
CDT = ".cproject"
LANGUAGE = ".settings/language.settings.xml"
ARCHIVE = "Lib/libtensorflow-microlite.a"
KERNEL = "CWRU_TotalSR/arm_nn_vec_mat_mult_t_per_ch_s8.c"
OUTPUT = Path(__file__).with_name("FIRMWARE_VERIFICADO.json")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def observed_hashes(directory, names):
    hashes = {}
    for name in sorted(names):
        relative = PurePosixPath(name)
        require(not relative.is_absolute() and ".." not in relative.parts
                and "\\" not in name and ":" not in name, f"Ruta inesperada en el registro {name}")
        path = directory / relative
        if path.is_file():
            hashes[name] = digest(path)
    return hashes


def compare(hashes, expected):
    absent = sorted(expected.keys() - hashes.keys())
    changed = sorted(name for name in expected.keys() & hashes.keys()
                     if expected[name] != hashes[name])
    return {"coinciden": len(expected) - len(absent) - len(changed),
            "faltan": absent, "difieren": changed}


def component(name):
    for prefix, group in (
        ("TFLM/third_party/cmsis_nn/", "CMSIS-NN"),
        ("CWRU_TotalSR/", "Kernel CMSIS-NN de aplicación"),
        ("TFLM/", "TFLM y dependencias vendorizadas restantes"),
        ("Middlewares/CMSIS-DSP/", "CMSIS-DSP"),
        ("Drivers/STM32F4xx_HAL_Driver/", "STM32 HAL"),
        ("Src/stm32f4xx_hal", "STM32 HAL compilado desde Src"),
        ("Drivers/CMSIS/", "CMSIS Core y dispositivos STM32"),
    ):
        if name.startswith(prefix):
            return group
    if name == ARCHIVE:
        return "Biblioteca TFLM precompilada"
    return "Otros insumos del proyecto"


def gcc_strings(path):
    return sorted({value.decode("ascii") for value in
                   re.findall(rb"GCC: [\x20-\x7e]{8,120}", path.read_bytes())})


def verify(root, review, sin_chm=False):
    campaign = root / CAMPAIGN_PATH
    project = root / PROJECT_PATH
    names = {
        "build": "COMPILACION_CANON42.json",
        "preparacion": "MANIFEST_PREPARACION_PROYECTO_SHA256.json",
        "cdt": "AUDITORIA_METADATA_CDT.json",
        "elf": "AUDITORIA_ELF_CANON42.json",
        "kernel": "AUDITORIA_INDEPENDIENTE_KERNEL_MEMORIA.json",
        "fisico": "AUDITORIA_FULL_CANON42.json",
        "sintaxis": "VERIFICACION_SINTAXIS_ARM.json",
        "log_build": "CUBEIDE_CANON42.log",
    }
    docs = {key: campaign / filename for key, filename in names.items()}
    hashes_docs = {key: digest(path) for key, path in docs.items()}
    build = read_json(docs["build"])
    prep = read_json(docs["preparacion"])
    audit_elf = read_json(docs["elf"])
    audit_cdt = read_json(docs["cdt"])
    audit_kernel = read_json(docs["kernel"])
    physical = read_json(docs["fisico"])
    syntax = read_json(docs["sintaxis"])
    kernel = audit_kernel["kernel"]
    require(build["returncode"] == 0 and build["integrity_after_build"]
            and build["zero_errors_and_warnings"] and build["configuration"] == CONFIGURATION,
            "La compilación registrada no es la canónica")
    for key, field in (("build", "build_report_sha256"),
                       ("preparacion", "preparation_manifest_sha256"),
                       ("cdt", "metadata_audit_sha256"), ("kernel", "kernel_audit_sha256")):
        require(hashes_docs[key] == audit_elf[field], f"Registro histórico con SHA-256 diferente {key}")
    require(audit_kernel["build_report_sha256"] == hashes_docs["build"],
            "El informe de enlace no corresponde a la compilación")
    reference = build["D_source_sha256_after_build"]
    manuals = sorted(name for name in reference if name.startswith("Drivers/STM32F4xx_HAL_Driver/")
                     and name.endswith(".chm"))
    require(len(manuals) == 8, "Cantidad inesperada de manuales HAL")
    expected_missing = manuals if sin_chm else []
    prefix = PROJECT_PATH.as_posix() + "/"
    prepared = {key[len(prefix):]: value for key, value in prep.items() if key.startswith(prefix)}
    require(len(reference) == 1877 and reference.keys() <= prepared.keys(),
            "Cobertura inesperada del manifiesto de preparación")
    require(audit_cdt["status"] == "PASS_GENERATED_METADATA_ONLY"
            and audit_cdt["added_configuration"] == CONFIGURATION
            and reference[LANGUAGE] == audit_cdt["before_sha256"] == prepared[LANGUAGE]
            and reference[CDT] != prepared[CDT], "Trazabilidad CDT inesperada")
    isolated_expected = {**prepared, LANGUAGE: audit_cdt["after_sha256"]}
    isolated = observed_hashes(project, isolated_expected)
    isolated_comparison = compare(isolated, isolated_expected)
    require(isolated_comparison["faltan"] == expected_missing and not isolated_comparison["difieren"],
            f"La copia aislada difiere de la preparación auditada {isolated_comparison}")
    ref_comparison = compare(isolated, reference)
    require(ref_comparison["faltan"] == expected_missing and ref_comparison["difieren"] == [CDT, LANGUAGE],
            "Hay diferencias no documentadas respecto del inventario de D")
    config_names = [node.get("name") for node in ElementTree.parse(project / CDT).iter("configuration")]
    require(config_names.count(CONFIGURATION) == 1, "Falta la configuración canónica de CDT")

    elf_rel = f"{CONFIGURATION}/{PROJECT}.elf"
    map_rel = f"{CONFIGURATION}/{PROJECT}.map"
    binaries = observed_hashes(project, (elf_rel, map_rel))
    require(binaries.get(elf_rel) == build["elf_sha256"] == audit_elf["elf_sha256"]
            == audit_kernel["ELF_sha256"] == physical["firmware_elf_sha256"],
            "El ELF no coincide con la validación física archivada")
    require(binaries.get(map_rel) == build["map_sha256"] == audit_elf["map_sha256"]
            == audit_kernel["map_sha256"], "El MAP no coincide con los registros de enlace")
    require(physical["status"] == "PASS_CANON42_STAGED_FULL_AUDITED",
            "No consta el cierre de validación física")
    require(isolated[ARCHIVE] == reference[ARCHIVE] == kernel["precompiled_archive_sha256"]
            and isolated[KERNEL] == kernel["source_sha256"]
            and kernel["application_object_is_map_provider"]
            and kernel["original_archive_member_not_extracted"],
            "No se pudo vincular la biblioteca y el kernel al informe de enlace")
    gcc_elf = gcc_strings(project / elf_rel)
    gcc_archive = gcc_strings(project / ARCHIVE)
    require(gcc_elf and all("14.3.rel1" in value for value in gcc_elf), "Identificador GCC inesperado en ELF")
    flags = audit_kernel["compile"]["expected_flags"]
    build_log = docs["log_build"].read_text(encoding="utf-8-sig")
    require(all(flag in build_log for flag in ("-mcpu=cortex-m4", "-O3", "-mfpu=fpv4-sp-d16",
                "-mfloat-abi=hard", "-DCMSIS_NN_USE_SINGLE_ROUNDING=1", "-DTFLITE_SINGLE_ROUNDING=1"))
            and "-ffast-math" not in build_log and audit_kernel["compile"]["all_C_CPP_expected_flags"],
            "El registro de compilación no confirma las opciones esperadas")
    tool_hashes = {PureWindowsPath(unit["command"][0]).name: unit["compiler_sha256"]
                   for unit in syntax["units"]}

    review_record = None
    if review is not None:
        require(review.is_dir(), "No se encontró la copia de revisión indicada")
        review_hashes = observed_hashes(review, isolated_expected.keys() | binaries.keys())
        review_preparation = compare(review_hashes, isolated_expected)
        require(review_preparation["faltan"] == manuals and not review_preparation["difieren"]
                and all(review_hashes.get(name) == value for name, value in binaries.items()),
                f"La copia de revisión tiene diferencias sin documentar {review_preparation}")
        review_record = {
            "comparacion_con_inventario_D": compare(review_hashes, reference),
            "comparacion_con_manifiesto_preparacion_y_CDT": review_preparation,
            "ELF_y_MAP_coinciden": True,
        }

    hal_src = "Src/stm32f4xx_hal.c"
    require(isolated[hal_src] == isolated["Drivers/STM32F4xx_HAL_Driver/Src/stm32f4xx_hal.c"],
            "El HAL en Src difiere del HAL de Drivers")
    identifiers = {}
    for name, path, pattern in (
        ("HAL STM32F4xx", hal_src, r"HAL Driver version number (V[\d.]+)"),
        ("CMSIS-DSP", "Middlewares/CMSIS-DSP/Include/arm_math.h", r"@version\s+(V[\d.]+)"),
        ("CMSIS-NN", "TFLM/third_party/cmsis_nn/Include/arm_nnfunctions.h", r"\$Revision:\s+(V[\d.]+)"),
        ("TFLM", "TFLM/tensorflow/lite/micro/micro_interpreter.h", r"TFLITE_SCHEMA_VERSION\s+\((\d+)\)"),
    ):
        require(path in isolated, f"Falta la fuente versionada {path}")
        match = re.search(pattern, (project / path).read_text(encoding="utf-8-sig"))
        require(match is not None, f"No se encontró el identificador declarado en {path}")
        identifiers[name] = {"declaracion_local": match[1], "archivo": path}

    return {
        "estado": "PASS_IDENTIDAD_LOCAL_FIRMWARE_CANONICO",
        "alcance": "Identidad de copias, configuración y binarios conservados",
        "proyecto_relativo": PROJECT_PATH.as_posix(),
        "configuracion": CONFIGURATION,
        "registros": {key: {"ruta": (CAMPAIGN_PATH / filename).as_posix(),
                           "sha256": hashes_docs[key]} for key, filename in names.items()},
        "copia_aislada": {
            "comparacion_con_inventario_D": compare(isolated, reference),
            "comparacion_con_manifiesto_preparacion_y_CDT": isolated_comparison,
            "rutas_agregadas_por_preparacion": sorted(prepared.keys() - reference.keys()),
            "archivos_por_componente": dict(Counter(component(name) for name in isolated)),
            "sha256_archivos": isolated,
        },
        "copia_revision_previa": review_record,
        "manuales_HAL_en_copia_aislada": manuals,
        "metadatos_CDT": {
            name: {"sha256_D": reference[name], "sha256_preparacion": prepared[name],
                   "sha256_copia_aislada": isolated[name]} for name in (CDT, LANGUAGE)
        },
        "artefactos": {
            **binaries,
            ARCHIVE: isolated[ARCHIVE],
            KERNEL: isolated[KERNEL],
        },
        "enlace": {
            "ELF_coincide_con_validacion_fisica_archivada": True,
            "kernel_enlazado_desde_objeto_de_aplicacion": True,
            "funcion_kernel_sha256_segun_auditoria": kernel["function_bytes_sha256"],
            "biblioteca_TFLM_ya_compilada_se_conservo": True,
        },
        "entorno_registrado": {
            "STM32CubeIDE": "2.2.0",
            "GCC_metadatos_en_ELF": gcc_elf,
            "GCC_metadatos_en_biblioteca": gcc_archive,
            "opciones_segun_auditoria": flags,
            "sha256_compiladores_registrados_en_prueba_de_sintaxis": tool_hashes,
            "notas_licencia_conservadas": sorted(name for name in isolated
                                                 if PurePosixPath(name).stem.lower() in ("license", "copying")),
        },
        "identificadores_locales": identifiers,
        "limites": [
            "La declaración HAL identifica una versión declarada en su fuente. Las declaraciones DSP y NN son revisiones de cabecera. El valor TFLM corresponde al esquema del modelo.",
            "No se acreditó un commit upstream exacto de TFLM, CMSIS-NN, CMSIS-DSP o HAL.",
            "No se ejecutó una compilación nueva, no se reconstruyó el archivo TFLM y no se utilizó la placa.",
        ],
    }


def adjust_expected_for_sin_chm(expected):
    """Ajusta el informe esperado cuando se verifican árboles públicos sin manuales .chm HAL."""
    manuals = expected["manuales_HAL_en_copia_aislada"]
    manual_set = set(manuals)
    isolated = expected["copia_aislada"]
    isolated["sha256_archivos"] = {
        name: sha for name, sha in isolated["sha256_archivos"].items() if name not in manual_set
    }
    isolated["archivos_por_componente"] = dict(Counter(component(name) for name in isolated["sha256_archivos"]))
    for key in ("comparacion_con_inventario_D", "comparacion_con_manifiesto_preparacion_y_CDT"):
        isolated[key]["coinciden"] -= len(manual_set)
        isolated[key]["faltan"] = sorted(manuals)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raiz", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--copia-revision", type=Path, help="Ruta alternativa al proyecto STM32 de la copia de revisión")
    parser.add_argument("--check", action="store_true", help="Coteja el JSON existente en modo solo lectura")
    parser.add_argument("--sin-chm", action="store_true",
                        help="Permite únicamente la ausencia de los 8 manuales .chm de ST HAL (ej. en public_release)")
    args = parser.parse_args()
    sin_chm = bool(args.sin_chm or (args.raiz / "MANIFEST_PUBLIC_SHA256.txt").is_file())
    default_review = args.raiz / REVIEW_PATH / CAMPAIGN / "copia_aislada" / PROJECT
    review = args.copia_revision if args.copia_revision else default_review if default_review.is_dir() else None
    try:
        report = verify(args.raiz, review, sin_chm=sin_chm)
        if args.check:
            expected = read_json(OUTPUT)
            if review is None:
                expected["copia_revision_previa"] = None
            if sin_chm:
                adjust_expected_for_sin_chm(expected)
            require(expected == report, "El informe registrado difiere de la comprobación actual")
        else:
            OUTPUT.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                              encoding="utf-8")
    except (ValueError, OSError, KeyError) as exc:
        print(f"Verificación fallida. {exc}", file=sys.stderr)
        return 1
    print(report["estado"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
