"""Consolida resultados existentes sin repetir experimentos.

Genera una matriz plana de evidencia y dos documentos de trazabilidad a partir
de los CSV/TXT ya guardados en el proyecto.
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path


CODIGO_DIR = Path(__file__).resolve().parents[2]
if str(CODIGO_DIR) not in sys.path:
    sys.path.insert(0, str(CODIGO_DIR))

from comun.rutas_proyecto import (  # noqa: E402
    DATOS_ARCINIEGAS_PROCESADO,
    RESULTADOS_ARCINIEGAS,
)


RAIZ = CODIGO_DIR.parent
SALIDA = RAIZ / "documentacion" / "consolidacion"
COLUMNAS = [
    "categoria",
    "experimento",
    "componente",
    "metrica",
    "valor",
    "unidad",
    "estado",
    "fuente",
    "observacion",
]


def relativo(path: Path) -> str:
    return path.relative_to(RAIZ).as_posix()


def leer_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as archivo:
        return list(csv.DictReader(archivo))


def leer_texto(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def buscar(texto: str, patron: str) -> float:
    coincidencia = re.search(
        patron,
        texto,
        re.MULTILINE | re.IGNORECASE | re.DOTALL,
    )
    if not coincidencia:
        raise ValueError(f"No se encontro el patron: {patron}")
    return float(coincidencia.group(1))


filas: list[dict[str, object]] = []


def agregar(
    categoria: str,
    experimento: str,
    componente: str,
    metrica: str,
    valor: object,
    unidad: str,
    fuente: Path,
    observacion: str = "",
    estado: str = "Verificado",
) -> None:
    filas.append(
        {
            "categoria": categoria,
            "experimento": experimento,
            "componente": componente,
            "metrica": metrica,
            "valor": valor,
            "unidad": unidad,
            "estado": estado,
            "fuente": relativo(fuente),
            "observacion": observacion,
        }
    )


# Datos y configuracion comun.
metadata_path = DATOS_ARCINIEGAS_PROCESADO / "metadata_features.csv"
metadata = leer_csv(metadata_path)
for split in ("training", "testing"):
    subset = [fila for fila in metadata if fila["split"] == split]
    agregar("Datos", "Comun", "Dataset", f"Ventanas {split}", len(subset), "ventanas", metadata_path)
    for clase in ("DETENIDO", "VELOCIDAD MEDIA", "VELOCIDAD ALTA"):
        cantidad = sum(fila["class"] == clase for fila in subset)
        agregar("Datos", "Comun", "Dataset", f"Ventanas {split} {clase}", cantidad, "ventanas", metadata_path)

auditoria_path = RESULTADOS_ARCINIEGAS / "componentes_compartidos" / "auditoria" / "arciniegas_dataset_audit.csv"
auditoria = leer_csv(auditoria_path)
agregar("Datos", "Comun", "Dataset", "Archivos JSON totales", len(auditoria), "archivos", auditoria_path)
for split in ("training", "testing"):
    agregar(
        "Datos",
        "Comun",
        "Dataset",
        f"Archivos JSON {split}",
        sum(fila["split"] == split for fila in auditoria),
        "archivos",
        auditoria_path,
    )

contexto_path = RAIZ / "CONTEXTO_PROYECTO.md"
for metrica, valor, unidad in (
    ("Frecuencia de muestreo", 100, "Hz"),
    ("Longitud de ventana", 16, "muestras"),
    ("Hop", 8, "muestras"),
    ("Solapamiento", 50, "%"),
    ("Caracteristicas", 21, "features"),
    ("Parametros DNN", 683, "parametros"),
):
    agregar("Configuracion", "Comun", "Pipeline", metrica, valor, unidad, contexto_path, estado="Documentado")


# Entrenamiento y cuantizacion E1-E4.
configuracion = {
    "E1": ("No", "No"),
    "E2": ("Si", "No"),
    "E3": ("Si", "Si"),
    "E4": ("No", "Si"),
}

for experimento, (scaler, class_weight) in configuracion.items():
    carpeta = RESULTADOS_ARCINIEGAS / experimento.lower() / "entrenamiento"
    metricas_path = carpeta / "metricas.csv"
    tinyml_path = carpeta / "metricas_tinyml_fp32_int8.csv"
    matriz_path = carpeta / "matriz_confusion.csv"
    metricas = leer_csv(metricas_path)[0]
    tinyml = {fila["modelo"]: fila for fila in leer_csv(tinyml_path)}
    keras = tinyml["Keras_FP32"]
    int8 = tinyml["TFLite_INT8"]

    agregar("Configuracion", experimento, "DNN", "StandardScaler", scaler, "booleano", carpeta / "replica_info.txt", estado="Documentado")
    agregar("Configuracion", experimento, "DNN", "class_weight", class_weight, "booleano", carpeta / "replica_info.txt", estado="Documentado")
    agregar("Desempeno", experimento, "Keras", "Accuracy", float(keras["accuracy_percent"]), "%", tinyml_path)
    agregar("Desempeno", experimento, "TFLite INT8", "Accuracy", float(int8["accuracy_percent"]), "%", tinyml_path)
    agregar("Desempeno", experimento, "TFLite INT8", "Cambio vs Keras", float(int8["delta_accuracy_pp_vs_keras"]), "pp", tinyml_path)
    errores_int8 = round(1332 * (1 - float(int8["accuracy"])))
    agregar("Desempeno", experimento, "TFLite INT8", "Errores de clasificacion", errores_int8, "ventanas", tinyml_path)
    agregar("Desempeno", experimento, "Keras", "Macro precision", float(metricas["macro_precision"]) * 100, "%", metricas_path)
    agregar("Desempeno", experimento, "Keras", "Macro recall", float(metricas["macro_recall"]) * 100, "%", metricas_path)
    agregar("Desempeno", experimento, "Keras", "Macro F1", float(metricas["macro_f1"]) * 100, "%", metricas_path)
    agregar("Recursos", experimento, "TFLite INT8", "Tamano de archivo", int(float(int8["archivo_bytes"])), "B", tinyml_path)
    agregar("Recursos", experimento, "TFLite INT8", "Memoria de tensores estimada", int(float(int8["memoria_tensores_estimada_bytes"])), "B", tinyml_path, "Estimacion en Python; no equivale a RAM real del MCU")

    matriz = leer_csv(matriz_path)
    errores = 0
    for fila in matriz:
        real = fila["clase_real"].replace("REAL_", "")
        for columna, valor in fila.items():
            if not columna.startswith("PRED_"):
                continue
            pred = columna.replace("PRED_", "")
            cantidad = int(valor)
            agregar("Matriz de confusion", experimento, "Keras", f"{real} -> {pred}", cantidad, "ventanas", matriz_path)
            if real != pred:
                errores += cantidad
    agregar("Desempeno", experimento, "Keras", "Errores de clasificacion", errores, "ventanas", matriz_path)


# Validacion numerica de features.
features_path = RESULTADOS_ARCINIEGAS / "componentes_compartidos" / "features" / "validacion_stm32_10_ventanas" / "resumen_general.txt"
features = leer_texto(features_path)
for metrica, patron, unidad in (
    ("Ventanas analizadas", r"Ventanas analizadas:\s*(\d+)", "ventanas"),
    ("Comparaciones", r"Comparaciones totales:\s*(\d+)", "comparaciones"),
    ("MAE global", r"Error absoluto medio global:\s*([0-9.eE+-]+)", "valor feature"),
    ("Error absoluto maximo", r"Error absoluto m.ximo global:\s*([0-9.eE+-]+)", "valor feature"),
    ("Coincidencia frecuencia dominante", r"Porcentaje de coincidencia:\s*([0-9.]+)%", "%"),
):
    agregar("Validacion STM32", "Comun", "Features", metrica, buscar(features, patron), unidad, features_path)


# K-means y validacion PC/STM32.
kmeans_dir = RESULTADOS_ARCINIEGAS / "componentes_compartidos" / "kmeans"
kmeans_path = kmeans_dir / "Analisis_P99" / "resumen_analisis_P99.txt"
kmeans = leer_texto(kmeans_path)
for metrica, patron, unidad in (
    ("Clusters", r"Clusters:\s*(\d+)", "clusters"),
    ("Umbral P99", r"Umbral P99:\s*([0-9.]+)", "d2"),
    ("Ventanas rechazadas P99", r"Ventanas rechazadas:\s*(\d+)", "ventanas"),
    ("Tasa de rechazo dominio conocido", r"False anomaly rate sobre dominio conocido:\s*([0-9.]+)", "%"),
):
    fuente = kmeans_path
    if metrica == "Clusters":
        fuente = kmeans_dir / "resumen_STANDARD.txt"
        valor = buscar(leer_texto(fuente), patron)
    else:
        valor = buscar(kmeans, patron)
    agregar("K-means", "Comun", "Detector", metrica, valor, unidad, fuente)

kmeans_stm32_path = kmeans_dir / "Validacion_STM32" / "resumen_kmeans_todas.txt"
kmeans_stm32 = leer_texto(kmeans_stm32_path)
for metrica, patron, unidad in (
    ("Clusters iguales PC/STM32", r"Clusters iguales:\s*(\d+)/1332", "ventanas"),
    ("Flags iguales PC/STM32", r"Flags anomaly iguales:\s*(\d+)/1332", "ventanas"),
    ("MAE score PC/STM32", r"MAE score:\s*([0-9.eE+-]+)", "d2"),
    ("Error maximo score PC/STM32", r"Max error score:\s*([0-9.eE+-]+)", "d2"),
):
    agregar("Validacion STM32", "Comun", "K-means", metrica, buscar(kmeans_stm32, patron), unidad, kmeans_stm32_path)


# Resultados STM32 E1 y E4, incluida latencia.
for experimento in ("E1", "E4"):
    carpeta = RESULTADOS_ARCINIEGAS / experimento.lower()
    rawall_path = carpeta / "validacion_rawall" / "rawall_todas_resumen.txt"
    bench_path = carpeta / "benchmark_end_to_end" / "benchall_resumen.txt"
    rawall = leer_texto(rawall_path)
    bench = leer_texto(bench_path)
    for metrica, patron, unidad in (
        ("Ventanas", r"Ventanas:\s*(\d+)", "ventanas"),
        ("Clases iguales PC/STM32", r"Clases iguales:\s*(\d+)/1332", "ventanas"),
        ("Accuracy STM32", r"Accuracy STM32:\s*([0-9.]+)\s*%", "%"),
        ("MAE probabilidades", r"MAE probabilidades:\s*([0-9.eE+-]+)", "probabilidad"),
        ("Error maximo probabilidades", r"Max error probabilidades:\s*([0-9.eE+-]+)", "probabilidad"),
    ):
        agregar("Validacion STM32", experimento, "Pipeline", metrica, buscar(rawall, patron), unidad, rawall_path)
    for metrica, patron in (
        ("Features media", r"FEATURE EXTRACTION.*?mean:\s*([0-9.]+)\s*us"),
        ("DNN media", r"DNN.*?mean:\s*([0-9.]+)\s*us"),
        ("K-means media", r"KMEANS.*?mean:\s*([0-9.]+)\s*us"),
        ("Pipeline media", r"PIPELINE TOTAL.*?mean:\s*([0-9.]+)\s*us"),
        ("Pipeline P95", r"PIPELINE TOTAL.*?p95:\s*([0-9.]+)\s*us"),
        ("Pipeline maximo", r"PIPELINE TOTAL.*?max:\s*([0-9.]+)\s*us"),
    ):
        agregar("Rendimiento", experimento, "STM32", metrica, buscar(bench, patron), "us", bench_path)
    agregar("Rendimiento", experimento, "STM32", "Ocupacion media", buscar(bench, r"Ocupacion media:\s*([0-9.]+)%"), "%", bench_path)


# Recursos medidos mediante builds controlados y firmware completo.
trabajo_path = RAIZ / "documentacion" / "investigacion" / "trabajo_de_investigacion_actual.txt"
builds = [
    ("B0", "Base", 20880, 2448, None, None),
    ("B1", "Base + Features", 32664, 2448, 1028, 2932.810295),
    ("B2", "Base + Features + DNN", 74348, 19312, 824, 733.125751),
    ("B3", "Base + Features + K-means", 35960, 2448, 156, 2640.621434),
    ("B4", "Pipeline completo controlado", 77644, 19312, 1036, 6309.258587),
]
for build, descripcion, flash, ram, stack, tiempo in builds:
    agregar("Recursos STM32", build, descripcion, "Flash total", flash, "B", trabajo_path, estado="Documentado")
    agregar("Recursos STM32", build, descripcion, "RAM estatica total", ram, "B", trabajo_path, estado="Documentado")
    if stack is not None:
        agregar("Recursos STM32", build, descripcion, "Stack dinamico", stack, "B", trabajo_path, estado="Documentado")
    if tiempo is not None:
        agregar("Recursos STM32", build, descripcion, "Tiempo medio", tiempo, "us", trabajo_path, estado="Documentado")

for experimento in ("E1", "E4"):
    agregar("Recursos STM32", experimento, "Firmware completo", "Flash", 94200, "B", contexto_path, "Build completo; no comparar directamente con B4", "Documentado")
    agregar("Recursos STM32", experimento, "Firmware completo", "RAM", 19312, "B", contexto_path, "Build completo", "Documentado")


SALIDA.mkdir(parents=True, exist_ok=True)
csv_path = SALIDA / "matriz_maestra_resultados.csv"
with csv_path.open("w", encoding="utf-8-sig", newline="") as archivo:
    writer = csv.DictWriter(archivo, fieldnames=COLUMNAS)
    writer.writeheader()
    writer.writerows(filas)


trazabilidad = f"""# Trazabilidad de evidencias

Fecha de consolidacion: 2026-10-11.

Se consolidaron **{len(filas)} evidencias** a partir de resultados existentes.
No se repitieron entrenamientos, inferencias ni mediciones STM32.

## Jerarquia de autoridad

1. CSV y TXT generados directamente por los experimentos.
2. `CONTEXTO_PROYECTO.md` para decisiones vigentes y mediciones sin CSV propio.
3. `HISTORIAL_COMPLETO.md` y la bitacora para detalle historico.

## Resultados centrales

| Experimento | StandardScaler | class_weight | Accuracy Keras | Accuracy INT8 |
|---|---|---|---:|---:|
| E1 | No | No | 97.5976 % | 97.8979 % |
| E2 | Si | No | 94.3694 % | 94.3694 % |
| E3 | Si | Si | 99.1742 % | 99.3243 % |
| E4 | No | Si | 99.9249 % | 99.9249 % |

E4 es la variante con mayor accuracy. E1 y E4 se validaron en STM32 sobre
1332 ventanas con coincidencia total frente a PC.

## Alcance de las cifras de memoria

- B0-B4 son builds controlados para atribuir costo incremental por componente.
- Los 94 200 B de Flash corresponden al firmware completo E1/E4.
- B4 usa 77 644 B porque pertenece al proyecto controlado de recursos.
- Ambos alcances usan 19 312 B de RAM estatica.

La ruta exacta de cada evidencia figura en `matriz_maestra_resultados.csv`.
"""
(SALIDA / "trazabilidad_evidencias.md").write_text(trazabilidad, encoding="utf-8")


inconsistencias = """# Inconsistencias y asuntos pendientes

## Resueltos en la consolidacion

1. **E2 INT8:** `CONTEXTO_PROYECTO.md` indicaba "no consolidado", pero
   `resultados/arciniegas/e2/entrenamiento/metricas_tinyml_fp32_int8.csv`
   registra 94.369369 %. La matriz utiliza el CSV experimental.
2. **Cuantizacion E4:** la bitacora asocio temporalmente `0.079673462/-55` a
   E4. Esa cuantizacion pertenecia al intento incorrecto con scaler. E1 y E4
   correctos usan input scale `0.9703094363212585` y zero point `-125`.
3. **K-means:** una ruta de validacion lo situaba bajo una carpeta E4 y otra
   bajo E1. Se trata del mismo componente compartido y se consolido una sola vez.
4. **BENCHALL E4:** el script lo guardaba bajo un nombre que mencionaba
   StandardScaler. Se clasifico como E4 sin scaler y con class_weight.
5. **Flash B4 frente a firmware completo:** 77 644 B y 94 200 B proceden de
   builds con alcances distintos; no deben presentarse como mediciones rivales.

## Pendientes que no requieren repetir experimentos

1. Integrar las tablas consolidadas en la redaccion final.
2. Seleccionar las figuras definitivas y eliminar redundancias editoriales.
3. Decidir si el analisis `analisis_probabilidades_con_class_weight` encontrado
   dentro de E2 se mantiene solo como artefacto historico; no hay evidencia
   suficiente para reasignarlo a E3.

## Trabajo futuro opcional

- Medir stack especifico de E4 si se desea documentarlo por separado.
- Adquisicion autonoma con sensor fisico.
- Aplicar posteriormente el pipeline a CWRU.
"""
(SALIDA / "inconsistencias_pendientes.md").write_text(inconsistencias, encoding="utf-8")

print(f"Evidencias consolidadas: {len(filas)}")
print(csv_path)
