"""Referencia PC explícita del bloque de entrada histórico CWRU, no entrenamiento V2."""
from __future__ import annotations
import hashlib
import re
import struct
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
RESOURCES = ROOT / "modelos/cwru/canonico_seed42/firmware"
MODEL = ROOT / "resultados/cwru/modelo_features_14/canonico_seed42_int8/modelo_cwru_de_fe_seed42_int8.tflite"
SCALER = MODEL.with_name("standard_scaler_train_only.npz")
FEATURES = ("de_rms", "de_skewness", "de_kurtosis", "de_spectral_energy",
            "de_dominant_frequency", "de_dominant_amplitude", "de_spectral_entropy",
            "fe_rms", "fe_skewness", "fe_kurtosis", "fe_spectral_energy",
            "fe_dominant_frequency", "fe_dominant_amplitude", "fe_spectral_entropy")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bits(values) -> list[str]:
    return [f"{int(value):08x}" for value in np.asarray(values, dtype="<f4").view("<u4").ravel()]


def from_bits(values) -> np.ndarray:
    return np.asarray([int(v, 16) for v in values], dtype="<u4").view("<f4")


def tflite_io(path: Path) -> dict:
    """Lee sólo las tablas IO siguiendo los offsets del esquema TFLite local.

    No es un intérprete ni ejecuta inferencias. Comprueba límites del buffer.
    """
    raw = path.read_bytes()
    def read(fmt, offset):
        size = struct.calcsize("<" + fmt)
        if offset < 0 or offset + size > len(raw):
            raise ValueError("FlatBuffer fuera de límites")
        return struct.unpack_from("<" + fmt, raw, offset)[0]
    def field(table, vt):
        vtable = table - read("i", table)
        if vt >= read("H", vtable):
            return None
        offset = read("H", vtable + vt)
        return table + offset if offset else None
    def pointer(offset):
        if offset is None:
            raise ValueError("Campo FlatBuffer ausente")
        return offset + read("I", offset)
    def vector(table, vt, fmt):
        vector_start = pointer(field(table, vt))
        length = read("I", vector_start)
        step = struct.calcsize("<" + fmt)
        if length > len(raw) // step:
            raise ValueError("Vector FlatBuffer inválido")
        return [read(fmt, vector_start + 4 + i * step) for i in range(length)]
    if raw[4:8] != b"TFL3":
        raise ValueError("Identificador TFL3 incorrecto")
    model = read("I", 0)
    graphs = pointer(field(model, 8))
    if read("I", graphs) != 1:
        raise ValueError("Se esperaba un subgrafo")
    graph = pointer(graphs + 4)
    tensors = pointer(field(graph, 4))
    result = {"schema_version": read("I", field(model, 4)), "model_bytes": len(raw)}
    for name, vt in (("input", 6), ("output", 8)):
        indices = vector(graph, vt, "i")
        if len(indices) != 1 or not 0 <= indices[0] < read("I", tensors):
            raise ValueError("Contrato de tensores IO inesperado")
        tensor = pointer(tensors + 4 + 4 * indices[0])
        quant = pointer(field(tensor, 12))
        scales, zeros = vector(quant, 8, "f"), vector(quant, 10, "q")
        if len(scales) != 1 or len(zeros) != 1 or read("b", field(tensor, 6)) != 9:
            raise ValueError("Se requiere INT8 con cuantización por tensor")
        result[name] = dict(shape=vector(tensor, 4, "i"), scale=scales[0], zero_point=zeros[0], dtype="int8")
    return result


def parameters() -> dict:
    text = (RESOURCES / "cwru_dnn_scaler.cpp").read_text(encoding="utf-8")
    def array(name):
        match = re.search(r"const float " + name + r"\[[^\]]+\]\s*=\s*\{([^}]+)\}", text)
        if not match:
            raise ValueError("Parámetro C no encontrado: " + name)
        return np.asarray([float(v.strip().removesuffix("f")) for v in match[1].split(",") if v.strip()], dtype=np.float32)
    def scalar(name):
        match = re.search(r"const (?:float|int) " + name + r"\s*=\s*([^;]+);", text)
        if not match:
            raise ValueError("Escalar C ausente: " + name)
        return float(match[1].strip().removesuffix("f"))
    mean, scale = array("cwru_dnn_scaler_mean"), array("cwru_dnn_scaler_scale")
    if mean.shape != (14,) or scale.shape != (14,) or not np.isfinite(mean).all() or not np.all(scale > 0):
        raise ValueError("Escalador inválido")
    with np.load(SCALER, allow_pickle=False) as stored:
        if bits(mean) != bits(stored["mean"].astype(np.float32)) or bits(scale) != bits(stored["scale"].astype(np.float32)):
            raise ValueError("Escalador C diferente del NPZ histórico")
    metadata = tflite_io(MODEL)
    quant_scale = np.float32(scalar("cwru_dnn_input_quant_scale"))
    zero = int(scalar("cwru_dnn_input_quant_zero_point"))
    if metadata["input"]["shape"] != [1, 14] or metadata["output"]["shape"] != [1, 3]:
        raise ValueError("Forma del modelo incompatible")
    if bits([quant_scale]) != bits([metadata["input"]["scale"]]) or zero != metadata["input"]["zero_point"]:
        raise ValueError("Cuantización C diferente de TFLite")
    return dict(mean=mean, scale=scale, quant_scale=quant_scale, zero_point=zero, metadata=metadata)


def transform(values, params: dict, quant_only: bool = False) -> dict:
    x = np.asarray(values, dtype=np.float32)
    if x.shape != (14,) or not np.isfinite(x).all():
        raise ValueError("Se requieren 14 float32 finitos")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        centered = np.subtract(x, params["mean"], dtype=np.float32)
        standardized = x.copy() if quant_only else np.divide(centered, params["scale"], dtype=np.float32)
        divided = np.divide(standardized, params["quant_scale"], dtype=np.float32)
        affine = np.add(divided, np.float32(params["zero_point"]), dtype=np.float32)
    if not np.isfinite(standardized).all() or not np.isfinite(affine).all():
        raise ValueError("Resultado intermedio no finito")
    rounded = np.rint(affine)
    clipped_low, clipped_high = rounded < -128, rounded > 127
    q = np.clip(rounded, -128, 127).astype(np.int8)
    return dict(standardized=standardized, affine=affine, quantized=q,
                saturated_low=clipped_low, saturated_high=clipped_high)


def quantize_affine_manual(values) -> np.ndarray:
    """Algoritmo escalar propuesto para C, independiente de np.rint."""
    result = []
    for value in np.asarray(values, dtype=np.float32).ravel():
        number = float(value)
        if not np.isfinite(number):
            raise ValueError("Afín no finito")
        if number >= 127.5:
            result.append(127)
        elif number < -128.5:
            result.append(-128)
        else:
            lower = int(np.floor(number))
            fraction = number - lower
            if fraction > .5 or (fraction == .5 and lower % 2 != 0):
                lower += 1
            result.append(lower)
    return np.asarray(result, dtype=np.int8)
