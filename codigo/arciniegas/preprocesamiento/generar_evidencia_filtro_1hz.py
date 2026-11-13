"""Genera evidencia reproducible sobre la especificacion LPF de 1 Hz.

El analisis contrasta la respuesta de un Butterworth digital de cuarto orden,
interpretado literalmente a partir del articulo de referencia, con los bins de
una RFFT de 16 puntos a 100 Hz. No entrena modelos ni modifica el dataset.

Para evitar una seleccion visual favorable, la demostracion sobre datos elige
la ventana de 16 muestras con mayor energia AC conjunta dentro de la captura
historicamente usada para comprobar el filtro. La proyeccion espectral aplica
la magnitud cero-fase |H|^2 a cada bin; no pretende reproducir transitorios de
borde particulares de scipy.signal.sosfiltfilt.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_JSON = (
    PROJECT_ROOT
    / "datos"
    / "arciniegas"
    / "raw"
    / "training"
    / "VELOCIDAD ALTA.json.51ir832t.ingestion-6f998cf944-t6sxf.s9.json"
)
FIGURE_DIR = PROJECT_ROOT / "articulo" / "figuras"
RESULT_DIR = PROJECT_ROOT / "articulo" / "resultados_derivados" / "filtro_1hz"

FS_HZ = 100.0
FC_HZ = 1.0
FILTER_ORDER = 4
FFT_SIZE = 16
HOP_SIZE = 8

INK = "#172033"
MUTED = "#596579"
GRID = "#DCE2EA"
BLUE = "#1666B1"
RED = "#C23B33"
ORANGE = "#D9822B"
PALE_BLUE = "#EAF2FA"
WHITE = "#FFFFFF"


def _font(size: int, bold: bool = False):
    filename = "arialbd.ttf" if bold else "arial.ttf"
    candidates = [
        Path("C:/Windows/Fonts") / filename,
        Path("C:/Windows/Fonts/calibri.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default(size=size)


FONT_TITLE = _font(42, bold=True)
FONT_SUBTITLE = _font(30, bold=True)
FONT_BODY = _font(24)
FONT_BODY_BOLD = _font(24, bold=True)
FONT_SMALL = _font(20)


def load_capture(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    payload = data["payload"]
    fs = 1000.0 / float(payload["interval_ms"])
    values = np.asarray(payload["values"], dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError(f"Forma inesperada: {values.shape}")
    if not math.isclose(fs, FS_HZ, rel_tol=1e-9):
        raise ValueError(f"Frecuencia inesperada: {fs} Hz")
    return values, fs


def butterworth_db(frequencies, fs=FS_HZ, fc=FC_HZ, order=FILTER_ORDER):
    """Respuesta digital Butterworth y respuesta cero-fase en dB.

    Reproduce la magnitud resultante de la transformacion bilineal usada para
    un Butterworth digital con Wn=fc/(fs/2). La pasada adelante-atras duplica
    los dB de la respuesta de una sola pasada.
    """
    f = np.asarray(frequencies, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        ratio = np.tan(np.pi * f / fs) / np.tan(np.pi * fc / fs)
        log_ratio = np.where(ratio > 0, np.log(ratio), -np.inf)
        log_denominator = np.logaddexp(0.0, 2.0 * order * log_ratio)
        single_db = -10.0 / np.log(10.0) * log_denominator
    single_db = np.where(f == 0, 0.0, single_db)
    zero_phase_db = 2.0 * single_db
    return single_db, zero_phase_db


def choose_window(values):
    candidates = []
    for start in range(0, len(values) - FFT_SIZE + 1, HOP_SIZE):
        window = values[start : start + FFT_SIZE]
        centered = window - window.mean(axis=0, keepdims=True)
        energy = float(np.sum(centered**2))
        candidates.append((energy, start, window))
    return max(candidates, key=lambda item: (item[0], -item[1]))


def map_x(value, vmin, vmax, left, right):
    return left + (value - vmin) / (vmax - vmin) * (right - left)


def map_y(value, vmin, vmax, top, bottom):
    return bottom - (value - vmin) / (vmax - vmin) * (bottom - top)


def dashed_line(draw, xy, fill, width=2, dash=10):
    x1, y1, x2, y2 = xy
    length = math.hypot(x2 - x1, y2 - y1)
    if length == 0:
        return
    dx = (x2 - x1) / length
    dy = (y2 - y1) / length
    cursor = 0.0
    while cursor < length:
        end = min(cursor + dash, length)
        draw.line(
            (x1 + dx * cursor, y1 + dy * cursor, x1 + dx * end, y1 + dy * end),
            fill=fill,
            width=width,
        )
        cursor += 2 * dash


def draw_response_image(path: Path, bin_rows):
    image = Image.new("RGB", (1800, 1000), WHITE)
    draw = ImageDraw.Draw(image)
    draw.text((70, 35), "Compatibilidad del LPF de 1 Hz con la FFT de 16 puntos", font=FONT_TITLE, fill=INK)
    draw.text(
        (70, 92),
        "Butterworth de orden 4; respuesta de fase cero (pasada adelante-atras)",
        font=FONT_BODY,
        fill=MUTED,
    )

    left, right, top, bottom = 105, 1190, 175, 875
    y_min, y_max = -240.0, 5.0
    for ytick in [-240, -200, -160, -120, -80, -40, 0]:
        py = map_y(ytick, y_min, y_max, top, bottom)
        draw.line((left, py, right, py), fill=GRID, width=2)
        draw.text((28, py - 13), f"{ytick}", font=FONT_SMALL, fill=MUTED)
    for xtick in range(0, 51, 5):
        px = map_x(xtick, 0, 50, left, right)
        draw.line((px, top, px, bottom), fill=GRID, width=1)
        draw.text((px - 14, bottom + 12), str(xtick), font=FONT_SMALL, fill=MUTED)

    frequencies = np.linspace(0, 50, 2001)
    _, zero_db = butterworth_db(frequencies)
    clipped = np.maximum(zero_db, y_min)
    points = [
        (
            map_x(float(f), 0, 50, left, right),
            map_y(float(db), y_min, y_max, top, bottom),
        )
        for f, db in zip(frequencies, clipped)
    ]
    draw.line(points, fill=BLUE, width=5)

    cutoff_x = map_x(FC_HZ, 0, 50, left, right)
    dashed_line(draw, (cutoff_x, top, cutoff_x, bottom), ORANGE, width=4, dash=13)
    draw.text((cutoff_x + 10, top + 8), "fc = 1 Hz", font=FONT_BODY_BOLD, fill=ORANGE)

    for row in bin_rows:
        px = map_x(row["frequency_hz"], 0, 50, left, right)
        py = map_y(max(row["zero_phase_db"], y_min), y_min, y_max, top, bottom)
        draw.ellipse((px - 7, py - 7, px + 7, py + 7), fill=RED)
        dashed_line(draw, (px, top, px, bottom), "#E8B4B0", width=2, dash=8)

    draw.rectangle((left, top, right, bottom), outline=INK, width=3)
    draw.text((520, 915), "Frecuencia [Hz]", font=FONT_BODY_BOLD, fill=INK)
    draw.text((20, 470), "Ganancia [dB]", font=FONT_BODY_BOLD, fill=INK)

    table_left = 1250
    draw.rounded_rectangle((table_left, 175, 1735, 875), radius=18, fill=PALE_BLUE, outline="#B8C8DA", width=2)
    draw.text((table_left + 30, 205), "Bins usados por la RFFT", font=FONT_SUBTITLE, fill=INK)
    draw.text((table_left + 30, 258), "Bin", font=FONT_BODY_BOLD, fill=INK)
    draw.text((table_left + 125, 258), "Hz", font=FONT_BODY_BOLD, fill=INK)
    draw.text((table_left + 250, 258), "Atenuacion", font=FONT_BODY_BOLD, fill=INK)
    for i, row in enumerate(bin_rows):
        y = 305 + i * 62
        draw.line((table_left + 25, y - 12, 1710, y - 12), fill="#C9D5E2", width=1)
        draw.text((table_left + 42, y), str(row["bin"]), font=FONT_BODY, fill=INK)
        draw.text((table_left + 118, y), f'{row["frequency_hz"]:.2f}', font=FONT_BODY, fill=INK)
        draw.text((table_left + 245, y), f'{row["zero_phase_db"]:.2f} dB', font=FONT_BODY, fill=RED)
    image.save(path, quality=95)


def draw_bins_image(path: Path, frequencies, original_db, filtered_db, start):
    image = Image.new("RGB", (1800, 1000), WHITE)
    draw = ImageDraw.Draw(image)
    draw.text((70, 35), "Efecto sobre una ventana real del corpus", font=FONT_TITLE, fill=INK)
    draw.text(
        (70, 92),
        f"Ventana de maxima energia AC de la captura de VELOCIDAD ALTA; muestras {start}-{start + FFT_SIZE - 1}",
        font=FONT_BODY,
        fill=MUTED,
    )
    axis_names = ["Eje X", "Eje Y", "Eje Z"]
    for axis in range(3):
        left, right = 105, 1690
        top = 180 + axis * 255
        bottom = top + 190
        y_min, y_max = -260.0, 5.0
        for ytick in [-240, -180, -120, -60, 0]:
            py = map_y(ytick, y_min, y_max, top, bottom)
            draw.line((left, py, right, py), fill=GRID, width=1)
            draw.text((25, py - 11), str(ytick), font=FONT_SMALL, fill=MUTED)
        for f in frequencies:
            px = map_x(float(f), 6.25, 50, left, right)
            draw.line((px, top, px, bottom), fill="#EEF1F5", width=1)
        p_original = [
            (map_x(float(f), 6.25, 50, left, right), map_y(float(v), y_min, y_max, top, bottom))
            for f, v in zip(frequencies, np.maximum(original_db[:, axis], y_min))
        ]
        p_filtered = [
            (map_x(float(f), 6.25, 50, left, right), map_y(float(v), y_min, y_max, top, bottom))
            for f, v in zip(frequencies, np.maximum(filtered_db[:, axis], y_min))
        ]
        draw.line(p_original, fill=BLUE, width=5)
        draw.line(p_filtered, fill=RED, width=5)
        for px, py in p_original:
            draw.ellipse((px - 6, py - 6, px + 6, py + 6), fill=BLUE)
        for px, py in p_filtered:
            draw.ellipse((px - 6, py - 6, px + 6, py + 6), fill=RED)
        draw.rectangle((left, top, right, bottom), outline=INK, width=2)
        draw.text((left + 12, top + 8), axis_names[axis], font=FONT_BODY_BOLD, fill=INK)
        if axis == 0:
            draw.line((1190, top + 25, 1260, top + 25), fill=BLUE, width=5)
            draw.text((1270, top + 10), "Original", font=FONT_SMALL, fill=INK)
            draw.line((1430, top + 25, 1500, top + 25), fill=RED, width=5)
            draw.text((1510, top + 10), "Tras LPF", font=FONT_SMALL, fill=INK)
        if axis == 2:
            for f in frequencies:
                px = map_x(float(f), 6.25, 50, left, right)
                draw.text((px - 24, bottom + 10), f"{f:g}", font=FONT_SMALL, fill=MUTED)
    draw.text((730, 930), "Frecuencia [Hz]", font=FONT_BODY_BOLD, fill=INK)
    draw.text((20, 500), "Magnitud relativa [dB]", font=FONT_BODY_BOLD, fill=INK)
    image.save(path, quality=95)


def combine_images(response_path: Path, bins_path: Path, output_path: Path):
    top = Image.open(response_path).convert("RGB")
    bottom = Image.open(bins_path).convert("RGB")
    canvas = Image.new("RGB", (1800, 2000), WHITE)
    canvas.paste(top, (0, 0))
    canvas.paste(bottom, (0, 1000))
    canvas.save(output_path, quality=95)


def generar_evidencia(json_path: Path = DEFAULT_JSON):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    values, fs = load_capture(json_path)
    window_energy, start, window = choose_window(values)
    centered = window - window.mean(axis=0, keepdims=True)
    spectrum = np.fft.rfft(centered, n=FFT_SIZE, axis=0)
    frequencies = np.fft.rfftfreq(FFT_SIZE, d=1.0 / fs)
    non_dc_frequencies = frequencies[1:]
    single_db, zero_db = butterworth_db(non_dc_frequencies, fs=fs)
    zero_magnitude = np.power(10.0, zero_db / 20.0)

    original_magnitude = np.abs(spectrum[1:, :])
    filtered_magnitude = original_magnitude * zero_magnitude[:, None]
    reference = np.maximum(np.max(original_magnitude, axis=0, keepdims=True), 1e-300)
    original_db = 20.0 * np.log10(np.maximum(original_magnitude / reference, 1e-300))
    filtered_db = 20.0 * np.log10(np.maximum(filtered_magnitude / reference, 1e-300))

    bin_rows = []
    for index, (frequency, single, zero) in enumerate(
        zip(non_dc_frequencies, single_db, zero_db), start=1
    ):
        bin_rows.append(
            {
                "bin": index,
                "frequency_hz": float(frequency),
                "single_pass_db": float(single),
                "zero_phase_db": float(zero),
                "amplitude_ratio_zero_phase": float(10.0 ** (zero / 20.0)),
            }
        )

    with (RESULT_DIR / "atenuacion_bins_fft16.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(bin_rows[0]))
        writer.writeheader()
        writer.writerows(bin_rows)

    energy_rows = []
    for axis, name in enumerate(["X", "Y", "Z"]):
        original_energy = float(np.sum(original_magnitude[:, axis] ** 2))
        filtered_energy = float(np.sum(filtered_magnitude[:, axis] ** 2))
        ratio = filtered_energy / original_energy if original_energy > 0 else 0.0
        energy_rows.append(
            {
                "axis": name,
                "original_non_dc_energy": original_energy,
                "projected_non_dc_energy": filtered_energy,
                "remaining_energy_ratio": ratio,
                "reduction_percent": 100.0 * (1.0 - ratio),
            }
        )
    with (RESULT_DIR / "energia_ventana_demostracion.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(energy_rows[0]))
        writer.writeheader()
        writer.writerows(energy_rows)

    response_path = FIGURE_DIR / "verificacion_filtro_1hz_respuesta.png"
    bins_path = FIGURE_DIR / "verificacion_filtro_1hz_bins.png"
    combined_path = FIGURE_DIR / "verificacion_filtro_1hz.png"
    draw_response_image(response_path, bin_rows)
    draw_bins_image(bins_path, non_dc_frequencies, original_db, filtered_db, start)
    combine_images(response_path, bins_path, combined_path)

    summary = [
        "VERIFICACION DE LA ESPECIFICACION LPF DE 1 HZ",
        "=" * 64,
        f"Captura: {json_path.name}",
        f"Fs: {fs:.6f} Hz",
        f"FFT: {FFT_SIZE} muestras; hop usado para seleccionar ventana: {HOP_SIZE}",
        f"Filtro interpretado: Butterworth digital orden {FILTER_ORDER}, fc={FC_HZ:.6f} Hz",
        "Respuesta evaluada: fase cero |H|^2",
        f"Ventana seleccionada automaticamente: inicio={start}, fin_exclusivo={start + FFT_SIZE}",
        f"Energia AC conjunta de seleccion: {window_energy:.12g}",
        "",
        "ATENUACION DEL PRIMER BIN NO-DC",
        f"Frecuencia: {bin_rows[0]['frequency_hz']:.6f} Hz",
        f"Una pasada: {bin_rows[0]['single_pass_db']:.6f} dB",
        f"Fase cero: {bin_rows[0]['zero_phase_db']:.6f} dB",
        f"Relacion de amplitud: {bin_rows[0]['amplitude_ratio_zero_phase']:.12e}",
        "",
        "ENERGIA NO-DC PROYECTADA EN LA VENTANA",
    ]
    for row in energy_rows:
        summary.append(
            f"Eje {row['axis']}: restante={row['remaining_energy_ratio']:.12e}; "
            f"reduccion={row['reduction_percent']:.12f}%"
        )
    summary.extend(
        [
            "",
            "INTERPRETACION",
            "Con N=16 y Fs=100 Hz, el primer bin no-DC se ubica en 6.25 Hz.",
            "Todos los bins utilizados para energia, frecuencia dominante, amplitud",
            "dominante y entropia quedan por encima de fc=1 Hz. Bajo la interpretacion",
            "literal ensayada, el LPF suprime practicamente todo el contenido espectral",
            "no-DC. Esto respalda omitirlo del pipeline principal y reportar la",
            "especificacion publicada como ambigua o probablemente incorrecta; no permite",
            "determinar si los autores pretendian indicar un pasa-altos u otro corte.",
        ]
    )
    (RESULT_DIR / "resumen_verificacion_filtro_1hz.txt").write_text(
        "\n".join(summary) + "\n", encoding="utf-8"
    )
    print("\n".join(summary))
    print(f"\nFigura compuesta: {combined_path}")


if __name__ == "__main__":
    generar_evidencia()
