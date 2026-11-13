"""Prueba del modelo de almacenamiento/aritmetica; NO ejecuta C/CMSIS/STM32."""
from pathlib import Path
import hashlib
import json
import math
import struct
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "despliegue_stm32/cache_potencias_180mhz_2026-10-23"
N, BINS, BYTES, GUARD = 4096, 2049, 16384, 32


def power_at(packed, k):
    real = float(packed[0] if k == 0 else packed[1] if k == N//2 else packed[2*k])
    imag = 0.0 if k in (0, N//2) else float(packed[2*k+1])
    return real*real + imag*imag


def metrics(powers):
    energy, entropy = 0.0, 0.0
    dominant, dominant_index = powers[1], 1
    for k, power in enumerate(powers):
        energy += power
        if k > 1 and power > dominant:
            dominant, dominant_index = power, k
    count = 0
    if energy > 0:
        for power in powers:
            if power > 0:
                p = power/energy
                entropy -= p*(math.log(p)/math.log(2.0))
                count += 1
        entropy = entropy/(math.log(BINS)/math.log(2.0)) if count > 1 else 0.0
    return struct.pack("<dddI", energy, entropy, dominant, dominant_index)


def run_case(packed):
    original = [power_at(packed, k) for k in range(BINS)]
    memory = bytearray(b"\xa5"*GUARD + b"\x00"*BYTES + b"\x5a"*GUARD)
    dc = original[0]
    for k in range(1, BINS):
        struct.pack_into("<d", memory, GUARD+(k-1)*8, original[k])
    recovered = [dc] + [struct.unpack_from("<d", memory, GUARD+(k-1)*8)[0] for k in range(1, BINS)]
    assert memory[:GUARD] == b"\xa5"*GUARD and memory[-GUARD:] == b"\x5a"*GUARD
    assert all(struct.pack("<d", a) == struct.pack("<d", b) for a, b in zip(original, recovered))
    assert metrics(original) == metrics(recovered)


def main():
    target = OUT / "PRUEBAS_CACHE_SOFTWARE.json"
    if target.exists():
        raise FileExistsError("No sobrescribir prueba")
    assert (BINS-1)*8 == BYTES == N*4
    cases = []
    zero = np.zeros(N, dtype=np.float32)
    cases.append(("zero", zero.copy()))
    for name, index in (("dc_only", 0), ("nyquist_only", 1), ("bin1_only", 2), ("last_ordinary_bin", N-2)):
        data = zero.copy(); data[index] = 1
        cases.append((name, data))
    tied = zero.copy(); tied[2] = tied[4] = 1
    cases.append(("equal_peak_tie", tied))
    for name, value in (("all_ones", 1), ("all_negative", -1), ("small", 1e-20), ("large", 1e10)):
        cases.append((name, np.full(N, value, dtype=np.float32)))
    rng = np.random.default_rng(42)
    for seed_index in range(12):
        cases.append(("random_"+str(seed_index), rng.normal(size=N).astype(np.float32)))
    with np.load(OUT / "Validacion_PC/ventanas_desarrollo.npz") as bundle:
        for i, window in enumerate(bundle["windows"]):
            for channel in range(2):
                signal = window[:, channel]
                mean = sum(float(v) for v in signal)/N
                centered = (signal.astype(np.float64)-mean).astype(np.float32)
                spectrum = np.fft.rfft(centered)
                packed = np.empty(N, dtype=np.float32)
                packed[0], packed[1] = spectrum[0].real, spectrum[-1].real
                packed[2::2], packed[3::2] = spectrum[1:-1].real, spectrum[1:-1].imag
                cases.append((f"development_{i}_channel_{channel}", packed))
    for name, data in cases:
        run_case(data)
    source = OUT / "copia_aislada/Prueba_2_CWRU_Extraccion/Src/cwru_feature_extraction.c"
    report = dict(status="PASS", cases=len(cases), realistic_spectra=84, synthetic_cases=len(cases)-84,
        source_c_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        all_double_power_representations_preserved=True, all_spectral_metrics_identical_in_python_model=True,
        guard_bytes_unchanged=True, cache_bytes=BYTES, extra_cache_array_bytes=0,
        first_slot_offset=0, last_slot_offset=16376, last_slot_end_exclusive=16384,
        dc_stored_separately=True, case_names=[name for name, _ in cases],
        actual_c_or_cmsis_executed=False, hardware_executed=False,
        limitations=["These tests validate a Python model of byte storage and unchanged arithmetic, not the compiled C implementation.",
            "NumPy-derived spectra are test inputs, not CMSIS hardware results or timing measurements.",
            "Compiler checks and physical paired validation remain necessary."])
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k != "case_names"}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
