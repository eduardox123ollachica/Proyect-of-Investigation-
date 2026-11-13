"""Modelo numerico host: no ejecuta el C ARM ni CMSIS ni mide hardware."""
from pathlib import Path
import ctypes
import hashlib
import json
import math
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "despliegue_stm32/mejoras_entropia_180mhz_2026-10-23"
N, BINS = 4096, 2049
CRT = ctypes.CDLL("ucrtbase.dll")
HOST_LOGF = CRT.logf
HOST_LOGF.argtypes, HOST_LOGF.restype = [ctypes.c_float], ctypes.c_float
FLOAT_MIN = float(np.finfo(np.float32).tiny)


def mixed_log(probability):
    with np.errstate(all="ignore"):
        pf = float(np.float32(probability))
    if not (FLOAT_MIN <= pf < 1.0):
        return math.log(probability), True
    return float(HOST_LOGF(pf)), False


def entropy(powers, mode):
    energy = 0.0
    for power in powers:
        energy += float(power)
    value, count, fallback = 0.0, 0, 0
    if energy > 0:
        for power in powers:
            if power > 0:
                p = float(power)/energy
                if mode == 0:
                    value -= p*(math.log(p)/math.log(2.0))
                elif mode == 1:
                    value -= p*math.log(p)
                else:
                    logp, used_fallback = mixed_log(p)
                    fallback += used_fallback
                    value -= p*logp
                count += 1
        if count > 1:
            value /= math.log(BINS)/math.log(2.0) if mode == 0 else math.log(BINS)
        else:
            value = 0.0
    return value, fallback


def main():
    target = OUT / "PRUEBAS_NUMERICAS_HOST.json"
    if target.exists():
        raise FileExistsError("No sobrescribir pruebas")
    cases = []
    for name, index in (("dc_only", 0), ("nyquist_only", BINS-1), ("ordinary_bin_only", 1)):
        powers = np.zeros(BINS); powers[index] = 1
        cases.append((name, powers))
    cases.append(("zeros", np.zeros(BINS)))
    cases.append(("uniform", np.ones(BINS)))
    for name, second in (("equal_peak", 1.0), ("near_one", 1e-10), ("float_subnormal", 1e-39), ("float_underflow", 1e-160)):
        powers = np.zeros(BINS); powers[0], powers[1] = 1, second
        cases.append((name, powers))
    rng = np.random.default_rng(42)
    for i in range(12):
        cases.append(("dynamic_random_"+str(i), np.power(10.0, rng.uniform(-80, 60, BINS))))
    bundle_path = OUT / "Validacion_Double/ventanas_desarrollo.npz"
    assert hashlib.sha256(bundle_path.read_bytes()).hexdigest() == "ff09491a486716b7176986cc504138a726e003389cb93b657565a3c422c234cb"
    with np.load(bundle_path) as bundle:
        for i, window in enumerate(bundle["windows"]):
            for channel in range(2):
                signal = window[:, channel]
                mean = sum(float(v) for v in signal)/N
                fft = np.fft.rfft((signal.astype(np.float64)-mean).astype(np.float32))
                real, imag = fft.real.astype(np.float32).astype(np.float64), fft.imag.astype(np.float32).astype(np.float64)
                cases.append((f"development_{i}_channel_{channel}", real*real+imag*imag))
    summaries = {}
    for mode in (1, 2):
        errors64, errors32, relative32, changed32, fallbacks = [], [], [], 0, 0
        for name, powers in cases:
            reference, _ = entropy(powers, 0)
            result, fallback = entropy(powers, mode)
            ref32, out32 = np.float32(reference), np.float32(result)
            assert math.isfinite(result) and np.isclose(out32, ref32, rtol=.001, atol=.00001), name
            errors64.append(abs(result-reference))
            errors32.append(abs(float(out32)-float(ref32)))
            relative32.append(abs(float(out32)-float(ref32))/max(abs(float(ref32)), 1e-30))
            changed32 += ref32.tobytes() != out32.tobytes()
            fallbacks += fallback
        summaries[str(mode)] = dict(cases=len(cases), all_within_original_tolerance=True,
            max_absolute_error_float64=max(errors64), max_absolute_error_float32=max(errors32),
            max_relative_error_float32=max(relative32), changed_float32_results=int(changed32),
            double_fallback_evaluations=int(fallbacks))
    for p in (1e-160, FLOAT_MIN/2, 0.9999999999, 1.0):
        logp, fallback = mixed_log(p)
        assert fallback and logp == math.log(p)
    assert not mixed_log(1e-5)[1]
    source = OUT / "copia_aislada/Prueba_2_CWRU_Extraccion/Src/cwru_feature_extraction.c"
    report = dict(status="PASS", source_c_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        cases_per_mode=len(cases), development_spectra=84, synthetic_cases=len(cases)-84,
        variants=summaries, extreme_probability_fallback_tests_passed=True,
        rtol=.001, atol=.00001, host_logf_library="Windows UCRT logf; NOT ARM/newlib logf",
        actual_arm_c_or_cmsis_executed=False, hardware_executed=False,
        limitations=["Python models and host UCRT logf test numerical plausibility, not the compiled ARM implementation.",
            "NumPy spectra are not CMSIS spectra; physical equivalence and timing must be measured.",
            "Bit identity is not required or presumed for either new mode; existing tolerances are unchanged."])
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
