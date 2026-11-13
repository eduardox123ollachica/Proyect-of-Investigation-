"""Prueba UART de escalado/cuantización; envía 14 float32, NO señales completas.

Usar la copia entregada junto a PARAMETROS_PC.json, casos_validacion.json y
VALIDACION_MANIFEST.json. --self-test no abre COM ni ejecuta el C ARM.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import sys
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
MODES = {"RUN": 0, "QUANT": 1, "ROUND": 2}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def words(values):
    return [f"{int(v):08x}" for v in np.asarray(values, dtype="<f4").view("<u4").ravel()]


def floats(values):
    if len(values) != 14 or any(len(v) != 8 or any(c not in "0123456789abcdefABCDEF" for c in v) for v in values):
        raise ValueError("Se requieren 14 palabras hex de ocho dígitos")
    return np.asarray([int(v, 16) for v in values], dtype="<u4").view("<f4")


def checksum(values):
    h = 2166136261
    for byte in np.asarray(values, dtype="<f4").tobytes():
        h = ((h ^ byte) * 16777619) & 0xffffffff
    return h


def bundle():
    manifest = json.loads((HERE / "VALIDACION_MANIFEST.json").read_text(encoding="utf-8"))
    for name, digest in manifest.items():
        if Path(name).name != name or sha(HERE / name) != digest:
            raise ValueError("Recurso del cliente modificado: " + name)
    params = json.loads((HERE / "PARAMETROS_PC.json").read_text(encoding="utf-8"))
    cases = json.loads((HERE / "casos_validacion.json").read_text(encoding="utf-8"))
    ties = np.arange(-140, 141, dtype=np.float32) + np.float32(.5)
    boundaries = np.concatenate([ties, np.nextafter(ties, -np.inf, dtype=np.float32),
                                 np.nextafter(ties, np.inf, dtype=np.float32),
                                 np.array([-1e30, 1e30, -0., 0.], dtype=np.float32)])
    for start in range(0, len(boundaries), 14):
        values = np.resize(boundaries[start:start + 14], 14)
        cases.append(dict(name=f"round_boundary_{start}", mode="ROUND", group="synthetic", input_bits=words(values)))
    return params, cases


def expected(case, params):
    x = floats(case["input_bits"])
    if not np.isfinite(x).all():
        raise ValueError("Entrada no finita")
    mean, scale = floats(params["mean_bits"]), floats(params["scale_bits"])
    quant_scale = np.asarray([int(params["quant_scale_bits"], 16)], dtype="<u4").view("<f4")[0]
    with np.errstate(over="ignore", invalid="ignore"):
        centered = np.subtract(x, mean, dtype=np.float32)
        z = np.divide(centered, scale, dtype=np.float32) if case["mode"] == "RUN" else x.copy()
        affine = x.copy() if case["mode"] == "ROUND" else np.add(np.divide(z, quant_scale, dtype=np.float32),
                                                                 np.float32(params["zero_point"]), dtype=np.float32)
    if not np.isfinite(z).all() or not np.isfinite(affine).all():
        raise ValueError("Intermedio no finito")
    rounded = np.rint(affine)
    return dict(standardized_bits=words(z), affine_bits=words(affine),
                quantized=np.clip(rounded, -128, 127).astype(np.int8).astype(int).tolist(),
                saturated_low=sum(1 << j for j, value in enumerate(rounded) if value < -128),
                saturated_high=sum(1 << j for j, value in enumerate(rounded) if value > 127))


def command(case):
    values = floats(case["input_bits"])
    return f'CWRUIN_{case["mode"]},{checksum(values):08x},' + ",".join(case["input_bits"])


def exchange(port, cmd, prefix, timeout, trace, allow_error=False):
    payload = (cmd + "\r\n").encode("ascii")
    if len(payload) >= 1024:
        raise ValueError("Trama demasiado larga")
    trace.append(dict(direction="TX", command=cmd, bytes=len(payload)))
    if port.write(payload) != len(payload):
        raise IOError("Escritura UART incompleta")
    port.flush()
    deadline, last = time.monotonic() + timeout, []
    while time.monotonic() < deadline:
        raw = port.readline()
        if not raw:
            continue
        line = raw.decode("ascii", errors="strict").strip()
        if not line:
            continue
        trace.append(dict(direction="RX", line=line)); last.append(line)
        if line.startswith("CWRUIN_ERROR,"):
            if allow_error:
                return line
            raise RuntimeError(line)
        if line.startswith(prefix):
            return line
    raise TimeoutError(f"No llegó {prefix}; últimas respuestas={last[-5:]}")


def info(port, params, timeout, trace):
    line = exchange(port, "CWRUIN_INFO", "CWRUIN_INFO,", timeout, trace)
    fields = line.split(",")
    if len(fields) != 9:
        raise ValueError("INFO incompatible: " + line)
    version, count, hclk, dwt = map(int, fields[1:5])
    fpscr = int(fields[7], 16)
    if (version, count, hclk, dwt) != (1, 14, 180000000, 1) or fields[5] != params["quant_scale_bits"] or int(fields[6]) != params["zero_point"]:
        raise ValueError("Firmware o cuantización incompatibles: " + line)
    if fields[8] != "24275e363e29573e7a158d44b00099d2224be9a783ac87654f2e1575fc86f1a5" or fpscr & 0x00c00000:
        raise ValueError("Parámetros distintos o modo de redondeo FPU no RNE")
    return dict(protocol=version, features=count, hclk_hz=hclk, dwt_enabled=True,
                quant_scale_bits=fields[5], zero_point=int(fields[6]), fpscr=f"{fpscr:08x}",
                flush_to_zero=bool(fpscr & (1 << 24)), params_source_sha256=fields[8])


def parse_result(line, case):
    fields = line.split(",")
    if len(fields) != 48 or fields[0] != "CWRUIN_RESULT" or int(fields[1]) != MODES[case["mode"]]:
        raise ValueError("Respuesta incompleta o modo incorrecto")
    if int(fields[2], 16) != checksum(floats(case["input_bits"])):
        raise ValueError("Respuesta corresponde a otra entrada")
    cycles, low, high = map(int, fields[3:6])
    z, affine = fields[6:20], fields[20:34]
    q = list(map(int, fields[34:]))
    if cycles <= 0 or not 0 <= low < (1 << 14) or not 0 <= high < (1 << 14) or any(v < -128 or v > 127 for v in q):
        raise ValueError("Ciclos, máscaras o códigos inválidos")
    if not np.isfinite(floats(z)).all() or not np.isfinite(floats(affine)).all():
        raise ValueError("Respuesta contiene intermedios no finitos")
    return dict(cycles=cycles, standardized_bits=z, affine_bits=affine, quantized=q,
                saturated_low=low, saturated_high=high)


def compare(observed, reference):
    return all(observed[key] == reference[key] for key in reference)


def negatives(params):
    zero = dict(mode="RUN", input_bits=["00000000"] * 14)
    valid = command(zero)
    bad_hash = valid.split(","); bad_hash[1] = f"{int(bad_hash[1], 16) ^ 1:08x}"
    nan = dict(mode="RUN", input_bits=["7fc00000"] + ["00000000"] * 13)
    inf = dict(mode="QUANT", input_bits=["7f800000"] + ["00000000"] * 13)
    huge = dict(mode="QUANT", input_bits=["7f7fffff"] * 14)
    return [(valid[:-1], "FORMAT"), (valid + ",00000000", "FORMAT"),
            (",".join(bad_hash), "CHECKSUM"), (valid.replace(",00000000", ",g0000000", 1), "FORMAT"),
            (command(nan), "NONFINITE"), (command(inf), "NONFINITE"),
            (command(huge), "INTERMEDIATE"), ("CWRUIN_BAD", "UNKNOWN")]


class ClientTests(unittest.TestCase):
    def test_resource_and_all_case_references(self):
        params, cases = bundle()
        self.assertEqual(len(cases), 252)
        for case in cases:
            self.assertLess(len(command(case)) + 2, 1024)
            ref = expected(case, params)
            self.assertEqual(len(ref["quantized"]), 14)
            self.assertLess(48 + 28 * 9 + sum(len(str(q)) + 1 for q in ref["quantized"]), 512)

    def test_parser_and_corruption_detection(self):
        params, cases = bundle(); case = cases[0]; ref = expected(case, params)
        fields = ["CWRUIN_RESULT", str(MODES[case["mode"]]), f'{checksum(floats(case["input_bits"])):08x}', "250",
                  str(ref["saturated_low"]), str(ref["saturated_high"]), *ref["standardized_bits"],
                  *ref["affine_bits"], *map(str, ref["quantized"])]
        self.assertTrue(compare(parse_result(",".join(fields), case), ref))
        fields[-1] = str(int(fields[-1]) + 1)
        self.assertFalse(compare(parse_result(",".join(fields), case), ref))
        with self.assertRaises(ValueError): parse_result(",".join(fields[:-1]), case)
        fields[2] = "00000000"
        with self.assertRaises(ValueError): parse_result(",".join(fields), case)

    def test_ties_even_and_saturation(self):
        params, _ = bundle()
        x = np.array([-129.5, -128.5, -127.5, -.5, .5, 1.5, 126.5, 127.5, 128.5, -0., 0., 2.5, -2.5, -126.5], dtype=np.float32)
        result = expected(dict(mode="ROUND", input_bits=words(x)), params)
        self.assertEqual(result["quantized"], [-128, -128, -128, 0, 0, 2, 126, 127, 127, 0, 0, 2, -2, -126])
        self.assertEqual(result["saturated_low"], 1)
        self.assertEqual(result["saturated_high"], (1 << 7) | (1 << 8))

    def test_rounding_is_after_zero_point(self):
        self.assertNotEqual(int(np.rint(np.float32(.5) + np.float32(-9))), int(np.rint(np.float32(.5))) - 9)

    def test_invalid_words_and_nonfinite(self):
        with self.assertRaises(ValueError): floats(["0"] * 14)
        params, _ = bundle()
        with self.assertRaises(ValueError): expected(dict(mode="RUN", input_bits=["7fc00000"] * 14), params)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--port")
    select = parser.add_mutually_exclusive_group()
    select.add_argument("--index", type=int)
    select.add_argument("--all", action="store_true")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=3.)
    args = parser.parse_args()
    if args.self_test:
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ClientTests))
        return 0 if result.wasSuccessful() else 1
    if not args.port or not 1 <= args.repeat <= 100 or args.timeout <= 0:
        parser.error("Indicar --port; repeat entre 1 y 100; timeout positivo")
    params, cases = bundle()
    index = 0 if args.index is None else args.index
    selected = cases if args.all else [c for c in cases if c.get("group") == "entm" and c.get("index") == index]
    if not selected:
        parser.error("Índice de desarrollo fuera de 0..41")
    import serial  # Sólo se requiere para la prueba física, no para --self-test.
    folder = HERE / "resultados" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    folder.mkdir(parents=True, exist_ok=False)
    trace, results, rejected = [], [], []
    report = dict(status="ERROR", selected_cases=len(selected), repetitions=args.repeat,
                  firmware_info=None, mode="all" if args.all else "single_development_index",
                  physical_extraction_repeated=False, classifier_executed=False, training_executed=False)
    try:
        with serial.Serial(args.port, 115200, timeout=.2, write_timeout=args.timeout) as port:
            time.sleep(.2); port.reset_input_buffer()
            report["firmware_info"] = info(port, params, args.timeout, trace)
            print("Firmware:", report["firmware_info"], flush=True)
            for case in selected:
                ref = expected(case, params)
                for repetition in range(1, args.repeat + 1):
                    line = exchange(port, command(case), "CWRUIN_RESULT,", args.timeout, trace)
                    observed = parse_result(line, case)
                    passed = compare(observed, ref)
                    row = dict(name=case["name"], group=case["group"], mode=case["mode"], repetition=repetition,
                               status="PASS" if passed else "FAIL", input_bits=case["input_bits"], reference=ref, observed=observed)
                    results.append(row)
                    if not passed:
                        raise RuntimeError("Diferencia numérica: " + case["name"] + "; revisar resultados.json")
                last = results[-1]["observed"]
                print(f'{case["name"]}: PASS; {args.repeat} repeticiones; última={last["cycles"]} ciclos; '
                      f'{last["cycles"] * 1e6 / report["firmware_info"]["hclk_hz"]:.4f} us', flush=True)
            if args.all:
                for cmd, reason in negatives(params):
                    line = exchange(port, cmd, "CWRUIN_ERROR,", args.timeout, trace, allow_error=True)
                    if line.split(",")[1] != reason:
                        raise RuntimeError("Rechazo incorrecto de trama inválida: " + line)
                    rejected.append(dict(command=cmd, expected_reason=reason, response=line))
            report["status"] = "PASS"
    except Exception as exc:
        report["error"] = str(exc)
        print("ERROR:", exc, file=sys.stderr)
    finally:
        report.update(results=results, negative_tests=rejected, completed_executions=len(results),
                      client_sha256=sha(Path(__file__)), cases_sha256=sha(HERE / "casos_validacion.json"),
                      parameters_sha256=sha(HERE / "PARAMETROS_PC.json"), numpy=np.__version__, python=sys.version)
        timing = {}
        for mode in MODES:
            values = [r["observed"]["cycles"] for r in results if r["mode"] == mode and r["status"] == "PASS"]
            if values:
                a = np.asarray(values, dtype=float)
                timing[mode] = dict(count=len(values), mean_cycles=float(a.mean()), min_cycles=int(a.min()), max_cycles=int(a.max()))
        report["timing_by_mode_not_full_pipeline"] = timing
        (folder / "resultados.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (folder / "uart_trace.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in trace), encoding="utf-8")
        (folder / "MANIFEST_SHA256.json").write_text(json.dumps({p.name: sha(p) for p in folder.iterdir() if p.is_file()}, indent=2) + "\n", encoding="utf-8")
        print("Evidencias:", folder, flush=True)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
