"""Genera referencias TFLite del modelo existente y 81 entradas; nunca entrena."""
from __future__ import annotations
import hashlib
import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "despliegue_stm32/dnn_int8_180mhz_2026-10-23"
PRIOR = ROOT / "despliegue_stm32/escalado_cuantizacion_180mhz_2026-10-23"
MODEL = ROOT / "resultados/cwru/modelo_features_14/canonico_seed42_int8/modelo_cwru_de_fe_seed42_int8.tflite"
MODEL_SHA = "cabc6e967b91691287c0cd8b899958d6784f0712b9bc956f68565273c9652596"


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def require(condition, message):
    if not condition: raise RuntimeError(message)
def fnv(raw):
    value = 2166136261
    for byte in raw: value = ((value ^ byte) * 16777619) & 0xffffffff
    return value
def save(name, data):
    with (OUT / name).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2); stream.write("\n")


def main():
    require(not (OUT / "REFERENCIA_DNN.json").exists(), "No sobrescribir referencia")
    model_bytes = MODEL.read_bytes()
    require(sha(MODEL) == MODEL_SHA and len(model_bytes) == 4064, "Modelo histórico diferente")
    export = ROOT / "modelos/cwru/canonico_seed42/firmware/cwru_model_data.cpp"
    exported_bytes = bytes(int(v, 16) for v in re.findall(r"0x([0-9a-fA-F]{2})\b", export.read_text(encoding="utf-8")))
    require(exported_bytes == model_bytes, "El modelo exportado no coincide")
    audit_path = PRIOR / "RESULTADOS_FISICOS_ENTRADA_COMPLETA.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    run_folder = PRIOR / "Validacion_PC/resultados" / audit["run"]
    run_path = run_folder / "resultados.json"
    require(sha(run_path) == audit["results_sha256"] and sha(run_folder / "uart_trace.jsonl") == audit["trace_sha256"],
            "Evidencia física de entrada distinta")
    run = json.loads(run_path.read_text(encoding="utf-8"))
    require(run["status"] == "PASS" and audit["passed_executions"] == 756, "Suite de entrada incompleta")
    cases = []
    for index in range(42):
        rows = [r for r in run["results"] if r["name"] == f"entm_{index:03d}"]
        require(len(rows) == 3 and all(r["observed"]["quantized"] == rows[0]["observed"]["quantized"] for r in rows),
                "Entradas EntM no estables")
        cases.append(dict(name=f"entm_{index:03d}", group="development", window_index=index,
                          input_int8=rows[0]["observed"]["quantized"]))
    for value in (-128, -9, 0, 1, 127):
        cases.append(dict(name=f"constant_{value}", group="synthetic", input_int8=[value]*14))
    for offset in (0, 1):
        cases.append(dict(name=f"alternating_{offset}", group="synthetic", input_int8=[-128 if (i+offset)%2 == 0 else 127 for i in range(14)]))
    rng = np.random.default_rng(20261023)
    for index in range(32):
        cases.append(dict(name=f"random_{index:02d}", group="synthetic", input_int8=rng.integers(-128,128,14,dtype=np.int16).astype(int).tolist()))
    require(len(cases) == 81, "Catálogo inesperado")
    resolvers = {"primary": tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
                 "secondary_reference": tf.lite.experimental.OpResolverType.BUILTIN_REF}
    metadata = {}
    for label, resolver in resolvers.items():
        interpreter = tf.lite.Interpreter(model_content=model_bytes, num_threads=1, experimental_op_resolver_type=resolver)
        interpreter.allocate_tensors()
        inp, out = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
        require(inp["dtype"] == out["dtype"] == np.int8 and inp["shape"].tolist() == [1,14] and out["shape"].tolist() == [1,3], "IO incompatible")
        require(inp["quantization"] == (0.033691830933094025,-9) and out["quantization"] == (0.00390625,-128), "Cuantización inesperada")
        ops = [o["op_name"] for o in interpreter._get_ops_details()]
        require(ops == ["FULLY_CONNECTED"]*3 + ["SOFTMAX"], "Operadores inesperados o delegado activo")
        metadata[label] = dict(resolver=resolver.name, num_threads=1, operators=ops,
            input_shape=[1,14], output_shape=[1,3], input_scale=inp["quantization"][0], input_zero_point=-9,
            output_scale=out["quantization"][0], output_zero_point=-128)
        for case in cases:
            data = np.asarray(case["input_int8"], dtype=np.int8)
            case["input_hex"] = data.tobytes().hex()
            case["checksum_fnv1a"] = f"{fnv(data.tobytes()):08x}"
            interpreter.set_tensor(inp["index"], data.reshape(1,14))
            interpreter.invoke()
            q = interpreter.get_tensor(out["index"])[0].astype(int)
            case[label] = dict(output_int8=q.tolist(), predicted_class=int(np.argmax(q)),
                               probabilities=((q+128)/256).tolist())
    differences = [np.abs(np.asarray(c["primary"]["output_int8"])-np.asarray(c["secondary_reference"]["output_int8"])) for c in cases]
    report = dict(status="PC_REFERENCES_READY_NOT_ARM_VALIDATED", model_sha256=MODEL_SHA,
        model_bytes=4064, model_fnv1a=f"{fnv(model_bytes):08x}", metadata=metadata,
        policy=dict(max_output_code_absolute_difference=1, probability_equivalent=1/256,
                    class_must_match_primary=True, repeated_outputs_must_be_exact=True,
                    primary_must_not_change_after_observing_hardware=True),
        cases=cases, development_cases=42, synthetic_cases=39,
        cpu_reference_comparison=dict(maximum_code_difference=int(np.max(differences)),
            exact_cases=sum(not np.any(d) for d in differences),
            class_disagreements=sum(c["primary"]["predicted_class"] != c["secondary_reference"]["predicted_class"] for c in cases)),
        tensorflow=tf.__version__, numpy=np.__version__, python=sys.version, executable=sys.executable,
        model_training_executed=False, model_conversion_executed=False, outer_test_loaded=False,
        firmware_executed=False, serial_port_opened=False, predictive_metrics_computed=False)
    inputs = [MODEL, export, audit_path, run_path, run_folder / "uart_trace.jsonl", Path(__file__).resolve(), OUT / "CONTRATO_PREVIO_DNN.md"]
    save("REFERENCIA_DNN.json", report)
    save("REFERENCIA_ENTRADAS_SHA256.json", {p.relative_to(ROOT).as_posix(): sha(p) for p in inputs})
    save("REFERENCIA_MANIFEST_SHA256.json", {name: sha(OUT/name) for name in ("REFERENCIA_DNN.json", "REFERENCIA_ENTRADAS_SHA256.json")})
    print(json.dumps({k:v for k,v in report.items() if k != "cases"}, ensure_ascii=True, indent=2), flush=True)


if __name__ == "__main__": main()
