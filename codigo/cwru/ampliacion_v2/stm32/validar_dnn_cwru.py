"""Valida la DNN INT8 aislada. No recibe señales ni ejecuta extracción/escalado.

La copia entregada requiere REFERENCIA_DNN.json y VALIDACION_MANIFEST.json.
--self-test no conecta COM ni ejecuta TFLM ARM.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
MODEL_SHA = "cabc6e967b91691287c0cd8b899958d6784f0712b9bc956f68565273c9652596"


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def fnv(raw):
    result = 2166136261
    for byte in raw: result = ((result ^ byte) * 16777619) & 0xffffffff
    return result
def bundle():
    manifest = json.loads((HERE / "VALIDACION_MANIFEST.json").read_text(encoding="utf-8"))
    for name, digest in manifest.items():
        if Path(name).name != name or sha(HERE / name) != digest: raise ValueError("Recurso modificado: " + name)
    ref = json.loads((HERE / "REFERENCIA_DNN.json").read_text(encoding="utf-8"))
    if ref["model_sha256"] != MODEL_SHA or len(ref["cases"]) != 81 or ref["policy"]["max_output_code_absolute_difference"] != 1:
        raise ValueError("Referencia o criterio distintos del contrato")
    return ref
def command(case):
    raw = bytes.fromhex(case["input_hex"])
    if len(raw) != 14 or np.frombuffer(raw, dtype=np.int8).astype(int).tolist() != case["input_int8"]:
        raise ValueError("Entrada INT8 incoherente")
    if f"{fnv(raw):08x}" != case["checksum_fnv1a"]: raise ValueError("Checksum de referencia distinto")
    return f'CWRUNN_RUN,{fnv(raw):08x},' + raw.hex()
def exchange(port, cmd, prefix, timeout, trace, allow_error=False):
    payload = (cmd + "\r\n").encode("ascii")
    trace.append(dict(direction="TX", command=cmd, bytes=len(payload)))
    if port.write(payload) != len(payload): raise IOError("Escritura incompleta")
    port.flush(); deadline = time.monotonic()+timeout; last=[]
    while time.monotonic() < deadline:
        raw = port.readline()
        if not raw: continue
        line = raw.decode("ascii", errors="strict").strip()
        if not line: continue
        trace.append(dict(direction="RX", line=line)); last.append(line)
        if line.startswith("CWRUNN_ERROR,"):
            if allow_error: return line
            parts=line.split(",")
            message=bytes.fromhex(parts[3]).decode("utf-8", errors="replace") if len(parts)>3 else ""
            raise RuntimeError(line + "; TFLM: " + message)
        if line.startswith(prefix): return line
    raise TimeoutError(f"No llegó {prefix}; últimas respuestas={last[-5:]}")
def info(port, ref, timeout, trace):
    fields=exchange(port,"CWRUNN_INFO","CWRUNN_INFO,",timeout,trace).split(",")
    if len(fields)!=16: raise ValueError("INFO incompatible")
    if tuple(map(int,fields[1:7]))!=(1,14,3,180000000,1,4064) or fields[7]!=ref["model_fnv1a"] or fields[8]!=MODEL_SHA:
        raise ValueError("Modelo, reloj o protocolo distintos")
    cap,used,state=map(int,fields[9:12])
    if cap!=16384 or not 0<=used<=cap or fields[12:]!=["3d0a0072","-9","3b800000","-128"]:
        raise ValueError("Arena o parámetros incompatibles")
    return dict(protocol=1,inputs=14,outputs=3,hclk_hz=180000000,dwt_enabled=True,
        model_bytes=4064,model_fnv1a=fields[7],model_sha256_declared=fields[8],
        arena_capacity=cap,arena_used=used,initialization_status=state)
def initialize(port, timeout, trace):
    fields=exchange(port,"CWRUNN_INIT","CWRUNN_READY,",timeout,trace).split(",")
    if len(fields)!=5: raise ValueError("READY incompatible")
    cycles,cap,used,reused=map(int,fields[1:])
    if cap!=16384 or not 0<used<=cap or reused not in (0,1) or (cycles<=0 if reused==0 else cycles!=0):
        raise ValueError("Inicialización o arena inválidas")
    return dict(cycles=cycles,arena_capacity=cap,arena_used=used,reused=bool(reused),
                measured_now=not bool(reused),milliseconds=cycles/180000 if not reused else None)
def parse_result(line,case):
    f=line.split(",")
    if len(f)!=9 or f[0]!="CWRUNN_RESULT" or f[1]!=case["checksum_fnv1a"]: raise ValueError("Respuesta incorrecta")
    invoke,call,pred,q0,q1,q2,arena=map(int,f[2:])
    q=[q0,q1,q2]
    if invoke<=0 or call<invoke or not 0<arena<=16384 or any(not -128<=v<=127 for v in q) or pred!=int(np.argmax(q)):
        raise ValueError("Salida, argmax o ciclos inválidos")
    return dict(invoke_cycles=invoke,call_cycles=call,predicted_class=pred,output_int8=q,arena_used=arena)
def compare(observed,case):
    delta=np.abs(np.asarray(observed["output_int8"])-np.asarray(case["primary"]["output_int8"]))
    class_match=observed["predicted_class"]==case["primary"]["predicted_class"]
    return dict(passed=bool(class_match and np.max(delta)<=1),class_match=class_match,
                output_bit_exact=bool(not np.any(delta)),max_code_difference=int(np.max(delta)))
def negatives(case):
    cmd=command(case); parts=cmd.split(","); parts[1]=f"{int(parts[1],16)^1:08x}"
    return [(cmd[:-1],"FORMAT"),(cmd+"00","FORMAT"),(",".join(parts),"CHECKSUM"),
            (cmd[:20]+"g"+cmd[21:],"FORMAT"),("CWRUNN_BAD","UNKNOWN")]


class ClientTests(unittest.TestCase):
    def test_bundle_and_encoding(self):
        ref=bundle(); self.assertEqual(len(ref["cases"]),81)
        for case in ref["cases"]:
            self.assertLess(len(command(case)),1024)
            self.assertEqual(len(bytes.fromhex(case["input_hex"])),14)
            self.assertEqual(case["primary"]["predicted_class"],int(np.argmax(case["primary"]["output_int8"])))
    def test_parser(self):
        case=bundle()["cases"][0]; q=case["primary"]["output_int8"]
        line=f'CWRUNN_RESULT,{case["checksum_fnv1a"]},1000,1100,{int(np.argmax(q))},'+",".join(map(str,q))+",3000"
        self.assertTrue(compare(parse_result(line,case),case)["passed"])
        with self.assertRaises(ValueError): parse_result(line[:-5],case)
        with self.assertRaises(ValueError): parse_result(line.replace(",1000,1100,",",1101,1100,"),case)
    def test_numeric_policy_and_class(self):
        case=dict(primary=dict(output_int8=[10,0,-10],predicted_class=0))
        self.assertTrue(compare(dict(output_int8=[9,1,-10],predicted_class=0),case)["passed"])
        self.assertFalse(compare(dict(output_int8=[8,0,-10],predicted_class=0),case)["passed"])
        self.assertFalse(compare(dict(output_int8=[10,0,-10],predicted_class=1),case)["passed"])
    def test_byte_corruption_rejected(self):
        case=dict(bundle()["cases"][0]); case["input_hex"]="00"*14
        with self.assertRaises(ValueError): command(case)


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--port"); p.add_argument("--self-test",action="store_true")
    group=p.add_mutually_exclusive_group(); group.add_argument("--index",type=int); group.add_argument("--all",action="store_true")
    p.add_argument("--repeat",type=int,default=3); p.add_argument("--timeout",type=float,default=5.)
    args=p.parse_args()
    if args.self_test:
        tests=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ClientTests))
        return 0 if tests.wasSuccessful() else 1
    if not args.port or not 1<=args.repeat<=100 or args.timeout<=0: p.error("Indicar --port y argumentos válidos")
    ref=bundle(); index=0 if args.index is None else args.index
    cases=ref["cases"] if args.all else [c for c in ref["cases"] if c.get("window_index")==index and c["group"]=="development"]
    if not cases: p.error("Índice fuera de 0..41")
    import serial
    folder=HERE/"resultados"/datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"); folder.mkdir(parents=True,exist_ok=False)
    trace=[]; results=[]; rejected=[]
    report=dict(status="ERROR",mode="all" if args.all else "single",cases=len(cases),repetitions=args.repeat,
                extraction_executed=False,scaling_executed=False,detector_executed=False,training_executed=False)
    try:
        with serial.Serial(args.port,115200,timeout=.2,write_timeout=args.timeout) as port:
            time.sleep(.2); port.reset_input_buffer()
            report["info_before_init"]=info(port,ref,args.timeout,trace)
            report["initialization"]=initialize(port,args.timeout,trace)
            report["firmware_info"]=info(port,ref,args.timeout,trace)
            if report["firmware_info"]["initialization_status"]!=0 or report["firmware_info"]["arena_used"]!=report["initialization"]["arena_used"]:
                raise RuntimeError("Estado incoherente después de inicializar")
            print("Firmware:",report["firmware_info"],flush=True)
            print("Inicialización:",report["initialization"],flush=True)
            for case in cases:
                previous=None
                for repetition in range(1,args.repeat+1):
                    observed=parse_result(exchange(port,command(case),"CWRUNN_RESULT,",args.timeout,trace),case)
                    comparison=compare(observed,case)
                    stable=previous is None or previous==observed["output_int8"]
                    passed=comparison["passed"] and stable and observed["arena_used"]==report["initialization"]["arena_used"]
                    results.append(dict(name=case["name"],group=case["group"],repetition=repetition,
                        status="PASS" if passed else "FAIL",input_hex=case["input_hex"],
                        primary_reference=case["primary"],secondary_reference=case["secondary_reference"],
                        observed=observed,comparison=comparison,repeated_output_stable=stable))
                    if not passed: raise RuntimeError("Diferencia de inferencia: "+case["name"]+"; conservar evidencia")
                    previous=observed["output_int8"]
                print(f'{case["name"]}: PASS; clase={observed["predicted_class"]}; salida={observed["output_int8"]}; '
                      f'última inferencia={observed["invoke_cycles"]/180:.4f} us; exacta PC={comparison["output_bit_exact"]}',flush=True)
            if args.all:
                for cmd,reason in negatives(cases[0]):
                    line=exchange(port,cmd,"CWRUNN_ERROR,",args.timeout,trace,allow_error=True)
                    if line.split(",")[1]!=reason: raise RuntimeError("Rechazo incorrecto: "+line)
                    rejected.append(dict(command=cmd,reason=reason,response=line))
            report["status"]="PASS"
    except Exception as exc:
        report["error"]=str(exc); print("ERROR:",exc,file=sys.stderr)
    finally:
        report.update(results=results,negative_tests=rejected,completed_executions=len(results),
            client_sha256=sha(Path(__file__)),reference_sha256=sha(HERE/"REFERENCIA_DNN.json"),
            numpy=np.__version__,python=sys.version)
        (folder/"resultados.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        (folder/"uart_trace.jsonl").write_text("".join(json.dumps(row)+"\n" for row in trace),encoding="utf-8")
        (folder/"MANIFEST_SHA256.json").write_text(json.dumps({f.name:sha(f) for f in folder.iterdir() if f.is_file()},indent=2)+"\n",encoding="utf-8")
        print("Evidencias:",folder,flush=True)
    return 0 if report["status"]=="PASS" else 1


if __name__=="__main__": raise SystemExit(main())
