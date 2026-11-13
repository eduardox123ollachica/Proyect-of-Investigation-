"""Auditoría estática de los ocho FlatBuffer; no ejecuta una red o un build.

Comprueba que los modelos congelados siguen dentro del dominio de la
corrección single-rounding ya validada. Esto no sustituye la prueba en ARM.
"""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import struct

import exportar_catalogo as catalog

ROOT = Path(__file__).resolve().parents[4]
READER = ROOT/'codigo/cwru/ampliacion_v2/analisis/auditar_elf_total_v2.py'
REPORT = catalog.OUT.parent/'DOMINIO_MODELOS_CANONICOS.json'


def multiplier(value):
    catalog.require(math.isfinite(value) and value > 0, 'Multiplicador no positivo/finito')
    fraction, shift = math.frexp(value)
    quantized = math.floor(fraction*(1 << 31)+.5)
    if quantized == 1 << 31:
        quantized //= 2
        shift += 1
    return quantized, shift


def audit(resources=catalog.OUT):
    resources = Path(resources)
    cat = catalog.verify_resources(resources)
    spec = importlib.util.spec_from_file_location('canon42_flatbuffer_reader', READER)
    reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reader)
    f32 = lambda value: struct.unpack('<f', struct.pack('<I', int(value, 16)))[0]
    rows = []
    for row in cat['representatives']:
        model = resources/row['model_path']
        fb = reader.FlatBuffer(model.read_bytes())
        topology = fb.validate(row)
        root = fb.unpack('<I', 0)[0]
        graph = fb.tables(root, 8)[0]
        tensors, operators = fb.tables(graph, 4), fb.tables(graph, 10)
        shifts = []
        for operator in operators[:3]:
            ins, outs = fb.vector(operator, 6, '<i'), fb.vector(operator, 8, '<i')
            inp, weights, out = (fb.tensor(tensors[i]) for i in (ins[0], ins[1], outs[0]))
            catalog.require(inp['type'] == weights['type'] == out['type'] == 9
                and len(out['shape']) == 2, 'FC fuera de la ruta INT8 2D auditada')
            catalog.require(len(weights['scale_bits']) == weights['shape'][0] > 1
                and all(z == 0 for z in weights['zero_points']), 'FC no per-channel simétrico')
            sin, sout = f32(inp['scale_bits'][0]), f32(out['scale_bits'][0])
            for scale in weights['scale_bits']:
                q, shift = multiplier(sin*f32(scale)/sout)
                catalog.require(q > 0 and -31 <= shift < 30, 'Shift FC fuera del dominio corregido')
                shifts.append(shift)
        catalog.require(len(shifts) == 33, 'Canales FC inesperados')
        soft_in = fb.tensor(tensors[fb.vector(operators[3], 6, '<i')[0]])
        beta = fb.scalar(fb.pointer(operators[3], 12), 4, '<f', 1.0)
        catalog.require(math.isfinite(beta) and beta > 0, 'Softmax beta no válido')
        real = min(beta*f32(soft_in['scale_bits'][0])*(1 << 26), (1 << 31)-1)
        q, shift = multiplier(real)
        catalog.require(real > 1 and q > 0 and 0 <= shift < 30, 'Softmax fuera del dominio auditado')
        rows.append(dict(id=row['id'], model_job_id=row['representative_id'],
            model_sha256=row['model_sha256'], topology=topology, FC_channels=33,
            FC_minimum_shift=min(shifts), FC_maximum_shift=max(shifts),
            softmax_quantized_multiplier=q, softmax_multiplier_shift=shift))
    catalog.require(len(rows) == 8, 'Ocho modelos requeridos')
    return dict(status='PASS_STATIC_MODEL_DOMAIN_PENDING_BUILD_AND_ARM', models=rows,
        total_FC_channels=264, catalog_sha256=cat['catalog_sha256'],
        resource_manifest_sha256=catalog.io.sha(resources/'MANIFEST_RECURSOS_SHA256.json'),
        reader_sha256=catalog.io.sha(READER), verifier_sha256=catalog.io.sha(__file__),
        training_fits=0, classifier_invocations=0, full_firmware_build=False,
        serial_opened=False, hardware_flashed=False, whole_TFLM_archive_rebuilt=False,
        limitation='Static tensor/shift checks only; compiled ELF review and physical equivalence remain pending.')


if __name__ == '__main__':
    result = audit()
    catalog.io.write(REPORT, result)
    print(json.dumps(dict(status=result['status'], models=8, FC_channels=264,
        report=str(REPORT)), ensure_ascii=True))
