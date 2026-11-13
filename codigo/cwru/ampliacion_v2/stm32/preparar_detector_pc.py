"""Prepara referencias explícitas sin entrenar, abrir test 3 HP o acceder a COM."""
import json
import re
import sys
from pathlib import Path
import numpy as np
from contrato_detector_cwru import INDICES, bits, floats, raw_hex, fnv, evaluate
from auditar_validacion_o3 import sha

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / 'despliegue_stm32/detector_normalidad_180mhz_2026-10-23'
PRIOR = ROOT / 'despliegue_stm32/escalado_cuantizacion_180mhz_2026-10-23'
RES = ROOT / 'modelos/cwru/canonico_seed42/firmware'
NPZ = ROOT / 'resultados/cwru/modelo_features_14/kmeans_normal_lolo_sin_frecuencia_dominante/kmeans_normal_final_float32.npz'


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def save(name, value):
    with (OUT/name).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def main():
    require(not (OUT/'REFERENCIA_DETECTOR.json').exists(), 'No sobrescribir referencia')
    source = RES/'cwru_normality_params.cpp'
    require(sha(source) == 'd04f04476227f48b6dc489088de5793cdb4235aa6965adbb126e6ca020493808', 'C histórico distinto')
    require(sha(NPZ) == '5aa9fd59fae5ed3c42fa3d8b51c4ef6b1123894fcd6c1145c9040cc9ae7d22ca', 'NPZ histórico distinto')
    text = source.read_text(encoding='utf-8')
    def array(name):
        match = re.search(r'\b'+name+r'\s*(?:\[[^\]]*\])+\s*=\s*\{(.*?)\};', text, re.S)
        require(match is not None, 'Array ausente: '+name)
        return np.asarray([float(v) for v in re.findall(r'([-+]?\d+\.\d+e[-+]\d+)f', match[1])], dtype=np.float32)
    mean = array('cwru_normality_scaler_mean')
    scale = array('cwru_normality_scaler_scale')
    centroid = array('cwru_normality_centroids')
    threshold = np.float32(re.search(r'threshold_d2\s*=\s*([^f;]+)f;', text)[1])
    exported_indices = [int(v) for v in re.search(r'feature_indices[^=]+\s*=\s*\{([^}]+)', text)[1].split(',') if v.strip()]
    require(exported_indices == INDICES and mean.shape == scale.shape == centroid.shape == (12,), 'Dimensiones/índices distintos')
    require(np.isfinite(mean).all() and np.isfinite(scale).all() and (scale > 0).all() and np.isfinite(centroid).all(), 'Parámetros no válidos')
    with np.load(NPZ, allow_pickle=False) as original:
        require(int(original['selected_k']) == 1, 'K distinto')
        for name, values in [('scaler_mean',mean),('scaler_scale',scale),('centroids',centroid),('threshold',[threshold])]:
            require(bits(original[name]) == bits(values), 'Exportación distinta del NPZ: '+name)
    param_raw = bytes(INDICES) + np.concatenate((mean, scale, centroid, [threshold])).astype('<f4').tobytes()
    params = dict(indices=INDICES, mean_bits=bits(mean), scale_bits=bits(scale), centroid_bits=bits(centroid),
        threshold_bits=bits([threshold])[0], threshold_value=float(threshold), k=1,
        params_fnv1a=f'{fnv(param_raw):08x}', source_sha256=sha(source), npz_sha256=sha(NPZ))
    case_path = PRIOR/'auditoria_pc/casos_validacion.json'
    prior_seal = json.loads((PRIOR/'MANIFEST_SHA256.json').read_text(encoding='utf-8'))
    # El sello previo usa rutas relativas al proyecto.
    require(sha(case_path) == prior_seal[case_path.relative_to(ROOT).as_posix()], 'Entradas previas modificadas')
    prior_cases = json.loads(case_path.read_text(encoding='utf-8'))
    cases, negatives, historical = [], [], []
    def add(name, mode, words, group, **extra):
        expected = evaluate(mode, words, params)
        cases.append(dict(name=name, mode=mode, group=group, input_bits=words,
            input_hex=raw_hex(words), checksum_fnv1a=f'{fnv(bytes.fromhex(raw_hex(words))):08x}', expected=expected, **extra))
    for label in ('entm', 'cache'):
        rows = [r for r in prior_cases if r['group'] == label]
        require(len(rows) == 42 and [r['index'] for r in rows] == list(range(42)), 'Desarrollo incompleto')
        for row in rows:
            add(row['name'], 'RUN', row['input_bits'], 'development_'+label, index=row['index'])
            current = cases[-1]['expected']
            terms = floats(current['term_bits'])
            old_score = np.sum(terms, dtype=np.float32)
            exact_sum = np.sum(terms.astype(np.float64), dtype=np.float64)
            sequential = floats([current['score_bits']])[0]
            require(bool(old_score > threshold) == current['rejected'], 'La acumulación cambia una decisión de desarrollo; revisar')
            historical.append(dict(name=row['name'], sequential_bits=current['score_bits'], numpy_sum_bits=bits([old_score])[0],
                absolute_difference=float(abs(float(sequential)-float(old_score))), historical_decision_equal=True,
                double_sum_diagnostic=float(exact_sum), rejected=current['rejected']))
    base = np.zeros(14, dtype=np.float32)
    base[INDICES] = mean
    add('means', 'RUN', bits(base), 'synthetic')
    for sign in (-1, 1):
        x = base.copy(); x[[4,11]] = np.float32(sign)*np.finfo(np.float32).max
        add('ignored_frequencies_'+str(sign), 'RUN', bits(x), 'synthetic')
        require(cases[-1]['expected'] == cases[-2 if sign == -1 else -3]['expected'], 'Columnas excluidas alteran referencia')
    for name, z in [('centroid',centroid),('zeros',np.zeros(12)),('ones',np.ones(12)),('minus_ones',-np.ones(12))]:
        add(name, 'DIST', bits(z), 'synthetic')
    for i in range(12):
        center = np.float32(centroid[i] + np.sqrt(threshold, dtype=np.float32))
        for suffix, value in [('below',np.nextafter(center,-np.inf,dtype=np.float32)),('center',center),('above',np.nextafter(center,np.inf,dtype=np.float32))]:
            z=centroid.copy(); z[i]=value
            add(f'boundary_{i:02d}_{suffix}', 'DIST', bits(z), 'boundary_diagnostic')
    rng = np.random.default_rng(20261023)
    for i in range(32):
        add(f'random_{i:02d}', 'DIST', bits(rng.uniform(-8,8,12).astype(np.float32)), 'synthetic')
    for name, score in [('zero',np.float32(0)),('negative_zero',np.float32(-0.)),
        ('below',np.nextafter(threshold,-np.inf,dtype=np.float32)),('equal',threshold),
        ('above',np.nextafter(threshold,np.inf,dtype=np.float32)),('maximum',np.finfo(np.float32).max)]:
        add('compare_'+name, 'CHECK', bits([score]), 'comparison_diagnostic')
    require(len(cases) == 165, 'Catálogo positivo distinto')
    by_name = {c['name']:c for c in cases}
    require(not by_name['compare_equal']['expected']['rejected'] and not by_name['compare_below']['expected']['rejected']
        and by_name['compare_above']['expected']['rejected'], 'Comparador estricto inválido')
    for i in range(42):
        a,b=by_name[f'entm_{i:03d}']['expected'],by_name[f'cache_{i:03d}']['expected']
        require(a['rejected'] == b['rejected'], 'EntM/control cambian decisión: revisar sin cambiar umbral')
    bad = []
    for label, value, index in [('nan_used',np.nan,0),('nan_ignored',np.nan,4),('inf',np.inf,0),('minus_inf',-np.inf,0),('overflow_scaling',np.finfo(np.float32).max,0)]:
        x=base.copy();x[index]=value;bad.append((label,'RUN',bits(x)))
    z=centroid.copy();z[0]=np.float32(1e30);bad.append(('overflow_square','DIST',bits(z)))
    z=np.full(12,np.sqrt(np.float32(np.finfo(np.float32).max/8)),dtype=np.float32)
    bad.append(('overflow_sum','DIST',bits(z)))
    for label,value in [('negative',-1.),('nan',np.nan),('inf',np.inf)]:
        bad.append(('score_'+label,'CHECK',bits([value])))
    for name,mode,words in bad:
        try:
            evaluate(mode,words,params)
        except ValueError as exc:
            negatives.append(dict(name=name,mode=mode,input_bits=words,input_hex=raw_hex(words),
                checksum_fnv1a=f'{fnv(bytes.fromhex(raw_hex(words))):08x}',expected_reason=str(exc)))
        else:
            raise RuntimeError('Caso negativo no se rechaza: '+name)
    require(len(negatives) == 10, 'Catálogo negativo numérico distinto')
    report = dict(status='PC_REFERENCE_READY_NOT_ARM_VALIDATED',params=params,cases=cases,numeric_negative_cases=negatives,
        historical_reduction_comparison=historical, historical_changed_decisions=0,
        historical_exact_score_cases=sum(r['sequential_bits']==r['numpy_sum_bits'] for r in historical),
        historical_maximum_absolute_score_difference=max(r['absolute_difference'] for r in historical),
        training_executed=False, outer_test_loaded=False, serial_port_opened=False,
        numpy=np.__version__,python=sys.version)
    save('REFERENCIA_DETECTOR.json', report)
    sources=[source,RES/'cwru_normality_params.h',NPZ,case_path,OUT/'CONTRATO_PREVIO_DETECTOR.md',Path(__file__),Path(__file__).with_name('contrato_detector_cwru.py')]
    save('ENTRADAS_REFERENCIA_SHA256.json',{p.relative_to(ROOT).as_posix():sha(p) for p in sources})
    print(json.dumps(dict(status=report['status'],positive_cases=len(cases),numeric_negatives=len(negatives),
        params=params,historical_exact=report['historical_exact_score_cases'],historical_compared=len(historical),
        historical_max_difference=report['historical_maximum_absolute_score_difference']),indent=2))


if __name__ == '__main__':
    main()
