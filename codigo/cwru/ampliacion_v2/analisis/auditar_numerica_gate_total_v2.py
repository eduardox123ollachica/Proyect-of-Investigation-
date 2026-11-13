"""Audita aritmetica y tiempos de la puerta TotalV2 sin inferir ni usar COM."""
from pathlib import Path
import hashlib
import json
import math
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT/'despliegue_stm32/costo_total_representantes_v2_180mhz_2026-10-25'
RUN = OUT/'Validacion_PC/resultados/20261026T040903_561072Z'
REPORT = OUT/'AUDITORIA_NUMERICA_GATE_TOTAL.json'
HCLK = 180000000

def require(ok, message):
    if not ok: raise ValueError(message)

def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def words(values): return [f'{int(x):08x}' for x in np.asarray(values,dtype='<f4').view('<u4').ravel()]
def floats(values):
    result = np.asarray([int(x,16) for x in values],dtype='<u4').view('<f4')
    require(np.isfinite(result).all(), 'No finito')
    return result

def manifest(path, base):
    digest, name = path.with_suffix('.sha256').read_text(encoding='ascii').split()
    require(name == path.name and sha(path) == digest, 'Sello distinto')
    entries = read(path)
    for relative, expected in entries.items():
        target = (base/relative).resolve()
        require(target.is_relative_to(base.resolve()) and target.is_file() and sha(target)==expected,
                'Entrada modificada: '+relative)
    return entries

def independent_replay(feature_bits, rep):
    source = floats(feature_bits)
    selected = source[rep['feature_indices']]
    centered = np.subtract(selected,floats(rep['dnn_mean32_bits']),dtype=np.float32)
    z = np.divide(centered,floats(rep['dnn_scale32_bits']),dtype=np.float32)
    divided = np.divide(z,floats([rep['input_quantization']['scale_bits']])[0],dtype=np.float32)
    affine = np.add(divided,np.float32(rep['input_quantization']['zero_point']),dtype=np.float32)
    rounded = np.rint(affine)
    q = np.clip(rounded,-128,127).astype(np.int8).astype(int).tolist()
    low = sum(1<<i for i,value in enumerate(rounded) if value < -128)
    high = sum(1<<i for i,value in enumerate(rounded) if value > 127)
    dc = np.subtract(source[rep['detector_indices']],floats(rep['detector_mean32_bits']),dtype=np.float32)
    dz = np.divide(dc,floats(rep['detector_scale32_bits']),dtype=np.float32)
    delta = np.subtract(dz,floats(rep['centroid32_bits']),dtype=np.float32)
    terms = np.multiply(delta,delta,dtype=np.float32)
    score = np.float32(0)
    for term in terms: score = np.add(score,term,dtype=np.float32)
    threshold = floats([rep['threshold_bits']])[0]
    return dict(standardized_bits=words(z),affine_bits=words(affine),quantized=q,
        saturated_low=low,saturated_high=high,detector_z_bits=words(dz),detector_terms_bits=words(terms),
        score_bits=words([score])[0],rejected=bool(score>threshold))

def close(a,b): return math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12)

def main():
    require(not REPORT.exists(), 'No sobrescribir auditoria')
    gate_manifest = manifest(RUN/'MANIFEST_EVIDENCIAS_SHA256.json',RUN)
    resource_manifest = manifest(OUT/'recursos/MANIFEST_RECURSOS_SHA256.json',OUT/'recursos')
    for name in ('CATALOGO_TOTAL.json','BANCO_TOTAL.json','CATALOGO_CANONICO.json'):
        require(sha(RUN/name) == sha(OUT/'recursos'/name), 'Referencia de otra exportacion')
    result = read(RUN/'RESULTADO_TOTAL.json'); plan = read(RUN/'PLAN_EJECUTADO.json')
    cat = read(RUN/'CATALOGO_TOTAL.json'); bank = read(RUN/'BANCO_TOTAL.json')
    rep = cat['representatives'][0]
    require(rep['id']==0 and rep['dims']==14 and rep['det_dims']==12 and rep['feature_indices']==list(range(14))
            and rep['detector_indices']==[i for i in range(14) if i not in (4,11)], 'Otra representacion')
    row = next(r for r in bank['windows'] if r['global_id']==568)
    ref = bank['reference_by_rep_gid']['0']['568']
    entries = [json.loads(line) for line in (RUN/'MEDICIONES.jsonl').read_text(encoding='utf-8').splitlines()]
    require([e['seq'] for e in entries]==list(range(1,len(entries)+1)), 'Orden de mediciones')
    bases = [e for e in entries if e['type']=='BASE']
    selections = [e for e in entries if e['type']=='SELECT']
    runs = [e for e in entries if e['type']=='RUN']; profiles = [e for e in entries if e['type']=='PROFILE']
    require(len(bases)==len(selections)==len(profiles)==1 and len(runs)==3, 'Conteos incompletos')
    require(result['status']=='GATE_PASS_NOT_TOTAL_COST_VALIDATED' and result['mode']=='gate'
            and plan['schedule']==[[568,[0]]] and result['completed_primary_RUN']==3
            and result['completed_PROFILE']==result['completed_BASE']==1 and result['negative_rejections']==17,
            'Estado o plan distinto')
    for info in (result['firmware_info'],result['firmware_info_final']):
        require(info['hclk_hz']==HCLK and not int(info['fpscr'],16)&((3<<22)|(1<<24))
                and info['paired_read_cycles_min']==5 and info['calibration_samples']==64
                and info['overhead_subtracted'] is False, 'Reloj/FPSCR/calibracion distinto')
    base = bases[0]; selection = selections[0]; profile = profiles[0]
    arm, pc = floats(base['features_bits']),floats(row['features_pc_cache14_bits'])
    require(base['global_id']==568 and base['features_bits']==row['arm_features14_bits']
            and np.allclose(arm,pc,rtol=.001,atol=.00001) and np.array_equal(arm[[4,11]],pc[[4,11]]),
            'BASE no coincide con referencia o tolerancia')
    require(selection['rep_id']==0 and selection['arena_used']==1188 and selection['reused'] is False
            and selection['cycles']>0 and selection['excluded_from_primary_cost'], 'Inicializacion distinta')
    replay = independent_replay(base['features_bits'],rep)
    require(replay['quantized']==ref['int8_inputs'], 'Entrada ARM cuantizada difiere: no sirve oracle cache en esta auditoria')
    expected_class = int(np.argmax(ref['int8_outputs']))
    comparisons=[]
    for e in runs+profiles:
        pages=e['pages']
        require(e['rep_id']==0 and e['global_id']==568 and e['checksum']==row['checksum_fnv1a']
                and e['extractions']==1 and e['arena_used']==1188 and e['status']=='PASS', 'Identidad/estado distinto')
        require(pages['features_bits']==base['features_bits'] and all(pages[k]==v for k,v in replay.items()),
                'Intermedio distinto del replay independiente')
        require(e['score_bits']==replay['score_bits'] and e['rejected']==replay['rejected']
                and e['predicted_class']==expected_class and e['output_int8']==ref['int8_outputs'],
                'Salida/clase/detector distintos del oracle sellado')
        require(close(e['milliseconds'],e['total_cycles']*1000/HCLK), 'Conversion ciclos incorrecta')
        comparison=dict(feature_bit_differences=sum(a!=b for a,b in zip(base['features_bits'],ref['raw_features14_bits'])),
            quantized_value_differences=sum(a!=b for a,b in zip(replay['quantized'],ref['int8_inputs'])),
            output_value_differences=sum(a!=b for a,b in zip(e['output_int8'],ref['int8_outputs'])),
            class_differs=e['predicted_class']!=expected_class,
            detector_score_bit_differs=e['score_bits']!=ref['detector_score_bits'],
            detector_rejection_differs=e['rejected']!=ref['outside_normality'])
        require(comparison==e['comparison']['cached_PC'], 'Comparacion cache declarada no coincide')
        comparisons.append(comparison)
    stage_keys=('extract_cycles','input_cycles','nn_cycles','detector_cycles','residual_cycles')
    require([e['run_id'] for e in runs]==[1,2,3] and [e['repetition'] for e in runs]==[1,2,3]
            and all(e['mode']==0 and all(e[k]==0 for k in stage_keys) for e in runs), 'Principal instrumentado')
    require(profile['run_id']==4 and profile['mode']==1 and sum(profile[k] for k in stage_keys)==profile['total_cycles'],
            'Perfil no cierra')
    cycles=np.asarray([e['total_cycles'] for e in runs],dtype=np.float64)
    stats=dict(count=3,min_cycles=int(cycles.min()),max_cycles=int(cycles.max()),mean_cycles=float(cycles.mean()),
        mean_ms=float(cycles.mean()/180000),median_ms=float(np.median(cycles)/180000),
        p95_ms=float(np.percentile(cycles,95)/180000),sample_std_ms=float(cycles.std(ddof=1)/180000))
    declared=result['timing_statistics'][0]
    require(len(result['timing_statistics'])==1 and all(close(v,declared[k]) for k,v in stats.items()), 'Estadistica inconsistente')
    require(all(declared[k] is False for k in ('PROFILE_included','BASE_included','initialization_included','UART_included',
        'WCET_claimed','selector_enabled')) and result['selector_enabled'] is False and result['total_cost_validated'] is False,
        'Alcance excedido')
    differences=[dict(index=i,ARM_bits=a,PC_cache_bits=b,absolute_error=float(abs(float(arm[i])-float(pc[i]))))
                 for i,(a,b) in enumerate(zip(base['features_bits'],row['features_pc_cache14_bits'])) if a!=b]
    report=dict(status='PASS_NUMERIC_TIMING_GATE_AUDIT_NOT_TOTAL_COST_VALIDATED',gate=str(RUN),
        gate_manifest_resources=len(gate_manifest),exported_resources=len(resource_manifest),
        gate_manifest_sha256=sha(RUN/'MANIFEST_EVIDENCIAS_SHA256.json'),script_sha256=sha(__file__),
        data_files_sha256={n:sha(RUN/n) for n in ('MEDICIONES.jsonl','RESULTADO_TOTAL.json','PLAN_EJECUTADO.json','BANCO_TOTAL.json','CATALOGO_TOTAL.json')},
        representative_id=rep['representative_id'],global_window_id=568,primary_RUN=3,PROFILE=1,BASE=1,
        primary_statistics=stats,primary_min_ms=float(cycles.min()/180000),primary_max_ms=float(cycles.max()/180000),
        profile=dict(cycles={k:profile[k] for k in ('total_cycles',*stage_keys)},
                     microseconds={k:profile[k]/180 for k in ('total_cycles',*stage_keys)},excluded_from_primary_statistics=True),
        initialization=dict(cycles=selection['cycles'],milliseconds=selection['cycles']/180000,excluded_from_primary_statistics=True),
        arena_used_bytes=1188,arena_capacity_bytes=16384,only_first_model_runtime_allocation_verified=True,
        same_INT8_input_output_exact=True,oracle='Original sealed PC cache output, because recomputed ARM quantized input is exactly the same',
        float32_ARM_feature_replay_all_intermediates_exact=True,all_four_publications_identical=True,
        features_bit_exact_historical_ARM=True,features_tolerance_PC_pass=True,dominant_frequency_exact=True,
        PC_cache_feature_bit_differences=differences,PC_cache_comparison=comparisons[0],
        score_ARM=float(floats([profile['score_bits']])[0]),score_PC_cache=float(floats([ref['detector_score_bits']])[0]),
        threshold=float(floats([rep['threshold_bits']])[0]),predicted_class=expected_class,output_INT8=ref['int8_outputs'],
        outside_normality=replay['rejected'],saturated_low=replay['saturated_low'],saturated_high=replay['saturated_high'],
        paired_read_cycles_min=5,calibration_samples=64,overhead_subtracted=False,
        PC_cache_end_to_end_bit_exact_claimed=False,WCET_claimed=False,confidence_interval_claimed=False,
        total_cost_validated=False,selector_enabled=False,TFLite_invocations=0,new_physical_runs=0,
        limitations=['One model and one window only; 31 model allocations and full banks remain pending.',
                     'No programmer verification report: copied/declared ELF is not a readback identity proof of all board Flash.',
                     'Three timings are an initial gate, not the 84 samples per representative required by selector.'])
    with REPORT.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(report,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(status=report['status'],primary_statistics=stats,profile=report['profile'],
        score=report['score_ARM'],threshold=report['threshold'],cache_feature_differences=differences,report=str(REPORT)),ensure_ascii=False))

if __name__=='__main__': main()
