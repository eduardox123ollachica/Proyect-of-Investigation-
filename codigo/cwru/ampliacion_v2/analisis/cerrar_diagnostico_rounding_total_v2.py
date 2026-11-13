"""Consolida la divergencia ya reproducida; no ejecuta inferencia ni hardware."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'despliegue_stm32/costo_total_representantes_v2_180mhz_2026-10-25'
DEST=OUT/'diagnostico_divergencia_20261026T042424Z'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,x):
    with Path(p).open('x',encoding='utf-8',newline='\n') as f:
        json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def main():
    target=DEST/'CONCLUSION_DIAGNOSTICO.json'
    assert not target.exists(), 'No sobrescribir el diagnostico cerrado'
    case=read(DEST/'CASO_FALLO.json')
    pc=read(DEST/'COMPARACION_KERNELS_PC.json')
    fc=read(DEST/'REPLAY_FC_ENTERAS.json')
    soft=read(DEST/'COMPARACION_SOFTMAX_AISLADO.json')
    audit=read(DEST/'AUDITORIA_RECUANTIZACION.json')
    assert case['input_exact_to_original_cache'] and case['same_predicted_class']
    assert case['input_INT8']==pc['input_INT8']==fc['input_INT8']==[-45,-15,-38,-36,-69,-39,-42]
    assert pc['model_sha256']==fc['model_sha256']==soft['original_model_sha256']=='b55aaa42f97a41ff1a8450780f86ad35f9d70ac685b134ca719525f06fb54784'
    checked=0
    for mode,variant in pc['variants'].items():
        assert mode in ('BUILTIN_WITHOUT_DEFAULT_DELEGATES','BUILTIN_REF')
        for expected,op in zip(fc['rounding_modes']['single']['layers'],variant['ops'][:3]):
            assert expected['output']==op['outputs'][0]['values'][0]
            checked+=len(expected['output'])
        assert variant['output_INT8']==case['PC_cache_output_INT8']==[41,-100,-69]
    assert checked==66 and len(pc['variants'])==2 and pc['layer_differences']==[]
    for mode,key in [('single','SINGLE'),('double','DOUBLE')]:
        assert fc['rounding_modes'][mode]['layers'][-1]['output']==soft['results'][key]['logits']
    assert soft['single_matches_PC'] and soft['double_matches_ARM']
    assert soft['results']['DOUBLE']['output_INT8']==case['ARM_output_INT8']==[42,-101,-69]
    assert soft['replay_FC_file_sha256']==sha(DEST/'REPLAY_FC_ENTERAS.json')
    source_paths=[DEST/n for n in ('CASO_FALLO.json','COMPARACION_KERNELS_PC.json','REPLAY_FC_ENTERAS.json',
                                  'COMPARACION_SOFTMAX_AISLADO.json','AUDITORIA_RECUANTIZACION.json')]
    report=dict(status='CAUSE_ROUNDING_MISMATCH_CONFIRMED_PENDING_ARM_FIX',
        date_local='2026-10-26',timezone='America/Lima',
        original_run='20261026T042424_323718Z',original_run_status='FAILED_INCOMPLETE',
        firmware_elf_sha256=case['firmware_elf_sha256'],model_sha256=pc['model_sha256'],
        representative_id=pc['representative_id'],global_window_id=232,
        input_INT8=case['input_INT8'],PC_output_INT8=case['PC_cache_output_INT8'],ARM_output_INT8=case['ARM_output_INT8'],
        output_delta_ARM_minus_PC=[1,-1,0],same_class_for_failed_case=True,
        single_PC_layers_exact=True,PC_backends_checked=2,neurons_per_backend=33,
        double_rounding_in_linked_CMSIS_FC_verified_by_assembly=True,
        double_softmax_matches_ARM=True,single_softmax_matches_PC=True,
        single_logits=[33,-29,-3],double_logits=[33,-30,-3],
        first_divergence=dict(layer=0,neuron=0,accumulator=6764,multiplier=2052107017,shift=-7,
            exact_product=13880451862988,exact_scaled=13880451862988/(1<<38),
            single_scaled=50,double_first_rounded=6464,double_second_scaled=51,
            output_zero_point=-128,single_output=-78,double_output=-77),
        exclusions_confirmed=['RAW/transport mismatch','selected feature mismatch','input scaling mismatch','input INT8 mismatch'],
        full_campaign_completed=False,preceding_saved_primary_RUN=66,preceding_saved_PROFILE=22,
        old_partial_costs_promoted=False,acceptance_tolerance_relaxed=False,
        proposed_change='Isolated SR configuration: original CMSIS per-channel matvec source compiled for single rounding, original archive preserved.',
        app_macro_alone_changes_precompiled_library=False,
        model_weights_or_training_changed=False,new_training_fits=0,
        diagnostic_TFLite_invocations=pc['TFLite_invocations']+soft['TFLite_invocations'],
        correction_physically_verified=False,serial_port_opened_by_diagnosis=False,
        selector_costs_enabled=False,sources_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in source_paths},
        external_primary_sources=[
            dict(url='https://github.com/ARM-software/CMSIS-NN/blob/main/README.md',
                 support='CMSIS_NN_USE_SINGLE_ROUNDING changes requantization; header-affecting options require corresponding runtime configuration.'),
            dict(url='https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/kernels/internal/common.cc',
                 support='Single and double integer multiplier equations; installed local sources and binary were audited separately.')],
        limits=['No intermediate activation captured on the board; assembly, integer replay and final output reproduce the mechanism.',
                'Compatibility of the proposed SR binary must pass a new physical regression gate.',
                'Same class is established for this case only; no global accuracy claim follows.',
                'Full timing samples must come from one verified firmware; prior partial run is retained as failed evidence.'])
    write(target,report)
    paths={p for p in DEST.iterdir() if p.is_file() and not p.name.startswith('MANIFEST_DIAGNOSTICO_ROUNDING_')}
    for name in ('extraer_divergencia_total_v2.py','diagnosticar_backends_int8_total_v2.py','reproducir_fc_entera_total_v2.py',
                 'diagnosticar_softmax_aislado_total_v2.py','cerrar_diagnostico_rounding_total_v2.py'):
        paths.add(Path(__file__).with_name(name))
    entries={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(paths)}
    manifest=DEST/'MANIFEST_DIAGNOSTICO_ROUNDING_SHA256.json'
    write(manifest,entries)
    with manifest.with_suffix('.sha256').open('x',encoding='ascii') as f:f.write(sha(manifest)+'  '+manifest.name+'\n')
    assert all(sha(ROOT/p)==v for p,v in entries.items())
    print(json.dumps(dict(status=report['status'],resources=len(entries),manifest_sha256=sha(manifest)),ensure_ascii=True))

if __name__=='__main__':main()
