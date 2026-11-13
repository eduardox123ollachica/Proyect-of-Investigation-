"""Consolida los 32 representantes desde artefactos; no entrena ni infiere DNN."""
from __future__ import annotations

from collections import Counter
import csv
from pathlib import Path
from statistics import median
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT/'codigo/cwru/ampliacion_v2/representantes'))
from ejecutar_representantes import inspect_campaign, read, write, sha, verify_manifest, PREP
from datos_representantes import load_verified_context
from numerica_representantes import normalize_f32, quantize_int8, integer_argmax

CAMPAIGN = ROOT/'resultados/cwru/ampliacion_v2/representantes_v2_2026-10-25/20261025T171847_957295Z'
OUTPUT = ROOT/'articulo/ampliacion_cwru/metodologia/cierre_representantes_v2_2026-10-25'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def main():
    plan, jobs, done, consumed = inspect_campaign(CAMPAIGN)
    require(len(done) == 32 and done == [j['job_id'] for j in jobs], 'Campana incompleta')
    require(not (CAMPAIGN/'EJECUCION_EN_CURSO.lock').exists(), 'Existe ejecucion activa')
    _, _, metadata, expected_hashes = load_verified_context()
    index = {int(row['global_window_id']):row for row in metadata}
    batches = sorted((CAMPAIGN/'lotes').iterdir())
    statuses = []
    summaries = []
    for batch in batches:
        summary = read(batch/'RESUMEN_LOTE.json')
        batch_plan = read(batch/'PLAN_LOTE.json')
        current = [read(batch/('ESTADO_'+name+'.json')) for name in batch_plan['job_ids']]
        require(all(s['status']=='PASS' and s['return_code']==0 and s['failure'] is None
                    and s['result_error'] is None for s in current), 'Proceso fallido')
        require(len(current)==summary['jobs_started']==summary['jobs_passed']
                and sum(s['worker_elapsed_seconds'] for s in current)==summary['worker_elapsed_seconds'],
                'Resumen temporal o conteos incompatibles')
        require(batch_plan['completed_before']==[s['job_id'] for s in statuses], 'Reanudacion incoherente')
        statuses.extend(current)
        summaries.append(summary)
    require([s['job_id'] for s in statuses]==done and sum(s['worker_elapsed_seconds'] for s in statuses)==consumed,
            'Estados faltantes/duplicados o tiempo incompatible')
    require(summaries[-1]['status']=='CAMPAIGN_COMPLETED' and summaries[-1]['total_completed']==32,
            'Falta cierre completo')
    input_paths = {item['path'] for j in jobs for item in read(CAMPAIGN/j['job_id']/'procedencia.json')['source_arrays']}
    for relative in input_paths:
        require(sha(ROOT/relative)==expected_hashes[relative], 'Cache de entrada modificado')
    table = []
    detector_parameters = {}
    versions = []
    for job, status in zip(jobs,statuses):
        directory = CAMPAIGN/job['job_id']
        result = read(directory/'resultado.json')
        state = read(directory/'estado_antes_conversion.json')
        provenance = read(directory/'procedencia.json')
        versions.append(result['versions'])
        require(result['versions']==plan['tested_versions'], 'Versiones no firmadas')
        require(provenance['train_global_window_ids']==job['training_global_window_ids']
                and provenance['validation_global_window_ids']==job['validation_global_window_ids'],
                'Procedencia y plan distintos')
        train_records = {index[k]['file_id'] for k in job['training_global_window_ids']}
        val_records = {index[k]['file_id'] for k in job['validation_global_window_ids']}
        require(len(train_records)==14 and len(val_records)==7 and train_records.isdisjoint(val_records),
                'Separacion MAT incorrecta')
        counts = np.bincount([int(index[k]['label']) for k in job['training_global_window_ids']],minlength=3)
        require(state['class_counts']==counts.tolist()
                and state['class_weights']==(len(job['training_global_window_ids'])/(3.*counts)).tolist(),
                'Pesos de clase calculados fuera del ajuste')
        require(state['optimizer_updates']==result['optimizer_updates']==result['epochs_executed']*
                ((len(job['training_global_window_ids'])+31)//32), 'Numero de updates inconsistente')
        history = state['history']
        require(len(history)==state['epochs_executed']
                and (len(history)==50 or history[-1]['patience_used']==10), 'Parada fuera de las reglas')
        epochs_on_disk = [read_line for read_line in (directory/'historial.jsonl').read_text(encoding='utf-8').splitlines()]
        import json
        require([json.loads(line) for line in epochs_on_disk]==history, 'Historial incremental distinto')
        for name,family in (('modelo_fp32.tflite','FP32'),('modelo_int8.tflite','INT8')):
            require((directory/name).stat().st_size==result['model_bytes'][family], 'Tamano de modelo incorrecto')
        for name in ('int8_details','benchmark_int8_details'):
            details = result[name]
            require(details['input_dtype']==details['output_dtype']=='int8'
                    and set(details['tensor_dtypes'])<={'int8','int32','int64'}
                    and details['operators']==['FULLY_CONNECTED']*3+['SOFTMAX']
                    and details['input_quantization'][0]>0 and details['output_quantization'][0]>0,
                    'INT8 declarado no cumple contrato')
        require(result['freeze']['comparison_batch_size']==1 and result['freeze']['weights_unchanged']
                and result['freeze']['class_changes']==0 and result['freeze']['resource_variable_ops_remaining']==0,
                'Freeze incompleto')
        require(result['outer_evaluation_executed'] is False and result['remaining_grid_enabled'] is False
                and result['measured_STM32_total_cost'] is None, 'Puerta externa/costo incorrecta')
        with np.load(directory/'predicciones_validacion.npz',allow_pickle=False) as saved:
            require(np.array_equal(integer_argmax(saved['fp32_probabilities']),saved['fp32_predictions'])
                    and np.array_equal(integer_argmax(saved['int8_outputs']),saved['int8_predictions']),
                    'Predicciones no corresponden a salidas guardadas')
            changes = int(np.count_nonzero(saved['fp32_predictions']!=saved['int8_predictions']))
            require(changes==result['FP32_INT8_class_changes'], 'Cambios de clase mal contados')
        with np.load(directory/'cuantizacion_train.npz',allow_pickle=False) as saved:
            require(saved['global_window_ids'].tolist()==job['training_global_window_ids']
                    and saved['int8_inputs'].shape==(len(job['training_global_window_ids']),job['input_dimension'])
                    and saved['int8_inputs'].dtype==np.int8, 'Cuantizacion de train incompleta')
        with np.load(directory/'scaler_train_only.npz',allow_pickle=False) as scaler, \
             np.load(directory/'banco_costos_referencia.npz',allow_pickle=False) as bank:
            normalized = normalize_f32(bank['raw_features14'][:,job['feature_indices']],scaler['mean32'],scaler['scale32'])
            quantized, clipped = quantize_int8(normalized,*result['int8_details']['input_quantization'])
            require(np.array_equal(normalized,bank['normalized_features']) and np.array_equal(quantized,bank['int8_inputs'])
                    and clipped==result['benchmark_int8_details']['input_clipped_values'], 'Banco numerico incoherente')
            bank_ids = bank['global_window_ids'].tolist()
            require(bank_ids==job['benchmark_global_window_ids'] and len(bank_ids)==28
                    and len({index[k]['file_id'] for k in bank_ids})==14, 'Banco no autorizado')
        with np.load(directory/'detector_parametros.npz',allow_pickle=False) as params:
            detector_parameters[job['outer_load_hp'],job['candidate']] = {
                key:params[key].copy() for key in ('indices','mean64','scale64','var64','mean32','scale32','centroid32','threshold32')}
        ep = result['epochs_executed']
        table.append(dict(job_id=job['job_id'],outer_load_hp=job['outer_load_hp'],inner_validation_load_hp=job['inner_validation_load_hp'],
            candidate=job['candidate'],seed=11,training_windows=len(job['training_global_window_ids']),
            validation_windows=len(job['validation_global_window_ids']),epochs_executed=ep,best_epoch=result['best_epoch'],
            stopped_by_patience=history[-1]['patience_used']>=10,hit_epoch_cap=ep==50,
            improvements_last_10=sum(row['improved'] for row in history[-10:]),
            macro_f1_FP32=result['metrics']['FP32']['macro_f1'],macro_f1_INT8=result['metrics']['INT8']['macro_f1'],
            accuracy_INT8=result['metrics']['INT8']['accuracy'],FP32_INT8_changes=changes,
            normal_spec=result['detector_metrics']['normal_spec'],ir_recall=result['detector_metrics']['ir_recall'],
            or_recall=result['detector_metrics']['or_recall'],detector_BA=result['detector_metrics']['balanced_accuracy'],
            train_clipped_values=result['training_saturated_input_values'],
            validation_clipped_values=result['int8_details']['input_clipped_values'],
            benchmark_clipped_values=result['benchmark_int8_details']['input_clipped_values'],
            model_INT8_bytes=result['model_bytes']['INT8'],process_and_validation_seconds=status['worker_elapsed_seconds']))
    aliases = []
    for outer in range(4):
        a,b = detector_parameters[outer,'F14'],detector_parameters[outer,'NF12']
        require(all(np.array_equal(a[key],b[key]) for key in a), 'Alias F14/NF12 no coincide')
        aliases.append(dict(outer_load_hp=outer,alias='NF12_detector_equals_F14',all_parameters_exact=True))
    val0 = [row for row in table if row['inner_validation_load_hp']==0]
    epochs = dict(executed_min=min(r['epochs_executed'] for r in table),executed_max=max(r['epochs_executed'] for r in table),
        executed_median=median(r['epochs_executed'] for r in table),best_median=median(r['best_epoch'] for r in table),
        best_epoch_histogram=dict(sorted(Counter(r['best_epoch'] for r in table).items())),
        stopped_by_patience=sum(r['stopped_by_patience'] for r in table),hit_cap50=sum(r['hit_epoch_cap'] for r in table),
        val0_best_epochs=[r['best_epoch'] for r in val0],
        cap_only_increase_with_same_ES_does_not_extend_already_stopped_trajectories=True)
    report = dict(status='TECHNICAL_AUDIT_PASS_PREDICTIVE_WARNINGS',campaign=CAMPAIGN.relative_to(ROOT).as_posix(),
        jobs_passed=32,batches=[dict(path=b.relative_to(ROOT).as_posix(),summary=s) for b,s in zip(batches,summaries)],
        process_and_verification_seconds=consumed,process_and_verification_minutes=consumed/60,epochs=epochs,
        diagnostics=dict(FP32_INT8_changes_sum=sum(r['FP32_INT8_changes'] for r in table),
            models_with_class_changes=sum(r['FP32_INT8_changes']>0 for r in table),
            validation_clipped_values_sum=sum(r['validation_clipped_values'] for r in table),
            models_with_validation_clipping=sum(r['validation_clipped_values']>0 for r in table),
            train_clipped_values_sum=sum(r['train_clipped_values'] for r in table),
            benchmark_clipped_values_sum=sum(r['benchmark_clipped_values'] for r in table),
            val0_models=len(val0),val0_models_normal_specificity_zero=sum(r['normal_spec']==0 for r in val0),
            all_IR_OR_recalls_one=all(r['ir_recall']==r['or_recall']==1 for r in table)),
        detector_alias_checks=aliases,representative_rows=table,
        new_training_fits=0,new_DNN_inferences=0,sklearn_or_tensorflow_imported=False,
        source_feature_arrays_materialized_in_audit=False,cached_source_file_hashes_checked=True,
        STM32_total_cost_measured=False,remaining_grid_enabled=False,outer_evaluation_executed=False,
        performance_or_cost_mask_selected=False,full_protocol_frozen=False,
        limitations=['One seed and one predefined inner fold per outer; not the complete three-by-three inner grid.',
            'Outer1/2/3 share validation load0: descriptive counts are not independent acquisitions.',
            'INT8 operators audited from recorded metadata, not a new FlatBuffer parser or runtime invocation.',
            'Clipping and class-change totals sum repeated model evaluations; not unique physical windows.',
            'Pure saved-array arithmetic and hashes cannot establish an independent OS access trace.'],
        next_priority='diagnose_validation_load0_FP32_INT8_and_normality_before_interpreting_epoch_extension',
        epochs_50_vs100_supplement='proposed_not_executed_no_main_protocol_change')
    OUTPUT.mkdir(exist_ok=False)
    write(OUTPUT/'CIERRE_REPRESENTANTES.json',report)
    with (OUTPUT/'RESULTADOS_REPRESENTANTES.csv').open('x',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    lines=['# Cierre de los 32 representantes V2','',
        'Auditoría técnica PASS: 32/32, dos lotes (1+31), sin fallos, timeouts o reintentos.',
        f'Tiempo acumulado de procesos y comprobación del padre: {consumed:.6f} s ({consumed/60:.4f} min).',
        'Se comprobaron sellos, firmas, roles e IDs, pesos de clase, ES, checkpoints, modelos,',
        'métricas, P95 y bancos. Sin reentrenamiento ni inferencia DNN adicional.','',
        '## Resultados internos descriptivos','',
        'Una semilla (11), un inner prefijado por outer. Outer1/2/3 validan la misma carga0.',
        'La tabla no permite seleccionar máscaras ni representa evaluación externa final.','',
        '| Outer / validación interna | F14 F1 INT8 | T6 F1 INT8 | NF12 F1 INT8 |',
        '|---|---:|---:|---:|']
    for outer in range(4):
        group={r['candidate']:r for r in table if r['outer_load_hp']==outer}
        lines.append(f"| {outer} / {group['F14']['inner_validation_load_hp']} HP | "+
            ' | '.join(f"{group[c]['macro_f1_INT8']:.6f}" for c in ('F14','T6','NF12'))+' |')
    lines += ['', 'La pérdida de rendimiento ya aparece en FP32: no puede atribuirse únicamente a INT8.',
        'Por ejemplo, F14 outer2 tiene F1 FP32 0.598474 y F1 INT8 0.261758. No se',
        'ha demostrado todavía una causa única ni corresponde modificar parámetros para mejorar esta tabla.','',
        '## Épocas y cuantización','',
        f"- {epochs['stopped_by_patience']}/32 pararon por paciencia antes de50; {epochs['hit_cap50']}/32 alcanzaron50.",
        '- Los24 que validan0 HP restauraron época1 o2; los6 limitados por50 validan1 HP.',
        '- Subir solo el máximo a100 conservando ES no prolongaría las26 trayectorias ya detenidas.',
        f"- Cambios FP32–INT8: {report['diagnostics']['FP32_INT8_changes_sum']} en {report['diagnostics']['models_with_class_changes']} modelos.",
        f"- Valores de entrada de validación recortados: {report['diagnostics']['validation_clipped_values_sum']} en {report['diagnostics']['models_with_validation_clipping']} modelos; ajuste/banco:0.",
        '- Los totales anteriores suman evaluaciones repetidas; no son ventanas físicas únicas.','',
        '## Detector de normalidad','',
        'IR/OR recall1.0 en los32; al validar0 HP la especificidad N es0 en18/24,',
        'y en los otros6 es0.071429–0.214286. En validación1 HP DE7 tiene0.448276;',
        'las otras siete máscaras1.0. La sensibilidad a fallas no compensa el rechazo',
        'de normales. Parámetros de los alias F14/NF12 idénticos en los cuatro scopes.','',
        '## Continuidad y alcance','',
        'Se recomienda diagnosticar el comportamiento de0 HP, la reducción FP32–INT8 y',
        'la calibración de normalidad mediante evidencia de desarrollo antes de interpretar',
        'el efecto de aumentar épocas. El ensayo50/100 está propuesto, sin ejecutarse.',
        'No se alteran decisiones, modelos o umbrales automáticamente a partir de estos resultados.',
        'El costo total STM32 sigue pendiente; no se habilitaron búsqueda restante, COST o',
        'evaluación externa. Las32 son subconjunto de288internos y se conservan sus artefactos.',
        'El manuscrito, firmware y recursos sellados anteriores permanecen intactos.','']
    (OUTPUT/'CIERRE_REPRESENTANTES.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')
    resources=dict(verify_manifest(PREP/'MANIFEST_PREPARACION_SHA256.json'))
    for batch in batches:
        resources.update(verify_manifest(batch/'MANIFEST_LOTE_SHA256.json'))
        for name in ('MANIFEST_LOTE_SHA256.json','MANIFEST_LOTE_SHA256.sha256'):
            path=batch/name;resources[path.relative_to(ROOT).as_posix()]=sha(path)
    prior=ROOT/'articulo/ampliacion_cwru/metodologia/cierre_primer_representante_2026-10-25/MANIFEST_CIERRE_SHA256.json'
    resources.update(verify_manifest(prior))
    for path in [Path(__file__),prior,prior.with_suffix('.sha256'),PREP/'MANIFEST_PREPARACION_SHA256.json',
                 PREP/'MANIFEST_PREPARACION_SHA256.sha256']+list(OUTPUT.iterdir()):
        resources[path.relative_to(ROOT).as_posix()]=sha(path)
    manifest=OUTPUT/'MANIFEST_CIERRE_SHA256.json';write(manifest,resources)
    manifest.with_suffix('.sha256').write_text(sha(manifest)+'  '+manifest.name+'\n',encoding='ascii')
    verify_manifest(manifest)
    print(f"Cierre32 PASS: {len(resources)} recursos; {consumed:.6f}s; ES{epochs['stopped_by_patience']}; cap50={epochs['hit_cap50']}.")


if __name__=='__main__':main()
