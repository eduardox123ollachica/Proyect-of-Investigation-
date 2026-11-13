"""Cierra la preparación DNN interna sin entrenar ni invocar TFLite.

--environment comprueba el Python real del usuario mediante metadatos.
--close contrasta los informes, fuentes, importaciones y presupuesto y sella
la entrega. No inicia campañas, modifica evidencias previas ni usa hardware.
"""
from __future__ import annotations

import argparse
import datetime
import importlib.abc
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT/'codigo/cwru/ampliacion_v2/busqueda_interna'
PREP = ROOT/'articulo/ampliacion_cwru/metodologia/busqueda_interna_v2_2026-10-26'
sys.path.insert(0, str(SOURCE))


class NoTrainingRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('tensorflow', 'keras', 'ai_edge_litert'):
            raise RuntimeError('Preparación de solo lectura: runtime ML bloqueado')
        return None


sys.meta_path.insert(0, NoTrainingRuntime())
import ejecutar_busqueda_interna as runner


def environment():
    started = time.perf_counter()
    plan = runner.preflight(check_environment=True)
    versions = runner.legacy.versions()
    runner.require('tensorflow' not in sys.modules, 'No importar TensorFlow')
    report = dict(status='READ_ONLY_USER_ENVIRONMENT_PREFLIGHT_PASS',
        python_executable=sys.executable, versions=versions,
        expected_versions=plan['tested_versions'], source_sha256=runner.source_hashes(),
        preparation_manifest_sha256=runner.sha(runner.PREP_MANIFEST),
        plan_sha256=runner.sha(runner.PLAN_FILE), logical_cells=288,
        imported_cells=48, new_fits=240, TensorFlow_imported=False,
        training_started=False, real_training_fits=0, real_TFLite_invocations=0,
        hardware_connected=False, seconds=time.perf_counter()-started)
    runner.write(PREP/'PREFLIGHT_ENTORNO_USUARIO.json', report)
    print(json.dumps(report, ensure_ascii=True))


def close():
    started = time.perf_counter()
    plan = runner.preflight()
    sources = runner.source_hashes()
    tests = runner.read(PREP/'PRUEBAS_PREPARACION.json')
    env = runner.read(PREP/'PREFLIGHT_ENTORNO_USUARIO.json')
    audit = runner.read(PREP/'AUDITORIA_REUTILIZACION.json')
    budget = runner.read(PREP/'PRESUPUESTO_Y_LIMITES.json')
    runner.require(tests['status'] == 'PASS' and tests['errors'] == tests['failures'] == 0
        and tests['tests_run'] >= 18 and tests['source_sha256'] == sources
        and tests['real_training_fits'] == tests['real_TFLite_invocations'] == 0
        and tests['TensorFlow_imported'] is False, 'Pruebas no corresponden a fuentes finales')
    runner.require(env['status'] == 'READ_ONLY_USER_ENVIRONMENT_PREFLIGHT_PASS'
        and env['versions'] == plan['tested_versions'] and env['source_sha256'] == sources
        and env['preparation_manifest_sha256'] == runner.sha(runner.PREP_MANIFEST)
        and env['plan_sha256'] == runner.sha(runner.PLAN_FILE)
        and env['TensorFlow_imported'] is False and env['training_started'] is False,
        'Entorno real pendiente o cambiado')
    runner.require(audit['status'] == 'PASS_48_INNER_DNN_FITS_REUSABLE_WITH_ORIGINAL_SIGNATURES'
        and audit['reusable_DNN_fits'] == 48 and audit['new_DNN_fits_required'] == 240,
        'Auditoría de reutilización incompatible')
    audit_script = ROOT/'codigo/cwru/ampliacion_v2/analisis/auditar_reutilizacion_busqueda_interna_v2.py'
    runner.require(all(runner.sha(ROOT/path) == digest
        for path,digest in audit['source_sha256'].items()), 'Fuentes de auditoría de origen cambiadas')
    by_directory = {v['source_directory']:v for v in plan['imported'].values()}
    runner.require(len(by_directory) == len(audit['links']) == 48, 'Referencias duplicadas')
    for link in audit['links']:
        imported = by_directory[link['source_directory']]
        runner.require(all(imported[k] == link[k] for k in (
            'source_job_id','source_result_sha256','source_fit_signature','model_sha256'))
            and imported['source_manifest_sha256'] == link['source_job_manifest_sha256']
            and link['source_signature_verified'] and not link['original_signature_rewritten'],
            'Discrepancia entre auditoría independiente e importación')
    runner.require(budget['n_new_fits'] == plan['new_fits'] == 240
        and budget['reused_DNN_fits'] == plan['imported_cells'] == 48
        and budget['limits'] == plan['limits'], 'Presupuesto distinto del inventario')
    from presupuesto_busqueda_interna import build_budget
    runner.require(build_budget(240) == budget, 'Presupuesto no reproduce evidencias temporales')
    runner.require(not runner.OUTPUT.exists() or not any(runner.OUTPUT.iterdir()),
        'La entrega preparada no debe contener campañas reales')
    runner.require('tensorflow' not in sys.modules, 'Runtime ML importado')
    manuscript = ROOT/'articulo/manuscrito/articulo_ieee_borrador.tex'
    report = dict(status='INNER_DNN_EXECUTOR_PREPARED_VERIFIED_PENDING_USER_FIRST_FIT',
        date_local='2026-10-26', timezone='America/Lima',
        closed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        logical_cells=288, reused_fits=48, new_fits_pending=240,
        original_representatives=32, additional_supplement_P0=16,
        inner_seeds=[11,22,33], maximum_epochs=50, early_stopping=plan['dnn']['early_stopping'],
        limits=plan['limits'], user_starts_training=True,
        static_review=dict(status='NO_BLOCKING_CODE_FINDINGS',
            scope='canonical_roles_imports_campaign_binding_resume_budget_checkpoint_scaler_quantization',
            bound_source_sha256=sources),
        synthetic_and_existing_artifact_tests_run=tests['tests_run'],
        actual_user_environment_verified=True,
        detector_inventory=audit['detector_inventory'],
        accumulated_budget_accounting='Controller run_one elapsed: subprocess launch/execution plus postprocess artifact verification. Conservative relative to pure subprocess time; not total calendar time.',
        forecast_minutes=budget['forecast'],
        current_manuscript_sha256=runner.sha(manuscript),
        manuscript_edited_by_this_preparation=False, dataset_modified=False,
        new_training_fits=0, new_TFLite_invocations=0, new_detector_fits=0,
        hardware_connected=False, serial_port_opened=False, firmware_built_or_flashed=False,
        predictive_selection_executed=False, outer_evaluation_executed=False,
        original_artifacts_overwritten=False,
        first_command=dict(script=(SOURCE/'ejecutar_busqueda_interna.py').relative_to(ROOT).as_posix(),
            python=r'C:\Users\LENOVO\AppData\Local\Microsoft\WindowsApps\python3.12.exe',
            arguments=['-B','--run','--max-new-jobs','1','--max-seconds','600'],
            expected_available_cells_if_pass=49),
        next_stage='User runs one new inner DNN fit; audit artifacts before expanding to <=16 per batch.',
        qualifications=['DNN matrix preparation is not completed predictive selection.',
            'Detector remaining rows, classical grids and outer evaluation remain separate stages.',
            'Seeds are repeated training, not independent motors or bearings.',
            'Source model signatures are retained; semantic wrapper signatures are separate.'],
        seconds=time.perf_counter()-started)
    closure=PREP/'CIERRE_PREPARACION_BUSQUEDA_INTERNA.json'
    runner.write(closure, report)
    files=[p for p in PREP.iterdir() if p.is_file()]
    files += [ROOT/p for p in set(sources)|set(audit['source_sha256'])]
    files += [Path(__file__).resolve(),audit_script,manuscript]
    # Preserve all source links without copying or rewriting their artifacts.
    for imported in plan['imported'].values():
        folder=ROOT/imported['source_directory']
        files += [folder/'resultado.json',folder/'MANIFEST_JOB_SHA256.json',
            folder/'MANIFEST_JOB_SHA256.sha256',folder/'modelo_fp32.tflite',folder/'modelo_int8.tflite']
    manifest=PREP/'MANIFEST_PREPARACION_COMPLETA_SHA256.json'
    runner.write_manifest(manifest, extra=files)
    print(json.dumps(dict(status=report['status'],logical_cells=288,reused_fits=48,
        new_fits_pending=240,tests_run=tests['tests_run'],
        manifest_files=len(runner.read(manifest)),manifest_sha256=runner.sha(manifest),
        real_training_fits=0,TensorFlow_imported=False),ensure_ascii=True))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--environment',action='store_true')
    mode.add_argument('--close',action='store_true')
    args=parser.parse_args()
    environment() if args.environment else close()
