"""Proyeccion de duracion del lote SR desde evidencias; no ejecuta el lote.

Solo lectura de capturas y recursos. Crea un informe nuevo, sin TF, COM,
entrenamiento, compilacion o cambios en los recursos sellados.
"""
from __future__ import annotations

from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import time

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT/'despliegue_stm32/costo_total_single_rounding_v2_180mhz_2026-10-26'
GATE = OUT/'Validacion_PC/resultados/20261026T143621_946818Z'
CLIENT = OUT/'Validacion_PC/validar_total_sr.py'
REPORT = OUT/'ESTIMACION_DURACION_CAMPANA_SR.json'
ELF_SHA = 'c0c3853d2b0e90bb20fe00be13bcf107d186088406be131b729a605ded44ecc7'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    started = time.perf_counter()
    require(not REPORT.exists(), 'No sobrescribir estimacion')
    spec = importlib.util.spec_from_file_location('sr_duration_client_readonly', CLIENT)
    client = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(client)
    catalog, bank, windows, _ = client.load_resources(OUT/'recursos')
    client.verify_gate(GATE, catalog, ELF_SHA, sha(OUT/'recursos/MANIFEST_RECURSOS_SHA256.json'))
    plan = read(GATE/'PLAN_EJECUTADO.json')
    gate = read(GATE/'RESULTADO_TOTAL.json')
    require(plan.get('baud', 115200) == 115200, 'Revisar baud distinto')
    trace = [json.loads(s) for s in (GATE/'TRAZA_UART.jsonl').read_text(encoding='utf-8').splitlines()]
    require(len(trace) % 2 == 0, 'Traza de longitud impar')
    exchanges = []
    for a, b in zip(trace[::2], trace[1::2]):
        require(a['direction'] == 'TX' and b['direction'] == 'RX' and b['seq'] == a['seq']+1,
                'Intercambio TX/RX invalido')
        dt = (datetime.fromisoformat(b['utc'])-datetime.fromisoformat(a['utc'])).total_seconds()
        require(dt >= 0, 'Timestamp decreciente')
        exchanges.append(dict(command=a['command'], response=b['line'], seconds=dt))
    span = (datetime.fromisoformat(trace[-1]['utc'])-datetime.fromisoformat(trace[0]['utc'])).total_seconds()
    startup_proxy = gate['wall_seconds']-span
    require(startup_proxy >= 0, 'Span UART mayor que tiempo wall')
    raw_sets = []
    current = []
    for row in exchanges:
        if row['command'] == 'CWRUFEAT_BEGIN':
            current = []
        elif row['command'].startswith('CWRUFEAT_DATA,'):
            current.append(row)
        elif row['command'] == 'CWRUFEAT_CHECK' and len(current) == 171:
            raw_sets.append(current)
            current = []
    require(len(raw_sets) == 2, 'Se esperaban dos transferencias RAW completas')
    raw = [row for block in raw_sets for row in block]
    wire_bytes = lambda row: len(row['command'].encode('ascii'))+2+len(row['response'].encode('ascii'))+2
    raw_wire = sum(wire_bytes(row) for row in raw)*10/115200
    raw_exchange = sum(row['seconds'] for row in raw)
    extra_per_frame = (raw_exchange-raw_wire)/len(raw)
    require(extra_per_frame >= 0, 'Tiempo observado menor al trafico UART nominal')
    full_plan = client.schedule(bank, True)
    by_id = {row['global_id']: row for row in bank['windows']}
    full_wire_bytes = 0
    full_frames = 0
    for gid, _ in full_plan:
        for command, count in client.frames(windows[by_id[gid]['npz_index']]):
            full_wire_bytes += len(command.encode('ascii'))+2+len(f'CWRUFEAT_ACK,{count}')+2
            full_frames += 1
    require(full_frames == 7182 and len(full_plan) == 42 and sum(len(ids) for _,ids in full_plan) == 896,
            'Plan completo distinto del previsto')

    def valid(command, response):
        return [row for row in exchanges if row['command'].split(',')[0] == command
                and row['response'].startswith(response+',')]

    def mean(rows):
        require(rows, 'Grupo temporal vacio')
        return statistics.mean(row['seconds'] for row in rows)

    runs = valid('CWRUTOTAL_RUN', 'CWRUTOTAL_RESULT')
    profiles = valid('CWRUTOTAL_PROFILE', 'CWRUTOTAL_RESULT')
    require(len(runs) == 6 and len(profiles) == 2, 'Conteos de resultados distintos')
    runs0 = [row for row in runs if row['response'].split(',')[2] == '0']
    runs22 = [row for row in runs if row['response'].split(',')[2] == '22']
    profile0 = [row for row in profiles if row['response'].split(',')[2] == '0']
    pages0 = {kind: [row for row in valid('CWRUTOTAL_'+kind, 'CWRUTOTAL_'+kind)
                    if row['response'].split(',')[2] == '0'] for kind in ('FEATURES','INPUT','DETECTOR')}
    require(len(runs0) == len(runs22) == 3 and len(profile0) == 1
            and all(len(rows) == 4 for rows in pages0.values()), 'Grupos por representante incompletos')
    three_pages0 = sum(mean(rows) for rows in pages0.values())
    raw_seconds = full_wire_bytes*10/115200+full_frames*extra_per_frame
    compute_and_pages = 896*(3*mean(runs0)+mean(profile0)+4*three_pages0)
    select = valid('CWRUTOTAL_SELECT', 'CWRUTOTAL_SELECTED')[:2]
    model = valid('CWRUTOTAL_MODEL', 'CWRUTOTAL_MODEL')
    base = valid('CWRUMASK_BASE', 'CWRUMASK_BASE')
    begin = valid('CWRUFEAT_BEGIN', 'CWRUFEAT_READY')[:2]
    check = [row for row in valid('CWRUFEAT_CHECK', 'CWRUFEAT_CHECK')
             if row['response'].split(',')[1] == '8192'][:2]
    ancillary = 896*mean(select)+32*mean(model)+42*(mean(base)+mean(begin)+mean(check))
    allowance = 10
    total = startup_proxy+raw_seconds+compute_and_pages+ancillary+allowance
    result = dict(
        status='RUNTIME_PLANNING_ESTIMATE_NOT_MEASURED_FULL_CAMPAIGN',
        date_local='2026-10-26', timezone='America/Lima',
        purpose='Personal execution budget only; not a measured cost table or selection criterion',
        source_gate=GATE.relative_to(ROOT).as_posix(),
        sources_sha256={name: sha(GATE/name) for name in
            ('TRAZA_UART.jsonl','RESULTADO_TOTAL.json','PLAN_EJECUTADO.json')},
        client_sha256=sha(CLIENT), script_sha256=sha(__file__),
        baud_assumption=115200, wire_bits_per_ASCII_byte=10,
        observed_gate=dict(wall_seconds=gate['wall_seconds'],serial_trace_seconds=span,
            startup_plus_post_trace_closure_seconds=startup_proxy, complete_RAW_transfers=2,
            RAW_frames=len(raw), RAW_wire_seconds=raw_wire,RAW_exchange_seconds=raw_exchange,
            extra_seconds_per_frame=extra_per_frame,
            F14_primary_RUN_mean_exchange_seconds=mean(runs0),
            DE7_primary_RUN_mean_exchange_seconds=mean(runs22),
            F14_PROFILE_exchange_seconds=mean(profile0),
            F14_three_page_mean_exchange_seconds=three_pages0,
            SELECT_mean_initial_model_exchange_seconds=mean(select),
            startup_note='Wall minus UART span includes startup and closure; not isolated TensorFlow profiling'),
        full_plan=dict(unique_RAW_windows=42,RAW_frames=full_frames,
            RAW_wire_bytes_including_ACK=full_wire_bytes,model_window_pairs=896,
            main_RUN=2688,diagnostic_PROFILE=896,pages=10752,MODEL_queries=32,
            main_SELECT=896,BASE=42,negative_rejections=17,planned_PC_TFLite_invocations=7168),
        forecast_seconds=dict(startup_proxy=startup_proxy,
            RAW_transfers_using_actual_42_ASCII_payload_lengths=raw_seconds,
            all_RUN_PROFILE_pages_if_all_cases_had_gate_F14_exchange_times=compute_and_pages,
            SELECT_MODEL_BASE_BEGIN_CHECK=ancillary,client_and_negative_suite_allowance=allowance,
            planning_reference_total=total,planning_reference_minutes=total/60),
        formulas=dict(raw='bytes_full*10/115200 + frames_full*observed_extra_seconds_per_frame',
            pipeline_and_pages='896*(3*mean_F14_RUN_exchange + F14_PROFILE_exchange + 4*mean_F14_three_pages_exchange)',
            total='startup_proxy + RAW + pipeline_and_pages + ancillary_commands + 10s_allowance'),
        recommended_user_expectation_minutes=[15,25],recommended_max_seconds=1800,
        recommended_limit_minutes=30,
        qualifications=[
            'Uses two physical gate cases, not measured complete-campaign duration.',
            'The all-F14 projection is a planning proxy; it is not an upper bound, WCET or confidence interval.',
            'DE7 was faster in this gate, but its one window does not establish all candidate costs.',
            'Wire time assumes the same 115200 baud,8N1 setup; USB,OS and PC load can add delays.',
            'The 1800s budget is checked around UART waits and does not preempt Python imports or external blocking calls.',
            'Startup and closure are not isolated from each other; client-PC overhead is represented approximately.',
        ],
        selector_costs_enabled=False,new_physical_RUN=0,TFLite_invocations=0,
        new_training_fits=0,serial_port_opened=False,hardware_flashed=False,
        seconds=time.perf_counter()-started,
    )
    with REPORT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(status=result['status'],planning_reference_minutes=total/60,
        recommended_user_expectation_minutes=[15,25],recommended_max_seconds=1800,
        RAW_wire_bytes=full_wire_bytes),ensure_ascii=True))


if __name__ == '__main__':
    main()
