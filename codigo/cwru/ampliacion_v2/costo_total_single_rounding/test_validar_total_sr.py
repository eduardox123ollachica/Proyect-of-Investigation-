"""Pruebas sin serial real, TensorFlow, ARM ni tiempos de hardware."""
import copy
import hashlib
import json
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

import validar_total_sr as v


TEST_ROOT=Path(__file__).resolve().parents[4]/'.codex_tmp/costo_total_sr_client_tests'


def temporary_folder():
    TEST_ROOT.mkdir(parents=True,exist_ok=True)
    directory=tempfile.TemporaryDirectory(dir=TEST_ROOT)
    # Cleanup is limited to a new child of this explicit workspace test root.
    assert Path(directory.name).resolve().is_relative_to(TEST_ROOT.resolve())
    return directory


def rawhex(words):
    return np.asarray([int(x,16) for x in words],dtype='<u4').tobytes().hex()


def fixture():
    indices=[0,1,2,3,4,5,6];det=[0,1,2,3,5,6]
    r=dict(id=6,mask_id=6,candidate='DE7',dims=7,det_dims=6,model_bytes=3500,
        model_fnv='12345678',params_fnv='00000000',model_sha256='a'*64,source_fit_signature='b'*64,
        feature_indices=indices,detector_indices=det,dnn_mean32_bits=v.bits([0]*7),dnn_scale32_bits=v.bits([1]*7),
        detector_mean32_bits=v.bits([0]*6),detector_scale32_bits=v.bits([1]*6),centroid32_bits=v.bits([0]*6),
        threshold_bits=v.bits([2])[0],input_quantization=dict(scale_bits=v.bits([1])[0],zero_point=0))
    r['params_fnv']=v.fnv(v.params_bytes(r));r['params_bytes_hex']=v.params_bytes(r).hex()
    row=dict(global_id=568,checksum_fnv1a='12345678',features_pc_cache14_bits=v.bits([.25]*14),
             arm_features14_bits=None)
    features=v.bits([.25]*7+[0]*7);expected=v.replay(features,r)
    result=dict(run_id=8,rep_id=6,global_id=568,checksum='12345678',mode=0,total_cycles=15000000,
        extract_cycles=0,input_cycles=0,nn_cycles=0,detector_cycles=0,residual_cycles=0,milliseconds=15000000/180000,
        predicted_class=1,output_int8=[-128,127,-128],score_bits=expected['score_bits'],rejected=expected['rejected'],
        arena_used=1188,extractions=1)
    pages=dict(features_bits=features,**expected)
    return r,row,result,pages


class BufferTrace:
    def __init__(self):self.items=[]
    def append(self,item):self.items.append(item)


class ChunkPort:
    def __init__(self,chunks):self.chunks=list(chunks);self.writes=[]
    @property
    def in_waiting(self):return len(self.chunks[0]) if self.chunks else 0
    def write(self,data):self.writes.append(data);return len(data)
    def flush(self):pass
    def read(self,count):
        if not self.chunks:return b''
        chunk=self.chunks.pop(0)
        # Honour read length so partial arrivals retain stream order.
        if len(chunk)>count:self.chunks.insert(0,chunk[count:])
        return chunk[:count]


class FakeLite:
    def predict(self,rid,q):return [-128,127,-128]


class NegativeLink:
    """Doble determinista del estado publicado/recepción para recorrer negativos."""
    def __init__(self,result,rep):
        self.result=result;self.rep=rep;self.valid=True;self.count=8192;self.commands=[]
    def ask(self,command,prefix,allow_error=False):
        self.commands.append(command);rid=self.result['run_id'];r=self.rep
        if command.startswith('CWRUTOTAL_FEATURES,'):
            text=command.split(',')[1:]
            if len(text)!=1 or text[0]=='0':return 'CWRUTOTAL_ERROR,FORMAT,-1'
            if not self.valid:return 'CWRUTOTAL_ERROR,NO_RESULT,-1'
            if text[0]!=str(rid):return 'CWRUTOTAL_ERROR,STALE_ID,-1'
            return f'CWRUTOTAL_FEATURES,{rid},{r["id"]},{self.result["checksum"]},'+'00'*56
        fixed={'CWRUTOTAL_INFO,1':'FORMAT','CWRUTOTAL_UNKNOWN':'UNKNOWN','CWRUTOTAL_MODEL,abc':'FORMAT',
               'CWRUTOTAL_MODEL,32':'REP_ID','CWRUTOTAL_SELECT,32':'REP_ID','CWRUTOTAL_SELECT,abc':'FORMAT'}
        if command in fixed:
            if command.startswith('CWRUTOTAL_SELECT,'):self.valid=False
            return 'CWRUTOTAL_ERROR,'+fixed[command]+',-1'
        if command.startswith('CWRUTOTAL_RUN,'):
            self.valid=False
            if command.endswith(',123'):return 'CWRUTOTAL_ERROR,FORMAT,-1'
            if self.count!=8192:return 'CWRUTOTAL_ERROR,INCOMPLETE,'+str(self.count)
            return 'CWRUTOTAL_ERROR,CHECKSUM,-1'
        if command==f'CWRUTOTAL_SELECT,{r["id"]}':
            self.valid=False
            return f'CWRUTOTAL_SELECTED,{r["id"]},0,1188,0,1,{r["model_fnv"]},{r["params_fnv"]}'
        if command=='CWRUFEAT_BEGIN':self.valid=False;self.count=0;return 'CWRUFEAT_READY,8192'
        if command=='CWRUFEAT_DATA,0,0.25,-0.5,1,-1':self.count=4;return 'CWRUFEAT_ACK,4'
        if command=='CWRUFEAT_DATA,3,1':return 'CWRUFEAT_ERROR,OFFSET,4'
        if command=='CWRUFEAT_DATA,4,5,nan':return 'CWRUFEAT_ERROR,BAD_VALUE,1'
        if command.startswith('CWRUFEAT_DATA,4,'):return 'CWRUFEAT_ERROR,FRAME_VALUES,48'
        if command=='CWRUFEAT_CHECK':return 'CWRUFEAT_CHECK,4,'+v.fnv(np.asarray([.25,-.5,1,-1],dtype='<f4').tobytes())
        raise AssertionError('Comando inesperado en doble: '+command)


class TestReplay(unittest.TestCase):
    def test_parameter_serialization_le_and_length(self):
        r,_,_,_=fixture();data=v.params_bytes(r)
        self.assertEqual(len(data),297);self.assertEqual(data[:3],bytes([7,6,6]))
        self.assertEqual(data[3:17],bytes(list(range(7))+[0]*7))
        self.assertEqual(data[17:29],bytes([0,1,2,3,5,6]+[0]*6))
        self.assertEqual(data[-4:],struct.pack('<i',0));self.assertEqual(r['params_fnv'],v.fnv(data))

    def test_ties_even_and_saturation_masks(self):
        r,_,_,_=fixture();f=v.bits([-128.5,127.5,126.5,-127.5,0,-129,129]+[0]*7)
        out=v.replay(f,r)
        self.assertEqual(out['quantized'],[-128,127,126,-128,0,-128,127])
        self.assertEqual(out['saturated_low'],1<<5)
        self.assertEqual(out['saturated_high'],(1<<1)|(1<<6))

    def test_float32_sequential_sum_not_float64(self):
        r,_,_,_=fixture();f=np.zeros(14,dtype=np.float32)
        f[0]=4096;f[1]=1;f[2]=1;f[3]=1;f[5]=1;f[6]=1
        out=v.replay(v.bits(f),r)
        # Every +1 is lost at2**24 with ties-to-even; f64 reduction would give+5.
        self.assertEqual(out['score_bits'],v.bits([16777216])[0])

    def test_detector_uses_selected_indices_excludes_frequency(self):
        r,_,_,_=fixture();f=np.zeros(14,np.float32);f[4]=1e9;f[11]=1e8;f[8]=1e7
        self.assertEqual(v.replay(v.bits(f),r)['score_bits'],'00000000')

    def test_strict_threshold(self):
        r,_,_,_=fixture();r['threshold_bits']=v.bits([1])[0];f=[1]+[0]*13
        self.assertFalse(v.replay(v.bits(f),r)['rejected'])
        f[1]=.5;self.assertTrue(v.replay(v.bits(f),r)['rejected'])


class TestProtocol(unittest.TestCase):
    def test_partial_reads_wait_for_crlf(self):
        port=ChunkPort([b'CWRUTO',b'',b'TAL_INFO,abc\r',b'\n'])
        trace=BufferTrace();link=v.Link(port,trace,.2)
        self.assertEqual(link.ask('CWRUTOTAL_INFO','CWRUTOTAL_INFO,'),'CWRUTOTAL_INFO,abc')
        self.assertEqual(port.writes,[b'CWRUTOTAL_INFO\r\n']);self.assertEqual(len(trace.items),2)

    def test_multiple_lines_single_chunk_preserved(self):
        link=v.Link(ChunkPort([b'boot\r\nCWRUTOTAL_INFO,ok\r\nCWRUTOTAL_MODEL,ok\r\n']),BufferTrace(),.2)
        self.assertEqual(link.ask('a','CWRUTOTAL_INFO,'),'CWRUTOTAL_INFO,ok')
        self.assertEqual(link.ask('b','CWRUTOTAL_MODEL,'),'CWRUTOTAL_MODEL,ok')

    def test_fragment_without_cr_is_rejected(self):
        link=v.Link(ChunkPort([b'CWRUTOTAL_INFO,ok\n']),BufferTrace(),.2)
        with self.assertRaises(ValueError):link.ask('a','CWRUTOTAL_INFO,')

    def test_timeout_and_error(self):
        with self.assertRaises(TimeoutError):v.Link(ChunkPort([]),BufferTrace(),.001).ask('a','b')
        link=v.Link(ChunkPort([b'CWRUTOTAL_ERROR,FORMAT,-1\r\n']),BufferTrace(),.2)
        self.assertEqual(link.ask('a','b',True),'CWRUTOTAL_ERROR,FORMAT,-1')

    def test_global_budget_before_write(self):
        port=ChunkPort([]);link=v.Link(port,BufferTrace(),.2,deadline=0)
        with self.assertRaises(ValueError):link.ask('a','b')
        self.assertEqual(port.writes,[])

    def test_info_catalog_and_fpscr(self):
        cat=dict(catalog_fnv='12345678',catalog_sha256='a'*64)
        line='CWRUTOTAL_INFO,1,32,12000,4096,2,180000000,1,00000000,12345678,'+'a'*64+',-1,0,0,13,64,0'
        self.assertEqual(v.parse_info(line,cat)['selected_id'],-1)
        for bad in (line.replace('00000000','01000000'),line.replace('180000000','16000000'),line.replace('13,64,0','13,64,1')):
            with self.assertRaises(ValueError):v.parse_info(bad,cat)

    def test_result_envelopes_and_binding(self):
        r,row,result,pages=fixture()
        line='CWRUTOTAL_RESULT,8,6,12345678,0,15000000,0,0,0,0,0,1,-128,127,-128,'+result['score_bits']+',0,1188,1'
        got=v.parse_result(line,r,row,7,0,1188);self.assertEqual(got['total_cycles'],15000000)
        with self.assertRaises(ValueError):v.parse_result(line,r,row,6,0,1188)
        with self.assertRaises(ValueError):v.parse_result(line.replace(',1188,1',',1188,2'),r,row,7,0,1188)
        with self.assertRaises(ValueError):v.parse_result(line.replace(',0,0,0,0,0,1,',',1,0,0,0,0,1,'),r,row,7,0,1188)
        profile=line.replace(',0,15000000,0,0,0,0,0,',',1,15000000,14000000,1000,800000,1000,198000,')
        self.assertEqual(v.parse_result(profile,r,row,7,1,1188)['residual_cycles'],198000)
        with self.assertRaises(ValueError):v.parse_result(profile.replace(',198000,',',198001,'),r,row,7,1,1188)

    def test_result_wraparound(self):
        r,row,result,_=fixture()
        line='CWRUTOTAL_RESULT,1,6,12345678,0,1,0,0,0,0,0,1,-128,127,-128,'+result['score_bits']+',0,1188,1'
        self.assertEqual(v.parse_result(line,r,row,0xffffffff,0,1188)['run_id'],1)

    def test_same_input_output_exact_independent_cache_report(self):
        r,row,result,pages=fixture();base=dict(features_bits=v.bits([.25]*14))
        pc=dict(raw_features14_bits=v.bits([.25]*14),int8_inputs=[9]*7,int8_outputs=[-128,127,-128],
                detector_score_bits='00000000',outside_normality=True)
        comparison=v.compare(result,pages,base,r,pc,FakeLite())
        self.assertEqual(comparison['cached_PC']['quantized_value_differences'],7)
        self.assertTrue(comparison['cached_PC']['detector_rejection_differs'])
        self.assertFalse(comparison['cached_PC_end_to_end_bit_exact_claimed'])
        corrupt=copy.deepcopy(pages);corrupt['features_bits'][12]='80000000'
        with self.assertRaises(ValueError):v.compare(result,corrupt,base,r,pc,FakeLite())
        corrupt=copy.deepcopy(pages);corrupt['quantized'][0]+=1
        with self.assertRaises(ValueError):v.compare(result,corrupt,base,r,pc,FakeLite())

    def test_pages_bind_run_model_checksum_and_float_byte_order(self):
        rep,row,result,pages=fixture();rid=result['run_id'];rep_id=rep['id'];ck=result['checksum']
        responses={
            'FEATURES':f'CWRUTOTAL_FEATURES,{rid},{rep_id},{ck},'+rawhex(pages['features_bits']),
            'INPUT':f'CWRUTOTAL_INPUT,{rid},{rep_id},{ck},'+rawhex(pages['standardized_bits'])+','+
                rawhex(pages['affine_bits'])+','+np.asarray(pages['quantized'],dtype=np.int8).tobytes().hex()+
                f',{pages["saturated_low"]:04x},{pages["saturated_high"]:04x}',
            'DETECTOR':f'CWRUTOTAL_DETECTOR,{rid},{rep_id},{ck},'+rawhex(pages['detector_z_bits'])+','+
                rawhex(pages['detector_terms_bits'])+','+pages['score_bits']+','+str(int(pages['rejected']))}
        class PagesLink:
            def ask(self,command,prefix):return responses[command.split(',')[0].split('_',1)[1]]
        self.assertEqual(v.get_pages(PagesLink(),result,rep),pages)
        responses['INPUT']=responses['INPUT'].replace(f',{rid},{rep_id},',f',{rid},{rep_id+1},')
        with self.assertRaises(ValueError):v.get_pages(PagesLink(),result,rep)

    def test_negative_suite_17_rejections_without_valid_extra_RUN(self):
        rep,row,result,pages=fixture();link=NegativeLink(result,rep)
        rows,support=v.negatives(link,result,rep)
        self.assertEqual(len(rows),17);self.assertTrue(all(r['status']=='PASS' for r in rows))
        self.assertEqual(support['rep_id'],rep['id']);self.assertTrue(support['reused'])
        self.assertEqual(link.count,4);self.assertFalse(link.valid)
        runs=[c for c in link.commands if c.startswith('CWRUTOTAL_RUN,')]
        self.assertEqual(len(runs),3);self.assertEqual(runs[-1],'CWRUTOTAL_RUN,'+result['checksum'])
    def test_frames_float32_roundtrip_and_size(self):
        rng=np.random.default_rng(3);window=rng.normal(size=(4096,2)).astype('<f4')
        items=list(v.frames(window));self.assertEqual(len(items),171);self.assertEqual(items[-1][1],8192)
        joined=np.concatenate([np.asarray(c.split(',')[2:],dtype='<f4') for c,_ in items])
        self.assertEqual(joined.tobytes(),window.tobytes());self.assertTrue(all(len(c)+2<1024 for c,_ in items))


class TestBankAndSeal(unittest.TestCase):
    def test_current_resource_preflight_and_schedule(self):
        folder=Path(__file__).resolve().parents[4]/'despliegue_stm32/costo_total_representantes_v2_180mhz_2026-10-25/recursos'
        if not folder.is_dir():self.skipTest('Recursos aún no exportados')
        cat,bank,windows,manifest=v.load_resources(folder)
        schedule=v.schedule(bank,True);self.assertEqual(len(schedule),42)
        self.assertEqual(sum(len(rids) for _,rids in schedule)*3,2688)
        for b in bank['banks']:
            self.assertEqual([g for g,_ in schedule if g in set(b['global_ids'])],b['global_ids'])
        self.assertEqual(v.schedule(bank,False),[(568,[0]),(232,[22])])
        for r in cat['representatives']:
            for gid in r['scope']['benchmark_global_window_ids']:
                ref=bank['reference_by_rep_gid'][str(r['id'])][str(gid)]
                replay=v.replay(ref['raw_features14_bits'],r)
                self.assertEqual(replay['standardized_bits'],ref['normalized_features_bits'])
                self.assertEqual(replay['quantized'],ref['int8_inputs'])
                self.assertEqual(replay['score_bits'],ref['detector_score_bits'])
                self.assertEqual(replay['rejected'],ref['outside_normality'])

    def test_statistics_excludes_profile(self):
        reps=[dict(id=0,candidate='F14',outer_load_hp=0)]
        rows=[dict(mode=0,rep_id=0,total_cycles=18),dict(mode=0,rep_id=0,total_cycles=36),
              dict(mode=1,rep_id=0,total_cycles=99999)]
        summary=v.statistics(rows,reps)[0]
        self.assertEqual(summary['count'],2);self.assertEqual(summary['mean_cycles'],27)
        self.assertFalse(summary['selector_enabled']);self.assertFalse(summary['PROFILE_included'])

    def test_gate_missing_report_seal_cannot_authorize_full(self):
        with temporary_folder() as temp:
            folder=Path(temp);v.write(folder/'RESULTADO_TOTAL.json',dict(status='GATE_PASS_NOT_TOTAL_COST_VALIDATED'))
            mp=folder/'MANIFEST_EVIDENCIAS_SHA256.json';v.write(mp,{})
            mp.with_suffix('.sha256').write_text(v.sha(mp)+'  '+mp.name+'\n',encoding='ascii')
            with self.assertRaises(ValueError):v.verify_gate(folder,{},'a'*64,'b'*64)

    def test_resource_path_escape_rejected(self):
        with temporary_folder() as temp:
            parent=Path(temp);(parent/'outer').write_text('x');(parent/'folder').mkdir()
            with self.assertRaises(ValueError):v.contained(parent/'folder','../outer')


class TestSingleRoundingGate(unittest.TestCase):
    """Ejercita la puerta nueva con transportes simulados y el banco congelado."""
    @classmethod
    def setUpClass(cls):
        cls.resources=Path(__file__).resolve().parents[4]/'despliegue_stm32/costo_total_representantes_v2_180mhz_2026-10-25/recursos'
        cls.cat,cls.bank,cls.windows,_=v.load_resources(cls.resources)

    def synthetic_gate(self,folder,report_change=None,plan_change=None):
        firmware=b'ELF_BINDING_TEST_ONLY'
        firmware_sha=hashlib.sha256(firmware).hexdigest()
        (folder/'firmware_declarado.elf').write_bytes(firmware)
        (folder/'validar_total_ejecutado.py').write_bytes(Path(v.__file__).read_bytes())
        (folder/'TRAZA_UART.jsonl').write_text('SIMULATED_TEST_ONLY\n',encoding='utf-8')
        (folder/'MEDICIONES.jsonl').write_text('SIMULATED_TEST_ONLY\n',encoding='utf-8')
        report=dict(status=v.GATE_STATUS,mode='gate',catalog_sha256=v.UPSTREAM_CATALOG_SHA256,
            firmware_elf_sha256=firmware_sha,completed_primary_RUN=6,completed_PROFILE=2,completed_BASE=2,
            negative_rejections=17,physical_unique_window_ids=[568,232],gate_cases=v.gate_plan(),
            model_selection_count=2,resource_manifest_sha256=v.UPSTREAM_RESOURCE_MANIFEST_SHA256,
            client_sha256=v.sha(v.__file__),timing_statistics=[dict(rep_id=0,count=3),dict(rep_id=22,count=3)],
            **v.rounding_provenance())
        plan=dict(mode='gate',schedule=v.gate_plan(),expected_primary_RUN=6,expected_PROFILE=2,expected_BASE=2,
            firmware_elf_sha256=firmware_sha,resource_manifest_sha256=v.UPSTREAM_RESOURCE_MANIFEST_SHA256,
            **v.rounding_provenance())
        if report_change:report.update(report_change)
        if plan_change:plan.update(plan_change)
        v.write(folder/'RESULTADO_TOTAL.json',report);v.write(folder/'PLAN_EJECUTADO.json',plan)
        entries={p.name:v.sha(p) for p in folder.iterdir() if p.is_file()}
        manifest=folder/'MANIFEST_EVIDENCIAS_SHA256.json';v.write(manifest,entries)
        manifest.with_suffix('.sha256').write_text(v.sha(manifest)+'  '+manifest.name+'\n',encoding='ascii')
        return firmware_sha

    def validate_synthetic(self,report_change=None,plan_change=None):
        with temporary_folder() as temp:
            folder=Path(temp)
            firmware_sha=self.synthetic_gate(folder,report_change,plan_change)
            return v.verify_gate(folder,self.cat,firmware_sha,v.UPSTREAM_RESOURCE_MANIFEST_SHA256)

    def test_new_two_case_gate_authorizes_matching_full_only(self):
        self.validate_synthetic()
        self.assertEqual(v.gate_plan(),[[568,[0]],[232,[22]]])
        self.assertFalse(v.rounding_provenance()['rounding_mode_verified_by_UART'])

    def test_legacy_status_and_three_run_gate_rejected(self):
        for update in (dict(status='GATE_PASS_NOT_TOTAL_COST_VALIDATED'),dict(completed_primary_RUN=3),
                       dict(completed_PROFILE=1),dict(completed_BASE=1)):
            with self.subTest(update=update),self.assertRaises(ValueError):self.validate_synthetic(update)

    def test_new_gate_requires_exact_control_and_regression_schedule(self):
        for plan in ([[568,[0]]],[[232,[22]],[568,[0]]],[[568,[0]],[232,[21]]]):
            with self.subTest(plan=plan),self.assertRaises(ValueError):self.validate_synthetic(plan_change=dict(schedule=plan))
        with self.assertRaises(ValueError):self.validate_synthetic(dict(gate_cases=[[568,[0]]]))
        with self.assertRaises(ValueError):self.validate_synthetic(dict(physical_unique_window_ids=[232,568]))

    def test_new_gate_requires_unchanged_rounding_provenance_and_client_binding(self):
        for change in (dict(client_sha256='0'*64),dict(new_rounding_mode='double'),
                       dict(upstream_client_sha256='0'*64),dict(rounding_mode_verified_by_UART=True),
                       dict(resource_manifest_sha256='0'*64)):
            with self.subTest(change=change),self.assertRaises(ValueError):self.validate_synthetic(change)

    def test_three_primary_timings_per_case_required(self):
        with self.assertRaises(ValueError):
            self.validate_synthetic(dict(timing_statistics=[dict(rep_id=0,count=6)]))

    def test_regression_case_must_be_authorized_by_its_original_bank(self):
        changed=copy.deepcopy(self.bank)
        row=next(r for r in changed['windows'] if r['global_id']==232)
        row['authorized_rep_ids'].remove(22)
        with self.assertRaises(ValueError):v.schedule(changed,False)

    def test_real_legacy_gate_is_rejected_without_modification(self):
        folder=self.resources.parent/'Validacion_PC/resultados/20261026T040903_561072Z'
        manifest=folder/'MANIFEST_EVIDENCIAS_SHA256.json';before=v.sha(manifest)
        with self.assertRaises(ValueError):
            v.verify_gate(folder,self.cat,v.sha(folder/'firmware_declarado.elf'),v.UPSTREAM_RESOURCE_MANIFEST_SHA256)
        self.assertEqual(v.sha(manifest),before)

    def execute_simulated_gate(self,temp,wrong_output=False):
        """Ejecuta run() real con I/O y Lite falsos; ninguna conexión o TF."""
        folder=Path(temp);elf=folder/'simulated.elf'
        elf.write_bytes(b'\x7fELF\x01\x01'+bytes(10)+struct.pack('<HH',2,40))
        args=SimpleNamespace(resources=self.resources,firmware_elf=elf,full=False,gate_evidence=None,
            output=folder/'results',build_report=None,programmer_report=None,max_seconds=600.,
            port='SIMULATED_NO_COM',baud=115200,timeout=15.)
        context=SimpleNamespace(gid=None,rid=None,commands=[],info_count=0)
        class SerialStub:
            def __enter__(self):return self
            def __exit__(self,*args):return False
        class LinkStub:
            def __init__(self,*args,**kwargs):pass
            def ask(self,command,prefix,**kwargs):
                context.commands.append(command)
                if command.startswith('CWRUTOTAL_SELECT,'):context.rid=int(command.split(',')[1])
                return command
        class LiteStub:
            def __init__(self,folder,reps):self.tf=SimpleNamespace(__version__='SIMULATED_NO_TENSORFLOW')
            def predict(self,rid,q):
                ref=TestSingleRoundingGate.bank['reference_by_rep_gid'][str(rid)][str(context.gid)]
                if q!=ref['int8_inputs']:raise AssertionError('Fake Lite requires exact frozen input')
                return ref['int8_outputs']
        def transmit(link,window,row):context.gid=row['global_id']
        def baseline(link,row):return dict(global_id=row['global_id'],features_bits=row['arm_features14_bits'])
        def info(*args):
            context.info_count+=1
            return dict(last_run_id=0 if context.info_count==1 else 8,result_valid=False,hclk_hz=v.HCLK)
        def selected(line,rep):return dict(rep_id=rep['id'],cycles=100,arena_used=1188,reused=False)
        def result(line,rep,row,last,mode,arena):
            qref=self.bank['reference_by_rep_gid'][str(rep['id'])][str(row['global_id'])]
            output=list(qref['int8_outputs'])
            if wrong_output and rep['id']==22:output=[42,-101,-69]
            f=[row['arm_features14_bits'][i] if i in rep['feature_indices'] else '00000000' for i in range(14)]
            replay=v.replay(f,rep)
            return dict(run_id=last+1,rep_id=rep['id'],global_id=row['global_id'],checksum=row['checksum_fnv1a'],
                mode=mode,total_cycles=180000,extract_cycles=170000 if mode else 0,input_cycles=2000 if mode else 0,
                nn_cycles=6000 if mode else 0,detector_cycles=1000 if mode else 0,residual_cycles=1000 if mode else 0,
                milliseconds=1.,predicted_class=int(np.argmax(output)),output_int8=output,
                score_bits=replay['score_bits'],rejected=replay['rejected'],arena_used=arena,extractions=1)
        def pages(link,result,rep):
            row=next(r for r in self.bank['windows'] if r['global_id']==result['global_id'])
            f=[row['arm_features14_bits'][i] if i in rep['feature_indices'] else '00000000' for i in range(14)]
            return dict(features_bits=f,**v.replay(f,rep))
        def negative_suite(link,result,rep):
            self.assertEqual((result['global_id'],rep['id'],result['run_id']),(232,22,8))
            return [dict(status='PASS') for _ in range(17)],dict(rep_id=22,cycles=0,reused=True,arena_used=1188)
        mocks=dict(Link=LinkStub,LiteModels=LiteStub,transmit=transmit,baseline=baseline,parse_info=info,
            validate_model=lambda *args:None,parse_selected=selected,parse_result=result,get_pages=pages,negatives=negative_suite)
        with mock.patch.dict(v.__dict__,mocks),mock.patch.dict('sys.modules',{'serial':SimpleNamespace(Serial=lambda *args,**kwargs:SerialStub())}):
            if wrong_output:
                with self.assertRaisesRegex(ValueError,'Salida INT8 STM32 distinta de PC para LA MISMA entrada'):
                    v.run(args)
                reports=list(args.output.glob('*/RESULTADO_TOTAL.json'));self.assertEqual(len(reports),1)
                saved=v.read(reports[0]);self.assertEqual(saved['status'],'FAILED_INCOMPLETE')
                self.assertEqual((saved['completed_primary_RUN'],saved['completed_PROFILE'],saved['completed_BASE']),(3,1,2))
                self.assertEqual(saved['negative_rejections'],0)
                valid_commands=[c for c in context.commands if c.startswith(('CWRUTOTAL_RUN,','CWRUTOTAL_PROFILE,'))]
                self.assertEqual(len(valid_commands),5)  # No retry after the first failing regression RUN.
                return
            report=v.run(args)
        self.assertEqual(report['status'],v.GATE_STATUS)
        self.assertEqual((report['completed_primary_RUN'],report['completed_PROFILE'],report['completed_BASE']),(6,2,2))
        self.assertEqual(report['negative_rejections'],17)
        run_folder=next(args.output.iterdir())
        v.verify_gate(run_folder,self.cat,v.sha(elf),v.UPSTREAM_RESOURCE_MANIFEST_SHA256)

    def test_run_collects_control_and_regression_then_17_negatives(self):
        with temporary_folder() as temp:self.execute_simulated_gate(temp)

    def test_one_LSB_output_difference_still_stops_gate_without_retry(self):
        with temporary_folder() as temp:self.execute_simulated_gate(temp,wrong_output=True)


if __name__=='__main__':
    unittest.main(verbosity=2)
