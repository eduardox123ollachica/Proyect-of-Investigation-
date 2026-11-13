"""Pruebas de copia/configuración/build simulado; nunca compilar o usar D real."""
from __future__ import annotations
import copy
from contextlib import contextmanager,redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import Mock,patch
import xml.etree.ElementTree as ET

import preparar_proyecto as p


def put(path,text):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(text,encoding='utf-8')


def catalog_files(resources,count=8,header=8):
    catalog=dict(count=count,catalog_sha256='a'*64,
        representatives=[dict(id=i,representative_id='synthetic-'+str(i)) for i in range(count)])
    put(resources/'CATALOGO_TOTAL.json',json.dumps(catalog))
    put(resources/'CATALOGO_CANONICO.json','{"synthetic":true}\n')
    put(resources/'cwru_total_catalog.h',f'#define CWRUTV_COUNT {header}U\n')
    put(resources/'cwru_total_catalog.cpp','/* Synthetic: not a model or build source. */\n')
    files={f.name:p.sha(f) for f in resources.iterdir() if f.is_file() and not f.name.startswith('MANIFEST_')}
    manifest=resources/'MANIFEST_RECURSOS_SHA256.json'
    put(manifest,json.dumps(files));put(manifest.with_suffix('.sha256'),p.sha(manifest)+'  '+manifest.name+'\n')


def fake_configuration(name,index):
    node=ET.Element('cconfiguration',id='cfg'+str(index))
    module=ET.SubElement(node,'storageModule',moduleId='cdtBuildSystem')
    cfg=ET.SubElement(module,'configuration',id='cfg'+str(index),name=name,artifactName='${ProjName}')
    folder=ET.SubElement(cfg,'folderInfo',id='folder'+str(index))
    chain=ET.SubElement(folder,'toolChain',id='chain'+str(index),targetTool='compiler'+str(index)+';linker'+str(index))
    for kind in ('c','cpp'):
        tool=ET.SubElement(chain,'tool',id=kind+'compiler'+str(index),superClass='x.tool.'+kind+'.compiler')
        opt=ET.SubElement(tool,'option',id=kind+'include'+str(index),superClass='x.includepaths')
        ET.SubElement(opt,'listOptionValue',value='../CWRU_TotalV2')
        ET.SubElement(opt,'listOptionValue',value='../CWRU_TotalSR')
    sources=ET.SubElement(cfg,'sourceEntries')
    for directory in ('Src','Inc','CWRU_Masks','CWRU_TotalV2','CWRU_TotalSR'):
        ET.SubElement(sources,'entry',name=directory,kind='sourcePath')
    return node


def fixture_cproject(path):
    root=ET.Element('cproject');settings=ET.SubElement(root,'storageModule',moduleId='org.eclipse.cdt.core.settings')
    names=['Debug','Release',*[f'Release_old_{i}' for i in range(14)],p.BASE]
    for index,name in enumerate(names):settings.append(fake_configuration(name,index))
    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(p.helper().xml_bytes(root))


def fixture_launch(path):
    root=ET.Element('launchConfiguration')
    values={'org.eclipse.cdt.launch.PROGRAM_NAME':p.BASE+'/'+p.NAME+'.elf',
        'org.eclipse.cdt.launch.PROJECT_BUILD_CONFIG_ID_ATTR':'old-config-id',
        'org.eclipse.cdt.launch.PROJECT_BUILD_CONFIG_AUTO_ATTR':'false',
        'com.st.stm32cube.ide.mcu.debug.launch.loadList':json.dumps({'fItems':[{'fPath':'old.elf','fBuild':False}]}),
        'synthetic.stlink.log_file':'D:/must-not-write-here.log'}
    for key,value in values.items():ET.SubElement(root,'stringAttribute',key=key,value=value)
    path.write_bytes(ET.tostring(root,encoding='utf-8'))


@contextmanager
def fixture():
    base=(p.ROOT/'tmp').resolve();base.mkdir(exist_ok=True)
    folder=Path(tempfile.mkdtemp(prefix='canon42_project_test_',dir=base)).resolve()
    source=folder/'source';out=folder/'output';resources=out/'recursos';templates=folder/'templates'
    project=out/'copia_aislada'/p.NAME
    source.mkdir();resources.mkdir(parents=True);templates.mkdir()
    fixture_cproject(source/'.cproject');put(source/'.project','<projectDescription><name>'+p.NAME+'</name></projectDescription>')
    put(source/'Src/main.c','/* synthetic original main; no firmware */\n')
    put(source/'Src/cmd_cwru_feat.c','/* synthetic receiver */\n')
    put(source/'Src/uart_commands.c','/* synthetic router */\n')
    put(source/'.settings/language.settings.xml','<settings/>\n')
    put(source/'Lib/libtensorflow-microlite.a','synthetic archive bytes; never linked')
    for name in p.SIMPLE_FILES:put(source/'CWRU_TotalV2'/name,'/* synthetic '+name+' */\n')
    put(source/'CWRU_TotalV2/cwru_total_nn.cpp','constexpr uint32_t kModelCount = 32U;\n')
    put(source/'CWRU_TotalV2/cwru_total_commands.inc',
        'static void helper_before_handler(void) {}\n'
        'void CWRUTOTAL_Command(char *command,char *out,size_t cap)\n{\n'
        '    if(command==NULL || out==NULL || cap==0U) return;\n'
        '        total_format(out,cap,"CWRUTOTAL_INFO,1,32,12000,4096,2,%lu",\n'
        '            (unsigned long)HAL_RCC_GetHCLKFreq(),0);\n}\n')
    for name in p.CATALOG_FILES:put(source/'CWRU_TotalV2'/name,'/* preserve original 32-bank '+name+' */\n')
    put(source/'CWRU_TotalSR/arm_nn_vec_mat_mult_t_per_ch_s8.c','/* original SR override */\n')
    for name in p.POST_FILES:put(templates/name,'/* synthetic POST interface '+name+' */\n')
    fixture_launch(source/(p.NAME+'_TotalV2_SR_180MHz_SinRecompilar.launch'))
    for name in ('Debug','Release',p.BASE):
        put(source/name/(p.NAME+'.elf'),'synthetic existing ELF '+name)
        put(source/name/'subdir.mk','not source')
    catalog_files(resources)
    try:
        with patch.object(p,'SOURCE',source),patch.object(p,'OUT',out),patch.object(p,'PROJECT',project),\
             patch.object(p,'RESOURCES',resources),patch.object(p,'HERE',templates),\
             patch.object(p,'PREPARATION',out/'PREPARACION_PROYECTO.json'),\
             patch.object(p,'MANIFEST',out/'MANIFEST_PREPARACION_PROYECTO_SHA256.json'):
            yield source,out,project,resources,templates
    finally:
        if folder.parent!=base or not folder.name.startswith('canon42_project_test_'):
            raise ValueError('Fixture fuera del temporal autorizado')
        shutil.rmtree(folder)


class CopyAndConfiguration(unittest.TestCase):
    def test_build_caches_are_excluded_but_legacy_sources_and_library_are_retained(self):
        for path in ('Debug/a.c','Release_180MHz_TotalV2_SR/a.o','.metadata/state','.git/config','x/__pycache__/a.pyc',
            'CWRU_TotalV2/a.o','Src/main.su','ReleaseLater/cache.bin'):
            self.assertTrue(p.skip_path(path),path)
        for path in ('Src/main.c','.cproject','.project','Lib/libtensorflow-microlite.a',
            'CWRU_TotalV2/cwru_total_catalog.cpp','CWRU_Canon8/cwru_canon_post.h'):
            self.assertFalse(p.skip_path(path),path)

    def test_configuration_clone_preserves_all_17_legacy_nodes_and_avoids_id_collisions(self):
        with fixture() as (source,_,_,_,_):
            helper=p.helper();root,settings,old=helper.configs(source/'.cproject')
            before={name:ET.tostring(node) for name,node in old.items()}
            original_ids={n.get('id') for cfg in old.values() for n in cfg.iter() if n.get('id')}
            new=p.add_configuration(helper,root,settings,old)
            self.assertEqual(len(settings),18)
            self.assertTrue(all(ET.tostring(old[name])==value for name,value in before.items()))
            self.assertFalse(original_ids.intersection(n.get('id') for n in new.iter() if n.get('id')))
            entries=[n.get('name') for n in new.findall('.//sourceEntries/entry')]
            self.assertIn(p.PRIVATE,entries);self.assertNotIn('CWRU_TotalV2',entries)
            self.assertIn('CWRU_TotalSR',entries)
            self.assertEqual(sum(n.get('value')=='../'+p.PRIVATE for n in new.iter()),2)

    def test_private_sources_use_catalog_count_protocol2_and_POST_without_editing_originals(self):
        with fixture() as (source,_,_,resources,templates):
            before={name:p.sha(source/'CWRU_TotalV2'/name) for name in (*p.SIMPLE_FILES,*p.CATALOG_FILES)}
            files=p.private_sources(source,resources,templates)
            self.assertEqual(len(files),11)
            self.assertIn(b'kModelCount = CWRUTV_COUNT;',files['cwru_total_nn.cpp'])
            commands=files['cwru_total_commands.inc'].decode()
            self.assertIn('CWRUTOTAL_INFO,2,%u,12000,4096,2,',commands)
            self.assertIn('(unsigned)CWRUTV_COUNT,(unsigned long)HAL_RCC_GetHCLKFreq()',commands)
            self.assertLess(commands.index('#include "cwru_canon_post_commands.inc"'),commands.index('void CWRUTOTAL_Command'))
            self.assertGreater(commands.index('if(CWRUCANON_Command(command,out,cap)) return;'),commands.index('if(command==NULL'))
            self.assertTrue(files['cwru_total.c'].rstrip().endswith(b'#include "cwru_canon_post_math.inc"'))
            self.assertEqual(before,{name:p.sha(source/'CWRU_TotalV2'/name) for name in before})

    def test_prepare_copies_source_only_preserves_D_snapshot_and_blocks_overwrite(self):
        with fixture() as (source,out,project,_,_),redirect_stdout(io.StringIO()):
            result=p.prepare()
            self.assertEqual(result['D_source_sha256_before'],result['D_source_sha256_after'])
            self.assertEqual(result['previous_configurations'],17);self.assertEqual(result['total_configurations'],18)
            self.assertFalse(result['build_executed']);self.assertFalse(result['D_files_modified'])
            self.assertFalse((project/'Debug').exists());self.assertFalse((project/p.BASE).exists())
            self.assertEqual((project/'Src/main.c').read_bytes(),(source/'Src/main.c').read_bytes())
            self.assertEqual((project/'CWRU_TotalV2/cwru_total_catalog.cpp').read_bytes(),
                (source/'CWRU_TotalV2/cwru_total_catalog.cpp').read_bytes())
            self.assertEqual(len(list((project/p.PRIVATE).iterdir())),11)
            p.verify()
            with self.assertRaises(ValueError):p.prepare()

    def test_modified_or_extra_copy_sources_and_changed_origin_block_build(self):
        for mutation in ('private','extra','origin','ELF'):
            with self.subTest(mutation=mutation),fixture() as (source,_,project,_,_),redirect_stdout(io.StringIO()):
                p.prepare()
                if mutation=='private':put(project/p.PRIVATE/'cwru_total.c','modified')
                elif mutation=='extra':put(project/p.PRIVATE/'unexpected.c','extra code')
                elif mutation=='origin':put(source/'Src/main.c','origin changed')
                else:put(source/p.BASE/(p.NAME+'.elf'),'origin ELF changed')
                with self.assertRaises(ValueError):p.verify()

    def test_headless_language_metadata_exception_does_not_allow_algorithm_edits(self):
        with fixture() as (_,_,project,_,_),redirect_stdout(io.StringIO()):
            p.prepare();put(project/'.settings/language.settings.xml','<new-generated-metadata/>')
            with self.assertRaises(ValueError):p.verify()
            p.verify(after_build=True)
            put(project/'Src/main.c','changed algorithm')
            with self.assertRaises(ValueError):p.verify(after_build=True)


class ResourceAndPathGuards(unittest.TestCase):
    def test_local_resource_seal_count_and_header_are_checked(self):
        for mutation in ('byte','count','header','traversal'):
            with self.subTest(mutation=mutation),fixture() as (_,_,_,resources,_):
                if mutation=='byte':put(resources/'cwru_total_catalog.cpp','tampered')
                elif mutation=='count':catalog_files(resources,count=9)
                elif mutation=='header':catalog_files(resources,header=32)
                else:
                    manifest=resources/'MANIFEST_RECURSOS_SHA256.json'
                    entries=p.read(manifest);entries['../escape.txt']='0'*64
                    put(manifest,json.dumps(entries));put(manifest.with_suffix('.sha256'),p.sha(manifest)+'  '+manifest.name+'\n')
                with self.assertRaises(ValueError):p.authenticate_resources(resources)

    def test_bounded_paths_and_exclusive_copy_prevent_escape_or_overwrite(self):
        with fixture() as (source,out,_,_,_):
            for value in ('../escape','.',str(source.resolve())):
                with self.subTest(value=value),self.assertRaises(ValueError):p.confined(out,value)
            target=out/'copy.c';p.copy_exclusive(source/'Src/main.c',target);before=target.read_bytes()
            with self.assertRaises(ValueError):p.copy_exclusive(source/'Src/main.c',target)
            self.assertEqual(target.read_bytes(),before)


class BoundedBuildSimulation(unittest.TestCase):
    def test_timeout_limits_are_fixed_and_checked_before_launch(self):
        for timeout in (True,0,59,1801,3600,'1800'):
            with self.subTest(timeout=timeout),patch.object(p,'verify') as verify,\
                 patch.object(p,'run_bounded') as launch,self.assertRaises(ValueError):
                p.build(timeout);verify.assert_not_called();launch.assert_not_called()

    def test_only_one_isolated_configuration_is_built_and_no_second_attempt_is_allowed(self):
        with fixture() as (_,out,project,_,_),redirect_stdout(io.StringIO()):
            p.prepare()
            def build(arguments,environment,logpath,timeout):
                self.assertEqual(timeout,1800)
                self.assertEqual(arguments[-2:],['-cleanBuild',p.NAME+'/'+p.CONFIG])
                self.assertIn(str(out/'workspace_aislado'),arguments)
                self.assertIn(str(project),arguments)
                put(logpath,'Build Finished. 0 errors, 0 warnings.\n')
                put(project/p.CONFIG/(p.NAME+'.elf'),'synthetic ELF; machine audit pending')
                put(project/p.CONFIG/(p.NAME+'.map'),'synthetic map')
                return dict(returncode=0,timed_out=False,error=None,seconds=1.)
            with patch.object(p,'toolchain',return_value=(out/'tools',out/'make')),\
                 patch.object(p,'run_bounded',side_effect=build) as launch:
                self.assertEqual(p.build(),0)
                report=p.read(out/'COMPILACION_CANON42.json')
                self.assertEqual(report['status'],'PASS_BUILD_PENDING_ELF_REVIEW_AND_ARM')
                self.assertTrue(report['D_source_unchanged_verified'])
                self.assertTrue(report['ELF_machine_and_memory_audit_pending'])
                with self.assertRaises(ValueError):p.build()
                launch.assert_called_once()

    def test_failed_build_is_preserved_and_never_retried(self):
        with fixture() as (_,out,_,_,_),redirect_stdout(io.StringIO()):
            p.prepare()
            def failed(arguments,environment,logpath,timeout):
                put(logpath,'synthetic.c: error: simulated build failure\n')
                return dict(returncode=1,timed_out=False,error=None,seconds=.1)
            with patch.object(p,'toolchain',return_value=(out/'tools',out/'make')),patch.object(p,'run_bounded',side_effect=failed) as launch:
                self.assertEqual(p.build(),1)
                self.assertEqual(p.read(out/'COMPILACION_CANON42.json')['status'],'FAIL_BUILD_REQUIRES_AUDIT')
                before=(out/'COMPILACION_CANON42.json').read_bytes()
                with self.assertRaises(ValueError):p.build()
                self.assertEqual((out/'COMPILACION_CANON42.json').read_bytes(),before);launch.assert_called_once()

    def test_origin_change_after_integrity_check_cannot_leave_a_pass_report(self):
        with fixture() as (source,out,project,_,_),redirect_stdout(io.StringIO()):
            p.prepare()
            real_verify=p.verify
            def verify_then_change_origin(after_build=False):
                result=real_verify(after_build=after_build)
                if after_build:
                    put(source/'Src/main.c','/* external origin change after integrity check */\n')
                return result
            def clean_build(arguments,environment,logpath,timeout):
                put(logpath,'Build Finished. 0 errors, 0 warnings.\n')
                put(project/p.CONFIG/(p.NAME+'.elf'),'synthetic ELF; not a hardware build')
                put(project/p.CONFIG/(p.NAME+'.map'),'synthetic map')
                return dict(returncode=0,timed_out=False,error=None,seconds=.1)
            with patch.object(p,'toolchain',return_value=(out/'tools',out/'make')),\
                 patch.object(p,'run_bounded',side_effect=clean_build) as launch,\
                 patch.object(p,'verify',side_effect=verify_then_change_origin):
                self.assertEqual(p.build(),1)
            report=p.read(out/'COMPILACION_CANON42.json')
            self.assertTrue(report['zero_errors_and_warnings'])
            self.assertTrue(report['integrity_after_build'])
            self.assertFalse(report['D_source_unchanged_verified'])
            self.assertEqual(report['status'],'FAIL_BUILD_REQUIRES_AUDIT')
            self.assertNotEqual(report['D_source_sha256_after_build'],
                p.read(out/'PREPARACION_PROYECTO.json')['D_source_sha256_before'])
            launch.assert_called_once()

    def test_background_process_is_hidden_and_watchdog_targets_only_its_mock_child_tree(self):
        with fixture() as (_,out,_,_,_),redirect_stdout(io.StringIO()):
            process=Mock();process.pid=12345;process.stdout=iter(())
            process.poll.return_value=None;process.wait.return_value=-9
            with patch.object(p.subprocess,'Popen',return_value=process) as launch,\
                 patch.object(p.subprocess,'run') as kill,patch.object(p.time,'monotonic',side_effect=[0.,1801.,1801.1]):
                result=p.run_bounded(['synthetic-IDE'],{},out/'watchdog.log',1800)
            self.assertTrue(result['timed_out']);self.assertEqual(result['error'],'TIMEOUT')
            self.assertEqual(launch.call_args.kwargs['creationflags'],getattr(p.subprocess,'CREATE_NO_WINDOW',0))
            self.assertEqual(kill.call_args.args[0],['taskkill','/PID','12345','/T','/F'])
            self.assertTrue((out/'watchdog.log').is_file())


if __name__=='__main__':unittest.main(verbosity=2)
