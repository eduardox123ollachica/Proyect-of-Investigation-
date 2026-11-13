"""Pruebas de integridad y carga simulada: nunca contactar hardware."""
from __future__ import annotations

from contextlib import contextmanager, redirect_stdout
import io
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

import cargar_firmware_canon42 as c


IDENTITY = (f'ST-LINK SN  : {c.PROBE_SN}\nBoard : NUCLEO-F446RE\n'
    'Device ID : 0x421\nDevice name : STM32F446xx\nNVM size : 512 KBytes\n')


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


@contextmanager
def fixture():
    base = c.ROOT/'tmp'
    base.mkdir(exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix='carga_canon42_test_', dir=base)).resolve()
    if root == base.resolve() or not root.is_relative_to(base.resolve()):
        raise RuntimeError('Fixture fuera del workspace')
    out = root/'entrega'; out.mkdir()
    own = root/'cargar.py'; own.write_text('# fixture\n', encoding='utf-8')
    cli = root/'programmer.exe'; cli.write_bytes(b'fake programmer, never execute')
    elf = out/'firmware.elf'; elf.write_bytes(b'fixture ELF, never flash')
    mapfile = out/'firmware.map'; mapfile.write_text('fixture', encoding='utf-8')
    build = out/'COMPILACION_CANON42.json'
    put(build, dict(status='PASS_BUILD_PENDING_ELF_REVIEW_AND_ARM', returncode=0,
        timed_out=False, zero_errors_and_warnings=True, integrity_after_build=True,
        D_source_unchanged_verified=True, elf_sha256=c.sha(elf), map_sha256=c.sha(mapfile)))
    audit = out/'AUDITORIA_ELF_CANON42.json'
    put(audit, dict(status='PASS_ELF_CANON42_VERIFIED_PENDING_ARM',
        firmware_elf_sha256=c.sha(elf), build_report_sha256=c.sha(build),
        configuration=c.CONFIG, catalog_sha256=c.CATALOG_SHA, model_count=8,
        zero_errors_and_warnings=True, source_integrity_after_build=True))
    manifest = out/'MANIFEST_CLIENTES_CANON42_SHA256.json'
    other = [out/'MANIFEST_PREPARACION_PROYECTO_SHA256.json',
        out/'recursos/MANIFEST_RECURSOS_SHA256.json',out/'bancos/MANIFEST_BANCOS_SHA256.json']
    for path in other: put(path, {'fixture': True})
    files = [own,audit,build,elf,mapfile,*other]
    def seal():
        put(manifest,{p.relative_to(root).as_posix():c.sha(p) for p in files})
        manifest.with_suffix('.sha256').write_text(c.sha(manifest)+'  '+manifest.name+'\n',encoding='ascii')
    seal()
    values=dict(ROOT=root,OUT=out,ELF=elf,MAP=mapfile,AUDIT=audit,BUILD=build,
        MANIFEST=manifest,MARKER=out/'INTENTO_CARGA_CANON42.json',CLI=cli,
        ELF_SHA=c.sha(elf),BUILD_SHA=c.sha(build),CLI_SHA=c.sha(cli),__file__=str(own))
    try:
        with patch.multiple(c,**values):yield dict(root=root,out=out,seal=seal,files=files)
    finally:
        if root != base.resolve() and root.is_relative_to(base.resolve()):shutil.rmtree(root)


def result(text=IDENTITY, **changes):
    row=dict(command=[],process_created=True,returncode=0,timed_out=False,error=None,
        seconds=.01,text=text,cleanup_error=None)
    row.update(changes)
    return row


class OfflineChecks(unittest.TestCase):
    def test_check_reads_seals_without_processes_files_or_hardware(self):
        with fixture() as f, patch.object(c.subprocess,'Popen') as process:
            before={str(p) for p in f['root'].rglob('*')}
            proof=c.check()
            self.assertEqual(proof['status'],'CANON42_OFFLINE_CHECK_PASS_NOT_PROGRAMMED')
            self.assertFalse(proof['hardware_contacted'])
            process.assert_not_called()
            self.assertEqual(before,{str(p) for p in f['root'].rglob('*')})

    def test_modified_or_unsealed_ELF_and_tool_are_rejected(self):
        with fixture():
            c.ELF.write_bytes(b'tampered')
            with self.assertRaisesRegex(ValueError,'sellado modificado'):c.check()
        with fixture():
            c.CLI.write_bytes(b'other tool')
            with self.assertRaisesRegex(ValueError,'Programador instalado distinto'):c.check()

    def test_failed_audit_or_wrong_catalog_cannot_be_resealed_into_approval(self):
        for key,value in [('status','FAIL'),('model_count',32),('catalog_sha256','0'*64)]:
            with self.subTest(key=key),fixture() as f:
                audit=c.read(c.AUDIT);audit[key]=value;put(c.AUDIT,audit);f['seal']()
                with self.assertRaisesRegex(ValueError,'Auditoría ELF no aprobada'):c.check()

    def test_manifest_escape_and_missing_required_entry_are_rejected(self):
        with fixture() as f:
            data=c.read(c.MANIFEST);data['../outside']='0'*64;put(c.MANIFEST,data)
            c.MANIFEST.with_suffix('.sha256').write_text(c.sha(c.MANIFEST)+'  '+c.MANIFEST.name+'\n')
            with self.assertRaisesRegex(ValueError,'Ruta de manifiesto insegura'):c.check()
        with fixture():
            data=c.read(c.MANIFEST);data.pop(c.relative(c.ELF));put(c.MANIFEST,data)
            c.MANIFEST.with_suffix('.sha256').write_text(c.sha(c.MANIFEST)+'  '+c.MANIFEST.name+'\n')
            with self.assertRaisesRegex(ValueError,'Sello de clientes incompleto'):c.check()


class ProgrammingSimulation(unittest.TestCase):
    def proof(self):return dict(manifest_sha256='a'*64)

    def report(self,f):
        paths=list((f['out']/'programacion_flash').rglob('RESULTADO_CARGA.json'))
        self.assertEqual(len(paths),1)
        return c.read(paths[0])

    def test_fixed_target_verified_program_and_reset_are_one_attempt(self):
        with fixture() as f,patch.object(c,'check',return_value=self.proof()),redirect_stdout(io.StringIO()), \
            patch.object(c,'run_cli',side_effect=[result(),result(IDENTITY+'Download verified successfully\nResetting target\n')]) as run:
            self.assertEqual(c.program(),0)
            self.assertEqual(run.call_count,2)
            first,second=[call.args[0] for call in run.call_args_list]
            self.assertNotIn('-d',first)
            self.assertIn('sn='+c.PROBE_SN,first)
            self.assertIn('sn='+c.PROBE_SN,second)
            self.assertEqual(second[second.index('-d')+1],str(c.ELF))
            self.assertIn('-v',second);self.assertIn('-rst',second)
            self.assertTrue(all(0<call.args[2]<=120 for call in run.call_args_list))
            report=self.report(f)
            self.assertEqual(report['status'],'PASS_CANON42_PROGRAMMED_VERIFIED_RESET_PENDING_UART_GATE')
            self.assertTrue(report['download_verification_marker'])
            self.assertTrue(report['reset_command_succeeded'])
            self.assertFalse(report['runtime_boot_confirmed'])
            self.assertFalse(report['COM_opened'])
            self.assertEqual(report['request_marker_sha256'],c.sha(c.MARKER))
            folder=Path(report['evidence_path'])
            manifest=folder/'MANIFEST_CARGA_SHA256.json'
            self.assertTrue(manifest.is_file())
            self.assertEqual(manifest.with_suffix('.sha256').read_text().split(),[c.sha(manifest),manifest.name])
            entries=c.read(manifest)
            self.assertIn('RESULTADO_CARGA.json',entries)
            self.assertNotIn(manifest.name,entries)
            self.assertTrue(all(c.sha(folder/name)==digest for name,digest in entries.items()))
            with self.assertRaisesRegex(ValueError,'Ya existe un intento'):c.program()
            self.assertEqual(run.call_count,2)

    def test_wrong_target_identity_blocks_the_program_command(self):
        for text in (IDENTITY.replace('0x421','0x413'),IDENTITY.replace(c.PROBE_SN,'OTHER'),
            IDENTITY.replace('NUCLEO-F446RE','OTHER'),IDENTITY.replace('512','256')):
            with self.subTest(text=text),fixture() as f,patch.object(c,'check',return_value=self.proof()), \
                redirect_stdout(io.StringIO()),patch.object(c,'run_cli',return_value=result(text)) as run:
                self.assertEqual(c.program(),1)
                self.assertEqual(run.call_count,1)
                self.assertEqual(self.report(f)['flash_state'],'NO_PROGRAM_COMMAND_STARTED')

    def test_missing_verification_or_failed_program_is_preserved_without_retry(self):
        for second in (result(IDENTITY+'File download complete\n'),
            result(IDENTITY+'Error: failed\n',returncode=1),
            result(IDENTITY,timed_out=True,error='TIMEOUT',returncode=-1)):
            with self.subTest(second=second),fixture() as f,patch.object(c,'check',return_value=self.proof()), \
                redirect_stdout(io.StringIO()),patch.object(c,'run_cli',side_effect=[result(),second]) as run:
                self.assertEqual(c.program(),1)
                self.assertEqual(run.call_count,2)
                report=self.report(f)
                self.assertEqual(report['flash_state'],'MAY_BE_MODIFIED')
                self.assertFalse(report['automatic_retry'])
                self.assertTrue(c.MARKER.exists())
                self.assertTrue((Path(report['evidence_path'])/'MANIFEST_CARGA_SHA256.json').is_file())


class ChildProcessSimulation(unittest.TestCase):
    def test_timeout_stops_mock_child_and_preserves_lossy_utf8_transcript(self):
        with fixture() as f:
            process=Mock();process.poll.return_value=None;process.returncode=-1
            process.communicate.side_effect=[subprocess.TimeoutExpired('mock',1),(b'partial \xff',None)]
            with patch.object(c.subprocess,'Popen',return_value=process) as start:
                row=c.run_cli(['mock-only'],f['out']/'timeout.txt',1)
            process.kill.assert_called_once()
            self.assertTrue(row['timed_out']);self.assertEqual(row['error'],'TIMEOUT')
            self.assertIn('\ufffd',row['text'])
            self.assertEqual(start.call_args.kwargs['creationflags'],getattr(c.subprocess,'CREATE_NO_WINDOW',0))

    def test_spawn_failure_or_no_remaining_budget_never_launches_a_retry(self):
        with fixture() as f,patch.object(c.subprocess,'Popen',side_effect=OSError('mock failure')) as start:
            row=c.run_cli(['mock-only'],f['out']/'spawn.txt',1)
            self.assertFalse(row['process_created']);self.assertIn('OSError',row['error'])
            self.assertEqual(start.call_count,1)
            with self.assertRaisesRegex(ValueError,'Presupuesto agotado'):c.run_cli(['mock-only'],f['out']/'none.txt',0)
            self.assertEqual(start.call_count,1)


if __name__=='__main__':unittest.main()
