"""Pruebas de guardas de instalacion; solo fixtures temporales, nunca D:."""
import hashlib
import json
from contextlib import contextmanager
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import instalar_total_sr_v2 as m


@contextmanager
def fixture():
    base = (m.ROOT/'tmp').resolve()
    base.mkdir(exist_ok=True)
    path = Path(tempfile.mkdtemp(prefix='sr_install_test_', dir=base)).resolve()
    try:
        yield path
    finally:
        # Verificacion del destino absoluto antes de borrar solo este fixture.
        m.require(path.parent == base and path.name.startswith('sr_install_test_')
                  and path.is_relative_to(base), 'Fixture fuera del directorio temporal previsto')
        shutil.rmtree(path)


class InstallationGuards(unittest.TestCase):
    def test_confines_targets_and_rejects_traversal_and_absolute(self):
        with fixture() as tmp:
            base = Path(tmp)
            self.assertEqual(m.confined(base, 'nested/file'), (base/'nested/file').resolve())
            for candidate in ('../outside', '', str(base/'absolute')):
                with self.assertRaises(RuntimeError):
                    m.confined(base, candidate)

    def test_launch_rejects_either_build_switch_or_wrong_elf(self):
        source = m.OUT/'archivos_para_D'/m.SAFE_LAUNCH
        original = source.read_bytes()
        root = ET.fromstring(original)
        attrs = {n.get('key'): n for n in root}
        cid = attrs['org.eclipse.cdt.launch.PROJECT_BUILD_CONFIG_ID_ATTR'].get('value')
        self.assertTrue(m.validate_launch(original, cid)['build_before_launch_disabled'])
        for key, value in ((m.BUILD_KEY, '2'), ('org.eclipse.cdt.launch.PROGRAM_NAME', 'wrong.elf')):
            mutated = ET.fromstring(original)
            next(n for n in mutated if n.get('key') == key).set('value', value)
            with self.assertRaises(RuntimeError):
                m.validate_launch(ET.tostring(mutated), cid)
        mutated = ET.fromstring(original)
        load = next(n for n in mutated if n.get('key') == m.LOAD_KEY)
        data = json.loads(load.get('value'))
        data['fItems'][0]['fPerformBuild'] = True
        load.set('value', json.dumps(data))
        with self.assertRaises(RuntimeError):
            m.validate_launch(ET.tostring(mutated), cid)

    def test_metadata_accepts_only_sr_append_and_detector_cache(self):
        before = m.OUT/'respaldo_antes_single_rounding/.settings/language.settings.xml'
        after = m.OUT/'copia_aislada'/m.NAME/'.settings/language.settings.xml'
        tree = ET.parse(after)
        added = next(n for n in tree.getroot() if n.get('name') == m.CONFIG)
        m.metadata_append(before, after, added.get('id'))
        next(iter(added)).set('unapproved', 'yes')
        with fixture() as tmp:
            mutated = Path(tmp)/'metadata.xml'
            tree.write(mutated, encoding='utf-8')
            with self.assertRaises(RuntimeError):
                m.metadata_append(before, mutated, added.get('id'))

    def test_copy_validates_hash_and_replaces_config_last(self):
        with fixture() as tmp:
            base = Path(tmp)
            source, staged = base/'dest', base/'stage'
            source.mkdir(); staged.mkdir()
            (source/'.cproject').write_bytes(b'old')
            (source/'untouched').write_bytes(b'control')
            payloads = {'new/object': b'object', '.cproject': b'newcfg'}
            targets = []
            for rel, data in payloads.items():
                file = staged/rel; file.parent.mkdir(parents=True, exist_ok=True); file.write_bytes(data)
                targets.append(dict(relative=rel, sha256=hashlib.sha256(data).hexdigest(), overwrite=rel=='.cproject'))
            before = {'.cproject': m.sha(source/'.cproject')}
            actual_replace = m.os.replace
            def guarded_replace(temp, dest):
                self.assertEqual((source/'new/object').read_bytes(), b'object')
                self.assertEqual((source/'untouched').read_bytes(), b'control')
                return actual_replace(temp, dest)
            with patch.object(m, 'SOURCE', source), patch.object(m, 'STAGED', staged), patch.object(m.os, 'replace', guarded_replace):
                self.assertEqual(m.write_targets(targets, before), ['new/object', '.cproject'])
            self.assertEqual((source/'.cproject').read_bytes(), b'newcfg')
            self.assertEqual((source/'untouched').read_bytes(), b'control')

    def test_existing_new_target_stops_before_config_replacement(self):
        with fixture() as tmp:
            base = Path(tmp)
            source, staged = base/'dest', base/'stage'
            source.mkdir(); staged.mkdir()
            (source/'.cproject').write_bytes(b'old')
            (source/'new').write_bytes(b'existing')
            targets = []
            for rel, data in (('new', b'replacement'), ('.cproject', b'newcfg')):
                (staged/rel).write_bytes(data)
                targets.append(dict(relative=rel, sha256=hashlib.sha256(data).hexdigest(), overwrite=rel=='.cproject'))
            with patch.object(m, 'SOURCE', source), patch.object(m, 'STAGED', staged):
                with self.assertRaises(RuntimeError):
                    m.write_targets(targets, {'.cproject': m.sha(source/'.cproject')})
            self.assertEqual((source/'.cproject').read_bytes(), b'old')
            self.assertEqual((source/'new').read_bytes(), b'existing')


if __name__ == '__main__':
    unittest.main(verbosity=2)
