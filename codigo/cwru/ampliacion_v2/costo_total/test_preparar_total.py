"""Pruebas de configuracion real en memoria; no escriben en D: ni compilan."""
import copy
import unittest
import xml.etree.ElementTree as ET

import preparar_total as p


class ConfigurationTests(unittest.TestCase):
    def test_configuration_isolation_and_flags(self):
        root, settings, old = p.configs(p.SOURCE / '.cproject')
        before = {name: ET.tostring(n) for name, n in old.items()}
        new = p.add_configuration(root, settings, old)
        self.assertEqual(len(settings), 16)
        for name, node in old.items():
            self.assertEqual(ET.tostring(node), before[name])
        self.assertIn(b'<?fileVersion 4.0.0?>', p.xml_bytes(root))
        entries = [n.get('name') for n in new.find('.//sourceEntries')]
        self.assertIn('CWRU_TotalV2', entries)
        self.assertIn('CWRU_Masks', entries)
        self.assertFalse(set(('CWRU_DNN', 'CWRU_Input', 'CWRU_Detector', 'CWRU_Pipeline')) & set(entries))
        for tool in new.iter('tool'):
            kind = tool.get('superClass', '')
            if kind.endswith(('tool.c.compiler', 'tool.cpp.compiler')):
                defines = [n.get('value') for n in p.option(tool, '.definedsymbols')]
                self.assertIn('CWRU_TOTALV2_TEST=1', defines)
                self.assertIn('CWRU_MASK_BENCH=1', defines)
                self.assertIn('CWRU_ENTROPY_MODE=2', defines)
                self.assertIn('CWRU_CACHE_SPECTRAL_POWER=1', defines)
                self.assertFalse(any(n.split('=')[0] in ('CWRU_DNN_TEST', 'CWRU_PIPELINE_TEST') for n in defines))
                self.assertIn('../CWRU_TotalV2', [n.get('value') for n in p.option(tool, '.includepaths')])
            if kind.endswith('tool.cpp.compiler'):
                self.assertEqual(p.option(tool, '.otherflags').get('value'), '-std=gnu++17 -fno-threadsafe-statics')
            if kind.endswith('tool.cpp.linker'):
                self.assertEqual([n.get('value') for n in p.option(tool, '.libraries')], ['tensorflow-microlite', 'm'])
                self.assertEqual([n.get('value') for n in p.option(tool, '.directories')], ['../Lib'])

    def test_config_rejects_changed_origin(self):
        root, settings, old = p.configs(p.SOURCE / '.cproject')
        damaged = copy.copy(old)
        damaged.pop(p.CONTROL)
        with self.assertRaises(RuntimeError):
            p.add_configuration(root, settings, damaged)

    def test_uart_guard_and_route_once(self):
        original = (p.SOURCE / p.MODIFIED[1]).read_text(encoding='utf-8')
        patched = p.patch_uart(original)
        self.assertEqual(patched.count('CWRUTOTAL_Command('), 1)
        self.assertLess(patched.index('CWRUTOTAL_Command('), patched.index('CWRUMASK_Command('))
        with self.assertRaises(RuntimeError):
            p.patch_uart(patched)

    def test_receiver_three_invalidations(self):
        original = (p.SOURCE / p.MODIFIED[2]).read_text(encoding='utf-8')
        patched = p.patch_receiver(original)
        self.assertEqual(patched.count('CWRUTOTAL_Invalidate();'), 3)
        self.assertEqual(patched.count('#include "cwru_total_commands.inc"'), 1)
        self.assertEqual(patched.count('CWRUMASK_Invalidate();'), 3)
        with self.assertRaises(RuntimeError):
            p.patch_receiver(patched)

    def test_flag_gate_excludes_fast_math_and_profiling(self):
        line = 'arm-none-eabi-gcc -c -O3 -DCWRU_TARGET_HCLK_HZ=180000000U -DCWRU_CACHE_SPECTRAL_POWER=1 -DCWRU_ENTROPY_MODE=2 -DCWRU_TOTALV2_TEST=1 -DCWRU_MASK_BENCH=1 -I../CWRU_TotalV2'
        self.assertTrue(p.flags([line], p.CONFIG))
        self.assertFalse(p.flags([line + ' -ffast-math'], p.CONFIG))
        self.assertFalse(p.flags([line + ' -DCWRU_PROFILE_EXTRACTOR=1'], p.CONFIG))
        self.assertFalse(p.flags([], p.CONFIG))


if __name__ == '__main__':
    unittest.main()
