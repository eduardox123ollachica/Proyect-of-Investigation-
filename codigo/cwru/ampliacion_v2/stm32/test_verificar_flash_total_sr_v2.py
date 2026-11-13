"""Pruebas sinteticas del verificador Flash; nunca conectan ST-LINK ni COM."""
from contextlib import contextmanager
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import verificar_flash_total_sr_v2 as m


@contextmanager
def fixture():
    base = (m.ROOT/'tmp').resolve()
    base.mkdir(exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix='sr_flash_guard_', dir=base)).resolve()
    try:
        yield folder
    finally:
        m.require(folder.parent == base and folder.name.startswith('sr_flash_guard_')
                  and folder.is_relative_to(base), 'Fixture fuera del temporal autorizado')
        shutil.rmtree(folder)


def synthetic_elf():
    payload = b'CODE'+b'INIT'
    entries = [dict(type=1, physical=m.FLASH_START+4, address=m.FLASH_START+4, size=4, offset=0),
               dict(type=1, physical=m.FLASH_START+16, address=0x20000000, size=4, offset=4),
               dict(type=1, physical=m.FLASH_START+20, address=0x20000004, size=0, offset=0)]
    return SimpleNamespace(segments=entries, range=lambda start, count: payload[start:start+count])


class FlashVerifierGuards(unittest.TestCase):
    def test_compares_physical_flash_and_initialized_RAM_while_ignoring_unspecified_bytes(self):
        segments = m.flash_segments(synthetic_elf())
        data = bytearray(b'\x35'*m.FLASH_SIZE)
        data[4:8], data[16:20] = b'CODE', b'INIT'
        result = m.compare_dump(bytes(data), segments)
        self.assertTrue(result['match'])
        self.assertEqual(result['compared_bytes'], 8)
        self.assertEqual(result['loaded_segments'][1]['virtual_address'], '0x20000000')
        self.assertEqual(result['bytes_not_specified_by_ELF'], m.FLASH_SIZE-8)
        self.assertFalse(result['gaps_and_tail_required_to_equal_FF'])

    def test_mismatch_and_incomplete_dump_are_rejected(self):
        segments = m.flash_segments(synthetic_elf())
        data = bytearray(b'\0'*m.FLASH_SIZE)
        data[4:8], data[16:20] = b'CODE', b'INIT'
        data[17] = ord('X')
        result = m.compare_dump(bytes(data), segments)
        self.assertFalse(result['match'])
        self.assertEqual(result['mismatch_count'], 1)
        self.assertEqual(result['first_mismatches'][0]['address'], '0x08000011')
        with self.assertRaises(ValueError):
            m.compare_dump(bytes(data[:-1]), segments)

    def test_cross_boundary_and_overlapping_segments_are_rejected(self):
        elf = synthetic_elf()
        elf.segments[0]['physical'] = m.FLASH_START-1
        with self.assertRaises(ValueError):
            m.flash_segments(elf)
        elf = synthetic_elf()
        elf.segments[1]['physical'] = m.FLASH_START+6
        with self.assertRaises(ValueError):
            m.flash_segments(elf)

    def test_command_is_read_only_fixed_and_confined(self):
        with fixture() as tmp, patch.object(m, 'OUT', tmp):
            folder = tmp/'lectura_flash/20261026T235959_123456Z'
            command = m.readback_command(folder)
            self.assertEqual(command[1:9], ['-c', 'port=SWD', 'mode=HOTPLUG', 'freq=4000',
                                          'sn='+m.STLINK_SN, '-u', '0x08000000', '0x80000'])
            self.assertEqual(command[10], '-log')
            self.assertFalse(set(command) & {'-w', '-d', '-e', '--erase', '-rst', '-halt', '-rdu'})
            for wrong in (tmp/'outside', tmp/'lectura_flash/../20261026T235959_123456Z',
                          tmp/'lectura_flash/20261026T235959_123456Z/nested'):
                with self.assertRaises(ValueError):
                    m.readback_command(wrong)

    def test_programmer_identity_and_completed_read_are_required(self):
        log = ('STM32CubeProgrammer v2.23.0\nST-LINK SN : '+m.STLINK_SN+
               '\nBoard : NUCLEO-F446RE\nDevice ID : 0x421\nDevice name : STM32F446xx\nData read successfully\n')
        self.assertTrue(m.check_programmer_log(log, 'detailed log')['read_success_confirmed'])
        for altered in (log.replace(m.STLINK_SN, 'wrong'), log.replace('0x421', '0x450'),
                        log.replace('Data read successfully', 'File download complete')):
            with self.assertRaises(ValueError):
                m.check_programmer_log(altered, 'detailed log')
        with self.assertRaises(ValueError):
            m.check_programmer_log(log, '')

    def test_timeout_terminates_without_retries_and_preserves_partial_output(self):
        class FakeProcess:
            def __init__(self):
                self.calls = []
                self.returncode = None
                self.terminated = False
            def communicate(self, timeout):
                self.calls.append(timeout)
                if len(self.calls) == 1:
                    raise subprocess.TimeoutExpired('synthetic', timeout)
                self.returncode = -15
                return b'partial stdout', b'partial stderr'
            def terminate(self):
                self.terminated = True
            def kill(self):
                raise AssertionError('Fake terminate should already complete')
        fake = FakeProcess()
        with patch.object(m.subprocess, 'Popen', return_value=fake) as launch:
            result = m.run_bounded(['synthetic-read-command'], 120)
        self.assertTrue(fake.terminated)
        self.assertTrue(result['timed_out'])
        self.assertEqual(fake.calls, [115, 3])
        self.assertEqual(result['stdout'], b'partial stdout')
        self.assertEqual(launch.call_count, 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
