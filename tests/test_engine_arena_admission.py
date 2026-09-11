"""UMA admission must not treat reclaimable cache as immediately free DRAM."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from engine.base.arena import GIB, prepare_allocation


class ArenaAdmissionTests(unittest.TestCase):
    def test_large_available_value_cannot_hide_low_immediately_free_memory(self):
        memory='MemFree: 8388608 kB\nMemAvailable: 104857600 kB\n'
        with patch.object(Path,'read_text',return_value=memory):
            with self.assertRaisesRegex(MemoryError,'immediately free'):
                prepare_allocation(48*GIB,[],16*GIB,lambda: 100*GIB)

    def test_reclaim_preserves_weight_bytes_and_device_free_also_limits_admission(self):
        with tempfile.TemporaryDirectory() as directory:
            weight=Path(directory)/'rank.safetensors'
            weight.write_bytes(bytes(range(256))*32)
            before=weight.read_bytes()
            memory='MemFree: 104857600 kB\nMemAvailable: 115343360 kB\n'
            with patch.object(Path,'read_text',return_value=memory):
                with self.assertRaises(MemoryError):
                    prepare_allocation(48*GIB,[weight],16*GIB,lambda: 60*GIB)
                report=prepare_allocation(48*GIB,[weight],16*GIB,lambda: 90*GIB)
            self.assertEqual(report['immediately_free'],90*GIB)
            self.assertEqual(weight.read_bytes(),before)


if __name__=='__main__':unittest.main()
