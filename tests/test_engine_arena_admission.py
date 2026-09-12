"""UMA admission must not treat reclaimable cache as immediately free DRAM."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from engine.base.arena import GIB, prepare_allocation, touch_pages


def meminfo(free, available):
    return f'MemFree: {free * GIB // 1024} kB\nMemAvailable: {available * GIB // 1024} kB\n'


class ArenaAdmissionTests(unittest.TestCase):
    def test_clean_model_cache_satisfies_admission_without_a_large_temporary_allocation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / 'model.safetensors'
            download = root / 'chunk.incomplete'
            other = root / 'other.db'
            for p in (model, download, other):
                p.write_bytes(bytes(range(256)) * 32)
            advised = []
            def advise(fd, *args):
                advised.append(Path('/proc/self/fd/' + str(fd)).resolve().name)
            with patch.object(Path, 'read_text', side_effect=[meminfo(80, 96), meminfo(92, 96)]), \
                 patch('os.posix_fadvise', side_effect=advise):
                report = prepare_allocation(74 * GIB, [], 16 * GIB, lambda: 120 * GIB,
                    reclaim=lambda n: self.fail('no anonymous pressure is needed'), cache_roots=(root,))
            self.assertEqual(set(advised), {'model.safetensors', 'chunk.incomplete'})
            self.assertEqual(report['cache_files'], 2)
            self.assertEqual(report['reclaimed'], 0)
            for p in (model, download, other):
                self.assertEqual(p.read_bytes(), bytes(range(256)) * 32)

    def test_the_transient_fault_is_bounded_by_the_box_not_by_the_engines_headroom(self):
        """The reclaim faults `need` for an instant and gives it back; the workspace and the OS
        reserve inside `headroom` are not allocated yet and nothing is serving behind them. Asking
        the node for `need + headroom` charged it twice, and once #710 raised the OS reserve to the
        box's own floor + 2 the second copy grew past what these nodes have: 2026-09-12, a boot
        refused at MemAvailable 92.74 GiB for an allocation of 60.99 that fits."""
        faulted = []
        # 60.99 + 20 = 80.99 needed, 78.47 immediately free: a 2.52 GiB shortfall out of 92.74
        # available. The old bound wanted 100.99 available; the box's SIGTERM line wants 86.99.
        with patch('engine.base.runtime_memory.oom_floor', return_value=(6 * GIB, 9 * GIB // 2)), \
             patch.object(Path, 'read_text', side_effect=[meminfo(78, 93), meminfo(82, 89)]):
            report = prepare_allocation(int(60.99 * GIB), [], 20 * GIB, lambda: 120 * GIB,
                                        reclaim=lambda n: faulted.append(n) or n)
        self.assertEqual(len(faulted), 1, "it reclaims instead of refusing")
        self.assertAlmostEqual(faulted[0] / GIB, 80.99, places=1, msg="it faults `need`, MemFree first")
        self.assertGreater(report['immediately_free'], report['allocation'])

    def test_a_fault_that_would_cross_the_boxs_kill_line_is_still_refused(self):
        """Loosening the bound is not removing it: below the SIGTERM line the box starts killing,
        and the engine is target number one."""
        with patch('engine.base.runtime_memory.oom_floor', return_value=(6 * GIB, 9 * GIB // 2)), \
             patch.object(Path, 'read_text', return_value=meminfo(8, 70)):
            with self.assertRaisesRegex(MemoryError, "SIGTERM line"):
                prepare_allocation(60 * GIB, [], 20 * GIB, lambda: 120 * GIB, reclaim=lambda n: n)

    def test_large_available_value_cannot_hide_low_immediately_free_memory(self):
        # MemAvailable 100 but only 8 free and the reclaim disabled: admission refuses
        with patch.object(Path, 'read_text', return_value=meminfo(8, 100)):
            with self.assertRaisesRegex(MemoryError, 'immediately free'):
                prepare_allocation(48 * GIB, [], 16 * GIB, lambda: 100 * GIB, reclaim=None)

    def test_reclaim_preserves_weight_bytes_and_device_free_also_limits_admission(self):
        with tempfile.TemporaryDirectory() as directory:
            weight = Path(directory) / 'rank.safetensors'
            weight.write_bytes(bytes(range(256)) * 32)
            before = weight.read_bytes()
            with patch.object(Path, 'read_text', return_value=meminfo(100, 110)):
                with self.assertRaises(MemoryError):
                    prepare_allocation(48 * GIB, [weight], 16 * GIB, lambda: 60 * GIB)
                report = prepare_allocation(48 * GIB, [weight], 16 * GIB, lambda: 90 * GIB)
            self.assertEqual(report['immediately_free'], 90 * GIB)
            self.assertEqual(report['reclaimed'], 0)
            self.assertEqual(weight.read_bytes(), before)

    def test_reclaim_faults_the_desired_extent_instead_of_reusing_free_pages(self):
        # srv4 on 09-12: a shortfall-only pump reused free pages, evicted no
        # cache, and repeatedly failed restart. The full extent forces eviction.
        touched = []

        def pump(nbytes):
            touched.append(nbytes)
            states.append(meminfo(100 if nbytes > 54*GIB else 54, 99))
            return nbytes

        states = [meminfo(54, 99)]
        with patch.object(Path, 'read_text', side_effect=lambda *a, **k: states[-1]):
            report = prepare_allocation(int(55.4 * GIB), [], 16 * GIB, lambda: 120 * GIB, reclaim=pump)
        self.assertEqual(touched, [int(55.4 * GIB) + 16 * GIB])
        self.assertEqual(report['reclaimed'], touched[0])
        self.assertEqual(report['immediately_free'], 100 * GIB)

    def test_reclaim_is_refused_when_it_would_starve_the_box(self):
        # The temporary extent must leave the BOX's line plus a margin -- not the engine's full
        # headroom, which is a post-boot requirement the check after the reclaim enforces. (66, 80)
        # used to be here: it leaves 8.6 GiB of MemAvailable during the fault, above the 6 GiB
        # SIGTERM line and above the 8 GiB floor, and refusing it is what stopped production
        # booting on 2026-09-12 once the OS reserve grew from 4 GiB to 8 (#710).
        for free, available in ((30, 60), (10, 72), (66, 76)):
            with self.subTest(free=free, available=available), \
                 patch.object(Path, 'read_text', return_value=meminfo(free, available)):
                with self.assertRaisesRegex(MemoryError, 'cannot be reclaimed'):
                    prepare_allocation(int(55.4 * GIB), [], 16 * GIB, lambda: 120 * GIB,
                                       reclaim=lambda n: (_ for _ in ()).throw(AssertionError('pump must not run')))

    def test_a_pump_that_did_not_free_enough_still_fails_closed(self):
        states = [meminfo(54, 99)]

        def pump(nbytes):
            states.append(meminfo(60, 99))                # the kernel swapped instead of dropping cache
            return nbytes

        with patch.object(Path, 'read_text', side_effect=lambda *a, **k: states[-1]):
            with self.assertRaisesRegex(MemoryError, 'after reclaiming'):
                prepare_allocation(int(55.4 * GIB), [], 16 * GIB, lambda: 120 * GIB, reclaim=pump)

    def test_touch_pages_rounds_to_pages_and_returns_the_memory(self):
        self.assertEqual(touch_pages(0), 0)
        self.assertEqual(touch_pages(1), touch_pages(4096))
        self.assertEqual(touch_pages(3 << 20), 3 << 20)


if __name__ == '__main__':
    unittest.main()
