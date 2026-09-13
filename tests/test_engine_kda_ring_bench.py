"""The KDA ring timing lane (probes/engine_kda_ring_bench.py): what the CPU can pin -- the lane is one the admitted
kernel check dispatches, and the byte and rate arithmetic its answer is read by."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class RingBenchTests(unittest.TestCase):
    def test_the_kernel_check_admits_and_dispatches_the_lane(self):
        source = (ROOT / "probes/engine_kernel_check.py").read_text()
        self.assertIn('"kda_ring_bench"}, selected', source)
        self.assertIn('if "kda_ring_bench" in selected:', source)
        self.assertIn("from probes.engine_kda_ring_bench import bench", source)

    def test_bytes_and_rates_match_the_timeline_s_accounting(self):
        import torch
        from probes import engine_kda_ring_bench as bench
        self.assertEqual(bench.state_bytes(torch.float32), 2**20)             # 16 x 128 x 128 x 4 B: one MiB a state
        self.assertEqual(bench.state_bytes(torch.float16), 2**19)
        self.assertEqual(bench.launch_bytes(1, torch.float32), 8 * 2**20)     # seven written, one read
        self.assertEqual(bench.launch_bytes(4, torch.float32), 32 * 2**20)
        row = bench.summary(4, torch.float32, [174.7, 170.0, 180.0], [120.0, 118.0, 122.0])
        self.assertEqual((row["rows"], row["storage"], row["mib"]), (4, "float32", 32.0))
        self.assertAlmostEqual(row["cold_us"], 174.7)
        self.assertAlmostEqual(row["per_row_cold_us"], 174.7 / 4)
        self.assertAlmostEqual(row["cold_gbps"], 32 * 2**20 / 174.7 / 1e3)   # the timeline's 192 GB/s
        self.assertEqual(round(row["cold_gbps"]), 192)
        self.assertEqual(row["samples"], 3)
        # the field the lane allocates stays inside a kernel check's budget beside production
        field_gib = bench.LAYERS * bench.SLOTS_PER_LAYER * bench.CELLS * bench.state_bytes(torch.float32) * 1.5 / 2**30
        self.assertLess(field_gib + bench.TRASH_MIB / 1024, 8)


if __name__ == "__main__":
    unittest.main()
