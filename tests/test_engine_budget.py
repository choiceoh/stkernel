"""The GLM-5.3 budget (D1): every line carries provenance, the remainder is the KV room."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


CONFIG = Path("/home/choiceoh/models/st-glm53-nvidia-tp4-9391/config.json")


@unittest.skipUnless(CONFIG.exists() and importlib.util.find_spec("torch") is not None, "needs the GLM-5.3 config and torch")
class BudgetTests(unittest.TestCase):
    def test_lines_are_all_provenanced_and_the_declared_kv_leaves_room(self):
        from engine.base.budget import ESTIMATED
        from engine.profiles.glm53 import budget
        b = budget.budget(8.73, 4, box_gib=121.6, drafter_dir=None)
        self.assertFalse([l for l in b.lines if l.source == ESTIMATED])
        self.assertTrue(b.is_gate())
        names = [l.name for l in b.lines]
        self.assertIn("weights (this rank, TP=4)", names)
        weights = next(l for l in b.lines if l.name.startswith("weights"))
        self.assertAlmostEqual(weights.gib, 44.353017, places=5)
        self.assertGreater(b.kv_gib, b.kv_declared_gib)                 # the box holds more KV than the boot declares
        self.assertGreater(b.kv_gib - b.kv_declared_gib, 30)
        text = budget.report(b)
        self.assertIn("unassigned +", text)
        self.assertIn("concurrency   4", text)

    def test_a_boot_ledger_puts_the_measured_workspace_peak_on_the_line(self):
        from engine.profiles.glm53 import budget
        with tempfile.TemporaryDirectory() as d:
            ledger = Path(d) / "memory-rank0.json"
            ledger.write_text(json.dumps({"phases": [{"phase": "prefill/6912/0/prepared", "peak_workspace_bytes": 3 << 30},
                                                     {"phase": "target/(4, 6, 4096)/captured", "peak_workspace_bytes": 5 << 30}]}))
            b = budget.budget(8.73, 4, box_gib=121.6, drafter_dir=None, ledger=ledger)
        workspace = next(l for l in b.lines if l.name.startswith("workspace"))
        self.assertEqual(workspace.gib, budget.WORKSPACE_GIB)
        self.assertIn("measured peak 5.00 GiB at target/(4, 6, 4096)/captured", workspace.evidence)

    @unittest.skipUnless(Path('/home/choiceoh/models/GLM-5.3-Flash-DFlash2/config.json').exists(), 'DFlash2 metadata')
    def test_native_kv_shards_remove_replicated_heads_from_every_snapshot(self):
        from engine.profiles.glm53 import budget
        replicated = budget.budget(16, 4, snapshots=96, draft_tp=1)
        native = budget.budget(16, 4, snapshots=96, draft_tp=4)
        def snapshots(b):
            return next(l.gib for l in b.lines if l.name.startswith('prefix snapshots'))
        # Stock batched SDPA stores block keys in an eight-cell scratch tail;
        # native attention reads those keys directly and needs no ring tail.
        saved_per_ring = 5 * 2 * ((2048 + 8) * 8 - 2048 * 2) * 128 * 2
        self.assertEqual(snapshots(replicated) - snapshots(native), 96 * saved_per_ring / (1 << 30))
        self.assertEqual(replicated.slot_bytes - native.slot_bytes, saved_per_ring)
        self.assertGreater(native.paged_gib, replicated.paged_gib)


if __name__ == "__main__":
    unittest.main()
