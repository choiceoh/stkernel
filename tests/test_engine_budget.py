"""The GLM-5.3 budget (D1): every line carries provenance, the remainder is the KV room."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


CONFIG = Path("/home/choiceoh/models/glm53-redhat-nvfp4/config.json")


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
        self.assertAlmostEqual(weights.gib, 44.50, places=1)
        self.assertGreater(b.kv_gib, b.kv_declared_gib)                 # the box holds more KV than the boot declares
        self.assertGreater(b.kv_gib - b.kv_declared_gib, 30)
        text = budget.report(b)
        self.assertIn("unassigned +", text)
        self.assertIn("concurrency   4", text)

    def test_a_boot_ledger_replaces_vllms_activation_slope_with_this_stacks_own_split(self):
        # The two lines that could never fill themselves: a runtime floor carried over from
        # vLLM's 40th-boot table, and a 12 GiB ceiling with vLLM's 0.52 GiB/1K slope quoted
        # under it. A ledger has both, and the ceiling stays the LINE because the allocator
        # will still hand out every byte of it -- what changes is that the table can now say
        # how much of it goes unspent (45차 §51).
        from engine.base.budget import DECLARED, MEASURED
        from engine.profiles.glm53 import budget
        report = {"arena_bytes": 90 << 30, "baseline_reserved_bytes": 0, "floor_bytes": 4 << 30,
                  "workspace_limit_bytes": budget.WORKSPACE_GIB * (1 << 30),
                  "phases": [{"phase": "prefill/6912/0/prepared", "peak_workspace_bytes": 5 << 30,
                              "reserved_bytes": 90 << 30},
                             {"phase": "target/(4, 6, 4096)/captured", "peak_workspace_bytes": 5 << 30,
                              "reserved_bytes": (90 << 30) + (1 << 30)}]}
        with tempfile.TemporaryDirectory() as d:
            ledger = Path(d) / "memory-rank0.json"
            ledger.write_text(json.dumps(report))
            b = budget.budget(8.73, 4, box_gib=121.6, drafter_dir=None, ledger=ledger)
        workspace = next(l for l in b.lines if l.name.startswith("workspace"))
        self.assertEqual(workspace.gib, budget.WORKSPACE_GIB)
        self.assertEqual(workspace.source, DECLARED)
        self.assertIn("memory-rank0.json peaked at 5.00 GiB through target/(4, 6, 4096)/captured", workspace.evidence)
        self.assertIn("prefill activations 5.00 GiB over 1 qualified shapes", workspace.evidence)
        self.assertIn("graphs and scratch +1.00 GiB retained", workspace.evidence)
        self.assertNotIn("vLLM's slope", workspace.evidence)
        floor = next(l for l in b.lines if l.name.startswith("runtime floor"))
        self.assertEqual(floor.gib, 4.0)
        self.assertEqual(floor.source, MEASURED)
        self.assertIn("+7.00 GiB of the ceiling unspent", budget.report(b))

    def test_the_report_dict_itself_is_a_ledger_so_boot_need_not_reread_what_it_just_wrote(self):
        from engine.profiles.glm53 import budget
        report = {"arena_bytes": 0, "baseline_reserved_bytes": 0,
                  "measured": {"floor_bytes": 3 << 30, "prefill_peak_bytes": 2 << 30, "prefill_shapes": ["prefill/6912/0/prepared"],
                               "graph_bytes": 1 << 30, "peak_workspace_bytes": 4 << 30,
                               "retained_workspace_bytes": 1 << 30,
                               "workspace_limit_bytes": budget.WORKSPACE_GIB * (1 << 30), "at_phase": "production/ready"}}
        b = budget.budget(8.73, 4, box_gib=121.6, drafter_dir=None, ledger=report)
        self.assertEqual(next(l for l in b.lines if l.name.startswith("runtime floor")).gib, 3.0)
        self.assertIn("this boot peaked at 4.00", budget.report(b))

    def test_a_ledger_without_the_two_lines_leaves_them_exactly_where_they_were(self):
        from engine.base.budget import LEDGER
        from engine.profiles.glm53 import budget
        b = budget.budget(8.73, 4, box_gib=121.6, drafter_dir=None, ledger=None)
        floor = next(l for l in b.lines if l.name.startswith("runtime floor"))
        self.assertEqual((floor.gib, floor.source), (budget.RUNTIME_FLOOR_GIB, LEDGER))
        self.assertIn("vLLM's slope", next(l for l in b.lines if l.name.startswith("workspace")).evidence)
        self.assertNotIn("this boot peaked", budget.report(b))

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
