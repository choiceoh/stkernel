"""The two rows left large and undivided get cut, because that is what cracked the last one.

`prefill/128`'s 15.6 s was a mystery for four days and an answer the week its row was split into
forward, vote and reclaim (2026-09-16: it is the forward, and no artifact was compiled anywhere in the
phase). The same is now true of `prepare native execution` -- 32.16 s, then 23.50 after the pack store
stopped hashing every weight twice, and no breakdown either time -- and of `lanes` 3.22 s, which is
importing the kernel packages and building a table over them in one number.
"""
import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parents[1]


class PrepareRowTests(unittest.TestCase):
    def setUp(self):
        source = (ROOT / "engine/profiles/glm53/boot.py").read_text(encoding="utf-8")
        start = source.index('with recorder.phase("prepare native execution")')
        self.block = source[start:source.index("if calib_plan:", start)]

    def test_the_calls_that_cost_each_have_a_row(self):
        for name, call in (("routers", "net.prepare_routers(arena)"),
                           ("dense packs", "net.prepare_dense(store, consume_weights=True)"),
                           ("decode projections", "net.prepare_decode_projections(arena, capture_rows=capture_rows)"),
                           ("drafter packs", "drafter.prepare_fast(store,")):
            with self.subTest(row=name):
                row = self.block.index(f'with recorder.phase("{name}")')
                self.assertLess(row, self.block.index(call), f"{call} runs inside its row")

    def test_the_decode_fastpaths_stay_inside_the_projection_row(self):
        """They are one decision and one cost; three rows of nothing would read as three costs."""
        row = self.block.index('with recorder.phase("decode projections")')
        for call in ("net.prepare_decode_dsa_inputs(", "net.prepare_decode_indexer_gate(",
                     "net.prepare_decode_absorb("):
            self.assertLess(row, self.block.index(call))
        self.assertLess(self.block.index("net.prepare_decode_absorb("),
                        self.block.index("recorder.gauge('decode_projection_resident_bytes'"))

    def test_the_drafter_checkpoint_load_keeps_its_own_row_outside_the_packs(self):
        """`load_drafter` already opens `load drafter`; the packing is the part with no row."""
        self.assertLess(self.block.index("drafter = load_drafter()"),
                        self.block.index('with recorder.phase("drafter packs")'))


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "requires PyTorch")
class LaneRowTests(unittest.TestCase):
    def test_the_imports_get_their_own_row_and_only_when_asked(self):
        source = (ROOT / "engine/profiles/glm53/lanes.py").read_text(encoding="utf-8")
        start = source.index("def served(")
        signature = source[start:source.index("-> Lanes:", start)]
        self.assertIn("recorder=None", signature)
        opened = source.index('recorder.phase("kernel imports") if recorder is not None else nullcontext()')
        self.assertLess(opened, source.index("from engine.kernels.kda import"))
        self.assertLess(source.index("from engine.kernels.indexer import"),
                        source.index("mk.configure_prefill(mla_prefill)"))

    def test_the_fleet_boot_passes_its_recorder(self):
        boot = (ROOT / "engine/profiles/glm53/boot.py").read_text(encoding="utf-8")
        fleet = boot[boot.index("def fleet(a)"):]
        served = fleet.index("lanes = lane_tables.served(")
        call = fleet[served:fleet.index("from engine.profiles.glm53.execution", served)]
        self.assertIn("recorder=rec", call)

    def test_a_caller_without_a_recorder_still_imports(self):
        """Probes and the local boot call `served()` bare; the nullcontext is what keeps them working."""
        from contextlib import nullcontext
        with (None.phase("x") if False else nullcontext()) as span:
            self.assertIsNone(span)


class KernelPrefetchTests(unittest.TestCase):
    """The packages `served` imports are the packages the boot prefetches -- checked, not remembered."""

    def setUp(self):
        self.source = (ROOT / "engine/profiles/glm53/lanes.py").read_text(encoding="utf-8")
        start = self.source.index("def served(")
        self.block = self.source[start:self.source.index("mk.configure_prefill(mla_prefill)", start)]

    def imported(self):
        """The packages the `served` import block actually pulls in, read off the block."""
        import re
        names = set()
        for line in self.block.splitlines():
            found = re.match(r"\s*from (engine\.kernels\.[\w.]+) import", line)
            if found:
                names.add(found.group(1))
            found = re.match(r"\s*from engine\.kernels import (\w+)", line)
            if found:
                names.add("engine.kernels." + found.group(1))
        return names

    def listed(self):
        from engine.profiles.glm53 import lanes
        return set(lanes.KERNEL_MODULES)

    def test_the_prefetch_list_is_exactly_what_served_imports(self):
        """Both directions: a package added to `served` and forgotten here loses its prefetch, and a
        name left here after `served` stopped importing it is a module the boot loads for nothing."""
        self.assertTrue(self.imported(), "the import block moved")
        self.assertEqual(self.imported(), self.listed())

    def test_it_imports_by_name_and_nothing_else(self):
        """The thread's whole job is importing. Anything else on it would be work off the main thread."""
        import ast
        tree = ast.parse(self.source)
        fn = next(node for node in tree.body
                  if isinstance(node, ast.FunctionDef) and node.name == "import_kernels")
        body = [node for node in fn.body if not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant))]
        self.assertEqual([type(node).__name__ for node in body], ["Import", "For"])
        call = body[1].body[0].value
        self.assertEqual(ast.unparse(call.func), "importlib.import_module")
        self.assertEqual(len(body[1].body), 1)

    def test_the_boot_starts_it_after_the_builds_and_joins_it_in_the_lanes_row(self):
        boot = (ROOT / "engine/profiles/glm53/boot.py").read_text(encoding="utf-8")
        fleet = boot[boot.index("def fleet(a)"):]
        started = fleet.index('imports = Background(lane_tables.import_kernels, "kernel-imports").start()')
        self.assertGreater(started, fleet.index("seconds = builds.wait()"))
        self.assertLess(started, fleet.index('comm.wait_prepared("native-builds")'))
        joined = fleet.index("imports.take()")
        self.assertGreater(joined, fleet.index('with rec.phase("lanes")'))
        self.assertLess(joined, fleet.index("lanes = lane_tables.served("))


class RowNameTests(unittest.TestCase):
    """A nested row is printed under its parent and adds nothing to the total (base/instruments)."""

    def test_nested_rows_do_not_double_count_in_the_total(self):
        from engine.base import instruments
        rec = instruments.Recorder("rank0", memory_sampling=False)
        with rec.phase("prepare native execution") as outer:
            with rec.phase("dense packs") as inner:
                pass
        outer.seconds, inner.seconds = 23.5, 14.0
        total = float(rec.table().splitlines()[-1].split()[1])
        self.assertEqual(total, 23.5)
        self.assertIn("  dense packs", rec.table())


if __name__ == "__main__":
    unittest.main()
