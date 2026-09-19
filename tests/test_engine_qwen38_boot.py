"""Qwen3.8's fleet boot does its host work where it is already waiting, and says where its seconds went.

The first fleet boot (2026-09-18) reported one number a rank -- "ready in 107.4 s", then 40.1 s -- and nothing inside
it; a later boot's rank 3 log puts 3.96 s between `collectives` and `lanes qualified`, and rank 0 built the door (7.0 s
in the ST image on the CPU, a fresh process) after the capture. GLM-5.3's boot had already moved the same host work off
its critical path (tests/test_engine_boot_breakdown.py); these hold Qwen3.8's to it.
"""
import ast
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FLEET = ROOT / "engine/profiles/qwen38/fleet.py"
LANES = ROOT / "engine/profiles/qwen38/lanes.py"


def function(path: Path, name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


def kernel_imports(fn: ast.FunctionDef) -> "set[str]":
    """The kernel packages a function's `from` lines pull in: `from engine.kernels import a, b` names two modules,
    `from engine.kernels.x import f` names engine.kernels.x."""
    names = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith(("engine.kernels", "engine.modules")):
            if node.module in ("engine.kernels", "engine.modules"):
                names.update(f"{node.module}.{alias.name}" for alias in node.names)
            else:
                names.add(node.module)
    return names


class KernelPrefetchTests(unittest.TestCase):
    def listed(self) -> "set[str]":
        tree = ast.parse(LANES.read_text(encoding="utf-8"))
        assign = next(node for node in tree.body if isinstance(node, ast.Assign)
                      and any(getattr(t, "id", None) == "KERNEL_MODULES" for t in node.targets))
        return set(ast.literal_eval(assign.value))

    def test_the_prefetch_list_is_exactly_what_served_imports(self):
        """Both directions, over the profile's `served` and the common lanes it starts from."""
        imported = kernel_imports(function(LANES, "served")) | kernel_imports(function(ROOT / "engine/base/lanes.py", "served"))
        self.assertTrue(imported, "the import block moved")
        self.assertEqual(imported, self.listed())

    def test_it_imports_by_name_and_nothing_else(self):
        fn = function(LANES, "import_kernels")
        body = [node for node in fn.body if not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant))]
        self.assertEqual([type(node).__name__ for node in body], ["Import", "For"])
        self.assertEqual(ast.unparse(body[1].body[0].value.func), "importlib.import_module")
        self.assertEqual(len(body[1].body), 1)


class BootOrderTests(unittest.TestCase):
    def setUp(self):
        self.main = ast.get_source_segment(FLEET.read_text(encoding="utf-8"), function(FLEET, "main"))
        self.build = ast.get_source_segment(FLEET.read_text(encoding="utf-8"), function(FLEET, "build"))

    def test_the_imports_start_after_the_box_and_under_the_rendezvous(self):
        """After check_box, which initialises CUDA on the main thread; before Comm.init, where the ranks meet."""
        started = self.main.index('imports = Background(lane_tables.import_kernels, "kernel-imports").start()')
        self.assertGreater(started, self.main.index("facts.check_box()"))
        self.assertLess(started, self.main.index("comm = Comm.init()"))
        joined = self.main.index("imports.take()")
        self.assertGreater(joined, self.main.index('with rec.phase("lanes")'))
        self.assertLess(joined, self.main.index("lanes = lane_tables.served("))

    def test_the_prelude_starts_before_the_load_and_joins_before_the_capture(self):
        started = self.main.index('"boot-prelude").start()')
        self.assertLess(started, self.main.index("= build("))
        self.assertIn("prelude=prelude", self.main[self.main.index("= build("):])
        joined = self.build.index("prelude.take()")
        self.assertLess(self.build.index('with recorder.phase("wait for the prelude")'), joined)
        self.assertLess(joined, self.build.index('with recorder.phase("capture decode")'))
        self.assertGreater(joined, self.build.index('with recorder.phase("prepare dense")'))

    def test_the_door_is_built_from_the_prelude_not_again(self):
        door = self.main[self.main.index('with rec.phase("door")'):]
        for call in ("tokenizer(", "chat_renderer(", "reasoning_marks(", "tool_formats.detect(", "effort_rungs_checked("):
            self.assertNotIn(call, self.main, f"{call} runs on the main thread again")
        self.assertIn("prelude.take()", door)

    def test_the_host_half_touches_no_device(self):
        source = ast.get_source_segment(FLEET.read_text(encoding="utf-8"), function(FLEET, "door_host_half"))
        for word in ("cuda", "torch.", "device"):
            self.assertNotIn(word, source.split('"""')[-1], f"the prelude thread reaches for {word}")

    def test_every_phase_between_the_box_and_the_door_has_a_row(self):
        for row in ("comm", "prepare one-shot", "lanes", "qualify lanes", "door"):
            with self.subTest(row=row):
                self.assertIn(f'with rec.phase("{row}")', self.main)
        self.assertLess(self.main.index("comm.prepare_oneshot()"), self.main.index('with rec.phase("lanes")'))
        self.assertGreater(self.main.index("lane_tables.qualify("), self.main.index('with rec.phase("qualify lanes")'))
        self.assertIn('rec.mark("front"', self.main)

    def test_rank_zero_prints_the_table_and_every_rank_dumps_before_serving(self):
        dumped = self.main.index("write_dumps(rec, model.memory, a.dump_dir, comm.rank)")
        self.assertLess(self.main.index("print(rec.table()"), self.main.index("server.loop()"))
        self.assertLess(dumped, self.main.index("server.loop()"))


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "requires PyTorch")
class DumpTests(unittest.TestCase):
    def test_a_dump_writes_the_table_and_the_ledger(self):
        from engine.base.instruments import Recorder
        from engine.profiles.qwen38 import fleet
        rec = Recorder("rank2", memory_sampling=False)
        with rec.phase("comm"):
            pass

        class Memory:
            def write(self, path):
                Path(path).write_text('{"phases": []}\n')

        with tempfile.TemporaryDirectory() as root:
            out = Path(root) / "dumps"
            fleet.write_dumps(rec, Memory(), out, 2)
            self.assertEqual(json.loads((out / "boot-rank2.json").read_text())["root"]["children"][0]["name"], "comm")
            self.assertTrue((out / "memory-rank2.json").exists())

    def test_a_dump_that_cannot_be_written_does_not_stop_the_boot(self):
        from engine.base.instruments import Recorder
        from engine.profiles.qwen38 import fleet
        with tempfile.TemporaryDirectory() as root:
            blocker = Path(root) / "file"
            blocker.write_text("")
            fleet.write_dumps(Recorder("rank0", memory_sampling=False), None, blocker / "dumps", 0)   # a file, not a dir


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "requires PyTorch")
class PackStoreTimingTests(unittest.TestCase):
    def test_every_entry_is_counted_and_answers_as_before(self):
        from engine.profiles.qwen38 import fleet

        class Store:
            def weight_digest(self, weight):
                return ("digest", weight)

            def pack(self, weight, name, digest=None):
                digest = digest or self.weight_digest(weight)      # the store's own call goes through the timer too
                return ("pack", name, digest)

            def pack_fp8(self, weight, name, digest=None):
                return None

        store = Store()
        spent = fleet.timed_store(store)
        self.assertEqual(store.pack("w", "L0.o"), ("pack", "L0.o", ("digest", "w")))
        store.pack("w", "L1.o", digest="d")
        self.assertIsNone(store.pack_fp8("w", "L0.o"))
        self.assertEqual({k: v[0] for k, v in spent.items()}, {"weight_digest": 1, "pack": 2, "pack_fp8": 1})
        self.assertTrue(all(seconds >= 0 for _, seconds in spent.values()))

    def test_the_prepare_dense_row_carries_them(self):
        source = FLEET.read_text(encoding="utf-8")
        row = source[source.index('with recorder.phase("prepare dense")'):source.index('with recorder.phase("caches")')]
        self.assertLess(row.index("spent = timed_store(store)"), row.index("net.prepare_dense("))
        self.assertIn('recorder.gauge(f"packs_{name}", count)', row)


if __name__ == "__main__":
    unittest.main()
