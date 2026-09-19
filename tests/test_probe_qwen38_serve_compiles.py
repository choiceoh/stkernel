"""The serve compile census: its kernel counting, its request loop against a fake runner, its lane, its imports.

The run needs a GB10 and a rank file (probes/engine_qwen38_serve_compiles.py). Held here: a kernel a step adds is named
with its step; a step that adds none costs no naming; a request runs until its row leaves the runner and is forgotten.
"""
from __future__ import annotations

import ast
import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from probes import engine_qwen38_serve_compiles as probe  # noqa: E402

torch = None
if importlib.util.find_spec("torch") is not None:
    import torch


class FakeCensus:
    """Kernels appear at scripted runner steps."""

    def __init__(self, appear_at):
        self.appear_at, self.steps, self.n, self.named = appear_at, 0, 0, 0

    def tick(self):
        self.steps += 1
        if self.steps in self.appear_at:
            self.n += 1

    def count(self):
        return self.n

    def new(self):
        self.named += 1
        return [{"kernel": "triton:_qsa_x", "key": f"k{self.n}"}]

    def sweep(self):
        return 0


class FakeRunner:
    """Two prefill steps, then decode steps until `made` tokens, then idle."""

    def __init__(self, census, model):
        self.census, self.model, self.state = census, model, SimpleNamespace(running=[], waiting=[])

    def submit(self, seq, prompt_len, ids=None):
        self.left, self.prefills = self.model.made, 2
        self.state.running = [seq]

    def step(self):
        if not self.state.running:
            return None
        self.census.tick()
        if self.prefills:
            self.prefills -= 1
            return SimpleNamespace(kind="prefill")
        self.left -= 1
        self.model.tokens[0].append(1)
        if self.left == 0:
            self.state.running = []
        return SimpleNamespace(kind="decode")


class FakeModel:
    def __init__(self):
        self.tokens, self.forgot, self.made = {}, 0, 0

    def add(self, seq, ids, max_new=None, temperature=None):
        if seq in self.tokens:
            raise ValueError("live")
        self.tokens[seq], self.made = list(ids), max_new

    def forget(self, seq):
        self.tokens.pop(seq)
        self.forgot += 1


@unittest.skipUnless(torch is not None, "requires torch")
class ServeTests(unittest.TestCase):
    def test_a_kernel_a_step_adds_is_named_with_its_step(self):
        census, model = FakeCensus(appear_at={4, 9}), FakeModel()
        rows = probe.serve(SimpleNamespace(vocab=1000), model, FakeRunner(census, model), census,
                           requests=(("a", 5, 3), ("b", 7, 4)))
        self.assertEqual([(r["request"], r["generated"], r["steps"], r["decode_steps"]) for r in rows],
                         [("a", 3, 5, 3), ("b", 4, 6, 4)])
        self.assertEqual([(c["step"], c["kind"]) for c in rows[0]["compiled"]], [(3, "decode")])
        self.assertEqual([(c["step"], c["kind"]) for c in rows[1]["compiled"]], [(3, "decode")])
        self.assertEqual(census.named, 2)                  # named only where the count moved
        self.assertEqual(model.forgot, 2)

    def test_the_requests_are_the_windows(self):
        self.assertEqual([(p, g) for _, p, g in probe.REQUESTS],
                         [(28, 4), (30, 21), (27, 256), (25, 474), (3223, 96), (25, 450), (27, 256)])
        self.assertLessEqual(max(p + g for _, p, g in probe.REQUESTS) + 2 * probe.SPEC_K, probe.BLOCKS * 768)


class LaneTests(unittest.TestCase):
    def test_the_kernel_check_runs_it(self):
        source = (ROOT / "probes/engine_kernel_check.py").read_text(encoding="utf-8")
        self.assertIn("args.lanes == 'qwen38_serve_compiles'", source)
        self.assertIn("qwen38_serve_compiles(args.output, args.ranks)", source)
        self.assertIn("--lanes qwen38_serve_compiles --ranks", probe.__doc__)

    def test_it_boots_in_the_fleets_order(self):
        source = (ROOT / "probes/engine_qwen38_serve_compiles.py").read_text(encoding="utf-8")
        body = source[source.index("def build("):source.index("def serve(")]
        order = [body.index(s) for s in ("warm.eager_moe(", "warm.warmup(", "capture(model, MAX_SEQS)")]
        self.assertEqual(order, sorted(order))

    def test_it_imports_only_what_the_lane_ships(self):
        tree = ast.parse((ROOT / "probes/engine_qwen38_serve_compiles.py").read_text(encoding="utf-8"))
        standard = ("__future__", "dataclasses", "gc", "inspect", "json", "pathlib", "sys", "time", "torch", "triton")
        for node in ast.walk(tree):
            names = ([node.module] if isinstance(node, ast.ImportFrom) else
                     [alias.name for alias in node.names] if isinstance(node, ast.Import) else [])
            for name in names:
                self.assertIn(name.split(".")[0], standard + ("engine", "probes", "tests"), name)


if __name__ == "__main__":
    unittest.main()
