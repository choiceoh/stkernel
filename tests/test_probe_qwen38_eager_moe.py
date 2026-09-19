"""The eager-MoE probe's verdicts, its lane in the kernel check, and that it imports only what the lane ships.

The run needs a GB10 and a rank file (probes/engine_qwen38_eager_moe.py). Held here: what the getter log says -- the
kernels the warm pass added, whether a request added one, whether every one-route launch kept one capacity.
"""
from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from probes import engine_qwen38_eager_moe as probe  # noqa: E402


def asked(calls):
    """An Asked over a scripted log, without a dispatcher to wrap."""
    a = probe.Asked.__new__(probe.Asked)
    a.calls = [dict(phase=p, m=m, topk=t, capacity=r, added=added, seconds=0.0) for p, m, t, r, added in calls]
    return a


class SummaryTests(unittest.TestCase):
    def test_the_fixed_boot_warms_eight_and_the_requests_add_none(self):
        log = [("warm", 8, 1, 8, True)] + [("warm", m, 1, 8, True) for m in range(1, 8)]
        log += [("requests", m, 1, 8, False) for m in probe.REQUEST_ORDER]
        verdict = probe.summarize(asked(log))
        self.assertEqual(verdict["warm_added"], [(m, 1, 8) for m in range(1, 9)])
        self.assertEqual(verdict["requests_added"], [])
        self.assertEqual(verdict["one_capacity"], [8])
        self.assertTrue(verdict["requests_within_warm"])

    def test_the_window_before_the_fix_is_caught(self):
        # 2026-09-19 K=3: nothing warmed, the requests compiled m2 r2, m1 r2, then m4 m3 m1 m2 at r4
        log = [("requests", 2, 1, 2, True), ("requests", 1, 1, 2, True), ("requests", 4, 1, 4, True),
               ("requests", 3, 1, 4, True), ("requests", 1, 1, 4, True), ("requests", 2, 1, 4, True)]
        verdict = probe.summarize(asked(log))
        self.assertEqual(len(verdict["requests_added"]), 6)
        self.assertEqual(verdict["one_capacity"], [2, 4])
        self.assertFalse(verdict["requests_within_warm"])

    def test_the_captured_launches_top_k_ten_do_not_count_as_a_capacity(self):
        log = [("warm", m, 1, 8, True) for m in range(1, 9)] + [("requests", 4, 10, 160, False)]
        self.assertEqual(probe.summarize(asked(log))["one_capacity"], [8])


class WrapTests(unittest.TestCase):
    def test_the_wrapper_sees_a_call_and_whether_it_added_a_kernel(self):
        cache = {}

        def getter(state_E, weight_E, m, k, n, num_topk, max_rows, **kwargs):
            cache.setdefault((m, num_topk, max_rows), object())
        md = SimpleNamespace(_MICRO_KERNEL_CACHE=cache, _get_micro_kernel=getter)
        a = probe.Asked(md)
        a.phase = "warm"
        md._get_micro_kernel(128, 128, 3, 2560, 640, 1, 8)
        a.phase = "requests"
        md._get_micro_kernel(128, 128, 3, 2560, 640, 1, 8)
        self.assertEqual([(c["phase"], c["m"], c["capacity"], c["added"]) for c in a.calls],
                         [("warm", 3, 8, True), ("requests", 3, 8, False)])


class LaneTests(unittest.TestCase):
    def test_the_kernel_check_runs_it(self):
        source = (ROOT / "probes/engine_kernel_check.py").read_text(encoding="utf-8")
        self.assertIn("args.lanes == 'qwen38_eager_moe'", source)
        self.assertIn("qwen38_eager_moe(args.output, args.ranks)", source)
        self.assertIn("--lanes qwen38_eager_moe --ranks", probe.__doc__)

    def test_it_imports_only_what_the_lane_ships(self):
        tree = ast.parse((ROOT / "probes/engine_qwen38_eager_moe.py").read_text(encoding="utf-8"))
        standard = ("__future__", "json", "pathlib", "sys", "time", "torch")
        for node in ast.walk(tree):
            names = ([node.module] if isinstance(node, ast.ImportFrom) else
                     [alias.name for alias in node.names] if isinstance(node, ast.Import) else [])
            for name in names:
                self.assertIn(name.split(".")[0], standard + ("engine", "probes", "tests"), name)


if __name__ == "__main__":
    unittest.main()
