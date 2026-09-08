"""CPU tensor checks of MHC fallback dispatch and exact failure diagnostics."""
from contextlib import contextmanager
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]


def driver():
    path = ROOT / "overlay/modules/glm53_megakernel/glm53_megakernel.py"
    spec = importlib.util.spec_from_file_location("mhc_ar_selftest_driver", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TorchProxy:
    def __init__(self, cuda):
        self.cuda = cuda

    def __getattr__(self, key):
        return getattr(torch, key)

    def randn(self, *shape, **kwargs):
        return torch.randn(*shape, **dict(kwargs, device="cpu"))

    def rand(self, *shape, **kwargs):
        return torch.rand(*shape, **dict(kwargs, device="cpu"))

    def zeros(self, *shape, **kwargs):
        return torch.zeros(*shape, **dict(kwargs, device="cpu"))

    def ones(self, *shape, **kwargs):
        return torch.ones(*shape, **dict(kwargs, device="cpu"))


class DiagnosticTests(unittest.TestCase):
    def test_named_output_count_first_value_and_nonfinite_are_visible(self):
        mk = driver()
        ref = tuple(torch.zeros(2, 4, dtype=torch.bfloat16) for _ in range(4))
        got = tuple(value.clone() for value in ref)
        got[1][1, 2] = 2
        got[2][0, 0] = float("nan")
        ref[3][0, 1] = float("inf")
        details = mk._mhc_ar_mismatch_details(ref, got)
        self.assertEqual([row["output"] for row in details], ["residual", "post_mix", "comb_mix", "layer_input"])
        self.assertTrue(details[0]["exact"])
        self.assertEqual(details[1]["different"], 1)
        self.assertEqual(details[1]["finite_max_abs"], 2)
        self.assertEqual(details[1]["first_flat_index"], 6)
        self.assertEqual((details[1]["first_ref"], details[1]["first_got"]), (0, 2))
        self.assertEqual(details[2]["got_nonfinite"], 1)
        self.assertEqual(details[3]["ref_nonfinite"], 1)
        self.assertNotEqual(details[1]["ref_ptr"], details[1]["got_ptr"])

    def test_large_fallback_keeps_t12_cache_warm_for_small_capture(self):
        mk = driver()
        # Small dimensions exercise the actual packing/cache/driver code with
        # CPU storage; only the CUDA predicate and native launch are mocked.
        mk.HIDDEN = 8
        mk.ENABLE_AR_CONSUMER = mk._AR_CONSUMER_OK = True
        mk.ENABLE_MHC_BF16 = mk._MHC_BF16_OK = True
        mk._ar_note = lambda *_: None
        capturing, launches = [False], []
        cuda = SimpleNamespace(is_current_stream_capturing=lambda: capturing[0])
        mk._EXT = SimpleNamespace(run_mhc=lambda ptrs, scalars, ints, bf16, early:
                                 launches.append((list(ptrs), ints[0], bf16, early)))
        fn = torch.randn(24, 32).bfloat16().float()
        def call(t, *, early):
            return mk._mhc_call(torch.zeros(t, 8, dtype=torch.bfloat16),
                torch.zeros(t, 4, 8, dtype=torch.bfloat16), torch.zeros(t, 4), torch.zeros(t, 16),
                fn, torch.ones(3), torch.zeros(24), torch.ones(8, dtype=torch.bfloat16),
                t, 1e-6, 1e-6, 1e-6, 1., 1e-6, 20, _ar_consumer=early)
        with patch.object(torch.Tensor, "is_cuda", property(lambda _: True)), \
             patch.dict(sys.modules, {"torch": TorchProxy(cuda)}):
            warm = call(12, early=False)
            entry = next(iter(mk._MHC_BF16_CACHE.values()))
            scalar, vector = entry[1], entry[2]
            self.assertEqual(tuple(scalar.shape), (24, 32))
            self.assertEqual(tuple(vector.shape), (24, 8, 4))
            self.assertEqual(launches[-1][1:], (12, False, False))
            self.assertEqual(launches[-1][0][4], fn.data_ptr())
            eager = call(16, early=False)
            capturing[0] = True
            captured = call(16, early=True)
            before, after = launches[-2:]
            held = [call(12, early=True)]
            self.assertEqual(launches[-1][1:], (12, False, False))
            self.assertEqual(launches[-1][0][4], fn.data_ptr())
            for t in (6, 8):
                held.append(call(t, early=True))
                self.assertEqual(launches[-1][1:], (t, True, True))
                self.assertEqual(launches[-1][0][4], vector.data_ptr())
            held.append(call(8, early=False))
            self.assertEqual(launches[-1][1:], (8, True, False))
            self.assertEqual(launches[-1][0][4], scalar.data_ptr())
        self.assertEqual(before[1:], (16, False, False))
        self.assertEqual(after[1:], before[1:])
        self.assertEqual(before[0][4], fn.data_ptr())
        self.assertEqual(after[0][4], before[0][4])
        self.assertEqual(before[0][12:], after[0][12:])
        self.assertEqual(len({value.data_ptr() for value in (*eager, *captured)}), 8)
        self.assertFalse({value.data_ptr() for value in (*eager, *captured)} & set(before[0][12:]))
        self.assertEqual(len(mk._MHC_BF16_CACHE), 1)
        self.assertIs(next(iter(mk._MHC_BF16_CACHE.values())), entry)
        self.assertIs(entry[0], fn)

    def test_original_selftest_launches_t16_fp32_fallback_and_small_bf16_consumer(self):
        # Exercise the original self-test, packing and dispatch. The fake
        # native launch only verifies routing/replay, never CUDA numerics.
        mk = driver()
        mk.HIDDEN = 8
        mk.ENABLE_AR_CONSUMER = mk._AR_CONSUMER_OK = True
        mk.ENABLE_MHC_BF16 = mk._MHC_BF16_OK = True
        mk._ar_note = lambda *_: None
        active, launches, replays, tensors = [None], [], [], {}
        class Graph:
            def replay(self):
                self.replay_work()
        @contextmanager
        def graph_context(graph):
            active[0] = graph
            try:
                yield
            finally:
                active[0] = None
        cuda = SimpleNamespace(CUDAGraph=Graph, graph=graph_context, synchronize=lambda: None,
            is_current_stream_capturing=lambda: active[0] is not None)
        class RecordingTorch(TorchProxy):
            def empty(self, *shape, **kwargs):
                result = torch.empty(*shape, **dict(kwargs, device="cpu"))
                tensors[result.data_ptr()] = result
                return result
            def empty_like(self, value, **kwargs):
                result = torch.empty_like(value, **kwargs)
                tensors[result.data_ptr()] = result
                return result
        def launch(ptrs, scalars, ints, bf16, early):
            t = ints[0]
            launches.append((t, bf16, early, active[0] is not None))
            outputs = [tensors[ptr] for ptr in ptrs[8:12]]
            def fill():
                for i, output in enumerate(outputs):
                    output.fill_(i + t)
            fill()
            if active[0] is not None:
                def replay():
                    replays.append((t, bf16, early))
                    fill()
                active[0].replay_work = replay
        mk._EXT = SimpleNamespace(run_mhc=launch)
        with patch.object(torch.Tensor, "is_cuda", property(lambda _: True)), \
             patch.dict(sys.modules, {"torch": RecordingTorch(cuda)}), \
             self.assertLogs(mk.logger.name, level="WARNING") as log:
            self.assertTrue(mk._selftest_ar_consumer())
        self.assertEqual(len(replays), 30)
        self.assertEqual([row for row in launches if row[0] == 16 and row[3]],
                         [(16, False, False, True)] * 2)
        self.assertEqual([row for row in replays if row[0] == 16],
                         [(16, False, False)] * 6)
        self.assertTrue(all(not bf16 and not early for t, bf16, early, _ in launches if t == 12))
        for t in (1, 2, 6, 8):
            self.assertIn((t, True, True, True), launches)
        self.assertEqual(len(mk._MHC_BF16_CACHE), 5)
        self.assertTrue(all(entry[1] is not None and entry[2] is not None
                            for entry in mk._MHC_BF16_CACHE.values()))
        self.assertIn("AR consumer MHC self-test PASS", "\n".join(log.output))

    def test_original_selftest_keeps_t16_all_replays_and_disarms_on_exact_failure(self):
        for poison in (False, True):
            with self.subTest(poison=poison):
                mk = driver()
                mk.HIDDEN = 8
                mk.ENABLE_MHC_BF16 = mk._MHC_BF16_OK = True
                active, calls, replays = [None], [], []
                class Graph:
                    def replay(self):
                        self.replay_work()
                @contextmanager
                def graph_context(graph):
                    active[0] = graph
                    try:
                        yield
                    finally:
                        active[0] = None
                cuda = SimpleNamespace(CUDAGraph=Graph, graph=graph_context, synchronize=lambda: None)
                mk._mhc_bf16_weight = lambda fn, ar_consumer=False: torch.zeros(
                    (24, 8, 4) if ar_consumer else (24, 32), dtype=torch.bfloat16)
                def compute(values):
                    x, residual, pm, cm = values[:4]
                    return residual.clone(), pm + x.float().sum(), cm.clone(), x.clone()
                def mhc_call(*values, _fp32_fn=False, _ar_consumer=None):
                    t = values[8]
                    calls.append((t, _fp32_fn, _ar_consumer, active[0] is not None))
                    outputs = compute(values)
                    if active[0] is not None:
                        def replay():
                            replays.append((t, _fp32_fn))
                            for target, value in zip(outputs, compute(values)):
                                target.copy_(value)
                            if poison and t == 16 and not _fp32_fn and values[0][0, 0] < 0:
                                outputs[1][0, 2] += 1
                        active[0].replay_work = replay
                    return outputs
                mk._mhc_call = mhc_call
                with patch.dict(sys.modules, {"torch": TorchProxy(cuda)}), \
                     self.assertLogs(mk.logger.name, level="WARNING") as log:
                    result = mk._selftest_ar_consumer()
                self.assertEqual(result, not poison)
                self.assertEqual(len(replays), 30)
                self.assertEqual({t for t, _, _, capture in calls if capture}, {1, 2, 6, 8, 16})
                self.assertEqual(sum(t == 16 and not fp32 for t, fp32 in replays), 3)
                text = "\n".join(log.output)
                if poison:
                    self.assertIn("T=16 fp32=False scale=-0.5 early=False", text)
                    self.assertIn("'output': 'post_mix'", text)
                    self.assertIn("'finite_max_abs': 1.0", text)
                    self.assertIn("mismatch storage", text)
                    self.assertNotIn("self-test PASS", text)
                else:
                    self.assertIn("AR consumer MHC self-test PASS", text)
                    self.assertNotIn("mismatch", text)


if __name__ == "__main__":
    unittest.main()
