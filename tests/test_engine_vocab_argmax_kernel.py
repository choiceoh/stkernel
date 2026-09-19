"""engine/kernels/common/vocab_candidates.argmax_key, the greedy pick's packet on a device (Qwen3.8 carry D5): both
launches at the declared warps, the same packet at any warps, and the key engine/modules/vocab.argmax computes off the
device -- zeros and NaNs canonical, the lowest id of equal scores, nothing past the decodable cut -- through one partial
and through several with the finishing launch. Until now only a probe ran the kernel path against anything.

    docker exec -w <repo> stk-test python3 -m unittest tests.test_engine_vocab_argmax_kernel
    docker exec -e TRITON_INTERPRET=1 -w <repo> stk-test python3 -m unittest tests.test_engine_vocab_argmax_kernel
"""
import ast
import importlib.util
import os
from pathlib import Path
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
torch = None
if importlib.util.find_spec("torch") is not None:
    import torch
TRITON = importlib.util.find_spec("triton") is not None
INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
RUNS = torch is not None and TRITON and (INTERPRET or torch.cuda.is_available())
DEVICE = "cpu" if INTERPRET or torch is None or not torch.cuda.is_available() else "cuda"


def reference(x, start, valid):
    """engine/modules/vocab.argmax's key off the device, line for line (tests/test_engine_vocab.py holds that branch
    to torch.argmax)."""
    value, index = x[..., :valid].float().cpu().max(dim=-1)
    value = torch.where(value == 0, torch.zeros_like(value), value)
    value = torch.where(torch.isnan(value), torch.full_like(value, float("nan")), value)
    bits = value.contiguous().view(torch.int32).to(torch.int64)
    ordered = torch.where(bits < 0, bits ^ 0x7fffffff, bits)
    return (ordered << 32) | (0xffffffff - (index + start))


class LaunchTests(unittest.TestCase):
    def test_both_launches_take_the_declared_warps(self):
        source = (ROOT / "engine" / "kernels" / "common" / "vocab_candidates.py").read_text()
        tree = ast.parse(source)
        declared = {t.id: n.value for n in tree.body if isinstance(n, ast.Assign) for t in n.targets
                    if isinstance(t, ast.Name)}
        self.assertEqual(ast.literal_eval(declared["ARGMAX_WARPS"]), 4)       # the record's (measurements/, D5)
        body = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "argmax_key")
        launches = [n for n in ast.walk(body) if isinstance(n, ast.Call) and isinstance(n.func, ast.Subscript)]
        self.assertEqual(sorted(n.func.value.id for n in launches), ["_argmax_finish", "_argmax_partials"])
        for launch in launches:
            warps = {k.arg: k.value for k in launch.keywords}["num_warps"]
            self.assertEqual(ast.unparse(warps), "ARGMAX_WARPS")

    def test_kernel_check_routes_the_lane_to_the_probe(self):
        text = (ROOT / "probes" / "engine_kernel_check.py").read_text()
        self.assertIn("args.lanes == 'vocab_argmax'", text)
        self.assertIn("from probes.engine_vocab_selection import argmax_run as vocab_argmax_check", text)


@unittest.skipUnless(RUNS, "requires CUDA and Triton, or TRITON_INTERPRET=1 with Triton")
class KeyTests(unittest.TestCase):
    def key(self, x, start, valid):
        from engine.kernels.common.vocab_candidates import argmax_key
        return argmax_key(x, start, valid).cpu()

    def test_the_exceptional_values_tie_as_torch_argmax_does(self):
        x = torch.tensor([
            [2., 2., 0., 2., 2., 2., 1., 2.],
            [-0., -1., 0., -0., 0., -2., -0., 0.],
            [float("-inf")] * 8,
            [0., float("inf"), 0., float("inf"), -1., -1., -1., -1.],
            [-1., float("nan"), 0., -float("nan"), -1., -1., -1., -1.],
            [-8., -7., -6., -5., -4., -3., -2., -1.],
        ], device=DEVICE)
        for valid in (8, 5, 1):
            for start in (0, 1000):
                with self.subTest(valid=valid, start=start):
                    got = self.key(x, start, valid)
                    self.assertTrue(torch.equal(got, reference(x, start, valid)))
                    self.assertEqual((0xffffffff - (got & 0xffffffff) - start).tolist(),
                                     x[:, :valid].cpu().argmax(-1).tolist())

    def test_several_partials_and_the_finishing_launch(self):
        gen = torch.Generator().manual_seed(95)
        width = 2500 if INTERPRET else 62080                                # three partials, or Qwen3.8's shard
        dtypes = (torch.float32, torch.float16) if INTERPRET else (torch.float32, torch.float16, torch.bfloat16)
        for dtype in dtypes:
            full = torch.randn(3, 2 * width, generator=gen).to(device=DEVICE, dtype=dtype)
            x = full[:, ::2]                                                 # a strided shard
            valid = width - 100
            x[:, valid:] = 1000                                             # past the cut: never chosen
            x[0, 1020:1030] = 50                                            # a tie across two partials: the lowest id
            x[1, valid - 1] = 60                                            # the last decodable column
            with self.subTest(dtype=dtype):
                got = self.key(x, 3 * width, valid)
                self.assertTrue(torch.equal(got, reference(x, 3 * width, valid)))
                self.assertEqual((0xffffffff - (got & 0xffffffff) - 3 * width).tolist()[:2], [1020, valid - 1])

    def test_no_valid_column_is_the_lowest_key(self):
        x = torch.zeros(4, 16, device=DEVICE)
        self.assertEqual(self.key(x, 0, 0).tolist(), [-(2 ** 63)] * 4)

    def test_the_packet_is_the_same_at_any_warps(self):
        """Integer keys: what the declared warps may never change."""
        from engine.kernels.common import vocab_candidates
        gen = torch.Generator().manual_seed(96)
        x = torch.randn(5, 2100, generator=gen).to(DEVICE)
        x[:, 7:90] = 3.5
        want = self.key(x, 11, 2050)
        for warps in (1, 2):
            with self.subTest(warps=warps), mock.patch.object(vocab_candidates, "ARGMAX_WARPS", warps):
                self.assertTrue(torch.equal(self.key(x, 11, 2050), want))


if __name__ == "__main__":
    unittest.main()
