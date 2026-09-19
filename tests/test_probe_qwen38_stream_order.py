"""gated_residual's stream launches over the grid (hc, rows) -- a row's streams adjacent -- and
probes/engine_qwen38_stream_order.py: the rule, the hook (inert until set, validated, restored, reaching every stream
launch's grid), the two orders the same bytes (CUDA, or TRITON_INTERPRET=1), and the lane routed.

    docker exec -w <repo> stk-test python3 -m unittest tests.test_probe_qwen38_stream_order
    TRITON_INTERPRET=1 CUDA_VISIBLE_DEVICES= python3 -m unittest tests.test_probe_qwen38_stream_order
"""
import importlib.util
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
torch = None
if importlib.util.find_spec("torch") is not None:
    import torch
TRITON = importlib.util.find_spec("triton") is not None
INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
KERNELS = torch is not None and TRITON
RUNS = KERNELS and (INTERPRET or torch.cuda.is_available())
DEVICE = "cpu" if INTERPRET or torch is None or not torch.cuda.is_available() else "cuda"
HC = 4


def probe():
    import probes.engine_qwen38_stream_order as module
    return module


class Recorder:
    def __init__(self):
        self.launches = []

    def __getitem__(self, grid):
        return lambda *args, **meta: self.launches.append((tuple(grid), meta))


@unittest.skipUnless(torch is not None, "requires torch")
class TableTests(unittest.TestCase):
    def test_the_probe_imports_without_a_gpu_or_the_kernel_package(self):
        code = ("import sys, probes.engine_qwen38_stream_order as p; "
                "print(callable(p.run), 'engine.kernels' in sys.modules)")
        result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True,
                                env={**os.environ, "PYTHONPATH": str(ROOT)})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.split(), ["True", "False"])

    def test_the_table_is_the_model_s_and_the_arms_are_the_two_orders(self):
        from probes.engine_qwen38_cells import facts
        p, F = probe(), facts()
        self.assertEqual((p.HIDDEN, p.HC), (F.hidden, F.hc))
        self.assertEqual([order for _, order in p.ORDERS], ["rows", None])
        self.assertTrue(all(rows >= 512 for rows in p.ROWS))              # mix_block's rows: the prefill site's leave
        self.assertEqual(p.FORMS, ("stream_scales", "leave_norm", "norm_streams"))

    def test_kernel_check_routes_the_lane(self):
        text = (ROOT / "probes" / "engine_kernel_check.py").read_text()
        self.assertIn("args.lanes == 'qwen38_stream_order'", text)
        self.assertIn("from probes.engine_qwen38_stream_order import run as qwen38_stream_order", text)


@unittest.skipUnless(KERNELS, "requires torch and triton")
class HookTests(unittest.TestCase):
    def setUp(self):
        from engine.kernels import gated_residual
        self.hcr = gated_residual

    def test_the_rule_is_a_row_s_streams_adjacent(self):
        hcr = self.hcr
        self.assertIsNone(hcr._STREAM_GRID_OVERRIDE)
        self.assertEqual(hcr._stream_grid(4096, HC), ((HC, 4096), False))
        self.assertEqual(hcr._stream_grid(1, 4), ((4, 1), False))

    def test_forced_takes_validates_and_restores(self):
        p, hcr = probe(), self.hcr
        with p.forced("rows"):
            self.assertEqual(hcr._stream_grid(4096, HC), ((4096, HC), True))
        self.assertIsNone(hcr._STREAM_GRID_OVERRIDE)
        for bad in ("streams", "", 0, True, ("rows",)):
            with self.subTest(bad=bad), p.forced(bad), self.assertRaisesRegex(ValueError, "_STREAM_GRID_OVERRIDE"):
                hcr._stream_grid(4096, HC)
        with self.assertRaises(RuntimeError):
            with p.forced("rows"):
                raise RuntimeError("the block dies")
        self.assertIsNone(hcr._STREAM_GRID_OVERRIDE)

    def test_the_order_reaches_every_stream_launch(self):
        """leave_norm, stream_scales after a leave, norm_streams and the scales of the streams as they are: the grid and
        the kernel's ROWS_FIRST agree, under the rule and forced."""
        p, hcr = probe(), self.hcr
        rows, hid = 20, 64
        gen = torch.Generator().manual_seed(3)
        h, out, inject, w = p.operands(rows, torch.device("cpu"), gen, hid, HC)
        for order, grid, rows_first in ((None, (HC, rows), False), ("rows", (rows, HC), True)):
            leaves, norms = Recorder(), Recorder()
            with self.subTest(order=order), patch.object(hcr, "_leave_norm", leaves), \
                    patch.object(hcr, "_norm_streams", norms), \
                    patch.object(torch.Tensor, "is_cuda", property(lambda tensor: True)), p.forced(order):
                hcr.leave_norm(h.clone(), out, inject, w, 1e-6, HC)
                hcr.stream_scales(h.clone(), out, inject, 1e-6, HC)
                hcr.norm_streams(h, w, 1e-6, HC)
                hcr.stream_scales(h, None, None, 1e-6, HC)
                self.assertEqual(len(leaves.launches), 2)
                self.assertEqual(len(norms.launches), 2)
                for launched, meta in leaves.launches + norms.launches:
                    self.assertEqual((launched, meta["ROWS_FIRST"]), (grid, rows_first))
                self.assertEqual([meta["SCALE_ONLY"] for _, meta in leaves.launches + norms.launches],
                                 [False, True, False, True])


@unittest.skipUnless(RUNS, "requires CUDA and Triton, or TRITON_INTERPRET=1 with Triton")
class BytesTests(unittest.TestCase):
    def test_the_two_orders_are_the_same_bytes(self):
        """Every form, from the same operands: the streams after the form and what it wrote. The interpreter runs the
        kernels for the scale-only forms (the others take the torch form off a device)."""
        p = probe()
        hidden = 80 if INTERPRET else p.HIDDEN                               # 80 = 5 x 16: whole lanes, and masked ones
        gen = torch.Generator().manual_seed(11)
        for rows in (1, 19):
            for name in p.FORMS:
                with self.subTest(rows=rows, form=name):
                    self.assertTrue(p.alike(name, rows, torch.device(DEVICE), gen, hidden, HC))

    def test_forced_rows_first_is_the_rule_s_bytes_through_site(self):
        """The prefill site whole (gated_residual.site, from PREFILL_ROWS) under each order: the same mixed output,
        injection and streams."""
        from engine.kernels import gated_residual as hcr
        p = probe()
        hidden, rank = (80, 16) if INTERPRET else (p.HIDDEN, 320)
        rows = hcr.PREFILL_ROWS
        gen = torch.Generator().manual_seed(5)
        h, out, inject, w = p.operands(rows, torch.device(DEVICE), gen, hidden, HC)
        down = (torch.randn(rank + HC, HC * hidden, generator=gen) * 0.02).bfloat16().to(DEVICE)
        up = (torch.randn(HC * hidden, rank, generator=gen) * 0.02).bfloat16().to(DEVICE)
        results = []
        for _, order in p.ORDERS:
            mine = h.clone()
            with p.forced(order):
                mixed, injection = hcr.site(mine, out, inject, w, 1e-6, HC, down, up, inject=True)
            results.append((mine, mixed, injection))
        (h_a, m_a, i_a), (h_b, m_b, i_b) = results
        self.assertTrue(torch.equal(h_a, h_b))
        self.assertTrue(torch.equal(m_a, m_b))
        self.assertTrue(torch.equal(i_a, i_b))


if __name__ == "__main__":
    unittest.main()
