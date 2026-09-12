"""Captured calibration retains every real input across flushes and shape changes."""
import tempfile
import unittest

import torch

from engine.kernels.dense.calibration import Calibration, GRAM_ROWS
from engine.kernels.dense.store import PackStore


class BudgetTests(unittest.TestCase):
    def test_staging_is_charged_before_attaching_a_layer(self):
        missing = PackStore.tiles("draft.fc", 20480)
        extra = Calibration.nbytes(missing, max_decode_rows=28) - Calibration.nbytes(missing)
        self.assertEqual(extra, (GRAM_ROWS + 27) * 20480 * 4)
        bf16_extra = Calibration.nbytes(missing, max_decode_rows=28, input_dtype=torch.bfloat16) - Calibration.nbytes(missing)
        self.assertEqual(bf16_extra * 2, extra)
        self.assertLess(Calibration.nbytes(missing, max_decode_rows=28), 2 << 30)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA graph and Triton execution required")
class BufferedGramTests(unittest.TestCase):
    def test_graph_replay_masks_wraps_shape_changes_and_partial_save(self):
        self._check_graph(torch.float32)

    def test_declared_bf16_inputs_preserve_all_rows_and_fp32_accumulation(self):
        self._check_graph(torch.bfloat16)

    def _check_graph(self, dtype):
        class Layer:
            observer = None
            input_dtype = dtype
        for width in (64, 95):
            with self.subTest(width=width):
                layer = Layer()
                c = Calibration("cuda", max_decode_rows=28)
                c.attach("x", layer, PackStore.tiles("x", width), small_rows=True)
                inputs = {n: torch.zeros(n, width, device="cuda", dtype=dtype) for n in (1, 7, 28)}
                masks = {n: torch.zeros(n, dtype=torch.bool, device="cuda") for n in inputs}
                graphs = {}
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for n, x in inputs.items():
                        layer.observer(x, masks[n])
                torch.cuda.current_stream().wait_stream(stream)
                for n, x in inputs.items():
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        layer.observer(x, masks[n])
                    graphs[n] = graph
                for graph in graphs.values():
                    graph.replay()
                self.assertEqual(c.progress(), 0)
                self.assertEqual(int(c.staging["x"][1]), 0)
                c.arm()
                kept = []
                gen = torch.Generator(device="cuda").manual_seed(812)
                for step in range(65):
                    n = (1, 7, 28)[step % 3]
                    # FP32 inputs exercise tf32x3, not only exactly representable BF16.
                    x = torch.randn(n, width, generator=gen, device="cuda", dtype=dtype)
                    mask = torch.arange(n, device="cuda") % 3 != step % 3
                    inputs[n].copy_(x)
                    masks[n].copy_(mask)
                    graphs[n].replay()
                    kept.append(x[mask])
                    if step == 32:
                        large = torch.randn(71, width, generator=gen, device="cuda", dtype=dtype)
                        layer.observer(large, None)  # eager prefill flushes the pending decode tail first
                        kept.append(large)
                expected_rows = torch.cat(kept)
                self.assertGreater(int(c.staging["x"][1]), 0)
                with tempfile.TemporaryDirectory() as root:
                    path, = c.save(root, 0)
                    saved = torch.load(path, weights_only=True)
                oracle = expected_rows.double().T @ expected_rows.double()
                torch.testing.assert_close(saved["H"].double(), oracle.cpu(), rtol=3e-5, atol=2e-4)
                self.assertEqual(saved["ntok"], len(expected_rows))
                # The persisted peak contract is FP32 even when the exact
                # input values arrive in BF16. Keep dtype checking enabled.
                torch.testing.assert_close(saved["amax"], expected_rows.float().abs().amax(0).cpu(), rtol=0, atol=0)
                self.assertEqual(int(c.staging["x"][1]), 0)
                before = c.H["x"].clone()
                c.flush()
                torch.testing.assert_close(c.H["x"], before, rtol=0, atol=0)

    def test_small_unmasked_target_rows_are_still_excluded(self):
        class Layer:
            observer = None
        layer = Layer()
        c = Calibration("cuda", max_decode_rows=28)
        c.attach("target", layer, PackStore.tiles("target", 64), small_rows=False)
        c.arm()
        layer.observer(torch.ones(28, 64, device="cuda"), None)
        self.assertEqual(c.staging, {})
        self.assertEqual(c.progress(), 0)

    def test_declared_bf16_staging_refuses_precision_loss(self):
        class Layer:
            observer = None
            input_dtype = torch.bfloat16
        layer = Layer()
        c = Calibration('cuda', max_decode_rows=28)
        c.attach('x', layer, PackStore.tiles('x', 64), small_rows=True)
        c.arm()
        for x, mask in ((torch.randn(7, 64, device='cuda'), None),
                        (torch.randn(7, 64, device='cuda').bfloat16(), torch.full((7,), .5, device='cuda'))):
            with self.assertRaises(ValueError):
                layer.observer(x, mask)
        self.assertEqual(c.progress(), 0)

    def test_peak_only_masked_strided_rows_need_no_hessian_staging(self):
        class Layer:
            observer = None
        layer = Layer()
        c = Calibration("cuda", max_decode_rows=28)
        c.attach("x", layer, [("peak", 3, 95, False)], small_rows=True)
        x = torch.randn(7, 128, device="cuda", dtype=torch.bfloat16)
        mask = torch.tensor([1, 0, 1, 0, 1, 1, 0], device="cuda", dtype=torch.bool)
        layer.observer(x, mask)
        self.assertEqual(c.progress(), 0)
        c.arm()
        layer.observer(x, mask)
        layer.observer(x, None)
        self.assertEqual(c.progress(), 11)
        self.assertEqual(c.H, {})
        self.assertEqual(c.staging, {})
        torch.testing.assert_close(c.amax["peak"], x[:, 3:98].float().abs().amax(0), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
