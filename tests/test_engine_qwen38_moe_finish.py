"""Qwen3.8's MoE output in one launch, with the torch composition's bytes (engine/QWEN38_CARRY.md M1).

`net._moe` finished every MoE layer -- 48 in the verify graph and the MTP head's one -- with
`(routed.float() + shared.float() * gate).to(x.dtype)`: two widenings, a product, a sum and a rounding, five launches
before the all-reduce. `engine/kernels/moe_output.gated_sum` does the same FP32 arithmetic in one launch, compiled
without fused multiply-add so the product and the sum round as torch's separate kernels do. The idea is GLM's MoE
output finalizer (#904/#906); GLM's own `combine` (the FP32 scatter plane's contract) is untouched.

    docker exec -e TRITON_INTERPRET=1 -w <repo> stk-test python3 -m unittest tests.test_engine_qwen38_moe_finish
"""
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
DEVICE = "cpu" if INTERPRET else "cuda"


@unittest.skipUnless(RUNS, "requires Triton with CUDA, or TRITON_INTERPRET=1")
class GatedSumTests(unittest.TestCase):
    H = 2560                                             # Qwen3.8's hidden width

    def setUp(self):
        torch.manual_seed(904)
        if INTERPRET:
            patch = mock.patch.object(torch.Tensor, "is_cuda", property(lambda tensor: True))
            patch.start()
            self.addCleanup(patch.stop)

    def operands(self, rows):
        routed = (torch.randn(rows, self.H, device=DEVICE) * 2).to(torch.bfloat16)
        shared = (torch.randn(rows, self.H, device=DEVICE) * 3).to(torch.bfloat16)
        gate = torch.sigmoid(torch.randn(rows, 1, device=DEVICE).to(torch.bfloat16).float())
        return routed, shared, gate

    @staticmethod
    def triton_bf16(x):
        """Triton's own FP32 -> BF16 conversion of `x`: the only step the served launch adds after the sum. On a GPU it
        is torch's rounding (the GPU case below, and GLM's moe_output rounding-boundary test); Triton's CPU interpreter
        does not round BF16 as a GPU does, so the CPU cases compare against this rather than against torch's cast."""
        import triton
        import triton.language as tl

        @triton.jit
        def cast(X, Y, sX, sY, H: tl.constexpr, BLOCK: tl.constexpr):
            r = tl.program_id(0)
            c = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
            m = c < H
            tl.store(Y + r * sY + c, tl.load(X + r * sX + c, m, 0).to(tl.bfloat16), m)

        y = torch.empty(x.shape, device=x.device, dtype=torch.bfloat16)
        cast[(x.shape[0], triton.cdiv(x.shape[1], 512))](x, y, x.stride(0), y.stride(0), H=x.shape[1], BLOCK=512)
        return y

    def test_the_sum_is_the_torch_composition_byte_for_byte(self):
        from engine.kernels import moe_output
        for rows in (1, 2, 4, 33):
            with self.subTest(rows=rows):
                routed, shared, gate = self.operands(rows)
                exact = torch.empty(rows, self.H, device=DEVICE, dtype=torch.float32)
                moe_output.gated_sum(routed, shared, gate, out=exact)
                self.assertTrue(torch.equal(exact, routed.float() + shared.float() * gate))

    def test_the_served_output_is_the_sum_rounded_once(self):
        from engine.kernels import moe_output
        for rows in (1, 33):
            with self.subTest(rows=rows):
                routed, shared, gate = self.operands(rows)
                self.assertTrue(torch.equal(moe_output.gated_sum(routed, shared, gate),
                                            self.triton_bf16(routed.float() + shared.float() * gate)))

    @unittest.skipIf(INTERPRET or torch is None or not torch.cuda.is_available(), "the GPU's BF16 rounding")
    def test_on_a_gpu_the_rounding_is_torchs(self):
        from engine.kernels import moe_output
        routed, shared, gate = self.operands(8)
        self.assertTrue(torch.equal(moe_output.gated_sum(routed, shared, gate),
                                    (routed.float() + shared.float() * gate).to(torch.bfloat16)))

    def test_a_row_view_and_an_empty_step(self):
        from engine.kernels import moe_output
        routed, shared, gate = self.operands(6)
        exact = torch.empty(3, self.H, device=DEVICE, dtype=torch.float32)
        moe_output.gated_sum(routed[::2], shared[::2], gate[::2], out=exact)      # strided rows, packed columns
        self.assertTrue(torch.equal(exact, (routed.float() + shared.float() * gate)[::2]))
        empty = moe_output.gated_sum(routed[:0], shared[:0], gate[:0])
        self.assertEqual(tuple(empty.shape), (0, self.H))

    def test_it_refuses_what_it_does_not_compute(self):
        from engine.kernels import moe_output
        routed, shared, gate = self.operands(2)
        cases = {
            "fp32 routed": (routed.float(), shared, gate),
            "a per-element gate": (routed, shared, gate.expand(2, self.H).contiguous()),
            "a bf16 gate": (routed, shared, gate.to(torch.bfloat16)),
            "strided columns": (routed.t().contiguous().t(), shared, gate),
        }
        for name, args in cases.items():
            with self.subTest(case=name), self.assertRaises(ValueError):
                moe_output.gated_sum(*args)
        with self.assertRaises(ValueError):
            moe_output.gated_sum(routed, shared, gate, out=routed)


@unittest.skipUnless(torch is not None, "requires torch")
class LaneTests(unittest.TestCase):
    def test_the_reference_lane_is_the_composition_the_layer_used(self):
        from engine.profiles.qwen38 import lanes
        routed = torch.randn(3, 8).to(torch.bfloat16)
        shared = torch.randn(3, 8).to(torch.bfloat16)
        gate = torch.rand(3, 1)
        self.assertTrue(torch.equal(lanes.reference().moe_finish(routed, shared, gate),
                                    (routed.float() + shared.float() * gate).to(torch.bfloat16)))

    def test_the_layer_finishes_through_the_lane_and_the_served_table_binds_the_kernel(self):
        net = (ROOT / "engine/profiles/qwen38/net.py").read_text()
        self.assertIn("Qwen38Net._sum(self, self.lanes.moe_finish(routed, shared, gate))", net)   # or the gated packets (X2)
        self.assertNotIn("routed.float() + shared.float() * gate", net)
        self.assertIn("moe_finish=on_main(moe_output.gated_sum)", (ROOT / "engine/profiles/qwen38/lanes.py").read_text())


if __name__ == "__main__":
    unittest.main()
