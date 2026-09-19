"""engine/kernels/router_fp32.router_logits_mma: the router projection on the tensor cores, FP32 accumulated and returned.

A BF16 activation times a BF16 weight is exact in FP32, so for Qwen3.8's router -- its weights are the checkpoint's
BF16 gates -- a BF16 MMA with FP32 sums is the IEEE FP32 projection up to the accumulation's rounding, which an FP32
GEMM has as well; an FP32 weight splits into three BF16 terms. Held to the float64 product at #1286's bounds, the
near-tie that BF16 logits lose kept, the split's use for weights with more than eight bits, and -- on a GPU -- a row
scoring the same bits in a decode step's launch and a prefill chunk's.

    TRITON_INTERPRET=1 CUDA_VISIBLE_DEVICES= python3 -m unittest tests.test_engine_router_mma
"""
import importlib.util
import os
import unittest
from types import SimpleNamespace as NS

torch = None
if importlib.util.find_spec("torch"):
    import torch
READY = torch is not None and importlib.util.find_spec("triton") is not None
INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
RUNS = READY and (INTERPRET or (torch is not None and torch.cuda.is_available()))
DEVICE = "cuda" if RUNS and not INTERPRET else "cpu"
EXPERTS, HIDDEN = (64, 256) if INTERPRET else (512, 2560)


def draw(seed, rows, *, fp32_weights=False):
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(rows, HIDDEN, generator=gen).bfloat16()
    w = torch.randn(EXPERTS, HIDDEN, generator=gen) * 0.03
    w = w if fp32_weights else w.bfloat16()
    return x.to(DEVICE), w.to(DEVICE)


@unittest.skipUnless(RUNS, "CUDA or the Triton interpreter required")
class RouterMMATests(unittest.TestCase):
    def test_it_is_the_fp64_product_at_the_ieee_router_s_bounds(self):
        from engine.kernels.router_fp32 import router_logits_mma
        for rows in (1, 5, 70):
            x, w = draw(rows, rows)
            got = router_logits_mma(x, w)
            ref = x.double() @ w.double().T
            with self.subTest(rows=rows):
                self.assertEqual((got.dtype, tuple(got.shape)), (torch.float32, (rows, EXPERTS)))
                torch.testing.assert_close(got.double(), ref, atol=2e-5, rtol=2e-5)
                # the FP32 widening of the same BF16 weights gives the same logits through the split's high term
                torch.testing.assert_close(router_logits_mma(x, w.float()), got, atol=0, rtol=0)

    def test_a_near_tie_bf16_logits_lose_is_kept(self):
        """#1286's case: two experts 2^-9 apart on logits of about one, which a BF16 logit rounds together."""
        from engine.kernels.router_fp32 import router_logits_mma
        x, w = draw(3, 4)
        x[0].zero_()
        x[0, :2] = 1
        w[:, :2] = -1
        w[0, :2] = torch.tensor([1.0, 0.0])
        w[1, :2] = torch.tensor([1.0, 2.0 ** -9])
        y = router_logits_mma(x, w)
        self.assertEqual(float(y[0, 1] - y[0, 0]), 2.0 ** -9)
        self.assertEqual(int(y[0].argmax()), 1)

    def test_fp32_weights_take_three_terms(self):
        """A weight with more than eight bits of mantissa: the three-term split holds the FP32 product; its high term
        alone (a BF16 router) does not."""
        from engine.kernels.router_fp32 import router_logits_mma
        x, w = draw(4, 9, fp32_weights=True)
        ref = x.double() @ w.double().T
        got = router_logits_mma(x, w)
        torch.testing.assert_close(got.double(), ref, atol=2e-5, rtol=2e-5)
        high = router_logits_mma(x, w.bfloat16())
        self.assertGreater(float((high.double() - ref).abs().max()), 20 * float((got.double() - ref).abs().max()))

    def test_a_row_scores_alike_in_a_decode_launch_and_a_prefill_launch(self):
        if DEVICE != "cuda":
            self.skipTest("the interpreter's numpy matmul sums an element by its matrix's shape; a GPU's MMA does not")
        from engine.kernels.router_fp32 import router_logits, router_logits_mma
        x, w = draw(5, 4096)
        whole = router_logits_mma(x, w)
        for rows in (1, 4, 16, 64, 700):
            with self.subTest(rows=rows):
                self.assertTrue(torch.equal(router_logits_mma(x[:rows].contiguous(), w), whole[:rows]))
        ieee = router_logits(x, w.float())
        torch.testing.assert_close(whole, ieee, atol=2e-5, rtol=2e-5)

    def test_it_refuses_what_it_cannot_take(self):
        from engine.kernels.router_fp32 import router_logits_mma
        x, w = draw(6, 2)
        for bad in ((x.half(), w), (x, w[:, :-1]), (x.t().contiguous().t()[:, :8], w[:, :8].t().contiguous().t())):
            with self.subTest(dtype=bad[0].dtype, shape=tuple(bad[1].shape)), self.assertRaises(ValueError):
                router_logits_mma(*bad)


@unittest.skipUnless(torch is not None, "requires torch")
class QwenRouterBindingTests(unittest.TestCase):
    def test_a_bf16_router_lane_admits_no_fp32_copy(self):
        from engine.profiles.qwen38.net import Qwen38Net
        F = NS(experts=4, hidden=8)
        gates = {f"L{L}.moe.gates": torch.randn(5, 8).bfloat16() for L in (0, 1)}
        gates["mtp.L0.moe.gates"] = torch.randn(5, 8).bfloat16()
        for bf16, nbytes in ((True, 0), (False, 3 * 4 * 8 * 4)):
            net = NS(F=F, layers=[0, 1], mtp=True, p=gates, lanes=NS(router_bf16=bf16), _router_weights={})
            self.assertEqual(Qwen38Net.router_nbytes(net), nbytes)
            arena = NS(carve=lambda n, name: torch.empty(n, dtype=torch.uint8))
            Qwen38Net.prepare_routers(net, arena)
            for prefix, w in net._router_weights.items():
                with self.subTest(bf16=bf16, prefix=prefix):
                    self.assertEqual(w.dtype, torch.bfloat16 if bf16 else torch.float32)
                    rows = 5 if bf16 else 4                  # the bf16 router carries the shared gate's row too
                    self.assertTrue(torch.equal(w.float(), gates[prefix + "moe.gates"][:rows].float()))

    def test_the_shared_gate_comes_out_of_the_router_launch(self):
        """With the gates' 513 rows bound, the router's last column is the shared gate's product: rounded to BF16 (a
        BF16 matmul's output) before the sigmoid, and the route sees only the experts' columns."""
        from engine.profiles.qwen38.net import Qwen38Net
        F = NS(experts=4, topk_experts=2)
        gates = torch.randn(5, 8).bfloat16()
        x = torch.randn(3, 8).bfloat16()
        seen = {}

        def route(scores, k):
            seen["columns"] = scores.shape[1]
            return torch.zeros(3, k, dtype=torch.int32), torch.ones(3, k)

        lanes = NS(router_logits=lambda x, w: x.double().matmul(w.double().t()).float(), route=route,
                   route_local=None, router_bf16=True)
        net = NS(F=F, p={"L0.moe.gates": gates}, lanes=lanes, _router_weights={"L0.": gates},
                 _experts={"L0.": lambda x, ids, w, **kw: x})
        _, gate = Qwen38Net._routed(net, "L0.", x, compact=True)
        want = torch.sigmoid((x.double() @ gates[4:].double().t()).float().bfloat16().float())
        self.assertEqual(seen["columns"], 4)
        self.assertTrue(torch.equal(gate, want))

    def test_the_served_lane_binds_the_mma_router(self):
        import inspect
        from engine.profiles.qwen38 import lanes
        source = inspect.getsource(lanes.served)
        self.assertIn("router_logits=on_main(router_fp32.router_logits_mma)", source)
        self.assertIn("router_bf16=True", source)
        self.assertFalse(lanes.reference().router_bf16)


if __name__ == "__main__":
    unittest.main()
