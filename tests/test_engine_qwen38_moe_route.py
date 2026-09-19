"""Qwen3.8's router in one launch, held to the torch composition it replaces (engine/QWEN38_CARRY.md M2).

A captured step routed every MoE layer -- 48 in the verify graph and the MTP head's one, in both graphs -- through
`engine/modules/moe.route_softmax_topk` (a widening, the softmax, the top-k, a sum, a division, a rounding), a widening
for the dispatcher and `lanes.local_routes`' eight small kernels. `engine/kernels/moe_route.softmax_topk` is one launch
a layer.

What is held: the weights are the composition's (FP32 arithmetic to a last-bits tolerance -- Triton's exp and this
program's reduction are not torch's -- and, rounded, never more than one BF16 step away); the experts are the
composition's wherever it has no tie, and at a tie the lowest ids of the tied set; the EP remap is the dispatcher
lane's. The shared expert's gate stays torch's sigmoid: it is consumed in FP32, and on a GPU Triton's sigmoid is not
torch's bytes (measured, kernels/moe_route's docstring). The idea is GLM's fused router (#789/#810/#779); GLM's own
router (sigmoid, groups, bias) is another formula and is untouched.

    docker exec -e TRITON_INTERPRET=1 -w <repo> stk-test python3 -m unittest tests.test_engine_qwen38_moe_route
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
E, K = 512, 10                                           # Qwen3.8's experts and routes a token


def reference(scores, k=K, experts=E):
    """(ids, unrounded FP32 weights, the FP32 probabilities) of engine/modules/moe.route_softmax_topk's arithmetic."""
    probs = torch.nn.functional.softmax(scores[:, :experts], dtype=torch.float, dim=-1)
    top, ids = torch.topk(probs, k, dim=-1)
    return ids, top / top.sum(dim=-1, keepdim=True), probs


@unittest.skipUnless(RUNS, "requires Triton with CUDA, or TRITON_INTERPRET=1")
class SoftmaxTopkTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(789)
        if INTERPRET:
            patch = mock.patch.object(torch.Tensor, "is_cuda", property(lambda tensor: True))
            patch.start()
            self.addCleanup(patch.stop)

    @staticmethod
    def scores(rows, columns=E + 1, scale=2.0):
        return (torch.randn(rows, columns, device=DEVICE) * scale).to(torch.bfloat16)

    def assert_routes(self, scores, ids, weights, k=K, experts=E):
        """The launch's routes against the composition's, tie-robustly: the same weights in order; every named expert
        holds the probability its place claims; no expert twice."""
        ref_ids, ref_w, probs = reference(scores, k, experts)
        # a few last FP32 bits: Triton's exp and this program's reduction are not torch's (8e-7 on an RTX 5050)
        torch.testing.assert_close(weights, ref_w, rtol=2e-6, atol=0)
        picked = probs.gather(1, ids.long())
        torch.testing.assert_close(picked / picked.sum(dim=-1, keepdim=True), ref_w, rtol=2e-6, atol=0)
        self.assertTrue(all(len(set(row)) == k for row in ids.tolist()))
        distinct = (probs.gather(1, ref_ids)[:, 1:] != probs.gather(1, ref_ids)[:, :-1]).all(dim=-1)
        boundary = probs.gather(1, ref_ids)[:, -1] > probs.scatter(1, ref_ids, -1.0).max(dim=-1).values
        clean = distinct & boundary                          # rows where torch.topk had no choice to make
        self.assertTrue(torch.equal(ids[clean].long(), ref_ids[clean]))
        return clean

    def test_the_routes_are_the_compositions(self):
        from engine.kernels import moe_route
        for rows in (1, 2, 8, 33):
            with self.subTest(rows=rows):
                scores = self.scores(rows)                  # BF16 logits tie often: 512 draws on a seven-bit fraction
                ids, weights = moe_route.softmax_topk(scores, K, experts=E, exact=True)
                self.assertEqual((ids.dtype, weights.dtype), (torch.int32, torch.float32))
                self.assert_routes(scores, ids, weights)

    def test_without_ties_the_experts_are_torchs_in_torchs_order(self):
        from engine.kernels import moe_route
        # every logit distinct and exact in BF16: a permutation of the multiples of 1/16 in [-16, 16)
        scores = torch.stack([torch.randperm(E, device=DEVICE) for _ in range(8)]).float().div(16).sub(16)
        scores = scores.to(torch.bfloat16)
        self.assertTrue(all(len(set(row)) == E for row in scores.float().tolist()))
        ids, weights = moe_route.softmax_topk(scores, K, exact=True)
        self.assertTrue(bool(self.assert_routes(scores, ids, weights).all()))
        self.assertTrue(torch.equal(ids.long(), reference(scores)[0]))

    def test_a_tie_names_the_lowest_experts_of_the_tied_set(self):
        from engine.kernels import moe_route
        scores = torch.full((2, E), -4.0, device=DEVICE, dtype=torch.bfloat16)
        scores[0, [400, 7, 300, 8]] = 3.0                  # four equal leaders, then a tie across the k-th place
        scores[0, 20:40] = 1.0
        scores[1, :] = 0.5                                 # every expert equal: the first k
        ids, weights = moe_route.softmax_topk(scores, K, exact=True)
        self.assertEqual(ids[0].tolist(), [7, 8, 300, 400, 20, 21, 22, 23, 24, 25])
        self.assertEqual(ids[1].tolist(), list(range(K)))
        torch.testing.assert_close(weights[1], torch.full((K,), 1 / K, device=DEVICE), rtol=1e-6, atol=0)
        self.assert_weights_sum_to_one(weights)

    def assert_weights_sum_to_one(self, weights):
        torch.testing.assert_close(weights.sum(dim=-1), torch.ones(weights.shape[0], device=weights.device),
                                   rtol=1e-6, atol=0)

    def test_the_served_weights_are_bf16_values_a_step_from_the_compositions_at_most(self):
        from engine.kernels import moe_route
        scores = self.scores(33)
        _, exact = moe_route.softmax_topk(scores, K, experts=E, exact=True)
        _, served = moe_route.softmax_topk(scores, K, experts=E)
        if not INTERPRET:                                   # the interpreter does not round BF16 as a GPU does
            self.assertTrue(torch.equal(served, served.to(torch.bfloat16).float()))
            self.assertTrue(torch.equal(served, exact.to(torch.bfloat16).float()))
        # BF16 keeps seven fraction bits: half a step when rounded to nearest, a whole one as the interpreter truncates
        torch.testing.assert_close(served, exact, rtol=2 ** -7 if INTERPRET else 2 ** -8, atol=0)

    def test_the_ep_remap_is_local_routes(self):
        from engine.kernels import moe_route
        from engine.profiles.qwen38.lanes import local_routes
        scores = self.scores(8)
        ids, weights = moe_route.softmax_topk(scores, K, experts=E, exact=True)
        for first, sentinel in ((0, None), (128, None), (384, 128)):
            with self.subTest(first=first, sentinel=sentinel):
                want_ids, want_w = local_routes(ids, weights, first, 128, sentinel)
                got_ids, got_w = moe_route.softmax_topk(scores, K, experts=E, first=first, local=128,
                                                        foreign=0 if sentinel is None else sentinel, exact=True)
                self.assertTrue(torch.equal(got_ids, want_ids))
                self.assertTrue(torch.equal(got_w, want_w))
                self.assertTrue(bool(((got_w == 0) == ((ids < first) | (ids >= first + 128))).all()))

    def test_the_gates_column_beside_the_experts_never_enters_the_softmax(self):
        from engine.kernels import moe_route
        scores = self.scores(8)
        scores[:, E] = 30.0                                 # would dominate every probability if it were read
        ids, weights = moe_route.softmax_topk(scores, K, experts=E, exact=True)
        self.assert_routes(scores, ids, weights)
        self.assertTrue(bool((ids < E).all()))

    def test_a_row_view_a_narrow_router_and_an_empty_step(self):
        from engine.kernels import moe_route
        scores = self.scores(6)
        ids, weights = moe_route.softmax_topk(scores[::2], K, experts=E, exact=True)      # strided rows
        self.assert_routes(scores[::2], ids, weights)
        small = self.scores(5, columns=9)                   # a width that is not a power of two, a gate beside it
        ids, weights = moe_route.softmax_topk(small, 3, experts=8, exact=True)
        self.assert_routes(small, ids, weights, k=3, experts=8)
        ids, weights = moe_route.softmax_topk(scores[:0], K, experts=E)
        self.assertEqual((tuple(ids.shape), tuple(weights.shape)), ((0, K), (0, K)))

    def test_it_refuses_what_it_does_not_compute(self):
        from engine.kernels import moe_route
        scores = self.scores(2)
        cases = {
            "fp16 scores": lambda: moe_route.softmax_topk(scores.half(), K),
            "strided columns": lambda: moe_route.softmax_topk(scores.t().contiguous().t(), K),
            "more routes than experts": lambda: moe_route.softmax_topk(scores, 9, experts=8),
            "more experts than columns": lambda: moe_route.softmax_topk(scores, K, experts=E + 2),
            "half a remap": lambda: moe_route.softmax_topk(scores, K, experts=E, first=128),
            "a foreign id past the sentinel": lambda: moe_route.softmax_topk(scores, K, experts=E, first=0, local=128,
                                                                             foreign=129),
        }
        for name, call in cases.items():
            with self.subTest(case=name), self.assertRaises(ValueError):
                call()


@unittest.skipUnless(RUNS, "requires Triton with CUDA, or TRITON_INTERPRET=1")
class LayerTests(unittest.TestCase):
    """net.Qwen38Net._moe hands a captured step's scores to the lane and the dispatcher its routes: the routes the
    composed layer built (route, then the dispatcher lane's local_routes), and the same layer output."""
    EXPERTS, LOCAL, TOPK, H = 8, 2, 3, 16

    def setUp(self):
        torch.manual_seed(810)
        if INTERPRET:
            patch = mock.patch.object(torch.Tensor, "is_cuda", property(lambda tensor: True))
            patch.start()
            self.addCleanup(patch.stop)

    def layer(self, x, gates, *, first, sentinel, fused, compact=False):
        from types import SimpleNamespace
        from engine.kernels import moe_route
        from engine.profiles.qwen38 import lanes
        from engine.profiles.qwen38.net import Qwen38Net
        seen = {}

        def experts(x, ids, weights, *, compact, local=False):
            if not local:                                   # the dispatcher lane's own remap (lanes.served's moe)
                ids, weights = lanes.local_routes(ids, weights, first, self.LOCAL, None if compact else sentinel)
            seen["routes"] = (ids, weights, local)
            return torch.zeros_like(x)

        def route_local(scores, k, *, experts, first_expert, w13, hidden):
            seen["shape"] = (tuple(scores.shape), w13.shape[0], hidden)
            return moe_route.softmax_topk(scores, k, experts=experts, first=first_expert, local=w13.shape[0],
                                          foreign=0 if sentinel is None else sentinel)

        table = SimpleNamespace(route=lanes.route_softmax_topk, route_local=route_local if fused else None,
                                swiglu=lambda fused_x, pad_to=None: fused_x[:, :self.H],
                                moe_finish=lambda routed, shared, gate: (seen.__setitem__("gate", gate), shared)[1])
        net = SimpleNamespace(F=SimpleNamespace(experts=self.EXPERTS, topk_experts=self.TOPK), lanes=table,
                              p={"L0.moe.gates": gates, "L0.moe.w13": torch.empty(self.LOCAL, 4, 2)},
                              _experts={"L0.": experts}, first_expert=first, dense={},
                              linear=lambda x, name: x, comm=SimpleNamespace(all_reduce=lambda t: t))
        Qwen38Net._moe(net, "L0.", x, compact=compact)
        return seen

    def test_a_captured_steps_routes_are_the_composed_layers_and_its_gate_is_torchs(self):
        x = torch.randn(5, self.H, device=DEVICE).to(torch.bfloat16)
        gates = torch.randn(self.EXPERTS + 1, self.H, device=DEVICE).to(torch.bfloat16)
        for first, sentinel in ((0, None), (2, None), (6, self.LOCAL)):
            with self.subTest(first=first, sentinel=sentinel):
                composed = self.layer(x, gates, first=first, sentinel=sentinel, fused=False)
                fused = self.layer(x, gates, first=first, sentinel=sentinel, fused=True)
                self.assertEqual(fused["shape"], ((5, self.EXPERTS), self.LOCAL, self.H))
                (ids, w, was_local), (want_ids, want_w, composed_local) = fused["routes"], composed["routes"]
                self.assertEqual((was_local, composed_local), (True, False))
                self.assertTrue(torch.equal(ids, want_ids))
                self.assertTrue(torch.equal(w == 0, want_w == 0))
                # FP32 router scores keep FP32 weights on both paths, including the fused EP remap.
                torch.testing.assert_close(w, want_w, rtol=2e-6, atol=0)
                self.assertTrue(torch.equal(fused["gate"], composed["gate"]))      # torch's sigmoid on both paths
                self.assertTrue(torch.equal(fused["gate"], torch.sigmoid((x @ gates.t())[:, self.EXPERTS:].float())))

    def test_an_eager_step_keeps_the_global_routes(self):
        x = torch.randn(5, self.H, device=DEVICE).to(torch.bfloat16)
        gates = torch.randn(self.EXPERTS + 1, self.H, device=DEVICE).to(torch.bfloat16)
        seen = self.layer(x, gates, first=2, sentinel=None, fused=True, compact=True)
        self.assertNotIn("shape", seen)                     # the dispatcher lane counts its pairs from global ids
        self.assertFalse(seen["routes"][2])


@unittest.skipUnless(RUNS, "requires Triton with CUDA, or TRITON_INTERPRET=1")
class CompactPairsTests(unittest.TestCase):
    """An eager step's compact MoE: the remap and the mask (`compact_routes`) and the pairs' gathers (`pair_rows`) in
    one launch each, bit for bit the torch launches they replace."""

    def routes(self, rows, seed):
        gen = torch.Generator().manual_seed(seed)
        ids = torch.stack([torch.randperm(E, generator=gen)[:K] for _ in range(rows)]).to(torch.int32).to(DEVICE)
        weights = torch.rand(rows, K, generator=gen).to(DEVICE)
        return ids, weights

    def test_the_remap_and_the_mask_are_local_routes_and_the_range(self):
        from engine.kernels import moe_route
        from engine.profiles.qwen38.lanes import local_routes
        for rows, first in ((1, 0), (37, 128), (300, 384)):
            ids, weights = self.routes(rows, rows)
            local_ids, w, mine = moe_route.compact_routes(ids, weights, first, 128)
            want_ids, want_w = local_routes(ids, weights, first, 128)
            shifted = ids - first
            with self.subTest(rows=rows, first=first):
                self.assertTrue(torch.equal(local_ids, want_ids) and torch.equal(w, want_w))
                self.assertTrue(torch.equal(mine, (shifted >= 0) & (shifted < 128)))

    def test_the_pairs_are_the_index_gathers(self):
        from engine.kernels import moe_route
        from engine.profiles.qwen38.lanes import local_routes
        gen = torch.Generator().manual_seed(5)
        for rows, h in ((40, 2560), (9, 100)):
            x = torch.randn(rows, h, generator=gen).to(torch.bfloat16).to(DEVICE)
            ids, weights = self.routes(rows, h)
            local_ids, w = local_routes(ids, weights, 128, 128)
            token, route = ((ids >= 128) & (ids < 256)).nonzero(as_tuple=True)
            xp, ip, wp = moe_route.pair_rows(x, local_ids, w, token, route)
            with self.subTest(rows=rows, h=h):
                self.assertTrue(torch.equal(xp, x.index_select(0, token)))
                self.assertTrue(torch.equal(ip, local_ids[token, route][:, None]))
                self.assertTrue(torch.equal(wp, w[token, route][:, None]))
        empty = torch.zeros(0, dtype=torch.int64, device=DEVICE)
        xp, ip, wp = moe_route.pair_rows(x, local_ids, w, empty, empty)
        self.assertEqual((tuple(xp.shape), tuple(ip.shape), tuple(wp.shape)), ((0, h), (0, 1), (0, 1)))

    def test_the_served_compact_moe_uses_both(self):
        served = (ROOT / "engine/profiles/qwen38/lanes.py").read_text(encoding="utf-8")
        body = served[served.index("    def moe(x, ids, weights, w13, w13_sf, w2, w2_sf, *, scales, first_expert, "
                                   "compact=False, local=False):"):]
        self.assertIn("moe_route.compact_routes(", body)
        self.assertIn("token, route = mine.nonzero(as_tuple=True)", body)
        self.assertIn("moe_route.pair_rows(x, local_ids, w, token, route)", body)


@unittest.skipUnless(torch is not None, "requires torch")
class LaneTests(unittest.TestCase):
    def test_the_reference_table_composes_and_the_served_table_binds_the_kernel(self):
        from engine.profiles.qwen38 import lanes
        self.assertIsNone(lanes.reference().route_local)     # the oracle's layer stays the composition
        served = (ROOT / "engine/profiles/qwen38/lanes.py").read_text(encoding="utf-8")
        self.assertIn("route_local=on_main(route_local)", served)
        self.assertIn("moe_route.softmax_topk(scores, k, experts=experts, first=first_expert", served)
        net = (ROOT / "engine/profiles/qwen38/net.py").read_text(encoding="utf-8")
        self.assertIn("lanes.route_local(scores, F.topk_experts", net)
        self.assertIn("compact=False, local=True)", net)
        self.assertIn("gate = torch.sigmoid(shared_score.float())", net)      # never the launch's exp


if __name__ == "__main__":
    unittest.main()
