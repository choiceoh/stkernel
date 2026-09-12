"""kernels/dense/smoothing: the fold that divides a norm and multiplies its readers' columns, exact up to bf16; the
groups the target and the drafter fold; the calibration filing its sums in the unsmoothed domain; the store's part."""
import tempfile
import unittest
from types import SimpleNamespace

import torch

from engine.kernels.dense.smoothing import CLAMP, fold, scales, smooth_hessian, smooth_weight


class SmoothingTests(unittest.TestCase):
    def test_the_fold_keeps_every_readers_product_and_evens_the_activation(self):
        from engine.profiles.glm53.net import rmsnorm
        g = torch.Generator().manual_seed(1)
        K = 256
        chan = torch.exp(torch.randn(K, generator=g) * 0.8); chan[:4] *= 20.0
        x = (torch.randn(64, K, generator=g) * chan).bfloat16()
        norm_w = (torch.rand(K, generator=g) + 0.5).bfloat16()
        readers = [(torch.randn(n, K, generator=g) * 0.05).bfloat16() for n in (96, 32)]
        before = [torch.nn.functional.linear(rmsnorm(x, norm_w, 1e-6), w).float() for w in readers]
        s = scales((x.float().abs().amax(0)), readers)
        self.assertEqual(tuple(s.shape), (K,))
        self.assertTrue(bool((s >= CLAMP[0]).all() and (s <= CLAMP[1]).all()))
        self.assertGreater(float(s[:4].mean()), float(s[4:].mean()), "outlier channels carry a bigger factor")
        self.assertTrue(bool((torch.log2(s) == torch.log2(s).round()).all()), "powers of two: the fold changes no product")
        s_eff = fold(norm_w, s)
        self.assertTrue(torch.equal(s_eff, s), "dividing a bf16 weight by a power of two is exact, so the undo is the factor itself")
        smoothed = [smooth_weight(w, s_eff) for w in readers]
        h = rmsnorm(x, norm_w, 1e-6)
        after = [torch.nn.functional.linear(h, w).float() for w in smoothed]
        for a, b in zip(after, before):
            self.assertTrue(torch.equal(a, b), "bit-identical: every product is the same number")
        peaks = h.float().abs().amax(0)
        self.assertLess(float(peaks.max() / peaks.median()), float((x.float().abs().amax(0)).max() / (x.float().abs().amax(0)).median()),
                        "the divided input's channel peaks are closer together")
        H = torch.randn(K, K, generator=g); H = H @ H.T
        Hs = smooth_hessian(H, s_eff)
        torch.testing.assert_close(Hs[3, 7], H[3, 7] / s_eff[3] / s_eff[7])

    def test_the_target_folds_exactly_the_readers_of_each_norm_output(self):
        from engine.base.comm import Comm
        from engine.profiles.glm53 import lanes
        from engine.profiles.glm53.net import Glm53Net
        from tests.test_engine_glm53 import tiny_facts
        F = tiny_facts()                                                    # kinds ("kda", "dsa", "dsa"), dense (0, 1, 2)
        net = Glm53Net(F, Comm(4, 0), lanes.reference(), layers=[0, 1, 2])
        groups = net.smoothing_groups()
        self.assertIn(("L0.in_norm", ["L0.kda.in_proj"], []), groups)
        self.assertIn(("L1.in_norm", ["L1.mla.qkv_a"], ["L1.idx.wk", "L1.idx.gate"]), groups)
        self.assertIn(("L1.mla.q_a_norm", ["L1.mla.q_b", "L1.idx.wq_b"], []), groups)
        self.assertIn(("L2.post_norm", ["L2.mlp.gate_up"], []), groups)
        self.assertEqual(len(groups), 8)
        # a MoE layer's post_norm is never folded: its experts and router read it too
        moe = SimpleNamespace(is_dsa=lambda L: False, is_moe=lambda L: True)
        net_moe = SimpleNamespace(F=moe, layers=[5])
        self.assertEqual(Glm53Net.smoothing_groups(net_moe), [("L5.in_norm", ["L5.kda.in_proj"], [])])
        g = torch.Generator().manual_seed(2)
        H, Q = F.hidden, F.q_lora
        net.p = {"L1.in_norm": (torch.rand(H, generator=g) + 0.5).bfloat16(), "L1.mla.qkv_a": (torch.randn(Q + F.kv_lora, H, generator=g) * 0.05).bfloat16(),
                 "L1.idx.wk": (torch.randn(F.idx_dim, H, generator=g) * 0.05).bfloat16(), "L1.idx.gate": (torch.randn(F.idx_dim, H, generator=g) * 0.05).bfloat16(),
                 "L1.mla.q_a_norm": torch.ones(Q).bfloat16(), "L1.mla.q_b": (torch.randn(8, Q, generator=g) * 0.05).bfloat16(), "L1.idx.wq_b": (torch.randn(8, Q, generator=g) * 0.05).bfloat16()}
        names = net.dense_weight_names(net.p)
        amax = {names["L1.mla.qkv_a"]: torch.rand(H, generator=g) * 4 + 0.1, names["L1.mla.q_b"]: torch.rand(Q, generator=g) + 0.1}
        originals = {k: v.clone() for k, v in net.p.items()}
        smoothed = net.smooth_inputs(lambda name: amax.get(name))
        self.assertEqual(set(smoothed), {"L1.mla.qkv_a", "L1.mla.q_b", "L1.idx.wq_b"})
        self.assertFalse(torch.equal(net.p["L1.in_norm"], originals["L1.in_norm"]))
        self.assertFalse(torch.equal(net.p["L1.idx.wk"], originals["L1.idx.wk"]), "a bf16 reader is rescaled in place")
        x = torch.randn(16, H, generator=g).bfloat16()
        from engine.profiles.glm53.net import rmsnorm
        for key in ("L1.mla.qkv_a", "L1.idx.wk", "L1.idx.gate"):
            w = smoothed[key][0] if key in smoothed else net.p[key]
            self.assertTrue(torch.equal(torch.nn.functional.linear(rmsnorm(x, net.p["L1.in_norm"], 1e-6), w),
                                        torch.nn.functional.linear(rmsnorm(x, originals["L1.in_norm"], 1e-6), originals[key])), key)

    def test_the_drafter_folds_the_block_path_and_leaves_the_context_projection(self):
        from tests.test_engine_drafter import DrafterTests
        d, field, dev = DrafterTests.make_full_drafter(DrafterTests(), seed=3)
        F, t = d.F, d.F.k + 1
        originals = {k: v.clone() for k, v in d.p.items()}
        g = torch.Generator().manual_seed(4)
        amax = torch.rand(F.hidden, generator=g) * 4 + 0.1
        ids = torch.tensor([7, F.mask_id, F.mask_id, F.mask_id], device=dev); positions = torch.arange(5, 9, device=dev)
        before = d.block(ids, positions, field[1], 5).float()
        weights, factors = d.smoothing_plan(lambda name: amax)
        self.assertEqual(set(factors), {"layers.0.input_layernorm.weight", "layers.0.post_attention_layernorm.weight"})
        self.assertEqual(set(weights), {"layers.0.attention_conv.kernel_projection.weight", "layers.0.self_attn.q_proj.weight", "layers.0.self_attn.k_proj.weight",
                                        "layers.0.self_attn.v_proj.weight", "layers.0.mlp_conv.kernel_projection.weight", "layers.0.mlp.gate_proj.weight", "layers.0.mlp.up_proj.weight"})
        for k in weights:
            self.assertTrue(torch.equal(d.p[k], originals[k]), "the source weights stay: the context projection reads k/v unsmoothed")
        d.p.update(weights)                                                 # the block path on the smoothed readers
        after = d.block(ids, positions, field[1], 5).float()
        self.assertTrue(torch.equal(after, before), "bit-identical through the grouped conv and the attention")

    def test_the_calibration_files_peaks_and_undoes_its_smoothing(self):
        from engine.kernels.dense.calibration import Calibration
        from engine.kernels.dense.store import Need, PackStore
        c = Calibration("cpu", budget_bytes=1 << 20)
        layer = SimpleNamespace(cols=32, name="T/model.layers.0.x", observer=None)
        s = torch.rand(32) + 0.5
        c.attach(layer.name, layer, PackStore.tiles(layer.name, 32), small_rows=True, unsmooth=s)
        c.arm()
        x = torch.randn(8, 32).bfloat16()
        layer.observer(x, None)
        torch.testing.assert_close(c.amax[layer.name], x.float().abs().amax(0))
        with tempfile.TemporaryDirectory() as tmp:
            c.save(tmp, rank=0)
            blob = torch.load(PackStore(tmp, 0).calibration_path(layer.name))
            torch.testing.assert_close(blob["amax"], x.float().abs().amax(0) * s)
            torch.testing.assert_close(blob["H"], (x.float().T @ x.float()) * s[:, None] * s[None, :], atol=1e-3, rtol=1e-3)
            store = PackStore(tmp, 0)
            torch.testing.assert_close(store.amax(layer.name), blob["amax"])
            self.assertEqual(store.missing_calibration(layer.name, 32), [])
            torch.save({"H": blob["H"], "ntok": 8, "name": layer.name}, store.calibration_path(layer.name))
            self.assertEqual(store.missing_calibration(layer.name, 32), [Need(layer.name, 0, 32, hessian=False)],
                             "a blob without peaks is summed again -- for the peaks alone, its Hessian stays")
            self.assertIsNone(store.amax(layer.name))
            self.assertTrue(store.calibrated(layer.name), "its Hessian still packs meanwhile")


if __name__ == "__main__":
    unittest.main()
