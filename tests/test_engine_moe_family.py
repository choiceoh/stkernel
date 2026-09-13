"""engine/modules/moe as a family: one router, one expert loop, the axes of MoE's docstring. Each model's block is
held to the transformers implementation that defines it, on the CPU: glm5_next (sigmoid + correction bias, clamped
swiglu, one wide shared expert, scaling on the weights; and its router with groups), deepseek_v3 (noaux_tc grouped
choice), minimax_m3_vl (model-dtype router, swigluoai, scaling on the output), inkling (shared experts scored by the
router as a sink, route_scale x global_scale); the dense MLPs of each. Qwen3.8's softmax router with its sigmoid-gated
shared expert is held by tests/test_engine_composition.py."""
import importlib.util
import unittest

torch = None
if importlib.util.find_spec("torch") is not None:
    import torch


def _present(model: str) -> bool:
    try:
        return importlib.util.find_spec(f"transformers.models.{model}") is not None
    except Exception:
        return False


HID, N = 64, 24


def draw(mod, seed=0):
    torch.manual_seed(seed)
    with torch.no_grad():
        for name, p in list(mod.named_parameters()) + list(mod.named_buffers()):
            if name.endswith("global_scale"):
                p.uniform_(0.5, 1.5)
            elif name.endswith("e_score_correction_bias"):
                p.normal_(0, 0.5)
            elif p.ndim >= 2:
                p.normal_(0, p.shape[-1] ** -0.5)
            else:
                p.normal_(0, 0.5)
    return mod


def source_of(mod):
    sd = {k: v.detach().clone() for k, v in mod.state_dict().items()}

    def source(hf):
        if f"{hf}.weight" in sd:
            return sd[f"{hf}.weight"]
        if hf in sd:
            return sd[hf]
        raise KeyError(hf)
    return source


def family(scheme, mod, **axes):
    from engine.modules.moe import MoE, experts_of, named, shared_of
    src = source_of(mod)
    experts, shared = experts_of(src), shared_of(scheme, src)
    return MoE(weights=lambda layer, name: named(scheme, src)(name), expert=lambda layer, e: experts(e),
               shared_expert=lambda layer, i: shared(i), **axes)


class Held(unittest.TestCase):
    def close(self, got, want, tol=1e-5):
        self.assertEqual(tuple(got.shape), tuple(want.shape))
        diff = float((got.float() - want.float()).abs().max())
        self.assertLessEqual(diff, tol, f"max |got - want| = {diff}")


@unittest.skipUnless(torch is not None and _present("glm5_next"), "requires transformers with glm5_next on the path")
class GLM5NextTests(Held):
    def build(self, groups=(1, 1), limit=0.8, seed=0):
        from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig
        from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextMLP, Glm5NextTextMoE
        cfg = Glm5NextTextConfig(vocab_size=256, hidden_size=HID, num_hidden_layers=4, num_attention_heads=4,
                                 num_key_value_heads=4, n_routed_experts=8, num_experts_per_tok=3, moe_intermediate_size=16,
                                 n_shared_experts=1, routed_scaling_factor=2.5, n_group=groups[0], topk_group=groups[1],
                                 norm_topk_prob=True, swiglu_limit=limit, intermediate_size=32, hidden_act="silu",
                                 linear_num_heads=4, linear_head_dim=16)
        return cfg, draw(Glm5NextTextMoE(cfg).eval(), seed), draw(Glm5NextTextMLP(cfg).eval(), seed + 1)

    def test_the_block_is_the_model(self):
        for groups in ((1, 1), (4, 2)):
            cfg, mod, _ = self.build(groups)
            feat = family("glm5_next", mod, experts=8, topk=3, score="sigmoid", bias=True, normalize=True, scaling=2.5,
                          groups=None if groups == (1, 1) else groups, shared=1, shared_mode="plain",
                          activation=("swiglu_clamped", 0.8))
            x = torch.randn(N, HID, generator=torch.Generator().manual_seed(1))
            with torch.no_grad():
                self.close(feat(0, x), mod(x[None])[0])

    def test_the_dense_mlp_is_the_model(self):
        from engine.modules.moe import Dense
        cfg, _, mlp = self.build()
        src = source_of(mlp)
        table = {"gate_up": lambda: torch.cat([src("gate_proj"), src("up_proj")]), "down": lambda: src("down_proj")}
        dense = Dense(activation=("swiglu_clamped", 0.8), weights=lambda layer, name: table[name]())
        x = torch.randn(N, HID, generator=torch.Generator().manual_seed(2))
        with torch.no_grad():
            self.close(dense(0, x), mlp(x))
        self.assertGreater(float(torch.nn.functional.linear(x, src("gate_proj")).abs().max()), 0.8)   # the clamp bites


@unittest.skipUnless(torch is not None and _present("deepseek_v3"), "requires transformers with deepseek_v3 on the path")
class DeepSeekTests(Held):
    def test_the_grouped_block_is_the_model(self):
        from transformers.models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config
        from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3MLP, DeepseekV3MoE
        from engine.modules.moe import Dense
        cfg = DeepseekV3Config(vocab_size=256, hidden_size=HID, num_hidden_layers=2, num_attention_heads=4,
                               num_key_value_heads=4, n_routed_experts=8, num_experts_per_tok=3, n_group=4, topk_group=2,
                               moe_intermediate_size=16, n_shared_experts=1, routed_scaling_factor=2.5, norm_topk_prob=True,
                               intermediate_size=32, hidden_act="silu", first_k_dense_replace=1, kv_lora_rank=32,
                               q_lora_rank=None, qk_rope_head_dim=8, qk_nope_head_dim=16, v_head_dim=16)
        mod = draw(DeepseekV3MoE(cfg).eval())
        feat = family("deepseek_v3", mod, experts=8, topk=3, score="sigmoid", bias=True, groups=(4, 2), normalize=True,
                      scaling=2.5, shared=1, shared_mode="plain")
        x = torch.randn(N, HID, generator=torch.Generator().manual_seed(3))
        with torch.no_grad():
            self.close(feat(0, x), mod(x[None])[0])
        mlp = draw(DeepseekV3MLP(cfg).eval(), 1)
        src = source_of(mlp)
        table = {"gate_up": lambda: torch.cat([src("gate_proj"), src("up_proj")]), "down": lambda: src("down_proj")}
        with torch.no_grad():
            self.close(Dense(weights=lambda layer, name: table[name]())(0, x), mlp(x))


@unittest.skipUnless(torch is not None and _present("minimax_m3_vl"), "requires transformers with minimax_m3_vl on the path")
class MiniMaxM3Tests(Held):
    def test_the_block_and_the_dense_mlp_are_the_model(self):
        from transformers.models.minimax_m3_vl.configuration_minimax_m3_vl import MiniMaxM3VLTextConfig
        from transformers.models.minimax_m3_vl.modeling_minimax_m3_vl import MiniMaxM3VLDenseMLP, MiniMaxM3VLSparseMoeBlock
        from engine.modules.moe import Dense
        cfg = MiniMaxM3VLTextConfig(vocab_size=256, hidden_size=HID, num_hidden_layers=2, num_attention_heads=4,
                                    num_key_value_heads=2, head_dim=16, num_local_experts=8, num_experts_per_tok=2,
                                    intermediate_size=16, shared_intermediate_size=16, dense_intermediate_size=32,
                                    swiglu_alpha=1.702, swiglu_limit=0.8, routed_scaling_factor=2.0)
        mod = draw(MiniMaxM3VLSparseMoeBlock(cfg).eval())
        feat = family("minimax_m3_vl", mod, experts=8, topk=2, score="sigmoid", bias=True, normalize=True, scaling=2.0,
                      scaling_on="output", router_fp32=False, shared=1, shared_mode="plain", activation=("swigluoai", 1.702, 0.8))
        x = torch.randn(N, HID, generator=torch.Generator().manual_seed(4))
        with torch.no_grad():
            self.close(feat(0, x), mod(x[None])[0])
        mlp = draw(MiniMaxM3VLDenseMLP(cfg).eval(), 1)
        src = source_of(mlp)
        table = {"gate_up": lambda: src("gate_up_proj"), "down": lambda: src("down_proj")}
        with torch.no_grad():
            self.close(Dense(activation=("swigluoai", 1.702, 0.8), weights=lambda layer, name: table[name]())(0, x), mlp(x))


@unittest.skipUnless(torch is not None and _present("inkling"), "requires transformers with inkling on the path")
class InklingTests(Held):
    def test_the_sink_block_and_the_scaled_mlp_are_the_model(self):
        from transformers.models.inkling.configuration_inkling import InklingTextConfig
        from transformers.models.inkling.modeling_inkling import InklingMLP, InklingMoE
        from engine.modules.moe import Dense
        cfg = InklingTextConfig(vocab_size=256, hidden_size=HID, num_hidden_layers=2, num_attention_heads=4,
                                num_key_value_heads=2, head_dim=16, n_routed_experts=8, num_experts_per_tok=3,
                                n_shared_experts=2, moe_intermediate_size=16, route_scale=8.0, intermediate_size=32,
                                hidden_act="silu", sconv_kernel_size=3)
        mod = draw(InklingMoE(cfg).eval())
        feat = family("inkling", mod, experts=8, topk=3, score="sigmoid", bias=True, scaling=8.0, router_fp32=False,
                      shared=2, shared_mode="sink")
        x = torch.randn(N, HID, generator=torch.Generator().manual_seed(5))
        with torch.no_grad():
            self.close(feat(0, x), mod(x[None])[0])
        mlp = draw(InklingMLP(cfg).eval(), 1)
        src = source_of(mlp)
        table = {"gate_up": lambda: torch.cat([src("gate_proj"), src("up_proj")]), "down": lambda: src("down_proj"),
                 "scale": lambda: src("global_scale")}
        with torch.no_grad():
            self.close(Dense(weights=lambda layer, name: table[name]())(0, x), mlp(x))
        self.assertNotEqual(float(src("global_scale")), 1.0)


@unittest.skipUnless(torch is not None, "requires torch")
class RouterAndAxesTests(unittest.TestCase):
    def test_the_bias_chooses_but_does_not_weight_and_groups_confine(self):
        from engine.modules.moe import route
        g = torch.Generator().manual_seed(6)
        x, w = torch.randn(N, HID, generator=g), torch.randn(8, HID, generator=g) * HID ** -0.5
        ids, weights, gammas = route(x, w, score="sigmoid", topk=3, normalize=True, scaling=2.5)
        self.assertIsNone(gammas)
        torch.testing.assert_close(weights.sum(-1), torch.full((N,), 2.5))
        big = torch.zeros(8); big[5] = 100.0
        ids_b, weights_b, _ = route(x, w, score="sigmoid", topk=3, bias=big, normalize=False)
        self.assertTrue(bool((ids_b == 5).any(-1).all()))                                   # chosen everywhere
        scores = torch.sigmoid(x @ w.T)
        torch.testing.assert_close(weights_b, scores.gather(1, ids_b))                       # weighted without the bias
        ids_g, _, _ = route(x, w, score="sigmoid", topk=2, groups=(4, 1))
        self.assertTrue(bool((ids_g // 2 == ids_g[:, :1] // 2).all()))                       # both from one group of two
        ids_s, weights_s, _ = route(x, w, score="softmax", topk=3, normalize=True, fp32=False)
        from engine.modules.moe import route_softmax_topk
        ids_r, weights_r = route_softmax_topk(x @ w.T, 3, True)
        self.assertTrue(torch.equal(ids_s, ids_r) and torch.equal(weights_s, weights_r))
        ids_k, weights_k, gammas_k = route(x, torch.randn(10, HID, generator=g) * HID ** -0.5, score="sigmoid", topk=3,
                                          bias=torch.zeros(8), scaling=8.0, sink=2)
        self.assertEqual((tuple(weights_k.shape), tuple(gammas_k.shape)), ((N, 3), (N, 2)))
        torch.testing.assert_close(weights_k.sum(-1) + gammas_k.sum(-1), torch.full((N,), 8.0))   # one normalisation

    def test_the_activations_and_the_axes_are_checked(self):
        from engine.modules.moe import Dense, MoE, gated_mlp
        g, u = torch.tensor([-2.0, 0.5, 3.0]), torch.tensor([2.0, -3.0, 0.5])
        torch.testing.assert_close(gated_mlp(g, u, "silu"), torch.nn.functional.silu(g) * u)
        torch.testing.assert_close(gated_mlp(g, u, ("swiglu_clamped", 1.0)),
                                   torch.nn.functional.silu(torch.tensor([-2.0, 0.5, 1.0])) * torch.tensor([1.0, -1.0, 0.5]))
        gc, uc = torch.tensor([-2.0, 0.5, 1.0]), torch.tensor([1.0, -1.0, 0.5])
        torch.testing.assert_close(gated_mlp(g, u, ("swigluoai", 1.702, 1.0)), (uc + 1) * gc * torch.sigmoid(1.702 * gc))
        with self.assertRaises(ValueError):
            gated_mlp(g, u, "gelu")
        ok = dict(experts=8, topk=2, weights=None, expert=None)
        MoE(**ok)
        for bad in (dict(topk=9), dict(score="relu"), dict(groups=(3, 1)), dict(groups=(4, 5)), dict(scaling_on="both"),
                    dict(shared=1), dict(shared_mode="plain"), dict(shared=1, shared_mode="sink", score="softmax", shared_expert=1),
                    dict(activation="gelu"), dict(shared=1, shared_mode="plain")):
            with self.assertRaises(ValueError):
                MoE(**{**ok, **bad})
        with self.assertRaises(ValueError):
            Dense(activation=("swiglu",), weights=None)


if __name__ == "__main__":
    unittest.main()
