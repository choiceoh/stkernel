"""engine/modules/residual as a family: the forms of engine/base/composition.Residual, each held to the transformers
decoder layer that defines it -- with the sublayers taken from the transformers layer itself, so only the residual
form differs: HyperStreams to glm5_next's layer (mHC, mean head) and deepseek_v4's weighted head, PreNorm to
deepseek_v3's layer and to inkling's (with the output convs and their per-sequence state). AttnRes (Kimi K3) has no
local oracle: its properties are checked against the quoted code. Qwen3.8's gated streams are held by the composition
tests. And the residual's own state rides the composition: pieces == whole, specs declared."""
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


HID, T = 64, 40


def draw(mod, seed=0):
    torch.manual_seed(seed)
    with torch.no_grad():
        for name, p in list(mod.named_parameters()) + list(mod.named_buffers()):
            if name.endswith("A_log"):
                p.copy_(torch.empty_like(p).uniform_(1, 16).log())
            elif name.endswith("scale"):
                p.uniform_(0.5, 1.5)
            elif name.endswith("base") or name.endswith("dt_bias") or name.endswith(".bias"):
                p.normal_(0, 0.5)
            elif name.endswith(("norm.weight", "layernorm.weight")):
                p.normal_(1, 0.3)
            elif p.ndim == 3:
                p.normal_(0, 0.5)
            elif p.ndim == 2:
                p.normal_(0, p.shape[-1] ** -0.5)
            else:
                p.normal_(0, 0.5)
    return mod


def steps(n, pieces):
    from engine.base.composition import Step
    at = 0
    for count in pieces:
        yield Step.of([(0, at, torch.zeros(count, dtype=torch.int64))])
        at += count


def drive(form, sublayers, x, pieces=None):
    """The composition's loop over `sublayers` [(mixer, mlp)] with the form, in one step or in `pieces`."""
    from engine.base.composition import State
    state, outs = State(), []
    for step in steps(x.shape[0], pieces or [x.shape[0]]):
        state.check(step)
        seg = step.segments[0]
        xs = x[seg.ctx:seg.ctx + seg.length]                                       # the piece's positions
        h = form.open(xs)
        for layer, (mixer, mlp) in enumerate(sublayers):
            for site, fn in (("mixer", mixer), ("mlp", mlp)):
                xin, carry = form.enter(layer, site, h, step, state)
                h = form.leave(layer, site, fn(xin), carry, step, state)
        outs.append(form.close(h))
        state.commit(step)
    return torch.cat(outs)


def causal_mask(t):
    keep = torch.arange(t)[None, :] <= torch.arange(t)[:, None]
    return torch.zeros(1, 1, t, t).masked_fill(~keep, torch.finfo(torch.float32).min)


class Held(unittest.TestCase):
    def close(self, got, want, tol=1e-5):
        self.assertEqual(tuple(got.shape), tuple(want.shape))
        diff = float((got.float() - want.float()).abs().max())
        self.assertLessEqual(diff, tol, f"max |got - want| = {diff}")


@unittest.skipUnless(torch is not None and _present("glm5_next"), "requires transformers with glm5_next on the path")
class HyperStreamsTests(Held):
    def test_two_glm_layers_are_the_model(self):
        from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig
        from transformers.models.glm5_next.modeling_glm5_next import (Glm5NextTextDecoderLayer, Glm5NextTextHyperHead,
                                                                         Glm5NextTextRMSNorm)
        from engine.modules.residual import HyperStreams
        cfg = Glm5NextTextConfig(vocab_size=256, hidden_size=HID, num_hidden_layers=2, num_attention_heads=4,
                                 num_key_value_heads=4, layer_types=["linear_attention"] * 2, mlp_layer_types=["dense"] * 2,
                                 intermediate_size=32, hc_mult=4, hc_sinkhorn_iters=3, hc_eps=1e-6, rms_norm_eps=1e-6,
                                 linear_num_heads=4, linear_head_dim=16, linear_conv_kernel_dim=4, swiglu_limit=10.0)
        layers = [draw(Glm5NextTextDecoderLayer(cfg, i).eval(), i) for i in range(2)]
        head_norm = draw(Glm5NextTextRMSNorm(HID, 1e-6), 9)
        x = torch.randn(T, HID, generator=torch.Generator().manual_seed(1))
        with torch.no_grad():
            streams = x[None].unsqueeze(2).expand(-1, -1, 4, -1).contiguous()
            for layer in layers:
                streams, _ = layer(streams)
            ref = head_norm(Glm5NextTextHyperHead()(streams))[0]
            table = {i: {"mixer": {"fn": L.attn_hc.fn, "base": L.attn_hc.base, "scale": L.attn_hc.scale, "norm": L.input_layernorm.weight},
                         "mlp": {"fn": L.ffn_hc.fn, "base": L.ffn_hc.base, "scale": L.ffn_hc.scale, "norm": L.post_attention_layernorm.weight}}
                     for i, L in enumerate(layers)}
            form = HyperStreams(hc=4, eps=1e-6, hc_eps=1e-6, sinkhorn=3, head="mean",
                                weights=lambda layer, site, name: table[layer][site][name],
                                final=lambda name: {"norm": head_norm.weight}[name])
            subs = [(lambda xin, L=L: L.self_attn(xin[None])[0], lambda xin, L=L: L.mlp(xin[None])[0]) for L in layers]
            got = drive(form, subs, x)
        self.close(got, ref, 1e-5)
        self.assertEqual(tuple(form.open(x).shape), (T, 4 * HID))

    @unittest.skipUnless(torch is not None and _present("deepseek_v4"), "requires transformers with deepseek_v4 on the path")
    def test_the_weighted_head_is_deepseek_v4(self):
        from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
        from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4HyperHead, DeepseekV4RMSNorm
        from engine.modules.residual import HyperStreams
        cfg = DeepseekV4Config(vocab_size=256, hidden_size=HID, num_hidden_layers=1, hc_mult=4, hc_eps=1e-6, rms_norm_eps=1e-6)
        head, norm = draw(DeepseekV4HyperHead(cfg), 2), draw(DeepseekV4RMSNorm(HID, 1e-6), 3)
        streams = torch.randn(1, T, 4, HID, generator=torch.Generator().manual_seed(4))
        with torch.no_grad():
            ref = norm(head(streams))[0]
            form = HyperStreams(hc=4, eps=1e-6, hc_eps=1e-6, sinkhorn=3, head="weighted", weights=None,
                                final=lambda name: {"fn": head.hc_fn, "base": head.hc_base, "scale": head.hc_scale, "norm": norm.weight}[name])
            got = form.close(streams[0].flatten(1))
        self.close(got, ref, 1e-5)


@unittest.skipUnless(torch is not None and _present("deepseek_v3"), "requires transformers with deepseek_v3 on the path")
class PreNormTests(Held):
    def test_a_deepseek_layer_is_the_model(self):
        from transformers.models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config
        from transformers.models.deepseek_v3.modeling_deepseek_v3 import (DeepseekV3DecoderLayer, DeepseekV3RMSNorm,
                                                                           DeepseekV3RotaryEmbedding)
        from engine.modules.residual import PreNorm
        cfg = DeepseekV3Config(vocab_size=256, hidden_size=HID, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
                               q_lora_rank=None, kv_lora_rank=32, qk_rope_head_dim=8, qk_nope_head_dim=16, v_head_dim=16,
                               rms_norm_eps=1e-6, rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
                               n_routed_experts=4, num_experts_per_tok=2, moe_intermediate_size=16, intermediate_size=32,
                               first_k_dense_replace=1)
        cfg._attn_implementation = "eager"
        layer, norm = draw(DeepseekV3DecoderLayer(cfg, 0).eval()), draw(DeepseekV3RMSNorm(HID, 1e-6), 5)
        x = torch.randn(T, HID, generator=torch.Generator().manual_seed(6))
        with torch.no_grad():
            cos, sin = DeepseekV3RotaryEmbedding(cfg)(x[None], torch.arange(T)[None])
            out = layer(x[None], position_embeddings=(cos, sin), attention_mask=causal_mask(T))
            ref = norm(out[0] if isinstance(out, tuple) else out)[0]
            table = {"mixer": {"norm": layer.input_layernorm.weight}, "mlp": {"norm": layer.post_attention_layernorm.weight}}
            form = PreNorm(eps=1e-6, weights=lambda l, site, name: table[site][name], final=lambda name: {"norm": norm.weight}[name])
            subs = [(lambda xin: layer.self_attn(xin[None], position_embeddings=(cos, sin), attention_mask=causal_mask(T))[0][0],
                     lambda xin: layer.mlp(xin[None])[0])]
            got = drive(form, subs, x)
        self.close(got, ref, 1e-5)


@unittest.skipUnless(torch is not None and _present("inkling"), "requires transformers with inkling on the path")
class InklingResidualTests(Held):
    def build(self):
        from transformers.models.inkling.configuration_inkling import InklingTextConfig
        from transformers.models.inkling.modeling_inkling import InklingDecoderLayer, InklingRMSNorm
        cfg = InklingTextConfig(vocab_size=256, hidden_size=HID, num_hidden_layers=2, num_attention_heads=4,
                                num_key_value_heads=2, head_dim=16, swa_num_attention_heads=4, swa_num_key_value_heads=2,
                                swa_head_dim=16, sliding_window_size=8, d_rel=4, rel_extent=12, log_scaling_n_floor=8,
                                sconv_kernel_size=3, layer_types=["hybrid", "hybrid"], mlp_layer_types=["dense", "dense"],
                                intermediate_size=32, rms_norm_eps=1e-6, n_routed_experts=4, num_experts_per_tok=2)
        cfg._attn_implementation = "eager"
        layers = [draw(InklingDecoderLayer(cfg, i).eval(), i) for i in range(2)]
        return cfg, layers, draw(InklingRMSNorm(HID, 1e-6), 7)

    def test_two_layers_with_output_convs_are_the_model(self):
        from engine.modules.residual import PreNorm
        cfg, layers, norm = self.build()
        x = torch.randn(T, HID, generator=torch.Generator().manual_seed(8))
        with torch.no_grad():
            h = x[None]
            for layer in layers:
                h = layer(h, attention_mask=causal_mask(T), conv_mask=None, past_key_values=None)
                h = h[0] if isinstance(h, tuple) else h
            ref = norm(h)[0]
            table = {i: {"mixer": {"norm": L.input_layernorm.weight, "conv": L.attn_sconv.conv1d.weight},
                         "mlp": {"norm": L.post_attention_layernorm.weight, "conv": L.mlp_sconv.conv1d.weight}} for i, L in enumerate(layers)}
            form = PreNorm(eps=1e-6, out_conv=3, hidden=HID, weights=lambda layer, site, name: table[layer][site][name],
                           final=lambda name: {"norm": norm.weight}[name])
            subs = [(lambda xin, L=L: L.self_attn(xin[None], attention_mask=causal_mask(T))[0][0], lambda xin, L=L: L.mlp(xin[None])[0])
                    for L in layers]
            got = drive(form, subs, x)
        self.close(got, ref, 1e-5)
        specs = form.cache_specs([0, 1])
        self.assertEqual([(s.name, s.key, s.bytes_per_seq) for s in specs],
                         [("residual mixer conv", "residual_mixer_conv", HID * 2 * 4), ("residual mlp conv", "residual_mlp_conv", HID * 2 * 4)])


@unittest.skipUnless(torch is not None, "requires torch")
class FormTests(Held):
    """Properties needing no oracle: the convs' state across pieces, AttnRes as quoted, the composition carrying it all."""

    def dummy(self, layers, seed=10):
        g = torch.Generator().manual_seed(seed)
        maps = [(torch.randn(HID, HID, generator=g) * HID ** -0.5, torch.randn(HID, HID, generator=g) * HID ** -0.5) for _ in range(layers)]
        return [(lambda xin, a=a: xin @ a, lambda xin, b=b: torch.tanh(xin @ b)) for a, b in maps]

    def test_output_convs_carry_across_pieces(self):
        from engine.modules.residual import PreNorm
        g = torch.Generator().manual_seed(11)
        table = {i: {site: {"norm": torch.rand(HID, generator=g) + 0.5, "conv": torch.randn(HID, 1, 3, generator=g) * 0.5}
                     for site in ("mixer", "mlp")} for i in range(2)}
        form = PreNorm(eps=1e-6, out_conv=3, hidden=HID, weights=lambda layer, site, name: table[layer][site][name])
        x = torch.randn(T, HID, generator=g)
        subs = self.dummy(2)
        whole = drive(form, subs, x)
        self.close(drive(form, subs, x, [13, 27]), whole, 1e-5)
        self.close(drive(form, subs, x, [T - 5] + [1] * 5), whole, 1e-5)
        plain = PreNorm(eps=1e-6, weights=lambda layer, site, name: table[layer][site][name])
        self.assertFalse(torch.allclose(drive(plain, subs, x), whole))                          # the convs matter
        self.assertEqual(plain.cache_specs([0, 1]), [])

    def test_attn_res_is_the_quoted_mix(self):
        from engine.modules.residual import AttnRes, attn_res
        g = torch.Generator().manual_seed(12)
        prefix, blocks = torch.randn(5, HID, generator=g), torch.randn(5, 3, HID, generator=g)
        mean = attn_res(prefix, blocks, torch.zeros(1, HID), torch.ones(HID), 1e-6)
        torch.testing.assert_close(mean, torch.cat([blocks, prefix[:, None]], 1).mean(1))      # zero scores: the mean
        proj, norm_w = torch.randn(1, HID, generator=g), torch.rand(HID, generator=g) + 0.5
        v = torch.cat([blocks, prefix[:, None]], 1)
        k = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + 1e-6)
        probs = torch.softmax((k * (norm_w * proj[0])).sum(-1), -1)
        torch.testing.assert_close(attn_res(prefix, blocks, proj, norm_w, 1e-6), torch.einsum("nb,nbh->nh", probs, v))
        table = {i: {site: {"norm": torch.rand(HID, generator=g) + 0.5, "res_proj": torch.randn(1, HID, generator=g) * 0.3,
                            "res_norm": torch.rand(HID, generator=g) + 0.5} for site in ("mixer", "mlp")} for i in range(3)}
        final = {"norm": torch.ones(HID), "res_proj": torch.randn(1, HID, generator=g) * 0.3, "res_norm": torch.ones(HID)}
        form = AttnRes(block=2, eps=1e-6, weights=lambda layer, site, name: table[layer][site][name], final=lambda name: final[name])
        x = torch.randn(T, HID, generator=g)
        subs = self.dummy(3, 13)
        from engine.base.composition import State, Step
        step, state = Step.of([(0, 0, torch.zeros(T, dtype=torch.int64))]), State()
        h = form.open(x)
        widths = []
        for layer, (mixer, mlp) in enumerate(subs):
            xin, carry = form.enter(layer, "mixer", h, step, state)
            if layer == 0:                                                                       # no block yet: the embedding, normalised
                from engine.modules.norm import rmsnorm
                torch.testing.assert_close(xin, rmsnorm(x, table[0]["mixer"]["norm"], 1e-6))
            h = form.leave(layer, "mixer", mixer(xin), carry, step, state)
            xin, carry = form.enter(layer, "mlp", h, step, state)
            h = form.leave(layer, "mlp", mlp(xin), carry, step, state)
            widths.append(h.shape[1] // HID)
        self.assertEqual(widths, [2, 2, 3])                                                     # blocks stored at layers 0 and 2
        self.assertEqual(tuple(form.close(h).shape), (T, HID))
        self.close(drive(form, subs, x, [13, 27]), drive(form, subs, x), 1e-6)                  # no sequence state
        with self.assertRaises(ValueError):
            AttnRes(block=0, eps=1e-6, weights=None, final=None)

    def test_the_composition_carries_the_residual_state(self):
        from engine.base.composition import Composition, Layer, Plan, State, Step
        from engine.modules.residual import PreNorm
        g = torch.Generator().manual_seed(14)
        table = {i: {site: {"norm": torch.rand(HID, generator=g) + 0.5, "conv": torch.randn(HID, 1, 3, generator=g) * 0.5}
                     for site in ("mixer", "mlp")} for i in range(2)}
        form = PreNorm(eps=1e-6, out_conv=3, hidden=HID, weights=lambda layer, site, name: table[layer][site][name])
        a, b = torch.randn(HID, HID, generator=g) * HID ** -0.5, torch.randn(HID, HID, generator=g) * HID ** -0.5
        embed_w, head_w = torch.randn(32, HID, generator=g), torch.randn(16, HID, generator=g)
        comp = Composition(Plan((Layer("mix", "mlp"), Layer("mix", "mlp"))), embed=lambda ids: embed_w[ids], residual=form,
                           features={"mix": lambda layer, x, step, state: x @ a, "mlp": lambda layer, x, step, state: torch.tanh(x @ b)},
                           head=lambda h: h @ head_w.T)
        paged, slots = comp.cache_specs()
        self.assertEqual(([s.key for s in paged], [s.key for s in slots]), ([], ["residual_mixer_conv", "residual_mlp_conv"]))
        self.assertEqual(comp.spec_layers(), {"residual_mixer_conv": [0, 1], "residual_mlp_conv": [0, 1]})
        ids = torch.randint(0, 32, (T,), generator=g)
        whole = comp.forward(Step.of([(0, 0, ids)]), State(), logits="all")
        state, outs, at = State(), [], 0
        for count in (13, 27):
            outs.append(comp.forward(Step.of([(0, at, ids[at:at + count])]), state, logits="all"))
            at += count
        self.close(torch.cat(outs), whole, 1e-6)


if __name__ == "__main__":
    unittest.main()
