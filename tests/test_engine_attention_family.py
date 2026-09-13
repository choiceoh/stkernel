"""engine/modules/attention as a family: one feature, the axes of its docstring. Each form and selection is held to
the transformers implementation that defines it, on the CPU: glm5_next (MLA + DSA k-pool indexer, no rotation),
deepseek_v3 (dense MLA, neox and interleaved rotation, with and without q_lora -- Kimi K3's and Ling-3.0's form),
minimax_m3_vl (GQA with unit-offset qk norms, partial rotation, the MSA block indexer), inkling (GQA with T5 qk
norms, k/v short convs, a sliding window or a relative position bias with log scaling); the sink to
sparse_attention.sparse_attn (DeepSeek-V4.1's kernel semantics). Qwen3.8's GQA + QSA + gate is held by
tests/test_engine_composition.py. And a prefill in pieces is the prefill whole, for every form."""
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
    """Every parameter drawn: linears at 1/sqrt(fan_in), norms around their identity, convs and biases wide."""
    torch.manual_seed(seed)
    with torch.no_grad():
        for name, p in mod.named_parameters():
            if name.endswith(("layernorm.weight", "q_norm.weight", "k_norm.weight")):
                p.normal_(1.0 if "kv_a_layernorm" in name or "q_a_layernorm" in name or "indexer" not in name and
                          not isinstance(mod, object) else 1.0, 0.3)
            elif name.endswith(".bias"):
                p.normal_(0, 0.3)
            elif p.ndim == 3:
                p.normal_(0, 0.5)
            elif p.ndim == 2:
                p.normal_(0, p.shape[-1] ** -0.5)
            else:
                p.normal_(0, 0.5)
    return mod


def gemma_draw(mod, seed=0):
    """MiniMax/Qwen norms are (1 + w): draw w around zero."""
    draw(mod, seed)
    with torch.no_grad():
        for name, p in mod.named_parameters():
            if name.endswith("norm.weight"):
                p.normal_(0, 0.3)
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


def feature(scheme, mod, **axes):
    from engine.modules.attention import Attention, named
    src = source_of(mod)
    dims = {k: axes[k] for k in ("heads", "head_dim") if k in axes}
    return Attention(dtype="float32", weights=lambda layer, name: named(scheme, src, **dims)(name), **axes)


def run(feat, x, pieces, state=None):
    from engine.base.composition import State, Step
    state = State() if state is None else state
    outs, at = [], 0
    for n in pieces:
        step = Step.of([(0, at, torch.zeros(n, dtype=torch.int64))])
        state.check(step)
        outs.append(feat(0, x[at:at + n], step, state))
        state.commit(step)
        at += n
    return torch.cat(outs), state


def causal_mask(t, window=None):
    """transformers' additive eager mask: 0 where a query may see a key, the dtype's minimum elsewhere."""
    q = torch.arange(t)[:, None]
    k = torch.arange(t)[None, :]
    keep = k <= q
    if window is not None:
        keep &= (q - k) < window
    return torch.zeros(1, 1, t, t).masked_fill(~keep, torch.finfo(torch.float32).min)


class Held(unittest.TestCase):
    def close(self, got, want, tol):
        self.assertEqual(tuple(got.shape), tuple(want.shape))
        diff = float((got.float() - want.float()).abs().max())
        self.assertLessEqual(diff, tol, f"max |got - want| = {diff}")

    def pieces(self, feat, x, tol=1e-5):
        with torch.no_grad():
            whole, s1 = run(feat, x, [T])
            chunked, _ = run(feat, x, [13, 27])
            decoded, s2 = run(feat, x, [T - 5] + [1] * 5)
        self.close(chunked, whole, tol)
        self.close(decoded, whole, tol)
        return whole


@unittest.skipUnless(torch is not None and _present("glm5_next"), "requires transformers with glm5_next on the path")
class GLM5NextTests(Held):
    """MLA without rotation + the DSA k-pool indexer == Glm5NextTextAttention."""

    def build(self, always_tail=True, seed=0):
        from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig
        from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextAttention
        cfg = Glm5NextTextConfig(vocab_size=256, hidden_size=HID, num_hidden_layers=4, num_attention_heads=4,
                                 num_key_value_heads=4, q_lora_rank=32, kv_lora_rank=32, qk_rope_head_dim=0,
                                 qk_nope_head_dim=16, v_head_dim=16, index_topk=8, index_head_dim=16, index_n_heads=16,
                                 index_kpool=4, index_kpool_always_select_tail=always_tail, indexer_types=["full"] * 4,
                                 rms_norm_eps=1e-6, linear_num_heads=4, linear_head_dim=16, linear_conv_kernel_dim=4)
        cfg._attn_implementation = "eager"
        mod = draw(Glm5NextTextAttention(cfg, layer_idx=0).eval(), seed)
        from engine.modules.attention import DSAKpool
        feat = feature("glm5_next", mod, form="mla", heads=4, latent=32, nope=16, rope=0, v_dim=16, q_lora=32,
                       rotary_dim=0, eps=1e-6,
                       select=DSAKpool(index_heads=16, index_head_dim=16, topk=8, kpool=4, always_tail=always_tail))
        return cfg, mod, feat

    def test_the_prefill_is_the_model(self):
        """16 index heads so no pool scores exactly zero for a query (relu ties are broken differently by torch.topk and
        by the reference). Without the tail the first kpool - 1 queries select nothing: transformers spreads them over
        every key, the reference (like the kernels) gives zero -- those rows are compared elsewhere."""
        for always_tail in (True, False):
            cfg, mod, feat = self.build(always_tail)
            x = torch.randn(T, HID, generator=torch.Generator().manual_seed(1))
            with torch.no_grad():
                ref = mod(x[None], attention_mask=torch.ones(1, T, dtype=torch.bool))[0][0]
                got, _ = run(feat, x, [T])
            first = 0 if always_tail else 3
            self.close(got[first:], ref[first:], 1e-5)
            if not always_tail:
                self.assertTrue(torch.equal(got[:3], torch.zeros(3, HID)))

    def test_pieces_are_the_whole(self):
        cfg, mod, feat = self.build()
        self.pieces(feat, torch.randn(T, HID, generator=torch.Generator().manual_seed(2)))

    def test_the_selection_is_sparse(self):
        """With a budget of 2 pools of 4 over 40 positions the query does not see everything."""
        from engine.modules.attention import Query
        cfg, mod, feat = self.build()
        from engine.base.composition import State, Step
        x = torch.randn(T, HID, generator=torch.Generator().manual_seed(3))
        state = State()
        step = Step.of([(0, 0, torch.zeros(T, dtype=torch.int64))])
        with torch.no_grad():
            feat(0, x, step, state)                                       # fills the index rows
            q_resid = torch.nn.functional.linear(x, feat.weights(0, "q_a"))
            from engine.modules.norm import rmsnorm
            q_resid = rmsnorm(q_resid, feat.weights(0, "q_a_norm"), 1e-6)
            allowed = feat.select.allowed(Query(feat, 0, 0, x, 0, T, state, None, q_resid))
        self.assertEqual(int(allowed[T - 1].sum()), 8)                    # two pools of four; T is a multiple of 4
        self.assertEqual(int(allowed[5].sum()), 6)                        # one pool (0..3) + the tail (4, 5)
        self.assertTrue(bool(allowed[2, :3].all()))                       # only the tail before the first pool


@unittest.skipUnless(torch is not None and _present("deepseek_v3"), "requires transformers with deepseek_v3 on the path")
class DenseMLATests(Held):
    """Dense MLA with its rope part rotated, neox or interleaved, with or without q_lora == DeepseekV3Attention."""

    def build(self, q_lora, interleave, seed=0):
        from transformers.models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config
        from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3Attention, DeepseekV3RotaryEmbedding
        cfg = DeepseekV3Config(vocab_size=256, hidden_size=HID, num_hidden_layers=2, num_attention_heads=4,
                               num_key_value_heads=4, q_lora_rank=q_lora, kv_lora_rank=32, qk_rope_head_dim=8,
                               qk_nope_head_dim=16, v_head_dim=16, rope_interleave=interleave, rms_norm_eps=1e-6,
                               rope_parameters={"rope_type": "default", "rope_theta": 10000.0}, n_routed_experts=4,
                               num_experts_per_tok=2, moe_intermediate_size=16, intermediate_size=32, first_k_dense_replace=1)
        cfg._attn_implementation = "eager"
        mod = draw(DeepseekV3Attention(cfg, layer_idx=0).eval(), seed)
        rot = DeepseekV3RotaryEmbedding(cfg)
        feat = feature("deepseek_v3", mod, form="mla", heads=4, latent=32, nope=16, rope=8, v_dim=16, q_lora=q_lora,
                       rotary_dim=8, theta=10000.0, interleaved=interleave, eps=1e-6)
        return cfg, mod, rot, feat

    def test_the_prefill_is_the_model(self):
        for q_lora in (None, 32):
            for interleave in (False, True):
                cfg, mod, rot, feat = self.build(q_lora, interleave)
                x = torch.randn(T, HID, generator=torch.Generator().manual_seed(4))
                with torch.no_grad():
                    cos, sin = rot(x[None], torch.arange(T)[None])
                    ref = mod(x[None], position_embeddings=(cos, sin), attention_mask=causal_mask(T))[0][0]
                    got, _ = run(feat, x, [T])
                self.close(got, ref, 1e-5)

    def test_pieces_are_the_whole(self):
        cfg, mod, rot, feat = self.build(32, True)
        self.pieces(feat, torch.randn(T, HID, generator=torch.Generator().manual_seed(5)))


@unittest.skipUnless(torch is not None and _present("minimax_m3_vl"), "requires transformers with minimax_m3_vl on the path")
class MiniMaxM3Tests(Held):
    """GQA with unit-offset qk norms and half rotation + the MSA block indexer == MiniMaxM3VLAttention."""

    def build(self, seed=0):
        from transformers.models.minimax_m3_vl.configuration_minimax_m3_vl import MiniMaxM3VLTextConfig
        from transformers.models.minimax_m3_vl.modeling_minimax_m3_vl import MiniMaxM3VLAttention, MiniMaxM3VLRotaryEmbedding
        cfg = MiniMaxM3VLTextConfig(vocab_size=256, hidden_size=HID, num_hidden_layers=2, num_attention_heads=4,
                                    num_key_value_heads=2, head_dim=16, index_n_heads=2, index_head_dim=16,
                                    index_block_size=4, index_topk_blocks=2, index_local_blocks=1,
                                    layer_types=["minimax_m3_sparse"] * 2, rms_norm_eps=1e-6,
                                    rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.5},
                                    num_local_experts=4, num_experts_per_tok=2, intermediate_size=16)
        cfg._attn_implementation = "eager"
        mod = gemma_draw(MiniMaxM3VLAttention(cfg, layer_idx=0).eval(), seed)
        rot = MiniMaxM3VLRotaryEmbedding(cfg)
        from engine.modules.attention import MSA
        feat = feature("minimax_m3_vl", mod, form="gqa", heads=4, kv_heads=2, head_dim=16, rotary_dim=8, theta=10000.0,
                       qk_norm="rms_unit_offset", eps=1e-6,
                       select=MSA(index_heads=2, index_head_dim=16, block=4, topk_blocks=2, local_blocks=1))
        return cfg, mod, rot, feat

    def test_the_prefill_is_the_model(self):
        cfg, mod, rot, feat = self.build()
        x = torch.randn(T, HID, generator=torch.Generator().manual_seed(6))
        with torch.no_grad():
            cos, sin = rot(x[None], torch.arange(T)[None])
            ref = mod(x[None], position_embeddings=(cos, sin), attention_mask=None)[0][0]
            got, _ = run(feat, x, [T])
        self.close(got, ref, 1e-5)

    def test_pieces_are_the_whole(self):
        cfg, mod, rot, feat = self.build()
        self.pieces(feat, torch.randn(T, HID, generator=torch.Generator().manual_seed(7)))


@unittest.skipUnless(torch is not None and _present("inkling"), "requires transformers with inkling on the path")
class InklingTests(Held):
    """GQA with T5 qk norms, k/v short convs, no rotation: the sliding-window layer with its window-wide relative bias
    and the global layer with its relative bias and log scaling == InklingAttention."""

    def build(self, layer, seed=0):
        from transformers.models.inkling.configuration_inkling import InklingTextConfig
        from transformers.models.inkling.modeling_inkling import InklingAttention
        cfg = InklingTextConfig(vocab_size=256, hidden_size=HID, num_hidden_layers=2, num_attention_heads=4,
                                num_key_value_heads=2, head_dim=16, swa_num_attention_heads=4, swa_num_key_value_heads=2,
                                swa_head_dim=16, sliding_window_size=8, d_rel=4, rel_extent=12, log_scaling_n_floor=8,
                                log_scaling_alpha=0.1, sconv_kernel_size=3, layer_types=["hybrid_sliding", "hybrid"],
                                rms_norm_eps=1e-6, n_routed_experts=4, num_experts_per_tok=2, intermediate_size=16)
        cfg._attn_implementation = "eager"
        mod = draw(InklingAttention(cfg, layer_idx=layer).eval(), seed)
        from engine.modules.attention import Window
        common = dict(form="gqa", heads=4, kv_heads=2, head_dim=16, rotary_dim=0, qk_norm="rms", scale=1.0 / 16, kv_conv=3, eps=1e-6)
        if layer == 0:
            feat = feature("inkling", mod, select=Window(8), relative=(4, 8), **common)
        else:
            feat = feature("inkling", mod, relative=(4, 12), log_scaling=(8, 0.1), **common)
        return cfg, mod, feat

    def test_the_sliding_layer_is_the_model(self):
        cfg, mod, feat = self.build(0)
        x = torch.randn(T, HID, generator=torch.Generator().manual_seed(8))
        with torch.no_grad():
            ref = mod(x[None], attention_mask=causal_mask(T, window=8))[0][0]
            got, _ = run(feat, x, [T])
        self.close(got, ref, 1e-5)

    def test_the_global_layer_is_the_model(self):
        cfg, mod, feat = self.build(1)
        x = torch.randn(T, HID, generator=torch.Generator().manual_seed(9))
        with torch.no_grad():
            ref = mod(x[None], attention_mask=causal_mask(T))[0][0]
            got, _ = run(feat, x, [T])
        self.close(got, ref, 1e-5)

    def test_pieces_are_the_whole(self):
        for layer in (0, 1):
            cfg, mod, feat = self.build(layer)
            self.pieces(feat, torch.randn(T, HID, generator=torch.Generator().manual_seed(10 + layer)))


@unittest.skipUnless(torch is not None, "requires torch")
class SinkAndAxesTests(Held):
    def test_the_sink_is_sparse_attn(self):
        """One KV head shared by every query head, k == v, o the identity: the family with a sink == DeepSeek-V4.1's
        sparse_attn over every causal position."""
        from engine.modules.attention import Attention
        from engine.modules.sparse_attention import sparse_attn
        H, D = 4, 16
        g = torch.Generator().manual_seed(11)
        w = {"q": torch.randn(H * D, HID, generator=g) * HID ** -0.5, "k": torch.randn(D, HID, generator=g) * HID ** -0.5,
             "o": torch.eye(H * D), "sink": torch.randn(H, generator=g)}
        w["v"] = w["k"]
        feat = Attention(form="gqa", heads=H, kv_heads=1, head_dim=D, rotary_dim=0, sink=True, dtype="float32",
                         weights=lambda layer, name: w[name])
        x = torch.randn(T, HID, generator=g)
        with torch.no_grad():
            got, _ = run(feat, x, [T])
            q = torch.nn.functional.linear(x, w["q"]).view(1, T, H, D)
            kv = torch.nn.functional.linear(x, w["k"]).view(1, T, D)
            idx = torch.full((1, T, T), -1, dtype=torch.int32)
            for j in range(T):
                idx[0, j, :j + 1] = torch.arange(j + 1, dtype=torch.int32)
            ref = sparse_attn(q, kv, w["sink"], idx, D ** -0.5).reshape(T, H * D)
        self.close(got, ref, 1e-5)

    def test_the_axes_are_checked(self):
        from engine.modules.attention import SCHEMES, Attention, DSAKpool, MSA, QSA, Window, named
        ok = dict(form="gqa", heads=4, kv_heads=2, head_dim=16, weights=None)
        Attention(**ok)
        for bad in (dict(form="mqa"), dict(kv_heads=3), dict(rotary_dim=32), dict(qk_norm="layer"), dict(gate="row"),
                    dict(rotary_dim=3), dict(log_scaling=(8, 0.1))):
            with self.assertRaises(ValueError):
                Attention(**{**ok, **bad})
        with self.assertRaises(ValueError):
            Attention(form="mla", heads=4, latent=32, nope=16, rope=8, v_dim=16, rotary_dim=4, weights=None)
        with self.assertRaises(ValueError):
            Attention(form="mla", heads=4, latent=32, nope=16, rope=8, v_dim=16, qk_norm="rms", weights=None)
        for bad in (lambda: Window(0), lambda: QSA(index_heads=2, index_head_dim=16, budget=6, ratio=4),
                    lambda: DSAKpool(index_heads=2, index_head_dim=16, topk=2, kpool=4),
                    lambda: MSA(index_heads=2, index_head_dim=16, block=0, topk_blocks=2)):
            with self.assertRaises(ValueError):
                bad()
        self.assertEqual(set(SCHEMES), {"qwen4_exp", "glm5_next", "deepseek_v3", "kimi", "minimax_m3_vl", "inkling"})
        get = named("qwen4_exp", lambda hf: torch.arange(4 * 2 * 3 * 5, dtype=torch.float32).view(4 * 2 * 3, 5), heads=4, head_dim=3)
        q, gate = get("q"), get("gate")
        self.assertEqual((tuple(q.shape), tuple(gate.shape)), ((12, 5), (12, 5)))
        self.assertTrue(torch.equal(q[:3], torch.arange(15, dtype=torch.float32).view(3, 5)))       # head 0's query rows
        self.assertTrue(torch.equal(gate[:3], torch.arange(15, 30, dtype=torch.float32).view(3, 5)))  # then its gate rows
        with self.assertRaises(KeyError):
            get("index_wk")

    def test_the_specs_follow_the_form(self):
        from engine.modules.attention import Attention, DSAKpool, MSA
        gqa = Attention(form="gqa", heads=4, kv_heads=2, head_dim=16, kv_conv=3, dtype="bfloat16", weights=None,
                        select=MSA(index_heads=2, index_head_dim=16, block=4, topk_blocks=2))
        names = [(s.name, s.key, s.bytes_per_token if hasattr(s, "bytes_per_token") else s.bytes_per_seq) for s in gqa.cache_specs([0, 1])]
        self.assertEqual(names, [("attention kv", "attention_kv", 2 * 2 * 16 * 2), ("attention k conv", "attention_k_conv", 32 * 2 * 4),
                                 ("attention v conv", "attention_v_conv", 32 * 2 * 4), ("msa index keys", "msa_index_keys", 16 * 2)])
        mla = Attention(form="mla", heads=4, latent=32, nope=16, rope=8, v_dim=16, q_lora=32, rotary_dim=8, dtype="float32",
                        weights=None, select=DSAKpool(index_heads=2, index_head_dim=16, topk=8, kpool=4))
        self.assertEqual([(s.name, s.bytes_per_token) for s in mla.cache_specs([0])],
                         [("attention latent", 40 * 4), ("dsa index keys", 2 * 16 * 4)])


if __name__ == "__main__":
    unittest.main()
