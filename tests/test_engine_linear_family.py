"""engine/modules/linear_attention as a family: one feature, six axes. The KDA variant is held to transformers'
Glm5NextTextLinearAttention (the oracle for GLM-5.3's linear attention, on the CPU) in both decay forms; the GDN variant
is held to qwen4_exp by tests/test_engine_composition.py. The axes that are only weight layout -- a full-rank decay or
gate for a low-rank pair (Kimi K3, Ling-3.0), separate projections and convs for fused ones (`named`) -- are shown to be
the same arithmetic. And a prefill in pieces is the prefill whole, the property the runner leans on."""
import importlib.util
import unittest

torch = None
if importlib.util.find_spec("torch") is not None:
    import torch


def _oracle_present() -> bool:
    try:
        return importlib.util.find_spec("transformers.models.glm5_next") is not None
    except Exception:
        return False


ORACLE = torch is not None and _oracle_present()
H, D, HID, CONV, EPS, T = 4, 16, 64, 4, 1e-6, 40


def tiny_glm5(lower_bound=-5.0, seed=0):
    """Glm5NextTextLinearAttention alone, at a small width, every parameter drawn (norm weights around one, A_log as
    the model initialises it)."""
    from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextLinearAttention
    torch.manual_seed(seed)
    cfg = Glm5NextTextConfig(vocab_size=256, hidden_size=HID, num_hidden_layers=4, num_attention_heads=4,
                             num_key_value_heads=4, linear_num_heads=H, linear_head_dim=D, linear_conv_kernel_dim=CONV,
                             linear_lower_bound=lower_bound, rms_norm_eps=EPS, hidden_act="silu")
    mod = Glm5NextTextLinearAttention(cfg, layer_idx=0).eval()
    with torch.no_grad():
        for name, p in mod.named_parameters():
            if name.endswith("A_log"):
                p.copy_(torch.empty_like(p).uniform_(1, 16).log())
            elif name.endswith("dt_bias"):
                p.normal_(0, 0.5)
            elif name.endswith("o_norm.weight"):
                p.normal_(1, 0.3)
            elif p.ndim == 3:                                        # the conv taps
                p.normal_(0, 0.5)
            else:
                p.normal_(0, p.shape[-1] ** -0.5)
    return cfg, mod


def sources(mod):
    """(glm5_next-named source, kimi-named source) over the module's parameters: the second splits the fused conv
    into q/k/v convs and composes the low-rank pairs into f_proj/g_proj as well."""
    sd = {k: v.detach().clone() for k, v in mod.state_dict().items()}
    conv = sd["conv1d.weight"]
    part = conv.shape[0] // 3
    kimi = {k: v for k, v in sd.items() if not k.startswith("forget_gate.") and k != "conv1d.weight"}
    kimi.update({"q_conv1d.weight": conv[:part], "k_conv1d.weight": conv[part:2 * part], "v_conv1d.weight": conv[2 * part:],
                 "f_a_proj.weight": sd["forget_gate.f_a_proj.weight"], "f_b_proj.weight": sd["forget_gate.f_b_proj.weight"],
                 "A_log": sd["forget_gate.A_log"], "dt_bias": sd["forget_gate.dt_bias"],
                 "f_proj.weight": sd["forget_gate.f_b_proj.weight"] @ sd["forget_gate.f_a_proj.weight"],
                 "g_proj.weight": sd["g_b_proj.weight"] @ sd["g_a_proj.weight"]})

    def over(table):
        def source(hf):
            if f"{hf}.weight" in table:
                return table[f"{hf}.weight"]
            if hf in table:
                return table[hf]
            raise KeyError(hf)
        return source
    return over(sd), over(kimi)


def feature(scheme, source, variant="kda", **axes):
    from engine.modules.linear_attention import VARIANTS, GatedDeltaNet, named
    settings = dict(VARIANTS[variant], **axes)
    return GatedDeltaNet(k_heads=H, v_heads=H, k_dim=D, v_dim=D, conv=CONV, eps=EPS, dtype="float32", **settings,
                         weights=lambda layer, name: named(scheme, source)(name))


def run(feat, x, pieces, state=None):
    """x [T, HID] through the feature as one sequence in `pieces` (token counts), the state continuing."""
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


@unittest.skipUnless(torch is not None, "requires torch")
class AxesTests(unittest.TestCase):
    def test_the_decays_are_the_named_ones(self):
        from engine.modules.linear_attention import gdn_decay, kda_gate, log_decay
        g = torch.Generator().manual_seed(0)
        a, A_log, bias = torch.randn(5, H, generator=g), torch.randn(H, generator=g), torch.randn(H, generator=g)
        self.assertTrue(torch.equal(gdn_decay(a, A_log, bias), log_decay(a, A_log, bias, None, False)))
        want = -A_log.exp() * torch.nn.functional.softplus(a + bias)
        torch.testing.assert_close(gdn_decay(a, A_log, bias), want)
        raw, cbias = torch.randn(1, 5, H, D, generator=g), torch.randn(H * D, generator=g)
        safe = kda_gate(raw, A_log, cbias, -5.0, True)
        torch.testing.assert_close(safe, -5.0 * torch.sigmoid(A_log.exp().view(1, 1, H, 1) * (raw + cbias.view(1, 1, H, D))))
        self.assertTrue(bool((safe < 0).all()) and bool((safe > -5.0).all()))
        soft = kda_gate(raw, A_log, cbias, -5.0, False)
        torch.testing.assert_close(soft, -A_log.exp().view(1, 1, H, 1) * torch.nn.functional.softplus(raw + cbias.view(1, 1, H, D)))
        self.assertTrue(torch.equal(soft, log_decay(raw, A_log, cbias, None, True)))

    def test_the_output_norms_differ_only_in_rounding(self):
        from engine.modules.linear_attention import kda_output_norm, output_norm
        g = torch.Generator().manual_seed(1)
        core, gate, w = torch.randn(6, D, generator=g), torch.randn(6, D, generator=g), torch.randn(D, generator=g)
        self.assertTrue(torch.equal(kda_output_norm(core, gate, w, EPS), output_norm(core, gate, w, EPS, "sigmoid", "strict")))
        torch.testing.assert_close(output_norm(core, gate, w, EPS, "silu", "rounded"), output_norm(core, gate, w, EPS, "silu", "strict"))
        b = core.to(torch.bfloat16)
        strict = output_norm(b, gate.to(torch.bfloat16), w.to(torch.bfloat16), EPS, "sigmoid", "strict")
        rounded = output_norm(b, gate.to(torch.bfloat16), w.to(torch.bfloat16), EPS, "sigmoid", "rounded")
        self.assertEqual((strict.dtype, rounded.dtype), (torch.bfloat16, torch.bfloat16))
        self.assertFalse(torch.equal(strict, rounded))                                 # the axis is real at bf16
        with self.assertRaises(ValueError):
            output_norm(core, gate, w, EPS, "sigmoid", "loose")

    def test_the_variants_and_the_axes_are_checked(self):
        from engine.modules.linear_attention import SCHEMES, VARIANTS, GatedDeltaNet, named
        self.assertEqual(set(VARIANTS), {"gdn", "kda", "kda_full_gate", "kda_full"})
        self.assertEqual({v["decay"] for k, v in VARIANTS.items() if k != "gdn"}, {"channel"})
        self.assertEqual((VARIANTS["kda_full_gate"]["lowrank_decay"], VARIANTS["kda_full_gate"]["lowrank_gate"]), (True, False))
        self.assertEqual((VARIANTS["kda_full"]["lowrank_decay"], VARIANTS["kda_full"]["lowrank_gate"]), (False, False))
        self.assertEqual(set(SCHEMES), {"qwen4_exp", "glm5_next", "kimi"})
        def mk(**kw):
            args = dict(k_heads=H, v_heads=H, k_dim=D, v_dim=D, conv=CONV, eps=EPS, weights=None)
            args.update(kw)
            return GatedDeltaNet(**args)
        for bad in (dict(decay="token"), dict(lower_bound=0.5), dict(norm_cast="loose"), dict(gate_activation="relu"),
                    dict(k_heads=3)):
            with self.assertRaises(ValueError):
                mk(**bad)
        get = named("kimi", lambda hf: torch.zeros(2))
        with self.assertRaises(KeyError):
            get("nothing")
        self.assertEqual(tuple(get("qkv").shape), (6,))                                # three concatenated
        with self.assertRaises(KeyError):
            named("unknown", lambda hf: None)


@unittest.skipUnless(ORACLE, "requires transformers with glm5_next (5.16.1) on the path")
class KDAOracleTests(unittest.TestCase):
    """The KDA variant is transformers' Glm5NextTextLinearAttention, in both decay forms."""

    def close(self, got, want, tol):
        self.assertEqual(got.shape, want.shape)
        diff = float((got.float() - want.float()).abs().max())
        self.assertLessEqual(diff, tol, f"max |got - want| = {diff}")

    def test_the_safe_gate_form_is_the_model(self):
        cfg, mod = tiny_glm5(-5.0)
        glm, _ = sources(mod)
        x = torch.randn(T, HID, generator=torch.Generator().manual_seed(2))
        with torch.no_grad():
            ref = mod(x[None])[0]
            got, _ = run(feature("glm5_next", glm, "kda", lower_bound=-5.0), x, [T])
        self.close(got, ref, 1e-5)

    def test_the_softplus_form_is_the_model(self):
        cfg, mod = tiny_glm5(None)
        self.assertIsNone(mod.forget_gate.safe_gate_lower_bound)
        glm, _ = sources(mod)
        x = torch.randn(T, HID, generator=torch.Generator().manual_seed(3))
        with torch.no_grad():
            ref = mod(x[None])[0]
            got, _ = run(feature("glm5_next", glm, "kda", lower_bound=None), x, [T])
        self.close(got, ref, 1e-5)

    def test_pieces_are_the_whole(self):
        cfg, mod = tiny_glm5(-5.0)
        glm, _ = sources(mod)
        feat = feature("glm5_next", glm, "kda")
        x = torch.randn(T, HID, generator=torch.Generator().manual_seed(4))
        with torch.no_grad():
            whole, state = run(feat, x, [T])
            chunked, _ = run(feat, x, [13, 27])
            decoded, state2 = run(feat, x, [T - 6] + [1] * 6)
        self.close(chunked, whole, 1e-5)
        self.close(decoded, whole, 1e-5)
        self.close(state2.get(0, "linear_state", 0), state.get(0, "linear_state", 0), 1e-5)
        self.close(state2.get(0, "linear_conv", 0), state.get(0, "linear_conv", 0), 1e-6)

    def test_separate_weights_are_the_fused_ones(self):
        cfg, mod = tiny_glm5(-5.0)
        glm, kimi = sources(mod)
        x = torch.randn(T, HID, generator=torch.Generator().manual_seed(5))
        with torch.no_grad():
            fused, _ = run(feature("glm5_next", glm, "kda"), x, [T])
            separate, _ = run(feature("kimi", kimi, "kda"), x, [T])
        self.assertTrue(torch.equal(fused, separate))

    def test_a_full_rank_gate_or_decay_is_the_low_rank_pair(self):
        cfg, mod = tiny_glm5(-5.0)
        glm, kimi = sources(mod)
        x = torch.randn(T, HID, generator=torch.Generator().manual_seed(6))
        with torch.no_grad():
            low, _ = run(feature("glm5_next", glm, "kda"), x, [T])
            full_gate, _ = run(feature("kimi", kimi, "kda_full_gate"), x, [T])    # Kimi K3's use_full_rank_gate
            full, _ = run(feature("kimi", kimi, "kda_full"), x, [T])              # Ling-3.0's no_kda_lora
        self.close(full_gate, low, 1e-5)
        self.close(full, low, 1e-5)


if __name__ == "__main__":
    unittest.main()
