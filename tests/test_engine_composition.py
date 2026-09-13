"""engine/base/composition: the decoder as a plan, a residual form and features -- the loop's contract on the CPU,
Qwen3.8 assembled from engine/modules and held to transformers' Qwen4ExpForCausalLM (the oracle, pinned in
engine/profiles/qwen38/plan.py) where transformers with qwen4_exp is importable.

    docker exec -e PYTHONPATH=<a site with transformers 5.16.1> -w <repo> stk-test python3 -m unittest tests.test_engine_composition
"""
import importlib.util
import unittest

torch = None
if importlib.util.find_spec("torch") is not None:
    import torch


def _oracle_present() -> bool:
    try:
        return importlib.util.find_spec("transformers.models.qwen4_exp") is not None
    except (ImportError, ValueError):
        return False


ORACLE = torch is not None and _oracle_present()

# Qwen3.8-Flash-Next's text config as the checkpoint on srv2 states it (/home/choiceoh/models/qwen38-flash-next-nvfp4/config.json,
# read 2026-09-13): the fields the composition reads.
QWEN38_TEXT = {
    "hidden_size": 2560, "num_hidden_layers": 48, "num_attention_heads": 24, "num_key_value_heads": 2, "head_dim": 256,
    "linear_num_key_heads": 16, "linear_num_value_heads": 48, "linear_key_head_dim": 128, "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4, "indexer_n_heads": 4, "indexer_kv_heads": 1, "indexer_head_dim": 128, "indexer_budget": 2048,
    "indexer_compress_ratio": 4, "num_experts": 512, "num_experts_per_tok": 10, "moe_intermediate_size": 640,
    "shared_expert_intermediate_size": 640, "hidden_act": "silu", "output_gate_type": "sigmoid", "hc_count": 4, "hc_lowrank": 320,
    "ple_layer_ids": [2], "ple_embed_dim": 2560, "ple_conv_kernel_size": 4, "ngram_size": 3, "heads_per_ngram": 8,
    "ngram_vocab_size_base": 20000000, "seed": 1234, "vocab_size": 248320, "eos_token_id": 248044, "rms_norm_eps": 1e-06,
    "norm_topk_prob": True, "rope_parameters": {"mrope_interleaved": True, "mrope_section": [11, 11, 10],
                                               "partial_rotary_factor": 0.25, "rope_theta": 10000000, "rope_type": "default"},
    "layer_types": ["full_attention" if (i + 1) % 4 == 0 else "linear_attention" for i in range(48)],
}


@unittest.skipUnless(torch is not None, "requires torch")
class LoopContractTests(unittest.TestCase):
    """The loop is model-free: it runs any plan over any residual form and features, in order, and keeps the state honest."""

    def toy(self, calls):
        from engine.base.composition import Composition, Layer, Plan

        class Streams:                                          # a plain residual: h + f(h)
            def open(self, x): return x
            def enter(self, layer, site, h): calls.append(("enter", layer, site)); return h, h
            def leave(self, layer, site, out, carry): return carry + out
            def close(self, h): return h

        class Feature:
            def __init__(self, name): self.name = name
            def __call__(self, layer, x, step, state):
                calls.append((self.name, layer, tuple((s.seq, s.ctx, s.length) for s in step.segments)))
                return x + 1

        plan = Plan((Layer("mix", "mlp", ("table",)), Layer("mix", "mlp")))
        return Composition(plan, embed=lambda ids: ids.float()[:, None].repeat(1, 2), residual=Streams(),
                           features={n: Feature(n) for n in ("mix", "mlp", "table")}, head=lambda h: h)

    def test_the_loop_runs_the_plan_in_order(self):
        from engine.base.composition import State, Step
        calls = []
        comp = self.toy(calls)
        state = State()
        out = comp.forward(Step.of([(4, 0, torch.tensor([1, 2, 3])), (9, 0, torch.tensor([7]))]), state)
        # every injection and sublayer adds f(h) = h + 1 to h, five times: embedding e -> 32e + 31, at each segment's last token
        self.assertEqual(out[:, 0].tolist(), [32 * 3 + 31, 32 * 7 + 31])
        self.assertEqual([c[:2] for c in calls if c[0] != "enter"],
                         [("table", 0), ("mix", 0), ("mlp", 0), ("mix", 1), ("mlp", 1)])
        self.assertEqual([c for c in calls if c[0] == "enter"],
                         [("enter", 0, "mixer"), ("enter", 0, "mlp"), ("enter", 1, "mixer"), ("enter", 1, "mlp")])
        self.assertEqual(tuple(out.shape), (2, 2))                          # the last token of each segment
        self.assertEqual(state.contexts, {4: 3, 9: 1})
        with self.assertRaisesRegex(ValueError, "at 3 tokens"):
            comp.forward(Step.of([(4, 2, torch.tensor([5]))]), state)       # does not continue sequence 4
        self.assertEqual(state.contexts, {4: 3, 9: 1})                       # and nothing moved
        out = comp.forward(Step.of([(9, 1, torch.tensor([8, 9]))]), state, logits="all")
        self.assertEqual((tuple(out.shape), state.contexts[9]), ((2, 2), 3))

    def test_steps_plans_and_compositions_are_validated(self):
        from engine.base.composition import Composition, Layer, Plan, Segment, State, Step
        with self.assertRaisesRegex(ValueError, "tile"):
            Step(torch.tensor([1, 2, 3]), (Segment(0, 0, 0, 2), Segment(1, 0, 1, 1)))
        with self.assertRaisesRegex(ValueError, "tile"):
            Step(torch.tensor([1, 2]), (Segment(0, 0, 0, 1), Segment(0, 1, 1, 1)))    # one segment per sequence
        with self.assertRaisesRegex(ValueError, "cover"):
            Step(torch.tensor([1, 2, 3]), (Segment(0, 0, 0, 2),))
        with self.assertRaises(ValueError):
            Segment(0, -1, 0, 1)
        with self.assertRaisesRegex(ValueError, "nonempty"):
            Plan(())
        with self.assertRaisesRegex(ValueError, "features the composition was not given"):
            Composition(Plan((Layer("a", "b"),)), embed=None, residual=None, features={"a": None}, head=lambda h: h)
        step = Step.of([(0, 5, torch.tensor([1, 2])), (1, 0, torch.tensor([3]))])
        self.assertEqual(step.positions().tolist(), [5, 6, 0])
        self.assertEqual(step.last().tolist(), [1, 2])
        state = State()
        state.put(0, "k", 0, 1); state.put(0, "k", 1, 2)
        state.drop(0)
        self.assertEqual((state.get(0, "k", 0), state.get(0, "k", 1)), (None, 2))


@unittest.skipUnless(torch is not None, "requires torch")
class Qwen38PlanTests(unittest.TestCase):
    """The profile's declarations from the real config: which layer runs what, and what it caches."""

    def test_the_plan_follows_the_layer_types(self):
        from engine.profiles.qwen38 import composition as qc
        plan = qc.plan(QWEN38_TEXT)
        self.assertEqual(len(plan.layers), 48)
        self.assertEqual(len(plan.layers_of("linear_attention")), 36)
        self.assertEqual(plan.layers_of("sparse_attention"), list(range(3, 48, 4)))
        self.assertEqual(plan.layers_of("ple"), [1])                        # ple_layer_ids are one-indexed
        self.assertEqual(len(plan.layers_of("moe")), 48)
        with self.assertRaisesRegex(ValueError, "not 'mamba'"):
            qc.plan(dict(QWEN38_TEXT, layer_types=["mamba"] * 48))

    def test_the_declared_caches_are_the_plan_arithmetic(self):
        """The features' cache specs, summed, are engine/profiles/qwen38/plan.state_bytes at TP 1 -- the hand-derived
        numbers the budget uses -- except the indexer keys: the reference keeps every position's raw key and pools at
        selection (as transformers does), the plan prices the vLLM layout's pooled keys, `ratio` times fewer."""
        from engine.base.composition import State  # noqa: F401  (the composition imports cleanly without weights)
        from engine.profiles.qwen38 import composition as qc
        from engine.profiles.qwen38.plan import state_bytes

        def no_weights(name):
            raise KeyError(name)
        paged, slots = qc.build(QWEN38_TEXT, no_weights).cache_specs()
        per_seq, kv_tok, idx_tok = state_bytes(QWEN38_TEXT, tp=1)
        by_name = lambda specs: {s.name: s for s in specs}
        slot, page = by_name(slots), by_name(paged)
        gdn = slot["linear conv state"], slot["linear recurrent state"]
        self.assertEqual({s.layers for s in gdn}, {36})
        self.assertEqual(sum(s.layers * s.bytes_per_seq for s in gdn), per_seq)
        self.assertEqual(page["attention kv"].layers * page["attention kv"].bytes_per_token, kv_tok)
        self.assertEqual(page["qsa raw keys"].layers * page["qsa raw keys"].bytes_per_token, idx_tok * QWEN38_TEXT["indexer_compress_ratio"])
        self.assertEqual((slot["ple token context"].layers, slot["ple conv state"].bytes_per_seq), (1, 4 * 2560 * 3 * 3 * 2))


def tiny_oracle(dtype="float32", seed=0):
    """A small Qwen4ExpForCausalLM with the real model's structure -- three GDN layers then one QSA layer, PLE before
    layer 2, a sparse indexer that drops blocks (budget 8, ratio 4) -- and every norm and the PLE conv nonzero."""
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpForCausalLM
    torch.manual_seed(seed)
    cfg = Qwen4ExpTextConfig(
        vocab_size=512, hidden_size=64, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, head_dim=32,
        linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16, linear_value_head_dim=16,
        linear_conv_kernel_dim=4, num_experts=8, num_experts_per_tok=2, moe_intermediate_size=32,
        shared_expert_intermediate_size=32, hc_count=4, hc_lowrank=16, ple_layer_ids=[2], ple_embed_dim=32, ngram_size=3,
        heads_per_ngram=8, ngram_vocab_size_base=1000, make_ngram_vocab_size_divisible_by=8, indexer_n_heads=2,
        indexer_kv_heads=1, indexer_head_dim=32, indexer_budget=8, indexer_compress_ratio=4, output_gate_type="sigmoid",
        eos_token_id=0, bos_token_id=0, rms_norm_eps=1e-6, hidden_act="silu",
        rope_parameters={"rope_type": "default", "rope_theta": 10000000.0, "partial_rotary_factor": 0.25,
                         "mrope_section": [2, 1, 1], "mrope_interleaved": True},
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"], dtype=dtype)
    cfg._attn_implementation = "eager"
    model = Qwen4ExpForCausalLM(cfg).eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith(("norm.weight", "layernorm.weight", "ple.conv1d.weight", "norm_key.weight",
                              "norm_query.weight", "norm_conv.weight")):
                p.normal_(0, 0.3)
        model.to(getattr(torch, dtype))
    return cfg, model


@unittest.skipUnless(ORACLE, "requires transformers with qwen4_exp (5.16.1) on the path")
class Qwen38OracleTests(unittest.TestCase):
    """Qwen3.8 assembled from engine/modules features equals transformers' model, token for token, on the CPU."""

    @classmethod
    def setUpClass(cls):
        from engine.profiles.qwen38 import composition as qc
        cls.cfg, cls.model = tiny_oracle()
        cls.sd = cls.model.state_dict()
        cls.comp = qc.build(cls.cfg.to_dict(), cls.sd.__getitem__)
        gen = torch.Generator().manual_seed(1)
        cls.ids = torch.randint(1, 512, (40,), generator=gen)
        with torch.no_grad():
            cls.ref = cls.model(input_ids=cls.ids[None]).logits[0]

    def close(self, got, want, tol=1e-6):
        self.assertEqual(got.shape, want.shape)
        self.assertLessEqual(float((got.float() - want.float()).abs().max()), tol)

    def test_a_prefill_is_the_model(self):
        from engine.base.composition import State, Step
        with torch.no_grad():
            got = self.comp.forward(Step.of([(0, 0, self.ids)]), State(), logits="all")
        self.close(got, self.ref)

    def test_the_indexer_drops_blocks_in_this_prefill(self):
        """The case above is sparse: at the last position there are 10 complete blocks and QSA keeps 2."""
        from engine.modules.sparse_indexer import qsa_select
        self.assertEqual((self.cfg.indexer_budget // self.cfg.indexer_compress_ratio, 40 // self.cfg.indexer_compress_ratio), (2, 10))
        raw = torch.randn(40, 32)
        cos, sin = torch.ones(40, 8), torch.zeros(40, 8)
        chosen = qsa_select(torch.randn(2, 32), raw, 38, 4, 2, cos, sin, torch.zeros(32), 1e-6)
        self.assertEqual(chosen.numel(), 2 * 4 + 3)                          # two blocks and the tail 36..38
        self.assertEqual(chosen[-3:].tolist(), [36, 37, 38])

    def test_decode_steps_are_the_model_with_its_cache(self):
        from engine.base.composition import State, Step
        from transformers.cache_utils import DynamicCache
        with torch.no_grad():
            out = self.model(input_ids=self.ids[None, :23], past_key_values=DynamicCache(config=self.cfg), use_cache=True)
            want = [out.logits[0, -1]]
            for j in range(23, 30):
                out = self.model(input_ids=self.ids[None, j:j + 1], past_key_values=out.past_key_values, use_cache=True)
                want.append(out.logits[0, -1])
            state = State()
            got = [self.comp.forward(Step.of([(0, 0, self.ids[:23])]), state)[0]]
            for j in range(23, 30):
                got.append(self.comp.forward(Step.of([(0, j, self.ids[j:j + 1])]), state)[0])
        for j, (a, b) in enumerate(zip(got, want)):
            with self.subTest(step=j):
                self.close(a, b)

    def test_a_chunked_prefill_is_the_same_prefill(self):
        from engine.base.composition import State, Step
        state, parts = State(), []
        with torch.no_grad():
            for lo, hi in ((0, 9), (9, 25), (25, 40)):
                parts.append(self.comp.forward(Step.of([(0, lo, self.ids[lo:hi])]), state, logits="all"))
        self.close(torch.cat(parts), self.ref)

    def test_sequences_in_one_step_are_independent(self):
        """Two sequences share a step, one with an EOS inside its prompt (PLE's n-grams stop at it), then continue with
        different lengths -- each equals its own run."""
        from engine.base.composition import State, Step
        other = torch.randint(1, 512, (17,), generator=torch.Generator().manual_seed(2))
        other[5] = self.cfg.eos_token_id
        with torch.no_grad():
            other_ref = self.model(input_ids=other[None]).logits[0]
            continued_ref = self.model(input_ids=torch.cat([other, self.ids[21:22]])[None]).logits[0, -1]
            state = State()
            both = self.comp.forward(Step.of([(7, 0, self.ids[:21]), (3, 0, other)]), state, logits="all")
            nxt = self.comp.forward(Step.of([(3, 17, self.ids[21:22]), (7, 21, self.ids[21:24])]), state, logits="all")
        self.close(both[:21], self.ref[:21])
        self.close(both[21:], other_ref)
        self.close(nxt[0], continued_ref)
        self.close(nxt[1:], self.ref[21:24])

    def test_bf16_follows_the_reference_casts(self):
        """In BF16 the features round where transformers rounds; the logits agree to BF16's resolution."""
        from engine.base.composition import State, Step
        from engine.profiles.qwen38 import composition as qc
        cfg, model = tiny_oracle("bfloat16")
        sd = model.state_dict()
        comp = qc.build(cfg.to_dict(), sd.__getitem__)
        with torch.no_grad():
            want = model(input_ids=self.ids[None]).logits[0].float()
            got = comp.forward(Step.of([(0, 0, self.ids)]), State(), logits="all").float()
        self.assertLessEqual(float((got - want).norm() / want.norm()), 2e-2)          # measured 5.1e-3


if __name__ == "__main__":
    unittest.main()
