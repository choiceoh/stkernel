"""engine/base/composed: the position-addressed store over blocks and slots, and the composed model the runner and
the door drive -- on the CPU, with a small Qwen3.8-shaped composition over random weights (no oracle needed: the
reference State's answer is the oracle for the store's, and a hand-run greedy loop is the oracle for the runner's).
"""
import importlib.util
import queue
import unittest

torch = None
if importlib.util.find_spec("torch") is not None:
    import torch

TINY = {
    "hidden_size": 64, "num_hidden_layers": 4, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 32,
    "linear_num_key_heads": 2, "linear_num_value_heads": 4, "linear_key_head_dim": 16, "linear_value_head_dim": 16,
    "linear_conv_kernel_dim": 4, "num_experts": 8, "num_experts_per_tok": 2, "moe_intermediate_size": 32,
    "shared_expert_intermediate_size": 32, "hidden_act": "silu", "output_gate_type": "sigmoid", "norm_topk_prob": True,
    "hc_count": 4, "hc_lowrank": 16, "ple_layer_ids": [2], "ple_embed_dim": 32, "ngram_size": 3, "heads_per_ngram": 8,
    "ngram_vocab_size_base": 1000, "make_ngram_vocab_size_divisible_by": 8, "seed": 1234,
    "indexer_n_heads": 2, "indexer_kv_heads": 1, "indexer_head_dim": 32, "indexer_budget": 8, "indexer_compress_ratio": 4,
    "vocab_size": 512, "eos_token_id": 0, "rms_norm_eps": 1e-6, "dtype": "float32",
    "rope_parameters": {"rope_type": "default", "rope_theta": 10000000.0, "partial_rotary_factor": 0.25,
                        "mrope_section": [2, 1, 1], "mrope_interleaved": True},
    "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
}


def tiny_composition(seed: int = 0):
    from engine.profiles.qwen38 import composition as qc
    from engine.profiles.qwen38.weights import random_weights
    cfg = dict(TINY, ple_conv_kernel_size=4)
    weights = random_weights(cfg, seed)
    return qc.build(cfg, weights.__getitem__), cfg


def prompt(seed: int, length: int) -> "list[int]":
    g = torch.Generator().manual_seed(seed)
    return torch.randint(1, TINY["vocab_size"], (length,), generator=g).tolist()


def reference_generate(comp, ids: "list[int]", count: int) -> "list[int]":
    """Greedy continuation of `ids`, `count` tokens, through the reference State one token at a time."""
    from engine.base.composition import State, Step
    state, out, ctx = State(), [], 0
    tokens = list(ids)
    with torch.no_grad():
        logits = comp.forward(Step.of([(0, 0, torch.tensor(tokens))]), state)
        for _ in range(count):
            out.append(int(logits[0].argmax()))
            ctx = len(tokens) + len(out) - 1
            logits = comp.forward(Step.of([(0, ctx, torch.tensor([out[-1]]))]), state)
    return out


@unittest.skipUnless(torch is not None, "requires torch")
class StoreTests(unittest.TestCase):
    """PositionStore is the reference State over blocks and slots: the same rows, the same values, by position."""

    def setUp(self):
        self.comp, self.cfg = tiny_composition()

    def test_the_layout_carves_every_keyed_spec(self):
        from engine.base.composed import ALIGN, Layout
        paged, slots = self.comp.cache_specs()
        layout = Layout(paged, slots, 4, self.comp.spec_layers())
        self.assertEqual(set(layout.paged), {"attention_kv", "qsa_raw_keys"})
        self.assertEqual(set(layout.slot), {"linear_conv", "linear_state", "ngram_context", "ngram_conv"})
        for at, spec in list(layout.paged.values()) + list(layout.slot.values()):
            self.assertEqual(at % ALIGN, 0)
        kv = layout.paged["attention_kv"][1]
        self.assertEqual((kv.dtype, kv.shape, kv.layers), ("float32", (2, 2, 32), 1))
        self.assertEqual((layout.region("attention_kv", 3), layout.region("ngram_context", 1)), (0, 0))
        with self.assertRaisesRegex(ValueError, "not kept for layer 0"):
            layout.region("attention_kv", 0)
        self.assertGreaterEqual(layout.block_bytes, 4 * (kv.bytes_per_token + layout.paged["qsa_raw_keys"][1].bytes_per_token))
        plan = layout.plan(kv_gib=0.001, max_seqs=3)
        self.assertEqual((plan.block_bytes % 4096, plan.num_slots), (0, 4))
        self.assertGreater(plan.num_blocks, 0)

    def test_the_store_answers_like_the_reference_state(self):
        from engine.base.composed import store_for
        from engine.base.composition import State, Step
        store, pool, slots, plan = store_for(self.comp, kv_gib=0.01, max_seqs=3, block_tokens=4)
        ref = State()
        a, b = prompt(1, 13), prompt(2, 6)
        with torch.no_grad():
            # two sequences, opened into slots with blocks reserved, prefilled in pieces, then decoded together
            for seq in (2, 1):
                store.open(seq, slots.take(seq))
            pool.reserve(2, 13 + 3); pool.reserve(1, 6 + 3)
            want, got = [], []
            for seq, ids in ((2, a), (1, b)):
                for lo, hi in ((0, 5), (5, len(ids))):
                    piece = torch.tensor(ids[lo:hi])
                    want.append(self.comp.forward(Step.of([(seq, lo, piece)]), ref, logits="all"))
                    got.append(self.comp.forward(Step.of([(seq, lo, piece)]), store, logits="all"))
            nxt = [(2, 13, torch.tensor([a[-1]])), (1, 6, torch.tensor([b[-1]]))]
            want.append(self.comp.forward(Step.of(nxt), ref, logits="all"))
            got.append(self.comp.forward(Step.of(nxt), store, logits="all"))
        for w, g in zip(want, got):
            self.assertLessEqual(float((w - g).abs().max()), 1e-5)
        self.assertEqual((store.contexts[2], store.contexts[1]), (14, 7))
        self.assertEqual(tuple(store.rows(3, "attention_kv", 2, 14).shape), (14, 2, 2, 32))
        with self.assertRaisesRegex(ValueError, "holds 14 positions"):
            store.rows(3, "attention_kv", 2, 15)
        with self.assertRaisesRegex(ValueError, "at 14 tokens"):
            self.comp.forward(Step.of([(2, 12, torch.tensor([1]))]), store)
        store.close(2)
        self.assertNotIn(2, store.contexts)

    def test_checkpoint_and_restore_carry_the_state_at_a_boundary(self):
        from engine.base.composed import store_for
        from engine.base.composition import State, Step
        store, pool, slots, _ = store_for(self.comp, kv_gib=0.01, max_seqs=3, block_tokens=4, snapshots=2)
        ids = prompt(3, 12)
        with torch.no_grad():
            store.open(1, slots.take(1)); pool.reserve(1, 13)
            self.comp.forward(Step.of([(1, 0, torch.tensor(ids[:8]))]), store)
            store.checkpoint(1, 8, snap=1)                                    # the boundary at block 2
            with self.assertRaisesRegex(ValueError, "stands at 8"):
                store.checkpoint(1, 4, snap=0)
            self.comp.forward(Step.of([(1, 8, torch.tensor(ids[8:]))]), store)
            want = self.comp.forward(Step.of([(1, 12, torch.tensor([ids[-1]]))]), store)
            # a second sequence adopts the first's blocks up to position 8 and the snapshot, then computes the rest
            store.open(2, slots.take(2))
            pool.adopt(2, list(pool.row(1)[:2]), 8)
            pool.reserve(2, 4 + 1)
            store.restore(2, 8, snap=1)
            self.assertEqual(store.contexts[2], 8)
            self.comp.forward(Step.of([(2, 8, torch.tensor(ids[8:]))]), store)
            got = self.comp.forward(Step.of([(2, 12, torch.tensor([ids[-1]]))]), store)
            ref = State()
            self.comp.forward(Step.of([(0, 0, torch.tensor(ids))]), ref)
            plain = self.comp.forward(Step.of([(0, 12, torch.tensor([ids[-1]]))]), ref)
        self.assertLessEqual(float((got - want).abs().max()), 1e-5)
        self.assertLessEqual(float((got - plain).abs().max()), 1e-5)
        self.assertEqual(store.snapshot_bytes(1).numel(), store.layout.slot_bytes)

    def test_the_store_refuses_what_a_kernel_would_read_wrong(self):
        from engine.base.composed import store_for
        from engine.base.composition import Step
        store, pool, slots, _ = store_for(self.comp, kv_gib=0.01, max_seqs=2, block_tokens=4)
        with self.assertRaisesRegex(ValueError, "not open"):
            self.comp.forward(Step.of([(1, 0, torch.tensor([1, 2]))]), store)
        store.open(1, slots.take(1))
        with self.assertRaisesRegex(ValueError, "fewer blocks"):
            self.comp.forward(Step.of([(1, 0, torch.tensor([1, 2]))]), store)   # no blocks reserved
        with self.assertRaisesRegex(ValueError, "slot 0"):
            store.open(2, 0)
        self.assertIsNone(store.get(0, "linear_state", 1))                          # nothing computed yet


@unittest.skipUnless(torch is not None, "requires torch")
class RunnerTests(unittest.TestCase):
    """The runner drives a ComposedModel as it drives GLM's adapter: homogeneous steps, one token a decode step, finish
    at the limit or an end token, more turns on a kept row, a cached prefix adopted."""

    def build(self, *, rows=2, keep_idle=False, prefix=None, max_new=6, block_tokens=4, seed=0):
        from engine.base.composed import ComposedModel, store_for
        from engine.base.record import Ring
        from engine.base.runner import STEP_RECORD, Runner
        from engine.base.scheduler import Contract
        comp, cfg = tiny_composition(seed)
        store, pool, slots, _ = store_for(comp, kv_gib=0.02, max_seqs=rows, block_tokens=block_tokens,
                                          snapshots=prefix.snapshots if prefix else 0)
        model = ComposedModel(comp, store, vocab=cfg["vocab_size"], eos_ids=[cfg["eos_token_id"]], max_new=max_new, temperature=0.0)
        runner = Runner(model, Contract(chunk_align=4, token_budget=8, draft_slots=0, max_wait_s=0.0, max_running=rows),
                        pool, slots, Ring(64, STEP_RECORD.size), keep_idle=keep_idle, prefix=prefix)
        return comp, model, runner

    def run_all(self, runner, limit=200):
        kinds = []
        for _ in range(limit):
            step = runner.step(now=0.0)
            if step is None:
                break
            kinds.append(step.kind)
        return kinds

    def test_greedy_generation_through_the_runner_is_the_reference_loop(self):
        comp, model, runner = self.build()
        a, b = prompt(1, 13), prompt(2, 5)
        model.add(0, a, max_new=6, temperature=0.0)
        model.add(1, b, max_new=4, temperature=0.0)
        runner.submit(0, len(a), now=0.0)
        runner.submit(1, len(b), now=0.0)
        kinds = self.run_all(runner)
        self.assertEqual(kinds[:2], ["prefill", "prefill"])                    # 13 tokens: 8 + 5 (chunk 8, aligned 4)
        self.assertEqual(set(kinds[2:]), {"prefill", "decode"})
        self.assertEqual(model.generated(0), reference_generate(comp, a, 6))
        self.assertEqual(model.generated(1), reference_generate(comp, b, 4))
        self.assertEqual((runner.state.running, runner.kv.available == runner.kv.num_blocks, runner.slots.available), ([], True, 2))

    def test_an_end_token_finishes_the_row_and_min_tokens_holds_it_back(self):
        comp, model, runner = self.build(rows=1)
        a = prompt(4, 6)
        plain = reference_generate(comp, a, 6)
        stop = plain[2]
        model.add(0, a, max_new=6, temperature=0.0, options={"stop_token_ids": [stop]})
        runner.submit(0, len(a), now=0.0)
        self.run_all(runner)
        self.assertEqual(model.generated(0), plain[:3])                       # ends with the stop token, then stops
        model.forget(0)
        model.add(0, a, max_new=6, temperature=0.0, min_new=5, options={"stop_token_ids": [stop]})
        runner.submit(0, len(a), now=0.0)
        self.run_all(runner)
        held = model.generated(0)
        self.assertGreaterEqual(len(held), 5)
        self.assertNotIn(stop, held[:5])                                       # replaced by the next best pick while held

    def test_a_second_turn_continues_the_kept_state(self):
        comp, model, runner = self.build(rows=1, keep_idle=True)
        a = prompt(5, 9)
        first = reference_generate(comp, a, 3)
        model.add(0, a, max_new=3, temperature=0.0)
        runner.submit(0, len(a), now=0.0)
        self.run_all(runner)
        self.assertEqual(model.generated(0), first)
        self.assertIn(0, runner.idle)
        more = prompt(6, 4)
        pending = model.extend(0, more, max_new=3, temperature=0.0)
        self.assertEqual(pending, 1 + 4)                                       # the last sampled token and the new ones
        runner.extend(0, pending, now=0.0)
        self.run_all(runner)
        self.assertEqual(model.generated(0), reference_generate(comp, a + first + more, 3))

    def test_a_cached_prefix_is_adopted_and_answers_the_same(self):
        from engine.base.prefix import PrefixCache
        comp, model, runner = self.build(rows=2, prefix=PrefixCache(4, 8, 4))
        a = prompt(7, 13)
        b = a[:10] + prompt(8, 3)                                              # shares two whole blocks (8 tokens)
        model.add(0, a, max_new=2, temperature=0.0)
        runner.submit(0, len(a), ids=a, now=0.0)
        self.run_all(runner)
        self.assertEqual(model.generated(0), reference_generate(comp, a, 2))
        model.add(1, b, max_new=3, temperature=0.0)
        runner.submit(1, len(b), ids=b, now=0.0)
        self.assertEqual(runner.reused_tokens, 8)
        self.assertEqual(runner.state.computed[1], 8)
        self.run_all(runner)
        self.assertEqual(model.generated(1), reference_generate(comp, b, 3))

    def test_a_parked_row_resumes_from_its_record_and_slot_bytes(self):
        comp, model, runner = self.build(rows=2, keep_idle=True)
        a = prompt(9, 7)
        model.add(0, a, max_new=2, temperature=0.0)
        runner.submit(0, len(a), now=0.0)
        self.run_all(runner)
        slot = runner.slot_of[0]
        held = model.state_bytes(slot).clone()
        record = model.park(0)
        self.assertEqual((record["context"], record["pending"], record["tokens"]), (8, 1, a + reference_generate(comp, a, 2)))
        self.assertNotIn(0, model.tokens)
        # the tier brings the bytes back into another slot; the row's blocks stay (this test moves no blocks)
        other = 2
        model.state_bytes(other).copy_(held)
        model.resume(0, other, record)
        self.assertEqual(model.context(0), 8)
        runner.kv.reserve_to((0,), (9,))                                       # the position the next token takes
        with torch.no_grad():
            from engine.base.composition import Step
            got = comp.forward(Step.of([(0, 8, torch.tensor([model.tokens[0][-1]]))]), model.store)
        self.assertEqual(int(got[0].argmax()), reference_generate(comp, a, 3)[2])


@unittest.skipUnless(torch is not None, "requires torch")
class DoorTests(unittest.TestCase):
    """The door (base/serve) over a ComposedModel, world 1: a request in, its tokens out, the same tokens as the loop."""

    def test_a_request_through_the_door(self):
        from engine.base.comm import Comm
        from engine.base.serve import Server
        comp, model, runner = RunnerTests().build(rows=2, max_new=5)
        s = Server(model, runner, Comm(), host="127.0.0.1", port=0)
        a = prompt(11, 9)
        request, event = s.submit(a, 4, 0.0)
        for _ in range(200):
            ran = s.once()
            if not ran and not s._waiting and not s._retiring and not s._resuming:
                break
        self.assertTrue(event.is_set())
        self.assertEqual(s.take_result(request), reference_generate(comp, a, 4))
        self.assertEqual(s.served, 1)
        model.validate_options({"logprobs": 2, "presence_penalty": 0.5, "logit_bias": {3: 1.0}})   # served now
        with self.assertRaisesRegex(Exception, "no grammar compiler is bound"):
            model.validate_options({"grammar": {"type": "json_object"}})


if __name__ == "__main__":
    unittest.main()
