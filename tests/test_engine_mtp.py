"""engine/modules/mtp: MTP heads as compositions and the drafter that runs them, on the CPU, over the small Qwen3.8-
shaped composition of tests/test_engine_composed.py with a random MTP head in Qwen3.8's layout.

Oracles: the fuse is vLLM's qwen3_8_flash_next MTP forward (nvidia/mtp.py) transcribed op for op; the drafter's
proposals -- through the store's lane, the offset layers, the provisional chain and its step back -- are a rollout
recomputed from nothing on reference States; and every run with the drafter is the run without it (base/composed's
guarantee, now with a drafter that writes into the same blocks). How good the drafts are is the real checkpoint's
question (engine/profiles/qwen38/boot.py --mtp on srv2)."""
import importlib.util
import unittest

torch = None
if importlib.util.find_spec("torch") is not None:
    import torch

from tests.test_engine_composed import TINY, prompt


def tiny_with_mtp(seed=0, layers=1):
    from engine.profiles.qwen38 import composition as qc
    from engine.profiles.qwen38.weights import random_weights
    cfg = dict(TINY, ple_conv_kernel_size=4, mtp_num_hidden_layers=layers)
    weights = random_weights(cfg, seed)
    return qc.build(cfg, weights.__getitem__), qc.build_mtp(cfg, weights.__getitem__), cfg, weights


def reference_drafts(comp, heads, tokens, k, vocab):
    """The drafts a row with history `tokens` (its last one pending, never fed) gets, recomputed from nothing: the
    target over tokens[:-1] on a fresh State, the head over (its states, tokens[1:]) on another, then the chain."""
    from engine.base.composition import State, Step
    context = len(tokens) - 1
    with torch.no_grad():
        _, hidden = comp.forward(Step.of([(0, 0, torch.tensor(tokens[:context]))]), State(), logits="all", hidden=True)
        head_state = State()
        logits, state = heads[0].forward(Step.of([(0, 0, torch.tensor(tokens[1:context + 1]))]), head_state,
                                         hidden=True, given=hidden)
        drafts, state = [int(logits[0, :vocab].argmax())], state[-1:]
        for i in range(1, k):
            logits, state = heads[i % len(heads)].forward(Step.of([(0, context + i - 1, torch.tensor([drafts[-1]]))]),
                                                          head_state, hidden=True, given=state)
            drafts.append(int(logits[0, :vocab].argmax()))
            state = state[-1:]
    return drafts


@unittest.skipUnless(torch is not None, "requires torch")
class FrameTests(unittest.TestCase):
    def test_a_head_at_an_offset_shares_the_layout_and_refuses_a_claimed_layer(self):
        import dataclasses
        from engine.base.composed import Layout, merged_specs
        comp, heads, cfg, _ = tiny_with_mtp()
        paged, slots, layers = merged_specs((comp, *heads))
        self.assertEqual(layers["attention_kv"], [3, 4])                    # the target's one QSA layer, then the head's
        self.assertEqual(layers["qsa_raw_keys"], [3, 4])
        self.assertEqual({s.key: s.layers for s in paged}, {"attention_kv": 2, "qsa_raw_keys": 2})
        self.assertEqual({s.key for s in slots}, {"linear_conv", "linear_state", "ngram_context", "ngram_conv"})
        layout = Layout(paged, slots, 4, layers)
        self.assertEqual((layout.region("attention_kv", 3), layout.region("attention_kv", 4)), (0, 1))
        clash = dataclasses.replace(heads[0], offset=3)
        with self.assertRaisesRegex(ValueError, "claim the same model layer"):
            merged_specs((comp, clash))

    def test_the_lane_writes_its_rows_and_leaves_the_targets(self):
        from engine.base.composed import store_for
        from engine.base.composition import Step
        comp, heads, cfg, _ = tiny_with_mtp()
        store, pool, slots, _ = store_for(comp, kv_gib=0.02, max_seqs=2, block_tokens=4, also=heads)
        ids = prompt(1, 10)
        with torch.no_grad():
            store.open(1, slots.take(1)); pool.reserve(1, 16)
            _, hidden = comp.forward(Step.of([(1, 0, torch.tensor(ids))]), store, logits="all", hidden=True)
            target_rows = store.rows(3, "attention_kv", 1, 10).clone()
            lane = store.lane()
            lane.place(1, 0)
            heads[0].forward(Step.of([(1, 0, torch.tensor(ids[1:] + [5]))]), lane, given=hidden)
            self.assertEqual((store.contexts[1], lane.contexts[1]), (10, 10))
            self.assertTrue(torch.equal(store.rows(3, "attention_kv", 1, 10), target_rows))
            self.assertFalse(torch.equal(lane.rows(4, "attention_kv", 1, 10), torch.zeros_like(target_rows)))
            lane.place(1, 6)
            self.assertEqual((store.contexts[1], lane.contexts[1]), (10, 6))
            with self.assertRaisesRegex(ValueError, "no fuse"):
                comp.forward(Step.of([(1, 10, torch.tensor([3]))]), store, given=hidden[:1])
            with self.assertRaisesRegex(ValueError, "holds 10 rows for a step of 1"):
                heads[0].forward(Step.of([(1, 6, torch.tensor([3]))]), lane, given=hidden)
            with self.assertRaisesRegex(ValueError, "not open"):
                lane.place(2, 0)


@unittest.skipUnless(torch is not None, "requires torch")
class FuseTests(unittest.TestCase):
    def test_fuse_streams_is_vllms_mtp_forward(self):
        """nvidia/mtp.py Qwen3_8FlashNextMultiTokenPredictor.forward, the fuse half, with GemmaRMSNorm written out."""
        from engine.modules.mtp import fuse_streams
        g = torch.Generator().manual_seed(2)
        N, hc, H, eps = 7, 4, 16, 1e-6
        w = {"embed_norm": torch.randn(H, generator=g) * 0.3, "embed_proj": torch.randn(H, H, generator=g) * 0.25,
             "hidden_norm": torch.randn(hc * H, generator=g) * 0.3, "hidden_proj": torch.randn(H, H, generator=g) * 0.25}
        embeds, given = torch.randn(N, H, generator=g), torch.randn(N, hc * H, generator=g)

        def gemma(x, weight):
            dtype = x.dtype
            x = x.float()
            x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
            return (x * (1.0 + weight.float())).to(dtype)
        for dtype in (torch.float32, torch.bfloat16):
            e, h = embeds.to(dtype), given.to(dtype)
            ww = {k: v.to(dtype) for k, v in w.items()}
            inputs_embeds = torch.nn.functional.linear(gemma(e, ww["embed_norm"]), ww["embed_proj"])
            hidden_states = gemma(h.view(N, hc, H).flatten(-2), ww["hidden_norm"]).view(N, hc, H)
            hidden_states = torch.nn.functional.linear(hidden_states, ww["hidden_proj"])
            want = (inputs_embeds.unsqueeze(-2) + hidden_states).flatten(-2)
            got = fuse_streams(e, h, ww.__getitem__, hc=hc, hidden=H, eps=eps)
            self.assertTrue(torch.equal(got, want), dtype)

    def test_fuse_concat_orders_and_zeroes(self):
        from engine.modules.mtp import fuse_concat
        g = torch.Generator().manual_seed(3)
        N, H = 5, 8
        w = {"embed_norm": torch.ones(H), "hidden_norm": torch.ones(H), "proj": torch.randn(H, 2 * H, generator=g)}
        e, h, pos = torch.randn(N, H, generator=g), torch.randn(N, H, generator=g), torch.arange(N)
        from engine.modules.norm import rmsnorm
        want = torch.nn.functional.linear(torch.cat([rmsnorm(e, w["embed_norm"], 1e-6).masked_fill((pos == 0)[:, None], 0),
                                                     rmsnorm(h, w["hidden_norm"], 1e-6)], -1), w["proj"])
        self.assertTrue(torch.equal(fuse_concat(e, h, w.__getitem__, eps=1e-6, positions=pos), want))
        flipped = fuse_concat(e, h, w.__getitem__, eps=1e-6, positions=pos + 1, state_first=True)
        self.assertEqual(tuple(flipped.shape), (N, H))


@unittest.skipUnless(torch is not None, "requires torch")
class DrafterTests(unittest.TestCase):
    K = 3

    def model(self, *, rows=2, prefix=None, max_new=10, drafter=True, seed=0, layers=1):
        from engine.base.composed import ComposedModel, store_for
        from engine.base.record import Ring
        from engine.base.runner import STEP_RECORD, Runner
        from engine.base.scheduler import Contract
        from engine.modules.mtp import MTPDrafter
        comp, heads, cfg, _ = tiny_with_mtp(seed, layers)
        k = self.K if drafter else 0
        store, pool, slots, _ = store_for(comp, kv_gib=0.04, max_seqs=rows, block_tokens=4, ring=k + 1,
                                          snapshots=prefix.snapshots if prefix else 0, also=heads if drafter else ())
        made = MTPDrafter(heads, store, k=self.K, vocab=cfg["vocab_size"]) if drafter else None
        model = ComposedModel(comp, store, vocab=cfg["vocab_size"], eos_ids=[cfg["eos_token_id"]], max_new=max_new,
                              temperature=0.0, drafter=made)
        runner = Runner(model, Contract(chunk_align=4, token_budget=16, draft_slots=k, max_wait_s=0.0, max_running=rows),
                        pool, slots, Ring(64, STEP_RECORD.size), prefix=prefix)
        return comp, heads, cfg, model, runner, pool, slots

    def run_all(self, runner):
        for _ in range(400):
            if runner.step(now=0.0) is None:
                return
        raise AssertionError("the runner did not finish")

    def test_proposals_are_the_rollout_from_nothing(self):
        comp, heads, cfg, model, runner, pool, slots = self.model(rows=1, max_new=12)
        ids = prompt(4, 11)
        model.add(0, ids, max_new=12, temperature=0.0)
        slot = slots.take(0)
        model.open(0, slot)
        pool.reserve(0, 64)
        with torch.no_grad():
            model.prefill(0, 0, 7, None, slot)                               # two prefill pieces: observed in pieces
            model.prefill(0, 7, 4, None, slot)
            for step in range(4):
                tokens = list(model.tokens[0])
                with self.subTest(step=step):
                    self.assertEqual(model.drafter.propose([0])[0], reference_drafts(comp, heads, tokens, self.K, 512))
                    self.assertEqual(model.drafter.lane.contexts[0], len(tokens) - 1)   # stepped back after the chain
                model.decode([0], None, None)
        self.assertGreater(model.drafted_total, 0)

    def test_accepted_drafts_are_observed_and_the_next_rollout_still_matches(self):
        """The head's own chain runs every step (its rows, its step back) and is checked against the rollout from
        nothing; the model is handed the true continuation instead, so drafts are accepted and the head observes
        several kept positions at once."""
        from engine.modules.mtp import MTPDrafter
        _, _, _, plain, plain_runner, _, _ = self.model(rows=1, drafter=False, max_new=12)
        ids = prompt(5, 9)
        plain.add(0, ids, max_new=12, temperature=0.0); plain_runner.submit(0, len(ids), now=0.0); self.run_all(plain_runner)
        truth = list(plain.tokens[0])
        comp, heads, cfg, model, runner, _, _ = self.model(rows=1, max_new=12)
        checked = []

        class Honest(MTPDrafter):
            def propose(inner, seqs):
                chains = MTPDrafter.propose(inner, seqs)
                tokens = list(model.tokens[0])
                if chains[0]:
                    checked.append((chains[0], reference_drafts(comp, heads, tokens, self.K, 512)))
                return [truth[len(tokens):len(tokens) + self.K]]
        honest = Honest(heads, model.store, k=self.K, vocab=512)
        model.drafter = honest
        model.add(0, ids, max_new=12, temperature=0.0); runner.submit(0, len(ids), now=0.0); self.run_all(runner)
        self.assertEqual(model.generated(0), plain.generated(0))
        self.assertGreater(model.accepted_total, 0)
        self.assertGreaterEqual(len(checked), 2)
        for chain, want in checked:
            self.assertEqual(chain, want)

    def test_runs_with_the_head_are_the_runs_without_it(self):
        requests = [(0, prompt(1, 13), 10), (1, prompt(2, 6), 8)]
        outs = []
        for drafter in (False, True):
            _, _, _, model, runner, _, _ = self.model(drafter=drafter)
            for seq, ids, max_new in requests:
                model.add(seq, ids, max_new=max_new, temperature=0.0)
                runner.submit(seq, len(ids), ids=ids, now=0.0)
            self.run_all(runner)
            outs.append([model.generated(seq) for seq, _, _ in requests])
            if drafter:
                self.assertEqual(model.drafts_total > 0 and model.drafted_total >= model.accepted_total, True)
        self.assertEqual(outs[1], outs[0])

    def test_two_head_layers_chain_and_an_adopted_prefix_carries_on(self):
        from engine.base.prefix import PrefixCache
        b = prompt(7, 13)
        c = b[:10] + prompt(8, 3)
        outs = []
        for drafter in (False, True):
            _, _, _, model, runner, _, _ = self.model(prefix=PrefixCache(4, 8, 4), drafter=drafter, layers=2)
            model.add(0, b, max_new=4, temperature=0.0); runner.submit(0, len(b), ids=b, now=0.0); self.run_all(runner)
            model.add(1, c, max_new=6, temperature=0.0); runner.submit(1, len(c), ids=c, now=0.0)
            self.assertEqual(runner.reused_tokens, 8)
            self.run_all(runner)
            outs.append((model.generated(0), model.generated(1)))
        self.assertEqual(outs[1], outs[0])

    def test_a_head_must_fuse_and_keep_no_per_sequence_values(self):
        import dataclasses
        from engine.base.composed import store_for
        from engine.modules.mtp import MTPDrafter
        comp, heads, cfg, _ = tiny_with_mtp()
        store, *_ = store_for(comp, kv_gib=0.02, max_seqs=1, block_tokens=4, also=heads)
        with self.assertRaisesRegex(ValueError, "needs a fuse"):
            MTPDrafter([dataclasses.replace(heads[0], fuse=None)], store, k=2, vocab=512)
        with self.assertRaisesRegex(ValueError, "verify rings of its own"):
            MTPDrafter([dataclasses.replace(comp, fuse=heads[0].fuse)], store, k=2, vocab=512)
        with self.assertRaisesRegex(ValueError, "k >= 1"):
            MTPDrafter(heads, store, k=0, vocab=512)


if __name__ == "__main__":
    unittest.main()
