"""Device option semantics against the synchronous sampler; model-free async integration."""
import unittest
from types import SimpleNamespace

import torch

from engine.base.sampler import History, process_logits, top_logprobs, top_logprobs_batch
from engine.base.sampling_options import SamplingState, needs_device_policy
from engine.profiles.glm53.pipeline import AsyncDecode


class PolicyCases:
    device = "cpu"

    def state(self, options=None, minimum=None, vocab=1030):
        return SamplingState.from_histories(
            options or [{"repetition_penalty": 1.3, "presence_penalty": .5, "frequency_penalty": -.2,
                         "logit_bias": {7: 1.5, 12: -2.0, 1025: .3}}, {}],
            [[5, 5, 7, 12, 12], [7, 8, 9]], [3, 2], minimum or [4, 0], vocab, self.device)

    def test_every_position_matches_the_reference_with_duplicate_drafts_and_shards(self):
        state = self.state()
        cpu = SamplingState.from_histories(state.options, [[5, 5, 7, 12, 12], [7, 8, 9]],
                                          [3, 2], [4, 0], 1030, "cpu")
        torch.manual_seed(5)
        full = torch.randn(16, 1030)
        drafts = torch.tensor([[12, 12, 7, 31, 6, 1025, 9], [8, 9, 8, 1, 2, 3, 4]])
        generated, ends = torch.tensor([2, 1]), torch.tensor([[31, 9, -1], [30, -1, -1]])
        for dtype in (torch.float32, torch.bfloat16, torch.float16):
            raw = full.to(dtype)
            reference = cpu.process(raw, drafts, generated, ends, decodable=1028).cpu()
            for start, width in ((0, 1030), (0, 515), (515, 515)):
                # Exercise noncontiguous input/output and ensure neighbours are untouched.
                storage = torch.full((16, 2*width+2), 4321., device=self.device)
                out = storage[:, 1:1+2*width:2]
                x = raw.to(self.device)[:, start:start+width]
                got = state.process(x, drafts.to(self.device), generated.to(self.device), ends.to(self.device),
                                    start=start, decodable=1028, out=out)
                torch.testing.assert_close(got.cpu(), reference[:, start:start+width], rtol=2e-6, atol=2e-6)
                self.assertTrue(bool((storage[:, ::2] == 4321).all()))
                self.assertTrue(bool((storage[:, -1] == 4321).all()))

    def test_only_accepted_clipped_tokens_enter_history(self):
        state = self.state()
        before = state.counts.clone()
        tokens = torch.tensor([[12, 12, 7, 9], [8, 9, 8, 9]], device=self.device)
        state.commit(tokens, torch.tensor([2, 0], device=self.device))
        expected = before.clone()
        expected[0, 12] += 2
        torch.testing.assert_close(state.counts, expected, rtol=0, atol=0)
        state.commit(tokens, torch.tensor([0, 0], device=self.device))
        torch.testing.assert_close(state.counts, expected, rtol=0, atol=0)
        # A newly emitted token also becomes "seen" for repetition penalties.
        state.commit(torch.tensor([[25], [26]], device=self.device), torch.ones(2, dtype=torch.int64, device=self.device))
        state.penalties[:, 0] = 2
        for opts in state.options:
            opts["repetition_penalty"] = 2
        result = state.process(torch.full((2, 1030), 4., device=self.device),
                               torch.empty(2, 0, dtype=torch.int64, device=self.device),
                               torch.tensor([5, 2], device=self.device),
                               torch.empty(2, 0, dtype=torch.int64, device=self.device))
        self.assertAlmostEqual(float(result[0, 25]), 1.7, places=5)
        self.assertEqual(float(result[1, 26]), 2.)

    def test_force_cannot_override_minimum_and_applies_after_bias(self):
        state = self.state(minimum=[3, 0])
        raw = torch.ones(6, 1030, device=self.device)
        drafts = torch.tensor([[7, 31], [8, 9]], device=self.device)
        generated = torch.tensor([2, 1], device=self.device)
        ends = torch.tensor([[31], [31]], device=self.device)
        forces = torch.tensor([31, 31, 1029, -1, 9, 31], device=self.device)
        got = state.process(raw, drafts, generated, ends, decodable=1028, forces=forces).cpu()
        self.assertTrue(torch.isneginf(got[0, 31]))
        self.assertGreater(int(torch.isfinite(got[0]).sum()), 1)
        for row, token in ((1, 31), (2, 1029), (4, 9), (5, 31)):
            self.assertEqual(torch.isfinite(got[row]).nonzero().flatten().tolist(), [token])
        self.assertEqual(float(got[2, 1029]), 0.)

    def test_compact_logprobs_match_existing_scores_including_zero_top_count(self):
        state = self.state(options=[{"logprobs": 0}, {"logprobs": 3}])
        raw = torch.arange(4*1030, dtype=torch.float32, device=self.device).view(4, 1030) / 713
        picks = torch.tensor([[1, 100], [200, 800]], device=self.device)
        packet = state.logprob_packet(raw, picks)
        for row in range(2):
            for pos in range(2):
                token = int(picks[row, pos])
                score, tops = top_logprobs(raw[row*2+pos], token, 3)
                self.assertAlmostEqual(float(packet["logprob"][row, pos]), score, places=6)
                self.assertEqual(packet["top_ids"][row, pos].tolist(), [i for i, _ in tops])
                torch.testing.assert_close(packet["top_logprobs"][row, pos].cpu(),
                                           torch.tensor([p for _, p in tops]), rtol=0, atol=0)
        state.logprobs = [0, 0]
        packet = state.logprob_packet(raw, picks)
        self.assertEqual(packet["top_ids"].shape, (2, 2, 0))
        for k in (0, 3):
            got = top_logprobs_batch(raw, picks.flatten().tolist(), k)
            self.assertEqual(got, [(i, *top_logprobs(row, i, k)) for row, i in zip(raw, picks.flatten().tolist())])

    def test_select_and_join_preserve_device_progress_and_option_ownership(self):
        state = self.state()
        state.commit(torch.tensor([[17, 17], [18, 19]], device=self.device),
                     torch.tensor([2, 1], device=self.device))
        selected = state.select(torch.tensor([1, 0], device=self.device), [1, 0])
        joined = selected.join(state.select(torch.tensor([0], device=self.device), [0]))
        self.assertEqual(joined.counts[:, 17].tolist(), [0, 2, 2])
        self.assertEqual(joined.counts[:, 18].tolist(), [1, 0, 0])
        self.assertEqual(joined.minimum.tolist(), [0, 4, 4])
        self.assertEqual(joined.options[0], {})
        self.assertEqual(joined.options[1], state.options[0])


class CpuPolicyTests(PolicyCases, unittest.TestCase):
    def test_a_new_turn_invalidates_cached_policy_and_stop_ids(self):
        from tests.test_engine_grammar import PickRichTests
        e, _, _ = PickRichTests().engine()
        e._rich_policies = {0: object()}
        e._ends_tensor[0] = torch.tensor([31])
        e.thinking = {}
        e.lps = {}
        e.min_new[0] = 1
        e._bind_options(0, {"stop_token_ids": [7], "frequency_penalty": .5})
        self.assertEqual(e._rich_policies, {})
        self.assertEqual(e._ends_tensor, {})
        result = e._row_logits(0, torch.ones(128), 0, [])
        self.assertTrue(torch.isneginf(result[7]))
        self.assertTrue(torch.isfinite(result[31]))


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class CudaPolicyTests(PolicyCases, unittest.TestCase):
    device = "cuda"

    @classmethod
    def setUpClass(cls):
        torch.cuda.set_per_process_memory_fraction(.12)

    def test_graph_replay_reads_new_history_generation_and_drafts(self):
        state = self.state()
        x = torch.ones(4, 1030, device="cuda")
        drafts = torch.tensor([[12], [9]], device="cuda")
        generated = torch.tensor([2, 1], device="cuda")
        ends = torch.tensor([[31], [31]], device="cuda")
        out = torch.empty_like(x)
        count = torch.tensor([1, 0], device="cuda")
        tokens = torch.tensor([[25, 9], [26, 9]], device="cuda")
        state.process(x, drafts, generated, ends, out=out)
        state.commit(tokens, torch.zeros_like(count))
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            state.process(x, drafts, generated, ends, out=out)
            state.commit(tokens, count)
        graph.replay()
        torch.cuda.synchronize()
        self.assertTrue(torch.isneginf(out[0, 31]))
        self.assertEqual(float(state.counts[0, 25]), 1.)
        generated.fill_(10)
        drafts[0, 0] = 25
        graph.replay()
        torch.cuda.synchronize()
        self.assertTrue(torch.isfinite(out[0, 31]))
        self.assertAlmostEqual(float(out[1, 25]), 1/1.3 + .4 - .5, places=5)
        self.assertEqual(float(state.counts[0, 25]), 2.)

    def test_grammar_live_span_penalties_and_logprobs_match_the_host_path(self):
        from tests.test_engine_grammar import PickRichTests
        fixture = PickRichTests()
        engines = [fixture.engine(refuse=(4,))[0] for _ in range(2)]
        results = []
        for e, device in zip(engines, ("cpu", "cuda")):
            e.options[0] = {"frequency_penalty": .5, "repetition_penalty": 1.2,
                            "logit_bias": {5: 2}, "logprobs": 3}
            e.ends[0] = set()
            e.lps = {0: []}
            # The fake matcher uses the real speculative walk and packed buffer.
            # Its masking reference uses CPU bit operations; upload just the mask.
            masks = e.grammars.prepare([(0, e.matchers[0], [5, 4, 3])], "cpu")
            if device == "cuda":
                apply = masks.apply
                def apply_on_device(seq, block):
                    reference = block.cpu()
                    apply(seq, reference)
                    block.copy_(reference)
                masks.apply = apply_on_device
            raw = fixture.logits([5, 70]).to(device)
            raw[:, 1] = .2  # top-k tie ordering differs between CPU and CUDA
            results.append(e._pick_rich([(0, raw, [5, 4, 3], None)], masks))
            self.assertEqual(e.matchers[0].m.accepted, [], "speculative matcher state rolled back")
            held = getattr(e, "_rich_policies", {}).get(0)
            if device == "cuda":
                self.assertIsNotNone(held)
                e._forget_history(0)
                self.assertNotIn(0, e._rich_policies)
        self.assertEqual(results[0][0][:2], results[1][0][:2])
        for cpu, cuda in zip(results[0][0][2], results[1][0][2]):
            self.assertEqual(cpu[0], cuda[0])
            self.assertAlmostEqual(cpu[1], cuda[1], places=5)
            self.assertEqual([i for i, _ in cpu[2]], [i for i, _ in cuda[2]])

    def test_boot_warmup_prepares_both_vocabulary_widths(self):
        from engine.base.sampling_options import warm_sampling_options
        logits = torch.empty(16, 515, dtype=torch.bfloat16, device="cuda")
        prepared = warm_sampling_options(logits, 1030, 8, 515, 1028)
        self.assertEqual([(offset, tuple(x.shape)) for offset, x in prepared],
                         [(515, (16, 515)), (0, (16, 1030))])
        self.assertTrue(all(x.dtype == torch.float32 for _, x in prepared))
        single_rank = warm_sampling_options(logits, 515, 8, 0, 513)
        self.assertEqual([(offset, tuple(x.shape)) for offset, x in single_rank], [(0, (16, 515))])

    def test_real_json_matcher_and_cuda_mask_commit_the_same_output(self):
        from tests.test_engine_grammar import GrammarTests, PickRichTests
        answers = []
        for device in ("cpu", "cuda"):
            grammar = GrammarTests()
            grammar.setUp()  # skips only when the optional grammar stack is absent
            e, _, _ = PickRichTests().engine(vocab=8)
            e.grammars = grammar.grammars
            e.matchers = {0: e.grammars.matcher({"type": "json_object"}, max_rollback=5)}
            e.tokens, e.ends, e.eos, e.decodable = {0: [2]}, {0: {7}}, {7}, 8
            e.options[0] = {"logprobs": 2, "frequency_penalty": .5, "repetition_penalty": 1.2,
                            "logit_bias": {0: .2}}
            e.lps, e.accepted_total, e.drafted_total, e.accepted_per_step = {0: []}, 0, 0, [0]*5
            raw = torch.arange(24, dtype=torch.float32, device=device).view(3, 8)
            raw[:, 2] = 100  # forbidden at the JSON boundaries
            for pos, tok in enumerate((0, 1, 7)):
                raw[pos, tok] = 50
            masks = e.grammars.prepare([(0, e.matchers[0], [0, 1])], device)
            (accepted, new, lp), = e._pick_rich([(0, raw, [0, 1], None)], masks)
            self.assertFalse(e.matchers[0].m.is_terminated(), "preparation must roll back")
            kept, done = e._commit(0, accepted, new, lp, 2)
            self.assertTrue(done)
            self.assertTrue(e.matchers[0].m.is_terminated(), "only committed EOS terminates")
            self.assertEqual(kept, [0, 1, 7])  # {} then EOS
            self.assertEqual(len(e.lps[0]), 3)
            answers.append(new)
            e.grammars._pool.shutdown()
        self.assertEqual(answers[0], answers[1])


class AsyncOptionTests(unittest.TestCase):
    def engine(self, options, logits, *, minimum=0, limit=10):
        from tests.test_engine_pipeline import BatchTransitionTests
        e = BatchTransitionTests().engine()
        e.options, e.min_new, e.prompt_len = {1: options}, {1: minimum}, {1: 1, 2: 1}
        e.lps = {1: []} if options.get("logprobs") is not None else {}
        e.limits[1] = (limit, 0.0)
        e.seeds, e.decodable = {}, 32
        e.net = SimpleNamespace(rank=0, vp=32, comm=SimpleNamespace(world_size=1,
                               all_gather=lambda x, dim: x, all_reduce_max=lambda x: x))
        e.decode_graphs.run_device = lambda shape, *args: (None, None, logits.repeat(shape[0], 1))
        return e

    def test_two_pending_steps_use_committed_counts_without_host_resolve(self):
        logits = (-.1 - torch.arange(32) * .001).repeat(2, 1)
        logits[:, 7], logits[:, 8] = 4, 3
        e = self.engine({"frequency_penalty": 2, "logprobs": 2}, logits)
        p = AsyncDecode(e)
        first = p.launch([1], [1])
        second = p.launch([1], [1])
        self.assertEqual(e.tokens[1], [5])
        self.assertEqual(e.lps[1], [])
        self.assertEqual(p.buf["sampling"].counts[0, [7, 8]].tolist(), [2, 2])
        first.resolve()
        self.assertEqual(e.tokens[1], [5, 7, 8])
        second.resolve()
        self.assertEqual(e.tokens[1], [5, 7, 8, 7, 8])
        oracle = History(32, "cpu")
        for i, (token, score, tops) in enumerate(e.lps[1]):
            seen, counts = oracle.of(1, e.tokens[1][:1+i], 1)
            row = process_logits(logits[i % 2], e.options[1], seen, counts)
            lp, expected = top_logprobs(row, token, 2)
            self.assertAlmostEqual(score, lp, places=6)
            self.assertEqual([t for t, _ in tops], [t for t, _ in expected])
            torch.testing.assert_close(torch.tensor([v for _, v in tops]),
                                       torch.tensor([v for _, v in expected]), rtol=2e-6, atol=2e-6)

    def test_minimum_crosses_inside_block_and_ghost_steps_publish_nothing(self):
        logits = torch.zeros(2, 32)
        logits[:, 31], logits[:, 7] = 5, 4
        e = self.engine({"logprobs": 0}, logits, minimum=3)
        p = AsyncDecode(e)
        first, second = p.launch([1], [1]), p.launch([1], [1])
        self.assertEqual(first.resolve(), [False])
        self.assertEqual(second.resolve(), [True])
        self.assertEqual(e.tokens[1], [5, 7, 7, 7, 31])
        self.assertEqual([r[0] for r in e.lps[1]], [7, 7, 7, 31])
        self.assertEqual([r[2] for r in e.lps[1]], [[], [], [], []])
        counts = p.buf["sampling"].counts.clone()
        p.launch([1], [1]).resolve()
        torch.testing.assert_close(p.buf["sampling"].counts, counts)
        self.assertEqual(len(e.lps[1]), 4)

    def test_limit_clips_logprob_packet_and_penalty_history(self):
        logits = torch.zeros(2, 32)
        logits[:, 7] = 5
        e = self.engine({"logprobs": 3, "presence_penalty": .5}, logits, limit=1)
        p = AsyncDecode(e)
        first, second = p.launch([1], [1]), p.launch([1], [1])
        first.resolve()
        second.resolve()
        self.assertEqual(e.tokens[1], [5, 7])
        self.assertEqual(len(e.lps[1]), 1)
        self.assertEqual(float(p.buf["sampling"].counts.sum()), 1.)

    def test_join_reorder_and_release_preserve_survivors_and_readback_ownership(self):
        logits = torch.zeros(2, 32)
        logits[:, 7], logits[:, 8] = 4, 3
        e = self.engine({"frequency_penalty": 2, "logprobs": 2}, logits)
        p = AsyncDecode(e)
        first = p.launch([1], [1])
        second = p.launch([2, 1], [2, 1])
        self.assertEqual(p.buf["sampling"].logprobs, [None, 2])
        self.assertEqual(p.buf["sampling"].counts[:, [7, 8]].tolist(), [[2, 0], [2, 2]])
        first.resolve()
        del e.tokens[2]
        second.resolve()
        self.assertEqual(e.tokens[1], [5, 7, 8, 7, 8])
        self.assertEqual(len(e.lps[1]), 4)
        p.launch([1], [1]).resolve()
        self.assertEqual(p.batch, (1,))
        self.assertEqual(p.buf["sampling"].logprobs, [2])

    def test_neutral_options_allocate_no_policy(self):
        logits = torch.zeros(2, 32)
        opts = {"seed": 0, "presence_penalty": 0, "frequency_penalty": 0, "repetition_penalty": 1, "logit_bias": {}}
        e = self.engine(opts, logits)
        self.assertFalse(needs_device_policy(opts))
        p = AsyncDecode(e)
        p.launch([1], [1]).resolve()
        self.assertIsNone(p.buf["sampling"])

    def test_plain_survivors_and_satisfied_minimum_return_to_the_captured_sampler(self):
        logits = torch.zeros(2, 32)
        logits[:, 7] = 4
        e = self.engine({"logprobs": 0}, logits)
        p = AsyncDecode(e)
        p.launch([1, 2], [1, 2]).resolve()
        p.launch([2], [2]).resolve()
        self.assertIsNone(p.buf["sampling"])
        logits[:, 31] = 5
        e = self.engine({}, logits, minimum=2)
        greedy = logits.argmax(-1).to(e.caches.device)
        e.sampling_graphs.greedy.run = lambda shape, fill: greedy.repeat(shape[0])
        p = AsyncDecode(e)
        self.assertEqual(p.launch([1], [1]).resolve(), [False])
        self.assertIsNotNone(p.buf["sampling"])
        self.assertEqual(p.launch([1], [1]).resolve(), [True])
        self.assertIsNone(p.buf["sampling"])
        self.assertEqual(e.tokens[1], [5, 7, 7, 31])

    def test_seeded_sampled_penalty_rows_match_synchronous_verification(self):
        from engine.base import draws
        from engine.base.sampler import block_verify, distribution
        logits = torch.arange(64, dtype=torch.float32).view(2, 32) / 30
        opts = {"seed": (1 << 80) + 3, "frequency_penalty": .3, "top_k": 12, "top_p": .9, "logprobs": 2}
        e = self.engine(opts, logits, limit=16)
        e.ends[1] = set()
        e.seeds[1], e.limits[1] = opts["seed"], (16, .7)
        e.note_ceilings = lambda *args: None
        def propose(field, slots, anchors, ctx, temps=None, uniforms=None, **kwargs):
            n, device = len(slots), slots.device
            candidates = torch.tensor([6, 7, 8, 9], device=device).view(1, 1, 4).repeat(n, 1, 1)
            q = torch.full((n, 1, 4), .25, device=device)
            chosen = (uniforms * 4).to(torch.int64).clamp_max(3)
            return candidates.gather(2, chosen.unsqueeze(2)).squeeze(2), candidates, q
        e.drafter.propose_rows = propose
        p, oracle_tokens = AsyncDecode(e), [5]
        history = History(32, "cpu")
        while len(oracle_tokens) < 17:
            key = draws.row_key(opts["seed"], 0, len(oracle_tokens)-1)
            proposed = [6 + min(3, int(draws.uniform(key, draws.DRAFT, 0) * 4))]
            seen, counts = history.of(1, oracle_tokens, 1)
            processed = [process_logits(logits[i], opts, seen, counts, proposed[:i]) for i in range(2)]
            target = torch.stack([distribution(row, .7, 12, .9) for row in processed])
            q = torch.zeros(1, 32)
            q[0, 6:10] = .25
            _, new = block_verify(target, proposed, q, draws.uniforms(key, draws.VERIFY, 1)
                                  + draws.uniforms(key, draws.FRESH, 1))
            new = new[:17-len(oracle_tokens)]
            oracle_tokens += new
            p.launch([1], [1]).resolve()
            self.assertEqual(e.tokens[1], oracle_tokens)
            self.assertEqual(len(e.lps[1]), len(oracle_tokens)-1)
            for pos, entry in enumerate(e.lps[1][-len(new):]):
                self.assertEqual(entry[0], new[pos])
                expected, _ = top_logprobs(processed[pos], entry[0], 2)
                self.assertAlmostEqual(entry[1], expected, places=5)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class CudaAsyncOptionTests(AsyncOptionTests):
    @classmethod
    def setUpClass(cls):
        torch.cuda.set_per_process_memory_fraction(.12)

    def engine(self, options, logits, **kw):
        e = super().engine(options, logits, **kw)
        e.caches.device = torch.device("cuda")
        e.caches.draft_field = lambda: torch.zeros(1, device="cuda")
        e.drafter.propose_rows = lambda field, slots, anchors, ctx, alive=None: torch.full((len(slots), 1), 7, device="cuda")
        gpu_logits = logits.cuda()
        e.decode_graphs.run_device = lambda shape, *args: (None, None, gpu_logits.repeat(shape[0], 1))
        e.sampling_graphs.greedy.run = lambda shape, fill: torch.tensor([7, 8] * shape[0], device="cuda")
        return e


if __name__ == "__main__":
    unittest.main()
