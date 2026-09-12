"""CPU checks for the OpenAI-dialect sampling options (45차 §23 A3/A4/B4): validation, penalties and bias on raw
logits, top-k/top-p distributions, seeded determinism, rejection sampling's output law, raw logprobs."""
import sys
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]  # noqa: E402

from engine.base.sampler import (distribution, draw, needs_rich_sampler, process_logits, speculative_pick,  # noqa: E402
                                 top_logprobs, validate_options)


class OptionTests(unittest.TestCase):
    def test_penalty_history_preserves_the_engines_prefix_history_protocol(self):
        from types import SimpleNamespace
        from engine.profiles.glm53.adapter import Glm53Engine
        e = Glm53Engine(None, SimpleNamespace(device=torch.device('cpu')), SimpleNamespace(spec_k=5))
        e.tokens[0], e.prompt_len[0] = [1, 2, 3], 2
        e.options[0] = {'presence_penalty': 1.0}
        self.assertEqual(e.history(0), [1, 2, 3])
        logits = e._row_logits(0, torch.ones(8), 0, [])
        self.assertEqual(logits.tolist(), [1., 1., 1., 0., 1., 1., 1., 1.])
        self.assertEqual(e.history(0), [1, 2, 3])
        e.forget(0)
        self.assertNotIn(0, e.sampling_history.rows)
        e.tokens[0], e.prompt_len[0] = [4, 5], 2
        e.options[0] = {'presence_penalty': 1.0}
        self.assertEqual(e._row_logits(0, torch.ones(8), 0, []).tolist(), [1.] * 8)
        self.assertEqual(e.history(0), [4, 5])

    def test_validate_refuses_unknown_and_out_of_range(self):
        validate_options({"top_p": 0.5, "top_k": 3, "seed": 1, "presence_penalty": 1.5, "frequency_penalty": -2,
                          "repetition_penalty": 1.2, "logit_bias": {3: -100.0}, "stop_token_ids": [7], "logprobs": 5,
                          "grammar": {"type": "json_object"}})
        for bad in ({"typo": 1}, {"top_p": 0}, {"top_k": 0}, {"presence_penalty": 2.5}, {"repetition_penalty": 0},
                    {"seed": -1}, {"logit_bias": {"3": 1}}, {"stop_token_ids": [-1]}, {"logprobs": 21}, {"grammar": {"type": "ebnf"}}):
            with self.assertRaises(ValueError, msg=bad):
                validate_options(bad)

    def test_rich_rows_are_exactly_the_ones_the_captured_sampler_cannot_serve(self):
        self.assertFalse(needs_rich_sampler({}, 0.0, drafts=True))
        # the captured sampler takes top-k and top-p as per-row arrays: a nucleus is not rich
        self.assertFalse(needs_rich_sampler({"top_p": 0.9}, 0.0, drafts=True))
        self.assertFalse(needs_rich_sampler({"top_k": 5}, 0.0, drafts=False))
        self.assertTrue(needs_rich_sampler({"top_p": 0.9, "seed": 3}, 0.0, drafts=False))   # its own generator
        self.assertTrue(needs_rich_sampler({"repetition_penalty": 1.1}, 0.0, drafts=False))
        self.assertTrue(needs_rich_sampler({"logprobs": 3}, 0.0, drafts=False))
        self.assertTrue(needs_rich_sampler({}, 0.8, drafts=True))          # rejection sampling needs the probabilities
        self.assertFalse(needs_rich_sampler({}, 0.8, drafts=False))

    def history(self, vocab, prompt, generated):
        """(seen, counts) as the adapter would hold them for a row with this history."""
        from engine.base.sampler import History
        return History(vocab, "cpu").of(0, list(prompt) + list(generated), len(prompt))

    def test_penalties_and_bias_act_on_the_named_tokens_only(self):
        logits = torch.tensor([2.0, -1.0, 0.5, 3.0, 0.0])
        seen, counts = self.history(5, [], [])
        out = process_logits(logits, {"logit_bias": {1: 5.0}}, seen, counts)
        self.assertEqual(out[1].item(), 4.0)
        seen, counts = self.history(5, [0], [1])
        out = process_logits(logits, {"repetition_penalty": 2.0}, seen, counts)
        self.assertEqual((out[0].item(), out[1].item(), out[3].item()), (1.0, -2.0, 3.0))   # positive / rp, negative * rp, untouched
        seen, counts = self.history(5, [], [2, 2, 4])
        out = process_logits(logits, {"presence_penalty": 1.0, "frequency_penalty": 0.5}, seen, counts)
        self.assertAlmostEqual(out[2].item(), 0.5 - 1.0 - 1.0)             # present once, counted twice
        self.assertAlmostEqual(out[4].item(), 0.0 - 1.0 - 0.5)
        self.assertEqual(out[3].item(), 3.0)
        seen, counts = self.history(5, [], [])
        out = process_logits(logits, {}, seen, counts, decodable=3)
        self.assertTrue(torch.isinf(out[3]) and torch.isinf(out[4]) and out[0] == 2.0)

    def test_this_step_drafts_count_without_being_written_into_the_history(self):
        logits = torch.zeros(5)
        seen, counts = self.history(5, [], [])
        plain = process_logits(logits, {"presence_penalty": 1.0}, seen, counts)
        drafted = process_logits(logits, {"presence_penalty": 1.0}, seen, counts, extra=[3])
        self.assertEqual(plain[3].item(), 0.0)
        self.assertEqual(drafted[3].item(), -1.0)
        # the correction must not stay behind: the next position sees the same history
        again = process_logits(logits, {"presence_penalty": 1.0}, seen, counts)
        self.assertEqual(again[3].item(), 0.0)

    def test_a_draft_repeats_for_the_repetition_penalty_too(self):
        logits = torch.tensor([2.0, 2.0])
        seen, counts = self.history(2, [], [])
        out = process_logits(logits, {"repetition_penalty": 2.0}, seen, counts, extra=[1])
        self.assertEqual((out[0].item(), out[1].item()), (2.0, 1.0))

    def test_the_history_is_grown_not_rebuilt_when_a_token_is_appended(self):
        from engine.base.sampler import History
        h = History(8, "cpu")
        tokens, walked = [1, 2, 3], []
        real = h._build

        def counting(seq, ids, prompt_len):
            walked.append(len(ids))
            return real(seq, ids, prompt_len)

        h._build = counting
        h.of(0, tokens, 3)
        for t in (4, 5, 6):                                  # one decode step each
            tokens.append(t)
            seen, counts = h.of(0, tokens, 3)
        self.assertEqual(walked, [3], "only the prompt was ever walked")
        self.assertTrue(bool(seen[6]) and bool(seen[1]))
        self.assertEqual([float(counts[i]) for i in (1, 4, 5, 6)], [0.0, 1.0, 1.0, 1.0])

    def test_a_prompt_that_moved_or_a_list_that_shrank_is_rebuilt(self):
        from engine.base.sampler import History
        h = History(8, "cpu")
        h.of(0, [1, 2, 3], 2)
        seen, counts = h.of(0, [1, 2, 3], 3)                  # the turn continued: output became prompt
        self.assertEqual(float(counts[3]), 0.0)
        h.of(1, [1, 2, 3], 2)
        seen, counts = h.of(1, [1, 2], 2)                     # a rejected draft came back off the end
        self.assertFalse(bool(seen[3]))

    def test_forgetting_a_row_drops_its_history(self):
        from engine.base.sampler import History
        h = History(8, "cpu")
        h.of(0, [1], 1)
        self.assertIn(0, h.rows)
        h.forget(0)
        self.assertNotIn(0, h.rows)

    def test_the_model_dtype_goes_in_and_float32_comes_out_unchanged(self):
        # the gather hands over the model's dtype now; the one copy here is the upcast
        raw = torch.tensor([2.0, -1.0, 0.5, 3.0])
        seen, counts = self.history(4, [0], [1])
        wide = process_logits(raw, {"repetition_penalty": 2.0}, seen, counts)
        narrow = process_logits(raw.to(torch.bfloat16), {"repetition_penalty": 2.0}, seen, counts)
        self.assertEqual(wide.dtype, torch.float32)
        self.assertEqual(narrow.dtype, torch.float32)
        self.assertEqual(wide.tolist(), narrow.tolist())
        self.assertEqual(raw.tolist(), [2.0, -1.0, 0.5, 3.0], "the caller's logits are untouched")

    def test_the_gather_does_not_upcast_what_every_row_copies_anyway(self):
        source = (ROOT / "engine/profiles/glm53/adapter.py").read_text()
        body = source[source.index("    def _gather(self"):]
        body = body[:body.index("\n    def ", 10)]
        self.assertIn("all_gather(local, dim=-1)", body)
        self.assertNotIn(".float()", body)

    def test_a_handful_of_forbidden_ids_needs_no_vocabulary_of_true(self):
        logits = torch.zeros(6)
        seen, counts = self.history(6, [], [])
        out = process_logits(logits, {}, seen, counts, forbid=torch.tensor([1, 4]))
        self.assertTrue(torch.isinf(out[1]) and torch.isinf(out[4]))
        self.assertEqual([float(out[i]) for i in (0, 2, 3, 5)], [0.0] * 4)

    def test_a_buffer_receives_the_row_instead_of_a_fresh_vocabulary(self):
        """The grammar mask crosses a whole row in one launch, which needs the row's positions in consecutive
        rows of one tensor -- so a position can be asked to land in one (base/grammar, 45차 §28)."""
        raw = torch.tensor([1.0, -2.0, 3.0, 4.0])
        seen, counts = self.history(4, [1], [1])
        buf = torch.empty(2, 4)
        got = process_logits(raw, {"repetition_penalty": 2.0}, seen, counts, forbid=torch.tensor([0]), out=buf[1])
        self.assertEqual(got.data_ptr(), buf[1].data_ptr(), "the answer is in the buffer, not beside it")
        fresh = process_logits(raw, {"repetition_penalty": 2.0}, seen, counts, forbid=torch.tensor([0]))
        self.assertTrue(torch.equal(torch.nan_to_num(buf[1], neginf=-1e9), torch.nan_to_num(fresh, neginf=-1e9)))
        self.assertTrue(torch.isinf(buf[1][0]))

    def test_picking_every_row_at_once_draws_what_picking_them_one_by_one_would(self):
        from engine.base.sampler import draw, pick_each
        rows = [torch.softmax(torch.randn(16, generator=torch.Generator().manual_seed(i)), -1) for i in range(4)]
        one_at_a_time = torch.Generator().manual_seed(9)
        together = torch.Generator().manual_seed(9)
        self.assertEqual(pick_each(rows, 1.0, together), [draw(r, one_at_a_time) for r in rows])

    def test_a_zero_temperature_row_set_picks_every_argmax(self):
        from engine.base.sampler import pick_each
        rows = [torch.tensor([0.1, 0.7, 0.2]), torch.tensor([0.6, 0.1, 0.3])]
        self.assertEqual(pick_each(rows, 0.0, None), [1, 0])

    def test_distribution_top_k_top_p_and_greedy(self):
        logits = torch.tensor([3.0, 2.0, 1.0, 0.0, -1.0])
        self.assertEqual(distribution(logits, 0.0, None, None).tolist(), [1, 0, 0, 0, 0])
        p = distribution(logits, 1.0, 2, None)
        self.assertEqual((p[2:] == 0).all().item(), True)
        self.assertAlmostEqual(p.sum().item(), 1.0, places=5)
        p = distribution(logits, 1.0, None, 0.5)
        self.assertGreater(p[0].item(), 0.99)                               # the first token already carries > 0.5 mass
        g1, g2 = torch.Generator().manual_seed(9), torch.Generator().manual_seed(9)
        probs = distribution(logits, 1.0, None, None)
        self.assertEqual([draw(probs, g1) for _ in range(20)], [draw(probs, g2) for _ in range(20)])

    def test_rejection_sampling_reproduces_the_target_law(self):
        torch.manual_seed(0)
        V, K = 6, 2
        target = torch.softmax(torch.randn(K + 1, V) * 2, dim=-1)
        draft = torch.softmax(torch.randn(K, V) * 2, dim=-1)
        g = torch.Generator().manual_seed(1)
        first = torch.zeros(V)
        n = 20000
        for _ in range(n):
            drafts = [draw(draft[i], g) for i in range(K)]
            accepted, out = speculative_pick(target, drafts, draft, g)
            self.assertEqual(len(out), accepted + 1)
            first[out[0]] += 1
        self.assertLess((first / n - target[0]).abs().max().item(), 0.02)   # the first committed token follows the target

    def test_raw_logprobs_and_top_candidates(self):
        logits = torch.tensor([1.0, 3.0, 2.0, 0.0])
        lp, top = top_logprobs(logits, 2, 2)
        self.assertAlmostEqual(lp, torch.log_softmax(logits, -1)[2].item(), places=6)
        self.assertEqual([i for i, _ in top], [1, 2])


class ReasoningBudgetTests(unittest.TestCase):
    """A thinking block that eats the whole limit leaves no answer (45차 §46)."""

    END = 6

    def engine(self, options):
        from types import SimpleNamespace
        from engine.profiles.glm53.adapter import Glm53Engine
        e = Glm53Engine(None, SimpleNamespace(device=torch.device("cpu")), SimpleNamespace(spec_k=5))
        e._bind_options(0, options)
        return e

    def allowed(self, e, generated, drafts=()):
        e.tokens[0], e.prompt_len[0] = [1, 2] + list(generated), 2
        out = e._row_logits(0, torch.ones(8), 0, list(drafts))
        return [i for i, v in enumerate(out.tolist()) if v != float("-inf")]

    def test_under_the_budget_nothing_is_forced(self):
        e = self.engine({"reasoning_budget": 4, "reasoning_end": self.END})
        self.assertEqual(self.allowed(e, [5, 5, 5]), list(range(8)))

    def test_the_budget_spent_leaves_only_the_way_out(self):
        e = self.engine({"reasoning_budget": 4, "reasoning_end": self.END})
        self.assertEqual(self.allowed(e, [5, 5, 5, 5]), [self.END])
        self.assertEqual(self.allowed(e, [5, 5, 5, 5, 5]), [self.END], "and it stays forced")

    def test_this_step_s_drafts_count_towards_it(self):
        e = self.engine({"reasoning_budget": 4, "reasoning_end": self.END})
        self.assertEqual(self.allowed(e, [5, 5, 5], drafts=[5, 5]), [self.END])

    def test_a_block_that_closed_itself_is_never_asked_again(self):
        e = self.engine({"reasoning_budget": 4, "reasoning_end": self.END})
        self.assertEqual(self.allowed(e, [5, 5, self.END, 5]), list(range(8)))
        self.assertFalse(e.thinking[0])
        self.assertEqual(self.allowed(e, [5, 5, self.END, 5, 5, 5, 5]), list(range(8)))

    def test_no_budget_is_no_bound(self):
        e = self.engine({})
        self.assertEqual(self.allowed(e, [5] * 6), list(range(8)))
        self.assertFalse(e.thinking[0])

    def test_min_tokens_outranks_the_budget(self):
        """`forbid` is a promise the caller made; a budget is the engine keeping room. If the two
        ever name the same token, the promise wins and nothing is forced."""
        from engine.base.sampler import process_logits
        seen, counts = torch.zeros(8, dtype=torch.bool), torch.zeros(8)
        out = process_logits(torch.ones(8), {}, seen, counts, forbid=torch.tensor([3]), force=3)
        self.assertEqual(out.tolist(), [1.0, 1.0, 1.0, float("-inf")] + [1.0] * 4)

    def test_the_option_pair_travels_together(self):
        from engine.base.sampler import validate_options
        validate_options({"reasoning_budget": 5, "reasoning_end": 3})
        for bad in ({"reasoning_budget": 5}, {"reasoning_end": 3},
                    {"reasoning_budget": -1, "reasoning_end": 3}, {"reasoning_budget": 5, "reasoning_end": -1}):
            with self.assertRaises(ValueError, msg=bad):
                validate_options(bad)

    def test_the_default_leaves_the_answer_a_share_of_the_limit(self):
        from engine.base.serve import ANSWER_FLOOR, RequestError, reasoning_budget
        for limit in (256, 1192, 2048):
            with self.subTest(limit=limit):
                budget = reasoning_budget({}, limit)
                self.assertGreaterEqual(limit - budget, min(ANSWER_FLOOR, limit - 1))
                self.assertGreater(budget, 0)
        self.assertIsNone(reasoning_budget({"reasoning_budget": -1}, 1000), "-1 asks for no bound")
        self.assertEqual(reasoning_budget({"reasoning_budget": 0}, 1000), 0, "0 asks for no thinking")
        with self.assertRaises(RequestError):
            reasoning_budget({"reasoning_budget": -2}, 1000)


if __name__ == "__main__":
    unittest.main()


class ValidateTests(unittest.TestCase):
    """adapter.validate runs on every rank inside the step loop: C-speed, and still strict."""

    def test_validate_rejects_what_it_must_and_accepts_the_rest_without_a_python_loop(self):
        from types import SimpleNamespace
        from engine.profiles.glm53.adapter import Glm53Engine
        e = SimpleNamespace(F=SimpleNamespace(vocab=100))
        ok = lambda ids: Glm53Engine.validate(e, ids, 4, 0.0)          # noqa: E731
        ok(list(range(100)))
        for bad in ([], [1, 100], [-1], [1.5, 2], [1, "2"], [None]):
            with self.assertRaises((ValueError, TypeError)):
                ok(bad)


def _as_candidates(dense: torch.Tensor):
    """[n, K, V] -> the candidates it puts mass on and that mass, which is all the verifier ever read of it."""
    n, K, V = dense.shape
    c = int((dense > 0).sum(-1).max())
    q, cand = dense.topk(c, dim=-1)
    return cand, q


def _verify_densely(target_probs, drafts, draft_probs, generator):
    """block_verify_batch as it was written before the draft became its candidates: the reference the sparse
    form is judged against, kept here so the equality is a test and not a comment."""
    from engine.base.constants import iota
    from engine.base.sampler import _inverse_cdf
    n, k1, _ = target_probs.shape
    K = k1 - 1
    device = target_probs.device
    rows = iota(n, device)
    on_draft = target_probs[:, :K].gather(2, drafts.unsqueeze(2)).squeeze(2)
    by_draft = draft_probs.gather(2, drafts.unsqueeze(2)).squeeze(2)
    step = torch.where(by_draft > 0, on_draft / by_draft.clamp_min(1e-30), torch.zeros_like(on_draft))
    carried = torch.empty_like(step)
    running = torch.ones(n, device=device, dtype=step.dtype)
    for i in range(K):
        running = (running * step[:, i]).clamp_max(1.0)
        carried[:, i] = running
    thresholds = carried.clone()
    if K > 1:
        ahead = carried[:, : K - 1].unsqueeze(-1)
        mass = (ahead * target_probs[:, 1:K] - draft_probs[:, 1:K]).clamp_min(0).sum(-1)
        denominator = mass + 1.0 - carried[:, : K - 1]
        thresholds[:, : K - 1] = torch.where(denominator > 0, mass / denominator.clamp_min(1e-30),
                                             torch.ones_like(mass))
    u = torch.rand(n, K, generator=generator, device=device)
    reach = iota(K, device).add(1).expand(n, K)
    accepted = torch.where(u <= thresholds, reach, torch.zeros_like(reach)).max(1).values
    at = accepted.clamp_max(K)
    before = torch.where(accepted > 0, carried.gather(1, (accepted - 1).clamp_min(0).unsqueeze(1)).squeeze(1),
                         torch.ones(n, device=device, dtype=carried.dtype))
    row_p = target_probs[rows, at]
    row_q = torch.where((at < K).unsqueeze(1), draft_probs[rows, at.clamp_max(K - 1)], torch.zeros_like(row_p))
    rest = (before.unsqueeze(1) * row_p - row_q).clamp_min(0)
    total = rest.sum(1, keepdim=True)
    rest = torch.where(total > 0, rest / total.clamp_min(1e-30),
                       row_p / row_p.sum(1, keepdim=True).clamp_min(1e-30))
    fresh = _inverse_cdf(rest, torch.rand(n, generator=generator, device=device))
    tokens = torch.cat([drafts, torch.zeros(n, 1, dtype=drafts.dtype, device=device)], 1)
    tokens.scatter_(1, at.unsqueeze(1), fresh.unsqueeze(1))
    return accepted, tokens, accepted + 1


class BlockVerifyKernelTests(unittest.TestCase):
    """The fused kernel must decide exactly what the torch reference decides.

    Compared on ONE device from ONE generator: a CPU generator and a CUDA generator do not agree at the same
    seed, and comparing across them reads as a kernel bug when it is two different streams of uniforms."""

    @unittest.skipUnless(torch.cuda.is_available(), "the kernel path needs a device")
    def test_the_kernel_accepts_and_picks_what_the_reference_does(self):
        from engine.base.sampler import _block_verify_by_torch, block_verify_batch
        for trial in range(12):
            torch.manual_seed(trial)
            n, K, V, C = (1 if trial % 3 else 4), 5, 2003, 16
            target = torch.softmax(torch.randn(n, K + 1, V), -1).cuda()
            cand = torch.stack([torch.stack([torch.randperm(V)[:C] for _ in range(K)]) for _ in range(n)]).cuda()
            qp = torch.softmax(torch.randn(n, K, C), -1).cuda()
            pick = torch.randint(0, C, (n, K))
            drafts = cand.cpu().gather(2, pick.unsqueeze(2)).squeeze(2).cuda()
            seed = lambda: torch.Generator(device="cuda").manual_seed(trial + 100)   # noqa: E731
            want = _block_verify_by_torch(target, drafts, cand, qp, seed())
            got = block_verify_batch(target, drafts, cand, qp, seed())
            for i, name in ((0, "accepted"), (1, "tokens"), (2, "count")):
                self.assertTrue(torch.equal(want[i], got[i]), f"{name} differ at trial {trial}")

    def test_the_reference_is_reachable_on_any_device(self):
        """It is the thing the kernel is judged against, so it must not be behind the `is_cuda` branch that
        chooses the kernel -- otherwise there is no way to run both on one device and compare."""
        from engine.base.sampler import _block_verify_by_torch
        n, K, V, C = 2, 3, 41, 5
        torch.manual_seed(4)
        target = torch.softmax(torch.randn(n, K + 1, V), -1)
        cand = torch.stack([torch.stack([torch.randperm(V)[:C] for _ in range(K)]) for _ in range(n)])
        qp = torch.softmax(torch.randn(n, K, C), -1)
        accepted, tokens, count = _block_verify_by_torch(target, cand[:, :, 0].contiguous(), cand, qp,
                                                         torch.Generator().manual_seed(1))
        self.assertEqual(tuple(tokens.shape), (n, K + 1))
        self.assertTrue(torch.equal(count, accepted + 1))


class SparseDraftDistributionTests(unittest.TestCase):
    """The draft puts mass on `sel_top_k` candidates a position and zero everywhere else, so carrying it as
    [n, K, vocab] was allocating and zeroing 12.4 MiB every decode step to hold 320 numbers (V=154,880, K=5,
    C=16, n=4) and then reading it twice. These say the shorter form is not an approximation of the longer."""

    def draft(self, n, K, V, C, seed):
        torch.manual_seed(seed)
        dense = torch.zeros(n, K, V)
        for r in range(n):
            for s in range(K):
                where = torch.randperm(V)[:C]
                dense[r, s, where] = torch.softmax(torch.randn(C), -1)
        return dense

    def test_the_candidates_verify_exactly_as_the_vocabulary_wide_row_did(self):
        from engine.base.sampler import block_verify_batch
        n, K, V, C = 4, 5, 61, 7
        for trial in range(30):
            dense = self.draft(n, K, V, C, trial)
            target = torch.softmax(torch.randn(n, K + 1, V), -1)
            ids = torch.stack([torch.multinomial(dense[r], 1).squeeze(1) for r in range(n)])
            cand, q = _as_candidates(dense)
            want = _verify_densely(target, ids, dense, torch.Generator().manual_seed(trial))
            got = block_verify_batch(target, ids, cand, q, torch.Generator().manual_seed(trial))
            self.assertTrue(torch.equal(want[0], got[0]), f"accepted differ at trial {trial}")
            self.assertTrue(torch.equal(want[1], got[1]), f"tokens differ at trial {trial}")
            self.assertTrue(torch.equal(want[2], got[2]))

    def test_it_holds_when_the_draft_is_a_point_mass(self):
        """A greedy row joining a sampled batch is one candidate carrying everything (pipeline._merge)."""
        from engine.base.sampler import block_verify_batch
        n, K, V, C = 3, 4, 23, 5
        torch.manual_seed(1)
        target = torch.softmax(torch.randn(n, K + 1, V), -1)
        ids = torch.randint(0, V, (n, K))
        dense = torch.zeros(n, K, V).scatter_(2, ids.unsqueeze(2), 1.0)
        cand = ids.unsqueeze(2).expand(n, K, C).contiguous()
        q = torch.zeros(n, K, C); q[..., 0] = 1.0
        want = _verify_densely(target, ids, dense, torch.Generator().manual_seed(4))
        got = block_verify_batch(target, ids, cand, q, torch.Generator().manual_seed(4))
        self.assertTrue(torch.equal(want[0], got[0]))
        self.assertTrue(torch.equal(want[1], got[1]))

    def test_both_ceilings_are_the_same_two_numbers(self):
        from engine.base.sampler import draft_ceilings, draft_ceilings_over
        n, K, V, C = 3, 4, 41, 6
        dense = self.draft(n, K, V, C, 11)
        target = torch.softmax(torch.randn(n, K + 1, V), -1)
        cand, q = _as_candidates(dense)
        a, b = draft_ceilings(target, dense)
        c, d = draft_ceilings_over(target, cand, q)
        self.assertAlmostEqual(a, c, places=6)
        self.assertAlmostEqual(b, d, places=6)


class BlockVerificationTests(unittest.TestCase):
    """Sun et al. 2024: a longer accepted prefix for the same output distribution.

    The claim is distributional, so the gate is distributional. The control is the
    token-level rule (`speculative_pick`), which the engine keeps as a reference.
    """

    def draws(self, seed, K, V, spread=1.3):
        torch.manual_seed(seed)
        return (torch.softmax(torch.randn(K + 1, V) * spread, -1),
                torch.softmax(torch.randn(K, V) * spread, -1))

    def emitted(self, pick, target, draft, rounds, seed):
        from engine.base.sampler import draw
        gen = torch.Generator().manual_seed(seed)
        K, V = draft.shape
        first = torch.zeros(V)
        accepted = 0
        for _ in range(rounds):
            drafts = [draw(draft[i], gen) for i in range(K)]
            got, new = pick(target, drafts, draft, gen)
            accepted += got
            first[new[0]] += 1
        return first / first.sum(), accepted / rounds

    def test_the_first_emitted_token_is_the_target_s_own(self):
        from engine.base.sampler import block_verify, speculative_pick
        target, draft = self.draws(72, 3, 5)
        blocked, _ = self.emitted(block_verify, target, draft, 20000, 3)
        token, _ = self.emitted(speculative_pick, target, draft, 20000, 3)
        # the control says how close 20,000 rounds gets; the block scheme must not be worse
        allowed = max(0.02, float((token - target[0]).abs().max()) * 1.5)
        self.assertLess(float((blocked - target[0]).abs().max()), allowed)

    def test_it_accepts_more_than_the_token_level_rule(self):
        from engine.base.sampler import block_verify, speculative_pick
        target, draft = self.draws(72, 3, 5)
        _, blocked = self.emitted(block_verify, target, draft, 8000, 11)
        _, token = self.emitted(speculative_pick, target, draft, 8000, 11)
        self.assertGreater(blocked, token)

    def test_one_draft_is_the_token_level_threshold(self):
        from engine.base.sampler import block_verify
        target = torch.tensor([[0.6, 0.4], [0.5, 0.5]])
        draft = torch.tensor([[0.2, 0.8]])
        # with K = 1 the threshold is min(p/q, 1) = min(0.6/0.2, 1) = 1: always accepted
        for seed in range(8):
            got, new = block_verify(target, [0], draft, torch.Generator().manual_seed(seed))
            self.assertEqual(got, 1)
            self.assertEqual(new[0], 0)

    def test_the_batch_accepts_what_the_row_by_row_rule_accepts(self):
        from engine.base.sampler import block_verify_batch
        torch.manual_seed(3)
        n, K, V = 4, 3, 7
        target = torch.softmax(torch.randn(n, K + 1, V), -1)
        draft = torch.softmax(torch.randn(n, K, V), -1)
        ids = torch.stack([torch.multinomial(draft[r], 1).squeeze(1) for r in range(n)])
        cand, q = _as_candidates(draft)
        accepted, tokens, count = block_verify_batch(target, ids, cand, q, torch.Generator().manual_seed(5))
        uniform = torch.rand(n, K, generator=torch.Generator().manual_seed(5))
        want = []
        for r in range(n):
            carried, running = [], 1.0
            for i in range(K):
                q = float(draft[r, i, ids[r, i]]); p = float(target[r, i, ids[r, i]])
                running = min(running * p / q, 1.0) if q > 0 else 0.0
                carried.append(running)
            thresholds = list(carried)
            for i in range(K - 1):
                mass = float((carried[i] * target[r, i + 1] - draft[r, i + 1]).clamp_min(0).sum())
                denominator = mass + 1.0 - carried[i]
                thresholds[i] = mass / denominator if denominator > 0 else 1.0
            want.append(max([i + 1 for i in range(K) if float(uniform[r, i]) <= thresholds[i]], default=0))
        self.assertEqual(accepted.tolist(), want)
        self.assertEqual(count.tolist(), [a + 1 for a in want])
        for r, a in enumerate(want):
            self.assertEqual(tokens[r, :a].tolist(), ids[r, :a].tolist(), "accepted drafts are committed as they were")


class DraftCeilingTests(unittest.TestCase):
    """Acceptance has three ceilings; the split is what makes "raise it" answerable."""

    def test_the_reachable_mass_is_the_overlap_and_the_covered_mass_is_the_support(self):
        from engine.base.sampler import draft_ceilings
        target = torch.tensor([[0.5, 0.3, 0.2], [0.1, 0.8, 0.1]])
        draft = torch.tensor([[0.4, 0.6, 0.0]])                       # one position, two candidates
        reachable, covered = draft_ceilings(target, draft)
        self.assertAlmostEqual(reachable, 0.4 + 0.3, places=6)        # min(.5,.4) + min(.3,.6) + min(.2,0)
        self.assertAlmostEqual(covered, 0.5 + 0.3, places=6)          # the target mass on the two candidates

    def test_a_draft_that_covers_nothing_reaches_nothing(self):
        from engine.base.sampler import draft_ceilings
        target = torch.tensor([[0.0, 0.0, 1.0], [0.5, 0.5, 0.0]])
        draft = torch.tensor([[0.5, 0.5, 0.0]])
        reachable, covered = draft_ceilings(target, draft)
        self.assertAlmostEqual(reachable, 0.0, places=6)
        self.assertAlmostEqual(covered, 0.0, places=6)

    def test_the_ceilings_bound_the_acceptance_they_explain(self):
        from engine.base.sampler import block_verify, draft_ceilings, draw
        torch.manual_seed(5)
        K, V, rounds = 3, 6, 4000
        target = torch.softmax(torch.randn(K + 1, V), -1)
        draft = torch.softmax(torch.randn(K, V), -1)
        gen = torch.Generator().manual_seed(7)
        accepted = 0
        for _ in range(rounds):
            ids = [draw(draft[i], gen) for i in range(K)]
            got, _ = block_verify(target, ids, draft, gen)
            accepted += got
        reachable, covered = draft_ceilings(target, draft)
        self.assertLessEqual(reachable, covered + 1e-6, "what a rule can accept sits under what the candidates cover")
        self.assertLessEqual(accepted / rounds, reachable + 0.05, "and acceptance sits under both")


class HistoryLifetimeTests(unittest.TestCase):
    """A kept history is only safe while the tokens it was built from are the row's own."""

    def test_a_row_reused_with_the_same_shape_does_not_inherit_the_old_counts(self):
        from engine.base.sampler import History
        h = History(8, "cpu")
        first = [1, 1, 2, 3]
        h.of(0, first, 2)
        self.assertEqual(float(h.of(0, first, 2)[1][1]), 0.0)     # token 1 is prompt here, not output
        # the row is handed a different conversation of the same length and prompt length
        h.forget(0)
        second = [4, 5, 1, 1]
        seen, counts = h.of(0, second, 2)
        self.assertEqual(float(counts[1]), 2.0)
        self.assertFalse(bool(seen[2]), "nothing of the old conversation survives")

    def test_the_adapter_drops_it_wherever_a_row_stops_owning_its_tokens(self):
        """The property, not a list of lines: every place a row is handed a different conversation, and the
        place a row leaves, drops the history first. The sites are found rather than enumerated, so a fourth
        one does not depend on anyone remembering this test."""
        import re
        source = (ROOT / "engine/profiles/glm53/adapter.py").read_text()
        sites = [m.start() for m in re.finditer(r"^ +self\.tokens\[seq\] = ", source, re.M)]
        self.assertGreaterEqual(len(sites), 2, "the reassignment sites moved: this test cannot see them")
        for at in sites + [source.index("rows.pop(seq, None)")]:
            self.assertIn("self._forget_history(seq)", source[at: at + 400], source[at: at + 60])

    def test_forgetting_reaches_the_history_and_tolerates_one_that_was_never_built(self):
        """What a source pin cannot see. A helper that is only a name passes the test above; and the history
        is lazy, so the callers that used to reach `self.history` directly died on the first request of a boot
        that had not built one yet (which is what the helper was introduced for)."""
        from engine.base.sampler import History
        from engine.profiles.glm53.adapter import Glm53Engine
        source = (ROOT / "engine/profiles/glm53/adapter.py").read_text()
        body = source[source.index("    def _forget_history"):]
        self.assertIn("self.sampling_history.forget(seq)", body[: body.index("\n    def ", 1)], "the helper must reach the History")
        engine = Glm53Engine.__new__(Glm53Engine)            # the method, none of the boot
        engine.sampling_history = None
        engine._forget_history(0)                            # a boot whose first request has not asked for penalties
        engine.sampling_history = History(8, "cpu")
        engine.sampling_history.of(0, [1, 1, 2, 3], 2)
        self.assertIn(0, engine.sampling_history.rows)
        engine._forget_history(0)
        self.assertNotIn(0, engine.sampling_history.rows, "the helper has to reach the history, not merely exist")


class DecodeChainGateTests(unittest.TestCase):
    """Why a decode step does or does not run ahead on the device, and whether anyone can find out.

    A step that cannot run ahead makes the runner empty every step in flight before it, so this gate decides how
    much of the time the engine is in the chain at all -- and that fraction is what any deeper fusion inside the
    chain would be multiplied by. It went uncounted until now."""

    def gate(self, **state):
        from engine.profiles.glm53.adapter import Glm53Engine
        e = Glm53Engine.__new__(Glm53Engine)                 # the gate, none of the boot
        e.chain_exits = {}
        e.options, e.matchers, e.gens, e.lps, e.min_new = {}, set(), {}, {}, {}
        e.tokens, e.prompt_len = {0: [1, 2, 3]}, {0: 2}
        e.pipeline = types.SimpleNamespace(ready_for=lambda seqs, slots=None: True)
        e.decode_graphs = object()
        e.drafter = types.SimpleNamespace(k=5)
        for name, value in state.items():
            setattr(e, name, value)
        return e

    def test_the_counters_exist_from_the_boot_and_not_from_this_fixture(self):
        """`gate()` sets them, so dropping the real initialiser kills no test above -- and a real boot would then
        die with AttributeError on the first decode step the chain refused. Found by tools/mutate.py."""
        source = (ROOT / "engine/profiles/glm53/adapter.py").read_text()
        init = source[source.index("    def __init__"):]
        init = init[: init.index("\n    def ", 1)]
        self.assertIn("self.chain_exits = {}", init)

    def test_a_plain_row_runs_ahead_and_is_counted(self):
        e = self.gate()
        self.assertTrue(e.async_ready([0]))
        self.assertEqual(e.chain_exits, {})

    def test_every_blocker_names_itself(self):
        for option in ("seed", "logprobs", "logit_bias", "min_p", "presence_penalty",
                       "frequency_penalty", "repetition_penalty", "grammar"):
            e = self.gate(options={0: {option: 1}})
            self.assertFalse(e.async_ready([0]), option)
            self.assertEqual(e.chain_exits, {option: 1}, option)

    def test_state_the_options_do_not_carry_names_itself_too(self):
        """A grammar, a seeded generator and a logprob request live in their own maps by then, not in `options`."""
        for state, reason in (({"matchers": {0: 1}}, "grammar"), ({"gens": {0: 1}}, "seed"), ({"lps": {0: 1}}, "logprobs")):
            e = self.gate(**state)
            self.assertFalse(e.async_ready([0]))
            self.assertEqual(e.chain_exits, {reason: 1})

    def test_min_tokens_is_named_apart_from_the_rest_because_it_passes(self):
        """min_tokens blocks only until the row has produced enough; the others never stop blocking. An operator
        reading one number cannot act on it, and reading the two apart tells them which."""
        e = self.gate(min_new={0: 9})
        self.assertFalse(e.async_ready([0]))
        self.assertEqual(e.chain_exits, {"min_tokens": 1})
        e.min_new[0] = 1                                     # one generated token is already enough
        self.assertTrue(e.async_ready([0]))

    def test_one_row_takes_the_whole_batch_off_the_chain(self):
        """The batch runs ahead together or not at all. At max_seqs 4 that is the cost of a single logprobs request,
        and the counter has to show it as such rather than as one row's business."""
        e = self.gate(options={0: {}, 1: {}, 2: {}, 3: {"logprobs": True}},
                      tokens={s: [1, 2, 3] for s in range(4)}, prompt_len={s: 2 for s in range(4)})
        self.assertFalse(e.async_ready([0, 1, 2, 3]))
        self.assertEqual(e.chain_exits, {"logprobs": 1})

    def test_rows_churning_mid_flight_is_its_own_reason(self):
        """PR #671 made most of these go away (independent new rows may now join without draining survivors), so
        telling them apart from a row's own options is what says whether any are left."""
        e = self.gate(pipeline=types.SimpleNamespace(ready_for=lambda seqs, slots=None: False))
        self.assertFalse(e.async_ready([0, 7]))
        self.assertEqual(e.chain_exits, {"rows_churned": 1})

    def test_no_drafter_is_not_a_row_s_fault(self):
        e = self.gate(drafter=types.SimpleNamespace(k=0))
        self.assertFalse(e.async_ready([0]))
        self.assertEqual(e.chain_exits, {"no_pipeline": 1})


class VerificationPathTests(unittest.TestCase):
    """A served row and a row with penalties must be verified by the same rule."""

    def test_the_row_path_and_the_batch_path_accept_the_same_prefix(self):
        from engine.base.sampler import block_verify, block_verify_batch
        torch.manual_seed(11)
        K, V = 4, 9
        for trial in range(25):
            target = torch.softmax(torch.randn(K + 1, V), -1)
            draft = torch.softmax(torch.randn(K, V), -1)
            ids = [int(torch.multinomial(draft[i], 1)) for i in range(K)]
            one, _ = block_verify(target, ids, draft, torch.Generator().manual_seed(trial))
            cand, q = _as_candidates(draft.unsqueeze(0))
            many, _, _ = block_verify_batch(target.unsqueeze(0), torch.tensor([ids]), cand, q,
                                            torch.Generator().manual_seed(trial))
            self.assertEqual(one, int(many[0]), f"trial {trial}")
