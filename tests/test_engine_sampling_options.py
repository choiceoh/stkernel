"""CPU checks for the OpenAI-dialect sampling options (45차 §23 A3/A4/B4): validation, penalties and bias on raw
logits, top-k/top-p distributions, seeded determinism, rejection sampling's output law, raw logprobs."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]  # noqa: E402

from engine.base.sampler import (distribution, draw, needs_rich_sampler, process_logits, speculative_pick,  # noqa: E402
                                 top_logprobs, validate_options)


class OptionTests(unittest.TestCase):
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
        self.assertTrue(needs_rich_sampler({"top_p": 0.9}, 0.0, drafts=True))    # the captured sampler has no nucleus branch
        self.assertTrue(needs_rich_sampler({"top_k": 5}, 0.0, drafts=False))
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
        accepted, tokens, count = block_verify_batch(target, ids, draft, torch.Generator().manual_seed(5))
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

    def test_the_adapter_drops_it_wherever_it_reassigns_a_row_s_tokens(self):
        source = (ROOT / "engine/profiles/glm53/adapter.py").read_text()
        for site in ("self.tokens[seq] = list(ids)", 'self.tokens[seq] = list(record["tokens"])'):
            after = source[source.index(site):]
            self.assertIn("self.history.forget(seq)", after[:400], site)


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
            many, _, _ = block_verify_batch(target.unsqueeze(0), torch.tensor([ids]), draft.unsqueeze(0),
                                            torch.Generator().manual_seed(trial))
            self.assertEqual(one, int(many[0]), f"trial {trial}")

