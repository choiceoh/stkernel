"""CPU checks for the OpenAI-dialect sampling options (45차 §23 A3/A4/B4): validation, penalties and bias on raw
logits, top-k/top-p distributions, seeded determinism, rejection sampling's output law, raw logprobs."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

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
        out = process_logits(logits, {}, seen, counts, decodable=3, mask=torch.tensor([True, False, True, True, True]))
        self.assertTrue(torch.isinf(out[1]) and torch.isinf(out[3]) and torch.isinf(out[4]) and out[0] == 2.0)

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
