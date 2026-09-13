"""A fast aligned prefix must neither lose tokens nor delay live decoders."""
import random
import unittest

from engine.base import scheduler as s


def contract(**kw):
    return s.Contract(chunk_align=2304, token_budget=32768, draft_slots=6,
                      max_wait_s=0, max_running=4, **kw)


class PrefillTailTests(unittest.TestCase):
    def test_long_requests_and_adopted_prefixes_keep_every_token(self):
        rng = random.Random(27)
        cases = [(32545, 0), (128559, 0), (2121, 0), (2128, 0)]
        for _ in range(500):
            end = rng.randrange(1, 134401)
            cases.append((end, rng.randrange(end)))
        for end, start in cases:
            with self.subTest(end=end, start=start):
                state = s.State()
                s.arrive(state, 1, end, 0, computed=start)
                total = 0
                while state.waiting:
                    step = s.plan(state, contract(prefill_tail_multiple=4), 0)
                    self.assertEqual(step.kind, s.PREFILL)
                    self.assertGreater(step.tokens, 0)
                    self.assertLessEqual(step.tokens, 32256)
                    if step.tokens % 4:
                        self.assertLessEqual(step.tokens, 2304)
                        self.assertEqual(start + total + step.tokens, end)
                    elif start + total + step.tokens < end:
                        self.assertEqual(step.tokens % 2304, 0)
                    total += step.tokens
                    s.advance(state, step)
                self.assertEqual(total, end - start)
                self.assertEqual(state.computed[1], end)
                self.assertEqual(state.running, [1])

    def test_opt_in_preserves_unsplit_fast_tails_and_other_profiles(self):
        for length in (2121, 2128, 4096, 8752):
            state = s.State()
            s.arrive(state, 1, length, 0)
            self.assertEqual(s.plan(state, contract(prefill_tail_multiple=4), 0).tokens, length)
        state = s.State()
        s.arrive(state, 1, 8751, 0)
        self.assertEqual(s.plan(state, contract(), 0).tokens, 8751)
        self.assertEqual(s.plan(state, contract(prefill_tail_multiple=4), 0).tokens, 6912)

    def test_decode_keeps_its_short_budget_and_next_turn(self):
        state = s.State(running=[2])
        s.arrive(state, 1, 128559, 0)
        c = contract(prefill_tail_multiple=4, decode_token_budget=2310)
        step = s.plan(state, c, 0)
        self.assertEqual(step.tokens, 2304)
        s.advance(state, step)
        step = s.plan(state, c, 0)
        self.assertEqual(step.kind, s.DECODE)
        self.assertEqual(step.seqs, (2,))

    def test_invalid_profile_alignment_refuses(self):
        for value in (True, -1, 5, 4.0):
            with self.assertRaises(ValueError):
                contract(prefill_tail_multiple=value)


if __name__ == '__main__':
    unittest.main()
