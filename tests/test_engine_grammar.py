"""Structured output (45차 §23 A2): the grammar masks over a toy tokenizer, the draft walk and its rollback.
Runs where xgrammar and transformers are importable (the ST image); skipped elsewhere."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class GrammarTests(unittest.TestCase):
    def setUp(self):
        try:
            import torch  # noqa: F401
            import xgrammar  # noqa: F401
            from tokenizers import Tokenizer, models, pre_tokenizers
            from transformers import PreTrainedTokenizerFast
        except ImportError as exc:
            self.skipTest(f"grammar stack unavailable here: {exc}")
        self.vocab = ["{", "}", "a", "1", ":", '"', ",", "<eos>"]
        tok = Tokenizer(models.WordLevel({t: i for i, t in enumerate(self.vocab)}, unk_token="<eos>"))
        tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
        self.hf = PreTrainedTokenizerFast(tokenizer_object=tok, eos_token="<eos>")
        from engine.base.grammar import Grammars
        self.grammars = Grammars(self.hf, len(self.vocab))

    def test_json_object_masks_follow_the_grammar_and_drafts_roll_back(self):
        import torch
        m = self.grammars.matcher({"type": "json_object"}, max_rollback=4)
        open_brace, close_brace, a, one, colon, quote = (self.vocab.index(x) for x in ("{", "}", "a", "1", ":", '"'))
        first = m.masks([], "cpu")[0]
        self.assertTrue(first[open_brace].item())
        self.assertFalse(first[close_brace].item() or first[a].item())     # an object must open first
        # drafts {, ": the second mask assumes the first draft was accepted; the matcher itself has not moved
        masks = m.masks([open_brace, quote], "cpu")
        self.assertEqual(len(masks), 3)
        self.assertTrue(masks[1][close_brace].item() and masks[1][quote].item())
        self.assertFalse(masks[1][colon].item())
        self.assertTrue(m.masks([], "cpu")[0][open_brace].item())       # rolled back: still at the start
        m.advance([open_brace, close_brace])
        done = m.masks([], "cpu")[0]                                     # {} is complete: only the stop token may follow
        self.assertTrue(done[self.vocab.index("<eos>")].item())
        self.assertFalse(done[open_brace].item() or done[a].item())

    def test_json_schema_is_enforced(self):
        m = self.grammars.matcher({"type": "json_schema", "schema": '{"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]}'},
                                  max_rollback=2)
        mask = m.masks([], "cpu")[0]
        self.assertTrue(mask[self.vocab.index("{")].item())
        m.advance([self.vocab.index("{")])
        mask = m.masks([], "cpu")[0]
        self.assertTrue(mask[self.vocab.index('"')].item())
        self.assertFalse(mask[self.vocab.index("}")].item())              # "a" is required: the object cannot close yet




class _Xgr:
    """Enough of xgrammar to drive the Matcher's buffers where the real one is absent."""

    def __init__(self, words, allow):
        self.words, self.allow = words, list(allow)
        self.accepted, self.rolled = [], 0

    # -- the module's half --------------------------------------------------------------
    def GrammarMatcher(self, compiled, max_rollback_tokens):
        self.max_rollback = max_rollback_tokens
        return self

    def allocate_token_bitmask(self, rows, vocab):
        import torch
        return torch.zeros(rows, self.words, dtype=torch.int32)

    def reset_token_bitmask(self, mask):
        mask.zero_()

    # -- the matcher's half -------------------------------------------------------------
    def fill_next_token_bitmask(self, mask):
        for t in self.allow:
            mask[0, t // 32] |= 1 << (t % 32)

    def is_terminated(self):
        return False

    def accept_token(self, token):
        self.accepted.append(token)
        return True

    def rollback(self, n):
        self.rolled += n
        del self.accepted[len(self.accepted) - n:]


class MaskTransferTests(unittest.TestCase):
    """The masks cross to the device once for the step, out of buffers that are kept."""

    def matcher(self, allow=(1, 5, 70), vocab=128):
        try:
            import torch  # noqa: F401
        except ImportError as exc:
            self.skipTest(str(exc))
        from types import SimpleNamespace
        from engine.base.grammar import Matcher
        xgr = _Xgr((vocab + 31) // 32, allow)
        return Matcher(SimpleNamespace(xgr=xgr, vocab_size=vocab), None, 5), xgr, allow, vocab

    def test_every_position_gets_the_grammar_it_asked_for(self):
        m, xgr, allow, vocab = self.matcher()
        masks = m.masks([3, 4], "cpu")
        self.assertEqual(len(masks), 3)
        for mask in masks:
            self.assertEqual(mask.shape[0], vocab)
            self.assertEqual(sorted(int(i) for i in mask.nonzero().flatten()), sorted(allow))
        self.assertEqual(xgr.rolled, 2, "the walk over the drafts is taken back")

    def test_the_buffers_are_kept_between_steps(self):
        m, _, _, _ = self.matcher()
        m.masks([1], "cpu")
        staging, landing = m.staging, m.landing
        m.masks([1], "cpu")
        self.assertIs(m.staging, staging)
        self.assertIs(m.landing, landing)

    def test_a_wider_step_grows_the_buffers_and_a_narrower_one_does_not(self):
        m, _, _, _ = self.matcher()
        m.masks([1], "cpu")
        narrow = m.staging
        m.masks([1, 2, 3], "cpu")
        self.assertIsNot(m.staging, narrow)
        wide = m.staging
        m.masks([1], "cpu")
        self.assertIs(m.staging, wide)

    def test_the_crossing_happens_once_after_the_walk_not_inside_it(self):
        source = (Path(__file__).resolve().parents[1] / "engine/base/grammar.py").read_text()
        body = source[source.index("    def masks(self"):source.index("    def advance(self")]
        self.assertLess(body.index("for i in range(positions)"), body.index("landing[:filled].copy_"))
        self.assertNotIn("torch.arange(32", body)          # the shift is a kept constant
        self.assertIn("iota(32", body)


if __name__ == "__main__":
    unittest.main()

