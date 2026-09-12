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


if __name__ == "__main__":
    unittest.main()
