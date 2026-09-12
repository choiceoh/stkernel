"""The door's tokenizer must not inherit tokenizer.json's truncation rule (45차 §23).

GLM-5.3's tokenizer.json carries {"truncation": {"max_length": 2048, "direction": "Right"}}. `tokenizers` applies
it on load; transformers' AutoTokenizer (vLLM's path) does not. The engine tokenizes prompts with the former, so
every prompt longer than 2,048 tokens was cut to its head and the question at its end vanished.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class TokenizerTests(unittest.TestCase):
    def test_boot_tokenizer_drops_the_checkpoints_truncation_rule(self):
        try:
            from tokenizers import Tokenizer, models, pre_tokenizers
        except ImportError as exc:
            self.skipTest(f"tokenizers unavailable: {exc}")
        vocab = {"[UNK]": 0, "a": 1, "b": 2}
        tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
        tok.pre_tokenizer = pre_tokenizers.Whitespace()
        tok.enable_truncation(4, direction="right")
        with tempfile.TemporaryDirectory() as d:
            tok.save(str(Path(d) / "tokenizer.json"))
            saved = json.loads((Path(d) / "tokenizer.json").read_text())
            self.assertEqual(saved["truncation"]["max_length"], 4)              # the rule is in the file, as in GLM-5.3's
            self.assertEqual(len(Tokenizer.from_file(str(Path(d) / "tokenizer.json")).encode("a b a b a b a b").ids), 4)
            from engine.profiles.glm53.boot import tokenizer
            self.assertEqual(len(tokenizer(d).encode("a b a b a b a b").ids), 8)  # the door sees the whole prompt


if __name__ == "__main__":
    unittest.main()
