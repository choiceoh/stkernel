"""GLM sampling defaults reach HTTP admission without overriding caller policy."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "requires PyTorch")
class GenerationDefaultsTests(unittest.TestCase):
    def defaults(self, config):
        from engine.profiles.glm53.boot import generation_defaults

        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "generation_config.json").write_text(json.dumps(config))
            return generation_defaults(tmp)

    def test_derivative_missing_top_p_inherits_original_glm_default(self):
        # Red Hat's production metadata carries temperature but no top_p.
        self.assertEqual(self.defaults({"temperature": 1.0, "eos_token_id": [154820]}),
                         {"temperature": 1.0, "top_p": 0.95})

    def test_explicit_checkpoint_policy_is_preserved(self):
        for top_p in (0.8, 0.95, 1.0):
            with self.subTest(top_p=top_p):
                config = {"temperature": 0.7, "top_p": top_p, "top_k": 40, "repetition_penalty": 1.1}
                self.assertEqual(self.defaults(config), config)

    def admitted(self, endpoint, body):
        from tests.test_engine_serve import OpenAIDialectTests, chat_server

        s = chat_server()
        s.generation = self.defaults({"temperature": 1.0})
        http = OpenAIDialectTests()
        with patch.object(s.engine, "add", wraps=s.engine.add) as add:
            http._serve(s, lambda base: http._post(base, endpoint, {**body, "max_tokens": 1}))
        self.assertEqual(add.call_count, 1)
        return add.call_args.kwargs["temperature"], add.call_args.kwargs.get("options", {})

    def test_chat_and_text_completions_use_default_or_explicit_override(self):
        for endpoint, prompt in (("/v1/chat/completions", {"messages": [{"role": "user", "content": "ab"}]}),
                                 ("/v1/completions", {"prompt": "ab"})):
            for override, temperature, top_p in (({}, 1.0, 0.95),
                                                ({"top_p": 0.8}, 1.0, 0.8),
                                                ({"top_p": 1.0}, 1.0, 1.0),
                                                ({"temperature": 0}, 0.0, 0.95)):
                with self.subTest(endpoint=endpoint, override=override):
                    actual_t, options = self.admitted(endpoint, {**prompt, **override})
                    self.assertEqual(actual_t, temperature)
                    self.assertEqual(options.get("top_p", 1.0), top_p)

    def test_raw_engine_dialect_keeps_explicit_replay_policy(self):
        for override, temperature, top_p in (({}, 0.0, 1.0),
                                            ({"temperature": 1.0}, 1.0, 1.0),
                                            ({"temperature": 1.0, "top_p": 0.95}, 1.0, 0.95)):
            with self.subTest(override=override):
                actual_t, options = self.admitted("/v1/engine/completions", {"ids": [97, 98], **override})
                self.assertEqual(actual_t, temperature)
                self.assertEqual(options.get("top_p", 1.0), top_p)


if __name__ == "__main__":
    unittest.main()
