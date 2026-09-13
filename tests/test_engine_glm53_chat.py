"""GLM effort policy through the real renderer and HTTP door, without model weights."""
import copy
import concurrent.futures
import json
import tempfile
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


class Glm53ChatTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import tokenizers
            import transformers
        except ImportError as exc:
            raise unittest.SkipTest(f"tokenizer dependencies unavailable: {exc}")

    def renderer(self, old_checkpoint):
        from tokenizers import Tokenizer, models
        from transformers import PreTrainedTokenizerFast
        from engine.profiles.glm53.boot import CHAT_TEMPLATE, chat_renderer

        checkpoint = tempfile.TemporaryDirectory()
        self.addCleanup(checkpoint.cleanup)
        if old_checkpoint:
            # The pinned upstream template would otherwise render max unchanged.
            (Path(checkpoint.name) / CHAT_TEMPLATE).write_bytes(
                (ROOT / "tests/fixtures/glm53_official_chat_template.jinja").read_bytes())
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]")),
            unk_token="[UNK]",
        )
        with patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer):
            return chat_renderer(checkpoint.name)

    def test_effort_policy_also_applies_to_an_old_checkpoint_template(self):
        messages = [{"role": "user", "content": "test"}]
        for old_checkpoint in (False, True):
            render = self.renderer(old_checkpoint)
            for kwargs, expected in (({}, "High"), ({"reasoning_effort": None}, "High"),
                                     ({"reasoning_effort": "low"}, "Low"),
                                     ({"reasoning_effort": "high"}, "High"),
                                     ({"reasoning_effort": "max"}, "High")):
                with self.subTest(old_checkpoint=old_checkpoint, kwargs=kwargs):
                    original = copy.deepcopy(kwargs)
                    self.assertIn("Reasoning Effort: " + expected, render(messages, kwargs))
                    self.assertEqual(kwargs, original)
            for effort in ("xhigh", "enormous", False):
                with self.subTest(old_checkpoint=old_checkpoint, effort=effort):
                    with self.assertRaisesRegex(ValueError, "reasoning_effort must be"):
                        render(messages, {"reasoning_effort": effort})

    def test_max_http_requests_succeed_with_high_reasoning(self):
        from engine.profiles.glm53.boot import REASONING_EFFORT_ALIASES
        from tests.test_engine_serve import chat_server, drive

        options = [{"reasoning_effort": "max"},
                   {"chat_template_kwargs": {"reasoning_effort": "max"}},
                   {"reasoning_effort": "max", "chat_template_kwargs": {"reasoning_effort": "max"}},
                   {"reasoning_effort": None, "chat_template_kwargs": {"reasoning_effort": "max"}},
                   {"reasoning_effort": "max", "chat_template_kwargs": {"reasoning_effort": None}},
                   {"reasoning_effort": "max", "chat_template_kwargs": {"reasoning_effort": "high"}},
                   {"reasoning_effort": "high", "chat_template_kwargs": {"reasoning_effort": "max"}},
                   {"reasoning_effort": "max", "chat_template_kwargs": {"thinking": False}}]
        for old_checkpoint in (False, True):
            server = chat_server(blocks=128, prefix=4, reasoning_effort_aliases=REASONING_EFFORT_ALIASES)
            render, prompts = self.renderer(old_checkpoint), []
            def chat(*args, **kwargs):
                prompt = render(*args, **kwargs)
                prompts.append(prompt)
                return prompt
            server.chat = chat
            httpd = server._serve_http()
            try:
                with concurrent.futures.ThreadPoolExecutor(1) as pool:
                    requests = [("/v1/chat/completions", dict(option, stream=stream))
                                for option in options for stream in (False, True)]
                    requests += [(route, {"chat_template_kwargs": {"reasoning_effort": "max"}})
                                 for route in ("/tokenize", "/v1/prefix/warm")]
                    for route, option in requests:
                        with self.subTest(old_checkpoint=old_checkpoint, route=route, option=option):
                            body = dict(option, messages=[{"role": "user", "content": "test"}], max_tokens=1)
                            request = urllib.request.Request(
                                f"http://127.0.0.1:{httpd.server_port}{route}",
                                data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
                            def post():
                                with urllib.request.urlopen(request, timeout=3) as response:
                                    return response.status, response.read()
                            status, response = drive(server, pool.submit(post))
                            self.assertEqual(status, 200, response)
                            self.assertIn("Reasoning Effort: High", prompts[-1])
                            self.assertNotIn("Reasoning Effort: Max", prompts[-1])
                    for top, nested in (("max", "low"), ("low", "max")):
                        request = urllib.request.Request(
                            f"http://127.0.0.1:{httpd.server_port}/v1/chat/completions",
                            data=json.dumps({"messages": [{"role": "user", "content": "test"}],
                                             "reasoning_effort": top,
                                             "chat_template_kwargs": {"reasoning_effort": nested}}).encode(),
                            headers={"Content-Type": "application/json"})
                        with self.assertRaises(urllib.error.HTTPError) as error:
                            urllib.request.urlopen(request, timeout=3)
                        self.assertEqual(error.exception.code, 400)
                        self.assertIn("must agree", json.load(error.exception)["error"])
            finally:
                httpd.shutdown()
                httpd.server_close()


if __name__ == "__main__":
    unittest.main()
