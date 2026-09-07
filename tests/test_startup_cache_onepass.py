"""Keep startup response recording compatible with canonical onepass arguments."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch


class ResponseRecorderTests(unittest.TestCase):
    def test_forwards_timing_and_decode_arguments_and_restores_original(self):
        timing = {"ctx": 2000}
        def request(url, model, content, max_tokens, received_timing, **kwargs):
            self.assertIs(received_timing, timing)
            self.assertEqual(kwargs, {"min_tokens": 3, "seed": 7})
            received_timing["elapsed_s"] = 0.25
            return "한국어 답변", 0.1, 2000, 3, "stop"
        original = Mock(side_effect=request)
        onepass = types.SimpleNamespace(ask_stream=original)
        onepass.main = lambda: onepass.ask_stream("url", "model", "문서", 400, timing, min_tokens=3, seed=7)
        path = Path(__file__).resolve().parents[1] / "bench/startup_cache_onepass.py"
        spec = importlib.util.spec_from_file_location("recorder_test", path)
        recorder = importlib.util.module_from_spec(spec)
        with tempfile.TemporaryDirectory() as root, patch.dict(sys.modules, {"onepass": onepass}), \
                patch.dict(os.environ, {"STARTUP_CACHE_RESPONSES": str(Path(root) / "responses.jsonl")}):
            spec.loader.exec_module(recorder)
            self.assertEqual(recorder.main()[0], "한국어 답변")
            self.assertIs(onepass.ask_stream, original)
            self.assertEqual(timing["elapsed_s"], 0.25)
            result = json.loads((Path(root) / "responses.jsonl").read_text())
            self.assertEqual(result["response"], "한국어 답변")
            self.assertEqual(result["completion_tokens"], 3)
            onepass.main = Mock(side_effect=RuntimeError("request failed"))
            with self.assertRaisesRegex(RuntimeError, "request failed"):
                recorder.main()
            self.assertIs(onepass.ask_stream, original)


if __name__ == "__main__":
    unittest.main()
