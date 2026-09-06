"""CPU checks for generation verdicts; no HTTP request or model is needed."""
import importlib.util
import json
from pathlib import Path
import unittest

from jsonschema import ValidationError

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("acceptance", ROOT / "probes/glm53_tool_choice_acceptance.py")
if spec is None or spec.loader is None:
    raise RuntimeError("Cannot load probes/glm53_tool_choice_acceptance.py")
acceptance = importlib.util.module_from_spec(spec)
spec.loader.exec_module(acceptance)


class VerdictTests(unittest.TestCase):
    schema = {"type": "object", "properties": {"tag": {"type": "string"}}, "required": ["tag"]}

    def packet(self, delta, finish=None):
        return ("data: " + json.dumps({"choices": [{"index": 0, "delta": delta,
            "finish_reason": finish}]}) + "\n").encode()

    def stream(self, call_type="function"):
        return [self.packet({"reasoning": "private reasoning"}), self.packet({"tool_calls": [
            {"index": 0, "id": "a", **({"type": call_type} if call_type else {}),
             "function": {"name": "record_job", "arguments": '{"tag":'}}]}),
            self.packet({"tool_calls": [{"index": 0, "function": {"arguments": '"서울"}'}}]}, "tool_calls"),
            b"data: [DONE]\n"]

    def test_complete_typed_tool_and_reasoning_separation(self):
        message, finish = acceptance.read_stream(self.stream())
        self.assertEqual(message["content"], "")
        acceptance.check_response(message, finish, "required", self.schema)

    def test_disconnect_and_id_change_do_not_pass(self):
        for lines in (self.stream()[:-1], self.stream()[:2] + [self.packet({"tool_calls": [
                {"index": 0, "id": "different", "function": {}}]})] + self.stream()[2:]):
            with self.assertRaises(ValueError):
                acceptance.read_stream(lines)

    def test_length_missing_required_and_partial_json_do_not_pass(self):
        message, _ = acceptance.read_stream(self.stream())
        with self.assertRaises(ValueError):
            acceptance.check_response(message, "length", "named", self.schema)
        for args in ('{"tag":', '{}'):
            message["tool_calls"][0]["function"]["arguments"] = args
            with self.assertRaises((ValueError, ValidationError)):
                acceptance.check_response(message, "tool_calls", "named", self.schema)

    def test_none_requires_final_content(self):
        for message in ({"reasoning": "only reasoning"}, {"content": "record_job"},
                        {"content": "literal <tool_call>"}):
            with self.assertRaises(ValueError):
                acceptance.check_response(message, "stop", "none", self.schema)
        acceptance.check_response({"content": "```xml\n<tool_call>record_job</tool_call>\n```"},
                                  "stop", "none", self.schema)

    def test_missing_or_wrong_wire_tool_type_cannot_pass(self):
        for kind in (None, "custom"):
            with self.subTest(stream=True, kind=kind), self.assertRaises(ValueError):
                message, finish = acceptance.read_stream(self.stream(call_type=kind))
                acceptance.check_response(message, finish, "required", self.schema)
            message, finish = acceptance.read_stream(self.stream())
            if kind:
                message["tool_calls"][0]["type"] = kind
            else:
                message["tool_calls"][0].pop("type")
            with self.subTest(stream=False, kind=kind), self.assertRaises(ValueError):
                acceptance.check_response(message, finish, "required", self.schema)


if __name__ == "__main__":
    unittest.main()
