"""GLM's tool-call wire format -> OpenAI tool calls (engine/profiles/glm53/tools.py)."""
import json
import unittest

from engine.profiles.glm53.tools import parse_tool_calls


class ToolCallTests(unittest.TestCase):
    def test_calls_in_order_with_json_and_text_values(self):
        text = ('Checking.<tool_call>get_weather<arg_key>city</arg_key><arg_value>Seoul</arg_value>'
                '<arg_key>days</arg_key><arg_value>3</arg_value><arg_key>opts</arg_key><arg_value>{"units": "C"}</arg_value></tool_call>'
                '<tool_call>noop</tool_call>')
        calls = parse_tool_calls(text)
        self.assertEqual([name for name, _ in calls], ["get_weather", "noop"])
        self.assertEqual(json.loads(calls[0][1]), {"city": "Seoul", "days": 3, "opts": {"units": "C"}})
        self.assertEqual(calls[1][1], "{}")

    def test_text_without_a_complete_call_is_not_a_call(self):
        self.assertIsNone(parse_tool_calls("plain answer"))
        self.assertIsNone(parse_tool_calls("<tool_call>get_weather<arg_key>city</arg_key>"))
        self.assertIsNone(parse_tool_calls("<tool_call></tool_call>"))

    def test_korean_values_survive(self):
        calls = parse_tool_calls("<tool_call>search<arg_key>q</arg_key><arg_value>서울 날씨</arg_value></tool_call>")
        self.assertEqual(json.loads(calls[0][1]), {"q": "서울 날씨"})


if __name__ == "__main__":
    unittest.main()
