"""GLM's tool-call wire format -> OpenAI tool calls (engine/profiles/glm53/tools.py)."""
import json
import unittest

from engine.profiles.glm53.tools import (parse_tool_calls, partial_tool_calls,
                                         tool_call_token, tool_grammar)


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


class PartialToolCallTests(unittest.TestCase):
    """`arguments` is streamed in fragments, so what this returns may only ever grow (45차 §44)."""

    SHAPES = {
        "korean and a number": '<tool_call>get_weather<arg_key>city</arg_key><arg_value>서울특별시</arg_value>'
                               '<arg_key>days</arg_key><arg_value>3</arg_value></tool_call>',
        "whitespace around a value": '<tool_call>write<arg_key>text</arg_key><arg_value>  안녕  하세요  </arg_value></tool_call>',
        "a value that is json": '<tool_call>send<arg_key>body</arg_key><arg_value>{"a": [1, 2]}</arg_value></tool_call>',
        "no arguments at all": "<tool_call>ping</tool_call>",
        "two calls": '<tool_call>a<arg_key>k</arg_key><arg_value>v1</arg_value></tool_call>'
                     '<tool_call>b<arg_key>k</arg_key><arg_value>한글</arg_value></tool_call>',
        "quotes and escapes": '<tool_call>echo<arg_key>s</arg_key><arg_value>he said "hi"\\n끝</arg_value></tool_call>',
    }

    def test_every_prefix_only_ever_grows(self):
        """Character by character, over the whole call: a client concatenates these fragments,
        so a fragment that contradicts one already sent cannot be taken back."""
        for label, text in self.SHAPES.items():
            with self.subTest(label):
                seen = {}
                for n in range(len(text) + 1):
                    for i, (_, args, _done) in enumerate(partial_tool_calls(text[:n])):
                        self.assertTrue(args.startswith(seen.get(i, "")),
                                        f"at {n}: {args!r} does not continue {seen.get(i, '')!r}")
                        seen[i] = args

    def test_the_end_of_the_stream_is_what_the_whole_parse_says(self):
        for label, text in self.SHAPES.items():
            with self.subTest(label):
                streamed = [(name, args) for name, args, done in partial_tool_calls(text) if done]
                self.assertEqual(streamed, [tuple(c) for c in (parse_tool_calls(text) or [])])
                for _, args in streamed:
                    json.loads(args)                        # and each one is JSON, whole

    def test_a_value_whose_type_is_not_settled_yet_waits(self):
        """`3` could become `3`, `30` or `3.5`, and `{` could become anything: a number or an
        object is only rendered once `</arg_value>` says what it was. A plain string cannot
        change its type, so it streams."""
        opening = '<tool_call>f<arg_key>n</arg_key><arg_value>'
        self.assertEqual(partial_tool_calls(opening + "3")[0][1], "{")
        self.assertEqual(partial_tool_calls(opening + '{"a"')[0][1], "{")
        self.assertEqual(partial_tool_calls(opening + "서울")[0][1], '{"n": "서울')

    def test_a_closing_tag_halfway_here_is_not_value_text(self):
        opening = '<tool_call>f<arg_key>n</arg_key><arg_value>서울'
        for cut in range(len("</arg_value>")):
            self.assertEqual(partial_tool_calls(opening + "</arg_value>"[:cut])[0][1], '{"n": "서울')

    def test_a_name_is_not_reported_until_it_is_whole(self):
        self.assertEqual(partial_tool_calls("<tool_call>get_wea"), [])
        self.assertEqual(partial_tool_calls("<tool_call>get_weather<arg_k"), [])
        self.assertEqual(partial_tool_calls("<tool_call>get_weather<arg_key>")[0][0], "get_weather")
        self.assertEqual(partial_tool_calls("<tool_call>ping</tool_call>"), [("ping", "{}", True)])

    def test_korean_is_not_escaped_into_six_bytes(self):
        text = '<tool_call>search<arg_key>q</arg_key><arg_value>서울 날씨'
        self.assertNotIn("\\u", partial_tool_calls(text)[0][1])


class ToolGrammarTests(unittest.TestCase):
    """Nothing held a call to the tools that were declared (45차 §45)."""

    TOOLS = [{"type": "function", "function": {"name": "get_weather", "parameters": {
                 "type": "object", "properties": {"days": {}, "city": {}}}}},
             {"type": "function", "function": {"name": "ping"}}]

    def test_it_binds_the_names_the_keys_and_the_shape(self):
        text = tool_grammar(self.TOOLS)
        self.assertIn('call0 ::= "get_weather" pairs0 "</tool_call>"', text)
        self.assertIn('key0 ::= "city" | "days"', text)                 # sorted, so the same tools are one cache entry
        self.assertIn('pairs1 ::= ""', text)                            # a tool that takes nothing takes nothing
        self.assertIn('root ::= call ("<tool_call>" call)*', text)

    def test_a_value_may_be_anything(self):
        """A value is raw text and may hold `<`, quotes, code. Its rule admits every character,
        and the ambiguity with the closing tag is the point: inside a value nothing is forbidden."""
        self.assertIn("value ::= [^\\u0000]*", tool_grammar(self.TOOLS))

    def test_nothing_to_bind_is_no_grammar(self):
        self.assertIsNone(tool_grammar([]))
        self.assertIsNone(tool_grammar(None))
        self.assertIsNone(tool_grammar([{"type": "function", "function": {}}]))
        self.assertIsNone(tool_grammar([{"type": "function"}]))

    def test_the_trigger_is_a_token_or_there_is_no_grammar(self):
        class Whole:
            def encode(self, text, add_special_tokens=True):
                return type("E", (), {"ids": [154843]})()
            def decode(self, ids, skip_special_tokens=True):
                return "<tool_call>"

        class InPieces(Whole):
            def encode(self, text, add_special_tokens=True):
                return type("E", (), {"ids": [1, 2, 3]})()

        class PlainList(Whole):
            def encode(self, text, add_special_tokens=True):
                return [154843]

        self.assertEqual(tool_call_token(Whole()), 154843)
        self.assertEqual(tool_call_token(PlainList()), 154843)          # a transformers tokenizer answers this way
        self.assertIsNone(tool_call_token(InPieces()))                  # armed in the middle of a marker: no
        self.assertIsNone(tool_call_token(None))


if __name__ == "__main__":
    unittest.main()
